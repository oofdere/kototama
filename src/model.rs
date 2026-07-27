use crate::Backend;
use llama_sys::*;
use std::{
    ffi::c_char,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
    sync::Arc,
};

pub struct ModelParams(pub llama_sys::llama_model_params);

impl ModelParams {
    pub fn new() -> Self {
        Self(unsafe { llama_sys::llama_model_default_params() })
    }

    pub fn as_ptr(&self) -> *const llama_sys::llama_model_params {
        &self.0
    }

    pub fn as_mut_ptr(&mut self) -> *mut llama_sys::llama_model_params {
        &mut self.0
    }
}

impl Deref for ModelParams {
    type Target = llama_sys::llama_model_params;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ModelParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Into<llama_sys::llama_model_params> for ModelParams {
    fn into(self) -> llama_sys::llama_model_params {
        self.0
    }
}

/// Errors returned by [`Model::try_tokenize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizeError {
    /// The text contains a NUL byte at this index, which llama.cpp cannot
    /// encode: it throws a C++ exception that would unwind into Rust.
    InteriorNul(usize),
}

struct ModelInner {
    model: *mut llama_model,
    pub(crate) vocab: *const llama_vocab,
    _backend: Backend,
}

// SAFETY: llama_model is immutable after creation. All methods on Model
// (tokenize, token_to_piece, vocab queries, desc, etc.) are read-only
// operations on the underlying C data. llama.cpp does not mutate the
// model during these calls.
unsafe impl Send for ModelInner {}
unsafe impl Sync for ModelInner {}

impl Drop for ModelInner {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.model) };
    }
}

/// Thread-safe handle to a loaded model.
///
/// Cloning is cheap (Arc bump). Use from any thread.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl Model {
    pub fn load_from_file(path: &str, params: ModelParams) -> Result<Self, ()> {
        let _backend = Backend::acquire();
        let path = std::ffi::CString::new(path).map_err(|_| ())?;
        let model = unsafe { llama_model_load_from_file(path.as_ptr(), params.into()) };

        if model.is_null() {
            return Err(());
        }

        let vocab = unsafe { llama_model_get_vocab(model) };

        Ok(Self {
            inner: Arc::new(ModelInner {
                model,
                vocab,
                _backend,
            }),
        })
    }

    pub fn as_ptr(&self) -> *const llama_model {
        self.inner.model
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut llama_model {
        self.inner.model
    }

    pub(crate) fn vocab_ptr(&self) -> *const llama_vocab {
        self.inner.vocab
    }

    pub fn chat_template(&self, name: Option<&str>) -> Option<String> {
        let name_cstr = name.map(|s| std::ffi::CString::new(s).unwrap());
        let name_ptr = name_cstr.as_ref().map(|s| s.as_ptr()).unwrap_or(null());
        let str = unsafe { llama_model_chat_template(self.inner.model, name_ptr) };
        if str.is_null() {
            None
        } else {
            let cstr = unsafe { std::ffi::CStr::from_ptr(str) };
            Some(cstr.to_string_lossy().to_string())
        }
    }

    pub fn desc(&self) -> String {
        let needed = unsafe { llama_model_desc(self.inner.model, null_mut(), 0) };
        if needed <= 0 {
            return String::new();
        }
        let buf_size = (needed as usize) + 1;
        let mut buf = vec![0u8; buf_size];
        let written = unsafe {
            llama_model_desc(self.inner.model, buf.as_mut_ptr() as *mut c_char, buf_size)
        };
        if written <= 0 {
            return String::new();
        }
        let written = (written as usize).min(buf_size - 1);
        String::from_utf8_lossy(&buf[..written]).into_owned()
    }

    #[inline]
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.inner.model) }
    }

    #[inline]
    pub fn decoder_start_token(&self) -> Option<i32> {
        let token = unsafe { llama_model_decoder_start_token(self.inner.model) };
        if token == LLAMA_TOKEN_NULL {
            None
        } else {
            Some(token)
        }
    }

    #[inline]
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.inner.model) }
    }

    #[inline]
    pub fn is_diffusion(&self) -> bool {
        unsafe { llama_model_is_diffusion(self.inner.model) }
    }

    #[inline]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_model_is_hybrid(self.inner.model) }
    }

    #[inline]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_model_is_recurrent(self.inner.model) }
    }

    pub fn token_to_piece(&self, token: i32) -> Result<String, ()> {
        let mut buf = [0u8; 64];
        let n = unsafe {
            llama_sys::llama_token_to_piece(
                self.inner.vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,
                true,
            )
        };
        if n < 0 {
            return Err(());
        }
        Ok(String::from_utf8_lossy(&buf[..n as usize]).to_string())
    }

    /// Tokenize `text`, returning an error instead of letting llama.cpp abort
    /// the process on input it cannot encode.
    ///
    /// A NUL byte is rejected: llama.cpp falls back to `byte_to_token`, which
    /// looks the byte up as a one-character `std::string` — for NUL that string
    /// is empty and never present in a vocab, so `unordered_map::at` throws
    /// `std::out_of_range`. That C++ exception would unwind through the
    /// `extern "C"` boundary into Rust, which is undefined behaviour and
    /// aborts with "Rust cannot catch foreign exceptions".
    pub fn try_tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, TokenizeError> {
        if let Some(pos) = text.as_bytes().iter().position(|&b| b == 0) {
            return Err(TokenizeError::InteriorNul(pos));
        }
        Ok(self.tokenize_unchecked(text, add_special, parse_special))
    }

    /// Tokenize `text`.
    ///
    /// # Panics
    ///
    /// Panics if `text` contains a NUL byte; see [`Model::try_tokenize`] for a
    /// non-panicking variant.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
        match self.try_tokenize(text, add_special, parse_special) {
            Ok(tokens) => tokens,
            Err(e) => panic!("tokenize failed: {e:?}"),
        }
    }

    fn tokenize_unchecked(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
        let len = -unsafe {
            llama_sys::llama_tokenize(
                self.inner.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                std::ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };
        let mut tokens = vec![0i32; len as usize];
        let n_tokens = unsafe {
            llama_sys::llama_tokenize(
                self.inner.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                parse_special,
            )
        };
        tokens.truncate(n_tokens as usize);
        tokens
    }
}
