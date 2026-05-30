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

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
        // `text.len() as i32` silently wraps for strings longer than
        // `i32::MAX` bytes. The C side does `std::string(text, text_len)`,
        // which converts the wrapped-negative length back to a huge `size_t`
        // and copies far past the end of the input buffer — undefined
        // behaviour reachable from purely safe Rust. Reject the call before
        // crossing the FFI boundary.
        let text_len = i32::try_from(text.len()).unwrap_or_else(|_| {
            panic!(
                "tokenize: text length ({} bytes) exceeds i32::MAX",
                text.len()
            )
        });

        let probe = unsafe {
            llama_sys::llama_tokenize(
                self.inner.vocab,
                text.as_ptr() as *const i8,
                text_len,
                std::ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };

        // `llama_tokenize` reserves `i32::MIN` to signal that the
        // tokenization result would exceed `i32::MAX` tokens. Negating that
        // in `i32` overflows: debug builds panic on the unary `-`; release
        // builds wrap back to `i32::MIN`, and `vec![0; i32::MIN as usize]`
        // then asks the allocator for ~9 EiB and aborts the process.
        let len = match probe {
            i32::MIN => panic!(
                "tokenize: tokenization result would exceed i32::MAX tokens"
            ),
            n if n < 0 => -n,
            // 0 means "nothing to write"; the probe call passes
            // `n_tokens_max == 0`, so a positive return is outside the
            // documented API contract — bail safely instead of negating to
            // a huge `usize` and tripping the allocator.
            _ => return Vec::new(),
        };

        let mut tokens = vec![0i32; len as usize];
        let n_tokens = unsafe {
            llama_sys::llama_tokenize(
                self.inner.vocab,
                text.as_ptr() as *const i8,
                text_len,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                parse_special,
            )
        };
        // A negative return from the second call means the C side hit an
        // error path with our sized buffer. `truncate(negative as usize)`
        // would cast to a huge value, do nothing, and silently leave the
        // caller with a vector full of zeros. Drop the buffer instead.
        if n_tokens < 0 {
            return Vec::new();
        }
        tokens.truncate(n_tokens as usize);
        tokens
    }
}
