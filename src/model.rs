//! Loading models.
//!
//! A [`Model`] is the loaded weights plus the vocabulary (the tokenizer). It is
//! read-only once loaded, which makes it the piece that's easiest to share:
//! many [`crate::Context`]s can run on top of one `Model` without copying the
//! weights.

use std::ffi::{c_char, CStr, CString};
use std::ops::{Deref, DerefMut};
use std::ptr::{null, null_mut};
use std::sync::Arc;

use llama_sys::*;

use crate::{Backend, Token};

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Settings used when loading a [`Model`].
///
/// Wraps llama.cpp's `llama_model_params` and derefs to it, so you can set
/// fields directly:
///
/// ```no_run
/// # use rusty_llama::ModelParams;
/// let mut params = ModelParams::new();
/// params.n_gpu_layers = 99; // offload as many layers to the GPU as possible
/// ```
pub struct ModelParams(pub llama_model_params);

impl ModelParams {
    /// llama.cpp's default model settings.
    pub fn new() -> Self {
        Self(unsafe { llama_model_default_params() })
    }

    pub fn as_ptr(&self) -> *const llama_model_params {
        &self.0
    }

    pub fn as_mut_ptr(&mut self) -> *mut llama_model_params {
        &mut self.0
    }
}

impl Default for ModelParams {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for ModelParams {
    type Target = llama_model_params;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ModelParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ModelParams> for llama_model_params {
    fn from(params: ModelParams) -> Self {
        params.0
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The part of a [`Model`] that's actually shared between clones.
struct ModelInner {
    model: *mut llama_model,
    /// Borrowed from `model`; llama.cpp frees it together with the model.
    vocab: *const llama_vocab,
    /// Keeps the llama.cpp backend initialised while this model exists.
    _backend: Backend,
}

// SAFETY: a `llama_model` is never modified after loading. Every method on
// `Model` (tokenizing, vocab lookups, metadata) only reads from it, and
// llama.cpp supports reading one model from many threads at once. That's what
// makes it sound to share the same weights across threads later on.
unsafe impl Send for ModelInner {}
unsafe impl Sync for ModelInner {}

impl Drop for ModelInner {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.model) };
    }
}

/// A loaded model: weights + vocabulary.
///
/// Cloning is cheap — it bumps a reference count and gives you another handle
/// to the *same* weights in memory. The model is freed when the last clone is
/// dropped (contexts hold a clone too, so they keep it alive).
///
/// Unlike [`crate::Context`], `Model` is `Send + Sync`. It already uses
/// `Arc` rather than `Rc`, so when threading arrives, every thread can hold a
/// clone and they'll all read the same weights with no copying.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl Model {
    /// Load a GGUF model file from disk.
    pub fn load_from_file(path: &str, params: ModelParams) -> Result<Self, ()> {
        // Make sure the backend is up before we ask llama.cpp to do anything.
        let backend = Backend::acquire();

        let path = CString::new(path).map_err(|_| ())?;
        let model = unsafe { llama_model_load_from_file(path.as_ptr(), params.into()) };
        if model.is_null() {
            return Err(());
        }
        let vocab = unsafe { llama_model_get_vocab(model) };

        Ok(Self {
            inner: Arc::new(ModelInner {
                model,
                vocab,
                _backend: backend,
            }),
        })
    }

    /// The raw `llama_model` pointer, for calling llama.cpp directly.
    pub fn as_ptr(&self) -> *const llama_model {
        self.inner.model
    }

    /// Some llama.cpp functions take `*mut llama_model` even though they don't
    /// modify it. Crate-private so users can't misuse it.
    pub(crate) fn as_mut_ptr(&self) -> *mut llama_model {
        self.inner.model
    }

    pub(crate) fn vocab_ptr(&self) -> *const llama_vocab {
        self.inner.vocab
    }

    // -- Metadata ----------------------------------------------------------

    /// The chat template stored in the model file, if it has one. Pass `None`
    /// for the default template, or `Some(name)` for a named one.
    pub fn chat_template(&self, name: Option<&str>) -> Option<String> {
        let name = name.map(|s| CString::new(s).unwrap());
        let name_ptr = name.as_ref().map_or(null(), |s| s.as_ptr());
        let ptr = unsafe { llama_model_chat_template(self.inner.model, name_ptr) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    /// A short human-readable description, e.g. `"llama 7B Q4_0"`.
    pub fn desc(&self) -> String {
        // First call with no buffer to learn how long the string is.
        let needed = unsafe { llama_model_desc(self.inner.model, null_mut(), 0) };
        if needed <= 0 {
            return String::new();
        }
        let buf_size = needed as usize + 1; // +1 for the trailing NUL
        let mut buf = vec![0u8; buf_size];
        let written =
            unsafe { llama_model_desc(self.inner.model, buf.as_mut_ptr() as *mut c_char, buf_size) };
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

    /// For encoder-decoder models: the token the decoder should start with.
    #[inline]
    pub fn decoder_start_token(&self) -> Option<Token> {
        let token = unsafe { llama_model_decoder_start_token(self.inner.model) };
        (token != LLAMA_TOKEN_NULL).then_some(token)
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

    // -- Tokenizing --------------------------------------------------------

    /// Turn text into tokens.
    ///
    /// * `add_special` — add the model's BOS/EOS tokens if it wants them. Use
    ///   `true` at the very start of a conversation.
    /// * `parse_special` — treat special-token text like `<|im_start|>` as the
    ///   special token rather than as ordinary characters.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<Token> {
        let tokenize_into = |buf: &mut [Token]| unsafe {
            llama_tokenize(
                self.inner.vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                add_special,
                parse_special,
            )
        };

        // With an empty buffer llama.cpp returns minus the number of tokens it
        // needs, so we call it once to size the buffer and once to fill it.
        let needed = -tokenize_into(&mut []);
        let mut tokens = vec![0; needed.max(0) as usize];
        let n = tokenize_into(&mut tokens);
        tokens.truncate(n.max(0) as usize);
        tokens
    }

    /// Turn one token back into text. Special tokens are rendered too.
    pub fn token_to_piece(&self, token: Token) -> Result<String, ()> {
        let mut buf = [0u8; 64];
        let n = unsafe {
            llama_token_to_piece(
                self.inner.vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,    // lstrip: don't strip leading spaces
                true, // special: render special tokens as text
            )
        };
        if n < 0 {
            return Err(());
        }
        Ok(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
    }
}
