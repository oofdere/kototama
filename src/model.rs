//! Model loading and vocabulary queries.

use std::ffi::{c_char, CStr, CString};
use std::ops::{Deref, DerefMut};
use std::ptr::{null, null_mut};
use std::sync::Arc;

use llama_sys::*;

use crate::{Backend, Error};

/// Parameters for [`Model::load_from_file`], mirroring `llama_model_params`.
///
/// Derefs to the raw `llama_model_params`, so fields can be set directly:
///
/// ```
/// use rusty_llama::ModelParams;
///
/// let mut params = ModelParams::new();
/// params.n_gpu_layers = 99;
/// ```
#[repr(transparent)]
pub struct ModelParams(llama_model_params);

impl ModelParams {
    /// Defaults from `llama_model_default_params`.
    pub fn new() -> Self {
        Self(unsafe { llama_model_default_params() })
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

struct ModelInner {
    model: *mut llama_model,
    vocab: *const llama_vocab,
    // Held so the backend stays initialized for as long as the model lives.
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

/// A loaded model. Cheap to clone; all clones share one `llama_model`.
///
/// Besides loading, this type is the vocabulary: tokenization, detokenization,
/// and special-token queries all live here.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl Model {
    /// Load a GGUF model from `path`.
    ///
    /// The [`Backend`] is acquired automatically and released when the last
    /// clone of the model is dropped.
    pub fn load_from_file(path: &str, params: ModelParams) -> Result<Self, Error> {
        let _backend = Backend::acquire();
        let path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        let model = unsafe { llama_model_load_from_file(path.as_ptr(), params.into()) };
        if model.is_null() {
            return Err(Error::ModelLoadFailed);
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

    /// Raw pointer to the underlying `llama_model`.
    pub fn as_ptr(&self) -> *const llama_model {
        self.inner.model
    }

    /// The model's chat template, or `None` when it has none.
    ///
    /// `name` selects a named template; `None` uses the default one.
    pub fn chat_template(&self, name: Option<&str>) -> Option<String> {
        let name_cstr = name.map(|s| CString::new(s).unwrap());
        let name_ptr = name_cstr.as_ref().map(|s| s.as_ptr()).unwrap_or(null());
        let str = unsafe { llama_model_chat_template(self.inner.model, name_ptr) };
        if str.is_null() {
            None
        } else {
            let cstr = unsafe { CStr::from_ptr(str) };
            Some(cstr.to_string_lossy().into_owned())
        }
    }

    /// Human-readable model description (`llama_model_desc`).
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

    /// `true` when the model needs `llama_decode` to produce output.
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.inner.model) }
    }

    /// Start-of-generation token for encoder-decoder models.
    pub fn decoder_start_token(&self) -> Option<i32> {
        let token = unsafe { llama_model_decoder_start_token(self.inner.model) };
        (token != LLAMA_TOKEN_NULL).then_some(token)
    }

    /// `true` when the model has an encoder (`llama_encode`).
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.inner.model) }
    }

    /// `true` for diffusion models.
    pub fn is_diffusion(&self) -> bool {
        unsafe { llama_model_is_diffusion(self.inner.model) }
    }

    /// `true` for hybrid (attention + recurrence) models.
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_model_is_hybrid(self.inner.model) }
    }

    /// `true` for recurrent models.
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_model_is_recurrent(self.inner.model) }
    }

    /// Render one token back to text.
    ///
    /// Fails with [`Error::NoPiece`] when the token has no text (e.g. some
    /// special tokens).
    pub fn token_to_piece(&self, token: i32) -> Result<String, Error> {
        let mut buf = vec![0u8; 64];
        let n = unsafe {
            llama_token_to_piece(
                self.inner.vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,
                true,
            )
        };
        if n < 0 {
            // The piece does not fit: llama.cpp reports the required size as a
            // negative count. Retry once with a buffer of exactly that size.
            buf.resize(-n as usize, 0);
            let n = unsafe {
                llama_token_to_piece(
                    self.inner.vocab,
                    token,
                    buf.as_mut_ptr() as *mut i8,
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n < 0 {
                return Err(Error::NoPiece);
            }
            buf.truncate(n as usize);
        } else {
            buf.truncate(n as usize);
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Split `text` into token ids.
    ///
    /// `add_special` prepends the BOS token when the vocabulary wants one;
    /// `parse_special` makes special-token markers like `<eos>` part of the
    /// tokenization instead of plain text.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
        let len = -unsafe {
            llama_tokenize(
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
            llama_tokenize(
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

    pub(crate) fn as_mut_ptr(&self) -> *mut llama_model {
        self.inner.model
    }

    pub(crate) fn vocab_ptr(&self) -> *const llama_vocab {
        self.inner.vocab
    }
}
