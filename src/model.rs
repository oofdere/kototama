//! Model loading and the immutable per-model API.
//!
//! [`Model`] is the entry point of the crate: load a GGUF file once, then hand
//! the cheap [`Arc`]-backed handle to as many threads as you like. Loading also
//! transparently acquires a [`Backend`] handle, so callers normally never need
//! to touch [`Backend`] directly.
//!
//! Construction parameters go through [`ModelParams`], a `#[repr(transparent)]`
//! wrapper around `llama_model_params`. [`ModelParams::new`] is the only safe
//! way to build one — the underlying llama.cpp struct is **not** zero-init-safe
//! (it carries function pointers and tagged enums). Tune fields directly via
//! [`Deref`] / [`DerefMut`]:
//!
//! ```ignore
//! let mut params = ModelParams::new();
//! params.n_gpu_layers = 0;            // direct field access via Deref
//! let model = Model::load_from_file("./model.gguf", params)?;
//! ```
//!
//! The token vocabulary accessors on `Model` (`bos_token`, `n_tokens`,
//! `tokenize`, `token_to_piece`, …) live in the `vocab` module but are
//! `impl Model` blocks so they appear directly on this type.

use crate::Backend;
use llama_sys::*;
use std::{
    ffi::c_char,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
    sync::Arc,
};

/// Construction parameters for [`Model::load_from_file`].
///
/// Wraps llama.cpp's `llama_model_params` so the raw fields can be tuned with
/// regular Rust syntax via [`Deref`] / [`DerefMut`] — e.g. `params.n_gpu_layers
/// = 0` to force CPU-only inference. The raw struct must never be constructed
/// by hand (it contains function pointers and tagged enums that are not
/// zero-init-safe); always start from [`ModelParams::new`], which seeds every
/// field with the upstream defaults from `llama_model_default_params`.
pub struct ModelParams(pub llama_sys::llama_model_params);

impl ModelParams {
    /// Build a [`ModelParams`] seeded with llama.cpp's default values.
    ///
    /// This is the only safe constructor; the underlying struct is not
    /// zero-init-safe.
    pub fn new() -> Self {
        Self(unsafe { llama_sys::llama_model_default_params() })
    }

    /// Borrow the inner `llama_model_params` as a raw const pointer.
    ///
    /// Intended for callers that need to interop with llama.cpp's C API
    /// directly; the pointer is valid for the lifetime of `&self`.
    pub fn as_ptr(&self) -> *const llama_sys::llama_model_params {
        &self.0
    }

    /// Borrow the inner `llama_model_params` as a raw mutable pointer.
    ///
    /// Intended for callers that need to interop with llama.cpp's C API
    /// directly; the pointer is valid for the lifetime of `&mut self`.
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

/// Thread-safe handle to a loaded llama.cpp model.
///
/// The underlying `llama_model` is immutable after load, so [`Model`] is
/// `Clone + Send + Sync` and every clone shares the same model via [`Arc`].
/// Cloning is cheap (a single atomic bump) — pass handles around freely
/// instead of re-loading.
///
/// Construct with [`Model::load_from_file`]; the loaded model is freed when
/// the last clone is dropped. The handle also keeps a [`Backend`] reference
/// alive internally, so the global backend stays initialised for as long as
/// any [`Model`] exists.
///
/// All methods on [`Model`] (tokenize, token_to_piece, vocab queries, desc,
/// etc.) are read-only — safe to call concurrently from multiple threads.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl Model {
    /// Load a GGUF model from `path`.
    ///
    /// Acquires a [`Backend`] handle internally, so the global llama.cpp
    /// backend is initialised on first call and torn down only after the last
    /// [`Model`] (and any [`crate::Context`] derived from it) is dropped.
    ///
    /// Returns `Err(())` if the path contains an interior NUL byte or if
    /// llama.cpp fails to load the file (e.g. missing file, corrupted GGUF,
    /// or unsupported architecture). The error type is intentionally
    /// unit — llama.cpp does not surface a structured load error.
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

    /// Raw pointer to the underlying `llama_model`.
    ///
    /// Intended for callers that need to call llama.cpp APIs not yet wrapped
    /// by this crate. The pointer is valid as long as `self` (or any other
    /// clone of this [`Model`]) is alive.
    pub fn as_ptr(&self) -> *const llama_model {
        self.inner.model
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut llama_model {
        self.inner.model
    }

    pub(crate) fn vocab_ptr(&self) -> *const llama_vocab {
        self.inner.vocab
    }

    /// Return the model's chat template, if any.
    ///
    /// Passing `None` requests the default template embedded in the GGUF
    /// metadata; passing `Some("name")` requests a named template
    /// (`tokenizer.chat_template.<name>` in GGUF). Returns `None` if the
    /// model carries no matching template.
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

    /// Human-readable description of the model architecture and size.
    ///
    /// Typical output is a string like `"llama 7B mostly Q4_0"` — the same
    /// format llama.cpp's CLI prints on load. Returns an empty string if
    /// llama.cpp could not produce a description.
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

    /// True if the model has a decoder stack (the common case for causal LMs).
    #[inline]
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.inner.model) }
    }

    /// Decoder start token for encoder–decoder models, if any.
    ///
    /// Returns `None` for decoder-only models (the common case).
    #[inline]
    pub fn decoder_start_token(&self) -> Option<i32> {
        let token = unsafe { llama_model_decoder_start_token(self.inner.model) };
        if token == LLAMA_TOKEN_NULL {
            None
        } else {
            Some(token)
        }
    }

    /// True if the model has an encoder stack (encoder-only or
    /// encoder–decoder architectures).
    #[inline]
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.inner.model) }
    }

    /// True if the model is a diffusion model.
    #[inline]
    pub fn is_diffusion(&self) -> bool {
        unsafe { llama_model_is_diffusion(self.inner.model) }
    }

    /// True if the model mixes attention and recurrent layers
    /// (e.g. Jamba-style hybrids).
    #[inline]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_model_is_hybrid(self.inner.model) }
    }

    /// True if the model is purely recurrent (e.g. Mamba / RWKV).
    #[inline]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_model_is_recurrent(self.inner.model) }
    }

    /// Decode a single token id to its piece (the model's tokenizer string).
    ///
    /// Renders with `render_special = true`, so control tokens like BOS/EOS
    /// produce their textual form (e.g. `"<s>"`). Returns `Err(())` if
    /// llama.cpp rejects the token (typically an out-of-range id) or if the
    /// piece does not fit in the internal 64-byte buffer.
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

    /// Tokenize `text` into a sequence of token ids.
    ///
    /// - `add_special`: prepend the model's BOS (and append EOS, if the
    ///   tokenizer is configured to do so) — the usual choice when tokenizing
    ///   a fresh prompt. See [`Model::get_add_bos`] / [`Model::get_add_eos`]
    ///   to inspect what the model itself prefers.
    /// - `parse_special`: if true, sequences like `"<|im_start|>"` are parsed
    ///   as their corresponding special token ids; if false, they tokenize
    ///   verbatim as bytes.
    ///
    /// Returns an empty `Vec` for empty input when `add_special` is false.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
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
