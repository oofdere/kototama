//! Loading models and inspecting their architecture.
//!
//! A [`Model`] is a thread-safe, reference-counted handle to a weights file
//! that has been mmapped/loaded by llama.cpp. Use [`Model::load_from_file`] to
//! construct one; clones are cheap (an [`Arc`] bump) and can be moved across
//! threads freely. Vocabulary inspection (special tokens, scores, token text)
//! lives on the same type — see the [`vocab`](crate::vocab) module.
//!
//! [`Model`] owns a [`Backend`] guard internally, so the process-wide
//! llama.cpp backend stays initialized for as long as any [`Model`] is alive.
//! Callers normally do not need to touch [`Backend`] directly.
//!
//! ## Construction parameters
//!
//! [`ModelParams`] is a thin wrapper over `llama_model_params` from the C API.
//! Tune fields like `n_gpu_layers` through [`Deref`]/[`DerefMut`] before
//! passing it to [`Model::load_from_file`]:
//!
//! ```ignore
//! let mut params = ModelParams::new();
//! params.n_gpu_layers = 99;
//! let model = Model::load_from_file("model.gguf", params)?;
//! ```

use crate::Backend;
use llama_sys::*;
use std::{
    ffi::c_char,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
    sync::Arc,
};

/// Parameters for loading a [`Model`].
///
/// Thin wrapper over `llama_model_params` from the C API. Construct one with
/// [`ModelParams::new`] (which calls `llama_model_default_params`), tune
/// fields like `n_gpu_layers` or `use_mmap` through [`Deref`]/[`DerefMut`],
/// then hand it to [`Model::load_from_file`].
pub struct ModelParams(pub llama_sys::llama_model_params);

impl ModelParams {
    /// Create a [`ModelParams`] seeded with llama.cpp's default values
    /// (`llama_model_default_params`).
    pub fn new() -> Self {
        Self(unsafe { llama_sys::llama_model_default_params() })
    }

    /// Borrow the inner `llama_model_params` as a `*const` pointer, for FFI
    /// calls that need a raw pointer to a read-only params struct.
    pub fn as_ptr(&self) -> *const llama_sys::llama_model_params {
        &self.0
    }

    /// Borrow the inner `llama_model_params` as a `*mut` pointer, for FFI
    /// calls that need a raw pointer to a mutable params struct.
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
    /// Load a model from a GGUF file on disk.
    ///
    /// Acquires a [`Backend`] guard internally and stashes it inside the
    /// returned [`Model`], so the llama.cpp backend stays initialized for as
    /// long as the model (or any clone of it) is alive.
    ///
    /// Returns `Err(())` if `path` is not a valid C string (contains an
    /// interior `\0`) or if llama.cpp fails to load the model (file missing,
    /// unsupported quantization, OOM, etc.). llama.cpp logs the underlying
    /// reason to stderr.
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

    /// Raw pointer to the underlying `llama_model`, for FFI calls that need
    /// one. The pointer is valid for as long as `self` (or any clone) is
    /// alive.
    pub fn as_ptr(&self) -> *const llama_model {
        self.inner.model
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut llama_model {
        self.inner.model
    }

    pub(crate) fn vocab_ptr(&self) -> *const llama_vocab {
        self.inner.vocab
    }

    /// Look up a Jinja chat template baked into the GGUF metadata.
    ///
    /// Passing `None` requests the model's default template (the
    /// `tokenizer.chat_template` key); passing `Some(name)` requests a named
    /// alternate template (the `tokenizer.chat_template.<name>` key).
    /// Returns [`None`] if the model has no template by that name — many
    /// non-chat models legitimately ship without one.
    ///
    /// # Panics
    ///
    /// Panics if `name` contains an interior `\0` byte.
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

    /// Human-readable architecture description, e.g. `"llama 7B Q4_0"`.
    ///
    /// Returns an empty string if llama.cpp reports no description (the
    /// underlying `llama_model_desc` returns `<= 0`).
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

    /// Whether this model has a decoder stack (i.e. it can produce tokens
    /// auto-regressively).
    ///
    /// True for decoder-only models like Llama and most chat models; true for
    /// the decoder half of encoder-decoder models like T5; false for
    /// encoder-only models like BERT.
    #[inline]
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.inner.model) }
    }

    /// The token an encoder-decoder model expects as the first decoder input
    /// (e.g. `<pad>` for T5), or [`None`] for models without a separate
    /// decoder-start token.
    #[inline]
    pub fn decoder_start_token(&self) -> Option<llama_token> {
        let token = unsafe { llama_model_decoder_start_token(self.inner.model) };
        if token == LLAMA_TOKEN_NULL {
            None
        } else {
            Some(token)
        }
    }

    /// Whether this model has an encoder stack.
    ///
    /// True for encoder-decoder architectures (T5, BART) and encoder-only
    /// models (BERT); false for plain decoder-only LMs like Llama.
    #[inline]
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.inner.model) }
    }

    /// Whether this is a diffusion language model.
    #[inline]
    pub fn is_diffusion(&self) -> bool {
        unsafe { llama_model_is_diffusion(self.inner.model) }
    }

    /// Whether this is a hybrid architecture (e.g. mixing attention with a
    /// recurrent or state-space backbone, like Jamba or Zamba).
    #[inline]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_model_is_hybrid(self.inner.model) }
    }

    /// Whether this model uses a recurrent (RNN-style or state-space)
    /// architecture rather than pure attention — e.g. Mamba or RWKV.
    #[inline]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_model_is_recurrent(self.inner.model) }
    }

    /// Render a single token id back to its displayable text piece.
    ///
    /// The returned string is the human-visible piece, with special-token
    /// rendering enabled (`special = true` in the underlying
    /// `llama_token_to_piece`). Bytes that aren't valid UTF-8 on their own
    /// (multi-byte tokens, partial code points) are mapped through
    /// [`String::from_utf8_lossy`].
    ///
    /// Returns `Err(())` if llama.cpp reports a negative byte count (invalid
    /// token id). The internal buffer is 64 bytes — long enough for every
    /// piece any current tokenizer emits, but a hard cap nonetheless.
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

    /// Tokenize `text` against this model's vocabulary.
    ///
    /// - `add_special`: prepend/append the tokenizer's configured special
    ///   tokens (BOS / EOS / separator). Equivalent to upstream
    ///   `llama_tokenize`'s `add_special` flag — see
    ///   [`get_add_bos`](Self::get_add_bos) and
    ///   [`get_add_eos`](Self::get_add_eos) for what the tokenizer would add
    ///   when this is `true`.
    /// - `parse_special`: when `true`, recognize special-token literals in
    ///   `text` (e.g. `"<|endoftext|>"`) and emit them as their special token
    ///   ids; when `false`, treat them as ordinary text.
    ///
    /// Sizes the output by calling `llama_tokenize` twice: first with a null
    /// buffer to learn the required length, then with the real buffer.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<llama_token> {
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
