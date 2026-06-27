//! Vocabulary accessors on [`Model`].
//!
//! These methods wrap llama.cpp's `llama_vocab_*` family and report what the
//! model's tokenizer knows about itself: which special tokens it exposes, how
//! many entries the vocabulary has, what raw text and score each id carries,
//! and which tokenizer family produced it.
//!
//! Although the impl block lives here for housekeeping reasons, every method
//! is hung off [`Model`] so callers see them as `model.bos_token()` rather
//! than going through a separate vocab handle.
//!
//! # Optional vs. required tokens
//!
//! Special-token accessors come in two flavours:
//!
//! - Mandatory tokens such as [`Model::n_tokens`] and [`Model::vocab_type`]
//!   return their value directly.
//! - Optional tokens — BOS, EOS, padding, fill-in-the-middle markers, etc. —
//!   return `Option<i32>`. llama.cpp signals "this model has no such token"
//!   by returning `LLAMA_TOKEN_NULL`; that sentinel is translated to `None`
//!   here so callers never see it leak into Rust code.
//!
//! Whether a given optional token is present is entirely a property of the
//! loaded model. A chat-tuned model usually exposes an end-of-turn token via
//! [`Model::eot_token`]; a base-only model may not. Code that relies on a
//! specific token should match on the `Option` rather than `unwrap` it.
//!
//! # Token id validity
//!
//! Per-token accessors ([`Model::get_attr`], [`Model::get_score`],
//! [`Model::get_text`], [`Model::is_control`], [`Model::is_eog`]) currently
//! forward the id to llama.cpp without bounds-checking. Pass only ids in
//! `0..Model::n_tokens()` — for example values returned by
//! [`Model::tokenize`], the special-token accessors above, or your own loop
//! over the vocabulary. Out-of-range ids are undefined behaviour in the
//! underlying C API.

use crate::*;
use llama_sys::*;
use std::ffi::CStr;

macro_rules! token_option {
    ($(#[$attr:meta])* $name:ident, $ffi_fn:ident) => {
        $(#[$attr])*
        #[inline]
        pub fn $name(&self) -> Option<i32> {
            let token = unsafe { $ffi_fn(self.vocab_ptr()) };
            if token == LLAMA_TOKEN_NULL {
                None
            } else {
                Some(token)
            }
        }
    };
}

impl Model {
    token_option!(
        /// Beginning-of-sequence (BOS) token id, or `None` if the model has
        /// no BOS.
        ///
        /// Most causal language models start every prompt with this token;
        /// see also [`Model::get_add_bos`] to check whether the tokenizer
        /// wants it prepended automatically by [`Model::tokenize`].
        bos_token, llama_vocab_bos
    );

    token_option!(
        /// Classifier (`[CLS]`) token id used by BERT-style encoders, or
        /// `None` if the model has none.
        cls_token, llama_vocab_cls
    );

    token_option!(
        /// End-of-sequence (EOS) token id, or `None` if the model has no
        /// EOS.
        ///
        /// Generation typically stops on this token; for chat-tuned models
        /// the finer-grained [`Model::eot_token`] may be more appropriate.
        eos_token, llama_vocab_eos
    );

    token_option!(
        /// End-of-turn token id used by chat-tuned models, or `None`.
        ///
        /// Distinct from EOS: end-of-turn marks the boundary between
        /// speaker turns in a conversation, while EOS marks the end of the
        /// entire sequence.
        eot_token, llama_vocab_eot
    );

    token_option!(
        /// Fill-in-the-middle "middle" marker token, or `None`.
        ///
        /// Used by code models that support FIM-style infilling, where the
        /// model is asked to predict text between a prefix and a suffix.
        fim_mid_token, llama_vocab_fim_mid
    );

    token_option!(
        /// Fill-in-the-middle padding token, or `None`.
        fim_pad_token, llama_vocab_fim_pad
    );

    token_option!(
        /// Fill-in-the-middle "prefix" marker token, or `None`.
        fim_pre_token, llama_vocab_fim_pre
    );

    token_option!(
        /// Fill-in-the-middle "repository" marker token, or `None`.
        ///
        /// Used by repo-aware code models to delimit cross-file context.
        fim_rep_token, llama_vocab_fim_rep
    );

    token_option!(
        /// Fill-in-the-middle separator token, or `None`.
        fim_sep_token, llama_vocab_fim_sep
    );

    token_option!(
        /// Fill-in-the-middle "suffix" marker token, or `None`.
        fim_suf_token, llama_vocab_fim_suf
    );

    /// Whether [`Model::tokenize`] should prepend [`Model::bos_token`] when
    /// `add_special = true`.
    ///
    /// Reflects the tokenizer's own preference recorded in the GGUF
    /// metadata. Most causal models return `true`; encoder-style models
    /// often return `false`.
    #[inline]
    pub fn get_add_bos(&self) -> bool {
        unsafe { llama_vocab_get_add_bos(self.vocab_ptr()) }
    }

    /// Whether [`Model::tokenize`] should append [`Model::eos_token`] when
    /// `add_special = true`.
    #[inline]
    pub fn get_add_eos(&self) -> bool {
        unsafe { llama_vocab_get_add_eos(self.vocab_ptr()) }
    }

    /// Whether [`Model::tokenize`] should insert [`Model::sep_token`]
    /// between segments when `add_special = true`.
    ///
    /// Relevant mainly to sentence-pair encoders such as BERT.
    #[inline]
    pub fn get_add_sep(&self) -> bool {
        unsafe { llama_vocab_get_add_sep(self.vocab_ptr()) }
    }

    /// Attribute bitset for `token` (normal, unknown, control, byte,
    /// user-defined, …).
    ///
    /// Returns the raw [`llama_token_attr`] reported by llama.cpp. Use the
    /// flag constants from `llama_sys` to interpret it — for example,
    /// testing the control flag is equivalent to [`Model::is_control`].
    ///
    /// `token` must be a valid id in `0..Model::n_tokens()`; passing an
    /// out-of-range id is undefined behaviour in the underlying C API.
    #[inline]
    pub fn get_attr(&self, token: i32) -> llama_token_attr {
        unsafe { llama_vocab_get_attr(self.vocab_ptr(), token) }
    }

    /// Tokenizer-assigned score for `token`.
    ///
    /// Meaningful for SentencePiece-style vocabularies (used as the
    /// log-probability that biases merges); other tokenizer families may
    /// report `0.0` or another placeholder.
    ///
    /// `token` must be a valid id in `0..Model::n_tokens()`.
    #[inline]
    pub fn get_score(&self, token: i32) -> f32 {
        unsafe { llama_vocab_get_score(self.vocab_ptr(), token) }
    }

    /// Raw vocabulary text for `token` as it is stored in the GGUF file.
    ///
    /// The returned [`CStr`] is borrowed from the model and is valid for
    /// the lifetime of `&self`. The bytes are the *internal* representation
    /// (e.g. a SentencePiece token may include a leading `▁` for
    /// whitespace) — prefer [`Model::token_to_piece`] when rendering output
    /// to a user.
    ///
    /// `token` must be a valid id in `0..Model::n_tokens()`.
    #[inline]
    pub fn get_text(&self, token: i32) -> &CStr {
        unsafe {
            let ptr = llama_vocab_get_text(self.vocab_ptr(), token);
            CStr::from_ptr(ptr)
        }
    }

    /// `true` if `token` is a control token (BOS, EOS, separators, chat
    /// role markers, …) rather than a regular vocabulary entry.
    ///
    /// Equivalent to testing the control bit on [`Model::get_attr`].
    /// `token` must be a valid id in `0..Model::n_tokens()`.
    #[inline]
    pub fn is_control(&self, token: i32) -> bool {
        unsafe { llama_vocab_is_control(self.vocab_ptr(), token) }
    }

    /// `true` if generation should stop after producing `token`.
    ///
    /// Covers EOS, end-of-turn and any other model-specific stop tokens
    /// flagged in the GGUF metadata — generation loops should check this
    /// rather than only comparing against [`Model::eos_token`], otherwise
    /// chat-tuned models that emit end-of-turn instead of EOS will run on.
    ///
    /// `token` must be a valid id in `0..Model::n_tokens()`.
    #[inline]
    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { llama_vocab_is_eog(self.vocab_ptr(), token) }
    }

    token_option!(
        /// Mask token id used by masked-language-modelling encoders, or
        /// `None`.
        mask_token, llama_vocab_mask
    );

    /// Number of entries in the vocabulary.
    ///
    /// Every valid token id falls in `0..n_tokens()`. This is also the
    /// length of a logits slice produced by [`crate::Sequence::logits`].
    #[inline]
    pub fn n_tokens(&self) -> i32 {
        unsafe { llama_vocab_n_tokens(self.vocab_ptr()) }
    }

    token_option!(
        /// Newline token id, or `None` if the tokenizer represents newlines
        /// only as part of larger pieces.
        nl_token, llama_vocab_nl
    );

    token_option!(
        /// Padding token id, or `None` if the model has none.
        pad_token, llama_vocab_pad
    );

    token_option!(
        /// Separator (`[SEP]`) token id used by sentence-pair encoders, or
        /// `None`.
        sep_token, llama_vocab_sep
    );

    /// The tokenizer family that produced this vocabulary (SPM, BPE,
    /// WordPiece, …).
    ///
    /// See [`llama_vocab_type`] for the full enum. `LLAMA_VOCAB_TYPE_NONE`
    /// indicates a model without a tokenizer (e.g. raw embedding models).
    #[inline]
    pub fn vocab_type(&self) -> llama_vocab_type {
        unsafe { llama_vocab_type(self.vocab_ptr()) }
    }
}
