//! Vocabulary inspection.
//!
//! Methods for querying a [`Model`]'s vocabulary: special token ids, per-token
//! metadata (text, score, attributes), and tokenizer flags. These all wrap the
//! `llama_vocab_*` family in the upstream C API and take no ownership — the
//! underlying `llama_vocab` lives as long as the [`Model`] it came from.
//!
//! ## Special-token accessors
//!
//! The `*_token` methods (e.g. [`Model::bos_token`], [`Model::eos_token`],
//! [`Model::fim_pre_token`]) return [`Option<llama_token>`]. Upstream signals
//! "this vocabulary doesn't define this token" by returning `LLAMA_TOKEN_NULL`,
//! and the wrappers translate that sentinel into [`None`]. Always handle the
//! [`None`] case — many models legitimately lack a PAD or MASK token, and the
//! fill-in-middle (`fim_*`) tokens only exist for code models trained with FIM.
//!
//! ## Token classification
//!
//! Two predicates classify token ids:
//!
//! - [`Model::is_eog`] — "end of generation" — covers EOS, EOT, and any other
//!   token the model treats as a stop signal. This is what generation loops
//!   should check, not just `token == eos_token`.
//! - [`Model::is_control`] — distinguishes control/special tokens (BOS, EOS,
//!   chat markers, etc.) from renderable content tokens.
//!
//! [`Model`]: crate::Model

use crate::*;
use llama_sys::*;
use std::ffi::CStr;

/// Defines a special-token accessor that wraps an `llama_vocab_*` getter and
/// maps the upstream `LLAMA_TOKEN_NULL` sentinel to [`None`].
macro_rules! token_option {
    ($name:ident, $ffi_fn:ident) => {
        #[inline]
        pub fn $name(&self) -> Option<llama_token> {
            let token = unsafe { $ffi_fn(self.vocab) };
            if token == LLAMA_TOKEN_NULL {
                None
            } else {
                Some(token)
            }
        }
    };
}

impl Model {
    /// Beginning-of-sentence token, or [`None`] if the vocabulary has none.
    token_option!(bos_token, llama_vocab_bos);

    /// Classification token. **Deprecated upstream** — `CLS` is equivalent to
    /// `BOS`; prefer [`Model::bos_token`].
    token_option!(cls_token, llama_vocab_cls);

    /// End-of-sentence token, or [`None`] if the vocabulary has none.
    ///
    /// Generation loops should usually check [`Model::is_eog`] instead, which
    /// also catches EOT and other model-specific stop tokens.
    token_option!(eos_token, llama_vocab_eos);

    /// End-of-turn token (used by chat-tuned models to delimit turns), or
    /// [`None`] if the vocabulary has none.
    token_option!(eot_token, llama_vocab_eot);

    /// Fill-in-middle "middle" marker. Only present on code models trained
    /// with FIM (e.g. CodeLlama, DeepSeek-Coder); [`None`] otherwise.
    token_option!(fim_mid_token, llama_vocab_fim_mid);

    /// Fill-in-middle padding marker, or [`None`] if the vocabulary has none.
    token_option!(fim_pad_token, llama_vocab_fim_pad);

    /// Fill-in-middle "prefix" marker. Only present on FIM-trained code
    /// models; [`None`] otherwise.
    token_option!(fim_pre_token, llama_vocab_fim_pre);

    /// Fill-in-middle repository/repeat marker, or [`None`] if the vocabulary
    /// has none.
    token_option!(fim_rep_token, llama_vocab_fim_rep);

    /// Fill-in-middle separator marker, or [`None`] if the vocabulary has
    /// none.
    token_option!(fim_sep_token, llama_vocab_fim_sep);

    /// Fill-in-middle "suffix" marker. Only present on FIM-trained code
    /// models; [`None`] otherwise.
    token_option!(fim_suf_token, llama_vocab_fim_suf);

    /// Whether the tokenizer is configured to prepend a BOS token when
    /// tokenizing.
    #[inline]
    pub fn get_add_bos(&self) -> bool {
        unsafe { llama_vocab_get_add_bos(self.vocab) }
    }

    /// Whether the tokenizer is configured to append an EOS token when
    /// tokenizing.
    #[inline]
    pub fn get_add_eos(&self) -> bool {
        unsafe { llama_vocab_get_add_eos(self.vocab) }
    }

    /// Whether the tokenizer is configured to insert a separator token when
    /// tokenizing.
    #[inline]
    pub fn get_add_sep(&self) -> bool {
        unsafe { llama_vocab_get_add_sep(self.vocab) }
    }

    /// Returns the attribute bit-flags for `token` (`UNKNOWN`, `NORMAL`,
    /// `CONTROL`, `USER_DEFINED`, `BYTE`, `NORMALIZED`, `LSTRIP`, `RSTRIP`,
    /// `SINGLE_WORD`, …).
    #[inline]
    pub fn get_attr(&self, token: llama_token) -> llama_token_attr {
        unsafe { llama_vocab_get_attr(self.vocab, token) }
    }

    /// Returns the tokenizer-assigned score for `token` (used by SentencePiece
    /// / Unigram models; `0.0` for tokenizers that don't track scores).
    #[inline]
    pub fn get_score(&self, token: llama_token) -> f32 {
        unsafe { llama_vocab_get_score(self.vocab, token) }
    }

    /// Returns the raw text piece for `token`, as stored by the tokenizer.
    ///
    /// The returned [`CStr`] borrows from the underlying `llama_vocab` and is
    /// valid for as long as `self`. For SentencePiece/BPE tokenizers the
    /// piece may contain marker characters (e.g. `▁` for word boundaries) and
    /// may not be valid UTF-8 on its own — joining multiple pieces and
    /// post-processing is what produces user-visible text.
    #[inline]
    pub fn get_text(&self, token: llama_token) -> &CStr {
        unsafe {
            let ptr = llama_vocab_get_text(self.vocab, token);
            CStr::from_ptr(ptr)
        }
    }

    /// Whether `token` is a control/special token rather than renderable
    /// content (BOS, EOS, chat markers, etc.).
    #[inline]
    pub fn is_control(&self, token: llama_token) -> bool {
        unsafe { llama_vocab_is_control(self.vocab, token) }
    }

    /// Whether `token` should terminate generation.
    ///
    /// Returns `true` for any "end of generation" token the model recognizes
    /// (EOS, EOT, and other model-specific stop tokens). Prefer this over
    /// comparing against a single special token id when implementing a
    /// generation loop.
    #[inline]
    pub fn is_eog(&self, token: llama_token) -> bool {
        unsafe { llama_vocab_is_eog(self.vocab, token) }
    }

    /// Mask token (used by BERT-style masked-language models), or [`None`] if
    /// the vocabulary has none.
    token_option!(mask_token, llama_vocab_mask);

    /// Total number of tokens in the vocabulary.
    #[inline]
    pub fn n_tokens(&self) -> i32 {
        unsafe { llama_vocab_n_tokens(self.vocab) }
    }

    /// Newline token, or [`None`] if the vocabulary has none.
    token_option!(nl_token, llama_vocab_nl);

    /// Padding token, or [`None`] if the vocabulary has none.
    token_option!(pad_token, llama_vocab_pad);

    /// Sentence-separator token, or [`None`] if the vocabulary has none.
    token_option!(sep_token, llama_vocab_sep);

    /// Tokenizer family for this vocabulary (e.g. `SPM`, `BPE`, `WPM`, `UGM`,
    /// `RWKV`, `PLAMO2`, or `NONE` for models without a vocabulary).
    #[inline]
    pub fn vocab_type(&self) -> llama_vocab_type {
        unsafe { llama_vocab_type(self.vocab) }
    }
}
