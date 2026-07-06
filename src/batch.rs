//! Owned RAII wrapper around llama.cpp's [`llama_batch`].
//!
//! A batch is the input container that [`llama_decode`] and [`llama_encode`]
//! consume in one FFI call. Under the hood it is a bag of raw pointers into
//! several parallel arrays — one entry per token slot — allocated by
//! [`llama_batch_init`] and freed by [`llama_batch_free`].
//!
//! [`Batch`] wraps that C struct so ownership is tied to Rust's scoping
//! rules: the underlying allocations are released on [`Drop`], never leaked
//! and never freed twice. The wrapper is `pub(crate)` because construction
//! and mutation are only sound at well-known call sites inside the context
//! actor (see [`crate::common`] for the low-level population helpers);
//! external callers drive batches indirectly through
//! [`Sequence::push`](crate::Sequence::push).
//!
//! Two constructor flavours are exposed, one per llama.cpp batch mode:
//!
//! - [`Batch::init_token`] — token-ID batches, the common case for
//!   causal language-model decoding.
//! - [`Batch::init_embd`] — embedding batches, where each slot holds a
//!   dense vector of `n_embd` floats instead of a token ID.
//!
//! The two paths are mutually exclusive: `llama_batch_init` only allocates
//! whichever of `batch.token` and `batch.embd` matches the mode requested,
//! leaving the other null. Callers must therefore keep track of which
//! constructor was used and only feed each batch to the matching
//! population helper.

use std::{
    num::NonZeroI32,
    ops::{Deref, DerefMut},
};

use llama_sys::*;

/// RAII wrapper over an owned [`llama_batch`].
///
/// The wrapped struct owns four parallel arrays (token/pos/seq_id/logits)
/// plus, in token mode, the `token` buffer and, in embedding mode, the
/// `embd` buffer. All of them are allocated by [`llama_batch_init`] and
/// released together by [`llama_batch_free`], which [`Drop`] calls.
///
/// The inner `n_tokens` field is the **live entry count** — how many
/// slots have been written by [`crate::common::batch_add`] since the last
/// [`crate::common::batch_clear`] — not the capacity passed at
/// construction. The capacity is fixed for the lifetime of the batch;
/// exceeding it surfaces as [`crate::common::BatchAddError::SizeExceeded`].
#[repr(transparent)]
pub struct Batch(pub llama_batch);

impl Batch {
    /// Allocate a token-mode batch with room for `n_tokens` token slots and
    /// up to `n_seq_max` sequence IDs per slot.
    ///
    /// `embd` is passed as `0` to [`llama_batch_init`], so `batch.token`
    /// is allocated and `batch.embd` is left null. Populate the batch with
    /// [`crate::common::batch_add`], which walks `batch.token`.
    pub fn init_token(n_tokens: i32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, 0, n_seq_max) })
    }

    /// Allocate an embedding-mode batch with room for `n_tokens` slots of
    /// `n_embd`-dimensional embeddings and up to `n_seq_max` sequence IDs
    /// per slot.
    ///
    /// `n_embd` is [`NonZeroI32`] because a zero-dimensional embedding
    /// batch is exactly a token batch — the constructor would forward `0`
    /// to `llama_batch_init` and produce something [`Batch::init_token`]
    /// already models, so the type system rules out that ambiguous case.
    pub fn init_embd(n_tokens: i32, n_embd: NonZeroI32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, n_embd.get(), n_seq_max) })
    }

    /// Borrow the wrapped [`llama_batch`] for read-only FFI use (e.g.
    /// passing to `llama_decode`).
    pub fn as_raw(&self) -> &llama_batch {
        &self.0
    }

    /// Borrow the wrapped [`llama_batch`] mutably, for helpers that write
    /// through its raw pointers (e.g. [`crate::common::batch_add`]).
    pub fn as_raw_mut(&mut self) -> &mut llama_batch {
        &mut self.0
    }
}

impl Deref for Batch {
    type Target = llama_batch;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Batch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for Batch {
    /// Release every allocation owned by the wrapped batch via
    /// [`llama_batch_free`].
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.0);
        }
    }
}
