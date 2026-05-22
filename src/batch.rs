//! Input batches for [`Context`](crate::Context) evaluation.
//!
//! A [`Batch`] is the unit of input handed to
//! [`Context::encode`](crate::Context::encode) and
//! [`Context::decode`](crate::Context::decode). It is a fixed-capacity,
//! pre-allocated buffer: the capacity is chosen at construction, entries are
//! appended with [`batch_add`](crate::common::batch_add), and the batch is
//! reset for reuse with [`batch_clear`](crate::common::batch_clear).

use std::{
    num::NonZeroI32,
    ops::{Deref, DerefMut},
};

use llama_sys::*;

/// An owned, fixed-capacity input batch for [`Context`](crate::Context) evaluation.
///
/// Wraps a raw `llama_batch` and frees it on drop. A batch holds a fixed
/// maximum number of entries — chosen when it is created — and starts empty.
///
/// The wrapped struct is reachable through [`Batch::as_raw`] /
/// [`Batch::as_raw_mut`] and directly via `Deref` / `DerefMut`. Note that its
/// `n_tokens` field is the number of entries *currently* in the batch, which
/// is distinct from the `n_tokens` *capacity* argument passed to
/// [`Batch::init_token`] and [`Batch::init_embd`].
#[repr(transparent)]
pub struct Batch(pub llama_batch);

impl Batch {
    /// Allocates a token batch that can hold up to `n_tokens` entries.
    ///
    /// Each entry is a single token id. `n_seq_max` is the maximum number of
    /// sequences any one entry may be assigned to. The returned batch is
    /// empty (its `n_tokens` field is `0`) and ready to be filled with
    /// [`batch_add`](crate::common::batch_add).
    pub fn init_token(n_tokens: i32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, 0, n_seq_max) })
    }

    /// Allocates an embedding batch that can hold up to `n_tokens` entries.
    ///
    /// Like [`Batch::init_token`], but each entry carries an
    /// `n_embd`-dimensional embedding vector instead of a token id. `n_embd`
    /// is a [`NonZeroI32`] because a zero embedding width is what the
    /// underlying C API uses to mean "token batch" — an embedding batch must
    /// have a positive width.
    ///
    /// Note that [`batch_add`](crate::common::batch_add) writes token entries
    /// and is therefore only valid for batches created with
    /// [`Batch::init_token`]; embedding batches must be filled by writing the
    /// `embd` field directly.
    pub fn init_embd(n_tokens: i32, n_embd: NonZeroI32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, n_embd.get(), n_seq_max) })
    }

    /// Returns a shared reference to the wrapped `llama_batch`.
    pub fn as_raw(&self) -> &llama_batch {
        &self.0
    }

    /// Returns a mutable reference to the wrapped `llama_batch`.
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

/// Frees the buffers owned by the wrapped `llama_batch` via `llama_batch_free`.
impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.0);
        }
    }
}

