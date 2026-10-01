//! RAII wrapper around `llama_batch`, the input buffer for `llama_decode`.

use std::ops::{Deref, DerefMut};

use llama_sys::*;

use crate::Error;

/// A batch of tokens waiting to be decoded.
///
/// Wraps `llama_batch` and owns the buffers allocated by `llama_batch_init`,
/// freeing them on drop. Use [`Batch::clear`] and [`Batch::add`] to fill it;
/// everything else goes through [`Deref`] to the raw `llama_batch`.
pub struct Batch(llama_batch);

impl Batch {
    /// Allocate a batch that can hold `n_tokens` tokens, each belonging to up
    /// to `n_seq_max` sequences.
    pub(crate) fn init_token(n_tokens: i32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, 0, n_seq_max) })
    }

    /// Drop all queued tokens, keeping the allocated buffers.
    pub(crate) fn clear(&mut self) {
        self.0.n_tokens = 0;
    }

    /// Append one token at position `pos`, belonging to `seq_ids`.
    ///
    /// When `logits` is set, the logits for this token are written to the
    /// context output and can be read back with `llama_get_logits_ith`.
    pub(crate) fn add(
        &mut self,
        id: llama_token,
        pos: llama_pos,
        seq_ids: &[llama_seq_id],
        logits: bool,
    ) -> Result<(), Error> {
        let slot = self.0.n_tokens as usize;
        let seq_ids_ptr = unsafe { *self.0.seq_id.add(slot) };
        if seq_ids_ptr.is_null() {
            return Err(Error::BatchFull);
        }
        unsafe {
            // TODO: get rid of this pointer arithmetic after wrapping the batches
            *self.0.token.add(slot) = id;
            *self.0.pos.add(slot) = pos;
            *self.0.n_seq_id.add(slot) = seq_ids.len() as i32;
            *self.0.logits.add(slot) = logits as i8;
        }
        for (i, seq_id) in seq_ids.iter().enumerate() {
            unsafe {
                *(*self.0.seq_id.add(slot)).add(i) = *seq_id;
            }
        }
        self.0.n_tokens += 1;
        Ok(())
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
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.0);
        }
    }
}
