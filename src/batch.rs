use std::{
    num::NonZeroI32,
    ops::{Deref, DerefMut},
};

use llama_sys::*;

pub struct Batch(pub llama_batch, i32);

impl Batch {
    pub fn init_token(n_tokens: i32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, 0, n_seq_max) }, n_seq_max)
    }

    pub fn init_embd(n_tokens: i32, n_embd: NonZeroI32, n_seq_max: i32) -> Self {
        Batch(
            unsafe { llama_batch_init(n_tokens, n_embd.get(), n_seq_max) },
            n_seq_max,
        )
    }

    pub fn as_raw(&self) -> &llama_batch {
        &self.0
    }

    pub fn as_raw_mut(&mut self) -> &mut llama_batch {
        &mut self.0
    }

    /// Maximum number of seq_ids per token slot, as passed to the constructor.
    ///
    /// `llama_batch_init` allocates a `seq_id` sub-array of exactly this size
    /// for every token slot, so `batch_add` rejects `seq_ids` slices longer
    /// than this to keep writes in-bounds.
    pub fn n_seq_max(&self) -> i32 {
        self.1
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
