use std::{
    num::NonZeroI32,
    ops::Deref,
};

use llama_sys::*;

/// Owning wrapper around `llama_batch`.
///
/// `llama_batch` is a plain C struct whose fields are raw pointers into
/// `malloc`-allocated buffers (see `llama_batch_init` in
/// `llama-sys/llama.cpp/src/llama-batch.cpp`). `Drop` calls
/// `llama_batch_free`, which `free()`s each non-null pointer field.
///
/// The inner `llama_batch` is not exposed mutably: replacing those pointers
/// from safe code (via field assignment, `DerefMut`, or a `&mut` to the inner
/// struct) would let callers feed null/dangling/non-`malloc` pointers to
/// `llama_batch_free` or to `batch_add`'s raw-pointer writes, both of which
/// are undefined behaviour. Read-only access via `Deref` is still provided so
/// callers can inspect counts and pass the batch (by `Copy`) into FFI calls
/// like `encode`/`decode`.
#[repr(transparent)]
pub struct Batch(pub(crate) llama_batch);

impl Batch {
    pub fn init_token(n_tokens: i32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, 0, n_seq_max) })
    }

    pub fn init_embd(n_tokens: i32, n_embd: NonZeroI32, n_seq_max: i32) -> Self {
        Batch(unsafe { llama_batch_init(n_tokens, n_embd.get(), n_seq_max) })
    }

    pub fn as_raw(&self) -> &llama_batch {
        &self.0
    }

    pub(crate) fn as_raw_mut(&mut self) -> &mut llama_batch {
        &mut self.0
    }
}

impl Deref for Batch {
    type Target = llama_batch;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.0);
        }
    }
}
