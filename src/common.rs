use std::ptr::null;

use llama_sys::*;

#[derive(Debug)]
pub enum BatchAddError {
    SizeExceeded,
    /// `batch_add` was called on a batch with no token buffer.
    ///
    /// `llama_batch_init` only allocates `batch.token` when `embd == 0`
    /// (i.e. `Batch::init_token`). Batches built with `Batch::init_embd`
    /// leave `batch.token` null, and writing through it would be UB.
    NotTokenBatch,
}

pub fn batch_add(
    batch: &mut llama_batch,
    id: llama_token,
    pos: llama_pos,
    seq_ids: &[llama_seq_id],
    logits: bool,
) -> Result<(), BatchAddError> {
    // `batch.token` is null for embedding batches (`Batch::init_embd`).
    // Writing `id` through that null pointer is undefined behaviour and is
    // reachable from purely safe Rust, so reject it before dereferencing.
    if batch.token.is_null() {
        return Err(BatchAddError::NotTokenBatch);
    }
    let seq_ids_ptr = unsafe { *batch.seq_id.add(batch.n_tokens as usize) };
    if seq_ids_ptr.is_null() {
        return Err(BatchAddError::SizeExceeded);
    }
    unsafe {
        // todo: get rid of this pointer arithmetic after wrapping the batches
        *batch.token.add(batch.n_tokens as usize) = id;
        *batch.pos.add(batch.n_tokens as usize) = pos;
        *batch.n_seq_id.add(batch.n_tokens as usize) = seq_ids.len() as i32;
    }
    for (i, seq_id) in seq_ids.iter().enumerate() {
        unsafe {
            *(*batch.seq_id.add(batch.n_tokens as usize)).add(i) = *seq_id;
        }
    }
    unsafe {
        *batch.logits.add(batch.n_tokens as usize) = logits as i8;
    }
    batch.n_tokens += 1;
    Ok(())
}

pub fn batch_clear(batch: &mut llama_batch) {
    batch.n_tokens = 0;
}

