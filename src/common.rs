use std::ptr::null;

use llama_sys::*;

use crate::Batch;

#[derive(Debug)]
pub enum BatchAddError {
    SizeExceeded,
    /// `seq_ids.len()` exceeded the per-slot `n_seq_max` the batch was allocated with.
    /// Writing past that allocation is a heap buffer overflow, so the add is rejected.
    TooManySeqIds,
}

pub fn batch_add(
    batch: &mut Batch,
    id: llama_token,
    pos: llama_pos,
    seq_ids: &[llama_seq_id],
    logits: bool,
) -> Result<(), BatchAddError> {
    // Per-token `seq_id[i]` sub-array is `malloc`d for exactly `n_seq_max` entries
    // (see `llama_batch_init` in `llama-sys/llama.cpp/src/llama-batch.cpp`).
    // Writing `seq_ids.len()` entries past that bound overflows the heap.
    if seq_ids.len() > batch.n_seq_max() as usize {
        return Err(BatchAddError::TooManySeqIds);
    }
    let raw = batch.as_raw_mut();
    let seq_ids_ptr = unsafe { *raw.seq_id.add(raw.n_tokens as usize) };
    if seq_ids_ptr.is_null() {
        return Err(BatchAddError::SizeExceeded);
    }
    unsafe {
        // todo: get rid of this pointer arithmetic after wrapping the batches
        *raw.token.add(raw.n_tokens as usize) = id;
        *raw.pos.add(raw.n_tokens as usize) = pos;
        *raw.n_seq_id.add(raw.n_tokens as usize) = seq_ids.len() as i32;
    }
    for (i, seq_id) in seq_ids.iter().enumerate() {
        unsafe {
            *(*raw.seq_id.add(raw.n_tokens as usize)).add(i) = *seq_id;
        }
    }
    unsafe {
        *raw.logits.add(raw.n_tokens as usize) = logits as i8;
    }
    raw.n_tokens += 1;
    Ok(())
}

pub fn batch_clear(batch: &mut Batch) {
    batch.as_raw_mut().n_tokens = 0;
}
