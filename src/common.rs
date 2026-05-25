use std::ptr::null;

use llama_sys::*;

use crate::Batch;

#[derive(Debug)]
pub enum BatchAddError {
    SizeExceeded,
}

pub fn batch_add(
    batch: &mut Batch,
    id: llama_token,
    pos: llama_pos,
    seq_ids: &[llama_seq_id],
    logits: bool,
) -> Result<(), BatchAddError> {
    let inner = batch.as_raw_mut();
    let seq_ids_ptr = unsafe { *inner.seq_id.add(inner.n_tokens as usize) };
    if seq_ids_ptr.is_null() {
        return Err(BatchAddError::SizeExceeded);
    }
    unsafe {
        // todo: get rid of this pointer arithmetic after wrapping the batches
        *inner.token.add(inner.n_tokens as usize) = id;
        *inner.pos.add(inner.n_tokens as usize) = pos;
        *inner.n_seq_id.add(inner.n_tokens as usize) = seq_ids.len() as i32;
    }
    for (i, seq_id) in seq_ids.iter().enumerate() {
        unsafe {
            *(*inner.seq_id.add(inner.n_tokens as usize)).add(i) = *seq_id;
        }
    }
    unsafe {
        *inner.logits.add(inner.n_tokens as usize) = logits as i8;
    }
    inner.n_tokens += 1;
    Ok(())
}

pub fn batch_clear(batch: &mut Batch) {
    batch.as_raw_mut().n_tokens = 0;
}
