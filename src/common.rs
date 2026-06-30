use llama_sys::*;

#[derive(Debug)]
pub enum BatchAddError {
    SizeExceeded,
}

pub fn batch_add(
    batch: &mut llama_batch,
    id: llama_token,
    pos: llama_pos,
    seq_ids: &[llama_seq_id],
    logits: bool,
) -> Result<(), BatchAddError> {
    // llama.cpp's `llama_batch_init` does no error reporting: if any of its
    // internal mallocs fails (OOM), the returned `llama_batch` keeps the
    // un-allocated field set to `nullptr` and the caller is expected to
    // notice. The existing `seq_id`-sentinel check below would itself
    // dereference a null pointer when `batch.seq_id` is null, and the
    // unconditional writes through `batch.token` / `pos` / `n_seq_id` /
    // `logits` would dereference null on the same path. Reject any batch
    // with a null buffer up front so a partially-allocated batch surfaces
    // as a clean `SizeExceeded` instead of FFI UB reachable from safe Rust
    // via `Sequence::push`.
    if batch.seq_id.is_null()
        || batch.token.is_null()
        || batch.pos.is_null()
        || batch.n_seq_id.is_null()
        || batch.logits.is_null()
    {
        return Err(BatchAddError::SizeExceeded);
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

#[cfg(test)]
mod tests {
    use super::{batch_add, BatchAddError};
    use llama_sys::{llama_batch, llama_seq_id};

    /// Simulate the OOM-failure mode of `llama_batch_init`: the returned
    /// `llama_batch` has one or more null internal buffers. Without the
    /// up-front null check in `batch_add`, the very first deref
    /// (`*batch.seq_id.add(0)`) would be UB. With the check, we get a
    /// clean `SizeExceeded` error and the host stays sound.
    #[test]
    fn batch_add_rejects_fully_null_batch() {
        // SAFETY: `llama_batch` is a bindgen-generated POD whose fields are
        // raw pointers and integers; the all-zero bit pattern is a valid
        // representation (null pointers, zero counts).
        let mut batch: llama_batch = unsafe { std::mem::zeroed() };
        let seq_ids = [0i32];
        let result = batch_add(&mut batch, 0, 0, &seq_ids, true);
        assert!(
            matches!(result, Err(BatchAddError::SizeExceeded)),
            "batch_add must reject a batch with null buffers instead of \
             dereferencing them"
        );
    }

    /// Pin that *any* null buffer is enough to make `batch_add` bail —
    /// `llama_batch_init` can fail one alloc while others succeed.
    #[test]
    fn batch_add_rejects_partially_null_batch() {
        // SAFETY: same as above; we only need a controlled `llama_batch`
        // value, the C side is never touched on the rejection path.
        let mut batch: llama_batch = unsafe { std::mem::zeroed() };
        // Pretend the seq_id outer alloc succeeded but token did not.
        let mut slots: [*mut llama_seq_id; 2] =
            [std::ptr::null_mut(), std::ptr::null_mut()];
        batch.seq_id = slots.as_mut_ptr();
        // batch.token left null — must still be rejected.
        let seq_ids = [0i32];
        let result = batch_add(&mut batch, 0, 0, &seq_ids, true);
        assert!(
            matches!(result, Err(BatchAddError::SizeExceeded)),
            "batch_add must reject a batch with any null buffer"
        );
    }
}
