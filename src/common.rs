//! Helpers for filling [`Batch`](crate::Batch)es.
//!
//! [`batch_add`] appends one entry to a batch and [`batch_clear`] empties it
//! for reuse. Both operate on the raw `llama_batch` behind a
//! [`Batch`](crate::Batch), which is reachable as `&mut *batch`.

use std::ptr::null;

use llama_sys::*;

/// Error returned by [`batch_add`] when an entry cannot be appended.
#[derive(Debug)]
pub enum BatchAddError {
    /// The batch is already at the capacity it was created with.
    SizeExceeded,
}

/// Appends a single token entry to a batch.
///
/// - `id` — the token id to add.
/// - `pos` — the position of this token within its sequence.
/// - `seq_ids` — the sequence id(s) this token belongs to.
/// - `logits` — whether the model should produce output (logits) for this
///   position. Set this to `true` only for positions you intend to sample
///   from; requesting logits for every position is wasteful.
///
/// Returns [`BatchAddError::SizeExceeded`] if the batch is already full. That
/// check works because [`Batch::init_token`](crate::Batch::init_token)
/// allocates one extra sequence-id slot and leaves it null as a sentinel;
/// once the batch reaches capacity, that null slot is what `batch_add` finds
/// at index `n_tokens`.
///
/// This writes a *token* entry, so the batch must be one created with
/// [`Batch::init_token`](crate::Batch::init_token) — embedding batches have
/// no `token` buffer.
///
/// The caller must ensure `seq_ids` is no longer than the `n_seq_max` the
/// batch was created with: the per-entry sequence-id storage is sized to
/// `n_seq_max`, and a longer slice would write out of bounds.
pub fn batch_add(
    batch: &mut llama_batch,
    id: llama_token,
    pos: llama_pos,
    seq_ids: &[llama_seq_id],
    logits: bool,
) -> Result<(), BatchAddError> {
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

/// Resets a batch to empty so it can be refilled.
///
/// Sets the entry count back to zero. It does not free or reallocate the
/// underlying buffers, so the batch keeps the capacity it was created with.
pub fn batch_clear(batch: &mut llama_batch) {
    batch.n_tokens = 0;
}

