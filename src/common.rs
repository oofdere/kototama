//! Low-level helpers for populating a [`llama_batch`].
//!
//! [`crate::Batch`] wraps the batch's ownership, but the underlying
//! [`llama_batch`] is still a plain C struct exposing raw pointers to
//! parallel arrays. The two helpers here — [`batch_add`] and
//! [`batch_clear`] — do the pointer arithmetic required to push a new
//! entry into those arrays or to reset the live-entry counter without
//! reallocating.
//!
//! Both helpers are `pub(crate)`: the context actor is the only place
//! that constructs and mutates a batch, so exposing this API outside the
//! crate would let safe callers write past the batch's capacity or into
//! the wrong array. External code drives batches through
//! [`Sequence::push`](crate::Sequence::push) and the actor-mediated
//! decode path.
//!
//! [`batch_add`] currently supports **token-mode** batches only — it
//! writes through `batch.token`, which is null in embedding-mode batches
//! (see [`Batch::init_embd`](crate::Batch::init_embd)). Passing an
//! embedding batch here would dereference a null pointer, so all in-tree
//! callers construct token batches via
//! [`Batch::init_token`](crate::Batch::init_token).

use llama_sys::*;

/// Reasons [`batch_add`] can refuse to append an entry.
#[derive(Debug)]
pub enum BatchAddError {
    /// The batch has no free slot. Detected via the null sentinel that
    /// [`llama_batch_init`] leaves in `batch.seq_id[n_tokens_alloc]` — as
    /// soon as `batch.n_tokens` reaches capacity, the next slot's
    /// `seq_id` sub-array pointer reads as null. Callers should either
    /// call [`batch_clear`] and reuse the batch or allocate a larger one.
    SizeExceeded,
}

/// Append a single token entry to `batch` and bump `batch.n_tokens`.
///
/// Writes the parallel arrays in one pass:
///
/// - `batch.token[n_tokens] = id`
/// - `batch.pos[n_tokens]   = pos`
/// - `batch.n_seq_id[n_tokens] = seq_ids.len() as i32`
/// - `batch.seq_id[n_tokens][i] = seq_ids[i]` for each `i in 0..seq_ids.len()`
/// - `batch.logits[n_tokens] = logits as i8`
///
/// Preconditions the caller must uphold — none of them are re-checked
/// here, so violating any is undefined behaviour:
///
/// - `batch` must be a token-mode batch built by
///   [`Batch::init_token`](crate::Batch::init_token). Embedding-mode
///   batches leave `batch.token` null and would trigger a null write on
///   the first line.
/// - `seq_ids.len()` must not exceed the `n_seq_max` passed to the
///   constructor. Each per-slot `seq_id` sub-array is exactly
///   `n_seq_max` entries wide.
///
/// Returns [`BatchAddError::SizeExceeded`] when the batch is full. In
/// that case no field is written and `n_tokens` is not incremented, so
/// the caller can recover cleanly by clearing or growing the batch.
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

/// Reset `batch.n_tokens` to zero so the batch can be repopulated.
///
/// The underlying token/pos/seq_id/logits buffers are **not** freed — the
/// slots they hold are simply marked as unused, ready for the next
/// [`batch_add`] pass. This is the intended way to reuse a batch across
/// consecutive decode calls without reallocating.
pub fn batch_clear(batch: &mut llama_batch) {
    batch.n_tokens = 0;
}
