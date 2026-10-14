//! A safe-ish wrapper around `llama_batch`.
//!
//! A `llama_batch` is the "envelope" you hand to `llama_decode`. It is a set of
//! parallel C arrays, one slot per token:
//!
//! ```text
//!   index:    0      1      2     ...
//!   token:   [t0]   [t1]   [t2]         which token to feed in
//!   pos:     [p0]   [p1]   [p2]         its position in its sequence
//!   n_seq_id:[1]    [1]    [1]          how many sequences it belongs to
//!   seq_id:  [[s0]] [[s1]] [[s2]]       which sequence(s) it belongs to
//!   logits:  [0]    [0]    [1]          do we want logits computed for it?
//! ```
//!
//! llama.cpp allocates these arrays for us in `llama_batch_init` and we must
//! free them with `llama_batch_free`, which is what [`Drop`] does here.
//!
//! This type is internal to the crate. Users interact with [`crate::Sequence`]
//! instead, which fills batches on their behalf.

use llama_sys::*;

use crate::{Pos, SeqId, Token};

/// Returned by [`Batch::add`] when the batch has no room left.
#[derive(Debug)]
pub(crate) struct BatchFull;

/// Owns a `llama_batch` and frees it on drop.
pub(crate) struct Batch {
    raw: llama_batch,
    /// How many token slots were allocated. llama.cpp doesn't store this, so
    /// we remember it ourselves to avoid writing past the end of the arrays.
    capacity: usize,
    /// How many sequence ids each token slot can hold.
    n_seq_max: usize,
}

impl Batch {
    /// Allocate a batch that holds up to `capacity` tokens, where each token can
    /// belong to up to `n_seq_max` sequences at once.
    pub(crate) fn new(capacity: usize, n_seq_max: usize) -> Self {
        // The middle argument (`embd`) is 0 because we feed token ids, not raw
        // embedding vectors.
        let raw = unsafe { llama_batch_init(capacity as i32, 0, n_seq_max as i32) };
        Self {
            raw,
            capacity,
            n_seq_max,
        }
    }

    /// The most tokens this batch can hold.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of tokens currently in the batch.
    pub(crate) fn len(&self) -> usize {
        self.raw.n_tokens as usize
    }

    /// Forget all tokens so the batch can be reused. Does not free memory.
    pub(crate) fn clear(&mut self) {
        self.raw.n_tokens = 0;
    }

    /// Append one token to the batch.
    ///
    /// * `pos` — where the token sits in its sequence (0 for the first token).
    /// * `seq_ids` — which sequence(s) the token belongs to. Usually just one.
    /// * `want_logits` — whether llama.cpp should compute logits for this
    ///   token. You normally only want them for the *last* token of a prompt.
    pub(crate) fn add(
        &mut self,
        token: Token,
        pos: Pos,
        seq_ids: &[SeqId],
        want_logits: bool,
    ) -> Result<(), BatchFull> {
        let i = self.len();
        if i >= self.capacity || seq_ids.len() > self.n_seq_max {
            return Err(BatchFull);
        }

        // SAFETY: `i < capacity` and `seq_ids.len() <= n_seq_max`, and
        // `llama_batch_init` allocated every array with exactly those sizes,
        // so all of these writes stay in bounds.
        unsafe {
            *self.raw.token.add(i) = token;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = seq_ids.len() as i32;
            let slot_seq_ids = *self.raw.seq_id.add(i);
            for (j, &seq_id) in seq_ids.iter().enumerate() {
                *slot_seq_ids.add(j) = seq_id;
            }
            *self.raw.logits.add(i) = want_logits as i8;
        }

        self.raw.n_tokens += 1;
        Ok(())
    }

    /// The raw struct, for passing to `llama_decode`. `llama_batch` is just a
    /// bundle of pointers, so copying it out is cheap and doesn't transfer
    /// ownership — we still free it in [`Drop`].
    pub(crate) fn raw(&self) -> llama_batch {
        self.raw
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.raw) };
    }
}
