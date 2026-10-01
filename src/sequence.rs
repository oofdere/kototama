//! Token sequences backed by a [`Context`].

use std::ops::{Index, Range};

use crate::{Context, Error, Sampler, Token};

/// One sequence of tokens inside a [`Context`], with its own KV-cache slot.
///
/// Obtain one with [`Context::sequence`]. The sequence owns its token list and
/// caches the logits from the last decode, so [`Sequence::logits`] and
/// [`Sequence::sample`] are plain local reads. Dropping the sequence frees its
/// sequence slot and removes its tokens from the context memory.
///
/// All methods are synchronous and safe to call from multiple threads (they
/// serialize on the context lock), but two handles to the *same* sequence
/// cannot exist at once: the sequence is not cloneable and every mutating
/// method takes `&mut self`.
pub struct Sequence {
    // Visible to the crate so the `asynchronous` facade can mirror these
    // methods without duplicating the state machine.
    pub(crate) ctx: Context,
    pub(crate) id: i32,
    pub(crate) tokens: Vec<i32>,
    pub(crate) logits: Option<Vec<f32>>,
}

impl Sequence {
    pub(crate) fn new(ctx: Context, id: i32) -> Self {
        Self {
            ctx,
            id,
            tokens: Vec::new(),
            logits: None,
        }
    }

    /// Logits produced by the last [`Sequence::push`] or [`Sequence::decode`],
    /// or `None` after operations that invalidate them ([`Sequence::pop`],
    /// [`Sequence::remove`], …).
    ///
    /// Use [`Sequence::decode`] to refresh them without pushing a new token.
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }

    /// `true` when the sequence holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Number of tokens in the sequence.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Decode `token` at the end of the sequence and cache its logits.
    ///
    /// Fails when the context cannot take the token (e.g. [`Error::ContextFull`]);
    /// on failure the sequence is left unchanged.
    pub fn push(&mut self, token: i32) -> Result<(), Error> {
        let pos = self.tokens.len() as i32;
        let logits = self.ctx.lock().decode_token(token, pos, self.id)?;
        self.logits = Some(logits);
        self.tokens.push(token);
        Ok(())
    }

    /// Decode several tokens in order. Stops at the first failure, leaving the
    /// sequence with the tokens decoded so far.
    pub fn extend(&mut self, tokens: &[i32]) -> Result<(), Error> {
        for &token in tokens {
            self.push(token)?;
        }
        Ok(())
    }

    /// Re-decode the last token to refresh the cached logits without pushing a
    /// new one. Useful after [`Sequence::pop`] or [`Sequence::remove`].
    ///
    /// The token's stale KV entry is replaced: llama.cpp requires consecutive
    /// positions and will not overwrite an existing one. Does nothing on an
    /// empty sequence.
    pub fn decode(&mut self) -> Result<(), Error> {
        if let Some(&last) = self.tokens.last() {
            let pos = (self.tokens.len() - 1) as i32;
            self.logits = Some(self.ctx.lock().refresh_token(last, pos, self.id)?);
        }
        Ok(())
    }

    /// Remove the last token, returning it. Invalidates the cached logits.
    ///
    /// Returns `None` on an empty sequence or when the KV-cache removal fails.
    pub fn pop(&mut self) -> Option<i32> {
        let len = self.tokens.len() as i32;
        if len == 0 {
            return None;
        }
        if self.kv_remove((len - 1)..len) {
            let token = self.tokens.pop();
            self.logits = None;
            token
        } else {
            None
        }
    }

    /// Token at `index`, or `None` when out of bounds.
    pub fn get(&self, index: usize) -> Option<i32> {
        self.tokens.get(index).copied()
    }

    /// All tokens, in order.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Remove the tokens in `range` (token-list indices). Invalidates the
    /// cached logits.
    ///
    /// Returns `false` — leaving the sequence unchanged — when the KV-cache
    /// removal fails.
    pub fn remove(&mut self, range: Range<usize>) -> bool {
        if self.ctx.lock().remove_token_range(self.id, &range) {
            self.tokens.drain(range);
            self.logits = None;
            true
        } else {
            false
        }
    }

    /// Copy this sequence's tokens in `range` (token-list indices) onto the
    /// end of `other`, replacing `other`'s tokens and logits cache with the
    /// copied range.
    ///
    /// Both sequences must belong to the same [`Context`].
    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        self.ctx
            .lock()
            .copy_token_range(self.id, other.id, &range);
        other.tokens.clear();
        other
            .tokens
            .extend_from_slice(&self.tokens[range.start..range.end]);
        other.logits = None;
    }

    /// Replace this sequence's tokens with `other`'s tokens in `range`
    /// (token-list indices). The inverse of [`Sequence::copy_to`].
    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    /// Smallest position present in the context memory for this sequence,
    /// or `-1` when the sequence is empty.
    pub fn pos_min(&self) -> i32 {
        self.ctx.lock().kv_pos_min(self.id)
    }

    /// Largest position present in the context memory for this sequence,
    /// or `-1` when the sequence is empty.
    pub fn pos_max(&self) -> i32 {
        self.ctx.lock().kv_pos_max(self.id)
    }

    /// Remove the KV-cache entries for positions in `range`. Invalidates the
    /// cached logits on success. Returns `false` when the range cannot be
    /// removed as a whole.
    pub fn kv_remove(&mut self, range: Range<i32>) -> bool {
        let ok = self
            .ctx
            .lock()
            .kv_remove(self.id, range.start, range.end);
        if ok {
            self.logits = None;
        }
        ok
    }

    /// Copy the KV-cache entries for positions in `range` to `other`, which
    /// must belong to the same [`Context`]. Invalidates `other`'s cached
    /// logits.
    pub fn kv_copy(&self, other: &mut Self, range: Range<i32>) {
        self.ctx.lock().kv_copy(self.id, other.id, range.start, range.end);
        other.logits = None;
    }

    /// Shift the positions of the KV-cache entries in `range` by `delta`.
    /// Invalidates the cached logits.
    pub fn kv_shift(&mut self, range: Range<i32>, delta: i32) {
        self.ctx
            .lock()
            .kv_shift(self.id, range.start, range.end, delta);
        self.logits = None;
    }

    /// Select the next token from the cached logits with `sampler`, or `None`
    /// when there are no cached logits (nothing has been decoded yet).
    ///
    /// The sampler is applied exactly once: its transform runs on a copy of
    /// the cached logits and its selector picks the token.
    pub fn sample<S: Sampler>(&self, sampler: &mut S) -> Option<Token> {
        let logits = self.logits.as_deref()?;
        Some(sampler.sample(logits))
    }
}

impl Index<usize> for Sequence {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tokens[index]
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        self.ctx.lock().release_seq(self.id);
    }
}
