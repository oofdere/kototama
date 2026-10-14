//! Sequences: one independent stream of tokens inside a [`Context`].
//!
//! Think of a [`Sequence`] as a single conversation. It remembers which tokens
//! you've fed in, keeps the logits from the most recent one, and lets you edit
//! its region of the KV cache (remove, copy, shift tokens).
//!
//! You get one from [`Context::sequence`]. Several sequences can be alive on
//! the same context at once; each has its own slot and doesn't see the others'
//! tokens.

use std::ops::{Index, Range};

use crate::{Context, Pos, Sampler, SeqId, Token};

/// One stream of tokens inside a [`Context`].
///
/// Dropping a `Sequence` clears its tokens from the KV cache and hands its slot
/// back to the context, so a new [`Context::sequence`] call can reuse it.
///
/// # Logits
///
/// After every [`push`](Self::push), the logits for the next token are copied
/// into a buffer owned by this sequence. That's why you can push to two
/// different sequences and still read both sets of logits afterwards:
/// llama.cpp's own output buffer is shared by the whole context and gets
/// overwritten on every decode, but each sequence keeps its own copy.
///
/// The buffer is allocated once and then reused, so generating tokens in a
/// loop doesn't allocate.
pub struct Sequence {
    /// Handle to the context this sequence lives in.
    ctx: Context,
    /// Our slot number inside the context.
    id: SeqId,
    /// Every token currently in this sequence, in order. `tokens[i]` sits at
    /// position `i` in the KV cache.
    tokens: Vec<Token>,
    /// Logits from the last decode. Only meaningful while `has_logits` is true;
    /// we keep the allocation around even when it's stale so it can be reused.
    logits: Vec<f32>,
    /// `false` before the first push and after any edit that makes the cached
    /// logits out of date (pop, remove, shift, ...).
    has_logits: bool,
}

impl Sequence {
    /// Called by [`Context::sequence`] after it has reserved `id` for us.
    pub(crate) fn new(ctx: Context, id: SeqId) -> Self {
        Self {
            ctx,
            id,
            tokens: Vec::new(),
            logits: Vec::new(),
            has_logits: false,
        }
    }

    /// This sequence's slot id inside the context.
    pub fn id(&self) -> SeqId {
        self.id
    }

    // -- Reading -----------------------------------------------------------

    /// Logits for the token that would come next, or `None` if there aren't
    /// any (nothing pushed yet, or the sequence was edited since the last
    /// push — call [`decode`](Self::decode) to refresh them).
    ///
    /// The slice has one entry per vocabulary token. Higher means more likely.
    pub fn logits(&self) -> Option<&[f32]> {
        self.has_logits.then_some(self.logits.as_slice())
    }

    /// All tokens in the sequence, in order.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The token at `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<Token> {
        self.tokens.get(index).copied()
    }

    /// Smallest position this sequence has in the KV cache, or -1 if empty.
    pub fn pos_min(&self) -> Pos {
        self.ctx.inner().seq_pos_min(self.id)
    }

    /// Largest position this sequence has in the KV cache, or -1 if empty.
    pub fn pos_max(&self) -> Pos {
        self.ctx.inner().seq_pos_max(self.id)
    }

    // -- Feeding tokens in -------------------------------------------------

    /// Feed one token into the model and update [`logits`](Self::logits).
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp fails to decode (for example, the KV cache is full).
    pub fn push(&mut self, token: Token) {
        let pos = self.tokens.len() as Pos;
        self.decode_at(token, pos);
        self.tokens.push(token);
    }

    /// Push each token in order. Equivalent to calling [`push`](Self::push) in
    /// a loop.
    pub fn extend(&mut self, tokens: &[Token]) {
        for &token in tokens {
            self.push(token);
        }
    }

    /// Recompute the logits for the last token without adding a new one.
    ///
    /// Useful after [`pop`](Self::pop), [`remove`](Self::remove) and friends,
    /// which clear the cached logits. Does nothing if the sequence is empty.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp fails to decode.
    pub fn decode(&mut self) {
        let Some(&last) = self.tokens.last() else {
            return;
        };
        let pos = (self.tokens.len() - 1) as Pos;
        // llama.cpp only accepts a position that directly follows what's
        // already in the cache, so drop the old entry for the last token first
        // and then feed it in again.
        self.ctx.inner().seq_rm(self.id, pos, -1);
        self.decode_at(last, pos);
    }

    /// Run one token through the model at `pos` and copy the resulting logits
    /// into our buffer.
    fn decode_at(&mut self, token: Token, pos: Pos) {
        let mut inner = self.ctx.inner();
        let logits = inner
            .decode_one(token, pos, self.id)
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));

        // Copy into our existing buffer. `clear` keeps the allocation, so after
        // the first push this never allocates.
        self.logits.clear();
        self.logits.extend_from_slice(logits);
        self.has_logits = true;
    }

    /// Get the next token by running `sampler` over the current logits.
    ///
    /// Returns `None` if there are no logits (see [`logits`](Self::logits)).
    /// This doesn't push the token — do that yourself if you want to keep it.
    pub fn sample<S: Sampler>(&self, sampler: &mut S) -> Option<Token> {
        let logits = self.logits()?;
        Some(sampler.sample(logits))
    }

    // -- Editing -----------------------------------------------------------

    /// Remove the last token. Returns it, or `None` if the sequence was empty.
    pub fn pop(&mut self) -> Option<Token> {
        let len = self.tokens.len();
        if len == 0 {
            return None;
        }
        if self.kv_remove((len - 1) as Pos..len as Pos) {
            self.tokens.pop()
        } else {
            None
        }
    }

    /// Remove the tokens in `range` from both the token list and the KV cache.
    ///
    /// Returns `false` (and changes nothing) if llama.cpp refuses, which can
    /// happen on recurrent models that can't remove from the middle.
    ///
    /// Note: tokens after the removed range keep their old KV cache positions,
    /// leaving a gap. Use [`kv_shift`](Self::kv_shift) to close it if you need
    /// to keep pushing.
    pub fn remove(&mut self, range: Range<usize>) -> bool {
        if self.kv_remove(range.start as Pos..range.end as Pos) {
            self.tokens.drain(range);
            true
        } else {
            false
        }
    }

    /// Make `other` contain a copy of `self.tokens[range]`, including the KV
    /// cache entries (so `other` doesn't have to re-decode them).
    ///
    /// Copying between sequences requires a unified KV cache
    /// (`ContextParams::kv_unified = true`).
    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        self.kv_copy(other, range.start as Pos..range.end as Pos);
        other.tokens.clear();
        other.tokens.extend_from_slice(&self.tokens[range]);
    }

    /// The reverse of [`copy_to`](Self::copy_to): copy from `other` into
    /// `self`.
    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    // -- Raw KV cache operations -------------------------------------------
    //
    // These only touch the KV cache and do *not* update `self.tokens`. Prefer
    // the methods above unless you know you need these.

    /// Remove positions `range` from the KV cache. Clears cached logits on
    /// success.
    pub fn kv_remove(&mut self, range: Range<Pos>) -> bool {
        let ok = self.ctx.inner().seq_rm(self.id, range.start, range.end);
        if ok {
            self.has_logits = false;
        }
        ok
    }

    /// Copy KV cache positions `range` from `self` into `other`. Clears
    /// `other`'s cached logits.
    pub fn kv_copy(&self, other: &mut Self, range: Range<Pos>) {
        self.ctx
            .inner()
            .seq_cp(self.id, other.id, range.start, range.end);
        other.has_logits = false;
    }

    /// Move KV cache positions `range` by `delta` (negative moves them
    /// earlier). Clears cached logits. Check [`Context::can_shift`] first.
    pub fn kv_shift(&mut self, range: Range<Pos>, delta: Pos) {
        self.ctx
            .inner()
            .seq_add(self.id, range.start, range.end, delta);
        self.has_logits = false;
    }
}

impl Index<usize> for Sequence {
    type Output = Token;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tokens[index]
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        self.ctx.inner().release_seq(self.id);
    }
}
