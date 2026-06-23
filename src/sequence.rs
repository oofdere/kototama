//! Per-sequence handle into a [`Context`]'s KV cache.
//!
//! A [`Sequence`] is the unit of generation: tokens are pushed onto it, the
//! resulting logits are cached locally, and a [`Sampler`] turns those logits
//! into the next token. Multiple sequences can coexist on a single
//! [`Context`] (up to `params.n_seq_max`), each occupying one slot of the
//! shared KV cache.
//!
//! ### Lifecycle
//!
//! Sequences are obtained from [`Context::sequence`](crate::Context::sequence) —
//! never constructed directly. The slot is released back to the context, and
//! its KV cache entries cleared, when the [`Sequence`] is dropped. Sequences
//! hold a clone of the parent [`Context`], so the context lives at least as
//! long as any sequence handed out from it.
//!
//! ### Logits cache
//!
//! Every mutating call ([`push`](Sequence::push), [`pop`](Sequence::pop),
//! [`remove`](Sequence::remove), the `kv_*` operations) refreshes — or
//! invalidates — the local logits cache. [`logits`](Sequence::logits) returns
//! `None` until at least one [`push`](Sequence::push) (or
//! [`decode`](Sequence::decode)) has produced fresh logits since the last
//! mutation. The cache means [`sample`](Sequence::sample) and
//! [`logits`](Sequence::logits) are zero-cost reads — no extra round-trip to
//! the context actor.
//!
//! ### Tokens vs. KV positions
//!
//! [`Sequence`] tracks the token ids that have been decoded into its slot;
//! [`tokens`](Sequence::tokens) returns them in push order and
//! [`len`](Sequence::len) is the count. The `kv_*` family operates one level
//! lower, on KV-cache positions — handy for advanced moves like sliding
//! windows or speculative-decoding rollbacks.

use crate::context::{context_protocol, ContextProtocol, SamplerPtr};
use crate::{Context, Sampler, Token};
use std::ops::{Index, Range};

/// A handle to one sequence slot in a [`Context`]'s KV cache.
///
/// Holds a clone of the parent [`Context`] and dispatches every FFI call
/// through the context actor, so a `Sequence` is `Send` and can be moved
/// across threads. It is **not** [`Sync`]: most methods take `&mut self`, and
/// each sequence corresponds to a single point of generation.
///
/// Construct via [`Context::sequence`](crate::Context::sequence); drop to
/// release the slot.
pub struct Sequence {
    ctx: Context,
    id: i32,
    tokens: Vec<i32>,
    logits: Option<Vec<f32>>,
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

    /// Logits produced by the most recent decode, or `None` if none are
    /// cached.
    ///
    /// Returns `None` before the first [`push`](Self::push), and after any
    /// mutation that invalidates the cache ([`pop`](Self::pop),
    /// [`remove`](Self::remove), [`kv_remove`](Self::kv_remove),
    /// [`kv_copy`](Self::kv_copy), [`kv_shift`](Self::kv_shift), or any
    /// successful [`copy_to`](Self::copy_to)/[`copy_from`](Self::copy_from)
    /// onto this sequence). Call [`decode`](Self::decode) to re-populate the
    /// cache without changing the token sequence.
    ///
    /// The returned slice has length `model.n_tokens()` — one f32 per vocab
    /// entry.
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }

    /// True if no tokens have been pushed (or all pushed tokens have been
    /// removed).
    ///
    /// Equivalent to `self.len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Append `token` to the sequence and decode it.
    ///
    /// Submits a one-token batch to the context actor, advances the KV cache,
    /// and caches the resulting logits — fetch them via [`logits`](Self::logits)
    /// or feed them straight into a sampler with [`sample`](Self::sample).
    ///
    /// **Panics** if the underlying decode fails (e.g. KV cache full, fatal
    /// error inside llama.cpp). Use [`Context::free_slots`](crate::Context::free_slots)
    /// to back-pressure callers before they grow a sequence beyond what the
    /// context can hold.
    pub fn push(&mut self, token: i32) {
        let pos = self.tokens.len() as i32;
        self.logits = Some(
            self.ctx
                .actor()
                .push_token(token, pos, self.id)
                .unwrap()
                .unwrap_or_else(|e| panic!("decode failed: {e:?}")),
        );
        self.tokens.push(token);
    }

    /// Re-decode the last token to refresh the logits cache without
    /// extending the sequence.
    ///
    /// Use after a mutation that invalidates the cache — typically
    /// [`pop`](Self::pop), [`remove`](Self::remove), or any of the `kv_*`
    /// operations — when you want to sample again from the same trailing
    /// position. No-op on an empty sequence.
    ///
    /// **Panics** if the decode fails (see [`push`](Self::push)).
    pub fn decode(&mut self) {
        if let Some(&last_token) = self.tokens.last() {
            let pos = (self.tokens.len() - 1) as i32;
            self.logits = Some(
                self.ctx
                    .actor()
                    .push_token(last_token, pos, self.id)
                    .unwrap()
                    .unwrap_or_else(|e| panic!("decode failed: {e:?}")),
            );
        }
    }

    /// Drop the most recently pushed token and its KV-cache entry.
    ///
    /// Returns the popped token, or `None` if the sequence was already empty
    /// or if llama.cpp refused to remove the slot. Invalidates the logits
    /// cache; call [`decode`](Self::decode) to repopulate it.
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

    /// Number of tokens currently in the sequence.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Push every token in `tokens` in order.
    ///
    /// Each token is decoded individually — equivalent to calling
    /// [`push`](Self::push) in a loop. After the call, [`logits`](Self::logits)
    /// reflects the final token's distribution.
    pub fn extend(&mut self, tokens: &[i32]) {
        for &token in tokens {
            self.push(token);
        }
    }

    /// Return the token id at `index`, or `None` if out of bounds.
    ///
    /// Indices match push order: `get(0)` is the first token pushed onto the
    /// sequence.
    pub fn get(&self, index: usize) -> Option<i32> {
        self.tokens.get(index).copied()
    }

    /// Remove the tokens in `range` from the sequence and its KV cache.
    ///
    /// Returns `true` on success and invalidates the logits cache. Returns
    /// `false` (and leaves the sequence untouched) if llama.cpp rejects the
    /// removal — typically because the requested range falls outside the
    /// sequence's current KV positions.
    pub fn remove(&mut self, range: Range<usize>) -> bool {
        if self.kv_remove(range.start as i32..range.end as i32) {
            self.tokens.drain(range);
            self.logits = None;
            true
        } else {
            false
        }
    }

    /// Copy a token range from this sequence into `other`, replacing
    /// `other`'s contents.
    ///
    /// Both the KV-cache positions and the local token list of `other` are
    /// overwritten so that `other.tokens()` ends equal to
    /// `&self.tokens()[range]`. `other`'s logits cache is invalidated.
    ///
    /// Requires the parent context to be configured with `kv_unified = true`
    /// — otherwise the underlying `llama_memory_seq_cp` is a no-op on the
    /// KV cache. The token list is updated either way.
    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        self.kv_copy(other, range.start as i32..range.end as i32);
        other.tokens.clear();
        other
            .tokens
            .extend_from_slice(&self.tokens[range.start..range.end]);
        other.logits = None;
    }

    /// Mirror of [`copy_to`](Self::copy_to) with the source and destination
    /// roles swapped: copy `other[range]` into `self`.
    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    /// Smallest position currently held in this sequence's KV cache.
    ///
    /// Returns `-1` when the sequence has no entries (e.g. immediately after
    /// checkout, before the first [`push`](Self::push)).
    pub fn pos_min(&self) -> i32 {
        self.ctx.actor().memory_seq_pos_min(self.id).unwrap()
    }

    /// Largest position currently held in this sequence's KV cache.
    ///
    /// Returns `-1` when the sequence has no entries. After `n` pushes onto
    /// a fresh sequence, this is `n - 1`.
    pub fn pos_max(&self) -> i32 {
        self.ctx.actor().memory_seq_pos_max(self.id).unwrap()
    }

    /// Borrow the full token-id list in push order.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Remove KV-cache positions in `range` directly, bypassing the local
    /// token list.
    ///
    /// Lower-level than [`remove`](Self::remove): the token list is left
    /// untouched, so its length can diverge from the KV positions. Use this
    /// only when you also intend to reconcile the token list yourself.
    /// Invalidates the logits cache on success.
    ///
    /// Returns the boolean reported by `llama_memory_seq_rm` — `false`
    /// typically means the range fell outside the sequence's current
    /// positions.
    pub fn kv_remove(&mut self, range: Range<i32>) -> bool {
        let ok = self
            .ctx
            .actor()
            .memory_seq_rm(self.id, range.start, range.end)
            .unwrap();
        if ok {
            self.logits = None;
        }
        ok
    }

    /// Copy KV-cache positions in `range` from `self` into `other`,
    /// bypassing the token list.
    ///
    /// Lower-level than [`copy_to`](Self::copy_to): the destination's token
    /// list is left untouched. Useful for advanced rollbacks where the
    /// caller manages token bookkeeping manually. Invalidates `other`'s
    /// logits cache.
    ///
    /// Requires the parent context to be configured with `kv_unified = true`.
    pub fn kv_copy(&self, other: &mut Self, range: Range<i32>) {
        self.ctx
            .actor()
            .memory_seq_cp(self.id, other.id, range.start, range.end)
            .unwrap();
        other.logits = None;
    }

    /// Shift the positions of KV-cache entries in `range` by `delta`.
    ///
    /// Used to implement sliding-window strategies — drop a prefix with
    /// [`kv_remove`](Self::kv_remove), then shift the surviving entries down
    /// so the next [`push`](Self::push) lands at a contiguous position.
    /// Only supported when the underlying memory backend reports
    /// [`Context::can_shift`](crate::Context::can_shift) — otherwise
    /// llama.cpp aborts the process.
    ///
    /// Invalidates the logits cache.
    pub fn kv_shift(&mut self, range: Range<i32>, delta: i32) {
        self.ctx
            .actor()
            .memory_seq_add(self.id, range.start, range.end, delta)
            .unwrap();
        self.logits = None;
    }

    /// Run the cached logits through `sampler` and return the chosen token.
    ///
    /// Returns `None` if no logits are cached — i.e. before the first
    /// [`push`](Self::push), or after a mutation that invalidated the cache
    /// (call [`decode`](Self::decode) first to refresh).
    ///
    /// The sampler's [`apply`](crate::Sampler::apply) is called first to
    /// transform the logits, then [`sample`](crate::Sampler::sample) selects
    /// a token from the transformed slice. The cache itself is not modified.
    pub fn sample<S: Sampler>(&self, sampler: &S) -> Option<Token> {
        let logits = self.logits.as_deref()?;
        let transformed = sampler.apply(logits);
        Some(sampler.sample(&transformed))
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
        let _ = self
            .ctx
            .actor()
            .request(context_protocol::ReleaseSeq { seq_id: self.id });
    }
}
