//! A typed handle to one KV-cache sequence slot inside a [`Context`].
//!
//! Each [`Sequence`] is a checkout from a [`Context`]'s pool of `n_seq_max`
//! slots, obtained via [`Context::sequence`]. It owns the running list of
//! token ids and (lazily) the logits from the most recent decode, so reading
//! [`Sequence::logits`] or calling [`Sequence::sample`] is a local operation
//! — no round-trip to the actor.
//!
//! Dropping the [`Sequence`] releases its slot back to the [`Context`] and
//! clears the corresponding KV-cache entries.
//!
//! ```ignore
//! let mut seq = ctx.sequence().expect("no free slots");
//! seq.extend(&model.tokenize("The cat sat on the", true, true));
//!
//! for _ in 0..32 {
//!     let next = seq.sample(&sampler);
//!     if model.is_eog(next) { break; }
//!     seq.push(next);
//! }
//! ```

use crate::context::{context_protocol, ContextProtocol, SamplerPtr};
use crate::{Context, LlamaSampler};
use llama_sys::*;
use std::ops::{Index, Range};

/// A handle to a single sequence slot in a [`Context`]'s KV cache.
///
/// Acquired via [`Context::sequence`]. Holds a cheap clone of the parent
/// [`Context`] handle and talks to the context actor via messages, so a
/// [`Sequence`] carries no lifetime parameter and can be moved around freely.
///
/// Logits from the last [`push`](Self::push) are cached in this struct, so
/// [`logits`](Self::logits) and [`sample`](Self::sample) don't have to round-
/// trip to the actor. Any mutation that invalidates them — [`pop`](Self::pop),
/// [`remove`](Self::remove), [`kv_remove`](Self::kv_remove),
/// [`kv_copy`](Self::kv_copy), [`kv_shift`](Self::kv_shift),
/// [`copy_to`](Self::copy_to), [`copy_from`](Self::copy_from) — clears the
/// cache. Use [`decode`](Self::decode) to re-populate it from the current
/// last token.
///
/// Dropping the [`Sequence`] releases its slot back to the parent context and
/// clears the matching KV cache range.
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

    /// Cached logits from the most recent decode on this sequence.
    ///
    /// Returns [`None`] if the sequence has never been decoded, or if the
    /// cache was invalidated by a mutation (see the cache-invalidation list
    /// on [`Sequence`]). Call [`decode`](Self::decode) to refresh.
    ///
    /// The slice length equals the model's vocabulary size
    /// ([`Model::n_tokens`](crate::Model::n_tokens)); entry `i` is the
    /// logit for token id `i`.
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }

    /// `true` if no tokens have been pushed to this sequence.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Push a single token onto the sequence and decode it, updating the
    /// cached [`logits`](Self::logits).
    ///
    /// # Panics
    ///
    /// Panics if the underlying `llama_decode` returns a
    /// [`DecodeError`](crate::DecodeError) (e.g. the KV cache is full, the
    /// input is rejected, the decode is aborted). This is intentional for
    /// the streaming generation loop —
    /// recover by avoiding the offending state (smaller `n_ctx`, fewer live
    /// sequences) rather than catching here.
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

    /// Re-decode the last token to refresh [`logits`](Self::logits) without
    /// pushing a new one.
    ///
    /// Useful after a mutation ([`pop`](Self::pop), [`remove`](Self::remove),
    /// [`kv_remove`](Self::kv_remove), [`kv_copy`](Self::kv_copy),
    /// [`kv_shift`](Self::kv_shift), [`copy_to`](Self::copy_to),
    /// [`copy_from`](Self::copy_from)) clears the logits cache and you need
    /// to sample a new token from the resulting state.
    ///
    /// No-op on an empty sequence — the cache stays [`None`].
    ///
    /// # Panics
    ///
    /// Same conditions as [`push`](Self::push).
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

    /// Remove the last token from the sequence and from the KV cache.
    ///
    /// Returns the popped token id, or [`None`] if the sequence is empty (or
    /// if the underlying KV-cache remove fails — e.g. the model doesn't
    /// support partial removal). Invalidates the [`logits`](Self::logits)
    /// cache; call [`decode`](Self::decode) to refresh it from the new last
    /// token.
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

    /// Push every token in `tokens` onto the sequence in order.
    ///
    /// Equivalent to calling [`push`](Self::push) in a loop; the final
    /// [`logits`](Self::logits) reflect the last token in `tokens`.
    ///
    /// # Panics
    ///
    /// Same conditions as [`push`](Self::push).
    pub fn extend(&mut self, tokens: &[i32]) {
        for &token in tokens {
            self.push(token);
        }
    }

    /// Token id at position `index`, or [`None`] if out of bounds.
    pub fn get(&self, index: usize) -> Option<i32> {
        self.tokens.get(index).copied()
    }

    /// Remove a half-open range of token positions from the sequence and
    /// from the KV cache.
    ///
    /// Returns `true` on success; `false` if the underlying
    /// `llama_memory_seq_rm` rejects the range (the model doesn't support
    /// partial KV removal at this position, etc.) — the token list is left
    /// unchanged in that case. Invalidates the [`logits`](Self::logits)
    /// cache on success.
    pub fn remove(&mut self, range: Range<usize>) -> bool {
        if self.kv_remove(range.start as i32..range.end as i32) {
            self.tokens.drain(range);
            self.logits = None;
            true
        } else {
            false
        }
    }

    /// Copy `range` of tokens (and their KV-cache entries) from `self` into
    /// `other`.
    ///
    /// `other`'s previous token list is replaced with the slice
    /// `self.tokens()[range]`; its [`logits`](Self::logits) cache is cleared.
    /// Call [`decode`](Self::decode) on `other` to refresh logits before
    /// sampling.
    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        self.kv_copy(other, range.start as i32..range.end as i32);
        other.tokens.clear();
        other
            .tokens
            .extend_from_slice(&self.tokens[range.start..range.end]);
        other.logits = None;
    }

    /// Mirror of [`copy_to`](Self::copy_to): copies `range` from `other`
    /// into `self`.
    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    /// Minimum position currently present in this sequence's KV cache.
    /// Wraps `llama_memory_seq_pos_min`.
    pub fn pos_min(&self) -> i32 {
        self.ctx.actor().memory_seq_pos_min(self.id).unwrap()
    }

    /// Maximum position currently present in this sequence's KV cache.
    /// Wraps `llama_memory_seq_pos_max`.
    pub fn pos_max(&self) -> i32 {
        self.ctx.actor().memory_seq_pos_max(self.id).unwrap()
    }

    /// Borrow the underlying token list as a slice.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Drop a half-open range of positions from this sequence's KV cache
    /// **without** touching the local token list.
    ///
    /// Lower-level than [`remove`](Self::remove): this only touches the
    /// cache, leaving [`tokens`](Self::tokens) untouched, which is useful
    /// for cache-management strategies that re-decode from a checkpoint.
    /// Returns `false` if the model rejects the range. Invalidates the
    /// [`logits`](Self::logits) cache on success.
    pub fn kv_remove(&mut self, range: Range<i32>) -> bool {
        let ok = self.ctx
            .actor()
            .memory_seq_rm(self.id, range.start, range.end)
            .unwrap();
        if ok {
            self.logits = None;
        }
        ok
    }

    /// Copy `range` of KV-cache positions from `self` into `other`'s slot,
    /// without touching either side's token list.
    ///
    /// Lower-level than [`copy_to`](Self::copy_to). Invalidates `other`'s
    /// [`logits`](Self::logits) cache. Wraps `llama_memory_seq_cp`.
    pub fn kv_copy(&self, other: &mut Self, range: Range<i32>) {
        self.ctx
            .actor()
            .memory_seq_cp(self.id, other.id, range.start, range.end)
            .unwrap();
        other.logits = None;
    }

    /// Shift the positions of every KV-cache entry in `range` by `delta`.
    ///
    /// Wraps `llama_memory_seq_add`. Used to implement rolling-window
    /// strategies (drop the oldest tokens, then shift the remaining cache
    /// down so positions stay contiguous). Only works when the underlying
    /// memory backend reports [`can_shift`](crate::Context::can_shift); on
    /// architectures that don't, the call still dispatches and may be a
    /// no-op or panic at the FFI layer. Invalidates the
    /// [`logits`](Self::logits) cache.
    pub fn kv_shift(&mut self, range: Range<i32>, delta: i32) {
        self.ctx
            .actor()
            .memory_seq_add(self.id, range.start, range.end, delta)
            .unwrap();
        self.logits = None;
    }

    /// Sample the next token using `sampler` against this sequence's most
    /// recent logits.
    ///
    /// Wraps `llama_sampler_sample` on the parent context at index `-1`
    /// (the last decoded position). Make sure [`logits`](Self::logits) is
    /// `Some` (i.e. the last operation was a [`push`](Self::push),
    /// [`extend`](Self::extend), or [`decode`](Self::decode)) — sampling
    /// after a cache-invalidating mutation without first calling
    /// [`decode`](Self::decode) reads stale state from llama.cpp's internal
    /// buffers.
    pub fn sample<S: LlamaSampler>(&self, sampler: &S) -> i32 {
        self.ctx
            .actor()
            .sample_token(SamplerPtr(sampler.as_ptr()))
            .unwrap()
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
