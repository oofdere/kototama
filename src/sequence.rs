use crate::context::SharedSequenceSnapshot;
use crate::{Context, Sampler, Token};
use std::ops::Range;
use std::sync::Arc;

/// A sequence handle backed by state owned by the context worker.
///
/// Sync and async operations share the same command path. The worker updates the
/// shared snapshot before replying, so canceling an async request cannot leave
/// the Rust-visible token/logit state behind the native KV state.
pub struct Sequence {
    ctx: Context,
    id: i32,
    snapshot: SharedSequenceSnapshot,
}

impl Sequence {
    pub(crate) fn new(ctx: Context, id: i32, snapshot: SharedSequenceSnapshot) -> Self {
        Self { ctx, id, snapshot }
    }

    pub fn logits(&self) -> Option<Arc<[f32]>> {
        self.snapshot.lock().unwrap().logits.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshot.lock().unwrap().tokens.is_empty()
    }

    pub fn push(&mut self, token: Token) {
        self.ctx
            .push_token(token, self.id, self.snapshot.clone())
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
    }

    pub async fn push_async(&mut self, token: Token) {
        self.ctx
            .push_token_async(token, self.id, self.snapshot.clone())
            .await
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
    }

    pub fn decode(&mut self) {
        self.ctx
            .decode_last(self.id, self.snapshot.clone())
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
    }

    pub async fn decode_async(&mut self) {
        self.ctx
            .decode_last_async(self.id, self.snapshot.clone())
            .await
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
    }

    pub fn pop(&mut self) -> Option<Token> {
        self.ctx.pop(self.id, self.snapshot.clone())
    }

    pub async fn pop_async(&mut self) -> Option<Token> {
        self.ctx.pop_async(self.id, self.snapshot.clone()).await
    }

    pub fn len(&self) -> usize {
        self.snapshot.lock().unwrap().tokens.len()
    }

    pub fn extend(&mut self, tokens: &[Token]) {
        for &token in tokens {
            self.push(token);
        }
    }

    pub async fn extend_async(&mut self, tokens: &[Token]) {
        for &token in tokens {
            self.push_async(token).await;
        }
    }

    pub fn get(&self, index: usize) -> Option<Token> {
        self.snapshot.lock().unwrap().tokens.get(index).copied()
    }

    pub fn remove(&mut self, range: Range<usize>) -> bool {
        self.ctx.remove(
            self.id,
            range.start as i32,
            range.end as i32,
            self.snapshot.clone(),
        )
    }

    pub async fn remove_async(&mut self, range: Range<usize>) -> bool {
        self.ctx
            .remove_async(
                self.id,
                range.start as i32,
                range.end as i32,
                self.snapshot.clone(),
            )
            .await
    }

    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        assert!(
            self.ctx.same_worker(&other.ctx),
            "cannot copy sequences between different contexts"
        );
        self.ctx.copy(
            self.id,
            other.id,
            range.start as i32,
            range.end as i32,
            self.snapshot.clone(),
            other.snapshot.clone(),
        );
    }

    pub async fn copy_to_async(&self, other: &mut Self, range: Range<usize>) {
        assert!(
            self.ctx.same_worker(&other.ctx),
            "cannot copy sequences between different contexts"
        );
        self.ctx
            .copy_async(
                self.id,
                other.id,
                range.start as i32,
                range.end as i32,
                self.snapshot.clone(),
                other.snapshot.clone(),
            )
            .await;
    }

    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    pub async fn copy_from_async(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to_async(self, range).await;
    }

    pub fn pos_min(&self) -> i32 {
        self.ctx.memory_seq_pos_min(self.id)
    }

    pub fn pos_max(&self) -> i32 {
        self.ctx.memory_seq_pos_max(self.id)
    }

    pub fn tokens(&self) -> Arc<[Token]> {
        self.snapshot.lock().unwrap().tokens.clone()
    }

    pub fn kv_remove(&mut self, range: Range<i32>) -> bool {
        self.ctx.memory_seq_rm(
            self.id,
            range.start,
            range.end,
            self.snapshot.clone(),
        )
    }

    pub fn kv_shift(&mut self, range: Range<i32>, delta: i32) {
        self.ctx.memory_seq_add(
            self.id,
            range.start,
            range.end,
            delta,
            self.snapshot.clone(),
        );
    }

    pub fn sample<S: Sampler>(&self, sampler: &mut S) -> Option<Token> {
        let logits = self.logits()?;
        Some(sampler.sample(&logits))
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        self.ctx.release_seq(self.id);
    }
}
