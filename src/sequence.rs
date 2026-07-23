use crate::context::SharedSequenceSnapshot;
use crate::{Context, Sampler, Token};
use std::ops::{Index, Range};
use std::sync::Arc;

/// A sequence handle whose native operations are serialized by the context
/// worker. Sync and async methods submit the same commands and differ only in
/// how they wait for the reply.
pub struct Sequence {
    ctx: Context,
    id: i32,
    snapshot: SharedSequenceSnapshot,
    tokens: Vec<Token>,
    logits: Option<Arc<[f32]>>,
}

impl Sequence {
    pub(crate) fn new(ctx: Context, id: i32, snapshot: SharedSequenceSnapshot) -> Self {
        Self {
            ctx,
            id,
            snapshot,
            tokens: Vec::new(),
            logits: None,
        }
    }

    fn refresh(&mut self) {
        let state = self.snapshot.lock().unwrap();
        self.tokens = state.tokens.to_vec();
        self.logits = state.logits.clone();
    }

    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn push(&mut self, token: Token) {
        self.ctx
            .push_token(token, self.id, self.snapshot.clone())
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
        self.refresh();
    }

    pub async fn push_async(&mut self, token: Token) {
        self.ctx
            .push_token_async(token, self.id, self.snapshot.clone())
            .await
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
        self.refresh();
    }

    pub fn decode(&mut self) {
        self.ctx
            .decode_last(self.id, self.snapshot.clone())
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
        self.refresh();
    }

    pub async fn decode_async(&mut self) {
        self.ctx
            .decode_last_async(self.id, self.snapshot.clone())
            .await
            .unwrap_or_else(|e| panic!("decode failed: {e:?}"));
        self.refresh();
    }

    pub fn pop(&mut self) -> Option<Token> {
        let token = self.ctx.pop(self.id, self.snapshot.clone());
        self.refresh();
        token
    }

    pub async fn pop_async(&mut self) -> Option<Token> {
        let token = self.ctx.pop_async(self.id, self.snapshot.clone()).await;
        self.refresh();
        token
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
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
        self.tokens.get(index).copied()
    }

    pub fn remove(&mut self, range: Range<usize>) -> bool {
        let removed = self.ctx.remove(
            self.id,
            range.start as i32,
            range.end as i32,
            self.snapshot.clone(),
        );
        self.refresh();
        removed
    }

    pub async fn remove_async(&mut self, range: Range<usize>) -> bool {
        let removed = self
            .ctx
            .remove_async(
                self.id,
                range.start as i32,
                range.end as i32,
                self.snapshot.clone(),
            )
            .await;
        self.refresh();
        removed
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
        other.refresh();
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
        other.refresh();
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

    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    pub fn kv_remove(&mut self, range: Range<i32>) -> bool {
        let removed =
            self.ctx
                .memory_seq_rm(self.id, range.start, range.end, self.snapshot.clone());
        self.refresh();
        removed
    }

    pub fn kv_shift(&mut self, range: Range<i32>, delta: i32) {
        self.ctx.memory_seq_add(
            self.id,
            range.start,
            range.end,
            delta,
            self.snapshot.clone(),
        );
        self.refresh();
    }

    pub fn sample<S: Sampler>(&self, sampler: &mut S) -> Option<Token> {
        let logits = self.logits()?;
        Some(sampler.sample(logits))
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
        self.ctx.release_seq(self.id);
    }
}
