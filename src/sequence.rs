use crate::context::{context_protocol, ContextProtocol, SamplerPtr};
use crate::{Context, LlamaSampler};
use llama_sys::*;
use std::ops::{Index, Range};

/// A sequence handle. No lifetime parameters — holds a clone of the Context
/// handle and communicates with the context actor via messages.
///
/// Logits from the last `push()` are cached locally, so `sample()` and
/// `logits()` don't need a second round-trip.
pub struct Sequence {
    ctx: Context,
    id: llama_seq_id,
    tokens: Vec<llama_token>,
    logits: Option<Vec<f32>>,
}

impl Sequence {
    pub(crate) fn new(ctx: Context, id: llama_seq_id) -> Self {
        Self {
            ctx,
            id,
            tokens: Vec::new(),
            logits: None,
        }
    }

    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn push(&mut self, token: llama_token) {
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

    /// Re-decode the last token to refresh logits without pushing a new one.
    /// Useful after `pop()`, `remove()`, or other mutations that invalidate logits.
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

    pub fn pop(&mut self) -> Option<llama_token> {
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

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn extend(&mut self, tokens: &[llama_token]) {
        for &token in tokens {
            self.push(token);
        }
    }

    pub fn get(&self, index: usize) -> Option<llama_token> {
        self.tokens.get(index).copied()
    }

    pub fn remove(&mut self, range: Range<usize>) -> bool {
        if self.kv_remove(range.start as i32..range.end as i32) {
            self.tokens.drain(range);
            self.logits = None;
            true
        } else {
            false
        }
    }

    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        self.kv_copy(other, range.start as i32..range.end as i32);
        other.tokens.clear();
        other
            .tokens
            .extend_from_slice(&self.tokens[range.start..range.end]);
        other.logits = None;
    }

    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    pub fn pos_min(&self) -> llama_pos {
        self.ctx.actor().memory_seq_pos_min(self.id).unwrap()
    }

    pub fn pos_max(&self) -> llama_pos {
        self.ctx.actor().memory_seq_pos_max(self.id).unwrap()
    }

    pub fn tokens(&self) -> &[llama_token] {
        &self.tokens
    }

    pub fn kv_remove(&mut self, range: Range<llama_pos>) -> bool {
        let ok = self.ctx
            .actor()
            .memory_seq_rm(self.id, range.start, range.end)
            .unwrap();
        if ok {
            self.logits = None;
        }
        ok
    }

    pub fn kv_copy(&self, other: &mut Self, range: Range<llama_pos>) {
        self.ctx
            .actor()
            .memory_seq_cp(self.id, other.id, range.start, range.end)
            .unwrap();
        other.logits = None;
    }

    pub fn kv_shift(&mut self, range: Range<llama_pos>, delta: llama_pos) {
        self.ctx
            .actor()
            .memory_seq_add(self.id, range.start, range.end, delta)
            .unwrap();
        self.logits = None;
    }

    pub fn sample<S: LlamaSampler>(&self, sampler: &S) -> llama_token {
        self.ctx
            .actor()
            .sample_token(SamplerPtr(sampler.as_ptr()))
            .unwrap()
    }
}

impl Index<usize> for Sequence {
    type Output = llama_token;

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
