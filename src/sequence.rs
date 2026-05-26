use crate::context::*;
use crate::{common, Context, LlamaSampler};
use llama_sys::*;
use std::ops::{Index, Range};

/// A sequence handle. No lifetime parameters — holds a clone of the Context
/// handle and the shared Mutex-guarded context state.
///
/// Hot-path operations (push, decode, sample, logits) go directly through
/// the Mutex — no actor message overhead. Slot checkout/release still goes
/// through the actor.
pub struct Sequence {
    ctx: Context,
    id: llama_seq_id,
    tokens: Vec<llama_token>,
    logits: Vec<f32>,
}

impl Sequence {
    pub(crate) fn new(ctx: Context, id: llama_seq_id) -> Self {
        Self {
            ctx,
            id,
            tokens: Vec::new(),
            logits: Vec::new(),
        }
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn push(&mut self, token: llama_token) {
        let pos = self.tokens.len() as i32;
        let mut shared = self.ctx.shared().lock().unwrap();
        common::batch_clear(&mut shared.batch);
        common::batch_add(&mut shared.batch, token, pos, &[self.id], true)
            .expect("batch add failed");
        shared.decode_batch().expect("decode failed");
        self.logits = shared.get_logits_ith(0).expect("no logits");
        drop(shared);
        self.tokens.push(token);
    }

    pub fn pop(&mut self) -> Option<llama_token> {
        let len = self.tokens.len() as i32;
        if len == 0 {
            return None;
        }
        if self.kv_remove((len - 1)..len) {
            self.tokens.pop()
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
    }

    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    pub fn pos_min(&self) -> llama_pos {
        let shared = self.ctx.shared().lock().unwrap();
        unsafe { llama_memory_seq_pos_min(shared.get_memory(), self.id) }
    }

    pub fn pos_max(&self) -> llama_pos {
        let shared = self.ctx.shared().lock().unwrap();
        unsafe { llama_memory_seq_pos_max(shared.get_memory(), self.id) }
    }

    pub fn tokens(&self) -> &[llama_token] {
        &self.tokens
    }

    pub fn kv_remove(&mut self, range: Range<llama_pos>) -> bool {
        let shared = self.ctx.shared().lock().unwrap();
        unsafe {
            llama_memory_seq_rm(shared.get_memory(), self.id, range.start, range.end)
        }
    }

    pub fn kv_copy(&self, other: &mut Self, range: Range<llama_pos>) {
        let shared = self.ctx.shared().lock().unwrap();
        unsafe {
            llama_memory_seq_cp(
                shared.get_memory(),
                self.id,
                other.id,
                range.start,
                range.end,
            )
        }
    }

    pub fn kv_shift(&mut self, range: Range<llama_pos>, delta: llama_pos) {
        let shared = self.ctx.shared().lock().unwrap();
        unsafe {
            llama_memory_seq_add(shared.get_memory(), self.id, range.start, range.end, delta)
        }
    }

    /// Sample a token using the cached logits from the last push().
    pub fn sample<S: LlamaSampler>(&self, sampler: &S) -> llama_token {
        let shared = self.ctx.shared().lock().unwrap();
        unsafe { llama_sampler_sample(sampler.as_ptr(), shared.ctx, -1) }
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
        // Tell the actor to clean up our KV cache entries and release the slot.
        // Ignore errors — the actor may have already stopped.
        let _ = self.ctx.actor().request(ReleaseSeq { seq_id: self.id });
    }
}
