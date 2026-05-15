use crate::{common::*, Batch, Context, LlamaSampler};
use llama_sys::*;
use std::{mem::ManuallyDrop, ops::{Index, Range}};

pub struct Sequence<'ctx, 'a> {
    ctx: ManuallyDrop<&'a Context<'ctx>>,
    id: llama_seq_id,
    batch: Batch,
    tokens: Vec<llama_token>,
    logits: Vec<f32>
}

impl<'ctx, 'a> Sequence<'ctx, 'a> {
    pub(crate) fn new(ctx: &'a Context<'ctx>, id: llama_seq_id) -> Self {
        let batch = Batch::init_token(1, ctx.params().n_seq_max as i32);
        unsafe {
            *batch.n_seq_id.add(0) = 1;
            *(*batch.seq_id.add(0)).add(0) = id;
            *batch.logits.add(0) = 1; // i8, 1 = give me logits
        }

        Self {
            ctx: ManuallyDrop::new(ctx),
            id,
            batch,
            tokens: Vec::new(),
            logits: Vec::new(),
        }
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn push(&mut self, token: llama_token) {
        let pos = self.tokens.len() as i32;
        batch_clear(&mut self.batch);
        batch_add(&mut self.batch, token, pos, &[self.id], true).unwrap();
        self.ctx.decode(*self.batch).unwrap();
        self.logits = self
            .ctx
            .get_logits_ith(0)
            .expect("logits should be available for a freshly decoded token")
            .to_vec();
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

    /// Removes all tokens that belong to the specified sequence and have positions in the provided range.
    ///
    /// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails.
    ///
    /// - if start is negative it will be treated as `0`
    /// - if end is negative it will be treated as the end of the sequence
    pub fn remove(&mut self, range: Range<usize>) -> bool {
        if self.kv_remove(range.start as i32..range.end as i32) {
            self.tokens.drain(range);
            true
        } else {
            false
        }
    }

    /// overwrites the other sequence with the tokens in this sequence
    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        self.kv_copy(other, range.start as i32..range.end as i32);
        other.tokens.clear();
        other
            .tokens
            .extend_from_slice(&self.tokens[range.start..range.end]);
    }

    /// overwrites the tokens in this sequence with the tokens from the other sequence
    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) {
        other.copy_to(self, range);
    }

    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    fn shift(&mut self, range: Range<llama_pos>, delta: llama_pos) {
        if self.ctx.can_shift() {
            unsafe {
                llama_memory_seq_add(
                    self.ctx.get_memory(),
                    self.id as i32,
                    range.start,
                    range.end,
                    delta,
                )
            }
        } else {
            todo!("add fallback for when seq_add is not supported")
        }
    }

    /// Returns the smallest position present in the memory for the specified sequence.
    ///
    /// This is typically non-zero only for SWA caches.
    ///
    /// Note that all positions in the range pos_min, pos_max are guaranteed to be present in the memory.
    ///
    /// Return -1 if the sequence is empty.
    pub fn pos_min(&self) -> llama_pos {
        unsafe { llama_memory_seq_pos_min(self.ctx.get_memory(), self.id as i32) }
    }

    /// Returns the largest position present in the memory for the specified sequence.
    ///
    /// Note that all positions in the range pos_min, pos_max are guaranteed to be present in the memory.
    ///
    /// Return -1 if the sequence is empty.
    pub fn pos_max(&self) -> llama_pos {
        unsafe { llama_memory_seq_pos_max(self.ctx.get_memory(), self.id as i32) }
    }

    /// get a reference to the tokens for this sequence
    pub fn tokens(&self) -> &[llama_token] {
        &self.tokens
    }

    /// Removes all tokens that belong to the specified sequence and have positions in the provided range.
    ///
    /// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails.
    ///
    /// - if start is negative it will be treated as `0`
    /// - if end is negative it will be treated as the end of the sequence
    pub fn kv_remove(&mut self, range: Range<llama_pos>) -> bool {
        unsafe { llama_memory_seq_rm(self.ctx.get_memory(), self.id, range.start, range.end) }
    }

    /// Copy all tokens that belong to the specified sequence to another sequence
    /// - if start is negative it will be treated as `0`
    /// - if end is negative it will be treated as the end of the sequence
    pub fn kv_copy(&self, other: &mut Self, range: Range<llama_pos>) {
        unsafe {
            llama_memory_seq_cp(
                self.ctx.get_memory(),
                self.id,
                other.id,
                range.start,
                range.end,
            )
        }
    }

    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    /// - if start is negative it will be treated as `0`
    /// - if end is negative it will be treated as the end of the sequence
    pub fn kv_shift(&mut self, range: Range<llama_pos>, delta: llama_pos) {
        unsafe {
            llama_memory_seq_add(
                self.ctx.get_memory(),
                self.id,
                range.start,
                range.end,
                delta,
            )
        }
    }
}

impl<'ctx, 'a> Index<usize> for Sequence<'ctx, 'a> {
    type Output = llama_token;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tokens[index]
    }
}

impl Drop for Sequence<'_, '_> {
    fn drop(&mut self) {
        unsafe { llama_memory_seq_rm(self.ctx.get_memory(), self.id, -1, -1) };
        self.ctx.checked_out[self.id as usize].set(false);
    }
}
