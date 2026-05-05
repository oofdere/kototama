use llama_sys::*;
use std::ops::{Index, Range};

use crate::Context;

pub struct Sequence<'ctx, 'a> {
    ctx: &'a Context<'ctx>,
    id: llama_seq_id,
}

impl<'ctx, 'a> Sequence<'ctx, 'a> {
    pub(crate) fn new(ctx: &'a Context<'ctx>, id: llama_seq_id) -> Self {
        Self { ctx, id }
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
        &self.ctx.tokens[self.id as usize]
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
    pub fn kv_add(&mut self, range: Range<llama_pos>, delta: llama_pos) {
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
        &self.ctx.tokens[self.id as usize][index]
    }
}
