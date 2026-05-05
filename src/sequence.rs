use crate::{Batch, Context, LlamaSampler, Model};
use llama_sys::*;
use std::{
    cell::RefCell,
    ops::{Deref, DerefMut, Range},
    rc::Weak,
    sync::OnceLock,
    vec,
};


pub struct Sequence<'ctx, 'a> {
    ctx: &'a Context<'ctx>,
    id: llama_seq_id,
}

impl<'ctx, 'a> Sequence<'ctx, 'a> {
    pub(crate) fn new(ctx: &'a Context<'ctx>, id: llama_seq_id) -> Self {
        Self { ctx, id }
    }

    /// Removes all tokens that belong to the specified sequence and have positions in the provided range.
    /// 
    /// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails.
    /// 
    /// - if start is negative it will be treated as `0`
    /// - if end is negative it will be treated as the end of the sequence
    pub fn remove(&mut self, range: Range<usize>) -> bool {
        if self.kv_remove(range.start as i32..range.end as i32) {
            self.tokens().borrow_mut().drain(range);
            true
        } else {
            false
        }
    }

    /// overwrites the other sequence with the tokens in this sequence
    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) {
        self.kv_copy(other, range.start as i32..range.end as i32);
        let mut tokens = other.tokens().borrow_mut();
        tokens.clear();
        tokens.extend_from_slice(&self.tokens().borrow()[range.start..range.end]);
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
            unsafe { llama_memory_seq_add(self.ctx.get_memory(), self.id as i32, range.start, range.end, delta) }
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

    /// get the RefCell containing the tokens for this sequence
    pub fn tokens(&self) -> &RefCell<Vec<llama_token>> {
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
