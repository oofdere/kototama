use llama_sys::*;
use std::{
    cell::RefCell,
    fmt,
    ops::{AddAssign, Range},
};

use crate::{common, context::TextBuf, Batch, Context, ContextDecodeResult, LlamaSampler};

pub struct Sequence<'ctx, 'a> {
    ctx: &'a Context<'ctx>,
    id: llama_seq_id,
}

impl<'ctx, 'a> Sequence<'ctx, 'a> {
    pub(crate) fn new(ctx: &'a Context<'ctx>, id: llama_seq_id) -> Self {
        Self { ctx, id }
    }

    /// Returns the sequence id.
    pub fn id(&self) -> llama_seq_id {
        self.id
    }

    // === KV Cache Operations ===

    /// Returns the smallest position present in the memory for the specified sequence.
    ///
    /// This is typically non-zero only for SWA caches.
    ///
    /// Note that all positions in the range pos_min, pos_max are guaranteed to be present in the memory.
    ///
    /// Return -1 if the sequence is empty.
    pub fn pos_min(&self) -> llama_pos {
        unsafe { llama_memory_seq_pos_min(self.ctx.get_memory(), self.id) }
    }

    /// Returns the largest position present in the memory for the specified sequence.
    ///
    /// Note that all positions in the range pos_min, pos_max are guaranteed to be present in the memory.
    ///
    /// Return -1 if the sequence is empty.
    pub fn pos_max(&self) -> llama_pos {
        unsafe { llama_memory_seq_pos_max(self.ctx.get_memory(), self.id) }
    }

    /// get the RefCell containing the tokens for this sequence
    pub fn tokens(&self) -> &RefCell<Vec<llama_token>> {
        &self.ctx.tokens[self.id as usize]
    }

    /// get the RefCell containing the text buffer for this sequence
    fn text_buf(&self) -> &RefCell<TextBuf> {
        &self.ctx.text_bufs[self.id as usize]
    }

    /// Resets the text buffer so the next `push_str` starts a fresh region at `new_start`.
    fn commit_text(&self, new_start: usize) {
        let mut buf = self.text_buf().borrow_mut();
        buf.text.clear();
        buf.token_start = new_start;
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

    // === Vec-like Operations ===

    /// Returns the number of tokens in the sequence.
    pub fn len(&self) -> usize {
        self.tokens().borrow().len()
    }

    /// Returns true if the sequence has no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens().borrow().is_empty()
    }

    /// Appends a token to the end of the sequence.
    ///
    /// Commits the text buffer so the next `push_str` starts a fresh region after this token.
    pub fn push(&self, token: llama_token) {
        let mut tokens = self.tokens().borrow_mut();
        tokens.push(token);
        let new_len = tokens.len();
        drop(tokens);
        self.commit_text(new_len);
    }

    /// Removes and returns the last token, also removing its KV cache entry.
    ///
    /// Commits the text buffer.
    pub fn pop(&mut self) -> Option<llama_token> {
        let mut tokens = self.tokens().borrow_mut();
        let token = tokens.pop()?;
        let pos = tokens.len() as llama_pos;
        let new_len = tokens.len();
        drop(tokens);
        self.commit_text(new_len);
        self.kv_remove(pos..pos + 1);
        Some(token)
    }

    /// Returns the token at the given index, or None if out of bounds.
    pub fn get(&self, index: usize) -> Option<llama_token> {
        self.tokens().borrow().get(index).copied()
    }

    /// Returns the last token, or None if empty.
    pub fn last(&self) -> Option<llama_token> {
        self.tokens().borrow().last().copied()
    }

    /// Clears all tokens, the text buffer, and KV cache entries.
    pub fn clear(&mut self) {
        self.tokens().borrow_mut().clear();
        self.commit_text(0);
        unsafe { llama_memory_seq_rm(self.ctx.get_memory(), self.id, 0, -1) };
    }

    /// Truncates the sequence to the given length, removing trailing tokens and their KV cache entries.
    ///
    /// Commits the text buffer.
    pub fn truncate(&mut self, len: usize) {
        let mut tokens = self.tokens().borrow_mut();
        if len >= tokens.len() {
            return;
        }
        let old_len = tokens.len();
        tokens.truncate(len);
        drop(tokens);
        self.commit_text(len);
        self.kv_remove(len as llama_pos..old_len as llama_pos);
    }

    /// Extends the sequence with tokens from an iterator.
    ///
    /// Commits the text buffer.
    pub fn extend(&self, iter: impl IntoIterator<Item = llama_token>) {
        let mut tokens = self.tokens().borrow_mut();
        tokens.extend(iter);
        let new_len = tokens.len();
        drop(tokens);
        self.commit_text(new_len);
    }

    /// Returns a copy of all tokens as a Vec.
    pub fn to_vec(&self) -> Vec<llama_token> {
        self.tokens().borrow().clone()
    }

    // === String-like Operations ===

    /// Tokenizes text and appends the resulting tokens.
    ///
    /// Tracks the raw text so that consecutive `push_str` calls produce the same
    /// tokenization as a single call with the concatenated string. For example,
    /// `push_str("hel"); push_str("lo")` produces the same tokens as `push_str("hello")`.
    ///
    /// If re-tokenization changes tokens that have already been decoded, the KV cache
    /// is automatically invalidated from the divergence point.
    ///
    /// Token-level operations (`push`, `pop`, `extend`, `truncate`) commit the text
    /// buffer so the next `push_str` starts a fresh region. This avoids ever needing
    /// to detokenize tokens back to text (which is lossy for byte-level BPE).
    ///
    /// Uses `add_special = false` and `parse_special = true`. To include BOS/EOS tokens,
    /// push them manually with `push(model.bos_token().unwrap())` before the first `push_str`.
    pub fn push_str(&self, text: &str) {
        let mut buf = self.text_buf().borrow_mut();
        buf.text.push_str(text);
        let new_tokens = self.ctx.model().tokenize(&buf.text, false, true);
        let token_start = buf.token_start;
        drop(buf);

        let mut tokens = self.tokens().borrow_mut();

        // Find common prefix within the text region
        let common_len = tokens[token_start..]
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let common_abs = token_start + common_len;

        // Invalidate KV cache if decoded tokens changed
        let n_decoded = self.n_decoded();
        if common_abs < n_decoded {
            unsafe {
                llama_memory_seq_rm(
                    self.ctx.get_memory(),
                    self.id,
                    common_abs as llama_pos,
                    -1,
                )
            };
        }

        // Replace tokens from divergence point
        tokens.truncate(common_abs);
        tokens.extend_from_slice(&new_tokens[common_len..]);
    }

    // === Inference Operations ===

    /// Returns the number of tokens that have been decoded (processed by the model).
    pub fn n_decoded(&self) -> usize {
        let pos = self.pos_max();
        if pos < 0 {
            0
        } else {
            (pos + 1) as usize
        }
    }

    /// Returns the number of tokens waiting to be decoded.
    pub fn n_pending(&self) -> usize {
        self.len().saturating_sub(self.n_decoded())
    }

    /// Decodes all pending tokens through the model.
    ///
    /// Automatically chunks the tokens according to the context's `n_batch` setting.
    /// Only requests logits for the very last token (needed for sampling).
    pub fn decode(&self) -> Result<(), ContextDecodeResult> {
        let tokens = self.tokens().borrow();
        let start_pos = self.n_decoded();
        if start_pos >= tokens.len() {
            return Ok(());
        }
        let pending: Vec<llama_token> = tokens[start_pos..].to_vec();
        drop(tokens);

        let n_batch = std::cmp::max(self.ctx.params().n_batch as usize, 1);

        for (chunk_idx, chunk) in pending.chunks(n_batch).enumerate() {
            let chunk_start = start_pos + chunk_idx * n_batch;
            let is_last_chunk = chunk_start + chunk.len() == start_pos + pending.len();

            let mut batch = Batch::init_token(chunk.len() as i32, 1);
            for (i, token) in chunk.iter().enumerate() {
                let is_last = is_last_chunk && i == chunk.len() - 1;
                common::batch_add(
                    batch.as_raw_mut(),
                    *token,
                    (chunk_start + i) as llama_pos,
                    vec![self.id],
                    is_last,
                )
                .expect("batch size exceeded");
            }

            self.ctx.decode(*batch)?;
        }

        Ok(())
    }

    /// Samples the next token using the given sampler.
    ///
    /// Reads logits from the last decoded position. The token is NOT automatically
    /// added to the sequence; call `push` to add it.
    pub fn sample<S: LlamaSampler>(&self, sampler: &S) -> llama_token {
        self.ctx.sample(sampler, -1)
    }

    /// Returns an iterator that generates tokens autoregressively.
    ///
    /// Each iteration decodes pending tokens, samples the next token, and yields it.
    /// The iterator stops when an end-of-generation token is produced.
    ///
    /// Use `.take(n)` to limit the number of generated tokens.
    pub fn generate<'s, S: LlamaSampler>(&'s self, sampler: &'s S) -> Generate<'s, 'ctx, 'a, S> {
        Generate {
            seq: self,
            sampler,
            done: false,
        }
    }
}

/// An iterator that generates tokens autoregressively from a sequence.
pub struct Generate<'seq, 'ctx, 'a, S: LlamaSampler> {
    seq: &'seq Sequence<'ctx, 'a>,
    sampler: &'seq S,
    done: bool,
}

impl<S: LlamaSampler> Iterator for Generate<'_, '_, '_, S> {
    type Item = llama_token;

    fn next(&mut self) -> Option<llama_token> {
        if self.done {
            return None;
        }
        if self.seq.decode().is_err() {
            self.done = true;
            return None;
        }
        let token = self.seq.sample(self.sampler);
        if self.seq.ctx.model().is_eog(token) {
            self.done = true;
            return None;
        }
        self.seq.push(token);
        Some(token)
    }
}

impl<S: LlamaSampler> std::iter::FusedIterator for Generate<'_, '_, '_, S> {}

impl fmt::Display for Sequence<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tokens = self.tokens().borrow();
        let model = self.ctx.model();
        for token in tokens.iter() {
            let piece = model.token_to_piece(*token).map_err(|_| fmt::Error)?;
            f.write_str(&piece)?;
        }
        Ok(())
    }
}

impl AddAssign<&str> for Sequence<'_, '_> {
    fn add_assign(&mut self, text: &str) {
        self.push_str(text);
    }
}


