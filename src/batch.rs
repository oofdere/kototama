use llama_sys::*;
use std::ops::Deref;

pub struct Batch {
    batch: llama_batch,
    owns_memory: bool,
    allocated: usize,
}

impl Batch {
    /// Initialize a new batch with the given parameters.
    /// n_tokens_alloc: maximum number of tokens the batch can hold
    /// embd: if non-zero, allocates embedding buffer of size n_tokens_alloc * embd
    /// n_seq_max: maximum number of sequence IDs per token
    pub fn init(n_tokens_alloc: i32, embd: i32, n_seq_max: i32) -> Self {
        let batch = unsafe { llama_batch_init(n_tokens_alloc, embd, n_seq_max) };
        Self {
            batch,
            owns_memory: true,
            allocated: n_tokens_alloc as usize,
        }
    }

    /// Create a batch from a single token (simple case for single-token generation)
    /// Note: This creates a batch that doesn't own the token memory - the token slice must outlive the batch
    pub fn from_one(token: &i32) -> Self {
        let batch = unsafe { llama_batch_get_one(token as *const i32 as *mut i32, 1) };
        Self {
            batch,
            owns_memory: false,
            allocated: 1,
        }
    }

    /// Create a batch from a slice of tokens
    /// Note: This creates a batch that doesn't own the token memory - the token slice must outlive the batch
    pub fn from_tokens(tokens: &[i32]) -> Self {
        let batch =
            unsafe { llama_batch_get_one(tokens.as_ptr() as *mut i32, tokens.len() as i32) };
        Self {
            batch,
            owns_memory: false,
            allocated: tokens.len(),
        }
    }

    /// Add a token to the batch at a specific position
    /// This requires the batch to have been initialized with sufficient capacity
    pub fn add_token(&mut self, token: i32, pos: i32, seq_id: i32, logits: bool) {
        if (self.batch.n_tokens as usize) < self.capacity() {
            unsafe {
                // Set token
                if !self.batch.token.is_null() {
                    *self.batch.token.add(self.batch.n_tokens as usize) = token;
                }
                // Set position
                if !self.batch.pos.is_null() {
                    *self.batch.pos.add(self.batch.n_tokens as usize) = pos;
                }
                // Set sequence ID
                if !self.batch.seq_id.is_null()
                    && !self
                        .batch
                        .seq_id
                        .add(self.batch.n_tokens as usize)
                        .is_null()
                {
                    let seq_ids = *self.batch.seq_id.add(self.batch.n_tokens as usize);
                    if !seq_ids.is_null() && !self.batch.n_seq_id.is_null() {
                        *self.batch.n_seq_id.add(self.batch.n_tokens as usize) = 1;
                        *seq_ids = seq_id;
                    }
                }
                // Set logits flag
                if !self.batch.logits.is_null() {
                    *self.batch.logits.add(self.batch.n_tokens as usize) =
                        if logits { 1 } else { 0 };
                }
                self.batch.n_tokens += 1;
            }
        }
    }

    /// Clear the batch (reset n_tokens to 0)
    pub fn clear(&mut self) {
        self.batch.n_tokens = 0;
    }

    /// Get the number of tokens currently in the batch
    pub fn n_tokens(&self) -> i32 {
        self.batch.n_tokens
    }

    /// Get the capacity of the batch (maximum number of tokens)
    pub fn capacity(&self) -> usize {
        self.allocated
    }
}

impl Deref for Batch {
    type Target = llama_batch;

    fn deref(&self) -> &Self::Target {
        &self.batch
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if self.owns_memory {
            unsafe {
                llama_batch_free(self.batch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_init() {
        let batch = Batch::init(512, 0, 1);
        assert_eq!(batch.n_tokens(), 0);
    }

    #[test]
    fn test_batch_from_one() {
        let token = 42i32;
        let batch = Batch::from_one(&token);
        assert_eq!(batch.n_tokens(), 1);
    }

    #[test]
    fn test_batch_from_tokens() {
        let tokens = vec![1, 2, 3, 4, 5];
        let batch = Batch::from_tokens(&tokens);
        assert_eq!(batch.n_tokens(), 5);
    }

    #[test]
    fn test_batch_clear() {
        let tokens = vec![1, 2, 3];
        let mut batch = Batch::from_tokens(&tokens);
        assert_eq!(batch.n_tokens(), 3);
        batch.clear();
        assert_eq!(batch.n_tokens(), 0);
    }
}
