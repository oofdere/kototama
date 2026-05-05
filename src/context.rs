use llama_sys::*;
use std::{
    ops::{Deref, DerefMut},
    vec,
};

use crate::{LlamaSampler, Model, Sequence};

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ContextParams(llama_context_params);

// todo builder pattern
impl ContextParams {
    pub fn new() -> Self {
        Self(unsafe { llama_context_default_params() })
    }

    pub fn as_ptr(&self) -> *const llama_context_params {
        &self.0
    }

    pub fn as_mut_ptr(&mut self) -> *mut llama_context_params {
        &mut self.0
    }
}

impl Deref for ContextParams {
    type Target = llama_context_params;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ContextParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct Context<'a> {
    ctx: *mut llama_context,
    params: &'a ContextParams,
    pub(crate) tokens: Box<[Vec<llama_token>]>,
    model: &'a Model,
}

#[derive(Debug)]
pub enum ContextDecodeResult {
    // could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    SlotNotFound,
    // aborted (processed ubatches will remain in the context's memory)
    Aborted,
    // invalid input batch
    InvalidInput,
    // fatal error (processed ubatches will remain in the context's memory)
    FatalError,
}

impl<'a> Context<'a> {
    pub fn new(model: &'a Model, params: &'a ContextParams) -> Result<Self, ()> {
        let ctx = unsafe { llama_init_from_model(model.as_ptr() as *mut _, params.0) };

        if ctx.is_null() {
            return Err(());
        }

        let ctx = Self {
            ctx,
            tokens: vec![Vec::new(); params.n_seq_max as usize].into_boxed_slice(),
            params,
            model,
        };

        Ok(ctx)
    }

    /// get a sequence by index
    pub fn sequence(&self, index: llama_seq_id) -> Sequence<'_, 'a> {
        Sequence::new(self, index)
    }

    /// get a mutable reference to the token list for a sequence
    pub fn tokens_mut(&mut self, seq_id: llama_seq_id) -> &mut Vec<llama_token> {
        &mut self.tokens[seq_id as usize]
    }

    /// get a reference back to the model this context is tied to
    pub fn model(&self) -> &'a Model {
        self.model
    }

    pub fn params(&self) -> &'a ContextParams {
        self.params
    }

    pub fn as_ptr(&self) -> *const llama_context {
        self.ctx
    }

    pub fn as_mut_ptr(&mut self) -> *mut llama_context {
        self.ctx
    }

    pub fn get_memory(&self) -> llama_memory_t {
        unsafe { llama_get_memory(self.ctx) }
    }

    /// Process a batch of tokens.
    /// In contrast to llama_decode() - this call does not use KV cache.
    /// For encode-decoder contexts, processes the batch using the encoder.
    /// Can store the encoder output internally for later use by the decoder's cross-attention layers.
    /// 0 - success
    /// < 0 - error. the memory state is restored to the state before this call
    pub fn encode(&self, batch: llama_sys::llama_batch) -> i32 {
        unsafe { llama_encode(self.ctx, batch) }
    }

    /// Process a batch of tokens. Requires the context to have a memory.
    /// For encode-decoder contexts, processes the batch using the decoder.
    ///
    /// Positive return values do not mean a fatal error, but rather a warning.
    /// Upon fatal-error or abort, the ubatches that managed to be processed will remain
    /// in the memory state of the context. To handle this correctly, query the memory
    /// state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max().
    /// Upon other return values, the memory state is restored to the state before this call.
    ///
    /// Return codes:
    /// - 0 - success
    /// - 1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    /// - 2 - aborted (processed ubatches will remain in the context's memory)
    /// - -1 - invalid input batch
    /// - < -1 - fatal error (processed ubatches will remain in the context's memory)
    pub fn decode(&self, batch: llama_sys::llama_batch) -> Result<(), ContextDecodeResult> {
        let result = unsafe { llama_decode(self.ctx, batch) };
        match result {
            0 => Ok(()),
            1 => Err(ContextDecodeResult::SlotNotFound),
            2 => Err(ContextDecodeResult::Aborted),
            -1 => Err(ContextDecodeResult::InvalidInput),
            _ => Err(ContextDecodeResult::FatalError),
        }
    }

    pub fn get_logits_ith(&self, idx: i32, n_vocab: usize) -> &[f32] {
        let ptr = unsafe { llama_get_logits_ith(self.ctx, idx) };
        unsafe { std::slice::from_raw_parts(ptr, n_vocab) }
    }

    /// Sample and accept a token from the idx-th output of the last evaluation
    pub fn sample<S: LlamaSampler>(&mut self, sampler: &S, idx: i32) -> i32 {
        unsafe { llama_sampler_sample(sampler.as_ptr(), self.ctx, idx) }
    }

    pub fn perf(&self) -> llama_perf_context_data {
        unsafe { llama_perf_context(self.ctx) }
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_n_ctx(self.ctx) }
    }

    pub fn can_shift(&self) -> bool {
        unsafe { llama_memory_can_shift(self.get_memory()) }
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}
