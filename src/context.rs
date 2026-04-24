use llama_sys::*;
use std::ops::{Deref, DerefMut};

use crate::{LlamaSamplerPtr, Model};

pub struct ContextParams(llama_context_params);

// todo builder pattern
impl ContextParams {
    pub fn new() -> Self {
        Self(unsafe { llama_context_default_params() })
    }
}

impl Deref for ContextParams {
    type Target = llama_sys::llama_context_params;
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
    _model: std::marker::PhantomData<&'a Model>,
}

impl<'a> Context<'a> {
    pub fn new(model: &'a Model, params: ContextParams) -> Self {
        let ctx = unsafe { llama_init_from_model(**model, params.0) };
        Self {
            ctx,
            _model: std::marker::PhantomData,
        }
    }

    /// Process a batch of tokens.
    /// In contrast to llama_decode() - this call does not use KV cache.
    /// For encode-decoder contexts, processes the batch using the encoder.
    /// Can store the encoder output internally for later use by the decoder's cross-attention layers.
    /// 0 - success
    /// < 0 - error. the memory state is restored to the state before this call
    pub fn encode(&self, batch: llama_sys::llama_batch) -> i32 {
        unsafe { llama_encode(**self, batch) }
    }

    /// Decode a batch of tokens.
    /// Processes the batch using the decoder.
    /// 0 - success
    /// < 0 - error. the memory state is restored to the state before this call
    pub fn decode(&self, batch: llama_sys::llama_batch) -> i32 {
        unsafe { llama_decode(**self, batch) }
    }

    pub fn get_logits_ith(&self, idx: i32, n_vocab: usize) -> &[f32] {
        let ptr = unsafe { llama_get_logits_ith(**self, idx) };
        unsafe { std::slice::from_raw_parts(ptr, n_vocab) }
    }

    /// Sample and accept a token from the idx-th output of the last evaluation
    pub fn sample<S: LlamaSamplerPtr>(&mut self, sampler: &S, idx: i32) -> i32 {
        unsafe { llama_sys::llama_sampler_sample(sampler.as_ptr(), **self, idx) }
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}

impl<'a> Deref for Context<'a> {
    type Target = *mut llama_sys::llama_context;
    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl<'a> DerefMut for Context<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}
