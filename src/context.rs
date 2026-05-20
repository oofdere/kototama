use llama_sys::*;
use std::{
    cell::Cell,
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
    pub(crate) checked_out: Box<[Cell<bool>]>,
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
            checked_out: vec![Cell::new(false); params.n_seq_max as usize].into_boxed_slice(),
            params,
            model,
        };

        Ok(ctx)
    }

    /// Get the next available sequence.
    ///
    /// Automatically assigns the first unchecked sequence id.
    /// Returns `None` if all sequences are checked out.
    pub fn sequence(&self) -> Option<Sequence<'_, 'a>> {
        for (i, slot) in self.checked_out.iter().enumerate() {
            if !slot.get() {
                slot.set(true);
                return Some(Sequence::new(self, i as llama_seq_id));
            }
        }
        None
    }

    // get the number of slots available to claim through `sequence()`
    pub fn free_slots(&self) -> usize {
        self.checked_out.iter().filter(|slot| !slot.get()).count()
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

    /// Get the logits for the `idx`-th output of the last evaluation.
    ///
    /// Returns `None` if no logits are available for that index — this happens
    /// when `idx` is out of range or logits were not requested for that
    /// position in the batch. The underlying llama.cpp API signals these cases
    /// by returning a null pointer, so checking for null here is required to
    /// avoid undefined behavior from constructing a slice over a null pointer.
    ///
    /// **Debug builds of llama.cpp:** when `NDEBUG` is not defined (the
    /// default for `cargo build`/`cargo test`), llama.cpp aborts the process
    /// via `GGML_ABORT` on an invalid `idx` instead of returning a null
    /// pointer. The `None` branch is only observable in release builds. The
    /// null check here is still load-bearing — it protects callers in release
    /// mode from a soundness hole.
    ///
    /// The logits are copied into an owned `Vec` rather than handed back as a
    /// borrowed slice. The buffer behind `llama_get_logits_ith` is owned by the
    /// `llama_context` and is overwritten in place by `encode`/`decode`. Those
    /// methods only take `&self` (so several `Sequence`s can share one
    /// `Context`), which means a borrowed `&[f32]` tied to `&self` could be
    /// mutated through the FFI while still live — a data race / aliasing
    /// violation, i.e. undefined behavior reachable from safe code. Returning
    /// an owned copy severs that alias and keeps this a sound safe API.
    ///
    /// The length is derived from the model's vocabulary size, which is the
    /// layout llama.cpp uses for the logits buffer.
    pub fn get_logits_ith(&self, idx: i32) -> Option<Vec<f32>> {
        let ptr = unsafe { llama_get_logits_ith(self.ctx, idx) };
        if ptr.is_null() {
            return None;
        }
        let n_vocab = self.model.n_tokens();
        if n_vocab <= 0 {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(ptr, n_vocab as usize) }.to_vec())
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
