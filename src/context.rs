//! Context creation and parameter types.
//!
//! A [`Context`] owns a `llama_context` and the decode batch that feeds it.
//! It is a plain synchronous handle: every call runs on the caller's thread.
//! The state is shared through `Arc<Mutex<_>>`, so `Context` is cheap to clone
//! and safe to use from several threads — calls simply serialize on the lock.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};

use llama_sys::*;

use crate::{Batch, Error, Model, Sequence};

// -- Params --

/// Parameters for [`Context::new`], mirroring `llama_context_params`.
///
/// Derefs to the raw `llama_context_params`, so fields can be set directly:
///
/// ```
/// use rusty_llama::ContextParams;
///
/// let mut params = ContextParams::new();
/// params.n_ctx = 2048;
/// ```
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ContextParams(llama_context_params);

impl ContextParams {
    /// Defaults from `llama_context_default_params`.
    pub fn new() -> Self {
        Self(unsafe { llama_context_default_params() })
    }
}

impl Default for ContextParams {
    fn default() -> Self {
        Self::new()
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

// -- Shared state --

/// Everything that touches the raw `llama_context`, guarded by one mutex.
///
/// SAFETY: `ctx` is only ever dereferenced while the mutex is held, so the
/// `llama_context` is never used from two threads at once. llama.cpp allows a
/// context to be used from any thread as long as calls are serialized.
pub(crate) struct ContextState {
    ctx: *mut llama_context,
    batch: Batch,
    n_vocab: usize,
    /// One entry per sequence id: `true` while a [`Sequence`] holds it.
    checked_out: Vec<bool>,
}

unsafe impl Send for ContextState {}

impl ContextState {
    fn memory(&self) -> llama_memory_t {
        unsafe { llama_get_memory(self.ctx) }
    }

    /// Decode one token of sequence `seq_id` at position `pos` and return the
    /// logits it produced.
    ///
    /// Each call resets the batch to hold exactly this token with its logits
    /// requested, so the output for the token lands at index 0.
    pub(crate) fn decode_token(
        &mut self,
        token: llama_token,
        pos: llama_pos,
        seq_id: llama_seq_id,
    ) -> Result<Vec<f32>, Error> {
        self.batch.clear();
        self.batch.add(token, pos, &[seq_id], true)?;

        let status = unsafe { llama_decode(self.ctx, *self.batch) };
        match status {
            0 => {}
            1 => return Err(Error::ContextFull),
            2 => return Err(Error::Aborted),
            -1 => return Err(Error::InvalidBatch),
            other => return Err(Error::Fatal(other)),
        }

        let ptr = unsafe { llama_get_logits_ith(self.ctx, 0) };
        if ptr.is_null() || self.n_vocab == 0 {
            return Err(Error::MissingLogits);
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) }.to_vec())
    }

    /// Free a checked-out sequence slot and drop all of its tokens from the
    /// context memory.
    pub(crate) fn release_seq(&mut self, seq_id: llama_seq_id) {
        unsafe { llama_memory_seq_rm(self.memory(), seq_id, -1, -1) };
        if let Some(slot) = self.checked_out.get_mut(seq_id as usize) {
            *slot = false;
        }
    }

    /// Remove the tokens of `seq_id` with positions in `[p0, p1)`.
    ///
    /// `p0 < 0` means "from the start", `p1 < 0` means "to the end".
    /// Returns `false` when the range cannot be removed as a whole.
    pub(crate) fn kv_remove(&mut self, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos) -> bool {
        unsafe { llama_memory_seq_rm(self.memory(), seq_id, p0, p1) }
    }

    /// Copy the tokens of `src` with positions in `[p0, p1)` over to `dst`.
    pub(crate) fn kv_copy(
        &mut self,
        src: llama_seq_id,
        dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) {
        unsafe { llama_memory_seq_cp(self.memory(), src, dst, p0, p1) };
    }

    /// Shift the positions of `seq_id`'s tokens in `[p0, p1)` by `delta`.
    pub(crate) fn kv_shift(
        &mut self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    ) {
        unsafe { llama_memory_seq_add(self.memory(), seq_id, p0, p1, delta) };
    }

    /// Smallest position present in memory for `seq_id`, or `-1` when empty.
    pub(crate) fn kv_pos_min(&self, seq_id: llama_seq_id) -> llama_pos {
        unsafe { llama_memory_seq_pos_min(self.memory(), seq_id) }
    }

    /// Largest position present in memory for `seq_id`, or `-1` when empty.
    pub(crate) fn kv_pos_max(&self, seq_id: llama_seq_id) -> llama_pos {
        unsafe { llama_memory_seq_pos_max(self.memory(), seq_id) }
    }
}

impl Drop for ContextState {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}

struct ContextInner {
    state: Mutex<ContextState>,
    /// Kept so the `llama_model` outlives the context: `llama_context` only
    /// borrows the model internally, so dropping the last `Model` handle first
    /// would leave the context pointing at freed memory. Cloning the handle is
    /// an Arc bump.
    _model: Model,
}

/// A loaded inference context. Cheap to clone; all clones share one
/// `llama_context`.
///
/// All methods are synchronous and safe to call from multiple threads: they
/// serialize on an internal lock. Create sequences with [`Context::sequence`].
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
    /// Create a context for `model`. Fails if llama.cpp rejects the parameters.
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, Error> {
        let ctx = unsafe { llama_init_from_model(model.as_mut_ptr(), params.0) };
        if ctx.is_null() {
            return Err(Error::ContextCreateFailed);
        }

        let state = ContextState {
            ctx,
            batch: Batch::init_token(1, params.n_seq_max as i32),
            n_vocab: model.n_tokens() as usize,
            checked_out: vec![false; params.n_seq_max as usize],
        };

        Ok(Self {
            inner: Arc::new(ContextInner {
                state: Mutex::new(state),
                _model: model.clone(),
            }),
        })
    }

    /// Check out a free sequence slot. Returns `None` when every slot is taken.
    ///
    /// The slot is returned to the pool when the [`Sequence`] is dropped.
    pub fn sequence(&self) -> Option<Sequence> {
        let mut state = self.lock();
        let id = state
            .checked_out
            .iter_mut()
            .position(|slot| !*slot)
            .map(|i| i as llama_seq_id)?;
        state.checked_out[id as usize] = true;
        Some(Sequence::new(self.clone(), id))
    }

    /// Number of sequence slots that are currently free.
    pub fn free_slots(&self) -> usize {
        let state = self.lock();
        state.checked_out.iter().filter(|&&s| !s).count()
    }

    /// Context size in tokens (`llama_n_ctx`).
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_n_ctx(self.lock().ctx) }
    }

    /// Whether the context's memory supports position shifting.
    pub fn can_shift(&self) -> bool {
        let state = self.lock();
        unsafe { llama_memory_can_shift(state.memory()) }
    }

    /// Performance counters for this context (`llama_perf_context`).
    pub fn perf(&self) -> llama_perf_context_data {
        unsafe { llama_perf_context(self.lock().ctx) }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, ContextState> {
        // Recover from a poisoned lock: a panic while holding it does not
        // touch the llama_context (the C state is unaffected), so there is
        // nothing to roll back.
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
