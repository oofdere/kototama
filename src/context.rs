//! The inference context — a thread-safe handle around a `llama_context`.
//!
//! A [`Context`] is what a loaded [`Model`] turns into when you want to
//! actually run inference: it owns the KV cache, the decode [`Batch`], and a
//! pool of sequence slots. Create one with [`Context::new`]; clones are cheap
//! ([`Arc`]-backed) and can be moved across threads.
//!
//! ## Actor model
//!
//! Internally a [`Context`] is a handle to an actor (powered by
//! [`spawned_concurrency`]) that owns the raw `*mut llama_context`. All FFI
//! calls are serialized on the actor's thread, which is what makes the public
//! API safe to call from any thread without external synchronization.
//!
//! Callers normally don't see the actor — they just call methods on [`Context`]
//! or on a [`Sequence`](crate::Sequence) acquired via [`Context::sequence`],
//! and the round-trip happens under the hood.
//!
//! ## Sequences
//!
//! Tokens are pushed and sampled through a [`Sequence`](crate::Sequence), not
//! through the [`Context`] itself. Each context has `n_seq_max` sequence slots
//! (configurable via [`ContextParams`]); [`Context::sequence`] checks one out,
//! and dropping the [`Sequence`] releases it back to the pool.
//!
//! ```ignore
//! let backend = Backend::acquire();
//! let model = Model::load_from_file("model.gguf", ModelParams::new())?;
//! let ctx = Context::new(&model, &ContextParams::new())?;
//!
//! let mut seq = ctx.sequence().expect("no free slots");
//! seq.extend(&model.tokenize("Hello", true, true));
//! let next = seq.logits().unwrap().iter().enumerate()
//!     .max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;
//! ```

use llama_sys::*;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use spawned_concurrency::protocol;
use spawned_concurrency::threads::{Actor, ActorRef, ActorStart, Context as ActorContext, Handler};
use spawned_concurrency::Response;

use crate::{common, Batch, Model};

// -- Params --

/// Parameters for constructing a [`Context`].
///
/// Thin wrapper over `llama_context_params` from the C API. Construct one
/// with [`ContextParams::new`] (which calls `llama_context_default_params`),
/// tune fields like `n_ctx`, `n_batch`, or `n_seq_max` through
/// [`Deref`]/[`DerefMut`], then hand it to [`Context::new`].
///
/// ```ignore
/// let mut params = ContextParams::new();
/// params.n_ctx = 2048;
/// params.n_batch = 1;
/// params.n_seq_max = 4;
/// let ctx = Context::new(&model, &params)?;
/// ```
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ContextParams(llama_context_params);

impl ContextParams {
    /// Create a [`ContextParams`] seeded with llama.cpp's default values
    /// (`llama_context_default_params`).
    pub fn new() -> Self {
        Self(unsafe { llama_context_default_params() })
    }

    /// Borrow the inner `llama_context_params` as a `*const` pointer, for FFI
    /// calls that need a raw pointer to a read-only params struct.
    pub fn as_ptr(&self) -> *const llama_context_params {
        &self.0
    }

    /// Borrow the inner `llama_context_params` as a `*mut` pointer, for FFI
    /// calls that need a raw pointer to a mutable params struct.
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

// -- Error type --

/// Reasons a `llama_decode` call (driven by
/// [`Sequence::push`](crate::Sequence::push)) can fail.
///
/// Variants mirror the numeric return codes from upstream `llama_decode`:
///
/// - [`SlotNotFound`](DecodeError::SlotNotFound) — `1`: no KV slot available
///   for the requested sequence (KV cache full).
/// - [`Aborted`](DecodeError::Aborted) — `2`: the decode was aborted by an
///   abort callback installed on the underlying context.
/// - [`InvalidInput`](DecodeError::InvalidInput) — `-1`: the input batch was
///   rejected as malformed (e.g. zero tokens, oversized batch, bad sequence
///   id, or a `batch_add` that overflowed `n_seq_max`).
/// - [`FatalError`](DecodeError::FatalError) — any other non-zero return.
///   Treat as unrecoverable on this context.
#[derive(Debug, Clone)]
pub enum DecodeError {
    SlotNotFound,
    Aborted,
    InvalidInput,
    FatalError,
}

// -- Send-safe wrapper for raw sampler pointer --

/// SAFETY: The pointer is only dereferenced inside the actor's handler
/// while the caller is blocked on the synchronous request().
pub(crate) struct SamplerPtr(pub *mut llama_sampler);
unsafe impl Send for SamplerPtr {}

// -- Protocol: defines what messages the actor handles --
//
// The #[protocol] macro generates:
//   - A message struct per method (e.g. checkout_seq -> CheckoutSeq)
//   - impl Message for each struct
//   - A blanket impl of ContextProtocol for any ActorRef<A> that handles all messages
//
// All generated types live in the `context_protocol` module.

#[protocol]
pub(crate) trait ContextProtocol: Send + Sync {
    fn checkout_seq(&self) -> Response<Option<llama_seq_id>>;
    fn release_seq(&self, seq_id: llama_seq_id) -> Response<()>;
    fn push_token(
        &self,
        token: llama_token,
        pos: llama_pos,
        seq_id: llama_seq_id,
    ) -> Response<Result<Vec<f32>, DecodeError>>;
    fn sample_token(&self, sampler: SamplerPtr) -> Response<llama_token>;
    fn memory_seq_rm(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> Response<bool>;
    fn memory_seq_cp(
        &self,
        src: llama_seq_id,
        dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> Response<()>;
    fn memory_seq_add(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    ) -> Response<()>;
    fn memory_seq_pos_min(&self, seq_id: llama_seq_id) -> Response<llama_pos>;
    fn memory_seq_pos_max(&self, seq_id: llama_seq_id) -> Response<llama_pos>;
    fn get_n_ctx(&self) -> Response<u32>;
    fn can_shift(&self) -> Response<bool>;
    fn free_slots(&self) -> Response<usize>;
    fn get_perf(&self) -> Response<llama_perf_context_data>;
}

// -- The Actor --

pub(crate) struct ContextActor {
    ctx: *mut llama_context,
    batch: Batch,
    n_vocab: i32,
    checked_out: Vec<bool>,
}

unsafe impl Send for ContextActor {}

impl ContextActor {
    fn get_memory(&self) -> llama_memory_t {
        unsafe { llama_get_memory(self.ctx) }
    }

    fn decode_batch(&mut self) -> Result<(), DecodeError> {
        let result = unsafe { llama_decode(self.ctx, *self.batch) };
        match result {
            0 => Ok(()),
            1 => Err(DecodeError::SlotNotFound),
            2 => Err(DecodeError::Aborted),
            -1 => Err(DecodeError::InvalidInput),
            _ => Err(DecodeError::FatalError),
        }
    }

    fn get_logits_ith(&self, idx: i32) -> Option<Vec<f32>> {
        let ptr = unsafe { llama_get_logits_ith(self.ctx, idx) };
        if ptr.is_null() {
            return None;
        }
        if self.n_vocab <= 0 {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(ptr, self.n_vocab as usize) }.to_vec())
    }
}

impl Actor for ContextActor {}

impl Drop for ContextActor {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}

// -- Handlers: one per protocol method --

use context_protocol::*;

impl Handler<CheckoutSeq> for ContextActor {
    fn handle(&mut self, _msg: CheckoutSeq, _ctx: &ActorContext<Self>) -> Option<llama_seq_id> {
        for (i, slot) in self.checked_out.iter_mut().enumerate() {
            if !*slot {
                *slot = true;
                return Some(i as llama_seq_id);
            }
        }
        None
    }
}

impl Handler<ReleaseSeq> for ContextActor {
    fn handle(&mut self, msg: ReleaseSeq, _ctx: &ActorContext<Self>) {
        unsafe { llama_memory_seq_rm(self.get_memory(), msg.seq_id, -1, -1) };
        if let Some(slot) = self.checked_out.get_mut(msg.seq_id as usize) {
            *slot = false;
        }
    }
}

impl Handler<PushToken> for ContextActor {
    fn handle(
        &mut self,
        msg: PushToken,
        _ctx: &ActorContext<Self>,
    ) -> Result<Vec<f32>, DecodeError> {
        common::batch_clear(&mut self.batch);
        common::batch_add(&mut self.batch, msg.token, msg.pos, &[msg.seq_id], true)
            .map_err(|_| DecodeError::InvalidInput)?;
        self.decode_batch()?;
        self.get_logits_ith(0).ok_or(DecodeError::FatalError)
    }
}

impl Handler<SampleToken> for ContextActor {
    fn handle(&mut self, msg: SampleToken, _ctx: &ActorContext<Self>) -> llama_token {
        unsafe { llama_sampler_sample(msg.sampler.0, self.ctx, -1) }
    }
}

impl Handler<MemorySeqRm> for ContextActor {
    fn handle(&mut self, msg: MemorySeqRm, _ctx: &ActorContext<Self>) -> bool {
        unsafe { llama_memory_seq_rm(self.get_memory(), msg.seq_id, msg.p0, msg.p1) }
    }
}

impl Handler<MemorySeqCp> for ContextActor {
    fn handle(&mut self, msg: MemorySeqCp, _ctx: &ActorContext<Self>) {
        unsafe { llama_memory_seq_cp(self.get_memory(), msg.src, msg.dst, msg.p0, msg.p1) }
    }
}

impl Handler<MemorySeqAdd> for ContextActor {
    fn handle(&mut self, msg: MemorySeqAdd, _ctx: &ActorContext<Self>) {
        unsafe {
            llama_memory_seq_add(self.get_memory(), msg.seq_id, msg.p0, msg.p1, msg.delta)
        }
    }
}

impl Handler<MemorySeqPosMin> for ContextActor {
    fn handle(&mut self, msg: MemorySeqPosMin, _ctx: &ActorContext<Self>) -> llama_pos {
        unsafe { llama_memory_seq_pos_min(self.get_memory(), msg.seq_id) }
    }
}

impl Handler<MemorySeqPosMax> for ContextActor {
    fn handle(&mut self, msg: MemorySeqPosMax, _ctx: &ActorContext<Self>) -> llama_pos {
        unsafe { llama_memory_seq_pos_max(self.get_memory(), msg.seq_id) }
    }
}

impl Handler<GetNCtx> for ContextActor {
    fn handle(&mut self, _msg: GetNCtx, _ctx: &ActorContext<Self>) -> u32 {
        unsafe { llama_n_ctx(self.ctx) }
    }
}

impl Handler<CanShift> for ContextActor {
    fn handle(&mut self, _msg: CanShift, _ctx: &ActorContext<Self>) -> bool {
        unsafe { llama_memory_can_shift(self.get_memory()) }
    }
}

impl Handler<FreeSlots> for ContextActor {
    fn handle(&mut self, _msg: FreeSlots, _ctx: &ActorContext<Self>) -> usize {
        self.checked_out.iter().filter(|&&s| !s).count()
    }
}

impl Handler<GetPerf> for ContextActor {
    fn handle(&mut self, _msg: GetPerf, _ctx: &ActorContext<Self>) -> llama_perf_context_data {
        unsafe { llama_perf_context(self.ctx) }
    }
}

// -- Public handle --

struct ContextInner {
    actor: ActorRef<ContextActor>,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        self.actor.context().stop();
        let _ = self.actor.send(FreeSlots);
        self.actor.join();
    }
}

/// A thread-safe handle to a running inference context.
///
/// Wraps an `*mut llama_context` owned by a background actor thread; all FFI
/// calls are dispatched as messages, so this handle is safely
/// [`Clone`] + [`Send`] + [`Sync`]. The actor (and the underlying
/// `llama_context`) is freed when the last clone of the [`Context`] is
/// dropped.
///
/// Construct with [`Context::new`]. Tokens are pushed and sampled through a
/// [`Sequence`](crate::Sequence) acquired via [`Context::sequence`].
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
    pub(crate) fn actor(&self) -> &ActorRef<ContextActor> {
        &self.inner.actor
    }

    /// Create a new [`Context`] bound to `model`, using the supplied
    /// [`ContextParams`].
    ///
    /// Returns `Err(())` if llama.cpp's `llama_init_from_model` returns a
    /// null pointer (out of memory, incompatible params, etc.). llama.cpp
    /// logs the underlying reason to stderr.
    ///
    /// The returned handle holds an internal [`Arc`] to the actor; cloning is
    /// cheap. The actor thread is shut down when the last clone is dropped.
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ()> {
        let ctx = unsafe { llama_init_from_model(model.as_mut_ptr(), params.0) };
        if ctx.is_null() {
            return Err(());
        }
        let n_seq_max = params.n_seq_max as usize;
        let n_vocab = model.n_tokens();

        let actor_inner = ContextActor {
            ctx,
            batch: Batch::init_token(1, params.n_seq_max as i32),
            n_vocab,
            checked_out: vec![false; n_seq_max],
        };
        let actor = actor_inner.start();

        Ok(Self {
            inner: Arc::new(ContextInner { actor }),
        })
    }

    /// Check out a fresh [`Sequence`](crate::Sequence) from this context's
    /// pool of slots.
    ///
    /// Returns [`None`] if every slot up to `params.n_seq_max` is already
    /// checked out — drop an outstanding [`Sequence`](crate::Sequence) (or
    /// raise `n_seq_max` in [`ContextParams`]) and try again.
    ///
    /// The returned [`Sequence`](crate::Sequence) shares ownership of this
    /// [`Context`] (cheap [`Arc`] clone), so it can outlive the borrow used
    /// to acquire it.
    pub fn sequence(&self) -> Option<crate::Sequence> {
        let seq_id = self.actor().checkout_seq().unwrap();
        seq_id.map(|id| crate::Sequence::new(self.clone(), id))
    }

    /// Number of sequence slots that are currently free (not checked out).
    ///
    /// Equivalent to `params.n_seq_max` minus the number of live
    /// [`Sequence`](crate::Sequence) handles.
    pub fn free_slots(&self) -> usize {
        self.actor().free_slots().unwrap()
    }

    /// Maximum context length (`n_ctx`) configured on the underlying
    /// `llama_context`. Wraps `llama_n_ctx`.
    pub fn n_ctx(&self) -> u32 {
        self.actor().get_n_ctx().unwrap()
    }

    /// Whether the KV cache supports position shifting (used by
    /// [`Sequence::kv_shift`](crate::Sequence::kv_shift)).
    ///
    /// Wraps `llama_memory_can_shift`. Some architectures (notably some
    /// recurrent/state-space models) report `false`.
    pub fn can_shift(&self) -> bool {
        self.actor().can_shift().unwrap()
    }

    /// Snapshot of llama.cpp's performance counters for this context.
    ///
    /// Wraps `llama_perf_context`. Counters are only populated when
    /// `ContextParams::no_perf` is `false` (it defaults to `false`).
    pub fn perf(&self) -> llama_perf_context_data {
        self.actor().get_perf().unwrap()
    }

    /// Sample the next token using `sampler` against this context's most
    /// recent logits.
    ///
    /// Equivalent to [`Sequence::sample`](crate::Sequence::sample) but
    /// dispatched directly on the context, without consulting a specific
    /// sequence. `_idx` is currently ignored — the underlying call always
    /// targets index `-1` (the last decoded position). Prefer
    /// [`Sequence::sample`](crate::Sequence::sample) when you have a
    /// [`Sequence`](crate::Sequence) in hand.
    pub fn sample<S: crate::LlamaSampler>(&self, sampler: &S, _idx: i32) -> i32 {
        self.actor()
            .sample_token(SamplerPtr(sampler.as_ptr()))
            .unwrap()
    }
}
