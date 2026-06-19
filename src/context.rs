use llama_sys::*;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use spawned_concurrency::protocol;
use spawned_concurrency::threads::{Actor, ActorRef, ActorStart, Context as ActorContext, Handler};
use spawned_concurrency::Response;

use crate::{common, Batch, Model};

// -- Params --

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ContextParams(llama_context_params);

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

// -- Error type --

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
    fn memory_seq_rm(&self, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos) -> Response<bool>;
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
        unsafe { llama_memory_seq_add(self.get_memory(), msg.seq_id, msg.p0, msg.p1, msg.delta) }
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
    can_shift: bool,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        self.actor.context().stop();
        let _ = self.actor.send(FreeSlots);
        self.actor.join();
    }
}

/// Handle to a running context actor. Clone + Send + Sync.
///
/// The actor thread is stopped when the last clone is dropped.
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
    pub(crate) fn actor(&self) -> &ActorRef<ContextActor> {
        &self.inner.actor
    }

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
        let can_shift = actor.can_shift().unwrap();

        Ok(Self {
            inner: Arc::new(ContextInner { actor, can_shift }),
        })
    }

    pub fn sequence(&self) -> Option<crate::Sequence> {
        let seq_id = self.actor().checkout_seq().unwrap();
        seq_id.map(|id| crate::Sequence::new(self.clone(), id))
    }

    pub fn free_slots(&self) -> usize {
        self.actor().free_slots().unwrap()
    }

    pub fn n_ctx(&self) -> u32 {
        self.actor().get_n_ctx().unwrap()
    }

    /// Whether this context's memory supports KV-shift (`Sequence::kv_shift`).
    ///
    /// Returns `false` for architectures llama.cpp does not implement shift for
    /// (e.g. MROPE/IMROPE multimodal models and STEP35). Cached at context
    /// creation since the underlying memory type cannot change.
    pub fn can_shift(&self) -> bool {
        self.inner.can_shift
    }

    pub fn perf(&self) -> llama_perf_context_data {
        self.actor().get_perf().unwrap()
    }
}
