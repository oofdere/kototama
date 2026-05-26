use llama_sys::*;
use std::ops::{Deref, DerefMut};

use spawned_concurrency::message::Message;
use spawned_concurrency::threads::{Actor, ActorStart, Context as ActorContext, Handler};

use crate::{common, Batch, Model};

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

#[derive(Debug, Clone)]
pub enum DecodeError {
    SlotNotFound,
    Aborted,
    InvalidInput,
    FatalError,
}

// -- Messages sent to the ContextActor --
// Each struct is a message; the actor handles them sequentially.

#[derive(Debug)]
pub(crate) struct CheckoutSeq;
impl Message for CheckoutSeq {
    type Result = Option<llama_seq_id>;
}

#[derive(Debug)]
pub(crate) struct ReleaseSeq {
    pub seq_id: llama_seq_id,
}
impl Message for ReleaseSeq {
    type Result = ();
}

pub(crate) struct PushToken {
    pub token: llama_token,
    pub pos: llama_pos,
    pub seq_id: llama_seq_id,
}
impl Message for PushToken {
    type Result = Result<Vec<f32>, DecodeError>;
}

pub(crate) struct SampleToken {
    pub sampler_ptr: *mut llama_sampler,
}
// SAFETY: The raw pointer is only used inside the actor's handler while
// the caller is blocked on the synchronous request(). The sampler is not
// accessed concurrently.
unsafe impl Send for SampleToken {}
impl Message for SampleToken {
    type Result = llama_token;
}

pub(crate) struct MemorySeqRm {
    pub seq_id: llama_seq_id,
    pub p0: llama_pos,
    pub p1: llama_pos,
}
impl Message for MemorySeqRm {
    type Result = bool;
}

pub(crate) struct MemorySeqCp {
    pub src: llama_seq_id,
    pub dst: llama_seq_id,
    pub p0: llama_pos,
    pub p1: llama_pos,
}
impl Message for MemorySeqCp {
    type Result = ();
}

pub(crate) struct MemorySeqAdd {
    pub seq_id: llama_seq_id,
    pub p0: llama_pos,
    pub p1: llama_pos,
    pub delta: llama_pos,
}
impl Message for MemorySeqAdd {
    type Result = ();
}

pub(crate) struct MemorySeqPosMin {
    pub seq_id: llama_seq_id,
}
impl Message for MemorySeqPosMin {
    type Result = llama_pos;
}

pub(crate) struct MemorySeqPosMax {
    pub seq_id: llama_seq_id,
}
impl Message for MemorySeqPosMax {
    type Result = llama_pos;
}

pub(crate) struct GetNCtx;
impl Message for GetNCtx {
    type Result = u32;
}

pub(crate) struct CanShift;
impl Message for CanShift {
    type Result = bool;
}

pub(crate) struct FreeSlots;
impl Message for FreeSlots {
    type Result = usize;
}

pub(crate) struct GetPerf;
impl Message for GetPerf {
    type Result = llama_perf_context_data;
}

// -- The Actor --

pub(crate) struct ContextActor {
    ctx: *mut llama_context,
    batch: Batch,
    n_vocab: i32,
    checked_out: Vec<bool>,
}

// SAFETY: The ContextActor runs on a dedicated thread. The *mut llama_context
// is only ever accessed from that thread, serialized through the actor mailbox.
unsafe impl Send for ContextActor {}

impl ContextActor {
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ()> {
        let ctx = unsafe { llama_init_from_model(model.as_mut_ptr(), params.0) };
        if ctx.is_null() {
            return Err(());
        }
        let n_seq_max = params.n_seq_max as usize;
        let n_vocab = model.n_tokens();
        Ok(Self {
            ctx,
            batch: Batch::init_token(1, params.n_seq_max as i32),
            n_vocab,
            checked_out: vec![false; n_seq_max],
        })
    }

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
        self.get_logits_ith(0)
            .ok_or(DecodeError::FatalError)
    }
}

impl Handler<SampleToken> for ContextActor {
    fn handle(&mut self, msg: SampleToken, _ctx: &ActorContext<Self>) -> llama_token {
        unsafe { llama_sampler_sample(msg.sampler_ptr, self.ctx, -1) }
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

impl Drop for ContextActor {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}

// -- Public handle to a running ContextActor --

use spawned_concurrency::threads::ActorRef;

/// Handle to a running context actor. Clone + Send + Sync.
///
/// Create via `Context::new()`. The underlying llama_context lives on
/// a dedicated thread; all operations are serialized through the actor mailbox.
#[derive(Clone)]
pub struct Context {
    pub(crate) actor: ActorRef<ContextActor>,
}

impl Context {
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ()> {
        let inner = ContextActor::new(model, params)?;
        let actor = inner.start();
        Ok(Self { actor })
    }

    pub fn sequence(&self) -> Option<crate::Sequence> {
        let seq_id = self.actor.request(CheckoutSeq).unwrap();
        seq_id.map(|id| crate::Sequence::new(self.clone(), id))
    }

    pub fn free_slots(&self) -> usize {
        self.actor.request(FreeSlots).unwrap()
    }

    pub fn n_ctx(&self) -> u32 {
        self.actor.request(GetNCtx).unwrap()
    }

    pub fn can_shift(&self) -> bool {
        self.actor.request(CanShift).unwrap()
    }

    pub fn perf(&self) -> llama_perf_context_data {
        self.actor.request(GetPerf).unwrap()
    }

    pub fn sample<S: crate::LlamaSampler>(&self, sampler: &S, _idx: i32) -> llama_token {
        self.actor
            .request(SampleToken {
                sampler_ptr: sampler.as_ptr(),
            })
            .unwrap()
    }
}
