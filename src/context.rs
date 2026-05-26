use llama_sys::*;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

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

// -- Shared context state, protected by Mutex --
// Both the actor and Sequence access this. The actor uses it for
// ReleaseSeq (llama_memory_seq_rm). Sequence uses it directly for
// the hot path (push/decode/sample/logits/kv ops).

pub(crate) struct ContextShared {
    pub ctx: *mut llama_context,
    pub batch: Batch,
    pub n_vocab: i32,
}

// SAFETY: Access is serialized by the Mutex. The raw pointer is only
// used while the Mutex is held.
unsafe impl Send for ContextShared {}

impl ContextShared {
    pub fn get_memory(&self) -> llama_memory_t {
        unsafe { llama_get_memory(self.ctx) }
    }

    pub fn decode_batch(&mut self) -> Result<(), DecodeError> {
        let result = unsafe { llama_decode(self.ctx, *self.batch) };
        match result {
            0 => Ok(()),
            1 => Err(DecodeError::SlotNotFound),
            2 => Err(DecodeError::Aborted),
            -1 => Err(DecodeError::InvalidInput),
            _ => Err(DecodeError::FatalError),
        }
    }

    pub fn get_logits_ith(&self, idx: i32) -> Option<Vec<f32>> {
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

impl Drop for ContextShared {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}

// -- Messages sent to the ContextActor --
// Only checkout/release/queries — hot-path ops go through the Mutex.

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

// -- The Actor (manages slot checkout only) --

pub(crate) struct ContextActor {
    shared: Arc<Mutex<ContextShared>>,
    checked_out: Vec<bool>,
}

// SAFETY: All fields are Send (Arc<Mutex<..>> is Send, Vec<bool> is Send).
unsafe impl Send for ContextActor {}

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
        let shared = self.shared.lock().unwrap();
        unsafe { llama_memory_seq_rm(shared.get_memory(), msg.seq_id, -1, -1) };
        drop(shared);
        if let Some(slot) = self.checked_out.get_mut(msg.seq_id as usize) {
            *slot = false;
        }
    }
}

impl Handler<GetNCtx> for ContextActor {
    fn handle(&mut self, _msg: GetNCtx, _ctx: &ActorContext<Self>) -> u32 {
        let shared = self.shared.lock().unwrap();
        unsafe { llama_n_ctx(shared.ctx) }
    }
}

impl Handler<CanShift> for ContextActor {
    fn handle(&mut self, _msg: CanShift, _ctx: &ActorContext<Self>) -> bool {
        let shared = self.shared.lock().unwrap();
        unsafe { llama_memory_can_shift(shared.get_memory()) }
    }
}

impl Handler<FreeSlots> for ContextActor {
    fn handle(&mut self, _msg: FreeSlots, _ctx: &ActorContext<Self>) -> usize {
        self.checked_out.iter().filter(|&&s| !s).count()
    }
}

impl Handler<GetPerf> for ContextActor {
    fn handle(&mut self, _msg: GetPerf, _ctx: &ActorContext<Self>) -> llama_perf_context_data {
        let shared = self.shared.lock().unwrap();
        unsafe { llama_perf_context(shared.ctx) }
    }
}

// -- Public handle to a running ContextActor --

use spawned_concurrency::threads::ActorRef;

/// Ref-counted inner: the actor stops when the last clone drops.
struct ContextInner {
    actor: ActorRef<ContextActor>,
    shared: Arc<Mutex<ContextShared>>,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        self.actor.context().stop();
    }
}

/// Handle to a running context actor. Clone + Send + Sync.
///
/// The actor manages sequence slot checkout/release. Hot-path operations
/// (push, decode, sample) go through a Mutex-guarded shared state for
/// near-zero overhead. The actor thread is stopped when the last clone
/// (including those held by Sequences) is dropped.
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
    pub(crate) fn actor(&self) -> &ActorRef<ContextActor> {
        &self.inner.actor
    }

    pub(crate) fn shared(&self) -> &Arc<Mutex<ContextShared>> {
        &self.inner.shared
    }

    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ()> {
        let ctx = unsafe { llama_init_from_model(model.as_mut_ptr(), params.0) };
        if ctx.is_null() {
            return Err(());
        }
        let n_seq_max = params.n_seq_max as usize;
        let n_vocab = model.n_tokens();

        let shared = Arc::new(Mutex::new(ContextShared {
            ctx,
            batch: Batch::init_token(1, params.n_seq_max as i32),
            n_vocab,
        }));

        let actor_inner = ContextActor {
            shared: shared.clone(),
            checked_out: vec![false; n_seq_max],
        };
        let actor = actor_inner.start();

        Ok(Self {
            inner: Arc::new(ContextInner { actor, shared }),
        })
    }

    pub fn sequence(&self) -> Option<crate::Sequence> {
        let seq_id = self.actor().request(CheckoutSeq).unwrap();
        seq_id.map(|id| crate::Sequence::new(self.clone(), id))
    }

    pub fn free_slots(&self) -> usize {
        self.actor().request(FreeSlots).unwrap()
    }

    pub fn n_ctx(&self) -> u32 {
        self.actor().request(GetNCtx).unwrap()
    }

    pub fn can_shift(&self) -> bool {
        self.actor().request(CanShift).unwrap()
    }

    pub fn perf(&self) -> llama_perf_context_data {
        self.actor().request(GetPerf).unwrap()
    }

    pub fn sample<S: crate::LlamaSampler>(&self, sampler: &S, _idx: i32) -> llama_token {
        let shared = self.shared().lock().unwrap();
        unsafe { llama_sampler_sample(sampler.as_ptr(), shared.ctx, -1) }
    }
}
