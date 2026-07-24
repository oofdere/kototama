use llama_sys::*;
use std::collections::HashMap;
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread::JoinHandle;

use crate::{common, Batch, Model, Token};

/// Parameters used to create a [`Context`].
///
/// This is a transparent wrapper around the pinned llama.cpp
/// [`llama_context_params`] structure. Most fields are plain configuration
/// values, but llama.cpp also exposes callbacks, callback user-data pointers,
/// and an experimental `samplers` pointer through this structure.
///
/// [`Context::new`] is the safe constructor. It accepts only the subset of
/// parameters that does not reference external state: evaluation callbacks,
/// abort callbacks, their user-data pointers, and native backend sampler
/// configurations must all be absent. If any of those fields are set,
/// [`Context::new`] returns [`ContextInitError::ThreadUnsafeParams`].
///
/// [`Context::new_unchecked`] exists for advanced FFI integrations that need
/// those raw fields and can uphold their cross-thread lifetime and exclusivity
/// requirements.
///
/// # Two different sampler APIs
///
/// `llama_context_params::samplers` is llama.cpp's experimental native
/// backend-sampling configuration. It points to per-sequence native
/// `llama_sampler` chains that llama.cpp may access from the context worker.
/// It is unrelated to this crate's [`crate::Sampler`] trait. Rust samplers are
/// applied explicitly with [`crate::Sequence::sample`] to an owned logits
/// snapshot on the caller's thread and are safe to use with [`Context::new`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ContextParams(pub(crate) llama_context_params);

impl ContextParams {
    pub fn new() -> Self {
        Self(unsafe { llama_context_default_params() })
    }

    pub fn as_ptr(&self) -> *const llama_context_params {
        &self.0
    }

    /// Returns a mutable pointer to the underlying llama.cpp parameter struct.
    ///
    /// # Safety
    /// The caller must preserve all invariants of [`llama_context_params`]. In
    /// particular, raw callback, user-data, sampler, and context pointers must
    /// remain valid for every native operation that can observe them.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut llama_context_params {
        &mut self.0
    }

    fn has_worker_thread_state(&self) -> bool {
        self.cb_eval.is_some()
            || !self.cb_eval_user_data.is_null()
            || self.abort_callback.is_some()
            || !self.abort_callback_data.is_null()
            || !self.samplers.is_null()
            || self.n_samplers != 0
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextError {
    WorkerStopped,
}

/// Errors that can occur while creating a [`Context`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextInitError {
    /// The safe constructor was given callbacks, callback user data, or native
    /// backend sampler pointers that may reference thread-affine external state.
    /// Use pointer-free parameters or, after auditing the requirements, call
    /// [`Context::new_unchecked`].
    ThreadUnsafeParams,
    /// The dedicated context worker thread could not be spawned.
    ThreadSpawnFailed,
    /// The worker stopped before reporting whether native initialization
    /// succeeded.
    WorkerStopped,
    /// `llama_init_from_model` returned a null context pointer.
    NativeInitFailed,
}

pub(crate) struct SequenceSnapshot {
    pub tokens: Vec<Token>,
    pub token_snapshot: Option<Arc<[Token]>>,
    pub logits: Option<Arc<[f32]>>,
}

impl SequenceSnapshot {
    fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            token_snapshot: Some(Arc::from([])),
            logits: None,
        }
    }
}

pub(crate) type SharedSequenceSnapshot = Arc<RwLock<SequenceSnapshot>>;

type Reply<T> = oneshot::Sender<Result<T, ContextError>>;

enum Command {
    CheckoutSeq {
        request_id: u64,
        reply: Reply<Option<SequenceReservation>>,
    },
    CommitCheckout {
        request_id: u64,
    },
    CancelCheckout {
        request_id: u64,
    },
    ReleaseSeq {
        seq_id: llama_seq_id,
    },
    PushToken {
        token: llama_token,
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
        reply: Reply<Result<Arc<[f32]>, DecodeError>>,
    },
    DecodeLast {
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
        reply: Reply<Result<Option<Arc<[f32]>>, DecodeError>>,
    },
    Pop {
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
        reply: Reply<Option<llama_token>>,
    },
    Remove {
        seq_id: llama_seq_id,
        start: llama_pos,
        end: llama_pos,
        snapshot: SharedSequenceSnapshot,
        reply: Reply<bool>,
    },
    Copy {
        src: llama_seq_id,
        dst: llama_seq_id,
        start: llama_pos,
        end: llama_pos,
        src_snapshot: SharedSequenceSnapshot,
        dst_snapshot: SharedSequenceSnapshot,
        reply: Reply<bool>,
    },
    MemorySeqRm {
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        snapshot: SharedSequenceSnapshot,
        reply: Reply<bool>,
    },
    MemorySeqAdd {
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
        snapshot: SharedSequenceSnapshot,
        reply: Reply<()>,
    },
    MemorySeqPosMin {
        seq_id: llama_seq_id,
        reply: Reply<llama_pos>,
    },
    MemorySeqPosMax {
        seq_id: llama_seq_id,
        reply: Reply<llama_pos>,
    },
    GetNCtx {
        reply: Reply<u32>,
    },
    CanShift {
        reply: Reply<bool>,
    },
    FreeSlots {
        reply: Reply<usize>,
    },
    GetPerf {
        reply: Reply<llama_perf_context_data>,
    },
    Shutdown,
}

pub(crate) struct SequenceReservation {
    id: llama_seq_id,
    snapshot: SharedSequenceSnapshot,
}

impl SequenceReservation {
    fn new(id: llama_seq_id, snapshot: SharedSequenceSnapshot) -> Self {
        Self { id, snapshot }
    }

    pub(crate) fn into_parts(self) -> (llama_seq_id, SharedSequenceSnapshot) {
        (self.id, self.snapshot)
    }
}

struct WorkerInit {
    model: Model,
    params: ContextParams,
}

// SAFETY: WorkerInit is private and is constructed only by Context::new,
// which rejects every callback or raw pointer-bearing parameter, or by
// Context::new_unchecked, whose safety contract requires all referenced state to
// be valid and safe to access on the worker thread for the full context lifetime.
unsafe impl Send for WorkerInit {}

struct Worker {
    ctx: *mut llama_context,
    _model: Model,
    batch: Batch,
    n_vocab: i32,
    checked_out: Vec<bool>,
    pending_checkouts: HashMap<u64, llama_seq_id>,
}

impl Worker {
    fn new(init: WorkerInit) -> Result<Self, ContextInitError> {
        let ctx = unsafe { llama_init_from_model(init.model.as_mut_ptr(), init.params.0) };
        if ctx.is_null() {
            return Err(ContextInitError::NativeInitFailed);
        }

        let n_seq_max = init.params.n_seq_max as usize;
        let n_vocab = init.model.n_tokens();

        Ok(Self {
            ctx,
            _model: init.model,
            batch: Batch::init_token(1, init.params.n_seq_max as i32),
            n_vocab,
            checked_out: vec![false; n_seq_max],
            pending_checkouts: HashMap::new(),
        })
    }

    fn get_memory(&self) -> llama_memory_t {
        unsafe { llama_get_memory(self.ctx) }
    }

    fn decode_batch(&mut self) -> Result<(), DecodeError> {
        match unsafe { llama_decode(self.ctx, *self.batch) } {
            0 => Ok(()),
            1 => Err(DecodeError::SlotNotFound),
            2 => Err(DecodeError::Aborted),
            -1 => Err(DecodeError::InvalidInput),
            _ => Err(DecodeError::FatalError),
        }
    }

    fn get_logits_ith(&self, idx: i32) -> Option<Arc<[f32]>> {
        let ptr = unsafe { llama_get_logits_ith(self.ctx, idx) };
        if ptr.is_null() || self.n_vocab <= 0 {
            return None;
        }

        Some(Arc::from(unsafe {
            std::slice::from_raw_parts(ptr, self.n_vocab as usize)
        }))
    }

    fn push_token(
        &mut self,
        token: llama_token,
        pos: llama_pos,
        seq_id: llama_seq_id,
    ) -> Result<Arc<[f32]>, DecodeError> {
        common::batch_clear(&mut self.batch);
        common::batch_add(&mut self.batch, token, pos, &[seq_id], true)
            .map_err(|_| DecodeError::InvalidInput)?;
        self.decode_batch()?;
        self.get_logits_ith(0).ok_or(DecodeError::FatalError)
    }

    fn checkout_sequence(&mut self, request_id: u64) -> Option<SequenceReservation> {
        let id = self.checked_out.iter().position(|used| !*used)?;
        self.checked_out[id] = true;
        let seq_id = id as llama_seq_id;
        self.pending_checkouts.insert(request_id, seq_id);
        Some(SequenceReservation::new(
            seq_id,
            Arc::new(RwLock::new(SequenceSnapshot::empty())),
        ))
    }

    fn release_sequence(&mut self, seq_id: llama_seq_id) {
        unsafe { llama_memory_seq_rm(self.get_memory(), seq_id, -1, -1) };
        if let Some(slot) = self.checked_out.get_mut(seq_id as usize) {
            *slot = false;
        }
    }

    fn run(mut self, commands: mpsc::Receiver<Command>) {
        while let Ok(command) = commands.recv() {
            match command {
                Command::CheckoutSeq { request_id, reply } => {
                    let result = self.checkout_sequence(request_id);
                    let allocated = result.as_ref().map(|reservation| reservation.id);
                    if reply.send(Ok(result)).is_err() {
                        self.pending_checkouts.remove(&request_id);
                        if let Some(seq_id) = allocated {
                            self.release_sequence(seq_id);
                        }
                    }
                }
                Command::CommitCheckout { request_id } => {
                    self.pending_checkouts.remove(&request_id);
                }
                Command::CancelCheckout { request_id } => {
                    if let Some(seq_id) = self.pending_checkouts.remove(&request_id) {
                        self.release_sequence(seq_id);
                    }
                }
                Command::ReleaseSeq { seq_id } => {
                    self.pending_checkouts
                        .retain(|_, pending_seq_id| *pending_seq_id != seq_id);
                    self.release_sequence(seq_id);
                }
                Command::PushToken {
                    token,
                    seq_id,
                    snapshot,
                    reply,
                } => {
                    let pos = snapshot.read().unwrap().tokens.len() as llama_pos;
                    let result = self.push_token(token, pos, seq_id);
                    if let Ok(logits) = &result {
                        let mut state = snapshot.write().unwrap();
                        state.tokens.push(token);
                        state.token_snapshot = None;
                        state.logits = Some(logits.clone());
                    }
                    let _ = reply.send(Ok(result));
                }
                Command::DecodeLast {
                    seq_id,
                    snapshot,
                    reply,
                } => {
                    let last = {
                        let state = snapshot.read().unwrap();
                        state
                            .tokens
                            .last()
                            .copied()
                            .map(|token| (token, (state.tokens.len() - 1) as llama_pos))
                    };
                    let result = match last {
                        Some((token, pos)) => self.push_token(token, pos, seq_id).map(Some),
                        None => Ok(None),
                    };
                    if let Ok(Some(logits)) = &result {
                        snapshot.write().unwrap().logits = Some(logits.clone());
                    }
                    let _ = reply.send(Ok(result));
                }
                Command::Pop {
                    seq_id,
                    snapshot,
                    reply,
                } => {
                    let len = snapshot.read().unwrap().tokens.len();
                    let result = if len == 0 {
                        None
                    } else {
                        let start = (len - 1) as llama_pos;
                        let ok = unsafe {
                            llama_memory_seq_rm(self.get_memory(), seq_id, start, len as i32)
                        };
                        if ok {
                            let mut state = snapshot.write().unwrap();
                            let token = state.tokens.pop();
                            state.token_snapshot = None;
                            state.logits = None;
                            token
                        } else {
                            None
                        }
                    };
                    let _ = reply.send(Ok(result));
                }
                Command::Remove {
                    seq_id,
                    start,
                    end,
                    snapshot,
                    reply,
                } => {
                    let len = snapshot.read().unwrap().tokens.len();
                    let valid = start >= 0 && end >= start && (end as usize) <= len;
                    let ok = valid
                        && unsafe { llama_memory_seq_rm(self.get_memory(), seq_id, start, end) };
                    if ok {
                        let removed = end - start;
                        if removed > 0 && (end as usize) < len {
                            unsafe {
                                llama_memory_seq_add(self.get_memory(), seq_id, end, -1, -removed)
                            };
                        }

                        let mut state = snapshot.write().unwrap();
                        state.tokens.drain(start as usize..end as usize);
                        state.token_snapshot = None;
                        state.logits = None;
                    }
                    let _ = reply.send(Ok(ok));
                }
                Command::Copy {
                    src,
                    dst,
                    start,
                    end,
                    src_snapshot,
                    dst_snapshot,
                    reply,
                } => {
                    let src_tokens = src_snapshot.read().unwrap().tokens.clone();
                    let valid = start >= 0 && end >= start && (end as usize) <= src_tokens.len();
                    if valid {
                        unsafe {
                            llama_memory_seq_rm(self.get_memory(), dst, -1, -1);
                            llama_memory_seq_cp(self.get_memory(), src, dst, start, end);
                            if start > 0 {
                                llama_memory_seq_add(self.get_memory(), dst, start, end, -start);
                            }
                        }
                        let mut dst_state = dst_snapshot.write().unwrap();
                        dst_state.tokens = src_tokens[start as usize..end as usize].to_vec();
                        dst_state.token_snapshot = None;
                        dst_state.logits = None;
                    }
                    let _ = reply.send(Ok(valid));
                }
                Command::MemorySeqRm {
                    seq_id,
                    p0,
                    p1,
                    snapshot,
                    reply,
                } => {
                    let ok = unsafe { llama_memory_seq_rm(self.get_memory(), seq_id, p0, p1) };
                    if ok {
                        snapshot.write().unwrap().logits = None;
                    }
                    let _ = reply.send(Ok(ok));
                }
                Command::MemorySeqAdd {
                    seq_id,
                    p0,
                    p1,
                    delta,
                    snapshot,
                    reply,
                } => {
                    unsafe { llama_memory_seq_add(self.get_memory(), seq_id, p0, p1, delta) };
                    snapshot.write().unwrap().logits = None;
                    let _ = reply.send(Ok(()));
                }
                Command::MemorySeqPosMin { seq_id, reply } => {
                    let value = unsafe { llama_memory_seq_pos_min(self.get_memory(), seq_id) };
                    let _ = reply.send(Ok(value));
                }
                Command::MemorySeqPosMax { seq_id, reply } => {
                    let value = unsafe { llama_memory_seq_pos_max(self.get_memory(), seq_id) };
                    let _ = reply.send(Ok(value));
                }
                Command::GetNCtx { reply } => {
                    let _ = reply.send(Ok(unsafe { llama_n_ctx(self.ctx) }));
                }
                Command::CanShift { reply } => {
                    let _ = reply.send(Ok(unsafe { llama_memory_can_shift(self.get_memory()) }));
                }
                Command::FreeSlots { reply } => {
                    let free = self.checked_out.iter().filter(|&&used| !used).count();
                    let _ = reply.send(Ok(free));
                }
                Command::GetPerf { reply } => {
                    let _ = reply.send(Ok(unsafe { llama_perf_context(self.ctx) }));
                }
                Command::Shutdown => break,
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}

struct ContextInner {
    commands: mpsc::Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
    next_checkout_id: AtomicU64,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

pin_project_lite::pin_project! {
    pub struct CheckoutFuture<F> {
        #[pin]
        inner: F,
        guard: Option<CheckoutGuard>,
    }
}

impl<F: Future<Output = Result<Option<crate::Sequence>, ContextError>>> Future
    for CheckoutFuture<F>
{
    type Output = Result<Option<crate::Sequence>, ContextError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.project();
        match this.inner.poll(cx) {
            std::task::Poll::Ready(result) => {
                if result.is_ok() {
                    if let Some(mut guard) = this.guard.take() {
                        guard.commit();
                    }
                } else {
                    drop(this.guard.take());
                }
                std::task::Poll::Ready(result)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

struct CheckoutGuard {
    context: Context,
    request_id: u64,
    active: bool,
}

impl CheckoutGuard {
    fn new(context: Context, request_id: u64) -> Self {
        Self {
            context,
            request_id,
            active: true,
        }
    }

    fn commit(&mut self) {
        let _ = self.context.inner.commands.send(Command::CommitCheckout {
            request_id: self.request_id,
        });
        self.active = false;
    }
}

impl Drop for CheckoutGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = self.context.inner.commands.send(Command::CancelCheckout {
                request_id: self.request_id,
            });
        }
    }
}

impl Context {
    /// Creates a context whose native lifetime is confined to a dedicated
    /// worker thread.
    ///
    /// The parameter structure is copied before the worker starts. This safe
    /// constructor therefore rejects every currently supported parameter field
    /// that can reach caller-owned state: evaluation callbacks, abort callbacks,
    /// callback user-data pointers, and native backend sampler configurations.
    /// Plain value fields such as context size, batch sizes, thread counts,
    /// cache types, and feature flags remain supported.
    ///
    /// This restriction does not affect the crate's Rust [`crate::Sampler`]
    /// implementations. Those operate on an owned logits snapshot when
    /// [`crate::Sequence::sample`] is called and are never stored in
    /// `llama_context_params`.
    ///
    /// # Errors
    ///
    /// Returns [`ContextInitError::ThreadUnsafeParams`] when `params` contains
    /// callbacks, callback user data, or native backend samplers. Other variants
    /// report worker creation or native initialization failures.
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ContextInitError> {
        if params.has_worker_thread_state() {
            return Err(ContextInitError::ThreadUnsafeParams);
        }

        // SAFETY: all callback and raw pointer-bearing fields were rejected.
        unsafe { Self::new_unchecked(model, params) }
    }

    /// Creates a context from raw llama.cpp parameters without rejecting
    /// callbacks, user-data pointers, or native backend samplers.
    ///
    /// This is an FFI escape hatch for features that [`Context::new`] cannot
    /// safely model. It performs the same initialization as [`Context::new`]:
    /// the parameter structure is copied, then `llama_init_from_model` is called
    /// on the dedicated worker thread. The function does not clone, retain, or
    /// synchronize anything referenced by raw pointers in `params`.
    ///
    /// `llama_context_params::samplers` is llama.cpp's experimental native
    /// backend-sampling configuration. It is not the crate's [`crate::Sampler`]
    /// trait. The native field points to per-sequence `llama_sampler` chains
    /// used by llama.cpp while evaluating the context.
    ///
    /// # Safety
    ///
    /// The caller must uphold all of the following for the entire period in
    /// which the native context may access the parameter state:
    ///
    /// - every non-null pointer reachable from `params` points to valid,
    ///   correctly initialized storage;
    /// - callback functions may be invoked on the worker thread, and their
    ///   user-data objects are valid and synchronized for that access;
    /// - when `samplers` is non-null, it points to at least `n_samplers` valid
    ///   sequence configurations, and every referenced native sampler is a
    ///   llama.cpp sampler chain suitable for backend sampling;
    /// - the sampler configuration storage, sampler chains, callback user data,
    ///   and anything they reference outlive this [`Context`];
    /// - none of that state is mutated, sampled through, reset, or destroyed
    ///   concurrently unless the upstream llama.cpp API explicitly permits it;
    /// - all remaining invariants required by the pinned `llama_context_params`
    ///   definition are satisfied.
    ///
    /// Violating these requirements can cause data races, use-after-free, or
    /// other undefined behavior inside llama.cpp.
    pub unsafe fn new_unchecked(
        model: &Model,
        params: &ContextParams,
    ) -> Result<Self, ContextInitError> {
        Self::start_worker(model, params)
    }

    fn start_worker(model: &Model, params: &ContextParams) -> Result<Self, ContextInitError> {
        let (commands, receiver) = mpsc::channel();
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let init = WorkerInit {
            model: model.clone(),
            params: *params,
        };

        let worker = std::thread::Builder::new()
            .name("rusty-llama-context".into())
            .spawn(move || match Worker::new(init) {
                Ok(worker) => {
                    let _ = started_tx.send(Ok(()));
                    worker.run(receiver);
                }
                Err(error) => {
                    let _ = started_tx.send(Err(error));
                }
            })
            .map_err(|_| ContextInitError::ThreadSpawnFailed)?;

        match started_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                inner: Arc::new(ContextInner {
                    commands,
                    worker: Mutex::new(Some(worker)),
                    next_checkout_id: AtomicU64::new(1),
                }),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err(ContextInitError::WorkerStopped)
            }
        }
    }

    fn submit<T>(
        &self,
        build: impl FnOnce(Reply<T>) -> Command,
    ) -> Result<oneshot::Receiver<Result<T, ContextError>>, ContextError> {
        let (reply, receiver) = oneshot::channel();
        self.inner
            .commands
            .send(build(reply))
            .map_err(|_| ContextError::WorkerStopped)?;
        Ok(receiver)
    }

    fn wait<T>(&self, build: impl FnOnce(Reply<T>) -> Command) -> Result<T, ContextError> {
        self.submit(build)?
            .recv()
            .map_err(|_| ContextError::WorkerStopped)?
    }

    async fn wait_async<T>(
        &self,
        build: impl FnOnce(Reply<T>) -> Command,
    ) -> Result<T, ContextError> {
        self.submit(build)?
            .await
            .map_err(|_| ContextError::WorkerStopped)?
    }

    pub(crate) fn same_worker(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn next_checkout_id(&self) -> u64 {
        self.inner.next_checkout_id.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn release_seq(&self, seq_id: llama_seq_id) {
        let _ = self.inner.commands.send(Command::ReleaseSeq { seq_id });
    }

    pub(crate) fn push_token(
        &self,
        token: Token,
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
    ) -> Result<Arc<[f32]>, DecodeError> {
        self.wait(|reply| Command::PushToken {
            token,
            seq_id,
            snapshot,
            reply,
        })
        .expect("context worker stopped")
    }

    pub(crate) async fn push_token_async(
        &self,
        token: Token,
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
    ) -> Result<Result<Arc<[f32]>, DecodeError>, ContextError> {
        self.wait_async(|reply| Command::PushToken {
            token,
            seq_id,
            snapshot,
            reply,
        })
        .await
    }

    pub(crate) fn decode_last(
        &self,
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
    ) -> Result<Option<Arc<[f32]>>, DecodeError> {
        self.wait(|reply| Command::DecodeLast {
            seq_id,
            snapshot,
            reply,
        })
        .expect("context worker stopped")
    }

    pub(crate) async fn decode_last_async(
        &self,
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
    ) -> Result<Result<Option<Arc<[f32]>>, DecodeError>, ContextError> {
        self.wait_async(|reply| Command::DecodeLast {
            seq_id,
            snapshot,
            reply,
        })
        .await
    }

    pub(crate) fn pop(
        &self,
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
    ) -> Option<Token> {
        self.wait(|reply| Command::Pop {
            seq_id,
            snapshot,
            reply,
        })
        .expect("context worker stopped")
    }

    pub(crate) async fn pop_async(
        &self,
        seq_id: llama_seq_id,
        snapshot: SharedSequenceSnapshot,
    ) -> Result<Option<Token>, ContextError> {
        self.wait_async(|reply| Command::Pop {
            seq_id,
            snapshot,
            reply,
        })
        .await
    }

    pub(crate) fn remove(
        &self,
        seq_id: llama_seq_id,
        start: llama_pos,
        end: llama_pos,
        snapshot: SharedSequenceSnapshot,
    ) -> bool {
        self.wait(|reply| Command::Remove {
            seq_id,
            start,
            end,
            snapshot,
            reply,
        })
        .expect("context worker stopped")
    }

    pub(crate) async fn remove_async(
        &self,
        seq_id: llama_seq_id,
        start: llama_pos,
        end: llama_pos,
        snapshot: SharedSequenceSnapshot,
    ) -> Result<bool, ContextError> {
        self.wait_async(|reply| Command::Remove {
            seq_id,
            start,
            end,
            snapshot,
            reply,
        })
        .await
    }

    pub(crate) fn copy(
        &self,
        src: llama_seq_id,
        dst: llama_seq_id,
        start: llama_pos,
        end: llama_pos,
        src_snapshot: SharedSequenceSnapshot,
        dst_snapshot: SharedSequenceSnapshot,
    ) -> bool {
        self.wait(|reply| Command::Copy {
            src,
            dst,
            start,
            end,
            src_snapshot,
            dst_snapshot,
            reply,
        })
        .expect("context worker stopped")
    }

    pub(crate) async fn copy_async(
        &self,
        src: llama_seq_id,
        dst: llama_seq_id,
        start: llama_pos,
        end: llama_pos,
        src_snapshot: SharedSequenceSnapshot,
        dst_snapshot: SharedSequenceSnapshot,
    ) -> Result<bool, ContextError> {
        self.wait_async(|reply| Command::Copy {
            src,
            dst,
            start,
            end,
            src_snapshot,
            dst_snapshot,
            reply,
        })
        .await
    }

    pub(crate) fn memory_seq_rm(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        snapshot: SharedSequenceSnapshot,
    ) -> bool {
        self.wait(|reply| Command::MemorySeqRm {
            seq_id,
            p0,
            p1,
            snapshot,
            reply,
        })
        .expect("context worker stopped")
    }

    pub(crate) fn memory_seq_add(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
        snapshot: SharedSequenceSnapshot,
    ) {
        self.wait(|reply| Command::MemorySeqAdd {
            seq_id,
            p0,
            p1,
            delta,
            snapshot,
            reply,
        })
        .expect("context worker stopped")
    }

    pub(crate) fn memory_seq_pos_min(&self, seq_id: llama_seq_id) -> llama_pos {
        self.wait(|reply| Command::MemorySeqPosMin { seq_id, reply })
            .expect("context worker stopped")
    }

    pub(crate) fn memory_seq_pos_max(&self, seq_id: llama_seq_id) -> llama_pos {
        self.wait(|reply| Command::MemorySeqPosMax { seq_id, reply })
            .expect("context worker stopped")
    }

    pub fn try_sequence(&self) -> Result<Option<crate::Sequence>, ContextError> {
        let request_id = self.next_checkout_id();
        let mut guard = CheckoutGuard::new(self.clone(), request_id);
        let reservation = self.wait(|reply| Command::CheckoutSeq { request_id, reply })?;
        guard.commit();
        Ok(reservation.map(|reservation| crate::Sequence::new(self.clone(), reservation)))
    }

    pub fn sequence(&self) -> Option<crate::Sequence> {
        self.try_sequence().expect("context worker stopped")
    }

    pub fn sequence_async(
        &self,
    ) -> impl Future<Output = Result<Option<crate::Sequence>, ContextError>> + Send + 'static {
        let request_id = self.next_checkout_id();
        let guard = Some(CheckoutGuard::new(self.clone(), request_id));
        let receiver = self.submit(|reply| Command::CheckoutSeq { request_id, reply });
        let context = self.clone();

        CheckoutFuture {
            inner: async move {
                let receiver = receiver?;
                let reservation = receiver.await.map_err(|_| ContextError::WorkerStopped)??;
                Ok(reservation.map(|reservation| crate::Sequence::new(context, reservation)))
            },
            guard,
        }
    }

    pub fn try_free_slots(&self) -> Result<usize, ContextError> {
        self.wait(|reply| Command::FreeSlots { reply })
    }

    pub fn free_slots(&self) -> usize {
        self.try_free_slots().expect("context worker stopped")
    }

    pub async fn free_slots_async(&self) -> Result<usize, ContextError> {
        self.wait_async(|reply| Command::FreeSlots { reply }).await
    }

    pub fn try_n_ctx(&self) -> Result<u32, ContextError> {
        self.wait(|reply| Command::GetNCtx { reply })
    }

    pub fn n_ctx(&self) -> u32 {
        self.try_n_ctx().expect("context worker stopped")
    }

    pub async fn n_ctx_async(&self) -> Result<u32, ContextError> {
        self.wait_async(|reply| Command::GetNCtx { reply }).await
    }

    pub fn try_can_shift(&self) -> Result<bool, ContextError> {
        self.wait(|reply| Command::CanShift { reply })
    }

    pub fn can_shift(&self) -> bool {
        self.try_can_shift().expect("context worker stopped")
    }

    pub async fn can_shift_async(&self) -> Result<bool, ContextError> {
        self.wait_async(|reply| Command::CanShift { reply }).await
    }

    pub fn try_perf(&self) -> Result<llama_perf_context_data, ContextError> {
        self.wait(|reply| Command::GetPerf { reply })
    }

    pub fn perf(&self) -> llama_perf_context_data {
        self.try_perf().expect("context worker stopped")
    }

    pub async fn perf_async(&self) -> Result<llama_perf_context_data, ContextError> {
        self.wait_async(|reply| Command::GetPerf { reply }).await
    }
}
