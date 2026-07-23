use llama_sys::*;
use std::ops::{Deref, DerefMut};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use crate::{common, Batch, Model, Token};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextError {
    WorkerStopped,
}

pub(crate) struct SequenceSnapshot {
    pub tokens: Arc<[Token]>,
    pub logits: Option<Arc<[f32]>>,
}

impl SequenceSnapshot {
    fn empty() -> Self {
        Self {
            tokens: Arc::from([]),
            logits: None,
        }
    }
}

pub(crate) type SharedSequenceSnapshot = Arc<Mutex<SequenceSnapshot>>;

type Reply<T> = oneshot::Sender<Result<T, ContextError>>;

enum Command {
    CheckoutSeq {
        reply: Reply<Option<(llama_seq_id, SharedSequenceSnapshot)>>,
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
        reply: Reply<()>,
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

struct WorkerInit {
    model: Model,
    params: ContextParams,
}

// SAFETY: the worker is the sole owner of the copied parameter struct after
// Context::new returns. This preserves the existing parameter API contract while
// ensuring the native context itself is created, used, and destroyed on one OS
// thread.
unsafe impl Send for WorkerInit {}

struct Worker {
    ctx: *mut llama_context,
    _model: Model,
    batch: Batch,
    n_vocab: i32,
    checked_out: Vec<bool>,
}

impl Worker {
    fn new(init: WorkerInit) -> Result<Self, ()> {
        let ctx = unsafe { llama_init_from_model(init.model.as_mut_ptr(), init.params.0) };
        if ctx.is_null() {
            return Err(());
        }

        let n_seq_max = init.params.n_seq_max as usize;
        let n_vocab = init.model.n_tokens();

        Ok(Self {
            ctx,
            _model: init.model,
            batch: Batch::init_token(1, init.params.n_seq_max as i32),
            n_vocab,
            checked_out: vec![false; n_seq_max],
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

        Some(Arc::from(
            unsafe { std::slice::from_raw_parts(ptr, self.n_vocab as usize) }.to_vec(),
        ))
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

    fn run(mut self, commands: mpsc::Receiver<Command>) {
        while let Ok(command) = commands.recv() {
            match command {
                Command::CheckoutSeq { reply } => {
                    let result = self
                        .checked_out
                        .iter_mut()
                        .enumerate()
                        .find(|(_, used)| !**used)
                        .map(|(id, used)| {
                            *used = true;
                            (
                                id as llama_seq_id,
                                Arc::new(Mutex::new(SequenceSnapshot::empty())),
                            )
                        });
                    let _ = reply.send(Ok(result));
                }
                Command::ReleaseSeq { seq_id } => {
                    unsafe { llama_memory_seq_rm(self.get_memory(), seq_id, -1, -1) };
                    if let Some(slot) = self.checked_out.get_mut(seq_id as usize) {
                        *slot = false;
                    }
                }
                Command::PushToken {
                    token,
                    seq_id,
                    snapshot,
                    reply,
                } => {
                    let pos = snapshot.lock().unwrap().tokens.len() as llama_pos;
                    let result = self.push_token(token, pos, seq_id);
                    if let Ok(logits) = &result {
                        let mut state = snapshot.lock().unwrap();
                        let mut tokens = state.tokens.to_vec();
                        tokens.push(token);
                        state.tokens = Arc::from(tokens);
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
                        let state = snapshot.lock().unwrap();
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
                        snapshot.lock().unwrap().logits = Some(logits.clone());
                    }
                    let _ = reply.send(Ok(result));
                }
                Command::Pop {
                    seq_id,
                    snapshot,
                    reply,
                } => {
                    let len = snapshot.lock().unwrap().tokens.len();
                    let result = if len == 0 {
                        None
                    } else {
                        let start = (len - 1) as llama_pos;
                        let ok = unsafe {
                            llama_memory_seq_rm(self.get_memory(), seq_id, start, len as i32)
                        };
                        if ok {
                            let mut state = snapshot.lock().unwrap();
                            let mut tokens = state.tokens.to_vec();
                            let token = tokens.pop();
                            state.tokens = Arc::from(tokens);
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
                    let len = snapshot.lock().unwrap().tokens.len();
                    let valid = start >= 0 && end >= start && (end as usize) <= len;
                    let ok = valid
                        && unsafe { llama_memory_seq_rm(self.get_memory(), seq_id, start, end) };
                    if ok {
                        let removed = end - start;
                        if removed > 0 && (end as usize) < len {
                            unsafe {
                                llama_memory_seq_add(
                                    self.get_memory(),
                                    seq_id,
                                    end,
                                    -1,
                                    -removed,
                                )
                            };
                        }

                        let mut state = snapshot.lock().unwrap();
                        let mut tokens = state.tokens.to_vec();
                        tokens.drain(start as usize..end as usize);
                        state.tokens = Arc::from(tokens);
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
                    let src_tokens = src_snapshot.lock().unwrap().tokens.clone();
                    let valid = start >= 0 && end >= start && (end as usize) <= src_tokens.len();
                    if valid {
                        unsafe {
                            llama_memory_seq_rm(self.get_memory(), dst, -1, -1);
                            llama_memory_seq_cp(self.get_memory(), src, dst, start, end);
                            if start > 0 {
                                llama_memory_seq_add(
                                    self.get_memory(),
                                    dst,
                                    start,
                                    end,
                                    -start,
                                );
                            }
                        }
                        let mut dst_state = dst_snapshot.lock().unwrap();
                        dst_state.tokens =
                            Arc::from(src_tokens[start as usize..end as usize].to_vec());
                        dst_state.logits = None;
                    }
                    let _ = reply.send(Ok(()));
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
                        snapshot.lock().unwrap().logits = None;
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
                    snapshot.lock().unwrap().logits = None;
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

impl Context {
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ()> {
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
                    let _ = started_tx.send(true);
                    worker.run(receiver);
                }
                Err(()) => {
                    let _ = started_tx.send(false);
                }
            })
            .map_err(|_| ())?;

        if started_rx.recv().map_err(|_| ())? {
            Ok(Self {
                inner: Arc::new(ContextInner {
                    commands,
                    worker: Mutex::new(Some(worker)),
                }),
            })
        } else {
            let _ = worker.join();
            Err(())
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
    ) -> Result<Arc<[f32]>, DecodeError> {
        self.wait_async(|reply| Command::PushToken {
            token,
            seq_id,
            snapshot,
            reply,
        })
        .await
        .expect("context worker stopped")
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
    ) -> Result<Option<Arc<[f32]>>, DecodeError> {
        self.wait_async(|reply| Command::DecodeLast {
            seq_id,
            snapshot,
            reply,
        })
        .await
        .expect("context worker stopped")
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
    ) -> Option<Token> {
        self.wait_async(|reply| Command::Pop {
            seq_id,
            snapshot,
            reply,
        })
        .await
        .expect("context worker stopped")
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
    ) -> bool {
        self.wait_async(|reply| Command::Remove {
            seq_id,
            start,
            end,
            snapshot,
            reply,
        })
        .await
        .expect("context worker stopped")
    }

    pub(crate) fn copy(
        &self,
        src: llama_seq_id,
        dst: llama_seq_id,
        start: llama_pos,
        end: llama_pos,
        src_snapshot: SharedSequenceSnapshot,
        dst_snapshot: SharedSequenceSnapshot,
    ) {
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
    ) {
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
        .expect("context worker stopped")
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

    pub fn sequence(&self) -> Option<crate::Sequence> {
        self.wait(|reply| Command::CheckoutSeq { reply })
            .expect("context worker stopped")
            .map(|(id, snapshot)| crate::Sequence::new(self.clone(), id, snapshot))
    }

    pub async fn sequence_async(&self) -> Option<crate::Sequence> {
        self.wait_async(|reply| Command::CheckoutSeq { reply })
            .await
            .expect("context worker stopped")
            .map(|(id, snapshot)| crate::Sequence::new(self.clone(), id, snapshot))
    }

    pub fn free_slots(&self) -> usize {
        self.wait(|reply| Command::FreeSlots { reply })
            .expect("context worker stopped")
    }

    pub async fn free_slots_async(&self) -> usize {
        self.wait_async(|reply| Command::FreeSlots { reply })
            .await
            .expect("context worker stopped")
    }

    pub fn n_ctx(&self) -> u32 {
        self.wait(|reply| Command::GetNCtx { reply })
            .expect("context worker stopped")
    }

    pub async fn n_ctx_async(&self) -> u32 {
        self.wait_async(|reply| Command::GetNCtx { reply })
            .await
            .expect("context worker stopped")
    }

    pub fn can_shift(&self) -> bool {
        self.wait(|reply| Command::CanShift { reply })
            .expect("context worker stopped")
    }

    pub async fn can_shift_async(&self) -> bool {
        self.wait_async(|reply| Command::CanShift { reply })
            .await
            .expect("context worker stopped")
    }

    pub fn perf(&self) -> llama_perf_context_data {
        self.wait(|reply| Command::GetPerf { reply })
            .expect("context worker stopped")
    }

    pub async fn perf_async(&self) -> llama_perf_context_data {
        self.wait_async(|reply| Command::GetPerf { reply })
            .await
            .expect("context worker stopped")
    }
}
