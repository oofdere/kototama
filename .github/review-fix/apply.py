from pathlib import Path
import re


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one exact match, found {count}\n{old[:160]}")
    file.write_text(text.replace(old, new, 1))


def sub_once(path: str, pattern: str, replacement: str) -> None:
    file = Path(path)
    text = file.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{path}: expected one regex match, found {count}\n{pattern[:160]}")
    file.write_text(updated)


# Context parameter and initialization safety.
replace_once(
    "src/context.rs",
    '''impl ContextParams {
    pub fn new() -> Self {
        Self(unsafe { llama_context_default_params() })
    }

    pub fn as_ptr(&self) -> *const llama_context_params {
        &self.0
    }

    pub fn as_mut_ptr(&mut self) -> *mut llama_context_params {
        &mut self.0
    }
}''',
    '''impl ContextParams {
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
            || !self.ctx_other.is_null()
    }
}''',
)

replace_once(
    "src/context.rs",
    '''#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextError {
    WorkerStopped,
}''',
    '''#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextError {
    WorkerStopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextInitError {
    ThreadUnsafeParams,
    ThreadSpawnFailed,
    WorkerStopped,
    NativeInitFailed,
}''',
)

replace_once(
    "src/context.rs",
    "        reply: Reply<()>,\n    },\n    MemorySeqRm {",
    "        reply: Reply<bool>,\n    },\n    MemorySeqRm {",
)

replace_once(
    "src/context.rs",
    '''    pub(crate) fn into_parts(self) -> (llama_seq_id, SharedSequenceSnapshot) {
        (self.id, self.snapshot.clone())
    }''',
    '''    pub(crate) fn into_parts(self) -> (llama_seq_id, SharedSequenceSnapshot) {
        (self.id, self.snapshot)
    }''',
)

replace_once(
    "src/context.rs",
    '''// SAFETY: the worker is the sole owner of the copied parameter struct after
// Context::new returns. This preserves the existing parameter API contract while
// ensuring the native context itself is created, used, and destroyed on one OS
// thread.
unsafe impl Send for WorkerInit {}''',
    '''// SAFETY: WorkerInit is private and is constructed only by Context::new,
// which rejects every callback or raw pointer-bearing parameter, or by
// Context::new_unchecked, whose safety contract requires all referenced state to
// be valid and safe to access on the worker thread for the full context lifetime.
unsafe impl Send for WorkerInit {}''',
)

replace_once(
    "src/context.rs",
    '''    fn new(init: WorkerInit) -> Result<Self, ()> {
        let ctx = unsafe { llama_init_from_model(init.model.as_mut_ptr(), init.params.0) };
        if ctx.is_null() {
            return Err(());
        }''',
    '''    fn new(init: WorkerInit) -> Result<Self, ContextInitError> {
        let ctx = unsafe { llama_init_from_model(init.model.as_mut_ptr(), init.params.0) };
        if ctx.is_null() {
            return Err(ContextInitError::NativeInitFailed);
        }''',
)

replace_once(
    "src/context.rs",
    '''                    let _ = reply.send(Ok(()));
                }
                Command::MemorySeqRm {''',
    '''                    let _ = reply.send(Ok(valid));
                }
                Command::MemorySeqRm {''',
)

sub_once(
    "src/context.rs",
    r'''impl<F: Future<Output = Option<crate::Sequence>>> Future for CheckoutFuture<F> \{.*?\n\}''',
    '''impl<F: Future<Output = Result<Option<crate::Sequence>, ContextError>>> Future
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
}''',
)

sub_once(
    "src/context.rs",
    r'''    pub fn new\(model: &Model, params: &ContextParams\) -> Result<Self, \(\)> \{.*?\n    \}\n\n    fn submit<T>''',
    '''    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ContextInitError> {
        if params.has_worker_thread_state() {
            return Err(ContextInitError::ThreadUnsafeParams);
        }

        // SAFETY: all callback and raw pointer-bearing fields were rejected.
        unsafe { Self::new_unchecked(model, params) }
    }

    /// Creates a context while allowing callback and raw pointer-bearing params.
    ///
    /// # Safety
    /// Every pointer reachable from `params` must remain valid until this
    /// [`Context`] is dropped and must be safe to access from the context worker
    /// thread. Callback functions must be callable on that thread. Sampler
    /// chains and `ctx_other`, when provided, must not be used concurrently in a
    /// way that violates llama.cpp's requirements.
    pub unsafe fn new_unchecked(
        model: &Model,
        params: &ContextParams,
    ) -> Result<Self, ContextInitError> {
        Self::start_worker(model, params)
    }

    fn start_worker(
        model: &Model,
        params: &ContextParams,
    ) -> Result<Self, ContextInitError> {
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

    fn submit<T>''',
)

# Async worker requests now propagate ContextError instead of panicking.
sub_once(
    "src/context.rs",
    r'''    pub\(crate\) async fn push_token_async\(.*?\n    \}''',
    '''    pub(crate) async fn push_token_async(
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
    }''',
)

sub_once(
    "src/context.rs",
    r'''    pub\(crate\) async fn decode_last_async\(.*?\n    \}''',
    '''    pub(crate) async fn decode_last_async(
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
    }''',
)

sub_once(
    "src/context.rs",
    r'''    pub\(crate\) async fn pop_async\(.*?\n    \}''',
    '''    pub(crate) async fn pop_async(
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
    }''',
)

sub_once(
    "src/context.rs",
    r'''    pub\(crate\) async fn remove_async\(.*?\n    \}''',
    '''    pub(crate) async fn remove_async(
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
    }''',
)

sub_once(
    "src/context.rs",
    r'''    pub\(crate\) fn copy\(.*?\n    \}\n\n    pub\(crate\) async fn copy_async\(.*?\n    \}''',
    '''    pub(crate) fn copy(
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
    }''',
)

sub_once(
    "src/context.rs",
    r'''    pub fn sequence\(&self\) -> Option<crate::Sequence> \{.*?\n    \}\n\n    pub fn sequence_async\(&self\) -> impl Future<Output = Option<crate::Sequence>> \+ Send \+ 'static \{.*?\n    \}''',
    '''    pub fn try_sequence(&self) -> Result<Option<crate::Sequence>, ContextError> {
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
                let reservation = receiver
                    .await
                    .map_err(|_| ContextError::WorkerStopped)??;
                Ok(reservation.map(|reservation| crate::Sequence::new(context, reservation)))
            },
            guard,
        }
    }''',
)

sub_once(
    "src/context.rs",
    r'''    pub fn free_slots\(&self\) -> usize \{.*?\n    \}\n\n    pub async fn free_slots_async\(&self\) -> usize \{.*?\n    \}\n\n    pub fn n_ctx\(&self\) -> u32 \{.*?\n    \}\n\n    pub async fn n_ctx_async\(&self\) -> u32 \{.*?\n    \}\n\n    pub fn can_shift\(&self\) -> bool \{.*?\n    \}\n\n    pub async fn can_shift_async\(&self\) -> bool \{.*?\n    \}\n\n    pub fn perf\(&self\) -> llama_perf_context_data \{.*?\n    \}\n\n    pub async fn perf_async\(&self\) -> llama_perf_context_data \{.*?\n    \}''',
    '''    pub fn try_free_slots(&self) -> Result<usize, ContextError> {
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
    }''',
)

# Sequence API: consistent range failure and fallible async operations.
replace_once(
    "src/sequence.rs",
    "use crate::{Context, Sampler, Token};",
    "use crate::{Context, ContextError, DecodeError, Sampler, Token};",
)

replace_once(
    "src/sequence.rs",
    '''pub struct Sequence {
    ctx: Context,
    id: i32,
    snapshot: SharedSequenceSnapshot,
}

impl Sequence {''',
    '''pub struct Sequence {
    ctx: Context,
    id: i32,
    snapshot: SharedSequenceSnapshot,
}

#[derive(Debug, Clone)]
pub enum SequenceError {
    Context(ContextError),
    Decode(DecodeError),
}

impl From<ContextError> for SequenceError {
    fn from(error: ContextError) -> Self {
        Self::Context(error)
    }
}

impl Sequence {''',
)

replace_once(
    "src/sequence.rs",
    '''    fn checked_copy_range(&self, range: Range<usize>) -> Range<i32> {
        let range = Self::checked_range(range).expect("sequence range exceeds llama_pos");
        assert!(
            range.start <= range.end && (range.end as usize) <= self.len(),
            "sequence range out of bounds"
        );
        range
    }''',
    '''    fn checked_copy_range(&self, range: Range<usize>) -> Option<Range<i32>> {
        let range = Self::checked_range(range)?;
        (range.start <= range.end && (range.end as usize) <= self.len()).then_some(range)
    }''',
)

sub_once(
    "src/sequence.rs",
    r'''    pub async fn push_async\(&mut self, token: Token\) \{.*?\n    \}''',
    '''    pub async fn push_async(&mut self, token: Token) -> Result<(), SequenceError> {
        self.ctx
            .push_token_async(token, self.id, self.snapshot.clone())
            .await?
            .map(|_| ())
            .map_err(SequenceError::Decode)
    }''',
)

sub_once(
    "src/sequence.rs",
    r'''    pub async fn decode_async\(&mut self\) \{.*?\n    \}''',
    '''    pub async fn decode_async(&mut self) -> Result<(), SequenceError> {
        self.ctx
            .decode_last_async(self.id, self.snapshot.clone())
            .await?
            .map(|_| ())
            .map_err(SequenceError::Decode)
    }''',
)

sub_once(
    "src/sequence.rs",
    r'''    pub async fn pop_async\(&mut self\) -> Option<Token> \{.*?\n    \}''',
    '''    pub async fn pop_async(&mut self) -> Result<Option<Token>, ContextError> {
        self.ctx.pop_async(self.id, self.snapshot.clone()).await
    }''',
)

sub_once(
    "src/sequence.rs",
    r'''    pub async fn extend_async\(&mut self, tokens: &\[Token\]\) \{.*?\n    \}''',
    '''    pub async fn extend_async(&mut self, tokens: &[Token]) -> Result<(), SequenceError> {
        for &token in tokens {
            self.push_async(token).await?;
        }
        Ok(())
    }''',
)

sub_once(
    "src/sequence.rs",
    r'''    pub async fn remove_async\(&mut self, range: Range<usize>\) -> bool \{.*?\n    \}''',
    '''    pub async fn remove_async(
        &mut self,
        range: Range<usize>,
    ) -> Result<bool, ContextError> {
        let Some(range) = Self::checked_range(range) else {
            return Ok(false);
        };
        self.ctx
            .remove_async(self.id, range.start, range.end, self.snapshot.clone())
            .await
    }''',
)

sub_once(
    "src/sequence.rs",
    r'''    pub fn copy_to\(&self, other: &mut Self, range: Range<usize>\) \{.*?\n    \}\n\n    pub async fn copy_to_async\(&self, other: &mut Self, range: Range<usize>\) \{.*?\n    \}\n\n    pub fn copy_from\(&mut self, other: &Self, range: Range<usize>\) \{.*?\n    \}\n\n    pub async fn copy_from_async\(&mut self, other: &Self, range: Range<usize>\) \{.*?\n    \}''',
    '''    pub fn copy_to(&self, other: &mut Self, range: Range<usize>) -> bool {
        assert!(
            self.ctx.same_worker(&other.ctx),
            "cannot copy sequences between different contexts"
        );
        assert_ne!(self.id, other.id, "cannot copy a sequence onto itself");
        let Some(range) = self.checked_copy_range(range) else {
            return false;
        };
        self.ctx.copy(
            self.id,
            other.id,
            range.start,
            range.end,
            self.snapshot.clone(),
            other.snapshot.clone(),
        )
    }

    pub async fn copy_to_async(
        &self,
        other: &mut Self,
        range: Range<usize>,
    ) -> Result<bool, ContextError> {
        assert!(
            self.ctx.same_worker(&other.ctx),
            "cannot copy sequences between different contexts"
        );
        assert_ne!(self.id, other.id, "cannot copy a sequence onto itself");
        let Some(range) = self.checked_copy_range(range) else {
            return Ok(false);
        };
        self.ctx
            .copy_async(
                self.id,
                other.id,
                range.start,
                range.end,
                self.snapshot.clone(),
                other.snapshot.clone(),
            )
            .await
    }

    pub fn copy_from(&mut self, other: &Self, range: Range<usize>) -> bool {
        other.copy_to(self, range)
    }

    pub async fn copy_from_async(
        &mut self,
        other: &Self,
        range: Range<usize>,
    ) -> Result<bool, ContextError> {
        other.copy_to_async(self, range).await
    }''',
)

# Public exports.
replace_once(
    "src/lib.rs",
    "pub use context::{Context, ContextError, ContextParams, DecodeError};",
    "pub use context::{Context, ContextError, ContextInitError, ContextParams, DecodeError};",
)

# Async API and initialization tests.
replace_once(
    "tests/worker.rs",
    "use rusty_llama::{Context, Model, ModelParams};",
    "use rusty_llama::{Context, ContextInitError, Model, ModelParams};",
)
replace_once(
    "tests/worker.rs",
    "        let mut seq = ctx.sequence_async().await.unwrap();\n        assert_eq!(ctx.free_slots_async().await, initial_slots - 1);",
    "        let mut seq = ctx.sequence_async().await.unwrap().unwrap();\n        assert_eq!(ctx.free_slots_async().await.unwrap(), initial_slots - 1);",
)
replace_once(
    "tests/worker.rs",
    "        seq.extend_async(&tokens).await;",
    "        seq.extend_async(&tokens).await.unwrap();",
)
replace_once(
    "tests/worker.rs",
    "        let popped = seq.pop_async().await;",
    "        let popped = seq.pop_async().await.unwrap();",
)
replace_once(
    "tests/worker.rs",
    "    let async_value = pollster::block_on(ctx.n_ctx_async());",
    "    let async_value = pollster::block_on(ctx.n_ctx_async()).unwrap();",
)
replace_once(
    "tests/worker.rs",
    '''#[test]
fn dropping_sequence_checkout_future_releases_the_slot() {''',
    '''#[test]
fn context_rejects_pointer_bearing_params() {
    let (model, mut params) = common::load_model_and_context();
    params.cb_eval_user_data = 1usize as *mut std::ffi::c_void;

    assert!(matches!(
        Context::new(&model, &params),
        Err(ContextInitError::ThreadUnsafeParams)
    ));
}

#[test]
fn dropping_sequence_checkout_future_releases_the_slot() {''',
)

replace_once(
    "tests/copy_ranges.rs",
    '''#[test]
#[should_panic(expected = "sequence range out of bounds")]
fn copy_to_rejects_out_of_bounds_range() {
    let ctx = context_with_copy_support();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    src.copy_to(&mut dst, 0..1);
}

#[test]
#[should_panic(expected = "sequence range out of bounds")]
fn copy_to_async_rejects_out_of_bounds_range() {
    let ctx = context_with_copy_support();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    pollster::block_on(src.copy_to_async(&mut dst, 0..1));
}''',
    '''#[test]
fn copy_to_rejects_out_of_bounds_range() {
    let ctx = context_with_copy_support();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    assert!(!src.copy_to(&mut dst, 0..1));
}

#[test]
fn copy_to_async_rejects_out_of_bounds_range() {
    let ctx = context_with_copy_support();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    assert!(!pollster::block_on(src.copy_to_async(&mut dst, 0..1)).unwrap());
}''',
)

replace_once(
    "tests/sequence.rs",
    '''#[test]
#[should_panic(expected = "sequence range exceeds llama_pos")]
fn copy_rejects_unrepresentable_range() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    src.copy_to(&mut dst, usize::MAX..usize::MAX);
}''',
    '''#[test]
fn copy_rejects_unrepresentable_range() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    assert!(!src.copy_to(&mut dst, usize::MAX..usize::MAX));
}''',
)
replace_once(
    "tests/sequence.rs",
    '''    assert_eq!(seq.is_empty(), seq.len() == 0);
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert_eq!(seq.is_empty(), seq.len() == 0);''',
    '''    assert!(seq.is_empty());
    assert_eq!(seq.len(), 0);
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(!seq.is_empty());
    assert_eq!(seq.len(), tokens.len());''',
)
