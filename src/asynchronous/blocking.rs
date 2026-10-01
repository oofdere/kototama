//! Offloading blocking work to a small background thread pool, and driving
//! futures from synchronous code.
//!
//! llama.cpp calls are synchronous and some take a long time (`llama_decode`
//! on a full batch runs for hundreds of milliseconds; loading a model takes
//! seconds). Running them inside a polled future would stall whatever executor
//! drives it, so every async operation in [`crate::asynchronous`] ships its
//! work to the pool here and awaits the result. The pool is std-only: this
//! crate does not depend on any async runtime.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::{pin, Pin};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::Thread;

type Job = Box<dyn FnOnce() + Send>;

/// A fixed pool of worker threads fed by one shared queue.
///
/// The pool lives for the whole process (idle workers park on a condvar), so
/// there is no shutdown path. That is deliberate: it backs short-lived
/// blocking calls, not long-running background work, and a process that exits
/// while a decode is still in flight does not care about its result.
struct Pool {
    queue: Mutex<VecDeque<Job>>,
    ready: Condvar,
}

impl Pool {
    fn global() -> &'static Arc<Pool> {
        static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
        POOL.get_or_init(|| {
            let pool = Arc::new(Pool {
                queue: Mutex::new(VecDeque::new()),
                ready: Condvar::new(),
            });
            // Decoding already uses several CPU threads internally (see
            // llama_set_n_threads), so a handful of workers keeps separate
            // sequences moving without oversubscribing the machine.
            let workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(4);
            for _ in 0..workers {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || pool.work());
            }
            pool
        })
    }

    fn work(self: Arc<Self>) {
        loop {
            let job = {
                let mut queue = lock(&self.queue);
                loop {
                    if let Some(job) = queue.pop_front() {
                        break job;
                    }
                    queue = self
                        .ready
                        .wait(queue)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            };
            job();
        }
    }

    fn submit(&self, job: Job) {
        lock(&self.queue).push_back(job);
        self.ready.notify_one();
    }
}

/// Handle shared between a [`Blocking`] future and its worker job.
struct Shared<T> {
    value: Mutex<Option<T>>,
    waker: Mutex<Option<Waker>>,
}

/// Future returned by [`run_blocking`]. Resolves with the closure's result
/// once a pool worker has run it.
///
/// Dropping the future does not cancel the job: the closure still runs to
/// completion on a worker and its result is discarded. This keeps the sync
/// core's state transitions (push, pop, slot release) all-or-nothing even when
/// a task is cancelled mid-operation, and matches how runtime
/// `spawn_blocking`-style helpers behave.
pub(crate) struct Blocking<F, T> {
    job: Option<(F, Arc<Shared<T>>)>,
    shared: Arc<Shared<T>>,
}

/// Run `f` on the thread pool and await its result.
///
/// Nothing runs until the returned future is first polled.
pub(crate) fn run_blocking<F, T>(f: F) -> Blocking<F, T>
where
    F: FnOnce() -> T + Send + Unpin + 'static,
    T: Send + 'static,
{
    let shared = Arc::new(Shared {
        value: Mutex::new(None),
        waker: Mutex::new(None),
    });
    Blocking {
        job: Some((f, Arc::clone(&shared))),
        shared,
    }
}

impl<F, T> Future for Blocking<F, T>
where
    F: FnOnce() -> T + Send + Unpin + 'static,
    T: Send + 'static,
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let this = self.get_mut();

        if let Some(value) = lock(&this.shared.value).take() {
            return Poll::Ready(value);
        }

        if let Some((f, shared)) = this.job.take() {
            Pool::global().submit(Box::new(move || {
                let value = f();
                *lock(&shared.value) = Some(value);
                if let Some(waker) = lock(&shared.waker).take() {
                    waker.wake();
                }
            }));
        }

        *lock(&this.shared.waker) = Some(cx.waker().clone());
        // Re-check after registering: the job may have finished between the
        // first check and the registration, and its wake would then be lost.
        if let Some(value) = lock(&this.shared.value).take() {
            Poll::Ready(value)
        } else {
            Poll::Pending
        }
    }
}

/// Run a future to completion on the current thread.
///
/// This is a minimal std-only executor: it polls the future and parks the
/// thread until the future's waker unparks it. It suits examples, tests, and
/// simple programs. Inside an async runtime, `.await` the futures instead.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Recover from a poisoned lock: a panic in one job does not invalidate
    // the queue or the shared state for the others.
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
