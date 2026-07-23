from pathlib import Path

context = Path("src/context.rs")
s = context.read_text()
s = s.replace(
    "use std::ops::{Deref, DerefMut};\nuse std::sync::{mpsc, Arc, Mutex};\n",
    "use std::collections::HashMap;\nuse std::future::Future;\nuse std::ops::{Deref, DerefMut};\nuse std::sync::atomic::{AtomicU64, Ordering};\nuse std::sync::{mpsc, Arc, Mutex, RwLock};\n",
)
s = s.replace(
    "pub(crate) type SharedSequenceSnapshot = Arc<Mutex<SequenceSnapshot>>;\n",
    "pub(crate) type SharedSequenceSnapshot = Arc<RwLock<SequenceSnapshot>>;\n",
)
s = s.replace(
    """enum Command {
    CheckoutSeq {
        reply: Reply<Option<(llama_seq_id, SharedSequenceSnapshot)>>,
    },
""",
    """enum Command {
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
""",
)
s = s.replace(
    """    Shutdown,
}

struct WorkerInit {
""",
    """    Shutdown,
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
        (self.id, self.snapshot.clone())
    }
}

struct WorkerInit {
""",
)
s = s.replace(
    """    checked_out: Vec<bool>,
}
""",
    """    checked_out: Vec<bool>,
    pending_checkouts: HashMap<u64, llama_seq_id>,
}
""",
    1,
)
s = s.replace(
    """            n_vocab,
            checked_out: vec![false; n_seq_max],
        })
""",
    """            n_vocab,
            checked_out: vec![false; n_seq_max],
            pending_checkouts: HashMap::new(),
        })
""",
    1,
)
s = s.replace(
    """    fn run(mut self, commands: mpsc::Receiver<Command>) {
""",
    """    fn checkout_sequence(&mut self, request_id: u64) -> Option<SequenceReservation> {
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
""",
    1,
)
s = s.replace(
    """                Command::CheckoutSeq { reply } => {
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
""",
    """                Command::CheckoutSeq { request_id, reply } => {
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
""",
    1,
)
s = s.replace("snapshot.lock().unwrap().tokens.len()", "snapshot.read().unwrap().tokens.len()")
s = s.replace("let state = snapshot.lock().unwrap();", "let state = snapshot.read().unwrap();")
s = s.replace("let mut state = snapshot.lock().unwrap();", "let mut state = snapshot.write().unwrap();")
s = s.replace("snapshot.lock().unwrap().logits =", "snapshot.write().unwrap().logits =")
s = s.replace("src_snapshot.lock().unwrap().tokens.clone()", "src_snapshot.read().unwrap().tokens.clone()")
s = s.replace("let mut dst_state = dst_snapshot.lock().unwrap();", "let mut dst_state = dst_snapshot.write().unwrap();")
s = s.replace(
    """struct ContextInner {
    commands: mpsc::Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
}
""",
    """struct ContextInner {
    commands: mpsc::Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
    next_checkout_id: AtomicU64,
}
""",
)
s = s.replace(
    """                    commands,
                    worker: Mutex::new(Some(worker)),
""",
    """                    commands,
                    worker: Mutex::new(Some(worker)),
                    next_checkout_id: AtomicU64::new(1),
""",
    1,
)
s = s.replace(
    """#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
""",
    """#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
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
        let _ = self
            .context
            .inner
            .commands
            .send(Command::CommitCheckout {
                request_id: self.request_id,
            });
        self.active = false;
    }
}

impl Drop for CheckoutGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = self
                .context
                .inner
                .commands
                .send(Command::CancelCheckout {
                    request_id: self.request_id,
                });
        }
    }
}

impl Context {
""",
    1,
)
s = s.replace(
    """    pub(crate) fn same_worker(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

""",
    """    pub(crate) fn same_worker(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn next_checkout_id(&self) -> u64 {
        self.inner.next_checkout_id.fetch_add(1, Ordering::Relaxed)
    }

""",
    1,
)
s = s.replace(
    """    pub fn sequence(&self) -> Option<crate::Sequence> {
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
""",
    """    pub fn sequence(&self) -> Option<crate::Sequence> {
        let request_id = self.next_checkout_id();
        let mut guard = CheckoutGuard::new(self.clone(), request_id);
        let reservation = self
            .wait(|reply| Command::CheckoutSeq { request_id, reply })
            .expect("context worker stopped");
        guard.commit();
        reservation.map(|reservation| crate::Sequence::new(self.clone(), reservation))
    }

    pub fn sequence_async(
        &self,
    ) -> impl Future<Output = Option<crate::Sequence>> + Send + 'static {
        let request_id = self.next_checkout_id();
        let mut guard = CheckoutGuard::new(self.clone(), request_id);
        let receiver = self.submit(|reply| Command::CheckoutSeq { request_id, reply });
        let context = self.clone();

        async move {
            let reservation = receiver
                .expect("context worker stopped")
                .await
                .map_err(|_| ContextError::WorkerStopped)
                .and_then(|result| result)
                .expect("context worker stopped");
            guard.commit();
            reservation.map(|reservation| crate::Sequence::new(context, reservation))
        }
    }
""",
    1,
)
context.write_text(s)
