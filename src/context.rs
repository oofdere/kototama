//! Inference contexts.
//!
//! A *context* is where inference actually happens. It owns the KV cache (the
//! model's "short-term memory" of tokens it has already seen) and the scratch
//! buffers llama.cpp needs to run the model. Loading a [`Model`] gives you the
//! weights; creating a [`Context`] from it gives you somewhere to run them.
//!
//! One context can hold several independent conversations at once, called
//! *sequences*. Each sequence gets a numeric id (its "slot") and its own region
//! of the KV cache. You don't manage slots directly: call
//! [`Context::sequence`] to check one out, and it's returned automatically when
//! the [`Sequence`] is dropped.
//!
//! # How this file is laid out
//!
//! There are two layers:
//!
//! 1. [`ContextInner`] — a plain struct that owns the raw `llama_context`
//!    pointer and does all of the `unsafe` FFI calls. Every method that changes
//!    something takes `&mut self`. It knows nothing about sharing.
//!
//! 2. [`Context`] — the public handle. It's just
//!    `Rc<RefCell<ContextInner>>`, which lets several [`Sequence`]s point at the
//!    same context and take turns borrowing it.
//!
//! This split is deliberate. When we add threading, only layer 2 changes:
//! `Rc<RefCell<_>>` becomes `Arc<Mutex<_>>` (or a dedicated worker thread that
//! owns the `ContextInner`), and `ContextInner` stays exactly the same. Keeping
//! all the FFI in one place means there's only one spot to audit for safety.

use std::cell::{RefCell, RefMut};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use llama_sys::*;

use crate::batch::Batch;
use crate::{Model, Pos, SeqId, Sequence, Token};

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Settings used when creating a [`Context`].
///
/// This wraps llama.cpp's `llama_context_params` and derefs to it, so you can
/// set fields directly:
///
/// ```no_run
/// # use rusty_llama::ContextParams;
/// let mut params = ContextParams::new();
/// params.n_ctx = 2048;    // total tokens the KV cache can hold
/// params.n_seq_max = 4;   // how many sequences can be checked out at once
/// ```
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ContextParams(llama_context_params);

impl ContextParams {
    /// llama.cpp's default context settings.
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

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a call to `llama_decode` failed.
///
/// The variants mirror the return codes documented in `llama.h`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// `1`: no room in the KV cache for this batch. Try a bigger `n_ctx`, or
    /// remove old tokens from a sequence.
    SlotNotFound,
    /// `2`: the abort callback stopped the decode.
    Aborted,
    /// `-1`: the batch was invalid (e.g. a position didn't follow on from the
    /// previous one).
    InvalidInput,
    /// Any other code: something went wrong inside llama.cpp.
    FatalError,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            DecodeError::SlotNotFound => "no space left in the KV cache",
            DecodeError::Aborted => "decode was aborted",
            DecodeError::InvalidInput => "invalid batch passed to llama_decode",
            DecodeError::FatalError => "llama_decode failed",
        };
        f.write_str(msg)
    }
}

// Lets `DecodeError` be used with `?` in functions returning
// `Result<_, Box<dyn std::error::Error>>`.
impl std::error::Error for DecodeError {}

// ---------------------------------------------------------------------------
// Layer 1: ContextInner — owns the pointer, does the FFI
// ---------------------------------------------------------------------------

/// The actual context state. See the module docs for why this is separate from
/// [`Context`].
///
/// Raw pointers make this type `!Send` and `!Sync` automatically, which is the
/// safe default: a `llama_context` must never be used by two threads at the
/// same time.
pub(crate) struct ContextInner {
    /// The llama.cpp context. Freed in [`Drop`].
    ctx: *mut llama_context,

    /// Reusable batch for decoding. Allocated once, with room for `n_batch`
    /// tokens (the most llama.cpp will accept in one `llama_decode` call), so
    /// decoding never allocates.
    batch: Batch,

    /// Size of the vocabulary, which is also the length of a logits row.
    n_vocab: usize,

    /// `slots[i]` is `true` while sequence id `i` is checked out.
    slots: Vec<bool>,

    /// Keeps the model (and its weights) alive for as long as this context
    /// exists. A `llama_context` points into the model's memory, so freeing the
    /// model first would leave the context dangling.
    ///
    /// Declared *after* `ctx` and only dropped after our `Drop::drop` has
    /// already freed `ctx`, so the order is always: context first, model second.
    _model: Model,
}

impl ContextInner {
    fn new(model: &Model, params: &ContextParams) -> Result<Self, ()> {
        let ctx = unsafe { llama_init_from_model(model.as_mut_ptr(), params.0) };
        if ctx.is_null() {
            return Err(());
        }

        let n_seq_max = params.n_seq_max as usize;
        // Ask the context rather than reading `params.n_batch`: llama.cpp may
        // adjust the value we asked for.
        let n_batch = unsafe { llama_n_batch(ctx) } as usize;
        Ok(Self {
            ctx,
            batch: Batch::new(n_batch, n_seq_max),
            n_vocab: model.n_tokens() as usize,
            slots: vec![false; n_seq_max],
            _model: model.clone(),
        })
    }

    fn memory(&self) -> llama_memory_t {
        unsafe { llama_get_memory(self.ctx) }
    }

    // -- Slot bookkeeping --------------------------------------------------

    /// Claim the lowest free sequence id, or `None` if all are taken.
    pub(crate) fn checkout_seq(&mut self) -> Option<SeqId> {
        let free = self.slots.iter().position(|taken| !taken)?;
        self.slots[free] = true;
        Some(free as SeqId)
    }

    /// Wipe a sequence's KV cache entries and mark its id as free again.
    pub(crate) fn release_seq(&mut self, seq_id: SeqId) {
        // -1, -1 means "the whole sequence".
        self.seq_rm(seq_id, -1, -1);
        if let Some(slot) = self.slots.get_mut(seq_id as usize) {
            *slot = false;
        }
    }

    pub(crate) fn free_slots(&self) -> usize {
        self.slots.iter().filter(|taken| !**taken).count()
    }

    // -- Decoding ----------------------------------------------------------

    /// The most tokens [`decode`](Self::decode) accepts in one call.
    pub(crate) fn batch_size(&self) -> usize {
        self.batch.capacity()
    }

    /// Run the model on `tokens`, which go into sequence `seq_id` at positions
    /// `start_pos`, `start_pos + 1`, ... and return the logits for the *last*
    /// token.
    ///
    /// All the tokens are processed in a single `llama_decode` call, which is
    /// much faster than one call per token. `tokens` must not be empty and
    /// must be no longer than [`batch_size`](Self::batch_size); split longer
    /// inputs into chunks first.
    ///
    /// The returned slice points straight into llama.cpp's own output buffer —
    /// nothing is copied. That buffer gets overwritten by the next decode, which
    /// is why the slice borrows `self`: the borrow checker won't let you decode
    /// again while you're still holding on to it. If you need the logits to
    /// stick around, copy them (that's what [`Sequence`] does).
    pub(crate) fn decode(
        &mut self,
        tokens: &[Token],
        start_pos: Pos,
        seq_id: SeqId,
    ) -> Result<&[f32], DecodeError> {
        if tokens.is_empty() {
            return Err(DecodeError::InvalidInput);
        }

        self.batch.clear();
        let last = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            // Only ask for logits on the last token: that's the only one we
            // return, and computing logits for every token would waste time.
            let want_logits = i == last;
            self.batch
                .add(token, start_pos + i as Pos, &[seq_id], want_logits)
                .map_err(|_| DecodeError::InvalidInput)?;
        }

        match unsafe { llama_decode(self.ctx, self.batch.raw()) } {
            0 => {}
            1 => return Err(DecodeError::SlotNotFound),
            2 => return Err(DecodeError::Aborted),
            -1 => return Err(DecodeError::InvalidInput),
            _ => return Err(DecodeError::FatalError),
        }

        // `ith` is an index into the batch we just decoded, so `last` is the
        // final token's row.
        let ptr = unsafe { llama_get_logits_ith(self.ctx, last as i32) };
        if ptr.is_null() {
            return Err(DecodeError::FatalError);
        }
        // SAFETY: llama.cpp guarantees a non-null logits row is `n_vocab` long,
        // and it stays valid until the next decode, which can't happen while
        // this `&self` borrow is alive.
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) })
    }

    // -- KV cache ("memory") operations --------------------------------------
    //
    // These are thin wrappers over `llama_memory_seq_*`. Positions use
    // half-open ranges `[p0, p1)`, and a negative value means "unbounded" on
    // that side.

    /// Remove positions `[p0, p1)` of a sequence from the KV cache.
    pub(crate) fn seq_rm(&mut self, seq_id: SeqId, p0: Pos, p1: Pos) -> bool {
        unsafe { llama_memory_seq_rm(self.memory(), seq_id, p0, p1) }
    }

    /// Copy positions `[p0, p1)` from one sequence to another.
    pub(crate) fn seq_cp(&mut self, src: SeqId, dst: SeqId, p0: Pos, p1: Pos) {
        unsafe { llama_memory_seq_cp(self.memory(), src, dst, p0, p1) }
    }

    /// Shift positions `[p0, p1)` of a sequence by `delta`.
    pub(crate) fn seq_add(&mut self, seq_id: SeqId, p0: Pos, p1: Pos, delta: Pos) {
        unsafe { llama_memory_seq_add(self.memory(), seq_id, p0, p1, delta) }
    }

    /// Smallest position stored for a sequence, or -1 if it's empty.
    pub(crate) fn seq_pos_min(&self, seq_id: SeqId) -> Pos {
        unsafe { llama_memory_seq_pos_min(self.memory(), seq_id) }
    }

    /// Largest position stored for a sequence, or -1 if it's empty.
    pub(crate) fn seq_pos_max(&self, seq_id: SeqId) -> Pos {
        unsafe { llama_memory_seq_pos_max(self.memory(), seq_id) }
    }

    // -- Simple queries ----------------------------------------------------

    fn n_ctx(&self) -> u32 {
        unsafe { llama_n_ctx(self.ctx) }
    }

    fn can_shift(&self) -> bool {
        unsafe { llama_memory_can_shift(self.memory()) }
    }

    fn perf(&self) -> llama_perf_context_data {
        unsafe { llama_perf_context(self.ctx) }
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // `batch` and `_model` are dropped automatically right after this.
        unsafe { llama_free(self.ctx) };
    }
}

// ---------------------------------------------------------------------------
// Layer 2: Context — the shareable public handle
// ---------------------------------------------------------------------------

/// A handle to an inference context.
///
/// Cloning a `Context` is cheap and gives you another handle to the *same*
/// context (it's reference counted). The context is freed when the last handle
/// — including the ones held inside [`Sequence`]s — is dropped.
///
/// `Context` is single-threaded: it is neither `Send` nor `Sync`. Everything
/// happens on the thread that created it, and each method call simply runs to
/// completion before returning.
///
/// ```no_run
/// # use rusty_llama::*;
/// # let model = Model::load_from_file("model.gguf", ModelParams::new()).unwrap();
/// let ctx = Context::new(&model, &ContextParams::new()).unwrap();
/// let mut seq = ctx.sequence().expect("no free sequence slots");
/// seq.extend(&model.tokenize("Once upon a time", true, false)).unwrap();
/// let logits = seq.logits().unwrap();
/// ```
#[derive(Clone)]
pub struct Context {
    inner: Rc<RefCell<ContextInner>>,
}

impl Context {
    /// Create a new context for `model`.
    ///
    /// The context keeps its own handle to the model, so it's fine to drop
    /// `model` afterwards.
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ()> {
        let inner = ContextInner::new(model, params)?;
        Ok(Self {
            inner: Rc::new(RefCell::new(inner)),
        })
    }

    /// Borrow the inner state for the duration of one call.
    ///
    /// `RefCell` checks at runtime that nobody else is borrowing it at the same
    /// time. That can't happen here because every method borrows, does one
    /// thing, and lets go before returning — nothing ever holds the borrow
    /// across a call back into user code.
    pub(crate) fn inner(&self) -> RefMut<'_, ContextInner> {
        self.inner.borrow_mut()
    }

    /// Check out a free sequence slot.
    ///
    /// Returns `None` if all `n_seq_max` slots are already in use. The slot is
    /// given back when the returned [`Sequence`] is dropped.
    pub fn sequence(&self) -> Option<Sequence> {
        let seq_id = self.inner().checkout_seq()?;
        Some(Sequence::new(self.clone(), seq_id))
    }

    /// How many sequence slots are still available.
    pub fn free_slots(&self) -> usize {
        self.inner().free_slots()
    }

    /// The actual context size in tokens. May be larger than what you asked
    /// for in `ContextParams::n_ctx` because llama.cpp rounds it up.
    pub fn n_ctx(&self) -> u32 {
        self.inner().n_ctx()
    }

    /// Whether the KV cache supports shifting positions (see
    /// [`Sequence::kv_shift`]).
    pub fn can_shift(&self) -> bool {
        self.inner().can_shift()
    }

    /// Timing counters collected by llama.cpp. Only meaningful if the context
    /// was created with `no_perf = false`.
    pub fn perf(&self) -> llama_perf_context_data {
        self.inner().perf()
    }
}
