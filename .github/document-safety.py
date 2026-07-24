from pathlib import Path

context = Path("src/context.rs")
text = context.read_text()

marker = "/// Parameters used to create a [`Context`]."
if marker not in text:
    anchor = "#[repr(transparent)]\n#[derive(Clone, Copy)]\npub struct ContextParams(pub(crate) llama_context_params);"
    replacement = '''/// Parameters used to create a [`Context`].
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
pub struct ContextParams(pub(crate) llama_context_params);'''
    if text.count(anchor) != 1:
        raise SystemExit(f"ContextParams anchor count: {text.count(anchor)}")
    text = text.replace(anchor, replacement, 1)

old_errors = '''#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextInitError {
    ThreadUnsafeParams,
    ThreadSpawnFailed,
    WorkerStopped,
    NativeInitFailed,
}'''
new_errors = '''/// Errors that can occur while creating a [`Context`].
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
}'''
if old_errors in text:
    text = text.replace(old_errors, new_errors, 1)
elif "/// Errors that can occur while creating a [`Context`]." not in text:
    raise SystemExit("ContextInitError block not found")

old_ctor = '''impl Context {
    pub fn new(model: &Model, params: &ContextParams) -> Result<Self, ContextInitError> {
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
    }'''
new_ctor = '''impl Context {
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
    }'''
if old_ctor in text:
    text = text.replace(old_ctor, new_ctor, 1)
elif "This is an FFI escape hatch for features" not in text:
    raise SystemExit("Context constructor block not found")

context.write_text(text)

samplers = Path("src/samplers/mod.rs")
text = samplers.read_text()
old_module = '''//! Sampling primitives.
//!
//! Each sampler implements [`Sampler`]. Logit-transforming samplers (e.g.
//! [`Temperature`], [`TopK`], [`MinP`]) override [`Sampler::apply_mut`], while
//! token-selecting samplers ([`Greedy`], [`Dist`]) override [`Sampler::sample`].
//! A [`Chain`] composes several samplers into a pipeline.
'''
new_module = '''//! Rust-side sampling primitives.
//!
//! Each sampler implements [`Sampler`]. Logit-transforming samplers (e.g.
//! [`Temperature`], [`TopK`], [`MinP`]) override [`Sampler::apply_mut`], while
//! token-selecting samplers ([`Greedy`], [`Dist`]) override [`Sampler::sample`].
//! A [`Chain`] composes several samplers into a pipeline.
//!
//! These samplers run in Rust on the thread that calls
//! [`crate::Sequence::sample`], using an owned logits snapshot returned by the
//! context worker. They are not llama.cpp's experimental native backend sampler
//! chains stored in `llama_context_params::samplers`. Native backend samplers
//! contain raw pointers and are rejected by the safe [`crate::Context::new`]
//! constructor; configuring them requires [`crate::Context::new_unchecked`] and
//! its documented safety contract.
'''
if old_module in text:
    text = text.replace(old_module, new_module, 1)
elif "They are not llama.cpp's experimental native backend sampler" not in text:
    raise SystemExit("samplers module docs not found")
samplers.write_text(text)

sequence = Path("src/sequence.rs")
text = sequence.read_text()
old_sample = '''    pub fn sample<S: Sampler>(&self, sampler: &mut S) -> Option<Token> {
        let logits = self.logits()?;
        Some(sampler.sample(&logits))
    }'''
new_sample = '''    /// Samples from the latest logits snapshot using a Rust [`Sampler`].
    ///
    /// Sampling runs synchronously on the caller's thread. The sampler receives
    /// an owned snapshot of the logits; it is not moved to or retained by the
    /// context worker. This API is unrelated to llama.cpp's experimental native
    /// backend sampler chains in `llama_context_params::samplers`.
    ///
    /// Returns `None` until this sequence has produced logits.
    pub fn sample<S: Sampler>(&self, sampler: &mut S) -> Option<Token> {
        let logits = self.logits()?;
        Some(sampler.sample(&logits))
    }'''
if old_sample in text:
    text = text.replace(old_sample, new_sample, 1)
elif "Sampling runs synchronously on the caller's thread" not in text:
    raise SystemExit("Sequence::sample block not found")
sequence.write_text(text)
