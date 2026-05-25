//! Sampler chains.
//!
//! A [`SamplerChain`] is a sequence of [`Sampler`]s applied in order to a
//! token distribution. It is itself a [`LlamaSampler`], so it can be passed
//! anywhere a single sampler is expected — most importantly to
//! [`Context::sample`](crate::Context::sample).
//!
//! ## Ownership
//!
//! Wraps the `llama_sampler_chain_*` family from upstream
//! `llama.cpp/include/llama.h` (pinned at `b9246` per the README). The
//! underlying llama.cpp object is a regular `llama_sampler` of the
//! chain-implementation type: it owns its child samplers and frees them when
//! the chain itself is freed.
//!
//! Mirroring that, [`SamplerChain::add`] takes a [`Sampler`] by value and
//! transfers ownership of the underlying `llama_sampler` to the chain. The
//! Rust [`Sampler`]'s [`Drop`](Drop) is suppressed via [`std::mem::forget`] so
//! the same object is not freed twice when the chain is later dropped.
//!
//! ## Building a chain
//!
//! Convention is to add transforming samplers first (`top_k`, `top_p`,
//! `temp`, `penalties`, …) and finish with a selecting sampler (`dist`,
//! `greedy`, `mirostat*`) that actually picks a token. See [`crate::Sampler`]
//! for the per-sampler distinction.

use crate::{LlamaSampler, Sampler};
use std::ops::{Deref, DerefMut};

/// Configuration for constructing a [`SamplerChain`].
///
/// Thin wrapper over `llama_sampler_chain_params`. Construct with
/// [`SamplerChainParams::new`] (which returns the upstream defaults from
/// `llama_sampler_chain_default_params`) and tweak fields through the
/// [`Deref`] / [`DerefMut`] impls — currently the only field is `no_perf`,
/// which disables per-sampler performance timing.
#[repr(transparent)]
pub struct SamplerChainParams(llama_sys::llama_sampler_chain_params);

impl SamplerChainParams {
    /// Returns the upstream default chain parameters
    /// (`llama_sampler_chain_default_params`).
    pub fn new() -> Self {
        Self(unsafe { llama_sys::llama_sampler_chain_default_params() })
    }

    /// Borrow the wrapped `llama_sampler_chain_params` as a raw pointer.
    ///
    /// Useful for handing the params straight to a llama.cpp FFI call that
    /// expects a `const` pointer. The pointer is valid for as long as `self`
    /// is borrowed.
    pub fn as_ptr(&self) -> *const llama_sys::llama_sampler_chain_params {
        &self.0
    }

    /// Mutably borrow the wrapped `llama_sampler_chain_params` as a raw
    /// pointer.
    ///
    /// The pointer is valid for as long as `self` is mutably borrowed.
    pub fn as_mut_ptr(&mut self) -> *mut llama_sys::llama_sampler_chain_params {
        &mut self.0
    }
}

impl Deref for SamplerChainParams {
    type Target = llama_sys::llama_sampler_chain_params;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SamplerChainParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An ordered chain of [`Sampler`]s.
///
/// Built with [`SamplerChain::new`] and extended with [`SamplerChain::add`],
/// which takes each [`Sampler`] by value and transfers ownership of the
/// underlying `llama_sampler` into the chain.
///
/// The chain is itself a `llama_sampler` (the chain-implementation variant)
/// and implements [`LlamaSampler`], so it can be passed to
/// [`Context::sample`](crate::Context::sample) like any other sampler.
///
/// When the chain is dropped, llama.cpp frees the chain *and* every sampler
/// it owns in a single `llama_sampler_free` call.
#[repr(transparent)]
pub struct SamplerChain(*mut llama_sys::llama_sampler);

impl SamplerChain {
    /// Create an empty sampler chain configured by `params`.
    ///
    /// Wraps `llama_sampler_chain_init`. The chain starts with no samplers;
    /// add them with [`SamplerChain::add`].
    pub fn new(params: &SamplerChainParams) -> Self {
        Self(unsafe { llama_sys::llama_sampler_chain_init(params.0) })
    }

    /// Append `sampler` to the chain and return the chain by value.
    ///
    /// Wraps `llama_sampler_chain_add`. The chain takes ownership of the
    /// underlying `llama_sampler`; the Rust [`Sampler`] is consumed and its
    /// [`Drop`] is suppressed via [`std::mem::forget`] so the same FFI object
    /// is not freed twice when the chain is later dropped.
    ///
    /// Returning `Self` lets callers build a chain fluently:
    ///
    /// ```ignore
    /// let chain = SamplerChain::new(&SamplerChainParams::new())
    ///     .add(Sampler::top_k(40))
    ///     .add(Sampler::top_p(0.95, 1))
    ///     .add(Sampler::temp(0.8))
    ///     .add(Sampler::dist(42));
    /// ```
    pub fn add(self, sampler: Sampler) -> Self {
        unsafe {
            llama_sys::llama_sampler_chain_add(self.0, sampler.as_ptr());
            std::mem::forget(sampler); // ownership of sampler gets moved to chain
        }
        self
    }

    /// Fetch llama.cpp's per-chain performance counters
    /// (`llama_perf_sampler`).
    ///
    /// Returns total sample time and the number of samples taken through
    /// this chain. The counters are populated only when the chain was built
    /// with `no_perf == false` (the upstream default).
    pub fn perf(&self) -> llama_sys::llama_perf_sampler_data {
        unsafe { llama_sys::llama_perf_sampler(self.0) }
    }

    /// Return the raw `*mut llama_sampler` for the chain.
    ///
    /// Intended for handing the chain to an FFI call that needs a raw
    /// pointer. Note that `self` is still dropped at the end of the
    /// surrounding scope and its [`Drop`] will call `llama_sampler_free` on
    /// this pointer — the returned pointer is therefore only valid for as
    /// long as the originating [`SamplerChain`] is alive.
    pub fn into_raw(self) -> *mut llama_sys::llama_sampler {
        let ptr = self.0;
        ptr
    }
}

impl LlamaSampler for SamplerChain {
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler {
        self.0
    }
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        unsafe {
            llama_sys::llama_sampler_free(self.0);
        }
    }
}
