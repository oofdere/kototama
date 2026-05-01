use crate::{LlamaSampler, Sampler};
use std::ops::{Deref, DerefMut};

#[repr(transparent)]
pub struct SamplerChainParams(llama_sys::llama_sampler_chain_params);

impl SamplerChainParams {
    pub fn new() -> Self {
        Self(unsafe { llama_sys::llama_sampler_chain_default_params() })
    }

    pub fn as_ptr(&self) -> *const llama_sys::llama_sampler_chain_params {
        &self.0
    }

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

#[repr(transparent)]
pub struct SamplerChain(*mut llama_sys::llama_sampler);

impl SamplerChain {
    pub fn new(params: &SamplerChainParams) -> Self {
        Self(unsafe { llama_sys::llama_sampler_chain_init(params.0) })
    }

    pub fn add(self, sampler: Sampler) -> Self {
        unsafe {
            llama_sys::llama_sampler_chain_add(self.0, sampler.as_ptr());
            std::mem::forget(sampler); // ownership of sampler gets moved to chain
        }
        self
    }

    pub fn perf(&self) -> llama_sys::llama_perf_sampler_data {
        unsafe { llama_sys::llama_perf_sampler(self.0) }
    }

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
