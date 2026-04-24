use std::ops::{Deref, DerefMut};

#[repr(transparent)]
pub struct Sampler(*mut llama_sys::llama_sampler);

impl Sampler {
    pub fn greedy() -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_greedy() })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            llama_sys::llama_sampler_free(self.0);
        }
    }
}

impl Deref for Sampler {
    type Target = *mut llama_sys::llama_sampler;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Sampler {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait LlamaSamplerPtr {
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler;
}

impl LlamaSamplerPtr for Sampler {
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler {
        self.0
    }
}

// Shorthand for: const auto * logits = llama_get_logits_ith(ctx, idx); llama_token_data_array cur_p = { ... init from logits ... }; llama_sampler_apply(smpl, &cur_p); auto token = cur_p.datacur_p.selected.id; llama_sampler_accept(smpl, token); return token; Returns the sampled token
