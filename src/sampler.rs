use llama_sys::{llama_logit_bias, llama_token, llama_vocab};

#[repr(transparent)]
pub struct Sampler(*mut llama_sys::llama_sampler);

impl Sampler {
    // todo these really are not rusty
    pub fn adaptive_p(target: f32, decay: f32, seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_adaptive_p(target, decay, seed) })
    }

    /// # Safety
    ///
    /// `vocab` must be a non-null pointer to a `llama_vocab` owned by a
    /// `Model` that remains alive for the lifetime of the returned `Sampler`
    /// (including any `SamplerChain` the sampler is later added to). The
    /// initializer dereferences `vocab` and the sampler retains it for
    /// later sampling calls.
    pub unsafe fn infill(vocab: *const llama_vocab) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_infill(vocab) })
    }

    /// # Safety
    ///
    /// `logit_bias` must either be null (when `n_logit_bias` is 0) or point
    /// to a contiguous array of at least `n_logit_bias` valid
    /// `llama_logit_bias` entries. The data is read by llama.cpp during
    /// initialization, so it must remain valid for the duration of the call.
    pub unsafe fn logit_bias(
        n_vocab: i32,
        n_logit_bias: i32,
        logit_bias: *const llama_logit_bias,
    ) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias) })
    }

    pub fn penalties(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Self {
        Self(unsafe {
            llama_sys::llama_sampler_init_penalties(
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            )
        })
    }

    pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_xtc(p, t, min_keep, seed) })
    }

    pub fn dist(seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_dist(seed) })
    }

    /// # Safety
    ///
    /// - `vocab` must be a non-null pointer to a `llama_vocab` owned by a
    ///   `Model` that remains alive for the lifetime of the returned
    ///   `Sampler` (including any `SamplerChain` the sampler is later added
    ///   to).
    /// - `seq_breakers` must either be null (when `num_breakers` is 0) or
    ///   point to a contiguous array of at least `num_breakers` valid,
    ///   non-null pointers to NUL-terminated C strings. Each string must
    ///   remain valid for the duration of the call.
    pub unsafe fn dry(
        vocab: *const llama_vocab,
        n_ctx_train: i32,
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: i32,
        dry_penalty_last_n: i32,
        seq_breakers: *mut *const ::std::os::raw::c_char,
        num_breakers: usize,
    ) -> Self {
        Self(unsafe {
            llama_sys::llama_sampler_init_dry(
                vocab,
                n_ctx_train,
                dry_multiplier,
                dry_base,
                dry_allowed_length,
                dry_penalty_last_n,
                seq_breakers,
                num_breakers,
            )
        })
    }

    /// # Safety
    ///
    /// - `vocab` must be a non-null pointer to a `llama_vocab` owned by a
    ///   `Model` that remains alive for the lifetime of the returned
    ///   `Sampler` (including any `SamplerChain` the sampler is later added
    ///   to).
    /// - `grammar_str` and `grammar_root` must each be non-null pointers to
    ///   NUL-terminated C strings that remain valid for the duration of
    ///   the call.
    pub unsafe fn grammar(
        vocab: *const llama_vocab,
        grammar_str: *const ::std::os::raw::c_char,
        grammar_root: *const ::std::os::raw::c_char,
    ) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_grammar(vocab, grammar_str, grammar_root) })
    }

    pub fn greedy() -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_greedy() })
    }

    pub fn min_p(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_min_p(p, min_keep) })
    }

    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) })
    }

    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_mirostat_v2(seed, tau, eta) })
    }

    pub fn temp(temp: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_temp(temp) })
    }

    pub fn typical(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_typical(p, min_keep) })
    }

    /// # Safety
    ///
    /// - `vocab` must be a non-null pointer to a `llama_vocab` owned by a
    ///   `Model` that remains alive for the lifetime of the returned
    ///   `Sampler` (including any `SamplerChain` the sampler is later added
    ///   to).
    /// - `grammar_str` and `grammar_root` must each be non-null pointers to
    ///   NUL-terminated C strings that remain valid for the duration of
    ///   the call.
    /// - `trigger_words` must either be null (when `num_trigger_words` is 0)
    ///   or point to a contiguous array of at least `num_trigger_words`
    ///   valid, non-null pointers to NUL-terminated C strings.
    /// - `trigger_tokens` must either be null (when `num_trigger_tokens` is
    ///   0) or point to a contiguous array of at least `num_trigger_tokens`
    ///   valid `llama_token` values.
    pub unsafe fn grammar_lazy(
        vocab: *const llama_vocab,
        grammar_str: *const ::std::os::raw::c_char,
        grammar_root: *const ::std::os::raw::c_char,
        trigger_words: *mut *const ::std::os::raw::c_char,
        num_trigger_words: usize,
        trigger_tokens: *const llama_token,
        num_trigger_tokens: usize,
    ) -> Self {
        Self(unsafe {
            llama_sys::llama_sampler_init_grammar_lazy(
                vocab,
                grammar_str,
                grammar_root,
                trigger_words,
                num_trigger_words,
                trigger_tokens,
                num_trigger_tokens,
            )
        })
    }

    pub fn temp_ext(temp: f32, delta: f32, exponent: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_temp_ext(temp, delta, exponent) })
    }

    pub fn top_k(k: i32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_k(k) })
    }

    pub fn top_n_sigma(n: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_n_sigma(n) })
    }

    pub fn top_p(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_p(p, min_keep) })
    }

    /// # Safety
    ///
    /// - `vocab` must be a non-null pointer to a `llama_vocab` owned by a
    ///   `Model` that remains alive for the lifetime of the returned
    ///   `Sampler` (including any `SamplerChain` the sampler is later added
    ///   to).
    /// - `grammar_str` and `grammar_root` must each be non-null pointers to
    ///   NUL-terminated C strings that remain valid for the duration of
    ///   the call.
    /// - `trigger_patterns` must either be null (when `num_trigger_patterns`
    ///   is 0) or point to a contiguous array of at least
    ///   `num_trigger_patterns` valid, non-null pointers to NUL-terminated
    ///   C strings.
    /// - `trigger_tokens` must either be null (when `num_trigger_tokens` is
    ///   0) or point to a contiguous array of at least `num_trigger_tokens`
    ///   valid `llama_token` values.
    pub unsafe fn grammar_lazy_patterns(
        vocab: *const llama_vocab,
        grammar_str: *const ::std::os::raw::c_char,
        grammar_root: *const ::std::os::raw::c_char,
        trigger_patterns: *mut *const ::std::os::raw::c_char,
        num_trigger_patterns: usize,
        trigger_tokens: *const llama_token,
        num_trigger_tokens: usize,
    ) -> Self {
        Self(unsafe {
            llama_sys::llama_sampler_init_grammar_lazy_patterns(
                vocab,
                grammar_str,
                grammar_root,
                trigger_patterns,
                num_trigger_patterns,
                trigger_tokens,
                num_trigger_tokens,
            )
        })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            llama_sys::llama_sampler_free(self.0);
        }
    }
}

pub trait LlamaSampler {
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler;
}

impl LlamaSampler for Sampler {
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler {
        self.0
    }
}

impl Clone for Sampler {
    fn clone(&self) -> Self {
        Self(unsafe { llama_sys::llama_sampler_clone(self.0) })
    }
}

// Shorthand for: const auto * logits = llama_get_logits_ith(ctx, idx); llama_token_data_array cur_p = { ... init from logits ... }; llama_sampler_apply(smpl, &cur_p); auto token = cur_p.datacur_p.selected.id; llama_sampler_accept(smpl, token); return token; Returns the sampled token
