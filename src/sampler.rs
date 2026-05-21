//! Token samplers.
//!
//! A [`Sampler`] wraps a single `llama_sampler` from llama.cpp. Samplers are
//! the building blocks of a sampling pipeline: most of them transform the
//! token probability distribution (truncation, temperature, penalties), while
//! a few — `greedy`, `dist`, `mirostat`, `mirostat_v2`, `adaptive_p` —
//! actually select the final token and therefore must come last in a chain.
//!
//! Combine several samplers with a [`SamplerChain`](crate::SamplerChain), then
//! draw a token with [`Context::sample`](crate::Context::sample).

use llama_sys::{llama_logit_bias, llama_token, llama_vocab};

/// A single llama.cpp sampler.
///
/// Construct one with the associated functions below and add it to a
/// [`SamplerChain`](crate::SamplerChain). The underlying sampler is freed when
/// the `Sampler` is dropped; moving it into a chain with
/// [`SamplerChain::add`](crate::SamplerChain::add) transfers ownership to the
/// chain instead.
#[repr(transparent)]
pub struct Sampler(*mut llama_sys::llama_sampler);

impl Sampler {
    // todo these really are not rusty

    /// Adaptive-p sampler: steadily selects tokens whose probability sits near
    /// a configurable target.
    ///
    /// It keeps an exponential moving average of the original probabilities of
    /// the tokens it picks and adapts the target accordingly. Because it
    /// selects a token outright, it must be the last sampler in the chain.
    ///
    /// - `target`: probability to aim for, in `0.0..=1.0` (negative disables it)
    /// - `decay`: EMA decay, in `0.0..=0.99`; history ≈ `1 / (1 - decay)` tokens
    /// - `seed`: RNG seed
    pub fn adaptive_p(target: f32, decay: f32, seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_adaptive_p(target, decay, seed) })
    }

    /// Fill-in-the-middle infill sampler.
    ///
    /// Meant to run after `top_k` + `top_p`. It picks an end-of-generation
    /// token when the EOG probability mass dominates, merges candidates that
    /// share a prefix, discards low-probability non-EOG tokens, and falls back
    /// to the end-of-text token if nothing is left.
    pub fn infill(vocab: *const llama_vocab) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_infill(vocab) })
    }

    /// Applies fixed per-token biases to the logits.
    ///
    /// `logit_bias` points to `n_logit_bias` `llama_logit_bias` entries; pass
    /// `0` and a null pointer for no bias. `n_vocab` is the model's vocabulary
    /// size.
    pub fn logit_bias(
        n_vocab: i32,
        n_logit_bias: i32,
        logit_bias: *const llama_logit_bias,
    ) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias) })
    }

    /// Repetition, frequency, and presence penalties over the last
    /// `penalty_last_n` tokens.
    ///
    /// Avoid applying this over the full vocabulary — scanning for repeated
    /// tokens is slow; truncate with `top_k`/`top_p` first.
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

    /// XTC ("exclude top choices") sampler.
    ///
    /// With probability `p` it removes all but the least likely of the tokens
    /// above threshold `t`, encouraging more varied output. `min_keep` is the
    /// minimum number of candidates to retain; `seed` seeds the RNG.
    pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_xtc(p, t, min_keep, seed) })
    }

    /// Samples a token from the (transformed) probability distribution.
    ///
    /// This is a token-selecting sampler and is typically the last link in a
    /// chain. Pass `LLAMA_DEFAULT_SEED` as `seed` to use a random seed.
    pub fn dist(seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_dist(seed) })
    }

    /// DRY ("don't repeat yourself") repetition penalty.
    ///
    /// Penalizes repeated token sequences more aggressively than the standard
    /// [`penalties`](Sampler::penalties) sampler. `seq_breakers` points to
    /// `num_breakers` C strings that reset the repetition window; pass a null
    /// pointer and `0` for none.
    pub fn dry(
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

    /// Constrains generation to a GBNF grammar.
    ///
    /// `grammar_str` holds the grammar's production rules and `grammar_root`
    /// names its start symbol, both as C strings. An empty grammar string
    /// yields an unconstrained grammar; a grammar that fails to parse makes the
    /// underlying constructor return a null pointer.
    pub fn grammar(
        vocab: *const llama_vocab,
        grammar_str: *const ::std::os::raw::c_char,
        grammar_root: *const ::std::os::raw::c_char,
    ) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_grammar(vocab, grammar_str, grammar_root) })
    }

    /// Always selects the most probable token (argmax).
    ///
    /// A token-selecting sampler; place it last in the chain.
    pub fn greedy() -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_greedy() })
    }

    /// Min-p sampling: keeps only tokens whose probability is at least `p`
    /// times the probability of the most likely token.
    ///
    /// `min_keep` is the minimum number of candidates to retain.
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_min_p(p, min_keep) })
    }

    /// Mirostat 1.0 sampling, which steers output toward a target perplexity.
    ///
    /// - `n_vocab`: the model's vocabulary size
    /// - `seed`: RNG seed
    /// - `tau`: target cross-entropy (surprise); higher is more unpredictable
    /// - `eta`: learning rate for the internal `mu` estimate
    /// - `m`: number of tokens used to estimate `s_hat` (the paper uses 100)
    ///
    /// Selects a token, so it must be last in the chain.
    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) })
    }

    /// Mirostat 2.0 sampling — like [`mirostat`](Sampler::mirostat) but without
    /// the `n_vocab`/`m` estimation step.
    ///
    /// - `seed`: RNG seed
    /// - `tau`: target cross-entropy (surprise)
    /// - `eta`: learning rate for the internal `mu` estimate
    ///
    /// Selects a token, so it must be last in the chain.
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_mirostat_v2(seed, tau, eta) })
    }

    /// Temperature scaling: divides every logit by `temp`.
    ///
    /// Lower values sharpen the distribution, higher values flatten it. When
    /// `temp <= 0.0` the most likely logit is kept and all others are set to
    /// negative infinity (equivalent to greedy selection).
    pub fn temp(temp: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_temp(temp) })
    }

    /// Locally typical sampling: keeps the smallest set of tokens whose
    /// information content is closest to the distribution's entropy.
    ///
    /// `min_keep` is the minimum number of candidates to retain.
    pub fn typical(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_typical(p, min_keep) })
    }

    /// Lazy GBNF grammar sampler activated by trigger words or trigger tokens.
    ///
    /// The grammar starts enforcing constraints only once a trigger appears in
    /// the output.
    ///
    /// Wraps `llama_sampler_init_grammar_lazy`, which is **deprecated upstream**
    /// in favor of [`grammar_lazy_patterns`](Sampler::grammar_lazy_patterns);
    /// prefer that constructor.
    pub fn grammar_lazy(
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

    /// Dynamic ("extended") temperature sampling, which varies the effective
    /// temperature with the entropy of the distribution.
    ///
    /// `temp` is the base temperature; `delta` and `exponent` control how far
    /// and how sharply it is adjusted.
    pub fn temp_ext(temp: f32, delta: f32, exponent: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_temp_ext(temp, delta, exponent) })
    }

    /// Top-k sampling: keeps only the `k` most probable tokens.
    ///
    /// Setting `k <= 0` makes this a no-op.
    pub fn top_k(k: i32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_k(k) })
    }

    /// Top-nσ sampling: keeps tokens whose logit lies within `n` standard
    /// deviations of the maximum logit.
    pub fn top_n_sigma(n: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_n_sigma(n) })
    }

    /// Top-p (nucleus) sampling: keeps the smallest set of most probable
    /// tokens whose cumulative probability is at least `p`.
    ///
    /// `min_keep` is the minimum number of candidates to retain.
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_p(p, min_keep) })
    }

    /// Lazy GBNF grammar sampler activated by regex patterns or trigger tokens.
    ///
    /// Each pattern in `trigger_patterns` is matched from the start of the
    /// output; once one matches, the grammar is fed content starting from its
    /// first capture group. A trigger token feeds content starting from the
    /// token itself. This is the preferred replacement for
    /// [`grammar_lazy`](Sampler::grammar_lazy).
    pub fn grammar_lazy_patterns(
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

/// Types that expose a raw `llama_sampler` pointer.
///
/// Implemented by [`Sampler`] and [`SamplerChain`](crate::SamplerChain) so both
/// can be passed to [`Context::sample`](crate::Context::sample).
pub trait LlamaSampler {
    /// Returns the raw `llama_sampler` pointer this wrapper owns.
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler;
}

impl LlamaSampler for Sampler {
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler {
        self.0
    }
}

/// Cloning a `Sampler` deep-copies the underlying llama.cpp sampler, including
/// any internal state it carries.
impl Clone for Sampler {
    fn clone(&self) -> Self {
        Self(unsafe { llama_sys::llama_sampler_clone(self.0) })
    }
}
