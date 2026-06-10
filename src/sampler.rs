//! Individual samplers.
//!
//! A [`Sampler`] is a thin owning wrapper around a single `llama_sampler`
//! object from upstream `llama.cpp/include/llama.h` (pinned at `b9246` per
//! the README). The constructors on `impl Sampler` mirror the
//! `llama_sampler_init_*` family one-for-one.
//!
//! ## Transformers vs pickers
//!
//! Most samplers in llama.cpp fall into one of two categories:
//!
//! - **Transformers** reshape the candidate token distribution but do not
//!   pick a token. Examples: [`Sampler::top_k`], [`Sampler::top_p`],
//!   [`Sampler::min_p`], [`Sampler::typical`], [`Sampler::temp`],
//!   [`Sampler::temp_ext`], [`Sampler::xtc`], [`Sampler::top_n_sigma`],
//!   [`Sampler::penalties`], [`Sampler::dry`], [`Sampler::logit_bias`],
//!   [`Sampler::infill`], and the `grammar*` family.
//! - **Pickers** consume the (possibly transformed) distribution and emit a
//!   token id. Examples: [`Sampler::greedy`], [`Sampler::dist`],
//!   [`Sampler::mirostat`], [`Sampler::mirostat_v2`],
//!   [`Sampler::adaptive_p`].
//!
//! A complete sampling pipeline is built by chaining transformers in front
//! of exactly one picker via [`SamplerChain`](crate::SamplerChain). A bare
//! [`Sampler`] is rarely useful on its own; see the [`SamplerChain`] module
//! docs for the canonical fluent-builder pattern.
//!
//! ## Ownership
//!
//! Each [`Sampler`] owns its underlying `*mut llama_sampler`. Dropping the
//! Rust value calls `llama_sampler_free` on the FFI object. Moving the
//! sampler into a [`SamplerChain`](crate::SamplerChain) via
//! [`SamplerChain::add`](crate::SamplerChain::add) transfers ownership to
//! the chain — the per-sampler `Drop` is suppressed via
//! [`std::mem::forget`], so the same FFI object is not freed twice.
//!
//! [`Clone`] calls `llama_sampler_clone`, producing an independent owning
//! handle (algorithm state — RNG, mirostat `mu`, adaptive-p EMA — is copied
//! at the moment of cloning).

use llama_sys::{llama_logit_bias, llama_token, llama_vocab};

/// An owning handle to a single llama.cpp sampler.
///
/// Construct via one of the `Sampler::*` associated functions, which mirror
/// the upstream `llama_sampler_init_*` family. Pass to a
/// [`SamplerChain`](crate::SamplerChain) via
/// [`SamplerChain::add`](crate::SamplerChain::add) to use it in a sampling
/// pipeline; the chain takes ownership.
///
/// See the module-level docs for the transformer / picker distinction and
/// the recommended chain shape.
#[repr(transparent)]
pub struct Sampler(*mut llama_sys::llama_sampler);

impl Sampler {
    // todo these really are not rusty
    /// Adaptive-p sampler — picks tokens whose probability tracks a moving
    /// target via an exponential moving average.
    ///
    /// Wraps `llama_sampler_init_adaptive_p`. This is a **picker**: it must
    /// be the last sampler in a chain (analogous to `mirostat` / `dist` /
    /// `greedy`).
    ///
    /// Upstream recommends running it with at most a single transformer in
    /// front (e.g. [`Sampler::min_p`]) — heavier truncation distorts the
    /// EMA it relies on.
    ///
    /// # Parameters
    /// - `target` — desired token probability in `[0.0, 1.0]`; a negative
    ///   value disables adaptation.
    /// - `decay` — EMA decay rate in `[0.0, 0.99]`; the effective history
    ///   length is roughly `1 / (1 - decay)` tokens.
    /// - `seed` — RNG seed (pass `llama_sys::LLAMA_DEFAULT_SEED` for a
    ///   random seed).
    pub fn adaptive_p(target: f32, decay: f32, seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_adaptive_p(target, decay, seed) })
    }

    /// Fill-in-the-middle infill sampler.
    ///
    /// Wraps `llama_sampler_init_infill`. Designed to sit **after** top-k +
    /// top-p in an infill / FIM (fill-in-the-middle) chain. It collapses
    /// candidates that share a prefix, boosts end-of-generation tokens
    /// whose summed probability dominates the alternatives, and discards
    /// non-EOG candidates with negligible probability.
    ///
    /// # Safety
    /// `vocab` must be a valid `*const llama_vocab` for the lifetime of the
    /// returned `Sampler` and any chain it joins. The recommended source is
    /// `Model::vocab_ptr()` — kept alive by holding the [`Model`] handle
    /// for as long as the sampler exists.
    ///
    /// [`Model`]: crate::Model
    pub fn infill(vocab: *const llama_vocab) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_infill(vocab) })
    }

    /// Per-token additive logit-bias transformer.
    ///
    /// Wraps `llama_sampler_init_logit_bias`. Adds a caller-supplied bias
    /// to the logits of specific token ids — commonly used to forbid a
    /// token (very negative bias) or favour one (positive bias).
    ///
    /// # Safety
    /// `logit_bias` must point to `n_logit_bias` contiguous
    /// `llama_logit_bias` entries and stay live for the lifetime of the
    /// returned `Sampler`.
    pub fn logit_bias(
        n_vocab: i32,
        n_logit_bias: i32,
        logit_bias: *const llama_logit_bias,
    ) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias) })
    }

    /// Repetition / frequency / presence penalty transformer.
    ///
    /// Wraps `llama_sampler_init_penalties`. Penalises tokens that occurred
    /// in the recent generation history. Upstream warns this can be slow
    /// against the full vocab — pair with [`Sampler::top_k`] or
    /// [`Sampler::top_p`] first.
    ///
    /// # Parameters
    /// - `penalty_last_n` — number of recent tokens to consider
    ///   (`0` disables; `-1` means full context).
    /// - `penalty_repeat` — multiplicative repeat penalty (`1.0` disables).
    /// - `penalty_freq` — additive frequency penalty (`0.0` disables).
    /// - `penalty_present` — additive presence penalty (`0.0` disables).
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

    /// XTC ("exclude top choices") transformer.
    ///
    /// Wraps `llama_sampler_init_xtc`. With probability `p`, removes the
    /// most likely candidates above the threshold `t` to encourage less
    /// predictable continuations. See
    /// <https://github.com/oobabooga/text-generation-webui/pull/6335>.
    ///
    /// # Parameters
    /// - `p` — probability of activating the exclusion step.
    /// - `t` — threshold above which top candidates may be excluded.
    /// - `min_keep` — minimum number of candidates that must survive.
    /// - `seed` — RNG seed (pass `llama_sys::LLAMA_DEFAULT_SEED` for a
    ///   random seed).
    pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_xtc(p, t, min_keep, seed) })
    }

    /// Multinomial-distribution picker (seeded RNG draw).
    ///
    /// Wraps `llama_sampler_init_dist`. This is the canonical **picker**:
    /// it samples one token from whatever distribution the preceding
    /// transformers produced. Place it last in the chain.
    ///
    /// Pass `llama_sys::LLAMA_DEFAULT_SEED` to use a random seed.
    pub fn dist(seed: u32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_dist(seed) })
    }

    /// DRY ("don't repeat yourself") repetition-suppression transformer.
    ///
    /// Wraps `llama_sampler_init_dry`. Penalises tokens that would extend
    /// a recently-seen n-gram, with configurable allowed-length and decay
    /// behaviour. See the upstream PR for design notes.
    ///
    /// # Parameters
    /// - `vocab` — vocabulary pointer (see safety note below).
    /// - `n_ctx_train` — the model's training context length.
    /// - `dry_multiplier` — penalty multiplier.
    /// - `dry_base` — exponential penalty base.
    /// - `dry_allowed_length` — n-gram length below which no penalty is
    ///   applied.
    /// - `dry_penalty_last_n` — number of recent tokens scanned for repeats.
    /// - `seq_breakers` / `num_breakers` — pointer to a C-string array of
    ///   tokens that reset the repetition window, and its length.
    ///
    /// # Safety
    /// `vocab` and `seq_breakers` (when `num_breakers > 0`) must point at
    /// valid C data that outlives the returned sampler and any chain it
    /// joins.
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

    /// GBNF grammar-constrained transformer.
    ///
    /// Wraps `llama_sampler_init_grammar`. Restricts the distribution to
    /// token sequences that match a [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
    /// grammar — useful for forcing JSON, code, or any other structured
    /// output.
    ///
    /// # Parameters
    /// - `vocab` — vocabulary pointer (see safety note below).
    /// - `grammar_str` — GBNF source as a C string. An empty string yields
    ///   an empty grammar; a malformed grammar yields a null
    ///   `llama_sampler` (which will fault when used).
    /// - `grammar_root` — name of the start rule, as a C string.
    ///
    /// # Safety
    /// `vocab`, `grammar_str`, and `grammar_root` must point at valid C
    /// data that outlives the returned sampler and any chain it joins.
    pub fn grammar(
        vocab: *const llama_vocab,
        grammar_str: *const ::std::os::raw::c_char,
        grammar_root: *const ::std::os::raw::c_char,
    ) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_grammar(vocab, grammar_str, grammar_root) })
    }

    /// Greedy (argmax) picker.
    ///
    /// Wraps `llama_sampler_init_greedy`. Always selects the token with
    /// the highest logit. Deterministic and parameter-free — useful as the
    /// final stage of any chain when reproducible output is required.
    pub fn greedy() -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_greedy() })
    }

    /// Min-P sampling transformer.
    ///
    /// Wraps `llama_sampler_init_min_p`. Keeps candidates whose probability
    /// is at least `p` times the most probable candidate's probability.
    /// See <https://github.com/ggml-org/llama.cpp/pull/3841>.
    ///
    /// # Parameters
    /// - `p` — minimum-probability ratio.
    /// - `min_keep` — minimum number of candidates that must survive.
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_min_p(p, min_keep) })
    }

    /// Mirostat 1.0 picker.
    ///
    /// Wraps `llama_sampler_init_mirostat`. Picker that targets a fixed
    /// cross-entropy ("surprise") level, updating an internal `mu` after
    /// each draw. See <https://arxiv.org/abs/2007.14966>.
    ///
    /// # Parameters
    /// - `n_vocab` — vocabulary size (typically `model.n_tokens()`).
    /// - `seed` — RNG seed (pass `llama_sys::LLAMA_DEFAULT_SEED` for a
    ///   random seed).
    /// - `tau` — target surprise; higher = more diverse output.
    /// - `eta` — learning rate for `mu`.
    /// - `m` — number of top tokens used to estimate `s_hat` (upstream
    ///   suggests `100`).
    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) })
    }

    /// Mirostat 2.0 picker.
    ///
    /// Wraps `llama_sampler_init_mirostat_v2`. Same target-surprise idea
    /// as [`Sampler::mirostat`] but without the `m` / `n_vocab` parameters
    /// — uses every candidate above the current threshold instead of a
    /// fixed top-`m` slice.
    ///
    /// # Parameters
    /// - `seed` — RNG seed (pass `llama_sys::LLAMA_DEFAULT_SEED` for a
    ///   random seed).
    /// - `tau` — target surprise.
    /// - `eta` — learning rate for `mu`.
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_mirostat_v2(seed, tau, eta) })
    }

    /// Temperature transformer.
    ///
    /// Wraps `llama_sampler_init_temp`. Divides every logit by `temp`
    /// before the next sampler sees them: lower values sharpen the
    /// distribution, higher values flatten it. When `temp <= 0.0` the
    /// argmax logit is kept and every other logit is set to `-inf` —
    /// effectively a greedy transform.
    pub fn temp(temp: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_temp(temp) })
    }

    /// Locally-typical sampling transformer.
    ///
    /// Wraps `llama_sampler_init_typical`. Keeps candidates whose
    /// information content is closest to the distribution's entropy. See
    /// <https://arxiv.org/abs/2202.00666>.
    ///
    /// # Parameters
    /// - `p` — typical-mass fraction to keep.
    /// - `min_keep` — minimum number of candidates that must survive.
    pub fn typical(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_typical(p, min_keep) })
    }

    /// Lazy GBNF grammar transformer — deprecated.
    ///
    /// Wraps the deprecated `llama_sampler_init_grammar_lazy`. The grammar
    /// is only enforced once a trigger word or token appears in the
    /// output. Prefer [`Sampler::grammar_lazy_patterns`] for new code; the
    /// upstream symbol is marked deprecated.
    ///
    /// # Safety
    /// All pointer parameters must reference valid C data that outlives
    /// the returned sampler and any chain it joins.
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

    /// Dynamic-temperature transformer.
    ///
    /// Wraps `llama_sampler_init_temp_ext`. Modulates the temperature
    /// based on the entropy of the current distribution. See
    /// <https://arxiv.org/abs/2309.02772>.
    ///
    /// # Parameters
    /// - `temp` — base temperature.
    /// - `delta` — half-width of the dynamic range around `temp`.
    /// - `exponent` — shape exponent applied to the entropy-derived weight.
    pub fn temp_ext(temp: f32, delta: f32, exponent: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_temp_ext(temp, delta, exponent) })
    }

    /// Top-K sampling transformer.
    ///
    /// Wraps `llama_sampler_init_top_k`. Keeps the `k` highest-logit
    /// candidates and discards the rest. `k <= 0` makes the call a no-op.
    /// See "The Curious Case of Neural Text Degeneration"
    /// (<https://arxiv.org/abs/1904.09751>).
    pub fn top_k(k: i32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_k(k) })
    }

    /// Top-nσ sampling transformer.
    ///
    /// Wraps `llama_sampler_init_top_n_sigma`. Keeps candidates whose
    /// logit lies within `n` standard deviations of the maximum logit.
    /// See <https://arxiv.org/pdf/2411.07641>.
    pub fn top_n_sigma(n: f32) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_n_sigma(n) })
    }

    /// Top-P (nucleus) sampling transformer.
    ///
    /// Wraps `llama_sampler_init_top_p`. Keeps the smallest candidate set
    /// whose cumulative probability is at least `p`. See "The Curious
    /// Case of Neural Text Degeneration"
    /// (<https://arxiv.org/abs/1904.09751>).
    ///
    /// # Parameters
    /// - `p` — cumulative-probability target.
    /// - `min_keep` — minimum number of candidates that must survive.
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        Self(unsafe { llama_sys::llama_sampler_init_top_p(p, min_keep) })
    }

    /// Lazy GBNF grammar transformer — pattern-triggered.
    ///
    /// Wraps `llama_sampler_init_grammar_lazy_patterns`. The grammar is
    /// only enforced once one of `trigger_patterns` matches the output so
    /// far (matched from the start; the grammar then sees from the first
    /// capture group), or one of `trigger_tokens` is emitted. Introduced
    /// in <https://github.com/ggml-org/llama.cpp/pull/9639>.
    ///
    /// # Safety
    /// All pointer parameters must reference valid C data that outlives
    /// the returned sampler and any chain it joins.
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
    /// Frees the underlying `llama_sampler` via `llama_sampler_free`.
    ///
    /// If the sampler has been moved into a
    /// [`SamplerChain`](crate::SamplerChain) via
    /// [`SamplerChain::add`](crate::SamplerChain::add) this destructor does
    /// not run — `add` calls [`std::mem::forget`] on the [`Sampler`] so
    /// the chain becomes the sole owner.
    fn drop(&mut self) {
        unsafe {
            llama_sys::llama_sampler_free(self.0);
        }
    }
}

/// Anything that can hand out a raw `*mut llama_sampler`.
///
/// Implemented by both [`Sampler`] and [`SamplerChain`](crate::SamplerChain)
/// so the public sampling APIs (e.g.
/// [`Sequence::sample`](crate::Sequence::sample),
/// [`Context::sample`](crate::Context::sample)) can accept either kind of
/// handle through a single bound.
pub trait LlamaSampler {
    /// Return the underlying `*mut llama_sampler`.
    ///
    /// The pointer is valid for as long as `self` is borrowed and remains
    /// owned by `self`; callers must not free it.
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler;
}

impl LlamaSampler for Sampler {
    fn as_ptr(&self) -> *mut llama_sys::llama_sampler {
        self.0
    }
}

impl Clone for Sampler {
    /// Clone via `llama_sampler_clone`.
    ///
    /// Produces an independent owning handle. Algorithm state (RNG cursor,
    /// mirostat `mu`, adaptive-p EMA, …) is copied at the moment of
    /// cloning; the two samplers diverge from there.
    fn clone(&self) -> Self {
        Self(unsafe { llama_sys::llama_sampler_clone(self.0) })
    }
}

// Shorthand for: const auto * logits = llama_get_logits_ith(ctx, idx); llama_token_data_array cur_p = { ... init from logits ... }; llama_sampler_apply(smpl, &cur_p); auto token = cur_p.datacur_p.selected.id; llama_sampler_accept(smpl, token); return token; Returns the sampled token
