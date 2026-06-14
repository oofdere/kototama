//! Pure-Rust logit transformers and token selectors.
//!
//! Each implementor of [`Sampler`] is one stage of a sampling pipeline. Stages
//! fall into two categories:
//!
//! - **Transformers** rewrite the raw logits — sharpening, smoothing, or
//!   masking out candidates — and pass the result on to the next stage.
//!   [`Temperature`] and [`MinP`] are transformers.
//! - **Selectors** consume the (possibly transformed) logits and emit a
//!   [`Token`]. [`Dist`] is a selector; the default [`Sampler::sample`] /
//!   [`Sampler::sample_mut`] implementations are argmax selectors.
//!
//! The canonical pipeline applies transformers in sequence, then asks the
//! selector for a token:
//!
//! ```ignore
//! let minp = MinP::new(0.05, 1);
//! let temp = Temperature::new(0.8);
//! let dist = Dist::new(seed);
//!
//! let l = minp.apply(logits);
//! let l = temp.apply(&l);
//! let token = dist.sample(&l);
//! ```
//!
//! Unlike the old [`llama_sampler`](llama_sys::llama_sampler)-backed wrappers,
//! these samplers are implemented entirely in Rust and operate directly on a
//! `&[f32]` slice of logits. They allocate only when borrowing through
//! [`Sampler::apply`] (which copies the input); [`Sampler::apply_mut`] is
//! allocation-free.

use crate::Token;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

/// A stage in a sampling pipeline.
///
/// Implementors decide how to rewrite logits ([`apply_mut`](Self::apply_mut))
/// and, optionally, how to pick a [`Token`] from them
/// ([`sample`](Self::sample)). The default selector implementations take the
/// argmax of the transformed logits — equivalent to a greedy decoder — so a
/// pure transformer only needs to provide [`apply_mut`](Self::apply_mut).
pub trait Sampler {
    /// Short type name of this sampler, useful for logging or debug output.
    ///
    /// Returns the last `::`-separated segment of [`std::any::type_name`],
    /// so `Temperature` instead of `rusty_llama::sampler::Temperature`.
    fn name(&self) -> &'static str {
        let full_name = std::any::type_name::<Self>();
        full_name.split("::").last().unwrap_or(full_name)
    }

    /// Return a transformed copy of `logits`.
    ///
    /// Allocates a new `Vec<f32>` and forwards to
    /// [`apply_mut`](Self::apply_mut). Prefer
    /// [`apply_mut`](Self::apply_mut) when the caller already owns a buffer.
    fn apply(&self, logits: &[f32]) -> Vec<f32> {
        let mut logits = logits.to_vec();
        self.apply_mut(&mut logits);
        logits
    }

    /// Transform `logits` in place.
    ///
    /// The only required method on the trait. Transformers rewrite values;
    /// selectors that don't reshape the distribution (such as [`Dist`]) leave
    /// the slice untouched.
    fn apply_mut(&self, logits: &mut [f32]);

    /// Apply this sampler to `logits` and pick a token.
    ///
    /// The default implementation copies `logits`, calls
    /// [`apply_mut`](Self::apply_mut), and returns the index of the largest
    /// value (argmax / greedy selection). Override for stochastic selectors
    /// such as [`Dist`].
    fn sample(&self, logits: &[f32]) -> Token {
        let mut logits = logits.to_vec();
        self.apply_mut(&mut logits);
        let (id, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        id as i32
    }

    /// Apply this sampler to `logits` in place and pick a token.
    ///
    /// Same as [`sample`](Self::sample) but consumes the caller's buffer
    /// instead of allocating a copy.
    fn sample_mut(&self, logits: &mut [f32]) -> Token {
        self.apply_mut(logits);

        let (id, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        id as i32
    }
}

/// Temperature scaling: divides every logit by `temp`.
///
/// Lower temperatures (`0 < temp < 1`) sharpen the post-softmax distribution
/// toward the argmax, making the model more deterministic. Higher temperatures
/// (`temp > 1`) flatten it, increasing the probability of less-likely tokens.
///
/// `temp == 0.0` is treated as a no-op (leaves logits unchanged) to avoid
/// division by zero; chain a selector that already implements greedy
/// argmax — such as the default [`Sampler::sample`] — to achieve "temperature
/// zero" behaviour.
pub struct Temperature{
    /// Temperature value. See the struct docs for behaviour at edge cases.
    pub temp: f32,
}

impl Temperature {
    /// Create a new temperature sampler with the given `temp` value.
    pub fn new(temp: f32) -> Self {
        Self { temp }
    }
}

impl Sampler for Temperature {
    fn apply_mut(&self, logits: &mut [f32]) {
        if self.temp == 0.0 {
            return;
        }

        let inv_temp = 1.0 / self.temp;

        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }
}

/// Min-P sampling: masks out any token whose probability is below `p`
/// times the probability of the most likely token.
///
/// Implemented directly on logits without an explicit softmax: a candidate is
/// kept iff `logit >= logit_max + ln(p)`. Masked candidates have their logit
/// set to `f32::NEG_INFINITY` so downstream transformers / selectors see them
/// as zero-probability.
///
/// `min_keep` is a floor on the number of surviving candidates: if the
/// `p`-based threshold would drop the candidate set below `min_keep`, the
/// threshold is relaxed to the `min_keep`-th largest logit so at least that
/// many tokens remain available.
///
/// See <https://arxiv.org/abs/2407.01082> for the original Min-P proposal.
pub struct MinP {
    /// Minimum probability ratio relative to the top candidate, in `[0, 1]`.
    pub p: f32,
    /// Minimum number of candidates to keep after thresholding.
    pub min_keep: usize,
}

impl MinP {
    /// Create a new Min-P sampler with the given `p` ratio and `min_keep` floor.
    pub fn new(p: f32, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl Sampler for MinP {
    fn apply_mut(&self, logits: &mut [f32]) {
        if logits.len() <= self.min_keep {
            return;
        }

        let Some(&logit_max) = logits.iter().max_by(|a, b| a.total_cmp(b)) else {
            return;
        };
        let mut thresh = logit_max + self.p.ln();

        if self.min_keep > 0 {
            let mut copy: Vec<f32> = logits.to_vec();
            copy.select_nth_unstable_by(self.min_keep - 1, |a, b| b.total_cmp(a));
            thresh = thresh.min(copy[self.min_keep - 1]);
        }

        for logit in logits.iter_mut() {
            if *logit < thresh {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

/// Categorical sampler over the softmax of the input logits.
///
/// `Dist` is a pure selector — its [`apply_mut`](Sampler::apply_mut) is a
/// no-op so it composes cleanly at the end of a chain of transformers
/// (typically [`MinP`] and [`Temperature`]). The transformed logits are
/// converted to probabilities via numerically-stable softmax (subtracting the
/// max before exponentiating) and a token is drawn proportionally.
///
/// The internal RNG is held in a [`RefCell`] so `sample` can take `&self`;
/// `Dist` is therefore **not** [`Sync`].
pub struct Dist {
    rng: RefCell<StdRng>,
}

impl Dist {
    /// Create a new categorical sampler seeded with the given `seed`.
    ///
    /// Reuse the same `Dist` across calls to keep the RNG state advancing.
    /// Constructing a fresh `Dist` with the same seed reproducibly restarts
    /// the stream.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Sampler for Dist {
    fn apply_mut(&self, _logits: &mut [f32]) {}

    fn sample(&self, logits: &[f32]) -> Token {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
        let sum: f32 = exps.iter().copied().sum();
        let r = self.rng.borrow_mut().random::<f32>() * sum;
        let mut acc = 0.0;
        for (id, &e) in exps.iter().enumerate() {
            acc += e;
            if acc >= r {
                return id as i32;
            }
        }
        (exps.len().saturating_sub(1)) as i32
    }
}
