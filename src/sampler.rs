//! Native Rust samplers for turning a `Vec<f32>` of logits into a next-token
//! decision.
//!
//! Everything in this module operates on plain `&[f32]` / `&mut [f32]` — there
//! is no llama.cpp sampler behind these types, so composition is just calling
//! each sampler's [`Sampler::apply`] / [`Sampler::apply_mut`] in whatever order
//! you want and then a terminal [`Sampler::sample`].
//!
//! The shipped example ([`examples/simple_chat.rs`]) wires the three concrete
//! samplers manually:
//!
//! ```ignore
//! let minp = MinP::new(0.05, 1);
//! let temp = Temperature::new(0.8);
//! let dist = Dist::new(seed);
//!
//! let l = minp.apply(logits);   // mask low-probability tokens to -inf
//! let l = temp.apply(&l);       // rescale surviving logits
//! let token = dist.sample(&l);  // stochastic softmax draw
//! ```
//!
//! When you drop a stage in that pipeline, the meaning of the remaining stages
//! is unchanged — `apply` / `apply_mut` operate on logits the same way
//! regardless of what preceded them.
//!
//! [`examples/simple_chat.rs`]: https://github.com/oofdere/rusty_llama/blob/trunk/examples/simple_chat.rs

use crate::Token;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

/// A logit transform (optional) plus a token picker.
///
/// Only [`Sampler::apply_mut`] is required. The rest — [`apply`](Sampler::apply),
/// [`sample`](Sampler::sample), [`sample_mut`](Sampler::sample_mut),
/// [`name`](Sampler::name) — have defaults built on top of it.
///
/// # Contract
///
/// - `apply_mut(logits)` may reorder, rescale, or mask entries of `logits`
///   (typically by writing [`f32::NEG_INFINITY`] into slots you want to
///   exclude), but the length of the slice must not change: the index of a
///   logit is the token id, and the terminal sampler picks a slot by index.
/// - `sample(logits)` returns a valid index into `logits` — always
///   `0 <= id < logits.len() as Token` when `logits` is non-empty.
///
/// # Composing samplers
///
/// There is no built-in chain type: to compose, call one sampler's `apply` /
/// `apply_mut` on the output of the previous one. Because the transforms
/// operate on a plain slice, the composed result is just as much a valid input
/// for a terminal sampler as raw model logits are.
pub trait Sampler {
    /// Short name for this sampler, used for logs and diagnostics.
    ///
    /// The default takes the trailing path segment of
    /// [`std::any::type_name`] — e.g. `"Temperature"` for
    /// [`Temperature`]. Override when the type name is not the user-facing
    /// name of the algorithm.
    fn name(&self) -> &'static str {
        let full_name = std::any::type_name::<Self>();
        full_name.split("::").last().unwrap_or(full_name)
    }

    /// Return the logits after this sampler's transform, without mutating the
    /// caller's slice.
    ///
    /// The default clones `logits` into a `Vec` and calls
    /// [`apply_mut`](Sampler::apply_mut) on the copy. Override only if a
    /// non-in-place implementation would be materially cheaper.
    fn apply(&self, logits: &[f32]) -> Vec<f32> {
        let mut logits = logits.to_vec();
        self.apply_mut(&mut logits);
        logits
    }

    /// Transform `logits` in place.
    ///
    /// Implementations may reorder, rescale, or mask entries (typically with
    /// [`f32::NEG_INFINITY`]) but must not change the length: downstream
    /// samplers treat the slot index as the token id.
    fn apply_mut(&self, logits: &mut [f32]);

    /// Apply the transform to a scratch copy of `logits` and return the picked
    /// token id.
    ///
    /// The default first calls [`apply_mut`](Sampler::apply_mut) on a copy and
    /// then returns the index of the maximum logit (greedy argmax). Terminal
    /// samplers that do stochastic draws (e.g. [`Dist`]) override this to skip
    /// the argmax and consume the transformed logits directly.
    ///
    /// # Panics
    ///
    /// The default implementation panics when `logits` is empty or contains a
    /// [`f32::NAN`] entry, because the underlying [`Iterator::max_by`] compares
    /// with [`f32::partial_cmp`] and then unwraps.
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

    /// Like [`sample`](Sampler::sample), but the transform is applied to the
    /// caller's slice rather than a scratch copy.
    ///
    /// After this call the slice holds the transformed logits, so it can be
    /// reused as input to another sampler.
    ///
    /// # Panics
    ///
    /// Same conditions as [`sample`](Sampler::sample): empty slice or a NaN
    /// entry.
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

/// Softmax temperature: multiplies every logit by `1.0 / temp`.
///
/// Higher `temp` flattens the distribution (more randomness), lower `temp`
/// sharpens it toward the argmax. `temp == 0.0` short-circuits to a no-op, so
/// pairing `Temperature::new(0.0)` with a terminal sampler that argmaxes the
/// transformed logits (e.g. the default [`Sampler::sample`]) reduces the whole
/// stage to greedy decoding.
///
/// The transform is a pure per-element multiply, so [`f32::NEG_INFINITY`]
/// masks placed by an earlier sampler (see [`MinP`]) survive unchanged.
pub struct Temperature{
    /// Softmax temperature; `0.0` is a special-cased no-op.
    pub temp: f32,
}

impl Temperature {
    /// Build a [`Temperature`] with the given temperature.
    ///
    /// See the type-level docs for the meaning of `temp` and the `0.0`
    /// special case.
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

/// Min-P mask: sets every logit below `max_logit + ln(p)` to
/// [`f32::NEG_INFINITY`].
///
/// This is the log-space form of "drop every token whose probability is less
/// than `p` times the top token's probability". Because the mask is
/// [`f32::NEG_INFINITY`], the softmax weight of a masked slot is exactly zero,
/// so downstream stochastic samplers such as [`Dist`] cannot draw a masked
/// index.
///
/// `min_keep` guarantees that at least that many logits survive the mask:
///
/// - If `logits.len() <= min_keep`, the sampler is a no-op — every entry
///   would need to survive, so masking would violate the guarantee.
/// - Otherwise, the threshold is clamped to the `min_keep`-th largest logit,
///   which keeps at least the top `min_keep` slots even when `p` would have
///   masked more.
///
/// `min_keep == 0` disables the clamp; the threshold is `max_logit + ln(p)`
/// alone.
pub struct MinP {
    /// Probability floor, expressed as a fraction of the top token's
    /// probability. Applied in log space, so the actual per-logit threshold
    /// is `max_logit + p.ln()`.
    pub p: f32,
    /// Lower bound on the number of surviving (non-`-inf`) logits after the
    /// mask, honoured even when `p` would have masked more.
    pub min_keep: usize,
}

impl MinP {
    /// Build a [`MinP`] with the given probability floor and minimum-keep
    /// guarantee. See the type-level docs for how the two interact.
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

/// Terminal stochastic sampler: draws a token id from the softmax of the
/// input logits, using a seeded [`StdRng`].
///
/// `Dist` treats logits as read-only — its [`apply_mut`](Sampler::apply_mut)
/// is a no-op, so putting it in the middle of a composition is meaningless.
/// Use it as the last stage after any transforms.
///
/// # Determinism
///
/// The seed is consumed once at construction; the RNG is stored in a
/// [`RefCell`], and each call to [`sample`](Sampler::sample) advances it.
/// Two `Dist`s built with the same seed and given the same sequence of logit
/// slices will draw the same sequence of tokens.
///
/// # Masked logits
///
/// A logit of [`f32::NEG_INFINITY`] contributes exactly `0.0` to the softmax
/// weight, so it can never be drawn. That's how [`MinP`] guarantees its mask
/// is honoured downstream.
pub struct Dist {
    rng: RefCell<StdRng>,
}

impl Dist {
    /// Build a [`Dist`] whose draws are reproducible for a given `seed`. See
    /// the type-level docs for the exact determinism guarantee.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Sampler for Dist {
    /// No-op: `Dist` is a terminal sampler and never rewrites the logits its
    /// [`sample`](Sampler::sample) consumes.
    fn apply_mut(&self, _logits: &mut [f32]) {}

    /// Softmax draw. Subtracts the maximum logit before exponentiating for
    /// numerical stability, then walks the cumulative weight until it crosses
    /// a `[0, sum)` random draw. Empty input yields `-1` (via
    /// `exps.len().saturating_sub(1)`).
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
