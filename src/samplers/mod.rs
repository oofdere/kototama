//! Sampling primitives.
//!
//! Each sampler implements [`Sampler`]. Logit-transforming samplers (e.g.
//! [`Temperature`], [`TopK`], [`MinP`]) override [`Sampler::apply_mut`], while
//! token-selecting samplers ([`Greedy`], [`Dist`]) override [`Sampler::sample`].
//! A [`Chain`] composes several samplers into a pipeline.

use crate::Token;

mod temperature;
mod min_p;
mod top_k;
mod top_p;
mod typical;
mod top_n_sigma;
mod xtc;
mod dist;
mod greedy;
mod chain;

pub use temperature::Temperature;
pub use min_p::MinP;
pub use top_k::TopK;
pub use top_p::TopP;
pub use typical::Typical;
pub use top_n_sigma::TopNSigma;
pub use xtc::Xtc;
pub use dist::Dist;
pub use greedy::Greedy;
pub use chain::Chain;

/// A sampler transforms a slice of logits and/or selects a token from them.
///
/// The default method implementations make it possible to write a
/// logit-transforming sampler by implementing only [`Sampler::apply_mut`]:
/// [`Sampler::sample`] will then apply the transform and pick the argmax.
pub trait Sampler {
    /// Short, human-readable name derived from the type name.
    fn name(&self) -> &'static str {
        let full_name = std::any::type_name::<Self>();
        full_name.split("::").last().unwrap_or(full_name)
    }

    /// Apply the sampler to a fresh copy of `logits` and return the result.
    fn apply(&mut self, logits: &[f32]) -> Vec<f32> {
        let mut logits = logits.to_vec();
        self.apply_mut(&mut logits);
        logits
    }

    /// Transform `logits` in place. This is the only method a
    /// logit-transforming sampler needs to implement.
    fn apply_mut(&mut self, logits: &mut [f32]);

    /// Apply the transform to a copy of `logits` and return the argmax token.
    fn sample(&mut self, logits: &[f32]) -> Token {
        let mut logits = logits.to_vec();
        self.apply_mut(&mut logits);
        argmax(&logits)
    }

    /// Transform `logits` in place and return the argmax token.
    fn sample_mut(&mut self, logits: &mut [f32]) -> Token {
        self.apply_mut(logits);
        argmax(logits)
    }
}

/// Pick the index of the largest logit, breaking ties towards the last.
///
/// NaN logits are ignored so a single non-comparable value cannot panic the
/// selection; if every logit is NaN or the slice is empty, token `0` is
/// returned as a safe fallback.
pub(crate) fn argmax(logits: &[f32]) -> Token {
    logits
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.is_nan())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(id, _)| id as i32)
        .unwrap_or(0)
}
