//! Rust-side sampling primitives.
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

use crate::Token;

mod chain;
mod dist;
mod greedy;
mod min_p;
mod temperature;
mod top_k;
mod top_n_sigma;
mod top_p;
mod typical;
mod xtc;

pub use chain::Chain;
pub use dist::Dist;
pub use greedy::Greedy;
pub use min_p::MinP;
pub use temperature::Temperature;
pub use top_k::TopK;
pub use top_n_sigma::TopNSigma;
pub use top_p::TopP;
pub use typical::Typical;
pub use xtc::Xtc;

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
pub(crate) fn argmax(logits: &[f32]) -> Token {
    let (id, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    id as i32
}
