use crate::samplers::{argmax, Sampler};
use crate::Token;

/// A pipeline of samplers applied in order.
///
/// Logit-transforming samplers (e.g. [`crate::TopK`], [`crate::Temperature`])
/// are run first, and the final sampler is expected to be a token selector such
/// as [`crate::Greedy`] or [`crate::Dist`].
///
/// ```
/// use rusty_llama::{Chain, Sampler, TopK, Temperature, Dist};
///
/// let mut chain = Chain::new()
///     .with(TopK::new(40))
///     .with(Temperature::new(0.8))
///     .with(Dist::new(42));
/// let token = chain.sample(&[1.0, 2.0, 0.5, 3.0]);
/// ```
pub struct Chain {
    samplers: Vec<Box<dyn Sampler>>,
}

impl Chain {
    pub fn new() -> Self {
        Self {
            samplers: Vec::new(),
        }
    }

    /// Append a sampler, returning `&mut self` for chained calls.
    pub fn push<S: Sampler + 'static>(&mut self, sampler: S) -> &mut Self {
        self.samplers.push(Box::new(sampler));
        self
    }

    /// Builder-style append, consuming and returning `self`.
    pub fn with<S: Sampler + 'static>(mut self, sampler: S) -> Self {
        self.samplers.push(Box::new(sampler));
        self
    }

    pub fn len(&self) -> usize {
        self.samplers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samplers.is_empty()
    }
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for Chain {
    fn apply_mut(&mut self, logits: &mut [f32]) {
        for sampler in &mut self.samplers {
            sampler.apply_mut(logits);
        }
    }

    fn sample(&mut self, logits: &[f32]) -> Token {
        let mut logits = logits.to_vec();
        for sampler in &mut self.samplers {
            sampler.apply_mut(&mut logits);
        }

        match self.samplers.last_mut() {
            // The last sampler is expected to be a selector (Greedy/Dist);
            // calling its own sample() applies no extra transform for those.
            Some(last) => last.sample(&logits),
            None => argmax(&logits),
        }
    }
}
