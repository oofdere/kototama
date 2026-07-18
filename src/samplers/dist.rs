use crate::samplers::Sampler;
use crate::Token;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Multinomial (weighted-random) token selection from the softmax distribution.
/// `apply_mut` is a no-op; the selection happens in [`Dist::sample`].
pub struct Dist {
    rng: StdRng,
}

impl Dist {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for Dist {
    fn apply_mut(&mut self, _logits: &mut [f32]) {}

    fn sample_mut(&mut self, logits: &mut [f32]) -> Token {
        self.sample(logits)
    }

    fn sample(&mut self, logits: &[f32]) -> Token {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
        let sum: f32 = exps.iter().copied().sum();
        let r = self.rng.random::<f32>() * sum;
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
