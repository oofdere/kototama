use crate::samplers::Sampler;
use crate::Token;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Multinomial (weighted-random) token selection from the softmax distribution.
/// `apply_mut` is a no-op; the selection happens in [`Dist::sample`].
///
/// Logits that are `-inf` (masked by an upstream sampler such as
/// [`crate::TopK`]) or `NaN` have zero weight and are never selected. `+inf`
/// logits take the whole mass. When no logit is selectable (empty slice, or all
/// logits masked/`NaN`), token `0` is returned.
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

    fn sample(&mut self, logits: &[f32]) -> Token {
        // `+inf` dominates the distribution: pick the first such token instead
        // of feeding `inf - inf = NaN` into the softmax below.
        if let Some(id) = logits.iter().position(|l| *l == f32::INFINITY) {
            return id as Token;
        }

        let max = logits
            .iter()
            .copied()
            .filter(|l| l.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return 0;
        }

        // Non-finite logits (masked `-inf`, `NaN`) get zero weight.
        let weights: Vec<f32> = logits
            .iter()
            .map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 })
            .collect();
        let sum: f32 = weights.iter().copied().sum();
        if !(sum > 0.0) || !sum.is_finite() {
            return 0;
        }

        let r = self.rng.random::<f32>() * sum;
        let mut acc = 0.0;
        let mut last = 0;
        for (id, &w) in weights.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            acc += w;
            last = id;
            if acc > r {
                return id as Token;
            }
        }

        // Only reachable through float rounding; `last` has non-zero weight.
        last as Token
    }
}
