use crate::samplers::Sampler;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// XTC (eXclude Top Choices): with probability `probability`, drop the tokens
/// whose softmax probability is `>= threshold` (keeping the rest). This is a
/// diversity-enhancing sampler; it only acts when at least two tokens clear the
/// threshold and at least `min_keep` would survive.
///
/// As in llama.cpp, the sampler is inert when `probability <= 0` or
/// `threshold > 0.5`.
pub struct Xtc {
    pub probability: f32,
    pub threshold: f32,
    pub min_keep: usize,
    rng: StdRng,
}

impl Xtc {
    pub fn new(probability: f32, threshold: f32, min_keep: usize, seed: u64) -> Self {
        Self {
            probability,
            threshold,
            min_keep,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for Xtc {
    fn apply_mut(&mut self, logits: &mut [f32]) {
        if self.probability <= 0.0 || self.threshold > 0.5 || logits.len() < 2 {
            return;
        }

        if self.rng.random::<f32>() > self.probability {
            return;
        }

        let n = logits.len();

        // softmax probabilities
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().copied().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        // indices sorted by descending logit
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        // pos_last = last rank (consecutive from the top) whose prob >= threshold
        let mut pos_last: usize = 0;
        for (rank, &i) in idx.iter().enumerate() {
            if probs[i] >= self.threshold {
                pos_last = rank;
            } else {
                break;
            }
        }

        // mask the top `pos_last` tokens, provided at least one is removed and
        // enough survive.
        if pos_last > 0 && n - pos_last >= self.min_keep {
            for &i in &idx[..pos_last] {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}
