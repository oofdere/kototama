use crate::samplers::Sampler;

/// Top-P (nucleus) sampling: keep the smallest set of highest-probability
/// tokens whose cumulative softmax probability reaches `p`, always keeping at
/// least `min_keep`. A `p >= 1.0` is a no-op.
pub struct TopP {
    pub p: f32,
    pub min_keep: usize,
}

impl TopP {
    pub fn new(p: f32, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl Sampler for TopP {
    fn apply_mut(&mut self, logits: &mut [f32]) {
        if self.p >= 1.0 {
            return;
        }

        let n = logits.len();
        if n == 0 {
            return;
        }

        // softmax probabilities
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().copied().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        // indices sorted by descending logit
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        let mut cum = 0.0f32;
        let mut keep = vec![false; n];
        for (rank, &i) in idx.iter().enumerate() {
            keep[i] = true;
            cum += probs[i];
            if cum >= self.p && rank + 1 >= self.min_keep {
                break;
            }
        }

        for (i, logit) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}
