use crate::samplers::Sampler;

/// Locally typical sampling: keep tokens whose negative-log-probability is
/// closest to the distribution entropy, accumulating until the cumulative
/// probability exceeds `p` (keeping at least `min_keep`). A `p >= 1.0` is a
/// no-op.
///
/// Reference: Meister et al., "Typical Decoding for Natural Language Generation".
pub struct Typical {
    pub p: f32,
    pub min_keep: usize,
}

impl Typical {
    pub fn new(p: f32, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl Sampler for Typical {
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

        // entropy H = -sum(p * ln(p))
        let entropy = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f32>();

        // shifted score = |(-ln p) - H|; sort ascending so the most "typical"
        // tokens come first.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            let sa = (-(probs[a].ln()) - entropy).abs();
            let sb = (-(probs[b].ln()) - entropy).abs();
            sa.total_cmp(&sb)
        });

        let mut cum = 0.0f32;
        let mut keep = vec![false; n];
        for (rank, &i) in idx.iter().enumerate() {
            keep[i] = true;
            cum += probs[i];
            if cum > self.p && (self.min_keep == 0 || rank + 1 >= self.min_keep) {
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
