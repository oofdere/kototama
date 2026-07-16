use crate::samplers::Sampler;

/// Top-N-Sigma sampling: mask every logit more than `n` standard deviations
/// below the maximum (considering only non-masked logits). A non-positive `n`
/// is a no-op.
///
/// Reference: "Simplifying Top-p Sampling with Top-nσ".
pub struct TopNSigma {
    pub n: f32,
}

impl TopNSigma {
    pub fn new(n: f32) -> Self {
        Self { n }
    }
}

impl Sampler for TopNSigma {
    fn apply_mut(&mut self, logits: &mut [f32]) {
        if self.n <= 0.0 || logits.len() <= 1 {
            return;
        }

        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for &l in logits.iter() {
            if l != f32::NEG_INFINITY {
                if l > max {
                    max = l;
                }
                sum += l;
                count += 1;
            }
        }

        if count == 0 {
            return;
        }
        let mean = sum / count as f32;

        let var = logits
            .iter()
            .filter(|&&l| l != f32::NEG_INFINITY)
            .map(|&l| (l - mean) * (l - mean))
            .sum::<f32>()
            / count as f32;
        let std = var.sqrt();

        let thresh = max - self.n * std;
        for logit in logits.iter_mut() {
            if *logit < thresh {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}
