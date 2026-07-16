use crate::samplers::Sampler;

/// Min-P sampling: mask every logit more than `p` (in probability space)
/// below the maximum. Keeps at least `min_keep` candidates.
pub struct MinP {
    pub p: f32,
    pub min_keep: usize,
}

impl MinP {
    pub fn new(p: f32, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl Sampler for MinP {
    fn apply_mut(&mut self, logits: &mut [f32]) {
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
