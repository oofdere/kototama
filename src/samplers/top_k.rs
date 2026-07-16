use crate::samplers::Sampler;

/// Top-K sampling: keep only the `k` highest logits, masking the rest to
/// `-inf`. Tokens tied at the threshold are kept, so the survivor set may
/// contain more than `k` entries. A non-positive `k` leaves the logits untouched.
pub struct TopK {
    pub k: i32,
}

impl TopK {
    pub fn new(k: i32) -> Self {
        Self { k }
    }
}

impl Sampler for TopK {
    fn apply_mut(&mut self, logits: &mut [f32]) {
        if self.k <= 0 {
            return;
        }

        let k = (self.k as usize).min(logits.len());
        if k >= logits.len() {
            return;
        }

        // Find the value of the k-th largest logit (1-indexed), then keep
        // everything greater-than-or-equal to it.
        let mut copy: Vec<f32> = logits.to_vec();
        copy.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        let thresh = copy[k - 1];

        for logit in logits.iter_mut() {
            if *logit < thresh {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}
