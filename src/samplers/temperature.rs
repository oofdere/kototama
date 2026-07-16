use crate::samplers::Sampler;

/// Scale logits by `1 / temp`. A temperature of `0.0` is a no-op
/// (use [`crate::Greedy`] for deterministic decoding instead).
pub struct Temperature {
    pub temp: f32,
}

impl Temperature {
    pub fn new(temp: f32) -> Self {
        Self { temp }
    }
}

impl Sampler for Temperature {
    fn apply_mut(&mut self, logits: &mut [f32]) {
        if self.temp == 0.0 {
            return;
        }

        let inv_temp = 1.0 / self.temp;

        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }
}
