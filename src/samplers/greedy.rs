use crate::samplers::Sampler;

/// Greedy / argmax selection. A pure token selector: `apply_mut` is a no-op.
pub struct Greedy;

impl Greedy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Greedy {
    fn default() -> Self {
        Self
    }
}

impl Sampler for Greedy {
    fn apply_mut(&mut self, _logits: &mut [f32]) {}
}
