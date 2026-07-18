//! Regression tests for `Dist::sample_mut`.
//!
//! `Dist` is documented as multinomial (weighted-random) selection, but it only
//! overrode `Sampler::sample`, not `Sampler::sample_mut`. The in-place entry
//! point therefore fell back to the trait default (`apply_mut` + `argmax`),
//! which is deterministic (seed-ignoring) and additionally panics on NaN/empty
//! logits. These tests pin the fixed behaviour: `sample_mut` must be stochastic,
//! agree with `sample` for the same seed, and never panic on adversarial input.

use rusty_llama::{Dist, Sampler};

#[test]
fn dist_sample_mut_is_stochastic_on_uniform_logits() {
    // The trait-default argmax on uniform logits deterministically returns the
    // last index (7) for every seed; a correct multinomial sampler must visit
    // other indices too.
    let mut dist = Dist::new(1);
    let mut seen_non_argmax = false;
    for _ in 0..200 {
        let mut logits = [0.0f32; 8];
        let tok = dist.sample_mut(&mut logits);
        assert!((0..8).contains(&tok), "token in range");
        if tok != 7 {
            seen_non_argmax = true;
        }
    }
    assert!(
        seen_non_argmax,
        "sample_mut ignored the distribution and always returned the argmax"
    );
}

#[test]
fn dist_sample_mut_matches_sample_with_same_seed() {
    let logits = [1.0f32, 2.0, 0.5, 3.0, 0.1];
    for seed in [0u64, 1, 42, 99_999] {
        let via_sample = Dist::new(seed).sample(&logits);
        let via_sample_mut = Dist::new(seed).sample_mut(&mut logits.to_vec());
        assert_eq!(
            via_sample, via_sample_mut,
            "sample and sample_mut diverged for seed {seed}"
        );
    }
}

#[test]
fn dist_sample_mut_leaves_logits_unchanged() {
    // Dist::apply_mut is a no-op, so the in-place entry point must not mutate.
    let original = [1.0f32, 2.0, 0.5, 3.0];
    let mut logits = original;
    let _ = Dist::new(7).sample_mut(&mut logits);
    assert_eq!(logits, original);
}

#[test]
fn dist_sample_mut_does_not_panic_on_nan() {
    let mut logits = [1.0f32, f32::NAN, 3.0];
    let tok = Dist::new(3).sample_mut(&mut logits);
    assert!((0..3).contains(&tok));
}

#[test]
fn dist_sample_mut_does_not_panic_on_empty() {
    let mut logits: [f32; 0] = [];
    let tok = Dist::new(3).sample_mut(&mut logits);
    assert_eq!(tok, 0);
}
