//! Regression tests for the default `Sampler::sample` / `sample_mut` argmax.
//!
//! Before the fix these used `partial_cmp(..).unwrap()` on the logits, which
//! panics when a logit is `NaN` (partial_cmp returns `None`) or when the slice
//! is empty (the outer `unwrap()` on `max_by`). Both are reachable through the
//! public `Sequence::sample` path once a sampler produces or forwards `NaN`s.

use rusty_llama::Sampler;

/// A no-op sampler so we can drive the default `sample` / `sample_mut` methods
/// with arbitrary logits without touching the llama.cpp backend.
struct Identity;

impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_does_not_panic_on_nan() {
    let logits = [0.1f32, f32::NAN, 0.5, f32::NAN, 0.2];
    // Highest non-NaN value is 0.5 at index 2.
    assert_eq!(Identity.sample(&logits), 2);
}

#[test]
fn sample_mut_does_not_panic_on_nan() {
    let mut logits = [f32::NAN, 3.0f32, 1.0, f32::NAN];
    assert_eq!(Identity.sample_mut(&mut logits), 1);
}

#[test]
fn sample_all_nan_falls_back_to_zero() {
    let logits = [f32::NAN, f32::NAN, f32::NAN];
    assert_eq!(Identity.sample(&logits), 0);
}

#[test]
fn sample_empty_logits_falls_back_to_zero() {
    let logits: [f32; 0] = [];
    assert_eq!(Identity.sample(&logits), 0);
}

#[test]
fn sample_normal_logits_picks_argmax() {
    let logits = [0.1f32, 0.9, 0.3, 0.4, 0.2];
    assert_eq!(Identity.sample(&logits), 1);
}
