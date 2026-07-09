//! Regression tests for the `Sampler` trait default `sample`/`sample_mut`
//! implementations. These previously used
//! `max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()`, which panics when the
//! logits contain a `NaN` (`partial_cmp` returns `None`) or when the slice is
//! empty (`max_by` returns `None`). Model logits can contain non-finite values
//! (e.g. after an ill-conditioned forward pass or a divide-by-zero in a custom
//! sampler), so the public `Sequence::sample` path must not panic on them.

use rusty_llama::Sampler;

/// A no-op sampler that leaves logits untouched, so `sample`/`sample_mut` fall
/// through to the trait's default argmax. This needs no llama.cpp backend.
struct Identity;

impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_picks_argmax() {
    let logits = [0.1, 0.9, 0.3, -1.0];
    assert_eq!(Identity.sample(&logits), 1);
}

#[test]
fn sample_does_not_panic_on_nan() {
    let logits = [f32::NAN, 0.5, f32::NAN, 0.2];
    // Must ignore the NaNs and pick the greatest real logit (index 1).
    assert_eq!(Identity.sample(&logits), 1);
}

#[test]
fn sample_handles_all_nan() {
    let logits = [f32::NAN, f32::NAN];
    // No valid maximum exists; fall back to token 0 instead of panicking.
    assert_eq!(Identity.sample(&logits), 0);
}

#[test]
fn sample_handles_empty() {
    let logits: [f32; 0] = [];
    assert_eq!(Identity.sample(&logits), 0);
}

#[test]
fn sample_mut_matches_sample() {
    let mut logits = [f32::NAN, -2.0, 3.0, f32::NAN, 1.0];
    let expected = Identity.sample(&logits);
    assert_eq!(Identity.sample_mut(&mut logits), expected);
    assert_eq!(expected, 2);
}

#[test]
fn sample_handles_infinities() {
    let logits = [f32::NEG_INFINITY, 1.0, f32::INFINITY, f32::NAN];
    assert_eq!(Identity.sample(&logits), 2);
}
