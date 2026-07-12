//! Regression tests for the default `Sampler::sample` / `sample_mut`
//! implementations. These previously used
//! `max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()`, which panics when the
//! logits contain a NaN (`partial_cmp` returns `None`) or when the slice is
//! empty (`max_by` returns `None`). Both are reachable through the public
//! `Sequence::sample` API, since model logits can legitimately be NaN and a
//! transforming sampler can also introduce them.

use rusty_llama::{Sampler, Token};

/// No-op sampler that exercises the trait's default `sample`/`sample_mut`
/// without needing a llama.cpp backend or model.
struct Identity;

impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_picks_max_ignoring_nan() {
    let s = Identity;
    let logits = vec![0.1f32, f32::NAN, 5.0, 2.0];
    assert_eq!(s.sample(&logits), 2 as Token);
}

#[test]
fn sample_does_not_panic_on_leading_nan() {
    // `total_cmp` ranks NaN above +inf, so a naive argmax would select the NaN.
    let s = Identity;
    let logits = vec![f32::NAN, 0.0f32];
    assert_eq!(s.sample(&logits), 1 as Token);
}

#[test]
fn sample_does_not_panic_on_all_nan() {
    let s = Identity;
    let logits = vec![f32::NAN, f32::NAN];
    assert_eq!(s.sample(&logits), 0 as Token);
}

#[test]
fn sample_does_not_panic_on_empty() {
    let s = Identity;
    let logits: Vec<f32> = Vec::new();
    assert_eq!(s.sample(&logits), 0 as Token);
}

#[test]
fn sample_mut_picks_max_ignoring_nan() {
    let s = Identity;
    let mut logits = vec![f32::NAN, 3.0f32, 1.0];
    assert_eq!(s.sample_mut(&mut logits), 1 as Token);
}

#[test]
fn sample_mut_does_not_panic_on_empty() {
    let s = Identity;
    let mut logits: Vec<f32> = Vec::new();
    assert_eq!(s.sample_mut(&mut logits), 0 as Token);
}
