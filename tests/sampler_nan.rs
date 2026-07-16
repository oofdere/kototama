//! Regression tests for the default `Sampler::sample`/`sample_mut` argmax.
//!
//! The trait defaults previously used `partial_cmp(...).unwrap()` inside
//! `max_by(...).unwrap()`, which panics when the logits contain a `NaN`
//! (`partial_cmp` returns `None`) or are empty (`max_by` returns `None`).
//! Both inputs are reachable from safe code via `Sequence::sample`, so the
//! defaults must degrade gracefully instead of panicking.

use rusty_llama::{Sampler, Token};

/// No-op sampler that leaves logits untouched, exercising the trait defaults.
struct Identity;

impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_picks_argmax_on_normal_logits() {
    let logits = [0.1_f32, 0.9, 0.3, 0.2];
    assert_eq!(Identity.sample(&logits), 1 as Token);
}

#[test]
fn sample_mut_picks_argmax_on_normal_logits() {
    let mut logits = [0.1_f32, 0.3, 0.2, 0.9];
    assert_eq!(Identity.sample_mut(&mut logits), 3 as Token);
}

#[test]
fn sample_does_not_panic_on_nan() {
    let logits = [1.0_f32, f32::NAN, 2.0, f32::NAN];
    // Must not panic; NaN entries are ignored so the real max (index 2) wins.
    assert_eq!(Identity.sample(&logits), 2 as Token);
}

#[test]
fn sample_mut_does_not_panic_on_nan() {
    let mut logits = [f32::NAN, 5.0, f32::NAN, 1.0];
    assert_eq!(Identity.sample_mut(&mut logits), 1 as Token);
}

#[test]
fn sample_falls_back_to_zero_on_empty() {
    let logits: [f32; 0] = [];
    assert_eq!(Identity.sample(&logits), 0 as Token);
}

#[test]
fn sample_falls_back_to_zero_when_all_nan() {
    let logits = [f32::NAN, f32::NAN, f32::NAN];
    assert_eq!(Identity.sample(&logits), 0 as Token);
}
