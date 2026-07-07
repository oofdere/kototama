//! Regression tests for the default `Sampler::sample`/`sample_mut` argmax.
//!
//! Logits produced by a model (or by numerically unstable sampler transforms)
//! can contain `NaN`. The default argmax used to compare with
//! `partial_cmp(..).unwrap()`, which panics on `NaN`. These tests ensure it
//! neither panics nor selects a `NaN` entry as the argmax.

use rusty_llama::{Sampler, Temperature};

#[test]
fn sample_ignores_nan_logits() {
    let sampler = Temperature::new(1.0);
    // Index 2 is the true maximum; index 1 is NaN and must be ignored.
    let logits = [0.1, f32::NAN, 5.0, 2.0];
    assert_eq!(sampler.sample(&logits), 2);
}

#[test]
fn sample_mut_ignores_nan_logits() {
    let sampler = Temperature::new(1.0);
    let mut logits = [f32::NAN, 3.0, f32::NAN, 1.0];
    assert_eq!(sampler.sample_mut(&mut logits), 1);
}

#[test]
fn sample_all_nan_does_not_panic() {
    let sampler = Temperature::new(1.0);
    let logits = [f32::NAN, f32::NAN, f32::NAN];
    // No comparable logits: falls back to token 0 instead of panicking.
    assert_eq!(sampler.sample(&logits), 0);
}

#[test]
fn sample_empty_does_not_panic() {
    let sampler = Temperature::new(1.0);
    assert_eq!(sampler.sample(&[]), 0);
}

#[test]
fn sample_normal_logits_pick_max() {
    let sampler = Temperature::new(1.0);
    let logits = [-1.0, 0.0, 3.5, 3.4];
    assert_eq!(sampler.sample(&logits), 2);
}
