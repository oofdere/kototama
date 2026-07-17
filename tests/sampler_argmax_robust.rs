//! Regression tests for the shared `argmax` used by token-selecting samplers.
//!
//! `Greedy`/`Chain`/`Sampler::sample` must not panic when the logits contain a
//! NaN or when the slice is empty. These are reachable from safe public API.

use rusty_llama::{Chain, Greedy, Sampler};

#[test]
fn greedy_sample_ignores_nan() {
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    assert_eq!(Greedy::new().sample(&logits), 2, "argmax should skip NaN");
}

#[test]
fn greedy_sample_all_nan_does_not_panic() {
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let token = Greedy::new().sample(&logits);
    assert!((0..3).contains(&token));
}

#[test]
fn greedy_sample_empty_does_not_panic() {
    let token = Greedy::new().sample(&[]);
    assert_eq!(token, 0);
}

#[test]
fn chain_sample_empty_does_not_panic() {
    // An empty chain falls back to argmax over the raw logits.
    let token = Chain::new().sample(&[]);
    assert_eq!(token, 0);
}

#[test]
fn sample_mut_ignores_nan() {
    let mut logits = vec![f32::NAN, 5.0, 1.0];
    assert_eq!(Greedy::new().sample_mut(&mut logits), 1);
}
