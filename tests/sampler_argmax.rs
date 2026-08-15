//! Regression tests: argmax-based token selection must not panic on NaN or
//! empty logits.

use rusty_llama::{Chain, Greedy, Sampler, Temperature, TopK};

#[test]
fn greedy_skips_nan_logits() {
    let mut g = Greedy::new();
    assert_eq!(g.sample(&[f32::NAN, 1.0, 0.5]), 1);
    assert_eq!(g.sample(&[1.0, f32::NAN, 2.0]), 2);
    assert_eq!(g.sample(&[2.0, 1.0, f32::NAN]), 0);
}

#[test]
fn greedy_handles_all_nan_logits() {
    let mut g = Greedy::new();
    let token = g.sample(&[f32::NAN, f32::NAN]);
    assert!((0..2).contains(&token));
}

#[test]
fn greedy_handles_empty_logits() {
    let mut g = Greedy::new();
    assert_eq!(g.sample(&[]), 0);

    let mut logits: Vec<f32> = Vec::new();
    assert_eq!(g.sample_mut(&mut logits), 0);
}

#[test]
fn greedy_sample_mut_skips_nan_logits() {
    let mut g = Greedy::new();
    let mut logits = vec![f32::NAN, 3.0, 1.0];
    assert_eq!(g.sample_mut(&mut logits), 1);
}

#[test]
fn nan_logits_are_ignored_over_neg_infinity() {
    // -inf is a valid (masked) logit; NaN must not shadow real candidates.
    let mut g = Greedy::new();
    assert_eq!(g.sample(&[f32::NEG_INFINITY, f32::NAN, -5.0]), 2);
}

#[test]
fn chain_masking_then_infinite_temperature_does_not_panic() {
    // TopK masks losers to -inf; scaling by 1/inf multiplies -inf by 0.0,
    // which yields NaN. Reachable entirely from the safe public API.
    let mut chain = Chain::new()
        .with(TopK::new(1))
        .with(Temperature::new(f32::INFINITY))
        .with(Greedy::new());
    let token = chain.sample(&[1.0, 5.0, 2.0]);
    assert!((0..3).contains(&token));
}

#[test]
fn empty_chain_handles_nan_logits() {
    // With no samplers, Chain::sample falls back to a bare argmax.
    let mut chain = Chain::new();
    assert_eq!(chain.sample(&[0.0, f32::NAN, 4.0]), 2);
    assert_eq!(chain.sample(&[]), 0);
}
