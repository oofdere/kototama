//! Regression tests for `Dist::sample` on degenerate logit distributions.
//!
//! `NaN` and all-`-inf` logit vectors are reachable from safe code (e.g.
//! `TopK` masks with `-inf`, and `Temperature::new(f32::INFINITY)` turns those
//! into `NaN`). The softmax used by `Dist` produced `NaN` in those cases, every
//! `acc >= r` comparison was false, and the fallback returned the *last* index
//! — a token whose probability was zero.

use rusty_llama::{Dist, Sampler, Temperature, TopK};

#[test]
fn nan_logit_does_not_hijack_selection() {
    let logits = [5.0, 1.0, f32::NAN];
    for seed in 0..64 {
        let mut d = Dist::new(seed);
        let t = d.sample(&logits);
        assert!(t == 0 || t == 1, "NaN token {t} picked (seed {seed})");
    }
}

#[test]
fn nan_logits_still_follow_the_distribution() {
    let logits = [10.0, 0.0, f32::NAN];
    let mut d = Dist::new(3);
    let picks = (0..200).map(|_| d.sample(&logits));
    let zeros = picks.filter(|&t| t == 0).count();
    assert!(zeros > 150, "expected token 0 to dominate, got {zeros}/200");
}

#[test]
fn all_masked_logits_return_zero() {
    let logits = [f32::NEG_INFINITY; 4];
    let mut d = Dist::new(11);
    assert_eq!(d.sample(&logits), 0);
}

#[test]
fn all_nan_logits_return_zero() {
    let logits = [f32::NAN, f32::NAN];
    let mut d = Dist::new(11);
    assert_eq!(d.sample(&logits), 0);
}

#[test]
fn empty_logits_return_zero() {
    let mut d = Dist::new(5);
    assert_eq!(d.sample(&[]), 0);
}

#[test]
fn single_survivor_is_always_picked() {
    let logits = [
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        0.5,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for seed in 0..64 {
        let mut d = Dist::new(seed);
        assert_eq!(d.sample(&logits), 2, "seed {seed}");
    }
}

#[test]
fn positive_infinity_takes_all_the_mass() {
    let logits = [1.0, f32::INFINITY, 2.0];
    for seed in 0..16 {
        let mut d = Dist::new(seed);
        assert_eq!(d.sample(&logits), 1, "seed {seed}");
    }
}

#[test]
fn multiple_positive_infinities_pick_an_infinite_token() {
    let logits = [f32::INFINITY, 0.0, f32::INFINITY];
    let mut d = Dist::new(7);
    let t = d.sample(&logits);
    assert!(t == 0 || t == 2, "finite token {t} picked over +inf");
}

#[test]
fn masked_then_infinite_temperature_stays_in_the_top_k() {
    // TopK masks with -inf, Temperature(inf) scales by 0 → -inf * 0 = NaN.
    let logits: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let masked = TopK::new(2).apply(&logits);
    let nan_logits = Temperature::new(f32::INFINITY).apply(&masked);
    assert!(nan_logits.iter().any(|l| l.is_nan()), "NaN is reachable");

    // Only the two surviving tokens (6, 7) became 0.0; the rest are NaN.
    for seed in 0..32 {
        let mut d = Dist::new(seed);
        let t = d.sample(&nan_logits);
        assert!(t == 6 || t == 7, "masked token {t} picked (seed {seed})");
    }
}
