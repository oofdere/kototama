use rusty_llama::{Dist, MinP, Sampler, Temperature};

#[test]
fn sample_nan_logits_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Must not panic — NaN logits can appear from numerical instability.
    let token = temp.sample(&logits);
    assert!((0..logits.len() as i32).contains(&token));
}

#[test]
fn sample_mut_nan_logits_does_not_panic() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = temp.sample_mut(&mut logits);
    assert!((0..4).contains(&token));
}

#[test]
fn sample_all_nan_logits_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN; 8];
    let token = temp.sample(&logits);
    assert!((0..8).contains(&token));
}

#[test]
fn sample_inf_logits_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::INFINITY, 2.0];
    let token = temp.sample(&logits);
    assert_eq!(token, 2, "should pick INFINITY as the max");
}

#[test]
fn temperature_zero_is_identity() {
    let temp = Temperature::new(0.0);
    let logits = vec![1.0, 2.0, 3.0];
    let out = temp.apply(&logits);
    assert_eq!(out, logits);
}

#[test]
fn minp_nan_logits_does_not_panic() {
    let minp = MinP::new(0.05, 1);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let out = minp.apply(&logits);
    assert_eq!(out.len(), logits.len());
}

#[test]
fn dist_sample_basic() {
    let dist = Dist::new(42);
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let token = dist.sample(&logits);
    assert!((0..4).contains(&token));
}
