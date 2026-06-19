//! Tests for sampler soundness — specifically NaN handling.
//!
//! The default `Sampler::sample` and `sample_mut` implementations must not
//! panic when logits contain NaN values. NaN can arise naturally when
//! Temperature scaling produces `0.0 * infinity` (IEEE 754 mandates NaN).

use rusty_llama::{Dist, MinP, Sampler, Temperature};

/// Applying Temperature with an extremely small value produces infinity
/// as the inverse, and any zero logit becomes NaN (0.0 * inf = NaN).
/// The sampler must not panic in this scenario.
#[test]
fn temperature_near_zero_does_not_panic() {
    let temp = Temperature::new(1e-45);
    // logits with a zero entry — will become NaN after temperature scaling
    let logits = vec![1.0_f32, 0.0, -1.0, 2.0, 0.0];
    // This must not panic
    let token = temp.sample(&logits);
    assert!(token >= 0 && (token as usize) < logits.len());
}

/// Directly verifying that sample/sample_mut handle NaN in the input
/// without panicking.
#[test]
fn sample_with_nan_logits_does_not_panic() {
    let temp = Temperature::new(1.0); // identity transform
    let logits = vec![1.0_f32, f32::NAN, -1.0, 2.0, f32::NAN];
    let token = temp.sample(&logits);
    assert!(token >= 0 && (token as usize) < logits.len());
}

/// sample_mut must also handle NaN without panicking.
#[test]
fn sample_mut_with_nan_logits_does_not_panic() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![f32::NAN, 1.0, 3.0, f32::NAN, 2.0];
    let token = temp.sample_mut(&mut logits);
    assert!(token >= 0 && (token as usize) < logits.len());
}

/// MinP sampler should also not panic with NaN logits.
#[test]
fn min_p_sample_with_nan_does_not_panic() {
    let min_p = MinP::new(0.05, 1);
    let logits = vec![1.0, f32::NAN, 3.0, 0.0, -1.0];
    let token = min_p.sample(&logits);
    assert!(token >= 0 && (token as usize) < logits.len());
}

/// Dist sampler handles NaN gracefully (NaN exps become NaN, but the
/// fallback return at the end of Dist::sample prevents infinite loops).
#[test]
fn dist_sample_with_nan_does_not_panic() {
    let dist = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 2.0, 0.0];
    let token = dist.sample(&logits);
    assert!(token >= 0 && (token as usize) < logits.len());
}

/// Verify that after fix, NaN is treated as "largest" by total_cmp,
/// so the argmax picks NaN's index. This confirms total_cmp behavior.
#[test]
fn total_cmp_treats_nan_as_largest() {
    let temp = Temperature::new(1.0);
    // With total_cmp, NaN > everything, so index 1 (the NaN) should be selected
    let logits = vec![1.0_f32, f32::NAN, 2.0];
    let token = temp.sample(&logits);
    // NaN is "greatest" under total_cmp, so token should be index 1
    assert_eq!(token, 1);
}
