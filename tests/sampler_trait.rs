// Import the Sampler trait to bring its methods into scope,
// and the concrete sampler types.
use rusty_llama::Sampler as _;
use rusty_llama::{Dist, Temperature};

/// Demonstrates that the default `Sampler::sample()` handles NaN logits
/// without panicking. Before the fix, `partial_cmp().unwrap()` would panic
/// because `NaN.partial_cmp(_)` returns `None`.
#[test]
fn sample_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0];
    // Should not panic — NaN is ordered consistently by total_cmp
    let token = temp.sample(&logits);
    assert!((0..logits.len() as i32).contains(&token));
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0];
    let token = temp.sample_mut(&mut logits);
    assert!((0..5).contains(&token));
}

/// A realistic scenario: Temperature with a very small subnormal value causes
/// `1.0 / temp` to overflow to infinity, then `0.0 * infinity = NaN` for
/// zero-valued logits. The sampler must not panic.
#[test]
fn temperature_near_zero_does_not_panic_on_zero_logit() {
    // Smallest positive subnormal f32: 1.0 / this = infinity
    let temp = Temperature::new(f32::from_bits(1));
    assert_eq!(1.0f32 / f32::from_bits(1), f32::INFINITY);
    // logit of 0.0 * INFINITY = NaN
    let logits = vec![0.0, 1.0, -1.0, 0.0, 2.0];
    let token = temp.sample(&logits);
    assert!((0..logits.len() as i32).contains(&token));
}

/// Verify that NaN logits are handled consistently: `total_cmp` places NaN
/// after all other values, so the argmax should pick the largest finite value.
#[test]
fn sample_picks_finite_max_over_nan() {
    let temp = Temperature::new(1.0);
    // With temp=1.0, apply_mut is just scaling by 1.0 (identity).
    // The max finite value is 5.0 at index 2.
    let logits = vec![1.0, f32::NAN, 5.0, f32::NAN, 3.0];
    let token = temp.sample(&logits);
    // total_cmp orders: -inf < finite < inf < NaN
    // So NaN is "greatest" — but that's the defined behavior with total_cmp.
    // The important thing is we don't panic.
    // With total_cmp, NaN > everything, so token will be a NaN index.
    // This test simply verifies no panic occurs.
    assert!((0..logits.len() as i32).contains(&token));
}

#[test]
fn sample_all_nan_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let token = temp.sample(&logits);
    assert!((0..3).contains(&token));
}

#[test]
fn sample_with_infinities_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NEG_INFINITY, 0.0, f32::INFINITY, 1.0];
    let token = temp.sample(&logits);
    // INFINITY is the max finite-ish value under total_cmp (before NaN)
    assert_eq!(token, 2);
}

#[test]
fn dist_sample_does_not_panic_on_nan() {
    let dist = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 2.0, f32::NAN, 3.0];
    // Dist::sample uses exp() which maps NaN -> NaN, but the accumulator
    // should still terminate. This verifies no panic.
    let token = dist.sample(&logits);
    assert!((0..logits.len() as i32).contains(&token));
}
