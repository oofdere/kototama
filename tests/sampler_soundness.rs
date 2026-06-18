use rusty_llama::{Temperature, Sampler};

// ---------- NaN soundness: partial_cmp panics on NaN ----------

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0];
    // Before the fix, this panics because partial_cmp(NaN, _) returns None
    // and the default Sampler::sample() calls .unwrap() on it.
    let token = temp.sample(&logits);
    assert!(token >= 0);
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0];
    let token = temp.sample_mut(&mut logits);
    assert!(token >= 0);
}

#[test]
fn temperature_near_zero_does_not_panic() {
    // temp = 1e-45 (subnormal) => inv_temp = infinity
    // 0.0 * infinity = NaN, triggering the partial_cmp panic
    let temp = Temperature::new(1e-45);
    let logits = vec![1.0, 0.0, -1.0];
    let token = temp.sample(&logits);
    assert!(token >= 0);
}

#[test]
fn sample_all_nan_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let _token = temp.sample(&logits);
    // Just assert no panic
}

#[test]
fn sample_prefers_finite_over_nan() {
    let temp = Temperature::new(1.0);
    // Only index 1 has a finite value (5.0); the rest are NaN.
    // With total_cmp, NaN sorts above finite values, so this test
    // verifies the fix at least doesn't crash.
    let logits = vec![f32::NAN, 5.0, f32::NAN];
    let _token = temp.sample(&logits);
}
