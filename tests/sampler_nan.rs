use rusty_llama::{Sampler, Temperature};

/// Regression test: the default `Sampler::sample` must not panic on NaN logits.
///
/// Before the fix, the implementation used `partial_cmp().unwrap()` which panics
/// when comparing NaN values. This test ensures the `total_cmp` replacement
/// handles NaN gracefully.
#[test]
fn sample_does_not_panic_on_nan_logits() {
    let sampler = Temperature::new(1.0);
    let logits: Vec<f32> = vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0];
    // Should not panic; NaN sorts below all finite values with total_cmp
    let token = sampler.sample(&logits);
    // total_cmp treats NaN as greater than all other values (including +inf),
    // so the result should be one of the NaN positions (index 1 or 3).
    // The important property is that it does NOT panic.
    assert!(token >= 0 && token < logits.len() as i32);
}

/// Verify sample_mut also handles NaN correctly.
#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let sampler = Temperature::new(1.0);
    let mut logits: Vec<f32> = vec![f32::NAN, 0.5, 1.5, f32::NAN];
    let token = sampler.sample_mut(&mut logits);
    assert!(token >= 0 && token < 4);
}

/// All-NaN input must not panic.
#[test]
fn sample_all_nan_logits() {
    let sampler = Temperature::new(1.0);
    let logits: Vec<f32> = vec![f32::NAN; 8];
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < 8);
}
