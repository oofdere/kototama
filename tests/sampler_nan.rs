use rusty_llama::{Temperature, MinP};

// The Sampler trait is the one from src/sampler.rs
use rusty_llama::Sampler;

/// Before the fix, the default `sample` and `sample_mut` implementations on the
/// Sampler trait used `partial_cmp().unwrap()` which panics when logits contain
/// NaN. This can happen when logits come from the FFI layer (llama_get_logits_ith)
/// or when sampler transformations produce NaN (e.g. 0.0 * inf).
///
/// The fix uses `f32::total_cmp` which defines a total ordering for all f32
/// values including NaN, preventing the panic.

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let sampler = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Before the fix this would panic with:
    // "called `Option::unwrap()` on a `None` value" inside partial_cmp
    let token = sampler.sample(&logits);
    // With total_cmp, NaN sorts above all other values (IEEE 754 totalOrder),
    // so the argmax returns the index of the NaN element.
    assert_eq!(token, 1);
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let sampler = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = sampler.sample_mut(&mut logits);
    assert_eq!(token, 1);
}

#[test]
fn sample_handles_all_nan_logits() {
    let sampler = Temperature::new(1.0);
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    // Should not panic; returns some valid index
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < 3);
}

#[test]
fn sample_still_finds_argmax_without_nan() {
    let sampler = Temperature::new(1.0);
    let logits = vec![1.0, 5.0, 3.0, 2.0];
    let token = sampler.sample(&logits);
    assert_eq!(token, 1);
}

#[test]
fn sample_handles_infinity() {
    let sampler = Temperature::new(1.0);
    let logits = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY];
    let token = sampler.sample(&logits);
    assert_eq!(token, 1);
}

#[test]
fn sample_nan_with_min_p() {
    let sampler = MinP::new(0.1, 1);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // MinP's apply_mut won't filter NaN (since `NaN < thresh` is false),
    // and then sample's argmax should not panic.
    let token = sampler.sample(&logits);
    assert_eq!(token, 1);
}
