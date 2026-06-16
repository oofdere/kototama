use rusty_llama::{Dist, MinP, Temperature};
use rusty_llama::Sampler;

// --- NaN soundness: sample/sample_mut must not panic on NaN logits ---

struct Identity;
impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let sampler = Identity;
    let logits = vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0];
    // Before the fix, this would panic due to partial_cmp().unwrap() on NaN.
    let token = sampler.sample(&logits);
    // total_cmp treats NaN as greater than all other values,
    // so the result should be one of the NaN indices (1 or 3).
    assert!(token == 1 || token == 3);
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let sampler = Identity;
    let mut logits = vec![f32::NAN, 0.5, -1.0];
    let token = sampler.sample_mut(&mut logits);
    // NaN is treated as greatest by total_cmp.
    assert_eq!(token, 0);
}

#[test]
fn sample_all_nan_does_not_panic() {
    let sampler = Identity;
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let token = sampler.sample(&logits);
    // Any index is valid since all values are NaN.
    assert!(token >= 0 && token < 3);
}

#[test]
fn sample_with_neg_infinity_works() {
    let sampler = Identity;
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
    let token = sampler.sample(&logits);
    assert_eq!(token, 1);
}

// --- MinP produces NEG_INFINITY but sample still works ---

#[test]
fn sample_after_min_p_filtering() {
    let min_p = MinP::new(0.5, 1);
    // After min_p, some logits become NEG_INFINITY.
    // The default sample must still find the max without panicking.
    let logits: Vec<f32> = vec![10.0, 1.0, 0.5, -1.0, -5.0];
    let token = min_p.sample(&logits);
    assert_eq!(token, 0, "should pick the highest logit");
}

// --- Temperature with NaN input ---

#[test]
fn temperature_sample_with_nan_input() {
    let temp = Temperature::new(0.8);
    let logits = vec![1.0, f32::NAN, 2.0];
    // NaN * anything = NaN; total_cmp handles NaN as max.
    let token = temp.sample(&logits);
    assert_eq!(token, 1, "NaN stays NaN after scaling; total_cmp treats it as max");
}

// --- Dist handles NaN gracefully in its custom sample impl ---

#[test]
fn dist_sample_with_all_neg_inf() {
    let dist = Dist::new(42);
    // All NEG_INFINITY → exp gives all 0 → sum = 0 → should not panic.
    let logits = vec![f32::NEG_INFINITY; 5];
    let token = dist.sample(&logits);
    assert!(token >= 0 && token < 5);
}
