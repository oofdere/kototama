use rusty_llama::{Dist, MinP, Sampler, Temperature};

// --- NaN soundness: sample must not panic on NaN logits ---

struct Identity;

impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_with_nan_does_not_panic() {
    let s = Identity;
    let logits = vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0];
    // Before the fix, this would panic due to partial_cmp().unwrap() on NaN.
    let token = s.sample(&logits);
    // total_cmp treats NaN as greater than all finite values,
    // so the result should be one of the NaN indices (1 or 3).
    assert!(token == 1 || token == 3);
}

#[test]
fn sample_mut_with_nan_does_not_panic() {
    let s = Identity;
    let mut logits = vec![f32::NAN, 0.5, -1.0];
    let token = s.sample_mut(&mut logits);
    // NaN sorts highest under total_cmp, so index 0.
    assert_eq!(token, 0);
}

#[test]
fn sample_all_nan_does_not_panic() {
    let s = Identity;
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let token = s.sample(&logits);
    assert!(token >= 0 && token < 3);
}

#[test]
fn sample_with_infinity_does_not_panic() {
    let s = Identity;
    let logits = vec![1.0, f32::INFINITY, f32::NEG_INFINITY, 2.0];
    let token = s.sample(&logits);
    // +INF is the largest finite-comparable value under total_cmp (before NaN)
    assert_eq!(token, 1);
}

// --- Double-application regression: Temperature should be applied exactly once ---

#[test]
fn temperature_sample_applies_once() {
    let temp = Temperature::new(2.0);
    let logits: Vec<f32> = vec![1.0, 2.0, 4.0, 3.0];

    // Calling sample() should apply temperature (divide by 2) then argmax.
    // The argmax of [0.5, 1.0, 2.0, 1.5] is index 2.
    let token = temp.sample(&logits);
    assert_eq!(token, 2, "argmax should be at index 2 regardless of temperature scaling");

    // Verify the logits are scaled correctly by apply()
    let applied = temp.apply(&logits);
    assert_eq!(applied, vec![0.5, 1.0, 2.0, 1.5]);
}

#[test]
fn min_p_sample_applies_once() {
    let min_p = MinP::new(0.5, 1);
    // logits: max is 4.0, threshold = 4.0 + ln(0.5) ≈ 4.0 - 0.693 = 3.307
    // Only index 2 (value 4.0) and index 3 (value 3.5) survive.
    let logits: Vec<f32> = vec![1.0, 2.0, 4.0, 3.5];

    let token = min_p.sample(&logits);
    // After filtering, indices 0 and 1 become NEG_INFINITY.
    // Argmax should be index 2 (value 4.0).
    assert_eq!(token, 2);
}

// --- Dist sampler should handle NaN gracefully ---

#[test]
fn dist_sample_with_neg_inf_does_not_panic() {
    let dist = Dist::new(42);
    // After MinP filtering, many logits are NEG_INFINITY. Dist should handle this.
    let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
    let token = dist.sample(&logits);
    // Only index 2 has non-zero probability after softmax
    assert_eq!(token, 2);
}

#[test]
#[should_panic(expected = "cannot sample from empty logits")]
fn sample_empty_logits_panics_with_message() {
    let s = Identity;
    let logits: Vec<f32> = vec![];
    s.sample(&logits);
}
