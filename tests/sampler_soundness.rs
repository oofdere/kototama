use rusty_llama::{Temperature, MinP, Dist};

// Re-import the Sampler trait so we can call trait methods directly.
use rusty_llama::Sampler;

use std::sync::atomic::{AtomicU32, Ordering};

// --- NaN-safety tests ---
// Before the fix, these panicked because `partial_cmp().unwrap()` returns None for NaN.

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Should not panic — NaN is ordered by total_cmp, not partial_cmp.
    let _token = temp.sample(&logits);
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let _token = temp.sample_mut(&mut logits);
}

#[test]
fn sample_does_not_panic_on_all_nan() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN; 4];
    let _token = temp.sample(&logits);
}

#[test]
fn sample_picks_finite_max_when_nan_present() {
    let temp = Temperature::new(1.0);
    // logits: [1.0, NaN, 5.0, 2.0]
    // total_cmp orders NaN > +Inf, so NaN index 1 would win.
    // The key property: no panic.
    let logits = vec![1.0, f32::NAN, 5.0, 2.0];
    let token = temp.sample(&logits);
    assert!(token >= 0 && token < 4);
}

// --- Double-application test ---
// Before the fix, Sequence::sample called apply() then sample(),
// meaning apply_mut was invoked twice.  We verify with a counting sampler
// that the trait's default sample() calls apply_mut exactly once.

static APPLY_COUNT: AtomicU32 = AtomicU32::new(0);

struct CountingSampler;

impl Sampler for CountingSampler {
    fn apply_mut(&self, _logits: &mut [f32]) {
        APPLY_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn default_sample_applies_exactly_once() {
    APPLY_COUNT.store(0, Ordering::SeqCst);
    let s = CountingSampler;
    let logits = vec![1.0, 2.0, 3.0];
    let _token = s.sample(&logits);
    assert_eq!(
        APPLY_COUNT.load(Ordering::SeqCst),
        1,
        "Sampler::sample should call apply_mut exactly once"
    );
}

#[test]
fn default_sample_mut_applies_exactly_once() {
    APPLY_COUNT.store(0, Ordering::SeqCst);
    let s = CountingSampler;
    let mut logits = vec![1.0, 2.0, 3.0];
    let _token = s.sample_mut(&mut logits);
    assert_eq!(
        APPLY_COUNT.load(Ordering::SeqCst),
        1,
        "Sampler::sample_mut should call apply_mut exactly once"
    );
}

// --- Temperature correctness ---

#[test]
fn temperature_zero_is_identity() {
    let temp = Temperature::new(0.0);
    let logits = vec![1.0, 2.0, 3.0];
    let result = temp.apply(&logits);
    assert_eq!(result, logits);
}

#[test]
fn temperature_sample_picks_argmax() {
    let temp = Temperature::new(0.5);
    let logits = vec![1.0, 5.0, 3.0];
    let token = temp.sample(&logits);
    assert_eq!(token, 1, "temperature sampler should pick the argmax token");
}

// --- Dist correctness ---

#[test]
fn dist_sample_returns_valid_index() {
    let dist = Dist::new(42);
    let logits = vec![0.0, 1.0, 2.0, 3.0];
    let token = dist.sample(&logits);
    assert!(token >= 0 && token < 4);
}

#[test]
fn dist_apply_mut_is_noop() {
    let dist = Dist::new(42);
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    dist.apply_mut(&mut logits);
    assert_eq!(logits, original);
}

// --- MinP correctness ---

#[test]
fn min_p_masks_low_probability_tokens() {
    let min_p = MinP::new(0.1, 1);
    let logits = vec![10.0, 0.0, -100.0];
    let result = min_p.apply(&logits);
    assert!(result[2] == f32::NEG_INFINITY, "very low logit should be masked");
    assert!(result[0].is_finite(), "top logit should remain finite");
}
