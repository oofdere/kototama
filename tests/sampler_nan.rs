use rusty_llama::{Sampler, Temperature, MinP, Dist};

// --- NaN safety in default Sampler::sample() ---

#[test]
fn sample_with_nan_logits_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Before fix: panicked at `partial_cmp().unwrap()` on NaN.
    // After fix: total_cmp treats NaN as greater than all values, so this
    // returns deterministically without panicking.
    let _token = temp.sample(&logits);
}

#[test]
fn sample_mut_with_nan_logits_does_not_panic() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let _token = temp.sample_mut(&mut logits);
}

#[test]
fn sample_all_nan_does_not_panic() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN; 8];
    let _token = temp.sample(&logits);
}

// --- NaN safety in Dist::sample() ---

#[test]
fn dist_sample_with_nan_logits_does_not_panic() {
    let dist = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Before fix: f32::max propagates NaN, making all exps NaN and sampling broken.
    // After fix: NaN logits are skipped when finding the max, producing a valid sample.
    let token = dist.sample(&logits);
    assert!(token >= 0);
}

// --- MinP with p=0 produces -inf threshold without panic ---

#[test]
fn minp_zero_p_does_not_panic() {
    let minp = MinP::new(0.0, 1);
    let mut logits = vec![1.0, 2.0, 3.0, 0.5];
    // p=0 means p.ln() = -inf; threshold = logit_max + (-inf) = -inf
    // All logits survive, no panic.
    minp.apply_mut(&mut logits);
}

// --- Normal operation still works correctly ---

#[test]
fn sample_picks_argmax_without_nan() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, 5.0, 3.0, 2.0];
    let token = temp.sample(&logits);
    assert_eq!(token, 1, "should pick index of max logit");
}

#[test]
fn sample_mut_picks_argmax_without_nan() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, 5.0, 3.0, 2.0];
    let token = temp.sample_mut(&mut logits);
    assert_eq!(token, 1, "should pick index of max logit");
}

#[test]
fn dist_sample_valid_range() {
    let dist = Dist::new(123);
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let token = dist.sample(&logits);
    assert!(token >= 0 && token < logits.len() as i32);
}
