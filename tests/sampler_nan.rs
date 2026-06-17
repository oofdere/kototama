use rusty_llama::{Temperature, MinP};

// Import the Sampler trait so we can call `.sample()` / `.sample_mut()`.
use rusty_llama::Sampler;

/// Logits containing NaN must not panic in the default `sample()` implementation.
/// Before the fix, `partial_cmp().unwrap()` on NaN would panic; now we use
/// `total_cmp()` which provides a total ordering for all f32 values.
#[test]
fn sample_does_not_panic_on_nan_logits() {
    let logits = vec![1.0_f32, f32::NAN, 3.0, 2.0];
    let temp = Temperature::new(1.0);
    let _token = temp.sample(&logits);
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let mut logits = vec![1.0_f32, f32::NAN, 3.0, 2.0];
    let temp = Temperature::new(1.0);
    let _token = temp.sample_mut(&mut logits);
}

#[test]
fn sample_returns_valid_index_with_nan() {
    let logits = vec![1.0_f32, f32::NAN, 3.0, 2.0];
    let temp = Temperature::new(1.0);
    let token = temp.sample(&logits);
    assert!(
        (token as usize) < logits.len(),
        "sampled token index {} should be within logits range 0..{}",
        token,
        logits.len()
    );
}

#[test]
fn sample_all_nan_does_not_panic() {
    let logits = vec![f32::NAN; 8];
    let temp = Temperature::new(1.0);
    let token = temp.sample(&logits);
    assert!(
        (token as usize) < logits.len(),
        "token index must be in range even when all logits are NaN"
    );
}

#[test]
fn sample_nan_with_minp_does_not_panic() {
    let logits = vec![1.0_f32, f32::NAN, 3.0, 2.0];
    let minp = MinP::new(0.05, 1);
    let _token = minp.sample(&logits);
}

/// Sanity check: when there are no NaN values, the argmax is correct.
#[test]
fn sample_selects_argmax_without_nan() {
    let logits = vec![1.0_f32, 5.0, 3.0, 2.0];
    let temp = Temperature::new(1.0);
    let token = temp.sample(&logits);
    assert_eq!(token, 1, "should select index 1 (logit 5.0)");
}
