use rusty_llama::Sampler;

struct Identity;

impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_with_nan_does_not_panic() {
    let s = Identity;
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = s.sample(&logits);
    assert!((token as usize) < logits.len());
}

#[test]
fn sample_mut_with_nan_does_not_panic() {
    let s = Identity;
    let mut logits = vec![f32::NAN, 1.0, f32::NAN];
    let token = s.sample_mut(&mut logits);
    assert!((token as usize) < logits.len());
}

#[test]
fn sample_all_nan_does_not_panic() {
    let s = Identity;
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let token = s.sample(&logits);
    assert!((token as usize) < logits.len());
}

#[test]
fn sample_normal_logits_picks_argmax() {
    let s = Identity;
    let logits = vec![1.0, 5.0, 3.0, 2.0];
    let token = s.sample(&logits);
    assert_eq!(token, 1);
}

#[test]
fn sample_with_neg_infinity() {
    let s = Identity;
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
    let token = s.sample(&logits);
    assert_eq!(token, 1);
}

#[test]
fn temperature_near_zero_produces_nan_but_sample_handles_it() {
    use rusty_llama::Temperature;
    let temp = Temperature::new(1e-45);
    let logits = vec![0.0, 1.0, -1.0];
    // With temp ≈ 0, inv_temp = INFINITY, so 0.0 * INF = NaN
    let transformed = temp.apply(&logits);
    assert!(transformed[0].is_nan(), "0.0 * INF should produce NaN");
    let s = Identity;
    let token = s.sample(&transformed);
    assert!((token as usize) < logits.len());
}
