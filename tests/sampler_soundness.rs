use rusty_llama::{Dist, MinP, Sampler, Temperature};

struct Identity;

impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_nan_logits_does_not_panic() {
    let sampler = Identity;
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Before the fix, this panicked because partial_cmp(NaN, _) returns None.
    let token = sampler.sample(&logits);
    assert!((0..4).contains(&token));
}

#[test]
fn sample_mut_nan_logits_does_not_panic() {
    let sampler = Identity;
    let mut logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = sampler.sample_mut(&mut logits);
    assert!((0..4).contains(&token));
}

#[test]
fn sample_all_nan_logits_returns_valid_index() {
    let sampler = Identity;
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let token = sampler.sample(&logits);
    assert!((0..3).contains(&token));
}

#[test]
fn sample_nan_with_temperature_does_not_panic() {
    let temp = Temperature::new(0.8);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = temp.sample(&logits);
    assert!((0..4).contains(&token));
}

#[test]
fn sample_nan_with_min_p_does_not_panic() {
    let min_p = MinP::new(0.05, 1);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = min_p.sample(&logits);
    assert!((0..4).contains(&token));
}

#[test]
fn sample_inf_logits_returns_inf_index() {
    let sampler = Identity;
    let logits = vec![1.0, f32::INFINITY, 3.0];
    let token = sampler.sample(&logits);
    assert_eq!(token, 1, "should pick the +inf logit");
}

#[test]
fn sample_neg_inf_logits_skips_them() {
    let sampler = Identity;
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
    let token = sampler.sample(&logits);
    assert_eq!(token, 1, "should pick the only finite logit");
}

#[test]
fn dist_sample_nan_logits_does_not_panic() {
    let dist = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Dist overrides sample() with its own softmax-based implementation.
    // NaN in exp() produces NaN, but it should not panic.
    let token = dist.sample(&logits);
    assert!((0..4).contains(&token));
}
