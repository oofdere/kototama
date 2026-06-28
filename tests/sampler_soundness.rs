use rusty_llama::{Dist, MinP, Sampler, Temperature};

// ---------- NaN safety in default sample() / sample_mut() ----------

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Before the fix, partial_cmp().unwrap() would panic here.
    let token = temp.sample(&logits);
    assert!((0..4).contains(&token));
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = temp.sample_mut(&mut logits);
    assert!((0..4).contains(&token));
}

#[test]
fn sample_with_all_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    // total_cmp places NaN at the end (greatest), so this should not panic
    let token = temp.sample(&logits);
    assert!((0..3).contains(&token));
}

#[test]
fn sample_with_inf_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::INFINITY];
    let token = temp.sample(&logits);
    assert_eq!(token, 2, "should pick the +inf logit");
}

// ---------- Dist softmax with degenerate inputs ----------

#[test]
fn dist_sample_all_neg_inf_does_not_produce_nan() {
    let dist = Dist::new(42);
    let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    // Before the fix, this produced NaN via NEG_INFINITY - NEG_INFINITY,
    // causing silent fallthrough to the last token every time.
    let token = dist.sample(&logits);
    assert!((0..3).contains(&token));
}

#[test]
fn dist_sample_with_nan_logit() {
    let dist = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 2.0];
    let token = dist.sample(&logits);
    assert!((0..3).contains(&token));
}

// ---------- MinP into Dist pipeline ----------

#[test]
fn minp_then_dist_does_not_panic() {
    let minp = MinP::new(0.1, 1);
    let dist = Dist::new(42);
    let logits = vec![10.0, 0.001, 0.001, 0.001];
    let filtered = minp.apply(&logits);
    // filtered should contain NEG_INFINITY for suppressed tokens
    let token = dist.sample(&filtered);
    assert!((0..4).contains(&token));
}
