use rusty_llama::{Dist, MinP, Sampler, Temperature};

// ---------- NaN safety ----------
// Before this fix, the default Sampler::sample() / sample_mut() used
// partial_cmp().unwrap() which panics when any logit is NaN.

struct Identity;
impl Sampler for Identity {
    fn apply_mut(&self, _logits: &mut [f32]) {}
}

#[test]
fn sample_nan_returns_some_instead_of_panicking() {
    let s = Identity;
    let logits = [1.0, f32::NAN, 3.0];
    let result = s.sample(&logits);
    assert!(result.is_some(), "sample should not panic on NaN logits");
}

#[test]
fn sample_mut_nan_returns_some_instead_of_panicking() {
    let s = Identity;
    let mut logits = [1.0, f32::NAN, 3.0];
    let result = s.sample_mut(&mut logits);
    assert!(result.is_some(), "sample_mut should not panic on NaN logits");
}

#[test]
fn sample_all_nan_returns_some() {
    let s = Identity;
    let logits = [f32::NAN, f32::NAN, f32::NAN];
    let result = s.sample(&logits);
    assert!(result.is_some());
}

// ---------- Empty-slice safety ----------

#[test]
fn sample_empty_returns_none() {
    let s = Identity;
    let logits: [f32; 0] = [];
    assert_eq!(s.sample(&logits), None);
}

#[test]
fn sample_mut_empty_returns_none() {
    let s = Identity;
    let mut logits: [f32; 0] = [];
    assert_eq!(s.sample_mut(&mut logits), None);
}

#[test]
fn dist_sample_empty_returns_none() {
    let d = Dist::new(42);
    let logits: [f32; 0] = [];
    assert_eq!(d.sample(&logits), None);
}

// ---------- Normal operation still works ----------

#[test]
fn sample_picks_argmax() {
    let s = Identity;
    let logits = [1.0, 5.0, 3.0, 2.0];
    assert_eq!(s.sample(&logits), Some(1));
}

#[test]
fn temperature_sample_picks_argmax() {
    let t = Temperature::new(0.5);
    let logits = [1.0, 5.0, 3.0];
    assert_eq!(t.sample(&logits), Some(1));
}

#[test]
fn min_p_sample_picks_argmax() {
    let mp = MinP::new(0.05, 1);
    let logits = [1.0, 5.0, 3.0];
    assert_eq!(mp.sample(&logits), Some(1));
}

#[test]
fn dist_sample_returns_valid_index() {
    let d = Dist::new(42);
    let logits = [1.0, 2.0, 3.0];
    let token = d.sample(&logits).unwrap();
    assert!(token >= 0 && token < 3);
}
