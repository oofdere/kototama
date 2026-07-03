use rusty_llama::{Sampler as SamplerTrait, Temperature, MinP, Dist};

// ---------- NaN handling in default sample()/sample_mut() ----------
// Before the fix, partial_cmp().unwrap() would panic on NaN logits.

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let sampler = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < logits.len() as i32);
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let sampler = Temperature::new(1.0);
    let mut logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = sampler.sample_mut(&mut logits);
    assert!(token >= 0 && token < 4);
}

#[test]
fn sample_does_not_panic_on_all_nan() {
    let sampler = Temperature::new(1.0);
    let logits = vec![f32::NAN; 4];
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < 4);
}

#[test]
fn sample_picks_valid_token_with_infinity() {
    let sampler = Temperature::new(1.0);
    let logits = vec![1.0, f32::NEG_INFINITY, f32::INFINITY, 2.0];
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < logits.len() as i32);
}

// NaN can arise from MinP filtering (sets filtered logits to NEG_INFINITY)
// followed by Temperature with infinite temp (inv_temp = 0, NEG_INFINITY * 0 = NaN).
#[test]
fn nan_from_minp_then_infinite_temp_does_not_panic() {
    let min_p = MinP::new(0.5, 1);
    let temp = Temperature::new(f32::INFINITY);

    let mut logits = vec![10.0, 1.0, 0.5, 0.1];
    min_p.apply_mut(&mut logits);
    // Some logits are now NEG_INFINITY after MinP filtering
    temp.apply_mut(&mut logits);
    // NEG_INFINITY * 0.0 = NaN
    assert!(logits.iter().any(|l| l.is_nan()), "expected NaN from NEG_INFINITY * 0");
    let token = temp.sample(&logits);
    assert!(token >= 0 && token < 4);
}

// ---------- Dist sampler with NaN ----------

#[test]
fn dist_sample_with_nan_does_not_panic() {
    let sampler = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < logits.len() as i32);
}
