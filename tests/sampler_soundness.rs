use rusty_llama::{MinP, Temperature};

// Import the Sampler trait methods (sample, sample_mut, apply)
use rusty_llama::Sampler;

/// Logits containing NaN must not cause a panic in sample() or sample_mut().
/// Before the fix, the default Sampler::sample and Sampler::sample_mut
/// implementations used `partial_cmp(b).unwrap()` which panics on NaN,
/// since f32::partial_cmp returns None when either operand is NaN.
#[test]
fn sample_does_not_panic_on_nan_logits() {
    let logits = vec![1.0_f32, f32::NAN, 3.0, 2.0];
    let temp = Temperature::new(1.0);
    let token = temp.sample(&logits);
    assert!((0..logits.len() as i32).contains(&token));
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let mut logits = vec![1.0_f32, f32::NAN, 3.0, 2.0];
    let temp = Temperature::new(1.0);
    let token = temp.sample_mut(&mut logits);
    assert!((0..logits.len() as i32).contains(&token));
}

/// MinP filtering can produce NEG_INFINITY entries; downstream arithmetic
/// (e.g. multiplying by 0) can turn those into NaN. Verify the full
/// MinP → Temperature → Dist chain handles this gracefully.
#[test]
fn minp_then_temperature_chain_with_nan() {
    let mut logits = vec![10.0, f32::NAN, -5.0, 0.0];
    let minp = MinP::new(0.05, 1);
    minp.apply_mut(&mut logits);
    let temp = Temperature::new(1.0);
    let token = temp.sample_mut(&mut logits);
    assert!((0..4).contains(&token));
}

/// All-NaN logits should still pick a token (the last one, per total_cmp
/// treating NaN as greater than all finite values).
#[test]
fn sample_all_nan_logits() {
    let logits = vec![f32::NAN; 4];
    let temp = Temperature::new(1.0);
    let token = temp.sample(&logits);
    assert!((0..4).contains(&token));
}

/// NEG_INFINITY * 0.0 = NaN — this can occur when MinP sets entries to
/// NEG_INFINITY and a subsequent temperature of infinity produces inv_temp = 0.
#[test]
fn neg_inf_times_zero_produces_nan_handled() {
    let mut logits = vec![5.0, f32::NEG_INFINITY, 3.0];
    let temp = Temperature::new(f32::INFINITY);
    temp.apply_mut(&mut logits);
    // logits[1] is now NaN (NEG_INFINITY * 0.0)
    assert!(logits[1].is_nan());
    // sample should still work without panicking
    let token = temp.sample_mut(&mut logits);
    assert!((0..3).contains(&token));
}
