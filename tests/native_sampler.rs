use rusty_llama::{Dist, MinP, Sampler, Temperature};

// ---- NaN safety: sample/sample_mut must not panic on NaN logits ----

#[test]
fn sample_does_not_panic_on_nan() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Before the fix this would panic via partial_cmp().unwrap()
    let token = temp.sample(&logits);
    assert!((token as usize) < logits.len());
}

#[test]
fn sample_mut_does_not_panic_on_nan() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![f32::NAN, 1.0, 2.0];
    let token = temp.sample_mut(&mut logits);
    assert!((token as usize) < 3);
}

#[test]
fn sample_does_not_panic_on_all_nan() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let token = temp.sample(&logits);
    assert!((token as usize) < logits.len());
}

#[test]
fn sample_returns_zero_on_empty_logits() {
    let temp = Temperature::new(1.0);
    let logits: Vec<f32> = vec![];
    let token = temp.sample(&logits);
    assert_eq!(token, 0);
}

// ---- NaN safety with MinP ----

#[test]
fn minp_sample_does_not_panic_on_nan() {
    let minp = MinP::new(0.05, 1);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    let token = minp.sample(&logits);
    assert!((token as usize) < logits.len());
}

// ---- NaN safety with Dist ----

#[test]
fn dist_sample_does_not_panic_on_nan() {
    let dist = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Dist has its own sample override; ensure it doesn't panic either
    let token = dist.sample(&logits);
    assert!((token as usize) < logits.len());
}

// ---- Correctness: total_cmp treats NaN as greater than all values ----

#[test]
fn sample_nan_is_deterministic() {
    let temp = Temperature::new(1.0);
    // With total_cmp, NaN sorts as greater than all finite values.
    // The last NaN in the list wins (max_by picks the last equal element).
    let logits = vec![100.0, f32::NAN, -1.0];
    let t1 = temp.sample(&logits);
    let t2 = temp.sample(&logits);
    assert_eq!(t1, t2, "sampling NaN logits should be deterministic");
}

// ---- Temperature scaling correctness ----

#[test]
fn temperature_preserves_argmax() {
    let temp = Temperature::new(0.5);
    let logits = vec![1.0, 5.0, 3.0, 2.0];
    let token = temp.sample(&logits);
    assert_eq!(token, 1, "argmax should be index 1 (value 5.0)");
}
