use rusty_llama::{Dist, Sampler, Temperature};

// ---------- NaN soundness ----------
//
// The default Sampler::sample() and sample_mut() use partial_cmp().unwrap()
// which panics on NaN. Logits from C models CAN contain NaN (numerical
// instability, corrupted weights, or user-constructed inputs). The sampler
// must not panic on valid f32 values.

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let sampler = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Should not panic — must produce a valid token index
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
fn sample_all_nan_logits() {
    let sampler = Temperature::new(1.0);
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    // Should still return a valid index, not panic
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < 3);
}

#[test]
fn sample_with_infinity_logits() {
    let sampler = Temperature::new(1.0);
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::INFINITY, 2.0];
    let token = sampler.sample(&logits);
    // INFINITY should be selected as the max
    assert_eq!(token, 2);
}

#[test]
fn sample_nan_mixed_with_neg_infinity() {
    // After MinP filtering, some logits become NEG_INFINITY.
    // If the model also produces NaN, both coexist.
    let sampler = Temperature::new(0.5);
    let logits = vec![f32::NEG_INFINITY, f32::NAN, 3.0, f32::NEG_INFINITY];
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < 4);
}

// ---------- Empty-slice soundness ----------
//
// The default sample() and sample_mut() call .unwrap() on max_by(),
// which panics on an empty iterator. While unusual, an empty logits
// slice should not crash the program.

#[test]
#[should_panic(expected = "cannot sample from empty logits")]
fn sample_empty_logits_panics_with_clear_message() {
    let sampler = Temperature::new(1.0);
    let logits: Vec<f32> = vec![];
    // Empty logits is a programming error (vocab size must be > 0).
    // The sampler should panic with a descriptive message rather than
    // an opaque "called unwrap() on None" from the iterator.
    sampler.sample(&logits);
}

// ---------- Dist sampler with NaN ----------

#[test]
fn dist_sample_does_not_panic_on_nan() {
    let sampler = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Dist does softmax sampling — NaN in exp() produces NaN,
    // which should be handled without panic
    let token = sampler.sample(&logits);
    assert!(token >= 0 && token < 4);
}
