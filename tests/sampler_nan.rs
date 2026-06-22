use rusty_llama::{Temperature, Dist};

// Minimal Sampler import: the trait itself is needed for `.sample()` / `.sample_mut()`.
use rusty_llama::Sampler as SamplerTrait;

/// Regression test: `Sampler::sample` previously used `partial_cmp().unwrap()`
/// which panics when logits contain NaN. After the fix it uses `total_cmp`,
/// which defines a total order over all f32 values (NaN sorts last).
#[test]
fn sample_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Should not panic; NaN is treated as > all finite values by total_cmp,
    // so the NaN element is chosen as the "max".
    let _token = temp.sample(&logits);
}

#[test]
fn sample_mut_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![f32::NAN, 1.0, 2.0];
    let _token = temp.sample_mut(&mut logits);
}

#[test]
fn sample_all_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN; 4];
    let _token = temp.sample(&logits);
}

#[test]
fn sample_nan_with_neg_infinity() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY];
    let _token = temp.sample(&logits);
}

#[test]
fn dist_sample_does_not_panic_on_nan_logits() {
    let dist = Dist::new(42);
    let logits = vec![1.0, f32::NAN, 3.0];
    // Dist has its own sample() override; verify it doesn't panic either.
    let _token = dist.sample(&logits);
}
