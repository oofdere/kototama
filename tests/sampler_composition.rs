//! Coverage for the manual sampler-composition pattern demonstrated in
//! `examples/simple_chat.rs`:
//!
//! ```ignore
//! let l = minp.apply(logits);   // mask low-probability tokens to -inf
//! let l = temp.apply(&l);       // rescale the surviving logits
//! let token = dist.sample(&l);  // stochastic softmax draw
//! ```
//!
//! Each sampler is unit-tested in isolation (`tests/sampler_trait.rs` in
//! PR #113, `tests/sampler_dist_sample_mut.rs` in PR #134), but the
//! *composition* used by the shipped example is untested. That leaves
//! several contracts unpinned:
//!
//! - `MinP` writes `f32::NEG_INFINITY` to masked slots. `Temperature`
//!   must not turn those into `NaN` (`(-inf) * inv_temp` is well-defined
//!   for positive `inv_temp` — it stays `-inf`), so `Dist::sample`
//!   downstream still sees zero softmax weight there.
//! - The composed pipeline is deterministic under a seeded `Dist`, so
//!   the example is reproducible when the user pins the seed.
//! - The pipeline never returns a masked token id, regardless of the
//!   temperature applied between masking and sampling.
//!
//! A silent regression in any of those (e.g. `Temperature::apply_mut`
//! adopting a different formula that produces `NaN` on `-inf` inputs, or
//! `Dist::sample`'s tail-fallback drifting to include masked indices)
//! would let the isolated tests still pass while breaking the shipped
//! example. These tests catch that class of regression.

use rusty_llama::{Dist, MinP, Sampler, Temperature};

// A logit distribution with one clearly dominant index, a couple of
// mid-pack candidates, and a long tail — the shape MinP is designed to
// prune. Index 2 is the dominant one.
fn baseline_logits() -> Vec<f32> {
    vec![0.5, 1.0, 5.0, 4.5, -3.0, -8.0, -12.0, -20.0]
}

// The example's exact composition, factored out so every test uses the
// same wiring as `examples/simple_chat.rs`.
fn compose_sample(minp: &MinP, temp: &Temperature, dist: &Dist, logits: &[f32]) -> i32 {
    let l = minp.apply(logits);
    let l = temp.apply(&l);
    dist.sample(&l)
}

#[test]
fn composed_pipeline_returns_valid_index() {
    let minp = MinP::new(0.05, 1);
    let temp = Temperature::new(0.8);
    let dist = Dist::new(42);
    let logits = baseline_logits();
    let token = compose_sample(&minp, &temp, &dist, &logits);
    assert!(
        token >= 0 && (token as usize) < logits.len(),
        "composed sampler must return a valid index into the logits, got {token}"
    );
}

#[test]
fn composed_pipeline_never_returns_a_masked_index() {
    // MinP prunes everything below `max + ln(p)`. With p = 0.5 and the
    // baseline (max = 5.0), the threshold is ~4.307, so indices 0, 1, 4,
    // 5, 6, 7 must all be masked to -inf. Temperature scales but cannot
    // resurrect -inf; Dist::sample must never draw a masked index.
    let minp = MinP::new(0.5, 1);
    let temp = Temperature::new(0.8);
    let dist = Dist::new(42);
    let logits = baseline_logits();

    // Identify which indices MinP masks up-front so the assertion is
    // independent of the exact threshold calculation.
    let masked = minp.apply(&logits);
    let masked_indices: Vec<usize> = masked
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_finite())
        .map(|(i, _)| i)
        .collect();
    assert!(
        !masked_indices.is_empty(),
        "test invariant: MinP must actually mask something on this input"
    );

    for _ in 0..500 {
        let token = compose_sample(&minp, &temp, &dist, &logits);
        assert!(
            !masked_indices.contains(&(token as usize)),
            "composed sampler drew masked index {token}; masked = {masked_indices:?}"
        );
    }
}

#[test]
fn composed_pipeline_is_deterministic_under_seeded_dist() {
    // The shipped example seeds `Dist` with `LLAMA_DEFAULT_SEED`; user
    // code that pins the seed expects reproducible token streams.
    let logits = baseline_logits();
    let make = || {
        (
            MinP::new(0.05, 1),
            Temperature::new(0.8),
            Dist::new(0xC0FFEE),
        )
    };

    let (m1, t1, d1) = make();
    let seq1: Vec<i32> = (0..32).map(|_| compose_sample(&m1, &t1, &d1, &logits)).collect();
    let (m2, t2, d2) = make();
    let seq2: Vec<i32> = (0..32).map(|_| compose_sample(&m2, &t2, &d2, &logits)).collect();

    assert_eq!(
        seq1, seq2,
        "same-seeded pipelines must produce identical token streams"
    );
}

#[test]
fn composed_pipeline_zero_temperature_still_masks_correctly() {
    // Temperature::apply_mut short-circuits on temp == 0.0 to avoid a
    // division by zero. That short-circuit must not accidentally undo
    // MinP's -inf mask (which would let Dist::sample draw masked
    // indices). This test would fail if Temperature ever chose to
    // "reset" its input on the zero path.
    let minp = MinP::new(0.5, 1);
    let temp = Temperature::new(0.0);
    let dist = Dist::new(99);
    let logits = baseline_logits();

    let masked = minp.apply(&logits);
    let masked_indices: Vec<usize> = masked
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_finite())
        .map(|(i, _)| i)
        .collect();
    assert!(!masked_indices.is_empty());

    for _ in 0..200 {
        let token = compose_sample(&minp, &temp, &dist, &logits);
        assert!(
            !masked_indices.contains(&(token as usize)),
            "temp = 0 must preserve MinP's mask; drew {token}"
        );
    }
}

#[test]
fn composed_pipeline_produces_no_nan_between_stages() {
    // MinP writes -inf to masked slots. If Temperature ever produced
    // NaN from `-inf * inv_temp`, `Dist::sample`'s `.fold(NEG_INFINITY,
    // f32::max)` would silently poison the softmax normalisation and
    // Dist's tail-fallback (`exps.len() - 1`) would be returned every
    // call. Guard the intermediate stage.
    let minp = MinP::new(0.5, 1);
    let temp = Temperature::new(0.8);
    let logits = baseline_logits();

    let after_minp = minp.apply(&logits);
    let after_temp = temp.apply(&after_minp);

    for (i, v) in after_temp.iter().enumerate() {
        assert!(
            !v.is_nan(),
            "composed pipeline produced NaN at index {i}: after_minp={after_minp:?}, after_temp={after_temp:?}"
        );
    }
}

#[test]
fn composed_pipeline_favours_dominant_index() {
    // With the baseline logits, index 2 sits ~0.5 nats above index 3
    // and ~4+ nats above every survivor. Under a moderate temperature
    // the softmax weight on index 2 is >55%, so a decisive majority of
    // draws should land there. This pins the "greedy-ish" character of
    // the pipeline; a bug that scrambled the softmax weights (e.g. a
    // sign flip in Temperature or a broken exp in Dist) would push the
    // hit rate toward the uniform 1/8 baseline and trip this bound.
    let minp = MinP::new(0.05, 1);
    let temp = Temperature::new(1.0);
    let dist = Dist::new(2024);
    let logits = baseline_logits();

    let n = 500;
    let hits = (0..n)
        .filter(|_| compose_sample(&minp, &temp, &dist, &logits) == 2)
        .count();
    assert!(
        hits * 2 >= n,
        "dominant index should win a majority of draws, got {hits}/{n}"
    );
}
