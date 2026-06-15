//! Unit tests for the native Sampler trait introduced in `baec429`
//! ("native sampler wip"). Covers the trait's default methods and the
//! mathematical / edge-case behaviour of `Temperature`, `MinP`, and `Dist`.
//!
//! NaN-handling is covered separately by `tests/sampler_nan.rs` (PR #109).

use rusty_llama::{Dist, MinP, Sampler, Temperature};

// ---------- Sampler trait defaults ----------

#[test]
fn name_returns_unqualified_type_name() {
    // Default `name()` strips the module path and returns just the type ident.
    assert_eq!(Temperature::new(1.0).name(), "Temperature");
    assert_eq!(MinP::new(0.5, 1).name(), "MinP");
    assert_eq!(Dist::new(0).name(), "Dist");
}

#[test]
fn apply_does_not_mutate_input() {
    let temp = Temperature::new(0.5);
    let input = vec![1.0, 2.0, 3.0];
    let _ = temp.apply(&input);
    assert_eq!(input, vec![1.0, 2.0, 3.0]);
}

#[test]
fn apply_returns_same_result_as_apply_mut() {
    let temp = Temperature::new(0.5);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let from_apply = temp.apply(&input);
    let mut copy = input.clone();
    temp.apply_mut(&mut copy);
    assert_eq!(from_apply, copy);
}

#[test]
fn sample_default_returns_argmax_after_transform() {
    // Temperature scaling preserves ordering for temp > 0, so argmax is index 1.
    let temp = Temperature::new(0.5);
    let logits = vec![1.0, 5.0, 3.0, 2.0];
    assert_eq!(temp.sample(&logits), 1);
}

#[test]
fn sample_default_does_not_mutate_input() {
    let temp = Temperature::new(0.5);
    let input = vec![1.0, 5.0, 3.0];
    let _ = temp.sample(&input);
    assert_eq!(input, vec![1.0, 5.0, 3.0]);
}

#[test]
fn sample_mut_applies_in_place_and_picks_argmax() {
    let temp = Temperature::new(2.0);
    let mut logits = vec![1.0, 5.0, 3.0];
    let token = temp.sample_mut(&mut logits);
    assert_eq!(token, 1);
    // sample_mut must apply the transform in place.
    assert_eq!(logits, vec![0.5, 2.5, 1.5]);
}

// ---------- Temperature ----------

#[test]
fn temperature_zero_is_no_op() {
    // Documented edge case: temp = 0.0 avoids div-by-zero by short-circuiting.
    let temp = Temperature::new(0.0);
    let mut logits = vec![1.0, -1.0, 3.5, 0.0];
    let before = logits.clone();
    temp.apply_mut(&mut logits);
    assert_eq!(logits, before, "temp = 0.0 must leave logits unchanged");
}

#[test]
fn temperature_one_is_identity() {
    let temp = Temperature::new(1.0);
    let mut logits = vec![1.0, -2.5, 3.5];
    temp.apply_mut(&mut logits);
    assert_eq!(logits, vec![1.0, -2.5, 3.5]);
}

#[test]
fn temperature_below_one_sharpens() {
    // temp < 1 multiplies by inv_temp > 1, so magnitudes grow.
    let temp = Temperature::new(0.5);
    let mut logits = vec![1.0, 2.0, -1.0];
    temp.apply_mut(&mut logits);
    assert_eq!(logits, vec![2.0, 4.0, -2.0]);
}

#[test]
fn temperature_above_one_softens() {
    // temp > 1 multiplies by inv_temp < 1, so magnitudes shrink.
    let temp = Temperature::new(2.0);
    let mut logits = vec![2.0, 4.0, -2.0];
    temp.apply_mut(&mut logits);
    assert_eq!(logits, vec![1.0, 2.0, -1.0]);
}

#[test]
fn temperature_preserves_argmax_for_positive_temp() {
    // Any temp > 0 scales every logit by the same positive constant, so
    // the argmax must be preserved.
    let temp = Temperature::new(0.25);
    let logits = vec![0.1, 5.0, 1.0, 4.9];
    let after = temp.apply(&logits);
    let argmax_after = after
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(argmax_after, 1);
}

// ---------- MinP ----------

#[test]
fn minp_noop_when_logits_at_most_min_keep() {
    // If logits.len() <= min_keep, MinP must not touch the slice.
    let m = MinP::new(0.5, 4);
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    let before = logits.clone();
    m.apply_mut(&mut logits);
    assert_eq!(logits, before);
}

#[test]
fn minp_p_one_keeps_only_max() {
    // thresh = logit_max + ln(1) = logit_max; without a min_keep floor,
    // only the single maximum logit survives.
    let m = MinP::new(1.0, 0);
    let mut logits = vec![1.0, 5.0, 3.0, 2.0];
    m.apply_mut(&mut logits);
    assert_eq!(logits[1], 5.0);
    assert_eq!(logits[0], f32::NEG_INFINITY);
    assert_eq!(logits[2], f32::NEG_INFINITY);
    assert_eq!(logits[3], f32::NEG_INFINITY);
}

#[test]
fn minp_threshold_prunes_low_logits() {
    // p = 0.5, max = 0.0, thresh = ln(0.5) ≈ -0.693
    // Logits >= -0.693 survive, lower ones go to -inf.
    let m = MinP::new(0.5, 0);
    let mut logits = vec![0.0, -0.5, -1.0, -2.0];
    m.apply_mut(&mut logits);
    assert_eq!(logits[0], 0.0);
    assert_eq!(logits[1], -0.5);
    assert_eq!(logits[2], f32::NEG_INFINITY);
    assert_eq!(logits[3], f32::NEG_INFINITY);
}

#[test]
fn minp_min_keep_floor_overrides_threshold() {
    // p = 1.0 would normally keep only the single max; min_keep = 3
    // forces the top 3 logits to survive regardless.
    let m = MinP::new(1.0, 3);
    let mut logits = vec![1.0, 5.0, 3.0, 2.0, 0.5];
    m.apply_mut(&mut logits);
    assert_eq!(logits[1], 5.0);
    assert_eq!(logits[2], 3.0);
    assert_eq!(logits[3], 2.0);
    assert_eq!(logits[0], f32::NEG_INFINITY);
    assert_eq!(logits[4], f32::NEG_INFINITY);
}

#[test]
fn minp_preserves_kept_logit_values() {
    // Surviving logits must be unchanged in magnitude — MinP is a mask,
    // not a rescaler.
    let m = MinP::new(0.5, 0);
    let original = vec![0.0, -0.3, -0.5, -2.0];
    let mut logits = original.clone();
    m.apply_mut(&mut logits);
    for (i, v) in logits.iter().enumerate() {
        if v.is_finite() {
            assert_eq!(*v, original[i], "kept logits must not be modified");
        }
    }
}

// ---------- Dist ----------

#[test]
fn dist_apply_mut_is_noop() {
    let d = Dist::new(42);
    let mut logits = vec![1.0, 2.0, 3.0, -1.0];
    let before = logits.clone();
    d.apply_mut(&mut logits);
    assert_eq!(logits, before, "Dist::apply_mut is a no-op");
}

#[test]
fn dist_sample_returns_valid_index() {
    let d = Dist::new(7);
    let logits = vec![0.1, 0.5, 0.3, 0.1];
    let token = d.sample(&logits);
    assert!(token >= 0 && (token as usize) < logits.len());
}

#[test]
fn dist_same_seed_same_sequence() {
    // Documented: Dist::new(seed) reproducibly restarts the stream.
    let a = Dist::new(123);
    let b = Dist::new(123);
    let logits = vec![1.0, 2.0, 3.0, 0.5];
    for _ in 0..10 {
        assert_eq!(a.sample(&logits), b.sample(&logits));
    }
}

#[test]
fn dist_different_seeds_diverge() {
    // With different seeds over a 4-way distribution, 20 draws are very
    // unlikely to fully agree.
    let a = Dist::new(1);
    let b = Dist::new(2);
    let logits = vec![1.0, 2.0, 3.0, 0.5];
    let mut differ = false;
    for _ in 0..20 {
        if a.sample(&logits) != b.sample(&logits) {
            differ = true;
            break;
        }
    }
    assert!(differ, "different seeds should produce different streams");
}

#[test]
fn dist_dominant_logit_wins_majority() {
    // A logit ~14 nats above the others has softmax weight > 99.999%,
    // so it should win nearly every draw.
    let d = Dist::new(99);
    let logits = vec![0.0, 14.0, 0.0, 0.0];
    let n = 200;
    let hits = (0..n).filter(|_| d.sample(&logits) == 1).count();
    assert!(
        hits >= 195,
        "dominant token should win nearly every draw, got {hits}/{n}"
    );
}

#[test]
fn dist_neg_infinity_logit_never_sampled() {
    // exp(-inf) = 0, so an index with -inf logit has zero weight and must
    // never be drawn. This is the contract MinP relies on.
    let d = Dist::new(42);
    let logits = vec![1.0, f32::NEG_INFINITY, 1.0, 1.0];
    for _ in 0..200 {
        let token = d.sample(&logits);
        assert_ne!(token, 1, "an index with -inf logit must never be sampled");
    }
}
