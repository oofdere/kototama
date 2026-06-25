use rusty_llama::{Dist, Greedy, MinP, Sampler, Temperature};

// ---------- Greedy ----------

#[test]
fn greedy_picks_argmax() {
    let mut g = Greedy::new();
    let logits = [0.1, 0.9, 0.5, 0.2];
    assert_eq!(g.sample(&logits), 1);
}

#[test]
fn greedy_tiebreak_is_last() {
    // max_by returns the last maximum on ties — pin the behavior
    let mut g = Greedy::new();
    let logits = [5.0, 5.0, 5.0];
    assert_eq!(g.sample(&logits), 2);
}

// ---------- Temperature ----------

#[test]
fn temperature_zero_is_identity() {
    let mut t = Temperature::new(0.0);
    let logits = [1.0, 3.0, 2.0];
    let out = t.apply(&logits);
    assert_eq!(out, logits);
}

#[test]
fn temperature_scales_logits() {
    let mut t = Temperature::new(2.0);
    let out = t.apply(&[1.0, 2.0, 4.0]);
    assert!((out[0] - 0.5).abs() < 1e-6);
    assert!((out[1] - 1.0).abs() < 1e-6);
    assert!((out[2] - 2.0).abs() < 1e-6);
}

#[test]
fn temperature_preserves_argmax() {
    let mut t = Temperature::new(0.8);
    let logits = [0.1, 0.9, 0.5];
    assert_eq!(t.sample(&logits), 1);
}

// ---------- MinP ----------

#[test]
fn min_p_masks_below_threshold() {
    let mut m = MinP::new(0.5, 0);
    let out = m.apply(&[4.0, 3.5, 3.0, 2.0]);
    assert!(out[0].is_finite(), "max survives");
    assert!(out[1].is_finite(), "3.5 >= thresh survives");
    assert!(out[2].is_infinite() && out[2].is_sign_negative(), "3.0 masked");
    assert!(out[3].is_infinite() && out[3].is_sign_negative(), "2.0 masked");
}

#[test]
fn min_p_min_keep_floor_forces_survivors() {
    // p=0.01 alone would keep only the max; min_keep forces 4
    let mut m = MinP::new(0.01, 4);
    let out = m.apply(&[10.0, 5.0, 4.0, 1.0, 0.0]);
    let finite = out.iter().filter(|x| x.is_finite()).count();
    assert_eq!(finite, 4);
}

#[test]
fn min_p_one_keeps_only_the_max() {
    let mut m = MinP::new(1.0, 0);
    let out = m.apply(&[1.0, 3.0, 3.0, 2.0]);
    assert!(out[1].is_finite() && out[2].is_finite(), "max(es) survive");
    assert!(out[0].is_infinite() && out[0].is_sign_negative());
    assert!(out[3].is_infinite() && out[3].is_sign_negative());
}

// ---------- Dist ----------

#[test]
fn dist_returns_valid_token() {
    let mut d = Dist::new(42);
    let token = d.sample(&[0.1, 0.5, 0.3, 0.2]);
    assert!((0..4).contains(&token));
}

#[test]
fn dist_is_deterministic_for_same_seed() {
    let logits = vec![1.0, 2.0, 0.5, 3.0, 1.5];
    let mut a = Dist::new(99);
    let mut b = Dist::new(99);
    assert_eq!(a.sample(&logits), b.sample(&logits));
}

#[test]
fn dist_advances_state_across_calls() {
    let logits = vec![1.0, 2.0, 0.5, 3.0, 1.5];
    let mut d = Dist::new(7);
    let distinct: std::collections::HashSet<i32> =
        (0..10).map(|_| d.sample(&logits)).collect();
    assert!(
        distinct.len() > 1,
        "RNG should advance, producing varied draws"
    );
}

#[test]
fn dist_never_picks_masked_tokens() {
    let mut d = Dist::new(1);
    let logits = vec![1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.5];
    for _ in 0..20 {
        let t = d.sample(&logits);
        assert!(t == 0 || t == 3, "masked token {t} picked");
    }
}

// ---------- name() ----------

#[test]
fn name_returns_short_type_name() {
    assert_eq!(Temperature::new(1.0).name(), "Temperature");
    assert_eq!(MinP::new(0.1, 1).name(), "MinP");
    assert_eq!(Greedy::new().name(), "Greedy");
    assert_eq!(Dist::new(0).name(), "Dist");
}
