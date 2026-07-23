use rusty_llama::{
    Chain, Dist, Greedy, MinP, Sampler, Temperature, TopK, TopNSigma, TopP, Typical, Xtc,
};

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
    assert!(
        out[2].is_infinite() && out[2].is_sign_negative(),
        "3.0 masked"
    );
    assert!(
        out[3].is_infinite() && out[3].is_sign_negative(),
        "2.0 masked"
    );
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
    let distinct: std::collections::HashSet<i32> = (0..10).map(|_| d.sample(&logits)).collect();
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
    assert_eq!(TopK::new(1).name(), "TopK");
    assert_eq!(TopP::new(0.9, 1).name(), "TopP");
    assert_eq!(Typical::new(0.9, 1).name(), "Typical");
    assert_eq!(TopNSigma::new(1.0).name(), "TopNSigma");
    assert_eq!(Xtc::new(0.1, 0.1, 1, 1).name(), "Xtc");
    assert_eq!(Chain::new().name(), "Chain");
}

// ---------- TopK ----------

fn finite_count(out: &[f32]) -> usize {
    out.iter().filter(|x| x.is_finite()).count()
}

#[test]
fn top_k_masks_all_but_k() {
    let mut k = TopK::new(2);
    let out = k.apply(&[4.0, 3.0, 2.0, 1.0, 0.0]);
    assert_eq!(finite_count(&out), 2, "only top-2 survive");
    assert!(out[0].is_finite() && out[1].is_finite(), "top two survive");
    assert!(out[3].is_infinite() && out[3].is_sign_negative());
}

#[test]
fn top_k_keeps_ties_at_threshold() {
    // 5.0 appears twice; k=2 keeps both ties (>= threshold)
    let mut k = TopK::new(2);
    let out = k.apply(&[5.0, 4.0, 5.0, 1.0]);
    assert_eq!(finite_count(&out), 2);
    assert!(out[0].is_finite() && out[2].is_finite());
}

#[test]
fn top_k_non_positive_is_noop() {
    let mut k = TopK::new(0);
    let out = k.apply(&[4.0, 3.0, 2.0]);
    assert_eq!(finite_count(&out), 3);
}

#[test]
fn top_k_larger_than_vocab_is_noop() {
    let mut k = TopK::new(100);
    let out = k.apply(&[4.0, 3.0, 2.0]);
    assert_eq!(finite_count(&out), 3);
}

// ---------- TopP ----------

#[test]
fn top_p_one_is_noop() {
    let mut p = TopP::new(1.0, 1);
    let out = p.apply(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(finite_count(&out), 4);
}

#[test]
fn top_p_keeps_nucleus() {
    // heavily peaked: argmax alone exceeds p=0.9
    let mut p = TopP::new(0.9, 1);
    let out = p.apply(&[10.0, 1.0, 1.0, 1.0]);
    assert_eq!(finite_count(&out), 1, "only the max clears p=0.9");
    assert!(out[0].is_finite());
}

#[test]
fn top_p_respects_min_keep() {
    // p would keep only 1, but min_keep forces at least 3
    let mut p = TopP::new(0.1, 3);
    let out = p.apply(&[10.0, 1.0, 1.0, 1.0]);
    assert!(finite_count(&out) >= 3, "min_keep floor honored");
}

// ---------- Typical ----------

#[test]
fn typical_one_is_noop() {
    let mut t = Typical::new(1.0, 1);
    let out = t.apply(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(finite_count(&out), 4);
}

#[test]
fn typical_masks_some_tokens() {
    let mut t = Typical::new(0.5, 1);
    let out = t.apply(&[0.5, 1.0, 0.2, 3.0, 0.1]);
    assert!(
        finite_count(&out) < 5,
        "typical sampling should mask at least one token"
    );
    assert!(finite_count(&out) >= 1, "at least one token survives");
}

// ---------- TopNSigma ----------

#[test]
fn top_n_sigma_non_positive_is_noop() {
    let mut s = TopNSigma::new(0.0);
    let out = s.apply(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(finite_count(&out), 4);
}

#[test]
fn top_n_sigma_masks_outliers() {
    // a tight cluster plus a far-below outlier
    let mut s = TopNSigma::new(1.0);
    let out = s.apply(&[10.0, 9.9, 10.1, -50.0]);
    assert!(
        out[3].is_infinite() && out[3].is_sign_negative(),
        "outlier far below the cluster is masked"
    );
    assert!(out[0].is_finite() && out[1].is_finite() && out[2].is_finite());
}

#[test]
fn top_n_sigma_respects_already_masked() {
    // -inf entries must not corrupt the mean/std: the result for the
    // finite entries should match a run that omits the -inf slot entirely.
    let mut s = TopNSigma::new(1.0);
    let with_inf = s.apply(&[10.0, 9.9, 10.1, f32::NEG_INFINITY]);
    let without = s.apply(&[10.0, 9.9, 10.1]);
    assert_eq!(with_inf[0], without[0]);
    assert_eq!(with_inf[1], without[1]);
    assert_eq!(with_inf[2], without[2]);
    assert!(with_inf[3].is_infinite() && with_inf[3].is_sign_negative());
}

// ---------- XTC ----------

#[test]
fn xtc_inert_when_threshold_too_high() {
    // threshold > 0.5 disables the sampler
    let mut x = Xtc::new(1.0, 0.9, 1, 1);
    let out = x.apply(&[10.0, 9.0, 8.0, 7.0]);
    assert_eq!(finite_count(&out), 4);
}

#[test]
fn xtc_can_mask_top_token() {
    // probability 1.0 (always triggers), threshold 0.1: the top two tokens
    // clear the threshold and are masked (XTC removes the top choices), the
    // remaining tokens survive.
    let mut x = Xtc::new(1.0, 0.1, 1, 1);
    let out = x.apply(&[2.0, 2.0, 1.0, 0.0]);
    assert!(
        out[0].is_infinite() && out[0].is_sign_negative(),
        "top token masked"
    );
    assert!(
        out[1].is_infinite() && out[1].is_sign_negative(),
        "second token masked"
    );
    assert!(out[2].is_finite(), "third token survives");
}

#[test]
fn xtc_probability_zero_is_noop() {
    let mut x = Xtc::new(0.0, 0.1, 1, 1);
    let out = x.apply(&[10.0, 9.0, 8.0, 7.0]);
    assert_eq!(finite_count(&out), 4);
}

// ---------- Chain ----------

#[test]
fn chain_applies_transforms_in_order() {
    // top_k(1) keeps only the max, so temperature is irrelevant -> argmax
    let mut chain = Chain::new()
        .with(TopK::new(1))
        .with(Temperature::new(0.8))
        .with(Greedy::new());
    let token = chain.sample(&[1.0, 5.0, 2.0, 3.0]);
    assert_eq!(token, 1, "argmax survives the chain");
}

#[test]
fn chain_dist_returns_valid_token() {
    let mut chain = Chain::new()
        .with(TopP::new(0.9, 1))
        .with(Temperature::new(0.8))
        .with(Dist::new(7));
    let token = chain.sample(&[1.0, 2.0, 0.5, 3.0]);
    assert!((0..4).contains(&token));
}

#[test]
fn chain_apply_mut_composes() {
    let mut chain = Chain::new().with(TopK::new(2));
    let mut logits = vec![4.0, 3.0, 2.0, 1.0];
    chain.apply_mut(&mut logits);
    assert_eq!(finite_count(&logits), 2);
}

#[test]
fn chain_empty_is_greedy() {
    let mut chain = Chain::new();
    let token = chain.sample(&[0.1, 0.9, 0.5]);
    assert_eq!(token, 1);
}
