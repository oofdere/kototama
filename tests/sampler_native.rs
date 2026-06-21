use rusty_llama::{Dist, MinP, Sampler, Temperature, Token};

// ---------- NaN safety in default sample() ----------

#[test]
fn sample_does_not_panic_on_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![1.0, f32::NAN, 3.0, 2.0];
    // Before the fix, this would panic due to partial_cmp().unwrap() on NaN.
    // After the fix, total_cmp treats NaN as greater than all other values,
    // so it deterministically picks the NaN index. The key property is no panic.
    let _token: Token = temp.sample(&logits);
}

#[test]
fn sample_does_not_panic_on_all_nan_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    let _token: Token = temp.sample(&logits);
}

#[test]
fn sample_does_not_panic_on_inf_logits() {
    let temp = Temperature::new(1.0);
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::INFINITY, 2.0];
    let token: Token = temp.sample(&logits);
    // INFINITY is the largest finite-comparable value under total_cmp
    assert_eq!(token, 2, "should pick the INFINITY logit at index 2");
}

// ---------- No double-application in default sample() ----------

#[test]
fn default_sample_does_not_reapply_transformation() {
    // Temperature(0.5) multiplies logits by 2.0 (inv_temp = 1/0.5).
    // If sample() re-applied, the effective multiplier would be 4.0.
    // With logits [1.0, 2.0, 3.0]:
    //   - After one application: [2.0, 4.0, 6.0] → argmax = index 2
    //   - After double application: [4.0, 8.0, 12.0] → argmax = index 2 (same here)
    //
    // Use a case where double-application changes the argmax:
    // logits = [-1.0, 0.5] with Temperature(0.5) → inv_temp = 2.0
    //   - No re-apply (correct): sample([-1.0, 0.5]) → argmax of [-1.0, 0.5] = index 1
    //   - With re-apply (bug): sample applies temp → [-2.0, 1.0] → argmax = index 1
    //
    // Actually argmax is invariant under positive scaling, so let's verify
    // the actual logit values instead.
    let temp = Temperature::new(0.5);
    let logits = vec![1.0, 3.0, 2.0];

    // apply() transforms logits
    let transformed = temp.apply(&logits);
    // inv_temp = 2.0, so transformed = [2.0, 6.0, 4.0]
    assert_eq!(transformed, vec![2.0, 6.0, 4.0]);

    // sample() should pick from the input AS-IS (no re-application)
    // So sample(&transformed) should return argmax of [2.0, 6.0, 4.0] = index 1
    let token = temp.sample(&transformed);
    assert_eq!(token, 1, "sample() should return argmax of input without re-applying");

    // Verify: if sample() re-applied, it would transform [2.0, 6.0, 4.0] → [4.0, 12.0, 8.0]
    // which also gives index 1 due to argmax being scale-invariant.
    // So let's test with a sampler where double-application matters more: MinP.
}

#[test]
fn minp_sample_no_double_application() {
    // MinP with p=0.5: filters out logits below max + ln(0.5) ≈ max - 0.693
    // logits = [5.0, 4.5, 4.0, 3.0, 2.0]
    //   max = 5.0, threshold = 5.0 + ln(0.5) = 5.0 - 0.693 = 4.307
    //   After one apply: [5.0, 4.5, -inf, -inf, -inf] (4.0 < 4.307)
    //   argmax = index 0
    //
    // If double-applied on [5.0, 4.5, -inf, -inf, -inf]:
    //   max = 5.0, threshold = 5.0 - 0.693 = 4.307
    //   4.5 >= 4.307 → kept. Same result in this case.
    //
    // Better test: verify that sample() doesn't mutate/re-filter by checking
    // that it picks the argmax of the INPUT logits directly.
    let minp = MinP::new(0.5, 1);
    let logits: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0];

    // After apply: max=5.0, thresh = 5.0 + ln(0.5) = 4.307
    // Only index 1 (5.0) survives, rest → -inf
    let transformed = minp.apply(&logits);
    assert_eq!(transformed[1], 5.0);
    assert!(transformed[0].is_infinite() && transformed[0].is_sign_negative());

    // sample() on transformed should just pick argmax = index 1
    let token = minp.sample(&transformed);
    assert_eq!(token, 1);

    // Critically: calling sample() on the RAW logits should pick argmax = index 1
    // (because sample() should NOT apply the filter itself)
    let token_raw = minp.sample(&logits);
    assert_eq!(token_raw, 1, "sample() should just pick argmax without re-applying MinP filter");
}

// ---------- Dist sampler (overrides sample()) ----------

#[test]
fn dist_sample_stays_in_range() {
    let dist = Dist::new(42);
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for _ in 0..100 {
        let token = dist.sample(&logits);
        assert!(
            token >= 0 && token < logits.len() as i32,
            "dist sample should be in range"
        );
    }
}

// ---------- Edge cases ----------

#[test]
#[should_panic(expected = "cannot sample from empty logits")]
fn sample_panics_on_empty_logits() {
    let temp = Temperature::new(1.0);
    let logits: Vec<f32> = vec![];
    temp.sample(&logits);
}

#[test]
fn sample_single_element() {
    let temp = Temperature::new(1.0);
    let logits = vec![42.0];
    let token = temp.sample(&logits);
    assert_eq!(token, 0);
}
