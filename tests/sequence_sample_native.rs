// Coverage for `Sequence::sample(&sampler)` against the native `Sampler`
// trait introduced in `baec429` ("native sampler wip"). The old C-wrapper
// `Sampler::greedy() / temp() / dist()` API was removed in that refactor,
// and the integration tests in `tests/integration.rs` and the
// `sequence_sample_*` tests in `tests/sequence.rs` still reference the
// deleted API and no longer compile. Open PR #113 covers the native
// `Sampler` trait and `Temperature`/`MinP`/`Dist` math in pure-Rust unit
// tests (no model, no actor). Open PR #119 covers `Sequence`'s logits
// cache invalidation. Neither exercises the `Sequence::sample` pipeline
// (`sampler.apply(cached_logits) -> sampler.sample(&transformed)`) with
// a real model's logits, so the end-to-end contract is currently
// unverified.

mod common;

use rusty_llama::{Context, Dist, MinP, Temperature};

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i32)
        .unwrap()
}

// ---------- Cache contract: sample() returns None without cached logits ----------

#[test]
fn sample_returns_none_when_no_logits_cached() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let seq = ctx.sequence().unwrap();
    assert!(
        seq.logits().is_none(),
        "precondition: fresh sequence has no cached logits"
    );
    let temp = Temperature::new(1.0);
    assert_eq!(
        seq.sample(&temp),
        None,
        "sample() must return None when no logits are cached"
    );
}

#[test]
fn sample_returns_none_after_pop_clears_cache() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some());

    let _ = seq.pop();
    let temp = Temperature::new(1.0);
    assert_eq!(
        seq.sample(&temp),
        None,
        "sample() must return None after pop() invalidates the cache"
    );
}

// ---------- Temperature: argmax preservation ----------

#[test]
fn sample_with_temperature_zero_matches_argmax_of_logits() {
    // `Temperature::apply_mut` short-circuits on temp == 0.0, leaving logits
    // untouched. `Sequence::sample` then calls the trait-default
    // `sampler.sample(transformed)` which returns the argmax of the
    // (unchanged) logits — i.e. exactly the argmax of the cached logits.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("Once upon a time", false, false);
    seq.extend(&tokens);

    let expected = argmax(seq.logits().unwrap());
    let temp = Temperature::new(0.0);
    assert_eq!(seq.sample(&temp), Some(expected));
}

#[test]
fn sample_with_temperature_one_matches_argmax_of_logits() {
    // temp = 1.0 multiplies every logit by 1, so argmax is unchanged and
    // the sampled token must equal the argmax of the cached logits.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let expected = argmax(seq.logits().unwrap());
    let temp = Temperature::new(1.0);
    assert_eq!(seq.sample(&temp), Some(expected));
}

#[test]
fn sample_with_positive_temperature_preserves_argmax() {
    // Any temp > 0 scales every logit by the same positive constant, so
    // the post-transform argmax matches the pre-transform argmax. Pinning
    // this against a real model's logits prevents a future refactor of
    // `Sequence::sample` from sampling pre- or non-transform logits by
    // mistake.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("the cat sat", false, false);
    seq.extend(&tokens);

    let expected = argmax(seq.logits().unwrap());
    for temp_val in [0.25_f32, 0.5, 0.8, 2.0, 10.0] {
        let temp = Temperature::new(temp_val);
        assert_eq!(
            seq.sample(&temp),
            Some(expected),
            "temp = {temp_val} must preserve argmax of logits"
        );
    }
}

#[test]
fn sample_with_temperature_does_not_mutate_cached_logits() {
    // `Sequence::sample` takes `&self`. Even though `Temperature::apply_mut`
    // mutates its slice, `Sequence::sample` calls `sampler.apply(logits)`
    // which clones the slice first. The cached logits must be untouched
    // afterwards.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let before: Vec<f32> = seq.logits().unwrap().to_vec();
    let temp = Temperature::new(0.5);
    let _ = seq.sample(&temp);
    let after: Vec<f32> = seq.logits().unwrap().to_vec();
    assert_eq!(before, after, "cached logits must not be mutated by sample()");
}

// ---------- MinP ----------

#[test]
fn sample_with_minp_returns_valid_vocab_token() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let minp = MinP::new(0.05, 1);
    let token = seq.sample(&minp).expect("logits cached, sample must yield");
    assert!(
        token >= 0 && token < model.n_tokens(),
        "MinP-sampled token {token} must lie in [0, {})",
        model.n_tokens()
    );
}

#[test]
fn sample_with_minp_p_one_picks_argmax() {
    // MinP(p=1.0, min_keep=0) sets thresh = logit_max + ln(1) = logit_max,
    // so only the single max logit survives. `sample` (default impl) then
    // returns the argmax of the surviving logits, which must equal the
    // original argmax.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let expected = argmax(seq.logits().unwrap());
    let minp = MinP::new(1.0, 0);
    assert_eq!(seq.sample(&minp), Some(expected));
}

// ---------- Dist ----------

#[test]
fn sample_with_dist_returns_valid_vocab_token() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let dist = Dist::new(42);
    let token = seq.sample(&dist).expect("logits cached, sample must yield");
    assert!(
        token >= 0 && token < model.n_tokens(),
        "Dist-sampled token {token} must lie in [0, {})",
        model.n_tokens()
    );
}

#[test]
fn sample_with_dist_same_seed_reproducible_across_sequences() {
    // Two sequences fed the same prompt produce identical cached logits.
    // Two `Dist`s seeded identically must then sample the same token from
    // those logits — this is the reproducibility contract that makes
    // greedy-replacing seeds useful for regression testing.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();

    let tokens = model.tokenize("the cat sat", false, false);

    let mut seq_a = ctx.sequence().unwrap();
    seq_a.extend(&tokens);
    let mut seq_b = ctx.sequence().unwrap();
    seq_b.extend(&tokens);

    assert_eq!(
        seq_a.logits().unwrap(),
        seq_b.logits().unwrap(),
        "precondition: same prompt -> same cached logits"
    );

    let dist_a = Dist::new(7);
    let dist_b = Dist::new(7);
    assert_eq!(seq_a.sample(&dist_a), seq_b.sample(&dist_b));
}

// ---------- &self contract ----------

#[test]
fn sample_takes_shared_borrow_compatible_with_logits_borrow() {
    // `Sequence::sample` takes `&self`. It must be callable while another
    // shared borrow (e.g. `seq.logits()`) is held — this is what makes
    // "inspect logits then sample" ergonomic and is the stated reason
    // `sample()` is `&self` rather than `&mut self`.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let logits_borrow = seq.logits().expect("logits cached after push");
    let temp = Temperature::new(1.0);
    let token = seq.sample(&temp).expect("sample must succeed");
    // Use both borrows after the sample call to prove they coexist.
    assert_eq!(logits_borrow.len(), model.n_tokens() as usize);
    assert!(token >= 0 && token < model.n_tokens());
}
