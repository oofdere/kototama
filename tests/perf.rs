mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};

// ---------- SamplerChain::perf ----------
//
// The only existing test of `SamplerChain::perf` (`sampler_chain_perf` in
// tests/sampler.rs) calls `chain.perf()` and discards the result with
// `let _perf = ...`. None of the fields on the returned
// `llama_perf_sampler_data` are verified, so any regression that returned
// nonsense (e.g. dropped sampler-count accounting) would slip past CI.

#[test]
fn sampler_chain_perf_n_sample_starts_at_zero() {
    let chain = SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());
    assert_eq!(
        chain.perf().n_sample,
        0,
        "a freshly-built chain should report zero sampled tokens"
    );
}

#[test]
fn sampler_chain_perf_n_sample_increments_per_sample() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let chain = SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());

    assert_eq!(chain.perf().n_sample, 0);
    let _ = seq.sample(&chain);
    assert_eq!(
        chain.perf().n_sample,
        1,
        "n_sample should equal 1 after a single sample"
    );

    for _ in 0..4 {
        let _ = seq.sample(&chain);
    }
    assert_eq!(
        chain.perf().n_sample,
        5,
        "n_sample should equal 5 after five total samples"
    );
}

#[test]
fn sampler_chain_perf_t_sample_ms_is_finite_and_nonnegative() {
    let chain = SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());
    let p = chain.perf();
    assert!(
        p.t_sample_ms.is_finite() && p.t_sample_ms >= 0.0,
        "t_sample_ms should be a finite non-negative number, got {}",
        p.t_sample_ms
    );
}

#[test]
fn sampler_chain_perf_independent_across_chains() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);

    let chain_a = SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());
    let chain_b = SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());

    let _ = seq.sample(&chain_a);
    let _ = seq.sample(&chain_a);
    let _ = seq.sample(&chain_b);

    assert_eq!(chain_a.perf().n_sample, 2);
    assert_eq!(chain_b.perf().n_sample, 1);
}

// ---------- Context::perf ----------
//
// The only existing test of `Context::perf` (`perf_does_not_crash` in
// tests/context.rs) calls `ctx.perf()` and discards the result with
// `let _ = ...`. The shape of `llama_perf_context_data` — that `t_start_ms`
// is set at construction, that `n_eval` grows as tokens are decoded, and
// that the count is shared between `Context::clone`d handles backed by the
// same actor — is therefore unverified.

#[test]
fn context_perf_t_start_ms_is_set_at_construction() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let p = ctx.perf();
    assert!(
        p.t_start_ms.is_finite() && p.t_start_ms > 0.0,
        "t_start_ms should be set at context construction, got {}",
        p.t_start_ms
    );
}

#[test]
fn context_perf_t_load_ms_is_finite_and_nonnegative() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let p = ctx.perf();
    assert!(
        p.t_load_ms.is_finite() && p.t_load_ms >= 0.0,
        "t_load_ms should be a finite non-negative number, got {}",
        p.t_load_ms
    );
}

#[test]
fn context_perf_n_eval_grows_after_push() {
    // n_eval is upstream's count of decoded tokens. Each call to
    // `Sequence::push` runs `llama_decode` on a single-token batch, so the
    // counter must strictly grow. We don't pin an exact ratio because
    // llama.cpp splits the count across `n_p_eval` and `n_eval` based on
    // batch-size heuristics that are not part of this crate's contract.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let before = ctx.perf().n_eval;

    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    assert!(!tokens.is_empty(), "tokenizer should produce at least one token");
    seq.extend(&tokens);

    let after = ctx.perf().n_eval;
    assert!(
        after > before,
        "n_eval should strictly grow after pushing tokens \
         (before={before}, after={after})"
    );
}

#[test]
fn context_perf_n_eval_grows_monotonically_across_pushes() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("once upon a time", false, false);
    assert!(tokens.len() >= 2, "test prompt must yield at least 2 tokens");

    seq.push(tokens[0]);
    let after_first = ctx.perf().n_eval;
    seq.push(tokens[1]);
    let after_second = ctx.perf().n_eval;

    assert!(
        after_second > after_first,
        "n_eval should strictly increase across two pushes \
         (after_first={after_first}, after_second={after_second})"
    );
}

#[test]
fn context_perf_shared_across_clones() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let ctx_clone = ctx.clone();

    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);

    let p1 = ctx.perf();
    let p2 = ctx_clone.perf();
    assert_eq!(
        p1.n_eval, p2.n_eval,
        "cloned Context handles share the actor and must observe the same n_eval"
    );
    assert_eq!(
        p1.t_start_ms, p2.t_start_ms,
        "cloned Context handles must observe the same t_start_ms"
    );
}

// ---------- Sampling through a raw `Sampler` (no SamplerChain) ----------
//
// `LlamaSampler` is implemented for both `Sampler` and `SamplerChain`, but
// every existing sample test wraps the sampler in a `SamplerChain` first.
// The `impl LlamaSampler for Sampler` code path — i.e. passing a bare
// `Sampler` to `Sequence::sample` — has zero test coverage. Underlying
// `llama_sampler_sample` works on any `llama_sampler *`, not just chains,
// so this path is part of the public contract.

#[test]
fn sample_with_raw_greedy_sampler_matches_argmax() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("once upon", false, false);
    seq.extend(&tokens);

    let argmax = seq
        .logits()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i32)
        .unwrap();

    let sampler = Sampler::greedy();
    let token = seq.sample(&sampler);
    assert_eq!(
        token, argmax,
        "a bare greedy Sampler (no chain) should still match manual argmax"
    );
}

#[test]
fn sample_with_raw_sampler_yields_token_in_vocab_range() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let sampler = Sampler::greedy();
    let token = seq.sample(&sampler);
    assert!(
        token >= 0 && token < model.n_tokens(),
        "bare-Sampler-sampled token should be in vocab range, got {token}"
    );
}
