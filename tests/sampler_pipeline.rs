mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};

// End-to-end coverage for the safe-FFI `Sampler::*` constructors that today
// have only init-only smoke tests. tests/sampler.rs verifies that each of
// `min_p`, `adaptive_p`, `mirostat`, `mirostat_v2`, `penalties`, `temp_ext`,
// `typical`, `xtc`, and `top_n_sigma` can be *constructed* without crashing,
// but nothing exercises them through `Sequence::sample` — i.e. nothing checks
// that `llama_sampler_sample` actually accepts these samplers and that the
// returned token id stays in-vocab.
//
// A regression that swapped an FFI symbol (e.g. so `llama_sampler_init_xtc`
// returned a null pointer, or `llama_sampler_init_mirostat_v2` returned a
// dangling one) would pass the existing init-only tests but segfault the
// first time a real caller chained the sampler into a `seq.sample(...)` loop.
// The tests below pin the public contract that every one of these samplers
// must produce a valid token through the canonical pipeline.

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

fn seeded_seq(
    model: &rusty_llama::Model,
    params: &rusty_llama::ContextParams,
) -> (Context, rusty_llama::Sequence) {
    let ctx = Context::new(model, params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    (ctx, seq)
}

fn assert_token_in_vocab(token: i32, model: &rusty_llama::Model, sampler_name: &str) {
    assert!(
        token >= 0 && token < model.n_tokens(),
        "{sampler_name}-sampled token {token} out of vocab range [0, {})",
        model.n_tokens()
    );
}

// ---------- Transformer samplers chained with a picker ----------
//
// These samplers transform the logits but don't pick a token, so they must be
// paired with a picker (greedy or dist) at the end of the chain. The
// assertion is the same in every case: the chain produces an in-vocab token.

#[test]
fn min_p_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::min_p(0.05, 1))
        .add(Sampler::greedy());
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "min_p");
}

#[test]
fn penalties_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::penalties(64, 1.1, 0.0, 0.0))
        .add(Sampler::greedy());
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "penalties");
}

#[test]
fn temp_ext_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::temp_ext(0.8, 0.1, 1.0))
        .add(Sampler::dist(42));
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "temp_ext");
}

#[test]
fn typical_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::typical(0.9, 1))
        .add(Sampler::dist(42));
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "typical");
}

#[test]
fn xtc_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::xtc(0.1, 0.1, 1, 42))
        .add(Sampler::dist(42));
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "xtc");
}

#[test]
fn top_n_sigma_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::top_n_sigma(1.0))
        .add(Sampler::dist(42));
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "top_n_sigma");
}

// ---------- Self-contained sampler-pickers ----------
//
// Mirostat (v1 and v2) and adaptive_p own both a logit transform and an RNG,
// so they can sit alone in a chain and pick a token directly.

#[test]
fn mirostat_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::mirostat(model.n_tokens(), 42, 5.0, 0.1, 100));
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "mirostat");
}

#[test]
fn mirostat_v2_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::mirostat_v2(42, 5.0, 0.1));
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "mirostat_v2");
}

#[test]
fn adaptive_p_in_chain_samples_valid_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::adaptive_p(0.1, 0.9, 42));
    let token = seq.sample(&chain);
    assert_token_in_vocab(token, &model, "adaptive_p");
}

// ---------- Seed actually threads through to dist() ----------
//
// `seeded_dist_sample_push_loop_is_deterministic` (in tests/sample_push_loop.rs
// from PR #87) confirms that the *same* seed reproduces the same token
// sequence — that's necessary but not sufficient. A regression that ignored
// the seed parameter (e.g. always using `0` or pulling from a global RNG)
// would also produce reproducible runs, just identical ones across seeds.
//
// The test below pins the *negative* property: two distinct seeds must, on
// the same prompt and sampler shape, eventually diverge. Without this, the
// `seed` parameter could rot into a no-op without any existing test failing.

#[test]
fn dist_different_seeds_produce_different_sequences() {
    let (model, params) = setup();
    let prompt = model.tokenize("Once upon a time", true, false);

    let run = |seed: u32| -> Vec<i32> {
        let ctx = Context::new(&model, &params).unwrap();
        let mut seq = ctx.sequence().unwrap();
        seq.extend(&prompt);
        // A non-trivial transformer in front of dist so the sampler has more
        // than one plausible token at each step — a degenerate distribution
        // (e.g. one logit dominating) would mask a seed regression by always
        // returning the argmax regardless of RNG state.
        let chain = SamplerChain::new(&SamplerChainParams::new())
            .add(Sampler::temp(1.5))
            .add(Sampler::top_k(40))
            .add(Sampler::dist(seed));
        let mut out = Vec::with_capacity(16);
        for _ in 0..16 {
            let t = seq.sample(&chain);
            if model.is_eog(t) {
                break;
            }
            out.push(t);
            seq.push(t);
        }
        out
    };

    let a = run(1);
    let b = run(2);
    assert!(
        !a.is_empty() && !b.is_empty(),
        "test is meaningless if either loop terminates immediately"
    );
    assert_ne!(
        a, b,
        "dist() with seeds 1 vs 2 must diverge — if these match, the seed parameter is not being threaded through to llama.cpp"
    );
}
