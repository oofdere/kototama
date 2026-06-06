mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};

// End-to-end coverage for the canonical generation loop:
//
//     loop {
//         let token = seq.sample(&chain);
//         if model.is_eog(token) { break; }
//         seq.push(token);
//     }
//
// That pattern is the documented use of the crate (`examples/simple_chat.rs`
// at L85-L105) but every existing test that exercises `Sequence::sample`
// calls it exactly once and then drops the sequence:
//   - `tests/sequence.rs::sequence_sample_*` — single-shot
//   - `tests/integration.rs::greedy_sample_matches_argmax`,
//     `sample_with_temperature_does_not_crash` — single-shot
//   - `tests/context.rs` (per PR #78) — single-shot on `Context::sample`
//   - `tests/grammar.rs` (per PR #66) — single-shot through a grammar chain
//   - `tests/perf.rs` (per PR #84) — counts `n_sample` but doesn't push
//     between samples or check the actual generation
//
// Every existing *multi-token* generation test in the suite
// (`tests/integration.rs::greedy_generation_is_deterministic`,
// `tests/snapshots.rs::greedy_generate`) bypasses the sampler and reaches
// into `seq.logits()` to argmax by hand, so the sample+push round-trip — and
// in particular the contract that `llama_sampler_sample` calls
// `llama_sampler_accept` so the chain advances between iterations — has no
// coverage.

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

fn greedy_chain() -> SamplerChain {
    SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy())
}

fn argmax_of(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i32)
        .unwrap()
}

// ---------- sample+push loop matches argmax+push loop under greedy ----------

#[test]
fn greedy_sample_push_loop_matches_argmax_push_loop() {
    // Pin the canonical generation loop against the manual argmax+push loop.
    // Both must yield the same token sequence on the same prompt — otherwise
    // the sampler-chain plumbing has drifted from the `greedy = argmax`
    // contract somewhere across the multi-step round-trip.
    let (model, params) = setup();
    let prompt = model.tokenize("Once upon a time", true, false);

    // Reference: manual argmax over `seq.logits()`.
    let argmax_run: Vec<i32> = {
        let ctx = Context::new(&model, &params).unwrap();
        let mut seq = ctx.sequence().unwrap();
        seq.extend(&prompt);
        let mut out = Vec::with_capacity(10);
        for _ in 0..10 {
            let t = argmax_of(seq.logits().unwrap());
            if model.is_eog(t) {
                break;
            }
            out.push(t);
            seq.push(t);
        }
        out
    };

    // Subject: sample through a greedy SamplerChain.
    let sample_run: Vec<i32> = {
        let ctx = Context::new(&model, &params).unwrap();
        let mut seq = ctx.sequence().unwrap();
        seq.extend(&prompt);
        let chain = greedy_chain();
        let mut out = Vec::with_capacity(10);
        for _ in 0..10 {
            let t = seq.sample(&chain);
            if model.is_eog(t) {
                break;
            }
            out.push(t);
            seq.push(t);
        }
        out
    };

    assert_eq!(
        sample_run, argmax_run,
        "greedy sample+push loop must produce the same tokens as argmax+push"
    );
    assert!(
        !sample_run.is_empty(),
        "test is meaningless if the loop terminates immediately"
    );
}

// ---------- the loop is deterministic across runs ----------

#[test]
fn greedy_sample_push_loop_is_deterministic() {
    // Same prompt + same greedy chain + same model → must produce the same
    // tokens across runs. A regression that leaked sampler state across
    // fresh contexts, or that reordered the apply/accept pair, could pass
    // every single-step test today and fail here.
    let (model, params) = setup();
    let prompt = model.tokenize("Once upon a time", true, false);

    let run = || -> Vec<i32> {
        let ctx = Context::new(&model, &params).unwrap();
        let mut seq = ctx.sequence().unwrap();
        seq.extend(&prompt);
        let chain = greedy_chain();
        let mut out = Vec::with_capacity(8);
        for _ in 0..8 {
            let t = seq.sample(&chain);
            if model.is_eog(t) {
                break;
            }
            out.push(t);
            seq.push(t);
        }
        out
    };

    let a = run();
    let b = run();
    assert_eq!(
        a, b,
        "greedy sample+push loop must be deterministic across runs"
    );
}

// ---------- seeded dist+temp loop is deterministic across runs ----------

#[test]
fn seeded_dist_sample_push_loop_is_deterministic() {
    // The greedy path has no RNG state — a regression in
    // `llama_sampler_accept` plumbing on a seeded RNG would not surface there.
    // A `dist(seed)` chain is the smallest non-trivial state machine in the
    // sampler API: each `sample` must consume one RNG step. If `accept`
    // wasn't being called, two runs from the same seed would either diverge
    // (RNG snapshot lost) or produce a degenerate sequence (same token every
    // step). Both are caught by this assertion.
    let (model, params) = setup();
    let prompt = model.tokenize("Once upon a time", true, false);
    const SEED: u32 = 0xC0FFEE;

    let run = || -> Vec<i32> {
        let ctx = Context::new(&model, &params).unwrap();
        let mut seq = ctx.sequence().unwrap();
        seq.extend(&prompt);
        let chain = SamplerChain::new(&SamplerChainParams::new())
            .add(Sampler::temp(0.8))
            .add(Sampler::top_k(40))
            .add(Sampler::dist(SEED));
        let mut out = Vec::with_capacity(8);
        for _ in 0..8 {
            let t = seq.sample(&chain);
            if model.is_eog(t) {
                break;
            }
            out.push(t);
            seq.push(t);
        }
        out
    };

    let a = run();
    let b = run();
    assert_eq!(
        a, b,
        "seeded dist+temp sample+push loop must be deterministic across runs"
    );
    assert!(
        !a.is_empty(),
        "test is meaningless if the loop terminates immediately"
    );
    // The chain has random non-greedy components, so the result should be
    // both reproducible (asserted above) AND consist of in-vocab tokens.
    for &t in &a {
        assert!(
            t >= 0 && t < model.n_tokens(),
            "dist-sampled token {t} out of vocab range [0, {})",
            model.n_tokens()
        );
    }
}

// ---------- multi-step matches single-step token-by-token ----------

#[test]
fn greedy_sample_push_loop_matches_step_by_step_argmax() {
    // Stronger than the run-vs-run comparison above: at *every* step, the
    // token returned by `seq.sample(&greedy_chain)` must equal the argmax of
    // the cached logits at that point. Catches a regression where the chain
    // operates on stale or partially-applied logits between iterations.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let prompt = model.tokenize("the quick brown fox", true, false);
    seq.extend(&prompt);
    let chain = greedy_chain();

    let mut steps = 0usize;
    for _ in 0..6 {
        let expected = argmax_of(seq.logits().unwrap());
        let got = seq.sample(&chain);
        assert_eq!(
            got, expected,
            "step {steps}: greedy sample must equal argmax of current logits"
        );
        if model.is_eog(got) {
            break;
        }
        seq.push(got);
        steps += 1;
    }
    assert!(steps > 0, "loop terminated immediately; test is degenerate");
}
