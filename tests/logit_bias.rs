mod common;

use llama_sys::llama_logit_bias;
use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};

// End-to-end behavioural coverage for `Sampler::logit_bias`.
//
// The constructor is exercised by no test on trunk today, and the only PR
// touching it (PR #43, open) is init-only — it builds the sampler with
// `n_logit_bias = 0` and never runs it through the sampling pipeline.
// That smoke test would pass even if:
//   * the per-entry FFI struct layout (`{ token: i32, bias: f32 }`) drifted;
//   * the `n_logit_bias` count were ignored (e.g. always read 0 entries);
//   * the bias values were applied to the wrong token id;
//   * the sampler were silently treated as a no-op in the chain.
//
// The tests below pin the *behaviour* contract: a positive bias must steer
// greedy onto the targeted token, a negative bias must steer it off, and an
// empty bias array must be a no-op. They live in their own file to avoid
// merge conflicts with the in-flight PRs touching `tests/sampler.rs`
// (#43, #66, #90).

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i32)
        .unwrap()
}

// ---------- n_logit_bias = 0 is a no-op ----------

#[test]
fn empty_logit_bias_is_noop_under_greedy() {
    // Chain: logit_bias(0 entries) -> greedy. Must agree with plain greedy
    // on the same logits. A regression that misread the count and walked
    // an uninitialised pointer would either crash or pick a different token.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", true, false);
    seq.extend(&tokens);

    let plain = {
        let chain = SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());
        seq.sample(&chain)
    };
    let biased = {
        let chain = SamplerChain::new(&SamplerChainParams::new())
            .add(Sampler::logit_bias(model.n_tokens(), 0, std::ptr::null()))
            .add(Sampler::greedy());
        seq.sample(&chain)
    };
    assert_eq!(
        plain, biased,
        "logit_bias with zero entries must leave the greedy choice unchanged"
    );
}

// ---------- positive bias steers greedy onto the target ----------

#[test]
fn positive_logit_bias_forces_target_token_under_greedy() {
    // Pick any token that is *not* the natural argmax, add an overwhelming
    // positive bias on it, and greedy must select it. A bias of 1e9 swamps
    // any plausible original logit (TinyStories logits live in roughly
    // [-30, 30]), so the assertion is independent of the model's actual
    // distribution.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", true, false);
    seq.extend(&tokens);

    let natural = argmax(seq.logits().unwrap());
    // Pick a different in-vocab token to bias toward.
    let target: i32 = if natural == 0 { 1 } else { 0 };
    assert!(target >= 0 && target < model.n_tokens());
    assert_ne!(target, natural, "test setup must bias a non-argmax token");

    let biases = vec![llama_logit_bias {
        token: target,
        bias: 1e9,
    }];
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::logit_bias(
            model.n_tokens(),
            biases.len() as i32,
            biases.as_ptr(),
        ))
        .add(Sampler::greedy());
    let sampled = seq.sample(&chain);
    // Keep `biases` alive across the sampler init (llama.cpp copies it
    // internally, but binding to a local makes that lifetime explicit).
    drop(biases);

    assert_eq!(
        sampled, target,
        "positive bias of 1e9 on token {target} must force greedy to pick it (got {sampled})"
    );
}

// ---------- negative bias on the argmax forces a different token ----------

#[test]
fn negative_logit_bias_on_argmax_excludes_it_under_greedy() {
    // A strongly negative bias on the natural argmax must push it below
    // every other token. Greedy then picks something else — we don't pin
    // *which* runner-up wins, only that the original winner is no longer
    // selected and the returned token is in-vocab.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", true, false);
    seq.extend(&tokens);

    let natural = argmax(seq.logits().unwrap());

    let biases = vec![llama_logit_bias {
        token: natural,
        bias: -1e9,
    }];
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::logit_bias(
            model.n_tokens(),
            biases.len() as i32,
            biases.as_ptr(),
        ))
        .add(Sampler::greedy());
    let sampled = seq.sample(&chain);
    drop(biases);

    assert_ne!(
        sampled, natural,
        "negative bias of -1e9 on the argmax token must force greedy off it"
    );
    assert!(
        sampled >= 0 && sampled < model.n_tokens(),
        "sampled token {sampled} must be in vocab range [0, {})",
        model.n_tokens()
    );
}

// ---------- multiple biased tokens: largest bias wins ----------

#[test]
fn strongest_of_several_biases_wins_under_greedy() {
    // Two non-argmax tokens, both biased positively. The token with the
    // larger bias must win under greedy — pins that multi-entry bias arrays
    // are read across the full `n_logit_bias` range (not truncated to 1)
    // and that each entry is matched to the right token id.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", true, false);
    seq.extend(&tokens);

    let natural = argmax(seq.logits().unwrap());

    // Two distinct in-vocab tokens, neither being the natural argmax.
    let targets: Vec<i32> = (0..model.n_tokens())
        .filter(|&t| t != natural)
        .take(2)
        .collect();
    assert_eq!(
        targets.len(),
        2,
        "vocab too small to pick two non-argmax targets"
    );
    let weaker = targets[0];
    let stronger = targets[1];

    let biases = vec![
        llama_logit_bias {
            token: weaker,
            bias: 1e7,
        },
        llama_logit_bias {
            token: stronger,
            bias: 1e9,
        },
    ];
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::logit_bias(
            model.n_tokens(),
            biases.len() as i32,
            biases.as_ptr(),
        ))
        .add(Sampler::greedy());
    let sampled = seq.sample(&chain);
    drop(biases);

    assert_eq!(
        sampled, stronger,
        "with bias +1e7 on {weaker} and +1e9 on {stronger}, greedy must pick {stronger} (got {sampled})"
    );
}

// ---------- compose: bias-then-bias agrees with single bias ----------

#[test]
fn double_logit_bias_in_chain_composes_under_greedy() {
    // Two `logit_bias` samplers in series should add their biases. Bias A
    // (+5e8 on `target`) then bias B (+5e8 on `target`) — net +1e9, which
    // is enough to overcome any plausible TinyStories logit and force the
    // greedy pick onto `target`. Catches a regression where stacking two
    // `logit_bias` samplers in a chain caused later entries to clobber
    // earlier ones (a plausible failure mode if the chain wrote to a
    // shared scratch buffer instead of mutating per-sampler state).
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", true, false);
    seq.extend(&tokens);

    let natural = argmax(seq.logits().unwrap());
    let target: i32 = if natural == 0 { 1 } else { 0 };
    assert_ne!(target, natural);

    let biases_a = vec![llama_logit_bias {
        token: target,
        bias: 5e8,
    }];
    let biases_b = vec![llama_logit_bias {
        token: target,
        bias: 5e8,
    }];
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::logit_bias(
            model.n_tokens(),
            biases_a.len() as i32,
            biases_a.as_ptr(),
        ))
        .add(Sampler::logit_bias(
            model.n_tokens(),
            biases_b.len() as i32,
            biases_b.as_ptr(),
        ))
        .add(Sampler::greedy());
    let sampled = seq.sample(&chain);
    drop(biases_a);
    drop(biases_b);

    assert_eq!(
        sampled, target,
        "two stacked +5e8 biases on token {target} should compose to +1e9 and force greedy onto it (got {sampled})"
    );
}
