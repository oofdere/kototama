mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};

// End-to-end coverage for `Sampler::clone()` on stateful samplers.
//
// The only existing `Clone` test (`tests/sampler.rs::greedy_clone`) clones
// a stateless `Sampler::greedy()` and immediately drops both handles. That
// catches a double-free in `Drop` for the stateless case but leaves the
// stateful samplers — `dist`, `mirostat`, `mirostat_v2`, `adaptive_p` —
// without any `Clone` coverage at all:
//
// - Nothing verifies that `llama_sampler_clone` produces a non-null,
//   non-dangling pointer for the stateful initializers (a regression in
//   the FFI binding that returned the original pointer would silently
//   pass `greedy_clone` but double-free here, where the underlying
//   allocations carry larger state structs).
// - Nothing exercises the cloned sampler through `Sequence::sample`, so
//   even if `clone()` returned a *valid* pointer, an upstream change that
//   broke the state copy (e.g. shallow-copied an RNG handle and aliased
//   it across clones) would not surface.
// - Nothing pins the documented contract that a clone is seeded
//   identically to the original — i.e. two fresh `Sampler::dist(SEED)`
//   handles, one obtained via `Sampler::dist(SEED)` and the other via
//   `Sampler::dist(SEED).clone()`, must produce the same first sample.
//
// I checked open PRs first: #43 adds *init* smoke tests for `mirostat` /
// `logit_bias` / `infill` / `dry` but does not touch `clone()`; #90 covers
// the stateful samplers through `Sequence::sample` but uses bare
// constructors (no `clone()` on the path); #84 sampler-perf and raw-
// `Sampler` sampling tests don't clone either; #98 documents `Clone`
// semantics but adds no test.

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

fn assert_in_vocab(token: i32, model: &rusty_llama::Model, label: &str) {
    assert!(
        token >= 0 && token < model.n_tokens(),
        "{label}-sampled token {token} out of vocab range [0, {})",
        model.n_tokens()
    );
}

// ---------- Smoke: clone + drop is sound for every stateful sampler ----------
//
// `Sampler::Drop` calls `llama_sampler_free` on the inner pointer. If
// `llama_sampler_clone` returned the *original* pointer (shallow alias
// instead of deep copy), dropping both handles would double-free. The
// stateless case is already covered; these pin the same invariant for
// each stateful initializer.

#[test]
fn dist_clone_then_drop_both_is_sound() {
    let s = Sampler::dist(42);
    let s2 = s.clone();
    drop(s);
    drop(s2);
}

#[test]
fn mirostat_clone_then_drop_both_is_sound() {
    // mirostat v1 requires the vocab size up front.
    let model = common::load_model();
    let s = Sampler::mirostat(model.n_tokens(), 42, 5.0, 0.1, 100);
    let s2 = s.clone();
    drop(s);
    drop(s2);
}

#[test]
fn mirostat_v2_clone_then_drop_both_is_sound() {
    let s = Sampler::mirostat_v2(42, 5.0, 0.1);
    let s2 = s.clone();
    drop(s);
    drop(s2);
}

#[test]
fn adaptive_p_clone_then_drop_both_is_sound() {
    let s = Sampler::adaptive_p(0.1, 0.9, 42);
    let s2 = s.clone();
    drop(s);
    drop(s2);
}

#[test]
fn dist_clone_drop_original_first_is_sound() {
    // Drop the original *before* the clone to catch a regression where
    // `clone()` shared a back-reference into the original's allocation.
    let s = Sampler::dist(42);
    let s2 = s.clone();
    drop(s);
    drop(s2);
}

#[test]
fn dist_clone_drop_clone_first_is_sound() {
    let s = Sampler::dist(42);
    let s2 = s.clone();
    drop(s2);
    drop(s);
}

// ---------- Cloned sampler is usable through Sequence::sample ----------
//
// A successful clone must produce a pointer that `llama_sampler_sample`
// accepts. If `llama_sampler_clone` returned NULL (e.g. an upstream FFI
// rename leaving us calling a stub), the wrapper would still construct a
// `Sampler(null_mut)` — `Drop` would not crash, but the first use through
// `Sequence::sample` would segfault. These tests force that path.

#[test]
fn cloned_dist_via_sequence_sample_yields_in_vocab_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let original = Sampler::dist(42);
    let cloned = original.clone();
    let token = seq.sample(&cloned);
    assert_in_vocab(token, &model, "cloned dist");
}

#[test]
fn cloned_mirostat_v2_via_sequence_sample_yields_in_vocab_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let original = Sampler::mirostat_v2(42, 5.0, 0.1);
    let cloned = original.clone();
    let token = seq.sample(&cloned);
    assert_in_vocab(token, &model, "cloned mirostat_v2");
}

#[test]
fn cloned_mirostat_via_sequence_sample_yields_in_vocab_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let original = Sampler::mirostat(model.n_tokens(), 42, 5.0, 0.1, 100);
    let cloned = original.clone();
    let token = seq.sample(&cloned);
    assert_in_vocab(token, &model, "cloned mirostat");
}

#[test]
fn cloned_adaptive_p_via_sequence_sample_yields_in_vocab_token() {
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let original = Sampler::adaptive_p(0.1, 0.9, 42);
    let cloned = original.clone();
    let token = seq.sample(&cloned);
    assert_in_vocab(token, &model, "cloned adaptive_p");
}

#[test]
fn cloned_dist_works_in_sampler_chain() {
    // The chain path takes ownership of the clone via `SamplerChain::add`,
    // so this verifies that the cloned pointer also survives the
    // ownership-transfer protocol used by the chain (`mem::forget` on add,
    // `llama_sampler_free` on chain drop).
    let (model, params) = setup();
    let (_ctx, seq) = seeded_seq(&model, &params);
    let original = Sampler::dist(42);
    let cloned = original.clone();
    let chain = SamplerChain::new(&SamplerChainParams::new()).add(cloned);
    let token = seq.sample(&chain);
    assert_in_vocab(token, &model, "chain-wrapped clone of dist");
}

// ---------- State copy: a clone is seeded identically to the original ----------
//
// llama.cpp's `llama_sampler_clone` deep-copies the sampler including its
// RNG state. The wrapper docs (PR #98) call this out explicitly. Pin the
// contract: a fresh `dist(seed)` and a `.clone()` of one must produce the
// same token on the same logits — otherwise the clone is silently producing
// fresh state, which would break "snapshot the sampler, branch generation"
// workflows.

#[test]
fn cloned_dist_first_sample_matches_original_first_sample() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let prompt = model.tokenize("hello world", false, false);

    let original = Sampler::dist(42);
    let cloned = original.clone();

    // Two sequences off the same context — both see identical logits, so
    // any token difference here can only come from sampler state.
    let mut seq_orig = ctx.sequence().unwrap();
    seq_orig.extend(&prompt);
    let mut seq_clone = ctx.sequence().unwrap();
    seq_clone.extend(&prompt);

    let t_orig = seq_orig.sample(&original);
    let t_clone = seq_clone.sample(&cloned);

    assert_eq!(
        t_orig, t_clone,
        "fresh `dist(seed).clone()` must be seeded identically to the original"
    );
}

#[test]
fn cloned_dist_after_advance_resumes_from_originals_state() {
    // Stronger than the fresh-clone case above: advance the original
    // through several `sample` calls (each one steps the RNG), *then*
    // clone, then sample once from each. The cloned sampler must see the
    // same next RNG output as the original — i.e. clone snapshots state
    // at the moment of clone, not at construction.
    //
    // A regression that re-seeded the clone from the constructor seed
    // (instead of copying current state) would produce a different token
    // here whenever the advanced-state token differs from the fresh-state
    // token, which is the common case for non-degenerate distributions.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let prompt = model.tokenize("once upon a time", true, false);

    let mut seq_orig = ctx.sequence().unwrap();
    seq_orig.extend(&prompt);

    let original = Sampler::dist(0xC0FFEE);

    // Advance original past its first few RNG steps.
    for _ in 0..3 {
        let _ = seq_orig.sample(&original);
    }

    // Clone now — should snapshot the advanced state.
    let cloned = original.clone();

    // Run the *next* sample on a fresh sequence with the same prompt, so
    // the logits seen by both samplers are identical and any difference
    // is purely sampler-state.
    let mut seq_clone = ctx.sequence().unwrap();
    seq_clone.extend(&prompt);

    let t_next_orig = seq_orig.sample(&original);
    let t_next_clone = seq_clone.sample(&cloned);

    assert_eq!(
        t_next_orig, t_next_clone,
        "clone must snapshot the advanced RNG state — not re-seed from construction"
    );
}

// ---------- Clone produces an independently-owned sampler ----------
//
// After clone, mutating one sampler (by sampling, which advances RNG /
// mirostat-mu / adaptive-p EMA state) must not silently advance the
// other. A regression that shared state across clones would surface here
// as the two samplers reporting the same token even after one has been
// stepped past the shared state.

#[test]
fn dist_clone_state_is_independent_from_original() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let prompt = model.tokenize("once upon a time", true, false);

    let original = Sampler::dist(42);
    let cloned = original.clone();

    let mut seq_orig = ctx.sequence().unwrap();
    seq_orig.extend(&prompt);
    let mut seq_clone = ctx.sequence().unwrap();
    seq_clone.extend(&prompt);

    // Advance `original` several steps; `cloned` should not move.
    for _ in 0..4 {
        let _ = seq_orig.sample(&original);
    }

    // The clone is still at its initial (fresh-seed) state, so its first
    // sample must equal a brand-new `dist(42)`'s first sample on the
    // same logits.
    let reference = Sampler::dist(42);
    let mut seq_ref = ctx.sequence().unwrap();
    seq_ref.extend(&prompt);

    let t_clone_first = seq_clone.sample(&cloned);
    let t_ref_first = seq_ref.sample(&reference);

    assert_eq!(
        t_clone_first, t_ref_first,
        "advancing the original must not advance the clone — clone should still see fresh-seed state"
    );
}
