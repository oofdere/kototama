mod common;

use rusty_llama::test_common::{model_path, test_ctx_params};
use rusty_llama::{Context, Model, ModelParams};

#[test]
fn context_new_ok() {
    let (model, params) = common::load_model_and_context();
    let _ctx = Context::new(&model, &params).expect("failed to create context");
}

#[test]
fn n_ctx_at_least_params() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    assert!(
        ctx.n_ctx() >= params.n_ctx,
        "n_ctx should be at least the requested size"
    );
}

#[test]
fn free_slots_starts_full() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    assert_eq!(ctx.free_slots(), params.n_seq_max as usize);
}

#[test]
fn sequence_checkout_reduces_free_slots() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let total = ctx.free_slots();
    let seq = ctx.sequence().expect("should be able to get a sequence");
    assert_eq!(ctx.free_slots(), total - 1);
    drop(seq);
    assert_eq!(ctx.free_slots(), total);
}

#[test]
fn sequence_drop_returns_slot() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let before = ctx.free_slots();
    {
        let _seq = ctx.sequence().unwrap();
        assert_eq!(ctx.free_slots(), before - 1);
    }
    assert_eq!(ctx.free_slots(), before);
}

#[test]
fn all_slots_exhausted() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let n = params.n_seq_max as usize;
    let seqs: Vec<_> = (0..n).map(|_| ctx.sequence().unwrap()).collect();
    assert_eq!(ctx.free_slots(), 0);
    assert!(
        ctx.sequence().is_none(),
        "should return None when all slots are taken"
    );
    drop(seqs);
    assert_eq!(ctx.free_slots(), n);
}

#[test]
fn can_shift_does_not_crash() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let _ = ctx.can_shift();
}

#[test]
fn perf_does_not_crash() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let _ = ctx.perf();
}

// ---------- Context::clone() shares actor state ----------

#[test]
fn context_clone_shares_free_slots() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let ctx_clone = ctx.clone();
    // Both handles should see the same free-slot count
    assert_eq!(ctx.free_slots(), ctx_clone.free_slots());
}

#[test]
fn context_clone_slot_checkout_visible_on_original() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let ctx_clone = ctx.clone();
    let total = ctx.free_slots();

    // Check out a sequence via the clone
    let _seq = ctx_clone.sequence().unwrap();

    // The original handle should observe the reduced count (shared actor)
    assert_eq!(ctx.free_slots(), total - 1);
}

#[test]
fn context_clone_slot_checkout_visible_on_clone() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let ctx_clone = ctx.clone();
    let total = ctx.free_slots();

    // Check out a sequence via the original
    let _seq = ctx.sequence().unwrap();

    // The clone handle should observe the reduced count
    assert_eq!(ctx_clone.free_slots(), total - 1);
}

#[test]
fn context_clone_sequence_drop_restores_on_both() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let ctx_clone = ctx.clone();
    let total = ctx.free_slots();

    {
        let _seq = ctx.sequence().unwrap();
        assert_eq!(ctx_clone.free_slots(), total - 1);
    }
    // After drop, both handles see the restored count
    assert_eq!(ctx.free_slots(), total);
    assert_eq!(ctx_clone.free_slots(), total);
}

#[test]
fn context_clone_n_ctx_matches() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let ctx_clone = ctx.clone();
    assert_eq!(ctx.n_ctx(), ctx_clone.n_ctx());
}

// ---------- Context from cloned Model ----------

#[test]
fn context_from_cloned_model() {
    let (model, params) = common::load_model_and_context();
    let model_clone = model.clone();
    // Context should be successfully created from a cloned model handle
    let ctx =
        Context::new(&model_clone, &params).expect("context from cloned model should succeed");
    assert!(ctx.free_slots() > 0);
    assert!(ctx.n_ctx() >= params.n_ctx);
}

// ---------- Model lifetime is extended by Context (use-after-free regression) ----------

// `llama_context` stores `const llama_model & model;` and reads `model.vocab` /
// `model.hparams` during `llama_decode` and inside `~llama_context()`. If the
// last `Model` handle were dropped while a `Context` was still alive, those
// reads would touch freed memory. Loads an independent (non-shared) `Model` so
// dropping it actually releases the underlying `llama_model`.
#[test]
fn context_outlives_dropped_model_handle() {
    let path = model_path();
    let mut model_params = ModelParams::new();
    model_params.n_gpu_layers = 0;
    let model = Model::load_from_file(&path, model_params).expect("load model");

    let params = test_ctx_params();
    let ctx = Context::new(&model, &params).expect("create context");
    drop(model);

    let mut seq = ctx.sequence().expect("checkout sequence");
    seq.push(0);
    drop(seq);
    drop(ctx);
}

#[test]
fn context_from_cloned_model_is_independent() {
    let (model, params) = common::load_model_and_context();
    let model_clone = model.clone();

    let ctx1 = Context::new(&model, &params).unwrap();
    let ctx2 = Context::new(&model_clone, &params).unwrap();

    // Independent contexts: checking out from one doesn't affect the other
    let _seq1 = ctx1.sequence().unwrap();
    assert_eq!(
        ctx2.free_slots(),
        params.n_seq_max as usize,
        "second context should have full slots independent of first"
    );
}
