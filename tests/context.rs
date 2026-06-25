mod common;

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

// ---------- Context keeps Model alive ----------
//
// Regression: Context::new used to take `&Model` without retaining the
// handle. The llama_context internally references the model's data and
// the global llama backend. If the user dropped every Model handle
// after constructing a Context, the underlying llama_model (and
// potentially the Backend) was freed while the context kept using it.

#[test]
fn context_keeps_model_alive_after_handles_dropped() {
    // Use a fresh Model so dropping it actually releases it. The shared
    // `load_model()` helper caches the Model in a OnceLock, so it would
    // mask this regression.
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    let model = Model::load_from_file(&common::model_path(), params).expect("load model");

    let ctx_params = common::test_ctx_params();
    let ctx = Context::new(&model, &ctx_params).expect("create context");

    // Grab a known-valid token before dropping the model handle.
    let tokens = model.tokenize("a", true, false);
    let first_token = *tokens.first().expect("tokenize produced at least one token");

    // Drop every Model handle in the user's scope. Before the fix this
    // released the llama_model while the actor's *mut llama_context was
    // still pointing into it.
    drop(model);

    // Use the context. Without the fix, the underlying llama_decode call
    // dereferences the freed llama_model — typically a segfault under
    // address sanitizer or simply garbage logits.
    let mut seq = ctx.sequence().expect("checkout sequence");
    seq.push(first_token);
    assert!(seq.logits().is_some(), "logits should be available after push");
}

#[test]
fn context_keeps_model_alive_across_clone() {
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    let model = Model::load_from_file(&common::model_path(), params).expect("load model");

    let ctx_params = common::test_ctx_params();
    let ctx = Context::new(&model, &ctx_params).expect("create context");
    let ctx_clone = ctx.clone();

    let tokens = model.tokenize("a", true, false);
    let first_token = *tokens.first().expect("tokenize produced at least one token");

    drop(ctx);
    drop(model);

    // The remaining Context clone should still hold the model alive.
    let mut seq = ctx_clone.sequence().expect("checkout sequence");
    seq.push(first_token);
    assert!(seq.logits().is_some());
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
