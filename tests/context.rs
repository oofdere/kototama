mod common;

use rusty_llama::{Context, ContextParams};

#[test]
fn context_new_ok() {
    let (model, params) = common::load_model_and_context();
    let _ctx = Context::new(&model, &params).expect("failed to create context");
}

#[test]
fn n_ctx_at_least_params() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    assert!(ctx.n_ctx() >= params.n_ctx, "n_ctx should be at least the requested size");
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
    assert!(ctx.sequence().is_none(), "should return None when all slots are taken");
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
    let ctx = Context::new(&model_clone, &params).expect("context from cloned model should succeed");
    assert!(ctx.free_slots() > 0);
    assert!(ctx.n_ctx() >= params.n_ctx);
}

#[test]
fn context_from_cloned_model_is_independent() {
    let (model, params) = common::load_model_and_context();
    let model_clone = model.clone();

    let ctx1 = Context::new(&model, &params).unwrap();
    let ctx2 = Context::new(&model_clone, &params).unwrap();

    // Independent contexts: checking out from one doesn't affect the other
    let _seq1 = ctx1.sequence().unwrap();
    assert_eq!(ctx2.free_slots(), params.n_seq_max as usize,
        "second context should have full slots independent of first");
}

// ---------- ContextParams setters ----------

#[test]
fn context_params_setters_apply_values() {
    let mut p = ContextParams::new();
    p.set_n_ctx(1024)
        .set_n_batch(256)
        .set_n_ubatch(128)
        .set_n_seq_max(7)
        .set_n_threads(2)
        .set_n_threads_batch(4)
        .set_embeddings(true)
        .set_offload_kqv(false)
        .set_no_perf(true);

    // Read back through Deref to confirm each setter mutated the field.
    assert_eq!(p.n_ctx, 1024);
    assert_eq!(p.n_batch, 256);
    assert_eq!(p.n_ubatch, 128);
    assert_eq!(p.n_seq_max, 7);
    assert_eq!(p.n_threads, 2);
    assert_eq!(p.n_threads_batch, 4);
    assert!(p.embeddings);
    assert!(!p.offload_kqv);
    assert!(p.no_perf);
}

#[test]
fn context_params_setters_are_chainable() {
    let mut p = ContextParams::new();
    let returned = p.set_n_ctx(99).set_n_batch(99);
    // The chain returns &mut Self at the end of the chain.
    assert_eq!(returned.n_ctx, 99);
    assert_eq!(returned.n_batch, 99);
}

#[test]
fn context_params_default_samplers_pointer_is_null() {
    // Soundness backstop: the C++ side at llama-context.cpp:91 only
    // dereferences `params.samplers[i]` when `n_samplers > 0`. The default
    // params must keep `samplers == null` and `n_samplers == 0`, and we no
    // longer expose a safe way to clobber them.
    let p = ContextParams::new();
    assert!(p.samplers.is_null());
    assert_eq!(p.n_samplers, 0);
    assert!(p.cb_eval_user_data.is_null());
    assert!(p.abort_callback_data.is_null());
}

#[test]
fn context_params_load_through_safe_setters_round_trips() {
    // Build params using only safe setters (no DerefMut / no as_mut_raw),
    // then drive a successful context creation. This exercises the
    // post-#85-style API end-to-end and proves the safe surface alone is
    // sufficient for the in-tree call sites.
    let model = common::load_model();
    let mut params = ContextParams::new();
    params
        .set_n_ctx(512)
        .set_n_batch(512)
        .set_n_seq_max(2)
        .set_no_perf(true);

    let ctx = Context::new(&model, &params).expect("context should build from safe-setter params");
    assert!(ctx.n_ctx() >= 512);
    assert_eq!(ctx.free_slots(), 2);
}
