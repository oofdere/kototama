mod common;

use rusty_llama::Context;

#[test]
fn context_new_ok() {
    let Some((model, params)) = common::try_load_model_and_context() else { return };
    let _ctx = Context::new(&model, &params).expect("failed to create context");
}

#[test]
fn n_ctx_at_least_params() {
    let Some((model, params)) = common::try_load_model_and_context() else { return };
    let ctx = Context::new(&model, &params).unwrap();
    // llama.cpp may round n_ctx up to a multiple of a hardware-dependent value
    assert!(ctx.n_ctx() >= params.n_ctx, "n_ctx should be at least the requested size");
}

#[test]
fn model_ref_round_trips() {
    let Some((model, params)) = common::try_load_model_and_context() else { return };
    let ctx = Context::new(&model, &params).unwrap();
    // ctx.model() should return a reference to the same model
    assert_eq!(ctx.model().n_tokens(), model.n_tokens());
}

#[test]
fn free_slots_starts_full() {
    let Some((model, params)) = common::try_load_model_and_context() else { return };
    let ctx = Context::new(&model, &params).unwrap();
    assert_eq!(ctx.free_slots(), params.n_seq_max as usize);
}

#[test]
fn sequence_checkout_reduces_free_slots() {
    let Some((model, params)) = common::try_load_model_and_context() else { return };
    let ctx = Context::new(&model, &params).unwrap();
    let total = ctx.free_slots();
    let seq = ctx.sequence().expect("should be able to get a sequence");
    assert_eq!(ctx.free_slots(), total - 1);
    drop(seq);
    assert_eq!(ctx.free_slots(), total);
}

#[test]
fn sequence_drop_returns_slot() {
    let Some((model, params)) = common::try_load_model_and_context() else { return };
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
    let Some((model, params)) = common::try_load_model_and_context() else { return };
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
    let Some((model, params)) = common::try_load_model_and_context() else { return };
    let ctx = Context::new(&model, &params).unwrap();
    let _ = ctx.can_shift();
}

#[test]
fn perf_does_not_crash() {
    let Some((model, params)) = common::try_load_model_and_context() else { return };
    let ctx = Context::new(&model, &params).unwrap();
    let _ = ctx.perf();
}
