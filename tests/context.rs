mod common;

use rusty_llama::Context;

#[test]
fn context_new_ok() {
    let (model, params) = common::load_model_and_context();
    let _ctx = Context::new(&model, &params).expect("failed to create context");
}

#[test]
fn n_ctx_at_least_params() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    // llama.cpp may round n_ctx up to a multiple of a hardware-dependent value
    assert!(ctx.n_ctx() >= params.n_ctx, "n_ctx should be at least the requested size");
}

#[test]
fn model_ref_round_trips() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    // ctx.model() should return a reference to the same model
    assert_eq!(ctx.model().n_tokens(), model.n_tokens());
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

#[test]
fn as_ptr_not_null() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    assert!(!ctx.as_ptr().is_null());
}

#[test]
fn as_mut_ptr_not_null() {
    let (model, params) = common::load_model_and_context();
    let mut ctx = Context::new(&model, &params).unwrap();
    assert!(!ctx.as_mut_ptr().is_null());
}

#[test]
fn params_round_trip() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    // ctx.params() should expose the same n_ctx we passed in.
    assert_eq!(ctx.params().n_ctx, params.n_ctx);
}

#[test]
fn context_params_as_ptr_not_null() {
    let params = common::test_ctx_params();
    assert!(!params.as_ptr().is_null());
}

#[test]
fn context_params_as_mut_ptr_not_null() {
    let mut params = common::test_ctx_params();
    assert!(!params.as_mut_ptr().is_null());
}

#[test]
fn context_params_deref_mut() {
    // Exercises the DerefMut impl for ContextParams (write through *params).
    let mut params = common::test_ctx_params();
    (*params).n_ctx = 256;
    assert_eq!(params.n_ctx, 256);
}

