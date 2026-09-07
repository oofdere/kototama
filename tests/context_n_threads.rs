mod common;

use llama_sys::GGML_MAX_N_THREADS;
use rusty_llama::Context;

fn push_some(ctx: &Context) {
    let mut seq = ctx.sequence().expect("free slot");
    seq.push(1);
    seq.push(2);
    assert_eq!(seq.len(), 2);
    assert!(seq.logits().is_some());
}

#[test]
fn n_threads_above_ggml_max_is_rejected() {
    let (model, mut params) = common::load_model_and_context();
    params.n_threads = i32::MAX;
    assert!(Context::new(&model, &params).is_err());

    let mut params = common::test_ctx_params();
    params.n_threads = GGML_MAX_N_THREADS as i32 + 1;
    assert!(Context::new(&model, &params).is_err());
}

#[test]
fn n_threads_batch_above_ggml_max_is_rejected() {
    let (model, mut params) = common::load_model_and_context();
    params.n_threads_batch = i32::MAX;
    assert!(Context::new(&model, &params).is_err());
}

#[test]
fn n_threads_at_ggml_max_is_accepted() {
    let (model, mut params) = common::load_model_and_context();
    params.n_threads = GGML_MAX_N_THREADS as i32;
    params.n_threads_batch = GGML_MAX_N_THREADS as i32;
    let ctx = Context::new(&model, &params).expect("context");
    push_some(&ctx);
}

#[test]
fn non_positive_n_threads_uses_ggml_default() {
    let (model, mut params) = common::load_model_and_context();
    params.n_threads = 0;
    params.n_threads_batch = -1;
    let ctx = Context::new(&model, &params).expect("context");
    push_some(&ctx);
}
