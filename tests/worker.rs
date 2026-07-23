mod common;

use rusty_llama::{Context, Model, ModelParams};

#[test]
fn context_keeps_its_model_alive() {
    let mut model_params = ModelParams::new();
    model_params.n_gpu_layers = 0;
    let model = Model::load_from_file(&common::model_path(), model_params).unwrap();
    let params = common::test_ctx_params();
    let ctx = Context::new(&model, &params).unwrap();

    drop(model);

    assert!(ctx.n_ctx() >= params.n_ctx);
    assert!(ctx.sequence().is_some());
}

#[test]
fn async_api_uses_the_same_worker_state() {
    pollster::block_on(async {
        let (model, params) = common::load_model_and_context();
        let ctx = Context::new(&model, &params).unwrap();
        let initial_slots = ctx.free_slots();

        let mut seq = ctx.sequence_async().await.unwrap();
        assert_eq!(ctx.free_slots_async().await, initial_slots - 1);

        let tokens = model.tokenize("hello", false, false);
        seq.extend_async(&tokens).await;
        assert_eq!(seq.tokens().as_ref(), tokens.as_slice());
        assert_eq!(seq.logits().unwrap().len(), model.n_tokens() as usize);

        let popped = seq.pop_async().await;
        assert!(popped.is_some());
        assert_eq!(seq.len(), tokens.len() - 1);
    });
}

#[test]
fn sync_and_async_calls_can_share_a_context() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let sync_value = ctx.n_ctx();
    let async_value = pollster::block_on(ctx.n_ctx_async());
    assert_eq!(sync_value, async_value);
}

#[test]
fn dropping_sequence_checkout_future_releases_the_slot() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let initial_slots = ctx.free_slots();

    let checkout = ctx.sequence_async();
    drop(checkout);

    // free_slots is submitted after the cancellation command, so its reply is
    // also a worker barrier for the abandoned checkout.
    assert_eq!(ctx.free_slots(), initial_slots);
}
