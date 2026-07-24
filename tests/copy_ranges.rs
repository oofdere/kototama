mod common;

use rusty_llama::Context;

fn context_with_copy_support() -> Context {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    Context::new(&model, &params).unwrap()
}

#[test]
fn copy_to_rejects_out_of_bounds_range() {
    let ctx = context_with_copy_support();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    assert!(!src.copy_to(&mut dst, 0..1));
}

#[test]
fn copy_to_async_rejects_out_of_bounds_range() {
    let ctx = context_with_copy_support();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    assert!(!pollster::block_on(src.copy_to_async(&mut dst, 0..1)).unwrap());
}
