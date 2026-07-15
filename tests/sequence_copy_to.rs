mod common;

use rusty_llama::Context;

#[test]
fn copy_to_replaces_destination_kv_state() {
    let model = common::load_model();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let token = model.tokenize("hi", false, false)[0];

    src.push(token);
    dst.extend(&[token; 4]);
    src.copy_to(&mut dst, 0..1);

    assert_eq!(dst.tokens(), &[token]);
    assert_eq!(dst.pos_min(), 0);
    assert_eq!(dst.pos_max(), 0);
}
