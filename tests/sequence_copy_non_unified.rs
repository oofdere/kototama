mod common;

use rusty_llama::{Context, ContextParams, Model};

fn setup(kv_unified: bool) -> (Model, ContextParams) {
    let mut params = common::test_ctx_params();
    params.kv_unified = kv_unified;
    (common::load_model(), params)
}

fn tokens(model: &Model, n: usize) -> Vec<i32> {
    let toks = model.tokenize("the quick brown fox jumps over", false, false);
    assert!(toks.len() >= n, "test model produced too few tokens");
    toks[..n].to_vec()
}

#[test]
fn default_params_are_not_kv_unified() {
    let (model, params) = setup(false);
    let ctx = Context::new(&model, &params).unwrap();
    assert!(!ctx.kv_unified());
}

#[test]
fn full_copy_works_with_a_non_unified_cache() {
    let (model, params) = setup(false);
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(src.copy_to(&mut dst, 0..toks.len()));
    assert_eq!(dst.tokens(), src.tokens());
    assert_eq!(dst.pos_min(), 0);
    assert_eq!(dst.pos_max(), toks.len() as i32 - 1);
}

#[test]
fn push_after_a_full_copy_decodes() {
    let (model, params) = setup(false);
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(src.copy_to(&mut dst, 0..toks.len()));
    dst.push(toks[0]);
    assert_eq!(dst.len(), toks.len() + 1);
    assert_eq!(dst.pos_max(), toks.len() as i32);
}

#[test]
fn partial_copy_is_rejected_with_a_non_unified_cache() {
    let (model, params) = setup(false);
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    // llama.cpp cannot copy a sub-range across KV streams; it must be reported
    // instead of aborting the process.
    assert!(!src.copy_to(&mut dst, 1..3));
    assert!(dst.tokens().is_empty());

    // Both sequences are still usable afterwards.
    src.push(toks[0]);
    assert_eq!(src.len(), toks.len() + 1);
    dst.push(toks[0]);
    assert_eq!(dst.tokens(), &toks[..1]);
}

#[test]
fn partial_copy_from_is_rejected_with_a_non_unified_cache() {
    let (model, params) = setup(false);
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(!dst.copy_from(&src, 2..4));
    assert!(dst.tokens().is_empty());
}

#[test]
fn raw_kv_copy_rejects_a_partial_range_with_a_non_unified_cache() {
    let (model, params) = setup(false);
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(!src.kv_copy(&mut dst, 1..3));
    assert!(src.kv_copy(&mut dst, -1..-1));
    assert_eq!(dst.pos_max(), toks.len() as i32 - 1);
}

#[test]
fn partial_copy_works_with_a_unified_cache() {
    let (model, params) = setup(true);
    let ctx = Context::new(&model, &params).unwrap();
    assert!(ctx.kv_unified());
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(src.copy_to(&mut dst, 1..3));
    assert_eq!(dst.tokens(), &toks[1..3]);
}

#[test]
fn copy_to_rejects_an_out_of_bounds_range() {
    let (model, params) = setup(true);
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 3);
    src.extend(&toks);

    assert!(!src.copy_to(&mut dst, 0..toks.len() + 1));
    assert!(!src.copy_to(&mut dst, 2..1));
    assert!(dst.tokens().is_empty());
}
