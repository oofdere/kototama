mod common;

use rusty_llama::{Context, ContextParams, Model};

// `llama_memory_seq_cp` on a non-unified KV cache copies across streams and
// aborts, so these tests use a unified cache (a separate pre-existing issue).
fn unified_params() -> ContextParams {
    let mut p = common::test_ctx_params();
    p.kv_unified = true;
    p
}

fn setup() -> (Model, ContextParams) {
    (common::load_model(), unified_params())
}

fn tokens(model: &Model, n: usize) -> Vec<i32> {
    let toks = model.tokenize("the quick brown fox jumps over", false, false);
    assert!(toks.len() >= n, "test model produced too few tokens");
    toks[..n].to_vec()
}

#[test]
fn copy_of_full_range_replicates_sequence() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 3);
    src.extend(&toks);

    assert!(src.copy_to(&mut dst, 0..toks.len()));
    assert_eq!(dst.tokens(), src.tokens());
    assert_eq!(dst.pos_min(), 0);
    assert_eq!(dst.pos_max(), toks.len() as i32 - 1);
}

#[test]
fn copy_of_suffix_rebases_kv_positions() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(src.copy_to(&mut dst, 1..3));
    assert_eq!(dst.tokens(), &toks[1..3]);
    assert_eq!(dst.pos_min(), 0);
    assert_eq!(dst.pos_max(), 1);
}

#[test]
fn push_after_partial_copy_decodes() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(src.copy_to(&mut dst, 2..4));
    dst.push(toks[0]);
    assert_eq!(dst.len(), 3);
    assert_eq!(dst.pos_max(), 2);
    assert!(dst.logits().is_some());
}

#[test]
fn copy_replaces_longer_destination() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks[..2]);
    dst.extend(&toks);

    assert!(src.copy_to(&mut dst, 0..2));
    assert_eq!(dst.len(), 2);
    assert_eq!(dst.pos_max(), 1);
    dst.push(toks[0]);
    assert_eq!(dst.pos_max(), 2);
}

#[test]
fn out_of_range_copy_is_rejected() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 2);
    src.extend(&toks);
    dst.extend(&toks);

    assert!(!src.copy_to(&mut dst, 0..99));
    // The destination is left exactly as it was.
    assert_eq!(dst.tokens(), toks.as_slice());
    assert_eq!(dst.pos_min(), 0);
    assert_eq!(dst.pos_max(), toks.len() as i32 - 1);
}

#[test]
fn inverted_range_copy_is_rejected() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 3);
    src.extend(&toks);

    assert!(!src.copy_to(&mut dst, 2..1));
    assert!(dst.is_empty());
    assert_eq!(dst.pos_max(), -1);
}

#[test]
fn empty_range_copy_clears_destination() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 3);
    src.extend(&toks);
    dst.extend(&toks);

    assert!(src.copy_to(&mut dst, 1..1));
    assert!(dst.is_empty());
    assert_eq!(dst.pos_max(), -1);
    dst.push(toks[0]);
    assert_eq!(dst.pos_max(), 0);
}

#[test]
fn copy_from_rebases_kv_positions() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let toks = tokens(&model, 4);
    src.extend(&toks);

    assert!(dst.copy_from(&src, 1..4));
    assert_eq!(dst.tokens(), &toks[1..4]);
    assert_eq!(dst.pos_min(), 0);
    assert_eq!(dst.pos_max(), 2);
}
