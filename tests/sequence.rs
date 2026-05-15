mod common;

use rusty_llama::Context;

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

#[test]
fn push_increases_len() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.len(), 0);
    let tokens = model.tokenize("hi", false, false);
    seq.push(tokens[0]);
    assert_eq!(seq.len(), 1);
}

#[test]
fn extend_fills_tokens() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    let n = tokens.len();
    seq.extend(&tokens);
    assert_eq!(seq.len(), n);
}

#[test]
fn tokens_accessor_matches_push_order() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("abc", false, false);
    seq.extend(&tokens);
    assert_eq!(seq.tokens(), tokens.as_slice());
}

#[test]
fn index_operator() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    assert_eq!(seq[0], tokens[0]);
}

#[test]
fn get_returns_token() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    assert_eq!(seq.get(0), Some(tokens[0]));
    assert_eq!(seq.get(999), None);
}

#[test]
fn pop_decreases_len() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let popped = seq.pop();
    assert!(popped.is_some());
    assert_eq!(seq.len(), len_before - 1);
}

#[test]
fn pop_empty_returns_none() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.pop(), None);
}

#[test]
fn logits_len_equals_vocab_size_after_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.push(tokens[0]);
    assert_eq!(seq.logits().len(), model.n_tokens() as usize);
}

#[test]
fn remove_range() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    let n = tokens.len();
    seq.extend(&tokens);
    assert_eq!(seq.len(), n);
    seq.remove(0..1);
    assert_eq!(seq.len(), n - 1);
}

#[test]
fn pos_min_max_after_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    // Before any push, should return -1 (empty)
    assert_eq!(seq.pos_min(), -1);
    assert_eq!(seq.pos_max(), -1);
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.pos_min() >= 0);
    assert!(seq.pos_max() >= seq.pos_min());
}

#[test]
fn copy_to() {
    let (model, _) = setup();
    // KV copies across sequences require kv_unified = true
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    // dst needs its own decoded state before we can overwrite it via KV copy
    dst.extend(&tokens);
    src.copy_to(&mut dst, 0..tokens.len());
    assert_eq!(dst.tokens(), src.tokens());
}
