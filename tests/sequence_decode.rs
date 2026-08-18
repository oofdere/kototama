mod common;

use rusty_llama::Context;

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

#[test]
fn decode_refreshes_logits_after_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.push(tokens[0]);
    let before = seq.logits().unwrap().to_vec();

    assert!(seq.decode());

    let after = seq.logits().expect("decode must repopulate logits");
    assert_eq!(before, after);
    assert_eq!(seq.len(), 1);
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), 0);
}

#[test]
fn decode_after_pop_restores_logits() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2);
    seq.extend(&tokens[..2]);
    seq.pop().unwrap();
    assert!(seq.logits().is_none());

    assert!(seq.decode());

    assert_eq!(
        seq.logits().map(|l| l.len()),
        Some(model.n_tokens() as usize)
    );
    assert_eq!(seq.len(), 1);
    assert_eq!(seq.pos_max(), 0);
}

#[test]
fn decode_keeps_positions_consecutive_for_next_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2);
    seq.push(tokens[0]);
    assert!(seq.decode());

    // The re-decode must not leave a duplicate or a hole in the KV cache,
    // otherwise this push is rejected by llama.cpp.
    seq.push(tokens[1]);
    assert_eq!(seq.len(), 2);
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), 1);
}

#[test]
fn decode_on_empty_sequence_is_a_noop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(!seq.decode());
    assert!(seq.logits().is_none());
    assert_eq!(seq.len(), 0);
}
