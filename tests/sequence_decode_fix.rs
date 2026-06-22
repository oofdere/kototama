mod common;

use rusty_llama::Context;

#[test]
fn decode_after_pop_repopulates_logits() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    seq.pop();
    assert!(seq.logits().is_none(), "pop() invalidates logits");
    seq.decode();
    assert!(
        seq.logits().is_some(),
        "decode() must repopulate logits after pop()"
    );
    assert_eq!(
        seq.logits().unwrap().len(),
        model.n_tokens() as usize,
        "decoded logits should match vocab size"
    );
}

#[test]
fn decode_after_trailing_remove_repopulates_logits() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    let n = seq.len();
    let ok = seq.remove((n - 1)..n);
    assert!(ok);
    assert!(seq.logits().is_none());
    seq.decode();
    assert!(
        seq.logits().is_some(),
        "decode() must repopulate logits after a trailing remove()"
    );
}

#[test]
fn decode_on_empty_sequence_is_noop() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(seq.is_empty());
    seq.decode();
    assert!(seq.is_empty(), "decode() on empty sequence must not push a token");
    assert!(
        seq.logits().is_none(),
        "decode() on empty sequence must not fabricate logits"
    );
}

#[test]
fn decode_preserves_token_vector() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let tokens_before = seq.tokens().to_vec();
    seq.decode();
    assert_eq!(seq.len(), len_before, "decode() must not change len()");
    assert_eq!(
        seq.tokens(),
        tokens_before.as_slice(),
        "decode() must not change the token vector"
    );
}
