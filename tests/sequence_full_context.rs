mod common;

use rusty_llama::{Context, DecodeError};

/// Filling the context window must surface a `DecodeError` instead of
/// panicking, and must leave the sequence usable.
#[test]
fn push_past_context_window_returns_error() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.n_ctx = 64;
    params.n_batch = 64;
    params.n_seq_max = 1;

    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let token = model.tokenize("hello", false, false)[0];

    let n_ctx = ctx.n_ctx() as usize;
    let mut err = None;
    for _ in 0..n_ctx + 8 {
        if let Err(e) = seq.push(token) {
            err = Some(e);
            break;
        }
    }

    let err = err.expect("pushing past the context window should fail");
    assert!(
        matches!(err, DecodeError::SlotNotFound),
        "unexpected error: {err:?}"
    );
    assert_eq!(
        seq.len(),
        n_ctx,
        "the failed push must not be recorded in the token list"
    );
    assert_eq!(seq.pos_max(), n_ctx as i32 - 1);

    // The sequence is still usable: freeing a slot lets decoding continue.
    assert!(seq.pop().is_some());
    seq.push(token).expect("push should succeed after pop");
}

#[test]
fn extend_past_context_window_returns_error() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.n_ctx = 64;
    params.n_batch = 64;
    params.n_seq_max = 1;

    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let token = model.tokenize("hello", false, false)[0];

    let n_ctx = ctx.n_ctx() as usize;
    let tokens = vec![token; n_ctx + 8];
    assert!(seq.extend(&tokens).is_err());
    assert_eq!(seq.len(), n_ctx, "tokens decoded before the failure remain");
}

#[test]
fn decode_on_empty_sequence_is_ok() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(seq.decode().is_ok());
    assert!(seq.logits().is_none());
}
