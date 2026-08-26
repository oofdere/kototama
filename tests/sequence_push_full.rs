mod common;

use rusty_llama::{Context, ContextParams, DecodeError, Model, Sequence};

/// A deliberately small context, so the KV cache fills up quickly.
fn setup() -> (Model, ContextParams) {
    let model = common::load_model();
    let mut params = ContextParams::new();
    params.n_ctx = 64;
    params.n_batch = 64;
    params.n_seq_max = 1;
    params.no_perf = true;
    (model, params)
}

/// Push the same token until the context is full. Returns the number of
/// tokens that were accepted and the error that stopped the loop.
fn fill(seq: &mut Sequence) -> (usize, DecodeError) {
    // The KV cache is padded, so the usable capacity is larger than `n_ctx`.
    for _ in 0..4096 {
        if let Err(e) = seq.push(1) {
            return (seq.len(), e);
        }
    }
    panic!("the context never filled up");
}

#[test]
fn push_reports_a_full_context_instead_of_panicking() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let (accepted, err) = fill(&mut seq);

    assert!(accepted >= params.n_ctx as usize);
    assert!(
        matches!(err, DecodeError::SlotNotFound),
        "unexpected error: {err:?}"
    );
}

#[test]
fn a_rejected_push_does_not_change_the_sequence() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let (accepted, _) = fill(&mut seq);

    assert_eq!(seq.len(), accepted);
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), accepted as i32 - 1);
    // The logits of the last accepted token are still the current ones.
    assert!(seq.logits().is_some());

    assert!(seq.push(1).is_err());
    assert_eq!(seq.len(), accepted);
    assert_eq!(seq.pos_max(), accepted as i32 - 1);
    assert!(seq.logits().is_some());
}

#[test]
fn a_sequence_stays_usable_after_a_rejected_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let (accepted, _) = fill(&mut seq);

    assert_eq!(seq.pop(), Some(1));
    assert_eq!(seq.len(), accepted - 1);
    assert!(seq.push(2).is_ok());
    assert_eq!(seq.len(), accepted);
    assert_eq!(seq.get(accepted - 1), Some(2));
}

#[test]
fn extend_stops_at_the_first_rejected_token() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();

    let capacity = {
        let mut seq = ctx.sequence().unwrap();
        fill(&mut seq).0
    };

    let mut seq = ctx.sequence().unwrap();
    let tokens = vec![1; capacity + 8];
    let result = seq.extend(&tokens);

    assert!(result.is_err());
    assert_eq!(seq.len(), capacity);
    assert_eq!(seq.pos_max(), capacity as i32 - 1);
}

#[test]
fn push_and_extend_succeed_below_the_context_size() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    assert!(seq.push(1).is_ok());
    assert!(seq.extend(&[2, 3, 4]).is_ok());
    assert_eq!(seq.len(), 4);
    assert!(seq.logits().is_some());
}
