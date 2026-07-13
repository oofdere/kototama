use rusty_llama::test_common::load_model_and_context;
use rusty_llama::Context;

// Regression test for `Sequence::copy_to`.
//
// `Sequence` maintains the invariant that a token's index in the local
// `tokens` vector equals its position in the KV cache (`push` decodes at
// `pos == tokens.len()`). `copy_to` resets the destination's `tokens` vector
// to a fresh 0-indexed slice, so the copied KV cells must also be rebased to
// start at position 0. Otherwise a later `push` on the destination reuses an
// already-occupied KV position.

#[test]
fn copy_to_rebases_kv_positions_to_zero() {
    let (model, mut params) = load_model_and_context();
    // `seq_cp` requires a unified (full) KV buffer.
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world this is a test", false, false);
    assert!(tokens.len() >= 4, "need at least 4 tokens, got {}", tokens.len());
    src.extend(&tokens);

    let mut dst = ctx.sequence().unwrap();
    // Copy an interior range that does NOT start at 0.
    src.copy_to(&mut dst, 1..3);

    assert_eq!(dst.len(), 2);

    // The copied cells must be rebased to [0, len): index == position.
    assert_eq!(dst.pos_min(), 0, "dst pos_min must be 0 after copy_to");
    assert_eq!(
        dst.pos_max(),
        (dst.len() - 1) as i32,
        "dst pos_max must be len-1 after copy_to"
    );

    // A subsequent push must land at the next free position and keep the
    // invariant intact (no collision with the copied cells).
    dst.push(tokens[0]);
    assert_eq!(dst.pos_min(), 0);
    assert_eq!(dst.pos_max(), (dst.len() - 1) as i32);
}
