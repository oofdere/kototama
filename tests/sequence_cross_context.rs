//! Copying between sequences that belong to different `Context`s.
//!
//! Sequence ids are only meaningful inside the context that issued them, so a
//! cross-context copy used to hand a foreign id to llama.cpp: with an id past
//! the destination context's stream count that trips a `GGML_ASSERT` and aborts
//! the process, and otherwise it silently rewrites KV cells of an unrelated
//! sequence in the *source* context.

mod common;

use rusty_llama::Context;

fn ctx_with_seq_max(model: &rusty_llama::Model, n_seq_max: u32) -> Context {
    let mut params = common::test_ctx_params();
    params.n_seq_max = n_seq_max;
    Context::new(model, &params).unwrap()
}

/// Same-context copies need a unified KV cache: copying between two streams of
/// a non-unified cache is only supported for full buffers in llama.cpp.
fn unified_ctx(model: &rusty_llama::Model, n_seq_max: u32) -> Context {
    let mut params = common::test_ctx_params();
    params.n_seq_max = n_seq_max;
    params.kv_unified = true;
    Context::new(model, &params).unwrap()
}

#[test]
fn copy_to_across_contexts_is_rejected() {
    let model = common::load_model();
    let ctx_a = ctx_with_seq_max(&model, 2);
    let ctx_b = ctx_with_seq_max(&model, 4);

    let mut src = ctx_a.sequence().unwrap();
    // Burn the low ids in ctx_b so that `dst` gets an id outside ctx_a's range.
    let _b0 = ctx_b.sequence().unwrap();
    let _b1 = ctx_b.sequence().unwrap();
    let _b2 = ctx_b.sequence().unwrap();
    let mut dst = ctx_b.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2);
    src.extend(&tokens[..2]);

    // On the unfixed code this aborts the process inside llama.cpp.
    assert!(!src.copy_to(&mut dst, 0..2));
    assert_eq!(dst.len(), 0);
    assert!(dst.tokens().is_empty());
}

#[test]
fn copy_from_across_contexts_is_rejected() {
    let model = common::load_model();
    let ctx_a = ctx_with_seq_max(&model, 2);
    let ctx_b = ctx_with_seq_max(&model, 4);

    let mut src = ctx_a.sequence().unwrap();
    let _b0 = ctx_b.sequence().unwrap();
    let _b1 = ctx_b.sequence().unwrap();
    let _b2 = ctx_b.sequence().unwrap();
    let mut dst = ctx_b.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    src.extend(&tokens[..2]);

    assert!(!dst.copy_from(&src, 0..2));
    assert_eq!(dst.len(), 0);
}

#[test]
fn kv_copy_across_contexts_is_rejected() {
    let model = common::load_model();
    let ctx_a = ctx_with_seq_max(&model, 2);
    let ctx_b = ctx_with_seq_max(&model, 4);

    let mut src = ctx_a.sequence().unwrap();
    let _b0 = ctx_b.sequence().unwrap();
    let _b1 = ctx_b.sequence().unwrap();
    let _b2 = ctx_b.sequence().unwrap();
    let mut dst = ctx_b.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    src.extend(&tokens[..2]);

    assert!(!src.kv_copy(&mut dst, 0..2));
}

#[test]
fn cross_context_copy_leaves_source_context_sequences_intact() {
    let model = common::load_model();
    let ctx_a = ctx_with_seq_max(&model, 2);
    let ctx_b = ctx_with_seq_max(&model, 2);

    let mut a0 = ctx_a.sequence().unwrap();
    let mut a1 = ctx_a.sequence().unwrap();
    let mut b1 = {
        let _b0 = ctx_b.sequence().unwrap();
        ctx_b.sequence().unwrap()
    };

    let tokens = model.tokenize("hello world", false, false);
    a0.extend(&tokens[..2]);
    a1.extend(&tokens[..2]);
    let a1_pos_max = a1.pos_max();

    // a1 shares ctx_a's stream 1 with b1's id; the copy must not touch it.
    assert!(!a0.copy_to(&mut b1, 0..2));
    assert_eq!(a1.pos_max(), a1_pos_max);
    assert_eq!(a1.len(), 2);
}

#[test]
fn same_context_copy_still_works() {
    let model = common::load_model();
    let ctx = unified_ctx(&model, 2);

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    src.extend(&tokens[..2]);

    assert!(src.copy_to(&mut dst, 0..2));
    assert_eq!(dst.tokens(), &tokens[..2]);
}

#[test]
fn cloned_context_handles_count_as_the_same_context() {
    let model = common::load_model();
    let ctx = unified_ctx(&model, 2);
    let ctx_clone = ctx.clone();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx_clone.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    src.extend(&tokens[..2]);

    assert!(src.copy_to(&mut dst, 0..2));
    assert_eq!(dst.tokens(), &tokens[..2]);
}
