//! Regression tests for the `kv_unified` precondition on cross-sequence
//! `Sequence::kv_copy` / `copy_to` / `copy_from`.
//!
//! Without the fix, calling these methods between two distinct sequences on a
//! `Context` built with the default `kv_unified = false` triggers
//! `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`
//! inside `llama_kv_cache::seq_cp` (see
//! `llama-sys/llama.cpp/src/llama-kv-cache.cpp:456`). That assert routes
//! through `ggml_abort()` / `std::abort()`, bypassing Rust unwinding and any
//! `Drop` impl — a process abort reachable from purely safe Rust.
//!
//! Placed in its own test target because the pre-existing `tests/sequence.rs`
//! doesn't compile on this branch (it references the removed `SamplerChain`
//! API from the in-progress "native sampler wip" refactor).

use rusty_llama::Context;
use std::panic::{catch_unwind, AssertUnwindSafe};

mod common;

fn ctx_params_non_unified() -> rusty_llama::ContextParams {
    let mut p = common::test_ctx_params();
    // The llama.cpp default — exactly the configuration that triggers the
    // abort below. Make it explicit so the test stays meaningful if the
    // upstream default ever changes.
    p.kv_unified = false;
    p
}

fn ctx_params_unified() -> rusty_llama::ContextParams {
    let mut p = common::test_ctx_params();
    p.kv_unified = true;
    p
}

// ---------- The regression: cross-seq copy on non-unified KV ----------

#[test]
fn kv_copy_across_distinct_seqs_panics_when_not_unified() {
    let model = common::load_model();
    let params = ctx_params_non_unified();
    let ctx = Context::new(&model, &params).unwrap();
    assert!(!ctx.kv_unified(), "precondition: kv_unified should be false");

    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    // Two distinct checkouts from the same context — different seq ids.

    let result = catch_unwind(AssertUnwindSafe(|| {
        src.kv_copy(&mut dst, 0..1);
    }));
    assert!(
        result.is_err(),
        "kv_copy between distinct seqs without kv_unified must panic in Rust \
         instead of aborting the process via GGML_ASSERT"
    );
}

#[test]
fn copy_to_across_distinct_seqs_panics_when_not_unified() {
    let model = common::load_model();
    let params = ctx_params_non_unified();
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);

    let result = catch_unwind(AssertUnwindSafe(|| {
        src.copy_to(&mut dst, 0..tokens.len());
    }));
    assert!(
        result.is_err(),
        "copy_to between distinct seqs without kv_unified must panic in Rust"
    );
}

#[test]
fn copy_from_across_distinct_seqs_panics_when_not_unified() {
    let model = common::load_model();
    let params = ctx_params_non_unified();
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);

    let result = catch_unwind(AssertUnwindSafe(|| {
        dst.copy_from(&src, 0..tokens.len());
    }));
    assert!(
        result.is_err(),
        "copy_from between distinct seqs without kv_unified must panic in Rust"
    );
}

// ---------- The pre-panic state stays observable (Drop runs) ----------

#[test]
fn copy_to_panic_does_not_clobber_destination_state() {
    let model = common::load_model();
    let params = ctx_params_non_unified();
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let src_tokens = model.tokenize("hi", false, false);
    let dst_tokens = model.tokenize("ok", false, false);
    src.extend(&src_tokens);
    dst.extend(&dst_tokens);
    let dst_tokens_before = dst.tokens().to_vec();
    let dst_logits_were_some = dst.logits().is_some();

    let result = catch_unwind(AssertUnwindSafe(|| {
        src.copy_to(&mut dst, 0..src_tokens.len());
    }));
    assert!(result.is_err());

    // The Rust-side guard fires before any FFI mutation or local state
    // bookkeeping, so dst's tokens and logits should be untouched.
    assert_eq!(
        dst.tokens(),
        dst_tokens_before.as_slice(),
        "dst.tokens must be unchanged after a panicking copy_to"
    );
    assert_eq!(
        dst.logits().is_some(),
        dst_logits_were_some,
        "dst.logits invalidation must not happen on the panicking path"
    );
}

// ---------- Positive paths the fix must not regress ----------

#[test]
fn kv_copy_across_distinct_seqs_works_when_unified() {
    let model = common::load_model();
    let params = ctx_params_unified();
    let ctx = Context::new(&model, &params).unwrap();
    assert!(ctx.kv_unified());

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);

    // Must not panic / abort.
    src.copy_to(&mut dst, 0..tokens.len());
    assert_eq!(dst.tokens(), src.tokens());
}

// ---------- Cached flag matches what we asked for ----------

#[test]
fn context_kv_unified_reflects_params() {
    let model = common::load_model();

    let ctx_off = Context::new(&model, &ctx_params_non_unified()).unwrap();
    assert!(!ctx_off.kv_unified());

    let ctx_on = Context::new(&model, &ctx_params_unified()).unwrap();
    assert!(ctx_on.kv_unified());
}

#[test]
fn context_kv_unified_is_consistent_across_clones() {
    let model = common::load_model();
    let ctx = Context::new(&model, &ctx_params_unified()).unwrap();
    let ctx2 = ctx.clone();
    assert_eq!(ctx.kv_unified(), ctx2.kv_unified());
}
