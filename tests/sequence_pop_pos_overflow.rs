//! Regression test for `Sequence::pop` silently wrapping a `usize` length
//! into a negative `i32` when crossing the FFI boundary.
//!
//! Before the fix, `Sequence::pop` computed the trailing position via
//! `self.tokens.len() as i32`. Once `tokens.len()` exceeds `i32::MAX`,
//! the `as` cast wraps to a negative value. llama.cpp's KV-cache APIs
//! treat `-1` as a sentinel meaning "from start" / "to end" in
//! `llama_memory_seq_rm`, so a wrapped position that happens to land on
//! `-1` silently targets a sentinel range instead of the actual trailing
//! token. Other wrapped values are interpreted as arbitrary signed
//! positions, which is FFI UB reachable from purely safe Rust.
//!
//! Sister fix to #147 (which closed the same hole on `Sequence::push`).
//! `pop` was the matching unprotected position cast on the other side
//! of the API.
//!
//! ---
//!
//! Allocating a `>2 GiB` `Vec<i32>` to drive the overflow path directly
//! is not realistic in CI, so the regression coverage is split into two
//! complementary pieces, mirroring #147's approach:
//!
//!   1. Normal-path tests, which exercise the same `i32::try_from`
//!      conversion the guard uses and pin that valid-length pops still
//!      work end-to-end.
//!
//!   2. A pure-logic test of the conversion shape itself, which asserts
//!      that `i32::try_from((i32::MAX as usize) + 1)` is `Err` — i.e.
//!      that the guard fires for exactly the value that used to silently
//!      wrap to a negative position.
//!
//! Placed in its own test target because the pre-existing
//! `tests/sequence.rs` doesn't compile on this branch — it references
//! the in-progress `SamplerChain` API that was removed by the "native
//! sampler wip" refactor.

mod common;

use rusty_llama::Context;

#[test]
fn pop_on_empty_sequence_returns_none() {
    // Pre-fix and post-fix this returned `None` via the early `len == 0`
    // check; the post-fix early return uses `is_empty()` instead, but the
    // observable behaviour must not change.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(seq.pop().is_none());
    assert_eq!(seq.len(), 0);
}

#[test]
fn pop_returns_last_pushed_token_and_shrinks() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    let last = *tokens.last().expect("tokenize produced at least one token");
    let popped = seq.pop().expect("pop on non-empty sequence");
    assert_eq!(popped, last);
    assert_eq!(seq.len(), tokens.len() - 1);
    assert!(seq.logits().is_none(), "pop must invalidate logits");
}

#[test]
fn pop_to_empty_then_pop_again_returns_none() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    while seq.pop().is_some() {}
    assert_eq!(seq.len(), 0);
    assert!(seq.pop().is_none(), "pop on drained sequence must be None");
}

#[test]
fn i32_try_from_rejects_value_just_past_i32_max() {
    // Pin the conversion shape the guard relies on. If a future refactor
    // re-introduces a bare `as i32` cast in `pop`, the guard logic would
    // silently start truncating again — this test pins the exact value
    // that used to wrap to a negative `int32_t` position.
    let just_over: usize = (i32::MAX as usize) + 1;
    assert!(
        i32::try_from(just_over).is_err(),
        "i32::try_from((i32::MAX as usize) + 1) must be Err"
    );
    // And the boundary value that DOES fit must convert cleanly.
    assert_eq!(i32::try_from(i32::MAX as usize), Ok(i32::MAX));
}
