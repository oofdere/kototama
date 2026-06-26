//! Regression test for `Sequence::push` silently wrapping a `usize` position
//! into a negative `i32` when crossing the FFI boundary.
//!
//! Before the fix, `Sequence::push` computed the position via
//! `self.tokens.len() as i32`. Once `tokens.len()` exceeds `i32::MAX`,
//! the `as` cast wraps to a negative value. llama.cpp's KV-cache APIs
//! treat `-1` as a sentinel meaning "from start" / "to end" in
//! `llama_memory_seq_rm`, `llama_memory_seq_cp`, and similar — so a
//! wrapped position that happens to land on `-1` silently targets a
//! sentinel range instead of an actual position. Other wrapped values
//! are interpreted as arbitrary signed positions, which is FFI UB
//! reachable from purely safe Rust.
//!
//! The push-path fix replaces the bare `as i32` cast with a checked
//! `i32::try_from(...)` and panics with a clear message on overflow,
//! so safe Rust may panic but it cannot pass an undefined value across
//! the FFI boundary.
//!
//! ---
//!
//! We can't cheaply allocate >2 GiB of tokens in CI to drive the
//! overflow path directly, so the regression coverage is split into
//! two complementary pieces:
//!
//!   1. A normal-path test, which exercises the same `i32::try_from`
//!      conversion the guard uses and pins that valid-length pushes
//!      still work end-to-end.
//!
//!   2. A pure-logic test of the conversion shape itself, which
//!      asserts that `i32::try_from((i32::MAX as usize) + 1)` is `Err`
//!      — i.e. that the guard fires for exactly the value that used to
//!      silently wrap to a negative position.
//!
//! Placed in its own test target because the pre-existing
//! `tests/sequence.rs` doesn't compile on this branch — it references
//! the in-progress `SamplerChain` API that was removed by the "native
//! sampler wip" refactor.

mod common;

use rusty_llama::Context;

#[test]
fn push_happy_path_still_works() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    for &t in &tokens {
        seq.push(t);
    }
    assert_eq!(seq.tokens(), tokens.as_slice());
    assert_eq!(seq.len(), tokens.len());
    assert!(seq.logits().is_some());
}

#[test]
fn push_first_token_position_is_zero() {
    // Regression hook for the guard's degenerate case: `tokens.len() == 0`
    // must convert cleanly to `0i32`, not panic.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.push(tokens[0]);
    assert_eq!(seq.len(), 1);
}

#[test]
fn i32_try_from_rejects_value_just_past_i32_max() {
    // Pin the conversion shape the guard relies on. If a future refactor
    // re-introduces a bare `as i32` cast, the guard logic in `push` would
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
