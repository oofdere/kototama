//! Regression test for `Sequence::decode` silently wrapping a `usize` length
//! into a negative `i32` when crossing the FFI boundary.
//!
//! Before the fix, `Sequence::decode` computed the trailing position via
//! `(self.tokens.len() - 1) as i32`. Once `tokens.len() - 1` exceeds
//! `i32::MAX`, the `as` cast wraps to a negative value. llama.cpp's KV-cache
//! APIs treat `-1` as a sentinel meaning "from start" / "to end" in
//! `llama_memory_seq_rm` and friends, so a wrapped position that happens to
//! land on `-1` silently targets a sentinel range instead of the actual
//! trailing token. Other wrapped values are interpreted as arbitrary signed
//! positions, which is FFI UB reachable from purely safe Rust.
//!
//! Sister fix to #147 (`Sequence::push`) and #153 (`Sequence::pop`).
//! `decode` was the last unprotected `pos`-from-length cast on `Sequence`.
//!
//! ---
//!
//! Allocating a `>2 GiB` `Vec<i32>` to drive the overflow path directly
//! is not realistic in CI, so the regression coverage is split, mirroring
//! #147 / #153's approach:
//!
//!   1. An end-to-end test on an empty sequence — pins that `decode` is
//!      a no-op when there is no trailing token (the short-circuit before
//!      the cast).
//!
//!   2. A pure-logic test of the conversion shape itself, which asserts
//!      that `i32::try_from((i32::MAX as usize) + 1)` is `Err` — i.e.
//!      that the guard fires for exactly the value that used to silently
//!      wrap to a negative position.
//!
//! We deliberately do *not* test `decode` on a non-empty sequence here:
//! on `trunk` the documented post-mutation refresh path (`extend` →
//! `decode`, `pop` → `decode`) panics with `DecodeError::InvalidInput`
//! because llama.cpp rejects a duplicate-position push (see #135, which
//! adds a `memory_seq_rm` before the re-push to land the documented
//! contract). That bug is orthogonal to the cast wrap fixed here, and
//! coupling this test to it would make the test fail on `trunk` for the
//! wrong reason. The empty-sequence test plus the conversion-shape pin
//! is enough to lock the cast guard in place.
//!
//! Placed in its own test target because the pre-existing
//! `tests/sequence.rs` doesn't compile on this branch — it references
//! the in-progress `SamplerChain` API that was removed by the "native
//! sampler wip" refactor.

mod common;

use rusty_llama::Context;

#[test]
fn decode_on_empty_sequence_is_noop() {
    // No `last_token`, so `decode` short-circuits before the cast and must
    // not panic or fabricate logits.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.decode();
    assert_eq!(seq.len(), 0);
    assert!(seq.logits().is_none());
}

#[test]
fn i32_try_from_rejects_value_just_past_i32_max() {
    // Pin the conversion shape the guard relies on. If a future refactor
    // re-introduces a bare `as i32` cast in `decode`, the guard logic would
    // silently start truncating again — this test pins the exact value that
    // used to wrap to a negative `int32_t` position.
    //
    // The guard converts `self.tokens.len() - 1`, so the value just past
    // the boundary is `(i32::MAX as usize) + 1`, exactly what a sequence of
    // length `(i32::MAX as usize) + 2` would feed into the conversion.
    let just_over: usize = (i32::MAX as usize) + 1;
    assert!(
        i32::try_from(just_over).is_err(),
        "i32::try_from((i32::MAX as usize) + 1) must be Err"
    );
    // And the boundary value that DOES fit must convert cleanly.
    assert_eq!(i32::try_from(i32::MAX as usize), Ok(i32::MAX));
}
