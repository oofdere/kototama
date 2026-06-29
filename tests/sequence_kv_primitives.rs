//! Coverage for the tokens-vs-KV-cache decoupling contract of
//! `Sequence::kv_remove` and `Sequence::kv_copy`.
//!
//! `Sequence` exposes two layers over the underlying KV cache:
//!
//!   * High-level: `pop()`, `remove(Range<usize>)`, `copy_to(&mut Self,
//!     Range<usize>)`. These keep `self.tokens` in lockstep with the KV
//!     cache — popping a token both frees the KV slot AND drops the
//!     token from `self.tokens`.
//!
//!   * Low-level: `kv_remove(Range<i32>)`, `kv_copy(&mut Self,
//!     Range<i32>)`, `kv_shift(Range<i32>, i32)`. These touch ONLY the
//!     KV cache and intentionally leave `self.tokens` untouched, so
//!     advanced callers can manage cache layout without the wrapper
//!     mirroring tokens for them.
//!
//! That decoupling is a load-bearing part of why the `kv_*` methods are
//! `pub` and distinct from their high-level wrappers. If a future
//! refactor "helpfully" added `self.tokens.drain(...)` to `kv_remove` or
//! made `kv_copy` mirror tokens, every advanced caller would silently
//! break.
//!
//! PR #119 (`tests/sequence_logits_lifecycle.rs`) covers the orthogonal
//! contract that these primitives invalidate cached logits. This file
//! covers the tokens-decoupling contract that #119 does not assert.

mod common;

use rusty_llama::Context;

// ---------- kv_remove: KV cache only, not self.tokens ----------

#[test]
fn kv_remove_does_not_shrink_tokens() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2, "test prompt must tokenize to >=2 tokens");
    seq.extend(&tokens);
    let len_before = seq.len();
    let tokens_before: Vec<i32> = seq.tokens().to_vec();

    let end = seq.len() as i32;
    let ok = seq.kv_remove((end - 1)..end);
    assert!(ok, "kv_remove of the last position should succeed");

    assert_eq!(
        seq.len(),
        len_before,
        "kv_remove must not change Sequence::len (it touches KV cache only)"
    );
    assert_eq!(
        seq.tokens(),
        tokens_before.as_slice(),
        "kv_remove must not modify Sequence::tokens (use remove() for that)"
    );
}

#[test]
fn kv_remove_full_range_leaves_tokens_intact() {
    // Wiping the whole KV range for a sequence must still leave the
    // user-facing token vector untouched. After this, `seq.tokens()`
    // continues to report the original tokens even though the KV cache
    // is empty — the intentional inconsistent state that distinguishes
    // the kv_* primitives from the high-level wrappers.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let tokens_before: Vec<i32> = seq.tokens().to_vec();

    let end = seq.len() as i32;
    let ok = seq.kv_remove(0..end);
    assert!(ok, "full-range kv_remove should succeed");

    assert_eq!(
        seq.len(),
        len_before,
        "kv_remove of the full range must not shrink Sequence::len"
    );
    assert_eq!(seq.tokens(), tokens_before.as_slice());
    assert!(
        !seq.is_empty(),
        "is_empty reflects the tokens vec, not the KV cache — kv_remove must not flip it"
    );
}

#[test]
fn kv_remove_empty_range_is_noop_on_tokens() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let tokens_before: Vec<i32> = seq.tokens().to_vec();

    // An empty range (start == end) is a degenerate input — it must not
    // mutate observable state.
    let _ = seq.kv_remove(0..0);

    assert_eq!(seq.len(), len_before);
    assert_eq!(seq.tokens(), tokens_before.as_slice());
}

#[test]
fn kv_remove_takes_i32_range_not_usize() {
    // This test is primarily a compile-time pin: `kv_remove` accepts
    // `Range<i32>` because positions in the KV cache are signed
    // (llama.cpp uses -1 as "from the start" / "to the end" sentinels in
    // related APIs). The high-level `remove()` takes `Range<usize>`.
    // If someone "harmonised" the signatures, this would stop compiling.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);

    let range: std::ops::Range<i32> = 0..(seq.len() as i32);
    let _ok: bool = seq.kv_remove(range);
}

// ---------- kv_copy: destination's KV cache only, not its tokens ----------

#[test]
fn kv_copy_does_not_overwrite_destination_tokens() {
    // `kv_copy` is the low-level primitive used by `copy_to`. Unlike
    // `copy_to` (which clears `other.tokens` and refills it from the
    // source's slice), `kv_copy` must leave `other.tokens` untouched.
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    let src_tokens = model.tokenize("hello", false, false);
    let dst_tokens = model.tokenize("world", false, false);
    src.extend(&src_tokens);
    dst.extend(&dst_tokens);

    let dst_tokens_before: Vec<i32> = dst.tokens().to_vec();
    let dst_len_before = dst.len();

    let end = src.len().min(dst.len()) as i32;
    src.kv_copy(&mut dst, 0..end);

    assert_eq!(
        dst.tokens(),
        dst_tokens_before.as_slice(),
        "kv_copy must not modify the destination's token vector"
    );
    assert_eq!(
        dst.len(),
        dst_len_before,
        "kv_copy must not change the destination's Sequence::len"
    );
}

#[test]
fn kv_copy_does_not_touch_source_tokens_or_len() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    let src_tokens = model.tokenize("hello world", false, false);
    let dst_tokens = model.tokenize("foo bar", false, false);
    src.extend(&src_tokens);
    dst.extend(&dst_tokens);

    let src_tokens_before: Vec<i32> = src.tokens().to_vec();
    let src_len_before = src.len();

    let end = src.len().min(dst.len()) as i32;
    src.kv_copy(&mut dst, 0..end);

    assert_eq!(
        src.tokens(),
        src_tokens_before.as_slice(),
        "kv_copy must not mutate the source's token vector"
    );
    assert_eq!(
        src.len(),
        src_len_before,
        "kv_copy must not change the source's Sequence::len"
    );
}

#[test]
fn kv_copy_takes_shared_borrow_on_source() {
    // `kv_copy` takes `&self` on the source (only `&mut self` on the
    // destination). That makes it usable while other shared borrows of
    // the source are alive — e.g. holding `src.logits()` across the
    // call. Pin this so a future refactor doesn't tighten the signature
    // to `&mut self`.
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);

    // Acquire and hold a shared borrow on the source.
    let logits_borrow = src.logits().expect("logits cached after extend");
    let logits_len_before = logits_borrow.len();

    let end = tokens.len() as i32;
    src.kv_copy(&mut dst, 0..end);

    // The shared borrow must still be valid for use after the call.
    assert_eq!(logits_borrow.len(), logits_len_before);
}

// ---------- High-level wrappers stay in sync (regression guard) ----------

#[test]
fn remove_wrapper_shrinks_tokens_while_kv_remove_does_not() {
    // Pin the contrast: `remove()` is the wrapper that drains
    // `self.tokens`, `kv_remove()` is the primitive that does not.
    // This is a regression guard against the two methods being
    // accidentally collapsed.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();

    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2);

    // Path A: high-level remove() shrinks Sequence::tokens.
    let mut seq_a = ctx.sequence().unwrap();
    seq_a.extend(&tokens);
    let a_len_before = seq_a.len();
    let ok_a = seq_a.remove(0..1);
    assert!(ok_a);
    assert_eq!(
        seq_a.len(),
        a_len_before - 1,
        "remove() must shrink Sequence::len by the range length"
    );

    // Path B: low-level kv_remove() leaves Sequence::tokens intact.
    let mut seq_b = ctx.sequence().unwrap();
    seq_b.extend(&tokens);
    let b_len_before = seq_b.len();
    let ok_b = seq_b.kv_remove(0..1);
    assert!(ok_b);
    assert_eq!(
        seq_b.len(),
        b_len_before,
        "kv_remove must NOT shrink Sequence::len — that's remove()'s job"
    );
}

#[test]
fn copy_to_wrapper_mirrors_tokens_while_kv_copy_does_not() {
    // Companion to the kv_remove vs remove pin above.
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();

    let src_tokens = model.tokenize("hello", false, false);
    let dst_tokens = model.tokenize("world", false, false);
    let n = src_tokens.len().min(dst_tokens.len());

    // Path A: high-level copy_to() rewrites dst.tokens from src.
    let mut src_a = ctx.sequence().unwrap();
    let mut dst_a = ctx.sequence().unwrap();
    src_a.extend(&src_tokens);
    dst_a.extend(&dst_tokens);
    src_a.copy_to(&mut dst_a, 0..n);
    assert_eq!(
        dst_a.tokens(),
        &src_tokens[..n],
        "copy_to() must mirror the source's tokens into the destination"
    );

    // Path B: low-level kv_copy() leaves dst.tokens unchanged.
    let mut src_b = ctx.sequence().unwrap();
    let mut dst_b = ctx.sequence().unwrap();
    src_b.extend(&src_tokens);
    dst_b.extend(&dst_tokens);
    let dst_b_tokens_before: Vec<i32> = dst_b.tokens().to_vec();
    src_b.kv_copy(&mut dst_b, 0..n as i32);
    assert_eq!(
        dst_b.tokens(),
        dst_b_tokens_before.as_slice(),
        "kv_copy must NOT mirror the source's tokens — that's copy_to()'s job"
    );
}
