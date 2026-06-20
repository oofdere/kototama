mod common;

use rusty_llama::Context;

// Regression tests for partial-state corruption in `Sequence::remove` and
// `Sequence::copy_to` when called with an out-of-bounds `range`.
//
// Before the fix, both methods performed the FFI mutation (`llama_memory_seq_rm`
// / `llama_memory_seq_cp`) BEFORE checking that `range` fits in the local
// `tokens` Vec. llama.cpp silently clips out-of-range positions, so the FFI
// call would succeed and only THEN the Rust-side slice/drain would panic —
// leaving:
//
//  * `Sequence::remove`: KV cache mutated, local `tokens` not drained →
//    next `push()` writes to a position the KV cache already holds, silently
//    corrupting attention.
//  * `Sequence::copy_to`: KV cache copied, destination `other.tokens` cleared,
//    destination `other.logits` still pointing at data unrelated to the now-
//    empty `tokens` → a subsequent `sample()` on `other` samples from stale
//    logits while the user reasonably assumes `other` is clean.
//
// After the fix, both methods bounds-check `range` up-front and panic with a
// clear message before any FFI call, so `Drop` runs and the data structures
// stay consistent.

#[test]
#[should_panic(expected = "exceeds sequence length")]
fn remove_panics_when_range_end_exceeds_len() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    // Range goes well past `seq.len()`.
    seq.remove(0..(seq.len() + 5));
}

#[test]
#[should_panic(expected = "exceeds sequence length")]
fn remove_panics_when_range_end_exceeds_len_on_empty_seq() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    // Range end > 0 on an empty sequence.
    seq.remove(0..1);
}

#[test]
#[should_panic(expected = "range start")]
fn remove_panics_when_range_start_gt_end() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    seq.remove(3..1);
}

#[test]
#[should_panic(expected = "exceeds sequence length")]
fn copy_to_panics_when_range_end_exceeds_src_len() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    // Range goes past `src.len()`.
    src.copy_to(&mut dst, 0..(src.len() + 10));
}

#[test]
#[should_panic(expected = "range start")]
fn copy_to_panics_when_range_start_gt_end() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    src.extend(&tokens);
    src.copy_to(&mut dst, 5..2);
}

// A bad-range `copy_to` must NOT clear `other.tokens` or invalidate FFI
// state. Without the fix, `kv_copy` would run, `other.tokens.clear()` would
// empty the destination, then the slice access would panic — leaving the
// destination in a broken state observable via `catch_unwind`.
#[test]
fn copy_to_bad_range_leaves_destination_unchanged() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let src_tokens = model.tokenize("hi", false, false);
    let dst_tokens = model.tokenize("hello world", false, false);
    src.extend(&src_tokens);
    dst.extend(&dst_tokens);

    let dst_tokens_before: Vec<i32> = dst.tokens().to_vec();
    let dst_had_logits = dst.logits().is_some();

    // Attempt a bad copy. Catch the panic so we can inspect `dst`.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        src.copy_to(&mut dst, 0..(src.len() + 50));
    }));
    assert!(res.is_err(), "out-of-range copy_to must panic");

    // After the panic, `dst` must be unchanged — neither `tokens` cleared
    // nor `logits` invalidated.
    assert_eq!(
        dst.tokens(),
        dst_tokens_before.as_slice(),
        "dst.tokens must be unchanged after a panicking copy_to"
    );
    assert_eq!(
        dst.logits().is_some(),
        dst_had_logits,
        "dst.logits must be unchanged after a panicking copy_to"
    );
}

// The happy path still works.
#[test]
fn remove_valid_range_works() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    let n = seq.len();
    assert!(seq.remove(0..1));
    assert_eq!(seq.len(), n - 1);
}

#[test]
fn remove_empty_range_is_noop_on_local_state() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    let n = seq.len();
    // Empty range at the end — within bounds, no-op.
    let _ = seq.remove(n..n);
    assert_eq!(seq.len(), n);
}

#[test]
fn copy_to_valid_full_range_works() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);
    src.copy_to(&mut dst, 0..src.len());
    assert_eq!(dst.tokens(), src.tokens());
}
