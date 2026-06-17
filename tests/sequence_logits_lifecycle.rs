// Coverage for the logits-invalidation contract of `Sequence`.
//
// `Sequence` caches the logits from the last successful decode so that
// `sample()` and `logits()` don't need a second round-trip to the context
// actor. Mutating operations that change what "the next" logits would be —
// `pop()`, `remove()`, `kv_remove()`, `kv_shift()`, and the destination of
// `kv_copy()` — must clear that cache so callers don't sample stale data.
//
// Before this file, none of that invalidation behaviour was exercised by
// the test suite. The cache could silently go stale and tests would still
// pass.

mod common;

use rusty_llama::Context;

#[test]
fn pop_clears_cached_logits() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some(), "logits cached after push");

    let _ = seq.pop();
    assert!(
        seq.logits().is_none(),
        "pop() must invalidate cached logits"
    );
}

#[test]
fn remove_clears_cached_logits() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2, "test prompt must tokenize to >=2 tokens");
    seq.extend(&tokens);
    assert!(seq.logits().is_some());

    let ok = seq.remove(0..1);
    assert!(ok);
    assert!(
        seq.logits().is_none(),
        "remove() must invalidate cached logits"
    );
}

#[test]
fn kv_remove_clears_cached_logits_when_successful() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some());

    let end = seq.len() as i32;
    let ok = seq.kv_remove((end - 1)..end);
    assert!(ok, "kv_remove of the last position should succeed");
    assert!(
        seq.logits().is_none(),
        "kv_remove must invalidate cached logits when it succeeds"
    );
}

#[test]
fn kv_shift_clears_cached_logits() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    // kv_shift requires the underlying memory implementation to support shifting.
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();

    if !ctx.can_shift() {
        // Some kv-cache backends don't implement shift; skip in that case so the
        // test remains portable across configurations.
        return;
    }

    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some());

    let end = seq.len() as i32;
    seq.kv_shift(0..end, 0);
    assert!(
        seq.logits().is_none(),
        "kv_shift must invalidate cached logits"
    );
}

#[test]
fn kv_copy_clears_destination_cached_logits() {
    let (model, _) = common::load_model_and_context();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();

    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);

    assert!(dst.logits().is_some(), "dst has logits before copy");

    let end = src.len() as i32;
    src.kv_copy(&mut dst, 0..end);
    assert!(
        dst.logits().is_none(),
        "kv_copy must invalidate the destination's cached logits"
    );
    // The source is purely read; its logits should stay intact.
    assert!(
        src.logits().is_some(),
        "kv_copy must not touch the source's cached logits"
    );
}

#[test]
fn pop_until_empty_clears_logits_each_step() {
    // Each pop along the way must clear the cache, even though we never
    // call sample() in between.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world!", false, false);
    assert!(tokens.len() >= 2);
    seq.extend(&tokens);

    while !seq.is_empty() {
        // Cache should always be invalidated immediately after pop, regardless
        // of whether anything was queried in between.
        let _ = seq.pop();
        assert!(
            seq.logits().is_none(),
            "logits must be invalidated after every pop"
        );
    }
}
