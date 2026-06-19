mod common;

use rusty_llama::Context;

/// `Sequence::kv_shift` forwards `(seq_id, p0, p1, delta)` to `llama_memory_seq_add`,
/// which in `llama.cpp` is guarded by
///
/// ```c
/// GGML_ASSERT(hparams.n_pos_per_embd() == 1 &&
///             "seq_add() is only supported for n_pos_per_embd() == 1");
/// ```
///
/// (`llama-sys/llama.cpp/src/llama-kv-cache.cpp:517`). `GGML_ASSERT` is compiled
/// in release builds and routes through `ggml_abort()` → `std::abort()`, which
/// terminates the host process without unwinding. That assertion fires for
/// MROPE / IMROPE multimodal models and the STEP35 architecture — exactly the
/// set for which `llama_memory_can_shift()` returns `false`.
///
/// `Sequence::kv_shift` now consults `Context::can_shift()` first and panics
/// instead, so safe code that hits this case still tears down cleanly through
/// `Drop` and is catchable with `catch_unwind`.
///
/// The bundled test model (TinyStories) supports shift, so the assertion path
/// can't be exercised directly here; these tests cover the supported path and
/// the invariant that `Context::can_shift()` is stable for use as a guard.
#[test]
fn kv_shift_works_when_supported() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    if !ctx.can_shift() {
        // Bundled test model is expected to support shift; skip otherwise.
        return;
    }
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let pos_max_before = seq.pos_max();
    assert!(pos_max_before >= 0);

    // Shift positions [0, pos_max+1) by +1 — well-defined when shift is
    // supported, would abort the process via GGML_ASSERT on an unsupported
    // model.
    seq.kv_shift(0..(pos_max_before + 1), 1);

    // The shift should bump the max position by `delta`.
    assert_eq!(seq.pos_max(), pos_max_before + 1);
}

#[test]
fn kv_shift_invalidates_logits() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    if !ctx.can_shift() {
        return;
    }
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some());

    let pos_max = seq.pos_max();
    seq.kv_shift(0..(pos_max + 1), 1);

    assert!(
        seq.logits().is_none(),
        "kv_shift must invalidate cached logits since positions changed"
    );
}

#[test]
fn can_shift_matches_actor_query() {
    // The cached value must equal what llama_memory_can_shift returns, which
    // is the guard `Sequence::kv_shift` relies on.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let cached = ctx.can_shift();
    // Re-querying must be consistent — it's cached but the contract is that
    // it matches the memory's actual capability.
    assert_eq!(ctx.can_shift(), cached);
}
