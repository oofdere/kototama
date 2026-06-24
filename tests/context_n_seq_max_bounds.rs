mod common;

use rusty_llama::{Context, ContextParams};

// Regression test for an FFI-soundness bug in `Context::new`:
//
// `ContextParams::n_seq_max` is a public `u32` field on the
// llama-bindgen'd `llama_context_params`. `Context::new` casts it to
// `i32` via `as` when handing it to `Batch::init_token`, which in turn
// passes it to `llama_batch_init(int32_t n_seq_max)`. For any
// `n_seq_max > i32::MAX as u32`, the `as` cast silently wraps to a
// negative `int32_t`. llama.cpp's batch allocator then computes
// `sizeof(llama_seq_id) * n_seq_max` where the signed `int32_t` is
// promoted to `size_t` — wrapping to an enormous number under modular
// `size_t` arithmetic. The resulting allocations are either NULL or
// successful but smaller than expected, and the actor's
// `common::batch_add` happily writes through those pointers without
// re-checking the buffer size — undefined behaviour reachable from
// purely safe Rust.
//
// The fix is to reject `params.n_seq_max > i32::MAX as u32` in
// `Context::new` before any FFI call, returning `Err(())`.
//
// (Placed in its own test target because the pre-existing
// `tests/sequence.rs` / `tests/sampler.rs` / `tests/integration.rs`
// don't compile on this branch — they reference the in-progress
// `SamplerChain` API that isn't in `trunk` yet.)

fn params_with_n_seq_max(n: u32) -> ContextParams {
    let mut p = common::test_ctx_params();
    p.n_seq_max = n;
    p
}

#[test]
fn context_new_rejects_n_seq_max_exceeding_i32_max() {
    let model = common::load_model();
    let params = params_with_n_seq_max((i32::MAX as u32) + 1);
    let result = Context::new(&model, &params);
    assert!(
        result.is_err(),
        "Context::new must reject n_seq_max > i32::MAX as u32 instead of \
         truncating it to a negative `int32_t` and calling llama_batch_init"
    );
}

#[test]
fn context_new_rejects_n_seq_max_u32_max() {
    let model = common::load_model();
    let params = params_with_n_seq_max(u32::MAX);
    let result = Context::new(&model, &params);
    assert!(
        result.is_err(),
        "Context::new must reject n_seq_max = u32::MAX (which would wrap \
         to -1 in `int32_t`)"
    );
}

#[test]
fn context_new_accepts_n_seq_max_at_i32_max() {
    // The exact boundary value — fits in `i32` (just barely) and so should
    // be accepted at the bounds-check layer. (Whether llama.cpp can actually
    // allocate ~2 GiB worth of seq_id slots is a separate question; if it
    // can't, we expect a clean `Err(())` from `llama_init_from_model` /
    // batch allocation rather than UB. This test pins that the Rust-side
    // guard does not over-reject the boundary.)
    let model = common::load_model();
    let params = params_with_n_seq_max(i32::MAX as u32);
    // We don't require success — the model/runtime may refuse the
    // allocation — but the guard itself must not be the thing rejecting it,
    // and crucially it must not invoke UB.
    let _ = Context::new(&model, &params);
}

#[test]
fn context_new_accepts_normal_n_seq_max() {
    // Sanity: the fix must not regress the normal path.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).expect("normal Context::new must still succeed");
    assert_eq!(ctx.free_slots(), params.n_seq_max as usize);
}
