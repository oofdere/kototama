mod common;

use rusty_llama::Context;

// Regression tests for safe-Rust UB in `Sequence::push`.
//
// Before the fix, calling `seq.push(token)` with a token outside the model's
// vocabulary range reached `llama_decode`, where `ggml_compute_forward_get_rows`
// asserts `0 <= id < ne01` and calls `ggml_abort` on the CPU backend (process
// abort — bypasses Rust unwinding and `Drop`). On CUDA, the get_rows kernel
// performs the embedding lookup without any bounds check at all, so the same
// safe-Rust call reads OOB GPU memory.
//
// After the fix, `Sequence::push` panics with a normal Rust assertion before
// crossing the FFI boundary, so `Drop` runs and the test harness can observe
// the failure.

#[test]
#[should_panic(expected = "out of vocabulary range")]
fn push_negative_token_panics_instead_of_aborting() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.push(-1);
}

#[test]
#[should_panic(expected = "out of vocabulary range")]
fn push_token_at_vocab_boundary_panics_instead_of_aborting() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.push(model.n_tokens());
}

#[test]
#[should_panic(expected = "out of vocabulary range")]
fn push_i32_max_token_panics_instead_of_aborting() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.push(i32::MAX);
}

#[test]
#[should_panic(expected = "out of vocabulary range")]
fn push_i32_min_token_panics_instead_of_aborting() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.push(i32::MIN);
}

#[test]
fn push_valid_token_still_works() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", true, false);
    for &t in &tokens {
        seq.push(t);
    }
    assert_eq!(seq.tokens(), tokens.as_slice());
}

#[test]
fn n_vocab_matches_model() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    assert_eq!(ctx.n_vocab(), model.n_tokens());
}
