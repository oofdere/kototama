//! Regression test: `Context::new` must not let `n_batch == 0` reach llama.cpp.
//!
//! `ContextParams` derefs to the raw `llama_context_params`, so safe code can
//! set `n_batch = 0`. llama.cpp only rejects that when `n_ubatch` is zero too;
//! with the default `n_ubatch` it computes `n_ubatch = min(n_batch, n_ubatch)`
//! == 0 and then fails `GGML_ASSERT(n_outputs >= 1)` in
//! `llama_context::graph_reserve`, aborting the whole process (SIGABRT) from
//! purely safe Rust. `llama_init_from_model`'s `try`/`catch` does not help: an
//! abort is not an exception.
//!
//! (Own test target because `tests/sequence.rs` / `tests/sampler.rs` /
//! `tests/integration.rs` on trunk reference an in-progress sampler API and do
//! not compile.)

use rusty_llama::{test_common, Context, ContextParams};

fn params() -> ContextParams {
    let mut p = ContextParams::new();
    p.n_ctx = 512;
    p.n_batch = 512;
    p.n_seq_max = 4;
    p.no_perf = true;
    p
}

#[test]
fn n_batch_zero_is_rejected() {
    let model = test_common::load_model();
    let mut p = params();
    p.n_batch = 0;
    assert!(Context::new(&model, &p).is_err());
}

#[test]
fn n_batch_zero_with_explicit_n_ubatch_is_rejected() {
    let model = test_common::load_model();
    let mut p = params();
    p.n_batch = 0;
    p.n_ubatch = 256;
    assert!(Context::new(&model, &p).is_err());
}

#[test]
fn valid_params_still_build_a_context() {
    let model = test_common::load_model();
    let ctx = Context::new(&model, &params()).expect("context should build");
    assert!(ctx.n_ctx() > 0);
    assert!(ctx.sequence().is_some());
}

#[test]
fn n_ubatch_zero_alone_is_still_allowed() {
    let model = test_common::load_model();
    let mut p = params();
    p.n_ubatch = 0;
    let ctx = Context::new(&model, &p).expect("n_ubatch == 0 defaults to n_batch");
    assert!(ctx.n_ctx() > 0);
}
