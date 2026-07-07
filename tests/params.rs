//! Public API surface tests for `ModelParams` and `ContextParams`.
//!
//! Existing tests only touch these types by mutating fields through the
//! auto-generated `DerefMut` and then passing the whole struct into
//! `Model::load_from_file` / `Context::new`. `cargo llvm-cov` confirms
//! that `as_ptr()`, `as_mut_ptr()`, the immutable `Deref` body, and the
//! `Copy` semantics on `ContextParams` are not exercised by any test.
//! These tests pin those contracts so a future refactor of the wrappers
//! (e.g. adding padding, changing the wrapper struct layout, or dropping
//! `Copy`) surfaces as a test failure, not a silent behaviour change in
//! `Model::load_from_file` or `Context::new`.
//!
//! No model load — everything here goes through the pure
//! `llama_*_default_params` fill functions and struct-field access.

use rusty_llama::{ContextParams, ModelParams};

// ---------- ModelParams ----------

#[test]
fn model_params_new_matches_llama_default() {
    let params = ModelParams::new();
    let default = unsafe { llama_sys::llama_model_default_params() };
    assert_eq!(params.n_gpu_layers, default.n_gpu_layers);
    assert_eq!(params.use_mmap, default.use_mmap);
    assert_eq!(params.use_mlock, default.use_mlock);
    assert_eq!(params.check_tensors, default.check_tensors);
    assert_eq!(params.vocab_only, default.vocab_only);
}

#[test]
fn model_params_as_ptr_reflects_wrapped_struct() {
    let params = ModelParams::new();
    let ptr = params.as_ptr();
    // as_ptr must point at the wrapped struct — every field visible
    // through Deref must match what the pointer sees.
    unsafe {
        assert_eq!((*ptr).n_gpu_layers, params.n_gpu_layers);
        assert_eq!((*ptr).use_mmap, params.use_mmap);
        assert_eq!((*ptr).vocab_only, params.vocab_only);
    }
}

#[test]
fn model_params_deref_reads_fields() {
    // Exercises the immutable Deref path (uncovered on trunk — all
    // existing sites use DerefMut for field assignment or move the whole
    // struct into `.into()`).
    let mut params = ModelParams::new();
    params.n_gpu_layers = 3;
    // Take an immutable reference so the field access goes through
    // Deref, not DerefMut.
    let borrowed: &ModelParams = &params;
    assert_eq!(borrowed.n_gpu_layers, 3);
    assert_eq!((*borrowed).n_gpu_layers, 3);
}

#[test]
fn model_params_deref_mut_writes_fields() {
    let mut params = ModelParams::new();
    params.n_gpu_layers = 42;
    assert_eq!(params.n_gpu_layers, 42);
    // The mutation must also be visible through the raw pointer.
    unsafe {
        assert_eq!((*params.as_ptr()).n_gpu_layers, 42);
    }
}

#[test]
fn model_params_as_mut_ptr_writes_through_to_deref() {
    let mut params = ModelParams::new();
    let mut_ptr = params.as_mut_ptr();
    unsafe {
        (*mut_ptr).n_gpu_layers = 7;
        (*mut_ptr).use_mmap = false;
    }
    assert_eq!(params.n_gpu_layers, 7);
    assert!(!params.use_mmap);
}

#[test]
fn model_params_into_carries_field_values() {
    // `Model::load_from_file` moves ModelParams by value and calls
    // `params.into()` — pin that the conversion preserves every field.
    let mut params = ModelParams::new();
    params.n_gpu_layers = 13;
    params.use_mmap = false;
    let raw: llama_sys::llama_model_params = params.into();
    assert_eq!(raw.n_gpu_layers, 13);
    assert!(!raw.use_mmap);
}

// ---------- ContextParams ----------

#[test]
fn context_params_new_matches_llama_default() {
    let params = ContextParams::new();
    let default = unsafe { llama_sys::llama_context_default_params() };
    assert_eq!(params.n_ctx, default.n_ctx);
    assert_eq!(params.n_batch, default.n_batch);
    assert_eq!(params.n_seq_max, default.n_seq_max);
    assert_eq!(params.no_perf, default.no_perf);
}

#[test]
fn context_params_as_ptr_reflects_wrapped_struct() {
    let params = ContextParams::new();
    let ptr = params.as_ptr();
    unsafe {
        assert_eq!((*ptr).n_ctx, params.n_ctx);
        assert_eq!((*ptr).n_batch, params.n_batch);
        assert_eq!((*ptr).n_seq_max, params.n_seq_max);
    }
}

#[test]
fn context_params_deref_reads_fields() {
    let mut params = ContextParams::new();
    params.n_ctx = 256;
    let borrowed: &ContextParams = &params;
    assert_eq!(borrowed.n_ctx, 256);
    assert_eq!((*borrowed).n_ctx, 256);
}

#[test]
fn context_params_deref_mut_writes_round_trip_through_ptr() {
    let mut params = ContextParams::new();
    params.n_ctx = 1024;
    params.n_batch = 256;
    params.n_seq_max = 2;
    params.no_perf = true;
    unsafe {
        assert_eq!((*params.as_ptr()).n_ctx, 1024);
        assert_eq!((*params.as_ptr()).n_batch, 256);
        assert_eq!((*params.as_ptr()).n_seq_max, 2);
        assert_eq!((*params.as_ptr()).no_perf, true);
    }
}

#[test]
fn context_params_as_mut_ptr_writes_through_to_deref() {
    let mut params = ContextParams::new();
    let mut_ptr = params.as_mut_ptr();
    unsafe {
        (*mut_ptr).n_ctx = 2048;
    }
    assert_eq!(params.n_ctx, 2048);
}

#[test]
fn context_params_is_copy_and_original_survives() {
    // ContextParams is `Copy` — a bindings user should be able to hand
    // the same params to multiple `Context::new` calls without cloning
    // by hand. Pin that this compiles and that both copies carry the
    // same field values.
    let mut original = ContextParams::new();
    original.n_ctx = 999;
    let copy = original; // by-value use — implicit Copy
    assert_eq!(original.n_ctx, 999, "Copy must leave the source usable");
    assert_eq!(copy.n_ctx, 999, "Copy must carry the modified field");
}

#[test]
fn context_params_copy_is_independent_write() {
    // Writes to a Copy must not alias back to the source.
    let mut a = ContextParams::new();
    a.n_ctx = 100;
    let mut b = a;
    b.n_ctx = 200;
    assert_eq!(a.n_ctx, 100);
    assert_eq!(b.n_ctx, 200);
}

#[test]
fn context_params_clone_matches_manual_copy() {
    let mut original = ContextParams::new();
    original.n_ctx = 512;
    original.n_batch = 128;
    let cloned = original.clone();
    assert_eq!(cloned.n_ctx, original.n_ctx);
    assert_eq!(cloned.n_batch, original.n_batch);
}
