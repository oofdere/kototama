use std::sync::Once;

use crate::{ContextParams, Model, ModelParams};

/// Path to the test model file. Reads `RUSTY_LLAMA_TEST_MODEL` or `RUSTY_LLAMA_BENCH_MODEL`,
/// falling back to `./test-models/TinyStories-656K.Q2_K.gguf` relative to the workspace root.
pub fn model_path() -> String {
    std::env::var("RUSTY_LLAMA_BENCH_MODEL")
        .or_else(|_| std::env::var("RUSTY_LLAMA_TEST_MODEL"))
        .unwrap_or_else(|_| "./test-models/TinyStories-656K.Q2_K.gguf".to_string())
}

/// Shared model instance loaded once and reused across all tests and benchmarks.
/// Safe because tests run with --test-threads=1 (single-threaded) and benchmarks
/// are also single-threaded by default.
#[allow(static_mut_refs)]
static mut SHARED_MODEL: Option<Model> = None;
static INIT: Once = Once::new();

/// Load the test model. Returns a reference to the shared model instance.
#[allow(static_mut_refs)]
pub fn load_model() -> &'static Model {
    unsafe {
        INIT.call_once(|| {
            let path = model_path();
            let mut params = ModelParams::new();
            params.n_gpu_layers = 0; // CPU-only for reproducibility in CI
            SHARED_MODEL = Some(Model::load_from_file(&path, params).expect("failed to load model"));
        });
        SHARED_MODEL.as_ref().expect("model not initialized")
    }
}

/// Build a minimal `ContextParams` suitable for testing (small context, CPU).
#[allow(dead_code)]
pub fn test_ctx_params() -> ContextParams {
    let mut p = ContextParams::new();
    p.n_ctx = 512;
    p.n_batch = 512;
    p.n_seq_max = 4;
    p.no_perf = true;
    p
}

/// Convenience: load model + create context.
#[allow(dead_code)]
pub fn load_model_and_context() -> (&'static Model, ContextParams) {
    let model = load_model();
    let params = test_ctx_params();
    (model, params)
}
