use std::sync::OnceLock;

use crate::{ContextParams, Model, ModelParams};

/// Path to the test model file.
pub fn model_path() -> String {
    std::env::var("RUSTY_LLAMA_BENCH_MODEL")
        .or_else(|_| std::env::var("RUSTY_LLAMA_TEST_MODEL"))
        .unwrap_or_else(|_| "./test-models/TinyStories-656K.Q2_K.gguf".to_string())
}

static SHARED_MODEL: OnceLock<Model> = OnceLock::new();

/// Load the test model. Returns a clone of the shared Model handle.
/// Model is Clone (Arc-backed), so this is cheap.
pub fn load_model() -> Model {
    SHARED_MODEL
        .get_or_init(|| {
            let path = model_path();
            let mut params = ModelParams::new();
            params.n_gpu_layers = 0;
            Model::load_from_file(&path, params).expect("failed to load model")
        })
        .clone()
}

/// Build a minimal `ContextParams` suitable for testing (small context, CPU).
#[allow(dead_code)]
pub fn test_ctx_params() -> ContextParams {
    let mut p = ContextParams::new();
    p.set_n_ctx(512)
        .set_n_batch(512)
        .set_n_seq_max(4)
        .set_no_perf(true);
    p
}

/// Convenience: load model + create context params.
#[allow(dead_code)]
pub fn load_model_and_context() -> (Model, ContextParams) {
    let model = load_model();
    let params = test_ctx_params();
    (model, params)
}
