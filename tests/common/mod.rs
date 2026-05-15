use rusty_llama::{ContextParams, Model, ModelParams};

/// Path to the test model file. Reads `RUSTY_LLAMA_TEST_MODEL`, falling back
/// to `./test-models/TinyStories-656K.Q2_K.gguf` relative to the workspace root.
pub fn model_path() -> String {
    std::env::var("RUSTY_LLAMA_TEST_MODEL")
        .unwrap_or_else(|_| "./test-models/TinyStories-656K.Q2_K.gguf".to_string())
}

/// Load the test model. Panics if the file doesn't exist.
pub fn load_model() -> Model {
    let path = model_path();
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0; // CPU-only for reproducibility in CI
    Model::load_from_file(&path, params).expect("failed to load model")
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
pub fn load_model_and_context() -> (Model, ContextParams) {
    let model = load_model();
    let params = test_ctx_params();
    (model, params)
}
