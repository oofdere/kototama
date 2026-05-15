use rusty_llama::{ContextParams, Model, ModelParams};

/// Path to the test model file. Reads `RUSTY_LLAMA_TEST_MODEL`, falling back
/// to `./test-models/smollm-135m.gguf` relative to the workspace root.
pub fn model_path() -> String {
    std::env::var("RUSTY_LLAMA_TEST_MODEL")
        .unwrap_or_else(|_| "./test-models/smollm-135m.gguf".to_string())
}

/// Load the test model, or skip/fail depending on whether the `require-model`
/// feature is enabled.
///
/// - Without `require-model`: returns `None` if the file doesn't exist and
///   prints a skip message.
/// - With `require-model`: panics if the file doesn't exist.
pub fn try_load_model() -> Option<Model> {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        #[cfg(feature = "require-model")]
        panic!(
            "Model file not found at '{}'. Set RUSTY_LLAMA_TEST_MODEL to a valid .gguf path.",
            path
        );
        #[cfg(not(feature = "require-model"))]
        {
            println!("SKIP: model not found at '{path}'. Set RUSTY_LLAMA_TEST_MODEL to run model tests.");
            return None;
        }
    }
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0; // CPU-only for reproducibility in CI
    Some(Model::load_from_file(&path, params).expect("failed to load model"))
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

/// Convenience: load model + create context. Returns `None` if model is absent
/// and `require-model` is not set.
#[allow(dead_code)]
pub fn try_load_model_and_context() -> Option<(Model, ContextParams)> {
    let model = try_load_model()?;
    let params = test_ctx_params();
    Some((model, params))
}
