// Re-export from the shared test_common module in src/
// This maintains compatibility with existing test code while sharing the model instance with benchmarks
pub use rusty_llama::test_common::{load_model, model_path, test_ctx_params, load_model_and_context};
