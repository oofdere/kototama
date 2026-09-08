use rusty_llama::test_common::*;
use rusty_llama::*;

fn load_no_alloc() -> Model {
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    params.no_alloc = true;
    Model::load_from_file(&model_path(), params).expect("failed to load no_alloc model")
}

#[test]
fn no_alloc_model_reports_flag() {
    assert!(load_no_alloc().is_no_alloc());
    assert!(!load_model().is_no_alloc());
}

// On trunk this aborted the whole process: `no_alloc` combined with the
// default `use_mmap = true` trips `GGML_ASSERT(!ml.no_alloc)` inside
// llama.cpp's load_tensors on the mmap-backed weight-buffer path.
#[test]
fn no_alloc_model_loads_and_can_tokenize() {
    let model = load_no_alloc();
    let tokens = model.tokenize("Once upon a time", true, false);
    assert!(!tokens.is_empty());
    assert!(model.n_tokens() > 0);
}

// On trunk this segfaulted: the loaded model's weight buffers are 0-size
// dummies, and `llama_init_from_model` dereferences tensor data while
// wiring up backends.
#[test]
fn context_new_rejects_no_alloc_model() {
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    params.no_alloc = true;
    params.use_mmap = false;
    let model = Model::load_from_file(&model_path(), params).expect("failed to load no_alloc model");
    assert!(Context::new(&model, &test_ctx_params()).is_err());
}

#[test]
fn full_model_context_still_works() {
    let (model, params) = load_model_and_context();
    let ctx = Context::new(&model, &params).expect("context");
    let mut seq = ctx.sequence().expect("sequence");
    let tokens = model.tokenize("Once upon", true, false);
    seq.push(tokens[0]);
    assert!(seq.logits().is_some());
}
