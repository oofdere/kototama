mod common;

use rusty_llama::{Context, Model, ModelParams};

/// Load a fresh, unshared `Model` handle so dropping it actually frees the
/// underlying `llama_model`.
fn load_owned_model() -> Model {
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    Model::load_from_file(&common::model_path(), params).expect("failed to load model")
}

#[test]
fn context_outlives_model_handle() {
    let model = load_owned_model();
    let params = common::test_ctx_params();
    let ctx = Context::new(&model, &params).unwrap();
    let tokens = model.tokenize("hello", true, false);
    assert!(!tokens.is_empty());

    drop(model);

    let mut seq = ctx.sequence().expect("sequence");
    seq.extend(&tokens);
    let logits = seq.logits().expect("logits after decode");
    assert!(logits.iter().any(|l| *l != 0.0));
}

#[test]
fn sequence_outlives_model_handle() {
    let model = load_owned_model();
    let params = common::test_ctx_params();
    let ctx = Context::new(&model, &params).unwrap();
    let token = model.tokenize("hello", true, false)[0];
    let mut seq = ctx.sequence().expect("sequence");
    seq.push(token);

    drop(model);
    drop(ctx);

    seq.push(token);
    assert_eq!(seq.len(), 2);
}
