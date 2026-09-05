use rusty_llama::test_common::*;
use rusty_llama::*;

fn load_vocab_only() -> Model {
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    params.vocab_only = true;
    Model::load_from_file(&model_path(), params).expect("failed to load vocab-only model")
}

#[test]
fn vocab_only_model_reports_flag() {
    assert!(load_vocab_only().is_vocab_only());
    assert!(!load_model().is_vocab_only());
}

#[test]
fn vocab_only_model_can_still_tokenize() {
    let model = load_vocab_only();
    let tokens = model.tokenize("Once upon a time", true, false);
    assert!(!tokens.is_empty());
    assert!(model.n_tokens() > 0);
}

// On trunk this aborted the whole process: `Context::new` returned `Ok`, and the
// first `Sequence::push` tripped `GGML_ASSERT(n_backends > 0)` inside llama.cpp
// because a vocab-only model has no weights or backend scheduler.
#[test]
fn context_new_rejects_vocab_only_model() {
    let model = load_vocab_only();
    let params = test_ctx_params();
    assert!(Context::new(&model, &params).is_err());
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
