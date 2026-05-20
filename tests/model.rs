mod common;

#[test]
fn load_model() {
    let model = common::load_model();
    assert!(!model.vocab.is_null());
}

#[test]
fn desc_nonempty() {
    let model = common::load_model();
    let desc = model.desc();
    assert!(!desc.is_empty(), "model description should not be empty");
}

#[test]
fn n_tokens_positive() {
    let model = common::load_model();
    assert!(model.n_tokens() > 0, "vocab should have tokens");
}

#[test]
fn has_decoder() {
    // TinyStories is a decoder-only model
    let model = common::load_model();
    assert!(model.has_decoder());
}

#[test]
fn no_encoder() {
    let model = common::load_model();
    assert!(!model.has_encoder());
}

#[test]
fn not_diffusion() {
    let model = common::load_model();
    assert!(!model.is_diffusion());
}

#[test]
fn not_recurrent() {
    let model = common::load_model();
    assert!(!model.is_recurrent());
}

#[test]
fn tokenize_nonempty_text() {
    let model = common::load_model();
    let tokens = model.tokenize("hello world", true, false);
    assert!(!tokens.is_empty());
}

#[test]
fn tokenize_empty_text() {
    let model = common::load_model();
    // With add_special=false, empty text should produce zero tokens
    let tokens = model.tokenize("", false, false);
    assert!(tokens.is_empty());
}

#[test]
fn tokenize_roundtrip() {
    let model = common::load_model();
    let text = " Hello, world!";
    let tokens = model.tokenize(text, false, false);
    let reconstructed: String = tokens
        .iter()
        .map(|&t| model.token_to_piece(t).unwrap())
        .collect();
    assert_eq!(reconstructed.trim(), text.trim());
}

#[test]
fn token_to_piece_bos() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let piece = model.token_to_piece(bos);
        assert!(piece.is_ok());
    }
}

#[test]
fn decoder_start_token_none_for_decoder_only() {
    let model = common::load_model();
    // Decoder-only models should return None for decoder_start_token
    assert!(model.decoder_start_token().is_none());
}

#[test]
fn chat_template_default() {
    let model = common::load_model();
    // TinyStories may or may not have a chat template; just verify it doesn't crash
    let _ = model.chat_template(None);
}

#[test]
fn chat_template_unknown_name_returns_none() {
    let model = common::load_model();
    // Requesting a template by a name that does not exist should yield None
    // (llama.cpp returns a null pointer, which the wrapper maps to None).
    assert!(model
        .chat_template(Some("definitely-not-a-real-template-name"))
        .is_none());
}

#[test]
fn not_hybrid() {
    let model = common::load_model();
    // TinyStories is a plain decoder-only transformer, not hybrid.
    assert!(!model.is_hybrid());
}

#[test]
fn as_ptr_and_mut_ptr_match() {
    let mut model_owned = {
        let mut params = rusty_llama::ModelParams::new();
        params.n_gpu_layers = 0;
        rusty_llama::Model::load_from_file(&common::model_path(), params)
            .expect("failed to load model")
    };
    // Both accessors should yield the same underlying pointer.
    let a = model_owned.as_ptr() as usize;
    let b = model_owned.as_mut_ptr() as usize;
    assert_eq!(a, b);
    assert_ne!(a, 0);
}

#[test]
fn model_params_deref_exposes_inner() {
    let params = rusty_llama::ModelParams::new();
    // Deref exposes the inner llama_model_params; just check a field is accessible.
    let _ = params.n_gpu_layers;
}

#[test]
fn model_params_deref_mut_allows_mutation() {
    let mut params = rusty_llama::ModelParams::new();
    (*params).n_gpu_layers = 0;
    assert_eq!(params.n_gpu_layers, 0);
}

#[test]
fn model_params_as_ptr_not_null() {
    let params = rusty_llama::ModelParams::new();
    assert!(!params.as_ptr().is_null());
}

#[test]
fn model_params_as_mut_ptr_not_null() {
    let mut params = rusty_llama::ModelParams::new();
    assert!(!params.as_mut_ptr().is_null());
}
