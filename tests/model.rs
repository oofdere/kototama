mod common;

#[test]
fn load_model() {
    let Some(model) = common::try_load_model() else { return };
    assert!(!model.vocab.is_null());
}

#[test]
fn desc_nonempty() {
    let Some(model) = common::try_load_model() else { return };
    let desc = model.desc();
    assert!(!desc.is_empty(), "model description should not be empty");
}

#[test]
fn n_tokens_positive() {
    let Some(model) = common::try_load_model() else { return };
    assert!(model.n_tokens() > 0, "vocab should have tokens");
}

#[test]
fn has_decoder() {
    // SmolLM is a decoder-only model
    let Some(model) = common::try_load_model() else { return };
    assert!(model.has_decoder());
}

#[test]
fn no_encoder() {
    let Some(model) = common::try_load_model() else { return };
    assert!(!model.has_encoder());
}

#[test]
fn not_diffusion() {
    let Some(model) = common::try_load_model() else { return };
    assert!(!model.is_diffusion());
}

#[test]
fn not_recurrent() {
    let Some(model) = common::try_load_model() else { return };
    assert!(!model.is_recurrent());
}

#[test]
fn tokenize_nonempty_text() {
    let Some(model) = common::try_load_model() else { return };
    let tokens = model.tokenize("hello world", true, false);
    assert!(!tokens.is_empty());
}

#[test]
fn tokenize_empty_text() {
    let Some(model) = common::try_load_model() else { return };
    // With add_special=false, empty text should produce zero tokens
    let tokens = model.tokenize("", false, false);
    assert!(tokens.is_empty());
}

#[test]
fn tokenize_roundtrip() {
    let Some(model) = common::try_load_model() else { return };
    let text = "Hello, world!";
    let tokens = model.tokenize(text, false, false);
    let reconstructed: String = tokens
        .iter()
        .map(|&t| model.token_to_piece(t).unwrap())
        .collect();
    assert_eq!(reconstructed, text);
}

#[test]
fn token_to_piece_bos() {
    let Some(model) = common::try_load_model() else { return };
    if let Some(bos) = model.bos_token() {
        let piece = model.token_to_piece(bos);
        assert!(piece.is_ok());
    }
}

#[test]
fn decoder_start_token_none_for_decoder_only() {
    let Some(model) = common::try_load_model() else { return };
    // Decoder-only models should return None for decoder_start_token
    assert!(model.decoder_start_token().is_none());
}

#[test]
fn chat_template_default() {
    let Some(model) = common::try_load_model() else { return };
    // SmolLM may or may not have a chat template; just verify it doesn't crash
    let _ = model.chat_template(None);
}
