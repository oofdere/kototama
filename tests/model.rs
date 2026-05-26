mod common;

#[test]
fn load_model() {
    let model = common::load_model();
    assert!(model.n_tokens() > 0);
}

#[test]
fn desc_nonempty() {
    let model = common::load_model();
    let desc = model.desc();
    assert!(!desc.is_empty(), "model description should not be empty");
}

#[test]
fn desc_matches_probe_length() {
    let model = common::load_model();
    let needed = unsafe {
        llama_sys::llama_model_desc(model.as_ptr() as *mut _, std::ptr::null_mut(), 0)
    };
    let desc = model.desc();
    assert_eq!(
        desc.len(),
        needed as usize,
        "desc() should return all {} bytes, got {:?} ({} bytes)",
        needed,
        desc,
        desc.len()
    );
    assert!(
        desc.ends_with("Medium"),
        "expected description to end with \"Medium\", got {:?}",
        desc
    );
}

#[test]
fn n_tokens_positive() {
    let model = common::load_model();
    assert!(model.n_tokens() > 0, "vocab should have tokens");
}

#[test]
fn has_decoder() {
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
    assert!(model.decoder_start_token().is_none());
}

#[test]
fn chat_template_default() {
    let model = common::load_model();
    let _ = model.chat_template(None);
}
