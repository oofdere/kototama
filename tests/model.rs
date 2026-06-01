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

// ---------- Clone (Arc-backed Model) ----------

#[test]
fn model_clone_has_same_n_tokens() {
    let model = common::load_model();
    let cloned = model.clone();
    assert_eq!(
        model.n_tokens(),
        cloned.n_tokens(),
        "cloned model should report the same vocabulary size"
    );
}

#[test]
fn model_clone_can_tokenize() {
    let model = common::load_model();
    let cloned = model.clone();
    let tokens_orig = model.tokenize("hello world", false, false);
    let tokens_clone = cloned.tokenize("hello world", false, false);
    assert_eq!(
        tokens_orig, tokens_clone,
        "cloned model should produce identical tokenization"
    );
}

#[test]
fn model_clone_desc_matches() {
    let model = common::load_model();
    let cloned = model.clone();
    assert_eq!(
        model.desc(),
        cloned.desc(),
        "cloned model description should match original"
    );
}

#[test]
fn model_clone_vocab_queries_match() {
    let model = common::load_model();
    let cloned = model.clone();
    assert_eq!(model.bos_token(), cloned.bos_token());
    assert_eq!(model.eos_token(), cloned.eos_token());
    assert_eq!(model.has_decoder(), cloned.has_decoder());
    assert_eq!(model.has_encoder(), cloned.has_encoder());
}

#[test]
fn model_clone_token_to_piece_matches() {
    let model = common::load_model();
    let cloned = model.clone();
    if let Some(bos) = model.bos_token() {
        assert_eq!(
            model.token_to_piece(bos),
            cloned.token_to_piece(bos),
            "cloned model token_to_piece should match original"
        );
    }
}

#[test]
fn model_original_drop_does_not_affect_clone() {
    // Verify that dropping the original doesn't invalidate the clone (Arc).
    let cloned = {
        let model = common::load_model();
        model.clone()
        // model drops here
    };
    // Clone should still be usable
    assert!(cloned.n_tokens() > 0);
    let tokens = cloned.tokenize("test", false, false);
    assert!(!tokens.is_empty());
}

// ---------- C-string error paths ----------
//
// `Model::load_from_file` and `Model::chat_template` both convert a Rust
// `&str` into a `CString` before handing it to llama.cpp. An interior NUL
// byte makes that conversion fail. The two methods react differently —
// `load_from_file` returns `Err(())`, `chat_template` panics on the
// `unwrap()` — and both branches were previously unexercised, so a refactor
// that swapped one for the other would silently change the public contract.

#[test]
fn load_from_file_path_with_interior_nul_returns_err() {
    use rusty_llama::{Model, ModelParams};
    // Path contains an embedded NUL, so CString::new fails before llama.cpp
    // is ever called — this exercises the `.map_err(|_| ())?` branch in
    // load_from_file rather than the llama.cpp load-failure branch.
    let result = Model::load_from_file("foo\0bar.gguf", ModelParams::new());
    assert!(
        result.is_err(),
        "path with interior NUL must be rejected before reaching llama.cpp"
    );
}

#[test]
#[should_panic]
fn chat_template_panics_on_name_with_interior_nul() {
    // chat_template unwraps the CString conversion, so a name containing an
    // interior NUL panics. The behaviour is documented; pin it down here so a
    // future switch to a Result-returning conversion is a deliberate API change
    // rather than an accidental one.
    let model = common::load_model();
    let _ = model.chat_template(Some("bad\0name"));
}
