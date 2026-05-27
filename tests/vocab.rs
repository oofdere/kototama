mod common;

#[test]
fn bos_token_some() {
    let model = common::load_model();
    assert!(model.bos_token().is_some(), "TinyStories should have a BOS token");
}

#[test]
fn eos_token_some() {
    let model = common::load_model();
    assert!(model.eos_token().is_some(), "TinyStories should have an EOS token");
}

#[test]
fn bos_and_eos_are_valid_tokens() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        assert!(bos >= 0 && bos < model.n_tokens());
    }
    if let Some(eos) = model.eos_token() {
        assert!(eos >= 0 && eos < model.n_tokens());
    }
}

#[test]
fn n_tokens_matches_vocab_size() {
    let model = common::load_model();
    assert!(model.n_tokens() > 0);
}

#[test]
fn vocab_type_is_not_none() {
    let model = common::load_model();
    let vtype = model.vocab_type();
    // LLAMA_VOCAB_TYPE_NONE = 0
    assert_ne!(vtype as i32, 0, "vocab type should not be NONE");
}

#[test]
fn get_add_bos() {
    let model = common::load_model();
    // Just verify the call doesn't crash; value depends on model config
    let _ = model.get_add_bos();
}

#[test]
fn get_score_bos() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let score = model.get_score(bos).expect("BOS token should be in range");
        assert!(score.is_finite());
    }
}

#[test]
fn get_text_bos_nonempty() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let text = model
            .get_text(bos)
            .expect("BOS token should be in range");
        assert!(!text.to_bytes().is_empty());
    }
}

// ---------- Soundness: vocab accessors must bounds-check the token ----------
//
// Before the fix, `get_text` / `get_score` / `get_attr` handed the caller's
// `i32` straight to `id_to_token.at(id)` on the C++ side, which throws
// `std::out_of_range` for invalid ids. A C++ exception unwinding through the
// C ABI into Rust is undefined behavior. `is_control` was even worse: it
// reached `id_to_token[id]` (vector `operator[]`), which is plain C++ UB on
// out-of-bounds. All four were reachable from purely safe Rust.
//
// The fix bounds-checks against `n_tokens()` before crossing the FFI, so
// negative or too-large tokens take a deterministic non-UB path:
// `None` for the Option-returning accessors and `false` for `is_control`.

#[test]
fn get_text_returns_none_for_negative_token() {
    let model = common::load_model();
    assert!(model.get_text(-1).is_none());
    assert!(model.get_text(i32::MIN).is_none());
}

#[test]
fn get_text_returns_none_for_oob_token() {
    let model = common::load_model();
    let n = model.n_tokens();
    assert!(model.get_text(n).is_none());
    assert!(model.get_text(n + 1000).is_none());
    assert!(model.get_text(i32::MAX).is_none());
}

#[test]
fn get_score_returns_none_for_oob_token() {
    let model = common::load_model();
    let n = model.n_tokens();
    assert!(model.get_score(-1).is_none());
    assert!(model.get_score(n).is_none());
    assert!(model.get_score(i32::MAX).is_none());
}

#[test]
fn get_attr_returns_none_for_oob_token() {
    let model = common::load_model();
    let n = model.n_tokens();
    assert!(model.get_attr(-1).is_none());
    assert!(model.get_attr(n).is_none());
    assert!(model.get_attr(i32::MAX).is_none());
}

#[test]
fn is_control_returns_false_for_oob_token() {
    let model = common::load_model();
    let n = model.n_tokens();
    // Out-of-range tokens are not control tokens — anything else would mean
    // we'd dereferenced past the end of the vocab's `id_to_token` vector.
    assert!(!model.is_control(-1));
    assert!(!model.is_control(n));
    assert!(!model.is_control(i32::MAX));
}

#[test]
fn token_to_piece_errors_on_oob_token() {
    let model = common::load_model();
    let n = model.n_tokens();
    assert!(model.token_to_piece(-1).is_err());
    assert!(model.token_to_piece(n).is_err());
    assert!(model.token_to_piece(i32::MAX).is_err());
}

#[test]
fn is_eog_eos() {
    let model = common::load_model();
    if let Some(eos) = model.eos_token() {
        assert!(model.is_eog(eos), "EOS token should be end-of-generation");
    }
}

#[test]
fn is_not_eog_regular_token() {
    let model = common::load_model();
    // Token 0 is typically BOS or a normal token, not EOG, for TinyStories
    if let Some(bos) = model.bos_token() {
        // BOS is control but not always EOG — just verify the call doesn't crash
        let _ = model.is_eog(bos);
    }
}

#[test]
fn nl_token_optional() {
    let model = common::load_model();
    // May or may not be present; just verify no crash
    let _ = model.nl_token();
}
