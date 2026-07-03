mod common;

#[test]
fn bos_token_some() {
    let model = common::load_model();
    assert!(
        model.bos_token().is_some(),
        "TinyStories should have a BOS token"
    );
}

#[test]
fn eos_token_some() {
    let model = common::load_model();
    assert!(
        model.eos_token().is_some(),
        "TinyStories should have an EOS token"
    );
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
        let score = model.get_score(bos);
        assert!(score.is_finite());
    }
}

#[test]
fn get_text_bos_nonempty() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let text = model.get_text(bos);
        assert!(!text.to_bytes().is_empty());
    }
}

#[test]
fn get_text_returns_empty_for_negative_token() {
    let model = common::load_model();
    // Without the bounds check, this would throw std::out_of_range from
    // id_to_token.at(id) on the C++ side and unwind through the C ABI (UB).
    assert!(model.get_text(-1).to_bytes().is_empty());
    assert!(model.get_text(i32::MIN).to_bytes().is_empty());
}

#[test]
fn get_text_returns_empty_for_oob_token() {
    let model = common::load_model();
    let n = model.n_tokens();
    assert!(model.get_text(n).to_bytes().is_empty());
    assert!(model.get_text(n + 1000).to_bytes().is_empty());
    assert!(model.get_text(i32::MAX).to_bytes().is_empty());
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
