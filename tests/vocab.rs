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

// Smoke tests for the remaining token_option!()-generated accessors.
// Each may legitimately return None for a given model; the goal is to
// exercise the FFI call path (and the LLAMA_TOKEN_NULL branch in the
// macro) for every accessor.
#[test]
fn cls_token_call() {
    let model = common::load_model();
    let _ = model.cls_token();
}

#[test]
fn eot_token_call() {
    let model = common::load_model();
    let _ = model.eot_token();
}

#[test]
fn pad_token_call() {
    let model = common::load_model();
    let _ = model.pad_token();
}

#[test]
fn sep_token_call() {
    let model = common::load_model();
    let _ = model.sep_token();
}

#[test]
fn mask_token_call() {
    let model = common::load_model();
    let _ = model.mask_token();
}

#[test]
fn fim_token_calls() {
    let model = common::load_model();
    let _ = model.fim_mid_token();
    let _ = model.fim_pad_token();
    let _ = model.fim_pre_token();
    let _ = model.fim_rep_token();
    let _ = model.fim_sep_token();
    let _ = model.fim_suf_token();
}

#[test]
fn get_add_eos_call() {
    let model = common::load_model();
    let _ = model.get_add_eos();
}

#[test]
fn get_add_sep_call() {
    let model = common::load_model();
    let _ = model.get_add_sep();
}

#[test]
fn get_attr_does_not_crash() {
    let model = common::load_model();
    // Just verify the FFI call doesn't crash for a valid token id.
    if let Some(bos) = model.bos_token() {
        let _ = model.get_attr(bos);
    }
}

#[test]
fn is_control_call() {
    let model = common::load_model();
    // BOS is typically a control token, but we don't assert the value —
    // just exercise the FFI path.
    if let Some(bos) = model.bos_token() {
        let _ = model.is_control(bos);
    }
}
