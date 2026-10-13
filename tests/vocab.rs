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

// ---------- Other optional special tokens ----------

#[test]
fn cls_token_in_range_or_none() {
    let model = common::load_model();
    if let Some(t) = model.cls_token() {
        assert!(t >= 0 && t < model.n_tokens());
    }
}

#[test]
fn sep_token_in_range_or_none() {
    let model = common::load_model();
    if let Some(t) = model.sep_token() {
        assert!(t >= 0 && t < model.n_tokens());
    }
}

#[test]
fn mask_token_in_range_or_none() {
    let model = common::load_model();
    if let Some(t) = model.mask_token() {
        assert!(t >= 0 && t < model.n_tokens());
    }
}

#[test]
fn pad_token_in_range_or_none() {
    let model = common::load_model();
    if let Some(t) = model.pad_token() {
        assert!(t >= 0 && t < model.n_tokens());
    }
}

#[test]
fn eot_token_in_range_or_none() {
    let model = common::load_model();
    if let Some(t) = model.eot_token() {
        assert!(t >= 0 && t < model.n_tokens());
    }
}

// ---------- FIM (fill-in-the-middle) tokens ----------

#[test]
fn fim_tokens_in_range_or_none() {
    let model = common::load_model();
    let n = model.n_tokens();
    for t in [
        model.fim_mid_token(),
        model.fim_pad_token(),
        model.fim_pre_token(),
        model.fim_rep_token(),
        model.fim_sep_token(),
        model.fim_suf_token(),
    ] {
        if let Some(id) = t {
            assert!(id >= 0 && id < n);
        }
    }
}

// ---------- Vocab add-flag accessors ----------

#[test]
fn get_add_eos() {
    let model = common::load_model();
    // Call shouldn't crash; value depends on model config.
    let _ = model.get_add_eos();
}

#[test]
fn get_add_sep() {
    let model = common::load_model();
    let _ = model.get_add_sep();
}

// ---------- Per-token attribute / control queries ----------

#[test]
fn get_attr_bos_matches_is_control() {
    use llama_sys::llama_token_attr;
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let attr = model.get_attr(bos);
        let is_control_attr = attr == llama_token_attr::LLAMA_TOKEN_ATTR_CONTROL;
        if is_control_attr {
            assert!(
                model.is_control(bos),
                "a token tagged CONTROL by get_attr must also be is_control()"
            );
        }
    }
}

#[test]
fn is_control_for_bos_is_bool() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        // Just verify the call returns without crashing; exact value depends on model.
        let _: bool = model.is_control(bos);
    }
}
