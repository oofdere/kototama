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

// ---------- is_control / get_attr ----------
//
// Both accessors take a token id and forward to llama.cpp's vocab attribute
// table. No other test exercises them today, so a regression here (e.g. a
// macro mis-wiring `is_control` to `llama_vocab_is_eog`, or `get_attr`
// reading past the vocab range) would slip through silently.

#[test]
fn is_control_bos() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        assert!(
            model.is_control(bos),
            "BOS should be flagged as a control token"
        );
    }
}

#[test]
fn is_control_eos() {
    let model = common::load_model();
    if let Some(eos) = model.eos_token() {
        assert!(
            model.is_control(eos),
            "EOS should be flagged as a control token"
        );
    }
}

#[test]
fn is_control_false_for_regular_text_tokens() {
    // Tokens produced from ordinary text (no add_special, no parse_special)
    // should not be control tokens. This pins is_control against the trivial
    // "always returns true" failure mode.
    let model = common::load_model();
    let tokens = model.tokenize("hello", false, false);
    assert!(
        !tokens.is_empty(),
        "tokenizing \"hello\" should produce at least one token"
    );
    for t in tokens {
        assert!(
            !model.is_control(t),
            "token {t} from regular text should not be a control token"
        );
    }
}

#[test]
fn is_control_stable() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        assert_eq!(model.is_control(bos), model.is_control(bos));
    }
}

#[test]
fn get_attr_bos_not_undefined() {
    // LLAMA_TOKEN_ATTR_UNDEFINED == 0 — every real vocab entry should set at
    // least one attribute bit.
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        assert_ne!(
            model.get_attr(bos) as i32,
            0,
            "BOS should have a defined token attribute"
        );
    }
}

#[test]
fn get_attr_stable() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        assert_eq!(model.get_attr(bos) as i32, model.get_attr(bos) as i32);
    }
}

#[test]
fn get_attr_differs_between_control_and_text_tokens() {
    // BOS (control) and a token from regular text ("hello") should not share
    // the same attribute mask — if they did, get_attr would be returning the
    // same value for every id, which would silently defeat the accessor.
    let model = common::load_model();
    let text_tokens = model.tokenize("hello", false, false);
    if let (Some(bos), Some(&text)) = (model.bos_token(), text_tokens.first()) {
        assert_ne!(
            model.get_attr(bos) as i32,
            model.get_attr(text) as i32,
            "BOS and a normal text token should expose different attributes"
        );
    }
}

#[test]
fn get_attr_consistent_with_is_control() {
    // If is_control(bos) is true, the CONTROL bit (LLAMA_TOKEN_ATTR_CONTROL == 1 << 3 == 8)
    // must be set in get_attr(bos). The two accessors must not disagree about
    // the same token.
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        if model.is_control(bos) {
            let attr = model.get_attr(bos) as i32;
            assert_ne!(
                attr & 8,
                0,
                "is_control(BOS) is true but CONTROL bit is not set in get_attr (attr={attr})"
            );
        }
    }
}
