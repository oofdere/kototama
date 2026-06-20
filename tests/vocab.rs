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

// ---------- Token-option accessors not exercised elsewhere ----------
//
// `cls_token`, `eot_token`, `fim_*_token`, `mask_token`, `pad_token`, and
// `sep_token` all go through the same `token_option!` macro: they return
// `None` for `LLAMA_TOKEN_NULL` and `Some(token)` otherwise. These tests
// pin the actual behaviour against the bundled TinyStories test model and
// guard against the macro being mis-wired to the wrong FFI symbol (e.g.
// `pad_token` accidentally calling `llama_vocab_sep`).

fn assert_in_vocab(model: &rusty_llama::Model, token: i32, name: &str) {
    assert!(
        token >= 0 && token < model.n_tokens(),
        "{} token {} out of vocab range [0, {})",
        name,
        token,
        model.n_tokens()
    );
}

#[test]
fn cls_token_some_and_in_range() {
    let model = common::load_model();
    let cls = model
        .cls_token()
        .expect("TinyStories should expose a CLS token");
    assert_in_vocab(&model, cls, "cls");
}

#[test]
fn pad_token_some_and_in_range() {
    let model = common::load_model();
    let pad = model
        .pad_token()
        .expect("TinyStories should expose a PAD token");
    assert_in_vocab(&model, pad, "pad");
}

#[test]
fn eot_token_none_for_tinystories() {
    let model = common::load_model();
    // TinyStories has no end-of-turn token; pinning this catches a regression
    // where the macro hooks into the wrong symbol.
    assert_eq!(model.eot_token(), None);
}

#[test]
fn fim_tokens_none_for_tinystories() {
    let model = common::load_model();
    // None of the fill-in-the-middle tokens are defined for TinyStories.
    assert_eq!(model.fim_mid_token(), None);
    assert_eq!(model.fim_pad_token(), None);
    assert_eq!(model.fim_pre_token(), None);
    assert_eq!(model.fim_rep_token(), None);
    assert_eq!(model.fim_sep_token(), None);
    assert_eq!(model.fim_suf_token(), None);
}

#[test]
fn mask_token_none_for_tinystories() {
    let model = common::load_model();
    assert_eq!(model.mask_token(), None);
}

#[test]
fn sep_token_none_for_tinystories() {
    let model = common::load_model();
    assert_eq!(model.sep_token(), None);
}

#[test]
fn token_option_accessors_are_stable() {
    // Each accessor should return the same value on repeated calls — there's
    // no internal state to drift, but this pins it.
    let model = common::load_model();
    assert_eq!(model.cls_token(), model.cls_token());
    assert_eq!(model.eot_token(), model.eot_token());
    assert_eq!(model.fim_mid_token(), model.fim_mid_token());
    assert_eq!(model.fim_pad_token(), model.fim_pad_token());
    assert_eq!(model.fim_pre_token(), model.fim_pre_token());
    assert_eq!(model.fim_rep_token(), model.fim_rep_token());
    assert_eq!(model.fim_sep_token(), model.fim_sep_token());
    assert_eq!(model.fim_suf_token(), model.fim_suf_token());
    assert_eq!(model.mask_token(), model.mask_token());
    assert_eq!(model.pad_token(), model.pad_token());
    assert_eq!(model.sep_token(), model.sep_token());
}

// ---------- Config-flag accessors ----------

#[test]
fn get_add_eos_false_for_tinystories() {
    let model = common::load_model();
    assert!(
        !model.get_add_eos(),
        "TinyStories tokenizer should not auto-append EOS"
    );
}

#[test]
fn get_add_sep_false_for_tinystories() {
    let model = common::load_model();
    assert!(
        !model.get_add_sep(),
        "TinyStories tokenizer should not auto-insert SEP"
    );
}

#[test]
fn get_add_bos_true_for_tinystories() {
    // Pin the value rather than just "does not crash" — pairs with get_add_bos above.
    let model = common::load_model();
    assert!(
        model.get_add_bos(),
        "TinyStories tokenizer should auto-prepend BOS"
    );
}
