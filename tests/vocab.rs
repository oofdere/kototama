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

// ---------- Optional special-token accessors ----------
//
// llama.cpp exposes a fixed set of "special" tokens (cls, sep, pad, mask,
// eot, the FIM family, ...). Each is optional — the model may or may not
// declare one. These wrappers convert `LLAMA_TOKEN_NULL` to `None`.
//
// We exercise every accessor here. Each test asserts the shape we observe on
// the bundled TinyStories model so a future model regression (or a
// llama.cpp change in vocab parsing) shows up as a failed assertion rather
// than as a silent change in returned value.

#[test]
fn cls_token_present_on_tinystories() {
    let model = common::load_model();
    let cls = model.cls_token().expect("TinyStories declares a CLS token");
    assert!(cls >= 0 && cls < model.n_tokens(),
        "CLS token id must be in vocab range");
}

#[test]
fn pad_token_present_on_tinystories() {
    let model = common::load_model();
    let pad = model.pad_token().expect("TinyStories declares a PAD token");
    assert!(pad >= 0 && pad < model.n_tokens(),
        "PAD token id must be in vocab range");
}

#[test]
fn nl_token_present_on_tinystories() {
    let model = common::load_model();
    let nl = model.nl_token().expect("TinyStories declares an NL token");
    assert!(nl >= 0 && nl < model.n_tokens(),
        "NL token id must be in vocab range");
    // The newline token's piece should contain a newline byte.
    let piece = model.token_to_piece(nl).expect("nl_token should decode");
    assert!(piece.contains('\n'),
        "nl_token piece should contain '\\n', got {piece:?}");
}

#[test]
fn eot_token_absent_on_tinystories() {
    // TinyStories does not declare an end-of-turn marker (it isn't a chat
    // model). The accessor must surface that as None, not as
    // LLAMA_TOKEN_NULL (-1) leaking through.
    let model = common::load_model();
    assert!(model.eot_token().is_none(),
        "TinyStories should not declare an EOT token");
}

#[test]
fn sep_token_absent_on_tinystories() {
    let model = common::load_model();
    assert!(model.sep_token().is_none());
}

#[test]
fn mask_token_absent_on_tinystories() {
    let model = common::load_model();
    assert!(model.mask_token().is_none());
}

#[test]
fn fim_tokens_all_absent_on_tinystories() {
    // TinyStories is not a fill-in-the-middle model — the entire FIM family
    // should be None.
    let model = common::load_model();
    assert!(model.fim_pre_token().is_none(), "fim_pre");
    assert!(model.fim_mid_token().is_none(), "fim_mid");
    assert!(model.fim_suf_token().is_none(), "fim_suf");
    assert!(model.fim_pad_token().is_none(), "fim_pad");
    assert!(model.fim_rep_token().is_none(), "fim_rep");
    assert!(model.fim_sep_token().is_none(), "fim_sep");
}

#[test]
fn special_tokens_all_in_vocab_range_when_present() {
    // Any accessor that returns Some(token) must return a valid index into
    // the vocab. If a future model declares one of these, this catches the
    // case where llama.cpp returns a sentinel that we forget to filter.
    let model = common::load_model();
    let n = model.n_tokens();
    let in_range = |t: Option<i32>, name: &str| {
        if let Some(t) = t {
            assert!(t >= 0 && t < n,
                "{name} token {t} out of vocab range [0, {n})");
        }
    };
    in_range(model.bos_token(), "bos");
    in_range(model.eos_token(), "eos");
    in_range(model.cls_token(), "cls");
    in_range(model.eot_token(), "eot");
    in_range(model.nl_token(), "nl");
    in_range(model.pad_token(), "pad");
    in_range(model.sep_token(), "sep");
    in_range(model.mask_token(), "mask");
    in_range(model.fim_pre_token(), "fim_pre");
    in_range(model.fim_mid_token(), "fim_mid");
    in_range(model.fim_suf_token(), "fim_suf");
    in_range(model.fim_pad_token(), "fim_pad");
    in_range(model.fim_rep_token(), "fim_rep");
    in_range(model.fim_sep_token(), "fim_sep");
}

// ---------- get_add_eos / get_add_sep ----------
//
// These mirror `get_add_bos` (which already has a callable-doesn't-crash
// test). Pin the observed value on TinyStories so a llama.cpp change in
// how these are read from the gguf header doesn't silently flip them.

#[test]
fn get_add_eos_false_on_tinystories() {
    let model = common::load_model();
    assert!(!model.get_add_eos(),
        "TinyStories does not request an automatic EOS on tokenize");
}

#[test]
fn get_add_sep_false_on_tinystories() {
    let model = common::load_model();
    assert!(!model.get_add_sep(),
        "TinyStories does not request an automatic SEP on tokenize");
}

#[test]
fn get_add_bos_true_on_tinystories() {
    // Already verified to not crash, but tighten to the observed value so a
    // regression in vocab parsing surfaces here.
    let model = common::load_model();
    assert!(model.get_add_bos(),
        "TinyStories requests an automatic BOS on tokenize");
}
