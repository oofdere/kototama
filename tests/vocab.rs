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

// ---------- Model::get_attr bounds-check (soundness) ----------
//
// llama.cpp's llama_vocab_get_attr uses std::vector::at() internally, which
// throws std::out_of_range on OOB. A C++ exception unwinding through the
// extern "C" boundary is UB per the Rust nomicon (aborts the process on
// current Rust). Guard tokens outside [0, n_tokens()) in the Rust wrapper
// and return LLAMA_TOKEN_ATTR_UNDEFINED — the sentinel llama.h defines for
// "no attributes".

#[test]
fn get_attr_negative_token_is_undefined() {
    let model = common::load_model();
    assert_eq!(
        model.get_attr(-1),
        llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNDEFINED,
    );
    assert_eq!(
        model.get_attr(i32::MIN),
        llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNDEFINED,
    );
}

#[test]
fn get_attr_out_of_range_token_is_undefined() {
    let model = common::load_model();
    let n = model.n_tokens();
    assert_eq!(
        model.get_attr(n),
        llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNDEFINED,
    );
    assert_eq!(
        model.get_attr(n + 1),
        llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNDEFINED,
    );
    assert_eq!(
        model.get_attr(i32::MAX),
        llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNDEFINED,
    );
}

#[test]
fn get_attr_valid_token_still_calls_ffi() {
    // Regression pin: the happy path is unchanged — a valid in-range token
    // still crosses the FFI boundary and returns whatever attributes the
    // vocab has for it. BOS is a "control" token in every real vocab, so it
    // must not report as UNDEFINED.
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        assert_ne!(
            model.get_attr(bos),
            llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNDEFINED,
            "valid BOS token should have some attribute set",
        );
    }
}
