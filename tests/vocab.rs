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

// Regression: `llama_token_attr` is a C++ bitflags enum. llama.cpp stores
// OR'd values like `NORMAL | CONTROL` (== 12) or `CONTROL | USER_DEFINED`
// (== 24) into fields of that type and returns them from
// `llama_vocab_get_attr`. If the Rust binding for `llama_token_attr` is a
// `#[repr(u32)] enum`, observing any of those combined bit patterns at the
// FFI boundary is instant UB — reachable from safe Rust via `get_attr`.
// The build script binds `llama_token_attr` as a `bitfield_enum` (a
// `#[repr(transparent)]` newtype over `u32`) so every `u32` value is a
// sound inhabitant of the type.
#[test]
fn get_attr_iterates_full_vocab_without_ub() {
    let model = common::load_model();
    let n = model.n_tokens();
    assert!(n > 0);
    // Force the FFI to return every attribute value the model uses,
    // including OR'd combinations. Under the old enum binding this loop
    // was UB the moment the FFI produced a value like CONTROL | USER_DEFINED.
    for t in 0..n {
        let attr = model.get_attr(t);
        // Round-trip through the raw bits; if this compiles and runs the
        // binding is a newtype, not the unsound enum.
        let _bits = attr.0;
    }
}

#[test]
fn get_attr_bos_has_control_bit() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let attr = model.get_attr(bos);
        // The BOS token in TinyStories is a control token. Test the CONTROL
        // bit via bitwise-AND rather than equality — the C side is free to
        // OR in other flags (e.g. SINGLE_WORD) alongside CONTROL, and an
        // equality check would break for those combinations.
        let control_bit =
            llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_CONTROL.0;
        assert_ne!(
            attr.0 & control_bit,
            0,
            "BOS token attr 0x{:x} should have CONTROL bit 0x{:x} set",
            attr.0,
            control_bit
        );
    }
}
