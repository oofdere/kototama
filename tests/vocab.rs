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

// ---------- Soundness: get_attr must accept bitflag combinations ----------
//
// `llama_token_attr` is a bitflags enum on the C++ side. Look at
// `src/llama-vocab.cpp` — the vocab init code freely stores combined values
// like `LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED` (== 24)
// into a `llama_token_attr` variable. If the Rust binding generates this as a
// `#[repr(u32)] enum` (bindgen `rustified_non_exhaustive_enum`), every
// observed value must match a declared variant. Combinations like 12, 20,
// 24, ... are not declared variants, so receiving such a value at the FFI
// boundary is immediate undefined behavior — reachable from purely safe Rust
// via `Model::get_attr(token)`. The build script now uses `bitfield_enum`,
// which emits a newtype struct (`llama_token_attr(u32)`); any `u32` is a
// sound inhabitant.

#[test]
fn get_attr_returns_valid_bitfield_for_every_token() {
    let model = common::load_model();
    let n = model.n_tokens();
    // Iterating the whole vocab forces `llama_vocab_get_attr` to return every
    // attribute value the model uses, including OR'd flag combinations. Pre-
    // fix, hitting a combined value would be UB at the FFI boundary; with the
    // newtype binding it is just a `u32`, so the loop completes and every
    // returned value round-trips through bitwise ops.
    for t in 0..n {
        let attr = model.get_attr(t);
        let bits = attr.0;
        // Re-decompose with the named flags as a smoke test that the
        // bitfield ops bindgen generated work as expected.
        let known_mask = llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNKNOWN.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_UNUSED.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_NORMAL.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_CONTROL.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_USER_DEFINED.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_BYTE.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_NORMALIZED.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_LSTRIP.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_RSTRIP.0
            | llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_SINGLE_WORD.0;
        // Every bit the C side sets should fall inside the documented mask.
        assert_eq!(
            bits & !known_mask,
            0,
            "token {t}: get_attr returned unknown bits 0x{:x} (full value 0x{:x})",
            bits & !known_mask,
            bits
        );
    }
}

#[test]
fn get_attr_bos_is_a_control_token() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let attr = model.get_attr(bos);
        // BOS is always a control token in TinyStories; verifying via bitwise
        // AND demonstrates the newtype's bitfield semantics rather than a
        // single-variant pattern match (which would be wrong for an OR'd
        // value like CONTROL | SINGLE_WORD).
        let is_control = (attr.0
            & llama_sys::llama_token_attr::LLAMA_TOKEN_ATTR_CONTROL.0)
            != 0;
        assert!(is_control, "BOS token should have the CONTROL bit set");
    }
}
