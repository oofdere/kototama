mod common;

#[test]
fn load_model() {
    let model = common::load_model();
    assert!(model.n_tokens() > 0);
}

#[test]
fn desc_nonempty() {
    let model = common::load_model();
    let desc = model.desc();
    assert!(!desc.is_empty(), "model description should not be empty");
}

#[test]
fn desc_matches_probe_length() {
    let model = common::load_model();
    let needed = unsafe {
        llama_sys::llama_model_desc(model.as_ptr() as *mut _, std::ptr::null_mut(), 0)
    };
    let desc = model.desc();
    assert_eq!(
        desc.len(),
        needed as usize,
        "desc() should return all {} bytes, got {:?} ({} bytes)",
        needed,
        desc,
        desc.len()
    );
    assert!(
        desc.ends_with("Medium"),
        "expected description to end with \"Medium\", got {:?}",
        desc
    );
}

#[test]
fn n_tokens_positive() {
    let model = common::load_model();
    assert!(model.n_tokens() > 0, "vocab should have tokens");
}

#[test]
fn has_decoder() {
    let model = common::load_model();
    assert!(model.has_decoder());
}

#[test]
fn no_encoder() {
    let model = common::load_model();
    assert!(!model.has_encoder());
}

#[test]
fn not_diffusion() {
    let model = common::load_model();
    assert!(!model.is_diffusion());
}

#[test]
fn not_recurrent() {
    let model = common::load_model();
    assert!(!model.is_recurrent());
}

#[test]
fn tokenize_nonempty_text() {
    let model = common::load_model();
    let tokens = model.tokenize("hello world", true, false);
    assert!(!tokens.is_empty());
}

#[test]
fn tokenize_empty_text() {
    let model = common::load_model();
    let tokens = model.tokenize("", false, false);
    assert!(tokens.is_empty());
}

// ---------- Soundness: `Model::tokenize` must guard the FFI `int32_t` cast ----------
//
// `Model::tokenize` calls `llama_tokenize(..., text_len: i32, ...)` twice.
// The previous implementation cast `text.len() as i32`, which silently
// wraps for any string longer than `i32::MAX` bytes. On the C side that
// wrapped negative is fed straight into `std::string(text, text_len)`,
// converting it back to a huge `size_t` and reading far past the end of the
// input buffer — undefined behaviour reachable from purely safe Rust.
//
// `llama_tokenize` also reserves `i32::MIN` to signal "tokenization result
// would exceed `i32::MAX` tokens" (see `llama.h` and
// `llama-vocab.cpp::llama_vocab::tokenize`). The old wrapper computed
// `-result`, which overflows for `i32::MIN`: debug panics with the cryptic
// "attempt to negate with overflow"; release wraps back to `i32::MIN`, and
// then `vec![0; i32::MIN as usize]` requests ~9 EiB and aborts the process.
//
// The fix guards both with explicit, message-bearing panics before any FFI
// call can be issued. Neither case is reachable on the bundled test model
// (it would need >2 GiB of input or a vocab with >2 G tokens), so these
// tests document the surrounding contract instead — `tokenize` must not
// regress for normal-sized inputs and must consistently report token counts.

#[test]
fn tokenize_no_truncation_zeros() {
    // Regression: the second `llama_tokenize` call may, in rare error paths,
    // return a negative value. The old wrapper did `tokens.truncate(n as
    // usize)`, which casts a negative `n` to a huge `usize`, does nothing,
    // and silently returns a vector full of zeros to the caller. The fix
    // drops the buffer on negative returns; for normal inputs the vector
    // should match what we'd get by tokenizing the same text twice.
    let model = common::load_model();
    let text = "the quick brown fox jumps over the lazy dog";
    let a = model.tokenize(text, false, false);
    let b = model.tokenize(text, false, false);
    assert_eq!(a, b, "tokenize must be deterministic for the same input");
    assert!(!a.is_empty(), "non-empty text should produce at least one token");
}

#[test]
fn tokenize_length_matches_probe() {
    // The two-pass tokenize must end up with exactly the number of tokens
    // the C probe call promised. If `truncate(n as usize)` were given a
    // negative `n`, the returned vector would be the over-allocated probe
    // size rather than the real count.
    let model = common::load_model();
    let text = "Once upon a time, in a galaxy far, far away.";
    let tokens = model.tokenize(text, true, false);
    let probe_needed = unsafe {
        let n = llama_sys::llama_tokenize(
            llama_sys::llama_model_get_vocab(model.as_ptr() as *mut _),
            text.as_ptr() as *const i8,
            text.len() as i32,
            std::ptr::null_mut(),
            0,
            true,
            false,
        );
        // Probe returns negative-of-needed when n_tokens_max=0 forces a
        // "too small buffer" response.
        (-n) as usize
    };
    assert_eq!(
        tokens.len(),
        probe_needed,
        "tokenize() result length must match the probe-call's predicted size"
    );
}

#[test]
fn tokenize_roundtrip() {
    let model = common::load_model();
    let text = " Hello, world!";
    let tokens = model.tokenize(text, false, false);
    let reconstructed: String = tokens
        .iter()
        .map(|&t| model.token_to_piece(t).unwrap())
        .collect();
    assert_eq!(reconstructed.trim(), text.trim());
}

#[test]
fn token_to_piece_bos() {
    let model = common::load_model();
    if let Some(bos) = model.bos_token() {
        let piece = model.token_to_piece(bos);
        assert!(piece.is_ok());
    }
}

#[test]
fn decoder_start_token_none_for_decoder_only() {
    let model = common::load_model();
    assert!(model.decoder_start_token().is_none());
}

#[test]
fn chat_template_default() {
    let model = common::load_model();
    let _ = model.chat_template(None);
}

// ---------- Clone (Arc-backed Model) ----------

#[test]
fn model_clone_has_same_n_tokens() {
    let model = common::load_model();
    let cloned = model.clone();
    assert_eq!(
        model.n_tokens(),
        cloned.n_tokens(),
        "cloned model should report the same vocabulary size"
    );
}

#[test]
fn model_clone_can_tokenize() {
    let model = common::load_model();
    let cloned = model.clone();
    let tokens_orig = model.tokenize("hello world", false, false);
    let tokens_clone = cloned.tokenize("hello world", false, false);
    assert_eq!(
        tokens_orig, tokens_clone,
        "cloned model should produce identical tokenization"
    );
}

#[test]
fn model_clone_desc_matches() {
    let model = common::load_model();
    let cloned = model.clone();
    assert_eq!(
        model.desc(),
        cloned.desc(),
        "cloned model description should match original"
    );
}

#[test]
fn model_clone_vocab_queries_match() {
    let model = common::load_model();
    let cloned = model.clone();
    assert_eq!(model.bos_token(), cloned.bos_token());
    assert_eq!(model.eos_token(), cloned.eos_token());
    assert_eq!(model.has_decoder(), cloned.has_decoder());
    assert_eq!(model.has_encoder(), cloned.has_encoder());
}

#[test]
fn model_clone_token_to_piece_matches() {
    let model = common::load_model();
    let cloned = model.clone();
    if let Some(bos) = model.bos_token() {
        assert_eq!(
            model.token_to_piece(bos),
            cloned.token_to_piece(bos),
            "cloned model token_to_piece should match original"
        );
    }
}

#[test]
fn model_original_drop_does_not_affect_clone() {
    // Verify that dropping the original doesn't invalidate the clone (Arc).
    let cloned = {
        let model = common::load_model();
        model.clone()
        // model drops here
    };
    // Clone should still be usable
    assert!(cloned.n_tokens() > 0);
    let tokens = cloned.tokenize("test", false, false);
    assert!(!tokens.is_empty());
}
