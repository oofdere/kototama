//! Soundness: text containing a NUL byte must not reach llama.cpp.
//!
//! `llama_tokenize` falls back to `llama_vocab::byte_to_token` for characters
//! that are not in the vocab. For a NUL byte that lookup uses an empty
//! `std::string` key and `unordered_map::at` throws `std::out_of_range`. The
//! exception unwinds through the `extern "C"` boundary into safe Rust, which
//! is undefined behaviour: the process aborts with
//! "fatal runtime error: Rust cannot catch foreign exceptions".

use rusty_llama::test_common::load_model;
use rusty_llama::TokenizeError;

#[test]
fn try_tokenize_rejects_leading_nul() {
    let model = load_model();
    assert_eq!(
        model.try_tokenize("\u{0}abc", false, false),
        Err(TokenizeError::InteriorNul(0))
    );
}

#[test]
fn try_tokenize_rejects_interior_nul() {
    let model = load_model();
    assert_eq!(
        model.try_tokenize("ab\u{0}cd", true, true),
        Err(TokenizeError::InteriorNul(2))
    );
}

#[test]
fn try_tokenize_accepts_nul_free_text() {
    let model = load_model();
    let tokens = model.try_tokenize("hello world", false, false).unwrap();
    assert!(!tokens.is_empty());
    assert_eq!(tokens, model.tokenize("hello world", false, false));
}

#[test]
fn tokenize_panics_instead_of_aborting_on_nul() {
    let model = load_model();
    let err = std::panic::catch_unwind(|| model.tokenize("a\u{0}b", false, false));
    assert!(err.is_err());
}
