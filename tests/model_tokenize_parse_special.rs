//! Coverage for `Model::tokenize(_, _, parse_special)` — the third
//! argument of the tokenizer.
//!
//! Every existing test passes `parse_special=false` (28 callsites across
//! `tests/` and `benches/`). The shipped examples, by contrast, pass
//! `parse_special=true` (`examples/simple.rs:51`,
//! `examples/simple_chat.rs:78`) because chat-formatted prompts contain
//! literal special-token markup that needs to collapse to single token
//! ids. Until now, that production-relied-on code path had no test
//! coverage at all — a silent change that made the flag a no-op
//! wouldn't fail anything.
//!
//! These tests pin the contract by exercising the flag end-to-end
//! against the bundled TinyStories model:
//!   - On plain text it must be a no-op (parity with `false`).
//!   - On the literal text of a special token (EOS), it must collapse
//!     that text to the special token id, and disabling the flag must
//!     prevent that collapse.
//!   - It must remain orthogonal to `add_special` — flipping
//!     `parse_special` must not implicitly toggle BOS prepending.

mod common;

#[test]
fn parse_special_does_not_change_ordinary_text() {
    // Plain text without any special-token markup tokenizes the same
    // regardless of `parse_special` — there's nothing for the special-
    // token matcher to find.
    let model = common::load_model();
    let text = "hello world";
    let off = model.tokenize(text, false, false);
    let on = model.tokenize(text, false, true);
    assert_eq!(
        off, on,
        "parse_special=true should not affect plain text without special-token strings"
    );
}

#[test]
fn parse_special_does_not_change_empty_text() {
    let model = common::load_model();
    let off = model.tokenize("", false, false);
    let on = model.tokenize("", false, true);
    assert_eq!(off, on);
}

#[test]
fn parse_special_on_recognizes_eos_literal_text() {
    // With `parse_special=true`, the literal text of a special token
    // (e.g. "</s>") must be recognized and emitted as that single
    // special token id, instead of as the underlying byte/character
    // pieces.
    let model = common::load_model();
    let Some(eos) = model.eos_token() else {
        // Model exposes no EOS; nothing to exercise.
        return;
    };
    let text = model.get_text(eos).to_string_lossy().into_owned();
    if text.is_empty() {
        // Some vocabularies don't expose a UTF-8 literal for EOS. Without
        // a printable form the parse_special path isn't exercisable here.
        return;
    }

    let with_special = model.tokenize(&text, false, true);
    let without_special = model.tokenize(&text, false, false);

    assert!(
        with_special.contains(&eos),
        "parse_special=true should produce the EOS token id when given its literal text;\n\
         got {with_special:?} for text {text:?} (eos={eos})"
    );
    assert_ne!(
        with_special, without_special,
        "tokenization with parse_special=true must differ from =false on the EOS literal text"
    );
}

#[test]
fn parse_special_off_does_not_collapse_eos_literal_text() {
    // Mirror of the previous test: with `parse_special=false`, the
    // literal EOS text must NOT be collapsed into the single EOS token
    // id. If both flags produced `[eos]`, the flag would be redundant.
    let model = common::load_model();
    let Some(eos) = model.eos_token() else {
        return;
    };
    let text = model.get_text(eos).to_string_lossy().into_owned();
    if text.is_empty() {
        return;
    }
    let without_special = model.tokenize(&text, false, false);
    assert!(
        without_special != vec![eos],
        "parse_special=false must not collapse the literal EOS text into [eos];\n\
         got {without_special:?} for text {text:?}"
    );
}

#[test]
fn parse_special_is_orthogonal_to_add_special() {
    // `add_special` controls BOS prepending; `parse_special` controls
    // collapsing literal special-token text inside the input. Toggling
    // `parse_special` alone must not silently flip BOS behaviour.
    let model = common::load_model();
    let Some(bos) = model.bos_token() else {
        return;
    };

    let parse_no_add = model.tokenize("hello", false, true);
    assert!(
        parse_no_add.first() != Some(&bos),
        "parse_special=true alone must not prepend BOS (that's add_special's job);\n\
         got {parse_no_add:?}, bos={bos}"
    );

    let add_parse = model.tokenize("hello", true, true);
    assert_eq!(
        add_parse.first(),
        Some(&bos),
        "add_special=true must prepend BOS even when parse_special=true"
    );
}
