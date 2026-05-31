mod common;

// Coverage for the two boolean flags on `Model::tokenize`:
//
//   pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32>
//
// The existing suite calls `tokenize` ~40 times but always with
// `parse_special = false`, so the parse-special FFI path (recognising literal
// special-token text like "<|start_story|>" and emitting its token id) was
// previously unexercised. `examples/simple.rs` ships with
// `parse_special = true`, so a regression here would silently break the
// example.
//
// The model under test, TinyStories-656K, has BOS=1 ("<|start_story|>") and
// EOS=2 ("<|end_story|>"). The tests below recover those literals from the
// vocab so they stay correct if the bundled test model is swapped.

fn bos_literal(model: &rusty_llama::Model) -> String {
    let bos = model.bos_token().expect("test model must have BOS");
    model.get_text(bos).to_string_lossy().into_owned()
}

fn eos_literal(model: &rusty_llama::Model) -> String {
    let eos = model.eos_token().expect("test model must have EOS");
    model.get_text(eos).to_string_lossy().into_owned()
}

// ---------- parse_special ----------

#[test]
fn parse_special_true_recognises_bos_literal_as_bos_token() {
    let model = common::load_model();
    let bos = model.bos_token().unwrap();
    let text = bos_literal(&model);
    let tokens = model.tokenize(&text, false, true);
    assert_eq!(
        tokens,
        vec![bos],
        "with parse_special=true, the BOS literal {text:?} should tokenize to exactly [bos]"
    );
}

#[test]
fn parse_special_true_recognises_eos_literal_as_eos_token() {
    let model = common::load_model();
    let eos = model.eos_token().unwrap();
    let text = eos_literal(&model);
    let tokens = model.tokenize(&text, false, true);
    assert_eq!(
        tokens,
        vec![eos],
        "with parse_special=true, the EOS literal {text:?} should tokenize to exactly [eos]"
    );
}

#[test]
fn parse_special_false_does_not_emit_bos_for_bos_literal() {
    let model = common::load_model();
    let bos = model.bos_token().unwrap();
    let text = bos_literal(&model);
    let tokens = model.tokenize(&text, false, false);
    assert!(
        !tokens.contains(&bos),
        "with parse_special=false, BOS literal {text:?} should be treated as ordinary text and never emit the BOS token id ({bos}); got {tokens:?}"
    );
}

#[test]
fn parse_special_flag_changes_tokenization_of_special_literal() {
    // Direct A/B: same input, only `parse_special` flipped. If the wrapper
    // ever forgot to pass the flag through to llama.cpp, these two would be
    // equal.
    let model = common::load_model();
    let text = bos_literal(&model);
    let with_parse = model.tokenize(&text, false, true);
    let without_parse = model.tokenize(&text, false, false);
    assert_ne!(
        with_parse, without_parse,
        "toggling parse_special on input {text:?} should change the tokenization"
    );
}

#[test]
fn parse_special_has_no_effect_on_plain_text() {
    // No special-token literal in the input -> the flag is a no-op.
    let model = common::load_model();
    let with_parse = model.tokenize("the cat sat on the mat", false, true);
    let without_parse = model.tokenize("the cat sat on the mat", false, false);
    assert_eq!(
        with_parse, without_parse,
        "parse_special must not change tokenization of text without special-token literals"
    );
}

#[test]
fn parse_special_true_in_mixed_text_keeps_surrounding_tokens() {
    // BOS literal sandwiched between plain text. With parse_special=true the
    // literal collapses to a single BOS id and the surrounding text still
    // tokenises normally — i.e. the output is longer than 1 and contains BOS.
    let model = common::load_model();
    let bos = model.bos_token().unwrap();
    let bos_text = bos_literal(&model);
    let mixed = format!("hello {bos_text} world");
    let tokens = model.tokenize(&mixed, false, true);
    assert!(
        tokens.contains(&bos),
        "mixed text {mixed:?} should contain the BOS token id ({bos}) under parse_special=true; got {tokens:?}"
    );
    assert!(
        tokens.len() > 1,
        "mixed text should still produce surrounding tokens, not just BOS; got {tokens:?}"
    );
}

// ---------- add_special ----------

#[test]
fn add_special_true_prepends_bos_when_vocab_adds_bos() {
    // TinyStories' tokenizer is configured with add_bos = true, so
    // `add_special = true` must prepend the BOS token to non-empty input.
    let model = common::load_model();
    assert!(
        model.get_add_bos(),
        "precondition: test model must be configured to add BOS; otherwise this test doesn't probe what it claims"
    );
    let bos = model.bos_token().unwrap();
    let tokens = model.tokenize("hello", true, false);
    assert_eq!(
        tokens.first().copied(),
        Some(bos),
        "with add_special=true and vocab.add_bos=true, output must start with BOS; got {tokens:?}"
    );
}

#[test]
fn add_special_false_omits_bos() {
    // Even though the vocab is configured to add BOS, passing
    // add_special=false must suppress it.
    let model = common::load_model();
    assert!(model.get_add_bos(), "precondition: test model must add BOS");
    let bos = model.bos_token().unwrap();
    let tokens = model.tokenize("hello", false, false);
    assert!(
        !tokens.is_empty(),
        "tokenizing a non-empty string should produce at least one token"
    );
    assert_ne!(
        tokens.first().copied(),
        Some(bos),
        "with add_special=false, output must not start with BOS; got {tokens:?}"
    );
}

#[test]
fn add_special_and_parse_special_are_independent() {
    // add_special toggles prepended specials; parse_special toggles inline
    // recognition. Each flag's effect should be visible in isolation:
    //
    //   - flipping add_special while parse_special=false: BOS appears/disappears at the head
    //   - flipping parse_special while add_special=false on a BOS literal: the
    //     payload tokens change but no prepended BOS is added by the wrapper
    let model = common::load_model();
    let bos = model.bos_token().unwrap();
    let bos_text = bos_literal(&model);

    let plain_no_add = model.tokenize("hello", false, false);
    let plain_add = model.tokenize("hello", true, false);
    assert_eq!(plain_add.first().copied(), Some(bos));
    assert_eq!(
        &plain_add[1..],
        plain_no_add.as_slice(),
        "add_special=true should only prepend BOS; remaining tokens must match"
    );

    let lit_no_parse = model.tokenize(&bos_text, false, false);
    let lit_parse = model.tokenize(&bos_text, false, true);
    assert_eq!(lit_parse, vec![bos]);
    assert_ne!(lit_parse, lit_no_parse);
}
