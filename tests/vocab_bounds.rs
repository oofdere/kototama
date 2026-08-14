//! Token-indexed vocabulary accessors must reject out-of-range ids.
//!
//! llama.cpp indexes its token table with `id_to_token.at(id)` (which throws
//! `std::out_of_range` — a foreign exception that aborts the process when it
//! unwinds into Rust) or with the unchecked `id_to_token[id]`, which reads out
//! of bounds. Every accessor below is reachable from safe code with an
//! arbitrary `i32`, so the bounds check has to live on the Rust side.

mod common;

use common::load_model;

fn out_of_range_ids(n_tokens: i32) -> Vec<i32> {
    vec![-1, -3, i32::MIN, n_tokens, n_tokens + 1, i32::MAX]
}

#[test]
fn is_valid_token_matches_vocab_range() {
    let model = load_model();
    let n = model.n_tokens();
    assert!(n > 0);

    assert!(model.is_valid_token(0));
    assert!(model.is_valid_token(n - 1));
    for id in out_of_range_ids(n) {
        assert!(!model.is_valid_token(id), "{id} should be invalid");
    }
}

#[test]
fn get_score_out_of_range_is_none() {
    let model = load_model();
    for id in out_of_range_ids(model.n_tokens()) {
        assert!(model.get_score(id).is_none(), "get_score({id})");
    }
}

#[test]
fn get_text_out_of_range_is_none() {
    let model = load_model();
    for id in out_of_range_ids(model.n_tokens()) {
        assert!(model.get_text(id).is_none(), "get_text({id})");
    }
}

#[test]
fn get_attr_out_of_range_is_none() {
    let model = load_model();
    for id in out_of_range_ids(model.n_tokens()) {
        assert!(model.get_attr(id).is_none(), "get_attr({id})");
    }
}

#[test]
fn predicates_out_of_range_are_false() {
    let model = load_model();
    for id in out_of_range_ids(model.n_tokens()) {
        assert!(!model.is_control(id), "is_control({id})");
        assert!(!model.is_eog(id), "is_eog({id})");
    }
}

#[test]
fn token_to_piece_out_of_range_is_err() {
    let model = load_model();
    for id in out_of_range_ids(model.n_tokens()) {
        assert!(model.token_to_piece(id).is_err(), "token_to_piece({id})");
    }
}

#[test]
fn in_range_accessors_still_work() {
    let model = load_model();
    let n = model.n_tokens();
    for id in [0, 1, n / 2, n - 1] {
        assert!(model.get_score(id).is_some());
        assert!(model.get_text(id).is_some());
        assert!(model.get_attr(id).is_some());
        assert!(model.token_to_piece(id).is_ok());
        let _ = model.is_control(id);
        let _ = model.is_eog(id);
    }
}
