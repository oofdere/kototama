// Tests for src/common.rs (batch_add / batch_clear helpers).
// Named common_utils.rs to avoid shadowing the tests/common/ helper module.

use rusty_llama::common::{batch_add, batch_clear, BatchAddError};
use rusty_llama::Batch;

fn make_batch(cap: i32) -> Batch {
    Batch::init_token(cap, 1)
}

#[test]
fn batch_add_ok() {
    let mut batch = make_batch(4);
    batch_add(&mut batch, 1, 0, &[0], true).unwrap();
    assert_eq!(batch.n_tokens, 1);
}

#[test]
fn batch_add_multiple_tokens() {
    let mut batch = make_batch(4);
    batch_add(&mut batch, 1, 0, &[0], false).unwrap();
    batch_add(&mut batch, 2, 1, &[0], false).unwrap();
    batch_add(&mut batch, 3, 2, &[0], true).unwrap();
    assert_eq!(batch.n_tokens, 3);
}

#[test]
fn batch_add_exceed_capacity_returns_err() {
    let mut batch = make_batch(2);
    batch_add(&mut batch, 1, 0, &[0], false).unwrap();
    batch_add(&mut batch, 2, 1, &[0], false).unwrap();
    // Third add exceeds capacity — seq_id pointer at that index will be null
    let result = batch_add(&mut batch, 3, 2, &[0], false);
    assert!(result.is_err());
}

#[test]
fn batch_clear_resets_n_tokens() {
    let mut batch = make_batch(4);
    batch_add(&mut batch, 1, 0, &[0], false).unwrap();
    batch_add(&mut batch, 2, 1, &[0], false).unwrap();
    assert_eq!(batch.n_tokens, 2);
    batch_clear(&mut batch);
    assert_eq!(batch.n_tokens, 0);
}

#[test]
fn batch_clear_allows_reuse() {
    let mut batch = make_batch(4);
    batch_add(&mut batch, 1, 0, &[0], false).unwrap();
    batch_clear(&mut batch);
    batch_add(&mut batch, 42, 0, &[0], true).unwrap();
    assert_eq!(batch.n_tokens, 1);
}

#[test]
fn batch_add_too_many_seq_ids_returns_err_instead_of_overflowing() {
    // Batch is allocated with n_seq_max=1, so each per-token seq_id sub-array
    // holds exactly one llama_seq_id. Passing more would have written past the
    // end of that sub-array (heap buffer overflow). batch_add must reject it.
    let mut batch = Batch::init_token(4, 1);
    let too_many = [0, 1, 2, 3];
    let result = batch_add(&mut batch, 1, 0, &too_many, true);
    assert!(matches!(result, Err(BatchAddError::TooManySeqIds)));
    // The rejected call must leave the batch untouched so the caller can recover.
    assert_eq!(batch.n_tokens, 0);
}

#[test]
fn batch_add_exact_n_seq_max_is_accepted() {
    // n_seq_max=3 means a seq_ids slice of length 3 fits exactly.
    let mut batch = Batch::init_token(2, 3);
    batch_add(&mut batch, 1, 0, &[0, 1, 2], true).unwrap();
    assert_eq!(batch.n_tokens, 1);
}

#[test]
fn batch_n_seq_max_round_trips() {
    let batch = Batch::init_token(4, 7);
    assert_eq!(batch.n_seq_max(), 7);
}
