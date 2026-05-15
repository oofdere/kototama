// Tests for src/common.rs (batch_add / batch_clear helpers).
// Named common_utils.rs to avoid shadowing the tests/common/ helper module.

use rusty_llama::common::{batch_add, batch_clear};
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
