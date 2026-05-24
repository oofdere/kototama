// Tests for src/common.rs (batch_add / batch_clear helpers).
// Named common_utils.rs to avoid shadowing the tests/common/ helper module.

use rusty_llama::common::{batch_add, batch_clear, BatchAddError};
use rusty_llama::Batch;
use std::num::NonZeroI32;

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
fn batch_add_on_embd_batch_returns_err_instead_of_writing_null() {
    // `Batch::init_embd` leaves `batch.token` null. Before the fix, `batch_add`
    // unconditionally wrote `id` through `batch.token`, which is reachable from
    // safe Rust and is undefined behaviour (a write through a null pointer).
    let n_embd = NonZeroI32::new(8).unwrap();
    let mut batch = Batch::init_embd(4, n_embd, 1);
    assert!(batch.token.is_null(), "embd batches must have a null token buffer");

    let res = batch_add(&mut batch, 1, 0, &[0], true);
    match res {
        Err(BatchAddError::NotTokenBatch) => {}
        other => panic!("expected NotTokenBatch, got {:?}", other),
    }
    // The batch must be left untouched on the error path.
    assert_eq!(batch.n_tokens, 0);
}
