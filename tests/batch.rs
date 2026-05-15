use rusty_llama::Batch;
use std::num::NonZeroI32;

#[test]
fn init_token_batch_starts_empty() {
    let batch = Batch::init_token(512, 1);
    assert_eq!(batch.n_tokens, 0);
    assert!(!batch.token.is_null());
    assert!(!batch.pos.is_null());
    assert!(!batch.logits.is_null());
}

#[test]
fn init_embd_batch_starts_empty() {
    let n_embd = NonZeroI32::new(64).unwrap();
    let batch = Batch::init_embd(32, n_embd, 1);
    assert_eq!(batch.n_tokens, 0);
    assert!(!batch.embd.is_null());
}

#[test]
fn deref_exposes_inner_n_tokens() {
    let batch = Batch::init_token(16, 1);
    assert_eq!((*batch).n_tokens, 0);
}

#[test]
fn deref_mut_allows_mutation() {
    let mut batch = Batch::init_token(16, 1);
    (*batch).n_tokens = 3;
    assert_eq!(batch.n_tokens, 3);
}
