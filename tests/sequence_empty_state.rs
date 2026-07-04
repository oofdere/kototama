mod common;

use rusty_llama::Context;

// ---------- extend(&[]) no-op contract ----------
//
// `Sequence::extend` is documented in `src/sequence.rs` as a loop of `push` over
// the input slice. That makes the empty-slice case a pure no-op — no decode,
// no KV mutation, no cached-logits change — but nothing in `tests/sequence.rs`
// pins that. A regression that eagerly decoded on `extend(&[])` (e.g. calling
// `self.decode()` unconditionally) would silently break callers who tokenize a
// possibly-empty prompt.

#[test]
fn extend_empty_slice_does_not_change_len() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.extend(&[]);
    assert_eq!(seq.len(), 0);
    assert!(seq.is_empty());
}

#[test]
fn extend_empty_slice_after_push_preserves_len_and_tokens() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let snapshot: Vec<i32> = seq.tokens().to_vec();

    seq.extend(&[]);

    assert_eq!(seq.len(), len_before);
    assert_eq!(seq.tokens(), snapshot.as_slice());
}

#[test]
fn extend_empty_slice_preserves_cached_logits() {
    // `push` sets `logits`; a no-op `extend(&[])` must NOT invalidate that cache.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    let logits_before: Vec<f32> = seq.logits().unwrap().to_vec();

    seq.extend(&[]);

    let logits_after = seq
        .logits()
        .expect("extend(&[]) must not invalidate cached logits");
    assert_eq!(logits_after, logits_before.as_slice());
}

#[test]
fn extend_empty_before_first_push_leaves_logits_none() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    seq.extend(&[]);

    assert!(
        seq.logits().is_none(),
        "extend(&[]) on a fresh sequence must not synthesize logits"
    );
}

// ---------- pos_min / pos_max reset after pop-to-empty ----------
//
// `tests/sequence.rs::pos_min_max_after_push` covers the fresh-sequence (-1)
// and post-push (>= 0) cases, but never the round-trip: after popping every
// token, do `pos_min`/`pos_max` return to `-1`? That's the invariant llama.cpp
// documents for a freshly-cleared KV slot, and it's what a caller checking
// "is this sequence empty at the KV layer?" via `pos_max == -1` would rely on.

#[test]
fn pos_min_returns_to_minus_one_after_pop_to_empty() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    assert!(seq.pos_min() >= 0);

    while seq.pop().is_some() {}
    assert_eq!(seq.len(), 0);
    assert_eq!(
        seq.pos_min(),
        -1,
        "pos_min must return to -1 after every token is popped"
    );
}

#[test]
fn pos_max_returns_to_minus_one_after_pop_to_empty() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.pos_max() >= seq.pos_min());

    while seq.pop().is_some() {}
    assert_eq!(seq.len(), 0);
    assert_eq!(
        seq.pos_max(),
        -1,
        "pos_max must return to -1 after every token is popped"
    );
}

#[test]
fn pos_min_max_reset_after_full_range_remove() {
    // Same contract but via `remove(0..n)` instead of repeated `pop()`.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    let n = seq.len();
    assert!(seq.pos_min() >= 0);
    assert!(seq.pos_max() >= 0);

    assert!(seq.remove(0..n));
    assert_eq!(seq.len(), 0);
    assert_eq!(
        seq.pos_min(),
        -1,
        "pos_min must return to -1 after removing the full range"
    );
    assert_eq!(
        seq.pos_max(),
        -1,
        "pos_max must return to -1 after removing the full range"
    );
}

// ---------- New sequence starts at pos == -1 on both accessors ----------
//
// `tests/sequence.rs` covers this for `pos_min` and `pos_max` together in a
// single assertion. Split them so a regression on one accessor doesn't get
// masked by the other still returning -1.

#[test]
fn fresh_sequence_pos_min_is_minus_one() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let seq = ctx.sequence().unwrap();
    assert_eq!(seq.pos_min(), -1);
}

#[test]
fn fresh_sequence_pos_max_is_minus_one() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let seq = ctx.sequence().unwrap();
    assert_eq!(seq.pos_max(), -1);
}
