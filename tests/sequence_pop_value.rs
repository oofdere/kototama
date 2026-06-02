mod common;

use rusty_llama::Context;

// Value-correctness coverage for `Sequence::pop()`.
//
// The existing suite checks that `pop()` reduces `len()` (sequence.rs::pop_decreases_len),
// that it returns `None` on an empty sequence (pop_empty_returns_none), that the
// sequence becomes empty after popping every token (is_empty_true_after_pop_clears_sequence),
// and that `pop()` invalidates the cached logits (PR #58, logits_none_after_pop).
//
// What is *not* covered is the value returned by `pop()` — i.e. LIFO semantics —
// nor that the remaining tokens after a `pop()` are the leading prefix. A
// regression that returned the wrong popped token, or accidentally popped the
// front instead of the back, would pass every existing test. The same holds
// for the `push`-after-`pop` round-trip: nothing currently asserts that
// re-pushing repopulates the logits cache or that the cache contents match a
// freshly-built sequence with the same tokens.
//
// These tests live in their own integration crate to avoid conflicting with
// the multiple in-flight PRs that touch tests/sequence.rs.

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

// ---------- pop() returns the actual last token (LIFO) ----------

#[test]
fn pop_returns_last_pushed_token() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2, "test prompt must produce >=2 tokens");
    seq.extend(&tokens);

    let last = *tokens.last().unwrap();
    let popped = seq.pop();
    assert_eq!(
        popped,
        Some(last),
        "pop() must return the most recently pushed token (LIFO)"
    );
}

#[test]
fn repeated_pop_returns_tokens_in_reverse_push_order() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world today", false, false);
    assert!(tokens.len() >= 3, "test prompt must produce >=3 tokens");
    seq.extend(&tokens);

    let mut popped = Vec::with_capacity(tokens.len());
    while let Some(t) = seq.pop() {
        popped.push(t);
    }

    let mut expected: Vec<i32> = tokens.iter().copied().rev().collect();
    assert_eq!(
        popped, expected,
        "repeated pop() must yield the pushed tokens in reverse order"
    );

    // And sanity: the sequence is now empty, matching the count we popped.
    assert!(seq.is_empty());
    expected.reverse();
    assert_eq!(expected, tokens, "test invariant: tokens slice was unchanged");
}

#[test]
fn pop_leaves_leading_prefix_intact() {
    // After popping the tail, `tokens()` must equal the leading prefix of the
    // original push order — never a suffix, never a permutation.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world today", false, false);
    assert!(tokens.len() >= 3);
    seq.extend(&tokens);

    let _ = seq.pop();
    assert_eq!(
        seq.tokens(),
        &tokens[..tokens.len() - 1],
        "after one pop the sequence must be the leading N-1 prefix"
    );

    let _ = seq.pop();
    assert_eq!(
        seq.tokens(),
        &tokens[..tokens.len() - 2],
        "after two pops the sequence must be the leading N-2 prefix"
    );
}

#[test]
fn pop_to_empty_then_pop_returns_none() {
    // Walking the sequence empty with pop() and then popping once more must
    // return None — i.e. the empty-check still works after a prior non-empty
    // run, not just on a freshly-checked-out sequence.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    for _ in 0..tokens.len() {
        assert!(seq.pop().is_some());
    }
    assert!(seq.is_empty());
    assert_eq!(seq.pop(), None, "pop() on an emptied sequence must return None");
}

// ---------- push after pop ----------

#[test]
fn push_after_pop_repopulates_logits() {
    // `pop()` clears the cached logits. A subsequent `push()` must rebuild
    // them — without this, callers would observe `logits() == None` even
    // after pushing a fresh token, which would break sampling.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let popped = seq.pop().expect("non-empty sequence must pop");
    assert!(seq.logits().is_none(), "pop() invalidates the logits cache");

    // Push the same token back at the same position.
    seq.push(popped);
    assert!(
        seq.logits().is_some(),
        "push() after pop() must repopulate the logits cache"
    );
    assert_eq!(
        seq.logits().unwrap().len(),
        model.n_tokens() as usize,
        "repopulated logits must span the full vocabulary"
    );
}

#[test]
fn pop_then_push_same_token_yields_same_logits_as_no_op() {
    // The pop+push round-trip ends in the same logical state as the original
    // sequence: same tokens in the same order. The logits cache, which is a
    // pure function of the decoded prefix, must therefore match. If pop()
    // failed to fully clear the KV slot (or push() landed on the wrong
    // position), the logits would diverge.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let before = seq.logits().expect("logits cached after extend").to_vec();

    let popped = seq.pop().unwrap();
    seq.push(popped);

    let after = seq.logits().expect("logits cached after re-push");
    assert_eq!(
        before.len(),
        after.len(),
        "logits vectors must match in length across a pop+push round-trip"
    );
    assert_eq!(
        before, after,
        "pop+push of the same token must produce identical logits"
    );
}

#[test]
fn pop_then_push_preserves_token_vector() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let popped = seq.pop().unwrap();
    seq.push(popped);

    assert_eq!(
        seq.tokens(),
        tokens.as_slice(),
        "pop+push of the same token must leave the token vector unchanged"
    );
}

// ---------- extend with an empty slice ----------

#[test]
fn extend_with_empty_slice_is_noop() {
    // `extend()` iterates its argument and calls `push()` per token, so an
    // empty slice should perform zero decodes — leaving the sequence empty
    // and the logits cache untouched (still `None`).
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(seq.is_empty());
    assert!(seq.logits().is_none());

    seq.extend(&[]);

    assert!(seq.is_empty(), "extend(&[]) must not push any tokens");
    assert!(
        seq.logits().is_none(),
        "extend(&[]) must not fabricate logits on an empty sequence"
    );
}

#[test]
fn extend_with_empty_slice_after_push_leaves_state_intact() {
    // Same property after the cache is populated: extending by nothing must
    // not invalidate or alter the existing cached logits.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let logits_before = seq.logits().expect("cached after extend").to_vec();

    seq.extend(&[]);

    assert_eq!(seq.len(), len_before, "extend(&[]) must not change len()");
    assert_eq!(
        seq.tokens(),
        tokens.as_slice(),
        "extend(&[]) must not change the token vector"
    );
    assert_eq!(
        seq.logits().unwrap(),
        logits_before.as_slice(),
        "extend(&[]) must not perturb the logits cache"
    );
}
