// Coverage for two public Sequence methods that had no tests:
//   - `Sequence::kv_shift` (llama_memory_seq_add wrapper): shifts KV-cache
//     positions of a range by a delta. Primitive behind sliding-window
//     context management.
//   - `Sequence::decode`: re-runs the last token through the model to
//     refresh cached logits after a mutation clears them.
//
// Both are stable public APIs. Missing coverage lets refactors silently
// break either one.
//
// Note: decode() interacts subtly with KV-cache state — re-decoding a
// token at a position that still has a live KV entry is rejected by the
// llama.cpp backend as `InvalidInput`. That failure mode is the subject
// of a separate outstanding PR (#58). This file exercises only the
// contract that decode() promises unconditionally: on an empty sequence
// it must be a no-op.

mod common;

use rusty_llama::Context;

// ---------- Sequence::kv_shift ----------

#[test]
fn kv_shift_invalidates_logits_cache() {
    let (model, params) = common::load_model_and_context();
    if !Context::new(&model, &params).unwrap().can_shift() {
        eprintln!("skipping: backend reports can_shift() == false");
        return;
    }
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    assert!(
        seq.logits().is_some(),
        "logits should be cached after extend()"
    );

    let end = seq.pos_max() + 1;
    seq.kv_shift(0..end, 0);

    assert!(
        seq.logits().is_none(),
        "kv_shift must invalidate the cached logits (they no longer describe the current KV state)"
    );
}

#[test]
fn kv_shift_positive_delta_moves_pos_max() {
    let (model, params) = common::load_model_and_context();
    if !Context::new(&model, &params).unwrap().can_shift() {
        eprintln!("skipping: backend reports can_shift() == false");
        return;
    }
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let pos_min_before = seq.pos_min();
    let pos_max_before = seq.pos_max();
    let end = pos_max_before + 1;

    seq.kv_shift(0..end, 3);

    assert_eq!(
        seq.pos_min(),
        pos_min_before + 3,
        "pos_min should shift by delta"
    );
    assert_eq!(
        seq.pos_max(),
        pos_max_before + 3,
        "pos_max should shift by delta"
    );
}

#[test]
fn kv_shift_negative_delta_moves_positions_back() {
    let (model, params) = common::load_model_and_context();
    if !Context::new(&model, &params).unwrap().can_shift() {
        eprintln!("skipping: backend reports can_shift() == false");
        return;
    }
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let end = seq.pos_max() + 1;
    seq.kv_shift(0..end, 5);
    let after_up_min = seq.pos_min();
    let after_up_max = seq.pos_max();

    seq.kv_shift(after_up_min..(after_up_max + 1), -5);

    assert_eq!(
        seq.pos_min(),
        after_up_min - 5,
        "negative delta should move pos_min back"
    );
    assert_eq!(
        seq.pos_max(),
        after_up_max - 5,
        "negative delta should move pos_max back"
    );
}

#[test]
fn kv_shift_leaves_tokens_slice_untouched() {
    // kv_shift touches KV-cache positions only; it must NOT alter the
    // sequence's local token vector or its len().
    let (model, params) = common::load_model_and_context();
    if !Context::new(&model, &params).unwrap().can_shift() {
        eprintln!("skipping: backend reports can_shift() == false");
        return;
    }
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let tokens_before: Vec<i32> = seq.tokens().to_vec();
    let len_before = seq.len();
    let end = seq.pos_max() + 1;

    seq.kv_shift(0..end, 2);

    assert_eq!(
        seq.tokens(),
        tokens_before.as_slice(),
        "kv_shift must not mutate the token vector"
    );
    assert_eq!(seq.len(), len_before, "kv_shift must not change len()");
}

// ---------- Sequence::decode ----------

#[test]
fn decode_on_empty_sequence_is_noop() {
    // Guards the `if let Some(&last) = self.tokens.last()` branch:
    // decode() on a freshly checked-out sequence must not panic and must
    // leave the (still-empty) logits cache untouched.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(seq.is_empty());
    assert!(seq.logits().is_none());

    seq.decode();

    assert!(seq.is_empty(), "decode() must not add tokens");
    assert!(
        seq.logits().is_none(),
        "decode() on an empty sequence must leave logits as None"
    );
}
