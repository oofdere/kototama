mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

#[test]
fn push_increases_len() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.len(), 0);
    let tokens = model.tokenize("hi", false, false);
    seq.push(tokens[0]);
    assert_eq!(seq.len(), 1);
}

#[test]
fn extend_fills_tokens() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    let n = tokens.len();
    seq.extend(&tokens);
    assert_eq!(seq.len(), n);
}

#[test]
fn tokens_accessor_matches_push_order() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("abc", false, false);
    seq.extend(&tokens);
    assert_eq!(seq.tokens(), tokens.as_slice());
}

#[test]
fn index_operator() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    assert_eq!(seq[0], tokens[0]);
}

#[test]
fn get_returns_token() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens);
    assert_eq!(seq.get(0), Some(tokens[0]));
    assert_eq!(seq.get(999), None);
}

#[test]
fn pop_decreases_len() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let popped = seq.pop();
    assert!(popped.is_some());
    assert_eq!(seq.len(), len_before - 1);
}

#[test]
fn pop_empty_returns_none() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.pop(), None);
}

#[test]
fn logits_len_equals_vocab_size_after_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.push(tokens[0]);
    assert_eq!(seq.logits().unwrap().len(), model.n_tokens() as usize);
}

#[test]
fn remove_range() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    let n = tokens.len();
    seq.extend(&tokens);
    assert_eq!(seq.len(), n);
    seq.remove(0..1);
    assert_eq!(seq.len(), n - 1);
}

#[test]
fn pos_min_max_after_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.pos_min(), -1);
    assert_eq!(seq.pos_max(), -1);
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.pos_min() >= 0);
    assert!(seq.pos_max() >= seq.pos_min());
}

#[test]
fn copy_to() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);
    src.copy_to(&mut dst, 0..tokens.len());
    assert_eq!(dst.tokens(), src.tokens());
}

#[test]
fn multiple_sequences_independent() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq1 = ctx.sequence().unwrap();
    let mut seq2 = ctx.sequence().unwrap();
    let tokens1 = model.tokenize("hello", false, false);
    let tokens2 = model.tokenize("world", false, false);
    seq1.extend(&tokens1);
    seq2.extend(&tokens2);
    assert_eq!(seq1.tokens(), tokens1.as_slice());
    assert_eq!(seq2.tokens(), tokens2.as_slice());
    assert_ne!(seq1.tokens(), seq2.tokens());
}

#[test]
fn multiple_sequences_generate_different_logits() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq1 = ctx.sequence().unwrap();
    let mut seq2 = ctx.sequence().unwrap();
    let tokens1 = model.tokenize("hello", false, false);
    let tokens2 = model.tokenize("world", false, false);
    seq1.extend(&tokens1);
    seq2.extend(&tokens2);
    assert_ne!(seq1.logits().unwrap(), seq2.logits().unwrap());
}

#[test]
fn free_slots_decreases_with_checkout() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let initial_slots = ctx.free_slots();
    let _seq1 = ctx.sequence().unwrap();
    assert_eq!(ctx.free_slots(), initial_slots - 1);
    let _seq2 = ctx.sequence().unwrap();
    assert_eq!(ctx.free_slots(), initial_slots - 2);
}

#[test]
fn sequence_checkout_up_to_n_seq_max() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.n_seq_max = 3;
    let ctx = Context::new(&model, &params).unwrap();
    let _seq1 = ctx.sequence().unwrap();
    let _seq2 = ctx.sequence().unwrap();
    let _seq3 = ctx.sequence().unwrap();
    assert!(ctx.sequence().is_none());
}

#[test]
fn dropping_sequence_frees_slot() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let initial_slots = ctx.free_slots();
    {
        let _seq = ctx.sequence().unwrap();
        assert_eq!(ctx.free_slots(), initial_slots - 1);
    }
    assert_eq!(ctx.free_slots(), initial_slots);
    let _new_seq = ctx.sequence().unwrap();
}

#[test]
fn copy_from() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);
    dst.copy_from(&src, 0..tokens.len());
    assert_eq!(dst.tokens(), src.tokens());
}

// ---------- Sequence::is_empty() (new in this PR) ----------

#[test]
fn is_empty_true_before_any_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let seq = ctx.sequence().unwrap();
    assert!(seq.is_empty(), "freshly created sequence should be empty");
}

#[test]
fn is_empty_false_after_push() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.push(tokens[0]);
    assert!(!seq.is_empty(), "sequence should not be empty after a push");
}

#[test]
fn is_empty_false_after_extend() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(!seq.is_empty());
}

#[test]
fn is_empty_true_after_pop_clears_sequence() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    // push exactly one token then pop it — sequence should be empty again
    for &t in &tokens {
        seq.push(t);
    }
    for _ in 0..tokens.len() {
        seq.pop();
    }
    assert!(seq.is_empty(), "sequence should be empty after all tokens are popped");
}

#[test]
fn is_empty_consistent_with_len() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.is_empty(), seq.len() == 0);
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert_eq!(seq.is_empty(), seq.len() == 0);
}

// ---------- Sequence::sample() (moved from Context to Sequence in this PR) ----------

#[test]
fn sequence_sample_greedy_is_valid_token() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::greedy());
    let token = seq.sample(&chain);
    assert!(token >= 0 && token < model.n_tokens(),
        "sampled token should be within vocab range");
}

#[test]
fn sequence_sample_matches_argmax() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("once upon", false, false);
    seq.extend(&tokens);

    let argmax = seq
        .logits()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i32)
        .unwrap();

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::greedy());
    let sampled = seq.sample(&chain);

    assert_eq!(sampled, argmax,
        "Sequence::sample with greedy should match manual argmax");
}

#[test]
fn sequence_sample_with_temperature_in_vocab_range() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::temp(0.8))
        .add(Sampler::top_k(40))
        .add(Sampler::dist(123));
    let token = seq.sample(&chain);
    assert!(token >= 0 && token < model.n_tokens(),
        "temperature-sampled token should be in vocab range");
}

#[test]
fn sequence_sample_does_not_require_mut() {
    // sample() takes &self — verify it compiles and runs with a shared borrow
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("test", false, false);
    seq.extend(&tokens);

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::greedy());

    // Both immutable borrows should coexist
    let _logits = seq.logits();
    let token = seq.sample(&chain);
    assert!(token >= 0 && token < model.n_tokens());
}

// ---------- Logits cache invalidation (regression coverage for 097db1d, 8436f38) ----------
//
// Mutations to the KV cache or the token vector must drop the cached logits,
// because the cache reflects the model's distribution *for the last decoded
// token*. After pop/remove/kv_remove/kv_copy/kv_shift the last token has
// either changed or moved, so any cached vector is stale.

#[test]
fn logits_none_after_pop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some(), "logits should be cached after push");
    seq.pop();
    assert!(seq.logits().is_none(), "pop() must invalidate cached logits");
}

#[test]
fn logits_none_after_remove() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some());
    let ok = seq.remove(0..1);
    assert!(ok);
    assert!(seq.logits().is_none(), "remove() must invalidate cached logits");
}

#[test]
fn logits_none_after_kv_remove() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some());
    let n = seq.len() as i32;
    let ok = seq.kv_remove(0..n);
    assert!(ok);
    assert!(seq.logits().is_none(), "kv_remove() must invalidate cached logits");
}

#[test]
fn logits_none_on_copy_to_destination() {
    // copy_to() writes into `other`'s KV slot; `other`'s cached logits no
    // longer match the new last token and must be cleared. (Regression for
    // 8436f38: "invalidate destination logits in kv_copy()".)
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);
    assert!(dst.logits().is_some(), "dst should have cached logits before copy");
    src.copy_to(&mut dst, 0..tokens.len());
    assert!(
        dst.logits().is_none(),
        "copy_to() must invalidate the destination's cached logits"
    );
}

#[test]
fn logits_none_on_copy_from_destination() {
    // copy_from() is implemented in terms of copy_to(), so the same
    // invalidation should reach the receiver.
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);
    dst.copy_from(&src, 0..tokens.len());
    assert!(dst.logits().is_none(), "copy_from() must invalidate the receiver's logits");
}

#[test]
fn logits_none_after_kv_shift() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    if !ctx.can_shift() {
        return;
    }
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    assert!(seq.logits().is_some());
    let n = seq.len() as i32;
    seq.kv_shift(0..n, 1);
    assert!(seq.logits().is_none(), "kv_shift() must invalidate cached logits");
}

// ---------- Sequence::decode() (added in cd5b7c6) ----------
//
// decode() re-runs decode on the *current* last token to repopulate the
// logits cache after a mutation. On an empty sequence it must be a no-op.

#[test]
fn decode_empty_sequence_is_noop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(seq.is_empty());
    seq.decode();
    assert!(seq.is_empty(), "decode() on empty sequence must not push a token");
    assert!(
        seq.logits().is_none(),
        "decode() on empty sequence must not fabricate logits"
    );
}

#[test]
fn decode_restores_logits_after_pop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    seq.pop();
    assert!(seq.logits().is_none(), "pop() invalidates logits");
    seq.decode();
    assert!(
        seq.logits().is_some(),
        "decode() must repopulate logits after pop()"
    );
    assert_eq!(
        seq.logits().unwrap().len(),
        model.n_tokens() as usize,
        "decoded logits should match vocab size"
    );
}

#[test]
fn decode_restores_logits_after_trailing_remove() {
    // Use a trailing-range remove so KV positions stay consistent with the
    // token vector — remove from the middle desyncs positions and is out of
    // scope for this test.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    let n = seq.len();
    let ok = seq.remove((n - 1)..n);
    assert!(ok);
    assert!(seq.logits().is_none());
    seq.decode();
    assert!(
        seq.logits().is_some(),
        "decode() must repopulate logits after a trailing remove()"
    );
}

#[test]
fn decode_preserves_token_count() {
    // decode() must not push a new token — it only refreshes the logits cache.
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens);
    let len_before = seq.len();
    let tokens_before = seq.tokens().to_vec();
    seq.decode();
    assert_eq!(seq.len(), len_before, "decode() must not change len()");
    assert_eq!(
        seq.tokens(),
        tokens_before.as_slice(),
        "decode() must not change the token vector"
    );
}
