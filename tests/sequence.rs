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

// ---------- Cross-context kv_copy / copy_to / copy_from guard ----------

#[test]
#[should_panic(expected = "same Context")]
fn kv_copy_across_contexts_panics_instead_of_aborting() {
    // Without the same-context guard, this routes `other.id` through
    // `self.ctx`'s actor and either corrupts an unrelated sequence inside
    // `self.ctx` or trips llama.cpp's `GGML_ASSERT(seq_id_dst < ...)` and
    // aborts the process — both reachable from purely safe code.
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx1 = Context::new(&model, &params).unwrap();
    let ctx2 = Context::new(&model, &params).unwrap();

    let src = ctx1.sequence().unwrap();
    let mut dst = ctx2.sequence().unwrap();

    src.kv_copy(&mut dst, 0..0);
}

#[test]
#[should_panic(expected = "same Context")]
fn copy_to_across_contexts_panics_instead_of_aborting() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx1 = Context::new(&model, &params).unwrap();
    let ctx2 = Context::new(&model, &params).unwrap();

    let src = ctx1.sequence().unwrap();
    let mut dst = ctx2.sequence().unwrap();

    src.copy_to(&mut dst, 0..0);
}

#[test]
#[should_panic(expected = "same Context")]
fn copy_from_across_contexts_panics_instead_of_aborting() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx1 = Context::new(&model, &params).unwrap();
    let ctx2 = Context::new(&model, &params).unwrap();

    let src = ctx1.sequence().unwrap();
    let mut dst = ctx2.sequence().unwrap();

    dst.copy_from(&src, 0..0);
}

#[test]
fn copy_to_across_cloned_context_handles_works() {
    // Cloned Context handles share the same actor + KV cache, so cross-handle
    // copies are well-defined and must still succeed.
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx1 = Context::new(&model, &params).unwrap();
    let ctx2 = ctx1.clone();

    let mut src = ctx1.sequence().unwrap();
    let mut dst = ctx2.sequence().unwrap();

    let tokens = model.tokenize("hi", false, false);
    src.extend(&tokens);
    dst.extend(&tokens);

    src.copy_to(&mut dst, 0..tokens.len());
    assert_eq!(dst.tokens(), src.tokens());
}

#[test]
fn context_ptr_eq_matches_clones_but_not_independent_news() {
    let (model, _) = setup();
    let params = common::test_ctx_params();
    let ctx1 = Context::new(&model, &params).unwrap();
    let ctx2 = ctx1.clone();
    let ctx3 = Context::new(&model, &params).unwrap();

    assert!(ctx1.ptr_eq(&ctx2), "Context::clone should compare ptr-equal");
    assert!(!ctx1.ptr_eq(&ctx3), "independent Context::new should not");
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
