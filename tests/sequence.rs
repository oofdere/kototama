mod common;

use rusty_llama::{Context, Dist, Greedy, Sampler, Temperature};

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
    seq.push(tokens[0]).unwrap();
    assert_eq!(seq.len(), 1);
}

#[test]
fn extend_fills_tokens() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    let n = tokens.len();
    seq.extend(&tokens).unwrap();
    assert_eq!(seq.len(), n);
}

#[test]
fn tokens_accessor_matches_push_order() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("abc", false, false);
    seq.extend(&tokens).unwrap();
    assert_eq!(seq.tokens(), tokens.as_slice());
}

#[test]
fn index_operator() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens).unwrap();
    assert_eq!(seq[0], tokens[0]);
}

#[test]
fn get_returns_token() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);
    seq.extend(&tokens).unwrap();
    assert_eq!(seq.get(0), Some(tokens[0]));
    assert_eq!(seq.get(999), None);
}

#[test]
fn pop_decreases_len() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens).unwrap();
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
    seq.push(tokens[0]).unwrap();
    assert_eq!(seq.logits().unwrap().len(), model.n_tokens() as usize);
}

#[test]
fn remove_range() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    let n = tokens.len();
    seq.extend(&tokens).unwrap();
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
    seq.extend(&tokens).unwrap();
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
    src.extend(&tokens).unwrap();
    dst.extend(&tokens).unwrap();
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
    seq1.extend(&tokens1).unwrap();
    seq2.extend(&tokens2).unwrap();
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
    seq1.extend(&tokens1).unwrap();
    seq2.extend(&tokens2).unwrap();
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
    src.extend(&tokens).unwrap();
    dst.extend(&tokens).unwrap();
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
    seq.push(tokens[0]).unwrap();
    assert!(!seq.is_empty(), "sequence should not be empty after a push");
}

#[test]
fn is_empty_false_after_extend() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens).unwrap();
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
        seq.push(t).unwrap();
    }
    for _ in 0..tokens.len() {
        seq.pop();
    }
    assert!(
        seq.is_empty(),
        "sequence should be empty after all tokens are popped"
    );
}

#[test]
fn is_empty_consistent_with_len() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.is_empty(), seq.len() == 0);
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens).unwrap();
    assert_eq!(seq.is_empty(), seq.len() == 0);
}

// ---------- Sequence::sample() (moved from Context to Sequence in this PR) ----------

#[test]
fn sequence_sample_greedy_is_valid_token() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    seq.extend(&tokens).unwrap();

    let mut greedy = Greedy::new();
    let token = seq.sample(&mut greedy).unwrap();
    assert!(
        token >= 0 && token < model.n_tokens(),
        "sampled token should be within vocab range"
    );
}

#[test]
fn sequence_sample_matches_argmax() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("once upon", false, false);
    seq.extend(&tokens).unwrap();

    let argmax = seq
        .logits()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i32)
        .unwrap();

    let mut greedy = Greedy::new();
    let sampled = seq.sample(&mut greedy).unwrap();

    assert_eq!(
        sampled, argmax,
        "Sequence::sample with greedy should match manual argmax"
    );
}

#[test]
fn sequence_sample_with_temperature_in_vocab_range() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens).unwrap();

    let mut temp = Temperature::new(0.8);
    let mut dist = Dist::new(123);
    let logits = seq.logits().unwrap();
    let l = temp.apply(logits);
    let token = dist.sample(&l);
    assert!(
        token >= 0 && token < model.n_tokens(),
        "temperature-sampled token should be in vocab range"
    );
}

#[test]
fn sequence_sample_does_not_require_mut() {
    // sample() takes &self — verify it compiles and runs with a shared borrow
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("test", false, false);
    seq.extend(&tokens).unwrap();

    let mut greedy = Greedy::new();

    // sample() takes &self on Sequence — verify it coexists with another shared borrow
    let _logits = seq.logits();
    let token = seq.sample(&mut greedy).unwrap();
    assert!(token >= 0 && token < model.n_tokens());
}

// ---------- Sequence::decode() ----------

#[test]
fn decode_after_pop_restores_logits() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens[..tokens.len() - 1]).unwrap();
    let expected = seq.logits().unwrap().to_vec();

    // Push one more, pop it, and re-decode: we should be back where we started.
    seq.push(tokens[tokens.len() - 1]).unwrap();
    seq.pop();
    assert!(seq.logits().is_none(), "pop should clear logits");
    seq.decode().unwrap();
    assert_eq!(seq.logits().unwrap(), expected.as_slice());
}

#[test]
fn decode_on_empty_sequence_is_noop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.decode().unwrap();
    assert!(seq.logits().is_none());
}

// ---------- Batched extend() ----------

/// Largest absolute difference between two logit rows.
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn extend_matches_push_loop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let tokens = model.tokenize("once upon a time there was", true, false);

    let mut batched = ctx.sequence().unwrap();
    batched.extend(&tokens).unwrap();

    let mut one_by_one = ctx.sequence().unwrap();
    for &t in &tokens {
        one_by_one.push(t).unwrap();
    }

    assert_eq!(batched.tokens(), one_by_one.tokens());
    // Batched and single-token decoding can round slightly differently, so
    // compare with a tolerance rather than exactly.
    let diff = max_diff(batched.logits().unwrap(), one_by_one.logits().unwrap());
    assert!(diff < 1e-3, "logits differ by {diff}");
}

#[test]
fn extend_longer_than_n_batch_is_chunked() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.n_batch = 4;
    params.n_ubatch = 4;
    let ctx = Context::new(&model, &params).unwrap();
    let tokens = model.tokenize("one two three four five six seven eight nine ten", true, false);
    assert!(tokens.len() > 4, "test needs more tokens than n_batch");

    let mut seq = ctx.sequence().unwrap();
    seq.extend(&tokens).unwrap();
    assert_eq!(seq.tokens(), tokens.as_slice());
    assert_eq!(seq.pos_max(), tokens.len() as i32 - 1);
    assert!(seq.logits().is_some());
}

#[test]
fn extend_empty_is_noop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    seq.extend(&[]).unwrap();
    assert!(seq.is_empty());
    assert!(seq.logits().is_none());
}

// ---------- Errors instead of panics ----------

#[test]
fn push_past_context_size_returns_error() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.n_ctx = 8;
    params.n_seq_max = 1;
    let ctx = Context::new(&model, &params).unwrap();
    let n_ctx = ctx.n_ctx() as usize;
    let mut seq = ctx.sequence().unwrap();
    let token = model.tokenize("a", false, false)[0];

    for _ in 0..n_ctx {
        seq.push(token).unwrap();
    }
    assert!(seq.push(token).is_err(), "push past n_ctx should fail");
    assert_eq!(seq.len(), n_ctx, "failed push must not add the token");
    assert!(seq.logits().is_none(), "failed push should clear logits");
}

// ---------- Positions after KV edits ----------

fn unified_ctx(model: &rusty_llama::Model) -> Context {
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    Context::new(model, &params).unwrap()
}

fn story(model: &rusty_llama::Model) -> Vec<i32> {
    let tokens = model.tokenize("Once upon a time there was a little girl", false, false);
    assert!(tokens.len() >= 8, "test needs at least 8 tokens");
    tokens
}

#[test]
fn positions_follow_pushes() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(seq.next_pos(), 0);
    let tokens = story(&model);
    seq.extend(&tokens).unwrap();
    let expected: Vec<i32> = (0..tokens.len() as i32).collect();
    assert_eq!(seq.positions(), expected.as_slice());
    assert_eq!(seq.next_pos(), tokens.len() as i32);
}

#[test]
fn push_after_removing_from_middle() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = story(&model);
    let n = tokens.len() as i32;
    seq.extend(&tokens).unwrap();

    assert!(seq.remove(2..4));
    assert_eq!(seq.len(), tokens.len() - 2);
    assert_eq!(&seq.positions()[..3], &[0, 1, 4]);

    seq.push(tokens[0]).unwrap();
    assert_eq!(*seq.positions().last().unwrap(), n);
    assert!(seq.logits().is_some());
}

#[test]
fn pop_after_remove_uses_real_position() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = story(&model);
    seq.extend(&tokens).unwrap();
    assert!(seq.remove(0..2));
    let before = seq.pos_max();
    seq.pop().unwrap();
    assert_eq!(seq.pos_max(), before - 1);
    seq.decode().unwrap();
    assert!(seq.logits().is_some());
}

#[test]
fn push_after_copying_offset_range() {
    let (model, _) = setup();
    let ctx = unified_ctx(&model);
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = story(&model);
    src.extend(&tokens).unwrap();

    src.copy_to(&mut dst, 3..6);
    assert_eq!(dst.tokens(), &tokens[3..6]);
    assert_eq!(dst.positions(), &[3, 4, 5]);
    assert_eq!(dst.pos_min(), 3);
    assert_eq!(dst.pos_max(), 5);

    dst.push(tokens[0]).unwrap();
    assert_eq!(*dst.positions().last().unwrap(), 6);
}

#[test]
fn copy_to_replaces_existing_contents() {
    let (model, _) = setup();
    let ctx = unified_ctx(&model);
    let mut src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    let tokens = story(&model);
    src.extend(&tokens[..2]).unwrap();
    dst.extend(&tokens).unwrap();

    src.copy_to(&mut dst, 0..2);
    assert_eq!(dst.tokens(), &tokens[..2]);
    assert_eq!(dst.pos_max(), 1, "dst's old KV entries must be gone");
    dst.push(tokens[2]).unwrap();
}

#[test]
fn copy_then_push_matches_original() {
    let (model, _) = setup();
    let ctx = unified_ctx(&model);
    let tokens = story(&model);
    let (prefix, next) = (&tokens[..tokens.len() - 1], *tokens.last().unwrap());

    let mut src = ctx.sequence().unwrap();
    src.extend(prefix).unwrap();
    let mut dst = ctx.sequence().unwrap();
    src.copy_to(&mut dst, 0..prefix.len());

    src.push(next).unwrap();
    dst.push(next).unwrap();
    let diff = max_diff(src.logits().unwrap(), dst.logits().unwrap());
    assert!(diff < 1e-3, "logits differ by {diff}");
}

#[test]
fn out_of_range_edits_panic_before_touching_kv() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let (model, _) = setup();
    let ctx = unified_ctx(&model);
    let mut seq = ctx.sequence().unwrap();
    let mut other = ctx.sequence().unwrap();
    let tokens = story(&model);
    seq.extend(&tokens).unwrap();
    other.extend(&tokens[..2]).unwrap();

    let n = tokens.len();
    let remove = catch_unwind(AssertUnwindSafe(|| seq.remove(0..n + 5)));
    assert!(remove.is_err());
    assert_eq!(seq.len(), n);
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), n as i32 - 1);

    let copy = catch_unwind(AssertUnwindSafe(|| seq.copy_to(&mut other, 0..n + 5)));
    assert!(copy.is_err());
    assert_eq!(other.len(), 2);
    assert_eq!(other.pos_max(), 1);
}

#[test]
fn push_after_sharing_another_sequences_cache() {
    // Two streams with overlapping positions: a "document" in one sequence,
    // shared into a shorter one with kv_copy. The next token has to go after
    // the largest shared position, not after the shorter sequence's own tokens.
    let (model, _) = setup();
    let ctx = unified_ctx(&model);
    let tokens = story(&model);
    let mut doc = ctx.sequence().unwrap();
    let mut q = ctx.sequence().unwrap();
    doc.extend(&tokens).unwrap();
    q.extend(&tokens[..3]).unwrap();

    doc.kv_copy(&mut q, 0..tokens.len() as i32);
    assert_eq!(q.next_pos(), tokens.len() as i32);
    q.push(tokens[0]).unwrap();
    assert_eq!(*q.positions().last().unwrap(), tokens.len() as i32);
}

#[test]
fn kv_shift_updates_positions() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    assert!(ctx.can_shift(), "test model should support shifting");
    let mut seq = ctx.sequence().unwrap();
    let tokens = story(&model);
    let n = tokens.len() as i32;
    seq.extend(&tokens).unwrap();

    assert!(seq.remove(0..2));
    seq.kv_shift(2..n, -2);
    let expected: Vec<i32> = (0..n - 2).collect();
    assert_eq!(seq.positions(), expected.as_slice());
    assert_eq!(seq.next_pos(), n - 2);
    seq.push(tokens[0]).unwrap();
}
