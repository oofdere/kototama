## Sequence tests -- the Nim counterpart of `tests/sequence.rs`.
##
## Run with: `nim c -r --path:src tests/tsequence.nim`

import std/[options, unittest]
import ../src/kototama
import testutil

suite "Sequence basics":
  test "push_increases_len":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    check seq.len() == 0
    let tokens = model.tokenize("hi", false, false)
    seq.push(tokens[0])
    check seq.len() == 1

  test "extend_fills_tokens":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hello world", false, false)
    let n = tokens.len
    seq.extend(tokens)
    check seq.len() == n

  test "tokens_accessor_matches_push_order":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("abc", false, false)
    seq.extend(tokens)
    check seq.tokens() == tokens

  test "index_operator":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hi", false, false)
    seq.extend(tokens)
    check seq[0] == tokens[0]

  test "get_returns_token":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hi", false, false)
    seq.extend(tokens)
    check seq.get(0) == some(tokens[0])
    check seq.get(999) == none(Token)

suite "Sequence editing":
  test "pop_decreases_len":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hello", false, false)
    seq.extend(tokens)
    let lenBefore = seq.len()
    check seq.pop().isSome()
    check seq.len() == lenBefore - 1

  test "pop_empty_returns_none":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let seq = ctx.sequence().get()
    check seq.pop().isNone()

  test "remove_range":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hello world", false, false)
    let n = tokens.len
    seq.extend(tokens)
    check seq.len() == n
    check seq.remove(0..0)   # inclusive: removes exactly one token
    check seq.len() == n - 1

  test "is_empty_true_before_any_push":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let seq = ctx.sequence().get()
    check seq.isEmpty()

  test "is_empty_false_after_push":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hi", false, false)
    seq.push(tokens[0])
    check not seq.isEmpty()

  test "is_empty_false_after_extend":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    seq.extend(model.tokenize("hello", false, false))
    check not seq.isEmpty()

  test "is_empty_true_after_pop_clears_sequence":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hi", false, false)
    for token in tokens:
      seq.push(token)
    for _ in 0 ..< tokens.len:
      discard seq.pop()
    check seq.isEmpty()

  test "is_empty_consistent_with_len":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    check seq.isEmpty() == (seq.len() == 0)
    seq.extend(model.tokenize("hello", false, false))
    check seq.isEmpty() == (seq.len() == 0)

suite "Sequence logits":
  test "logits_len_equals_vocab_size_after_push":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    let tokens = model.tokenize("hi", false, false)
    seq.push(tokens[0])
    check seq.logits().get().len == model.nTokens().int

  test "logits_empty_before_push":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let seq = ctx.sequence().get()
    check seq.logits().isNone()

suite "Sequence KV memory":
  test "pos_min_max_after_push":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    check seq.posMin() == -1
    check seq.posMax() == -1
    seq.extend(model.tokenize("hello", false, false))
    check seq.posMin() >= 0
    check seq.posMax() >= seq.posMin()

  test "copy_to":
    let (model, _) = loadModelAndContext()
    var params = testCtxParams()
    params.kvUnified = true
    let ctx = Context.new(model, params)
    var src = ctx.sequence().get()
    var dst = ctx.sequence().get()
    let tokens = model.tokenize("hi", false, false)
    src.extend(tokens)
    dst.extend(tokens)
    src.copyTo(dst, 0 .. tokens.high)   # inclusive end
    check dst.tokens() == src.tokens()

  test "copy_from":
    let (model, _) = loadModelAndContext()
    var params = testCtxParams()
    params.kvUnified = true
    let ctx = Context.new(model, params)
    var src = ctx.sequence().get()
    var dst = ctx.sequence().get()
    let tokens = model.tokenize("hi", false, false)
    src.extend(tokens)
    dst.extend(tokens)
    dst.copyFrom(src, 0 .. tokens.high)
    check dst.tokens() == src.tokens()

suite "Sequence sampling":
  test "sequence_sample_greedy_is_valid_token":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    seq.extend(model.tokenize("hello", false, false))
    let token = seq.sample(Greedy.new()).get()
    check token >= 0 and token < model.nTokens()

  test "sequence_sample_matches_argmax":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    seq.extend(model.tokenize("once upon", false, false))
    let argmax = argmaxToken(seq.logits().get())
    let sampled = seq.sample(Greedy.new()).get()
    check sampled == argmax

  test "sequence_sample_with_temperature_in_vocab_range":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    seq.extend(model.tokenize("hello world", false, false))
    let temp = Temperature.new(0.8)
    let dist = Dist.new(123)
    let l = temp.apply(seq.logits().get())
    let token = dist.sample(l)
    check token >= 0 and token < model.nTokens()

  test "sequence_sample_without_logits_is_none":
    # Sampling before any push has no logits to draw from.
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let seq = ctx.sequence().get()
    check seq.sample(Greedy.new()).isNone()

suite "Multi-sequence":
  test "multiple_sequences_independent":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq1 = ctx.sequence().get()
    var seq2 = ctx.sequence().get()
    let tokens1 = model.tokenize("hello", false, false)
    let tokens2 = model.tokenize("world", false, false)
    seq1.extend(tokens1)
    seq2.extend(tokens2)
    check seq1.tokens() == tokens1
    check seq2.tokens() == tokens2
    check seq1.tokens() != seq2.tokens()

  test "multiple_sequences_generate_different_logits":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq1 = ctx.sequence().get()
    var seq2 = ctx.sequence().get()
    seq1.extend(model.tokenize("hello", false, false))
    seq2.extend(model.tokenize("world", false, false))
    check seq1.logits().get() != seq2.logits().get()

  test "free_slots_decreases_with_checkout":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let initial = ctx.freeSlots()
    let seq1 = ctx.sequence().get()
    check ctx.freeSlots() == initial - 1
    let seq2 = ctx.sequence().get()
    check ctx.freeSlots() == initial - 2

  test "sequence_checkout_up_to_n_seq_max":
    let (model, _) = loadModelAndContext()
    var params = testCtxParams()
    params.nSeqMax = 3
    let ctx = Context.new(model, params)
    let seq1 = ctx.sequence().get()
    let seq2 = ctx.sequence().get()
    let seq3 = ctx.sequence().get()
    check ctx.sequence().isNone()

  test "dropping_sequence_frees_slot":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let initial = ctx.freeSlots()
    block:
      let seq = ctx.sequence().get()
      check ctx.freeSlots() == initial - 1
    check ctx.freeSlots() == initial
    let newSeq = ctx.sequence().get()   # the slot is reusable
    check not newSeq.isNil
