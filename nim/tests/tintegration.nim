## Integration tests -- the Nim counterpart of `tests/integration.rs` and
## `tests/backend.rs`.
##
## Run with: `nim c -r --path:src tests/tintegration.nim`

import std/[options, unittest]
import ../src/kototama
import testutil

suite "Backend lifecycle":
  # Mirrors tests/backend.rs. The backend is global state shared by every model.
  test "init":
    let b = acquire()
    check not b.isNil

  test "acquire_twice_coexist":
    var a = acquire()
    var b = acquire()
    # Both alive -- just confirm no crash.
    a = nil
    b = nil

  test "acquire_drop_acquire":
    # Dropping every handle and re-acquiring should re-init cleanly.
    block:
      let b = acquire()
    let b2 = acquire()
    check not b2.isNil

suite "Model error handling":
  test "load_model_invalid_path_raises":
    expect ModelLoadError:
      discard loadFromFile("/nonexistent/path/to/model.gguf", defaultModelParams())

suite "Tokenization edge cases":
  test "tokenize_empty_with_bos_gives_bos":
    let model = loadModel()
    let tokens = model.tokenize("", addSpecial = true, parseSpecial = false)
    check tokens.len == 1
    check tokens[0] == model.bosToken().get()

  test "tokenize_deterministic":
    let model = loadModel()
    let a = model.tokenize("the cat sat on the mat", false, false)
    let b = model.tokenize("the cat sat on the mat", false, false)
    check a == b

suite "Sampler integration":
  test "greedy_sample_matches_argmax":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    seq.extend(model.tokenize("Once upon a time", true, false))

    let argmax = argmaxToken(seq.logits().get())
    let sampled = seq.sample(Greedy.new()).get()
    check sampled == argmax

  test "sample_with_temperature_does_not_crash":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seq = ctx.sequence().get()
    seq.extend(model.tokenize("hello", true, false))

    let temp = Temperature.new(0.8)
    let dist = Dist.new(42)
    let l = temp.apply(seq.logits().get())
    let token = dist.sample(l)
    check token >= 0 and token < model.nTokens()

suite "Multi-sequence":
  test "two_sequences_independent":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    var seqA = ctx.sequence().get()
    var seqB = ctx.sequence().get()

    let tokensA = model.tokenize("hello", true, false)
    let tokensB = model.tokenize("world", true, false)
    seqA.extend(tokensA)
    seqB.extend(tokensB)

    check seqA.tokens() == tokensA
    check seqB.tokens() == tokensB
    check seqA.logits().get() != seqB.logits().get()

suite "Generation determinism":
  test "greedy_generation_is_deterministic":
    let (model, params) = loadModelAndContext()

    proc generate(): seq[Token] =
      let ctx = Context.new(model, params)
      var seq = ctx.sequence().get()
      seq.extend(model.tokenize("the", true, false))
      for _ in 0 ..< 5:
        let token = argmaxToken(seq.logits().get())
        if model.isEog(token):
          break
        result.add(token)
        seq.push(token)

    check generate() == generate()

suite "Token-to-piece coverage":
  test "most_vocab_tokens_have_pieces":
    let model = loadModel()
    let n = model.nTokens()
    var ok = 0
    for i in 0 ..< n:
      try:
        discard model.tokenToPiece(i)
        inc ok
      except KototamaError:
        discard
    check ok.float64 / n.float64 > 0.99
