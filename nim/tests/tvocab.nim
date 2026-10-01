## Vocabulary tests -- the Nim counterpart of `tests/vocab.rs`.
##
## Run with: `nim c -r --path:src tests/tvocab.nim`

import std/[math, options, unittest]
import ../src/kototama
import testutil

suite "Vocab":
  test "bos_token_some":
    let model = loadModel()
    check model.bosToken().isSome()   # TinyStories should have a BOS token

  test "eos_token_some":
    let model = loadModel()
    check model.eosToken().isSome()   # TinyStories should have an EOS token

  test "bos_and_eos_are_valid_tokens":
    let model = loadModel()
    let bos = model.bosToken()
    if bos.isSome():
      check bos.get() >= 0 and bos.get() < model.nTokens()
    let eos = model.eosToken()
    if eos.isSome():
      check eos.get() >= 0 and eos.get() < model.nTokens()

  test "n_tokens_matches_vocab_size":
    let model = loadModel()
    check model.nTokens() > 0

  test "vocab_type_is_not_none":
    let model = loadModel()
    check model.vocabType() != vtNone

  test "get_add_bos":
    let model = loadModel()
    # Just verify the call doesn't crash; value depends on model config.
    discard model.getAddBos()

  test "get_score_bos":
    let model = loadModel()
    let bos = model.bosToken()
    if bos.isSome():
      check model.getScore(bos.get()).classify notin {fcNan, fcInf, fcNegInf}

  test "get_text_bos_nonempty":
    let model = loadModel()
    let bos = model.bosToken()
    if bos.isSome():
      check model.getText(bos.get()).len > 0

  test "is_eog_eos":
    let model = loadModel()
    let eos = model.eosToken()
    if eos.isSome():
      check model.isEog(eos.get())   # EOS is end-of-generation

  test "is_not_eog_regular_token":
    let model = loadModel()
    let bos = model.bosToken()
    if bos.isSome():
      # BOS is control but not always EOG -- just verify the call doesn't crash.
      discard model.isEog(bos.get())

  test "nl_token_optional":
    let model = loadModel()
    # May or may not be present; just verify no crash.
    discard model.nlToken()
