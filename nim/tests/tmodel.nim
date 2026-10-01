## Model tests -- the Nim counterpart of `tests/model.rs`.
##
## Run with: `nim c -r --path:src tests/tmodel.nim`

import std/[math, options, strutils, unittest]
import ../src/kototama
import testutil

suite "Model":
  test "load_model":
    let model = loadModel()
    check model.nTokens() > 0

  test "desc_nonempty":
    let model = loadModel()
    check model.desc().len > 0

  test "desc_matches_probe_length":
    let model = loadModel()
    # The probe call asks how many bytes the description needs; desc() should
    # return exactly that many.
    let needed = llamaModelDesc(model.handle(), nil, 0)
    let desc = model.desc()
    check desc.len == needed.int
    check desc.endsWith("Medium")

  test "n_tokens_positive":
    let model = loadModel()
    check model.nTokens() > 0

  test "has_decoder":
    let model = loadModel()
    check model.hasDecoder()

  test "no_encoder":
    let model = loadModel()
    check not model.hasEncoder()

  test "not_diffusion":
    let model = loadModel()
    check not model.isDiffusion()

  test "not_recurrent":
    let model = loadModel()
    check not model.isRecurrent()

  test "tokenize_nonempty_text":
    let model = loadModel()
    check model.tokenize("hello world", addSpecial = true, parseSpecial = false).len > 0

  test "tokenize_empty_text":
    let model = loadModel()
    check model.tokenize("", addSpecial = false, parseSpecial = false).len == 0

  test "tokenize_roundtrip":
    let model = loadModel()
    let text = " Hello, world!"
    let tokens = model.tokenize(text, addSpecial = false, parseSpecial = false)
    var reconstructed = ""
    for token in tokens:
      reconstructed.add(model.tokenToPiece(token))
    check reconstructed.strip() == text.strip()

  test "token_to_piece_bos":
    let model = loadModel()
    let bos = model.bosToken()
    if bos.isSome():
      # Just verify it renders without raising.
      discard model.tokenToPiece(bos.get())

  test "decoder_start_token_none_for_decoder_only":
    let model = loadModel()
    check model.decoderStartToken().isNone()

  test "chat_template_default":
    let model = loadModel()
    # Just verify the call doesn't crash; value depends on model config.
    discard model.chatTemplate("")

suite "Model sharing":
  # These mirror the Rust `Model::clone()` tests. Nim's `Model` is a shared
  # ref, so "clone" is just an assignment.
  test "shared model has same n_tokens":
    let model = loadModel()
    let copy = model
    check model.nTokens() == copy.nTokens()

  test "shared model can tokenize":
    let model = loadModel()
    let copy = model
    check model.tokenize("hello world", false, false) ==
          copy.tokenize("hello world", false, false)

  test "shared model desc matches":
    let model = loadModel()
    let copy = model
    check model.desc() == copy.desc()

  test "shared model vocab queries match":
    let model = loadModel()
    let copy = model
    check model.bosToken() == copy.bosToken()
    check model.eosToken() == copy.eosToken()
    check model.hasDecoder() == copy.hasDecoder()
    check model.hasEncoder() == copy.hasEncoder()

  test "shared model token_to_piece matches":
    let model = loadModel()
    let copy = model
    let bos = model.bosToken()
    if bos.isSome():
      check model.tokenToPiece(bos.get()) == copy.tokenToPiece(bos.get())

  test "releasing one handle does not affect another":
    # Rust pins the same property for Arc-backed Model: dropping the "original"
    # must not invalidate the clone. In Nim the copy holds its own reference
    # count, so the model outlives any single handle.
    let copy =
      block:
        let model = loadModel()
        model
    check copy.nTokens() > 0
    check copy.tokenize("test", false, false).len > 0
