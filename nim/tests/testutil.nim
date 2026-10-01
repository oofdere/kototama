## Shared test helpers -- the Nim counterpart of `src/test_common.rs` and
## `tests/common/mod.rs`.
##
## The model is loaded once per test binary and shared by every test (the same
## shape as Rust's `OnceLock<Model>`), since `Model` is a cheap-to-copy handle.

import std/os
import ../src/kototama

proc modelPath*(): string =
  ## Path to the test model. Honors the same override the Rust suite uses, plus
  ## a kototama-namespaced one.
  result = getEnv("KOTOTAMA_TEST_MODEL",
                  getEnv("RUSTY_LLAMA_TEST_MODEL",
                         "./test-models/TinyStories-656K.Q2_K.gguf"))

var sharedModel: Model

proc loadModel*(): Model =
  ## Load the test model once and hand out the shared handle thereafter.
  if sharedModel.isNil:
    var params = defaultModelParams()
    params.nGpuLayers = 0
    sharedModel = loadFromFile(modelPath(), params)
  sharedModel

proc testCtxParams*(): LlamaContextParams =
  ## A minimal `ContextParams` for testing: small context, CPU.
  result = defaultContextParams()
  result.nCtx = 512
  result.nBatch = 512
  result.nSeqMax = 4
  result.noPerf = true

proc loadModelAndContext*(): (Model, LlamaContextParams) =
  ## Convenience: load the shared model and build test context params.
  (loadModel(), testCtxParams())

proc argmaxToken*(logits: openArray[float32]): Token =
  ## Argmax over logits, ties going to the last maximum (matching the Rust
  ## tests' `max_by` usage).
  var best = 0
  var bestLogit = NegInf
  for i, logit in logits:
    if logit >= bestLogit:
      bestLogit = logit
      best = i
  Token(best)

proc finiteCount*(logits: openArray[float32]): int =
  ## How many logits are unmasked (not -inf).
  for logit in logits:
    if logit != NegInf:
      inc result

proc isMasked*(logit: float32): bool =
  ## True when a logit has been masked to -inf.
  logit == NegInf
