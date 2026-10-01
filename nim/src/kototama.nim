## kototama: safe, readable Nim bindings for llama.cpp.
##
## This is the Nim port of the Rust crate `rusty-llama` in this repository's
## root. Both wrap the same pinned llama.cpp submodule, so the two are directly
## comparable; see `nim/COMPARISON.md` for a walkthrough of how they differ and
## why.
##
## Quick start:
##
## ```nim
## import kototama
##
## let model = Model.loadFromFile("model.gguf")
## echo model.desc()
##
## var ctxParams = defaultContextParams()
## ctxParams.nCtx = 2048
## let ctx = Context.new(model, ctxParams)
##
## var seq = ctx.sequence().get()
## seq.extend(model.tokenize("Once upon a time", addSpecial = true, parseSpecial = true))
##
## let sampler = Chain.new()
##   .with(TopK.new(40))
##   .with(Temperature.new(0.8))
##   .with(Dist.new(42))
##
## while not model.isEog(sampled):
##   let sampled = seq.sample(sampler).get()
##   stdout.write model.tokenToPiece(sampled)
##   seq.push(sampled)
## ```
##
## Module map:
##
##   * `model`     -- loaded weights, vocabulary queries, tokenization
##   * `context`   -- one inference session; hands out sequences
##   * `sequence`  -- one generation stream: push, sample, edit
##   * `samplers`  -- logit transforms and token selectors, composable
##   * `raw`       -- the bare FFI layer (the `llama-sys` counterpart)
##   * `backend`, `batch`, `errors`, `types` -- supporting pieces

import kototama/[types, errors, raw, backend, batch, model, context, sequence, samplers]

export types, errors, raw, backend, batch, model, context, sequence, samplers

proc defaultModelParams*(): LlamaModelParams =
  ## Default model-load parameters (from llama.cpp). Field names are camelCase
  ## on the Nim side: `nGpuLayers`, `useMmap`, ...
  llamaModelDefaultParams()

proc defaultContextParams*(): LlamaContextParams =
  ## Default context parameters (from llama.cpp): `nCtx`, `nSeqMax`, ...
  llamaContextDefaultParams()
