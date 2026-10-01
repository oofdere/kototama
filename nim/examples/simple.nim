## Minimal generation example: load a model, feed a prompt, greedily decode.
##
## The Nim counterpart of `examples/simple.rs` in this repository's root.
##
## Usage:
##   nim c -r --path:src examples/simple.nim -m:MODEL.gguf [-n:32] [--ngl:99] "your prompt"

import std/[options, os, parseopt, strutils]
import ../src/kototama

type Args = object
  model: string
  nPredict: int32
  nGpuLayers: int32
  prompt: string

proc parseArgs(): Args =
  var p = initOptParser()
  result = Args(model: "", nPredict: 32, nGpuLayers: 99, prompt: "Hello my name is")
  var positionals: seq[string]
  for kind, key, val in p.getopt():
    case kind
    of cmdArgument:
      positionals.add(key)
    of cmdLongOption, cmdShortOption:
      case key
      of "m", "model": result.model = val
      of "n", "n_predict": result.nPredict = val.parseInt.int32
      of "ngl": result.nGpuLayers = val.parseInt.int32
      else:
        stderr.writeLine("unknown option: " & key)
        quit(1)
    of cmdEnd:
      discard
  if positionals.len > 0:
    result.prompt = positionals.join(" ")

proc main() =
  let args = parseArgs()
  if args.model.len == 0:
    stderr.writeLine("usage: simple -m:MODEL.gguf [-n:N] [--ngl:N] [prompt]")
    quit(1)
  if args.nPredict <= 0:
    stderr.writeLine("n_predict must be positive, got " & $args.nPredict)
    quit(1)

  # The backend is initialized and freed automatically by Model's lifetime.

  var modelParams = defaultModelParams()
  modelParams.nGpuLayers = args.nGpuLayers

  let model = loadFromFile(args.model, modelParams)
  echo "Model: ", model.desc()

  if model.hasEncoder():
    raise newException(KototamaError, "Model has encoder, which is not supported in this example")

  # Tokenize the prompt. `parseSpecial = true` lets control strings pass through
  # as their special tokens.
  let promptTokens = model.tokenize(args.prompt, addSpecial = true, parseSpecial = true)
  let nPrompt = promptTokens.len

  var ctxParams = defaultContextParams()
  # nCtx is the context size. This example decodes one token at a time, so the
  # batch size is 1 (see below).
  ctxParams.nCtx = (nPrompt + args.nPredict - 1).uint32
  ctxParams.nBatch = 1
  ctxParams.noPerf = false

  let ctx = Context.new(model, ctxParams)
  var seq = ctx.sequence().get()
  seq.extend(promptTokens)

  # Print the prompt token by token.
  for token in promptTokens:
    stdout.write model.tokenToPiece(token)
  stdout.flushFile()

  # Main loop: argmax the cached logits, stop at end-of-generation.
  for _ in 0 ..< args.nPredict:
    var best = 0
    var bestLogit = NegInf
    let logits = seq.logits().get()
    for i, logit in logits:
      if logit > bestLogit:
        bestLogit = logit
        best = i
    let token = Token(best)
    if model.isEog(token):
      break
    stdout.write model.tokenToPiece(token)
    stdout.flushFile()
    seq.push(token)

  echo ""

  seq = nil

main()
