## Interactive chat example: a prompt loop with a manual sampler chain.
##
## The Nim counterpart of `examples/simple_chat.rs` in this repository's root
## (itself a port of llama.cpp's `simple_chat`, minus chat templates).
##
## Usage:
##   nim c -r --path:src examples/simple_chat.nim -m:MODEL.gguf [-c:2048] [--ngl:99]

import std/[options, os, parseopt, strutils]
import ../src/kototama

type
  Message = object
    isUser: bool
    text: string

  Args = object
    model: string
    context: uint32
    nGpuLayers: int32

proc parseArgs(): Args =
  var p = initOptParser()
  result = Args(model: "", context: 2048, nGpuLayers: 99)
  for kind, key, val in p.getopt():
    case kind
    of cmdArgument:
      discard
    of cmdLongOption, cmdShortOption:
      case key
      of "m", "model": result.model = val
      of "c", "context": result.context = val.parseInt.uint32
      of "ngl": result.nGpuLayers = val.parseInt.int32
      else:
        stderr.writeLine("unknown option: " & key)
        quit(1)
    of cmdEnd:
      discard

proc format(messages: seq[Message]): string =
  ## Render the conversation the way the Rust example does: one line per
  ## message, then an `assistant:` cue for the model to continue.
  var lines: seq[string]
  for m in messages:
    let who = if m.isUser: "user" else: "assistant"
    lines.add(who & ": " & m.text)
  result = lines.join("\n")
  result.add("\nassistant:")
  echo result

proc main() =
  let args = parseArgs()
  if args.model.len == 0:
    stderr.writeLine("usage: simple_chat -m:MODEL.gguf [-c:N] [--ngl:N]")
    quit(1)

  var modelParams = defaultModelParams()
  modelParams.nGpuLayers = args.nGpuLayers

  let model = loadFromFile(args.model, modelParams)

  var ctxParams = defaultContextParams()
  ctxParams.nCtx = args.context
  ctxParams.nBatch = args.context

  let ctx = Context.new(model, ctxParams)
  var seq = ctx.sequence().get()

  # Samplers, applied by hand below: minP -> temperature -> dist.
  let minp = MinP.new(0.05, 1)
  let temp = Temperature.new(0.8)
  let dist = Dist.new(int64 LlamaDefaultSeed)

  var messages: seq[Message] = @[]

  while true:
    # Get user input.
    var input = ""
    if stdin.readLine(input):
      discard
    else:
      break
    if input.len == 0:
      break

    messages.add(Message(isUser: true, text: input.strip()))

    # Tokenize the prompt; only the very first tokenization adds a BOS.
    let prompt = format(messages)
    let isFirst = seq.isEmpty()
    let tokens = model.tokenize(prompt, addSpecial = isFirst, parseSpecial = true)
    seq.extend(tokens)

    echo ""
    var response = ""
    while true:
      # Sample the next token: minP -> temperature -> dist.
      let logits = seq.logits().get()
      let l1 = minp.apply(logits)
      let l2 = temp.apply(l1)
      let token = dist.sample(l2)

      if model.isEog(token):
        break

      let piece = model.tokenToPiece(token)
      stdout.write piece
      response.add(piece)

      seq.push(token)

      if piece.contains('\n'):
        break

    echo ""
    messages.add(Message(isUser: false, text: response))

main()
