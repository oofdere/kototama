## Benchmarks -- the Nim counterpart of `benches/inference.rs`.
##
## Both benchmarks run the same six operations against the same model on the
## same machine, so the numbers compare wrappers rather than workloads:
##
##   tokenize, token_to_piece, decode_single_token,
##   generate_10_tokens, context_creation, sequence_extend_100_tokens
##
## There is no criterion in Nim, so this is a small honest harness instead.
## Two details keep it comparable with the Rust numbers:
##
##   1. Each benchmark splits setup from the measured body (`benchWithSetup`),
##      matching criterion's `iter_batched`: setup cost never lands in the
##      measurement.
##   2. Every body stores its result in `benchSink`, which is printed at the
##      end, so an optimizing build cannot delete the work being timed.
##
## Warmup runs first; we report the median and mean of the timed samples.
##
## Run with: nim c -r -d:release --path:nim/src bench/bench.nim

import std/[algorithm, math, options, os, strformat, times]
import ../src/kototama

var benchSink: float   ## Opaque sink; keeps bodies from being optimized away.

type
  BenchResult = object
    name: string
    medianMs: float
    meanMs: float
    iterations: int

proc summarize(name: string; iterations: int; samples: var openArray[float]): BenchResult =
  samples.sort()
  result = BenchResult(
    name: name,
    medianMs: samples[samples.len div 2],
    meanMs: samples.sum() / samples.len.float,
    iterations: iterations)

proc bench(name: string; iterations: int; body: proc ()) : BenchResult =
  ## Time `body` `iterations` times, with no per-iteration setup.
  for _ in 0 ..< min(iterations div 10, 50):
    body()

  var samples = newSeq[float](iterations)
  for i in 0 ..< iterations:
    let t0 = epochTime()
    body()
    samples[i] = (epochTime() - t0) * 1000.0
  summarize(name, iterations, samples)

proc benchWithSetup(name: string; iterations: int;
                    setup: proc (): Sequence;
                    body: proc (s: Sequence)) : BenchResult =
  ## Like `bench`, but each iteration's `setup` runs outside the timer and its
  # result is passed to the measured `body` -- the shape of criterion's
  ## `iter_batched`, and the shape the Rust benchmarks use.
  for _ in 0 ..< min(iterations div 10, 50):
    body(setup())

  var samples = newSeq[float](iterations)
  for i in 0 ..< iterations:
    let s = setup()
    let t0 = epochTime()
    body(s)
    samples[i] = (epochTime() - t0) * 1000.0
    s.close()   # Teardown sits outside the timer, as in criterion's iter_batched.
  summarize(name, iterations, samples)

proc main() =
  let modelPath = getEnv("KOTOTAMA_BENCH_MODEL",
                         getEnv("RUSTY_LLAMA_BENCH_MODEL",
                                getEnv("KOTOTAMA_TEST_MODEL",
                                       "./test-models/TinyStories-656K.Q2_K.gguf")))

  # Match the Rust benchmark's settings exactly (see benches/inference.rs and
  # src/test_common.rs): CPU-only model, 512-token context, perf counters off.
  var modelParams = defaultModelParams()
  modelParams.nGpuLayers = 0
  let model = loadFromFile(modelPath, modelParams)

  var ctxParams = defaultContextParams()
  ctxParams.nCtx = 512
  ctxParams.nBatch = 512
  ctxParams.nSeqMax = 1
  ctxParams.noPerf = true

  let prompt = "The quick brown fox jumps over the lazy dog."
  let bos = block:
    let b = model.bosToken()
    if b.isSome(): b.get() else: 1.Token
  let promptTokens = model.tokenize("Once upon a time", true, false)
  doAssert promptTokens.len > 0
  var manyTokens = newSeq[Token](100)
  for i in 0 ..< 100: manyTokens[i] = bos

  # The shared context for the benchmarks that reuse one (as the Rust benches
  # do). `decode_single_token`, `generate_10_tokens`, and
  # `sequence_extend_100_tokens` each check out a fresh sequence per iteration.
  let ctx = Context.new(model, ctxParams)

  var results: seq[BenchResult] = @[]

  results.add bench("tokenize", 2000, proc () =
    benchSink += model.tokenize(prompt, false, false).len.float)

  results.add bench("token_to_piece", 2000, proc () =
    benchSink += model.tokenToPiece(bos).len.float)

  results.add benchWithSetup("decode_single_token", 200,
    proc (): Sequence =
      # Unmeasured setup: acquire a sequence and push the seed token, so the
      # measured part is exactly argmax + one decode (as in Rust).
      let s = ctx.sequence().get()
      s.push(bos)
      s,
    proc (s: Sequence) =
      let token = argmax(s.logits().get())
      s.push(token)
      benchSink += token.float)

  results.add benchWithSetup("generate_10_tokens", 200,
    proc (): Sequence =
      let s = ctx.sequence().get()
      s.extend(promptTokens)
      s,
    proc (s: Sequence) =
      for _ in 0 ..< 10:
        let token = argmax(s.logits().get())
        if model.isEog(token): break
        s.push(token)
        benchSink += token.float)

  results.add bench("context_creation", 200, proc () =
    let c = Context.new(model, ctxParams)
    benchSink += c.freeSlots().float
    c.close())

  results.add benchWithSetup("sequence_extend_100_tokens", 100,
    proc (): Sequence = ctx.sequence().get(),
    proc (s: Sequence) =
      s.extend(manyTokens)
      benchSink += s.len().float)

  echo ""
  echo &"""{"benchmark":<28} {"median(ms)":>12} {"mean(ms)":>12} {"iters":>8}"""
  for r in results:
    echo &"{r.name:<28} {r.medianMs:>12.4} {r.meanMs:>12.4} {r.iterations:>8}"
  echo &"\n(checksum, to foil the optimizer: {benchSink})"

main()
