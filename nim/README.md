# kototama in Nim

A port of this repository's Rust bindings (`rusty-llama`) to [Nim](https://nim-lang.org),
written to explore how the two languages feel for FFI-heavy systems glue code.

- **Same backend.** Both ports compile the same pinned `llama.cpp` submodule
  with the same CMake flags, so behaviour and benchmarks compare wrappers, not
  workloads. Generation output is byte-identical between the two.
- **Same scope.** Full parity with the Rust API: `Model`, `Context`,
  `Sequence`, `Batch`, all nine samplers, vocab queries, two examples, and the
  complete test suite — including the Rust suite's insta snapshot expectations,
  which the Nim tests read directly.
- **Documentation first.** Every module opens with prose that explains *why*
  the code is shaped the way it is, and every proc carries a doc comment. If
  you find a comment that just restates the code, that's a bug.

For the design walkthrough and the honest "which is better" discussion, see
[COMPARISON.md](COMPARISON.md).

## Layout

```
nim/
  build_llama_cpp.sh        builds the pinned llama.cpp into llama-cpp-build/
  src/kototama.nim          public entry point (import kototama)
  src/kototama/
    raw.nim                 FFI layer (llama-sys counterpart) -- start here
    backend.nim             refcounted process-wide llama.cpp init
    model.nim               weights, vocab queries, tokenization
    context.nim             one inference session; thread-safe via one lock
    sequence.nim            one generation stream: push, sample, edit
    batch.nim               batch construction for llama_decode
    samplers.nim            nine samplers + Chain, composed into one file
    errors.nim              typed exceptions
    types.nim               Token / Pos / SeqId aliases
  examples/                 simple.nim, simple_chat.nim
  tests/                    full suite (see below)
  bench/bench.nim           benchmarks matching benches/inference.rs
  COMPARISON.md             the Rust-vs-Nim writeup
```

## Building

One-time setup: a Nim toolchain (tested with 2.2.12) and the submodule.

```sh
git submodule update --init --recursive
nim/build_llama_cpp.sh          # ~25s; installs into nim/llama-cpp-build/
```

Compile anything with `--path:nim/src`:

```sh
nim c -d:release --path:nim/src nim/examples/simple.nim
./nim/examples/simple -m:test-models/TinyStories-656K.Q2_K.gguf "Once upon a time"
```

If you build llama.cpp somewhere else, point the linker at it with
`-d:llamaLibDir=/your/lib/dir`.

## Testing

```sh
nim/test.sh                 # builds and runs all suites, like `cargo test`
```

The suites mirror the Rust tests file for file: `tsamplers.nim`,
`tmodel.nim`, `tvocab.nim`, `tcontext.nim`, `tsequence.nim`,
`tintegration.nim`, `tsnapshots.nim` — 122 tests, all passing.

`tsnapshots.nim` asserts against the *Rust* suite's insta snapshots in
`tests/snapshots/`, so both languages are held to identical token-level
expectations. When you bump llama.cpp and retake those snapshots, both ports
are checked against the new values at once.

The test model is the same bundled
`test-models/TinyStories-656K.Q2_K.gguf`; override with
`KOTOTAMA_TEST_MODEL` (the Rust suite's `RUSTY_LLAMA_TEST_MODEL` is honored
too).

Note that tests run llama.cpp on CPU (`n_gpu_layers = 0`), matching the Rust
suite. The examples default to GPU offload, as the Rust examples do.

## Benchmarks

```sh
nim c -d:release --path:nim/src nim/bench/bench.nim
./nim/bench/bench
```

Same six operations as `benches/inference.rs`, measured with the same
setup-excluded shape criterion's `iter_batched` uses. Numbers from an M4 Pro
Mac (the full table is in [COMPARISON.md](COMPARISON.md)).

## A tour of the API

```nim
import std/options
import kototama

# Load a model. `Model` is a cheap-to-copy shared handle.
let model = loadFromFile("model.gguf")

# Tokenize. addSpecial prepends BOS; parseSpecial lets "<s>" be a real token.
let prompt = model.tokenize("Once upon a time", addSpecial = true, parseSpecial = true)

# Create a context and take a sequence slot (up to nSeqMax of them).
var ctxParams = defaultContextParams()
ctxParams.nCtx = 2048
let ctx = Context.new(model, ctxParams)
var seq = ctx.sequence().get()

# Decode tokens; the last push's logits are cached on the sequence.
seq.extend(prompt)

# Compose a sampler pipeline. Logit transforms first, token selector last.
let sampler = Chain.new()
  .with(TopK.new(40))
  .with(Temperature.new(0.8))
  .with(Dist.new(42))

while true:
  let token = seq.sample(sampler).get()
  if model.isEog(token): break
  stdout.write model.tokenToPiece(token)
  seq.push(token)

# Free the slot now rather than at scope exit.
seq.close()
```

## Known divergences from the Rust API

Kept deliberately small; each one is explained in [COMPARISON.md](COMPARISON.md).

- **Errors are exceptions**, not `Result<T, ()>`. `DecodeError` carries the
  same fine-grained kind the Rust `DecodeError` enum does.
- **Ranges differ where the languages do.** Token-history edits take a Nim
  `Slice` (inclusive both ends); KV-level ops take llama.cpp's own
  `p0`/`p1` (end-exclusive, `-1` = to the end).
- **`Sequence.sample` applies the sampler exactly once.** The Rust version
  applies it twice (see the COMPARISON writeup).
- **`Context` holds its `Model` alive**, so a context can never outlive its
  weights.
- **`close()`** on `Sequence`/`Context`/`Model` gives deterministic early
  release, because Nim's temporaries live to the end of their scope (Rust's
  `drop(x)` has no direct equivalent).
- The `Context` is **lock-guarded** instead of actor-threaded. Same
  serialization guarantee, stated directly.
- The `Dist`/`Xtc` samplers use Nim's `std/random`, so equal seeds do not
  mean equal draws across languages (determinism holds within each).
