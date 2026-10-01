# kototama/swift — a human-readable Swift port

A port of the `rusty-llama` Rust crate (in this repository's root) to Swift,
sharing the exact same pinned llama.cpp checkout. It is not a rewrite of
llama.cpp — it is the same thin, safe layer over the same C API, expressed in
Swift and documented for humans.

**Start with [`COMPARISON.md`](COMPARISON.md)** — that is the write-up this
port was made for: how the two languages feel to work in, what the code looks
like side by side, and how fast each one runs on the same machine.

## Why Swift

The Rust crate's most interesting design decision is that `Context` is an
actor: llama.cpp contexts are not thread-safe, so every operation becomes a
message to a dedicated thread, and a `#[protocol]` macro generates the
message types. Swift has actors as a language feature, which made the port a
good probe of a simple question — how much of the Rust code is solving the
problem, and how much is working around the lack of a native answer?

The short version is in COMPARISON.md. The one-word preview: less than you
might expect, and the actor is not where the port ends up.

## Layout

| Path | What it is |
|---|---|
| `Package.swift` | SwiftPM manifest; wires in the prebuilt llama.cpp archives |
| `scripts/build-llama.sh` | Builds llama.cpp (static, Metal) once, for `swift build` |
| `Sources/CLlama` | C shim: one header over `llama.h`, plus a module map |
| `Sources/Kototama` | The library — `Model`, `Vocabulary`, `Context`, `TokenSequence`, samplers |
| `Sources/simple` | Example: minimal text generation (ports `examples/simple.rs`) |
| `Sources/SimpleChat` | Example: interactive chat loop (ports `examples/simple_chat.rs`) |
| `Sources/kototama-bench` | Benchmark harness (mirrors `benches/inference.rs`) |
| `Tests/KototamaTests` | Test suite (mirrors `tests/*.rs`), including snapshot tests |
| `COMPARISON.md` | The Rust ↔ Swift comparison: design, code, and numbers |

## Building

One-time setup, from the repository root:

```sh
git submodule update --init --recursive   # llama.cpp, shared with llama-sys
cd swift
scripts/build-llama.sh                    # builds llama.cpp as static libs
```

Then:

```sh
swift build                               # debug
swift build -c release                    # optimized
swift test                                # 120 tests, incl. 6 snapshot tests
```

`scripts/build-llama.sh` mirrors the CMake configuration in
`llama-sys/build.rs` (same llama.cpp commit, same flags, same Metal settings),
so the Rust and Swift sides really do run the same engine and the comparison
means something.

## Running the examples

```sh
swift run simple -m ../test-models/TinyStories-656K.Q2_K.gguf "Once upon a time"
swift run simple-chat -m ../test-models/TinyStories-656K.Q2_K.gguf
```

Both accept the same flags as their Rust counterparts (`-n`, `--ngl`, `-c`).

## A tour of the API

```swift
import Kototama

// Load a model. Failures are real errors, not `Result<_, ()>`.
var modelParams = ModelParams()
modelParams.nGPULayers = 99
let model = try Model.load(from: "model.gguf", params: modelParams)

print(try model.describe())                  // "llama ?B Q2_K - Medium"
let tokens = try model.tokenize("Hello!", addSpecial: true)

// A context holds the compute state; sequences are slots checked out of it.
let context = try Context(model: model, params: ContextParams())
guard let sequence = context.checkoutSequence() else {
    fatalError("no free sequence slots")
}

// Feed the prompt, then generate.
try sequence.push(contentsOf: tokens)

while let logits = sequence.logits {
    let token = argmax(logits)               // or any Sampler
    if model.isEndOfGeneration(token) { break }
    print(model.pieceOrNil(of: token) ?? "", terminator: "")
    try sequence.push(token)
}
```

Sampling is a small composable pipeline, mirroring the Rust `Sampler` trait:

```swift
let chain = Chain()
    .pushing(TopK(k: 40))
    .pushing(Temperature(temp: 0.8))
    .pushing(Dist(seed: 42))

let token = chain.sample(logits)
```

All ten samplers are ported: `Greedy`, `Temperature`, `TopK`, `TopP`, `MinP`,
`Typical`, `TopNSigma`, `Xtc`, `Dist`, and `Chain`.

## Testing and snapshots

```sh
swift test                                  # everything
swift test --filter samplers                # one suite
KOTOTAMA_UPDATE_SNAPSHOTS=1 swift test --filter SnapshotTests   # re-record
```

The six snapshot tests carry the token streams ported **verbatim** from the
Rust crate's `insta` snapshots (see `Tests/KototamaTests/Snapshots/*.json`).
They pass unchanged, which is the strongest available evidence that both
languages tokenize and generate identically on the bundled test model.

## Deliberate differences from the Rust crate

These are the only places the Swift port does not mirror the Rust code
one-to-one. Each is discussed in COMPARISON.md.

1. **Error types instead of `Result<_, ()>`.** Rust returns `Result<_, ()>`
   from most fallible calls. Swift throws `KototamaError` with real diagnoses.
2. **`Mutex` instead of an actor thread.** `Context` uses `Synchronization`'s
   `Mutex` to serialize access to the C pointer, keeping the API synchronous.
   Rust's actor serializes the same way, over message-passing.
3. **A small seeded PRNG instead of ChaCha12.** `Dist` and `Xtc` draw from
   SplitMix64 rather than `rand`'s `StdRng`. Same determinism contract,
   different specific draws for a given seed.
4. **One throwing `piece(of:)` plus `pieceOrNil(of:)`.** Two overloads named
   `piece(of:)` recursed into each other at runtime (a silent SIGBUS); see
   COMPARISON.md's "traps" section.
5. **Sort-based `TopK`/`MinP` thresholds** instead of `select_nth_unstable_by`
   (no quickselect in Swift's standard library). Same results, slightly worse
   asymptotics on a step that is noise next to `llama_decode`.

## Requirements

- macOS 15+ on Apple Silicon (Metal backend), Swift 6.4 / Xcode 27 toolchain
- `cmake` for the one-time llama.cpp build
- The llama.cpp submodule checked out (shared with the Rust crate)
