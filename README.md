# kototama: polylot llama.cpp bindings

upstream version: b9246

api stability: no. not yet.

this should just build without any special work if you've followed the [upstream build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for your specific backend. currently, only the vulkan and metal backends are supported.

## Goals
1. provide high quality rust bindings for llama.cpp
2. keep bindings as safe as possible without neutering functionality
3. document things properly and clearly
4. provide common compile flags as cargo features

## Versioning
semver (no effort will be made to match the upstream version numbers whatsoever for obvious reasons)

## Quick start

The crate wraps four core pieces of the llama.cpp API:

- `Model` — a loaded GGUF model. Cheap to `clone()` (Arc-backed) and safe to share across threads.
- `Context` — a KV-cache and decoder attached to a model. Actor-backed, so cloning shares the same underlying context; the actor is stopped when the last handle drops.
- `Sequence` — a checked-out slot on a context. Push tokens, read cached logits, sample the next token, and pop/copy/shift when you need to rewind or fork.
- `Sampler` — a trait implemented by `Temperature`, `MinP`, and `Dist`. Apply samplers by hand, or compose them into your own pipeline.

Minimal greedy completion:

```rust
use rusty_llama::{Context, ContextParams, Model, ModelParams};

fn main() {
    let model = Model::load_from_file(
        "./test-models/TinyStories-656K.Q2_K.gguf",
        ModelParams::new(),
    )
    .expect("failed to load model");

    let mut ctx_params = ContextParams::new();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    let ctx = Context::new(&model, &ctx_params).expect("failed to create context");

    let mut seq = ctx.sequence().expect("no free slot");
    let prompt = model.tokenize("Once upon a time", true, false);
    seq.extend(&prompt);

    for _ in 0..32 {
        let (token, _) = seq
            .logits()
            .expect("logits should be populated after a push")
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        let token = token as i32;
        if model.is_eog(token) {
            break;
        }
        print!("{}", model.token_to_piece(token).unwrap());
        seq.push(token);
    }
}
```

Sampling with the native `Sampler` trait (min-p → temperature → categorical draw):

```rust
use rusty_llama::{Dist, MinP, Sampler, Temperature};

let min_p = MinP::new(0.05, 1);
let temp = Temperature::new(0.8);
let dist = Dist::new(42);

let logits = seq.logits().expect("no logits");
let l = min_p.apply(logits);
let l = temp.apply(&l);
let token = dist.sample(&l);
seq.push(token);
```

See [`examples/simple.rs`](./examples/simple.rs) for a runnable greedy CLI and
[`examples/simple_chat.rs`](./examples/simple_chat.rs) for a chat-style loop.

## Requirements

- Rust (edition 2021)
- CMake and a C++ toolchain (needed to build the vendored llama.cpp submodule)
- A backend supported by llama.cpp — currently `vulkan` and `metal` are exercised in this crate
- The `llama-sys/llama.cpp` git submodule must be checked out:

```sh
git submodule update --init --recursive
```

## Testing, benchmarks, and coverage

Tests live under [`tests/`](./tests). There are no inline `#[cfg(test)]` blocks in the
source — every test is an integration test that talks to the public API.
A tiny [TinyStories-656K Q2_K](https://huggingface.co/mradermacher/TinyStories-656K-GGUF)
model (~540 KB) is bundled at `./test-models/TinyStories-656K.Q2_K.gguf`.

```sh
# tests — --test-threads=1 is required (llama.cpp uses global state)
cargo test -- --test-threads=1

# benchmarks (criterion)
cargo bench

# line coverage (needs cargo-llvm-cov)
cargo install cargo-llvm-cov
cargo llvm-cov --html -- --test-threads=1
```

Point `RUSTY_LLAMA_TEST_MODEL` (or `RUSTY_LLAMA_BENCH_MODEL`) at a different
GGUF file to override the bundled model. See [`AGENTS.md`](./AGENTS.md) for the
full developer workflow, including snapshot-test conventions.
