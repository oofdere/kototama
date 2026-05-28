# rusty-llama — developer notes

## Build

```sh
cargo build
```

## Tests

Tests live entirely in `tests/`. There are no inline `#[cfg(test)]` blocks in source files.

A test model is bundled at `./test-models/TinyStories-656K.Q2_K.gguf`. To use a different model, set `RUSTY_LLAMA_TEST_MODEL`.

### Run tests

```sh
cargo test -- --test-threads=1
```

`--test-threads=1` is required because the llama.cpp backend uses global state.

### Snapshot tests (insta)

Initial snapshots are created on first run with `INSTA_UPDATE=new`:

```sh
INSTA_UPDATE=new cargo test -- --test-threads=1
cargo insta review
```

After bumping llama.cpp, retake snapshots the same way and review the diff.

## Benchmarks

```sh
cargo bench
```

To use a different model, set `RUSTY_LLAMA_BENCH_MODEL`. Falls back to `RUSTY_LLAMA_TEST_MODEL`, then `./test-models/TinyStories-656K.Q2_K.gguf`.

HTML reports are written to `target/criterion/`.

## Coverage

Install `cargo-llvm-cov` once:

```sh
cargo install cargo-llvm-cov
```

Then:

```sh
cargo llvm-cov -- --test-threads=1
# HTML report:
cargo llvm-cov --html -- --test-threads=1
cargo llvm-cov --html -- --test-threads=1
# opens target/llvm-cov/html/index.html
```

## Model used for testing

The canonical test model is **TinyStories-656K Q2_K** (~540KB):

- HuggingFace: https://huggingface.co/mradermacher/TinyStories-656K-GGUF
- Bundled at `./test-models/TinyStories-656K.Q2_K.gguf`
- Override with `RUSTY_LLAMA_TEST_MODEL` if needed

## Generated files (alef)

The following directories are **auto-generated** by [alef](https://github.com/kreuzberg-dev/alef) and should not be edited by hand. Regenerate with `alef generate && alef docs`.

- `packages/elixir/` — Rustler NIF crate and Elixir modules
- `packages/node/` — NAPI-RS crate and TypeScript declarations
- `packages/typescript/` — TypeScript re-export stub
- `docs/reference/` — API reference docs (Elixir, TypeScript, Rust)
- `.alef/` — cached IR (gitignored)

Within this section, `alef.toml` is the only hand-maintained file; everything else is regenerated from it.
