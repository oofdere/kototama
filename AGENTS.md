# rusty-llama — developer notes

## Build

```sh
cargo build
```

## Tests

Tests live entirely in `tests/`. There are no inline `#[cfg(test)]` blocks in source files.

### Without a model (always passes)

```sh
cargo test
```

Model-requiring tests soft-skip with a printed message when no model is found.

### With a model

```sh
RUSTY_LLAMA_TEST_MODEL=/path/to/model.gguf cargo test -- --test-threads=1
```

`--test-threads=1` is required because the llama.cpp backend uses global state.

### Hard-fail if model is missing (for CI)

```sh
RUSTY_LLAMA_TEST_MODEL=/path/to/model.gguf cargo test --features require-model -- --test-threads=1
```

### Snapshot tests (insta)

Initial snapshots are created on first run with `INSTA_UPDATE=new`:

```sh
RUSTY_LLAMA_TEST_MODEL=/path/to/model.gguf INSTA_UPDATE=new cargo test -- --test-threads=1
cargo insta review
```

After bumping llama.cpp, retake snapshots the same way and review the diff.

## Benchmarks

```sh
RUSTY_LLAMA_BENCH_MODEL=/path/to/model.gguf cargo bench
```

Falls back to `RUSTY_LLAMA_TEST_MODEL`, then `./test-models/smollm-135m.gguf`.

HTML reports are written to `target/criterion/`.

## Coverage

Install `cargo-llvm-cov` once:

```sh
cargo install cargo-llvm-cov
```

Then:

```sh
RUSTY_LLAMA_TEST_MODEL=/path/to/model.gguf cargo llvm-cov --features require-model -- --test-threads=1
# HTML report:
RUSTY_LLAMA_TEST_MODEL=/path/to/model.gguf cargo llvm-cov --html --features require-model -- --test-threads=1
# opens target/llvm-cov/html/index.html
```

## Model used for testing

The canonical test model is **SmolLM-135M Q2_K**:

- HuggingFace: `HuggingFaceTB/SmolLM-135M-GGUF` → `SmolLM-135M.Q2_K.gguf`
- Place it at `./test-models/smollm-135m.gguf` or set `RUSTY_LLAMA_TEST_MODEL`.

For CI, add a download step before running tests:

```sh
mkdir -p test-models
huggingface-cli download HuggingFaceTB/SmolLM-135M-GGUF SmolLM-135M.Q2_K.gguf --local-dir test-models
mv test-models/SmolLM-135M.Q2_K.gguf test-models/smollm-135m.gguf
```
