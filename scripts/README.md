# scripts

## Binding coverage

`check-binding-coverage.ts` verifies that the Elixir (Rustler) and Node
(NAPI-RS) bindings keep up with the core `rusty-llama` API. It parses source as
text, so it needs **no llama.cpp submodule, no GPU backend, and no build** —
cheap enough to run as a CI gate (`.github/workflows/binding-coverage.yml`).

It runs unmodified in both runtimes (it only uses `node:` builtins, which Deno
supports via its node-compat layer):

```sh
node scripts/check-binding-coverage.ts
# or
deno run --allow-read scripts/check-binding-coverage.ts
```

Node >= 22.18 strips TypeScript types automatically; on older Node add
`--experimental-strip-types`.

It reports, per handle type (Model / Context / Sequence), how many public
methods are bound in each language, then lists gaps.

### How gaps are classified — `scripts/binding-coverage.json`

- `ignore` — methods intentionally never bound (raw-pointer accessors, generics
  over `LlamaSampler`, raw C enums). Dropped from the core surface.
- `todo` — known-missing bindings we intend to add. Reported as warnings; they
  do **not** fail CI. Delete an entry once you bind the method.

The check **fails** only when a core method is neither bound, nor ignored, nor
listed as a todo (new, untracked drift), or when a binding refers to a method
that no longer exists in core (a stale binding).

### When you add or rename a core method

1. Add the matching binding in `packages/elixir/...` and `packages/node/...`, or
2. record it in `binding-coverage.json` under `ignore` or `todo`.

## Binding generator — IR (`binding/`)

Toward generating the bindings (and their docs) from one source, the API is
extracted into a language-neutral IR that backends consume. The extractor reads
**rustdoc JSON** — the compiler's own output — so macros are expanded (e.g. the
`token_option!` token getters become real methods) and types/docs resolved,
which a text parser can't do.

```sh
npm run gen:ir
# = cargo +nightly rustdoc -p rusty-llama -- --output-format json -Z unstable-options
#   then: node scripts/binding/extract.ts   (Deno: deno run -RW scripts/binding/extract.ts)
```

This needs the llama.cpp submodule and a nightly toolchain, but **no GPU** — a
CPU build (`GGML_NATIVE`) is enough:

```sh
git submodule update --init --depth 1 llama-sys/llama.cpp
```

Pieces:

- `binding/ir.ts` — the typed-JSON IR schema (the contract backends implement against).
- `binding/extract.ts` — rustdoc JSON → `binding/api.json`.
- `binding/api.json` — the committed IR (regenerate with `gen:ir`).
- `binding-spec.json` — hints rustdoc can't infer: handle kind/mutability,
  `cpuBound` (BEAM dirty scheduler), and per-backend `manual` escape hatches.
  Exclusions are read from `binding-coverage.json`'s `ignore` (single source).

Backends (Elixir, Node, future languages) are `api.json → files` and are not
implemented yet — this is the producer half of the pipeline.
