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
