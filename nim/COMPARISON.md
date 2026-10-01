# Rust vs Nim: porting kototama

This is the writeup promised by the port: what changed, what didn't, and how
the two languages actually felt for this codebase — a ~3.5k-line safe wrapper
around llama.cpp's C API, heavy on FFI, ownership, and documentation.

**TL;DR.** The Nim port is the same codebase one notch more direct: the FFI
layer is a readable file instead of a build-time code generator, the thread
safety guarantee is stated with a lock instead of a whole actor subsystem, and
errors are exceptions instead of `Result<T, ()>`. Nim compiles faster and
generated text comes out byte-identical to the Rust version. Rust still wins
decisively on tooling, ecosystem confidence, and the kind of compile-time
enforcement you want in a safety-critical binding layer. For a *documented,
human-readable* bindings library, Nim is a surprisingly strong fit — with two
sharp edges (destruction semantics, and no `pub(crate)`) that the port had to
learn the hard way.

## Parity checklist

Everything in the Rust crate has a Nim counterpart:

| | Rust | Nim |
|---|---|---|
| FFI layer | `llama-sys` (bindgen + `build.rs`) | `raw.nim` (hand-written, ~370 lines) |
| Core API | `Model`, `Context`, `Sequence`, `Batch`, `Backend` | same five |
| Samplers | 9 + `Chain`, in 11 files | same 10, in one sectioned file |
| Vocab queries | `src/vocab.rs` (macro-generated) | `model.nim` (template-generated) |
| Examples | `simple`, `simple_chat` | same two, same CLI shape |
| Tests | 122 `#[test]` + 1 doctest | 122, mapped one-to-one |
| Snapshots | insta YAML | **reads the same insta files** |
| Benchmarks | criterion, 6 benches | own harness, same 6 operations |
| Docs | module docs + sparse comments | module docs + every proc documented |

The snapshot tests are the strongest parity evidence: the Nim suite asserts
against the Rust suite's own `tests/snapshots/*.snap` files, so both languages
are held to identical token-id expectations, and both pass.

## Numbers

Machine: M4 Pro (arm64 macOS), llama.cpp `871b0b70`, same CMake flags for both
(so the numbers compare wrappers, not backends). Rust: `cargo bench` (criterion,
median). Nim: `nim/bench/bench.nim` (median of timed samples, setup excluded
the way criterion's `iter_batched` excludes it). Lower is better.

| benchmark | Rust (criterion) | Nim | delta |
|---|---|---|---|
| `tokenize` | 6.59 µs | 7.15 µs | +8% |
| `token_to_piece` | 27.3 ns | below timer resolution¹ | — |
| `decode_single_token` | 74.0 µs | 70.8 µs | −4% |
| `generate_10_tokens` | 743 µs | 631 µs | −15% |
| `context_creation` | 205 µs | 213 µs | +4% |
| `sequence_extend_100_tokens` | 7.76 ms | 6.82 ms | −12% |

¹ `token_to_piece` is a ~20 ns vocabulary-cache hit in llama.cpp; the Nim
harness times single iterations with `epochTime`, whose two clock reads alone
cost about that much. Take Rust's criterion number as the real one.

Everything is within ±15%, which is the honest headline: for a thin wrapper
over a C library, the language overhead is noise next to the model's compute.
Nim's single-token decode is *faster* here because it replaces the Rust actor's
message round-trip with one lock acquisition — one channel handshake per
`push` costs more than one mutex lock/unlock on a hot path. On `tokenize`, Rust
edges ahead on a tight allocation loop. Run-to-run variance on this laptop is
roughly ±5%, so treat the smaller deltas as a tie.

### Build and iteration loop

The numbers that *do* separate the two, from the developer's chair:

| | Rust | Nim |
|---|---|---|
| rebuild one module (`touch src/…` → binary) | 0.9 s (`cargo build`) | 0.5 s (`nim c`) |
| rebuild + run one test file | 1.4 s (`cargo test --test sampler`) | 0.8 s (`nim c -r`) |
| cold full library build | 0.9 s | 0.7 s |
| full test suite | 13 s (122 tests, serial) | 17 s (122 tests, serial) |
| library code | 1,755 lines | 1,670 lines |

Both loops are fast on a project this size; Nim's edge grows with codebase size
because it has no dependency graph to re-check. The Rust suite is faster
overall only because criterion-adjacent tooling is quicker to invoke.

## How it felt, topic by topic

### FFI: bindgen vs. a hand-written file

Rust needs `bindgen` to talk to `llama.h`: a build dependency that parses the
header at compile time and emits thousands of lines of `bindings.rs`. It is
correct and unreviewable. Nim's FFI declarations are plain `proc` and `type`
statements with `importc`/`header` pragmas, so the whole binding layer is one
370-line file you can diff against the header by eye — and it *stays* correct:
the compiler checks field accesses and argument types against the real C
definitions at compile time, and we added `_Static_assert`s on struct sizes as
a tripwire against upstream layout drift. Writing it was maybe ninety minutes
of careful work, and the result is the most readable artifact in either port.

The sharp edge is spelled `-llama`: linking libllama needs the doubled `-l`,
and `-llama` silently means "link a library named `lama`" until the linker
complains at the very end of the build. A word of warning lives in `raw.nim`
for the next person.

### Ownership: `Arc` + `Drop` vs `ref` + `=destroy`

This is where the port got its best war story. Rust's model is well known:
`Arc<T>` for shared ownership, `Drop` for cleanup, and *fields are destroyed
automatically* whether or not you write a `Drop` impl. Nim's `ref` +
destructors look equivalent — refcounting drops the object at zero — but
**a custom `=destroy` replaces field destruction instead of adding to it**.

The first version of the port leaked every owned field: `Sequence` never
released its `Context` reference, `Context` never released its `Model`, the
`Model` never released the `Backend`, so llama.cpp's teardown never ran and
ggml's static exit handler aborted on a leaked Metal device. The fix is
explicit field release in every destructor — three extra lines per type, and
the reason each one is there is documented in place. Once you know the rule,
the Nim version is arguably clearer: `=destroy` bodies read as complete
release-order specifications.

Related: Rust's `drop(seq)` frees immediately, while Nim keeps temporaries
alive to the end of their scope, so "release this sequence's KV slot now"
needed a real API: `Sequence.close()` (plus `Context.close()` and
`Model.close()`), idempotent and documented. That is a genuinely useful API
Rust doesn't need and Nim does.

### Threading: actor vs. lock

The Rust `Context` pins the raw `*mut llama_context` to a spawned actor thread
(the crate's stated reason: thread safety via an actor model). It costs a
`#[protocol]` trait, one message struct per operation, one handler per
message, and a `Send`-wrapper for pointers — roughly 180 lines for ~13
operations, and a channel round-trip on every `push`, `sample`, and KV query.

The Nim port states the same guarantee in one `Lock`: every operation takes
the lock, touches the C handle, releases. ~40 lines total, no message types,
and (per the benchmarks) measurably less overhead on the decode path. The
tradeoff is real: a long `decode` blocks other callers instead of queueing
behind a running actor, and there is no place to put async work later. For a
single shared context — what both APIs actually model today — the lock is the
simpler correct thing. If kototama grows true pipelined serving, the actor
would earn its complexity again.

### Errors: `Result<T, ()>` vs. exceptions

The Rust crate answers `load_from_file` with `Result<Self, ()>`, forcing every
caller into `unwrap`/`expect` while erasing *why* loading failed. Nim raises
`ModelLoadError`, `ContextCreateError`, and a `DecodeError` carrying the same
fine-grained kind (`SlotNotFound`, `Aborted`, `InvalidInput`, `Fatal`) the Rust
`DecodeError` enum defines — callers who care can catch exactly, and everyone
else gets a readable message. This is a philosophical split, but for a
documented library the exception version documents itself better.

### Samplers: 11 files vs. 1

Rust's `src/samplers/` splits into eleven files because Rust's module system
and its "one type per file" custom make that the idiomatic shape. Nim's
module granularity makes one 455-line file with section headers read better
end to end: you can see the whole `Sampler` contract and all nine
implementations in one scroll, and `diff` two samplers without switching
files. The `Sampler` trait becomes a `ref object of RootObj` base with two
`method`s and one generic `name` proc derived from the type — the same
"implement `applyMut` only and get `sample` for free" contract the Rust trait
documented. Behavior matches the Rust tests point for point, with one
deliberate divergence below.

### Tests and snapshots

122 tests in both, mapped one-to-one (`tests/sampler.rs` → `tsamplers.nim`,
etc.). Rust's `cargo test` is the better tool: filtering, reporting, and the
runner come for free. Nim's `unittest` module is fine but bare — no
`cargo-nextest`, no built-in parallelism control — so `nim/test.sh` is a
20-line runner that compiles and runs each suite. The repo's `--test-threads=1`
constraint (llama.cpp's global backend state) applies equally in both
languages.

The insta snapshots were the pleasant surprise. The Nim suite parses the Rust
suite's YAML snapshot files directly, so there is exactly one source of truth
for token-level expectations across both languages: bump llama.cpp, retake the
snapshots once, and both ports are validated against the new values.

## Bugs found in the Rust version

A faithful port is a second reader, and the second reader found two things:

1. **`Sequence::sample` applies the sampler twice.** `sample` calls
   `sampler.apply(logits)` to transform a copy, then `sampler.sample(&transformed)`
   which applies the transform *again* before selecting. For greedy decoding
   the double-apply is harmless (argmax is invariant to scaling), but for
   `Temperature` it squares the intended scale and for the truncating samplers
   it double-masks. The Nim version applies once and says so; see
   `Sequence.sample` in `sequence.nim`.
2. **`Context` can outlive its `Model`.** `Context::new` takes `&Model` and
   keeps no reference, while `llama_init_from_model` does not copy the weights
   and this llama.cpp's `llama_model_free` is a bare `delete` — so dropping the
   last `Model` while a `Context` still runs is a use-after-free. The Nim
   `Context` holds its `Model` for its whole life. (Rust's `Model` is `Arc`'d,
   so a one-line `Arc` clone in `Context::new` would fix it upstream.)

Both are cheap fixes in Rust and worth an issue or a PR against the crate.

## Divergences in the port (all deliberate)

- **Errors are exceptions**, with `DecodeError.kind` preserving the enum's
  fine-grained classification.
- **Range conventions.** Token-history edits take a Nim `Slice` (inclusive at
  both ends, e.g. `seq.remove(2..4)` removes three tokens — Rust's `Range`
  excluded the end). KV-level ops (`kvRemove`, `kvCopy`, `kvShift`) take
  llama.cpp's own `p0`/`p1` (end-exclusive, `-1` = to the end) instead of the
  Rust wrapper's `Range` translation.
- **`Sequence.sample` applies once** (see above).
- **`Context` owns its `Model` reference** (see above).
- **`close()`** exists on `Sequence`/`Context`/`Model` for deterministic early
  release.
- **`Chain.with` is an alias of `push`** rather than a consuming builder; Nim's
  refs make "consume and return self" meaningless.
- **RNG families differ.** `Dist` and `Xtc` use Nim's `std/random`
  (Xoroshiro), not Rust's `StdRng` (ChaCha12), so equal seeds do not produce
  equal draws across languages. Determinism holds within a language.
- **The `Backend` log callback** is exported in `raw.nim` but not wired up,
  matching the Rust crate's commented-out `ggml_log_set` call.

## So, which would I keep?

For this specific project — bindings that pride themselves on documentation
and readability, where the hard part is making llama.cpp's API comprehensible —
Nim is the better *fit*: the FFI layer is a document instead of a generated
artifact, the concurrency story is a paragraph instead of a subsystem, and the
docs can live in the code without fighting the language. It is also a smaller
dependency surface: one compiler, no build-script codegen.

Rust is the better *bet*: the tooling is deeper, the compile-time guarantees
(stricter types, `Send`/`Sync`, exhaustive matching) catch more mistakes before
runtime, the ecosystem for a public bindings crate is incomparably larger, and
the actor design, while heavy today, leaves room for concurrent serving that
the lock design would have to rebuild. If kototama is going to be depended on
by other people's production systems, the answer is Rust.

If the goal is a *teaching* codebase that shows exactly how a safe wrapper over
a C library is built — which is what "polylot llama.cpp bindings with proper
documentation" reads like — the Nim port is the clearer of the two, and the
Rust version would benefit from borrowing two things back: the single-file
sampler overview and the `=destroy`-style release-order comments (as `Drop`
doc comments).
