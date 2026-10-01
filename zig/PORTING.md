# Porting kototama from Rust to Zig — how it feels, how it compares

This is the write-up behind the port: what it felt like to build kototama's
Rust design in Zig 0.16, where the languages genuinely diverge, and what I'd
tell someone choosing between them for this kind of FFI-heavy library. The
short version up front:

> Zig is the better *deployment* story and the worse *library-ergonomics*
> story. The core port was pleasantly direct — explicit ownership reads like a
> well-commented C codebase that happens to have a real build system and real
> types. But two things Rust gave us for free (RAII and traits) each cost a
> deliberate design decision in Zig, and the zig 0.16.0 C++ toolchain is
> currently broken in a way that would have shipped a broken build if I hadn't
> tested on real hardware.

Scope: the safe layer (`Model`, `Vocab`, `Batch`, `Context`, `Sequence`,
`Backend`), all 10 samplers, both examples, and a 28-test suite — behaviourally
verified against the Rust version (greedy generation is byte-identical on the
test model). The actor-based concurrency layer was intentionally left out,
per the sync-first choice; see the green-threading section for where that's
going.

## What feels better in Zig

**One toolchain, no glue layer.** The Rust crate needs `llama-sys`: a build.rs
that drives CMake, bindgen to generate FFI bindings, feature flags to thread
the right link lines through, and a wrapper crate around all of it. The Zig
build compiles llama.cpp's C/C++/ObjC sources directly with `zig cc`, and
`@cImport("llama.h")` replaces bindgen entirely — the C header *is* the type
declaration, no generator, no generated code to review. The build script reads
like a manifest of what llama.cpp contains (it mirrors the CMake source lists,
verified against `compile_commands.json`) instead of like a program that runs
another build system.

**Explicit ownership is genuinely readable.** Rust hides the free calls in
`Drop`; Zig makes you write `deinit`, but every file also *shows* the
ownership graph. `Sequence.deinit` calls `ctx.releaseSeq`, `Model.deinit`
calls `backend.release`, and a reader can see the whole lifetime story in two
screens without chasing trait impls. For a bindings library whose whole job is
managing C lifetimes, having the C-shaped API do the C-shaped thing is
honest — and the `pub fn deinit` contracts fit in doc comments where Rust fits
them in `Drop` review comments.

**Errors are smaller.** The Rust `Sequence` methods panic on decode failure
(`panic!("decode failed: {e:?}")`) because the message-passing API made
error propagation awkward. The Zig port returns `DecodeError!` everywhere; the
absence of an actor meant the awkwardness wasn't there to work around. This is
an honest advantage of the sync design, not of the language.

**The debug allocator is excellent.** `std.heap.DebugAllocator` plus explicit
`deinit` calls catches leaks and double-frees at test time in a way that made
the ownership contracts verifiable as I wrote them. It found a real double-free
in my own test code within one run.

## What feels worse in Zig

**RAII is gone, and you notice it everywhere.** Every API that allocates
becomes two calls with a documented contract (`Model.load`/`deinit`,
`Batch.init`/`deinit`, `Context.init`/`deinit`, `Sequence.init`/`deinit`), and
every call site carries a `defer X.deinit(...)` you must not forget. The Rust
port expresses the same contracts once, in `Drop`, and the compiler enforces
them at *every* call site including code nobody is reviewing today. For a
library whose job is exactly "hold a `*mut llama_context` safely", that
enforcement is the product. Zig trusts you; the test suite is the only thing
that will catch the mistake.

**Traits disappear, and the replacement is a tradeoff either way.** Rust's
`samplers/` modules implement a `Sampler` trait — ten files, each self-contained,
open to extension by downstream users, composed by `Chain`. In Zig I chose a
tagged union: one file, one exhaustive dispatch table, no indirect calls, and
every sampler's behaviour visible while reading. The cost is real: a downstream
user can't add a sampler without editing `samplers.zig`. The alternative
(a vtable struct) buys openness at the cost of indirect calls and the same
"where is this actually implemented" hunt that a trait has. For a closed set
of samplers the union wins; for a plugin surface it loses. The Rust design
didn't have to make this call.

**Allocator-threading is unavoidable.** Every allocating function takes
`alloc` as a parameter, so call chains look like `model.tokenize(alloc, ...)`
and `seq.push(alloc, ...)` where Rust says `model.tokenize(...)`. An arena
absorbs most of it (the examples use `init.arena` and never free
piecemeal), but it's a visible tax on the whole API surface.

**The type system helps less with errors.** `DecodeError` in Rust is a `Debug`
enum in a `Result`, with `map_err` available at every step. Zig error sets are
fine but have no payloads, so `DecodeError` carries its meaning in the error
name and the docs, and `catch` branches compare sets rather than values.

## The build story (the biggest surprise of the port)

I expected the pure-Zig build to be the hard part; it was the *best* part.
`zig cc` compiled all 189 llama.cpp translation units (C, C++, and
Objective-C), `@cImport` consumed the headers, and `.incbin` assembly handled
the embedded Metal shader — no CMake, no bindgen, no `xcrun` at build time.

The landmines were toolchain bugs, not design problems:

- **zig 0.16.0's bundled libc++ cannot compile on this machine.** Its runtime
  sources fail against the macOS 27 SDK (`INFINITY` disappears in its own
  `random.cpp`), and zig builds that libc++ on *any* C++ link — so a
  plain `zig build` of llama.cpp (C++) dies before your code runs. Workaround
  used here: compile C++ against zig's libc++ headers, link Apple's system
  `libc++` via `-lc++.1` (a versioned name zig's `isLibCxxLibName` doesn't
  intercept). The Rust build never sees this because CMake uses Apple's
  toolchain end to end.
- **zig's C UBSan fires on llama.cpp's `offsetof` idiom.** `incr_ptr_aligned`
  computes struct layout offsets from a null pointer — defined in practice but
  not by the C standard — and zig aborts at startup in safe builds. Setting
  `sanitize_c = .off` on the llama module fixes it.

Both are documented in `build.zig` and `README.md`. Neither is an argument
against the approach; both are arguments for testing a port on real hardware
early.

## Concurrency and the green-threading question

The Rust version serializes all llama.cpp access through an actor thread
(Spawned): `Context` is `Clone`, and every operation is a message. It's a
sound design — llama.cpp contexts are not thread-safe — but it was, as noted,
an attempt to do sync and async at once, and it shows in the code: a protocol
trait with a macro generating message structs, `unwrap()` on every call site,
and a `SamplerPtr` newtype whose `unsafe impl Send` carries a paragraph of
justification.

The Zig port is synchronous by choice: `Context` is a plain value the caller
doesn't share across threads, and if you want shared access you wrap one in a
`std.Thread.Mutex`. That deleted the entire actor layer — the protocol trait,
the message enum, the join-on-drop logic — and made error propagation trivial.

The question you raised about the Zig team's green threading is the right one
to ask here, and the answer is "almost, not yet":

- Zig 0.16 shipped `std.Io`, a colorless I/O-and-concurrency interface: the
  same code runs against `std.Io.Threaded` (an OS thread pool, the stable
  implementation) or `std.Io.Evented` (io_uring on Linux, kqueue on the BSDs,
  GCD on macOS) which is built on **stackful coroutines — userspace stack
  switching, i.e. the green threads/fibers you're thinking of**.
- The Zig compiler itself runs on `std.Io.Evented` today, so the approach is
  real and the "swap the implementation without coloring your functions"
  property works.
- But the Zig team marks it experimental: error handling is incomplete, a few
  functions are unimplemented, there's a not-yet-diagnosed performance
  regression when the compiler runs on it, and it needs a builtin for maximum
  stack sizes to be practical when `overcommit` is off.

So for kototama: when `std.Io.Evented` stabilizes, the interesting move is to
keep the sync `Context` API and let a *caller* run many of them under
`std.Io` — the interface is designed so libraries don't have to pick a
concurrency style at all. That's a materially better end state than an actor
per context, and it's what the Rust actor design was reaching for. For now,
a mutex is the honest answer.

## Numbers

(On an M4 Pro MacBook Pro, macOS 26, zig 0.16.0.)

| | Rust (Release) | Zig (ReleaseFast) |
|---|---|---|
| language surface | ~3,200 lines (lib + samplers + examples + tests) | ~2,100 lines (same scope) |
| FFI | bindgen (generated), ~1,000 lines of build glue | `@cImport`, zero glue |
| test suite | 100+ tests, insta snapshots | 28 tests, 1 snapshot matched to Rust |
| llama.cpp build | CMake + cargo build.rs | `zig cc` direct, no CMake |
| clean build (llama.cpp + everything) | 2m28s | 33s wall clock (2m20s CPU) |
| generated greedy text (test model) | byte-identical | byte-identical |

Line counts are a weak metric, but the gap is mostly the actor layer and
bindgen glue, which is also where the design differences are. The build-time
gap is mostly parallelism: `zig build` schedules all translation units across
cores on its own, while the CMake build inside `build.rs` was effectively
serialized behind cargo's phases.

## How it feels, in one sentence

Porting this felt less like translating Rust and more like *deciding*, file by
file, which guarantees the compiler should have — and living with the ones it
doesn't give you. I'd use the Zig port for deployment, where one toolchain and
a static binary matter; I'd keep the Rust crate for library ergonomics, where
RAII and traits are doing real work. The two learn from each other: the Zig
build has no reason to be harder than Cargo + CMake, and the Rust samplers
have no reason to be harder to read than one dispatch table.

## Verified behaviour

- Greedy generation on `test-models/TinyStories-656K.Q2_K.gguf` is
  byte-identical between the Rust and Zig `simple` examples.
- The tokenizer snapshot test matches the Rust insta snapshot
  (`80 1443 410 555 4` for `"Hello, world!"`).
- `Dist`/`Xtc` streams differ between the two by design (Rust `StdRng` vs Zig
  Xoshiro256) — seeded randomness is not comparable across languages.
