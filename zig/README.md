# kototama (Zig port)

A Zig port of the [Rust kototama bindings](../README.md) for llama.cpp, made
to see how the same design feels in Zig and how the two compare. Behavioural
parity was verified against the Rust version on the bundled test model; see
[PORTING.md](./PORTING.md) for the full comparison write-up.

Same goals as the Rust crate:

1. high-quality bindings for llama.cpp
2. as safe as possible without neutering functionality
3. documented properly and clearly
4. common compile flags surfaced as build options

Upstream llama.cpp is the same pinned submodule the Rust crate uses
(`../llama-sys/llama.cpp`, b9246). API stability: none, like the original.

## Layout

```
zig/
├── build.zig          # compiles llama.cpp directly (no CMake), then the Zig layer
├── src/               # the library: root, backend, model, batch, context, sequence, samplers
├── examples/          # simple.zig (greedy), simple_chat.zig (sampled chat loop)
└── tests/             # the test suite (run with `zig build test`)
```

## Building

Requires Zig 0.16.x and, on macOS, the Xcode command line tools (Metal and
Accelerate frameworks). From this directory:

```sh
zig build                              # debug
zig build -Doptimize=ReleaseFast       # what you want for inference
zig build test                         # run the test suite
zig build simple -- -m ../test-models/TinyStories-656K.Q2_K.gguf "Once upon a time"
zig build simple-chat -- -m ../test-models/TinyStories-656K.Q2_K.gguf
```

`build.zig` compiles llama.cpp's C, C++, and Objective-C sources directly with
`zig cc` — the source lists and per-target defines mirror llama.cpp's CMake
build (verified against its `compile_commands.json`). The Metal shader is
spliced and embedded the same way CMake's `GGML_METAL_EMBED_LIBRARY` does, so
no build-time `xcrun` is needed. Platform scope: macOS arm64 with the Metal
backend (CPU and BLAS too); other targets get a CPU-only build.

### Toolchain note: zig 0.16.0's bundled libc++ is broken

Any C++ *link* through zig 0.16.0 builds zig's bundled libc++ runtime, and that
runtime fails to compile against the macOS 27 SDK (`INFINITY` disappears in its
own `random.cpp`). Since llama.cpp is C++, the whole build is affected. The
workaround used here: compile C++ against zig's libc++ **headers** (with the
defines `zig c++` normally passes) and link Apple's system `libc++` via
`-lc++.1` — a versioned library name zig passes to the linker untouched. When
a future zig ships a fixed libc++, delete the `libcxx_defs` workaround block
and `linkSystemLibrary("c++.1", ...)` in `build.zig` and use plain
`link_libcpp = true`.

A second surprise: zig runs its C undefined-behavior sanitizer over C code in
safe builds, and llama.cpp's `incr_ptr_aligned` computes struct offsets from a
null pointer (the classic `offsetof` idiom), which UBSan aborts on. The llama
module sets `sanitize_c = .off` for this reason. Neither issue affects the
Rust build, which uses Apple's toolchain throughout via CMake.

## Using the library

```zig
const std = @import("std");
const kototama = @import("kototama");

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    // Load a model. `deinit` frees it and releases the global backend lease.
    var model = try kototama.Model.load(alloc, "model.gguf", .{ .n_gpu_layers = 99 });
    defer model.deinit();

    // A context owns its llama.cpp context plus a pool of sequence slots.
    var ctx = try kototama.Context.init(alloc, &model, .{ .n_ctx = 2048, .n_batch = 2048 });
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit(alloc);

    const prompt = try model.tokenize(alloc, "Once upon a time", true, true);
    defer alloc.free(prompt);
    try seq.extend(alloc, prompt);

    // Greedy decode 32 tokens.
    var sampler = kototama.Sampler{ .greedy = .{} };
    for (0..32) |_| {
        const token = (try seq.sample(&sampler, alloc)).?;
        if (model.isEog(token)) break;
        const piece = try model.tokenToPiece(alloc, token);
        defer alloc.free(piece);
        // ... print `piece`
        _ = try seq.push(alloc, token);
    }
}
```

### Ownership and lifetime rules

Zig has no destructors, so every wrapper spells its contract out:

- `Model.load` returns the sole owner; `deinit` frees the llama.cpp model and
  releases the global backend lease. `retain` gives a non-owning copy (its
  `deinit` is a no-op) for code that just needs to borrow the handle.
- `Context.init` takes a borrowed `*const Model`; `deinit` frees the context.
  A `Sequence` borrows its context and **must** be deinitialized first.
- `Sequence.deinit` returns the slot to the context's pool and clears that
  sequence's KV state.
- Every function returning allocated text or token slices (`tokenize`,
  `tokenToPiece`, `desc`) documents the allocator it used; in the examples
  that is the process arena, so nothing is freed piecemeal.

### Concurrency

`Context` is synchronous and not thread-safe: the Rust version routes all
llama.cpp calls through a dedicated actor thread, and this port deliberately
does not (see PORTING.md). Guard a single `Context` with a
`std.Thread.Mutex` if you need to share one across threads.

### Samplers

Samplers are a tagged union (`kototama.Sampler`) rather than a trait
implementing the Rust-style open extension point. Every sampler's behaviour is
visible in one dispatch table in `src/samplers.zig`; a chain composes
transforms with a trailing selector. `Greedy` and all logit transforms are
deterministic and shared with the Rust version; `Dist` and `Xtc` use zig's
Xoshiro256, so a given seed produces a different stream than Rust's `StdRng`.

## Differences from the Rust crate at a glance

| | Rust | Zig |
|---|---|---|
| llama.cpp build | CMake via `build.rs`, bindgen FFI | `zig cc` compiles sources directly, `@cImport` |
| `Context` | actor thread (Spawned) | synchronous, caller-serialized |
| samplers | `Sampler` trait | tagged union + one dispatch table |
| ownership | RAII, `Drop` | explicit `deinit`, documented contracts |
| errors | `Result` + panics in `Sequence` | error sets, no panics in the API |
| tests | `cargo test -- --test-threads=1`, insta snapshots | `zig build test`, 28 tests, one snapshot matched to the Rust one |

The full "how does it feel" comparison, including the green-threading
question, lives in [PORTING.md](./PORTING.md).
