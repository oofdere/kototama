//! Build script for the Zig port of kototama.
//!
//! The Rust version drives CMake from `llama-sys/build.rs`; here the same
//! llama.cpp sources are compiled directly by `zig cc`, one toolchain for the
//! whole build. The source lists, per-target defines, and include paths below
//! mirror what llama.cpp's CMake produces (verified against the actual
//! `compile_commands.json` of a reference CMake build) — keep them in sync when
//! bumping the llama.cpp submodule.
//!
//! Platform scope: macOS arm64 with the Metal backend (plus CPU and BLAS,
//! matching what `llama-sys/build.rs` enables on macOS). Other targets get a
//! CPU-only build.
//!
//! Toolchain note: zig 0.16.0 ships a bundled libc++ whose runtime fails to
//! compile against the macOS 27 SDK (`INFINITY` disappears in its own
//! `random.cpp`), which breaks every C++ link. We compile C++ against zig's
//! libc++ *headers* but link the system `libc++` instead — see README.md.
const std = @import("std");

/// The pinned llama.cpp submodule, shared with the Rust `llama-sys` crate.
const llama_cpp = "../llama-sys/llama.cpp";

/// Mirrors ggml/CMakeLists.txt's GGML_VERSION and the GGML_COMMIT CMake
/// derives from the pinned submodule (`git rev-parse --short=9 HEAD`).
const ggml_version = "0.12.0";
const ggml_commit = "871b0b70f";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const io = b.graph.io;

    const is_macos = target.result.os.tag == .macos;
    const is_arm64 = target.result.cpu.arch == .aarch64;
    const is_native = target.result.cpu.arch == b.graph.host.result.cpu.arch and
        target.result.os.tag == b.graph.host.result.os.tag;

    // ---- paths ----------------------------------------------------------
    const ll = b.path(llama_cpp);
    const ll_abs = ll.getPath(b);

    const inc_ggml = b.fmt("-I{s}/ggml/include", .{ll_abs});
    const inc_ggml_src = b.fmt("-I{s}/ggml/src", .{ll_abs});
    const inc_llama = b.fmt("-I{s}/include", .{ll_abs});
    const inc_llama_src = b.fmt("-I{s}/src", .{ll_abs});
    const inc_cpu = b.fmt("-I{s}/ggml/src/ggml-cpu", .{ll_abs});

    // zig's libc++ headers, needed to compile the C++ translation units. The
    // *library* is never linked: see the toolchain note at the top.
    const zig_lib = b.graph.zig_lib_directory.path.?;
    const inc_zig_include = b.fmt("-I{s}/include", .{zig_lib});
    const inc_libcxx = b.fmt("-I{s}/libcxx/include", .{zig_lib});
    const inc_libcxxabi = b.fmt("-I{s}/libcxxabi/include", .{zig_lib});

    const cxx_flags: []const []const u8 = &.{ inc_zig_include, inc_libcxx, inc_libcxxabi };

    const ggml_version_def = b.fmt("-DGGML_VERSION=\"{s}\"", .{ggml_version});
    const ggml_commit_def = b.fmt("-DGGML_COMMIT=\"{s}\"", .{ggml_commit});

    // ---- the llama.cpp static library ----------------------------------
    // One module holds every C/C++/Objective-C translation unit, grouped by the
    // flags each group needs (llama.cpp's CMake compiles these as separate
    // targets: ggml-base, ggml, ggml-cpu, ggml-metal, ggml-blas, llama).
    //
    // `sanitize_c = .off`: zig runs its C undefined-behavior sanitizer over C
    // code in safe builds, and llama.cpp's `incr_ptr_aligned` computes struct
    // layout offsets from a NULL pointer (the classic offsetof idiom). That
    // pattern is defined in practice but not by the C standard, so UBSan
    // aborts at startup inside `ggml_graph_overhead_custom`.
    const ll_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .sanitize_c = .off,
    });

    // -- ggml-base: core tensors, allocator, backend API ------------------
    ll_mod.addCSourceFiles(.{
        .root = ll,
        .files = &.{
            "ggml/src/ggml.c",
            "ggml/src/ggml-alloc.c",
            "ggml/src/ggml-quants.c",
        },
        .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml, ggml_version_def, ggml_commit_def }, ggml_common_defs }),
    });
    ll_mod.addCSourceFiles(.{
        .root = ll,
        .files = &.{
            "ggml/src/ggml.cpp",
            "ggml/src/ggml-backend.cpp",
            "ggml/src/ggml-backend-meta.cpp",
            "ggml/src/ggml-opt.cpp",
            "ggml/src/ggml-threading.cpp",
            "ggml/src/gguf.cpp",
        },
        .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml, ggml_version_def, ggml_commit_def }, cxx_flags, libcxx_defs, ggml_common_defs }),
    });

    // -- ggml: backend registry (which backends are compiled in) ----------
    var backend_defs: std.ArrayList([]const u8) = .empty;
    backend_defs.appendSlice(b.allocator, &.{
        "-DGGML_USE_CPU",
    }) catch @panic("OOM");
    if (is_macos) {
        backend_defs.appendSlice(b.allocator, &.{
            "-DGGML_USE_METAL",
            "-DGGML_USE_BLAS",
        }) catch @panic("OOM");
    }
    ll_mod.addCSourceFiles(.{
        .root = ll,
        .files = &.{
            "ggml/src/ggml-backend-dl.cpp",
            "ggml/src/ggml-backend-reg.cpp",
        },
        .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml }, cxx_flags, libcxx_defs, ggml_common_defs, backend_defs.items }),
    });

    // -- ggml-cpu: the CPU backend --------------------------------------
    const cpu_arch_flags: []const []const u8 = if (is_arm64) &.{
        // ggml-cpu/CMakeLists.txt probes the machine for dotprod/i8mm/sme and
        // disables SVE. On Apple silicon `-mcpu=native` already implies the
        // first two and the chips have no SVE, so one flag covers it.
        if (is_native) "-mcpu=native" else "-mcpu=generic",
    } else &.{};

    const cpu_defs: []const []const u8 = &.{
        "-DGGML_USE_CPU_REPACK",
        "-DGGML_USE_LLAMAFILE",
        "-DGGML_USE_ACCELERATE",
        "-DACCELERATE_NEW_LAPACK",
        "-DACCELERATE_LAPACK_ILP64",
    };

    ll_mod.addCSourceFiles(.{
        .root = ll,
        .files = &.{
            "ggml/src/ggml-cpu/ggml-cpu.c",
            "ggml/src/ggml-cpu/quants.c",
            "ggml/src/ggml-cpu/arch/arm/quants.c",
        },
        .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml, inc_cpu }, ggml_common_defs, cpu_defs, cpu_arch_flags }),
    });
    ll_mod.addCSourceFiles(.{
        .root = ll,
        .files = &.{
            "ggml/src/ggml-cpu/ggml-cpu.cpp",
            "ggml/src/ggml-cpu/repack.cpp",
            "ggml/src/ggml-cpu/hbm.cpp",
            "ggml/src/ggml-cpu/traits.cpp",
            "ggml/src/ggml-cpu/binary-ops.cpp",
            "ggml/src/ggml-cpu/unary-ops.cpp",
            "ggml/src/ggml-cpu/vec.cpp",
            "ggml/src/ggml-cpu/ops.cpp",
            "ggml/src/ggml-cpu/amx/amx.cpp",
            "ggml/src/ggml-cpu/amx/mmq.cpp",
            "ggml/src/ggml-cpu/llamafile/sgemm.cpp",
            "ggml/src/ggml-cpu/arch/arm/repack.cpp",
        },
        .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml, inc_cpu }, cxx_flags, libcxx_defs, ggml_common_defs, cpu_defs, cpu_arch_flags }),
    });

    // -- ggml-metal: the Metal backend (and ggml-blas via Accelerate) ----
    if (is_macos) {
        writeMetalEmbed(b, io, ll_abs);
        ll_mod.addAssemblyFile(b.path("zig-gen/ggml-metal-embed.s"));

        const metal_defs: []const []const u8 = &.{
            "-DGGML_METAL_EMBED_LIBRARY",
            "-DGGML_METAL_NDEBUG",
        };
        ll_mod.addCSourceFiles(.{
            .root = ll,
            .files = &.{
                "ggml/src/ggml-metal/ggml-metal-common.cpp",
                "ggml/src/ggml-metal/ggml-metal-device.cpp",
                "ggml/src/ggml-metal/ggml-metal-ops.cpp",
                "ggml/src/ggml-metal/ggml-metal.cpp",
            },
            .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml }, cxx_flags, libcxx_defs, ggml_common_defs, metal_defs }),
        });
        ll_mod.addCSourceFiles(.{
            .root = ll,
            .files = &.{
                "ggml/src/ggml-metal/ggml-metal-context.m",
                "ggml/src/ggml-metal/ggml-metal-device.m",
            },
            .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml }, ggml_common_defs, metal_defs }),
        });

        const blas_defs: []const []const u8 = &.{
            "-DGGML_BLAS_USE_ACCELERATE",
            "-DACCELERATE_NEW_LAPACK",
            "-DACCELERATE_LAPACK_ILP64",
        };
        ll_mod.addCSourceFiles(.{
            .root = ll,
            .files = &.{"ggml/src/ggml-blas/ggml-blas.cpp"},
            .flags = flagsOf(b, &.{ &.{ inc_ggml_src, inc_ggml }, cxx_flags, libcxx_defs, ggml_common_defs, blas_defs }),
        });
    }

    // -- llama: the model/inference core --------------------------------
    // src/models/*.cpp is one file per supported architecture; llama.cpp's
    // CMake globs the directory, so scan it at configure time the same way.
    var llama_defs: std.ArrayList([]const u8) = .empty;
    llama_defs.appendSlice(b.allocator, &.{
        "-DGGML_USE_CPU",
        "-DNDEBUG",
    }) catch @panic("OOM");
    if (is_macos) {
        llama_defs.appendSlice(b.allocator, &.{
            "-DGGML_USE_METAL",
            "-DGGML_USE_BLAS",
        }) catch @panic("OOM");
    }

    const model_sources = collectModelSources(b, io, ll_abs);
    ll_mod.addCSourceFiles(.{
        .root = ll,
        .files = &.{
            "src/llama.cpp",
            "src/llama-adapter.cpp",
            "src/llama-arch.cpp",
            "src/llama-batch.cpp",
            "src/llama-chat.cpp",
            "src/llama-context.cpp",
            "src/llama-cparams.cpp",
            "src/llama-grammar.cpp",
            "src/llama-graph.cpp",
            "src/llama-hparams.cpp",
            "src/llama-impl.cpp",
            "src/llama-io.cpp",
            "src/llama-kv-cache.cpp",
            "src/llama-kv-cache-iswa.cpp",
            "src/llama-memory.cpp",
            "src/llama-memory-hybrid.cpp",
            "src/llama-memory-hybrid-iswa.cpp",
            "src/llama-memory-recurrent.cpp",
            "src/llama-mmap.cpp",
            "src/llama-model-loader.cpp",
            "src/llama-model-saver.cpp",
            "src/llama-model.cpp",
            "src/llama-quant.cpp",
            "src/llama-sampler.cpp",
            "src/llama-vocab.cpp",
            "src/unicode.cpp",
            "src/unicode-data.cpp",
        },
        .flags = flagsOf(b, &.{ &.{ inc_ggml, inc_llama_src, inc_llama }, cxx_flags, libcxx_defs, llama_defs.items }),
    });
    ll_mod.addCSourceFiles(.{
        .root = ll,
        .files = model_sources.items,
        .flags = flagsOf(b, &.{ &.{ inc_ggml, inc_llama_src, inc_llama }, cxx_flags, libcxx_defs, llama_defs.items }),
    });

    const llama_lib = b.addLibrary(.{
        .name = "llama",
        .linkage = .static,
        .root_module = ll_mod,
    });
    b.installArtifact(llama_lib);

    // ---- the Zig wrapper module ----------------------------------------
    const kototama = b.addModule("kototama", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    kototama.linkLibrary(llama_lib);
    // See the toolchain note at the top: system libc++, never zig's bundled one.
    // `c++.1` is the versioned library name, which zig passes through to the
    // linker untouched (it only intercepts the bare name `c++`).
    kototama.linkSystemLibrary("c++.1", .{ .use_pkg_config = .no });

    if (is_macos) {
        kototama.linkFramework("Metal", .{});
        kototama.linkFramework("MetalKit", .{});
        kototama.linkFramework("Foundation", .{});
        kototama.linkFramework("Accelerate", .{});
    }
    kototama.addIncludePath(ll.path(b, "include"));
    kototama.addIncludePath(ll.path(b, "ggml/include"));

    // ---- examples ------------------------------------------------------
    inline for (.{ "simple", "simple-chat" }) |name| {
        const root = if (std.mem.eql(u8, name, "simple"))
            "examples/simple.zig"
        else
            "examples/simple_chat.zig";

        const exe = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(root),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "kototama", .module = kototama }},
            }),
        });
        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        run.step.dependOn(b.getInstallStep());
        if (b.args) |args| run.addArgs(args);
        const run_step = b.step(name, b.fmt("Run the {s} example", .{name}));
        run_step.dependOn(&run.step);
    }

    // ---- tests ---------------------------------------------------------
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "kototama", .module = kototama }},
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run the library test suite");
    test_step.dependOn(&run_tests.step);
}

/// Defines present on every ggml translation unit (ggml/CMakeLists.txt).
const ggml_common_defs: []const []const u8 = &.{
    "-DGGML_SCHED_MAX_COPIES=4",
    "-D_DARWIN_C_SOURCE",
    "-D_XOPEN_SOURCE=600",
    "-DNDEBUG",
};

/// zig's libc++ is compiled with flags that make the C99 `INFINITY` macro
/// disappear in its runtime sources, so linking it is impossible on this
/// toolchain. These flags replicate what `zig c++ -v` passes to user code so
/// C++ compiles against zig's libc++ headers; the final link uses the system
/// libc++ via `-lc++.1`.
const libcxx_defs: []const []const u8 = &.{
    "-D_LIBCPP_ABI_NAMESPACE=__1",
    "-D_LIBCPP_ABI_VERSION=1",
    "-D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS",
    "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE",
    "-D_LIBCPP_HAS_FILESYSTEM=1",
    "-D_LIBCPP_HAS_THREADS=1",
    "-D_LIBCPP_HAS_LOCALIZATION",
    "-D_LIBCPP_HAS_MONOTONIC_CLOCK",
    "-D_LIBCPP_HAS_MUSL_LIBC=0",
    "-D_LIBCPP_HAS_NO_STD_MODULES",
    "-D_LIBCPP_HAS_RANDOM_DEVICE",
    "-D_LIBCPP_HAS_TERMINAL",
    "-D_LIBCPP_HAS_UNICODE",
    "-D_LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS=0",
    "-D_LIBCPP_HAS_WIDE_CHARACTERS",
    "-D_LIBCPP_PSTL_BACKEND_SERIAL",
    "-D_LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS",
};

/// Flatten groups of compile flags into one allocated slice. The build script
/// is regular Zig, so flag lists that depend on the target (runtime values)
/// cannot use `++`; this keeps every call site readable.
fn flagsOf(b: *std.Build, groups: []const []const []const u8) []const []const u8 {
    var flat: std.ArrayList([]const u8) = .empty;
    for (groups) |group| {
        flat.appendSlice(b.allocator, group) catch @panic("OOM");
    }
    return flat.items;
}

/// Reimplements llama.cpp's `file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")`.
fn collectModelSources(
    b: *std.Build,
    io: std.Io,
    ll_abs: []const u8,
) std.ArrayList([]const u8) {
    var sources: std.ArrayList([]const u8) = .empty;

    const models_dir = b.fmt("{s}/src/models", .{ll_abs});
    var dir = std.Io.Dir.cwd().openDir(io, models_dir, .{ .iterate = true }) catch |err|
        std.debug.panic("cannot open {s}: {s}", .{ models_dir, @errorName(err) });
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch |err|
        std.debug.panic("cannot read {s}: {s}", .{ models_dir, @errorName(err) })) |entry|
    {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".cpp")) continue;
        sources.append(b.allocator, b.fmt("src/models/{s}", .{entry.name})) catch @panic("OOM");
    }
    std.mem.sort([]const u8, sources.items, {}, lessThanAscii);
    return sources;
}

fn lessThanAscii(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Reproduces the Metal shader embedding that llama.cpp's CMake performs for
/// GGML_METAL_EMBED_LIBRARY: splice `ggml-common.h` and `ggml-metal-impl.h`
/// into `ggml-metal.metal` (the runtime source compiler cannot resolve
/// #include), then bake the merged text into the binary via `.incbin`. At
/// runtime `ggml-metal-device.m` reads the bytes between the two exported
/// symbols and hands them to Metal as source.
fn writeMetalEmbed(b: *std.Build, io: std.Io, ll_abs: []const u8) void {
    const gen_dir = b.pathFromRoot("zig-gen");
    std.Io.Dir.cwd().createDirPath(io, gen_dir) catch |err|
        std.debug.panic("cannot create {s}: {s}", .{ gen_dir, @errorName(err) });

    const metal_src = readWholeFile(b, io, b.fmt("{s}/ggml/src/ggml-metal/ggml-metal.metal", .{ll_abs}));
    const common_h = readWholeFile(b, io, b.fmt("{s}/ggml/src/ggml-common.h", .{ll_abs}));
    const impl_h = readWholeFile(b, io, b.fmt("{s}/ggml/src/ggml-metal/ggml-metal-impl.h", .{ll_abs}));

    var merged = replaceLine(b, metal_src, "__embed_ggml-common.h__", common_h);
    merged = replaceLine(b, merged, "#include \"ggml-metal-impl.h\"", impl_h);

    writeFileIfChanged(b, io, b.fmt("{s}/ggml-metal-embed.metal", .{gen_dir}), merged);

    // The assembler resolves `.incbin` names through its include paths, not
    // relative to the .s file, so bake in an absolute path.
    const embed_asm = b.fmt(
        \\    .section __DATA,__ggml_metallib
        \\    .globl _ggml_metallib_start
        \\_ggml_metallib_start:
        \\    .incbin "{s}/ggml-metal-embed.metal"
        \\    .globl _ggml_metallib_end
        \\_ggml_metallib_end:
        \\
    , .{gen_dir});
    writeFileIfChanged(b, io, b.fmt("{s}/ggml-metal-embed.s", .{gen_dir}), embed_asm);
}

fn readWholeFile(b: *std.Build, io: std.Io, path: []const u8) []const u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, b.allocator, .unlimited) catch |err|
        std.debug.panic("cannot read {s}: {s}", .{ path, @errorName(err) });
}

fn replaceLine(b: *std.Build, source: []const u8, marker: []const u8, replacement: []const u8) []const u8 {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        if (std.mem.eql(u8, std.mem.trim(u8, line, " \t"), marker)) {
            out.appendSlice(b.allocator, replacement) catch @panic("OOM");
        } else {
            out.appendSlice(b.allocator, line) catch @panic("OOM");
            out.append(b.allocator, '\n') catch @panic("OOM");
        }
    }
    return out.items;
}

fn writeFileIfChanged(b: *std.Build, io: std.Io, path: []const u8, content: []const u8) void {
    if (std.Io.Dir.cwd().readFileAlloc(io, path, b.allocator, .unlimited)) |existing| {
        if (std.mem.eql(u8, existing, content)) return;
    } else |_| {}
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = content }) catch |err|
        std.debug.panic("cannot write {s}: {s}", .{ path, @errorName(err) });
}
