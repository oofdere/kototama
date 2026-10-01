//! Global llama.cpp backend lifecycle.
//!
//! llama.cpp needs one-time global initialization (`llama_backend_init`) and
//! backend registration (`ggml_backend_load_all`) before any model is loaded,
//! plus a matching shutdown. The Rust crate wraps this in an RAII `Backend`
//! handle that refcounts to zero; Zig has no destructors, so the same design
//! is spelled out explicitly:
//!
//! ```zig
//! const b = backend.acquire(); // first caller loads backends
//! defer backend.release(b);    // last caller shuts llama.cpp down
//! ```
//!
//! Most programs never touch this: `Model.load` acquires the backend for you
//! and `Model.deinit` releases it.
const std = @import("std");
const c = @import("root.zig").c;

/// Number of live `Backend` handles. The first acquire initializes, the last
/// release shuts down — same refcounting the Rust `BACKEND_HANDLES` static did.
var live_handles: std.atomic.Value(usize) = .init(0);

/// A lease on the global llama.cpp backend. Pass it to `release` when done.
pub const Backend = struct {
    /// Opaque token identifying this lease. Only the identity matters.
    id: usize,

    /// Initialize llama.cpp and load all available backends (CPU, Metal,
    /// BLAS, ...) if this is the first handle.
    pub fn acquire() Backend {
        const previous = live_handles.fetchAdd(1, .monotonic);
        if (previous == 0) {
            c.ggml_backend_load_all();
            c.llama_backend_init();
        }
        return .{ .id = previous + 1 };
    }

    /// Drop this lease; shuts llama.cpp down when the last handle goes away.
    /// All models and contexts must already be freed at that point.
    pub fn release(self: Backend) void {
        _ = self;
        const previous = live_handles.fetchSub(1, .monotonic);
        std.debug.assert(previous > 0);
        if (previous == 1) {
            c.llama_backend_free();
        }
    }

    /// How many handles are currently alive (diagnostics and tests).
    pub fn liveCount() usize {
        return live_handles.load(.monotonic);
    }
};

test "acquire and release refcount" {
    const b = Backend.acquire();
    try std.testing.expectEqual(@as(usize, 1), Backend.liveCount());
    b.release();
    try std.testing.expectEqual(@as(usize, 0), Backend.liveCount());
}
