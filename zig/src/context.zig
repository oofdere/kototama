//! The inference context: a llama.cpp context plus its sequence slots.
//!
//! # Design difference from the Rust version
//!
//! Rust's `Context` wraps a dedicated actor thread (via the Spawned crate):
//! every operation is a message, and the actor serializes all llama.cpp calls
//! so a `Context` can be shared across threads with only `Clone`. That design
//! exists because llama.cpp contexts are not thread-safe and Rust's ownership
//! model needs *some* way to hand a `*mut llama_context` to multiple callers
//! safely.
//!
//! The Zig port is synchronous by choice (see the porting notes in README.md).
//! `Context` is a plain value that the caller does not share across threads;
//! if you need concurrent access, guard a single `Context` with a
//! `std.Thread.Mutex` at the application level. The API surface below is
//! otherwise the same set of operations the actor exposed.
//!
//! # Ownership
//!
//! `Context.init` takes a borrowed model pointer and creates its own
//! `llama_context`; `deinit` frees it. Sequences are checked out from the
//! context (`sequence()`), hold a *pointer* to their context, and must be
//! deinitialized before the context is.
const std = @import("std");
const c = @import("root.zig").c;
const model = @import("model.zig");
const batch = @import("batch.zig");
const sequence = @import("sequence.zig");

const Model = model.Model;
const Batch = batch.Batch;

/// Options for `Context.init`. Mirrors `llama_context_params`.
pub const ContextOptions = struct {
    /// Context size (tokens of history per sequence). 0 = model default.
    n_ctx: u32 = 0,
    /// Maximum tokens per `llama_decode` call. The current API decodes one
    /// token at a time, so 1 is typical.
    n_batch: u32 = 1,
    /// Maximum number of concurrent sequences.
    n_seq_max: u32 = 1,
    /// Disable llama.cpp's performance counters.
    no_perf: bool = true,
};

/// What went wrong during decoding. Mirrors the Rust `DecodeError` enum and
/// the documented `llama_decode` return codes.
pub const DecodeError = error{
    ///  1: no KV slot free for the batch — shrink the batch or grow `n_ctx`.
    SlotNotFound,
    ///  2: an abort callback stopped decoding (partial state may remain).
    Aborted,
    /// -1: the batch itself was invalid.
    InvalidInput,
    /// < -1: fatal error inside llama.cpp.
    FatalError,
} || error{OutOfMemory};

/// A llama.cpp context with a pool of checkable-out sequences.
pub const Context = struct {
    handle: *c.llama_context,
    batch: Batch,
    n_vocab: u32,
    checked_out: []bool,
    alloc: std.mem.Allocator,

    /// Create a context from `model`. Free with `deinit`.
    ///
    /// The sequence-slot pool is sized from `n_seq_max` *after* defaults are
    /// merged, so `0` ("use the library default") yields a working pool rather
    /// than an empty one.
    pub fn init(
        alloc: std.mem.Allocator,
        m: *const Model,
        opts: ContextOptions,
    ) error{ContextInitFailed}!Context {
        var params = c.llama_context_default_params();
        if (opts.n_ctx != 0) params.n_ctx = opts.n_ctx;
        if (opts.n_batch != 0) params.n_batch = opts.n_batch;
        if (opts.n_seq_max != 0) params.n_seq_max = opts.n_seq_max;
        params.no_perf = opts.no_perf;

        const n_seq_max = params.n_seq_max;

        // Allocate the slot bitmap *first* and zero it: Zig's allocator hands
        // back undefined memory, and every slot must start free (Rust's
        // `vec![false; n]` did this for free). Allocating it before the C
        // handles exist also keeps the failure paths trivially clean — a
        // later failure cannot leak a half-initialized batch or context.
        const checked_out = alloc.alloc(bool, n_seq_max) catch return error.ContextInitFailed;
        errdefer alloc.free(checked_out);
        @memset(checked_out, false);

        const handle = c.llama_init_from_model(m.handle, params);
        if (handle == null) return error.ContextInitFailed;

        return .{
            .handle = handle.?,
            .batch = Batch.init(1, @intCast(n_seq_max)),
            .n_vocab = @intCast(m.nTokens()),
            .checked_out = checked_out,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Context) void {
        self.batch.deinit();
        c.llama_free(self.handle);
        self.alloc.free(self.checked_out);
        self.* = undefined;
    }

    /// Check out a free sequence slot. Returns `null` when every slot is in
    /// use; release slots by deinitializing their `Sequence`.
    pub fn checkoutSequence(self: *Context) error{OutOfMemory}!?sequence.Sequence {
        for (self.checked_out, 0..) |*slot, i| {
            if (!slot.*) {
                // Claim the slot first so a concurrent checkout can't take
                // it, but give it back if Sequence.init fails — otherwise a
                // failed checkout would leak the slot for the context's life.
                slot.* = true;
                const seq = sequence.Sequence.init(self, @intCast(i)) catch |err| {
                    slot.* = false;
                    return err;
                };
                return seq;
            }
        }
        return null;
    }

    /// Return a slot to the pool and clear its KV state. Called by
    /// `Sequence.deinit`; not usually called directly.
    pub fn releaseSeq(self: *Context, seq_id: c.llama_seq_id) void {
        _ = c.llama_memory_seq_rm(self.memory(), seq_id, -1, -1);
        if (seq_id >= 0 and seq_id < self.checked_out.len) {
            self.checked_out[@intCast(seq_id)] = false;
        }
    }

    /// Free sequence slots remaining.
    pub fn freeSlots(self: *const Context) usize {
        var n: usize = 0;
        for (self.checked_out) |slot| {
            if (!slot) n += 1;
        }
        return n;
    }

    pub fn nCtx(self: *const Context) u32 {
        return c.llama_n_ctx(self.handle);
    }

    /// Whether this context's memory supports shifting positions.
    pub fn canShift(self: *const Context) bool {
        return c.llama_memory_can_shift(self.memory());
    }

    /// llama.cpp performance counters for this context.
    pub fn perf(self: *const Context) c.llama_perf_context_data {
        return c.llama_perf_context(self.handle);
    }

    // ---- internal, used by Sequence ------------------------------------

    pub fn memory(self: *const Context) c.llama_memory_t {
        return c.llama_get_memory(self.handle);
    }

    /// Decode the current one-token batch and return the logits vector for
    /// the decoded token. This is the synchronous core the Rust actor's
    /// `PushToken` handler performed.
    pub fn pushToken(
        self: *Context,
        token: c.llama_token,
        pos: c.llama_pos,
        seq_id: c.llama_seq_id,
    ) DecodeError![]const f32 {
        self.batch.clear();
        self.batch.addSingle(token, pos, seq_id) catch return error.InvalidInput;

        const result = c.llama_decode(self.handle, self.batch.raw);
        switch (result) {
            0 => {},
            1 => return error.SlotNotFound,
            2 => return error.Aborted,
            -1 => return error.InvalidInput,
            else => return error.FatalError,
        }

        return self.logitsIth(0) orelse error.FatalError;
    }

    /// Borrow the logits for batch index `i` straight from llama.cpp's
    /// context-wide output buffer. The slice is invalidated by the next
    /// `llama_decode` on this context — `Sequence.push` copies out of it.
    pub fn logitsIth(self: *const Context, i: i32) ?[]const f32 {
        const ptr = c.llama_get_logits_ith(self.handle, i);
        if (ptr == null or self.n_vocab == 0) return null;
        return ptr[0..self.n_vocab];
    }

    /// Sample with a raw `llama_sampler` (used by llama.cpp's own sampler
    /// chains; kototama's samplers work on logits directly).
    pub fn sampleRaw(self: *Context, sampler: *c.llama_sampler) c.llama_token {
        return c.llama_sampler_sample(sampler, self.handle, -1);
    }
};
