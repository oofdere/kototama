//! `Sequence`: one conversation or generation thread inside a `Context`.
//!
//! # Logits ownership
//!
//! llama.cpp hands out a pointer to the context's single output buffer, which
//! the next `llama_decode` anywhere in the context overwrites. The Rust
//! version copies into a per-sequence `Vec` for exactly this reason; this port
//! preallocates one `n_vocab`-sized buffer per sequence at checkout and copies
//! into it on every push. `lastLogits()` therefore stays valid while other
//! sequences decode — and because the buffer is allocated up front, `push`
//! has no allocation left to fail after the KV cache has already advanced.
//!
//! Ownership: a `Sequence` borrows its `Context` and must be deinitialized
//! first. `deinit` returns the slot to the context's pool and clears that
//! sequence's KV state.
const std = @import("std");
const c = @import("root.zig").c;
const context = @import("context.zig");
const samplers = @import("samplers.zig");

const Context = context.Context;
const DecodeError = context.DecodeError;
const Token = @import("root.zig").Token;

pub const Sequence = struct {
    ctx: *Context,
    id: c.llama_seq_id,
    tokens: std.ArrayList(Token),
    /// Owned buffer holding a copy of the logits from the most recent push.
    /// `logits` points into this buffer, or is `null` after mutations that
    /// invalidate it (pop, remove, kv_*).
    logits_buf: []f32,
    logits: ?[]const f32,
    alloc: std.mem.Allocator,

    /// Check out sequence slot `id` of `ctx`. The logits buffer is allocated
    /// here so `push` never has to allocate mid-generation; free the sequence
    /// with `deinit` before the context goes away.
    pub fn init(ctx: *Context, id: c.llama_seq_id) error{OutOfMemory}!Sequence {
        const logits_buf = try ctx.alloc.alloc(f32, ctx.n_vocab);
        return .{
            .ctx = ctx,
            .id = id,
            .tokens = .empty,
            .logits_buf = logits_buf,
            .logits = null,
            .alloc = ctx.alloc,
        };
    }

    /// Return the sequence's slot to the context and clear its KV state.
    pub fn deinit(self: *Sequence) void {
        self.tokens.deinit(self.alloc);
        self.alloc.free(self.logits_buf);
        self.ctx.releaseSeq(self.id);
        self.* = undefined;
    }

    /// Decode one token at the next position. Returns the logits vector for
    /// that token: a slice of this sequence's own buffer, valid until the next
    /// `push`/`decode` on *this* sequence (other sequences decoding in the
    /// same context cannot invalidate it).
    ///
    /// The only allocation (growing the token list) happens before the decode,
    /// so once llama.cpp has advanced the KV cache no failure can desync the
    /// token list from it.
    pub fn push(self: *Sequence, token: Token) DecodeError![]const f32 {
        try self.tokens.ensureUnusedCapacity(self.alloc, 1);

        const pos: c.llama_pos = @intCast(self.tokens.items.len);
        const borrowed = try self.ctx.pushToken(token, pos, self.id);

        @memcpy(self.logits_buf, borrowed);
        self.logits = self.logits_buf;

        self.tokens.appendAssumeCapacity(token);
        return self.logits_buf;
    }

    /// Decode several tokens in order (the Rust `extend`).
    pub fn extend(self: *Sequence, tokens: []const Token) DecodeError!void {
        for (tokens) |token| {
            _ = try self.push(token);
        }
    }

    /// Re-decode the last token to refresh logits without pushing a new one.
    /// Useful after `pop`, `remove`, or other mutations that invalidate them.
    pub fn decode(self: *Sequence) DecodeError!void {
        if (self.tokens.items.len == 0) return;
        const last = self.tokens.items[self.tokens.items.len - 1];
        const pos: c.llama_pos = @intCast(self.tokens.items.len - 1);
        const borrowed = try self.ctx.pushToken(last, pos, self.id);

        @memcpy(self.logits_buf, borrowed);
        self.logits = self.logits_buf;
    }

    /// Remove the last token and its KV state. Returns the removed token, or
    /// `null` when the sequence is empty or the KV removal was refused.
    pub fn pop(self: *Sequence) ?Token {
        const n = self.tokens.items.len;
        if (n == 0) return null;
        if (!self.kvRemove(@intCast(n - 1), @intCast(n))) return null;
        const token = self.tokens.pop().?;
        self.logits = null;
        return token;
    }

    /// Remove a half-open range of tokens and their KV state. Returns `false`
    /// when llama.cpp refused the removal (tokens then stay in place).
    pub fn remove(self: *Sequence, start: usize, end: usize) bool {
        if (!self.kvRemove(@intCast(start), @intCast(end))) return false;
        for (start..end) |_| {
            _ = self.tokens.orderedRemove(start);
        }
        self.logits = null;
        return true;
    }

    /// Copy this sequence's tokens and KV state in `[start, end)` into
    /// `other`, replacing `other`'s contents.
    pub fn copyTo(self: *Sequence, other: *Sequence, start: usize, end: usize) !void {
        self.kvCopy(other, @intCast(start), @intCast(end));
        other.tokens.clearRetainingCapacity();
        try other.tokens.appendSlice(other.alloc, self.tokens.items[start..end]);
        other.logits = null;
    }

    // ---- accessors ------------------------------------------------------

    pub fn len(self: *const Sequence) usize {
        return self.tokens.items.len;
    }

    pub fn isEmpty(self: *const Sequence) bool {
        return self.tokens.items.len == 0;
    }

    pub fn get(self: *const Sequence, index: usize) ?Token {
        return if (index < self.tokens.items.len) self.tokens.items[index] else null;
    }

    pub fn items(self: *const Sequence) []const Token {
        return self.tokens.items;
    }

    /// The logits of the last push, or `null` when nothing has been decoded
    /// (or a KV mutation invalidated them). Borrowed from this sequence's own
    /// buffer — see the module docs.
    pub fn lastLogits(self: *const Sequence) ?[]const f32 {
        return self.logits;
    }

    // ---- KV-cache manipulation -----------------------------------------
    //
    // These map onto llama.cpp's llama_memory_seq_* family. Ranges are
    // half-open [p0, p1); a negative bound means "through the end".

    /// Drop KV state for positions in [p0, p1). Returns llama.cpp's answer.
    pub fn kvRemove(self: *Sequence, p0: c.llama_pos, p1: c.llama_pos) bool {
        const ok = c.llama_memory_seq_rm(self.ctx.memory(), self.id, p0, p1);
        if (ok) self.logits = null;
        return ok;
    }

    /// Copy KV state in [p0, p1) from this sequence to `other`.
    pub fn kvCopy(self: *Sequence, other: *Sequence, p0: c.llama_pos, p1: c.llama_pos) void {
        c.llama_memory_seq_cp(self.ctx.memory(), self.id, other.id, p0, p1);
        other.logits = null;
    }

    /// Shift positions in [p0, p1) by `delta`.
    pub fn kvShift(self: *Sequence, p0: c.llama_pos, p1: c.llama_pos, delta: c.llama_pos) void {
        c.llama_memory_seq_add(self.ctx.memory(), self.id, p0, p1, delta);
        self.logits = null;
    }

    pub fn posMin(self: *const Sequence) c.llama_pos {
        return c.llama_memory_seq_pos_min(self.ctx.memory(), self.id);
    }

    pub fn posMax(self: *const Sequence) c.llama_pos {
        return c.llama_memory_seq_pos_max(self.ctx.memory(), self.id);
    }

    // ---- sampling -------------------------------------------------------

    /// Apply a sampler to the cached logits and pick a token. Returns `null`
    /// when nothing has been decoded yet.
    pub fn sample(self: *const Sequence, s: *samplers.Sampler, alloc: std.mem.Allocator) !?Token {
        const logits = self.logits orelse return null;
        return try s.sample(logits, alloc);
    }
};
