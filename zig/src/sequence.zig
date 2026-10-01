//! `Sequence`: one conversation or generation thread inside a `Context`.
//!
//! The Rust version caches the logits from the last `push()` locally, so
//! sampling needs no extra round-trip; this port keeps that design — `push`
//! returns the logits slice borrowed from llama.cpp (valid until the next
//! decode on the context).
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
    /// Logits of the most recent push, borrowed from the context's output
    /// buffer. `null` after mutations that invalidate them (pop, remove, kv_*)
    logits: ?[]const f32,

    pub fn init(ctx: *Context, id: c.llama_seq_id) error{OutOfMemory}!Sequence {
        return .{
            .ctx = ctx,
            .id = id,
            .tokens = .empty,
            .logits = null,
        };
    }

    /// Return the sequence's slot to the context and clear its KV state.
    pub fn deinit(self: *Sequence, alloc: std.mem.Allocator) void {
        self.tokens.deinit(alloc);
        self.ctx.releaseSeq(self.id);
        self.* = undefined;
    }

    /// Decode one token at the next position. Returns the logits vector for
    /// that token (borrowed; valid until the next decode on this context).
    pub fn push(self: *Sequence, alloc: std.mem.Allocator, token: Token) DecodeError![]const f32 {
        const pos: c.llama_pos = @intCast(self.tokens.items.len);
        const logits = try self.ctx.pushToken(token, pos, self.id);
        try self.tokens.append(alloc, token);
        self.logits = logits;
        return logits;
    }

    /// Decode several tokens in order (the Rust `extend`).
    pub fn extend(self: *Sequence, alloc: std.mem.Allocator, tokens: []const Token) DecodeError!void {
        for (tokens) |token| {
            _ = try self.push(alloc, token);
        }
    }

    /// Re-decode the last token to refresh logits without pushing a new one.
    /// Useful after `pop`, `remove`, or other mutations that invalidate them.
    pub fn decode(self: *Sequence) DecodeError!void {
        const last = if (self.tokens.items.len == 0) return else self.tokens.items[self.tokens.items.len - 1];
        const pos: c.llama_pos = @intCast(self.tokens.items.len - 1);
        self.logits = try self.ctx.pushToken(last, pos, self.id);
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
    pub fn copyTo(self: *Sequence, other: *Sequence, alloc: std.mem.Allocator, start: usize, end: usize) !void {
        self.kvCopy(other, @intCast(start), @intCast(end));
        other.tokens.clearRetainingCapacity();
        try other.tokens.appendSlice(alloc, self.tokens.items[start..end]);
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

    /// The logits of the last push, or `null` when there is nothing decoded.
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
