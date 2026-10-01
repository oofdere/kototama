//! `llama_batch`: the container llama.cpp decodes.
//!
//! The Rust crate wraps `llama_batch` in a `Batch` with a `Drop` impl and
//! internal `batch_add`/`batch_clear` helpers. Zig spells the ownership out:
//! `Batch.init` allocates through llama.cpp, `deinit` frees it.
const std = @import("std");
const c = @import("root.zig").c;

const Token = @import("root.zig").Token;
const Pos = @import("root.zig").Pos;
const SeqId = @import("root.zig").SeqId;

/// A pre-allocated batch of tokens to feed to `llama_decode`.
pub const Batch = struct {
    raw: c.llama_batch,

    /// Allocate a batch that holds up to `n_tokens` tokens across
    /// `n_seq_max` sequences. Free with `deinit`.
    pub fn init(n_tokens: i32, n_seq_max: i32) Batch {
        return .{ .raw = c.llama_batch_init(n_tokens, 0, n_seq_max) };
    }

    pub fn deinit(self: *Batch) void {
        c.llama_batch_free(self.raw);
        self.* = undefined;
    }

    /// Reset the batch to empty without freeing its storage.
    pub fn clear(self: *Batch) void {
        self.raw.n_tokens = 0;
    }

    /// Append one token at `pos`, belonging to `seq_ids`, requesting logits
    /// if `logits` is set. Returns `error.SizeExceeded` when the batch is
    /// already full — the same guard as the Rust `batch_add`.
    pub fn add(
        self: *Batch,
        token: Token,
        pos: Pos,
        seq_ids: []const SeqId,
        logits: bool,
    ) error{SizeExceeded}!void {
        const n: usize = @intCast(self.raw.n_tokens);
        // Each slot carries a pre-allocated seq_id array; a null one means
        // this slot is past the capacity llama.cpp allocated for us.
        const slot = self.raw.seq_id[n];
        if (slot == null) return error.SizeExceeded;

        self.raw.token[n] = token;
        self.raw.pos[n] = pos;
        self.raw.n_seq_id[n] = @intCast(seq_ids.len);
        for (seq_ids, 0..) |seq_id, i| {
            slot[i] = seq_id;
        }
        self.raw.logits[n] = @intFromBool(logits);
        self.raw.n_tokens = @intCast(n + 1);
    }

    /// Convenience: append a single-sequence token requesting logits.
    /// This is the shape `Sequence.push` uses for token-by-token decoding.
    pub fn addSingle(self: *Batch, token: Token, pos: Pos, seq_id: SeqId) error{SizeExceeded}!void {
        const seq_ids = [_]SeqId{seq_id};
        try self.add(token, pos, &seq_ids, true);
    }
};

test "batch add up to capacity then reject" {
    var b = Batch.init(2, 2);
    defer b.deinit();

    try b.addSingle(1, 0, 0);
    try b.addSingle(2, 1, 0);
    try std.testing.expectError(error.SizeExceeded, b.addSingle(3, 2, 0));

    b.clear();
    try b.addSingle(3, 0, 0);
}
