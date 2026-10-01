//! kototama — low-level but safe Zig bindings for llama.cpp.
//!
//! This is a port of the Rust crate of the same name. It provides thin, safe
//! wrappers around the llama.cpp C API while staying close to the original:
//! - Memory safety through explicit `deinit` methods (Zig has no destructors;
//!   each wrapper documents who frees what)
//! - Type safety where Zig's type system allows it
//! - A synchronous `Context`: callers serialize their own access (see
//!   `Context` for why this differs from the Rust actor design)
//!
//! The module layout mirrors the Rust crate:
//!
//! | Rust                    | Zig                     |
//! |-------------------------|-------------------------|
//! | `src/backend.rs`        | `backend.zig`           |
//! | `src/model.rs`, `vocab.rs` | `model.zig`         |
//! | `src/batch.rs`, `common.rs` | `batch.zig`        |
//! | `src/context.rs`        | `context.zig`           |
//! | `src/sequence.rs`       | `sequence.zig`          |
//! | `src/samplers/*`        | `samplers.zig`          |
//!
//! Typical use:
//!
//! ```zig
//! var model = try Model.load(alloc, "model.gguf", .{});
//! defer model.deinit();
//!
//! var ctx = try Context.init(alloc, &model, .{ .n_ctx = 2048 });
//! defer ctx.deinit();
//!
//! var seq = try ctx.sequence();
//! defer seq.deinit();
//! ```
const std = @import("std");

/// The raw C API. Everything here is unsafe; the wrappers below are the
/// supported surface.
pub const c = @cImport(@cInclude("llama.h"));

pub const backend = @import("backend.zig");
pub const model = @import("model.zig");
pub const batch = @import("batch.zig");
pub const context = @import("context.zig");
pub const sequence = @import("sequence.zig");
pub const samplers = @import("samplers.zig");

pub const Backend = backend.Backend;
pub const Model = model.Model;
pub const ModelOptions = model.ModelOptions;
pub const Batch = batch.Batch;
pub const Context = context.Context;
pub const ContextOptions = context.ContextOptions;
pub const Sequence = sequence.Sequence;
pub const DecodeError = context.DecodeError;

pub const Sampler = samplers.Sampler;
pub const Temperature = samplers.Temperature;
pub const TopK = samplers.TopK;
pub const TopP = samplers.TopP;
pub const MinP = samplers.MinP;
pub const Typical = samplers.Typical;
pub const TopNSigma = samplers.TopNSigma;
pub const Xtc = samplers.Xtc;
pub const Dist = samplers.Dist;
pub const Greedy = samplers.Greedy;
pub const Chain = samplers.Chain;

/// Type alias for llama.cpp token IDs.
pub const Token = c.llama_token;
/// Type alias for llama.cpp position indices.
pub const Pos = c.llama_pos;
/// Type alias for llama.cpp sequence IDs.
pub const SeqId = c.llama_seq_id;

test {
    std.testing.refAllDecls(@This());
}
