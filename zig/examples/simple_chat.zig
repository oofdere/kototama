//! Interactive chat loop — the Zig port of examples/simple_chat.rs.
//!
//! ```sh
//! zig build simple-chat -- -m ../test-models/TinyStories-656K.Q2_K.gguf
//! ```
//!
//! Like the Rust example (itself a port of llama.cpp's simple_chat), the chat
//! template is applied by hand as a plain `user:` / `assistant:` transcript.
//! Sampling is min_p -> temperature -> dist, applied manually per token,
//! exactly as the Rust version does it.
const std = @import("std");
const Io = std.Io;
const kototama = @import("kototama");

const Model = kototama.Model;

const Message = union(enum) {
    user: []const u8,
    assistant: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    // ---- arguments: simple-chat -m <model> [-c <context>] [--ngl <n_gpu_layers>]
    const args = try init.minimal.args.toSlice(arena);

    var model_path: ?[:0]const u8 = null;
    var n_ctx: u32 = 2048;
    var n_gpu_layers: i32 = 99;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return usage("missing value for -m");
            model_path = args[i];
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--context")) {
            i += 1;
            if (i >= args.len) return usage("missing value for -c");
            n_ctx = std.fmt.parseInt(u32, args[i], 10) catch return usage("bad -c value");
        } else if (std.mem.eql(u8, arg, "--ngl")) {
            i += 1;
            if (i >= args.len) return usage("missing value for --ngl");
            n_gpu_layers = std.fmt.parseInt(i32, args[i], 10) catch return usage("bad --ngl value");
        } else {
            return usage("unknown argument");
        }
    }
    const path = model_path orelse return usage("no model given");

    var out_buf: [4096]u8 = undefined;
    var out_file: Io.File.Writer = .init(.stdout(), io, &out_buf);
    const out = &out_file.interface;

    var in_buf: [4096]u8 = undefined;
    var in_file: Io.File.Reader = .init(.stdin(), io, &in_buf);
    const in = &in_file.interface;

    // ---- load model + context -----------------------------------------
    var model = try Model.load(arena, path, .{ .n_gpu_layers = n_gpu_layers });
    defer model.deinit();

    var ctx = try kototama.Context.init(arena, &model, .{
        .n_ctx = n_ctx,
        .n_batch = n_ctx,
    });
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit(arena);

    // ---- samplers: min_p -> temperature -> dist, applied by hand --------
    var minp = kototama.Sampler{ .min_p = .init(0.05, 1) };
    var temp = kototama.Sampler{ .temperature = .init(0.8) };
    var dist = kototama.Sampler{ .dist = .init(kototama.c.LLAMA_DEFAULT_SEED) };

    var messages: std.ArrayList(Message) = .empty;
    defer messages.deinit(arena);

    while (true) {
        // Read one line of user input; Ctrl-D ends the loop.
        const line = (try in.takeDelimiter('\n')) orelse break;
        const input = std.mem.trim(u8, line, " \t\r");
        if (input.len == 0) break;

        try messages.append(arena, .{ .user = try arena.dupe(u8, input) });

        // Format the whole transcript by hand, as the Rust example does.
        const prompt = try format(arena, messages.items);
        try out.print("{s}\n", .{prompt});

        const is_first = seq.isEmpty();
        const prompt_tokens = try model.tokenize(arena, prompt, is_first, true);
        try seq.extend(arena, prompt_tokens);

        // ---- generate until end-of-line or end-of-generation ------------
        var response: std.ArrayList(u8) = .empty;

        try out.print("\n", .{});

        while (true) {
            const logits = seq.lastLogits() orelse break;

            // min_p -> temperature -> dist, each stage on its own buffer.
            const l1 = try arena.dupe(f32, logits);
            try minp.applyMut(l1, arena);

            const l2 = try arena.dupe(f32, l1);
            try temp.applyMut(l2, arena);

            const token = try dist.sample(l2, arena);
            if (model.isEog(token)) break;

            const piece = try model.tokenToPiece(arena, token);
            try out.print("{s}", .{piece});
            try response.appendSlice(arena, piece);

            _ = try seq.push(arena, token);

            // The Rust example stops at the end of each generated line.
            if (std.mem.indexOfScalar(u8, piece, '\n') != null) break;
        }
        try out.print("\n", .{});
        try messages.append(arena, .{ .assistant = try response.toOwnedSlice(arena) });
    }

    try out.flush();
}

/// Render the transcript the way the Rust `format` does: one `role: text`
/// line per message, then the `assistant:` cue.
fn format(arena: std.mem.Allocator, messages: []const Message) ![]u8 {
    var out: Io.Writer.Allocating = .init(arena);
    const w = &out.writer;

    for (messages, 0..) |m, i| {
        const body = switch (m) {
            .user => |s| s,
            .assistant => |s| s,
        };
        const role = switch (m) {
            .user => "user",
            .assistant => "assistant",
        };
        if (i > 0) try w.writeAll("\n");
        try w.print("{s}: {s}", .{ role, body });
    }
    try w.writeAll("\n");
    try w.writeAll("assistant:");
    return try out.toOwnedSlice();
}

fn usage(msg: []const u8) error{Usage} {
    std.debug.print("error: {s}\nusage: simple-chat -m <model.gguf> [-c N] [--ngl N]\n", .{msg});
    return error.Usage;
}
