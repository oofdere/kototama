//! Generate a continuation for a prompt — the Zig port of examples/simple.rs.
//!
//! ```sh
//! zig build simple -- -m ../test-models/TinyStories-656K.Q2_K.gguf "Once upon a time"
//! ```
//!
//! Greedy (argmax) decoding, exactly like the Rust example: the port's
//! behavioural baseline.
const std = @import("std");
const Io = std.Io;
const kototama = @import("kototama");

const Model = kototama.Model;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    // ---- arguments: simple -m <model> [-n <n_predict>] [--ngl <n_gpu_layers>] [prompt]
    const args = try init.minimal.args.toSlice(arena);

    var model_path: ?[:0]const u8 = null;
    var n_predict: i32 = 32;
    var n_gpu_layers: i32 = 99;
    var prompt: []const u8 = "Hello my name is";

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return usage("missing value for -m");
            model_path = args[i];
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--n-predict")) {
            i += 1;
            if (i >= args.len) return usage("missing value for -n");
            n_predict = std.fmt.parseInt(i32, args[i], 10) catch return usage("bad -n value");
        } else if (std.mem.eql(u8, arg, "--ngl")) {
            i += 1;
            if (i >= args.len) return usage("missing value for --ngl");
            n_gpu_layers = std.fmt.parseInt(i32, args[i], 10) catch return usage("bad --ngl value");
        } else if (arg.len > 0 and arg[0] != '-') {
            prompt = arg;
        } else {
            return usage("unknown argument");
        }
    }
    const path = model_path orelse return usage("no model given");
    if (n_predict <= 0) return usage("-n must be positive");

    // ---- output buffering ---------------------------------------------
    var out_buf: [4096]u8 = undefined;
    var out_file: Io.File.Writer = .init(.stdout(), io, &out_buf);
    const out = &out_file.interface;

    // ---- load the model ------------------------------------------------
    var model = try Model.load(arena, path, .{ .n_gpu_layers = n_gpu_layers });
    defer model.deinit();

    const desc = try model.desc(arena);
    try out.print("Model: {s}\n", .{desc});

    if (model.hasEncoder()) {
        return error.EncoderModelUnsupported;
    }

    // ---- tokenize the prompt ------------------------------------------
    const prompt_tokens = try model.tokenize(arena, prompt, true, true);

    // ---- decode the prompt --------------------------------------------
    // The Rust example sizes the context to exactly the prompt plus the
    // requested output and decodes one token at a time; same here.
    const params = kototama.ContextOptions{
        .n_ctx = @intCast(prompt_tokens.len + @as(usize, @intCast(n_predict)) - 1),
        .n_batch = 1,
        .no_perf = false,
    };
    var ctx = try kototama.Context.init(arena, &model, params);
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit();

    // Decode the prompt into the sequence, then echo it back token by token.
    try seq.extend(prompt_tokens);
    for (prompt_tokens) |token| {
        const piece = try model.tokenToPiece(arena, token);
        try out.print("{s}", .{piece});
    }

    // ---- main loop: greedy decode --------------------------------------
    for (0..@intCast(n_predict)) |_| {
        const logits = seq.lastLogits() orelse break;
        const token = argmax(logits);
        if (model.isEog(token)) break;

        const piece = try model.tokenToPiece(arena, token);
        try out.print("{s}", .{piece});

        _ = try seq.push(token);
    }
    try out.print("\n", .{});
    try out.flush();
}

/// The same tie-breaking argmax the Rust example uses inline (last max wins).
fn argmax(logits: []const f32) kototama.Token {
    var best: usize = 0;
    var best_val: f32 = -std.math.inf(f32);
    for (logits, 0..) |l, i| {
        if (l >= best_val) {
            best_val = l;
            best = i;
        }
    }
    return @intCast(best);
}

fn usage(msg: []const u8) error{Usage} {
    std.debug.print("error: {s}\nusage: simple -m <model.gguf> [-n N] [--ngl N] [prompt]\n", .{msg});
    return error.Usage;
}
