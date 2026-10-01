//! Tests for the Zig port, mirroring the Rust `tests/` suite in intent:
//! sampling behaviour, sequence bookkeeping, and tokenizer/model queries
//! against the bundled test model.
//!
//! The Rust suite runs single-threaded because llama.cpp has global backend
//! state (`--test-threads=1`); Zig runs tests in one process per file here,
//! so the shared model is loaded once per run.
const std = @import("std");
const kototama = @import("kototama");

const Model = kototama.Model;

/// Path to the test model, overridable like the Rust suite's
/// RUSTY_LLAMA_TEST_MODEL. The default is resolved against both the
/// repository root and `zig/`, since `zig build test` runs from the latter.
fn testModelPath(alloc: std.mem.Allocator) ![:0]const u8 {
    if (std.c.getenv("RUSTY_LLAMA_TEST_MODEL")) |p| {
        return try alloc.dupeZ(u8, std.mem.span(p));
    }
    const candidates = [_][]const u8{
        "test-models/TinyStories-656K.Q2_K.gguf",
        "../test-models/TinyStories-656K.Q2_K.gguf",
    };
    for (candidates) |candidate| {
        std.Io.Dir.cwd().access(std.testing.io, candidate, .{}) catch continue;
        return try alloc.dupeZ(u8, candidate);
    }
    return try alloc.dupeZ(u8, candidates[0]);
}

var shared_model: Model = undefined;
var shared_loaded: bool = false;

fn loadModel(alloc: std.mem.Allocator) !*Model {
    if (shared_loaded) return &shared_model;
    const path = try testModelPath(alloc);
    defer alloc.free(path);
    shared_model = try Model.load(alloc, path, .{ .n_gpu_layers = 0 });
    shared_loaded = true;
    return &shared_model;
}

// ---- model + vocab ------------------------------------------------------

test "model loads and reports a description" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);
    const desc = try model.desc(alloc);
    defer alloc.free(desc);
    try std.testing.expect(desc.len > 0);
    // The Rust suite loads TinyStories-656K; its description names the llama
    // architecture and quantization.
    try std.testing.expect(std.mem.indexOf(u8, desc, "llama") != null);
}

test "load failure on missing file" {
    const alloc = std.testing.allocator;
    const result = Model.load(alloc, "nonexistent/model.gguf", .{});
    try std.testing.expectError(error.LoadFailed, result);
}

test "vocab size matches n_tokens" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);
    // TinyStories-656K has a 2048-token vocabulary.
    try std.testing.expectEqual(@as(i32, 2048), model.nTokens());
}

test "special tokens are present" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    // The model's config marks BOS = 1 and EOS = 2 ('<|start_story|>',
    // '<|end_story|>'), matching what the Rust suite observes.
    try std.testing.expectEqual(@as(?kototama.Token, 1), model.bosToken());
    try std.testing.expectEqual(@as(?kototama.Token, 2), model.eosToken());
    try std.testing.expect(model.eosToken().? != kototama.c.LLAMA_TOKEN_NULL);
}

test "is_eog distinguishes the eos token" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    try std.testing.expect(model.isEog(model.eosToken().?));
    try std.testing.expect(!model.isEog(model.bosToken().?));
}

test "add_bos flag matches the model config" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);
    // TinyStories-656K's tokenizer.ggml.add_bos_token is true.
    try std.testing.expect(model.addBos());
    try std.testing.expect(!model.addEos());
}

// ---- tokenization -------------------------------------------------------

test "tokenize empty text with bos gives exactly bos" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    const tokens = try model.tokenize(alloc, "", true, false);
    defer alloc.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(model.bosToken().?, tokens[0]);
}

test "tokenize is deterministic" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    const a = try model.tokenize(alloc, "the cat sat on the mat", false, false);
    defer alloc.free(a);
    const b = try model.tokenize(alloc, "the cat sat on the mat", false, false);
    defer alloc.free(b);

    try std.testing.expectEqualSlices(kototama.Token, a, b);
}

test "tokenize and detokenize round-trip basic text" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    const text = "Hello, world!";
    const tokens = try model.tokenize(alloc, text, false, false);
    defer alloc.free(tokens);
    try std.testing.expect(tokens.len > 0);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    for (tokens) |token| {
        const piece = try model.tokenToPiece(alloc, token);
        defer alloc.free(piece);
        try out.appendSlice(alloc, piece);
    }
    // The SPM tokenizer reconstructs text with a leading space (the
    // "add_space_prefix" behaviour llama.cpp documents for token_to_piece),
    // so detokenize(tokenize(x)) == " " + x for text that does not start
    // with whitespace. This is a property of llama.cpp, not of the port.
    try std.testing.expectEqualStrings(" Hello, world!", out.items);
}

test "tokenize matches the Rust snapshot for Hello, world!" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    // Kept in lockstep with tests/snapshots/snapshots__snapshot_tokenize_hello_world.snap.
    // If either side changes after a llama.cpp bump, both must be updated.
    const tokens = try model.tokenize(alloc, "Hello, world!", false, false);
    defer alloc.free(tokens);

    const expected = [_]kototama.Token{ 80, 1443, 410, 555, 4 };
    try std.testing.expectEqualSlices(kototama.Token, &expected, tokens);
}

// ---- samplers -----------------------------------------------------------

test "argmax picks the largest logit with last-tie-break" {
    // Ties go to the LAST maximum, matching the Rust `max_by` behaviour
    // that both examples rely on.
    try std.testing.expectEqual(@as(kototama.Token, 3), kototama.samplers.argmax(&.{ 1.0, 0.5, 3.0, 3.0 }));
    try std.testing.expectEqual(@as(kototama.Token, 0), kototama.samplers.argmax(&.{ -1.0, -2.0 }));
}

test "temperature scales logits by 1/temp" {
    const alloc = std.testing.allocator;

    var logits = [_]f32{ 1.0, 2.0 };
    var s = kototama.Sampler{ .temperature = .init(0.5) };
    try s.applyMut(&logits, alloc);

    try std.testing.expectEqual(@as(f32, 2.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 4.0), logits[1]);
}

test "temperature zero is a no-op" {
    const alloc = std.testing.allocator;

    var logits = [_]f32{ 1.0, 2.0 };
    var s = kototama.Sampler{ .temperature = .init(0.0) };
    try s.applyMut(&logits, alloc);

    try std.testing.expectEqual(@as(f32, 1.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]);
}

test "top_k keeps exactly k highest logits" {
    const alloc = std.testing.allocator;

    var logits = [_]f32{ 1.0, 5.0, 3.0, 2.0 };
    var s = kototama.Sampler{ .top_k = .init(2) };
    try s.applyMut(&logits, alloc);

    // 5.0 and 3.0 survive; 1.0 and 2.0 are masked.
    try std.testing.expectEqual(-std.math.inf(f32), logits[0]);
    try std.testing.expectEqual(@as(f32, 5.0), logits[1]);
    try std.testing.expectEqual(@as(f32, 3.0), logits[2]);
    try std.testing.expectEqual(-std.math.inf(f32), logits[3]);
}

test "top_k non-positive is a no-op" {
    const alloc = std.testing.allocator;

    var logits = [_]f32{ 1.0, 2.0 };
    var s = kototama.Sampler{ .top_k = .init(0) };
    try s.applyMut(&logits, alloc);

    try std.testing.expectEqual(@as(f32, 1.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]);
}

test "min_p masks tokens below p of the max" {
    const alloc = std.testing.allocator;

    // Max logit is 4.0. p = 0.5 -> threshold = 4.0 + ln(0.5) ≈ 3.307, so
    // only the max survives given min_keep = 1.
    var logits = [_]f32{ 1.0, 4.0, 3.0 };
    var s = kototama.Sampler{ .min_p = .init(0.5, 1) };
    try s.applyMut(&logits, alloc);

    try std.testing.expectEqual(-std.math.inf(f32), logits[0]);
    try std.testing.expectEqual(@as(f32, 4.0), logits[1]);
    try std.testing.expectEqual(-std.math.inf(f32), logits[2]);
}

test "greedy samples the argmax" {
    const alloc = std.testing.allocator;

    var s = kototama.Sampler{ .greedy = .{} };
    const token = try s.sample(&.{ 0.1, 0.9, 0.3 }, alloc);
    try std.testing.expectEqual(@as(kototama.Token, 1), token);
}

test "dist samples within the vocabulary range" {
    const alloc = std.testing.allocator;

    var s = kototama.Sampler{ .dist = .init(42) };
    for (0..100) |_| {
        const token = try s.sample(&.{ 1.0, 1.0, 1.0 }, alloc);
        try std.testing.expect(token >= 0 and token < 3);
    }
}

test "chain applies transforms in order then selects" {
    const alloc = std.testing.allocator;

    var chain = kototama.Chain.init(alloc);
    defer chain.deinit(alloc);

    // top_k(1) isolates the maximum; temperature then scales it; greedy
    // selects. Any correct ordering yields the same winner here, so also
    // check the transform order by inspecting the scratch space.
    try chain.push(alloc, .{ .top_k = .init(1) });
    try chain.push(alloc, .{ .temperature = .init(2.0) });
    try chain.push(alloc, .{ .greedy = .{} });

    try std.testing.expectEqual(@as(usize, 3), chain.len());

    var s = kototama.Sampler{ .chain = chain };
    const token = try s.sample(&.{ 1.0, 3.0, 2.0 }, alloc);
    try std.testing.expectEqual(@as(kototama.Token, 1), token);
}

test "chain of transforms alone falls back to argmax" {
    const alloc = std.testing.allocator;

    var chain = kototama.Chain.init(alloc);
    defer chain.deinit(alloc);
    try chain.push(alloc, .{ .top_k = .init(2) });

    var s = kototama.Sampler{ .chain = chain };
    const token = try s.sample(&.{ 5.0, 1.0, 4.0 }, alloc);
    try std.testing.expectEqual(@as(kototama.Token, 0), token);
}

test "sampler names are stable" {
    const alloc = std.testing.allocator;
    _ = alloc;

    var s = kototama.Sampler{ .top_p = .init(0.9, 1) };
    try std.testing.expectEqualStrings("TopP", s.name());
}

// ---- context + sequence bookkeeping ------------------------------------

test "sequence checkout and release returns slots" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 128,
        .n_batch = 128,
        .n_seq_max = 2,
    });
    defer ctx.deinit();

    try std.testing.expectEqual(@as(usize, 2), ctx.freeSlots());

    var a = (try ctx.checkoutSequence()).?;
    defer a.deinit();
    try std.testing.expectEqual(@as(usize, 1), ctx.freeSlots());

    var b = (try ctx.checkoutSequence()).?;
    try std.testing.expectEqual(@as(usize, 0), ctx.freeSlots());

    try std.testing.expect((try ctx.checkoutSequence()) == null);

    b.deinit();
    try std.testing.expectEqual(@as(usize, 1), ctx.freeSlots());
}

test "push grows the sequence and produces logits" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 128,
        .n_batch = 128,
    });
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit();

    try std.testing.expectEqual(@as(usize, 0), seq.len());
    try std.testing.expect(seq.lastLogits() == null);

    const tokens = try model.tokenize(alloc, "hello world", false, false);
    defer alloc.free(tokens);
    try seq.extend(tokens);

    try std.testing.expectEqual(tokens.len, seq.len());
    const logits = seq.lastLogits().?;
    try std.testing.expectEqual(@as(usize, @intCast(model.nTokens())), logits.len);
}

test "greedy decode matches manual argmax over logits" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 128,
        .n_batch = 128,
    });
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit();

    const tokens = try model.tokenize(alloc, "Once upon a time", true, false);
    defer alloc.free(tokens);
    try seq.extend(tokens);

    const logits = seq.lastLogits().?;
    var manual: kototama.Token = 0;
    var best: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        if (v > best) {
            best = v;
            manual = @intCast(i);
        }
    }

    var greedy = kototama.Sampler{ .greedy = .{} };
    const sampled = (try seq.sample(&greedy, alloc)).?;
    try std.testing.expectEqual(manual, sampled);
}

test "pop removes the last token and its kv state" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 128,
        .n_batch = 128,
    });
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit();

    const tokens = try model.tokenize(alloc, "one two three", true, false);
    defer alloc.free(tokens);
    try seq.extend(tokens);
    try std.testing.expectEqual(tokens.len, seq.len());

    const popped = seq.pop();
    try std.testing.expectEqual(tokens[tokens.len - 1], popped.?);
    try std.testing.expectEqual(tokens.len - 1, seq.len());

    // Logits are invalidated by a pop until the next decode.
    try std.testing.expect(seq.lastLogits() == null);
}

test "remove drops a range of tokens" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 128,
        .n_batch = 128,
    });
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit();

    const tokens = try model.tokenize(alloc, "one two three four", true, false);
    defer alloc.free(tokens);
    try seq.extend(tokens);

    try std.testing.expect(seq.remove(1, 3));
    try std.testing.expectEqual(tokens.len - 2, seq.len());
    try std.testing.expectEqual(tokens[0], seq.get(0).?);
    try std.testing.expectEqual(tokens[3], seq.get(1).?);
}

test "empty sequence reports zero length" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 64,
        .n_batch = 64,
    });
    defer ctx.deinit();

    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit();

    try std.testing.expect(seq.isEmpty());
    try std.testing.expectEqual(@as(?kototama.Token, null), seq.pop());
}

test "n_seq_max zero falls back to llama default with a working pool" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    // Regression: the slot bitmap used to be sized from opts.n_seq_max
    // (0 here) while the C context got the library default, so checkout
    // always failed. Zero must also mean "all slots start free" — Zig's
    // allocator hands back undefined memory.
    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 64,
        .n_batch = 64,
        .n_seq_max = 0,
    });
    defer ctx.deinit();

    try std.testing.expect(ctx.freeSlots() > 0);
    var seq = (try ctx.checkoutSequence()).?;
    defer seq.deinit();
}

test "sequence logits stay valid while another sequence decodes" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    // Regression: logits used to be a pointer into the context-wide output
    // buffer, so B's push silently overwrote A's cached logits.
    var ctx = try kototama.Context.init(alloc, model, .{
        .n_ctx = 128,
        .n_batch = 128,
        .n_seq_max = 2,
    });
    defer ctx.deinit();

    var a = (try ctx.checkoutSequence()).?;
    defer a.deinit();
    var b = (try ctx.checkoutSequence()).?;
    defer b.deinit();

    const tokens_a = try model.tokenize(alloc, "first", true, false);
    defer alloc.free(tokens_a);
    try a.extend(tokens_a);

    const before = try alloc.dupe(f32, a.lastLogits().?);
    defer alloc.free(before);

    // Decode on the *other* sequence; A's cached logits must not move.
    const tokens_b = try model.tokenize(alloc, "second", true, false);
    defer alloc.free(tokens_b);
    try b.extend(tokens_b);

    const after = a.lastLogits().?;
    try std.testing.expectEqualSlices(f32, before, after);
}

test "failed checkout does not leak a pool slot" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    // Regression: checkoutSequence claims the slot before Sequence.init can
    // fail on its logits-buffer allocation. Let Context.init's slot-bitmap
    // allocation succeed (allocation #1), then fail the next one.
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 1 });
    var ctx = try kototama.Context.init(failing.allocator(), model, .{
        .n_ctx = 64,
        .n_batch = 64,
        .n_seq_max = 2,
    });
    defer ctx.deinit();

    try std.testing.expectEqual(@as(usize, 2), ctx.freeSlots());
    try std.testing.expectError(error.OutOfMemory, ctx.checkoutSequence());
    // The slot must be back in the pool, not silently consumed.
    try std.testing.expectEqual(@as(usize, 2), ctx.freeSlots());
}

test "greedy generation is deterministic across runs" {
    const alloc = std.testing.allocator;
    const model = try loadModel(alloc);

    // Generate 8 tokens twice with greedy decoding; both runs must agree,
    // which is the same guarantee the Rust snapshot tests encode.
    var runs: [2][]kototama.Token = undefined;
    for (&runs) |*out| {
        var ctx = try kototama.Context.init(alloc, model, .{
            .n_ctx = 256,
            .n_batch = 256,
        });
        defer ctx.deinit();

        var seq = (try ctx.checkoutSequence()).?;
        defer seq.deinit();

        const tokens = try model.tokenize(alloc, "Once upon a time", true, false);
        defer alloc.free(tokens);
        try seq.extend(tokens);

        var generated: std.ArrayList(kototama.Token) = .empty;
        for (0..8) |_| {
            const logits = seq.lastLogits().?;
            const token = kototama.samplers.argmax(logits);
            if (model.isEog(token)) break;

            try generated.append(alloc, token);
            _ = try seq.push(token);
        }
        out.* = try generated.toOwnedSlice(alloc);
    }
    defer alloc.free(runs[0]);
    defer alloc.free(runs[1]);

    try std.testing.expectEqualSlices(kototama.Token, runs[0], runs[1]);
    try std.testing.expectEqual(@as(usize, 8), runs[0].len);
}
