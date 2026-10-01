//! Sampling primitives — the Zig port of `src/samplers/*.rs`.
//!
//! # Design: tagged union instead of a trait
//!
//! Rust's samplers implement a `Sampler` trait with default methods
//! (`apply_mut` for transforms, `sample` for selectors), so any user type can
//! plug into a `Chain`. Zig has no traits, and the two idiomatic replacements
//! are a vtable struct (open for extension, every call indirect) or a tagged
//! union (closed set, one exhaustive switch). This port uses the tagged union:
//! every sampler's behaviour is visible in one dispatch table (`Sampler.applyMut`,
//! `Sampler.sample`), there are no indirect calls, and a reader never has to
//! hunt through ten files to learn what a chain will do. The cost is that a
//! downstream user cannot add a sampler without editing this file — see the
//! porting notes in README.md for the vtable variant if you need that.
//!
//! # The pipeline
//!
//! A sampler either transforms logits in place (Temperature, TopK, TopP,
//! MinP, Typical, TopNSigma, Xtc) or selects a token from them (Greedy, Dist).
//! `Chain` composes them: transforms run in order and the last element is
//! expected to be a selector.
const std = @import("std");
const Token = @import("root.zig").Token;

// ---- transform samplers -------------------------------------------------

/// Scales logits by `1 / temp`. A temperature of `0.0` is a no-op (use
/// `Greedy` for deterministic decoding instead).
pub const Temperature = struct {
    temp: f32,

    pub fn init(temp: f32) Temperature {
        return .{ .temp = temp };
    }

    pub fn applyMut(self: Temperature, logits: []f32) void {
        if (self.temp == 0.0) return;
        const inv_temp = 1.0 / self.temp;
        for (logits) |*logit| {
            logit.* *= inv_temp;
        }
    }
};

/// Keep only the `k` highest logits, masking the rest to `-inf`. Tokens tied
/// at the threshold are kept, so the survivor set may contain more than `k`
/// entries. A non-positive `k` leaves the logits untouched.
pub const TopK = struct {
    k: i32,

    pub fn init(k: i32) TopK {
        return .{ .k = k };
    }

    pub fn applyMut(self: TopK, logits: []f32, alloc: std.mem.Allocator) !void {
        if (self.k <= 0) return;

        const requested: usize = @intCast(self.k);
        const k: usize = @min(requested, logits.len);
        if (k == 0 or k >= logits.len) return;

        // Find the value of the k-th largest logit (1-indexed), then keep
        // everything greater-than-or-equal to it. Same partial-select
        // approach as the Rust `select_nth_unstable_by`.
        const copy = try alloc.dupe(f32, logits);
        defer alloc.free(copy);
        const thresh = kthLargest(copy, k);

        for (logits) |*logit| {
            if (logit.* < thresh) logit.* = -std.math.inf(f32);
        }
    }
};

/// Top-P (nucleus) sampling: keep the smallest set of highest-probability
/// tokens whose cumulative softmax probability reaches `p`, always keeping at
/// least `min_keep`. A `p >= 1.0` is a no-op.
pub const TopP = struct {
    p: f32,
    min_keep: usize,

    pub fn init(p: f32, min_keep: usize) TopP {
        return .{ .p = p, .min_keep = min_keep };
    }

    pub fn applyMut(self: TopP, logits: []f32, alloc: std.mem.Allocator) !void {
        if (self.p >= 1.0) return;
        const n = logits.len;
        if (n == 0) return;

        const probs = try softmaxProbs(logits, alloc);
        defer alloc.free(probs);

        // indices sorted by descending logit
        const idx = try sortedIndices(logits, alloc, .desc);
        defer alloc.free(idx);

        const keep = try alloc.alloc(bool, n);
        defer alloc.free(keep);
        @memset(keep, false);

        var cum: f32 = 0.0;
        for (idx, 0..) |i, rank| {
            keep[i] = true;
            cum += probs[i];
            if (cum >= self.p and rank + 1 >= self.min_keep) break;
        }

        for (logits, 0..) |*logit, i| {
            if (!keep[i]) logit.* = -std.math.inf(f32);
        }
    }
};

/// Min-P sampling: mask every logit more than `p` (in probability space)
/// below the maximum. Keeps at least `min_keep` candidates.
pub const MinP = struct {
    p: f32,
    min_keep: usize,

    pub fn init(p: f32, min_keep: usize) MinP {
        return .{ .p = p, .min_keep = min_keep };
    }

    pub fn applyMut(self: MinP, logits: []f32, alloc: std.mem.Allocator) !void {
        if (logits.len <= self.min_keep) return;

        var logit_max: f32 = -std.math.inf(f32);
        for (logits) |l| {
            if (l > logit_max) logit_max = l;
        }
        var thresh = logit_max + @log(self.p);

        if (self.min_keep > 0) {
            // Never mask below the (min_keep)-th best candidate.
            const copy = try alloc.dupe(f32, logits);
            defer alloc.free(copy);
            const floor = kthLargest(copy, self.min_keep);
            thresh = @min(thresh, floor);
        }

        for (logits) |*logit| {
            if (logit.* < thresh) logit.* = -std.math.inf(f32);
        }
    }
};

/// Locally typical sampling: keep tokens whose negative-log-probability is
/// closest to the distribution entropy, accumulating until the cumulative
/// probability exceeds `p` (keeping at least `min_keep`). A `p >= 1.0` is a
/// no-op.
///
/// Reference: Meister et al., "Typical Decoding for Natural Language Generation".
pub const Typical = struct {
    p: f32,
    min_keep: usize,

    pub fn init(p: f32, min_keep: usize) Typical {
        return .{ .p = p, .min_keep = min_keep };
    }

    pub fn applyMut(self: Typical, logits: []f32, alloc: std.mem.Allocator) !void {
        if (self.p >= 1.0) return;
        const n = logits.len;
        if (n == 0) return;

        const probs = try softmaxProbs(logits, alloc);
        defer alloc.free(probs);

        // entropy H = -sum(p * ln(p))
        var entropy: f32 = 0.0;
        for (probs) |p| {
            if (p > 0.0) entropy += -p * @log(p);
        }

        // score = |(-ln p) - H|; sort ascending so the most "typical"
        // tokens come first.
        const idx = try alloc.alloc(usize, n);
        defer alloc.free(idx);
        for (idx, 0..) |*slot, i| slot.* = i;

        const ScoreCtx = struct { probs: []const f32, entropy: f32 };
        std.mem.sort(usize, idx, ScoreCtx{ .probs = probs, .entropy = entropy }, struct {
            fn lessThan(ctx: ScoreCtx, a: usize, b: usize) bool {
                const sa = @abs((-@log(ctx.probs[a])) - ctx.entropy);
                const sb = @abs((-@log(ctx.probs[b])) - ctx.entropy);
                return sa < sb;
            }
        }.lessThan);

        const keep = try alloc.alloc(bool, n);
        defer alloc.free(keep);
        @memset(keep, false);

        var cum: f32 = 0.0;
        for (idx, 0..) |i, rank| {
            keep[i] = true;
            cum += probs[i];
            if (cum > self.p and (self.min_keep == 0 or rank + 1 >= self.min_keep)) break;
        }

        for (logits, 0..) |*logit, i| {
            if (!keep[i]) logit.* = -std.math.inf(f32);
        }
    }
};

/// Top-N-Sigma sampling: mask every logit more than `n` standard deviations
/// below the maximum (considering only non-masked logits). A non-positive `n`
/// is a no-op.
///
/// Reference: "Simplifying Top-p Sampling with Top-nσ".
pub const TopNSigma = struct {
    n: f32,

    pub fn init(n: f32) TopNSigma {
        return .{ .n = n };
    }

    pub fn applyMut(self: TopNSigma, logits: []f32) void {
        if (self.n <= 0.0 or logits.len <= 1) return;

        var max: f32 = -std.math.inf(f32);
        var sum: f32 = 0.0;
        var count: usize = 0;
        for (logits) |l| {
            if (l != -std.math.inf(f32)) {
                if (l > max) max = l;
                sum += l;
                count += 1;
            }
        }
        if (count == 0) return;

        const mean = sum / @as(f32, @floatFromInt(count));
        var var_sum: f32 = 0.0;
        for (logits) |l| {
            if (l != -std.math.inf(f32)) var_sum += (l - mean) * (l - mean);
        }
        const std_dev = @sqrt(var_sum / @as(f32, @floatFromInt(count)));

        const thresh = max - self.n * std_dev;
        for (logits) |*logit| {
            if (logit.* < thresh) logit.* = -std.math.inf(f32);
        }
    }
};

/// XTC (eXclude Top Choices): with probability `probability`, drop the tokens
/// whose softmax probability is `>= threshold` (keeping the rest). This is a
/// diversity-enhancing sampler; it only acts when at least two tokens clear
/// the threshold and at least `min_keep` would survive.
///
/// As in llama.cpp, the sampler is inert when `probability <= 0` or
/// `threshold > 0.5`. The masking rule matches the Rust implementation
/// exactly (`idx[..pos_last]`, i.e. the run above the threshold minus its
/// last element).
pub const Xtc = struct {
    probability: f32,
    threshold: f32,
    min_keep: usize,
    prng: std.Random.DefaultPrng,

    pub fn init(probability: f32, threshold: f32, min_keep: usize, seed: u64) Xtc {
        return .{
            .probability = probability,
            .threshold = threshold,
            .min_keep = min_keep,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn applyMut(self: *Xtc, logits: []f32, alloc: std.mem.Allocator) !void {
        if (self.probability <= 0.0 or self.threshold > 0.5 or logits.len < 2) return;
        if (self.prng.random().float(f32) > self.probability) return;

        const n = logits.len;
        const probs = try softmaxProbs(logits, alloc);
        defer alloc.free(probs);

        const idx = try sortedIndices(logits, alloc, .desc);
        defer alloc.free(idx);

        // pos_last = last rank (consecutive from the top) whose prob >= threshold
        var pos_last: usize = 0;
        for (idx, 0..) |i, rank| {
            if (probs[i] >= self.threshold) {
                pos_last = rank;
            } else {
                break;
            }
        }

        // Mask the top `pos_last` tokens, provided at least one is removed
        // and enough survive.
        if (pos_last > 0 and n - pos_last >= self.min_keep) {
            for (idx[0..pos_last]) |i| {
                logits[i] = -std.math.inf(f32);
            }
        }
    }
};

// ---- selection samplers -------------------------------------------------

/// Greedy / argmax selection. A pure token selector: `applyMut` is a no-op.
pub const Greedy = struct {};

/// Multinomial (weighted-random) token selection from the softmax
/// distribution. `applyMut` is a no-op; the selection happens in `sample`.
///
/// Note: the RNG is zig's Xoshiro256, seeded with the same `seed` value, so
/// draws differ from Rust's `StdRng` even with an identical seed. Seeded
/// streams are not comparable across languages — deterministic parity in the
/// test suite rests on Greedy, not Dist.
pub const Dist = struct {
    prng: std.Random.DefaultPrng,

    pub fn init(seed: u64) Dist {
        return .{ .prng = std.Random.DefaultPrng.init(seed) };
    }

    pub fn sample(self: *Dist, logits: []const f32, alloc: std.mem.Allocator) !Token {
        const probs = try softmaxProbs(logits, alloc);
        defer alloc.free(probs);

        var sum: f32 = 0.0;
        for (probs) |p| sum += p;
        const r = self.prng.random().float(f32) * sum;

        var acc: f32 = 0.0;
        for (probs, 0..) |e, id| {
            acc += e;
            if (acc >= r) return @intCast(id);
        }
        return @intCast(probs.len - 1);
    }
};

/// A pipeline of samplers applied in order: transforms run first, and the
/// final element is expected to be a selector ([`Greedy`] or [`Dist`]).
pub const Chain = struct {
    steps: std.ArrayList(Sampler),

    pub fn init(alloc: std.mem.Allocator) Chain {
        _ = alloc;
        return .{ .steps = .empty };
    }

    pub fn deinit(self: *Chain, alloc: std.mem.Allocator) void {
        self.steps.deinit(alloc);
    }

    /// Append a sampler to the pipeline.
    pub fn push(self: *Chain, alloc: std.mem.Allocator, step: Sampler) !void {
        try self.steps.append(alloc, step);
    }

    pub fn len(self: *const Chain) usize {
        return self.steps.items.len;
    }

    pub fn isEmpty(self: *const Chain) bool {
        return self.steps.items.len == 0;
    }
};

// ---- the union + dispatch ----------------------------------------------

/// The sampler set. Each variant carries its parameters (and RNG state where
/// needed); `applyMut` and `sample` below are the single dispatch table.
pub const Sampler = union(enum) {
    temperature: Temperature,
    top_k: TopK,
    top_p: TopP,
    min_p: MinP,
    typical: Typical,
    top_n_sigma: TopNSigma,
    xtc: Xtc,
    greedy: Greedy,
    dist: Dist,
    chain: Chain,

    /// Transform `logits` in place. Selection samplers are a no-op here.
    pub fn applyMut(self: *Sampler, logits: []f32, alloc: std.mem.Allocator) !void {
        switch (self.*) {
            .temperature => |s| s.applyMut(logits),
            .top_k => |s| try s.applyMut(logits, alloc),
            .top_p => |s| try s.applyMut(logits, alloc),
            .min_p => |s| try s.applyMut(logits, alloc),
            .typical => |s| try s.applyMut(logits, alloc),
            .top_n_sigma => |s| s.applyMut(logits),
            .xtc => |*s| try s.applyMut(logits, alloc),
            .greedy => {},
            .dist => {},
            .chain => |*chain| {
                for (chain.steps.items) |*step| {
                    try step.applyMut(logits, alloc);
                }
            },
        }
    }

    /// Transform a copy of `logits` and pick a token. For a plain transform
    /// this is "apply, then argmax"; selectors do their own picking; a chain
    /// runs every transform and then hands off to its final selector (or
    /// argmax when the chain ends in a transform).
    pub fn sample(self: *Sampler, logits: []const f32, alloc: std.mem.Allocator) !Token {
        switch (self.*) {
            .greedy => return argmax(logits),
            .dist => |*s| return try s.sample(logits, alloc),
            .chain => |*chain| {
                const scratch = try alloc.dupe(f32, logits);
                defer alloc.free(scratch);
                for (chain.steps.items) |*step| {
                    try step.applyMut(scratch, alloc);
                }
                if (chain.steps.items.len == 0) return argmax(scratch);
                return switch (chain.steps.items[chain.steps.items.len - 1]) {
                    .dist => |*d| try d.sample(scratch, alloc),
                    else => argmax(scratch),
                };
            },
            else => {
                const scratch = try alloc.dupe(f32, logits);
                defer alloc.free(scratch);
                try self.applyMut(scratch, alloc);
                return argmax(scratch);
            },
        }
    }

    /// Short, human-readable name of the active sampler.
    pub fn name(self: *const Sampler) []const u8 {
        return switch (self.*) {
            .temperature => "Temperature",
            .top_k => "TopK",
            .top_p => "TopP",
            .min_p => "MinP",
            .typical => "Typical",
            .top_n_sigma => "TopNSigma",
            .xtc => "Xtc",
            .greedy => "Greedy",
            .dist => "Dist",
            .chain => "Chain",
        };
    }
};

// ---- shared helpers -----------------------------------------------------

/// Pick the index of the largest logit, breaking ties towards the last —
/// matching the Rust `argmax`, which iterates with `max_by`.
pub fn argmax(logits: []const f32) Token {
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

const SortOrder = enum { asc, desc };

/// Values of the `k`-th largest element (1-indexed) in `buf`; `buf` is
/// permuted in place (the Rust code used `select_nth_unstable_by` the same
/// way).
fn kthLargest(buf: []f32, k: usize) f32 {
    std.debug.assert(k >= 1 and k <= buf.len);
    const want = buf.len - k;
    std.mem.sortUnstable(f32, buf, {}, struct {
        fn lessThan(_: void, a: f32, b: f32) bool {
            return a < b;
        }
    }.lessThan);
    return buf[want];
}

/// Indices of `logits` sorted by value (descending by default).
fn sortedIndices(
    logits: []const f32,
    alloc: std.mem.Allocator,
    order: SortOrder,
) ![]usize {
    const idx = try alloc.alloc(usize, logits.len);
    errdefer alloc.free(idx);
    for (idx, 0..) |*slot, i| slot.* = i;

    const Ctx = struct { vals: []const f32, order: SortOrder };
    std.mem.sort(usize, idx, Ctx{ .vals = logits, .order = order }, struct {
        fn lessThan(ctx: Ctx, a: usize, b: usize) bool {
            return switch (ctx.order) {
                .desc => ctx.vals[a] > ctx.vals[b],
                .asc => ctx.vals[a] < ctx.vals[b],
            };
        }
    }.lessThan);
    return idx;
}

/// Numerically stable softmax probabilities of `logits`.
fn softmaxProbs(logits: []const f32, alloc: std.mem.Allocator) ![]f32 {
    const probs = try alloc.alloc(f32, logits.len);
    errdefer alloc.free(probs);

    var max: f32 = -std.math.inf(f32);
    for (logits) |l| {
        if (l > max) max = l;
    }
    var sum: f32 = 0.0;
    for (logits, 0..) |l, i| {
        probs[i] = @exp(l - max);
        sum += probs[i];
    }
    for (probs) |*p| {
        p.* /= sum;
    }
    return probs;
}
