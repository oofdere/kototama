//! Model loading plus the vocabulary queries the Rust crate splits across
//! `src/model.rs` and `src/vocab.rs`.
//!
//! A loaded `Model` owns the llama.cpp model handle and, transitively, a lease
//! on the global backend. It is a plain value with `init`/`deinit` — clone the
//! underlying handle with `retain` if you need shared ownership.
const std = @import("std");
const c = @import("root.zig").c;
const backend = @import("backend.zig");

/// Options for `Model.load`. Mirrors `llama_model_params`; unset fields take
/// llama.cpp's defaults (`llama_model_default_params`).
pub const ModelOptions = struct {
    /// Layers to offload to the GPU; 0 = CPU only, negative = all layers.
    n_gpu_layers: i32 = 0,
};

/// A loaded model: weights plus vocabulary. Free with `deinit`.
pub const Model = struct {
    handle: *c.llama_model,
    vocab: *const c.llama_vocab,
    owned: bool,
    backend_lease: backend.Backend,

    /// Load a model from a GGUF file. The caller receives the only owner;
    /// `deinit` frees the model and releases the backend lease.
    ///
    /// Returns `error.LoadFailed` when llama.cpp rejects the file (missing
    /// file, corrupt data, unsupported architecture).
    pub fn load(alloc: std.mem.Allocator, path: [:0]const u8, opts: ModelOptions) !Model {
        _ = alloc;
        var params = c.llama_model_default_params();
        params.n_gpu_layers = opts.n_gpu_layers;

        const lease = backend.Backend.acquire();
        const handle = c.llama_model_load_from_file(path.ptr, params);
        if (handle == null) {
            lease.release();
            return error.LoadFailed;
        }

        return .{
            .handle = handle.?,
            .vocab = c.llama_model_get_vocab(handle).?,
            .owned = true,
            .backend_lease = lease,
        };
    }

    /// Free the model. Only the owner may call this (see `retain`).
    pub fn deinit(self: *Model) void {
        if (!self.owned) return;
        c.llama_model_free(self.handle);
        self.backend_lease.release();
        self.* = undefined;
    }

    /// A second handle to the same model that does *not* own it: calling
    /// `deinit` on the copy is a no-op. The original must outlive the copy.
    pub fn retain(self: *const Model) Model {
        return .{
            .handle = self.handle,
            .vocab = self.vocab,
            .owned = false,
            .backend_lease = self.backend_lease,
        };
    }

    // ---- metadata -------------------------------------------------------

    /// Short human-readable description, e.g. `"llama ?B Q2_K - Medium"`.
    pub fn desc(self: *const Model, alloc: std.mem.Allocator) ![]u8 {
        const needed = c.llama_model_desc(self.handle, null, 0);
        if (needed <= 0) return try alloc.dupe(u8, "");
        var buf = try alloc.alloc(u8, @intCast(needed + 1));
        defer alloc.free(buf);
        const written = c.llama_model_desc(self.handle, buf.ptr, buf.len);
        if (written <= 0) return try alloc.dupe(u8, "");
        return try alloc.dupe(u8, buf[0..@intCast(written)]);
    }

    /// The chat template string, or `null` when the model carries none.
    /// Pass `name` to request a named template, or `null` for the default.
    pub fn chatTemplate(self: *const Model, alloc: std.mem.Allocator, name: ?[]const u8) !?[]u8 {
        var name_buf: [:0]u8 = undefined;
        defer if (name != null) alloc.free(name_buf);
        const name_ptr: [*:0]const u8 = if (name) |n| blk: {
            name_buf = try alloc.dupeZ(u8, n);
            break :blk name_buf.ptr;
        } else null;

        const text = c.llama_model_chat_template(self.handle, name_ptr);
        if (text == null) return null;
        return try alloc.dupe(u8, std.mem.span(text));
    }

    pub fn hasDecoder(self: *const Model) bool {
        return c.llama_model_has_decoder(self.handle);
    }

    pub fn hasEncoder(self: *const Model) bool {
        return c.llama_model_has_encoder(self.handle);
    }

    pub fn isDiffusion(self: *const Model) bool {
        return c.llama_model_is_diffusion(self.handle);
    }

    pub fn isHybrid(self: *const Model) bool {
        return c.llama_model_is_hybrid(self.handle);
    }

    pub fn isRecurrent(self: *const Model) bool {
        return c.llama_model_is_recurrent(self.handle);
    }

    /// The token that starts a decoder, or `null` when there is none.
    pub fn decoderStartToken(self: *const Model) ?Token {
        const token = c.llama_model_decoder_start_token(self.handle);
        return if (token == c.LLAMA_TOKEN_NULL) null else token;
    }

    // ---- tokenization ---------------------------------------------------

    /// Tokenize `text`. `add_special` prepends BOS, `parse_special` treats
    /// special tokens in the text as tokens instead of literal characters.
    ///
    /// The returned slice is allocated with `alloc`; free it with
    /// `alloc.free`.
    pub fn tokenize(
        self: *const Model,
        alloc: std.mem.Allocator,
        text: []const u8,
        add_special: bool,
        parse_special: bool,
    ) ![]Token {
        // First call with a null buffer returns the negated token count —
        // the same two-call pattern the Rust `tokenize` uses.
        const needed = c.llama_tokenize(
            self.vocab,
            text.ptr,
            @intCast(text.len),
            null,
            0,
            add_special,
            parse_special,
        );
        if (needed == 0) return try alloc.alloc(Token, 0);
        const count: usize = @intCast(-needed);

        const tokens = try alloc.alloc(Token, count);
        errdefer alloc.free(tokens);
        const written = c.llama_tokenize(
            self.vocab,
            text.ptr,
            @intCast(text.len),
            tokens.ptr,
            @intCast(tokens.len),
            add_special,
            parse_special,
        );
        if (written < 0) {
            alloc.free(tokens);
            return error.TokenizeFailed;
        }
        return tokens[0..@intCast(written)];
    }

    /// The text piece a token decodes to. The returned slice is allocated
    /// with `alloc`; free it with `alloc.free`.
    pub fn tokenToPiece(self: *const Model, alloc: std.mem.Allocator, token: Token) ![]u8 {
        var buf: [128]u8 = undefined;
        var n = c.llama_token_to_piece(self.vocab, token, &buf, buf.len, 0, true);
        if (n > 0) return try alloc.dupe(u8, buf[0..@intCast(n)]);
        if (n == 0) return try alloc.dupe(u8, "");

        // A negative return is the negated size the token needs; retry once
        // with exactly that much room.
        const needed: usize = @intCast(-n);
        const big = try alloc.alloc(u8, needed);
        defer alloc.free(big);
        n = c.llama_token_to_piece(self.vocab, token, big.ptr, @intCast(big.len), 0, true);
        if (n < 0) return error.PieceFailed;
        return try alloc.dupe(u8, big[0..@intCast(n)]);
    }

    /// Whether a token marks end-of-generation (EOS or another EOG marker).
    pub fn isEog(self: *const Model, token: Token) bool {
        return c.llama_vocab_is_eog(self.vocab, token);
    }

    // ---- vocabulary queries --------------------------------------------
    //
    // These mirror the `token_option!` macro in the Rust vocab.rs: each
    // returns `null` when the vocabulary has no such token.

    pub fn bosToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_bos(self.vocab));
    }
    pub fn eosToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_eos(self.vocab));
    }
    pub fn clsToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_cls(self.vocab));
    }
    pub fn eotToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_eot(self.vocab));
    }
    pub fn nlToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_nl(self.vocab));
    }
    pub fn padToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_pad(self.vocab));
    }
    pub fn sepToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_sep(self.vocab));
    }
    pub fn maskToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_mask(self.vocab));
    }
    pub fn fimPreToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_fim_pre(self.vocab));
    }
    pub fn fimSufToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_fim_suf(self.vocab));
    }
    pub fn fimMidToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_fim_mid(self.vocab));
    }
    pub fn fimPadToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_fim_pad(self.vocab));
    }
    pub fn fimRepToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_fim_rep(self.vocab));
    }
    pub fn fimSepToken(self: *const Model) ?Token {
        return optionalToken(c.llama_vocab_fim_sep(self.vocab));
    }

    pub fn addBos(self: *const Model) bool {
        return c.llama_vocab_get_add_bos(self.vocab);
    }
    pub fn addEos(self: *const Model) bool {
        return c.llama_vocab_get_add_eos(self.vocab);
    }
    pub fn addSep(self: *const Model) bool {
        return c.llama_vocab_get_add_sep(self.vocab);
    }

    pub fn tokenAttr(self: *const Model, token: Token) c.llama_token_attr {
        return c.llama_vocab_get_attr(self.vocab, token);
    }

    pub fn tokenScore(self: *const Model, token: Token) f32 {
        return c.llama_vocab_get_score(self.vocab, token);
    }

    /// The raw text of a vocabulary entry, as a borrowed C string.
    pub fn tokenText(self: *const Model, token: Token) [*:0]const u8 {
        return c.llama_vocab_get_text(self.vocab, token);
    }

    pub fn isControl(self: *const Model, token: Token) bool {
        return c.llama_vocab_is_control(self.vocab, token);
    }

    pub fn nTokens(self: *const Model) i32 {
        return c.llama_vocab_n_tokens(self.vocab);
    }

    pub fn vocabType(self: *const Model) c.llama_vocab_type {
        return c.llama_vocab_type(self.vocab);
    }
};

/// `LLAMA_TOKEN_NULL` means "no such token"; map it to `null`.
fn optionalToken(token: Token) ?Token {
    return if (token == c.LLAMA_TOKEN_NULL) null else token;
}

const Token = @import("root.zig").Token;
