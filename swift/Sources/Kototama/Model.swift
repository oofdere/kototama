//
//  Model.swift
//  Kototama
//
//  Loading a GGUF file and asking it questions.
//

import CLlama

/// Parameters for loading a model.
///
/// Mirrors the Rust crate's `ModelParams`, which is a transparent wrapper over
/// `llama_model_params` that derefs straight to the C struct. Swift cannot
/// overload `.` for struct fields the way Rust's `Deref` works, so instead we
/// expose the handful of fields a caller realistically tunes and hand
/// everything else llama.cpp's defaults.
public struct ModelParams: Sendable {
    /// How many model layers to offload to the GPU. Negative means "all".
    /// Mirrors the Rust API's `params.n_gpu_layers` (set before loading).
    public var nGPULayers: Int32

    /// Load only the vocabulary, no weights. Useful for tokenize-only tools.
    public var vocabOnly: Bool

    public init(nGPULayers: Int32 = 99, vocabOnly: Bool = false) {
        self.nGPULayers = nGPULayers
        self.vocabOnly = vocabOnly
    }

    /// Builds the C parameter struct. Every field we do not model here keeps
    /// llama.cpp's own default (mmap on, no lock, no overrides, ...).
    func makeCParams() -> llama_model_params {
        var p = llama_model_default_params()
        p.n_gpu_layers = nGPULayers
        p.vocab_only = vocabOnly
        return p
    }
}

/// A loaded model. Cheap to copy: all copies share one llama.cpp model.
///
/// This is the Swift counterpart of the Rust crate's `Arc<ModelInner>` model
/// handle. Reference semantics come free from the language — no `#[derive(Clone)]`,
/// no `Deref`, no explicit `Send`/`Sync` proofs. Model queries are read-only
/// on the underlying C object, which is exactly why sharing is safe here.
public final class Model: @unchecked Sendable {
    /// The raw llama.cpp model handle.
    let handle: OpaquePointer

    /// The model's vocabulary, for token IDs, special tokens, and queries.
    ///
    /// Cached here so callers never have to re-query it. The Rust crate
    /// exposes these same queries as methods on `Model` itself (see
    /// `Vocabulary` for why this port collects them in one type instead).
    public let vocabulary: Vocabulary

    /// Keeps the process-wide backend alive for as long as any model exists.
    private let backend: Backend

    /// Wraps an already-loaded handle. Internal: use `Model.load(from:)`.
    init(handle: OpaquePointer, backend: Backend) {
        self.handle = handle
        self.backend = backend
        self.vocabulary = Vocabulary(handle: llama_model_get_vocab(handle))
    }

    /// Loads a model from a GGUF file.
    ///
    /// - Parameter path: Filesystem path to a `.gguf` file.
    /// - Parameter params: Load options; `ModelParams()` gives llama.cpp defaults.
    /// - Throws: `KototamaError.modelLoadFailed` when the file cannot be loaded.
    public static func load(from path: String, params: ModelParams = ModelParams()) throws -> Model {
        let backend = Backend.shared
        var cParams = params.makeCParams()
        guard let handle = llama_model_load_from_file(path, cParams) else {
            throw KototamaError.modelLoadFailed(path: path)
        }
        return Model(handle: handle, backend: backend)
    }

    deinit {
        llama_model_free(handle)
    }

    // MARK: Description and architecture

    /// Human-readable model description, e.g. `"TinyStories-656K Q2_K ..."`.
    ///
    /// The two-call pattern (probe the length, then read) mirrors
    /// `Model::desc` in the Rust crate.
    public func describe() throws -> String {
        let needed = llama_model_desc(handle, nil, 0)
        guard needed > 0 else { throw KototamaError.emptyModelDescription }
        var buffer = [CChar](repeating: 0, count: Int(needed) + 1)
        let written = llama_model_desc(handle, &buffer, buffer.count)
        guard written > 0 else { throw KototamaError.emptyModelDescription }
        // Truncate to what C actually wrote — the buffer is zero-padded past
        // that, and decoding the padding would append NUL characters. Rust's
        // `Model::desc` does the same `&buf[..written]` slice. The decode is
        // lossy, so a stray byte cannot trap the caller.
        return lossyString(buffer.prefix(Int(written)), count: Int(written))
    }

    /// Does this model use a decoder (i.e. can it generate text)?
    public var hasDecoder: Bool { llama_model_has_decoder(handle) }

    /// Does this model use an encoder (e.g. T5-style)? Generation via
    /// `TokenSequence` is decoder-only.
    public var hasEncoder: Bool { llama_model_has_encoder(handle) }

    /// The token that starts decoding, when the model defines one
    /// (encoder-decoder models). `nil` for decoder-only models.
    public var decoderStartToken: Token? {
        let token = llama_model_decoder_start_token(handle)
        return token == LLAMA_TOKEN_NULL ? nil : Token(token)
    }

    /// Is this a diffusion language model?
    public var isDiffusion: Bool { llama_model_is_diffusion(handle) }

    /// Is this a hybrid attention/recurrent model?
    public var isHybrid: Bool { llama_model_is_hybrid(handle) }

    /// Is this a recurrent (state-based) model?
    public var isRecurrent: Bool { llama_model_is_recurrent(handle) }

    /// The model's chat template in Jinja form, or `nil` if it has none.
    ///
    /// - Parameter name: Optional template name, for models that ship several.
    public func chatTemplate(named name: String? = nil) -> String? {
        guard let text = name.flatMap({ llama_model_chat_template(handle, $0) })
            ?? llama_model_chat_template(handle, nil)
        else {
            return nil
        }
        // Lossy, for the same reason `Vocabulary.text(of:)` is.
        return lossyString(text)
    }

    // MARK: Tokenization

    /// Splits text into token IDs.
    ///
    /// - Parameters:
    ///   - text: The text to tokenize.
    ///   - addSpecial: Insert BOS/EOS when the vocabulary is configured to.
    ///   - parseSpecial: Treat special tokens in `text` as tokens, not text.
    /// - Returns: The token IDs. Empty input with `addSpecial == false`
    ///   yields an empty array.
    ///
    /// The two-pass call (probe the token count, then fill) is how the C API
    /// wants to be driven, and matches `Model::tokenize` in the Rust crate.
    public func tokenize(
        _ text: String,
        addSpecial: Bool = true,
        parseSpecial: Bool = false
    ) throws -> [Token] {
        try text.withCString { cText in
            let byteCount = strlen(cText)
            // First call: with no output buffer, llama_tokenize returns the
            // token count as a negative number (the Rust crate mirrors this
            // convention with its `let len = -unsafe { ... }` probe).
            let probe = llama_tokenize(
                vocabulary.handle, cText, Int32(byteCount), nil, 0,
                addSpecial, parseSpecial
            )
            guard probe <= 0 else {
                throw KototamaError.tokenizationFailed(text: text)
            }
            let needed = Int(-probe)
            if needed == 0 { return [] }

            var buffer = [llama_token](repeating: 0, count: needed)
            let written = buffer.withUnsafeMutableBufferPointer { buf in
                llama_tokenize(
                    vocabulary.handle, cText, Int32(byteCount),
                    buf.baseAddress, Int32(buf.count),
                    addSpecial, parseSpecial
                )
            }
            guard written >= 0 else {
                throw KototamaError.tokenizationFailed(text: text)
            }
            return buffer.prefix(Int(written)).map { Token($0) }
        }
    }

    /// Converts a single token back to its text piece.
    ///
    /// - Parameter token: The token to decode.
    /// - Returns: The piece of text this token represents. May be empty for
    ///   special tokens (e.g. BOS), which is why the throwing form exists
    ///   alongside `piece(of:)`.
    public func piece(of token: Token) throws -> String {
        // Byte-fallback tokens decode to raw bytes (0x80...0xFF) that are not
        // valid UTF-8 on their own, and `String(cString:)` traps on those.
        // Rust uses `String::from_utf8_lossy` here, which never fails —
        // `String(decoding:as:)` is the Swift equivalent (U+FFFD for bad
        // bytes). Decoding the `written` prefix also avoids depending on the
        // buffer's zero terminator, since the C API says it does not write one.
        var buffer = [UInt8](repeating: 0, count: 64)
        let written = buffer.withUnsafeMutableBufferPointer { buf -> Int32 in
            guard let base = buf.baseAddress else { return -1 }
            let cChars = UnsafeMutableRawPointer(base).assumingMemoryBound(to: CChar.self)
            return llama_token_to_piece(
                vocabulary.handle, token.rawValue,
                cChars, Int32(buf.count), 0, true
            )
        }
        guard written >= 0 else {
            throw KototamaError.pieceConversionFailed(token: token)
        }
        return String(decoding: buffer.prefix(Int(written)), as: UTF8.self)
    }

    /// Non-throwing variant that maps failure to `nil`.
    ///
    /// A second overload named `piece(of:)` returning `String?` was the
    /// obvious design and is a trap: inside it, `try? piece(of: token)` resolves
    /// back to *itself*, so the call recurses until the stack overflows (a
    /// SIGBUS in practice, with no diagnostic pointing at the cause). Distinct
    /// names make the choice explicit instead of leaving it to overload
    /// resolution.
    public func pieceOrNil(of token: Token) -> String? {
        try? piece(of: token)
    }

    /// Decodes a run of tokens back into text. Inverse of `tokenize(_:...)`.
    public func detokenize(_ tokens: [Token]) -> String {
        tokens.map { pieceOrNil(of: $0) ?? "" }.joined()
    }

    /// Is this token an end-of-generation marker (EOS, EOT, ...)? Use it to
    /// stop a generation loop.
    public func isEndOfGeneration(_ token: Token) -> Bool {
        vocabulary.isEndOfGeneration(token)
    }
}
