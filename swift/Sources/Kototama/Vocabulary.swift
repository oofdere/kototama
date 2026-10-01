//
//  Vocabulary.swift
//  Kototama
//
//  Read-only queries about a model's token vocabulary.
//

import CLlama

/// Token attributes, mirroring `llama_token_attr` bit flags.
public struct TokenAttributes: OptionSet, Sendable, Hashable {
    public let rawValue: Int32

    public init(rawValue: Int32) {
        self.rawValue = rawValue
    }

    public static let unknown = TokenAttributes(rawValue: 1 << 0)
    public static let unused = TokenAttributes(rawValue: 1 << 1)
    public static let normal = TokenAttributes(rawValue: 1 << 2)
    public static let control = TokenAttributes(rawValue: 1 << 3)
    public static let userDefined = TokenAttributes(rawValue: 1 << 4)
    public static let byte = TokenAttributes(rawValue: 1 << 5)
    public static let normalized = TokenAttributes(rawValue: 1 << 6)
    public static let leftStrip = TokenAttributes(rawValue: 1 << 7)
    public static let rightStrip = TokenAttributes(rawValue: 1 << 8)
    public static let singleWord = TokenAttributes(rawValue: 1 << 9)
}

/// Which tokenizer family a model uses, mirroring `llama_vocab_type`.
public enum VocabularyType: Sendable, Hashable {
    case none
    case sentencePiece
    case bytePairEncoding
    case wordPiece
    case unigram
    case rwkv
    case plamo2

    init(clangValue: llama_vocab_type) {
        // C enums import as non-frozen raw-value types, so map explicitly.
        switch Int32(clangValue.rawValue) {
        case 1: self = .sentencePiece
        case 2: self = .bytePairEncoding
        case 3: self = .wordPiece
        case 4: self = .unigram
        case 5: self = .rwkv
        case 6: self = .plamo2
        default: self = .none
        }
    }
}

/// A read-only view onto a model's vocabulary.
///
/// Lives as a plain value type: the underlying `llama_vocab` is owned by the
/// model and outlives every query, so this view holds no resources and needs
/// no cleanup. (The Rust crate exposes these same queries as methods on
/// `Model` via a `token_option!` macro; here they collect in one type with a
/// clear job, which is easier to read than a macro.)
///
/// `Sendable` is sound here for the same reason the Rust crate can mark its
/// model `Send + Sync`: every query is read-only on immutable C data. The
/// pointer itself is opaque to Swift's checker, hence the unchecked opt-in.
public struct Vocabulary: @unchecked Sendable {
    /// The raw llama.cpp vocabulary pointer. Owned by the `Model`.
    let handle: OpaquePointer

    init(handle: OpaquePointer) {
        self.handle = handle
    }

    // MARK: Size and type

    /// Number of tokens in the vocabulary.
    public var count: Int32 { llama_vocab_n_tokens(handle) }

    /// Which tokenizer family this vocabulary belongs to.
    public var type: VocabularyType {
        VocabularyType(clangValue: llama_vocab_type(handle))
    }

    // MARK: Special tokens

    /// Beginning-of-sentence token, if the vocabulary defines one.
    public var bosToken: Token? { specialToken(llama_vocab_bos) }
    /// End-of-sentence token, if defined.
    public var eosToken: Token? { specialToken(llama_vocab_eos) }
    /// End-of-turn token, if defined.
    public var eotToken: Token? { specialToken(llama_vocab_eot) }
    /// Separator token, if defined.
    public var separatorToken: Token? { specialToken(llama_vocab_sep) }
    /// Newline token, if defined.
    public var newlineToken: Token? { specialToken(llama_vocab_nl) }
    /// Padding token, if defined.
    public var paddingToken: Token? { specialToken(llama_vocab_pad) }
    /// Mask token, if defined.
    public var maskToken: Token? { specialToken(llama_vocab_mask) }
    /// Class token, if defined.
    public var classToken: Token? { specialToken(llama_vocab_cls) }

    // MARK: Fill-in-the-middle tokens

    public var fimPrefixToken: Token? { specialToken(llama_vocab_fim_pre) }
    public var fimSuffixToken: Token? { specialToken(llama_vocab_fim_suf) }
    public var fimMiddleToken: Token? { specialToken(llama_vocab_fim_mid) }
    public var fimPaddingToken: Token? { specialToken(llama_vocab_fim_pad) }
    public var fimRepeatToken: Token? { specialToken(llama_vocab_fim_rep) }
    public var fimSeparatorToken: Token? { specialToken(llama_vocab_fim_sep) }

    // MARK: Insertion policy

    /// Does the tokenizer insert BOS automatically?
    public var addsBOS: Bool { llama_vocab_get_add_bos(handle) }
    /// Does the tokenizer insert EOS automatically?
    public var addsEOS: Bool { llama_vocab_get_add_eos(handle) }
    /// Does the tokenizer insert a separator automatically?
    public var addsSeparator: Bool { llama_vocab_get_add_sep(handle) }

    // MARK: Per-token queries

    /// Attribute flags for a token (control, byte, normalized, ...).
    public func attributes(of token: Token) -> TokenAttributes {
        TokenAttributes(rawValue: Int32(llama_vocab_get_attr(handle, token.rawValue).rawValue))
    }

    /// The unembedding score llama.cpp stores for the token, if any.
    public func score(of token: Token) -> Float {
        llama_vocab_get_score(handle, token.rawValue)
    }

    /// The raw vocabulary text for a token. May be empty for special tokens;
    /// use `Model.piece(of:)` when you want the rendered form.
    public func text(of token: Token) -> String {
        guard let text = llama_vocab_get_text(handle, token.rawValue) else { return "" }
        // Lossy: byte-fallback tokens hold raw bytes, not valid UTF-8.
        return lossyString(text)
    }

    /// Is this a control token (BOS, EOS, special markup, ...)?
    public func isControl(_ token: Token) -> Bool {
        llama_vocab_is_control(handle, token.rawValue)
    }

    /// Is this an end-of-generation token? `Model.isEndOfGeneration(_:)`
    /// forwards here.
    public func isEndOfGeneration(_ token: Token) -> Bool {
        llama_vocab_is_eog(handle, token.rawValue)
    }

    // MARK: Private helpers

    /// Wraps a `llama_vocab_*` query that returns a token or `LLAMA_TOKEN_NULL`.
    private func specialToken(
        _ query: (OpaquePointer) -> llama_token
    ) -> Token? {
        let token = query(handle)
        return token == LLAMA_TOKEN_NULL ? nil : Token(token)
    }
}
