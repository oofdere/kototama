//
//  KototamaError.swift
//  Kototama
//
//  Every way an operation can fail, in one place.
//
//  The Rust crate returns `Result<_, ()>` from most fallible operations —
//  an empty error type that says *that* something failed but never *why*.
//  Swift's error protocol is the idiomatic counterpart, so this port takes
//  the opportunity to carry real diagnoses instead.
//

/// Errors thrown by Kototama operations.
public enum KototamaError: Error, Equatable, Sendable, CustomStringConvertible {
    /// A model file could not be loaded (missing file, bad GGUF, etc.).
    case modelLoadFailed(path: String)

    /// `llama_model_desc` returned no description text.
    case emptyModelDescription

    /// A context could not be created from the model and parameters.
    case contextCreationFailed

    /// Tokenizing `text` failed inside llama.cpp.
    case tokenizationFailed(text: String)

    /// Converting a token back to text failed.
    case pieceConversionFailed(token: Token)

    /// No text could be produced for a token (special tokens can be empty).
    case pieceIsEmpty(token: Token)

    /// The requested sequence slot range was invalid.
    case invalidRange(start: Int32, end: Int32)

    /// `llama_decode` failed with the given status.
    case decodeFailed(DecodeError)

    /// Logits were requested before any token had been decoded.
    case logitsUnavailable

    public var description: String {
        switch self {
        case .modelLoadFailed(let path):
            return "failed to load model from '\(path)'"
        case .emptyModelDescription:
            return "llama_model_desc returned an empty description"
        case .contextCreationFailed:
            return "failed to create a context from the model"
        case .tokenizationFailed(let text):
            return "failed to tokenize text: \(String(text.prefix(64)))"
        case .pieceConversionFailed(let token):
            return "failed to convert \(token) back to a text piece"
        case .pieceIsEmpty(let token):
            return "token \(token) has no text piece (special token?)"
        case .invalidRange(let start, let end):
            return "invalid position range \(start)..<\(end)"
        case .decodeFailed(let err):
            return "llama_decode failed: \(err)"
        case .logitsUnavailable:
            return "no logits yet — push a token before sampling"
        }
    }
}

/// Why a `llama_decode` call failed. Mirrors the return codes llama.cpp
/// documents for `llama_decode` (and the Rust crate's `DecodeError`).
public enum DecodeError: Error, Equatable, Sendable, CustomStringConvertible {
    /// No KV slot was available for the batch (status 1).
    case slotNotFound
    /// The decode was aborted (status 2).
    case aborted
    /// The batch was malformed (status -1).
    case invalidInput
    /// Anything below -1: an unrecoverable llama.cpp error.
    case fatalError

    public var description: String {
        switch self {
        case .slotNotFound: return "no KV slot available for the batch"
        case .aborted: return "decode aborted"
        case .invalidInput: return "invalid input batch"
        case .fatalError: return "fatal llama.cpp error"
        }
    }

    /// Maps a raw `llama_decode` return value to a `DecodeError`.
    init(status: Int32) {
        switch status {
        case 1: self = .slotNotFound
        case 2: self = .aborted
        case -1: self = .invalidInput
        default: self = .fatalError
        }
    }
}
