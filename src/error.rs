//! Error types returned by this crate.

use std::fmt;

/// Anything that can go wrong when loading a model, creating a context,
/// decoding tokens, or detokenizing.
///
/// Decode failures map llama.cpp's documented `llama_decode` status codes;
/// the rest come from this crate's own checks and from llama.cpp returning
/// `NULL` on creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The path contains an interior NUL byte and cannot be passed to llama.cpp.
    InvalidPath,
    /// llama.cpp refused to load the model: the file is missing, is not a valid
    /// GGUF, or the parameters were rejected.
    ModelLoadFailed,
    /// llama.cpp refused to create the context: the parameters were rejected.
    ContextCreateFailed,
    /// The token has no textual piece in this vocabulary.
    NoPiece,
    /// The batch already holds its maximum number of tokens.
    BatchFull,
    /// `llama_decode` could not find a free KV slot for the batch — the
    /// context is full. Reduce the batch size or increase the context size.
    ContextFull,
    /// Decoding was aborted through the abort callback.
    Aborted,
    /// The batch handed to `llama_decode` was malformed (bad positions or
    /// sequence ids).
    InvalidBatch,
    /// The decode succeeded but produced no logits for the decoded token.
    MissingLogits,
    /// `llama_decode` failed with an unexpected status code.
    Fatal(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidPath => write!(f, "path contains an interior NUL byte"),
            Error::ModelLoadFailed => write!(f, "llama.cpp failed to load the model"),
            Error::ContextCreateFailed => write!(f, "llama.cpp failed to create the context"),
            Error::NoPiece => write!(f, "token has no text piece"),
            Error::BatchFull => write!(f, "batch is full"),
            Error::ContextFull => {
                write!(f, "no free KV slot for the batch (context is full)")
            }
            Error::Aborted => write!(f, "decoding was aborted"),
            Error::InvalidBatch => write!(f, "invalid batch passed to llama_decode"),
            Error::MissingLogits => write!(f, "decode produced no logits for the token"),
            Error::Fatal(code) => write!(f, "llama_decode failed with status {code}"),
        }
    }
}

impl std::error::Error for Error {}
