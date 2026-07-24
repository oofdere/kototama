//! rusty-llama - Low-level but safe Rust bindings for llama.cpp
//!
//! This crate provides thin, safe wrappers around the llama.cpp C API.
//! It maintains close fidelity to the original API while providing:
//! - Memory safety through RAII
//! - Type safety where possible
//! - Runtime-agnostic sync and async access through a dedicated worker thread

mod model;
pub use model::*;

mod backend;
pub use backend::*;

mod context;
pub use context::{Context, ContextError, ContextInitError, ContextParams, DecodeError};

mod samplers;
pub use samplers::*;

mod vocab;

pub(crate) mod common;

mod batch;
pub(crate) use batch::*;

mod sequence;
pub use sequence::*;

/// Type alias for llama.cpp token IDs.
pub type Token = i32;
/// Type alias for llama.cpp position indices.
pub type Pos = i32;
/// Type alias for llama.cpp sequence IDs.
pub type SeqId = i32;

pub mod test_common;
