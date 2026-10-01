//! rusty-llama - Low-level but safe Rust bindings for llama.cpp
//!
//! This crate provides thin, safe wrappers around the llama.cpp C API.
//! It maintains close fidelity to the original API while providing:
//! - Memory safety through RAII
//! - Type safety where possible
//! - Thread safety through cheaply-cloned handles that share state behind a lock
//!
//! Everything is synchronous: calls run on the caller's thread and simply
//! serialize when a handle is shared between threads.
//!
//! # Example
//!
//! ```no_run
//! use rusty_llama::*;
//!
//! let model = Model::load_from_file("model.gguf", ModelParams::new())?;
//! let ctx = Context::new(&model, &ContextParams::new())?;
//! let mut seq = ctx.sequence().expect("no free sequence slot");
//!
//! seq.extend(&model.tokenize("Hello", true, true))?;
//! let mut sampler = Chain::new()
//!     .with(TopK::new(40))
//!     .with(Temperature::new(0.8))
//!     .with(Dist::new(42));
//! let token = seq.sample(&mut sampler).expect("no logits yet");
//! # Ok::<(), Error>(())
//! ```

mod error;
pub use error::Error;

mod model;
pub use model::*;

mod backend;
pub use backend::*;

mod context;
pub use context::{Context, ContextParams};

mod samplers;
pub use samplers::*;

mod vocab;

mod batch;
use batch::Batch;

mod sequence;
pub use sequence::*;

/// Async façade over the synchronous core, offloading blocking llama.cpp
/// calls to a small background thread pool. No async runtime required.
pub mod asynchronous;

/// Type alias for llama.cpp token IDs.
pub type Token = i32;
/// Type alias for llama.cpp position indices.
pub type Pos = i32;
/// Type alias for llama.cpp sequence IDs.
pub type SeqId = i32;

pub mod test_common;
