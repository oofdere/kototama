//! rusty-llama — safe, low-level Rust bindings for llama.cpp.
//!
//! The wrappers stay close to the C API, but add:
//! - automatic cleanup (everything is freed when it's dropped)
//! - lifetimes enforced by reference counting, so e.g. a model can't be freed
//!   while a context still uses it
//! - Rust types and `Option`/`Result` instead of sentinel values
//!
//! # The pieces
//!
//! ```text
//!   Backend   global llama.cpp init; you rarely touch it directly
//!      ▲
//!   Model     the weights + tokenizer          (Arc, Send + Sync, read-only)
//!      ▲
//!   Context   KV cache + scratch buffers       (Rc<RefCell>, single-threaded)
//!      ▲
//!   Sequence  one stream of tokens in a context, with its own logits
//! ```
//!
//! Each arrow means "keeps the thing above it alive".
//!
//! # Example
//!
//! ```no_run
//! use rusty_llama::*;
//!
//! let model = Model::load_from_file("model.gguf", ModelParams::new()).unwrap();
//! let ctx = Context::new(&model, &ContextParams::new()).unwrap();
//! let mut seq = ctx.sequence().unwrap();
//!
//! seq.extend(&model.tokenize("Once upon a time", true, false)).unwrap();
//! let mut sampler = Greedy::new();
//! for _ in 0..16 {
//!     let token = seq.sample(&mut sampler).unwrap();
//!     if model.is_eog(token) {
//!         break;
//!     }
//!     print!("{}", model.token_to_piece(token).unwrap());
//!     seq.push(token).unwrap();
//! }
//! ```
//!
//! # Threading (not yet — here's the plan)
//!
//! Everything is currently synchronous and single-threaded: every call does its
//! work and returns. The types are laid out so threads can be added without
//! rewriting them:
//!
//! - [`Model`] is already `Arc`-based and `Send + Sync`. Threads will share one
//!   copy of the weights in memory.
//! - [`Context`] is a thin `Rc<RefCell<ContextInner>>` handle. All of the FFI
//!   lives in `ContextInner` (see `src/context.rs`), so a threaded version only
//!   has to swap the handle for `Arc<Mutex<_>>` or a worker thread that owns
//!   the inner state.
//! - [`Sequence`] copies logits into its own reusable buffer, so reading them
//!   never needs access to the context.

mod backend;
mod batch;
mod context;
mod model;
mod samplers;
mod sequence;
mod vocab;

pub use backend::*;
pub use context::{Context, ContextParams, DecodeError};
pub use model::*;
pub use samplers::*;
pub use sequence::*;

/// A token id: an index into the model's vocabulary.
pub type Token = i32;
/// A token's position within its sequence (0 for the first token).
pub type Pos = i32;
/// A sequence's slot id within a context.
pub type SeqId = i32;

/// Helpers shared by the tests and benchmarks. Not part of the stable API.
#[doc(hidden)]
pub mod test_common;
