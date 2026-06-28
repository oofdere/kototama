//! rusty-llama - Low-level but safe Rust bindings for llama.cpp
//!
//! This crate provides thin, safe wrappers around the llama.cpp C API.
//! It maintains close fidelity to the original API while providing:
//! - Memory safety through RAII
//! - Type safety where possible
//! - Thread safety via an actor model (powered by Spawned)
//!
//! # The canonical workflow
//!
//! Every program builds the same four-layer stack, top to bottom:
//!
//! 1. [`Backend`] — global llama.cpp init. Acquired transparently when the
//!    first [`Model`] is loaded and freed when the last handle drops; you
//!    rarely touch it directly.
//! 2. [`Model`] — a loaded GGUF file. Cheap to [`Clone`] (it's `Arc`-backed)
//!    and `Send + Sync`. Provides tokenization and vocab queries.
//! 3. [`Context`] — owns the KV cache and runs decode on a dedicated actor
//!    thread. Cheap to [`Clone`]; the actor stays alive until the last
//!    handle drops.
//! 4. [`Sequence`] — a single conversational slot inside a [`Context`].
//!    Owns its own token list and caches the logits from its last decode.
//!    Drop returns the slot to the context.
//!
//! On top of that, a [`Sampler`] turns the logits cached in a [`Sequence`]
//! into the next [`Token`]. The crate ships [`Temperature`], [`MinP`], and
//! [`Dist`] as building blocks; callers compose them by hand (see
//! `examples/simple_chat.rs`).
//!
//! # End-to-end example
//!
//! Greedy generation against a local model file. The code below requires a
//! real model on disk and a working llama.cpp build, so the doctest is
//! marked `no_run`.
//!
//! ```no_run
//! use rusty_llama::{Context, ContextParams, Model, ModelParams};
//!
//! // 1. Load the model. The backend is acquired automatically.
//! let mut model_params = ModelParams::new();
//! model_params.n_gpu_layers = 99;
//! let model = Model::load_from_file("model.gguf", model_params)
//!     .expect("failed to load model");
//!
//! // 2. Tokenize a prompt (add_special = true emits the BOS token).
//! let prompt = model.tokenize("Once upon a time", true, false);
//!
//! // 3. Open a context sized for prompt + generation.
//! let n_predict = 32;
//! let mut ctx_params = ContextParams::new();
//! ctx_params.n_ctx = (prompt.len() + n_predict) as u32;
//! ctx_params.n_batch = 1;
//! let ctx = Context::new(&model, &ctx_params).expect("failed to create context");
//!
//! // 4. Check out a sequence and feed it the prompt.
//! let mut seq = ctx.sequence().expect("no free slots");
//! seq.extend(&prompt);
//!
//! // 5. Greedy decode loop (argmax over the cached logits).
//! for _ in 0..n_predict {
//!     let (token, _) = seq
//!         .logits()
//!         .expect("no logits")
//!         .iter()
//!         .enumerate()
//!         .max_by(|(_, a), (_, b)| a.total_cmp(b))
//!         .unwrap();
//!     let token = token as i32;
//!     if model.is_eog(token) {
//!         break;
//!     }
//!     print!("{}", model.token_to_piece(token).unwrap());
//!     seq.push(token);
//! }
//! ```
//!
//! For a stochastic sampler pipeline (min-p → temperature → distribution)
//! see `examples/simple_chat.rs`.
//!
//! # Threading
//!
//! [`Model`] and [`Context`] are `Send + Sync` and cheaply cloneable, so a
//! single loaded model can back many contexts and a single context can be
//! shared between threads — each [`Sequence`] is an independent
//! conversational slot. Calls into a [`Context`] are serialized by its
//! actor thread.

mod model;
pub use model::*;

mod backend;
pub use backend::*;

mod context;
pub use context::{Context, ContextParams, DecodeError};

mod sampler;
pub use sampler::*;

mod sampler_chain;
pub use sampler_chain::*;

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
