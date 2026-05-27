#![allow(unused_imports)]

//! rusty-llama - Low-level but safe Rust bindings for llama.cpp
//!
//! This crate provides thin, safe wrappers around the llama.cpp C API.
//! It maintains close fidelity to the original API while providing:
//! - Memory safety through RAII
//! - Type safety where possible
//! - Thread safety via an actor model (powered by Spawned)

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
pub use vocab::*;

pub(crate) mod common;

mod batch;
pub(crate) use batch::*;

mod sequence;
pub use sequence::*;

pub mod test_common;
