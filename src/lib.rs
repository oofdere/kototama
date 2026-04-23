//! rusty-llama - Low-level but safe Rust bindings for llama.cpp
//!
//! This crate provides thin, safe wrappers around the llama.cpp C API.
//! It maintains close fidelity to the original API while providing:
//! - Memory safety through RAII
//! - Type safety where possible
//! - Comprehensive documentation

use llama_sys::*;

mod model;
pub use model::*;

mod backend;
pub use backend::*;

mod context;
pub use context::*;


