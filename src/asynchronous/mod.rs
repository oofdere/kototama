//! Async façade over the synchronous core.
//!
//! This module does not change how inference works — the same
//! [`crate::Model`], [`crate::Context`], and [`crate::Sequence`] run
//! underneath. The difference is *where* the expensive llama.cpp calls run:
//! loading a model, creating a context, and decoding tokens are shipped to a
//! small background thread pool and awaited, so they never stall the async
//! executor driving your code. Cheap local reads ([`Sequence::logits`],
//! [`Sequence::sample`], [`Sequence::tokens`]) stay synchronous — they touch
//! only memory and would gain nothing from the pool.
//!
//! Because the core is shared with the sync API, every method here is a thin
//! wrapper: clone the [`Context`] handle, run the blocking call on the pool,
//! then apply the result to the sequence's local token list. Nothing is held
//! across an `.await`, so the futures are [`Send`] and cannot deadlock against
//! the context lock.
//!
//! Use the `*_async` methods from any executor, or drive futures without one
//! using [`block_on`], a minimal std-only executor for examples and tests.
//!
//! # Example
//!
//! ```no_run
//! use rusty_llama::asynchronous::*;
//! use rusty_llama::*;
//!
//! # async fn run() -> Result<(), Error> {
//! let model = Model::load_from_file_async("model.gguf", ModelParams::new()).await?;
//! let ctx = Context::new_async(&model, &ContextParams::new()).await?;
//! let mut seq = ctx.sequence().expect("no free sequence slot");
//!
//! seq.extend_async(&model.tokenize("Hello", true, true)).await?;
//! let mut sampler = Chain::new()
//!     .with(TopK::new(40))
//!     .with(Temperature::new(0.8))
//!     .with(Dist::new(42));
//! let token = seq.sample(&mut sampler).expect("no logits yet");
//! # Ok(()) }
//! ```

mod blocking;

use std::future::Future;
use std::ops::Range;

use crate::{Context, ContextParams, Error, Model, ModelParams, Sequence};

pub use blocking::block_on;

use blocking::run_blocking;

/// `llama_model_params` is a plain C parameter bundle passed by value into
/// `llama_model_load_from_file`. Its raw pointer fields make Rust consider it
/// `!Send`, but moving the bundle to another thread for one call is fine: the
/// pointers name data the caller keeps alive for the duration of that call,
/// and nothing is shared across threads concurrently. This wrapper exists
/// only so the blocking closure can be `Send`; it is never dereferenced.
struct SendModelParams(ModelParams);

// SAFETY: see the type's doc comment.
unsafe impl Send for SendModelParams {}

/// Same as [`SendModelParams`], for `llama_context_params`.
struct SendContextParams(ContextParams);

// SAFETY: see [`SendModelParams`].
unsafe impl Send for SendContextParams {}

impl SendModelParams {
    fn into_inner(self) -> ModelParams {
        self.0
    }
}

impl SendContextParams {
    fn into_inner(self) -> ContextParams {
        self.0
    }
}

impl Model {
    /// Load a GGUF model from `path` on the thread pool.
    ///
    /// The async counterpart of [`Model::load_from_file`], with identical
    /// semantics.
    pub fn load_from_file_async(
        path: &str,
        params: ModelParams,
    ) -> impl Future<Output = Result<Self, Error>> + Send {
        let path = path.to_owned();
        let params = SendModelParams(params);
        run_blocking(move || Model::load_from_file(&path, params.into_inner()))
    }
}

impl Context {
    /// Create a context for `model` on the thread pool.
    ///
    /// The async counterpart of [`Context::new`], with identical semantics.
    /// Sequence checkout ([`Context::sequence`]) and other slot bookkeeping
    /// stay synchronous: they are plain memory operations.
    pub fn new_async(
        model: &Model,
        params: &ContextParams,
    ) -> impl Future<Output = Result<Self, Error>> + Send {
        let model = model.clone();
        let params = SendContextParams(*params);
        run_blocking(move || {
            let cfg = params.into_inner();
            Context::new(&model, &cfg)
        })
    }
}

impl Sequence {
    /// Decode `token` at the end of the sequence on the thread pool.
    ///
    /// The async counterpart of [`Sequence::push`], with identical semantics:
    /// on failure the sequence is left unchanged.
    pub fn push_async(&mut self, token: i32) -> impl Future<Output = Result<(), Error>> + Send + '_ {
        let ctx = self.ctx.clone();
        let id = self.id;
        let pos = self.tokens.len() as i32;
        async move {
            let logits = run_blocking(move || ctx.lock().decode_token(token, pos, id)).await?;
            self.logits = Some(logits);
            self.tokens.push(token);
            Ok(())
        }
    }

    /// Decode several tokens in order on the thread pool, stopping at the
    /// first failure.
    ///
    /// The async counterpart of [`Sequence::extend`], with identical
    /// semantics: the sequence is left with the tokens decoded so far.
    pub fn extend_async(
        &mut self,
        tokens: &[i32],
    ) -> impl Future<Output = Result<(), Error>> + Send + '_ {
        let tokens = tokens.to_vec();
        async move {
            for token in tokens {
                self.push_async(token).await?;
            }
            Ok(())
        }
    }

    /// Re-decode the last token on the thread pool to refresh the cached
    /// logits without pushing a new one. Does nothing on an empty sequence.
    ///
    /// The async counterpart of [`Sequence::decode`], with identical
    /// semantics.
    pub fn decode_async(&mut self) -> impl Future<Output = Result<(), Error>> + Send + '_ {
        let ctx = self.ctx.clone();
        let id = self.id;
        let last = self.tokens.last().copied();
        let pos = self.tokens.len().saturating_sub(1) as i32;
        async move {
            if let Some(token) = last {
                let logits = run_blocking(move || ctx.lock().refresh_token(token, pos, id)).await?;
                self.logits = Some(logits);
            }
            Ok(())
        }
    }

    /// Remove the last token on the thread pool, returning it. Invalidates the
    /// cached logits. Returns `None` on an empty sequence or when the KV-cache
    /// removal fails.
    ///
    /// The async counterpart of [`Sequence::pop`], with identical semantics.
    pub fn pop_async(&mut self) -> impl Future<Output = Option<i32>> + Send + '_ {
        let ctx = self.ctx.clone();
        let id = self.id;
        let len = self.tokens.len() as i32;
        async move {
            if len == 0 {
                return None;
            }
            let ok = run_blocking(move || ctx.lock().kv_remove(id, len - 1, len)).await;
            if ok {
                let token = self.tokens.pop();
                self.logits = None;
                token
            } else {
                None
            }
        }
    }

    /// Remove the tokens in `range` on the thread pool (token-list indices).
    ///
    /// The async counterpart of [`Sequence::remove`], with identical
    /// semantics: returns `false`, leaving the sequence unchanged, when the
    /// KV-cache removal fails.
    pub fn remove_async(&mut self, range: Range<usize>) -> impl Future<Output = bool> + Send + '_ {
        let ctx = self.ctx.clone();
        let id = self.id;
        let kv_range = range.clone();
        async move {
            let ok = run_blocking(move || ctx.lock().remove_token_range(id, &kv_range)).await;
            if ok {
                self.tokens.drain(range);
                self.logits = None;
            }
            ok
        }
    }
}
