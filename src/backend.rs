//! Process-wide llama.cpp backend lifecycle.
//!
//! llama.cpp keeps its ggml backends in global state that must be initialized
//! once per process (via `llama_backend_init` / `ggml_backend_load_all`) and
//! torn down with `llama_backend_free`. [`Backend`] is a reference-counted
//! guard around that state: the first live guard performs the one-time
//! initialization, and freeing happens only when the last guard is dropped.
//!
//! ## You rarely call this directly
//!
//! [`Model::load_from_file`](crate::Model::load_from_file) acquires its own
//! [`Backend`] and stores it inside the [`Model`](crate::Model), so as long as
//! a [`Model`](crate::Model) is alive the backend stays initialized. Reach for
//! [`Backend::acquire`] directly only when you need to keep the backend
//! resident across model loads or before any model exists — e.g. to amortize
//! initialization cost in a server that may construct and drop several
//! [`Model`](crate::Model)s over its lifetime.
//!
//! ## Ordering and thread-safety
//!
//! The reference count uses [`Ordering::SeqCst`], so concurrent
//! [`Backend::acquire`] / [`Drop`] calls agree on which call sees the
//! `0 → 1` transition (and runs the init) and which sees the `1 → 0`
//! transition (and runs the free). Note however that this only sequences the
//! Rust-side counter — it does not make llama.cpp's own init/free routines
//! reentrant.

use llama_sys::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of live [`Backend`] guards in this process.
///
/// The `0 → 1` transition (in [`Backend::acquire`]) triggers backend
/// initialization; the `1 → 0` transition (in [`Drop`]) tears it down.
static BACKEND_HANDLES: AtomicUsize = AtomicUsize::new(0);

/// RAII guard for the process-wide llama.cpp backend.
///
/// Each live [`Backend`] keeps a reference on the shared backend state. The
/// first guard initializes the backend (`ggml_backend_load_all` +
/// `llama_backend_init`); dropping the last guard tears it down with
/// `llama_backend_free`. Construct one with [`Backend::acquire`] —
/// [`Model::load_from_file`](crate::Model::load_from_file) already does this
/// internally and stores the guard in the [`Model`](crate::Model), so callers
/// normally do not need a [`Backend`] of their own.
pub struct Backend();

impl Backend {
    /// Acquire a guard on the llama.cpp backend, initializing it on the first
    /// call.
    ///
    /// Increments the live-guard count. If this is the first live guard, the
    /// backend is initialized (`ggml_backend_load_all` then
    /// `llama_backend_init`); subsequent calls only bump the count. The
    /// backend stays initialized for as long as any [`Backend`] is alive.
    pub fn acquire() -> Backend {
        if BACKEND_HANDLES.fetch_add(1, Ordering::SeqCst) == 0 {
            unsafe {
                //ggml_log_set(Some(llama_log_callback), std::ptr::null_mut());
                //llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
                ggml_backend_load_all();
                llama_backend_init();
            }
        }
        Backend()
    }
}

/// Releases this guard. When the last live guard is dropped, the llama.cpp
/// backend is torn down via `llama_backend_free`.
impl Drop for Backend {
    fn drop(&mut self) {
        if BACKEND_HANDLES.fetch_sub(1, Ordering::SeqCst) == 1 {
            println!("Freeing backend");
            unsafe { llama_sys::llama_backend_free() };
        }
    }
}

/// Bridges llama.cpp / ggml log messages to the Rust process's stdout/stderr.
///
/// Designed to be installed via `ggml_log_set` and `llama_log_set` (the calls
/// are currently commented out in [`Backend::acquire`]). Errors and warnings
/// are written to stderr with a tag (`[ERROR]`, `[WARN]`); info messages and
/// everything else go to stdout. The incoming message already includes its own
/// trailing newline, so the callback uses `print!`/`eprint!` rather than
/// `println!`/`eprintln!`.
///
/// # Safety
///
/// This is an FFI callback. `msg` must be a non-null, NUL-terminated C string
/// that remains valid for the duration of the call; llama.cpp upholds this
/// when invoking the callback. `_user_data` is the cookie passed at install
/// time and is ignored here.
#[allow(non_upper_case_globals)]
pub extern "C" fn llama_log_callback(
    level: ggml_log_level,
    msg: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    use std::ffi::CStr;
    let msg_str = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    match level {
        ggml_log_level::GGML_LOG_LEVEL_ERROR => eprint!("[ERROR] {}", msg_str),
        ggml_log_level::GGML_LOG_LEVEL_WARN => eprint!("[WARN] {}", msg_str),
        ggml_log_level::GGML_LOG_LEVEL_INFO => print!("[INFO] {}", msg_str),
        _ => print!("{}", msg_str),
    }
}


