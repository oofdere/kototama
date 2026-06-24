//! Global llama.cpp / ggml runtime lifecycle.
//!
//! llama.cpp owns process-global state — the ggml backend registry and the
//! `llama_backend_init` bookkeeping — that has to be initialized exactly
//! once before any model is loaded and freed once after the last model is
//! gone. [`Backend`] is the RAII handle that drives that lifecycle.
//!
//! ## When to acquire one
//!
//! Most callers never need to. [`Model::load_from_file`](crate::Model::load_from_file)
//! holds its own `Backend` internally for the lifetime of the loaded model,
//! so as long as at least one [`Model`](crate::Model) is alive the backend
//! stays initialized. Acquire one manually only when you need to keep the
//! backend warm across the gap between dropping one model and loading the
//! next — without an outstanding handle, that gap will trigger a full
//! `llama_backend_free` / re-init cycle.
//!
//! ## Reference counting
//!
//! Handles are tracked by a process-global `AtomicUsize`. The first
//! [`Backend::acquire`] performs `ggml_backend_load_all` +
//! `llama_backend_init`; the last [`Drop::drop`] performs
//! `llama_backend_free`. Acquires in between are cheap atomic bumps.
//!
//! `Backend` is `Send` / `Sync` (it's a zero-sized handle into global
//! state) and clone-equivalent via `acquire()`.

use llama_sys::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static BACKEND_HANDLES: AtomicUsize = AtomicUsize::new(0);

/// RAII handle to llama.cpp's process-global runtime.
///
/// The first live handle initializes the ggml backend registry and
/// `llama_backend_init`; the last handle dropped tears them down. Holding
/// one keeps the runtime alive across model reloads, which avoids the
/// init/free cost when you're loading more than one model in sequence.
///
/// See the [module docs](self) for when you actually need to construct one
/// yourself versus letting [`Model`](crate::Model) manage it for you.
pub struct Backend();

impl Backend {
    /// Acquire a handle to the global runtime, initializing it on the
    /// first call.
    ///
    /// Subsequent calls — while any prior handle is still alive — are
    /// cheap atomic bumps that skip re-initialization. The handle becomes
    /// invalid (i.e. the runtime can be torn down on the next drop) only
    /// once it is itself dropped.
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

impl Drop for Backend {
    /// Decrement the refcount; when the last handle drops, call
    /// `llama_backend_free` to tear down the global runtime.
    ///
    /// The `Freeing backend` line printed here is a temporary debug marker
    /// — useful while the lifecycle was being shaken out, and worth
    /// keeping until the matching log-callback wiring lands.
    fn drop(&mut self) {
        if BACKEND_HANDLES.fetch_sub(1, Ordering::SeqCst) == 1 {
            println!("Freeing backend");
            unsafe { llama_sys::llama_backend_free() };
        }
    }
}

/// `ggml_log_callback`-compatible adapter that routes llama.cpp / ggml log
/// messages to Rust's standard streams.
///
/// Not wired up by default — the call sites in [`Backend::acquire`] are
/// commented out so the crate doesn't take over the host process's stdio.
/// To opt in, pass this function to `ggml_log_set` and/or `llama_log_set`
/// directly after acquiring a [`Backend`].
///
/// Routing:
/// - `ERROR` → `stderr`, prefixed with `[ERROR]`
/// - `WARN`  → `stderr`, prefixed with `[WARN]`
/// - `INFO`  → `stdout`, prefixed with `[INFO]`
/// - anything else (debug, continuation lines) → `stdout`, no prefix
///
/// # Safety
///
/// `msg` must be a NUL-terminated C string. llama.cpp upholds this for
/// every call it makes through `ggml_log_callback`; do not invoke this
/// function manually from Rust without that guarantee.
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
