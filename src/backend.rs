//! Global llama.cpp setup and teardown.
//!
//! llama.cpp needs `llama_backend_init` called once before anything else, and
//! `llama_backend_free` called once at the end. That's global, process-wide
//! state, so we reference count it: the first [`Backend`] handle initialises
//! the backend and the last one to be dropped frees it.
//!
//! You rarely need this directly — [`crate::Model::load_from_file`] acquires a
//! handle for you and the model keeps it until it's freed.

use std::sync::atomic::{AtomicUsize, Ordering};

use llama_sys::*;

/// How many [`Backend`] handles currently exist.
///
/// An atomic rather than a plain integer so this stays correct once models are
/// created and dropped from several threads.
static BACKEND_HANDLES: AtomicUsize = AtomicUsize::new(0);

/// A handle that keeps llama.cpp's global backend initialised while it exists.
pub struct Backend(());

impl Backend {
    /// Get a handle, initialising the backend if this is the first one.
    pub fn acquire() -> Backend {
        // `fetch_add` returns the value *before* adding, so 0 means "we're first".
        if BACKEND_HANDLES.fetch_add(1, Ordering::SeqCst) == 0 {
            unsafe {
                ggml_backend_load_all();
                llama_backend_init();
            }
        }
        Backend(())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        // 1 before subtracting means we were the last handle.
        if BACKEND_HANDLES.fetch_sub(1, Ordering::SeqCst) == 1 {
            unsafe { llama_backend_free() };
        }
    }
}

/// A log callback that forwards llama.cpp's messages to stdout/stderr.
///
/// Not installed by default. To use it:
///
/// ```no_run
/// unsafe {
///     llama_sys::llama_log_set(Some(rusty_llama::llama_log_callback), std::ptr::null_mut());
/// }
/// ```
pub extern "C" fn llama_log_callback(
    level: ggml_log_level,
    msg: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    let msg = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    match level {
        ggml_log_level::GGML_LOG_LEVEL_ERROR => eprint!("[ERROR] {msg}"),
        ggml_log_level::GGML_LOG_LEVEL_WARN => eprint!("[WARN] {msg}"),
        ggml_log_level::GGML_LOG_LEVEL_INFO => print!("[INFO] {msg}"),
        _ => print!("{msg}"),
    }
}
