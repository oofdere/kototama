//! Global llama.cpp backend lifecycle.

use llama_sys::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static BACKEND_HANDLES: AtomicUsize = AtomicUsize::new(0);

/// Handle that keeps the llama.cpp backend initialized.
///
/// Normally created for you: [`crate::Model::load_from_file`] acquires one and
/// holds it for the lifetime of the model. The backend is initialized when the
/// first handle is acquired and freed when the last one is dropped.
pub struct Backend();

impl Backend {
    /// Acquire a backend handle, initializing the backend on first use.
    pub fn acquire() -> Backend {
        if BACKEND_HANDLES.fetch_add(1, Ordering::SeqCst) == 0 {
            unsafe {
                ggml_backend_load_all();
                llama_backend_init();
            }
        }
        Backend()
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if BACKEND_HANDLES.fetch_sub(1, Ordering::SeqCst) == 1 {
            unsafe { llama_backend_free() };
        }
    }
}

/// Log callback that routes llama.cpp log lines to stdout/stderr.
///
/// Not installed by default (llama.cpp logs to stderr on its own). To use it,
/// pass it to `llama_sys::llama_log_set` once at startup.
///
/// # Safety
///
/// Called by llama.cpp with a valid `msg` pointer; call it only through
/// `llama_log_set`.
#[allow(non_upper_case_globals)]
pub unsafe extern "C" fn llama_log_callback(
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
