use llama_sys::*;
use std::sync::Once;

static BACKEND_INIT: Once = Once::new();

pub struct Backend;

impl Backend {
    pub fn acquire() -> Backend {
        BACKEND_INIT.call_once(|| {
            unsafe {
                ggml_backend_load_all();
                llama_backend_init();
            }
        });
        Backend
    }
}

/// # Safety
///
/// `msg` must be a valid, non-null, null-terminated C string for the
/// duration of this call. This is guaranteed by the llama.cpp log callback
/// contract.
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
