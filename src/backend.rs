use llama_sys::*;
use std::sync::Mutex;

// SAFETY: This counter, plus the global `llama_backend_init`/`llama_backend_free`
// FFI calls that consume it, must always be observed together. Two threads that
// both see the same transition (0→1 or 1→0) but order their init/free against
// each other's observation race on the backend state. We therefore guard the
// whole acquire/drop sequence — counter update AND FFI call — under a single
// mutex, rather than relying on a lock-free atomic and a separate FFI call.
static BACKEND_HANDLES: Mutex<usize> = Mutex::new(0);

pub struct Backend();

impl Backend {
    pub fn acquire() -> Backend {
        let mut count = BACKEND_HANDLES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *count == 0 {
            unsafe {
                //ggml_log_set(Some(llama_log_callback), std::ptr::null_mut());
                //llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
                ggml_backend_load_all();
                llama_backend_init();
            }
        }
        *count += 1;
        Backend()
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let mut count = BACKEND_HANDLES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *count -= 1;
        if *count == 0 {
            println!("Freeing backend");
            unsafe { llama_sys::llama_backend_free() };
        }
    }
}

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


