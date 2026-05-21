use llama_sys::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static BACKEND_HANDLES: AtomicUsize = AtomicUsize::new(0);

// The private `()` field seals the tuple constructor: a `Backend` can only be
// obtained through `acquire()`. A `Backend` is a guard whose `Drop` decrements
// `BACKEND_HANDLES`, so every live value must correspond to an `acquire()` that
// incremented the count. Were the constructor public (as it is for a
// zero-field tuple struct), safe code could mint a `Backend` that never bumped
// the counter; dropping it would then free the backend out from under a live
// `Model`, or underflow the counter so the next `acquire()` skips init.
pub struct Backend(());

impl Backend {
    pub fn acquire() -> Backend {
        if BACKEND_HANDLES.fetch_add(1, Ordering::SeqCst) == 0 {
            unsafe {
                //ggml_log_set(Some(llama_log_callback), std::ptr::null_mut());
                //llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
                ggml_backend_load_all();
                llama_backend_init();
            }
        }
        Backend(())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if BACKEND_HANDLES.fetch_sub(1, Ordering::SeqCst) == 1 {
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


