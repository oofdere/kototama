use llama_sys::*;
use std::sync::{Mutex, PoisonError};

/// Number of live [`Backend`] handles. Guarded by a mutex so that
/// initializing/freeing the backend and updating the count happen together:
/// otherwise a second thread could observe a non-zero count and start using
/// llama.cpp while the first thread is still populating the global backend
/// registry.
static BACKEND_HANDLES: Mutex<usize> = Mutex::new(0);

fn handles() -> std::sync::MutexGuard<'static, usize> {
    BACKEND_HANDLES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// RAII handle keeping the global llama.cpp backend initialized.
///
/// The backend is initialized when the first handle is acquired and freed when
/// the last one is dropped. Handles can only be created through
/// [`Backend::acquire`], because dropping a handle that never initialized the
/// backend would free it while models and contexts are still alive:
///
/// ```compile_fail
/// // Not constructible: this would decrement the handle count on drop.
/// let backend = rusty_llama::Backend();
/// ```
pub struct Backend(());

impl Backend {
    pub fn acquire() -> Backend {
        let mut handles = handles();
        if *handles == 0 {
            unsafe {
                //ggml_log_set(Some(llama_log_callback), std::ptr::null_mut());
                //llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
                ggml_backend_load_all();
                llama_backend_init();
            }
        }
        *handles += 1;
        Backend(())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let mut handles = handles();
        *handles -= 1;
        if *handles == 0 {
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
