//! Tests for the llama.cpp log callback.

use std::ffi::CString;
use std::ptr;

use llama_sys::ggml_log_level;
use rusty_llama::llama_log_callback;

// On trunk this segfaults: `llama_log_callback` was a *safe* public function
// that dereferenced the caller-supplied `msg` pointer.
#[test]
fn null_msg_is_ignored() {
    unsafe {
        llama_log_callback(
            ggml_log_level::GGML_LOG_LEVEL_ERROR,
            ptr::null(),
            ptr::null_mut(),
        );
    }
}

#[test]
fn valid_msg_all_levels() {
    let msg = CString::new("log callback test\n").unwrap();
    for level in [
        ggml_log_level::GGML_LOG_LEVEL_NONE,
        ggml_log_level::GGML_LOG_LEVEL_DEBUG,
        ggml_log_level::GGML_LOG_LEVEL_INFO,
        ggml_log_level::GGML_LOG_LEVEL_WARN,
        ggml_log_level::GGML_LOG_LEVEL_ERROR,
        ggml_log_level::GGML_LOG_LEVEL_CONT,
    ] {
        unsafe { llama_log_callback(level, msg.as_ptr(), ptr::null_mut()) };
    }
}

#[test]
fn non_utf8_msg_is_lossy() {
    let bytes: &[u8] = &[0x66, 0x6f, 0xff, 0x6f, 0x00];
    unsafe {
        llama_log_callback(
            ggml_log_level::GGML_LOG_LEVEL_INFO,
            bytes.as_ptr().cast(),
            ptr::null_mut(),
        )
    };
}
