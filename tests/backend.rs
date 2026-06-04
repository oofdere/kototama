use rusty_llama::Backend;

#[test]
fn init() {
    let _b = Backend::acquire();
}

#[test]
fn acquire_twice_coexist() {
    let a = Backend::acquire();
    let b = Backend::acquire();
    // Both alive — just confirm no crash
    drop(a);
    drop(b);
}

#[test]
fn acquire_drop_acquire() {
    // Dropping all handles and re-acquiring should re-init cleanly
    {
        let _b = Backend::acquire();
    }
    let _b2 = Backend::acquire();
}

// ---------- llama_log_callback ----------
//
// The `llama_log_callback` function is the C-ABI bridge that llama.cpp /
// ggml use to surface log messages to the host. It is publicly exported
// (re-exported via `pub use backend::*`) and intended to be installed via
// `ggml_log_set` / `llama_log_set`. Until now nothing exercised it, so
// every match arm — ERROR, WARN, INFO, and the catch-all for
// NONE/DEBUG/CONT — was uncovered.

use std::ffi::CString;
use std::os::raw::{c_char, c_void};

fn invoke_log_callback(level: llama_sys::ggml_log_level, msg: &str) {
    let cstr = CString::new(msg).expect("test message must not contain NUL bytes");
    rusty_llama::llama_log_callback(level, cstr.as_ptr() as *const c_char, std::ptr::null_mut());
}

#[test]
fn log_callback_error_level_does_not_crash() {
    // ERROR is the eprint! "[ERROR] ..." arm.
    invoke_log_callback(llama_sys::ggml_log_level::GGML_LOG_LEVEL_ERROR, "test error\n");
}

#[test]
fn log_callback_warn_level_does_not_crash() {
    // WARN is the eprint! "[WARN] ..." arm.
    invoke_log_callback(llama_sys::ggml_log_level::GGML_LOG_LEVEL_WARN, "test warn\n");
}

#[test]
fn log_callback_info_level_does_not_crash() {
    // INFO is the print! "[INFO] ..." arm.
    invoke_log_callback(llama_sys::ggml_log_level::GGML_LOG_LEVEL_INFO, "test info\n");
}

#[test]
fn log_callback_default_arm_for_debug_level() {
    // DEBUG / NONE / CONT all fall through to the print!("{}", ...) catch-all.
    invoke_log_callback(llama_sys::ggml_log_level::GGML_LOG_LEVEL_DEBUG, "test debug\n");
}

#[test]
fn log_callback_default_arm_for_none_level() {
    invoke_log_callback(llama_sys::ggml_log_level::GGML_LOG_LEVEL_NONE, "test none\n");
}

#[test]
fn log_callback_default_arm_for_cont_level() {
    // CONT means "continue previous log line"; the callback prints the raw
    // message without a level tag.
    invoke_log_callback(llama_sys::ggml_log_level::GGML_LOG_LEVEL_CONT, "...continued\n");
}

#[test]
fn log_callback_ignores_user_data() {
    // The user_data cookie passed at install time is `_user_data` in the
    // callback signature — i.e. unused. A non-null pointer must therefore
    // be handled identically to a null one.
    let cstr = CString::new("user_data test\n").unwrap();
    let mut cookie: u32 = 0xDEADBEEF;
    rusty_llama::llama_log_callback(
        llama_sys::ggml_log_level::GGML_LOG_LEVEL_INFO,
        cstr.as_ptr() as *const c_char,
        &mut cookie as *mut u32 as *mut c_void,
    );
    // The cookie must be untouched.
    assert_eq!(cookie, 0xDEADBEEF);
}

#[test]
fn log_callback_accepts_empty_message() {
    // An empty (but valid, NUL-terminated) C string should not panic; the
    // CStr::from_ptr / to_string_lossy path must handle zero-length input.
    invoke_log_callback(llama_sys::ggml_log_level::GGML_LOG_LEVEL_INFO, "");
}

#[test]
fn log_callback_handles_non_utf8_bytes() {
    // llama.cpp delivers raw C bytes; lone non-UTF-8 bytes must go through
    // `to_string_lossy` without panicking.
    let bytes: &[u8] = b"bad-utf8: \xFF\xFE\xFD\0";
    // SAFETY: `bytes` is NUL-terminated and lives for the duration of the call.
    rusty_llama::llama_log_callback(
        llama_sys::ggml_log_level::GGML_LOG_LEVEL_WARN,
        bytes.as_ptr() as *const c_char,
        std::ptr::null_mut(),
    );
}
