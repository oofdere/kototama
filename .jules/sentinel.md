## 2024-05-09 - FFI String Conversion and Length Integer Overflows in Rust

**Vulnerability:** The code performed unsafeguarded conversions of Rust `&str` length via `.len() as i32` when calling C functions, creating risks of integer overflow (leading to negative values) if a string's length exceeded `i32::MAX`. Additionally, untrusted inputs casted via `CString::new(s).unwrap()` created DoS vulnerability (panic) due to potential null bytes in strings.

**Learning:** When interacting with `llama_sys` FFI functions, Rust `usize` cannot be safely cast to C `int32_t` directly. Furthermore, `unwrap` should never be used on `CString::new` when processing strings provided at runtime, as a single null byte will crash the program. Replacing `vec![0; len]` with `Vec::with_capacity(len)` and `.set_len()` provides a performance benefit but requires extreme caution to never read uninitialized memory on failure.

**Prevention:**
1. Always use `i32::try_from(len).unwrap_or(i32::MAX)` (or proper error handling) rather than `len as i32`.
2. Replace `CString::new(s).unwrap()` with `.and_then(|s| std::ffi::CString::new(s).ok())` or equivalent safe handling.
3. Validate C function return values before calling `set_len()` on `Vec`s initialized with `with_capacity`.
