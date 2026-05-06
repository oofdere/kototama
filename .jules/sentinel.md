## 2024-05-06 - FFI Integer Overflow and DoS via Null Bytes
**Vulnerability:** Found `text.len() as i32` casting which can overflow, and `.unwrap()` on `CString::new(s)` which can panic on null bytes.
**Learning:** Rust-to-C FFI bindings are prone to unchecked lengths causing memory safety issues in C, and unwrap-panics on FFI boundary inputs.
**Prevention:** Always use `i32::try_from(len)` or similar validation before passing to C, and handle `CString::new` results using `.ok()` or pattern matching to avoid runtime panics.
