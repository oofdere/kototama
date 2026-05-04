## 2025-02-27 - [Fix FFI null pointer dereferences]
**Vulnerability:** Null pointer dereferences when converting C strings to Rust strings (CStr::from_ptr) in `src/backend.rs` and `src/vocab.rs`.
**Learning:** Raw pointers received from FFI calls must be validated as non-null before dereferencing, even when assumed to be valid strings, to prevent DoS via crashes.
**Prevention:** Always use `is_null()` checks on raw pointers prior to calling `CStr::from_ptr` or wrap the FFI calls in safe abstractions that gracefully handle null.
