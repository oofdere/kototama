## 2024-05-24 - [Fix CString unwrap and length overflow]
**Vulnerability:** `unwrap()` on `CString::new()` could panic on null bytes, and casting `text.len()` to `i32` could cause integer overflow for strings larger than 2GB, potentially leading to out-of-bounds memory read by the C backend.
**Learning:** Rust FFI string conversions require defensive programming; `text.len()` must be safely converted to signed integers used by C to prevent unintended negative values or overflow.
**Prevention:** Always use safe conversions like `.ok()` for `CString` and `i32::try_from().unwrap_or()` for lengths passed to C functions.
