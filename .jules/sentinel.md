## 2024-05-24 - FFI Memory Safety in Rusty Llama
**Vulnerability:** Memory safety risks when calling C FFI from Rust.
1. `CString::new()` calling `.unwrap()` on unsanitized string which could contain null bytes, causing panic (DoS).
2. `llama_model_desc` returning string length without null-terminator, but code allocated length exactly, which truncates the string and passes unallocated bound memory to FFI.
3. String lengths implicitly cast to `i32` when passed to FFI without bounds checks (`text.len() as i32`), which causes an integer overflow if strings exceed `i32::MAX`.
**Learning:** Working with C FFI code in Rust implies maintaining C semantics (e.g. null terminators and string limitations). Rust provides strong types, but conversion across boundaries requires extreme care for truncation, overflow, and panics.
**Prevention:**
- Replace `.unwrap()` with `.and_then(|s| std::ffi::CString::new(s).ok())` to silently drop or filter bad strings.
- Add `+ 1` to buffer lengths strictly when C expects a null-terminator, and read bytes manually rather than relying on `CStr`.
- Explicitly validate buffer dimensions (e.g., `text.len() <= i32::MAX`) before casting to smaller integer types for FFI.
