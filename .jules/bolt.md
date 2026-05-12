## 2024-05-24 - Zero-initialization overhead in FFI
**Learning:** When writing Rust bindings for C functions that overwrite output buffers, `vec![0; len]` unnecessarily zero-initializes the entire buffer, adding overhead.
**Action:** Use `Vec::with_capacity(len)` and `vec.set_len()` after safely validating the C function's return code (e.g., handling negative values instead of panicked casts) to ensure performance without Undefined Behavior.
