## 2026-05-14 - Optimize FFI Vec initialization
**Learning:** Avoid `vec![0; len]` zero-initialization when allocating buffers for FFI calls that overwrite the entire array.
**Action:** Use `Vec::with_capacity(len)` combined with `unsafe { vec.set_len(len) }` after checking for negative error codes from the FFI call.
