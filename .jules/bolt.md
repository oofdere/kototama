## 2024-05-13 - [Vec zero-initialization overhead before C FFI]
**Learning:** Using `vec![0; len]` before passing the buffer to a C function via FFI that fully overwrites it is a performance anti-pattern. It incurs O(n) memory allocation overhead for zero-initialization which pollutes the CPU cache unnecessarily.
**Action:** Use `Vec::with_capacity(len)` followed by passing `vec.as_mut_ptr()` to the C function, and only after the C function completes, update the length with `unsafe { vec.set_len(n_tokens) }`. Ensure safety constraints (like checking for failure) are properly handled.
