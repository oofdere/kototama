## 2024-05-18 - [Eliminate vector zero-initialization overhead in FFI]
**Learning:** When interacting with FFI boundaries in Rust that expect a mutable output buffer (like `llama_tokenize`), zero-initializing vectors (`vec![0i32; len]`) adds an unnecessary O(N) memory memset overhead.
**Action:** Use `Vec::with_capacity(len)` and update its size cleanly via `tokens.set_len(n_tokens)` after ensuring the FFI call succeeds (i.e. checking `n_tokens > 0`). This relies on Rust inferring the scalar element type directly to match the C function signature.
