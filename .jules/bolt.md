## 2026-05-01 - Avoid double tokenization
**Learning:** Calling llama_tokenize with a null pointer to find buffer size causes a full redundant pass over the input. Pre-allocating based on string length avoids this O(N) cost in the vast majority of cases.
**Action:** For FFI bindings that return needed size on failure, allocate a reasonable heuristic first and fallback to resizing.
