## 2025-05-05 - [HIGH] Fix input validation in model string APIs
**Vulnerability:** DoS via unhandled panic (`unwrap`) when parsing strings with null bytes in `chat_template`. Buffer over-read via integer overflow when `text.len() as i32` is negative for string > 2GB in `tokenize`.
**Learning:** Rust string length conversion to C signed integers is unsafe if length is unbounded. `CString` fails if `\0` is in string, which was unchecked.
**Prevention:** Bound string sizes for C FFI before conversion to signed types. Use `ok().flatten()` or proper error matching when generating CString from user string variables.
