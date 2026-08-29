// llama.cpp reports some failures by throwing C++ exceptions. Unwinding an
// exception across the FFI boundary into Rust aborts the process, so calls that
// can throw are routed through a C++ try/catch here.

#include <cstdint>

#include <llama.h>

extern "C" bool rusty_llama_tokenize(const llama_vocab * vocab, const char * text, int32_t text_len,
                                     llama_token * tokens, int32_t n_tokens_max, bool add_special,
                                     bool parse_special, int32_t * n_tokens_out) {
    try {
        *n_tokens_out =
            llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special);
        return true;
    } catch (...) {
        return false;
    }
}
