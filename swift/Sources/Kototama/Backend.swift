//
//  Backend.swift
//  Kototama
//
//  Process-wide llama.cpp backend lifecycle.
//

import CLlama

/// Keeps the llama.cpp backend alive.
///
/// llama.cpp requires `llama_backend_init()` before any model is loaded and
/// `llama_backend_free()` once the last one is gone. Rather than making the
/// caller manage that (the Rust crate does this with a reference count behind
/// an opaque `Backend` token), we hold one shared instance for the whole
/// process and initialize it lazily on first use.
///
/// This is also where Metal, the CPU backend, and BLAS get registered:
/// `ggml_backend_load_all()` links every backend compiled into the archive.
///
/// You never construct this yourself — `Model` does it internally.
public final class Backend: Sendable {
    /// The single process-wide instance. First touch runs `llama_backend_init`.
    static let shared = Backend()

    private init() {
        ggml_backend_load_all()
        llama_backend_init()
    }

    deinit {
        llama_backend_free()
    }
}
