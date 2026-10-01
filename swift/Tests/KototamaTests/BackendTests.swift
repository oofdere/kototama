//
//  BackendTests.swift
//  KototamaTests
//
//  Port of tests/backend.rs.
//
//  The Rust crate exposes an explicit `Backend::acquire()` token with
//  reference-counted init/free. The Swift port keeps one lazy process-wide
//  instance instead (see Backend.swift), so the meaningful invariant left to
//  test is that repeated access is safe and models keep working — which is
//  what these cover.
//

import Foundation
import Testing
@testable import Kototama

@Suite("backend")
struct BackendTests {
    @Test func init_is_idempotent() {
        // Touching the shared backend twice must not double-initialize.
        let a = Backend.shared
        let b = Backend.shared
        #expect(a === b, "there is exactly one process-wide backend")
    }

    @Test func model_load_after_backend_touch() throws {
        // Loading through the shared backend works and produces a usable model.
        let model = try Model.load(from: bundledModelPath(), params: {
            var p = ModelParams()
            p.nGPULayers = 0
            return p
        }())
        #expect(model.vocabulary.count > 0)
    }
}

/// Path to the bundled test model. Delegates to the shared fixture, which
/// resolves the repo root correctly from this file's location.
private func bundledModelPath() -> String {
    // `sharedModel`'s initializer already searched `#filePath`-relative paths,
    // so re-derive the same location rather than duplicating the walk.
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // KototamaTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // swift
        .deletingLastPathComponent()  // repo root
        .appendingPathComponent("test-models/TinyStories-656K.Q2_K.gguf")
        .path
}
