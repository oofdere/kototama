//
//  TestSupport.swift
//  KototamaTests
//
//  Shared fixtures, mirroring tests/common/mod.rs and src/test_common.rs.
//

import Foundation
import Kototama

/// Path to the test model file.
///
/// Same lookup order as the Rust crate's `test_common::model_path`:
/// `RUSTY_LLAMA_BENCH_MODEL`, then `RUSTY_LLAMA_TEST_MODEL`, then the
/// bundled TinyStories model.
func modelPath() -> String {
    ProcessInfo.processInfo.environment["RUSTY_LLAMA_BENCH_MODEL"]
        ?? ProcessInfo.processInfo.environment["RUSTY_LLAMA_TEST_MODEL"]
        ?? "./test-models/TinyStories-656K.Q2_K.gguf"
}

/// The shared test model, loaded once per test process.
///
/// Rust keeps a `OnceLock<Model>` for this; Swift's `static let` on a type is
/// the same lazy-once pattern. The repo root is located relative to this test
/// bundle's source so tests run from any working directory.
let sharedModel: Model = {
    // Tests run with the package root as CWD when invoked via `swift test`,
    // but be forgiving and search upward from the test bundle.
    let bundle = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // KototamaTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // swift
        .deletingLastPathComponent()   // repo root
    let defaultPath = bundle
        .appendingPathComponent("test-models/TinyStories-656K.Q2_K.gguf")
        .path

    let path: String
    if FileManager.default.fileExists(atPath: modelPath()) {
        path = modelPath()
    } else {
        path = defaultPath
    }

    var params = ModelParams()
    params.nGPULayers = 0
    do {
        return try Model.load(from: path, params: params)
    } catch {
        fatalError("failed to load test model at \(path): \(error)")
    }
}()

/// Minimal `ContextParams` for testing: small context, CPU.
///
/// Mirrors `test_common::test_ctx_params`.
func testContextParams() -> ContextParams {
    var params = ContextParams()
    params.nContext = 512
    params.nBatch = 512
    params.nSequencesMax = 4
    params.noPerf = true
    return params
}

/// Number of finite (unmasked) logits.
func finiteCount(_ logits: [Float]) -> Int {
    logits.filter { $0.isFinite }.count
}
