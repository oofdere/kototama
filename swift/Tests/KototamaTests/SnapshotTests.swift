//
//  SnapshotTests.swift
//  KototamaTests
//
//  Port of tests/snapshots.rs.
//
//  The Rust crate uses `insta` to snapshot token streams as YAML. The
//  expected values here are ported verbatim from those .snap files, which
//  makes this suite a cross-language equality check: if the Swift port is
//  faithful, the same model and llama.cpp build must produce the same token
//  IDs in both languages.
//
//  To record new snapshots (after a llama.cpp bump, say), run:
//      KOTOTAMA_UPDATE_SNAPSHOTS=1 swift test --filter SnapshotTests
//  which writes Snapshots/*.json, the counterpart of `INSTA_UPDATE=new`.
//

import Foundation
import Testing
@testable import Kototama

@Suite("snapshots")
struct SnapshotTests {
    // MARK: Tokenization snapshots

    @Test func snapshot_tokenize_hello_world() throws {
        let tokens = try sharedModel.tokenize("Hello, world!", addSpecial: false, parseSpecial: false)
        assertSnapshot(tokens, named: "tokenize_hello_world")
    }

    @Test func snapshot_tokenize_with_bos() throws {
        let tokens = try sharedModel.tokenize("Hello, world!", addSpecial: true, parseSpecial: false)
        assertSnapshot(tokens, named: "tokenize_with_bos")
    }

    @Test func snapshot_tokenize_multiline() throws {
        let tokens = try sharedModel.tokenize("line one\nline two\nline three", addSpecial: false, parseSpecial: false)
        assertSnapshot(tokens, named: "tokenize_multiline")
    }

    @Test func snapshot_tokenize_numbers() throws {
        let tokens = try sharedModel.tokenize("1, 2, 3, 4, 5", addSpecial: false, parseSpecial: false)
        assertSnapshot(tokens, named: "tokenize_numbers")
    }

    // MARK: Generation snapshots
    //
    // Greedy sampling is deterministic for a given model and llama.cpp
    // version. We snapshot token IDs rather than decoded text, so the
    // snapshots stay valid even if piece rendering changes.

    @Test func snapshot_generate_10_tokens() throws {
        let tokens = try greedyGenerate(prompt: "Once upon a time", count: 10)
        assertSnapshot(tokens, named: "generate_10_tokens")
    }

    @Test func snapshot_generate_numbers() throws {
        let tokens = try greedyGenerate(prompt: "1, 2, 3,", count: 8)
        assertSnapshot(tokens, named: "generate_numbers")
    }

    // MARK: Helpers

    /// Greedy generation loop, mirroring `greedy_generate` in
    /// tests/snapshots.rs: prompt, then argmax until end-of-generation.
    private func greedyGenerate(prompt: String, count: Int) throws -> [Token] {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())

        let promptTokens = try sharedModel.tokenize(prompt, addSpecial: true, parseSpecial: false)
        try seq.push(contentsOf: promptTokens)

        var generated: [Token] = []
        for _ in 0..<count {
            let token = argmax(try #require(seq.logits))
            if sharedModel.isEndOfGeneration(token) { break }
            generated.append(token)
            try seq.push(token)
        }
        return generated
    }
}

/// Compares `tokens` against a stored snapshot, or records it.
///
/// - When `KOTOTAMA_UPDATE_SNAPSHOTS` is set, writes the snapshot file and
///   passes (the `INSTA_UPDATE=new` workflow).
/// - Otherwise reads Snapshots/<name>.json and fails on any difference.
private func assertSnapshot(_ tokens: [Token], named name: String) {
    let file = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Snapshots/\(name).json")

    let actual = tokens.map(\.rawValue)

    if ProcessInfo.processInfo.environment["KOTOTAMA_UPDATE_SNAPSHOTS"] != nil {
        let data = try! JSONSerialization.data(
            withJSONObject: actual, options: [.prettyPrinted, .sortedKeys]
        )
        try! data.write(to: file)
        return
    }

    guard
        let data = try? Data(contentsOf: file),
        let expected = (try? JSONSerialization.jsonObject(with: data)) as? [Int32]
    else {
        Issue.record("snapshot '\(name)' is missing — run with KOTOTAMA_UPDATE_SNAPSHOTS=1 to record")
        return
    }

    #expect(actual == expected,
            "snapshot '\(name)' mismatch:\n  expected \(expected)\n  actual   \(actual)")
}
