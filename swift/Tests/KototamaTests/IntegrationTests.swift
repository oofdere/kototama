//
//  IntegrationTests.swift
//  KototamaTests
//
//  Port of tests/integration.rs: model + context + samplers together.
//

import Testing
@testable import Kototama

@Suite("integration")
struct IntegrationTests {
    // MARK: Model error handling

    @Test func load_model_invalid_path_returns_err() {
        #expect(throws: (any Error).self) {
            _ = try Model.load(from: "/nonexistent/path/to/model.gguf")
        }
    }

    // MARK: Tokenization edge cases

    @Test func tokenize_empty_with_bos_gives_bos() throws {
        let tokens = try sharedModel.tokenize("", addSpecial: true, parseSpecial: false)
        #expect(tokens.count == 1, "empty text with addSpecial should produce exactly BOS")
        #expect(tokens[0] == sharedModel.vocabulary.bosToken)
    }

    @Test func tokenize_deterministic() throws {
        let a = try sharedModel.tokenize("the cat sat on the mat", addSpecial: false, parseSpecial: false)
        let b = try sharedModel.tokenize("the cat sat on the mat", addSpecial: false, parseSpecial: false)
        #expect(a == b, "tokenization should be deterministic")
    }

    // MARK: Sampler integration

    @Test func greedy_sample_matches_argmax() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("Once upon a time", addSpecial: true, parseSpecial: false)
        try seq.push(contentsOf: tokens)

        var bestToken = Token(0)
        var bestLogit = -Float.infinity
        for (index, value) in try #require(seq.logits).enumerated() {
            if value > bestLogit {
                bestLogit = value
                bestToken = Token(Int32(index))
            }
        }

        var greedy = Greedy()
        let sampled = try #require(seq.sample(greedy))
        #expect(sampled == bestToken, "greedy sampler should pick the argmax token")
    }

    @Test func sample_with_temperature_does_not_crash() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hello", addSpecial: true, parseSpecial: false)
        try seq.push(contentsOf: tokens)

        let temp = Temperature(temp: 0.8)
        var dist = Dist(seed: 42)
        let scaled = temp.transform(try #require(seq.logits))
        let token = dist.sample(scaled)
        #expect(token.rawValue >= 0 && token.rawValue < sharedModel.vocabulary.count)
    }

    // MARK: Multi-sequence

    @Test func two_sequences_independent() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seqA = try #require(context.checkoutSequence())
        let seqB = try #require(context.checkoutSequence())

        let tokensA = try sharedModel.tokenize("hello", addSpecial: true, parseSpecial: false)
        let tokensB = try sharedModel.tokenize("world", addSpecial: true, parseSpecial: false)
        try seqA.push(contentsOf: tokensA)
        try seqB.push(contentsOf: tokensB)

        #expect(seqA.tokens == tokensA)
        #expect(seqB.tokens == tokensB)
        let logitsA = try #require(seqA.logits)
        let logitsB = try #require(seqB.logits)
        #expect(logitsA != logitsB, "different prompts should produce different logits")
    }

    // MARK: Sequence logits state

    @Test func logits_empty_before_push() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        #expect(seq.logits == nil, "logits should be nil before any token is pushed")
    }

    // MARK: Generation determinism

    @Test func greedy_generation_is_deterministic() throws {
        // Two independent runs of the same greedy generation must agree
        // token for token.
        func generate() throws -> [Token] {
            let context = try Context(model: sharedModel, params: testContextParams())
            let seq = try #require(context.checkoutSequence())
            let prompt = try sharedModel.tokenize("the", addSpecial: true, parseSpecial: false)
            try seq.push(contentsOf: prompt)

            var generated: [Token] = []
            for _ in 0..<5 {
                let logits = try #require(seq.logits)
                let token = argmax(logits)
                if sharedModel.isEndOfGeneration(token) { break }
                generated.append(token)
                try seq.push(token)
            }
            return generated
        }

        let run1 = try generate()
        let run2 = try generate()
        #expect(run1 == run2, "greedy generation should be deterministic across runs")
    }

    // MARK: Token-to-piece coverage

    @Test func most_vocab_tokens_have_pieces() {
        let count = sharedModel.vocabulary.count
        var ok = 0
        for index in 0..<count {
            if sharedModel.pieceOrNil(of: Token(index)) != nil {
                ok += 1
            }
        }
        #expect(Double(ok) / Double(count) > 0.99,
                "at least 99% of vocab tokens should decode to a piece, got \(ok)/\(count)")
    }
}
