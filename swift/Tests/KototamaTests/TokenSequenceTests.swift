//
//  TokenSequenceTests.swift
//  KototamaTests
//
//  Port of tests/sequence.rs.
//

import Testing
@testable import Kototama

@Suite("sequence")
struct TokenSequenceTests {
    @Test func push_increases_len() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        #expect(seq.count == 0)
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try seq.push(tokens[0])
        #expect(seq.count == 1)
    }

    @Test func extend_fills_tokens() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hello world", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        #expect(seq.count == tokens.count)
    }

    @Test func tokens_accessor_matches_push_order() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("abc", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        #expect(seq.tokens == tokens)
    }

    @Test func index_operator() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        #expect(seq[0] == tokens[0])
    }

    @Test func get_returns_token() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        #expect(seq.token(at: 0) == tokens[0])
        #expect(seq.token(at: 999) == nil)
    }

    @Test func pop_decreases_len() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hello", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        let lenBefore = seq.count
        let popped = seq.pop()
        #expect(popped != nil)
        #expect(seq.count == lenBefore - 1)
    }

    @Test func pop_empty_returns_none() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        #expect(seq.pop() == nil)
    }

    @Test func logits_len_equals_vocab_size_after_push() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try seq.push(tokens[0])
        #expect(try #require(seq.logits).count == Int(sharedModel.vocabulary.count))
    }

    @Test func remove_range() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hello world", addSpecial: false, parseSpecial: false)
        let n = tokens.count
        try seq.push(contentsOf: tokens)
        #expect(seq.count == n)
        seq.remove(0..<1)
        #expect(seq.count == n - 1)
    }

    @Test func pos_min_max_after_push() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        #expect(seq.minPosition == -1)
        #expect(seq.maxPosition == -1)
        let tokens = try sharedModel.tokenize("hello", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        #expect(seq.minPosition >= 0)
        #expect(seq.maxPosition >= seq.minPosition)
    }

    @Test func copy_to() throws {
        var params = testContextParams()
        params.kvUnified = true
        let context = try Context(model: sharedModel, params: params)
        let src = try #require(context.checkoutSequence())
        let dst = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try src.push(contentsOf: tokens)
        try dst.push(contentsOf: tokens)
        src.copy(to: dst, range: 0..<tokens.count)
        #expect(dst.tokens == src.tokens)
    }

    @Test func multiple_sequences_independent() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq1 = try #require(context.checkoutSequence())
        let seq2 = try #require(context.checkoutSequence())
        let tokens1 = try sharedModel.tokenize("hello", addSpecial: false, parseSpecial: false)
        let tokens2 = try sharedModel.tokenize("world", addSpecial: false, parseSpecial: false)
        try seq1.push(contentsOf: tokens1)
        try seq2.push(contentsOf: tokens2)
        #expect(seq1.tokens == tokens1)
        #expect(seq2.tokens == tokens2)
        #expect(seq1.tokens != seq2.tokens)
    }

    @Test func multiple_sequences_generate_different_logits() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq1 = try #require(context.checkoutSequence())
        let seq2 = try #require(context.checkoutSequence())
        let tokens1 = try sharedModel.tokenize("hello", addSpecial: false, parseSpecial: false)
        let tokens2 = try sharedModel.tokenize("world", addSpecial: false, parseSpecial: false)
        try seq1.push(contentsOf: tokens1)
        try seq2.push(contentsOf: tokens2)
        let logits1 = try #require(seq1.logits)
        let logits2 = try #require(seq2.logits)
        #expect(logits1 != logits2)
    }

    @Test func free_slots_decreases_with_checkout() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let initialSlots = context.freeSlots
        let seq1 = try #require(context.checkoutSequence())
        #expect(context.freeSlots == initialSlots - 1)
        let seq2 = try #require(context.checkoutSequence())
        #expect(context.freeSlots == initialSlots - 2)
        _ = (seq1, seq2)
    }

    @Test func sequence_checkout_up_to_n_seq_max() throws {
        var params = testContextParams()
        params.nSequencesMax = 3
        let context = try Context(model: sharedModel, params: params)
        let seq1 = try #require(context.checkoutSequence())
        let seq2 = try #require(context.checkoutSequence())
        let seq3 = try #require(context.checkoutSequence())
        #expect(context.checkoutSequence() == nil)
        _ = (seq1, seq2, seq3)
    }

    @Test func dropping_sequence_frees_slot() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let initialSlots = context.freeSlots
        do {
            let seq = try #require(context.checkoutSequence())
            #expect(context.freeSlots == initialSlots - 1)
            _ = seq
        }
        #expect(context.freeSlots == initialSlots)
        let newSeq = try #require(context.checkoutSequence())
        _ = newSeq
    }

    @Test func copy_from() throws {
        var params = testContextParams()
        params.kvUnified = true
        let context = try Context(model: sharedModel, params: params)
        let src = try #require(context.checkoutSequence())
        let dst = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try src.push(contentsOf: tokens)
        try dst.push(contentsOf: tokens)
        dst.copy(from: src, range: 0..<tokens.count)
        #expect(dst.tokens == src.tokens)
    }

    // MARK: isEmpty

    @Test func is_empty_true_before_any_push() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        #expect(seq.isEmpty, "freshly created sequence should be empty")
    }

    @Test func is_empty_false_after_push() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try seq.push(tokens[0])
        #expect(!seq.isEmpty, "sequence should not be empty after a push")
    }

    @Test func is_empty_false_after_extend() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hello", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        #expect(!seq.isEmpty)
    }

    @Test func is_empty_true_after_pop_clears_sequence() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hi", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        for _ in 0..<tokens.count {
            seq.pop()
        }
        #expect(seq.isEmpty, "sequence should be empty after all tokens are popped")
    }

    @Test func is_empty_consistent_with_len() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        #expect(seq.isEmpty == (seq.count == 0))
        let tokens = try sharedModel.tokenize("hello", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)
        #expect(seq.isEmpty == (seq.count == 0))
    }

    // MARK: Sampling from a sequence

    @Test func sequence_sample_greedy_is_valid_token() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hello", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)

        var greedy = Greedy()
        let token = try #require(seq.sample(greedy))
        #expect(token.rawValue >= 0 && token.rawValue < sharedModel.vocabulary.count,
                "sampled token should be within vocab range")
    }

    @Test func sequence_sample_matches_argmax() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("once upon", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)

        let logits = try #require(seq.logits)
        let argmaxToken = logits.enumerated().max(by: { $0.element < $1.element })!.offset

        var greedy = Greedy()
        let sampled = try #require(seq.sample(greedy))
        #expect(sampled.rawValue == Int32(argmaxToken),
                "sampling with greedy should match manual argmax")
    }

    @Test func sequence_sample_with_temperature_in_vocab_range() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        let tokens = try sharedModel.tokenize("hello world", addSpecial: false, parseSpecial: false)
        try seq.push(contentsOf: tokens)

        let temp = Temperature(temp: 0.8)
        var dist = Dist(seed: 123)
        let logits = try #require(seq.logits)
        let scaled = temp.transform(logits)
        let token = dist.sample(scaled)
        #expect(token.rawValue >= 0 && token.rawValue < sharedModel.vocabulary.count,
                "temperature-sampled token should be in vocab range")
    }

    @Test func sequence_sample_without_logits_returns_nil() throws {
        // Nothing decoded yet — no logits to sample from. Ports the
        // "sample() takes &self" contract test with the sharper assertion
        // available in Swift: a fresh sequence has no logits at all.
        let context = try Context(model: sharedModel, params: testContextParams())
        let seq = try #require(context.checkoutSequence())
        var greedy = Greedy()
        #expect(seq.sample(greedy) == nil)
    }
}
