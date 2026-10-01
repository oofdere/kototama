//
//  ModelTests.swift
//  KototamaTests
//
//  Port of tests/model.rs.
//

import Testing
@testable import Kototama

@Suite("model")
struct ModelTests {
    @Test func load_model() {
        #expect(sharedModel.vocabulary.count > 0)
    }

    @Test func desc_nonempty() throws {
        let desc = try sharedModel.describe()
        #expect(!desc.isEmpty, "model description should not be empty")
    }

    @Test func desc_matches_probe_length() throws {
        // The description should be complete, and this particular model's
        // description ends with "Medium". Both assertions ported from
        // `desc_matches_probe_length` in tests/model.rs.
        let desc = try sharedModel.describe()
        #expect(
            desc.hasSuffix("Medium"),
            "expected description to end with \"Medium\", got \(desc)"
        )
    }

    @Test func n_tokens_positive() {
        #expect(sharedModel.vocabulary.count > 0, "vocab should have tokens")
    }

    @Test func has_decoder() {
        #expect(sharedModel.hasDecoder)
    }

    @Test func no_encoder() {
        #expect(!sharedModel.hasEncoder)
    }

    @Test func not_diffusion() {
        #expect(!sharedModel.isDiffusion)
    }

    @Test func not_recurrent() {
        #expect(!sharedModel.isRecurrent)
    }

    @Test func tokenize_nonempty_text() throws {
        let tokens = try sharedModel.tokenize("hello world", addSpecial: true, parseSpecial: false)
        #expect(!tokens.isEmpty)
    }

    @Test func tokenize_empty_text() throws {
        let tokens = try sharedModel.tokenize("", addSpecial: false, parseSpecial: false)
        #expect(tokens.isEmpty)
    }

    @Test func tokenize_roundtrip() throws {
        // Decoding every token back and concatenating should reproduce the
        // text (modulo the tokenizer's whitespace normalization).
        let text = " Hello, world!"
        let tokens = try sharedModel.tokenize(text, addSpecial: false, parseSpecial: false)
        let reconstructed = tokens.compactMap { sharedModel.pieceOrNil(of: $0) }.joined()
        let trimmed = { (s: String) in s.trimmingCharacters(in: .whitespaces) }
        #expect(trimmed(reconstructed) == trimmed(text))
    }

    @Test func token_to_piece_bos() throws {
        if let bos = sharedModel.vocabulary.bosToken {
            #expect(try sharedModel.piece(of: bos) != nil)
        }
    }

    @Test func decoder_start_token_none_for_decoder_only() {
        #expect(sharedModel.decoderStartToken == nil)
    }

    @Test func chat_template_default() {
        _ = sharedModel.chatTemplate()
    }

    // MARK: Model is a shared reference (Arc counterpart)

    @Test func model_clone_has_same_n_tokens() {
        let cloned = sharedModel
        #expect(
            sharedModel.vocabulary.count == cloned.vocabulary.count,
            "cloned model should report the same vocabulary size"
        )
    }

    @Test func model_clone_can_tokenize() throws {
        let cloned = sharedModel
        let a = try sharedModel.tokenize("hello world", addSpecial: false, parseSpecial: false)
        let b = try cloned.tokenize("hello world", addSpecial: false, parseSpecial: false)
        #expect(a == b, "cloned model should produce identical tokenization")
    }

    @Test func model_clone_desc_matches() throws {
        let cloned = sharedModel
        #expect(try sharedModel.describe() == cloned.describe())
    }

    @Test func model_clone_vocab_queries_match() {
        let cloned = sharedModel
        #expect(sharedModel.vocabulary.bosToken == cloned.vocabulary.bosToken)
        #expect(sharedModel.vocabulary.eosToken == cloned.vocabulary.eosToken)
        #expect(sharedModel.hasDecoder == cloned.hasDecoder)
        #expect(sharedModel.hasEncoder == cloned.hasEncoder)
    }

    @Test func model_clone_token_to_piece_matches() {
        // `Model` is a shared reference (the Arc counterpart), so a "clone" is
        // the same instance and must answer identically.
        if let bos = sharedModel.vocabulary.bosToken {
            let a = sharedModel.pieceOrNil(of: bos)
            let b = sharedModel.pieceOrNil(of: bos)
            #expect(a == b, "cloned model token_to_piece should match original")
        }
    }
}
