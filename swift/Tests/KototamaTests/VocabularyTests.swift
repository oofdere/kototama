//
//  VocabularyTests.swift
//  KototamaTests
//
//  Port of tests/vocab.rs.
//

import Testing
@testable import Kototama

@Suite("vocabulary")
struct VocabularyTests {
    private var vocab: Vocabulary { sharedModel.vocabulary }

    @Test func bos_token_some() {
        #expect(vocab.bosToken != nil, "TinyStories should have a BOS token")
    }

    @Test func eos_token_some() {
        #expect(vocab.eosToken != nil, "TinyStories should have an EOS token")
    }

    @Test func bos_and_eos_are_valid_tokens() {
        if let bos = vocab.bosToken {
            #expect(bos.rawValue >= 0 && bos.rawValue < vocab.count)
        }
        if let eos = vocab.eosToken {
            #expect(eos.rawValue >= 0 && eos.rawValue < vocab.count)
        }
    }

    @Test func n_tokens_matches_vocab_size() {
        #expect(vocab.count > 0)
    }

    @Test func vocab_type_is_not_none() {
        #expect(vocab.type != .none, "vocab type should not be NONE")
    }

    @Test func get_add_bos() {
        // Just verify the call doesn't crash; value depends on model config.
        _ = vocab.addsBOS
    }

    @Test func get_score_bos() {
        if let bos = vocab.bosToken {
            #expect(vocab.score(of: bos).isFinite)
        }
    }

    @Test func get_text_bos_nonempty() {
        if let bos = vocab.bosToken {
            #expect(!vocab.text(of: bos).isEmpty)
        }
    }

    @Test func is_eog_eos() {
        if let eos = vocab.eosToken {
            #expect(vocab.isEndOfGeneration(eos), "EOS token should be end-of-generation")
        }
    }

    @Test func is_not_eog_regular_token() {
        // BOS is control but not always EOG — just verify the call doesn't crash.
        if let bos = vocab.bosToken {
            _ = vocab.isEndOfGeneration(bos)
        }
    }

    @Test func nl_token_optional() {
        // May or may not be present; just verify no crash.
        _ = vocab.newlineToken
    }
}
