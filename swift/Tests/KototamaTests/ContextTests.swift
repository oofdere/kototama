//
//  ContextTests.swift
//  KototamaTests
//
//  Port of tests/context.rs: sequence slot checkout and shared-state
//  semantics. In Rust these exercise the context actor's slot table; here
//  they exercise the same table behind `Context`'s lock.
//

import Testing
@testable import Kototama

@Suite("context")
struct ContextTests {
    @Test func context_new_ok() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        _ = context
    }

    @Test func n_ctx_at_least_params() throws {
        let params = testContextParams()
        let context = try Context(model: sharedModel, params: params)
        #expect(context.contextSize >= params.nContext,
                "context size should be at least the requested size")
    }

    @Test func free_slots_starts_full() throws {
        let params = testContextParams()
        let context = try Context(model: sharedModel, params: params)
        #expect(context.freeSlots == Int(params.nSequencesMax))
    }

    @Test func sequence_checkout_reduces_free_slots() throws {
        let params = testContextParams()
        let context = try Context(model: sharedModel, params: params)
        let total = context.freeSlots
        let seq = try #require(context.checkoutSequence())
        #expect(context.freeSlots == total - 1)
        _ = seq
    }

    @Test func sequence_drop_returns_slot() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let before = context.freeSlots
        do {
            let seq = try #require(context.checkoutSequence())
            #expect(context.freeSlots == before - 1)
            _ = seq
        }
        #expect(context.freeSlots == before)
    }

    @Test func all_slots_exhausted() throws {
        let params = testContextParams()
        let context = try Context(model: sharedModel, params: params)
        let n = Int(params.nSequencesMax)
        var seqs: [TokenSequence] = []
        for _ in 0..<n {
            seqs.append(try #require(context.checkoutSequence()))
        }
        #expect(context.freeSlots == 0)
        #expect(context.checkoutSequence() == nil,
                "should return nil when all slots are taken")
        seqs.removeAll()
        #expect(context.freeSlots == n)
    }

    @Test func can_shift_does_not_crash() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        _ = context.canShift
    }

    @Test func perf_does_not_crash() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        _ = context.performance
    }

    // MARK: Context is a shared reference (clone shares state)

    @Test func context_clone_shares_free_slots() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let contextClone = context
        #expect(context.freeSlots == contextClone.freeSlots)
    }

    @Test func context_clone_slot_checkout_visible_on_original() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let contextClone = context
        let total = context.freeSlots

        let seq = try #require(contextClone.checkoutSequence())
        #expect(context.freeSlots == total - 1)
        _ = seq
    }

    @Test func context_clone_slot_checkout_visible_on_clone() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let contextClone = context
        let total = context.freeSlots

        let seq = try #require(context.checkoutSequence())
        #expect(contextClone.freeSlots == total - 1)
        _ = seq
    }

    @Test func context_clone_sequence_drop_restores_on_both() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let contextClone = context
        let total = context.freeSlots

        do {
            let seq = try #require(context.checkoutSequence())
            #expect(contextClone.freeSlots == total - 1)
            _ = seq
        }
        #expect(context.freeSlots == total)
        #expect(contextClone.freeSlots == total)
    }

    @Test func context_clone_n_ctx_matches() throws {
        let context = try Context(model: sharedModel, params: testContextParams())
        let contextClone = context
        #expect(context.contextSize == contextClone.contextSize)
    }

    // MARK: Context from a shared Model

    @Test func context_from_cloned_model() throws {
        let modelClone = sharedModel
        let context = try Context(model: modelClone, params: testContextParams())
        #expect(context.freeSlots > 0)
        #expect(context.contextSize >= testContextParams().nContext)
    }

    @Test func context_from_cloned_model_is_independent() throws {
        let modelClone = sharedModel
        let params = testContextParams()

        let context1 = try Context(model: sharedModel, params: params)
        let context2 = try Context(model: modelClone, params: params)

        let seq1 = try #require(context1.checkoutSequence())
        #expect(context2.freeSlots == Int(params.nSequencesMax),
                "second context should have full slots independent of first")
        _ = seq1
    }
}
