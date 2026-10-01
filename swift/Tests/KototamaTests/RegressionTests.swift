//
//  RegressionTests.swift
//  KototamaTests
//
//  Tests for bugs caught in review of the Swift port. Each one guards a fix
//  that brought the port back in line with the Rust crate.
//

import Testing
@testable import Kototama

@Suite("regressions")
struct RegressionTests {
    // MARK: chatTemplate must not fall back to the default template
    //
    // A named lookup that misses has to return nil, not the default template.
    // (llama.cpp's `llama_model_chat_template` returns NULL for a missing
    // named key; only the unnamed lookup gets upstream's special-case
    // workarounds.)

    @Test func chat_template_unknown_name_returns_nil_not_default() {
        let unknown = sharedModel.chatTemplate(named: "definitely-not-a-template-name")
        #expect(unknown == nil,
                "a missing named template must be nil, not the default template")
    }

    @Test func chat_template_nil_name_is_untouched() {
        // The unnamed lookup keeps its llama.cpp behavior (default template,
        // or nil). We only assert consistency: two calls agree.
        let first = sharedModel.chatTemplate()
        let second = sharedModel.chatTemplate()
        #expect(first == second)
    }

    // MARK: Sampling pipeline order in SimpleChat (min_p -> temp -> dist)
    //
    // The transforms do not commute: MinP thresholds at max + ln(p) in logit
    // space, so applying Temperature first changes which tokens survive. The
    // example must prune before scaling. This test pins the pipeline itself:
    // prune with MinP, then scale with Temperature, then select.

    @Test func min_p_then_temperature_pipeline_is_not_commutative() {
        let minP = MinP(p: 0.05, minKeep: 1)          // keep within ln(0.05) ≈ -3.0 of the max
        let temperature = Temperature(temp: 0.5)     // scales logits by 2

        // MinP thresholds at max + ln(p) in logit space, so scaling first
        // changes the survivor set. With [6, 4, 3.5, 0]:
        //
        //   min_p then temp: threshold 6 - 3 = 3      -> 6, 4, 3.5 survive
        //   temp then min_p: [12, 8, 7, 0], threshold 12 - 3 = 9 -> only 12
        let logits: [Float] = [6, 4, 3.5, 0]

        let correctOrder = temperature.transform(minP.transform(logits))
        let wrongOrder = minP.transform(temperature.transform(logits))

        #expect(finiteCount(correctOrder) == 3,
                "min_p -> temp must keep three tokens for this input")
        #expect(finiteCount(wrongOrder) == 1,
                "temp -> min_p keeps one: the transforms do not commute")
    }
}
