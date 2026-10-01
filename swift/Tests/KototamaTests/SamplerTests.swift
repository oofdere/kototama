//
//  SamplerTests.swift
//  KototamaTests
//
//  Port of tests/sampler.rs. Pure math: no model required.
//

import Testing
@testable import Kototama

@Suite("samplers")
struct SamplerTests {
    // MARK: Greedy

    @Test func greedy_picks_argmax() {
        var sampler = Greedy()
        #expect(sampler.sample([0.1, 0.9, 0.5, 0.2]) == 1)
    }

    @Test func greedy_tiebreak_is_last() {
        // Rust's `max_by` returns the last maximum on ties — pin the behavior.
        var sampler = Greedy()
        #expect(sampler.sample([5.0, 5.0, 5.0]) == 2)
    }

    // MARK: Temperature

    @Test func temperature_zero_is_identity() {
        let sampler = Temperature(temp: 0.0)
        let logits: [Float] = [1.0, 3.0, 2.0]
        #expect(sampler.transform(logits) == logits)
    }

    @Test func temperature_scales_logits() {
        let sampler = Temperature(temp: 2.0)
        let out = sampler.transform([1.0, 2.0, 4.0])
        #expect(abs(out[0] - 0.5) < 1e-6)
        #expect(abs(out[1] - 1.0) < 1e-6)
        #expect(abs(out[2] - 2.0) < 1e-6)
    }

    @Test func temperature_preserves_argmax() {
        let sampler = Temperature(temp: 0.8)
        #expect(sampler.sample([0.1, 0.9, 0.5]) == 1)
    }

    // MARK: MinP

    @Test func min_p_masks_below_threshold() {
        let sampler = MinP(p: 0.5, minKeep: 0)
        let out = sampler.transform([4.0, 3.5, 3.0, 2.0])
        #expect(out[0].isFinite, "max survives")
        #expect(out[1].isFinite, "3.5 >= thresh survives")
        #expect(out[2].isInfinite && out[2].sign == .minus, "3.0 masked")
        #expect(out[3].isInfinite && out[3].sign == .minus, "2.0 masked")
    }

    @Test func min_p_min_keep_floor_forces_survivors() {
        // p=0.01 alone would keep only the max; minKeep forces 4.
        let sampler = MinP(p: 0.01, minKeep: 4)
        let out = sampler.transform([10.0, 5.0, 4.0, 1.0, 0.0])
        #expect(finiteCount(out) == 4)
    }

    @Test func min_p_one_keeps_only_the_max() {
        let sampler = MinP(p: 1.0, minKeep: 0)
        let out = sampler.transform([1.0, 3.0, 3.0, 2.0])
        #expect(out[1].isFinite && out[2].isFinite, "max(es) survive")
        #expect(out[0].isInfinite && out[0].sign == .minus)
        #expect(out[3].isInfinite && out[3].sign == .minus)
    }

    // MARK: Dist

    @Test func dist_returns_valid_token() {
        let sampler = Dist(seed: 42)
        let token = sampler.sample([0.1, 0.5, 0.3, 0.2])
        #expect((0..<4).contains(token.rawValue))
    }

    @Test func dist_is_deterministic_for_same_seed() {
        let logits: [Float] = [1.0, 2.0, 0.5, 3.0, 1.5]
        let a = Dist(seed: 99)
        let b = Dist(seed: 99)
        #expect(a.sample(logits) == b.sample(logits))
    }

    @Test func dist_advances_state_across_calls() {
        let logits: [Float] = [1.0, 2.0, 0.5, 3.0, 1.5]
        let sampler = Dist(seed: 7)
        let distinct = Set((0..<10).map { _ in sampler.sample(logits).rawValue })
        #expect(distinct.count > 1, "RNG should advance, producing varied draws")
    }

    @Test func dist_never_picks_masked_tokens() {
        let sampler = Dist(seed: 1)
        let logits: [Float] = [1.0, -.infinity, -.infinity, 0.5]
        for _ in 0..<20 {
            let token = sampler.sample(logits).rawValue
            #expect(token == 0 || token == 3, "masked token \(token) picked")
        }
    }

    // MARK: name()

    @Test func name_returns_short_type_name() {
        #expect(Temperature(temp: 1.0).name == "Temperature")
        #expect(MinP(p: 0.1, minKeep: 1).name == "MinP")
        #expect(Greedy().name == "Greedy")
        #expect(Dist(seed: 0).name == "Dist")
        #expect(TopK(k: 1).name == "TopK")
        #expect(TopP(p: 0.9, minKeep: 1).name == "TopP")
        #expect(Typical(p: 0.9, minKeep: 1).name == "Typical")
        #expect(TopNSigma(n: 1.0).name == "TopNSigma")
        #expect(Xtc(probability: 0.1, threshold: 0.1, minKeep: 1, seed: 1).name == "Xtc")
        #expect(Chain().name == "Chain")
    }

    // MARK: TopK

    @Test func top_k_masks_all_but_k() {
        let sampler = TopK(k: 2)
        let out = sampler.transform([4.0, 3.0, 2.0, 1.0, 0.0])
        #expect(finiteCount(out) == 2, "only top-2 survive")
        #expect(out[0].isFinite && out[1].isFinite, "top two survive")
        #expect(out[3].isInfinite && out[3].sign == .minus)
    }

    @Test func top_k_keeps_ties_at_threshold() {
        // 5.0 appears twice; k=2 keeps both ties (>= threshold).
        let sampler = TopK(k: 2)
        let out = sampler.transform([5.0, 4.0, 5.0, 1.0])
        #expect(finiteCount(out) == 2)
        #expect(out[0].isFinite && out[2].isFinite)
    }

    @Test func top_k_non_positive_is_noop() {
        let sampler = TopK(k: 0)
        let out = sampler.transform([4.0, 3.0, 2.0])
        #expect(finiteCount(out) == 3)
    }

    @Test func top_k_larger_than_vocab_is_noop() {
        let sampler = TopK(k: 100)
        let out = sampler.transform([4.0, 3.0, 2.0])
        #expect(finiteCount(out) == 3)
    }

    // MARK: TopP

    @Test func top_p_one_is_noop() {
        let sampler = TopP(p: 1.0, minKeep: 1)
        let out = sampler.transform([1.0, 2.0, 3.0, 4.0])
        #expect(finiteCount(out) == 4)
    }

    @Test func top_p_keeps_nucleus() {
        // Heavily peaked: the argmax alone exceeds p=0.9.
        let sampler = TopP(p: 0.9, minKeep: 1)
        let out = sampler.transform([10.0, 1.0, 1.0, 1.0])
        #expect(finiteCount(out) == 1, "only the max clears p=0.9")
        #expect(out[0].isFinite)
    }

    @Test func top_p_respects_min_keep() {
        // p would keep only 1, but minKeep forces at least 3.
        let sampler = TopP(p: 0.1, minKeep: 3)
        let out = sampler.transform([10.0, 1.0, 1.0, 1.0])
        #expect(finiteCount(out) >= 3, "minKeep floor honored")
    }

    // MARK: Typical

    @Test func typical_one_is_noop() {
        let sampler = Typical(p: 1.0, minKeep: 1)
        let out = sampler.transform([1.0, 2.0, 3.0, 4.0])
        #expect(finiteCount(out) == 4)
    }

    @Test func typical_masks_some_tokens() {
        let sampler = Typical(p: 0.5, minKeep: 1)
        let out = sampler.transform([0.5, 1.0, 0.2, 3.0, 0.1])
        #expect(finiteCount(out) < 5, "typical sampling should mask at least one token")
        #expect(finiteCount(out) >= 1, "at least one token survives")
    }

    // MARK: TopNSigma

    @Test func top_n_sigma_non_positive_is_noop() {
        let sampler = TopNSigma(n: 0.0)
        let out = sampler.transform([1.0, 2.0, 3.0, 4.0])
        #expect(finiteCount(out) == 4)
    }

    @Test func top_n_sigma_masks_outliers() {
        // A tight cluster plus a far-below outlier.
        let sampler = TopNSigma(n: 1.0)
        let out = sampler.transform([10.0, 9.9, 10.1, -50.0])
        #expect(out[3].isInfinite && out[3].sign == .minus, "outlier far below the cluster is masked")
        #expect(out[0].isFinite && out[1].isFinite && out[2].isFinite)
    }

    @Test func top_n_sigma_respects_already_masked() {
        // -inf entries must not corrupt the mean/std: the result for the
        // finite entries should match a run that omits the -inf slot entirely.
        let sampler = TopNSigma(n: 1.0)
        let withInf = sampler.transform([10.0, 9.9, 10.1, -.infinity])
        let without = sampler.transform([10.0, 9.9, 10.1])
        #expect(withInf[0] == without[0])
        #expect(withInf[1] == without[1])
        #expect(withInf[2] == without[2])
        #expect(withInf[3].isInfinite && withInf[3].sign == .minus)
    }

    // MARK: Xtc

    @Test func xtc_inert_when_threshold_too_high() {
        // threshold > 0.5 disables the sampler.
        let sampler = Xtc(probability: 1.0, threshold: 0.9, minKeep: 1, seed: 1)
        let out = sampler.transform([10.0, 9.0, 8.0, 7.0])
        #expect(finiteCount(out) == 4)
    }

    @Test func xtc_can_mask_top_token() {
        // probability 1.0 (always triggers), threshold 0.1: the top two tokens
        // clear the threshold and are masked (XTC removes the top choices), the
        // remaining tokens survive.
        let sampler = Xtc(probability: 1.0, threshold: 0.1, minKeep: 1, seed: 1)
        let out = sampler.transform([2.0, 2.0, 1.0, 0.0])
        #expect(out[0].isInfinite && out[0].sign == .minus, "top token masked")
        #expect(out[1].isInfinite && out[1].sign == .minus, "second token masked")
        #expect(out[2].isFinite, "third token survives")
    }

    @Test func xtc_probability_zero_is_noop() {
        let sampler = Xtc(probability: 0.0, threshold: 0.1, minKeep: 1, seed: 1)
        let out = sampler.transform([10.0, 9.0, 8.0, 7.0])
        #expect(finiteCount(out) == 4)
    }

    // MARK: Chain

    @Test func chain_applies_transforms_in_order() {
        // top_k(1) keeps only the max, so temperature is irrelevant -> argmax.
        let chain = Chain()
            .pushing(TopK(k: 1))
            .pushing(Temperature(temp: 0.8))
            .pushing(Greedy())
        let token = chain.sample([1.0, 5.0, 2.0, 3.0])
        #expect(token == 1, "argmax survives the chain")
    }

    @Test func chain_dist_returns_valid_token() {
        let chain = Chain()
            .pushing(TopP(p: 0.9, minKeep: 1))
            .pushing(Temperature(temp: 0.8))
            .pushing(Dist(seed: 7))
        let token = chain.sample([1.0, 2.0, 0.5, 3.0])
        #expect((0..<4).contains(token.rawValue))
    }

    @Test func chain_apply_mut_composes() {
        let chain = Chain().pushing(TopK(k: 2))
        var logits: [Float] = [4.0, 3.0, 2.0, 1.0]
        chain.transformInPlace(&logits)
        #expect(finiteCount(logits) == 2)
    }

    @Test func chain_empty_is_greedy() {
        let chain = Chain()
        let token = chain.sample([0.1, 0.9, 0.5])
        #expect(token == 1)
    }
}
