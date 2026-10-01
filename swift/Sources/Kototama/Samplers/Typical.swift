//
//  Typical.swift
//  Kototama
//

/// Locally typical sampling: keep tokens whose negative-log-probability is
/// closest to the distribution entropy, accumulating until the cumulative
/// probability exceeds `p` (keeping at least `minKeep`). A `p >= 1.0` is a
/// no-op.
///
/// Reference: Meister et al., "Typical Decoding for Natural Language Generation".
///
/// Mirrors `src/samplers/typical.rs`.
import Foundation

public struct Typical: Sampler {
    /// Cumulative probability mass to retain, in `(0, 1)`.
    public var p: Float
    /// Never keep fewer than this many candidates.
    public var minKeep: Int

    public init(p: Float, minKeep: Int) {
        self.p = p
        self.minKeep = minKeep
    }

    public func transformInPlace(_ logits: inout [Float]) {
        guard p < 1.0 else { return }
        guard !logits.isEmpty else { return }

        let probabilities = softmax(logits)

        // Entropy H = -sum(p * ln p) over nonzero probabilities.
        let entropy = probabilities.reduce(into: 0.0 as Float) { total, probability in
            if probability > 0 {
                total += -probability * log(probability)
            }
        }

        // Rank by |(-ln p) - H| ascending: the most "typical" tokens first.
        // Ties break toward lower indices, like the Rust port's stable sort.
        let ranked = logits.indices.sorted { lhs, rhs in
            let lhsScore = abs(-log(probabilities[lhs]) - entropy)
            let rhsScore = abs(-log(probabilities[rhs]) - entropy)
            return lhsScore < rhsScore
        }

        var cumulative = 0.0 as Float
        var keep = [Bool](repeating: false, count: logits.count)
        for (rank, index) in ranked.enumerated() {
            keep[index] = true
            cumulative += probabilities[index]
            if cumulative > p && (minKeep == 0 || rank + 1 >= minKeep) {
                break
            }
        }

        for index in logits.indices where !keep[index] {
            logits[index] = -.infinity
        }
    }
}
