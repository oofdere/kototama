//
//  TopP.swift
//  Kototama
//

/// Top-P (nucleus) sampling: keep the smallest set of highest-probability
/// tokens whose cumulative softmax probability reaches `p`, always keeping at
/// least `minKeep`. A `p >= 1.0` is a no-op.
///
/// Mirrors `src/samplers/top_p.rs`.
public struct TopP: Sampler {
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

        // Rank tokens by descending logit.
        let ranked = logits.indices.sorted { logits[$0] > logits[$1] }

        var cumulative = 0.0 as Float
        var keep = [Bool](repeating: false, count: logits.count)
        for (rank, index) in ranked.enumerated() {
            keep[index] = true
            cumulative += probabilities[index]
            if cumulative >= p && rank + 1 >= minKeep {
                break
            }
        }

        for index in logits.indices where !keep[index] {
            logits[index] = -.infinity
        }
    }
}
