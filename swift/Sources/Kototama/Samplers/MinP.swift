//
//  MinP.swift
//  Kototama
//

/// Min-P sampling: mask every logit more than `p` (in probability space) below
/// the maximum. Keeps at least `minKeep` candidates.
///
/// Compared with `TopP`, which keeps a *cumulative* mass, Min-P keeps every
/// token whose probability is at least `p` times the most likely token's.
///
/// Mirrors `src/samplers/min_p.rs`.
import Foundation

public struct MinP: Sampler {
    /// Relative probability floor, in `(0, 1]`: the smallest probability,
    /// as a fraction of the top token's, that survives.
    public var p: Float
    /// Never keep fewer than this many candidates.
    public var minKeep: Int

    public init(p: Float, minKeep: Int) {
        self.p = p
        self.minKeep = minKeep
    }

    public func transformInPlace(_ logits: inout [Float]) {
        guard logits.count > minKeep else { return }

        // ln(p) converts the probability floor into logit space:
        // keep logits within ln(p) of the maximum.
        guard let maxLogit = logits.max() else { return }
        var threshold = maxLogit + log(p)

        // The floor forces the `minKeep` best logits to survive even when the
        // threshold would have cut them.
        if minKeep > 0 {
            let floor = logits.sorted(by: >)[minKeep - 1]
            threshold = min(threshold, floor)
        }

        for index in logits.indices where logits[index] < threshold {
            logits[index] = -.infinity
        }
    }
}
