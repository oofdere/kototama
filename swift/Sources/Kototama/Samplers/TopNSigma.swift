//
//  TopNSigma.swift
//  Kototama
//

/// Top-N-Sigma sampling: mask every logit more than `n` standard deviations
/// below the maximum (considering only non-masked logits). A non-positive `n`
/// is a no-op.
///
/// Reference: "Simplifying Top-p Sampling with Top-nσ".
///
/// Mirrors `src/samplers/top_n_sigma.rs`.
public struct TopNSigma: Sampler {
    /// Width of the band below the maximum, in standard deviations.
    public var n: Float

    public init(n: Float) {
        self.n = n
    }

    public func transformInPlace(_ logits: inout [Float]) {
        guard n > 0.0, logits.count > 1 else { return }

        // Statistics over the finite logits only: previously masked entries
        // must not drag the mean and standard deviation around.
        let finite = logits.filter { $0.isFinite }
        guard !finite.isEmpty else { return }

        let maximum = finite.max()!
        let mean = finite.reduce(0.0 as Float, +) / Float(finite.count)
        let variance = finite.reduce(0.0 as Float) { total, value in
            total + (value - mean) * (value - mean)
        } / Float(finite.count)
        let standardDeviation = variance.squareRoot()

        let threshold = maximum - n * standardDeviation
        for index in logits.indices where logits[index] < threshold {
            logits[index] = -.infinity
        }
    }
}
