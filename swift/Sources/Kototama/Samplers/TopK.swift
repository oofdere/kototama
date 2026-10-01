//
//  TopK.swift
//  Kototama
//

/// Top-K sampling: keep only the `k` highest logits and mask the rest to
/// `-infinity`.
///
/// Tokens tied at the threshold are kept too, so the survivor set can hold
/// more than `k` entries. A non-positive `k` leaves the logits untouched.
///
/// Mirrors `src/samplers/top_k.rs`.
public struct TopK: Sampler {
    /// How many highest logits to keep. Non-positive disables the sampler.
    public var k: Int32

    public init(k: Int32) {
        self.k = k
    }

    public func transformInPlace(_ logits: inout [Float]) {
        guard k > 0 else { return }

        let survivorCount = min(Int(k), logits.count)
        // Asking for every token (or more) keeps everything.
        guard survivorCount < logits.count else { return }

        // Find the value of the k-th largest logit, then keep everything
        // greater than or equal to it. Rust reaches for
        // `select_nth_unstable_by` (linear time); Swift has no quickselect in
        // the standard library, so we sort a copy instead. Same threshold,
        // same tie behavior, slightly worse complexity — samplers run once per
        // generated token over a vocabulary-sized slice, where this is noise
        // next to the decode itself.
        let threshold = logits.sorted(by: >)[survivorCount - 1]

        for index in logits.indices where logits[index] < threshold {
            logits[index] = -.infinity
        }
    }
}
