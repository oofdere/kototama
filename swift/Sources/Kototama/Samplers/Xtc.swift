//
//  Xtc.swift
//  Kototama
//

/// XTC (eXclude Top Choices): with probability `probability`, drop the tokens
/// whose softmax probability is at least `threshold` (keeping the rest).
///
/// This is a diversity-enhancing sampler: it only acts when at least two
/// tokens clear the threshold and at least `minKeep` would survive. As in
/// llama.cpp, it is inert when `probability <= 0` or `threshold > 0.5`.
///
/// Unlike the stateless transforms, XTC rolls dice on every call, so it owns
/// its random number generator and is a class (see `Dist` for why).
///
/// Mirrors `src/samplers/xtc.rs`.
public final class Xtc: Sampler {
    /// Chance the sampler acts at all, in `[0, 1]`. `<= 0` disables it.
    public var probability: Float
    /// Probability floor for a token to count as a "top choice", at most `0.5`.
    public var threshold: Float
    /// Never leave fewer than this many candidates.
    public var minKeep: Int

    /// RNG for the trigger roll. Seeded at construction.
    private var rng: SeededRandomNumberGenerator

    public init(probability: Float, threshold: Float, minKeep: Int, seed: UInt64) {
        self.probability = probability
        self.threshold = threshold
        self.minKeep = minKeep
        self.rng = SeededRandomNumberGenerator(seed: seed)
    }

    public func transformInPlace(_ logits: inout [Float]) {
        guard probability > 0.0, threshold <= 0.5, logits.count >= 2 else { return }

        // Roll to decide whether to act at all.
        if rng.nextFloat() > probability {
            return
        }

        let probabilities = softmax(logits)

        // Tokens ranked by descending logit.
        let ranked = logits.indices.sorted { logits[$0] > logits[$1] }

        // Rank of the lowest token that clears the threshold (it must be part
        // of the leading run — the scan stops at the first miss).
        var lastTopChoice = 0
        for (rank, index) in ranked.enumerated() {
            if probabilities[index] >= threshold {
                lastTopChoice = rank
            } else {
                break
            }
        }

        // Drop the top choices, provided at least one goes and enough stay.
        //
        // `lastTopChoice` is the *rank* of the lowest top choice — the same
        // `pos_last` accounting as both the Rust crate and upstream
        // llama.cpp's `llama_sample_xtc_apply`. Masking `prefix(lastTopChoice)`
        // therefore always leaves the last qualifying token alive:
        // `xtc_can_mask_top_token` pins that on [2, 2, 1, 0], where three
        // tokens clear the threshold but exactly the top two are masked.
        if lastTopChoice > 0 && logits.count - lastTopChoice >= minKeep {
            for index in ranked.prefix(lastTopChoice) {
                logits[index] = -.infinity
            }
        }
    }
}
