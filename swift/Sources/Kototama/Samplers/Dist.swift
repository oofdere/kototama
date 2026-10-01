//
//  Dist.swift
//  Kototama
//

/// Multinomial (weighted-random) token selection from the softmax distribution.
///
/// A pure token selector: the logits are left untouched and the choice happens
/// in `sample(_:)`. Ties and order don't matter — every token is drawn with
/// probability proportional to `exp(logit)`.
///
/// Owns its random number generator, so this is a class: a struct would need
/// `mutating` methods, which the `Sampler` protocol (shared with stateless
/// transforms) does not require. Rust reaches the same shape with `&mut self`.
///
/// Mirrors `src/samplers/dist.rs`.
import Foundation

public final class Dist: Sampler {
    /// The seed llama.cpp uses when it wants a non-deterministic draw
    /// (`LLAMA_DEFAULT_SEED`, 0xFFFFFFFF). Passing it gives a "random-ish"
    /// stream without reaching for the C header.
    public static let defaultSeed: UInt64 = 0xFFFF_FFFF

    /// RNG for the draw. Seeded at construction.
    private var rng: SeededRandomNumberGenerator

    public init(seed: UInt64) {
        self.rng = SeededRandomNumberGenerator(seed: seed)
    }

    /// Identity: selection is the whole point of this sampler.
    public func transformInPlace(_ logits: inout [Float]) {}

    /// Draws a token with probability proportional to `exp(logit)`.
    public func sample(_ logits: [Float]) -> Token {
        // Softmax by hand: exponentiate relative to the maximum (numerical
        // stability), then walk the cumulative weights. Masked tokens
        // (`-infinity`) exponentiate to zero and can never be drawn except
        // for the degenerate `draw == 0` edge, which mirrors the Rust port.
        let maxLogit = logits.max() ?? -.infinity
        let weights = logits.map { exp($0 - maxLogit) }
        let total = weights.reduce(0.0 as Float, +)

        let draw = rng.nextFloat() * total
        var cumulative = 0.0 as Float
        for (index, weight) in weights.enumerated() {
            cumulative += weight
            if cumulative >= draw {
                return Token(Int32(index))
            }
        }
        return Token(Int32(weights.count - 1))
    }
}
