//
//  SeededRandomNumberGenerator.swift
//  Kototama
//
//  A small, deterministic random number generator for the samplers.
//

/// A reproducible RNG based on SplitMix64.
///
/// **This is a deliberate divergence from the Rust crate.** There, `Dist` and
/// `Xtc` draw from `rand`'s `StdRng`, which is ChaCha12 keyed by a PCG32-based
/// expansion of the seed. Porting ChaCha12 byte-for-byte would buy exact
/// cross-language draw parity at the cost of ~100 lines of bit manipulation
/// that has nothing to do with llama.cpp.
///
/// What is preserved is the *contract* the sampler tests pin: the same seed
/// always produces the same stream, different seeds diverge, and the stream
/// advances on every draw. The exact token a seeded `Dist` picks may differ
/// from the Rust version's — see COMPARISON.md for the trade-off write-up.
struct SeededRandomNumberGenerator {
    /// SplitMix64 state.
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    /// Next 64 random bits.
    mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var mixed = state
        mixed = (mixed ^ (mixed >> 30)) &* 0xBF58_476D_1CE4_E5B9
        mixed = (mixed ^ (mixed >> 27)) &* 0x94D0_49BB_1331_11EB
        return mixed ^ (mixed >> 31)
    }

    /// Uniform float in `[0, 1)`, using the top 24 bits (float precision).
    mutating func nextFloat() -> Float {
        Float(nextUInt64() >> 40) / Float(1 << 24)
    }
}
