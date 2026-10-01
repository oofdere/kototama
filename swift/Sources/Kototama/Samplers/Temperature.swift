//
//  Temperature.swift
//  Kototama
//

/// Scales logits by `1 / temperature`.
///
/// Higher temperature flattens the distribution (more surprising tokens);
/// lower temperature sharpens it (more predictable tokens). A temperature of
/// `0.0` is a no-op — use `Greedy` for deterministic decoding instead.
///
/// Mirrors `src/samplers/temperature.rs`.
public struct Temperature: Sampler {
    /// The temperature to divide by. `0.0` leaves logits untouched.
    public var temp: Float

    public init(temp: Float) {
        self.temp = temp
    }

    public func transformInPlace(_ logits: inout [Float]) {
        // Zero temperature means "no scaling", not "infinite sharpness";
        // that is Greedy's job. Pinned by SamplerTests.temperature_zero_is_identity.
        guard temp != 0.0 else { return }

        let inverseTemperature = 1.0 / temp
        for index in logits.indices {
            logits[index] *= inverseTemperature
        }
    }
}
