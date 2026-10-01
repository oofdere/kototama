//
//  Softmax.swift
//  Kototama
//

/// The shared softmax helper.
///
/// Several samplers need probabilities rather than logits. Computing them
/// relative to the maximum keeps the exponentials from overflowing — the
/// same trick `TopP`, `Typical`, and `Dist` each use inline in the Rust code.
/// Keeping it in one place here is the only structural change from the Rust
/// layout.
///
/// - Returns: One probability per logit, summing to 1. Masked entries
///   (`-infinity`) come out as 0.
import Foundation

func softmax(_ logits: [Float]) -> [Float] {
    guard let maxLogit = logits.max() else { return [] }
    let exponentials = logits.map { exp($0 - maxLogit) }
    let total = exponentials.reduce(0.0 as Float, +)
    return exponentials.map { $0 / total }
}
