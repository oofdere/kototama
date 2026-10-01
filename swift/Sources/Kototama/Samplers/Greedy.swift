//
//  Greedy.swift
//  Kototama
//

/// Greedy (argmax) selection: always take the highest-scoring token.
///
/// A pure token selector — it never touches the logits. Deterministic, which
/// makes it the sampler of choice for snapshot tests and evaluation.
///
/// Mirrors `src/samplers/greedy.rs`.
public struct Greedy: Sampler {
    public init() {}

    /// Identity: greedy selection happens in `sample(_:)` via argmax.
    public func transformInPlace(_ logits: inout [Float]) {}
}
