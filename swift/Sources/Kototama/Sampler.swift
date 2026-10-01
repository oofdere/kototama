//
//  Sampler.swift
//  Kototama
//
//  The sampling pipeline: transforms over logits, and how to pick a token.
//

/// A sampling step.
///
/// This protocol mirrors the Rust `Sampler` trait one-for-one. Two kinds of
/// sampler implement it:
///
/// - **Logit transforms** reshape the scores — `Temperature`, `TopK`, `MinP`,
///   and friends. They implement `transformInPlace(_:)` and inherit
///   `sample(_:)`, which applies the transform and picks the highest-scoring
///   remaining token.
///
/// - **Token selectors** choose a token and leave the logits alone —
///   `Greedy` and `Dist`. They implement `sample(_:)` directly and leave
///   `transformInPlace(_:)` as the identity.
///
/// A `Chain` stacks transforms and ends in a selector. That split is exactly
/// how the Rust crate's trait defaults work: implement `apply_mut` and you get
/// argmax for free; override `sample` and you are a selector.
///
/// Like Rust's `&mut self` methods, a sampler is a single-use object: `Dist`
/// and `Xtc` carry RNG state that advances as they are used, so do not share
/// one instance between threads.
public protocol Sampler {
    /// Short, human-readable name (the bare type name).
    ///
    /// Defaults to the type name, which is Swift's counterpart of the Rust
    /// trait's `std::any::type_name` default.
    var name: String { get }

    /// Reshapes logits in place. The only method a logit transform needs.
    ///
    /// Naming note: the Rust trait calls this `apply_mut`. Swift's API design
    /// guidelines express the in-place/returning pair through argument labels,
    /// so the pair here reads `transform(_:)` and `transformInPlace(_:)`.
    func transformInPlace(_ logits: inout [Float])

    /// Returns a transformed copy of `logits`.
    func transform(_ logits: [Float]) -> [Float]

    /// Applies the transform to a copy of `logits` and returns the chosen token.
    func sample(_ logits: [Float]) -> Token

    /// Applies the transform to `logits` in place and returns the chosen token.
    func sampleInPlace(_ logits: inout [Float]) -> Token
}

extension Sampler {
    /// The bare type name, e.g. `"Temperature"` or `"Chain"`.
    public var name: String {
        let full = String(describing: type(of: self))
        return full.split(separator: ".").last.map(String.init) ?? full
    }

    /// Returns a transformed copy of `logits`.
    public func transform(_ logits: [Float]) -> [Float] {
        var scratch = logits
        transformInPlace(&scratch)
        return scratch
    }

    /// Applies the transform to a copy of `logits` and picks the argmax.
    public func sample(_ logits: [Float]) -> Token {
        var scratch = logits
        transformInPlace(&scratch)
        return argmax(scratch)
    }

    /// Applies the transform to `logits` in place and picks the argmax.
    public func sampleInPlace(_ logits: inout [Float]) -> Token {
        transformInPlace(&logits)
        return argmax(logits)
    }
}

/// Picks the index of the largest logit, breaking ties toward the last.
///
/// Identical to the Rust crate's `argmax` — `max_by` there and `>=` here both
/// keep the last of several equal maxima. The snapshot tests depend on this
/// tie-break, so `SamplerTests` pins it explicitly.
public func argmax(_ logits: [Float]) -> Token {
    var bestIndex = 0
    var bestValue = -Float.infinity
    for (index, value) in logits.enumerated() where value >= bestValue {
        bestValue = value
        bestIndex = index
    }
    return Token(Int32(bestIndex))
}
