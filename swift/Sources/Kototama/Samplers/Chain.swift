//
//  Chain.swift
//  Kototama
//

/// A pipeline of samplers applied in order.
///
/// Logit-transforming samplers (e.g. `TopK`, `Temperature`) run first; the
/// final sampler is expected to be a token selector such as `Greedy` or
/// `Dist`, which makes the actual choice from the transformed logits.
///
/// ```
/// let chain = Chain()
///     .pushing(TopK(k: 40))
///     .pushing(Temperature(temp: 0.8))
///     .pushing(Dist(seed: 42))
/// let token = chain.sample(logits)
/// ```
///
/// Mirrors `src/samplers/chain.rs`.
public final class Chain: Sampler {
    /// The stages, applied front to back.
    private var stages: [any Sampler]

    public init() {
        self.stages = []
    }

    /// Appends a sampler and returns `self`, for builder-style setup.
    /// (Rust calls this `with`; `pushing` reads as the Swift verb for it.)
    @discardableResult
    public func pushing(_ sampler: some Sampler) -> Chain {
        stages.append(sampler)
        return self
    }

    /// Appends a sampler in place. (Rust's `push`.)
    public func push(_ sampler: some Sampler) {
        stages.append(sampler)
    }

    /// How many stages the chain holds.
    public var count: Int { stages.count }

    /// Is this chain empty (it will behave like `Greedy`)?
    public var isEmpty: Bool { stages.isEmpty }

    /// Runs every stage's transform, front to back.
    public func transformInPlace(_ logits: inout [Float]) {
        for stage in stages {
            stage.transformInPlace(&logits)
        }
    }

    /// Applies every stage in order, then lets the last one choose the token.
    ///
    /// The last stage is usually a selector (`Greedy`, `Dist`) whose own
    /// transform is a no-op. With no stages at all this degenerates to plain
    /// argmax, matching `Chain::sample` in Rust and pinned by
    /// `SamplerTests.chain_empty_is_greedy`.
    public func sample(_ logits: [Float]) -> Token {
        var scratch = logits
        for stage in stages {
            stage.transformInPlace(&scratch)
        }
        guard let last = stages.last else {
            return argmax(scratch)
        }
        return last.sample(scratch)
    }
}
