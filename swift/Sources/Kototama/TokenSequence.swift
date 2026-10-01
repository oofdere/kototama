//
//  TokenSequence.swift
//  Kototama
//
//  One generation in progress: a run of tokens plus its cached logits.
//

import Foundation

/// A handle to one sequence slot inside a `Context`.
///
/// This is the Swift counterpart of the Rust crate's `Sequence`. The shape is
/// the same: a token list and the logits from the last decode, kept locally so
/// `sample(_:)` does not need a second round trip to the model.
///
/// The lifetime rule mirrors Rust's exactly — the slot returns to the pool
/// when the handle dies (`Sequence::drop` there, `deinit` here). Where the
/// Rust version talks to the context actor over messages, this one calls the
/// `Context` methods directly; the context's lock is what makes that safe.
///
/// A `TokenSequence` is deliberately **not** `Sendable`. It carries mutable
/// per-sequence state (`tokens`, cached `logits`), and nothing in the design
/// wants two threads racing on one generation — that is a different sequence's
/// job. Rust makes the same call: `Sequence` is not `Sync`.
public final class TokenSequence {
    /// The context this slot belongs to. Held so the context outlives the slot.
    let context: Context

    /// This sequence's slot ID within the context.
    public let id: SequenceID

    /// The tokens pushed so far, in order.
    ///
    /// The Rust crate exposes this as `Sequence::tokens()` returning `&[i32]`;
    /// here it is a plain array.
    public private(set) var tokens: [Token] = []

    /// Logits produced by the most recent `push(_:)`.
    ///
    /// `nil` until something is decoded, and cleared by any KV-cache edit
    /// that would invalidate it (`pop()`, `remove(_:)`, the `kv*` methods).
    /// This cache is what lets `logits` and `sample(_:)` avoid re-decoding.
    private var cachedLogits: [Float]?

    init(context: Context, id: SequenceID) {
        self.context = context
        self.id = id
    }

    deinit {
        context.releaseSlot(id)
    }

    // MARK: Reading state

    /// The logits from the last `push(_:)`, or `nil` if none is valid.
    public var logits: [Float]? {
        cachedLogits
    }

    /// Is this sequence empty (no tokens pushed)?
    public var isEmpty: Bool {
        tokens.isEmpty
    }

    /// Number of tokens pushed so far.
    public var count: Int {
        tokens.count
    }

    /// The token at `index`, or `nil` if out of range.
    public func token(at index: Int) -> Token? {
        indices.contains(index) ? tokens[index] : nil
    }

    /// Subscript access to the token list. Traps on out-of-range, like
    /// `Sequence`'s `Index` impl in Rust.
    public subscript(index: Int) -> Token {
        tokens[index]
    }

    /// Valid subscript indices, so `sequence[0..<n]` works like Rust's.
    public var indices: Range<Int> { tokens.indices }

    /// Lowest position in the sequence's KV cache (`-1` when empty).
    public var minPosition: Position { context.minPosition(seq: id) }

    /// Highest position in the sequence's KV cache (`-1` when empty).
    public var maxPosition: Position { context.maxPosition(seq: id) }

    // MARK: Editing the sequence

    /// Decodes one token and appends it, refreshing the cached logits.
    ///
    /// This is the workhorse: every generated token goes through here before
    /// the next one can be sampled. The token's position is its index in the
    /// list, exactly as `Sequence::push` computes it.
    public func push(_ token: Token) throws {
        let position = Position(Int32(tokens.count))
        cachedLogits = try context.pushToken(token, at: position, seq: id)
        tokens.append(token)
    }

    /// Appends several tokens in order. Equivalent to calling `push(_:)`
    /// once per token (`Sequence::extend` in Rust).
    public func push(contentsOf newTokens: [Token]) throws {
        for token in newTokens {
            try push(token)
        }
    }

    /// Re-decodes the most recent token to refresh logits without adding one.
    ///
    /// Useful after `pop()` or `remove(_:)`, which leave the cache empty. Does
    /// nothing when the sequence has no tokens. Mirrors `Sequence::decode`.
    public func redecodeLastToken() throws {
        guard let last = tokens.last else { return }
        let position = Position(Int32(tokens.count - 1))
        cachedLogits = try context.pushToken(last, at: position, seq: id)
    }

    /// Removes the last token, dropping its KV cache entry too.
    ///
    /// - Returns: The removed token, or `nil` if the sequence was empty.
    ///   Also `nil` when llama.cpp refuses to drop the cache entry — the same
    ///   contract as `Sequence::pop`.
    @discardableResult
    public func pop() -> Token? {
        guard let lastToken = tokens.last else { return nil }
        let lastPosition = Int32(tokens.count - 1)
        guard removeKVCache(lastPosition..<(lastPosition + 1)) else { return nil }
        tokens.removeLast()
        cachedLogits = nil
        return lastToken
    }

    /// Removes the tokens in `range`, dropping their KV cache entries.
    ///
    /// - Returns: `false` (and changes nothing) when llama.cpp refuses to
    ///   drop the cache range, matching `Sequence::remove`.
    @discardableResult
    public func remove(_ range: Range<Int>) -> Bool {
        guard removeKVCache(Int32(range.lowerBound)..<Int32(range.upperBound)) else {
            return false
        }
        tokens.removeSubrange(range)
        cachedLogits = nil
        return true
    }

    /// Copies the tokens (and KV cache) of `range` into `other`, replacing
    /// whatever `other` held. Mirrors `Sequence::copy_to`.
    public func copy(to other: TokenSequence, range: Range<Int>) {
        copyKVCache(
            to: other,
            from: Int32(range.lowerBound),
            to: Int32(range.upperBound)
        )
        other.tokens = Array(tokens[range])
        other.cachedLogits = nil
    }

    /// The mirror of `copy(to:range:)` — copies `other`'s range onto this
    /// sequence (`Sequence::copy_from`).
    public func copy(from other: TokenSequence, range: Range<Int>) {
        other.copy(to: self, range: range)
    }

    // MARK: KV-cache editing
    //
    // These edit llama.cpp's attention cache directly. They are named `kv...`
    // in the Rust crate; the names are kept here so the two APIs stay easy to
    // diff. Each one invalidates the cached logits, because the next-logit
    // computation depends on the cache that was just edited.

    /// Drops the KV cache for positions `range` (end exclusive).
    @discardableResult
    public func removeKVCache(_ range: Range<Int32>) -> Bool {
        let ok = context.removeKVCache(seq: id, from: range.lowerBound, to: range.upperBound)
        if ok { cachedLogits = nil }
        return ok
    }

    /// Copies the KV cache for positions `range` (end exclusive) into `other`.
    public func copyKVCache(to other: TokenSequence, from start: Int32, to end: Int32) {
        context.copyKVCache(from: id, to: other.id, from: start, to: end)
        other.cachedLogits = nil
    }

    /// Shifts KV-cache positions in `range` (end exclusive) by `delta`.
    public func shiftKVCache(_ range: Range<Int32>, by delta: Int32) {
        context.shiftKVCache(seq: id, from: range.lowerBound, to: range.upperBound, by: delta)
        cachedLogits = nil
    }

    // MARK: Sampling

    /// Picks the next token using `sampler`, from the cached logits.
    ///
    /// - Returns: `nil` when nothing has been decoded yet (no logits to
    ///   sample from), which is `Sequence::sample`'s behavior in Rust.
    public func sample(_ sampler: any Sampler) -> Token? {
        guard let logits = cachedLogits else { return nil }
        return sampler.sample(logits)
    }
}
