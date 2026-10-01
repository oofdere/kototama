//
//  Context.swift
//  Kototama
//
//  The decoding engine: one llama.cpp context, plus the sequence slots
//  checked out of it.
//

import CLlama
import Synchronization

/// Parameters for creating a context.
///
/// Mirrors the Rust crate's `ContextParams`, which wraps `llama_context_params`
/// and lets callers mutate any C field through `Deref`. Swift has no `Deref`,
/// so the five fields callers actually reach for are named here and every
/// other llama.cpp knob (threads, RoPE, flash attention, ...) keeps the value
/// `llama_context_default_params()` gives it.
public struct ContextParams: Sendable {
    /// Text context size. `0` means "whatever the model was trained for".
    public var nContext: UInt32
    /// Largest batch `llama_decode` accepts per call. The port feeds one token
    /// per call, so this mainly bounds prompt ingestion.
    public var nBatch: UInt32
    /// How many independent sequences (generation slots) the context serves.
    /// This is the size of the checkout pool `Context.checkoutSequence()` deals out.
    public var nSequencesMax: UInt32
    /// Skip llama.cpp's performance counters. The C name is a double negative
    /// (`no_perf`), kept verbatim so the mapping to the C struct stays obvious:
    /// set it to `false` to enable timing, as the Rust examples do.
    public var noPerf: Bool
    /// Share one KV buffer across sequences. Helps when sequences share long
    /// prefixes; hurts when they don't (see llama.cpp's notes on `kv_unified`).
    public var kvUnified: Bool

    /// Builds parameters from llama.cpp's own defaults.
    public init() {
        let defaults = llama_context_default_params()
        self.nContext = defaults.n_ctx
        self.nBatch = defaults.n_batch
        self.nSequencesMax = defaults.n_seq_max
        self.noPerf = defaults.no_perf
        self.kvUnified = defaults.kv_unified
    }

    /// Builds parameters field by field. Fields not listed here keep the
    /// llama.cpp defaults.
    public init(
        nContext: UInt32,
        nBatch: UInt32,
        nSequencesMax: UInt32,
        noPerf: Bool,
        kvUnified: Bool
    ) {
        self.nContext = nContext
        self.nBatch = nBatch
        self.nSequencesMax = nSequencesMax
        self.noPerf = noPerf
        self.kvUnified = kvUnified
    }

    /// The C parameter struct to hand to `llama_init_from_model`.
    func makeCParams() -> llama_context_params {
        var p = llama_context_default_params()
        p.n_ctx = nContext
        p.n_batch = nBatch
        p.n_seq_max = nSequencesMax
        p.no_perf = noPerf
        p.kv_unified = kvUnified
        return p
    }
}

/// A snapshot of llama.cpp's performance counters for one context.
///
/// The Rust crate returns the raw `llama_perf_context_data` bindgen struct
/// from `Context::perf()`; here the fields get names a human can read.
public struct ContextPerformance: Sendable, Hashable {
    /// Absolute time the context was created, in milliseconds.
    public let startMilliseconds: Double
    /// Time spent loading the model, in milliseconds.
    public let loadMilliseconds: Double
    /// Time spent processing prompt tokens, in milliseconds.
    public let promptEvalMilliseconds: Double
    /// Time spent generating tokens, in milliseconds.
    public let evalMilliseconds: Double
    /// Number of prompt tokens processed.
    public let promptTokensEvaluated: Int32
    /// Number of tokens generated.
    public let tokensGenerated: Int32
    /// How many compute graphs were reused instead of rebuilt.
    public let graphsReused: Int32

    init(_ data: llama_perf_context_data) {
        self.startMilliseconds = data.t_start_ms
        self.loadMilliseconds = data.t_load_ms
        self.promptEvalMilliseconds = data.t_p_eval_ms
        self.evalMilliseconds = data.t_eval_ms
        self.promptTokensEvaluated = data.n_p_eval
        self.tokensGenerated = data.n_eval
        self.graphsReused = data.n_reused
    }
}

/// A llama.cpp context: the compute state that turns tokens into logits.
///
/// **Thread safety.** The Rust crate refuses to let anyone touch the
/// `llama_context` pointer directly — every operation is a message to a
/// dedicated actor thread (`ContextActor`, via the `#[protocol]` macro).
/// llama.cpp contexts are not thread-safe, so *someone* has to serialize
/// access; Rust picks an actor.
///
/// The Swift equivalent guarantee is much smaller: a lock. Every entry point
/// below takes `state.withLock { ... }`, so the pointer is only ever reached
/// by one caller at a time — the same "one mutation at a time" invariant the
/// actor enforces, without a thread hop per token. Callers keep a synchronous
/// API (`sequence.push(token)` blocks until the decode is done), which is
/// what a generation loop wants anyway.
///
/// Sharing one `Context` across threads is safe; sharing one `TokenSequence`
/// is not (it carries per-sequence state), so `TokenSequence` is deliberately
/// not `Sendable`.
public final class Context: @unchecked Sendable {
    /// The model this context was built from. Held so the model outlives the
    /// C context that points into it.
    public let model: Model

    /// The parameters this context was created with.
    public let params: ContextParams

    /// Everything that must stay consistent under one lock: the C pointer,
    /// the shared one-token batch, and the slot checkout table.
    ///
    /// (`Mutex` is `~Copyable`, which is fine in a `let` property: one owner,
    /// no copies, and every access funnels through `withLock`.)
    private let state: Mutex<ContextState>

    /// Creates a context from a loaded model.
    ///
    /// - Throws: `KototamaError.contextCreationFailed` if llama.cpp refuses.
    public init(model: Model, params: ContextParams) throws {
        var cParams = params.makeCParams()
        guard let handle = llama_init_from_model(model.handle, cParams) else {
            throw KototamaError.contextCreationFailed
        }
        self.model = model
        self.params = params
        self.state = Mutex(
            ContextState(
                handle: handle,
                batch: TokenBatch(capacity: 1, maxSequences: Int32(params.nSequencesMax)),
                nVocabulary: model.vocabulary.count,
                checkedOut: [Bool](repeating: false, count: Int(params.nSequencesMax))
            )
        )
    }

    deinit {
        state.withLock { llama_free($0.handle) }
    }

    // MARK: Sequence slots

    /// Checks out a free sequence slot, or `nil` if they are all taken.
    ///
    /// Dropping the returned `TokenSequence` (or letting it go out of scope)
    /// hands the slot back. Mirrors `Context::sequence` in the Rust crate,
    /// which checks out a slot from the actor and wraps it in a `Sequence`.
    public func checkoutSequence() -> TokenSequence? {
        state.withLock { state in
            guard let slot = state.checkedOut.firstIndex(of: false) else { return nil }
            state.checkedOut[slot] = true
            return TokenSequence(context: self, id: SequenceID(Int32(slot)))
        }
    }

    /// How many sequence slots are still free.
    public var freeSlots: Int {
        state.withLock { $0.checkedOut.filter { !$0 }.count }
    }

    /// The context size llama.cpp actually granted (at least `params.nContext`).
    public var contextSize: UInt32 {
        state.withLock { llama_n_ctx($0.handle) }
    }

    /// Can this context's memory be shifted (needed for some KV-cache edits)?
    public var canShift: Bool {
        state.withLock { llama_memory_can_shift(llama_get_memory($0.handle)) }
    }

    /// A snapshot of llama.cpp's performance counters.
    public var performance: ContextPerformance {
        state.withLock { ContextPerformance(llama_perf_context($0.handle)) }
    }

    // MARK: Operations used by TokenSequence
    //
    // All internal: the public surface is `TokenSequence`, which owns the
    // per-sequence bookkeeping (token list, cached logits) and calls in here
    // for anything that touches the shared C state.

    /// Decodes a single token and returns the resulting logits.
    ///
    /// The Rust actor does exactly one token per `push_token` message, using
    /// its batch of capacity 1; this mirrors that.
    func pushToken(_ token: Token, at pos: Position, seq id: SequenceID) throws -> [Float] {
        try state.withLock { state in
            state.batch.clear()
            try state.batch.add(
                token: token, pos: pos, seqIDs: [id], wantsLogits: true
            )

            let status = llama_decode(state.handle, state.batch.cBatch)
            guard status == 0 else { throw KototamaError.decodeFailed(DecodeError(status: status)) }

            guard let logits = llama_get_logits_ith(state.handle, 0) else {
                throw KototamaError.logitsUnavailable
            }
            return Array(UnsafeBufferPointer(start: logits, count: Int(state.nVocabulary)))
        }
    }

    /// Asks llama.cpp to forget a range of positions in a sequence's KV cache.
    /// Returns `false` when the range was not present.
    func removeKVCache(seq id: SequenceID, from start: Int32, to end: Int32) -> Bool {
        state.withLock {
            llama_memory_seq_rm(llama_get_memory($0.handle), id.rawValue, start, end)
        }
    }

    /// Copies a range of one sequence's KV cache into another sequence.
    func copyKVCache(from src: SequenceID, to dst: SequenceID, from start: Int32, to end: Int32) {
        state.withLock {
            llama_memory_seq_cp(llama_get_memory($0.handle), src.rawValue, dst.rawValue, start, end)
        }
    }

    /// Shifts positions of a sequence's KV cache by `delta`.
    func shiftKVCache(seq id: SequenceID, from start: Int32, to end: Int32, by delta: Int32) {
        state.withLock {
            llama_memory_seq_add(llama_get_memory($0.handle), id.rawValue, start, end, delta)
        }
    }

    /// Lowest position present in a sequence's KV cache (`-1` when empty).
    func minPosition(seq id: SequenceID) -> Position {
        state.withLock { Position(llama_memory_seq_pos_min(llama_get_memory($0.handle), id.rawValue)) }
    }

    /// Highest position present in a sequence's KV cache (`-1` when empty).
    func maxPosition(seq id: SequenceID) -> Position {
        state.withLock { Position(llama_memory_seq_pos_max(llama_get_memory($0.handle), id.rawValue)) }
    }

    /// Returns a slot to the pool and wipes its KV cache. Called from
    /// `TokenSequence.deinit`, matching the Rust crate's `Drop` impl.
    func releaseSlot(_ id: SequenceID) {
        state.withLock { state in
            llama_memory_seq_rm(llama_get_memory(state.handle), id.rawValue, -1, -1)
            state.checkedOut[Int(id.rawValue)] = false
        }
    }
}

/// The locked state behind one `Context`.
private struct ContextState {
    /// The raw llama.cpp context. Freed by `Context.deinit`.
    let handle: OpaquePointer
    /// Reused one-token batch (the Rust actor keeps one the same way).
    let batch: TokenBatch
    /// Vocabulary size — the width of the logits vector `llama_decode` fills.
    let nVocabulary: Int32
    /// Which sequence slots are checked out. Index = sequence ID.
    var checkedOut: [Bool]
}
