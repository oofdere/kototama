//
//  Batch.swift
//  Kototama
//
//  A batch of tokens waiting to be decoded.
//

import CLlama

/// Errors from filling a batch.
enum BatchError: Error {
    /// The batch already holds as many tokens as it was allocated for.
    case full
}

/// Owns a `llama_batch`: the C-side array of tokens handed to `llama_decode`.
///
/// The Rust crate keeps this internal (`pub(crate)`, in `src/batch.rs` and
/// `src/common.rs`) and only ever fills one token at a time, from inside its
/// context actor. The Swift port does the same — `Context` owns one instance
/// and every decode goes through it.
///
/// `llama_batch` is a struct of raw pointers into memory that `llama_batch_init`
/// allocates, so this class exists mostly to pair that allocation with its
/// `llama_batch_free` in `deinit`.
final class TokenBatch: @unchecked Sendable {
    /// How many tokens this batch was allocated for. Slots beyond it are
    /// unallocated memory — the fill methods refuse to write there.
    let capacity: Int32

    /// The raw C batch. Only touched while `Context`'s lock is held.
    private var raw: llama_batch

    init(capacity: Int32, maxSequences: Int32) {
        self.capacity = capacity
        self.raw = llama_batch_init(capacity, 0, maxSequences)
    }

    deinit {
        llama_batch_free(raw)
    }

    /// Empties the batch without freeing its buffers.
    func clear() {
        raw.n_tokens = 0
    }

    /// Appends one token at the next free slot.
    ///
    /// - Parameters:
    ///   - token: The token ID to decode.
    ///   - pos: Its position in the sequence.
    ///   - seqIDs: Which sequence(s) this token belongs to.
    ///   - wantsLogits: Ask llama.cpp to compute logits for this slot.
    ///
    /// Mirrors `common::batch_add` in the Rust crate. That version checks a
    /// `seq_id` pointer for null to detect overflow (with a `todo: get rid of
    /// this pointer arithmetic` note); here the capacity check is just a
    /// comparison, because Swift can express "is this index inside the
    /// allocation?" directly.
    func add(token: Token, pos: Position, seqIDs: [SequenceID], wantsLogits: Bool) throws {
        guard raw.n_tokens < capacity else { throw BatchError.full }
        let index = Int(raw.n_tokens)

        // Every column of `llama_batch` is a raw pointer that `llama_batch_init`
        // allocated, so all of them arrive as optionals and get unwrapped here.
        guard
            let tokenColumn = raw.token,
            let positionColumn = raw.pos,
            let sequenceCountColumn = raw.n_seq_id,
            let seqIDColumn = raw.seq_id,
            let logitsColumn = raw.logits
        else {
            throw BatchError.full
        }

        tokenColumn[index] = token.rawValue
        positionColumn[index] = pos.rawValue
        sequenceCountColumn[index] = Int32(seqIDs.count)
        // `seq_id` is `llama_seq_id **`: a column of row pointers, each row
        // holding the sequence IDs for one token. Rows are allocated by
        // `llama_batch_init`, so they arrive as optionals too.
        guard let seqIDRow = seqIDColumn[index] else {
            throw BatchError.full
        }
        for (slot, id) in seqIDs.enumerated() {
            seqIDRow[slot] = id.rawValue
        }
        logitsColumn[index] = wantsLogits ? 1 : 0

        raw.n_tokens += 1
    }

    /// The C batch, for handing to `llama_decode`.
    var cBatch: llama_batch { raw }
}
