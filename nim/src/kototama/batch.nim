## Batch: a stack of tokens fed to `llama_decode` in one call.
##
## A `llama_batch` is a bundle of parallel arrays (tokens, positions, sequence
## ids, "should I output logits here" flags) that llama.cpp processes in a
## single forward pass. The C API hands out those arrays from
## `llama_batch_init` and expects `llama_batch_free` on the way out; this module
## owns that pairing so the arrays can never leak or double-free.
##
## kototama currently decodes one token at a time (see `context.nim`), so a
## batch here holds exactly one row, sized at construction.

import ./raw

type
  Batch* = ref BatchObj
    ## Owns a `llama_batch`. Dropping it frees the underlying arrays.

  BatchObj = object
    data: LlamaBatch

  BatchAddError* = object of CatchableError
    ## Raised when a row is appended to a batch that is already full.

proc `=destroy`(self: var BatchObj) =
  # `llama_batch_free` frees each array unconditionally but tolerates nulls,
  # so this is safe even if the object never held a real batch.
  llamaBatchFree(self.data)

proc initBatch*(nTokens, nSeqMax: int32): Batch =
  ## Allocate a batch with room for `nTokens` token rows.
  ## `nSeqMax` is how many sequences a single row may belong to.
  ##
  ## Row capacity is not stored in the C struct: `llama_batch_init` appends a
  ## null sentinel just past the last `seq_id` slot, and a full batch is found
  ## by seeing that sentinel where the next row would go.
  Batch(data: llamaBatchInit(nTokens, 0, nSeqMax))

proc asRaw*(self: Batch): LlamaBatch =
  ## The underlying struct, for passing to C by value (Rust calls this
  ## `as_raw`).
  self.data

proc clear*(self: Batch) =
  ## Forget every row, keeping the allocated arrays for reuse.
  self.data.nTokens = 0

proc add*(self: Batch; token: LlamaToken; pos: LlamaPos;
          seqIds: openArray[LlamaSeqId]; logits: bool) =
  ## Append one row: which token, at what position, belonging to which
  ## sequences, and whether to write its output logits.
  ##
  ## Raises `BatchAddError` when the batch is already at capacity.
  let slot = self.data.nTokens.int
  # The parallel arrays are raw C buffers; Nim indexes them through
  # `UncheckedArray`, the typed view of "pointer to many".
  let rows = cast[ptr UncheckedArray[ptr LlamaSeqId]](self.data.seqId)
  if rows[slot].isNil:
    # The sentinel marks the end of the allocated rows.
    raise newException(BatchAddError, "batch is full")

  cast[ptr UncheckedArray[LlamaToken]](self.data.token)[slot] = token
  cast[ptr UncheckedArray[LlamaPos]](self.data.pos)[slot] = pos
  cast[ptr UncheckedArray[int32]](self.data.nSeqId)[slot] = seqIds.len.int32
  let row = cast[ptr UncheckedArray[LlamaSeqId]](rows[slot])
  for i, seqId in seqIds:
    row[i] = seqId
  cast[ptr UncheckedArray[int8]](self.data.logits)[slot] = (if logits: 1'i8 else: 0'i8)

  inc self.data.nTokens
