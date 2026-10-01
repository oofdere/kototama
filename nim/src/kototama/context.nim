## Context: a running inference session over a model.
##
## This is the Nim counterpart of `src/context.rs`. It owns the
## `llama_context` and hands out `Sequence` slots (see `sequence.nim`).
##
## ## Threading: a lock, not an actor
##
## The Rust crate hides its raw `*mut llama_context` behind a spawned actor
## thread. That design exists for one reason: the pointer is not `Send`, so the
## actor pins every llama.cpp call to one thread while other threads talk to it
## over channels. It costs a message type and a handler per operation.
##
## Nim can state the same guarantee more directly. A `Context` is a shared
## `ref` that serializes access to the C handle with one mutex: every operation
## takes the lock, touches llama.cpp, and releases. The effect is identical --
## the raw pointer is only ever dereferenced by one thread at a time -- without
## the channel plumbing. The tradeoff is that a long `decode` blocks other
## callers instead of queueing behind a running actor, which for a single
## shared context is what the actor would do anyway.
##
## ## Lifetime
##
## A `Context` holds a reference to its `Model`. This is load-bearing:
## `llama_init_from_model` does not copy the weights, and `llama_model_free` in
## this llama.cpp version does not refcount, so a context must keep its model
## alive for its whole life. See `model.nim` and COMPARISON.md.
##
## ## About the exported internals
##
## The procs marked "internal" below (everything in the lower half) exist for
## `sequence.nim` to call. Rust hides the equivalents with `pub(crate)`; Nim
## has no visibility tier between private and public, so they are exported and
## documented as off-limits to library users.

import std/[locks, options]
import ./raw, ./batch, ./model, ./types, ./errors

type
  Context* = ref ContextObj
    ## Shared handle to a running context. Cheap to copy; all copies share one
    ## set of sequence slots.

  ContextObj = object
    handle: ptr raw.LlamaContext
    batch: Batch
    nVocab: int32
    checkedOut: seq[bool]
    model: Model          ## Keeps the weights alive (see module docs).
    lock: Lock

proc `=destroy`(self: var ContextObj) =
  if not self.handle.isNil:
    llamaFree(self.handle)
    self.handle = nil
  # A custom `=destroy` replaces the compiler's field destruction entirely
  # (unlike Rust's Drop), so release each owned field by hand.
  self.batch = nil
  self.model = nil
  reset(self.checkedOut)
  deinitLock(self.lock)

proc close*(self: Context) =
  ## Free the underlying llama.cpp context now instead of at scope exit.
  ##
  ## Nim keeps temporaries alive to the end of their scope, so early release
  ## is an explicit call here, mirroring `Sequence.close`. Idempotent. After
  ## closing, the handle (and its sequences) must not be used again.
  withLock self.lock:
    if not self.handle.isNil:
      llamaFree(self.handle)
      self.handle = nil

proc nCtx*(self: Context): uint32 =
  ## The context's size in tokens.
  withLock self.lock:
    result = llamaNCtx(self.handle)

proc freeSlots*(self: Context): int =
  ## How many sequence slots are available right now.
  withLock self.lock:
    var free = 0
    for taken in self.checkedOut:
      if not taken: inc free
    result = free

proc canShift*(self: Context): bool =
  ## Whether this context's memory supports shifting positions.
  withLock self.lock:
    result = llamaMemoryCanShift(llamaGetMemory(self.handle))

proc perf*(self: Context): LlamaPerfContextData =
  ## Performance counters for this context.
  withLock self.lock:
    result = llamaPerfContext(self.handle)

proc new*(T: typedesc[Context]; model: Model;
          params = llamaContextDefaultParams()): Context =
  ## Create a context over `model`.
  ##
  ## `params` comes from `raw.llamaContextDefaultParams()`. The context has
  ## `params.nSeqMax` sequence slots.
  let handle = llamaInitFromModel(model.handle, params)
  if handle.isNil:
    raise newException(ContextCreateError, "failed to create context")

  result = Context(
    handle: handle,
    batch: initBatch(1'i32, params.nSeqMax.int32),
    nVocab: model.nTokens(),
    checkedOut: newSeq[bool](params.nSeqMax.int),
    model: model,
  )
  initLock(result.lock)

# ---------------------------------------------------------------------------
# Internal operations (for `sequence.nim`)
#
# These correspond one-for-one to the Rust actor's message handlers; each takes
# the lock for its duration instead of running on an actor thread.
# ---------------------------------------------------------------------------

proc checkoutSeq*(self: Context): Option[SeqId] {.discardable.} =
  ## Internal. Take a free sequence slot.
  withLock self.lock:
    for i, taken in self.checkedOut.mpairs:
      if not taken:
        taken = true
        return some(SeqId(i))
  none(SeqId)

proc releaseSeq*(self: Context; seqId: SeqId) =
  ## Internal. Return a sequence slot and clear its memory. Safe to call after
  ## `close()`: a closed context only gives the slot back locally, since its
  ## llama.cpp memory is already gone.
  withLock self.lock:
    if not self.handle.isNil:
      discard llamaMemorySeqRm(llamaGetMemory(self.handle), seqId, -1, -1)
    if seqId.int in 0 ..< self.checkedOut.len:
      self.checkedOut[seqId.int] = false

proc pushToken*(self: Context; token: Token; pos: Pos; seqId: SeqId): seq[float32] =
  ## Internal. Decode one token into a sequence and return its output logits.
  withLock self.lock:
    if self.handle.isNil:
      raise decodeError(deFatal)   # Context is closed.
    self.batch.clear()
    self.batch.add(token, pos, [seqId], logits = true)

    let rc = llamaDecode(self.handle, self.batch.asRaw())
    case rc
    of 0: discard
    of 1: raise decodeError(deSlotNotFound)
    of 2: raise decodeError(deAborted)
    of -1: raise decodeError(deInvalidInput)
    else: raise decodeError(deFatal)

    let logitsPtr = llamaGetLogitsIth(self.handle, 0)
    if logitsPtr.isNil:
      raise decodeError(deFatal)
    result = newSeq[float32](self.nVocab.int)
    copyMem(addr result[0], logitsPtr, self.nVocab.int * sizeof(float32))

proc memorySeqRm*(self: Context; seqId: SeqId; p0, p1: Pos): bool =
  ## Internal. Remove a range of positions from a sequence's memory.
  ## `p1` is exclusive; `-1` means "to the end".
  withLock self.lock:
    result = llamaMemorySeqRm(llamaGetMemory(self.handle), seqId, p0, p1)

proc memorySeqCp*(self: Context; src, dst: SeqId; p0, p1: Pos) =
  ## Internal. Copy a position range from one sequence's memory to another's.
  withLock self.lock:
    llamaMemorySeqCp(llamaGetMemory(self.handle), src, dst, p0, p1)

proc memorySeqAdd*(self: Context; seqId: SeqId; p0, p1: Pos; delta: Pos) =
  ## Internal. Shift a position range of a sequence's memory by `delta`.
  withLock self.lock:
    llamaMemorySeqAdd(llamaGetMemory(self.handle), seqId, p0, p1, delta)

proc memorySeqPosMin*(self: Context; seqId: SeqId): Pos =
  ## Internal. Lowest position currently held in a sequence's memory, `-1`
  ## when empty.
  withLock self.lock:
    result = llamaMemorySeqPosMin(llamaGetMemory(self.handle), seqId)

proc memorySeqPosMax*(self: Context; seqId: SeqId): Pos =
  ## Internal. Highest position currently held in a sequence's memory, `-1`
  ## when empty.
  withLock self.lock:
    result = llamaMemorySeqPosMax(llamaGetMemory(self.handle), seqId)

# The public `Context.sequence()` accessor lives in `sequence.nim`, beside the
# `Sequence` type it returns. This keeps the module graph acyclic.