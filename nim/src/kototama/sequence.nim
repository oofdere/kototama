## Sequence: one generation stream inside a context.
##
## This is the Nim counterpart of `src/sequence.rs`. A `Sequence` owns a
## sequence id inside a `Context` and mirrors its token history locally, so
## questions like "what tokens did I push?" never need a round trip.
##
## Logits from the last `push` are cached here too, so `sample` and `logits`
## run without an extra decode.
##
## Dropping a `Sequence` returns its slot to the context and clears its KV
## memory -- the same behavior as the Rust `Drop` impl.
##
## ## Range conventions
##
## Token-history edits (`remove`, `copyTo`, `copyFrom`) take a Nim `Slice`,
## which is *inclusive* at both ends: `seq.remove(2..4)` removes three tokens.
## The Rust API used `Range` (end-exclusive) here. KV-level operations
## (`kvRemove`, `kvCopy`, `kvShift`) mirror llama.cpp's own `p0`/`p1` convention
## exactly instead: `p1` is exclusive and `-1` means "to the end".

import std/[options, sequtils]
import ./context, ./types, ./samplers

type
  Sequence* = ref SequenceObj
    ## A checked-out sequence slot. All copies refer to the same slot.

  SequenceObj = object
    ctx: Context
    id: SeqId
    tokens: seq[Token]
    cachedLogits: Option[seq[float32]]
    closed: bool

proc `=destroy`(self: var SequenceObj) =
  # Return the slot and clear its memory. `ctx` keeps the context alive until
  # this runs, so the call is always safe.
  if not self.ctx.isNil and not self.closed:
    self.ctx.releaseSeq(self.id)
  # A custom `=destroy` replaces the compiler's field destruction entirely
  # (unlike Rust's Drop), so release the held context reference by hand.
  self.ctx = nil

proc close*(self: Sequence) =
  ## Release this sequence's slot now instead of at scope exit.
  ##
  ## Nim keeps temporaries alive to the end of their scope, so dropping a
  ## sequence early is not as simple as Rust's `drop(seq)`. Call this when the
  ## KV cache slot should be reclaimed deterministically. Idempotent.
  if not self.closed and not self.ctx.isNil:
    self.ctx.releaseSeq(self.id)
    self.closed = true

proc newSequence*(ctx: Context; id: SeqId): Sequence {.inline.} =
  ## Internal. Build a handle for a slot already checked out of `ctx`.
  Sequence(ctx: ctx, id: id, tokens: @[], cachedLogits: none(seq[float32]), closed: false)

# ---------------------------------------------------------------------------
# Token history
# ---------------------------------------------------------------------------

proc len*(self: Sequence): int = self.tokens.len
proc isEmpty*(self: Sequence): bool = self.tokens.len == 0
proc tokens*(self: Sequence): seq[Token] = self.tokens

proc get*(self: Sequence; index: int): Option[Token] =
  ## The token at `index`, or `none` when out of range.
  if index in 0 ..< self.tokens.len: some(self.tokens[index]) else: none(Token)

proc `[]`*(self: Sequence; index: int): Token = self.tokens[index]

# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

proc logits*(self: Sequence): Option[seq[float32]] =
  ## The full vocabulary logits produced by the last `push`, or `none` when
  ## they are stale (after `pop` or `remove`, or on a fresh sequence).
  self.cachedLogits

proc push*(self: Sequence; token: Token) =
  ## Decode one token at the end of this sequence and cache its logits.
  let pos = self.tokens.len.Pos
  self.cachedLogits = some(self.ctx.pushToken(token, pos, self.id))
  self.tokens.add(token)

proc extend*(self: Sequence; toks: openArray[Token]) =
  ## Push each token in order.
  for token in toks:
    self.push(token)

proc decode*(self: Sequence) =
  ## Re-decode the last token to refresh the cached logits without pushing a
  ## new one. Useful after `pop` or `remove` invalidates them.
  if self.tokens.len == 0:
    return
  let pos = (self.tokens.len - 1).Pos
  self.cachedLogits = some(self.ctx.pushToken(self.tokens[^1], pos, self.id))

# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

proc pop*(self: Sequence): Option[Token] =
  ## Drop the last token (and its KV memory). Answers the dropped token, or
  ## `none` on an empty sequence or when the KV removal fails.
  if self.tokens.len == 0:
    return none(Token)
  let lastPos = self.tokens.len.Pos - 1
  if not self.ctx.memorySeqRm(self.id, lastPos, lastPos + 1):
    return none(Token)
  result = some(self.tokens[^1])
  self.tokens.setLen(self.tokens.len - 1)
  self.cachedLogits = none(seq[float32])

proc remove*(self: Sequence; range: Slice[int]): bool =
  ## Drop `range` (inclusive at both ends) from the token history and KV
  ## memory. Answers whether the KV removal succeeded.
  result = self.ctx.memorySeqRm(self.id, range.a.Pos, range.b.Pos + 1)
  if result:
    self.tokens.delete(range)
    self.cachedLogits = none(seq[float32])

proc copyTo*(self: Sequence; other: Sequence; range: Slice[int]) =
  ## Copy a position range (inclusive) of KV memory and token history into
  ## `other`, replacing whatever `other` held.
  self.ctx.memorySeqCp(self.id, other.id, range.a.Pos, range.b.Pos + 1)
  other.tokens.setLen(0)
  other.tokens.add(self.tokens[range])
  other.cachedLogits = none(seq[float32])

proc copyFrom*(self: Sequence; other: Sequence; range: Slice[int]) =
  ## The mirror of `copyTo`: pull `range` (inclusive) from `other` into `self`.
  other.copyTo(self, range)

# ---------------------------------------------------------------------------
# KV memory operations. These mirror llama.cpp exactly: `p1` is exclusive and
# `-1` means "to the end". They touch KV memory only, never the token history.
# ---------------------------------------------------------------------------

proc kvRemove*(self: Sequence; p0, p1: Pos): bool =
  ## Remove KV memory in `[p0, p1)`. Answers whether the removal succeeded.
  result = self.ctx.memorySeqRm(self.id, p0, p1)
  if result:
    self.cachedLogits = none(seq[float32])

proc kvCopy*(self: Sequence; other: Sequence; p0, p1: Pos) =
  ## Copy KV memory in `[p0, p1)` from this sequence into `other`.
  self.ctx.memorySeqCp(self.id, other.id, p0, p1)
  other.cachedLogits = none(seq[float32])

proc kvShift*(self: Sequence; p0, p1: Pos; delta: Pos) =
  ## Shift KV memory in `[p0, p1)` by `delta` positions.
  self.ctx.memorySeqAdd(self.id, p0, p1, delta)
  self.cachedLogits = none(seq[float32])

# ---------------------------------------------------------------------------
# KV memory views
# ---------------------------------------------------------------------------

proc posMin*(self: Sequence): Pos =
  ## Lowest position in this sequence's KV memory, `-1` when empty.
  self.ctx.memorySeqPosMin(self.id)

proc posMax*(self: Sequence): Pos =
  ## Highest position in this sequence's KV memory, `-1` when empty.
  self.ctx.memorySeqPosMax(self.id)

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

proc sample*(self: Sequence; sampler: Sampler): Option[Token] =
  ## Draw a token from the cached logits through `sampler`.
  ##
  ## The sampler sees the raw logits exactly once. (The Rust API applies the
  ## sampler's transform twice here, which over-scales logits for Temperature
  ## and over-masks for the truncating samplers; see COMPARISON.md.) Answers
  ## `none` when the cached logits are stale.
  if self.cachedLogits.isNone:
    return none(Token)
  some(sampler.sample(self.cachedLogits.get()))

# ---------------------------------------------------------------------------
# Context-side accessor, placed here to keep the module graph acyclic.
# ---------------------------------------------------------------------------

proc sequence*(ctx: Context): Option[Sequence] =
  ## Check out a sequence slot, or `none` when every slot is taken.
  let id = ctx.checkoutSeq()
  if id.isNone:
    none(Sequence)
  else:
    some(newSequence(ctx, id.get()))
