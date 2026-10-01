## Context tests -- the Nim counterpart of `tests/context.rs`.
##
## Run with: `nim c -r --path:src tests/tcontext.nim`

import std/[options, unittest]
import ../src/kototama
import testutil

suite "Context":
  test "context_new_ok":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    check not ctx.isNil

  test "n_ctx_at_least_params":
    # llama.cpp may round n_ctx up, but never below the request.
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    check ctx.nCtx() >= params.nCtx

  test "free_slots_starts_full":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    check ctx.freeSlots() == params.nSeqMax.int

  test "sequence_checkout_reduces_free_slots":
    # close() releases the slot immediately. (Rust tests this with `drop(seq)`;
    # in Nim, assignment alone defers release to scope end because the value's
    # temporary may still be alive. See Sequence.close.)
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let total = ctx.freeSlots()
    let seq = ctx.sequence().get()
    check ctx.freeSlots() == total - 1
    seq.close()
    check ctx.freeSlots() == total

  test "sequence_drop_returns_slot":
    # Same as above, exercised through scope exit rather than an explicit nil.
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let before = ctx.freeSlots()
    block:
      let seq = ctx.sequence().get()
      check ctx.freeSlots() == before - 1
    check ctx.freeSlots() == before

  test "all_slots_exhausted":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let n = params.nSeqMax.int
    var seqs: seq[Sequence] = @[]
    for _ in 0 ..< n:
      seqs.add(ctx.sequence().get())
    check ctx.freeSlots() == 0
    check ctx.sequence().isNone()   # no slot left
    seqs = @[]
    check ctx.freeSlots() == n

  test "can_shift_does_not_crash":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    discard ctx.canShift()

  test "perf_does_not_crash":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    discard ctx.perf()

suite "Context sharing":
  # Mirrors the Rust `Context::clone()` tests: a shared `ref` sees the same
  # slots from every copy.
  test "context copies share free slots":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let ctxCopy = ctx
    check ctx.freeSlots() == ctxCopy.freeSlots()

  test "context copy slot checkout visible on original":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let ctxCopy = ctx
    let total = ctx.freeSlots()
    let seq = ctxCopy.sequence().get()
    check ctx.freeSlots() == total - 1

  test "context copy slot checkout visible on copy":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let ctxCopy = ctx
    let total = ctx.freeSlots()
    let seq = ctx.sequence().get()
    check ctxCopy.freeSlots() == total - 1

  test "context copy sequence drop restores on both":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let ctxCopy = ctx
    let total = ctx.freeSlots()
    block:
      let seq = ctx.sequence().get()
      check ctxCopy.freeSlots() == total - 1
    check ctx.freeSlots() == total
    check ctxCopy.freeSlots() == total

  test "context copy n_ctx matches":
    let (model, params) = loadModelAndContext()
    let ctx = Context.new(model, params)
    let ctxCopy = ctx
    check ctx.nCtx() == ctxCopy.nCtx()

suite "Context from shared Model":
  test "context_from_shared_model":
    let (model, params) = loadModelAndContext()
    let modelCopy = model
    let ctx = Context.new(modelCopy, params)
    check ctx.freeSlots() > 0
    check ctx.nCtx() >= params.nCtx

  test "context_from_shared_model_is_independent":
    # Independent contexts do not share sequence slots.
    let (model, params) = loadModelAndContext()
    let modelCopy = model
    let ctx1 = Context.new(model, params)
    let ctx2 = Context.new(modelCopy, params)
    let seq1 = ctx1.sequence().get()
    check ctx2.freeSlots() == params.nSeqMax.int
