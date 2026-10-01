## Error types. Every failure mode in the library raises one of these.
##
## The Rust crate answers many of these conditions with `Result<T, ()>`, which
## forces every caller into `match`/`unwrap` ceremony while erasing *why*
## something failed. Nim's idiomatic channel is exceptions, so this port raises
## a small hierarchy instead: each type says exactly what went wrong, and
## callers that care can catch precisely.
##
## `DecodeError` keeps the fine-grained kind from the Rust `DecodeError` enum,
## because llama.cpp's `llama_decode` distinguishes recoverable conditions
## (no KV slot, aborted) from genuine misuse.

type
  KototamaError* = object of CatchableError
    ## Base for every error raised by this library.

  ModelLoadError* = object of KototamaError
    ## `Model.loadFromFile` could not load the model file.

  ContextCreateError* = object of KototamaError
    ## `Context.new` could not create a llama.cpp context.

  DecodeErrorKind* = enum
    deSlotNotFound
      ## `llama_decode` could not find a KV slot for the batch: the context is
      ## full. Try a larger context or free a sequence.
    deAborted
      ## The decode was aborted by an abort callback.
    deInvalidInput
      ## The batch was malformed (llama.cpp answered -1).
    deFatal
      ## Any other llama.cpp failure (return codes below -1).

  DecodeError* = object of KototamaError
    ## `llama_decode` failed. `kind` narrows down why.
    kind*: DecodeErrorKind

proc decodeError*(kind: DecodeErrorKind): ref DecodeError =
  ## Build a `DecodeError` with a human-readable message for `kind`.
  let msg = case kind
    of deSlotNotFound: "llama_decode: no free KV slot (context full?)"
    of deAborted: "llama_decode: aborted"
    of deInvalidInput: "llama_decode: invalid input batch"
    of deFatal: "llama_decode: fatal error"
  result = newException(DecodeError, msg)
  result.kind = kind
