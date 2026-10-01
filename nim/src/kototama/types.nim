## Public scalar types shared across the library.
##
## These are the same aliases the Rust crate defines at its root
## (`Token = i32`, `Pos = i32`, `SeqId = i32`): plain aliases, not distinct
## types, so the binding layer and the high-level API stay interchangeable.

type
  Token* = int32
    ## A llama.cpp token id.

  Pos* = int32
    ## A position inside a sequence.

  SeqId* = int32
    ## A sequence id within one context.
