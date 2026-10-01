## Backend lifecycle: one handle per process-wide llama.cpp init.
##
## llama.cpp keeps global state (registered backends, the ggml log hook), so
## `llama_backend_init`/`llama_backend_free` must run exactly once per process
## regardless of how many models and contexts exist. The Rust crate does this
## with an `AtomicUsize` counter and a `Backend` zero-sized token whose `Drop`
## frees the backend when the last handle goes away.
##
## Nim reaches the same design with less machinery: a plain `ref` object whose
## `=destroy` runs when the refcount hits zero. Copying the ref is free and
## share-safe; `acquire()` takes a lock only on the first and last call.

import std/locks
import ./raw

type
  Backend* = ref BackendObj
    ## Shared, refcounted handle to the process-wide backend. Grab one with
    ## `acquire`; the backend shuts down when the last handle is released.

  BackendObj = object
    ## Destructor target. Never used directly.

var
  backendLock: Lock
  backendHeld: int    ## Number of live `Backend` handles.
  backendUp = false   ## Whether llama.cpp global state is currently initialized.

initLock(backendLock)

proc `=destroy`(_: var BackendObj) =
  withLock backendLock:
    dec backendHeld
    if backendHeld == 0 and backendUp:
      llamaBackendFree()
      backendUp = false

proc acquire*(): Backend =
  ## Get a handle on the global backend, initializing it on first use.
  ## Everything that touches llama.cpp (models, contexts) holds one of these
  ## internally, so users rarely call this directly.
  withLock backendLock:
    if backendHeld == 0 and not backendUp:
      ggmlBackendLoadAll()
      llamaBackendInit()
      backendUp = true
    inc backendHeld
  result = Backend()
