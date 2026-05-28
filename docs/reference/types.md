---
title: "Types Reference"
---

## Types Reference

All types defined by the library, grouped by category. Types are shown using Rust as the canonical representation.

### Other Types

#### Model

Thread-safe handle to a loaded model.

Cloning is cheap (Arc bump). Use from any thread.

*Opaque type — fields are not directly accessible.*

---

#### Context

Handle to a running context actor. Clone + Send + Sync.

The actor thread is stopped when the last clone is dropped.

*Opaque type — fields are not directly accessible.*

---

#### Sequence

A sequence handle. No lifetime parameters — holds a clone of the Context
handle and communicates with the context actor via messages.

Logits from the last `push()` are cached locally, so `sample()` and
`logits()` don't need a second round-trip.

*Opaque type — fields are not directly accessible.*

---

### Enums

#### DecodeError

| Variant | Description |
|---------|-------------|
| `SlotNotFound` | Slot not found |
| `Aborted` | Aborted |
| `InvalidInput` | Invalid input |
| `FatalError` | Fatal error |

---
