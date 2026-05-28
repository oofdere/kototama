---
title: "Elixir API Reference"
---

## Elixir API Reference <span class="version-badge">v0.1.0</span>

### Types

#### Context

Handle to a running context actor. Clone + Send + Sync.

The actor thread is stopped when the last clone is dropped.

### Functions

#### new()

**Signature:**

```elixir
def new(model, params)
```

#### sequence()

**Signature:**

```elixir
def sequence()
```

#### free_slots()

**Signature:**

```elixir
def free_slots()
```

#### n_ctx()

**Signature:**

```elixir
def n_ctx()
```

#### can_shift()

**Signature:**

```elixir
def can_shift()
```

#### perf()

**Signature:**

```elixir
def perf()
```

---

#### Model

Thread-safe handle to a loaded model.

Cloning is cheap (Arc bump). Use from any thread.

### Functions

#### load_from_file()

**Signature:**

```elixir
def load_from_file(path, params)
```

#### as_ptr()

**Signature:**

```elixir
def as_ptr()
```

#### chat_template()

**Signature:**

```elixir
def chat_template(name)
```

#### desc()

**Signature:**

```elixir
def desc()
```

#### has_decoder()

**Signature:**

```elixir
def has_decoder()
```

#### decoder_start_token()

**Signature:**

```elixir
def decoder_start_token()
```

#### has_encoder()

**Signature:**

```elixir
def has_encoder()
```

#### is_diffusion()

**Signature:**

```elixir
def is_diffusion()
```

#### is_hybrid()

**Signature:**

```elixir
def is_hybrid()
```

#### is_recurrent()

**Signature:**

```elixir
def is_recurrent()
```

#### token_to_piece()

**Signature:**

```elixir
def token_to_piece(token)
```

#### tokenize()

**Signature:**

```elixir
def tokenize(text, add_special, parse_special)
```

#### get_add_bos()

**Signature:**

```elixir
def get_add_bos()
```

#### get_add_eos()

**Signature:**

```elixir
def get_add_eos()
```

#### get_add_sep()

**Signature:**

```elixir
def get_add_sep()
```

#### get_attr()

**Signature:**

```elixir
def get_attr(token)
```

#### get_score()

**Signature:**

```elixir
def get_score(token)
```

#### get_text()

**Signature:**

```elixir
def get_text(token)
```

#### is_control()

**Signature:**

```elixir
def is_control(token)
```

#### is_eog()

**Signature:**

```elixir
def is_eog(token)
```

#### n_tokens()

**Signature:**

```elixir
def n_tokens()
```

#### vocab_type()

**Signature:**

```elixir
def vocab_type()
```

---

#### Sequence

A sequence handle. No lifetime parameters — holds a clone of the Context
handle and communicates with the context actor via messages.

Logits from the last `push()` are cached locally, so `sample()` and
`logits()` don't need a second round-trip.

### Functions

#### logits()

**Signature:**

```elixir
def logits()
```

#### is_empty()

**Signature:**

```elixir
def is_empty()
```

#### push()

**Signature:**

```elixir
def push(token)
```

#### decode()

Re-decode the last token to refresh logits without pushing a new one.
Useful after `pop()`, `remove()`, or other mutations that invalidate logits.

**Signature:**

```elixir
def decode()
```

#### pop()

**Signature:**

```elixir
def pop()
```

#### len()

**Signature:**

```elixir
def len()
```

#### extend()

**Signature:**

```elixir
def extend(tokens)
```

#### get()

**Signature:**

```elixir
def get(index)
```

#### remove()

**Signature:**

```elixir
def remove(range)
```

#### copy_to()

**Signature:**

```elixir
def copy_to(other, range)
```

#### copy_from()

**Signature:**

```elixir
def copy_from(other, range)
```

#### pos_min()

**Signature:**

```elixir
def pos_min()
```

#### pos_max()

**Signature:**

```elixir
def pos_max()
```

#### tokens()

**Signature:**

```elixir
def tokens()
```

#### kv_remove()

**Signature:**

```elixir
def kv_remove(range)
```

#### kv_copy()

**Signature:**

```elixir
def kv_copy(other, range)
```

#### kv_shift()

**Signature:**

```elixir
def kv_shift(range, delta)
```

#### index()

**Signature:**

```elixir
def index(index)
```

#### drop()

**Signature:**

```elixir
def drop()
```

---

### Enums

#### DecodeError

| Value | Description |
|-------|-------------|
| `slot_not_found` | Slot not found |
| `aborted` | Aborted |
| `invalid_input` | Invalid input |
| `fatal_error` | Fatal error |

---
