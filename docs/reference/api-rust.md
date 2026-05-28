---
title: "Rust API Reference"
---

## Rust API Reference <span class="version-badge">v0.1.0</span>

### Types

#### Context

Handle to a running context actor. Clone + Send + Sync.

The actor thread is stopped when the last clone is dropped.

### Methods

#### new()

**Signature:**

```rust
pub fn new(model: Model, params: &str) -> Context
```

#### sequence()

**Signature:**

```rust
pub fn sequence(&self) -> Option<Sequence>
```

#### free_slots()

**Signature:**

```rust
pub fn free_slots(&self) -> usize
```

#### n_ctx()

**Signature:**

```rust
pub fn n_ctx(&self) -> u32
```

#### can_shift()

**Signature:**

```rust
pub fn can_shift(&self) -> bool
```

#### perf()

**Signature:**

```rust
pub fn perf(&self) -> String
```

---

#### Model

Thread-safe handle to a loaded model.

Cloning is cheap (Arc bump). Use from any thread.

### Methods

#### load_from_file()

**Signature:**

```rust
pub fn load_from_file(path: &str, params: &str) -> Model
```

#### as_ptr()

**Signature:**

```rust
pub fn as_ptr(&self) -> String
```

#### chat_template()

**Signature:**

```rust
pub fn chat_template(&self, name: Option<String>) -> Option<String>
```

#### desc()

**Signature:**

```rust
pub fn desc(&self) -> String
```

#### has_decoder()

**Signature:**

```rust
pub fn has_decoder(&self) -> bool
```

#### decoder_start_token()

**Signature:**

```rust
pub fn decoder_start_token(&self) -> Option<String>
```

#### has_encoder()

**Signature:**

```rust
pub fn has_encoder(&self) -> bool
```

#### is_diffusion()

**Signature:**

```rust
pub fn is_diffusion(&self) -> bool
```

#### is_hybrid()

**Signature:**

```rust
pub fn is_hybrid(&self) -> bool
```

#### is_recurrent()

**Signature:**

```rust
pub fn is_recurrent(&self) -> bool
```

#### token_to_piece()

**Signature:**

```rust
pub fn token_to_piece(&self, token: i32) -> String
```

#### tokenize()

**Signature:**

```rust
pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<String>
```

#### get_add_bos()

**Signature:**

```rust
pub fn get_add_bos(&self) -> bool
```

#### get_add_eos()

**Signature:**

```rust
pub fn get_add_eos(&self) -> bool
```

#### get_add_sep()

**Signature:**

```rust
pub fn get_add_sep(&self) -> bool
```

#### get_attr()

**Signature:**

```rust
pub fn get_attr(&self, token: &str) -> String
```

#### get_score()

**Signature:**

```rust
pub fn get_score(&self, token: &str) -> f32
```

#### get_text()

**Signature:**

```rust
pub fn get_text(&self, token: &str) -> String
```

#### is_control()

**Signature:**

```rust
pub fn is_control(&self, token: &str) -> bool
```

#### is_eog()

**Signature:**

```rust
pub fn is_eog(&self, token: &str) -> bool
```

#### n_tokens()

**Signature:**

```rust
pub fn n_tokens(&self) -> i32
```

#### vocab_type()

**Signature:**

```rust
pub fn vocab_type(&self) -> String
```

---

#### Sequence

A sequence handle. No lifetime parameters — holds a clone of the Context
handle and communicates with the context actor via messages.

Logits from the last `push()` are cached locally, so `sample()` and
`logits()` don't need a second round-trip.

### Methods

#### logits()

**Signature:**

```rust
pub fn logits(&self) -> Option<Vec<f32>>
```

#### is_empty()

**Signature:**

```rust
pub fn is_empty(&self) -> bool
```

#### push()

**Signature:**

```rust
pub fn push(&self, token: &str)
```

#### decode()

Re-decode the last token to refresh logits without pushing a new one.
Useful after `pop()`, `remove()`, or other mutations that invalidate logits.

**Signature:**

```rust
pub fn decode(&self)
```

#### pop()

**Signature:**

```rust
pub fn pop(&self) -> Option<String>
```

#### len()

**Signature:**

```rust
pub fn len(&self) -> usize
```

#### extend()

**Signature:**

```rust
pub fn extend(&self, tokens: Vec<String>)
```

#### get()

**Signature:**

```rust
pub fn get(&self, index: usize) -> Option<String>
```

#### remove()

**Signature:**

```rust
pub fn remove(&self, range: &str) -> bool
```

#### copy_to()

**Signature:**

```rust
pub fn copy_to(&self, other: Sequence, range: &str)
```

#### copy_from()

**Signature:**

```rust
pub fn copy_from(&self, other: Sequence, range: &str)
```

#### pos_min()

**Signature:**

```rust
pub fn pos_min(&self) -> String
```

#### pos_max()

**Signature:**

```rust
pub fn pos_max(&self) -> String
```

#### tokens()

**Signature:**

```rust
pub fn tokens(&self) -> Vec<String>
```

#### kv_remove()

**Signature:**

```rust
pub fn kv_remove(&self, range: &str) -> bool
```

#### kv_copy()

**Signature:**

```rust
pub fn kv_copy(&self, other: Sequence, range: &str)
```

#### kv_shift()

**Signature:**

```rust
pub fn kv_shift(&self, range: &str, delta: &str)
```

#### index()

**Signature:**

```rust
pub fn index(&self, index: usize) -> String
```

#### drop()

**Signature:**

```rust
pub fn drop(&self)
```

---

### Enums

#### DecodeError

| Value | Description |
|-------|-------------|
| `SlotNotFound` | Slot not found |
| `Aborted` | Aborted |
| `InvalidInput` | Invalid input |
| `FatalError` | Fatal error |

---
