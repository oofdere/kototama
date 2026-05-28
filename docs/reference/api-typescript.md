---
title: "TypeScript API Reference"
---

## TypeScript API Reference <span class="version-badge">v0.1.0</span>

### Types

#### Context

Handle to a running context actor. Clone + Send + Sync.

The actor thread is stopped when the last clone is dropped.

### Methods

#### new()

**Signature:**

```typescript
static new(model: Model, params: string): Context
```

#### sequence()

**Signature:**

```typescript
sequence(): Sequence | null
```

#### freeSlots()

**Signature:**

```typescript
freeSlots(): number
```

#### nCtx()

**Signature:**

```typescript
nCtx(): number
```

#### canShift()

**Signature:**

```typescript
canShift(): boolean
```

#### perf()

**Signature:**

```typescript
perf(): string
```

---

#### Model

Thread-safe handle to a loaded model.

Cloning is cheap (Arc bump). Use from any thread.

### Methods

#### loadFromFile()

**Signature:**

```typescript
static loadFromFile(path: string, params: string): Model
```

#### asPtr()

**Signature:**

```typescript
asPtr(): string
```

#### chatTemplate()

**Signature:**

```typescript
chatTemplate(name: string): string | null
```

#### desc()

**Signature:**

```typescript
desc(): string
```

#### hasDecoder()

**Signature:**

```typescript
hasDecoder(): boolean
```

#### decoderStartToken()

**Signature:**

```typescript
decoderStartToken(): string | null
```

#### hasEncoder()

**Signature:**

```typescript
hasEncoder(): boolean
```

#### isDiffusion()

**Signature:**

```typescript
isDiffusion(): boolean
```

#### isHybrid()

**Signature:**

```typescript
isHybrid(): boolean
```

#### isRecurrent()

**Signature:**

```typescript
isRecurrent(): boolean
```

#### tokenToPiece()

**Signature:**

```typescript
tokenToPiece(token: number): string
```

#### tokenize()

**Signature:**

```typescript
tokenize(text: string, addSpecial: boolean, parseSpecial: boolean): Array<string>
```

#### getAddBos()

**Signature:**

```typescript
getAddBos(): boolean
```

#### getAddEos()

**Signature:**

```typescript
getAddEos(): boolean
```

#### getAddSep()

**Signature:**

```typescript
getAddSep(): boolean
```

#### getAttr()

**Signature:**

```typescript
getAttr(token: string): string
```

#### getScore()

**Signature:**

```typescript
getScore(token: string): number
```

#### getText()

**Signature:**

```typescript
getText(token: string): string
```

#### isControl()

**Signature:**

```typescript
isControl(token: string): boolean
```

#### isEog()

**Signature:**

```typescript
isEog(token: string): boolean
```

#### nTokens()

**Signature:**

```typescript
nTokens(): number
```

#### vocabType()

**Signature:**

```typescript
vocabType(): string
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

```typescript
logits(): Array<number> | null
```

#### isEmpty()

**Signature:**

```typescript
isEmpty(): boolean
```

#### push()

**Signature:**

```typescript
push(token: string): void
```

#### decode()

Re-decode the last token to refresh logits without pushing a new one.
Useful after `pop()`, `remove()`, or other mutations that invalidate logits.

**Signature:**

```typescript
decode(): void
```

#### pop()

**Signature:**

```typescript
pop(): string | null
```

#### len()

**Signature:**

```typescript
len(): number
```

#### extend()

**Signature:**

```typescript
extend(tokens: Array<string>): void
```

#### get()

**Signature:**

```typescript
get(index: number): string | null
```

#### remove()

**Signature:**

```typescript
remove(range: string): boolean
```

#### copyTo()

**Signature:**

```typescript
copyTo(other: Sequence, range: string): void
```

#### copyFrom()

**Signature:**

```typescript
copyFrom(other: Sequence, range: string): void
```

#### posMin()

**Signature:**

```typescript
posMin(): string
```

#### posMax()

**Signature:**

```typescript
posMax(): string
```

#### tokens()

**Signature:**

```typescript
tokens(): Array<string>
```

#### kvRemove()

**Signature:**

```typescript
kvRemove(range: string): boolean
```

#### kvCopy()

**Signature:**

```typescript
kvCopy(other: Sequence, range: string): void
```

#### kvShift()

**Signature:**

```typescript
kvShift(range: string, delta: string): void
```

#### index()

**Signature:**

```typescript
index(index: number): string
```

#### drop()

**Signature:**

```typescript
drop(): void
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
