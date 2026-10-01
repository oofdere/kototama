## Sampling primitives: transforms over logits and token selectors.
##
## This is the Nim counterpart of `src/samplers/`. The design is the same one
## the Rust trait defines, and the doc comments there apply here too:
##
##   * A sampler either *transforms* logits in place (`applyMut`), *selects* a
##     token from them (`sample`), or both.
##   * `apply` runs the transform on a copy, leaving the caller's logits alone.
##   * `Chain` composes several samplers into one pipeline.
##
## The default `sample` applies the transform and picks the argmax, so a pure
## logit transformer (Temperature, TopK, ...) needs to implement `applyMut`
## only; a selector like `Greedy` or `Dist` overrides `sample` instead.
##
## Where the Rust code lives in eleven small files (one per sampler), Nim's
## module granularity makes one sectioned file easier to read end to end. The
## section order below matches the Rust file order.

import std/[algorithm, math, random, typetraits]
import ./types

proc argmax*(logits: openArray[float32]): Token =
  ## Index of the largest logit. Ties resolve to the *last* maximum, matching
  ## Rust's `Iterator::max_by` (which the tests pin explicitly).
  if logits.len == 0:
    raise newException(ValueError, "argmax of empty logits")
  var best = logits[0]
  result = 0
  for i in 1 ..< logits.len:
    if logits[i] >= best:
      best = logits[i]
      result = Token(i)

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

type
  Sampler* = ref object of RootObj
    ## Base type of every sampler. See the module docs for the contract.

method applyMut*(self: Sampler; logits: var openArray[float32]) {.base.} =
  ## Transform `logits` in place. The base implementation does nothing, so a
  ## token selector can override `sample` alone.
  discard

method sample*(self: Sampler; logits: openArray[float32]): Token {.base.} =
  ## Select a token from `logits`. The default implementation runs `applyMut`
  ## on a copy and returns the argmax, which is exactly right for transform-only
  ## samplers.
  var transformed = @logits
  self.applyMut(transformed)
  argmax(transformed)

proc apply*(self: Sampler; logits: openArray[float32]): seq[float32] =
  ## Run the sampler's transform on a copy of `logits` and return it.
  result = @logits
  self.applyMut(result)

proc sampleMut*(self: Sampler; logits: var openArray[float32]): Token =
  ## Transform `logits` in place, then select the argmax.
  self.applyMut(logits)
  argmax(logits)

proc name*[T: Sampler](s: T): string =
  ## The sampler's type name, e.g. `"TopK"`. Mirrors the Rust `Sampler::name`,
  ## which is likewise derived from the type.
  typetraits.name(T)

# ---------------------------------------------------------------------------
# Temperature (src/samplers/temperature.rs)
# ---------------------------------------------------------------------------

type
  Temperature* = ref object of Sampler
    ## Scale logits by `1 / temp`. `temp == 0.0` is a no-op -- use `Greedy` for
    ## deterministic decoding instead.
    temp*: float32

proc new*(T: typedesc[Temperature]; temp: float32): Temperature =
  Temperature(temp: temp)

method applyMut*(self: Temperature; logits: var openArray[float32]) =
  if self.temp == 0.0:
    return
  let invTemp = 1.0'f32 / self.temp
  for logit in logits.mitems:
    logit *= invTemp

# ---------------------------------------------------------------------------
# Greedy (src/samplers/greedy.rs)
# ---------------------------------------------------------------------------

type
  Greedy* = ref object of Sampler
    ## Greedy / argmax selection. `applyMut` does nothing.

proc new*(T: typedesc[Greedy]): Greedy = Greedy()

# Nothing to implement: the base `sample` is already apply + argmax.

# ---------------------------------------------------------------------------
# TopK (src/samplers/top_k.rs)
# ---------------------------------------------------------------------------

type
  TopK* = ref object of Sampler
    ## Keep only the `k` highest logits, masking the rest to `-inf`. Tokens tied
    ## at the threshold are kept, so the survivor set may hold more than `k`
    ## entries. `k <= 0` leaves the logits untouched.
    k*: int32

proc new*(T: typedesc[TopK]; k: int32): TopK = TopK(k: k)

method applyMut*(self: TopK; logits: var openArray[float32]) =
  if self.k <= 0:
    return
  let k = min(self.k.int, logits.len)
  if k >= logits.len:
    return

  # The threshold is the k-th largest logit (1-indexed): keep everything
  # greater-than-or-equal to it. A full sort is O(n log n) versus the O(n)
  # partial selection Rust uses, but vocab-sized arrays are small and the
  # intent reads far more clearly this way.
  var copy = @logits
  copy.sort(Descending)
  let thresh = copy[k - 1]

  for logit in logits.mitems:
    if logit < thresh:
      logit = NegInf

# ---------------------------------------------------------------------------
# MinP (src/samplers/min_p.rs)
# ---------------------------------------------------------------------------

type
  MinP* = ref object of Sampler
    ## Mask every logit more than `p` (in probability space) below the maximum.
    ## Keeps at least `minKeep` candidates.
    p*: float32
    minKeep*: Natural

proc new*(T: typedesc[MinP]; p: float32; minKeep: Natural): MinP =
  MinP(p: p, minKeep: minKeep)

method applyMut*(self: MinP; logits: var openArray[float32]) =
  if logits.len <= self.minKeep:
    return

  var logitMax = NegInf
  for logit in logits:
    if logit > logitMax:
      logitMax = logit

  # Everything within a factor of `p` of the max survives: p_max * p in
  # probability space is `ln(max) + ln(p)` in logit space.
  var thresh = logitMax + ln(self.p)

  if self.minKeep > 0:
    # Never drop below the minKeep-th best logit.
    var copy = @logits
    copy.sort(Descending)
    thresh = min(thresh, copy[self.minKeep - 1])

  for logit in logits.mitems:
    if logit < thresh:
      logit = NegInf

# ---------------------------------------------------------------------------
# TopP (src/samplers/top_p.rs)
# ---------------------------------------------------------------------------

type
  TopP* = ref object of Sampler
    ## Top-P (nucleus) sampling: keep the smallest set of highest-probability
    ## tokens whose cumulative softmax probability reaches `p`, always keeping at
    ## least `minKeep`. `p >= 1.0` is a no-op.
    p*: float32
    minKeep*: Natural

proc new*(T: typedesc[TopP]; p: float32; minKeep: Natural): TopP =
  TopP(p: p, minKeep: minKeep)

proc softmaxProbs(logits: openArray[float32]): seq[float32] =
  ## Numerically stable softmax. Returns one probability per logit.
  var maxLogit = NegInf
  for logit in logits:
    if logit > maxLogit:
      maxLogit = logit
  result = newSeq[float32](logits.len)
  var sum = 0.0'f32
  for i, logit in logits:
    result[i] = exp(logit - maxLogit)
    sum += result[i]
  for p in result.mitems:
    p /= sum

method applyMut*(self: TopP; logits: var openArray[float32]) =
  if self.p >= 1.0:
    return
  if logits.len == 0:
    return

  let probs = softmaxProbs(logits)

  # Walk candidates from most to least likely, keeping everything until the
  # cumulative probability reaches p. (The sort needs a capture-friendly view
  # of the logits, hence the local seq copy.)
  let lg = @logits
  var order = newSeq[int](lg.len)
  for i in 0 ..< lg.len: order[i] = i
  order.sort(proc (a, b: int): int = cmp(lg[b], lg[a]))

  var
    cum = 0.0'f32
    keep = newSeq[bool](logits.len)
  for rank, i in order:
    keep[i] = true
    cum += probs[i]
    if cum >= self.p and rank + 1 >= self.minKeep:
      break

  for i, logit in logits.mpairs:
    if not keep[i]:
      logit = NegInf

# ---------------------------------------------------------------------------
# Typical (src/samplers/typical.rs)
# ---------------------------------------------------------------------------

type
  Typical* = ref object of Sampler
    ## Locally typical sampling: keep tokens whose negative-log-probability is
    ## closest to the distribution entropy, accumulating until the cumulative
    ## probability exceeds `p` (keeping at least `minKeep`). `p >= 1.0` is a
    ## no-op.
    ##
    ## Reference: Meister et al., "Typical Decoding for Natural Language
    ## Generation".
    p*: float32
    minKeep*: Natural

proc new*(T: typedesc[Typical]; p: float32; minKeep: Natural): Typical =
  Typical(p: p, minKeep: minKeep)

method applyMut*(self: Typical; logits: var openArray[float32]) =
  if self.p >= 1.0:
    return
  if logits.len == 0:
    return

  let probs = softmaxProbs(logits)

  # Entropy H = -sum(p * ln p).
  var entropy = 0.0'f32
  for p in probs:
    if p > 0.0:
      entropy += -p * ln(p)

  # Rank by |(-ln p) - H|, ascending: the most "typical" tokens first.
  var order = newSeq[int](probs.len)
  for i in 0 ..< probs.len: order[i] = i
  order.sort(proc (a, b: int): int =
    let sa = abs(-ln(probs[a]) - entropy)
    let sb = abs(-ln(probs[b]) - entropy)
    cmp(sa, sb))

  var
    cum = 0.0'f32
    keep = newSeq[bool](logits.len)
  for rank, i in order:
    keep[i] = true
    cum += probs[i]
    if cum > self.p and (self.minKeep == 0 or rank + 1 >= self.minKeep):
      break

  for i, logit in logits.mpairs:
    if not keep[i]:
      logit = NegInf

# ---------------------------------------------------------------------------
# TopNSigma (src/samplers/top_n_sigma.rs)
# ---------------------------------------------------------------------------

type
  TopNSigma* = ref object of Sampler
    ## Mask every logit more than `n` standard deviations below the maximum,
    ## considering only non-masked logits. `n <= 0` is a no-op.
    ##
    ## Reference: "Simplifying Top-p Sampling with Top-nσ".
    n*: float32

proc new*(T: typedesc[TopNSigma]; n: float32): TopNSigma =
  TopNSigma(n: n)

method applyMut*(self: TopNSigma; logits: var openArray[float32]) =
  if self.n <= 0.0 or logits.len <= 1:
    return

  # -inf entries are already masked by an earlier sampler and must not skew
  # the statistics, so gather them out first.
  var
    max = NegInf
    sum = 0.0'f32
    count = 0
  for logit in logits:
    if logit != NegInf:
      if logit > max: max = logit
      sum += logit
      inc count
  if count == 0:
    return

  let mean = sum / count.float32
  var varSum = 0.0'f32
  for logit in logits:
    if logit != NegInf:
      varSum += (logit - mean) * (logit - mean)
  let std = sqrt(varSum / count.float32)

  let thresh = max - self.n * std
  for logit in logits.mitems:
    if logit < thresh:
      logit = NegInf

# ---------------------------------------------------------------------------
# Xtc (src/samplers/xtc.rs)
# ---------------------------------------------------------------------------

type
  Xtc* = ref object of Sampler
    ## XTC (eXclude Top Choices): with probability `probability`, drop the
    ## tokens whose softmax probability is `>= threshold`, keeping the rest.
    ## A diversity sampler: it only acts when at least two tokens clear the
    ## threshold and at least `minKeep` would survive.
    ##
    ## As in llama.cpp, the sampler is inert when `probability <= 0` or
    ## `threshold > 0.5`.
    probability*: float32
    threshold*: float32
    minKeep*: Natural
    rng: Rand

proc new*(T: typedesc[Xtc]; probability, threshold: float32;
          minKeep: Natural; seed: int64): Xtc =
  Xtc(probability: probability, threshold: threshold, minKeep: minKeep,
      rng: initRand(seed))

method applyMut*(self: Xtc; logits: var openArray[float32]) =
  if self.probability <= 0.0 or self.threshold > 0.5 or logits.len < 2:
    return

  if rand(self.rng, 1.0).float32 > self.probability:
    return

  let probs = softmaxProbs(logits)

  let lg = @logits
  var order = newSeq[int](lg.len)
  for i in 0 ..< lg.len: order[i] = i
  order.sort(proc (a, b: int): int = cmp(lg[b], lg[a]))

  # posLast = last rank (consecutive from the top) whose prob >= threshold.
  var posLast = 0
  for rank, i in order:
    if probs[i] >= self.threshold:
      posLast = rank
    else:
      break

  # Mask the top `posLast` tokens, provided at least one is removed and enough
  # survive.
  if posLast > 0 and logits.len - posLast >= self.minKeep:
    for i in order[0 ..< posLast]:
      logits[i] = NegInf

# ---------------------------------------------------------------------------
# Dist (src/samplers/dist.rs)
# ---------------------------------------------------------------------------

type
  Dist* = ref object of Sampler
    ## Multinomial (weighted-random) selection from the softmax distribution.
    ## `applyMut` is a no-op; the selection happens in `sample`.
    ##
    ## Note: Nim's `std/random` generator is not bit-identical to Rust's
    ## `StdRng`, so equal seeds do not produce equal draws across languages.
    ## Determinism holds within one language: same seed, same sequence.
    rng: Rand

proc new*(T: typedesc[Dist]; seed: int64): Dist =
  Dist(rng: initRand(seed))

method sample*(self: Dist; logits: openArray[float32]): Token =
  let probs = softmaxProbs(logits)
  let r = rand(self.rng, 1.0).float32
  var acc = 0.0'f32
  for id, p in probs:
    acc += p
    if acc >= r:
      return Token(id)
  Token(probs.len - 1)   # Rounding can push the last token over; fall back.

# ---------------------------------------------------------------------------
# Chain (src/samplers/chain.rs)
# ---------------------------------------------------------------------------

type
  Chain* = ref object of Sampler
    ## A pipeline of samplers applied in order.
    ##
    ## Logit-transforming samplers (TopK, Temperature, ...) run first and the
    ## final sampler is expected to be a selector such as `Greedy` or `Dist`:
    ##
    ## ```
    ## let chain = Chain.new()
    ##   .with(TopK.new(40))
    ##   .with(Temperature.new(0.8))
    ##   .with(Dist.new(42));
    ## let token = chain.sample(logits)
    ## ```
    samplers: seq[Sampler]

proc new*(T: typedesc[Chain]): Chain = Chain(samplers: @[])

proc push*(self: Chain; sampler: Sampler): Chain =
  ## Append a sampler, returning `self` for chained calls.
  self.samplers.add(sampler)
  self

proc with*(self: Chain; sampler: Sampler): Chain =
  ## Builder-style append. Identical to `push`; kept for the fluent form that
  ## the Rust API uses.
  self.push(sampler)

proc len*(self: Chain): int = self.samplers.len
proc isEmpty*(self: Chain): bool = self.samplers.len == 0

method applyMut*(self: Chain; logits: var openArray[float32]) =
  for sampler in self.samplers:
    sampler.applyMut(logits)

method sample*(self: Chain; logits: openArray[float32]): Token =
  var transformed = @logits
  for sampler in self.samplers:
    sampler.applyMut(transformed)
  case self.samplers.len
  of 0:
    argmax(transformed)
  else:
    # The last sampler is expected to be a selector (Greedy/Dist). For those,
    # its own `sample` runs no extra transform.
    self.samplers[^1].sample(transformed)
