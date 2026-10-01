## Sampler tests -- the Nim counterpart of `tests/sampler.rs`.
##
## Run with: `nim c -r --path:src tests/tsamplers.nim`

import std/unittest
import ../src/kototama
import testutil

suite "Greedy":
  test "greedy picks argmax":
    let g = Greedy.new()
    check g.sample([0.1'f32, 0.9, 0.5, 0.2]) == 1

  test "greedy tiebreak is last":
    # Ties resolve to the last maximum -- pinned behavior.
    let g = Greedy.new()
    check g.sample([5.0'f32, 5.0, 5.0]) == 2

suite "Temperature":
  test "temperature zero is identity":
    let t = Temperature.new(0.0)
    let res = t.apply([1.0'f32, 3.0, 2.0])
    check res == @[1.0'f32, 3.0, 2.0]

  test "temperature scales logits":
    let t = Temperature.new(2.0)
    let res = t.apply([1.0'f32, 2.0, 4.0])
    check abs(res[0] - 0.5) < 1e-6
    check abs(res[1] - 1.0) < 1e-6
    check abs(res[2] - 2.0) < 1e-6

  test "temperature preserves argmax":
    let t = Temperature.new(0.8)
    check t.sample([0.1'f32, 0.9, 0.5]) == 1

suite "MinP":
  test "min_p masks below threshold":
    let m = MinP.new(0.5, 0)
    let res = m.apply([4.0'f32, 3.5, 3.0, 2.0])
    check res[0] != NegInf   # max survives
    check res[1] != NegInf   # 3.5 >= thresh survives
    check isMasked(res[2])   # 3.0 masked
    check isMasked(res[3])   # 2.0 masked

  test "min_p min_keep floor forces survivors":
    # p=0.01 alone would keep only the max; min_keep forces 4.
    let m = MinP.new(0.01, 4)
    let res = m.apply([10.0'f32, 5.0, 4.0, 1.0, 0.0])
    check finiteCount(res) == 4

  test "min_p one keeps only the max":
    let m = MinP.new(1.0, 0)
    let res = m.apply([1.0'f32, 3.0, 3.0, 2.0])
    check res[1] != NegInf and res[2] != NegInf   # max(es) survive
    check isMasked(res[0])
    check isMasked(res[3])

suite "Dist":
  test "dist returns valid token":
    let d = Dist.new(42)
    let token = d.sample([0.1'f32, 0.5, 0.3, 0.2])
    check token in 0 .. 3

  test "dist is deterministic for same seed":
    let logits = @[1.0'f32, 2.0, 0.5, 3.0, 1.5]
    let a = Dist.new(99)
    let b = Dist.new(99)
    check a.sample(logits) == b.sample(logits)

  test "dist advances state across calls":
    let logits = @[1.0'f32, 2.0, 0.5, 3.0, 1.5]
    let d = Dist.new(7)
    var seen: set[int16] = {}
    for _ in 0 ..< 10:
      seen.incl(d.sample(logits).int16)
    check seen.card > 1   # RNG should advance, producing varied draws

  test "dist never picks masked tokens":
    let d = Dist.new(1)
    let logits = @[1.0'f32, NegInf, NegInf, 0.5]
    for _ in 0 ..< 20:
      let t = d.sample(logits)
      check t == 0 or t == 3

suite "name()":
  test "name returns short type name":
    check Temperature.new(1.0).name() == "Temperature"
    check MinP.new(0.1, 1).name() == "MinP"
    check Greedy.new().name() == "Greedy"
    check Dist.new(0).name() == "Dist"
    check TopK.new(1).name() == "TopK"
    check TopP.new(0.9, 1).name() == "TopP"
    check Typical.new(0.9, 1).name() == "Typical"
    check TopNSigma.new(1.0).name() == "TopNSigma"
    check Xtc.new(0.1, 0.1, 1, 1).name() == "Xtc"
    check Chain.new().name() == "Chain"

suite "TopK":
  test "top_k masks all but k":
    let k = TopK.new(2)
    let res = k.apply([4.0'f32, 3.0, 2.0, 1.0, 0.0])
    check finiteCount(res) == 2   # only top-2 survive
    check res[0] != NegInf and res[1] != NegInf
    check isMasked(res[3])

  test "top_k keeps ties at threshold":
    # 5.0 appears twice; k=2 keeps both ties (>= threshold).
    let k = TopK.new(2)
    let res = k.apply([5.0'f32, 4.0, 5.0, 1.0])
    check finiteCount(res) == 2
    check res[0] != NegInf and res[2] != NegInf

  test "top_k non-positive is noop":
    let k = TopK.new(0)
    let res = k.apply([4.0'f32, 3.0, 2.0])
    check finiteCount(res) == 3

  test "top_k larger than vocab is noop":
    let k = TopK.new(100)
    let res = k.apply([4.0'f32, 3.0, 2.0])
    check finiteCount(res) == 3

suite "TopP":
  test "top_p one is noop":
    let p = TopP.new(1.0, 1)
    let res = p.apply([1.0'f32, 2.0, 3.0, 4.0])
    check finiteCount(res) == 4

  test "top_p keeps nucleus":
    # Heavily peaked: argmax alone exceeds p=0.9.
    let p = TopP.new(0.9, 1)
    let res = p.apply([10.0'f32, 1.0, 1.0, 1.0])
    check finiteCount(res) == 1   # only the max clears p=0.9
    check res[0] != NegInf

  test "top_p respects min_keep":
    # p would keep only 1, but min_keep forces at least 3.
    let p = TopP.new(0.1, 3)
    let res = p.apply([10.0'f32, 1.0, 1.0, 1.0])
    check finiteCount(res) >= 3

suite "Typical":
  test "typical one is noop":
    let t = Typical.new(1.0, 1)
    let res = t.apply([1.0'f32, 2.0, 3.0, 4.0])
    check finiteCount(res) == 4

  test "typical masks some tokens":
    let t = Typical.new(0.5, 1)
    let res = t.apply([0.5'f32, 1.0, 0.2, 3.0, 0.1])
    check finiteCount(res) < 5   # should mask at least one token
    check finiteCount(res) >= 1  # at least one token survives

suite "TopNSigma":
  test "top_n_sigma non-positive is noop":
    let s = TopNSigma.new(0.0)
    let res = s.apply([1.0'f32, 2.0, 3.0, 4.0])
    check finiteCount(res) == 4

  test "top_n_sigma masks outliers":
    # A tight cluster plus a far-below outlier.
    let s = TopNSigma.new(1.0)
    let res = s.apply([10.0'f32, 9.9, 10.1, -50.0])
    check isMasked(res[3])       # outlier far below the cluster is masked
    check res[0] != NegInf and res[1] != NegInf and res[2] != NegInf

  test "top_n_sigma respects already masked":
    # -inf entries must not corrupt the mean/std: the finite entries should
    # match a run that omits the -inf slot entirely.
    let s = TopNSigma.new(1.0)
    let withInf = s.apply([10.0'f32, 9.9, 10.1, NegInf])
    let without = s.apply([10.0'f32, 9.9, 10.1])
    check withInf[0] == without[0]
    check withInf[1] == without[1]
    check withInf[2] == without[2]
    check isMasked(withInf[3])

suite "XTC":
  test "xtc inert when threshold too high":
    # threshold > 0.5 disables the sampler.
    let x = Xtc.new(1.0, 0.9, 1, 1)
    let res = x.apply([10.0'f32, 9.0, 8.0, 7.0])
    check finiteCount(res) == 4

  test "xtc can mask top token":
    # probability 1.0 (always triggers), threshold 0.1: the top two tokens
    # clear the threshold and are masked; the rest survive.
    let x = Xtc.new(1.0, 0.1, 1, 1)
    let res = x.apply([2.0'f32, 2.0, 1.0, 0.0])
    check isMasked(res[0])   # top token masked
    check isMasked(res[1])   # second token masked
    check res[2] != NegInf   # third token survives

  test "xtc probability zero is noop":
    let x = Xtc.new(0.0, 0.1, 1, 1)
    let res = x.apply([10.0'f32, 9.0, 8.0, 7.0])
    check finiteCount(res) == 4

suite "Chain":
  test "chain applies transforms in order":
    # top_k(1) keeps only the max, so temperature is irrelevant -> argmax.
    let chain = Chain.new()
      .with(TopK.new(1))
      .with(Temperature.new(0.8))
      .with(Greedy.new())
    check chain.sample([1.0'f32, 5.0, 2.0, 3.0]) == 1

  test "chain dist returns valid token":
    let chain = Chain.new()
      .with(TopP.new(0.9, 1))
      .with(Temperature.new(0.8))
      .with(Dist.new(7))
    let token = chain.sample([1.0'f32, 2.0, 0.5, 3.0])
    check token in 0 .. 3

  test "chain apply_mut composes":
    let chain = Chain.new().with(TopK.new(2))
    var logits = @[4.0'f32, 3.0, 2.0, 1.0]
    chain.applyMut(logits)
    check finiteCount(logits) == 2

  test "chain empty is greedy":
    let chain = Chain.new()
    check chain.sample([0.1'f32, 0.9, 0.5]) == 1
