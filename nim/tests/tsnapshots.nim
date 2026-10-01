## Snapshot tests -- the Nim counterpart of `tests/snapshots.rs`.
##
## These pin exact token ids against `tests/snapshots/*.snap`, the insta
## snapshots the Rust suite records. Reading the same files means both
## languages are held to the same expectations, and a llama.cpp bump that
## changes tokenization shows up here in both at once.
##
## Run with: `nim c -r --path:src tests/tsnapshots.nim`

import std/[options, os, strutils, unittest]
import ../src/kototama
import testutil

const snapshotDir = currentSourcePath().parentDir / "../../tests/snapshots"
  ## Resolved from this file's location so tests pass from any working
  ## directory. These are the Rust suite's own insta snapshots.

proc readSnapshot(name: string): seq[Token] =
  ## Parse an insta YAML snapshot: a header followed by `- <id>` lines.
  let path = snapshotDir / name
  doAssert fileExists(path), "missing snapshot: " & path
  for line in path.readFile().splitLines():
    let t = line.strip()
    if t.startsWith("- "):
      let value = parseInt(t[2 .. ^1])
      result.add(Token(value))

proc greedyGenerate(prompt: string; nTokens: int): seq[Token] =
  ## Greedy (argmax) generation -- deterministic for a given model and
  ## llama.cpp build, which is exactly what makes it snapshot-worthy.
  let (model, params) = loadModelAndContext()
  let ctx = Context.new(model, params)
  var seq = ctx.sequence().get()
  seq.extend(model.tokenize(prompt, addSpecial = true, parseSpecial = false))
  for _ in 0 ..< nTokens:
    let token = argmaxToken(seq.logits().get())
    if model.isEog(token):
      break
    result.add(token)
    seq.push(token)

suite "Tokenization snapshots":
  test "snapshot_tokenize_hello_world":
    let model = loadModel()
    let tokens = model.tokenize("Hello, world!", false, false)
    check tokens == readSnapshot("snapshots__snapshot_tokenize_hello_world.snap")

  test "snapshot_tokenize_with_bos":
    let model = loadModel()
    let tokens = model.tokenize("Hello, world!", true, false)
    check tokens == readSnapshot("snapshots__snapshot_tokenize_with_bos.snap")

  test "snapshot_tokenize_multiline":
    let model = loadModel()
    let tokens = model.tokenize("line one\nline two\nline three", false, false)
    check tokens == readSnapshot("snapshots__snapshot_tokenize_multiline.snap")

  test "snapshot_tokenize_numbers":
    let model = loadModel()
    let tokens = model.tokenize("1, 2, 3, 4, 5", false, false)
    check tokens == readSnapshot("snapshots__snapshot_tokenize_numbers.snap")

suite "Generation snapshots":
  test "snapshot_generate_10_tokens":
    let tokens = greedyGenerate("Once upon a time", 10)
    check tokens == readSnapshot("snapshots__snapshot_generate_10_tokens.snap")

  test "snapshot_generate_numbers":
    let tokens = greedyGenerate("1, 2, 3,", 8)
    check tokens == readSnapshot("snapshots__snapshot_generate_numbers.snap")
