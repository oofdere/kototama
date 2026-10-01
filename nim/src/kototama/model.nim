## Model: a loaded GGUF file and its vocabulary.
##
## This is the Nim counterpart of `src/model.rs` + `src/vocab.rs`. A `Model` is
## a shared `ref` (like Rust's `Arc<ModelInner>`): copying it is free, all
## copies refer to the same loaded weights, and the weights are freed when the
## last copy is dropped.
##
## Two ownership facts shape the types here:
##
##   1. The model owns a `Backend` handle, so llama.cpp's global state stays
##      initialized for as long as any model exists.
##   2. `llama_model_free` in this llama.cpp version is a plain `delete` with no
##      refcounting, so every consumer must keep the model alive itself. This
##      module holds the handle; `Context` holds a `Model` reference for the
##      same reason. (The Rust crate's `Context` does not, which is a latent
##      use-after-free there: dropping the last `Model` while a `Context` still
##      runs leaves that context pointing into freed weights. See COMPARISON.md.)

import std/options
import ./raw, ./backend, ./types, ./errors

type
  Model* = ref ModelObj
    ## Shared handle to a loaded model. Cheap to copy.

  ModelObj = object
    handle: ptr raw.LlamaModel
    vocab: ptr raw.LlamaVocab
    backend: Backend

proc `=destroy`(self: var ModelObj) =
  if not self.handle.isNil:
    llamaModelFree(self.handle)
    self.handle = nil
  # A custom `=destroy` replaces the compiler's field destruction entirely
  # (unlike Rust's Drop, which runs after the fields are gone), so every owned
  # field must be released by hand: dropping the last model handle is what
  # shuts llama.cpp's global state down.
  self.backend = nil

proc close*(self: Model) =
  ## Free the underlying weights now instead of at scope exit.
  ##
  ## Idempotent. Any context created from this model must already be closed;
  ## after closing, the model must not be used again. (Rust has no counterpart:
  ## its `Drop` runs whenever the Arc count reaches zero.)
  if not self.handle.isNil:
    llamaModelFree(self.handle)
    self.handle = nil
    self.vocab = nil
    self.backend = nil

proc loadFromFile*(path: string; params = llamaModelDefaultParams()): Model =
  ## Load a model from a GGUF file.
  ##
  ## `params` comes from `raw.llamaModelDefaultParams()`; set fields on it
  ## (such as `nGpuLayers`) before passing it in.
  let backend = acquire()
  let handle = llamaModelLoadFromFile(cstring(path), params)
  if handle.isNil:
    raise newException(ModelLoadError, "failed to load model: " & path)
  result = Model(handle: handle, vocab: llamaModelGetVocab(handle), backend: backend)

proc handle*(self: Model): ptr raw.LlamaModel =
  ## The raw llama.cpp pointer. Escape hatch for advanced use.
  self.handle

proc vocabHandle*(self: Model): ptr raw.LlamaVocab =
  ## The raw vocabulary pointer.
  self.vocab

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

proc desc*(self: Model): string =
  ## A short human-readable description of the model, e.g.
  ## `"llama 7B Q4_0 (Medium)"`. Empty if llama.cpp cannot describe it.
  ##
  ## The two-call pattern is the C API's buffer sizing protocol: a null buffer
  ## of size 0 answers how many bytes the description needs.
  let needed = llamaModelDesc(self.handle, nil, 0)
  if needed <= 0:
    return ""
  var buf = newString(needed + 1)   # + 1 for the NUL terminator C writes.
  let written = llamaModelDesc(self.handle, cstring(buf), (needed + 1).LlamaSizeT)
  if written <= 0:
    return ""
  buf.setLen(written)   # llama_model_desc answers the length written, sans NUL.
  buf

proc chatTemplate*(self: Model; name = ""): Option[string] =
  ## The model's Jinja chat template, or `none` when it has none.
  ## `name` selects a named template; empty means the default one.
  let key = if name.len == 0: nil else: cstring(name)
  let s = llamaModelChatTemplate(self.handle, key)
  if s.isNil:
    none(string)
  else:
    some($s)

proc hasEncoder*(self: Model): bool = llamaModelHasEncoder(self.handle)
proc hasDecoder*(self: Model): bool = llamaModelHasDecoder(self.handle)
proc isRecurrent*(self: Model): bool = llamaModelIsRecurrent(self.handle)
proc isHybrid*(self: Model): bool = llamaModelIsHybrid(self.handle)
proc isDiffusion*(self: Model): bool = llamaModelIsDiffusion(self.handle)

proc decoderStartToken*(self: Model): Option[Token] =
  ## The token a decoder starts from, for encoder-decoder models.
  let token = llamaModelDecoderStartToken(self.handle)
  if token == LlamaTokenNull: none(Token) else: some(token)

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

proc tokenize*(self: Model; text: string; addSpecial, parseSpecial: bool): seq[Token] =
  ## Split text into token ids.
  ##
  ## `addSpecial` prepends BOS (and appends EOS) when the model wants it;
  ## `parseSpecial` lets control strings like `<s>` tokenize as special tokens
  ## instead of ordinary text.
  ##
  ## The two-call pattern is llama.cpp's buffer sizing protocol: a call with
  ## room for zero tokens answers how many tokens the text *would* need.
  let needed = llamaTokenize(self.vocab, cstring(text), text.len.int32,
                             nil, 0, addSpecial, parseSpecial)
  if needed >= 0:
    # Fits in zero slots: the text tokenizes to nothing at all.
    return @[]
  result = newSeq[Token](-needed)
  let n = llamaTokenize(self.vocab, cstring(text), text.len.int32,
                        addr result[0], result.len.int32, addSpecial, parseSpecial)
  if n < 0:
    raise newException(KototamaError, "tokenize failed for text: " & text)
  result.setLen(n)

proc tokenToPiece*(self: Model; token: Token): string =
  ## Render a single token id back to the text it represents.
  ##
  ## Most pieces fit in 64 bytes; the rare longer one is handled by growing the
  ## buffer to the size llama.cpp asks for.
  var buf = newString(64)
  while true:
    let n = llamaTokenToPiece(self.vocab, token, cstring(buf), buf.len.int32, 0, true)
    if n >= 0:
      buf.setLen(n)
      return buf
    if n == low(int32):
      raise newException(KototamaError, "token_to_piece overflow for token " & $token)
    buf.setLen(-n)   # Negative answer: the size the piece actually needs.

# ---------------------------------------------------------------------------
# Vocabulary queries
# ---------------------------------------------------------------------------

type
  VocabType* = enum
    ## The tokenizer family a model uses. Mirrors `enum llama_vocab_type`.
    vtNone, vtSpm, vtBpe, vtWpm, vtUgm, vtRwkv, vtPlamo2

  TokenAttr* = enum
    ## What kind of thing a token is. The *values* mirror the bit positions in
    ## `enum llama_token_attr` (1 shl ord), so a set of these can be produced
    ## from the C bitfield directly.
    taUnknown = 0
    taUnused = 1
    taNormal = 2
    taControl = 3
    taUserDefined = 4
    taByte = 5
    taNormalized = 6
    taLstrip = 7
    taRstrip = 8
    taSingleWord = 9

proc nTokens*(self: Model): int32 =
  ## Size of the vocabulary.
  llamaVocabNTokens(self.vocab)

proc vocabType*(self: Model): VocabType =
  VocabType(llamaVocabType(self.vocab))

proc getAddBos*(self: Model): bool = llamaVocabGetAddBos(self.vocab)
proc getAddEos*(self: Model): bool = llamaVocabGetAddEos(self.vocab)
proc getAddSep*(self: Model): bool = llamaVocabGetAddSep(self.vocab)

proc getScore*(self: Model; token: Token): float32 =
  ## The token's merge score, as recorded in the vocabulary.
  llamaVocabGetScore(self.vocab, token).float32

proc getText*(self: Model; token: Token): string =
  ## The raw text of a vocabulary entry (before rendering rules apply).
  $llamaVocabGetText(self.vocab, token)

proc isControl*(self: Model; token: Token): bool = llamaVocabIsControl(self.vocab, token)

proc isEog*(self: Model; token: Token): bool =
  ## "End of generation": true for tokens that stop a generation, such as EOS.
  llamaVocabIsEog(self.vocab, token)

proc getAttr*(self: Model; token: Token): set[TokenAttr] =
  ## Attribute bits of a vocabulary entry.
  result = {}
  let bits = llamaVocabGetAttr(self.vocab, token).uint32
  for attr in TokenAttr:
    if (bits and (1'u32 shl attr.ord)) != 0:
      result.incl(attr)

# Special-token accessors. Each answers `none` when the model lacks that role.
# This mirrors the `token_option!` macro in Rust's src/vocab.rs.

template tokenOption(name: untyped, ffiProc: untyped) =
  proc name*(self: Model): Option[Token] =
    let token = ffiProc(self.vocab)
    if token == LlamaTokenNull: none(Token) else: some(token)

tokenOption bosToken, llamaVocabBos
tokenOption eosToken, llamaVocabEos
tokenOption eotToken, llamaVocabEot
tokenOption clsToken, llamaVocabCls
tokenOption sepToken, llamaVocabSep
tokenOption nlToken, llamaVocabNl
tokenOption padToken, llamaVocabPad
tokenOption maskToken, llamaVocabMask
tokenOption fimPreToken, llamaVocabFimPre
tokenOption fimSufToken, llamaVocabFimSuf
tokenOption fimMidToken, llamaVocabFimMid
tokenOption fimPadToken, llamaVocabFimPad
tokenOption fimRepToken, llamaVocabFimRep
tokenOption fimSepToken, llamaVocabFimSep
