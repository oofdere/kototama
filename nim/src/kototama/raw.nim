## Raw FFI bindings to llama.cpp -- the Nim counterpart of the Rust `llama-sys` crate.
##
## Where Rust runs `bindgen` at build time to generate bindings from `llama.h`,
## Nim's foreign-function interface is good enough that the whole binding layer
## is a single readable file you can check against the header by eye. Every
## declaration below names a symbol in `llama.h`, and every `importc` struct uses
## the *real* C definition via the `header` pragma, so the C compiler resolves
## field accesses by name. A typo'd field or an incompatible pointer type is a
## build error, not a corrupt read.
##
## What is deliberately *not* here: no safe wrappers, no RAII, no ownership.
## That is `model.nim`, `context.nim`, and friends. This module only translates
## the C API one-for-one, so it stays boring and auditable.
##
## Safety contract: nearly every proc here touches raw pointers and unchecked
## memory. Callers must uphold llama.cpp's documented preconditions. Nothing in
## this module can enforce them.

import std/os

const
  thisDir = currentSourcePath().parentDir
  repoRoot = thisDir / ".." / ".." / ".."
  llamaCppDir = repoRoot / "llama-sys" / "llama.cpp"
    ## Both language bindings build against the same pinned llama.cpp submodule.
  llamaInclude* = llamaCppDir / "include"
  ggmlInclude* = llamaCppDir / "ggml" / "include"

  llamaLibDir* {.strdefine.} = repoRoot / "nim" / "llama-cpp-build" / "install" / "lib"
    ## Static libraries produced by `nim/build_llama_cpp.sh`. Override at compile
    ## time with `-d:llamaLibDir=/your/lib/dir` if you build llama.cpp elsewhere.

{.passC: "-I" & llamaInclude.}
{.passC: "-I" & ggmlInclude.}

# Link order matters for static archives: dependents before their dependencies.
# Note the spelling: `-llama` would parse as "-l lama" (a library called lama),
# so linking libllama takes the doubled `-lllama`.
{.passL: "-L" & llamaLibDir.}
{.passL: "-lllama".}
{.passL: "-lllama-common".}
{.passL: "-lggml".}
{.passL: "-lggml-cpu".}
{.passL: "-lggml-base".}

when defined(macosx):
  # Metal backend is compiled in on macOS, mirroring llama-sys/build.rs.
  {.passL: "-lggml-metal".}
  {.passL: "-lggml-blas".}
  {.passL: "-framework Metal".}
  {.passL: "-framework Foundation".}
  {.passL: "-framework Accelerate".}
  {.passL: "-lc++".}
elif defined(linux):
  # whole-archive keeps ggml's self-registering CPU backend from being dropped
  # by the linker as unreferenced. Same reason llama-sys/build.rs uses it.
  {.passL: "-Wl,--whole-archive -lggml-cpu -lggml -lggml-base -Wl,--no-whole-archive".}
  {.passL: "-lgomp".}
  {.passL: "-lstdc++".}

# ---------------------------------------------------------------------------
# Scalar aliases. These are typedefs in llama.h and carry no extra meaning here;
# the high-level modules give them the names `Token`, `Pos`, and `SeqId`.
# ---------------------------------------------------------------------------

type
  LlamaToken* = int32
  LlamaPos* = int32
  LlamaSeqId* = int32
  LlamaSizeT* = uint

const
  LlamaTokenNull* = -1.LlamaToken
  LlamaDefaultSeed* = 0xFFFFFFFF'u32

# ---------------------------------------------------------------------------
# Opaque handles. Only ever used behind a pointer; the C struct bodies are
# private to llama.cpp.
# ---------------------------------------------------------------------------

type
  LlamaModel* {.importc: "struct llama_model", header: "llama.h".} = object
  LlamaContext* {.importc: "struct llama_context", header: "llama.h".} = object
  LlamaVocab* {.importc: "struct llama_vocab", header: "llama.h".} = object
  LlamaMemoryObj* {.importc: "struct llama_memory_i", header: "llama.h".} = object
  LlamaMemory* = ptr LlamaMemoryObj

# ---------------------------------------------------------------------------
# Enums used at the boundary. The public vocab/token enums live in `model.nim`.
# ---------------------------------------------------------------------------

type
  GgmlLogLevel* = enum
    gllNone = 0
    gllDebug = 1
    gllInfo = 2
    gllWarn = 3
    gllError = 4
    gllCont = 5

  GgmlLogCallback* = proc (level: cint; text: cstring; userData: pointer) {.cdecl.}

# ---------------------------------------------------------------------------
# Structs. `importc` + `header` means Nim emits the field accesses against the
# real C definition, and `bycopy` passes them by value exactly like C does.
# Field order below mirrors llama.h so a diff against the header is trivial.
# ---------------------------------------------------------------------------

type
  # Each Nim field carries its C name via a field-level `importc` pragma, so
  # generated C code reads `params.n_seq_max` where Nim writes `params.nSeqMax`.
  LlamaModelParams* {.importc: "struct llama_model_params", header: "llama.h",
                     bycopy.} = object
    devices* {.importc.}: pointer
    tensorBuftOverrides* {.importc: "tensor_buft_overrides".}: pointer
    nGpuLayers* {.importc: "n_gpu_layers".}: int32
    splitMode* {.importc: "split_mode".}: cint
    mainGpu* {.importc: "main_gpu".}: int32
    tensorSplit* {.importc: "tensor_split".}: ptr cfloat
    progressCallback* {.importc: "progress_callback".}: pointer
    progressCallbackUserData* {.importc: "progress_callback_user_data".}: pointer
    kvOverrides* {.importc: "kv_overrides".}: pointer
    vocabOnly* {.importc: "vocab_only".}: bool
    useMmap* {.importc: "use_mmap".}: bool
    useDirectIo* {.importc: "use_direct_io".}: bool
    useMlock* {.importc: "use_mlock".}: bool
    checkTensors* {.importc: "check_tensors".}: bool
    useExtraBufts* {.importc: "use_extra_bufts".}: bool
    noHost* {.importc: "no_host".}: bool
    noAlloc* {.importc: "no_alloc".}: bool

  LlamaContextParams* {.importc: "struct llama_context_params", header: "llama.h",
                       bycopy.} = object
    nCtx* {.importc: "n_ctx".}: uint32
    nBatch* {.importc: "n_batch".}: uint32
    nUbatch* {.importc: "n_ubatch".}: uint32
    nSeqMax* {.importc: "n_seq_max".}: uint32
    nRsSeq* {.importc: "n_rs_seq".}: uint32
    nThreads* {.importc: "n_threads".}: int32
    nThreadsBatch* {.importc: "n_threads_batch".}: int32
    ctxType* {.importc: "ctx_type".}: cint
    ropeScalingType* {.importc: "rope_scaling_type".}: cint
    poolingType* {.importc: "pooling_type".}: cint
    attentionType* {.importc: "attention_type".}: cint
    flashAttnType* {.importc: "flash_attn_type".}: cint
    ropeFreqBase* {.importc: "rope_freq_base".}: cfloat
    ropeFreqScale* {.importc: "rope_freq_scale".}: cfloat
    yarnExtFactor* {.importc: "yarn_ext_factor".}: cfloat
    yarnAttnFactor* {.importc: "yarn_attn_factor".}: cfloat
    yarnBetaFast* {.importc: "yarn_beta_fast".}: cfloat
    yarnBetaSlow* {.importc: "yarn_beta_slow".}: cfloat
    yarnOrigCtx* {.importc: "yarn_orig_ctx".}: uint32
    defragThold* {.importc: "defrag_thold".}: cfloat
    cbEval* {.importc: "cb_eval".}: pointer
    cbEvalUserData* {.importc: "cb_eval_user_data".}: pointer
    typeK* {.importc: "type_k".}: cint
    typeV* {.importc: "type_v".}: cint
    abortCallback* {.importc: "abort_callback".}: pointer
    abortCallbackData* {.importc: "abort_callback_data".}: pointer
    embeddings* {.importc.}: bool
    offloadKqv* {.importc: "offload_kqv".}: bool
    noPerf* {.importc: "no_perf".}: bool
    opOffload* {.importc: "op_offload".}: bool
    swaFull* {.importc: "swa_full".}: bool
    kvUnified* {.importc: "kv_unified".}: bool
    samplers* {.importc.}: pointer
    nSamplers* {.importc: "n_samplers".}: LlamaSizeT

  LlamaBatch* {.importc: "struct llama_batch", header: "llama.h", bycopy.} = object
    nTokens* {.importc: "n_tokens".}: int32
    token* {.importc.}: ptr LlamaToken
    embd* {.importc.}: ptr cfloat
    pos* {.importc.}: ptr LlamaPos
    nSeqId* {.importc: "n_seq_id".}: ptr int32
    seqId* {.importc: "seq_id".}: ptr ptr LlamaSeqId
    logits* {.importc.}: ptr int8

  LlamaPerfContextData* {.importc: "struct llama_perf_context_data",
                         header: "llama.h", bycopy.} = object
    tStartMs* {.importc: "t_start_ms".}: float64
    tLoadMs* {.importc: "t_load_ms".}: float64
    tPEvalMs* {.importc: "t_p_eval_ms".}: float64
    tEvalMs* {.importc: "t_eval_ms".}: float64
    nPEval* {.importc: "n_p_eval".}: int32
    nEval* {.importc: "n_eval".}: int32
    nReused* {.importc: "n_reused".}: int32

# Layout tripwires. The `importc` structs above resolve their fields against
# the real llama.h definition at C compile time, but the field *list* is still
# ours to maintain. Nim's `sizeof` cannot see through `importc`, so the checks
# are stated in C where the truth lives: if the pinned llama.cpp ever changes
# one of these layouts, the build stops instead of corrupting memory. The
# numbers were measured against the pinned submodule; update both together.
{.emit: """
#include "llama.h"
_Static_assert(sizeof(struct llama_model_params)      ==  72, "llama_model_params layout changed -- update raw.nim");
_Static_assert(sizeof(struct llama_context_params)    == 144, "llama_context_params layout changed -- update raw.nim");
_Static_assert(sizeof(struct llama_batch)             ==  56, "llama_batch layout changed -- update raw.nim");
_Static_assert(sizeof(struct llama_perf_context_data) ==  48, "llama_perf_context_data layout changed -- update raw.nim");
""".}

# ---------------------------------------------------------------------------
# Backend lifecycle. llama.cpp keeps global state, so init/free are refcounted
# in `backend.nim`, never called directly by library users.
# ---------------------------------------------------------------------------

proc ggmlBackendLoadAll*() {.importc: "ggml_backend_load_all", header: "ggml-backend.h".}
proc llamaBackendInit*() {.importc: "llama_backend_init", header: "llama.h".}
proc llamaBackendFree*() {.importc: "llama_backend_free", header: "llama.h".}

proc ggmlLogSet*(cb: GgmlLogCallback; userData: pointer)
  {.importc: "ggml_log_set", header: "ggml.h".}

# ---------------------------------------------------------------------------
# Default parameter structs. Always start from these rather than a zeroed
# struct: llama.cpp's defaults are not all-zero.
# ---------------------------------------------------------------------------

proc llamaModelDefaultParams*(): LlamaModelParams
  {.importc: "llama_model_default_params", header: "llama.h".}
proc llamaContextDefaultParams*(): LlamaContextParams
  {.importc: "llama_context_default_params", header: "llama.h".}

# ---------------------------------------------------------------------------
# Model lifecycle and metadata.
# ---------------------------------------------------------------------------

proc llamaModelLoadFromFile*(path: cstring; params: LlamaModelParams): ptr LlamaModel
  {.importc: "llama_model_load_from_file", header: "llama.h".}
proc llamaModelFree*(model: ptr LlamaModel)
  {.importc: "llama_model_free", header: "llama.h".}
proc llamaModelGetVocab*(model: ptr LlamaModel): ptr LlamaVocab
  {.importc: "llama_model_get_vocab", header: "llama.h".}

proc llamaModelDesc*(model: ptr LlamaModel; buf: cstring; bufSize: LlamaSizeT): int32
  {.importc: "llama_model_desc", header: "llama.h".}
proc llamaModelChatTemplate*(model: ptr LlamaModel; name: cstring): cstring
  {.importc: "llama_model_chat_template", header: "llama.h".}

proc llamaModelHasEncoder*(model: ptr LlamaModel): bool
  {.importc: "llama_model_has_encoder", header: "llama.h".}
proc llamaModelHasDecoder*(model: ptr LlamaModel): bool
  {.importc: "llama_model_has_decoder", header: "llama.h".}
proc llamaModelDecoderStartToken*(model: ptr LlamaModel): LlamaToken
  {.importc: "llama_model_decoder_start_token", header: "llama.h".}
proc llamaModelIsRecurrent*(model: ptr LlamaModel): bool
  {.importc: "llama_model_is_recurrent", header: "llama.h".}
proc llamaModelIsHybrid*(model: ptr LlamaModel): bool
  {.importc: "llama_model_is_hybrid", header: "llama.h".}
proc llamaModelIsDiffusion*(model: ptr LlamaModel): bool
  {.importc: "llama_model_is_diffusion", header: "llama.h".}

# ---------------------------------------------------------------------------
# Context lifecycle and decoding.
# ---------------------------------------------------------------------------

proc llamaInitFromModel*(model: ptr LlamaModel; params: LlamaContextParams): ptr LlamaContext
  {.importc: "llama_init_from_model", header: "llama.h".}
proc llamaFree*(ctx: ptr LlamaContext)
  {.importc: "llama_free", header: "llama.h".}
proc llamaNCtx*(ctx: ptr LlamaContext): uint32
  {.importc: "llama_n_ctx", header: "llama.h".}
proc llamaGetMemory*(ctx: ptr LlamaContext): LlamaMemory
  {.importc: "llama_get_memory", header: "llama.h".}
proc llamaDecode*(ctx: ptr LlamaContext; batch: LlamaBatch): int32
  {.importc: "llama_decode", header: "llama.h".}
proc llamaGetLogitsIth*(ctx: ptr LlamaContext; i: int32): ptr cfloat
  {.importc: "llama_get_logits_ith", header: "llama.h".}

# ---------------------------------------------------------------------------
# Batch construction.
# ---------------------------------------------------------------------------

proc llamaBatchInit*(nTokens: int32; embd: int32; nSeqMax: int32): LlamaBatch
  {.importc: "llama_batch_init", header: "llama.h".}
proc llamaBatchFree*(batch: LlamaBatch)
  {.importc: "llama_batch_free", header: "llama.h".}

# ---------------------------------------------------------------------------
# Sequence memory (the KV cache). `p1` is exclusive; -1 means "to the end".
# ---------------------------------------------------------------------------

proc llamaMemorySeqRm*(mem: LlamaMemory; seqId: LlamaSeqId; p0: LlamaPos; p1: LlamaPos): bool
  {.importc: "llama_memory_seq_rm", header: "llama.h".}
proc llamaMemorySeqCp*(mem: LlamaMemory; src: LlamaSeqId; dst: LlamaSeqId; p0: LlamaPos; p1: LlamaPos)
  {.importc: "llama_memory_seq_cp", header: "llama.h".}
proc llamaMemorySeqAdd*(mem: LlamaMemory; seqId: LlamaSeqId; p0: LlamaPos; p1: LlamaPos; delta: LlamaPos)
  {.importc: "llama_memory_seq_add", header: "llama.h".}
proc llamaMemorySeqPosMin*(mem: LlamaMemory; seqId: LlamaSeqId): LlamaPos
  {.importc: "llama_memory_seq_pos_min", header: "llama.h".}
proc llamaMemorySeqPosMax*(mem: LlamaMemory; seqId: LlamaSeqId): LlamaPos
  {.importc: "llama_memory_seq_pos_max", header: "llama.h".}
proc llamaMemoryCanShift*(mem: LlamaMemory): bool
  {.importc: "llama_memory_can_shift", header: "llama.h".}

# ---------------------------------------------------------------------------
# Vocabulary queries. Token ids outside the vocab answer `LlamaTokenNull`.
# ---------------------------------------------------------------------------

proc llamaVocabType*(vocab: ptr LlamaVocab): cint
  {.importc: "llama_vocab_type", header: "llama.h".}
proc llamaVocabNTokens*(vocab: ptr LlamaVocab): int32
  {.importc: "llama_vocab_n_tokens", header: "llama.h".}
proc llamaVocabGetText*(vocab: ptr LlamaVocab; token: LlamaToken): cstring
  {.importc: "llama_vocab_get_text", header: "llama.h".}
proc llamaVocabGetScore*(vocab: ptr LlamaVocab; token: LlamaToken): cfloat
  {.importc: "llama_vocab_get_score", header: "llama.h".}
proc llamaVocabGetAttr*(vocab: ptr LlamaVocab; token: LlamaToken): cint
  {.importc: "llama_vocab_get_attr", header: "llama.h".}
proc llamaVocabIsEog*(vocab: ptr LlamaVocab; token: LlamaToken): bool
  {.importc: "llama_vocab_is_eog", header: "llama.h".}
proc llamaVocabIsControl*(vocab: ptr LlamaVocab; token: LlamaToken): bool
  {.importc: "llama_vocab_is_control", header: "llama.h".}

proc llamaVocabBos*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_bos", header: "llama.h".}
proc llamaVocabEos*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_eos", header: "llama.h".}
proc llamaVocabEot*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_eot", header: "llama.h".}
proc llamaVocabCls*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_cls", header: "llama.h".}
proc llamaVocabSep*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_sep", header: "llama.h".}
proc llamaVocabNl*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_nl", header: "llama.h".}
proc llamaVocabPad*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_pad", header: "llama.h".}
proc llamaVocabMask*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_mask", header: "llama.h".}

proc llamaVocabGetAddBos*(vocab: ptr LlamaVocab): bool
  {.importc: "llama_vocab_get_add_bos", header: "llama.h".}
proc llamaVocabGetAddEos*(vocab: ptr LlamaVocab): bool
  {.importc: "llama_vocab_get_add_eos", header: "llama.h".}
proc llamaVocabGetAddSep*(vocab: ptr LlamaVocab): bool
  {.importc: "llama_vocab_get_add_sep", header: "llama.h".}

proc llamaVocabFimPre*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_fim_pre", header: "llama.h".}
proc llamaVocabFimSuf*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_fim_suf", header: "llama.h".}
proc llamaVocabFimMid*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_fim_mid", header: "llama.h".}
proc llamaVocabFimPad*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_fim_pad", header: "llama.h".}
proc llamaVocabFimRep*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_fim_rep", header: "llama.h".}
proc llamaVocabFimSep*(vocab: ptr LlamaVocab): LlamaToken
  {.importc: "llama_vocab_fim_sep", header: "llama.h".}

# ---------------------------------------------------------------------------
# Tokenization. Both procs answer the token/byte count on success and a
# negative "would-have-been" count when the buffer is too small, which is how
# the wrappers size their buffers.
# ---------------------------------------------------------------------------

proc llamaTokenize*(vocab: ptr LlamaVocab; text: cstring; textLen: int32;
                    tokens: ptr LlamaToken; nTokensMax: int32;
                    addSpecial: bool; parseSpecial: bool): int32
  {.importc: "llama_tokenize", header: "llama.h".}

proc llamaTokenToPiece*(vocab: ptr LlamaVocab; token: LlamaToken; buf: cstring;
                        length: int32; lstrip: int32; special: bool): int32
  {.importc: "llama_token_to_piece", header: "llama.h".}

# ---------------------------------------------------------------------------
# Performance counters.
# ---------------------------------------------------------------------------

proc llamaPerfContext*(ctx: ptr LlamaContext): LlamaPerfContextData
  {.importc: "llama_perf_context", header: "llama.h".}
