#!/usr/bin/env bash
# Build and run the full Nim test suite -- the counterpart of `cargo test`.
#
# Nim's unittest runs every suite in one process per file, so this compiles and
# runs each test file in turn. Like the Rust suite (see AGENTS.md), tests are
# not parallelized: llama.cpp's backend is process-global state.
#
# Usage: nim/test.sh [testname]   e.g. nim/test.sh tsamplers
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f test-models/TinyStories-656K.Q2_K.gguf ]; then
  echo "warning: bundled test model missing; set KOTOTAMA_TEST_MODEL" >&2
fi

if [ ! -f nim/llama-cpp-build/install/lib/libllama.a ]; then
  echo "error: llama.cpp not built; run: nim/build_llama_cpp.sh" >&2
  exit 1
fi

TESTS=(tsamplers tmodel tvocab tcontext tsequence tintegration tsnapshots)
if [ $# -gt 0 ]; then
  TESTS=("$@")
fi

total_pass=0
total_fail=0

for t in "${TESTS[@]}"; do
  echo "=== $t"
  nim c --hints:off --path:nim/src -o:"/tmp/kototama-$t" "nim/tests/$t.nim" >&2
  # llama.cpp's log stream is verbose; keep the unittest result lines.
  out=$("/tmp/kototama-$t" 2>/dev/null | grep -E "^\s+\[(OK|FAILED)\]|^\[Suite\]" || true)
  echo "$out"
  pass=$(echo "$out" | grep -c "\[OK\]" || true)
  fail=$(echo "$out" | grep -c "\[FAILED\]" || true)
  total_pass=$((total_pass + pass))
  total_fail=$((total_fail + fail))
done

echo
echo "test result: $total_pass passed; $total_fail failed"
[ "$total_fail" -eq 0 ]
