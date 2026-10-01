#!/usr/bin/env bash
# Build the static llama.cpp libraries that kototama's Nim port links against.
#
# This mirrors llama-sys/build.rs (the Rust build script) flag for flag: both
# language ports compile the same pinned submodule at llama-sys/llama.cpp with
# the same configuration, so benchmark numbers compare wrappers, not backends.
#
# Output: nim/llama-cpp-build/install/lib/*.a  (override with --prefix DIR)
set -euo pipefail

cd "$(dirname "$0")/.."

SRC=llama-sys/llama.cpp
BUILD=nim/llama-cpp-build
PREFIX="$PWD/$BUILD/install"

if [ "${1:-}" = "--prefix" ] && [ -n "${2:-}" ]; then
  PREFIX="$2"
fi

if [ ! -f "$SRC/include/llama.h" ]; then
  echo "error: $SRC is empty; run: git submodule update --init --recursive" >&2
  exit 1
fi

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_STATIC=ON \
  -DGGML_PERF=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=ON \
  -DGGML_METAL_NDEBUG=ON

cmake --build "$BUILD" --config Release -j
cmake --install "$BUILD"

echo
echo "llama.cpp static libraries installed in: $PREFIX/lib"
echo "Point the Nim build at them with: -d:llamaLibDir=$PREFIX/lib"
