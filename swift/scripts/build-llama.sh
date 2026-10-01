#!/usr/bin/env bash
#
# Builds the llama.cpp checkout pinned by the Rust `llama-sys` crate
# (../llama-sys/llama.cpp) into static libraries for the Swift package.
#
# This mirrors the CMake configuration in llama-sys/build.rs so that the
# Rust and Swift sides run the same llama.cpp with the same backends,
# which is what makes the side-by-side comparison meaningful.
#
# Outputs:
#   .build/llama/prefix/lib/*.a   static libraries (merged into
#                                 libkototama-llama.a)
#   Sources/CLlama/include/*.h    C headers, copied so the SwiftPM
#                                 CLlama target stays self-contained
#
# Usage: scripts/build-llama.sh

set -euo pipefail

cd "$(dirname "$0")/.."
SWIFT_ROOT="$(pwd)"
REPO_ROOT="$(cd .. && pwd)"
LLAMA_SRC="$REPO_ROOT/llama-sys/llama.cpp"

if [ ! -f "$LLAMA_SRC/CMakeLists.txt" ]; then
    echo "error: llama.cpp submodule is missing at $LLAMA_SRC" >&2
    echo "run: git submodule update --init --recursive" >&2
    exit 1
fi

BUILD_ROOT="$SWIFT_ROOT/.build/llama"
OBJ_DIR="$BUILD_ROOT/obj"
PREFIX="$BUILD_ROOT/prefix"

echo "==> configuring llama.cpp (Release, static, Metal on macOS)"
cmake -S "$LLAMA_SRC" -B "$OBJ_DIR" \
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
    -DGGML_METAL_NDEBUG=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON

echo "==> building (this takes a minute or two)"
cmake --build "$OBJ_DIR" --config Release -j "$(sysctl -n hw.ncpu)"
cmake --install "$OBJ_DIR" --config Release

echo "==> merging static libraries into libkototama-llama.a"
LIB_DIR="$BUILD_ROOT/lib"
mkdir -p "$LIB_DIR"
find "$OBJ_DIR" -name '*.a' -print0 | xargs -0 libtool -static -o "$LIB_DIR/libkototama-llama.a"

echo "==> copying headers into Sources/CLlama/include"
HEADER_DIR="$SWIFT_ROOT/Sources/CLlama/include"
mkdir -p "$HEADER_DIR"
# CLlama.h is the shim we own; everything else is regenerated from the submodule.
find "$HEADER_DIR" -name '*.h' ! -name 'CLlama.h' -delete
for header in "$LLAMA_SRC/include"/*.h "$LLAMA_SRC/ggml/include"/*.h; do
    cp "$header" "$HEADER_DIR/"
done

echo "==> done. next: swift build"
