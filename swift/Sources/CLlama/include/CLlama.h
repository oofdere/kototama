#ifndef CLlama_h
#define CLlama_h

/*
 * CLlama — the thin C interop shim for the Kototama Swift package.
 *
 * This single header is the only entry point the Swift Clang importer needs.
 * It pulls in llama.h, which transitively brings in the ggml headers that were
 * copied into this directory by scripts/build-llama.sh.
 *
 * We deliberately keep the module map pointed at *this* header only (see
 * module.modulemap). That way the importer parses llama.h and its includes and
 * nothing else — the extra backend headers that also land in include/ (cuda.h,
 * sycl.h, vulkan.h, ...) are never parsed, so they can never break the build
 * even though they require SDKs this package does not use.
 */

#include "llama.h"

#endif /* CLlama_h */
