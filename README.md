# kototama: polylot llama.cpp bindings

upstream version: b9246

supported languages: rust, elixir, typescript

api stability: no. not yet.

this should just build without any special work if you've followed the [upstream build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for your specific backend. currently, only the vulkan and metal backends are supported.

## Goals
1. provide high quality rust bindings for llama.cpp
2. keep bindings as safe as possible without neutering functionality
3. document things properly and clearly
4. provide common compile flags as cargo features

## Versioning
semver (no effort will be made to match the upstream version numbers whatsoever for obvious reasons)

