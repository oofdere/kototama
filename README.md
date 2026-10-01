# kototama: polylot llama.cpp bindings

upstream version: b9246

api stability: no. not yet.

this should just build without any special work if you've followed the [upstream build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for your specific backend. currently, only the vulkan and metal backends are supported.

## Goals
1. provide high quality rust bindings for llama.cpp
2. keep bindings as safe as possible without neutering functionality
3. document things properly and clearly
4. provide common compile flags as cargo features

## Versioning
semver (no effort will be made to match the upstream version numbers whatsoever for obvious reasons)

## sync and async

the core api (`Model`, `Context`, `Sequence`) is synchronous: calls run on your thread and serialize on a lock when you share handles. no runtime needed.

the `rusty_llama::asynchronous` module is a thin facade over the same core, for use inside async code: its `*_async` methods ship the blocking llama.cpp calls (model loading, decoding, kv-cache edits) to a small built-in thread pool and return `Send` futures, so awaiting them never stalls your executor. no tokio/async-std dependency - `asynchronous::block_on` runs the futures without any runtime at all. see `examples/simple_async.rs`.

