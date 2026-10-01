//
//  kototama-bench — benchmark harness for the Swift port.
//
//  Mirrors benches/inference.rs (the criterion benchmarks) one-to-one:
//
//    tokenize                  split a fixed prompt into tokens
//    token_to_piece            decode a single token
//    decode_single_token       push one token and read logits
//    generate_10_tokens        greedy generation of 10 tokens
//    context_creation          build a Context from the model
//    sequence_extend_100       push 100 tokens into a fresh sequence
//
//  Usage:
//      swift run -c release kototama-bench [--json]
//
//  Methodology note: criterion warms up, then collects samples of many
//  iterations each and reports the mean with bootstrap confidence intervals.
//  This harness warms up too and reports the *median* of per-iteration wall
//  times over the same kind of sampling, which is the robust counterpart and
//  comparable across the two languages. See COMPARISON.md for the discussion.
//

import Foundation
import Kototama

// MARK: Timing

/// Times `body` over several samples and returns per-iteration wall times.
///
/// When `setup` is given, it runs before every iteration but is *not* timed —
/// this matches criterion's `iter_batched(_, _, PerIteration)`, where the
/// setup closure prepares the fixture and only the routine is measured. This
/// matters: criterion's `generate_10_tokens` measures 10 decodes of a
/// pre-filled sequence, not the prompt ingestion too.
///
/// With no `setup`, each sample runs `iterationsPerSample` iterations in one
/// timed block and divides down, so clock overhead does not dominate fast
/// operations (which is how criterion's plain `iter` behaves).
@discardableResult
func measure<T>(
    name: String,
    iterationsPerSample: Int = 20,
    samples: Int = 30,
    warmup: Int = 5,
    setup: (() -> T)? = nil,
    _ body: (T) throws -> Void
) rethrows -> Result {
    for _ in 0..<warmup {
        if let fixture = setup?() {
            try body(fixture)
        }
    }

    var perIteration: [Double] = []
    perIteration.reserveCapacity(samples)

    let seconds: (Duration) -> Double = { duration in
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }

    for _ in 0..<samples {
        if setup != nil {
            // Per-iteration timing; setup excluded.
            for _ in 0..<iterationsPerSample {
                let fixture = setup!()
                let start = ContinuousClock.now
                try body(fixture)
                perIteration.append(seconds(ContinuousClock.now - start))
            }
        } else {
            let start = ContinuousClock.now
            for _ in 0..<iterationsPerSample {
                try body(() as! T)
            }
            perIteration.append(seconds(ContinuousClock.now - start) / Double(iterationsPerSample))
        }
    }

    perIteration.sort()
    let median = perIteration[perIteration.count / 2]
    let mean = perIteration.reduce(0.0, +) / Double(perIteration.count)
    let minimum = perIteration.first!
    let maximum = perIteration.last!

    return Result(
        name: name, samples: samples, iterationsPerSample: iterationsPerSample,
        medianSeconds: median, meanSeconds: mean, minSeconds: minimum, maxSeconds: maximum
    )
}

struct Result {
    let name: String
    let samples: Int
    let iterationsPerSample: Int
    let medianSeconds: Double
    let meanSeconds: Double
    let minSeconds: Double
    let maxSeconds: Double

    /// Human-readable line, mirroring criterion's "time: [a b c]" output scale.
    var description: String {
        let fmt = { (value: Double) in
            if value >= 1e-3 {
                return String(format: "%.3f ms", value * 1e3)
            } else {
                return String(format: "%.3f µs", value * 1e6)
            }
        }
        let paddedName = name.padding(toLength: 24, withPad: " ", startingAt: 0)
        return "\(paddedName) median \(fmt(medianSeconds))  mean \(fmt(meanSeconds))  min \(fmt(minSeconds))  max \(fmt(maxSeconds))"
    }
}

// MARK: Setup
//
// The fixture matches benches/inference.rs: the bundled TinyStories model,
// CPU only (load_model in src/test_common.rs forces n_gpu_layers = 0), and a
// 512/512/1 context.

func modelPath() -> String {
    let env = ProcessInfo.processInfo.environment
    if let path = env["RUSTY_LLAMA_BENCH_MODEL"] ?? env["RUSTY_LLAMA_TEST_MODEL"] {
        return path
    }
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    return root.appendingPathComponent("test-models/TinyStories-656K.Q2_K.gguf").path
}

var modelParams = ModelParams()
modelParams.nGPULayers = 0
let model = try! Model.load(from: modelPath(), params: modelParams)

func makeContextParams() -> ContextParams {
    var params = ContextParams()
    params.nContext = 512
    params.nBatch = 512
    params.nSequencesMax = 1
    params.noPerf = true
    return params
}

// MARK: Benchmarks
//
// Each bench pairs the fixture setup with the routine being measured, in the
// same arrangement as benches/inference.rs so the two sets of numbers compare
// like for like.

var results: [Result] = []

// tokenize — plain iter: nothing to set up, timed as a batch.
do {
    let prompt = "The quick brown fox jumps over the lazy dog."
    results.append(measure(name: "tokenize") { (_: Void) in
        _ = try! model.tokenize(prompt, addSpecial: false, parseSpecial: false)
    })
}

// token_to_piece — plain iter.
do {
    let bos = model.vocabulary.bosToken ?? Token(1)
    results.append(measure(name: "token_to_piece") { (_: Void) in
        _ = model.pieceOrNil(of: bos)
    })
}

// decode_single_token — iter_batched PerIteration, exactly as
// bench_decode_single_token in benches/inference.rs: the setup checks out a
// fresh sequence and decodes the seed token, then the timed part is reading
// the logits, argmax, and pushing that one token.
do {
    let params = makeContextParams()
    let context = try! Context(model: model, params: params)
    let seedToken = model.vocabulary.bosToken ?? Token(1)

    results.append(measure(
        name: "decode_single_token",
        setup: { () -> TokenSequence in
            let sequence = context.checkoutSequence()!
            try! sequence.push(seedToken)
            return sequence
        }
    ) { sequence in
        let token = argmax(sequence.logits!)
        try! sequence.push(token)
        _ = sequence
    })
}

// generate_10_tokens — iter_batched PerIteration: the prompt is ingested in
// setup and only the 10 greedy decodes are timed.
do {
    let params = makeContextParams()
    let context = try! Context(model: model, params: params)
    let promptTokens = try! model.tokenize("Once upon a time", addSpecial: true, parseSpecial: false)

    results.append(measure(
        name: "generate_10_tokens",
        setup: { () -> TokenSequence in
            let sequence = context.checkoutSequence()!
            try! sequence.push(contentsOf: promptTokens)
            return sequence
        }
    ) { sequence in
        for _ in 0..<10 {
            let token = argmax(sequence.logits!)
            if model.isEndOfGeneration(token) { break }
            try! sequence.push(token)
        }
    })
}

// context_creation — iter_batched PerIteration with an empty setup.
do {
    let params = makeContextParams()
    results.append(
        measure(
            name: "context_creation",
            setup: { () -> Void in () }
        ) { (_: Void) in
            _ = try! Context(model: model, params: params)
        }
    )
}

// sequence_extend_100_tokens — iter_batched PerIteration: fresh sequence per
// iteration (setup), then a 100-token extend (timed).
do {
    let params = makeContextParams()
    let context = try! Context(model: model, params: params)
    let bos = model.vocabulary.bosToken ?? Token(1)
    let tokens = [Token](repeating: bos, count: 100)

    results.append(measure(
        name: "sequence_extend_100_tokens",
        iterationsPerSample: 5,
        samples: 20,
        setup: { context.checkoutSequence()! }
    ) { sequence in
        try! sequence.push(contentsOf: tokens)
    })
}

// MARK: Report

if CommandLine.arguments.contains("--json") {
    let payload: [[String: Any]] = results.map {
        [
            "name": $0.name,
            "samples": $0.samples,
            "iterations_per_sample": $0.iterationsPerSample,
            "median_seconds": $0.medianSeconds,
            "mean_seconds": $0.meanSeconds,
            "min_seconds": $0.minSeconds,
            "max_seconds": $0.maxSeconds,
        ]
    }
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
    FileHandle.standardOutput.write(data)
} else {
    for result in results {
        print(result.description)
    }
}
