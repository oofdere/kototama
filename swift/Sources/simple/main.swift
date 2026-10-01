//
//  simple — generate text with a model.
//
//  The Swift counterpart of examples/simple.rs. Same flags, same output.
//
//      swift run simple -m ../test-models/TinyStories-656K.Q2_K.gguf "Hello"
//

import Foundation
import Kototama

// MARK: Command line

struct Options {
    var modelPath: String = ""
    var prompt = "Hello my name is"
    var nPredict = 32
    var nGPULayers: Int32 = 99

    static func parse(_ arguments: [String]) -> Options? {
        var options = Options()
        var index = 0
        var positional: [String] = []

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-m", "--model":
                index += 1
                guard index < arguments.count else { return nil }
                options.modelPath = arguments[index]
            case "-n", "--n-predict":
                index += 1
                guard index < arguments.count, let value = Int32(arguments[index]) else { return nil }
                options.nPredict = Int(value)
            case "--ngl":
                index += 1
                guard index < arguments.count, let value = Int32(arguments[index]) else { return nil }
                options.nGPULayers = value
            case "-h", "--help":
                print("""
                OVERVIEW: generate text with a llama.cpp model

                USAGE: simple [options] [prompt]

                OPTIONS:
                  -m, --model <path>   path to the model file (required)
                  -n <count>           number of tokens to predict (default: 32)
                  --ngl <count>        number of GPU layers (default: 99)
                  -h, --help           show this help
                """)
                exit(0)
            default:
                positional.append(argument)
            }
            index += 1
        }

        if let first = positional.first {
            options.prompt = first
        }
        return options.modelPath.isEmpty ? nil : options
    }
}

// MARK: Main

guard let options = Options.parse(Array(CommandLine.arguments.dropFirst())) else {
    FileHandle.standardError.write(Data("usage: simple -m <model.gguf> [prompt]\n".utf8))
    exit(1)
}

guard options.nPredict > 0 else {
    FileHandle.standardError.write(Data("n_predict must be positive, got \(options.nPredict)\n".utf8))
    exit(1)
}

do {
    // The backend loads lazily on first model load.

    var modelParams = ModelParams()
    modelParams.nGPULayers = options.nGPULayers

    let model = try Model.load(from: options.modelPath, params: modelParams)
    print("Model: \(try model.describe())")

    if model.hasEncoder {
        print("Model has encoder, which is not supported in this example")
        exit(1)
    }

    // Tokenize the prompt.
    let promptTokens = try model.tokenize(options.prompt, addSpecial: true, parseSpecial: true)

    // One context sized for prompt + output, decoding a single token per
    // step (the same `n_batch = 1` choice examples/simple.rs documents).
    var contextParams = ContextParams()
    contextParams.nContext = UInt32(promptTokens.count + options.nPredict - 1)
    contextParams.nBatch = 1
    contextParams.noPerf = false

    let context = try Context(model: model, params: contextParams)
    guard let sequence = context.checkoutSequence() else {
        FileHandle.standardError.write(Data("failed to acquire sequence\n".utf8))
        exit(1)
    }
    try sequence.push(contentsOf: promptTokens)

    // Echo the prompt, then generate.
    for token in promptTokens {
        print(model.pieceOrNil(of: token) ?? "", terminator: "")
    }
    fflush(stdout)

    for _ in 0..<options.nPredict {
        guard let logits = sequence.logits else { break }
        let token = argmax(logits)
        if model.isEndOfGeneration(token) { break }
        print(model.pieceOrNil(of: token) ?? "", terminator: "")
        fflush(stdout)
        try sequence.push(token)
    }

    print()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
