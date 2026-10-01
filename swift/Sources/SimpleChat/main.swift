//
//  simple-chat — an interactive chat loop.
//
//  The Swift counterpart of examples/simple_chat.rs (itself a port of
//  llama.cpp's simple_chat example, ignoring chat templates): user and
//  assistant turns are formatted by hand, and each reply is generated
//  through MinP -> Temperature -> Dist until a newline or end-of-generation.
//
//      swift run simple-chat -m ../test-models/TinyStories-656K.Q2_K.gguf
//

import Foundation
import Kototama

// MARK: Command line

struct Options {
    var modelPath = ""
    var context = 2048
    var nGPULayers: Int32 = 99

    static func parse(_ arguments: [String]) -> Options? {
        var options = Options()
        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "-m", "--model":
                index += 1
                guard index < arguments.count else { return nil }
                options.modelPath = arguments[index]
            case "-c", "--context":
                index += 1
                guard index < arguments.count, let value = Int32(arguments[index]) else { return nil }
                options.context = Int(value)
            case "--ngl":
                index += 1
                guard index < arguments.count, let value = Int32(arguments[index]) else { return nil }
                options.nGPULayers = value
            case "-h", "--help":
                print("""
                OVERVIEW: interactive chat with a llama.cpp model

                USAGE: simple-chat [options]

                OPTIONS:
                  -m, --model <path>   path to the model file (required)
                  -c, --context <n>    context size (default: 2048)
                  --ngl <count>        number of GPU layers (default: 99)
                  -h, --help           show this help
                """)
                exit(0)
            default:
                break
            }
            index += 1
        }
        return options.modelPath.isEmpty ? nil : options
    }
}

// MARK: Conversation formatting

enum Message {
    case user(String)
    case assistant(String)

    var rendered: String {
        switch self {
        case .user(let text): return "user: \(text)"
        case .assistant(let text): return "assistant: \(text)"
        }
    }
}

/// Renders the conversation so far as the prompt for the next reply.
/// Ports the `format` closure in examples/simple_chat.rs.
func format(_ messages: [Message]) -> String {
    var prompt = messages.map(\.rendered).joined(separator: "\n")
    prompt += "\nassistant:"
    print(prompt)
    return prompt
}

// MARK: Main

guard let options = Options.parse(Array(CommandLine.arguments.dropFirst())) else {
    FileHandle.standardError.write(Data("usage: simple-chat -m <model.gguf>\n".utf8))
    exit(1)
}

do {
    var modelParams = ModelParams()
    modelParams.nGPULayers = options.nGPULayers
    let model = try Model.load(from: options.modelPath, params: modelParams)

    var contextParams = ContextParams()
    contextParams.nContext = UInt32(options.context)
    contextParams.nBatch = UInt32(options.context)

    let context = try Context(model: model, params: contextParams)
    guard let sequence = context.checkoutSequence() else {
        FileHandle.standardError.write(Data("failed to acquire sequence\n".utf8))
        exit(1)
    }

    // Samplers, applied by hand: min_p -> temp -> dist.
    let minP = MinP(p: 0.05, minKeep: 1)
    let temperature = Temperature(temp: 0.8)
    var dist = Dist(seed: Dist.defaultSeed)

    var messages: [Message] = []

    while let line = readLine(strippingNewline: true) {
        if line.isEmpty { break }
        messages.append(.user(line))

        let prompt = format(messages)
        let isFirst = sequence.isEmpty

        let tokens = try model.tokenize(prompt, addSpecial: isFirst, parseSpecial: true)
        try sequence.push(contentsOf: tokens)

        var response = ""
        print()

        while true {
            guard let logits = sequence.logits else { break }
            let scaled = minP.transform(temperature.transform(logits))
            let token = dist.sample(scaled)

            if model.isEndOfGeneration(token) { break }

            let piece = model.pieceOrNil(of: token) ?? ""
            print(piece, terminator: "")
            response += piece

            try sequence.push(token)

            if piece.contains("\n") { break }
        }

        print()
        messages.append(.assistant(response))
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
