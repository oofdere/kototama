// Port of examples/simple.rs — greedy argmax generation
//
// Usage:
//   npx ts-node examples/simple.ts --model path/to/model.gguf --prompt "Hello my name is"
//
// NOTE: ModelParams/ContextParams are not yet exposed through the bindings
// (their C struct fields are invisible to alef's syn parser). This example
// shows the target API shape; params handling will work once rusty_llama
// exports Rust-native param structs.

import { Model, Context, type Sequence } from "rusty-llama";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    model: { type: "string", short: "m" },
    prompt: { type: "string", default: "Hello my name is" },
    "n-predict": { type: "string", default: "32" },
    ngl: { type: "string", default: "99" },
  },
});

const modelPath = values.model;
if (!modelPath) throw new Error("missing --model");

const prompt = values.prompt!;
const nPredict = parseInt(values["n-predict"]!, 10);

if (nPredict <= 0) throw new Error(`n_predict must be positive, got ${nPredict}`);

// Initialize the model
// TODO: pass ModelParams once params types are exposed
const model = Model.loadFromFile(modelPath, "{}");

console.log(`Model: ${model.desc()}`);

if (model.hasEncoder()) {
  throw new Error("Model has encoder, which is not supported in this example");
}

// Tokenize the prompt
const promptTokens = model.tokenize(prompt, true, true);

// Initialize the context
// TODO: pass ContextParams once params types are exposed
const ctx = Context.new(model, "{}");

const seq = ctx.sequence()!;

seq.extend(promptTokens);

// Print the prompt token-by-token
for (const token of promptTokens) {
  process.stdout.write(model.tokenToPiece(Number(token)));
}

// Main loop — greedy argmax sampling
for (let i = 0; i < nPredict; i++) {
  const logits = seq.logits();
  if (!logits) break;

  let bestToken = 0;
  let bestScore = -Infinity;
  for (let j = 0; j < logits.length; j++) {
    if (logits[j] > bestScore) {
      bestScore = logits[j];
      bestToken = j;
    }
  }

  if (model.isEog(bestToken.toString())) break;

  process.stdout.write(model.tokenToPiece(bestToken));
  seq.push(bestToken.toString());
}

console.log();
