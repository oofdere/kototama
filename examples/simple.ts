// Port of examples/simple.rs — greedy argmax generation
//
// Usage:
//   node examples/simple.js --model path/to/model.gguf --prompt "Hello my name is"
//
// Build the NAPI addon first:
//   cd packages/node && npm run build

import { Model, Context } from "../packages/node";
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
const nPredictRaw = values["n-predict"]!;
const nPredict = Number(nPredictRaw);

if (!Number.isInteger(nPredict) || nPredict <= 0) {
  throw new Error(`n_predict must be a positive integer, got "${nPredictRaw}"`);
}

const ngl = Number(values.ngl ?? "99");

const model = Model.loadFromFile(modelPath, ngl);

console.log(`Model: ${model.desc()}`);

if (model.hasEncoder()) {
  throw new Error("Model has encoder, which is not supported in this example");
}

const promptTokens = model.tokenize(prompt, true, true);

const ctx = Context.new(model, 2048, 512);

const seq = ctx.sequence()!;

seq.extend(promptTokens);

for (const token of promptTokens) {
  process.stdout.write(model.tokenToPiece(token));
}

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

  if (model.isEog(bestToken)) break;

  process.stdout.write(model.tokenToPiece(bestToken));
  seq.push(bestToken);
}

console.log();
