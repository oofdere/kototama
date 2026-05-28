// Port of examples/simple_chat.rs — interactive chat loop
//
// Usage:
//   npx ts-node examples/simple_chat.ts --model path/to/model.gguf
//
// NOTE: ModelParams/ContextParams/SamplerChain are not yet exposed through
// the bindings. This example shows the target API shape using greedy
// sampling as a stand-in for the sampler chain.

import { Model, Context, type Sequence } from "rusty-llama";
import * as readline from "node:readline";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    model: { type: "string", short: "m" },
    context: { type: "string", default: "2048" },
    ngl: { type: "string", default: "99" },
  },
});

const modelPath = values.model;
if (!modelPath) throw new Error("missing --model");

// Initialize the model
// TODO: pass ModelParams with n_gpu_layers once params are exposed
const model = Model.loadFromFile(modelPath, "{}");

// Initialize the context
// TODO: pass ContextParams with n_ctx/n_batch once params are exposed
const ctx = Context.new(model, "{}");

const seq = ctx.sequence()!;

interface Message {
  role: "user" | "assistant";
  content: string;
}

const messages: Message[] = [];

function formatMessages(msgs: Message[]): string {
  const formatted = msgs
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n");
  const result = `${formatted}\nassistant:`;
  console.log(result);
  return result;
}

function generateResponse(): string {
  let response = "";

  while (true) {
    const logits = seq.logits();
    if (!logits) break;

    // Greedy argmax as stand-in for SamplerChain (min_p + temp + dist)
    let bestToken = 0;
    let bestScore = -Infinity;
    for (let j = 0; j < logits.length; j++) {
      if (logits[j] > bestScore) {
        bestScore = logits[j];
        bestToken = j;
      }
    }

    if (model.isEog(bestToken.toString())) break;

    const piece = model.tokenToPiece(bestToken);
    process.stdout.write(piece);
    response += piece;

    seq.push(bestToken.toString());

    if (piece.includes("\n")) break;
  }

  return response;
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function prompt() {
  rl.question("", (input) => {
    if (!input.trim()) {
      rl.close();
      return;
    }

    messages.push({ role: "user", content: input.trim() });

    const promptText = formatMessages(messages);
    const isFirst = seq.isEmpty();

    const tokens = model.tokenize(promptText, isFirst, true);
    seq.extend(tokens);

    console.log();
    const response = generateResponse();
    console.log();

    messages.push({ role: "assistant", content: response });
    prompt();
  });
}

prompt();
