// Port of examples/simple_chat.rs — interactive chat loop
//
// Usage:
//   node examples/simple_chat.js --model path/to/model.gguf
//
// Build the NAPI addon first:
//   cd packages/node && npm run build

import { Model, Context, type Sequence } from "../packages/node";
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

const ngl = Number(values.ngl ?? "99");
const nCtx = Number(values.context ?? "2048");

const model = Model.loadFromFile(modelPath, ngl);

const ctx = Context.new(model, nCtx, 512);

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

    let bestToken = 0;
    let bestScore = -Infinity;
    for (let j = 0; j < logits.length; j++) {
      if (logits[j] > bestScore) {
        bestScore = logits[j];
        bestToken = j;
      }
    }

    if (model.isEog(bestToken)) break;

    const piece = model.tokenToPiece(bestToken);
    process.stdout.write(piece);
    response += piece;

    seq.push(bestToken);

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
