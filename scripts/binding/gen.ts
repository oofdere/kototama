/**
 * Binding generator driver: api.json + a backend -> source files.
 *
 *   node scripts/binding/gen.ts <backend>            # write the files
 *   node scripts/binding/gen.ts <backend> --check    # fail if files are stale
 *
 * `--check` writes the would-be output to `<file>.gen` for inspection on drift.
 * Runs in Node and Deno.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import process from "node:process";
import type { Api } from "./ir.ts";
import type { Backend, ManualLoader } from "./backends/types.ts";
import { nodeBackend } from "./backends/node.ts";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
const BACKENDS: Record<string, Backend> = { node: nodeBackend };

function manualLoader(id: string): ManualLoader {
  return (key) => {
    const [type, method] = key.split("::");
    const path = `scripts/binding/manual/${id}/${type}.${method}.rs`;
    try {
      return readFileSync(REPO + path, "utf8").replace(/\n+$/, "");
    } catch {
      console.error(`error: missing manual snippet ${path} (method is marked manual for "${id}")`);
      process.exit(1);
    }
  };
}

const id = process.argv[2];
const check = process.argv.includes("--check");
const backend = BACKENDS[id];
if (!backend) {
  console.error(`usage: gen.ts <backend> [--check]\nknown backends: ${Object.keys(BACKENDS).join(", ")}`);
  process.exit(1);
}

const api: Api = JSON.parse(readFileSync(REPO + "scripts/binding/api.json", "utf8"));
const files = backend.generate(api, manualLoader(id));

let drift = 0;
for (const f of files) {
  if (check) {
    let cur = "";
    try { cur = readFileSync(REPO + f.path, "utf8"); } catch { /* missing */ }
    if (cur === f.contents) {
      console.log(`ok:    ${f.path}`);
    } else {
      writeFileSync(REPO + f.path + ".gen", f.contents);
      console.error(`DRIFT: ${f.path}  (wrote ${f.path}.gen)`);
      drift++;
    }
  } else {
    writeFileSync(REPO + f.path, f.contents);
    console.log(`wrote ${f.path}`);
  }
}
if (check && drift) process.exit(1);
