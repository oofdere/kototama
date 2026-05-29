#!/usr/bin/env -S node --no-warnings
/**
 * Check that the Elixir and Node bindings expose the core rusty-llama API.
 *
 * Runs unmodified in both Node (>= 22.18, which strips TS types) and Deno.
 * It only uses `node:` builtins, which Deno supports via its node-compat layer.
 *
 *   node scripts/check-binding-coverage.ts
 *   deno run --allow-read scripts/check-binding-coverage.ts
 *
 * This is a *build-free* checker: it parses Rust/Elixir source as text, so it
 * needs no llama.cpp submodule, no GPU backend, and no compilation, making it
 * cheap to run in CI as a drift gate.
 *
 * What it does:
 *   1. Extracts the public methods of the core handle types (Model, Context,
 *      Sequence) from `src/`, including macro-generated `token_option!` getters.
 *   2. Extracts the symbols actually bound by:
 *        - Elixir: `def <type>_<method>` stubs in native.ex
 *        - Node:   `pub fn <method>` inside each `#[napi] impl Js<Type>` block
 *   3. Compares them, honoring scripts/binding-coverage.json:
 *        ignore -> dropped from the core surface (never expected to be bound)
 *        todo   -> reported as a warning, not a failure (tracked backlog)
 *
 * Exit status is non-zero when a core method is neither bound nor tracked (new
 * drift), or when a binding refers to a method that no longer exists in core.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const REPO = fileURLToPath(new URL("..", import.meta.url));
const CONFIG = "scripts/binding-coverage.json";

// Core handle types and the source files whose `impl <Type> { .. }` blocks
// define their public surface.
const CORE_TYPES: Record<string, string[]> = {
  Model: ["src/model.rs", "src/vocab.rs"],
  Context: ["src/context.rs"],
  Sequence: ["src/sequence.rs"],
};

const ELIXIR_NATIVE = "packages/elixir/lib/rusty_llama/native.ex";
const NODE_LIB = "packages/node/src/lib.rs";

const PUB_FN = /\bpub\s+fn\s+([a-z_][a-z0-9_]*)\s*[(<]/g;
const TOKEN_OPTION = /\btoken_option!\s*\(\s*([a-z_][a-z0-9_]*)/g;

type Bindings = Record<string, Set<string>>;
type Config = { ignore: Record<string, string>; todo: Record<string, string> };

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function read(rel: string): string {
  try {
    return readFileSync(join(REPO, rel), "utf8");
  } catch {
    console.error(`error: expected file not found: ${rel}`);
    process.exit(1);
  }
}

function matchAll(re: RegExp, text: string): string[] {
  const out: string[] = [];
  for (const m of text.matchAll(re)) out.push(m[1]);
  return out;
}

/**
 * Return the bodies of every inherent `impl <typeName> {` block. Skips trait
 * impls (`impl Foo for <typeName>`) by anchoring on a `{` right after the name.
 */
function implBlocks(source: string, typeName: string): string[] {
  const blocks: string[] = [];
  const header = new RegExp(`\\bimpl\\s+${escapeRegExp(typeName)}\\s*\\{`, "g");
  for (const m of source.matchAll(header)) {
    const start = m.index + m[0].length - 1; // position of the opening brace
    let depth = 0;
    for (let i = start; i < source.length; i++) {
      const c = source[i];
      if (c === "{") depth++;
      else if (c === "}") {
        depth--;
        if (depth === 0) {
          blocks.push(source.slice(start + 1, i));
          break;
        }
      }
    }
  }
  return blocks;
}

function coreMethods(): Bindings {
  const out: Bindings = {};
  for (const [typeName, files] of Object.entries(CORE_TYPES)) {
    const methods = new Set<string>();
    for (const rel of files) {
      const src = read(rel);
      for (const body of implBlocks(src, typeName)) {
        for (const name of matchAll(PUB_FN, body)) methods.add(name);
        for (const name of matchAll(TOKEN_OPTION, body)) methods.add(name);
      }
    }
    out[typeName] = methods;
  }
  return out;
}

function elixirBindings(): Bindings {
  const src = read(ELIXIR_NATIVE);
  const out: Bindings = {};
  for (const t of Object.keys(CORE_TYPES)) out[t] = new Set<string>();
  const prefixes: Record<string, string> = {
    model: "Model",
    context: "Context",
    sequence: "Sequence",
  };
  for (const m of src.matchAll(/\bdef\s+([a-z_][a-z0-9_]*)\s*\(/g)) {
    const name = m[1];
    const prefix = name.split("_", 1)[0];
    const typeName = prefixes[prefix];
    if (typeName && name.includes("_")) {
      out[typeName].add(name.slice(prefix.length + 1));
    }
  }
  return out;
}

function nodeBindings(): Bindings {
  const src = read(NODE_LIB);
  const out: Bindings = {};
  for (const typeName of Object.keys(CORE_TYPES)) {
    const names = new Set<string>();
    for (const body of implBlocks(src, `Js${typeName}`)) {
      for (const name of matchAll(PUB_FN, body)) names.add(name);
    }
    out[typeName] = names;
  }
  return out;
}

function loadConfig(): Config {
  const cfg = JSON.parse(read(CONFIG));
  return { ignore: cfg.ignore ?? {}, todo: cfg.todo ?? {} };
}

function main(): number {
  const { ignore, todo } = loadConfig();
  const core = coreMethods();
  const elixir = elixirBindings();
  const node = nodeBindings();

  // Core methods minus the [ignore] bucket, per type.
  const qualified: Bindings = {};
  for (const [typeName, methods] of Object.entries(core)) {
    qualified[typeName] = new Set(
      [...methods].filter((m) => !(`${typeName}::${m}` in ignore)),
    );
  }

  const failures: string[] = [];
  const warnings: string[] = [];

  // Stale bindings: bound symbols that no longer exist in core.
  for (const [lang, bound] of [["elixir", elixir], ["node", node]] as const) {
    for (const [typeName, names] of Object.entries(bound)) {
      for (const name of [...names].sort()) {
        if (!core[typeName].has(name)) {
          failures.push(
            `stale ${lang} binding: ${typeName}::${name} is bound but not present in core`,
          );
        }
      }
    }
  }

  // Missing bindings.
  for (const [lang, bound] of [["elixir", elixir], ["node", node]] as const) {
    for (const typeName of Object.keys(CORE_TYPES)) {
      const missing = [...qualified[typeName]]
        .filter((m) => !bound[typeName].has(m))
        .sort();
      for (const name of missing) {
        const key = `${typeName}::${name}`;
        const msg = `${lang}: ${typeName}::${name} not bound`;
        if (key in todo) warnings.push(`${msg}  (todo: ${todo[key]})`);
        else failures.push(msg);
      }
    }
  }

  // Report.
  const total = Object.values(qualified).reduce((n, s) => n + s.size, 0);
  console.log(`core methods (after ignore): ${total}`);
  for (const typeName of Object.keys(CORE_TYPES)) {
    const q = qualified[typeName];
    const e = [...q].filter((m) => elixir[typeName].has(m)).length;
    const n = [...q].filter((m) => node[typeName].has(m)).length;
    console.log(
      `  ${typeName.padEnd(9)} elixir ${e}/${q.size}   node ${n}/${q.size}`,
    );
  }

  if (warnings.length) {
    console.log(`\n${warnings.length} tracked todo(s):`);
    for (const w of warnings) console.log(`  - ${w}`);
  }

  if (failures.length) {
    console.log(`\n${failures.length} FAILURE(s):`);
    for (const f of failures) console.log(`  - ${f}`);
    console.log(
      "\nAdd the method to the bindings, or record it in " +
        "scripts/binding-coverage.json (ignore or todo).",
    );
    return 1;
  }

  console.log("\nOK: no untracked binding drift.");
  return 0;
}

process.exit(main());
