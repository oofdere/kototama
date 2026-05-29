/**
 * Extractor: rustdoc JSON -> the language-neutral API IR (scripts/binding/api.json).
 *
 * Runs in both Node (>= 22.18) and Deno. Reads a rustdoc JSON file (produced by
 * the compiler, so macros are expanded and types resolved) and maps the
 * inherent methods of the core handle types into the IR that backends consume.
 *
 * Produce the rustdoc JSON first (CPU build, no GPU):
 *
 *   git submodule update --init --depth 1 llama-sys/llama.cpp
 *   cargo +nightly rustdoc -p rusty-llama -- --output-format json -Z unstable-options
 *   node scripts/binding/extract.ts          # reads target/doc/rusty_llama.json
 *   deno run --allow-read --allow-write scripts/binding/extract.ts
 *
 * Pass a different rustdoc JSON path as the first argument if needed.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import process from "node:process";
import type {
  Api,
  ApiType,
  BindingSpec,
  IrType,
  Method,
  Param,
  Receiver,
} from "./ir.ts";
import { IR_SCHEMA_VERSION } from "./ir.ts";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
const DEFAULT_RUSTDOC = "target/doc/rusty_llama.json";
const SPEC_PATH = "scripts/binding-spec.json";
const COVERAGE_PATH = "scripts/binding-coverage.json";
const OUT_PATH = "scripts/binding/api.json";

function readJson(rel: string): any {
  try {
    return JSON.parse(readFileSync(REPO + rel, "utf8"));
  } catch (e) {
    console.error(`error: cannot read ${rel}: ${(e as Error).message}`);
    process.exit(1);
  }
}

const INT_BITS: Record<string, number> = {
  i8: 8, i16: 16, i32: 32, i64: 64, isize: 64,
  u8: 8, u16: 16, u32: 32, u64: 64, usize: 64,
};

function mapPrim(p: string): IrType {
  if (p === "bool") return { k: "bool" };
  if (p === "str") return { k: "string" };
  if (p in INT_BITS) return { k: "int", rust: p, signed: p.startsWith("i"), bits: INT_BITS[p] };
  if (p === "f32") return { k: "float", rust: "f32", bits: 32 };
  if (p === "f64") return { k: "float", rust: "f64", bits: 64 };
  return { k: "opaque", rust: p };
}

/** rustdoc Type -> IR type. `selfName` resolves `Self`; `isHandle` flags handle types. */
function mapType(t: any, selfName: string, isHandle: (n: string) => boolean): IrType {
  if (t == null) return { k: "unit" };
  if ("primitive" in t) return mapPrim(t.primitive);
  if ("tuple" in t) return t.tuple.length === 0 ? { k: "unit" } : { k: "opaque", rust: "tuple" };
  if ("slice" in t) return { k: "vec", of: mapType(t.slice, selfName, isHandle) };
  if ("borrowed_ref" in t) {
    const inner = t.borrowed_ref.type;
    const h = handleName(inner, selfName, isHandle);
    if (h) return { k: "handle", type: h, ref: t.borrowed_ref.is_mutable ? "mut" : "shared" };
    return mapType(inner, selfName, isHandle); // &str, &[i32], &CStr: ref is irrelevant
  }
  if ("generic" in t) {
    if (t.generic === "Self") return { k: "handle", type: selfName, ref: "owned" };
    return { k: "opaque", rust: t.generic };
  }
  if ("resolved_path" in t) {
    const rp = t.resolved_path;
    const name: string = rp.path;
    const short = name.split("::").pop() ?? name; // rustdoc may give crate::Sequence
    const args: any[] = (rp.args?.angle_bracketed?.args ?? [])
      .map((a: any) => a.type)
      .filter((x: any) => x !== undefined);
    switch (short) {
      case "Option": return { k: "option", of: mapType(args[0], selfName, isHandle) };
      case "Vec": return { k: "vec", of: mapType(args[0], selfName, isHandle) };
      case "Range": return { k: "range", elem: mapType(args[0], selfName, isHandle) };
      case "Result":
        return { k: "result", ok: mapType(args[0], selfName, isHandle), err: mapType(args[1] ?? null, selfName, isHandle) };
      case "String": case "CString": case "CStr": case "str":
        return { k: "string" };
      case "Token": case "Pos": case "SeqId":
        return { k: "int", rust: "i32", signed: true, bits: 32 };
    }
    if (short === selfName || isHandle(short)) return { k: "handle", type: short, ref: "owned" };
    return { k: "opaque", rust: name };
  }
  return { k: "opaque", rust: JSON.stringify(t) };
}

/** If `t` denotes a handle type (Self or a known handle), return its name. */
function handleName(t: any, selfName: string, isHandle: (n: string) => boolean): string | null {
  if (t && "generic" in t && t.generic === "Self") return selfName;
  if (t && "resolved_path" in t) {
    const n = (t.resolved_path.path as string).split("::").pop() ?? t.resolved_path.path;
    if (n === selfName || isHandle(n)) return n;
  }
  return null;
}

function mapFn(fn: any, selfName: string, isHandle: (n: string) => boolean): { receiver: Receiver; params: Param[]; ret: IrType } {
  let receiver: Receiver = "none";
  const params: Param[] = [];
  for (const [pname, ptype] of fn.sig.inputs as [string, any][]) {
    if (pname === "self") {
      receiver = ptype && "borrowed_ref" in ptype
        ? (ptype.borrowed_ref.is_mutable ? "mut" : "ref")
        : "mut"; // owned `self`
    } else {
      params.push({ name: pname, type: mapType(ptype, selfName, isHandle) });
    }
  }
  return { receiver, params, ret: mapType(fn.sig.output, selfName, isHandle) };
}

export function extract(rustdocRel: string = DEFAULT_RUSTDOC): Api {
  const doc = readJson(rustdocRel);
  const idx = doc.index as Record<string, any>;
  const spec = readJson(SPEC_PATH) as BindingSpec;
  const ignore = (readJson(COVERAGE_PATH).ignore ?? {}) as Record<string, string>;

  const handleTypes = Object.keys(spec.types);
  const isHandle = (n: string) => handleTypes.includes(n);

  // Collect inherent (trait == null) impl methods, grouped by the type they're on.
  const byType: Record<string, any[]> = {};
  for (const t of handleTypes) byType[t] = [];
  for (const item of Object.values(idx)) {
    const im = item?.inner?.impl;
    if (!im || im.trait !== null) continue;
    const forName: string | undefined = im.for?.resolved_path?.path;
    if (!forName || !isHandle(forName)) continue;
    for (const fid of im.items) {
      const fnItem = idx[String(fid)];
      if (fnItem?.inner?.function) byType[forName].push(fnItem);
    }
  }

  const typeDoc = (name: string): string | null => {
    for (const item of Object.values(idx)) {
      if (item?.inner?.struct && item.name === name) return item.docs ?? null;
    }
    return null;
  };

  const types: ApiType[] = handleTypes.map((typeName) => {
    const cfg = spec.types[typeName];
    const methods: Method[] = byType[typeName]
      .filter((fnItem) => !(`${typeName}::${fnItem.name}` in ignore))
      .map((fnItem): Method => {
        const key = `${typeName}::${fnItem.name}`;
        const { receiver, params, ret } = mapFn(fnItem.inner.function, typeName, isHandle);
        return {
          name: fnItem.name,
          doc: fnItem.docs ?? null,
          receiver,
          params,
          ret,
          hints: spec.hints?.[key] ?? {},
          manual: spec.manual?.[key] ?? [],
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));
    return { name: typeName, kind: cfg.kind, mutable: cfg.mutable, doc: typeDoc(typeName), methods };
  });

  const crate = idx[String(doc.root)]?.name ?? "rusty_llama";
  return { schema: IR_SCHEMA_VERSION, crate, types };
}

function isMain(): boolean {
  const g = globalThis as any;
  if (g.Deno) return (import.meta as any).main === true;
  return process.argv[1] !== undefined && pathToFileURL(process.argv[1]).href === import.meta.url;
}

if (isMain()) {
  const rustdocRel = process.argv[2] ?? DEFAULT_RUSTDOC;
  const api = extract(rustdocRel);
  writeFileSync(REPO + OUT_PATH, JSON.stringify(api, null, 2) + "\n");
  const n = api.types.reduce((s, t) => s + t.methods.length, 0);
  console.log(`wrote ${OUT_PATH}: ${api.types.length} types, ${n} methods (from ${rustdocRel})`);
}
