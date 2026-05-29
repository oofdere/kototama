/**
 * Node / NAPI-RS backend: IR -> packages/node/src/lib.rs.
 *
 * Generates a `#[napi]` class per handle type with one method per IR method.
 * Methods marked `manual` for "node" are spliced from scripts/binding/manual/node/.
 * Doc comments come from the IR (forwarded to JSDoc in the generated .d.ts).
 */

import type { ApiType, IrType, Method, Param } from "../ir.ts";
import type { Backend, GeneratedFile, ManualLoader } from "./types.ts";

const ID = "node";
const OUT = "packages/node/src/lib.rs";

/** Render a doc string as `///` lines at `indent`, avoiding trailing spaces. */
function docComment(doc: string | null, indent: string): string {
  if (!doc) return "";
  return doc.split("\n").map((l) => (l ? `${indent}/// ${l}` : `${indent}///`)).join("\n") + "\n";
}

function camel(s: string): string {
  const [head, ...rest] = s.split("_");
  return head + rest.map((p) => (p ? p[0].toUpperCase() + p.slice(1) : "")).join("");
}

function napiAttr(name: string): string {
  return name.includes("_") ? `#[napi(js_name = "${camel(name)}")]` : "#[napi]";
}

/** Owned napi Rust type for a parameter. */
function paramType(t: IrType): string {
  switch (t.k) {
    case "bool": return "bool";
    case "int": return t.rust === "usize" ? "u32" : t.rust;
    case "float": return t.rust === "f32" ? "f64" : t.rust;
    case "string": return "String";
    case "option": return `Option<${paramType(t.of)}>`;
    case "vec": return `Vec<${paramType(t.of)}>`;
    default: return "()";
  }
}

/** napi Rust return type (no `-> ` for unit). */
function retType(t: IrType): string {
  switch (t.k) {
    case "unit": return "";
    case "bool": return "bool";
    case "int": return t.rust === "usize" ? "i64" : t.rust;
    case "float": return "f64";
    case "string": return "String";
    case "option": return `Option<${retType(t.of)}>`;
    case "vec": return `Vec<${retType(t.of)}>`;
    case "result": return `Result<${retType(t.ok)}>`;
    case "handle": return `Js${t.type}`;
    default: return "()";
  }
}

/** A parameter's napi declaration(s) and the expression passed to the core call. */
function param(p: Param): { decls: string[]; arg: string } {
  const t = p.type;
  if (t.k === "range") {
    const elem = t.elem.k === "int" ? t.elem.rust : "i32";
    if (elem === "usize") return { decls: ["start: u32", "end: u32"], arg: "start as usize..end as usize" };
    return { decls: [`start: ${elem}`, `end: ${elem}`], arg: "start..end" };
  }
  let arg = p.name;
  if (t.k === "int" && t.rust === "usize") arg = `${p.name} as usize`;
  else if (t.k === "string" && t.borrowed) arg = `&${p.name}`;
  else if (t.k === "vec" && t.borrowed) arg = `&${p.name}`;
  else if (t.k === "option" && t.of.k === "string" && t.of.borrowed) arg = `${p.name}.as_deref()`;
  return { decls: [`${p.name}: ${paramType(t)}`], arg };
}

interface Ctx {
  methodName: string;
  typeMut: (name: string) => boolean;
}

/** Convert a core-typed expression `expr` into the napi return type. */
function lower(expr: string, t: IrType, ctx: Ctx): string {
  switch (t.k) {
    case "unit":
    case "bool":
      return expr;
    case "int": return t.rust === "usize" ? `${expr} as i64` : expr;
    case "float": return `${expr} as f64`;
    case "string": return t.repr === "cstr" ? `${expr}.to_string_lossy().into_owned()` : expr;
    case "option": {
      const inner = lower("x", t.of, ctx);
      return inner === "x" ? expr : `${expr}.map(|x| ${inner})`;
    }
    case "vec":
      if (t.of.k === "float") return `${expr}.iter().map(|&e| e as f64).collect()`;
      return t.borrowed ? `${expr}.to_vec()` : expr;
    case "result":
      return `${expr}\n            .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "${ctx.methodName} failed"))`;
    case "handle":
      return ctx.typeMut(t.type)
        ? `Js${t.type} { inner: Arc::new(Mutex::new(${expr})) }`
        : `Js${t.type} { inner: Arc::new(${expr}) }`;
    default: return expr;
  }
}

function emitMethod(t: ApiType, m: Method, typeMut: (n: string) => boolean): string {
  const ctx: Ctx = { methodName: m.name, typeMut };
  const decls: string[] = [];
  const args: string[] = [];
  for (const p of m.params) {
    const c = param(p);
    decls.push(...c.decls);
    args.push(c.arg);
  }
  const sig = ["&self", ...decls].join(", ");
  const target = t.mutable ? "self.inner.lock().unwrap()" : "self.inner";
  const call = `${target}.${m.name}(${args.join(", ")})`;
  const arrow = m.ret.k === "unit" ? "" : ` -> ${retType(m.ret)}`;
  const body = m.ret.k === "unit" ? `        ${call};` : `        ${lower(call, m.ret, ctx)}`;
  const doc = docComment(m.doc, "    ");
  return `${doc}    ${napiAttr(m.name)}\n    pub fn ${m.name}(${sig})${arrow} {\n${body}\n    }`;
}

function emitType(t: ApiType, manual: ManualLoader, typeMut: (n: string) => boolean): string {
  const inner = t.mutable ? `Arc<Mutex<rusty_llama::${t.name}>>` : `Arc<rusty_llama::${t.name}>`;
  const doc = docComment(t.doc, "");
  const methods = t.methods
    .map((m) => (m.manual.includes(ID) ? manual(`${t.name}::${m.name}`) : emitMethod(t, m, typeMut)))
    .join("\n\n");
  return `${doc}#[derive(Clone)]\n#[napi(js_name = "${t.name}")]\npub struct Js${t.name} {\n    inner: ${inner},\n}\n\n#[napi]\nimpl Js${t.name} {\n${methods}\n}`;
}

const HEADER = `// @generated by scripts/binding/backends/node.ts from scripts/binding/api.json.
// Do not edit by hand. Regenerate: npm run gen:ir && node scripts/binding/gen.ts node
// Methods marked \`manual\` live in scripts/binding/manual/node/ and are spliced in.
#![allow(dead_code, unused_imports, unused_variables)]
#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::let_unit_value, clippy::needless_borrow)]

use napi::*;
use napi_derive::napi;
use std::sync::Arc;
use std::sync::Mutex;
`;

const DECODE_ERROR = `#[napi(string_enum, js_name = "DecodeError")]
#[derive(Clone)]
pub enum JsDecodeError {
    SlotNotFound,
    Aborted,
    InvalidInput,
    FatalError,
}`;

export const nodeBackend: Backend = {
  id: ID,
  generate(api, manual): GeneratedFile[] {
    const mut = new Map(api.types.map((t) => [t.name, t.mutable]));
    const typeMut = (n: string) => mut.get(n) ?? false;
    const blocks = api.types.map((t) => emitType(t, manual, typeMut));
    const contents = `${HEADER}\n${blocks.join("\n\n")}\n\n${DECODE_ERROR}\n`;
    return [{ path: OUT, contents }];
  },
};
