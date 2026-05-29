/**
 * The language-neutral API IR — the "typed JSON" contract.
 *
 * Backends (Elixir, Node/JS, future languages) consume this and never parse
 * Rust themselves. Produced by scripts/binding/extract.ts from the core crate
 * source plus scripts/binding-spec.json (hints text can't infer) and the
 * `ignore` list in scripts/binding-coverage.json (shared with the coverage
 * checker, so exclusions have a single source of truth).
 */

export const IR_SCHEMA_VERSION = "1";

/** A normalized type. Backends map each `k` to their own wire/surface type. */
export type IrType =
  | { k: "unit" }
  | { k: "bool" }
  | { k: "int"; rust: string; signed: boolean; bits: number }
  | { k: "float"; rust: string; bits: number }
  /** `borrowed` = came from `&str`/`&CStr` (needs owning); `repr:"cstr"` = `CStr`. */
  | { k: "string"; borrowed?: boolean; repr?: "cstr" }
  | { k: "option"; of: IrType }
  /** `borrowed` = came from `&[T]` (needs `.to_vec()`); else an owned `Vec<T>`. */
  | { k: "vec"; of: IrType; borrowed?: boolean }
  | { k: "range"; elem: IrType }
  | { k: "result"; ok: IrType; err: IrType }
  /** A reference to another handle type (e.g. `&mut Sequence`, owned return). */
  | { k: "handle"; type: string; ref: "owned" | "shared" | "mut" }
  /** Anything not yet modeled; backends must treat as manual. */
  | { k: "opaque"; rust: string };

/** Method receiver: associated fn, `&self`, or `&mut self`. */
export type Receiver = "none" | "ref" | "mut";

export interface Param {
  name: string;
  type: IrType;
}

export interface MethodHints {
  /** BEAM: schedule on a dirty CPU scheduler (long-running / FFI round-trip). */
  cpuBound?: boolean;
}

export interface Method {
  name: string;
  doc: string | null;
  receiver: Receiver;
  params: Param[];
  ret: IrType;
  hints: MethodHints;
  /** Backend ids that hand-write this method instead of generating it. */
  manual: string[];
}

export type TypeKind = "handle" | "value";

export interface ApiType {
  name: string;
  kind: TypeKind;
  /** Shared-mut: needs interior mutability (Mutex) when shared across calls. */
  mutable: boolean;
  doc: string | null;
  methods: Method[];
}

export interface Api {
  schema: string;
  crate: string;
  types: ApiType[];
}

/** Shape of scripts/binding-spec.json (hints the text extractor can't infer). */
export interface BindingSpec {
  types: Record<string, { kind: TypeKind; mutable: boolean }>;
  hints: Record<string, MethodHints>;
  manual: Record<string, string[]>;
}
