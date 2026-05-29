import type { Api } from "../ir.ts";

export interface GeneratedFile {
  /** Repo-relative path. */
  path: string;
  contents: string;
}

/** Loads a hand-written snippet for a `Type::method` marked `manual`. */
export type ManualLoader = (key: string) => string;

/** A language backend: turns the IR into source files. */
export interface Backend {
  id: string;
  generate(api: Api, manual: ManualLoader): GeneratedFile[];
}
