#!/usr/bin/env python3
"""Check that the Elixir and Node bindings expose the core rusty-llama API.

This is a *build-free* checker: it parses Rust/Elixir source as text, so it
needs no llama.cpp submodule, no GPU backend, and no compilation. That makes it
cheap to run in CI as a drift gate.

What it does:
  1. Extracts the public methods of the core handle types (Model, Context,
     Sequence) from `src/`, including macro-generated `token_option!` getters.
  2. Extracts the symbols actually bound by:
       - Elixir: `def <type>_<method>` stubs in native.ex
       - Node:   `pub fn <method>` inside each `#[napi] impl Js<Type>` block
  3. Compares them, honoring scripts/binding-coverage.toml:
       [ignore] -> dropped from the core surface (never expected to be bound)
       [todo]   -> reported as a warning, not a failure (tracked backlog)

Exit status is non-zero when a core method is neither bound nor tracked
(new drift), or when a binding refers to a method that no longer exists in
core (stale binding).
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "scripts" / "binding-coverage.toml"

# Core handle types and the source files whose `impl <Type> { .. }` blocks
# define their public surface.
CORE_TYPES = {
    "Model": ["src/model.rs", "src/vocab.rs"],
    "Context": ["src/context.rs"],
    "Sequence": ["src/sequence.rs"],
}

ELIXIR_NATIVE = "packages/elixir/lib/rusty_llama/native.ex"
NODE_LIB = "packages/node/src/lib.rs"

PUB_FN = re.compile(r"\bpub\s+fn\s+([a-z_][a-z0-9_]*)\s*[(<]")
TOKEN_OPTION = re.compile(r"\btoken_option!\s*\(\s*([a-z_][a-z0-9_]*)")


def read(rel: str) -> str:
    path = REPO / rel
    if not path.is_file():
        sys.exit(f"error: expected file not found: {rel}")
    return path.read_text()


def impl_blocks(source: str, type_name: str) -> list[str]:
    """Return the bodies of every inherent `impl <type_name> {` block.

    Skips trait impls (`impl Foo for <type_name>`) by anchoring on a `{`
    immediately following the type name.
    """
    blocks: list[str] = []
    header = re.compile(rf"\bimpl\s+{re.escape(type_name)}\s*\{{")
    for m in header.finditer(source):
        depth = 0
        start = m.end() - 1  # position of the opening brace
        for i in range(start, len(source)):
            c = source[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(source[start + 1 : i])
                    break
    return blocks


def core_methods() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for type_name, files in CORE_TYPES.items():
        methods: set[str] = set()
        for rel in files:
            src = read(rel)
            for body in impl_blocks(src, type_name):
                methods.update(PUB_FN.findall(body))
                methods.update(TOKEN_OPTION.findall(body))
        out[type_name] = methods
    return out


def elixir_bindings() -> dict[str, set[str]]:
    src = read(ELIXIR_NATIVE)
    out: dict[str, set[str]] = {t: set() for t in CORE_TYPES}
    prefixes = {"model": "Model", "context": "Context", "sequence": "Sequence"}
    for m in re.finditer(r"\bdef\s+([a-z_][a-z0-9_]*)\s*\(", src):
        name = m.group(1)
        prefix = name.split("_", 1)[0]
        type_name = prefixes.get(prefix)
        if type_name and "_" in name:
            out[type_name].add(name[len(prefix) + 1 :])
    return out


def node_bindings() -> dict[str, set[str]]:
    src = read(NODE_LIB)
    out: dict[str, set[str]] = {t: set() for t in CORE_TYPES}
    for type_name in CORE_TYPES:
        for body in impl_blocks(src, f"Js{type_name}"):
            out[type_name].update(PUB_FN.findall(body))
    return out


def load_config() -> tuple[dict[str, str], dict[str, str]]:
    with CONFIG.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("ignore", {}), cfg.get("todo", {})


def main() -> int:
    ignore, todo = load_config()
    core = core_methods()
    elixir = elixir_bindings()
    node = node_bindings()

    # Build the set of all qualified core methods (Type::method).
    qualified: dict[str, set[str]] = {}
    for type_name, methods in core.items():
        qualified[type_name] = {m for m in methods if f"{type_name}::{m}" not in ignore}

    failures: list[str] = []
    warnings: list[str] = []

    # Stale bindings: bound symbols that no longer exist in core.
    for lang, bound in (("elixir", elixir), ("node", node)):
        for type_name, names in bound.items():
            for name in sorted(names):
                if name not in core[type_name]:
                    failures.append(
                        f"stale {lang} binding: {type_name}::{name} is bound but "
                        f"not present in core"
                    )

    # Missing bindings.
    for lang, bound in (("elixir", elixir), ("node", node)):
        for type_name in CORE_TYPES:
            for name in sorted(qualified[type_name] - bound[type_name]):
                key = f"{type_name}::{name}"
                msg = f"{lang}: {type_name}::{name} not bound"
                if key in todo:
                    warnings.append(f"{msg}  (todo: {todo[key]})")
                else:
                    failures.append(msg)

    # Report.
    total = sum(len(v) for v in qualified.values())
    print(f"core methods (after [ignore]): {total}")
    for type_name in CORE_TYPES:
        e = len(qualified[type_name] & elixir[type_name])
        n = len(qualified[type_name] & node[type_name])
        tot = len(qualified[type_name])
        print(f"  {type_name:<9} elixir {e}/{tot}   node {n}/{tot}")

    if warnings:
        print(f"\n{len(warnings)} tracked todo(s):")
        for w in warnings:
            print(f"  - {w}")

    if failures:
        print(f"\n{len(failures)} FAILURE(s):")
        for fmsg in failures:
            print(f"  - {fmsg}")
        print(
            "\nAdd the method to the bindings, or record it in "
            "scripts/binding-coverage.toml ([ignore] or [todo])."
        )
        return 1

    print("\nOK: no untracked binding drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
