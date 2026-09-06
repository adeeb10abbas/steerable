#!/usr/bin/env python3
"""Derive a strict V4 lane spec with audited dotted-path overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_override(value: str) -> tuple[list[str], Any]:
    path, separator, raw = value.partition("=")
    keys = path.split(".")
    if separator != "=" or not all(keys):
        raise ValueError(f"invalid dotted-path override: {value}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"override value must be JSON: {value}") from exc
    return keys, parsed


def derive_spec(
    *,
    source_path: Path,
    overrides: list[str],
) -> dict[str, Any]:
    spec = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("source lane spec must be a JSON object")
    source_dir = source_path.parent.resolve()
    for role in ("policy", "simulator"):
        for binding in spec.get(role, {}).get("file_bindings", []):
            source = binding.get("source")
            if isinstance(source, str) and not Path(source).is_absolute():
                binding["source"] = str((source_dir / source).resolve())
    for override in overrides:
        keys, value = parse_override(override)
        cursor: Any = spec
        for key in keys[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                raise ValueError(f"override path does not exist: {'.'.join(keys)}")
            cursor = cursor[key]
        leaf = keys[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ValueError(f"override path does not exist: {'.'.join(keys)}")
        cursor[leaf] = value
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = derive_spec(
        source_path=args.source.resolve(),
        overrides=args.set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(spec, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
