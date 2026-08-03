#!/usr/bin/env python3
"""Merge model-specific native guard ledgers without concurrent writes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


KNOWN_SCHEMAS = {
    "vla-wam-shared-v2-native-thermal-interventions-v1",
    "vla-wam-shared-v2-native-thermal-invalid-attempts-v1",
}


def load_ledger(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    schema = payload.get("schema_version")
    events = payload.get("events")
    if schema not in KNOWN_SCHEMAS:
        raise RuntimeError(f"Unsupported native ledger schema in {path}: {schema!r}")
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise RuntimeError(f"Native ledger events must be a list of objects: {path}")
    return schema, events


def merge_ledgers(paths: list[Path], model_id: str) -> dict[str, Any]:
    if not paths:
        raise RuntimeError("At least one --input ledger is required")
    schema: str | None = None
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        source_schema, events = load_ledger(path)
        if schema is None:
            schema = source_schema
        elif source_schema != schema:
            raise RuntimeError(
                f"Cannot merge different native ledger schemas: {schema!r} and {source_schema!r}"
            )
        for event in events:
            event_id = event.get("id")
            if not isinstance(event_id, str) or not event_id:
                raise RuntimeError(f"Native ledger event in {path} has no string id")
            if event.get("model_id") != model_id:
                raise RuntimeError(
                    f"Native ledger event {event_id!r} has model_id={event.get('model_id')!r}, "
                    f"expected {model_id!r}"
                )
            if event_id in by_id:
                raise RuntimeError(f"Duplicate native ledger event id across inputs: {event_id}")
            by_id[event_id] = event
    assert schema is not None
    events = sorted(
        by_id.values(),
        key=lambda event: (str(event.get("started_at_utc", "")), event["id"]),
    )
    return {"schema_version": schema, "events": events}


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model_id not in args.output.name:
        raise RuntimeError(
            f"Model-specific output filename must include {args.model_id!r}: {args.output}"
        )
    write_atomic(args.output, merge_ledgers(args.input, args.model_id))


if __name__ == "__main__":
    main()
