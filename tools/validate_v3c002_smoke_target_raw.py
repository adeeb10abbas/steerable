#!/usr/bin/env python3
"""Rehash an existing V3-C002 excluded smoke block on its target PVC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


SCHEMA = "vla-wam-shared-v3c002-smoke-target-raw-rehash-receipt-v1"
EXPECTED_ORDER = [
    "canonical_left",
    "inverse_reference_right",
    "canonical_right",
    "inverse_reference_left",
]


class ValidationError(RuntimeError):
    """Raised when retained smoke evidence does not match its binding."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValidationError(f"required target file is absent: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def resolve_bound_path(path_value: object, *, source_root: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else source_root / path


def rehash_bound_value(
    value: Any,
    *,
    label: str,
    source_root: Path,
    seen: dict[tuple[str, str], int],
) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = resolve_bound_path(value["path"], source_root=source_root).resolve()
            if not path.is_file():
                raise ValidationError(f"{label} target artifact is absent: {path}")
            if type(value["bytes"]) is not int or value["bytes"] != path.stat().st_size:
                raise ValidationError(f"{label} byte count changed: {path}")
            if value["sha256"] != sha256_file(path):
                raise ValidationError(f"{label} digest changed: {path}")
            seen[(str(path), str(value["sha256"]))] = path.stat().st_size
            return
        for key, item in value.items():
            rehash_bound_value(
                item,
                label=f"{label}.{key}",
                source_root=source_root,
                seen=seen,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rehash_bound_value(
                item,
                label=f"{label}[{index}]",
                source_root=source_root,
                seen=seen,
            )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON at {path}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.smoke_root.resolve()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValidationError(f"refusing to overwrite target receipt: {output}")
    if not source_root.is_dir():
        raise ValidationError(f"source root is absent: {source_root}")
    observed_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed_commit != args.expected_source_commit:
        raise ValidationError(
            f"source checkout differs: {observed_commit} != {args.expected_source_commit}"
        )

    marker_path = root / "excluded_smoke/seed12000/completed_block.json"
    marker = read_json(marker_path)
    if not isinstance(marker, dict):
        raise ValidationError("completed marker is not an object")
    records = marker.get("raw_episodes")
    if (
        marker.get("schema_version") != "vla-wam-shared-v3c002-completed-block-v1"
        or marker.get("status") != "completed_excluded_smoke_block"
        or marker.get("authorization_mode") != "excluded_smoke"
        or marker.get("execution_order") != EXPECTED_ORDER
        or not isinstance(records, list)
        or len(records) != 4
    ):
        raise ValidationError("completed excluded smoke marker changed or is incomplete")

    seen: dict[tuple[str, str], int] = {}
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValidationError(f"raw record {index} is not an object")
        raw_path = Path(str(record.get("path", ""))).resolve()
        raw_binding = file_binding(raw_path)
        if any(record.get(key) != raw_binding[key] for key in ("bytes", "sha256")):
            raise ValidationError(f"raw episode binding {index} changed")
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1:
            raise ValidationError(f"raw episode {index} is not exactly one JSONL row")
        row = json.loads(lines[0])
        if not isinstance(row, dict):
            raise ValidationError(f"raw episode {index} is not an object")
        if row.get("cell_id") != record.get("cell_id"):
            raise ValidationError(f"raw episode {index} cell identity changed")
        if row.get("prompt_condition") != EXPECTED_ORDER[index]:
            raise ValidationError(f"raw episode {index} order changed")
        if (
            row.get("authorization_mode") != "excluded_smoke"
            or row.get("excluded_from_behavioral_denominators") is not True
            or type(row.get("model_request_count")) is not int
            or row["model_request_count"] < 1
        ):
            raise ValidationError(f"raw episode {index} is not a valid excluded smoke row")
        rehash_bound_value(
            row,
            label=f"raw_episode[{index}]",
            source_root=source_root,
            seen=seen,
        )
        seen[(raw_binding["path"], raw_binding["sha256"])] = int(raw_binding["bytes"])
        rows.append(row)

    initial_hashes = {row.get("initial_state_sha256") for row in rows}
    request0_hashes = {row.get("request0_pair_identity_sha256") for row in rows}
    if len(initial_hashes) != 1 or None in initial_hashes:
        raise ValidationError("smoke cells do not share one initial state")
    if len(request0_hashes) != 1 or None in request0_hashes:
        raise ValidationError("smoke cells do not share one request-zero identity")

    auxiliary_names = [
        "smoke_runner.command.sh",
        "smoke_runner.environment.json",
        "smoke_runner.log",
        "smoke_runner.exit_code",
        "runtime_observed.json",
        "runtime_manifest.json",
        "lane_plan.json",
        "excluded_smoke/seed12000/completed_block.json",
    ]
    auxiliary_bindings = []
    for name in auxiliary_names:
        binding = file_binding(root / name)
        auxiliary_bindings.append(binding)
        seen[(binding["path"], binding["sha256"])] = int(binding["bytes"])
    invocation_binding = file_binding(args.invocation)
    environment_binding = file_binding(args.environment)
    validator_binding = file_binding(Path(__file__))
    for binding in (invocation_binding, environment_binding, validator_binding):
        seen[(binding["path"], binding["sha256"])] = int(binding["bytes"])

    receipt = {
        "schema_version": SCHEMA,
        "status": "passed_excluded_smoke_target_raw_rehash",
        "passed": True,
        "completed_cells": 4,
        "cell_ids": [row["cell_id"] for row in rows],
        "execution_order": [row["prompt_condition"] for row in rows],
        "model_request_count": sum(row["model_request_count"] for row in rows),
        "behavioral_episode_count": 0,
        "excluded_from_behavioral_denominators": True,
        "initial_state_sha256": rows[0]["initial_state_sha256"],
        "request0_pair_identity_sha256": rows[0]["request0_pair_identity_sha256"],
        "source_root": str(source_root),
        "source_commit": observed_commit,
        "validator": validator_binding,
        "invocation": invocation_binding,
        "environment": environment_binding,
        "unique_raw_bindings_rehashed": len(seen),
        "raw_bytes_rehashed": sum(seen.values()),
        "auxiliary_bindings": auxiliary_bindings,
        "raw_episode_bindings": records,
        "validation_location": (
            "target simulator PVC/runtime with direct access to every absolute smoke "
            "raw path; repository-relative source bindings resolved against source_root"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "sha256": sha256_file(output),
                "bindings": receipt["unique_raw_bindings_rehashed"],
                "bytes": receipt["raw_bytes_rehashed"],
                "requests": receipt["model_request_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
