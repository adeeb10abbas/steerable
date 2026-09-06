#!/usr/bin/env python3
"""Compile complete path-seed evidence for one horizontal G3 scale candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.model_blind_g3 import (  # noqa: E402
    canonical_json_bytes,
    compile_path_scale_receipt,
    path_receipt_schema,
    plan_schema,
    sha256_file,
    validate_path_scale_receipt,
    validate_plan_payload,
)


def _load_canonical(path: Path, *, schema: str) -> dict[str, Any]:
    body = path.read_bytes()
    value = json.loads(body)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"{path}: schema differs")
    if body != canonical_json_bytes(value):
        raise ValueError(f"{path}: JSON is not canonical")
    return value


def _verify_artifact_records(value: Any, *, label: str) -> int:
    verified = 0
    if isinstance(value, Mapping):
        if {"path", "sha256", "bytes"}.issubset(value):
            path = Path(str(value["path"]))
            if not path.is_file():
                raise ValueError(f"{label}: artifact is missing: {path}")
            if sha256_file(path) != value["sha256"]:
                raise ValueError(f"{label}: artifact hash differs: {path}")
            if path.stat().st_size != value["bytes"]:
                raise ValueError(f"{label}: artifact byte count differs: {path}")
            verified += 1
        for key, item in value.items():
            verified += _verify_artifact_records(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            verified += _verify_artifact_records(item, label=f"{label}[{index}]")
    return verified


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(dict(payload))
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def compile_receipts(
    *,
    plan_path: Path,
    scale: float,
    receipts_root: Path,
    output_path: Path,
    verify_external: bool = False,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan_body = plan_path.read_bytes()
    plan = json.loads(plan_body)
    if not isinstance(plan, dict):
        raise ValueError(f"{plan_path}: plan must be a JSON object")
    fixture_id = plan.get("fixture_id")
    if not isinstance(fixture_id, str):
        raise ValueError(f"{plan_path}: plan lacks fixture_id")
    plan = _load_canonical(plan_path, schema=plan_schema(fixture_id))
    validate_plan_payload(plan)
    plan_receipt = {
        "path": str(plan_path),
        "sha256": sha256_file(plan_path),
    }
    receipt_paths = sorted(receipts_root.resolve().rglob("g3_path_seed_receipt.json"))
    if not receipt_paths:
        raise ValueError("no G3 path seed receipts found")

    receipts: list[dict[str, Any]] = []
    receipt_files: dict[int, dict[str, Any]] = {}
    verified_artifact_count = 0
    for path in receipt_paths:
        receipt = _load_canonical(path, schema=path_receipt_schema(fixture_id))
        if verify_external:
            verified_artifact_count += _verify_artifact_records(
                receipt.get("checks"), label=str(path)
            )
            verified_artifact_count += _verify_artifact_records(
                receipt.get("artifacts"), label=f"{path}.artifacts"
            )
        seed = receipt.get("environment_seed")
        if type(seed) is not int:
            raise ValueError(f"{path}: environment_seed is invalid")
        receipt_files[seed] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        receipts.append(receipt)

    aggregate = compile_path_scale_receipt(
        plan=plan,
        plan_receipt=plan_receipt,
        scale=scale,
        path_seed_receipts=receipts,
        path_seed_receipt_files=receipt_files,
    )
    validate_path_scale_receipt(aggregate, plan=plan)
    aggregate.update(
        {
            "plan_file": {
                "path": str(plan_path),
                "sha256": plan_receipt["sha256"],
                "bytes": plan_path.stat().st_size,
            },
            "verified_external_artifact_count": verified_artifact_count,
        }
    )
    _write_exclusive(output_path.resolve(), aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verify-external",
        action="store_true",
        help="Verify external evidence files referenced by each seed receipt",
    )
    args = parser.parse_args(argv)
    report = compile_receipts(
        plan_path=args.plan,
        scale=args.scale,
        receipts_root=args.receipts_root,
        output_path=args.out,
        verify_external=args.verify_external,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
