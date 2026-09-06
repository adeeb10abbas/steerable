#!/usr/bin/env python3
"""Compile horizontal G3 path-scale ladder and scripted-controller evidence."""

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
    _scripted_check_key,
    canonical_json_bytes,
    compile_g3_aggregate_receipt,
    path_scale_receipt_schema,
    scripted_receipt_schema,
    sha256_file,
    validate_g3_aggregate_receipt,
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


def _discover_path_scale_receipts(root: Path) -> list[Path]:
    paths = sorted(root.resolve().rglob("g3_path_scale_receipt.json"))
    if paths:
        return paths
    return sorted(root.resolve().glob("g3_path_scale_receipt*.json"))


def _discover_scripted_receipts(root: Path) -> list[Path]:
    paths = sorted(root.resolve().rglob("g3_scripted_check_receipt.json"))
    if paths:
        return paths
    return sorted(root.resolve().rglob("*scripted*receipt*.json"))


def compile_receipts(
    *,
    plan_path: Path,
    path_scale_receipts_root: Path,
    scripted_receipts_root: Path | None,
    output_path: Path,
    verify_external: bool = False,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan_body = plan_path.read_bytes()
    plan = json.loads(plan_body)
    if not isinstance(plan, dict):
        raise ValueError(f"{plan_path}: plan must be a JSON object")
    fixture_id = str(plan.get("fixture_id", "horizontal"))
    if plan_body != canonical_json_bytes(plan):
        raise ValueError(f"{plan_path}: JSON is not canonical")
    validate_plan_payload(plan)
    plan_receipt = {
        "path": str(plan_path),
        "sha256": sha256_file(plan_path),
    }

    path_scale_paths = _discover_path_scale_receipts(path_scale_receipts_root)
    path_scale_receipts: list[dict[str, Any]] = []
    path_scale_files: dict[float, dict[str, Any]] = {}
    verified_artifact_count = 0
    path_scale_schema = path_scale_receipt_schema(fixture_id)
    for path in path_scale_paths:
        receipt = _load_canonical(path, schema=path_scale_schema)
        scale = float(receipt["scale"])
        if verify_external:
            verified_artifact_count += _verify_artifact_records(
                receipt.get("path_seed_receipt_files_by_env_seed"),
                label=str(path),
            )
        path_scale_files[scale] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        path_scale_receipts.append(receipt)

    scripted_receipts: list[dict[str, Any]] | None = None
    scripted_files: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    if scripted_receipts_root is not None:
        scripted_receipts = []
        scripted_schema = scripted_receipt_schema(fixture_id)
        for path in _discover_scripted_receipts(scripted_receipts_root):
            receipt = _load_canonical(path, schema=scripted_schema)
            if verify_external:
                verified_artifact_count += _verify_artifact_records(
                    receipt.get("evidence"), label=str(path)
                )
            key = _scripted_check_key(receipt)
            scripted_files[key] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            scripted_receipts.append(receipt)

    aggregate = compile_g3_aggregate_receipt(
        plan=plan,
        plan_receipt=plan_receipt,
        path_scale_receipts=path_scale_receipts,
        path_scale_receipt_files=path_scale_files,
        scripted_check_receipts=scripted_receipts,
        scripted_check_receipt_files=scripted_files or None,
    )
    validate_g3_aggregate_receipt(aggregate, plan=plan)
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
    parser.add_argument("--path-scale-receipts-root", type=Path, required=True)
    parser.add_argument("--scripted-receipts-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verify-external",
        action="store_true",
        help="Verify external evidence files referenced by receipts",
    )
    args = parser.parse_args(argv)
    report = compile_receipts(
        plan_path=args.plan,
        path_scale_receipts_root=args.path_scale_receipts_root,
        scripted_receipts_root=args.scripted_receipts_root,
        output_path=args.out,
        verify_external=args.verify_external,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
