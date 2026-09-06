#!/usr/bin/env python3
"""Compile fixed-scale C7 pilot G3 evidence against confirmatory scale selection."""

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
    aggregate_receipt_schema_g3,
    canonical_json_bytes,
    expected_scripted_check_keys,
    path_scale_receipt_schema,
    scripted_receipt_schema,
    sha256_file,
    validate_g3_aggregate_receipt,
    validate_path_scale_receipt,
    validate_plan_payload,
    validate_scripted_check_receipt,
)


FIXTURE_ID = "object_pair"
CONFIRMATORY_SCOPE = "confirmatory"
PILOT_SCOPE = "engineering_pilot"


def load_canonical(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    body = path.read_bytes()
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if body != canonical_json_bytes(value):
        raise ValueError(f"{path}: JSON is not canonical")
    if schema is not None and value.get("schema_version") != schema:
        raise ValueError(f"{path}: schema differs")
    return value


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def verify_artifact_records(value: Any, *, label: str) -> int:
    count = 0
    if isinstance(value, Mapping):
        if {"path", "sha256", "bytes"}.issubset(value):
            path = Path(str(value["path"]))
            if not path.is_file():
                raise ValueError(f"{label}: artifact is missing: {path}")
            if sha256_file(path) != value["sha256"]:
                raise ValueError(f"{label}: artifact hash differs: {path}")
            if path.stat().st_size != value["bytes"]:
                raise ValueError(f"{label}: artifact byte count differs: {path}")
            count += 1
        for key, item in value.items():
            count += verify_artifact_records(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            count += verify_artifact_records(item, label=f"{label}[{index}]")
    return count


def discover_scripted_receipts(root: Path) -> list[Path]:
    paths = sorted(root.resolve().rglob("receipts/*.json"))
    if not paths:
        paths = sorted(root.resolve().rglob("g3_scripted_check_receipt.json"))
    if not paths:
        raise ValueError(f"no scripted check receipts found under {root}")
    return paths


def compile_pilot_aggregate(
    *,
    pilot_plan_path: Path,
    confirmatory_aggregate_path: Path,
    pilot_path_scale_path: Path,
    scripted_receipts_root: Path,
    output_path: Path,
    verify_external: bool,
) -> dict[str, Any]:
    plan_path = pilot_plan_path.resolve()
    plan = load_canonical(plan_path)
    validate_plan_payload(plan)
    if (
        plan.get("fixture_id") != FIXTURE_ID
        or plan.get("qualification_scope") != PILOT_SCOPE
    ):
        raise ValueError("pilot plan identity differs")

    confirmatory_path = confirmatory_aggregate_path.resolve()
    confirmatory = load_canonical(
        confirmatory_path,
        schema=aggregate_receipt_schema_g3(FIXTURE_ID),
    )
    validate_g3_aggregate_receipt(confirmatory)
    if (
        confirmatory.get("fixture_id") != FIXTURE_ID
        or confirmatory.get("qualification_scope", CONFIRMATORY_SCOPE)
        != CONFIRMATORY_SCOPE
        or confirmatory.get("passed") is not True
        or confirmatory.get("status") != "passed"
    ):
        raise ValueError("confirmatory G3 aggregate is not passing")
    selected_scale = confirmatory.get("selected_scale")
    if not isinstance(selected_scale, (int, float)) or isinstance(
        selected_scale,
        bool,
    ):
        raise ValueError("confirmatory selected scale is missing")
    selected_scale = float(selected_scale)

    path_scale_path = pilot_path_scale_path.resolve()
    path_scale = load_canonical(
        path_scale_path,
        schema=path_scale_receipt_schema(FIXTURE_ID),
    )
    validate_path_scale_receipt(path_scale, plan=plan)
    if (
        path_scale.get("qualification_scope") != PILOT_SCOPE
        or path_scale.get("passed") is not True
        or float(path_scale.get("scale")) != selected_scale
    ):
        raise ValueError("pilot path-scale receipt does not pass selected scale")

    expected_keys = expected_scripted_check_keys(plan)
    observed: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    receipt_files: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    verified_artifact_count = 0
    scripted_schema = scripted_receipt_schema(FIXTURE_ID)
    selected_displacement = float(path_scale["displacement_m"])
    for path in discover_scripted_receipts(scripted_receipts_root):
        receipt = load_canonical(path, schema=scripted_schema)
        validate_scripted_check_receipt(receipt)
        if receipt.get("fixture_id") != FIXTURE_ID:
            raise ValueError(f"{path}: fixture differs")
        if float(receipt.get("scale")) != selected_scale:
            raise ValueError(f"{path}: scale differs")
        if (
            abs(float(receipt.get("displacement_m")) - selected_displacement)
            > 1e-12
        ):
            raise ValueError(f"{path}: displacement differs")
        key = _scripted_check_key(receipt)
        if key in observed:
            raise ValueError(f"duplicate scripted check receipt: {key}")
        observed[key] = receipt
        receipt_files[key] = file_record(path)
        if verify_external:
            verified_artifact_count += verify_artifact_records(
                receipt.get("evidence"),
                label=str(path),
            )

    missing = [key for key in expected_keys if key not in observed]
    unexpected = sorted(set(observed) - set(expected_keys))
    failed = [
        key for key in expected_keys if observed.get(key, {}).get("passed") is not True
    ]
    passed = not missing and not unexpected and not failed
    aggregate = {
        "schema_version": aggregate_receipt_schema_g3(FIXTURE_ID),
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "qualification_scope": PILOT_SCOPE,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "plan_receipt": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "candidate_scales_descending": list(
            plan["scale_selection"]["candidate_scales_descending"]
        ),
        "rejected_scales": list(confirmatory.get("rejected_scales") or []),
        "selected_scale": selected_scale,
        "selected_displacement_m": selected_displacement,
        "selected_scale_source": {
            "mode": "confirmatory_fixture_scale_reused_for_disjoint_pilot_resets",
            "confirmatory_g3_aggregate": file_record(confirmatory_path),
        },
        "selected_path_scale_receipt_sha256": sha256_file(path_scale_path),
        "pilot_path_scale_receipt": file_record(path_scale_path),
        "expected_scripted_check_count": len(expected_keys),
        "observed_scripted_check_count": len(observed),
        "missing_scripted_check_keys": [list(key) for key in missing],
        "unexpected_scripted_check_keys": [list(key) for key in unexpected],
        "scripted_passed_check_count": sum(
            observed.get(key, {}).get("passed") is True for key in expected_keys
        ),
        "scripted_failed_check_count": len(failed),
        "scripted_check_receipt_files": [
            {"key": list(key), **receipt_files[key]}
            for key in expected_keys
            if key in receipt_files
        ],
        "verified_external_artifact_count": verified_artifact_count,
        "scientific_failure_summary": {
            "path_scale": list(
                path_scale.get("scientific_failure_summary") or []
            ),
            "scripted": [
                {
                    "key": list(key),
                    "reasons": list(observed.get(key, {}).get("reasons") or []),
                }
                for key in failed
            ],
        },
        "release_boundary": (
            "A passing aggregate proves the confirmatory-selected C7 scale on the "
            "24 disjoint engineering-pilot resets. It does not select a new scale, "
            "authorize confirmatory inference, or replace G7-G8."
        ),
    }
    validate_g3_aggregate_receipt(aggregate, plan=plan)
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(aggregate))
        handle.flush()
        os.fsync(handle.fileno())
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--confirmatory-aggregate", type=Path, required=True)
    parser.add_argument("--pilot-path-scale", type=Path, required=True)
    parser.add_argument("--scripted-receipts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-external", action="store_true")
    args = parser.parse_args()
    aggregate = compile_pilot_aggregate(
        pilot_plan_path=args.pilot_plan,
        confirmatory_aggregate_path=args.confirmatory_aggregate,
        pilot_path_scale_path=args.pilot_path_scale,
        scripted_receipts_root=args.scripted_receipts_root,
        output_path=args.out,
        verify_external=args.verify_external,
    )
    print(
        json.dumps(
            {
                "path": str(args.out.resolve()),
                "sha256": sha256_file(args.out.resolve()),
                "passed": aggregate["passed"],
                "selected_scale": aggregate["selected_scale"],
                "observed_scripted_check_count": aggregate[
                    "observed_scripted_check_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
