#!/usr/bin/env python3
"""Compile a passing model-blind G3 gate receipt from path-scale evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.fixture_qualification import (  # noqa: E402
    qualification_profile,
)
from experiments.online_correction_v4.model_blind_g3 import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        rel = str(resolved.relative_to(ROOT))
    except ValueError:
        rel = str(resolved)
    return {
        "path": rel,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_passing(path: Path, *, schema: str, fixture_id: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != schema:
        raise ValueError(f"{path}: schema differs from {schema}")
    if payload.get("fixture_id") != fixture_id:
        raise ValueError(f"{path}: fixture mismatch")
    if payload.get("passed") is not True or payload.get("status") != "passed":
        raise ValueError(f"{path}: receipt is not passing")
    return payload


def compile_gate_receipt(
    *,
    fixture_id: str,
    plan_path: Path,
    g2_path: Path,
    path_scale_receipt_path: Path,
    output_path: Path,
    attempt_id: str,
) -> dict[str, Any]:
    profile = qualification_profile(fixture_id)
    if profile.g3_basis != "path_scale":
        raise ValueError(f"{fixture_id} does not use path-scale G3 basis")
    plan = load_json(plan_path)
    if plan.get("fixture_id") != fixture_id:
        raise ValueError("plan fixture differs")
    g2 = load_json(g2_path)
    if g2.get("fixture_id") != fixture_id or g2.get("passed") is not True:
        raise ValueError("G2 aggregate is not passing for fixture")
    path_scale = require_passing(
        path_scale_receipt_path,
        schema=profile.g3_path_scale_schema,
        fixture_id=fixture_id,
    )
    if path_scale.get("observed_seed_count") != path_scale.get("expected_seed_count"):
        raise ValueError("path-scale receipt seed coverage is incomplete")
    if path_scale.get("information_gate_failed_seeds"):
        raise ValueError("path-scale receipt reports information-gate failures")
    payload: dict[str, Any] = {
        "schema_version": f"v4-{fixture_id.replace('_', '-')}-g3-gate-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": profile.family_id,
        "fixture_id": fixture_id,
        "gate": "G3",
        "attempt_id": attempt_id,
        "status": "passed",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "qualification_scope": "model_blind_motion_and_feasibility_no_policy",
        "selected_scale": path_scale.get("scale"),
        "selected_displacement_m": path_scale.get("displacement_m"),
        "observed_seed_count": path_scale.get("observed_seed_count"),
        "expected_seed_count": path_scale.get("expected_seed_count"),
        "information_gate_failed_seeds": path_scale.get(
            "information_gate_failed_seeds", []
        ),
        "plan_receipt": artifact(plan_path),
        "g2_aggregate": artifact(g2_path),
        "path_scale_receipt": artifact(path_scale_receipt_path),
        "scripted_checks": {
            "status": "not_required_for_model_blind_path_scale_gate",
            "reason": (
                "Containment G3 closes on complete path-scale coverage with "
                "information-gate pass on both translation-sign halves."
            ),
        },
        "release_boundary": (
            f"A pass completes {profile.family_id} model-blind G3 only. "
            "G4-G8 and a released runtime lock remain required before confirmatory "
            "policy episodes."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    with output_path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--g2-aggregate", type=Path, required=True)
    parser.add_argument("--path-scale-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite gate receipt: {args.out}")
    report = compile_gate_receipt(
        fixture_id=args.fixture_id,
        plan_path=args.plan,
        g2_path=args.g2_aggregate,
        path_scale_receipt_path=args.path_scale_receipt,
        output_path=args.out,
        attempt_id=args.attempt_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
