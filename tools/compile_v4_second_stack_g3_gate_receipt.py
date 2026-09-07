#!/usr/bin/env python3
"""Bind passing C8 second-stack G3 path-scale and scripted evidence."""

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

from tools.build_v4_second_stack_g3_plan import canonical_json_bytes, sha256_file  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_passing(path: Path, *, schema: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != schema:
        raise ValueError(f"{path}: schema differs from {schema}")
    if payload.get("passed") is not True or payload.get("status") != "passed":
        raise ValueError(f"{path}: receipt is not passing")
    return payload


def compile_gate_receipt(
    *,
    plan_path: Path,
    g2_path: Path,
    path_scale_receipts: list[Path],
    scripted_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    if plan.get("fixture_id") != "second_stack":
        raise ValueError("plan fixture differs")
    g2 = load_json(g2_path)
    if g2.get("passed") is not True:
        raise ValueError("G2 aggregate is not passing")
    path_payloads = [
        require_passing(path, schema="v4-second-stack-g3-path-aggregate-v1")
        for path in path_scale_receipts
    ]
    scripted = require_passing(
        scripted_receipt_path,
        schema="v4-second-stack-g3-scripted-aggregate-v1",
    )
    selected_scale = scripted.get("selected_scale")
    if not isinstance(selected_scale, (int, float)):
        raise ValueError("scripted receipt lacks selected_scale")
    scales = sorted({float(item["displacement_m"]) for item in path_payloads})
    rejected_scales = [2.0, 1.5, 1.0, 0.75]
    rejected = [scale for scale in rejected_scales if scale not in scales]
    payload = {
        "schema_version": "v4-second-stack-g3-gate-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C8",
        "fixture_id": "second_stack",
        "gate": "G3",
        "status": "passed",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "selected_scale": float(selected_scale),
        "rejected_scales": rejected,
        "path_scale_receipts": [artifact(path) for path in path_scale_receipts],
        "scripted_receipt": artifact(scripted_receipt_path),
        "geometry_plan": artifact(plan_path),
        "g2_aggregate": artifact(g2_path),
        "release_boundary": (
            "A passing gate receipt completes C8 G3 only. G4-G6 are separately "
            "receipted; G7 engineering pilots, G8 miniature rehearsal, and a "
            "released runtime lock remain required before confirmatory dispatch."
        ),
    }
    body = canonical_json_bytes(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/setup/second_stack_g3_plan.candidate.json",
    )
    parser.add_argument(
        "--g2",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/qualification/20260906_second_stack_g2_aggregate_g2c8q20260906e.json",
    )
    parser.add_argument(
        "--path-scale",
        type=Path,
        action="append",
        required=True,
        help="Passing path-scale aggregate receipt (repeat for each accepted scale).",
    )
    parser.add_argument(
        "--scripted",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/qualification/20260906_second_stack_g3_scripted_scale_0p5_full_g3c8s20260906w.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = compile_gate_receipt(
        plan_path=args.plan.resolve(),
        g2_path=args.g2.resolve(),
        path_scale_receipts=[path.resolve() for path in args.path_scale],
        scripted_receipt_path=args.scripted.resolve(),
        output_path=args.out.resolve(),
    )
    print(json.dumps({"status": receipt["status"], "selected_scale": receipt["selected_scale"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
