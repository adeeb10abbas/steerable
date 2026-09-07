#!/usr/bin/env python3
"""Bind passing C2 reference_binding G3 path-scale and scripted evidence."""

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
    scripted_receipt_path: Path | None,
    aggregate_receipt_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    if plan.get("fixture_id") != "reference_binding":
        raise ValueError("plan fixture differs")
    g2 = load_json(g2_path)
    if g2.get("passed") is not True:
        raise ValueError("G2 aggregate is not passing")
    path_payloads = [
        require_passing(path, schema="v4-reference-binding-g3-path-scale-receipt-v1")
        for path in path_scale_receipts
    ]
    scales = sorted({float(item["scale"]) for item in path_payloads})
    payload: dict[str, Any] = {
        "schema_version": "v4-reference-binding-g3-gate-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C2",
        "fixture_id": "reference_binding",
        "gate": "G3",
        "status": "passed",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "path_scale_receipts": [artifact(path) for path in path_scale_receipts],
        "geometry_plan": artifact(plan_path),
        "g2_aggregate": artifact(g2_path),
        "release_boundary": (
            "A passing gate receipt completes C2 G3 only. Verified common-prefix "
            "replay, G4-G6, G7 engineering pilots, G8 miniature rehearsal, and a "
            "released runtime lock remain required before confirmatory dispatch."
        ),
    }
    if scripted_receipt_path is not None:
        scripted = require_passing(
            scripted_receipt_path,
            schema="v4-reference-binding-g3-scripted-check-receipt-v1",
        )
        payload["scripted_receipt"] = artifact(scripted_receipt_path)
        payload["selected_scale"] = float(scripted["scale"])
    elif aggregate_receipt_path is not None:
        aggregate = require_passing(
            aggregate_receipt_path,
            schema="v4-reference-binding-g3-aggregate-receipt-v1",
        )
        payload["aggregate_receipt"] = artifact(aggregate_receipt_path)
        payload["selected_scale"] = float(aggregate["selected_scale"])
    else:
        raise ValueError("scripted or aggregate receipt required")
    payload["rejected_scales"] = [
        scale
        for scale in (2.0, 1.5, 1.0, 0.75, 0.5)
        if scale not in scales
    ]
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
        default=ROOT
        / "artifacts/online_correction_v4/setup/reference_binding_g3_plan.candidate.json",
    )
    parser.add_argument(
        "--g2",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/qualification/20260906_reference_binding_g2_aggregate_g2c2q20260906a.json",
    )
    parser.add_argument("--path-scale", type=Path, action="append", required=True)
    parser.add_argument("--scripted", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = compile_gate_receipt(
        plan_path=args.plan.resolve(),
        g2_path=args.g2.resolve(),
        path_scale_receipts=[path.resolve() for path in args.path_scale],
        scripted_receipt_path=args.scripted.resolve() if args.scripted else None,
        aggregate_receipt_path=args.aggregate.resolve() if args.aggregate else None,
        output_path=args.out.resolve(),
    )
    print(json.dumps({"status": receipt["status"], "selected_scale": receipt.get("selected_scale")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
