#!/usr/bin/env python3
"""Compare model-blind G2/G3 seed receipts across GPU products under frozen tolerances."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.model_blind_g2 import POSITION_TOLERANCE_M  # noqa: E402

SCHEMA = "v4-model-blind-gpu-determinism-comparison-v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: receipt must be an object")
    return payload


def _gpu_name(receipt: Mapping[str, Any]) -> str:
    runtime = receipt.get("runtime_identity") or {}
    gpu = runtime.get("gpu") or {}
    name = gpu.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("receipt lacks runtime_identity.gpu.name")
    return name


def _position_errors(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    attestation = receipt.get("reset_attestation") or {}
    errors = attestation.get("position_errors_m")
    if not isinstance(errors, Mapping):
        raise ValueError("receipt lacks reset_attestation.position_errors_m")
    return errors


def _max_position_error(errors: Mapping[str, Any]) -> float:
    values = [float(v) for v in errors.values() if isinstance(v, (int, float))]
    if not values:
        raise ValueError("position_errors_m is empty")
    return max(values)


def compare_receipts(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tolerance_m: float = POSITION_TOLERANCE_M,
) -> dict[str, Any]:
    baseline_seed = baseline.get("environment_seed")
    candidate_seed = candidate.get("environment_seed")
    if baseline_seed != candidate_seed:
        raise ValueError("environment_seed differs between baseline and candidate")
    baseline_errors = _position_errors(baseline)
    candidate_errors = _position_errors(candidate)
    if set(baseline_errors) != set(candidate_errors):
        raise ValueError("position error keys differ between receipts")
    per_object: dict[str, float] = {}
    for key in baseline_errors:
        delta = abs(float(baseline_errors[key]) - float(candidate_errors[key]))
        per_object[key] = delta
    max_delta = max(per_object.values())
    passed = max_delta <= tolerance_m and math.isfinite(max_delta)
    return {
        "schema_version": SCHEMA,
        "environment_seed": baseline_seed,
        "baseline_gpu_name": _gpu_name(baseline),
        "candidate_gpu_name": _gpu_name(candidate),
        "registry_position_tolerance_m": tolerance_m,
        "max_position_error_delta_m": max_delta,
        "per_object_position_error_delta_m": per_object,
        "passed_within_tolerance": passed,
        "baseline_receipt_sha256": baseline.get("receipt_sha256"),
        "candidate_receipt_sha256": candidate.get("receipt_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_receipts(
        baseline=_load(args.baseline),
        candidate=_load(args.candidate),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
