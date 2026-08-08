"""Analytic, zero-model-request gate for a V3-E004 layout candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .layout_contract import ASYMMETRY_LEVELS, canonical_json_bytes, load_candidate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_static_candidate(candidate_path: Path, candidate_sha256: str) -> dict[str, Any]:
    candidate = load_candidate(candidate_path, candidate_sha256)
    levels = []
    for s in ASYMMETRY_LEVELS:
        poses = candidate.layout(s)
        levels.append(
            {
                "symmetry_level_s": s,
                "active_inventory": sorted(poses),
                "inventory_transition": not math.isclose(s, 0.0, abs_tol=1e-12),
                "asymmetry_metric_A": candidate.asymmetry_A(poses),
                "analytic_residuals": candidate.residuals(poses),
            }
        )
    s0 = levels[0]
    positive = levels[1:]
    if s0["active_inventory"] != sorted(candidate.control_poses):
        raise RuntimeError("s0 inventory does not equal the hash-bound control")
    expected_positive = sorted(candidate.symmetric_poses)
    if any(row["active_inventory"] != expected_positive for row in positive):
        raise RuntimeError("positive-level companion inventory is inconsistent")
    full = levels[-1]
    if any(abs(value) > 1e-12 for value in full["analytic_residuals"].values()):
        raise RuntimeError("registered s=1 endpoint is not analytically symmetric")
    return {
        "schema_version": "vla-wam-shared-v3e004-static-layout-gate-v1",
        "status": "passed_model_blind_analytic_not_released_for_inference",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_sha256": candidate_sha256,
        "levels": levels,
        "s0_exact_control_inventory": True,
        "inventory_transition_disclosed": True,
        "H1_design_limitation": "The confirmatory s0-to-s1 contrast includes the registered same-asset companion activation.",
        "H3_primary_levels": [0.25, 0.5, 0.75, 1.0],
        "camera_gate_status": "pending_live_camera_geometry_or_instance_segmentation",
        "arm_reset_pose_status": "pending_live_settled_reset",
        "release_boundary": "Analytic pose checks do not release inference; every used level still requires a live camera, reset-pose, stability, and runtime gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite static gate: {args.output}")
    value = evaluate_static_candidate(args.candidate, args.candidate_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(value))
    print(json.dumps({"gate": str(args.output.resolve()), "sha256": _sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
