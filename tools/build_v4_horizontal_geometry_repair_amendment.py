#!/usr/bin/env python3
"""Freeze the disclosed V4 horizontal geometry-repair amendment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.horizontal_geometry_repair import (  # noqa: E402
    COHORT,
    FIXTURE_VERSION,
    MINIMUM_DISPLACEMENT_M,
    MINIMUM_SCALE,
    REPAIR_INCREMENT_M,
    SUPPORT_EDGE_GUARD_M,
    canonical_aabb_freshness_audit,
    minimum_cube_repair_offset_m,
)

SPEC = importlib.util.spec_from_file_location(
    "build_v4_horizontal_reset_registry",
    ROOT / "tools/build_v4_horizontal_reset_registry.py",
)
reset_builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reset_builder)

DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_amendment.candidate.json"
)
FORENSIC_RECEIPT = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260907_horizontal_g3_collision_forensic_g3p20260905h.json"
)
WITNESS_RECEIPT = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_horizontal_g3_contact_geometry_witness_g3p20260905h.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    body = resolved.read_bytes()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registry = reset_builder.build_registry(
        campaign_path=reset_builder.DEFAULT_CAMPAIGN,
        queue_path=reset_builder.DEFAULT_QUEUE,
        source_report_path=reset_builder.DEFAULT_SOURCE,
    )
    offset_m, clearance_audit = minimum_cube_repair_offset_m(
        base_positions_robot_base_m=registry["source_identity"]["base_positions_robot_base_m"],
        resets_by_env_seed=registry["resets_by_env_seed"],
    )
    payload = {
        "schema_version": "v4-horizontal-geometry-repair-amendment-v2",
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "fixture_version": FIXTURE_VERSION,
        "cohort": COHORT,
        "status": "model_blind_candidate_not_released_for_inference",
        "amendment_status": "frozen_for_model_blind_requalification",
        "post_result_amendment": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "triggering_evidence": {
            "path_scale_receipt": artifact(
                ROOT
                / "artifacts/online_correction_v4/qualification/20260905_horizontal_g3_path_scale_0p5_g3p20260905h.json"
            ),
            "forensic_receipt": artifact(FORENSIC_RECEIPT),
            "geometry_witness_receipt": artifact(WITNESS_RECEIPT),
            "attempt_id": "g3p20260905h",
            "failed_scale": MINIMUM_SCALE,
            "original_reset_registry": artifact(
                ROOT
                / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
            ),
        },
        "repair": {
            "selection_rule": (
                "Move only rubiks_cube along robot-base -X in 1 cm increments before "
                "deterministic common XY jitter. Select the smallest increment such that "
                "PVC witness-delayed first rubiks_cube__bowl contact on front/behind -X "
                "paths reaches the 0.5-scale displacement plus the existing 5 mm guard, "
                "verified across all 128 jittered resets for front/behind only."
            ),
            "cube_robot_base_x_offset_m": offset_m,
            "increment_m": REPAIR_INCREMENT_M,
            "support_edge_guard_m": SUPPORT_EDGE_GUARD_M,
            "minimum_scale": MINIMUM_SCALE,
            "minimum_displacement_m": MINIMUM_DISPLACEMENT_M,
            "application_order": "cube_offset_before_common_xy_jitter",
            "unchanged": [
                "bowl",
                "banana",
                "scene_asset",
                "physics",
                "task_frame",
                "prompts",
                "scoring",
                "timing",
                "scale_ladder",
                "thresholds",
            ],
            "policy_outcome_used": False,
            "clearance_audit": clearance_audit,
        },
        "aabb_freshness_audit": canonical_aabb_freshness_audit(),
        "supersedes_fixture_version": "horizontal_geometry_repair_v1",
        "required_requalification": {
            "fresh_g2_required": True,
            "fresh_g3_required": True,
            "registered_seed_count": 128,
            "affected_confirmatory_rows": 9728,
            "affected_families": ["C1", "C3", "C4"],
            "reuse_prior_horizontal_g2_or_g3_receipts": False,
        },
        "authorization_boundary": {
            "authorizes_repaired_model_blind_g2_g3": True,
            "authorizes_original_layout_c1_c3_c4_inference": False,
            "authorizes_policy_inference": False,
            "authorizes_behavioral_episode": False,
            "c7_execution_inspected_only": True,
        },
        "disclosure": (
            "Disclosed V4-only geometry repair after conclusive model-blind G3 "
            "collision evidence on the original layout. v1 used a defective 3D AABB "
            "clearance model; v2 binds to PVC contact timing and live world AABB. "
            "Original failed receipts and raw PVC traces are preserved."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "path": str(args.out.resolve()),
                "sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
                "cube_robot_base_x_offset_m": offset_m,
                "fixture_version": FIXTURE_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
