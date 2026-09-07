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
    minimum_cube_repair_offset_m,
    root_pose_aabb_center_mismatch_audit,
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
    forensic = json.loads(FORENSIC_RECEIPT.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "v4-horizontal-geometry-repair-amendment-v1",
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
                "deterministic common XY jitter until the minimum 0.5-scale conservative "
                "bowl swept-AABB separation meets the existing 5 mm support-edge guard."
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
        "aabb_freshness_audit": forensic["aabb_freshness_audit"],
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
            "collision evidence on the original layout. Original failed receipts "
            "and raw PVC traces are preserved. V2/V3 and unrelated families remain "
            "unchanged. C7 confirmatory execution counts were inspected for PVC "
            "forensics only; no horizontal policy outcomes were used."
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
