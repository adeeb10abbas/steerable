#!/usr/bin/env python3
"""Freeze the disclosed V4 natural-grasp trigger observation wiring repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/natural_grasp_trigger_wiring_repair_amendment.candidate.json"
)
BLOCKING_RECEIPT = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_object_pair_natural_grasp_live_positive_control_g3ngp20260908m.json"
)
RECLASSIFICATION = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_c7_natural_grasp_trigger_wiring_reclassification.json"
)
DROID_ROBOLAB = ROOT / "experiments/online_correction_v4/droid_robolab.py"
FIXTURE_VERSION = "natural_grasp_trigger_wiring_repair_v1"


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

    payload = {
        "schema_version": "v4-natural-grasp-trigger-wiring-repair-amendment-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": "object_pair",
        "fixture_version": FIXTURE_VERSION,
        "cohort": "confirmatory_natural_grasp_trigger_wiring_repair_v1",
        "status": "model_blind_candidate_not_released_for_inference",
        "amendment_status": "frozen_for_model_blind_requalification",
        "post_result_amendment": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "triggering_evidence": {
            "blocking_positive_control": artifact(BLOCKING_RECEIPT),
            "attempt_id": "g3ngp20260908m",
            "scripted_trajectory_passed": True,
            "trigger_eligible": False,
            "c7_reclassification": artifact(RECLASSIFICATION),
            "episode_count_reclassified": 279,
        },
        "defect": {
            "component": "experiments/online_correction_v4/droid_robolab.py::object_kinematic_state",
            "summary": (
                "NaturalGraspDetector received robot-base pose as gripper position and "
                "object_grabbed() as the sole contact source, so trigger_eligible never "
                "fired despite successful scripted grasps in live Isaac."
            ),
            "observed_symptoms": [
                "contact remained false across all detector samples",
                "lift_m stayed near 0 versus registered min_lift_m=0.04",
                "relative_drift_m ~0.52 m because object pose was compared to robot base",
            ],
        },
        "repair": {
            "component": "experiments/online_correction_v4/droid_robolab.py",
            "changed": [
                "gripper_x/y/z now sourced from scene eef_frame target_pos_w",
                "contact uses object_grabbed, then gripper-target contact forces, then closed-gripper proximity",
                "_anchor_initial_supported_z() anchors initial_supported_z at reset",
            ],
            "unchanged": [
                "NaturalGraspDetector thresholds",
                "natural_grasp_min_lift_m",
                "natural_grasp_dwell_s",
                "kinematic_grasp_relative_drift_max_m",
                "trigger_deadline_s",
                "prompts",
                "scoring",
                "timing",
                "scale_ladder",
            ],
            "policy_outcome_used": False,
            "observation_wiring_constants": {
                "GRIPPER_CONTACT_FORCE_THRESHOLD_N": 0.05,
                "GRIPPER_OBJECT_PROXIMITY_M": 0.12,
            },
            "repaired_module": artifact(DROID_ROBOLAB),
        },
        "required_requalification": {
            "live_positive_control_required": True,
            "live_negative_control_required": True,
            "fresh_g4_policy_session_required": True,
            "fresh_g5_trigger_branch_required": True,
            "affected_confirmatory_rows": 768,
            "affected_families": ["C7"],
            "blocked_until_repair_families": ["C2", "C8"],
            "reuse_prior_c7_accepted_ledger_rows": False,
        },
        "authorization_boundary": {
            "authorizes_repaired_model_blind_trigger_validation": True,
            "authorizes_policy_inference": False,
            "authorizes_behavioral_episode": False,
            "authorizes_c7_confirmatory_before_live_controls": False,
            "shared_code_affects_all_droid_families": True,
        },
        "disclosure": (
            "Disclosed V4-only observation wiring repair after conclusive live Isaac "
            "positive control showed scripted grasp success with trigger_eligible=false. "
            "The failed receipt, raw PVC traces, and 279 C7 accepted ledger rows are "
            "preserved as infrastructure-invalid provenance; confirmatory C7 must rerun "
            "on the repaired trigger path after live positive and negative controls pass."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "path": str(args.out.resolve()),
                "sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
                "fixture_version": FIXTURE_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
