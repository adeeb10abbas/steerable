from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tools.validate_v3e006_r008_closure import gate_summary


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008/results/results.json"
)


def test_compact_gate_summary_is_exact_and_mutations_differ() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    stage = value["candidate_attempts"][0]["stages"]["canonical_grasp"]
    synthetic = {
        "passed": stage["passed"],
        "normalized_state_sha256": stage["normalized_state_sha256"],
        "physics_gate": deepcopy(stage["physics_gate"]),
        "ood_gate": deepcopy(stage["ood_gate"]),
        "camera_evidence": {"passed": stage["camera_gate_passed"]},
        "companion_pose_gate": deepcopy(stage["companion_gate"]),
        "base_link_to_eef_frame_identity": {
            "passed": stage["frame_identity_passed"]
        },
    }
    assert gate_summary(synthetic) == stage
    for key in ("physics_gate", "ood_gate", "normalized_state_sha256"):
        bad = deepcopy(synthetic)
        if key == "normalized_state_sha256":
            bad[key] = "0" * 64
        else:
            bad[key]["passed"] = not bad[key]["passed"]
        assert gate_summary(bad) != stage


def test_closure_binds_authoritative_and_failed_receipts() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert value["raw_result"] == {
        "bytes": 64704588,
        "path": "/data/users/ali/vla_wam/raw/v3e006_r008/state_repair/ae06a1f-a40r06-attempt01/raw/state_repair_result.json",
        "sha256": "58f8e54994348634d8793f0448c49144b35ab3b601065ab4be298cbd9ebc6548",
    }
    assert (
        value["authoritative_target_validation_receipt"]["sha256"]
        == "afb5228e5abb82c1fc87581d500a72adfa5d44a36f3aa04082dabe518085d562"
    )
    assert value["failed_zero_byte_frozen_validation_receipt"]["bytes"] == 0
    assert (
        value["postexecution_validator_amendment"]["sha256"]
        == "980b5540947a747f231213a033b6fb2de1a39e1aaa421b0052a8dd891776bbb4"
    )


def test_common_stage_pass_counts_are_frozen() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert value["stage_gate_pass_counts"] == {
        "camera_gate_pass_count": 8,
        "companion_gate_pass_count": 2,
        "evaluated_stage_count": 8,
        "frame_identity_pass_count": 8,
        "full_state_pass_count": 0,
        "ood_gate_pass_count": 6,
        "physics_check_pass_counts": {
            "arm_joint_speed": 2,
            "cube_angular_speed": 2,
            "cube_gripper_relative_drift": 8,
            "cube_linear_speed": 6,
            "cube_midline": 0,
            "intended_cube_gripper_contact_force": 0,
            "no_unintended_contacts": 0,
            "normal_gripper_contact": 0,
        },
        "physics_gate_pass_count": 0,
    }
