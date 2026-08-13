"""Pure validation of the immutable V3-E006-R008 state-search closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R008_RAW_RESULT_SHA256 = "58f8e54994348634d8793f0448c49144b35ab3b601065ab4be298cbd9ebc6548"
R008_RESULTS_SHA256 = "5f86b77b721806b0c19416b622e9a6aab68b13de6d596a6dadec89ed41a081d5"
R008_CLOSURE_COMMIT = "a13d5b7f8a4be6374a22483a436b9a41aadd1c9f"
R008_RECEIPT_SHA256 = "afb5228e5abb82c1fc87581d500a72adfa5d44a36f3aa04082dabe518085d562"
R008_AMENDMENT_SHA256 = "980b5540947a747f231213a033b6fb2de1a39e1aaa421b0052a8dd891776bbb4"


def validate_r008_exhaustion_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r008-state-repair-closure-v1",
        "amendment_id": "V3-E006-R008",
        "status": "r008_candidate_budget_exhausted_no_valid_state_pair",
        "passed": False,
        "accepted_candidate_rank": None,
        "accepted_state_hashes": None,
        "registered_diagnostic_budget": 4,
        "diagnostic_evaluation_count": 4,
        "diagnostics_all_passed": True,
        "registered_candidate_budget": 4,
        "candidate_pair_evaluation_count": 4,
        "first_passing_rule_obeyed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"R008 exhaustion closure field differs: {key}")
    attempts = payload.get("candidate_attempts")
    if not isinstance(attempts, list) or [
        row.get("candidate_rank") for row in attempts
    ] != [1, 2, 3, 4]:
        raise ValueError("R008 candidate order differs")
    for attempt in attempts:
        if attempt.get("passed") is not False:
            raise ValueError("R008 exhausted rank outcome differs")
        stages = attempt.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != {
            "canonical_grasp",
            "canonical_carry",
        }:
            raise ValueError("R008 stage inventory differs")
        for state in stages.values():
            if not isinstance(state, Mapping) or state.get("passed") is not False:
                raise ValueError("R008 rejected stage differs")
            if state.get("physics_gate", {}).get("passed") is not False:
                raise ValueError("R008 rejected physics gate differs")
            if state.get("camera_gate_passed") is not True:
                raise ValueError("R008 camera outcome differs")
            if state.get("frame_identity_passed") is not True:
                raise ValueError("R008 frame outcome differs")
    counts = payload.get("stage_gate_pass_counts")
    if counts != {
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
    }:
        raise ValueError("R008 common stage-gate counts differ")
    raw = payload.get("raw_result")
    if not isinstance(raw, Mapping) or raw.get("sha256") != R008_RAW_RESULT_SHA256:
        raise ValueError("R008 raw-result binding differs")
    if payload.get("authoritative_target_validation_receipt", {}).get(
        "sha256"
    ) != R008_RECEIPT_SHA256:
        raise ValueError("R008 authoritative validator receipt differs")
    if payload.get("postexecution_validator_amendment", {}).get(
        "sha256"
    ) != R008_AMENDMENT_SHA256:
        raise ValueError("R008 validator amendment differs")
