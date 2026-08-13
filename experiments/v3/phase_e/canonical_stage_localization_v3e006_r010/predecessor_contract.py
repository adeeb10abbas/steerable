"""Pure validation of immutable R009 attachment-invalid exhaustion closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R009_RAW_RESULT_SHA256 = "0753b4fe44ed479d4575804f3b5f38e3244a1cea25de3f1a8564dca4cd26ab3c"
R009_RESULTS_SHA256 = "10dcccacce21f0412e37a7f33fbab0357cfae5cdd161705128eb15d5652da852"
R009_CLOSURE_COMMIT = "9000d2897e634eee9469d02c9449baf85fe15729"
R009_RECEIPT_SHA256 = "5d46ced5926652ac21943ef70be26d99940b3e1906cc5f4ece565a48d17c94ef"
R009_MANIFEST_SHA256 = "601dba6a367117a7aba8f8df838f5c2d54e73a19bd09d61156ca28110f89eab4"


def validate_r009_attachment_invalid_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r009-state-repair-closure-v1",
        "amendment_id": "V3-E006-R009",
        "status": "r009_candidate_budget_exhausted_no_valid_state_pair",
        "passed": False,
        "accepted_candidate_rank": None,
        "accepted_state_hashes": None,
        "registered_diagnostic_budget": 4,
        "diagnostic_evaluation_count": 4,
        "diagnostics_all_passed": True,
        "registered_candidate_budget": 4,
        "candidate_pair_evaluation_count": 4,
        "first_passing_rule_obeyed": True,
        "mechanically_valid_frozen_execution": True,
        "intended_collision_pinch_semantics_attachment_valid": False,
        "intended_collision_pinch_algorithm_scientifically_exhausted": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"R009 closure field differs: {key}")
    attempts = payload.get("candidate_attempts")
    if not isinstance(attempts, list) or [row.get("candidate_rank") for row in attempts] != [1, 2, 3, 4]:
        raise ValueError("R009 candidate order differs")
    for attempt in attempts:
        if attempt.get("passed") is not False:
            raise ValueError("R009 exhausted rank differs")
        stages = attempt.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != {"canonical_grasp", "canonical_carry"}:
            raise ValueError("R009 stage inventory differs")
        for state in stages.values():
            if not isinstance(state, Mapping) or state.get("passed") is not False:
                raise ValueError("R009 rejected stage differs")
            if state.get("physics_gate", {}).get("passed") is not False:
                raise ValueError("R009 physics outcome differs")
            if state.get("ood_gate", {}).get("passed") is not True:
                raise ValueError("R009 OOD outcome differs")
            if state.get("camera_gate_passed") is not True:
                raise ValueError("R009 camera outcome differs")
            if state.get("companion_gate", {}).get("passed") is not True:
                raise ValueError("R009 companion outcome differs")
            if state.get("frame_identity_passed") is not True:
                raise ValueError("R009 frame outcome differs")
    if payload.get("stage_gate_pass_counts") != {
        "camera_gate_pass_count": 8,
        "companion_gate_pass_count": 8,
        "evaluated_stage_count": 8,
        "frame_identity_pass_count": 8,
        "full_state_pass_count": 0,
        "ood_gate_pass_count": 8,
        "physics_check_pass_counts": {
            "arm_joint_speed": 0,
            "cube_angular_speed": 0,
            "cube_gripper_relative_drift": 8,
            "cube_linear_speed": 6,
            "cube_midline": 8,
            "intended_cube_gripper_contact_force": 6,
            "no_unintended_contacts": 0,
            "normal_gripper_contact": 6,
        },
        "physics_gate_pass_count": 0,
    }:
        raise ValueError("R009 common stage-gate counts differ")
    raw = payload.get("raw_result")
    if not isinstance(raw, Mapping) or raw.get("sha256") != R009_RAW_RESULT_SHA256:
        raise ValueError("R009 raw-result binding differs")
    if payload.get("authoritative_target_validation_receipt", {}).get("sha256") != R009_RECEIPT_SHA256:
        raise ValueError("R009 target receipt differs")
    finding = payload.get("geometry_attachment_finding")
    if not isinstance(finding, Mapping) or finding.get("classification") != "attachment_invalid_intended_collision_pinch_semantics":
        raise ValueError("R009 attachment classification differs")
    signature = finding.get("quantitative_signature")
    if not isinstance(signature, Mapping) or signature != {
        "all_stages_table_loaded": True,
        "cube_table_contact_force_n_max": 9.805905349787599,
        "cube_table_contact_force_n_min": 5.569357287640638,
        "evaluated_stage_count": 8,
        "finger_body_origin_separation_m_max": 0.08178667449434165,
        "finger_body_origin_separation_m_min": 0.07925181825673311,
        "grabbed_all_final_steps_stage_count": 6,
        "reconstructed_pad_center_separation_m_max": 0.34517768097091517,
        "reconstructed_pad_center_separation_m_min": 0.34249134116656277,
    }:
        raise ValueError("R009 attachment signature differs")
