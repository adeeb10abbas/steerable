"""Pure validation of the immutable R010 pre-action oracle-failure closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R010_RAW_RESULT_SHA256 = "4792f9c0f60a3af2c2549a16f89b45c4493e0ce0a303e588fb5be0d864ff1b10"
R010_RAW_PREFLIGHT_SHA256 = "c20070cc4276afee824bb54b311bba8b78794bda29ece9a51df2e745320ae0b4"
R010_RESULTS_SHA256 = "d4762cd8f4db539e760a79ce9e36f81d49455cb76836d8df5dadf902e6b78869"
R010_CLOSURE_COMMIT = "45a90fa93b1df2b3a07c0e974527a211814d6cb9"
R010_RECEIPT_SHA256 = "076fa2ae87ef549601126af1000dec53dcf8d32d6d3ec1301b79731fa0dfd0ad"
R010_MANIFEST_SHA256 = "569efa7250241d1bdcb31bcd0b85d99cb3db63953e9945122a2b87d594db589a"


def validate_r010_oracle_failure_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r010-state-repair-closure-v1",
        "amendment_id": "V3-E006-R010",
        "status": "r010_geometry_attachment_preflight_failed_candidates_not_evaluated",
        "passed": False,
        "accepted_candidate_rank": None,
        "accepted_state_hashes": None,
        "geometry_attachment_preflight_count": 1,
        "diagnostic_evaluation_count": 0,
        "candidate_pair_evaluation_count": 0,
        "mechanically_valid_fail_closed_execution": True,
        "relative_bound_attachment_validated": False,
        "relative_bound_controller_evaluated": False,
        "intended_r010_construction_scientifically_exhausted": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"R010 closure field differs: {key}")

    bindings = {
        "raw_result": R010_RAW_RESULT_SHA256,
        "raw_preflight": R010_RAW_PREFLIGHT_SHA256,
        "authoritative_target_validation_receipt": R010_RECEIPT_SHA256,
    }
    for key, digest in bindings.items():
        row = payload.get(key)
        if not isinstance(row, Mapping) or row.get("sha256") != digest:
            raise ValueError(f"R010 closure binding differs: {key}")

    finding = payload.get("geometry_oracle_finding")
    if not isinstance(finding, Mapping):
        raise ValueError("R010 geometry-oracle finding is absent")
    finding_expected = {
        "classification": "pre_action_geometry_oracle_synchronization_failure",
        "mechanically_valid_fail_closed_execution": True,
        "relative_bound_attachment_validated": False,
        "relative_bound_controller_evaluated": False,
        "intended_r010_construction_scientifically_exhausted": False,
        "behavioral_release_permitted": False,
    }
    for key, value in finding_expected.items():
        if finding.get(key) != value:
            raise ValueError(f"R010 geometry-oracle classification differs: {key}")
    synchronization = finding.get("one_shot_synchronization")
    if synchronization != {
        "call": "omni.physx.get_physx_interface().update_transformations(False, True, False, False)",
        "intervening_action_or_physics_steps": 0,
        "reset_steps": 75,
        "setting_after": False,
        "setting_before": False,
        "setting_path": "/physics/updateToUsd",
        "setting_unchanged": True,
    }:
        raise ValueError("R010 failed synchronization signature differs")
    if finding.get("quantitative_signature") != {
        "collision_aabb_error_m_max": 0.12691410402921288,
        "collision_aabb_error_m_min": 0.0009491238381734712,
        "failed_collision_prim_count": 5,
        "failed_owner_count": 3,
        "owner_orientation_error_deg_max": 0.49780630137334264,
        "owner_orientation_error_deg_min": 0.09479201431562724,
        "owner_position_error_m_max": 0.12693187594413757,
        "owner_position_error_m_min": 0.0017656832933425903,
    }:
        raise ValueError("R010 geometry-oracle mismatch signature differs")
    owner_rows = finding.get("owner_pose_rows")
    collision_rows = finding.get("collision_oracle_rows")
    if (
        not isinstance(owner_rows, list)
        or len(owner_rows) != 3
        or any(row.get("passed") is not False for row in owner_rows)
        or not isinstance(collision_rows, list)
        or len(collision_rows) != 5
        or any(row.get("passed") is not False for row in collision_rows)
    ):
        raise ValueError("R010 failed oracle row inventory differs")
