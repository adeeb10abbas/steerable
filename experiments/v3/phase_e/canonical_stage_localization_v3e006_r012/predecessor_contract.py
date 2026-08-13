"""Pure validation of the immutable R011 final scene-sync closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R011_RAW_RESULT_SHA256 = "160a253999d0b3ae55709bdebca742172fbd87ece22d5201062bffd4dd9af1a0"
R011_RAW_PREFLIGHT_SHA256 = "6ba9cae267f852cafa6c20e7f9e3e5aa4bfcb3d3a0dc24f6053e435ca37e0c23"
R011_RESULTS_SHA256 = "f6e869939d5003c175abb21c0544d2d02313fe00e7a0dd7831d60e0c7f192054"
R011_CLOSURE_COMMIT = "3c3650199a38cf850a7d0df5e371df9c28b39f6a"
R011_RECEIPT_SHA256 = "3d28c7f2cb76fbfeefd7f2020abb7a6ed8eef4d87c72884b4b63ad52821e38a7"
R011_MANIFEST_SHA256 = "d6f41c624edffa54f7b82743184a3b0d5145b6c8ee9952838b3bac66cad184d9"


def validate_r011_scene_sync_failure_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r011-state-repair-closure-v1",
        "amendment_id": "V3-E006-R011",
        "status": "r011_geometry_attachment_preflight_failed_candidates_not_evaluated",
        "passed": False,
        "accepted_candidate_rank": None,
        "accepted_state_hashes": None,
        "geometry_attachment_preflight_count": 1,
        "diagnostic_evaluation_count": 0,
        "candidate_pair_evaluation_count": 0,
        "mechanically_valid_fail_closed_execution": True,
        "relative_bound_attachment_validated": False,
        "relative_bound_controller_evaluated": False,
        "intended_r011_construction_scientifically_exhausted": False,
        "final_state_construction_blocker": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"R011 closure field differs: {key}")
    for key, digest in {
        "raw_result": R011_RAW_RESULT_SHA256,
        "raw_preflight": R011_RAW_PREFLIGHT_SHA256,
        "authoritative_target_validation_receipt": R011_RECEIPT_SHA256,
    }.items():
        row = payload.get(key)
        if not isinstance(row, Mapping) or row.get("sha256") != digest:
            raise ValueError(f"R011 closure binding differs: {key}")
    finding = payload.get("geometry_oracle_finding")
    if not isinstance(finding, Mapping):
        raise ValueError("R011 scene-sync finding absent")
    for key, value in {
        "classification": "final_scene_specific_usd_tensor_oracle_synchronization_failure",
        "mechanically_valid_fail_closed_execution": True,
        "relative_bound_attachment_validated": False,
        "relative_bound_controller_evaluated": False,
        "intended_r011_construction_scientifically_exhausted": False,
        "behavioral_release_permitted": False,
    }.items():
        if finding.get(key) != value:
            raise ValueError(f"R011 scene-sync classification differs: {key}")
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
        raise ValueError("R011 mismatch signature differs")
