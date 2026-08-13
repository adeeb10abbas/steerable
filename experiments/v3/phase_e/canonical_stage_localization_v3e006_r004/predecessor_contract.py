"""Pure validation of the immutable V3-E006-R003 diagnostic closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R003_RAW_RESULT_SHA256 = "0bac94d3d1e5b93f3eb00f94f1c4a6cc989cbe54d01f2b9247ed6f5ecf5a9392"


def validate_r003_diagnostic_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r003-diagnostic-closure-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E006-R003",
        "status": "r003_known_reachable_diagnostic_failed_candidates_not_evaluated",
        "passed": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "diagnostic_evaluation_count": 1,
        "candidate_pair_evaluation_count": 0,
        "state_candidate_count": 0,
        "accepted_candidate_rank": None,
        "first_passing_rule_obeyed": True,
        "behavioral_activation_released": False,
        "scientific_gate_thresholds_unchanged": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"R003 diagnostic closure field differs: {key}")
    failure = payload.get("diagnostic_failure")
    if not isinstance(failure, Mapping) or not (
        failure.get("diagnostic_index_one_based") == 1
        and failure.get("stage") == "canonical_grasp"
        and failure.get("source_side") == "left"
        and failure.get("position_tolerance_m_inclusive") == 0.001
        and failure.get("orientation_tolerance_deg_inclusive") == 1.0
        and failure.get("final_position_error_m") == 0.0012277967696529603
        and failure.get("final_orientation_geodesic_error_deg") == 0.2210353088374109
        and failure.get("fresh_reset_passed") is True
        and failure.get("camera_evidence_passed") is True
        and failure.get("all_base_link_to_eef_frame_identity_checks_passed") is True
    ):
        raise ValueError("R003 diagnostic failure summary differs")
    raw = payload.get("raw_evidence")
    expected_names = {"harness", "launch", "result", "runtime_log", "target_validation_receipt", "video"}
    if not isinstance(raw, Mapping) or set(raw) != expected_names:
        raise ValueError("R003 raw-evidence inventory differs")
    if raw["result"].get("sha256") != R003_RAW_RESULT_SHA256:
        raise ValueError("R003 raw-result digest differs")
    for name, row in raw.items():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError(f"R003 raw-evidence binding differs: {name}")
        if not isinstance(row.get("bytes"), int) or isinstance(row.get("bytes"), bool):
            raise ValueError(f"R003 raw-evidence bytes differ: {name}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"R003 raw-evidence digest differs: {name}")
