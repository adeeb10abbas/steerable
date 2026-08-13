"""Pure validation of the immutable V3-E006-R004 timeout closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R004_RAW_RESULT_SHA256 = "54c2335c5c4339037bd5f7e7e76ab15c5485191d82de253cc61d27dd66ddb81d"


def validate_r004_timeout_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r004-exhaustion-closure-v1",
        "amendment_id": "V3-E006-R004",
        "status": "r004_candidate_budget_exhausted_construction_time_limit_before_materialization",
        "passed": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "diagnostic_evaluation_count": 4,
        "diagnostics_all_passed": True,
        "candidate_pair_evaluation_count": 4,
        "stage_solve_count": 8,
        "state_candidate_count": 0,
        "accepted_candidate_rank": None,
        "first_passing_rule_obeyed": True,
        "behavioral_activation_released": False,
        "scientific_gate_thresholds_unchanged": True,
        "registered_diagnostic_budget": 4,
        "registered_candidate_budget": 4,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"R004 timeout closure field differs: {key}")
    signature = payload.get("termination_signature")
    if signature != {
        "all_eight_stages_identical": True,
        "common_step_counter": 450,
        "max_episode_length": 450,
        "phase": "registered_waypoint_07_of_08_correction_round_01",
        "phase_step_one_based": 15,
        "success": False,
        "terminated": False,
        "time_out": True,
        "truncated": True,
    }:
        raise ValueError("R004 timeout signature differs")
    raw = payload.get("raw_evidence")
    expected_names = {"harness", "launch", "result", "runtime_log", "target_validation_receipt"}
    if not isinstance(raw, Mapping) or set(raw) != expected_names:
        raise ValueError("R004 raw-evidence inventory differs")
    if raw["result"].get("sha256") != R004_RAW_RESULT_SHA256:
        raise ValueError("R004 raw-result digest differs")
    for name, row in raw.items():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError(f"R004 raw-evidence binding differs: {name}")
        if not isinstance(row.get("bytes"), int) or isinstance(row.get("bytes"), bool):
            raise ValueError(f"R004 raw-evidence bytes differ: {name}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"R004 raw-evidence digest differs: {name}")
