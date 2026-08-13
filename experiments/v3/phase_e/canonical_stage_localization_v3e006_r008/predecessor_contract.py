"""Pure validation of the immutable V3-E006-R007 state-search closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R007_RAW_RESULT_SHA256 = "f20314dcd32d9d6503dd5c2bd3777a08563fea3ca2a47c5f91bc26f4483a5cd6"
R007_RESULTS_SHA256 = "3a6ab612919fd9e5eeef2f4bd030b74c25f6b6f871b6c25c13f92efac5ba9b7d"
R007_CLOSURE_COMMIT = "7cc3acc120027bdd181340b443633d8a03d6858d"


def validate_r007_exhaustion_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r007-state-repair-closure-v1",
        "amendment_id": "V3-E006-R007",
        "status": "r007_candidate_budget_exhausted_no_valid_state_pair",
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
            raise ValueError(f"R007 exhaustion closure field differs: {key}")
    attempts = payload.get("candidate_attempts")
    if not isinstance(attempts, list) or [row.get("candidate_rank") for row in attempts] != [1, 2, 3, 4]:
        raise ValueError("R007 candidate order differs")
    for attempt in attempts:
        if attempt.get("passed") is not False:
            raise ValueError("R007 exhausted rank outcome differs")
        stages = attempt.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != {"canonical_grasp", "canonical_carry"}:
            raise ValueError("R007 stage inventory differs")
        for state in stages.values():
            if not isinstance(state, Mapping) or state.get("passed") is not False:
                raise ValueError("R007 rejected stage differs")
            if state.get("physics_gate", {}).get("passed") is not False:
                raise ValueError("R007 rejected physics gate differs")
            if state.get("ood_gate", {}).get("passed") is not True:
                raise ValueError("R007 OOD outcome differs")
            if state.get("camera_gate_passed") is not True:
                raise ValueError("R007 camera outcome differs")
            if state.get("companion_gate", {}).get("passed") is not True:
                raise ValueError("R007 companion outcome differs")
            if state.get("frame_identity_passed") is not True:
                raise ValueError("R007 frame outcome differs")
    raw = payload.get("raw_result")
    if not isinstance(raw, Mapping) or raw.get("sha256") != R007_RAW_RESULT_SHA256:
        raise ValueError("R007 raw-result binding differs")
    if payload.get("authoritative_target_validation_receipt", {}).get("sha256") != (
        "5b02a70eee847773a314f7a37da2f6b2575d04c0e095dcb90a9cda6e2327a476"
    ):
        raise ValueError("R007 authoritative validator receipt differs")
    if payload.get("postexecution_validator_amendment", {}).get("sha256") != (
        "c34486692c4d7c451500e1925d731769f33a7fc14e1804c594ccadb2e3a9367a"
    ):
        raise ValueError("R007 validator amendment differs")
