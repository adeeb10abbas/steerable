"""Pure validation of the immutable V3-E006-R006 state-search closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R006_RAW_RESULT_SHA256 = "7eae75c38a7b65ba4b8fbc44f3ca4c565c3af5675134c93570b1dc0e85176011"


def validate_r006_exhaustion_closure(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "vla-wam-shared-v3e006-r006-state-repair-closure-v2",
        "amendment_id": "V3-E006-R006",
        "status": "r006_candidate_budget_exhausted_no_valid_state_pair",
        "passed": False,
        "accepted_candidate_rank": None,
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
            raise ValueError(f"R006 exhaustion closure field differs: {key}")
    diagnostics = payload.get("known_reachable_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 4:
        raise ValueError("R006 diagnostic summary differs")
    if any(row.get("passed") is not True for row in diagnostics):
        raise ValueError("R006 diagnostic outcome differs")
    attempts = payload.get("candidate_attempts")
    if not isinstance(attempts, list) or [row.get("candidate_rank") for row in attempts] != [1, 2, 3, 4]:
        raise ValueError("R006 candidate order differs")
    for attempt in attempts:
        if attempt.get("passed") is not False:
            raise ValueError("R006 exhausted rank outcome differs")
        stages = attempt.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != {"canonical_grasp", "canonical_carry"}:
            raise ValueError("R006 stage inventory differs")
        for state in stages.values():
            if not isinstance(state, Mapping) or state.get("passed") is not False:
                raise ValueError("R006 rejected stage differs")
            physics = state.get("physics_gate")
            if not isinstance(physics, Mapping) or physics.get("passed") is not False:
                raise ValueError("R006 rejected physics gate differs")
            for name in ("ood_gate_passed", "camera_gate_passed", "companion_gate_passed"):
                if state.get(name) is not True:
                    raise ValueError(f"R006 unchanged {name} differs")
    raw = payload.get("raw_result")
    if not isinstance(raw, Mapping) or raw.get("sha256") != R006_RAW_RESULT_SHA256:
        raise ValueError("R006 raw-result binding differs")
    for name in (
        "raw_result", "raw_harness", "raw_launch", "raw_runtime_log",
        "raw_target_validation_receipt", "registration", "candidate_schedule", "source_push_gate",
    ):
        row = payload.get(name)
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError(f"R006 binding differs: {name}")
        if not isinstance(row.get("bytes"), int) or isinstance(row.get("bytes"), bool):
            raise ValueError(f"R006 binding bytes differ: {name}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"R006 binding digest differs: {name}")
