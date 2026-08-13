"""Pure validation of the immutable V3-E006-R002 exhaustion closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


R002_RAW_RESULT_SHA256 = "afb8c3ba2b53f1513bd22fd6135b16cbfe3e4dd9de3fe3d818ac05e458311fe7"


def validate_r002_exhaustion(payload: Mapping[str, Any]) -> None:
    """Fail closed unless *payload* is the exact compact R002 exhaustion shape.

    The committed compact closure records ``candidate_pair_count`` and
    ``stage_solve_count``.  It never contained a ``candidate_budget`` field.
    """

    expected_scalars = {
        "schema_version": "vla-wam-shared-v3e006-r002-state-repair-closure-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E006-R002",
        "status": "r002_candidate_budget_exhausted_no_valid_state_pair",
        "passed": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_pair_count": 8,
        "stage_solve_count": 16,
        "accepted_candidate_rank": None,
        "first_passing_rule_obeyed": True,
        "behavioral_activation_released": False,
        "scientific_gate_thresholds_unchanged": True,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(f"R002 compact exhaustion field differs: {key}")

    selection = payload.get("selection_rule")
    if not isinstance(selection, Mapping) or selection.get("maximum_candidate_pairs") != 8:
        raise ValueError("R002 compact exhaustion selection rule differs")
    if not isinstance(selection.get("first_pass_rule"), str) or not isinstance(
        selection.get("exhaustion"), str
    ):
        raise ValueError("R002 compact exhaustion first-pass contract is absent")

    stage_solves = payload.get("stage_solves")
    if not isinstance(stage_solves, list) or len(stage_solves) != 16:
        raise ValueError("R002 compact exhaustion stage-solve ledger differs")
    expected_grid = [
        (rank, stage)
        for rank in range(1, 9)
        for stage in ("canonical_grasp", "canonical_carry")
    ]
    observed_grid = [(row.get("candidate_rank"), row.get("stage")) for row in stage_solves]
    if observed_grid != expected_grid:
        raise ValueError("R002 compact exhaustion rank/stage grid differs")

    raw_evidence = payload.get("raw_evidence")
    expected_names = {
        "harness", "launch", "result", "runtime_log", "target_validation_receipt"
    }
    if not isinstance(raw_evidence, Mapping) or set(raw_evidence) != expected_names:
        raise ValueError("R002 compact exhaustion raw-evidence inventory differs")
    for name, row in raw_evidence.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"R002 raw-evidence binding is invalid: {name}")
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError(f"R002 raw-evidence path is invalid: {name}")
        if not isinstance(row.get("bytes"), int) or isinstance(row.get("bytes"), bool) or row["bytes"] < 0:
            raise ValueError(f"R002 raw-evidence byte count is invalid: {name}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"R002 raw-evidence digest is invalid: {name}")
    if raw_evidence["result"]["sha256"] != R002_RAW_RESULT_SHA256:
        raise ValueError("R002 raw state-repair result digest differs")
