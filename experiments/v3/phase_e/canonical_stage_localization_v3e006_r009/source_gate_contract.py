"""Prospective source-push contract for V3-E006-R009."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "vla-wam-shared-v3e006-r009-source-push-gate-v1"
STATUS = "passed_before_first_r009_live_diagnostic_candidate_or_model_request"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or path.stat().st_size != row.get("bytes") or _sha(path) != row.get("sha256"):
        raise ValueError(f"{label} binding changed: {path}")
    return path


def validate_source_gate(value: Mapping[str, Any], *, study_root: Path, verify_raw_history: bool) -> None:
    if value.get("schema_version") != SCHEMA or value.get("status") != STATUS:
        raise ValueError("R009 source-push identity/status differs")
    for key in (
        "model_request_count", "behavioral_episode_count", "r009_live_diagnostic_count",
        "r009_live_candidate_evaluation_count", "accepted_state_candidate_count",
        "infrastructure_invalid_attempt_count",
    ):
        if value.get(key) != 0:
            raise ValueError(f"R009 source gate is not prospective: {key}")
    if value.get("r008_closure_commit") != "a13d5b7f8a4be6374a22483a436b9a41aadd1c9f":
        raise ValueError("R008 closure commit differs")
    if value.get("r008_results_sha256") != "5f86b77b721806b0c19416b622e9a6aab68b13de6d596a6dadec89ed41a081d5":
        raise ValueError("R008 results digest differs")
    registration = json.loads(_verify(study_root, value["repair_registration"], "registration").read_text())
    schedule = json.loads(_verify(study_root, value["candidate_schedule"], "schedule").read_text())
    if registration.get("counts_at_registration") != {
        "r009_live_diagnostics": 0, "r009_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        raise ValueError("R009 registration counts differ")
    if schedule.get("pinch_geometry_contract", {}).get("closed_stage_transport_steps") != 300:
        raise ValueError("R009 pinch schedule differs")
    if (
        schedule.get("pinch_geometry_contract", {}).get("algorithm_version")
        != "uniform-tensor-reconstructed-collision-pinch-acquisition-v1"
        or schedule.get("pinch_geometry_contract", {}).get(
            "dynamic_usd_world_bounds_used"
        )
        is not False
        or schedule.get("selection_rule", {}).get("algorithm_version")
        != "r009-uniform-tensor-collision-pinch-first-passing-pair-v1"
    ):
        raise ValueError("R009 tensor-geometry/selection contract differs")
    if schedule.get("joint_handoff_contract", {}).get("settle_steps") != 600:
        raise ValueError("R009 handoff schedule differs")
    lifecycle = schedule.get("construction_lifecycle_contract", {})
    if (
        lifecycle.get("worst_case_steps") != 1695
        or lifecycle.get("registered_max_episode_length_steps") != 1800
        or lifecycle.get("registered_margin_steps") != 105
        or "construction_horizon_contract" in schedule
        or "open_contact_construction_contract" in schedule
        or schedule.get("archived_predecessor_contracts", {}).get("status")
        != "archived_lineage_only_not_active_r009_runtime_evidence"
    ):
        raise ValueError("R009 authoritative construction lifecycle differs")
    files = value.get("implementation_files")
    if not isinstance(files, list) or len(files) != value.get("implementation_file_count"):
        raise ValueError("R009 source inventory differs")
    for row in files:
        _verify(study_root, row, "implementation file")
    predecessor = registration.get("r008_predecessor", {})
    result_path = _verify(study_root, predecessor.get("results", {}), "R008 results")
    if verify_raw_history:
        result = json.loads(result_path.read_text())
        for key in (
            "raw_result", "raw_harness", "raw_launch", "raw_runtime_log",
            "authoritative_target_validation_receipt", "postexecution_validator_amendment",
        ):
            _verify(study_root, result[key], f"R008 {key}")
