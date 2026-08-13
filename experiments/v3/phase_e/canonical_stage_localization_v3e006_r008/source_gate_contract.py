"""Prospective source-push contract for V3-E006-R008."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "vla-wam-shared-v3e006-r008-source-push-gate-v1"
STATUS = "passed_before_first_r008_live_diagnostic_candidate_or_model_request"


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
        raise ValueError("R008 source-push identity/status differs")
    for key in (
        "model_request_count", "behavioral_episode_count", "r008_live_diagnostic_count",
        "r008_live_candidate_evaluation_count", "accepted_state_candidate_count",
        "infrastructure_invalid_attempt_count",
    ):
        if value.get(key) != 0:
            raise ValueError(f"R008 source gate is not prospective: {key}")
    if value.get("r007_closure_commit") != "7cc3acc120027bdd181340b443633d8a03d6858d":
        raise ValueError("R007 closure commit differs")
    if value.get("r007_results_sha256") != "3a6ab612919fd9e5eeef2f4bd030b74c25f6b6f871b6c25c13f92efac5ba9b7d":
        raise ValueError("R007 results digest differs")
    registration = json.loads(_verify(study_root, value["repair_registration"], "registration").read_text())
    schedule = json.loads(_verify(study_root, value["candidate_schedule"], "schedule").read_text())
    if registration.get("counts_at_registration") != {
        "r008_live_diagnostics": 0, "r008_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        raise ValueError("R008 registration counts differ")
    if schedule.get("object_space_servo_contract", {}).get("servo_steps") != 360:
        raise ValueError("R008 servo schedule differs")
    if schedule.get("joint_handoff_contract", {}).get("settle_steps") != 600:
        raise ValueError("R008 handoff schedule differs")
    lifecycle = schedule.get("construction_lifecycle_contract", {})
    if (
        lifecycle.get("worst_case_steps") != 1365
        or lifecycle.get("registered_max_episode_length_steps") != 1500
        or lifecycle.get("registered_margin_steps") != 135
        or "construction_horizon_contract" in schedule
        or "open_contact_construction_contract" in schedule
        or schedule.get("archived_predecessor_contracts", {}).get("status")
        != "archived_lineage_only_not_active_r008_runtime_evidence"
    ):
        raise ValueError("R008 authoritative construction lifecycle differs")
    files = value.get("implementation_files")
    if not isinstance(files, list) or len(files) != value.get("implementation_file_count"):
        raise ValueError("R008 source inventory differs")
    for row in files:
        _verify(study_root, row, "implementation file")
    predecessor = registration.get("r007_predecessor", {})
    result_path = _verify(study_root, predecessor.get("results", {}), "R007 results")
    if verify_raw_history:
        result = json.loads(result_path.read_text())
        for key in (
            "raw_result", "raw_harness", "raw_launch", "raw_runtime_log",
            "authoritative_target_validation_receipt",
        ):
            _verify(study_root, result[key], f"R007 {key}")
