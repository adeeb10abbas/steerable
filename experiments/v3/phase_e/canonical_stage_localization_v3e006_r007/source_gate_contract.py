"""Prospective source-gate contract for V3-E006-R007."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "vla-wam-shared-v3e006-r007-source-push-gate-v1"
STATUS = "passed_before_first_r007_live_diagnostic_candidate_or_model_request"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding.get("path", "")))
    if not path.is_absolute():
        path = root / path
    if (
        not path.is_file()
        or path.stat().st_size != binding.get("bytes")
        or _sha256(path) != binding.get("sha256")
    ):
        raise ValueError(f"{label} binding changed: {path}")
    return path


def validate_source_gate(
    value: Mapping[str, Any], *, study_root: Path, verify_raw_history: bool
) -> None:
    if value.get("schema_version") != SCHEMA or value.get("status") != STATUS:
        raise ValueError("R007 source-push gate identity/status differs")
    for key in (
        "model_request_count", "behavioral_episode_count",
        "r007_live_diagnostic_count", "r007_live_candidate_evaluation_count",
        "accepted_state_candidate_count", "infrastructure_invalid_attempt_count",
    ):
        if value.get(key) != 0:
            raise ValueError(f"R007 source-push gate is not prospective: {key}")
    if value.get("r006_closure_commit") != "125e8f0d231ebd2e3c7d0d9b54dce83e1080cea1":
        raise ValueError("R006 closure lineage changed")
    if value.get("r006_results_sha256") != "3c58721d11f669243690aaf3619121d1c348bf788ca56aacd2a009f727065e63":
        raise ValueError("R006 closure result changed")
    registration = json.loads(_verify(
        study_root, value.get("repair_registration", {}), "R007 registration"
    ).read_text(encoding="utf-8"))
    schedule = json.loads(_verify(
        study_root, value.get("candidate_schedule", {}), "R007 schedule"
    ).read_text(encoding="utf-8"))
    if registration.get("counts_at_registration") != {
        "r007_live_diagnostics": 0, "r007_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        raise ValueError("R007 registration counts changed")
    contract = schedule.get("open_contact_construction_contract", {})
    if contract.get("candidate_action_steps") != 810 or contract.get("phase_steps") != {
        "open_approach": 120, "open_descent": 120, "normal_close": 90,
        "closed_lift_to_registered_stage_target": 180,
        "closed_settle_at_registered_stage_target": 300,
    }:
        raise ValueError("R007 open-contact action schedule changed")
    files = value.get("implementation_files")
    if not isinstance(files, list) or not files:
        raise ValueError("R007 source-push inventory is absent")
    for row in files:
        _verify(study_root, row, "R007 implementation file")
    predecessor = registration.get("r006_predecessor", {})
    results_path = _verify(study_root, predecessor.get("results", {}), "R006 closure")
    if verify_raw_history:
        predecessor_result = json.loads(results_path.read_text(encoding="utf-8"))
        for key in (
            "raw_result", "raw_harness", "raw_launch", "raw_runtime_log",
            "raw_target_validation_receipt",
        ):
            _verify(study_root, predecessor_result[key], f"R006 {key}")
