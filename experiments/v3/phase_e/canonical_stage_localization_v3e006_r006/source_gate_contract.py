"""Prospective source-gate contract for V3-E006-R006."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "vla-wam-shared-v3e006-r006-source-push-gate-v1"
STATUS = "passed_before_first_r006_live_diagnostic_candidate_or_model_request"


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
        raise ValueError("R006 source-push gate identity/status differs")
    for key in (
        "model_request_count", "behavioral_episode_count",
        "r006_live_diagnostic_count", "r006_live_candidate_evaluation_count",
        "accepted_state_candidate_count", "infrastructure_invalid_attempt_count",
    ):
        if value.get(key) != 0:
            raise ValueError(f"R006 source-push gate is not prospective: {key}")
    if value.get("r005_closure_commit") != "040cf75c1d83a2e5f8383d87247fb096e8d2491a":
        raise ValueError("R005 closure lineage changed")
    if value.get("r005_results_sha256") != "550665a234c378cbcb5c8022d16249a980d1a5b5368b08900568c959c51fb9f2":
        raise ValueError("R005 closure result changed")
    registration = json.loads(_verify(
        study_root, value.get("repair_registration", {}), "R006 registration"
    ).read_text(encoding="utf-8"))
    schedule = json.loads(_verify(
        study_root, value.get("candidate_schedule", {}), "R006 schedule"
    ).read_text(encoding="utf-8"))
    if registration.get("counts_at_registration") != {
        "r006_live_diagnostics": 0, "r006_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        raise ValueError("R006 registration counts changed")
    if schedule.get("joint_equilibrium_hold_contract", {}).get("settle_steps") != 780:
        raise ValueError("R006 equilibrium settle count changed")
    files = value.get("implementation_files")
    if not isinstance(files, list) or not files:
        raise ValueError("R006 source-push inventory is absent")
    for row in files:
        _verify(study_root, row, "R006 implementation file")
    predecessor = registration.get("r005_predecessor", {})
    results_path = _verify(study_root, predecessor.get("results", {}), "R005 closure")
    if verify_raw_history:
        predecessor_result = json.loads(results_path.read_text(encoding="utf-8"))
        for key in (
            "raw_result", "raw_harness", "raw_launch", "raw_runtime_log",
            "raw_target_validation_receipt",
        ):
            _verify(study_root, predecessor_result[key], f"R005 {key}")
