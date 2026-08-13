"""Prospective retry source-gate contract for V3-E006-R005."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


V1_SCHEMA = "vla-wam-shared-v3e006-r005-source-push-gate-v1"
V1_STATUS = "passed_before_first_r005_live_diagnostic_candidate_or_model_request"
V2_SCHEMA = "vla-wam-shared-v3e006-r005-source-push-gate-v2"
V2_STATUS = "passed_before_identical_r005_retry_after_terminal_serialization_fix"
ATTEMPT_SCHEMA = "vla-wam-shared-v3e006-r005-infrastructure-attempt-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
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


def _read_single_attempt(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise ValueError("R005 retry requires exactly one retained infrastructure attempt")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid R005 infrastructure ledger: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("R005 infrastructure ledger row is not an object")
    return value


def validate_retry_source_gate(
    value: Mapping[str, Any], *, study_root: Path, verify_raw_history: bool
) -> dict[str, Any]:
    """Validate the sole prospective retry gate and its zero-model predecessor."""

    if value.get("schema_version") != V2_SCHEMA or value.get("status") != V2_STATUS:
        raise ValueError("R005 retry source-push gate identity/status differs")
    expected_counts = {
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r005_live_diagnostic_count": 4,
        "r005_live_candidate_evaluation_count": 4,
        "completed_candidate_pair_count": 0,
        "raw_candidate_pair_compute_count": 4,
        "accepted_state_candidate_count": 0,
        "infrastructure_invalid_search_attempt_count": 1,
    }
    for key, expected in expected_counts.items():
        if value.get(key) != expected:
            raise ValueError(f"R005 retry source-push history differs: {key}")
    if value.get("retry_scope") != "identical_zero_model_search_after_reporting_only_fix":
        raise ValueError("R005 retry scope differs")

    v1_path = _resolve_binding(
        study_root, value.get("superseded_source_push_gate", {}), "R005 v1 source gate"
    )
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    if v1.get("schema_version") != V1_SCHEMA or v1.get("status") != V1_STATUS:
        raise ValueError("R005 v1 source gate identity changed")
    for key in (
        "model_request_count",
        "behavioral_episode_count",
        "r005_live_diagnostic_count",
        "r005_live_candidate_evaluation_count",
        "completed_candidate_pair_count",
        "accepted_state_candidate_count",
        "infrastructure_invalid_search_attempt_count",
    ):
        if v1.get(key) != 0:
            raise ValueError(f"R005 v1 source gate was not prospective: {key}")

    ledger_path = _resolve_binding(
        study_root, value.get("infrastructure_attempts", {}), "R005 infrastructure ledger"
    )
    attempt = _read_single_attempt(ledger_path)
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA
        or attempt.get("status")
        != "infrastructure_invalid_after_complete_compute_terminal_serialization_failure"
        or attempt.get("model_request_count") != 0
        or attempt.get("behavioral_episode_count") != 0
        or attempt.get("state_candidate_count") != 0
        or attempt.get("candidate_gate_passed") is not False
        or attempt.get("behavioral_denominator_included") is not False
    ):
        raise ValueError("R005 infrastructure attempt semantics differ")
    completeness = attempt.get("completeness", {})
    if (
        completeness.get("diagnostic_compute_count") != 4
        or completeness.get("candidate_pair_compute_count") != 4
        or completeness.get("scientifically_completed_candidate_pair_count") != 0
        or completeness.get("offline_terminal_recovery_permitted") is not False
    ):
        raise ValueError("R005 infrastructure attempt completeness differs")
    failure = attempt.get("failure", {})
    if (
        failure.get("exception_type") != "KeyError"
        or failure.get("exception_message") != "'candidate_search'"
        or failure.get("source_expression") != 'repair_registration["candidate_search"]'
    ):
        raise ValueError("R005 retained terminal-serialization cause differs")

    if verify_raw_history:
        raw = attempt.get("raw_bindings", {})
        expected_names = {
            "failure_report",
            "harness_result",
            "launch",
            "runtime_log",
            "serialized_invocation",
            "target_validation_receipt",
            "video",
        }
        if set(raw) != expected_names:
            raise ValueError("R005 infrastructure raw-binding inventory differs")
        paths = {
            name: _resolve_binding(study_root, binding, f"R005 attempt {name}")
            for name, binding in raw.items()
        }
        harness = json.loads(paths["harness_result"].read_text(encoding="utf-8"))
        failure_report = json.loads(paths["failure_report"].read_text(encoding="utf-8"))
        receipt = json.loads(paths["target_validation_receipt"].read_text(encoding="utf-8"))
        if (
            harness.get("status") != "infrastructure_invalid_r005_state_repair"
            or harness.get("process_completed") is not False
            or harness.get("model_request_count") != 0
            or harness.get("behavioral_episode_count") != 0
            or harness.get("r005_live_diagnostic_count") != 4
            or harness.get("repair_candidate_evaluation_count") != 4
        ):
            raise ValueError("R005 raw harness history differs")
        if (
            failure_report.get("status") != "infrastructure_invalid_r005_state_repair"
            or failure_report.get("model_request_count") != 0
            or failure_report.get("behavioral_episode_count") != 0
            or failure_report.get("state_candidate_count") != 0
            or failure_report.get("error", {}).get("type") != "KeyError"
            or failure_report.get("error", {}).get("message") != "'candidate_search'"
        ):
            raise ValueError("R005 raw child failure history differs")
        evidence = receipt.get("candidate_evidence", {})
        if (
            receipt.get("passed") is not True
            or evidence.get("passed") is not True
            or evidence.get("child_report", {}).get("sha256")
            != raw["failure_report"].get("sha256")
            or evidence.get("harness", {}).get("sha256")
            != raw["harness_result"].get("sha256")
            or evidence.get("launch", {}).get("sha256") != raw["launch"].get("sha256")
        ):
            raise ValueError("R005 target raw-validation receipt differs")
    return attempt
