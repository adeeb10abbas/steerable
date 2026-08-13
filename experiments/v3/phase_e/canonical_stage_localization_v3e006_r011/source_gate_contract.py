"""Prospective source-push contract for V3-E006-R011."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r011.pinch_geometry import (
    validate_attachment_preflight_contract,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r011.predecessor_contract import (
    R010_CLOSURE_COMMIT,
    R010_RESULTS_SHA256,
    validate_r010_oracle_failure_closure,
)

SCHEMA = "vla-wam-shared-v3e006-r011-source-push-gate-v1"
STATUS = "passed_before_first_r011_live_preflight_diagnostic_candidate_or_model_request"


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
        raise ValueError("R011 source-push identity/status differs")
    for key in (
        "model_request_count",
        "behavioral_episode_count",
        "r011_geometry_attachment_preflight_count",
        "r011_live_diagnostic_count",
        "r011_live_candidate_evaluation_count",
        "accepted_state_candidate_count",
        "infrastructure_invalid_attempt_count",
    ):
        if value.get(key) != 0:
            raise ValueError(f"R011 source gate is not prospective: {key}")
    if value.get("r010_closure_commit") != R010_CLOSURE_COMMIT:
        raise ValueError("R010 closure commit differs")
    if value.get("r010_results_sha256") != R010_RESULTS_SHA256:
        raise ValueError("R010 results digest differs")

    registration = json.loads(_verify(study_root, value["repair_registration"], "registration").read_text())
    schedule = json.loads(_verify(study_root, value["candidate_schedule"], "schedule").read_text())
    if registration.get("counts_at_registration") != {
        "r011_geometry_attachment_preflights": 0,
        "r011_live_diagnostics": 0,
        "r011_live_candidate_evaluations": 0,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }:
        raise ValueError("R011 registration counts differ")
    validate_contract(schedule.get("pinch_geometry_contract", {}))
    validate_attachment_preflight_contract(
        schedule.get("geometry_attachment_preflight_contract", {})
    )
    if (
        schedule.get("selection_rule", {}).get("algorithm_version")
        != "r011-scene-sync-validated-relative-bound-collision-pinch-first-passing-pair-v1"
        or schedule.get("joint_handoff_contract", {}).get("settle_steps") != 600
        or schedule.get("construction_lifecycle_contract", {}).get("worst_case_steps") != 1695
        or schedule.get("construction_lifecycle_contract", {}).get(
            "registered_max_episode_length_steps"
        )
        != 1800
    ):
        raise ValueError("R011 retained controller/selection contract differs")
    sources = schedule.get("scene_sync_source_bindings", {})
    if set(sources) != {
        "physx_python_interface",
        "nvidia_physx_scene_specific_transform_test",
        "isaac_sim_core_simulation_context",
        "isaaclab_simulation_context",
    }:
        raise ValueError("R011 scene-sync source inventory differs")
    for label, row in sources.items():
        _verify(study_root, row, f"R011 scene-sync source {label}")

    files = value.get("implementation_files")
    if not isinstance(files, list) or len(files) != value.get("implementation_file_count"):
        raise ValueError("R011 source inventory differs")
    for row in files:
        _verify(study_root, row, "implementation file")
    predecessor = registration.get("r010_predecessor", {})
    result_path = _verify(study_root, predecessor.get("results", {}), "R010 results")
    result = json.loads(result_path.read_text())
    validate_r010_oracle_failure_closure(result)
    if verify_raw_history:
        for key in (
            "raw_result",
            "raw_preflight",
            "raw_harness",
            "raw_launch",
            "raw_runtime_log",
            "authoritative_target_validation_receipt",
        ):
            _verify(study_root, result[key], f"R010 {key}")
