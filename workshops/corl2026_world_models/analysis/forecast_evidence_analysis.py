#!/usr/bin/env python3
"""Fail-closed confirmation analysis for WMF-ABLATION-001.

The command consumes a signed evidence manifest.  It revalidates the complete
request-selection/recording chain, the confirmation annotation freeze, and the
mechanically reproduced human consensus before computing any forecast metric.
Technical-invalid, safety-censored, and unrun cells remain separate throughout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parents[1]
ANNOTATION_MODULE = PACKAGE / "analysis" / "forecast_annotation_workflow.py"
RELEASE_FREEZE_MODULE = PACKAGE / "analysis" / "freeze_development_release.py"
RECORDING_MODULE = PACKAGE / "experiments" / "forecast_layout" / "recording_adapter.py"
SPEC_PATH = PACKAGE / "experiments" / "forecast_layout" / "ablation_spec.json"
STUDY_ID = "WMF-ABLATION-001"
EVIDENCE_SCHEMA = "wmf-forecast-analysis-evidence-manifest-v1"
ENDPOINT_SCHEMA = "wmf-forecast-endpoint-trace-receipt-v1"
HISTORY_SCHEMA = "wmf-forecast-request-history-receipt-v1"
OUTPUT_SCHEMA = "wmf-forecast-final-analysis-v1"
ANALYSIS_SEED = 2026091301
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
CONDITIONS = (
    "original_left",
    "original_right",
    "reflected_left",
    "reflected_right",
)
MODELS = ("N3", "D1")
MODEL_BRANCHES = {
    "full_two_model": ("N3", "D1"),
    "reduced_n3": ("N3",),
    "reduced_d1": ("D1",),
}
STATUS_VALUES = ("valid_complete", "valid_censored", "technical_invalid", "not_run")
REFERENCE_KEYS = {"path", "sha256"}
EVIDENCE_KEYS = {
    "schema_version",
    "study_id",
    "stage",
    "cohort_branch",
    "sources",
    "endpoint_trace_receipts",
    "request_history_receipts",
    "payload_sha256",
}
SOURCE_KEYS = {
    "ablation_spec",
    "development_release_freeze",
    "request_selection",
    "annotation_freeze",
    "restricted_map",
    "final_consensus",
}
ENDPOINT_KEYS = {
    "schema_version",
    "study_id",
    "stage",
    "cell_id",
    "recording_id",
    "model_id",
    "recording_status",
    "recording_receipt_sha256",
    "action_manifest_sha256",
    "executed_action_count",
    "first_success_action_index",
    "action_zero",
    "action_450",
    "first_success_or_action_450",
    "action_zero_observation_id",
    "action_450_observation_id",
    "first_success_or_action_450_observation_id",
    "source_adapter_completion",
    "source_adapter_journal",
    "payload_sha256",
}
POINT_KEYS = {"action_index", "cube_robot_xyz", "bowl_robot_xyz"}
HISTORY_KEYS = {
    "schema_version",
    "study_id",
    "stage",
    "source_request_id",
    "cell_id",
    "alignment_receipt_id",
    "alignment_receipt_sha256",
    "camera_id",
    "preceding_observation_id",
    "current_observation_id",
    "preceding_camera_capture_time_ns",
    "current_camera_capture_time_ns",
    "preceding_physics_time_s",
    "current_physics_time_s",
    "preceding_observation_interval_s",
    "source_adapter_completion",
    "source_adapter_journal",
    "payload_sha256",
}
ANNOTATION_QUALITY_KEYS = {
    "unit",
    "images",
    "first_pass_exact_agreements",
    "independently_adjudicated",
    "first_pass_exact_agreement_rate",
    "independent_adjudication_rate",
    "decision_inventory_sha256",
    "final_consensus_sha256",
    "source_restricted_map_sha256",
    "adjudication_map_sha256",
    "first_pass_response_sha256_by_slot",
    "rater_code_sha256_by_slot",
    "adjudicator_response_sha256",
}
_ANNOTATION_VALIDATOR: Any | None = None
_RELEASE_FREEZE_VALIDATOR: Any | None = None
_RECORDING_VALIDATOR: Any | None = None


class AnalysisContractError(ValueError):
    """Evidence is insufficient or inconsistent for scientific analysis."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisContractError(message)


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    require(not missing, f"{label} missing keys: {sorted(missing)}")
    require(not extra, f"{label} has disallowed keys: {sorted(extra)}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_hash(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("payload_sha256", None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def sign_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result["payload_sha256"] = payload_hash(result)
    return result


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def verify_signed(document: Mapping[str, Any], label: str) -> None:
    observed = require_sha256(document.get("payload_sha256"), f"{label} payload_sha256")
    require(payload_hash(document) == observed, f"{label} payload hash mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_reference(base: Path, reference: Mapping[str, Any], label: str) -> Path:
    exact_keys(reference, REFERENCE_KEYS, label)
    expected = require_sha256(reference.get("sha256"), f"{label} sha256")
    raw_path = reference.get("path")
    require(isinstance(raw_path, str) and raw_path.strip(), f"{label} path must be nonempty")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    require(not path.is_symlink(), f"{label} must not be a symlink")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise AnalysisContractError(f"{label} file is missing: {path}") from error
    require(path.is_file(), f"{label} file is missing: {path}")
    require(sha256_file(path) == expected, f"{label} file hash mismatch")
    return path


def resolve_already_validated_reference(
    base: Path,
    reference: Mapping[str, Any],
    label: str,
    *,
    expected_path: Path,
    expected_sha256: str,
) -> Path:
    """Bind a repeated reference to a source already hashed in this process.

    History receipts repeat their cell's native completion/journal references.
    Rehashing the same journal once per selected request would scale with the
    number of requests instead of recordings.  This validates the exact path
    and digest declaration; ``load_evidence`` hashes every unique source again
    after all history checks to close the final evidence gate.
    """

    exact_keys(reference, REFERENCE_KEYS, label)
    observed_sha256 = require_sha256(reference.get("sha256"), f"{label} sha256")
    require(observed_sha256 == expected_sha256, f"{label} hash differs from validated cell evidence")
    raw_path = reference.get("path")
    require(isinstance(raw_path, str) and raw_path.strip(), f"{label} path must be nonempty")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    require(not candidate.is_symlink(), f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise AnalysisContractError(f"{label} file is missing: {candidate}") from error
    require(resolved.is_file(), f"{label} file is missing: {resolved}")
    require(resolved == expected_path, f"{label} path differs from validated cell evidence")
    return resolved


def finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    return float(value)


def finite_xyz(value: Any, label: str) -> tuple[float, float, float]:
    require(isinstance(value, list) and len(value) == 3, f"{label} must contain three coordinates")
    return tuple(finite_number(coordinate, label) for coordinate in value)  # type: ignore[return-value]


def validate_endpoint_point(value: Any, label: str, *, expected_action: int | None = None) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    exact_keys(value, POINT_KEYS, label)
    action_index = value.get("action_index")
    require(type(action_index) is int and action_index >= 0, f"{label} action_index is invalid")
    if expected_action is not None:
        require(action_index == expected_action, f"{label} action_index must be {expected_action}")
    cube = finite_xyz(value.get("cube_robot_xyz"), f"{label} cube_robot_xyz")
    bowl = finite_xyz(value.get("bowl_robot_xyz"), f"{label} bowl_robot_xyz")
    return {"action_index": action_index, "cube_robot_xyz": cube, "bowl_robot_xyz": bowl}


def _validate_endpoint_fields(
    receipt: Mapping[str, Any],
    *,
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    exact_keys(receipt, ENDPOINT_KEYS, "endpoint trace receipt")
    require(receipt.get("schema_version") == ENDPOINT_SCHEMA, "endpoint trace receipt schema mismatch")
    require(receipt.get("study_id") == STUDY_ID, "endpoint trace receipt study mismatch")
    require(receipt.get("stage") == "confirmation", "endpoint trace receipt must be confirmation evidence")
    verify_signed(receipt, "endpoint trace receipt")
    for key in (
        "cell_id",
        "recording_id",
        "model_id",
        "recording_status",
        "recording_receipt_sha256",
        "action_manifest_sha256",
        "executed_action_count",
    ):
        require(receipt.get(key) == roster.get(key), f"endpoint trace receipt {key} differs from recording roster")
    status = receipt["recording_status"]
    require(status in {"valid_complete", "valid_censored"}, "endpoint receipt cannot relabel invalid or unrun evidence")
    action_count = receipt["executed_action_count"]
    action_zero = validate_endpoint_point(receipt.get("action_zero"), "endpoint action_zero", expected_action=0)
    first_index = receipt.get("first_success_action_index")
    require(first_index is None or (type(first_index) is int and 1 <= first_index <= action_count), "first-success action is invalid")
    if status == "valid_complete":
        require(action_count == 450, "complete endpoint receipt must contain 450 actions")
        action_450 = validate_endpoint_point(receipt.get("action_450"), "endpoint action_450", expected_action=450)
    else:
        require(0 < action_count < 450, "censored endpoint receipt action count is invalid")
        require(receipt.get("action_450") is None, "censored endpoint receipt fabricates action 450")
        action_450 = None
    selected = receipt.get("first_success_or_action_450")
    if first_index is not None:
        selected = validate_endpoint_point(selected, "first_success_or_action_450", expected_action=first_index)
    elif status == "valid_complete":
        selected = validate_endpoint_point(selected, "first_success_or_action_450", expected_action=450)
        require(selected == action_450, "no-success endpoint must equal the recorded action-450 endpoint")
    else:
        require(selected is None, "censored no-success trace cannot substitute its last state for action 450")
    return {
        "cell_id": receipt["cell_id"],
        "action_zero": action_zero,
        "action_450": action_450,
        "first_success_or_action_450": selected,
        "first_success_action_index": first_index,
    }


def _read_adapter_source(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    completion_path = resolve_reference(
        receipt_path.parent,
        receipt.get("source_adapter_completion"),
        "source adapter completion",
    )
    journal_path = resolve_reference(
        receipt_path.parent,
        receipt.get("source_adapter_journal"),
        "source adapter journal",
    )
    require(
        completion_path.parent == journal_path.parent,
        "adapter completion and journal do not belong to one attempt directory",
    )
    completion = load_json(completion_path)
    require(completion.get("schema_version") == "wmf-forecast-recording-attempt-v1", "adapter completion schema mismatch")
    require(completion.get("study_id") == STUDY_ID, "adapter completion study mismatch")
    identity = completion.get("identity")
    require(isinstance(identity, Mapping), "adapter completion lacks attempt identity")
    expected_identity = {
        "cell_id": roster["cell_id"],
        "stage": "confirmation",
        "layout_pair_id": roster["layout_pair_id"],
        "model_config": roster["model_id"],
    }
    for key, expected in expected_identity.items():
        require(identity.get(key) == expected, f"adapter completion {key} differs from roster")
    arm, command = roster["condition_id"].split("_", maxsplit=1)
    require(identity.get("layout_arm") == arm and identity.get("command") == command, "adapter completion condition differs from roster")
    require(completion.get("actions_executed") == roster["executed_action_count"], "adapter completion action count differs from roster")
    require(completion.get("validation_errors") == [], "adapter completion contains technical validation errors")
    require(
        completion.get("success_configured_as_termination") is False,
        "adapter completion did not verify fixed-duration success handling",
    )
    status = roster["recording_status"]
    if status == "valid_complete":
        require(
            completion.get("behavioral_result_valid") is True
            and completion.get("stop_reason") == "action_cap"
            and completion.get("right_censored") is False
            and completion.get("technical_invalid") is False,
            "complete roster cell is not a valid completed adapter attempt",
        )
    else:
        require(
            status == "valid_censored"
            and completion.get("behavioral_result_valid") is False
            and completion.get("stop_reason") == "safety_abort"
            and completion.get("right_censored") is True
            and completion.get("technical_invalid") is False,
            "censored roster cell is not a safety-censored adapter attempt",
        )
    recording = load_recording_module()
    try:
        journal_identity = recording.verify_journal(journal_path)
    except Exception as error:
        raise AnalysisContractError(f"adapter journal validation failed: {error}") from error
    require(completion.get("journal_path") is not None, "adapter completion lacks journal path")
    recorded_journal = Path(completion["journal_path"])
    require(not recorded_journal.is_symlink(), "adapter completion journal path is a symlink")
    require(recorded_journal.resolve(strict=True) == journal_path, "adapter completion cites another journal")
    require(completion.get("event_count") == journal_identity["event_count"], "adapter completion event count differs")
    require(completion.get("journal_tail_sha256") == journal_identity["tail_sha256"], "adapter completion journal tail differs")
    events = []
    try:
        with journal_path.open("r", encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisContractError("adapter journal became unreadable after hash validation") from error
    return {
        "completion_path": completion_path,
        "journal_path": journal_path,
        "completion_sha256": receipt["source_adapter_completion"]["sha256"],
        "journal_sha256": receipt["source_adapter_journal"]["sha256"],
        "attempt_directory": completion_path.parent,
        "completion": completion,
        "events": events,
        "recording_module": recording,
    }


def _observation_events(adapter: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    steps = set()
    for event in adapter["events"]:
        if event.get("kind") != "observation_captured":
            continue
        row = event.get("payload")
        require(isinstance(row, Mapping), "observation journal event payload is invalid")
        observation_id = row.get("observation_id")
        control_step = row.get("control_step")
        require(isinstance(observation_id, str) and observation_id, "observation event lacks identity")
        require(type(control_step) is int and control_step >= 0, "observation event control step is invalid")
        require(observation_id not in result and control_step not in steps, "adapter journal duplicates an observation")
        result[observation_id] = row
        steps.add(control_step)
    completion = adapter["completion"]
    require(len(result) == completion.get("observation_count"), "observation journal count differs from completion")
    require(steps == set(range(completion["actions_executed"] + 1)), "observation journal steps are not contiguous")
    return result


def _derive_first_success(adapter: Mapping[str, Any], command: str) -> Mapping[str, Any] | None:
    """Reproduce first success from the per-action native journal snapshots."""

    completed: dict[int, Mapping[str, Any]] = {}
    first_events: list[Mapping[str, Any]] = []
    for event in adapter["events"]:
        payload = event.get("payload")
        if event.get("kind") == "environment_step_completed":
            require(isinstance(payload, Mapping), "environment completion event payload is invalid")
            action_step = payload.get("action_step")
            require(type(action_step) is int and action_step > 0, "environment completion action step is invalid")
            require(action_step not in completed, "adapter journal duplicates an environment completion")
            completed[action_step] = payload
        elif event.get("kind") == "first_success":
            require(isinstance(payload, Mapping), "first-success journal event payload is invalid")
            first_events.append(payload)
    actions = adapter["completion"]["actions_executed"]
    require(set(completed) == set(range(1, actions + 1)), "environment completion steps do not cover every executed action")
    expected = None
    for action_step in range(1, actions + 1):
        success = completed[action_step].get("success_predicates")
        require(isinstance(success, Mapping), "environment completion lacks success predicates")
        require(
            type(success.get(command)) is bool and type(success.get("released")) is bool,
            "environment completion success predicates are invalid",
        )
        if success[command]:
            expected = {
                "action_step": action_step,
                "requested_relation": command,
                "released": success["released"],
                "observation_id": f"obs_{action_step:06d}",
            }
            break
    if expected is None:
        require(not first_events, "journal records first success absent from action snapshots")
    else:
        require(len(first_events) == 1, "journal does not record exactly one first-success event")
        require(dict(first_events[0]) == expected, "first-success event differs from action snapshots")
    require(adapter["completion"].get("first_success") == expected, "adapter completion first success differs from journal")
    return expected


def _payload_xyz(value: Any, label: str) -> tuple[float, float, float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    require(isinstance(value, list), f"{label} is not an array")
    return finite_xyz(value, label)


def _point_from_observation(
    adapter: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    observation_id: str,
) -> dict[str, Any]:
    payload = _load_observation_payload(adapter, observations, observation_id)
    row = observations[observation_id]
    state = payload.get("state")
    require(
        isinstance(state, Mapping)
        and state.get("simulator_state_sample_only_not_policy_input") is True,
        "observation lacks measurement-only simulator state attestation",
    )
    objects = state.get("objects")
    require(isinstance(objects, Mapping), "observation simulator state lacks objects")
    cube = objects.get("rubiks_cube")
    bowl = objects.get("bowl")
    require(isinstance(cube, Mapping) and isinstance(bowl, Mapping), "observation lacks cube or bowl state")
    return {
        "action_index": row["control_step"],
        "cube_robot_xyz": _payload_xyz(cube.get("position_robot_base_m"), "cube robot position"),
        "bowl_robot_xyz": _payload_xyz(bowl.get("position_robot_base_m"), "bowl robot position"),
    }


def _load_observation_payload(
    adapter: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    observation_id: str,
) -> Mapping[str, Any]:
    """Load once and cross-check a journal observation against its NPZ payload."""

    require(observation_id in observations, f"adapter journal lacks observation {observation_id}")
    row = observations[observation_id]
    descriptor = row.get("artifact")
    require(isinstance(descriptor, Mapping), f"observation {observation_id} lacks payload descriptor")
    require(descriptor.get("role") == "observation", f"observation {observation_id} cites another payload role")
    cache = adapter.get("observation_payloads")
    if cache is None:
        cache = {}
        adapter["observation_payloads"] = cache
    require(isinstance(cache, dict), "adapter observation payload cache is invalid")
    if observation_id in cache:
        return cache[observation_id]
    try:
        payload = adapter["recording_module"].load_payload(adapter["attempt_directory"], descriptor)
    except Exception as error:
        raise AnalysisContractError(f"observation payload validation failed for {observation_id}: {error}") from error
    require(isinstance(payload, Mapping), "observation payload is not a mapping")
    require(
        payload.get("observation_id") == observation_id,
        f"observation payload identity differs for {observation_id}",
    )
    require(payload.get("phase") == row.get("phase"), f"observation payload phase differs for {observation_id}")
    require(payload.get("clock") == row.get("clock"), f"observation payload clock differs for {observation_id}")
    cache[observation_id] = payload
    return payload


def validate_endpoint_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce every endpoint from the hash-checked native recorder state."""

    result = _validate_endpoint_fields(receipt, roster=roster)
    adapter = _read_adapter_source(receipt, receipt_path=receipt_path, roster=roster)
    observations = _observation_events(adapter)
    adapter["observations"] = observations
    completion = adapter["completion"]
    first = _derive_first_success(adapter, roster["condition_id"].split("_", maxsplit=1)[1])
    first_index = first.get("action_step") if isinstance(first, Mapping) else None
    require(first_index == result["first_success_action_index"], "endpoint first-success index differs from adapter completion")
    zero_id = "obs_000000"
    require(receipt.get("action_zero_observation_id") == zero_id, "endpoint action-zero observation identity changed")
    expected_zero = _point_from_observation(adapter, observations, zero_id)
    require(result["action_zero"] == expected_zero, "endpoint action-zero positions do not reproduce from recorder payload")
    if roster["recording_status"] == "valid_complete":
        action_450_id = "obs_000450"
        expected_450 = _point_from_observation(adapter, observations, action_450_id)
        require(receipt.get("action_450_observation_id") == action_450_id, "endpoint action-450 observation identity changed")
        require(result["action_450"] == expected_450, "endpoint action-450 positions do not reproduce from recorder payload")
    else:
        require(receipt.get("action_450_observation_id") is None, "censored endpoint cites an action-450 observation")
    if first_index is not None:
        selected_id = f"obs_{first_index:06d}"
        require(first.get("observation_id") == selected_id, "adapter first-success observation identity is inconsistent")
        expected_selected = _point_from_observation(adapter, observations, selected_id)
    elif roster["recording_status"] == "valid_complete":
        selected_id = "obs_000450"
        expected_selected = result["action_450"]
    else:
        selected_id = None
        expected_selected = None
    require(
        receipt.get("first_success_or_action_450_observation_id") == selected_id,
        "alternative endpoint observation identity changed",
    )
    require(
        result["first_success_or_action_450"] == expected_selected,
        "alternative endpoint positions do not reproduce from recorder payload",
    )
    result["adapter_source"] = adapter
    result["source_adapter_completion_sha256"] = adapter["completion_sha256"]
    result["source_adapter_journal_sha256"] = adapter["journal_sha256"]
    return result


def validate_history_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    request: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    exact_keys(receipt, HISTORY_KEYS, "request history receipt")
    require(receipt.get("schema_version") == HISTORY_SCHEMA, "request history receipt schema mismatch")
    require(receipt.get("study_id") == STUDY_ID, "request history receipt study mismatch")
    require(receipt.get("stage") == "confirmation", "request history receipt must be confirmation evidence")
    verify_signed(receipt, "request history receipt")
    for key in ("source_request_id", "cell_id", "alignment_receipt_id", "alignment_receipt_sha256", "camera_id"):
        require(receipt.get(key) == request.get(key), f"request history receipt {key} differs from selection")
    resolve_already_validated_reference(
        receipt_path.parent,
        receipt["source_adapter_completion"],
        "history adapter completion",
        expected_path=adapter["completion_path"],
        expected_sha256=adapter["completion_sha256"],
    )
    resolve_already_validated_reference(
        receipt_path.parent,
        receipt["source_adapter_journal"],
        "history adapter journal",
        expected_path=adapter["journal_path"],
        expected_sha256=adapter["journal_sha256"],
    )
    packed = [
        event["payload"]
        for event in adapter["events"]
        if event.get("kind") == "model_request_packed"
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("request_index") == request.get("request_index")
    ]
    require(len(packed) == 1, "adapter journal does not contain exactly one selected model request")
    packed_request = packed[0]
    preceding_id = packed_request.get("preceding_observation_id")
    current_id = packed_request.get("current_observation_id")
    require(
        receipt.get("preceding_observation_id") == preceding_id
        and receipt.get("current_observation_id") == current_id,
        "history receipt observation identities differ from the model request",
    )
    require(isinstance(preceding_id, str) and isinstance(current_id, str), "selected history request lacks two observations")
    observations = adapter["observations"]
    preceding = observations.get(preceding_id)
    current = observations.get(current_id)
    require(preceding is not None and current is not None, "history receipt observations are absent from journal")
    require(current["control_step"] == request.get("request_start_action_index"), "current observation action index differs from selection")
    require(preceding["control_step"] == current["control_step"] - 1, "preceding observation is not the immediately preceding recorded action")
    preceding_payload = _load_observation_payload(adapter, observations, preceding_id)
    current_payload = _load_observation_payload(adapter, observations, current_id)
    camera = request["camera_id"]
    try:
        preceding_capture = preceding_payload["clock"]["cameras"][camera]["capture_time_ns"]
        current_capture = current_payload["clock"]["cameras"][camera]["capture_time_ns"]
        preceding_physics = preceding_payload["clock"]["physics_time_s"]
        current_physics = current_payload["clock"]["physics_time_s"]
    except (KeyError, TypeError) as error:
        raise AnalysisContractError("native camera/physics timestamps are missing from history observations") from error
    require(type(preceding_capture) is int and type(current_capture) is int, "native camera timestamps are invalid")
    preceding_physics = finite_number(preceding_physics, "preceding physics time")
    current_physics = finite_number(current_physics, "current physics time")
    require(current_capture > preceding_capture, "native camera capture time did not advance")
    require(current_physics > preceding_physics, "native physics time did not advance")
    expected_interval = (current_capture - preceding_capture) / 1_000_000_000
    expected_fields = {
        "preceding_camera_capture_time_ns": preceding_capture,
        "current_camera_capture_time_ns": current_capture,
        "preceding_physics_time_s": preceding_physics,
        "current_physics_time_s": current_physics,
        "preceding_observation_interval_s": expected_interval,
    }
    for key, expected in expected_fields.items():
        observed = finite_number(receipt.get(key), f"history receipt {key}")
        require(observed == float(expected), f"history receipt {key} does not reproduce from native clocks")
    interval = finite_number(receipt.get("preceding_observation_interval_s"), "preceding observation interval")
    require(interval > 0, "preceding observation interval must be strictly positive")
    return {"source_request_id": receipt["source_request_id"], "preceding_observation_interval_s": interval}


def load_annotation_module():
    global _ANNOTATION_VALIDATOR
    if _ANNOTATION_VALIDATOR is not None:
        return _ANNOTATION_VALIDATOR
    require(ANNOTATION_MODULE.is_file(), "forecast annotation validator is missing")
    spec = importlib.util.spec_from_file_location("wmf_forecast_annotation_validator", ANNOTATION_MODULE)
    require(spec is not None and spec.loader is not None, "cannot load forecast annotation validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ANNOTATION_VALIDATOR = module
    return _ANNOTATION_VALIDATOR


def load_release_freeze_module():
    global _RELEASE_FREEZE_VALIDATOR
    if _RELEASE_FREEZE_VALIDATOR is not None:
        return _RELEASE_FREEZE_VALIDATOR
    require(RELEASE_FREEZE_MODULE.is_file(), "development-to-confirmation release validator is missing")
    spec = importlib.util.spec_from_file_location("wmf_development_release_validator", RELEASE_FREEZE_MODULE)
    require(spec is not None and spec.loader is not None, "cannot load development release validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RELEASE_FREEZE_VALIDATOR = module
    return _RELEASE_FREEZE_VALIDATOR


def load_recording_module():
    global _RECORDING_VALIDATOR
    if _RECORDING_VALIDATOR is not None:
        return _RECORDING_VALIDATOR
    require(RECORDING_MODULE.is_file(), "forecast recording validator is missing")
    spec = importlib.util.spec_from_file_location("wmf_forecast_recording_validator", RECORDING_MODULE)
    require(spec is not None and spec.loader is not None, "cannot load forecast recording validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RECORDING_VALIDATOR = module
    return _RECORDING_VALIDATOR


def validate_machine_spec(specification: Mapping[str, Any]) -> None:
    require(specification.get("spec_id") == STUDY_ID, "ablation specification identity mismatch")
    analysis = specification.get("analysis")
    require(isinstance(analysis, dict), "ablation specification lacks analysis contract")
    require(analysis.get("bootstrap_resamples") == BOOTSTRAP_RESAMPLES, "bootstrap count differs from specification")
    require(analysis.get("analysis_seed") == ANALYSIS_SEED, "analysis seed differs from specification")
    require(analysis.get("confidence_level") == CONFIDENCE_LEVEL, "confidence level differs from specification")
    require(analysis.get("models_reported_separately") is True, "specification permits forbidden pooled model reporting")
    require(analysis.get("request_sample_cap_per_episode") == 4, "request cap differs from specification")
    require(
        analysis.get("missing_condition_policy")
        == "continuous complete-layout estimate labeledconditional; full-layout weighted win bounds",
        "missing-condition policy differs from specification",
    )


def validate_release_annotation_binding(
    release_annotation: Any,
    confirmation_freeze: Mapping[str, Any],
    freeze_info: Mapping[str, Any],
) -> None:
    """Require both independently validated freezes to name identical dev evidence."""

    require(isinstance(release_annotation, Mapping), "development release freeze lacks annotation evidence")
    require(
        release_annotation.get("movement_resolution", {}).get("threshold_relative_image_diagonal")
        == confirmation_freeze["movement_resolution"]["threshold_relative_image_diagonal"],
        "annotation and development-release movement thresholds differ",
    )
    require(
        release_annotation.get("development_summary", {}).get("sha256")
        == confirmation_freeze["movement_resolution"]["development_summary_sha256"],
        "annotation and development-release label summaries differ",
    )
    require(
        release_annotation.get("rubric", {}).get("sha256") == freeze_info["rubric_sha256"],
        "annotation and development-release rubric hashes differ",
    )
    require(
        release_annotation.get("final_consensus", {}).get("sha256")
        == confirmation_freeze["development_final_consensus"]["sha256"],
        "annotation and development-release final consensus hashes differ",
    )
    require(
        release_annotation.get("usability_decision")
        == confirmation_freeze["development_validation_decision"],
        "annotation and development-release usability decisions differ",
    )


def annotation_quality_summary(
    consensus: Mapping[str, Any], *, final_consensus_sha256: str
) -> dict[str, Any]:
    """Normalize independently verified rater agreement and adjudication evidence."""

    final_sha = require_sha256(final_consensus_sha256, "final consensus sha256")
    counts = consensus.get("counts")
    require(isinstance(counts, Mapping), "final consensus lacks annotation counts")
    exact_keys(
        counts,
        {"images", "first_pass_exact_agreements", "independently_adjudicated"},
        "final consensus counts",
    )
    images = counts.get("images")
    agreements = counts.get("first_pass_exact_agreements")
    adjudicated = counts.get("independently_adjudicated")
    require(
        type(images) is int
        and type(agreements) is int
        and type(adjudicated) is int
        and images >= 0
        and agreements >= 0
        and adjudicated >= 0
        and agreements + adjudicated == images,
        "final consensus annotation counts are inconsistent",
    )
    labels = consensus.get("labels")
    require(isinstance(labels, list) and len(labels) == images, "final consensus label count differs")
    decision_rows = []
    seen_assets = set()
    for row in labels:
        require(isinstance(row, Mapping), "final consensus label is invalid")
        asset_id = row.get("restricted_asset_id")
        decision = row.get("decision_source")
        require(isinstance(asset_id, str) and asset_id, "final consensus label lacks an asset identity")
        require(asset_id not in seen_assets, "final consensus duplicates an asset identity")
        require(
            decision in {"first_pass_exact_agreement", "independent_adjudicator"},
            "final consensus label has an invalid decision source",
        )
        seen_assets.add(asset_id)
        decision_rows.append(
            {"restricted_asset_id": asset_id, "decision_source": decision}
        )
    decision_rows.sort(key=lambda row: row["restricted_asset_id"])
    decisions = Counter(row["decision_source"] for row in decision_rows)
    require(
        len(decisions) <= 2
        and decisions["first_pass_exact_agreement"] == agreements
        and decisions["independent_adjudicator"] == adjudicated,
        "final consensus decision sources differ from annotation counts",
    )
    response_hashes = consensus.get("first_pass_response_sha256_by_slot")
    require(
        isinstance(response_hashes, Mapping) and set(response_hashes) == {"rater_a", "rater_b"},
        "final consensus first-pass response hashes are incomplete",
    )
    response_hashes = {
        slot: require_sha256(response_hashes[slot], f"{slot} response sha256")
        for slot in ("rater_a", "rater_b")
    }
    rater_hashes = consensus.get("rater_code_sha256_by_slot")
    require(
        isinstance(rater_hashes, Mapping)
        and set(rater_hashes) == {"rater_a", "rater_b", "adjudicator"},
        "final consensus rater-code hashes are incomplete",
    )
    normalized_rater_hashes = {
        "rater_a": require_sha256(rater_hashes["rater_a"], "rater_a code sha256"),
        "rater_b": require_sha256(rater_hashes["rater_b"], "rater_b code sha256"),
        "adjudicator": rater_hashes["adjudicator"],
    }
    require(
        normalized_rater_hashes["rater_a"] != normalized_rater_hashes["rater_b"],
        "first-pass rater identities are not independent",
    )
    adjudicator_response_sha = consensus.get("adjudicator_response_sha256")
    if adjudicated:
        normalized_rater_hashes["adjudicator"] = require_sha256(
            normalized_rater_hashes["adjudicator"], "adjudicator code sha256"
        )
        require(
            normalized_rater_hashes["adjudicator"]
            not in {
                normalized_rater_hashes["rater_a"],
                normalized_rater_hashes["rater_b"],
            },
            "adjudicator identity is not independent",
        )
        adjudicator_response_sha = require_sha256(
            adjudicator_response_sha, "adjudicator response sha256"
        )
    else:
        require(
            normalized_rater_hashes["adjudicator"] is None
            and adjudicator_response_sha is None,
            "agreement-only consensus introduces adjudicator evidence",
        )
    denominator = float(images)
    return {
        "unit": "distinct_blinded_annotation_images",
        "images": images,
        "first_pass_exact_agreements": agreements,
        "independently_adjudicated": adjudicated,
        "first_pass_exact_agreement_rate": agreements / denominator if images else None,
        "independent_adjudication_rate": adjudicated / denominator if images else None,
        "decision_inventory_sha256": hashlib.sha256(
            canonical_bytes(decision_rows)
        ).hexdigest(),
        "final_consensus_sha256": final_sha,
        "source_restricted_map_sha256": require_sha256(
            consensus.get("source_restricted_map_sha256"),
            "consensus restricted-map sha256",
        ),
        "adjudication_map_sha256": require_sha256(
            consensus.get("adjudication_map_sha256"),
            "consensus adjudication-map sha256",
        ),
        "first_pass_response_sha256_by_slot": response_hashes,
        "rater_code_sha256_by_slot": normalized_rater_hashes,
        "adjudicator_response_sha256": adjudicator_response_sha,
    }


def validate_annotation_quality_summary(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "analysis context lacks annotation quality evidence")
    exact_keys(value, ANNOTATION_QUALITY_KEYS, "annotation quality summary")
    images = value.get("images")
    agreements = value.get("first_pass_exact_agreements")
    adjudicated = value.get("independently_adjudicated")
    require(
        type(images) is int
        and type(agreements) is int
        and type(adjudicated) is int
        and images >= 0
        and agreements >= 0
        and adjudicated >= 0
        and agreements + adjudicated == images,
        "annotation quality counts are inconsistent",
    )
    require(value.get("unit") == "distinct_blinded_annotation_images", "annotation quality unit changed")
    expected_agreement_rate = agreements / float(images) if images else None
    expected_adjudication_rate = adjudicated / float(images) if images else None
    require(
        value.get("first_pass_exact_agreement_rate") == expected_agreement_rate
        and value.get("independent_adjudication_rate") == expected_adjudication_rate,
        "annotation quality rates differ from counts",
    )
    for key in (
        "final_consensus_sha256",
        "source_restricted_map_sha256",
        "adjudication_map_sha256",
        "decision_inventory_sha256",
    ):
        require_sha256(value.get(key), f"annotation quality {key}")
    response_hashes = value.get("first_pass_response_sha256_by_slot")
    require(
        isinstance(response_hashes, Mapping) and set(response_hashes) == {"rater_a", "rater_b"},
        "annotation quality response hashes are incomplete",
    )
    for slot in ("rater_a", "rater_b"):
        require_sha256(response_hashes[slot], f"annotation quality {slot} response sha256")
    rater_hashes = value.get("rater_code_sha256_by_slot")
    require(
        isinstance(rater_hashes, Mapping)
        and set(rater_hashes) == {"rater_a", "rater_b", "adjudicator"},
        "annotation quality rater hashes are incomplete",
    )
    first_pass_codes = {
        require_sha256(rater_hashes[slot], f"annotation quality {slot} code sha256")
        for slot in ("rater_a", "rater_b")
    }
    require(len(first_pass_codes) == 2, "annotation quality first-pass raters are not independent")
    if adjudicated:
        adjudicator = require_sha256(
            rater_hashes["adjudicator"], "annotation quality adjudicator code sha256"
        )
        require(adjudicator not in first_pass_codes, "annotation quality adjudicator is not independent")
        require_sha256(
            value.get("adjudicator_response_sha256"),
            "annotation quality adjudicator response sha256",
        )
    else:
        require(
            rater_hashes["adjudicator"] is None
            and value.get("adjudicator_response_sha256") is None,
            "agreement-only annotation quality introduces adjudicator evidence",
        )
    return dict(value)


def annotation_quality_decisions(consensus: Mapping[str, Any]) -> list[dict[str, str]]:
    labels = consensus.get("labels")
    require(isinstance(labels, list), "final consensus labels are missing")
    rows = []
    for label in labels:
        require(isinstance(label, Mapping), "final consensus label is invalid")
        rows.append(
            {
                "restricted_asset_id": str(label.get("restricted_asset_id", "")),
                "decision_source": str(label.get("decision_source", "")),
            }
        )
    return sorted(rows, key=lambda row: row["restricted_asset_id"])


def validate_annotation_quality_context(
    value: Any,
    *,
    decisions: Any,
    labels: Mapping[str, Any],
    sources: Any,
    consensus_bytes: Any,
) -> dict[str, Any]:
    quality = validate_annotation_quality_summary(value)
    require(isinstance(sources, Mapping), "analysis context lacks source evidence")
    final_source = sources.get("final_consensus")
    restricted_source = sources.get("restricted_map")
    require(
        isinstance(final_source, Mapping)
        and final_source.get("sha256") == quality["final_consensus_sha256"],
        "annotation quality is detached from final consensus evidence",
    )
    require(
        isinstance(restricted_source, Mapping)
        and restricted_source.get("sha256") == quality["source_restricted_map_sha256"],
        "annotation quality is detached from restricted-map evidence",
    )
    require(
        isinstance(consensus_bytes, bytes),
        "analysis context lacks exact final-consensus source bytes",
    )
    require(
        hashlib.sha256(consensus_bytes).hexdigest() == final_source["sha256"],
        "annotation quality final-consensus bytes differ from source evidence",
    )
    try:
        consensus = json.loads(consensus_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisContractError("annotation quality final-consensus bytes are invalid JSON") from error
    require(isinstance(consensus, dict), "annotation quality final consensus is not an object")
    load_annotation_module().verify_signed(consensus, "annotation quality final consensus")
    expected_quality = annotation_quality_summary(
        consensus,
        final_consensus_sha256=final_source["sha256"],
    )
    require(quality == expected_quality, "annotation quality differs from final consensus")
    require(isinstance(decisions, list), "analysis context lacks annotation decision evidence")
    observed_decisions: list[dict[str, str]] = []
    seen_assets = set()
    for row in decisions:
        require(isinstance(row, Mapping), "annotation decision row is invalid")
        exact_keys(row, {"restricted_asset_id", "decision_source"}, "annotation decision row")
        asset_id = row.get("restricted_asset_id")
        decision = row.get("decision_source")
        require(isinstance(asset_id, str) and asset_id, "annotation decision lacks an asset identity")
        require(asset_id not in seen_assets, "annotation decision inventory duplicates an asset")
        require(
            decision in {"first_pass_exact_agreement", "independent_adjudicator"},
            "annotation decision source is invalid",
        )
        seen_assets.add(asset_id)
        observed_decisions.append(
            {"restricted_asset_id": asset_id, "decision_source": decision}
        )
    observed_decisions.sort(key=lambda row: row["restricted_asset_id"])
    require(
        observed_decisions == annotation_quality_decisions(consensus),
        "annotation decision inventory differs from final consensus",
    )
    require(
        hashlib.sha256(canonical_bytes(observed_decisions)).hexdigest()
        == quality["decision_inventory_sha256"],
        "annotation decision inventory differs from its commitment",
    )
    decision_counts = Counter(row["decision_source"] for row in observed_decisions)
    require(
        len(observed_decisions) == quality["images"]
        and decision_counts["first_pass_exact_agreement"]
        == quality["first_pass_exact_agreements"]
        and decision_counts["independent_adjudicator"]
        == quality["independently_adjudicated"],
        "annotation decision inventory differs from quality counts",
    )
    label_asset_ids = set()
    for request_id, role_rows in labels.items():
        require(isinstance(role_rows, Mapping), f"label roles are invalid for {request_id}")
        for role, label in role_rows.items():
            require(isinstance(label, Mapping), f"label {request_id}:{role} is invalid")
            asset_id = label.get("restricted_asset_id")
            require(
                isinstance(asset_id, str) and asset_id,
                f"label {request_id}:{role} lacks a restricted asset identity",
            )
            label_asset_ids.add(asset_id)
    require(
        label_asset_ids == seen_assets,
        "annotation quality assets differ from validated request labels",
    )
    return quality


def _artifact_ref_rows(
    manifest_path: Path,
    references: Any,
    label: str,
) -> list[tuple[Path, dict[str, Any]]]:
    require(isinstance(references, list), f"{label} must be a list")
    result = []
    for index, reference in enumerate(references):
        path = resolve_reference(manifest_path.parent, reference, f"{label}[{index}]")
        result.append((path, load_json(path)))
    return result


def load_evidence(manifest_path: Path) -> dict[str, Any]:
    """Validate every source chain and return normalized evidence.

    No numerical forecast result is returned unless this gate succeeds.
    """

    require(not manifest_path.is_symlink(), "analysis evidence manifest must not be a symlink")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as error:
        raise AnalysisContractError(f"analysis evidence manifest is missing: {manifest_path}") from error
    require(manifest_path.is_file(), "analysis evidence manifest must be a file")
    manifest_sha256 = sha256_file(manifest_path)
    manifest = load_json(manifest_path)
    exact_keys(manifest, EVIDENCE_KEYS, "analysis evidence manifest")
    require(manifest.get("schema_version") == EVIDENCE_SCHEMA, "analysis evidence manifest schema mismatch")
    require(manifest.get("study_id") == STUDY_ID, "analysis evidence manifest study mismatch")
    require(manifest.get("stage") == "confirmation", "forecast accuracy is confirmation-only")
    branch = manifest.get("cohort_branch")
    require(branch in MODEL_BRANCHES, "analysis evidence cohort branch is invalid")
    verify_signed(manifest, "analysis evidence manifest")
    sources = manifest.get("sources")
    exact_keys(sources, SOURCE_KEYS, "analysis evidence sources")
    paths = {
        key: resolve_reference(manifest_path.parent, sources[key], f"analysis source {key}")
        for key in sorted(SOURCE_KEYS)
    }
    specification = load_json(paths["ablation_spec"])
    validate_machine_spec(specification)

    annotation = load_annotation_module()
    selection = load_json(paths["request_selection"])
    selected = annotation._selected_requests(selection)
    require(selection.get("stage") == "confirmation", "selection is not a confirmation selection")
    require(selection.get("cohort_branch") == branch, "selection cohort branch differs from evidence manifest")

    freeze = load_json(paths["annotation_freeze"])
    freeze_info = annotation.validate_freeze(
        freeze,
        freeze_path=paths["annotation_freeze"],
        stage="confirmation",
    )
    require(freeze_info["cohort_branch"] == branch, "annotation freeze cohort branch differs from evidence manifest")

    release_validator = load_release_freeze_module()
    release_freeze = release_validator.validate_release_freeze(
        paths["development_release_freeze"],
        sources["development_release_freeze"]["sha256"],
    )
    require(release_freeze.get("cohort_branch") == branch, "development release freeze cohort branch differs")
    require(
        release_freeze.get("qualified_model_ids") == list(MODEL_BRANCHES[branch]),
        "development release freeze qualified models differ",
    )
    selection_alignments = {
        contract["model_id"]: contract
        for contract in selection["alignment_contracts"]
    }
    release_alignments = release_freeze.get("alignment_contracts_by_model")
    require(isinstance(release_alignments, Mapping), "development release freeze lacks alignment contracts")
    for model in MODEL_BRANCHES[branch]:
        require(
            release_alignments[model]["contract_sha256"]
            == selection_alignments[model]["contract_sha256"],
            f"{model} selection alignment differs from the development release freeze",
        )
    validate_release_annotation_binding(release_freeze.get("annotation"), freeze, freeze_info)

    mapping = load_json(paths["restricted_map"])
    exact_keys(mapping, annotation.RESTRICTED_MAP_KEYS, "confirmation restricted mapping")
    annotation.verify_signed(mapping, "confirmation restricted mapping")
    require(mapping.get("stage") == "confirmation", "restricted mapping is not confirmation evidence")
    require(mapping.get("cohort_branch") == branch, "restricted mapping cohort branch differs")
    require(
        mapping.get("selection_manifest_sha256") == sha256_file(paths["request_selection"]),
        "restricted mapping does not bind this request selection",
    )
    require(
        mapping.get("freeze_sha256") == sha256_file(paths["annotation_freeze"]),
        "restricted mapping does not bind this annotation freeze",
    )
    require(
        mapping.get("qualified_model_ids") == list(MODEL_BRANCHES[branch]),
        "restricted mapping qualified models differ from branch",
    )

    consensus = load_json(paths["final_consensus"])
    require(consensus.get("schema_version") == annotation.FINAL_CONSENSUS_SCHEMA, "final consensus schema mismatch")
    require(consensus.get("stage") == "confirmation", "final consensus is not confirmation evidence")
    annotation.verify_signed(consensus, "confirmation final consensus")
    adjudication_path_value = consensus.get("adjudication_map_path")
    require(isinstance(adjudication_path_value, str) and adjudication_path_value, "consensus lacks adjudication map path")
    adjudication_path = Path(adjudication_path_value)
    if not adjudication_path.is_absolute():
        adjudication_path = paths["final_consensus"].parent / adjudication_path
    require(not adjudication_path.is_symlink(), "final consensus adjudication map must not be a symlink")
    try:
        adjudication_path = adjudication_path.resolve(strict=True)
    except OSError as error:
        raise AnalysisContractError("final consensus adjudication map is missing") from error
    require(adjudication_path.is_file(), "final consensus adjudication map is missing")
    require(
        sha256_file(adjudication_path) == consensus.get("adjudication_map_sha256"),
        "final consensus adjudication map hash mismatch",
    )
    adjudicator_path_value = consensus.get("adjudicator_response_path")
    adjudicator_path = None
    if adjudicator_path_value is not None:
        require(isinstance(adjudicator_path_value, str) and adjudicator_path_value, "adjudicator response path is invalid")
        adjudicator_path = Path(adjudicator_path_value)
        if not adjudicator_path.is_absolute():
            adjudicator_path = paths["final_consensus"].parent / adjudicator_path
        require(not adjudicator_path.is_symlink(), "adjudicator response must not be a symlink")
        try:
            adjudicator_path = adjudicator_path.resolve(strict=True)
        except OSError as error:
            raise AnalysisContractError("adjudicator response is missing") from error
    reproduced = annotation.merge_adjudication(
        adjudication_map_path=adjudication_path,
        adjudicator_response_path=adjudicator_path,
    )
    require(reproduced == consensus, "final consensus does not reproduce from locked blind human responses")
    require(
        consensus.get("source_restricted_map_sha256") == sha256_file(paths["restricted_map"]),
        "final consensus does not bind the supplied restricted map",
    )

    labels_by_asset = {row["restricted_asset_id"]: row["annotation"] for row in consensus["labels"]}
    require(len(labels_by_asset) == len(consensus["labels"]), "final consensus duplicates an asset")
    mapped_ids = {asset["restricted_asset_id"] for asset in mapping.get("assets", [])}
    require(set(labels_by_asset) == mapped_ids, "final consensus does not cover the restricted map exactly")
    label_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for asset in mapping["assets"]:
        annotation_value = labels_by_asset[asset["restricted_asset_id"]]
        for source in asset["source_records"]:
            request_id = source["source_request_id"]
            role = source["image_role"]
            require(request_id in selected, "restricted mapping contains an unselected request")
            require(role not in label_records[request_id], f"duplicate {role} label for {request_id}")
            label_records[request_id][role] = {
                "annotation": annotation_value,
                "width_px": asset["width_px"],
                "height_px": asset["height_px"],
                "restricted_asset_id": asset["restricted_asset_id"],
            }
    require(set(label_records) == set(selected), "human consensus omits a selected request")

    roster = {row["cell_id"]: row for row in selection["episode_roster"]}
    endpoints = {}
    endpoint_evidence = {}
    for path, receipt in _artifact_ref_rows(
        manifest_path,
        manifest.get("endpoint_trace_receipts"),
        "endpoint_trace_receipts",
    ):
        cell_id = receipt.get("cell_id")
        require(cell_id in roster, f"endpoint receipt references unknown cell: {cell_id}")
        require(cell_id not in endpoints, f"duplicate endpoint receipt for {cell_id}")
        endpoints[cell_id] = validate_endpoint_receipt(
            receipt,
            receipt_path=path,
            roster=roster[cell_id],
        )
        adapter = endpoints[cell_id]["adapter_source"]
        endpoint_evidence[cell_id] = {
            "endpoint_receipt": {"path": str(path), "sha256": sha256_file(path)},
            "native_completion": {
                "path": str(adapter["completion_path"]),
                "sha256": adapter["completion_sha256"],
            },
            "native_journal": {
                "path": str(adapter["journal_path"]),
                "sha256": adapter["journal_sha256"],
            },
        }
    expected_endpoints = {
        cell_id
        for cell_id, row in roster.items()
        if row["recording_status"] in {"valid_complete", "valid_censored"}
    }
    require(set(endpoints) == expected_endpoints, "endpoint receipts must cover every valid complete/censored recording exactly")

    histories = {}
    history_evidence = {}
    for path, receipt in _artifact_ref_rows(
        manifest_path,
        manifest.get("request_history_receipts"),
        "request_history_receipts",
    ):
        request_id = receipt.get("source_request_id")
        require(request_id in selected, f"history receipt references unknown selected request: {request_id}")
        require(request_id not in histories, f"duplicate request history receipt for {request_id}")
        cell_id = selected[request_id]["cell_id"]
        require(cell_id in endpoints, "selected request has no validated endpoint/adapter evidence")
        histories[request_id] = validate_history_receipt(
            receipt,
            receipt_path=path,
            request=selected[request_id],
            adapter=endpoints[cell_id]["adapter_source"],
        )
        history_evidence[request_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    expected_histories = {
        request_id
        for request_id, request in selected.items()
        if request["history_mode"] == "preceding_observation"
    }
    require(set(histories) == expected_histories, "history receipts must cover every selected real-preceding-observation request exactly")

    # Close the evidence gate only after rehashing each unique native source.
    # The in-memory journal events and endpoint payloads above therefore cannot
    # outlive a changed completion/journal without this final check failing.
    for cell_id, endpoint in endpoints.items():
        adapter = endpoint["adapter_source"]
        require(
            sha256_file(adapter["completion_path"]) == adapter["completion_sha256"],
            f"native completion changed during analysis for {cell_id}",
        )
        require(
            sha256_file(adapter["journal_path"]) == adapter["journal_sha256"],
            f"native journal changed during analysis for {cell_id}",
        )
    for key, path in paths.items():
        require(
            sha256_file(path) == sources[key]["sha256"],
            f"analysis source {key} changed during validation",
        )
    require(
        sha256_file(adjudication_path) == consensus["adjudication_map_sha256"],
        "final consensus adjudication map changed during validation",
    )
    if adjudicator_path is not None:
        require(
            sha256_file(adjudicator_path) == consensus["adjudicator_response_sha256"],
            "adjudicator response changed during validation",
        )
    require(sha256_file(manifest_path) == manifest_sha256, "analysis evidence manifest changed during validation")

    try:
        annotation_quality_consensus_bytes = paths["final_consensus"].read_bytes()
    except OSError as error:
        raise AnalysisContractError("final consensus became unreadable") from error
    require(
        hashlib.sha256(annotation_quality_consensus_bytes).hexdigest()
        == sources["final_consensus"]["sha256"],
        "final consensus changed before annotation-quality handoff",
    )
    annotation_quality = annotation_quality_summary(
        consensus,
        final_consensus_sha256=sources["final_consensus"]["sha256"],
    )
    return {
        "evidence_gate": {
            "recording_and_action_chain": "VALIDATED",
            "development_release_freeze": "VALIDATED",
            "confirmation_annotation_freeze": "VALIDATED",
            "blind_human_consensus": "REPRODUCED",
            "technical_statuses_preserved": "VALIDATED",
        },
        "branch": branch,
        "selection": selection,
        "selected": selected,
        "labels": dict(label_records),
        "annotation_quality": annotation_quality,
        "annotation_quality_decisions": annotation_quality_decisions(consensus),
        "annotation_quality_consensus_bytes": annotation_quality_consensus_bytes,
        "endpoints": endpoints,
        "histories": histories,
        "movement_threshold": freeze["movement_resolution"]["threshold_relative_image_diagonal"],
        "model_specification": specification["models"],
        "frozen_resource_budget": release_freeze["resource_budget"],
        "sources": {
            key: {"path": str(path), "sha256": sources[key]["sha256"]}
            for key, path in paths.items()
        },
        "receipt_evidence": {
            "endpoint_trace_receipts_by_cell": endpoint_evidence,
            "request_history_receipts_by_request": history_evidence,
        },
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
    }


def vector_from_label(record: Mapping[str, Any]) -> dict[str, tuple[float, float]] | None:
    annotation = record["annotation"]
    if annotation["cube_resolvability"] != "resolvable" or annotation["bowl_resolvability"] != "resolvable":
        return None
    diagonal = math.hypot(record["width_px"], record["height_px"])
    require(diagonal > 0, "annotation image diagonal is zero")
    cube = tuple(float(value) / diagonal for value in annotation["cube_center_px"])
    bowl = tuple(float(value) / diagonal for value in annotation["bowl_center_px"])
    relative = (cube[0] - bowl[0], cube[1] - bowl[1])
    return {"cube": cube, "bowl": bowl, "relative": relative}


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.dist(tuple(left), tuple(right))


def request_metrics(
    request: Mapping[str, Any],
    labels: Mapping[str, Mapping[str, Any]],
    history: Mapping[str, Any] | None,
    movement_threshold: float,
) -> dict[str, Any]:
    vectors = {role: vector_from_label(record) for role, record in labels.items()}
    missing_roles = sorted(role for role in ("current", "predicted", "executed") if vectors.get(role) is None)
    result: dict[str, Any] = {
        "source_request_id": request["source_request_id"],
        "cell_id": request["cell_id"],
        "model_id": request["model_id"],
        "layout_pair_id": request["layout_pair_id"],
        "condition_id": request["condition_id"],
        "primary_observable": not missing_roles,
        "missing_primary_roles": missing_roles,
        "cv_observable": False,
        "early_observable": False,
        "ambiguity_codes_by_role": {
            role: list(record["annotation"].get("ambiguity_codes", []))
            for role, record in labels.items()
        },
    }
    if missing_roles:
        return result
    current = vectors["current"]
    predicted = vectors["predicted"]
    executed = vectors["executed"]
    assert current is not None and predicted is not None and executed is not None
    forecast_error = distance(executed["relative"], predicted["relative"])
    persistence_error = distance(executed["relative"], current["relative"])
    result.update(
        {
            "forecast_error": forecast_error,
            "persistence_error": persistence_error,
            "skill": persistence_error - forecast_error,
            "strict_forecast_win": forecast_error < persistence_error,
            "actual_relative_motion": persistence_error,
            "predicted_relative_motion": distance(predicted["relative"], current["relative"]),
            "actual_cube_motion": distance(executed["cube"], current["cube"]),
            "actual_bowl_motion": distance(executed["bowl"], current["bowl"]),
            "predicted_cube_motion": distance(predicted["cube"], current["cube"]),
            "predicted_bowl_motion": distance(predicted["bowl"], current["bowl"]),
            "cube_forecast_error": distance(executed["cube"], predicted["cube"]),
            "bowl_forecast_error": distance(executed["bowl"], predicted["bowl"]),
            "movement_stratum": "moving" if persistence_error > movement_threshold else "stationary",
        }
    )
    if request["history_mode"] == "persistence_at_initial_request":
        cv_relative = current["relative"]
        result["cv_observable"] = True
    else:
        preceding = vectors.get("preceding")
        if preceding is not None:
            require(history is not None, "preceding label lacks its hash-bound timing receipt")
            elapsed = history["preceding_observation_interval_s"]
            horizon = finite_number(request["target_physical_time_s"], "request physical horizon")
            cv_relative = tuple(
                current["relative"][axis]
                + horizon * (current["relative"][axis] - preceding["relative"][axis]) / elapsed
                for axis in range(2)
            )
            result["cv_observable"] = True
    if result["cv_observable"]:
        result["constant_velocity_error"] = distance(executed["relative"], cv_relative)
        result["skill_vs_constant_velocity"] = result["constant_velocity_error"] - forecast_error
    if request.get("early_horizon_supported"):
        early_predicted = vectors.get("early_predicted")
        early_executed = vectors.get("early_executed")
        if early_predicted is not None and early_executed is not None:
            early_forecast_error = distance(early_executed["relative"], early_predicted["relative"])
            early_persistence_error = distance(early_executed["relative"], current["relative"])
            early_skill = early_persistence_error - early_forecast_error
            result.update(
                {
                    "early_observable": True,
                    "early_forecast_error": early_forecast_error,
                    "early_persistence_error": early_persistence_error,
                    "early_skill": early_skill,
                    # This copy is deliberately created only when the same
                    # request has observable primary and earlier targets.  It
                    # prevents a later aggregation from subtracting marginal
                    # means drawn from different request populations.
                    "primary_skill_paired_with_early": result["skill"],
                    "primary_minus_early_skill": result["skill"] - early_skill,
                }
            )
    return result


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def percentile(values: Sequence[float], probability: float) -> float:
    require(values, "cannot compute a percentile from no values")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_layout_mean(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "ci95": None, "layout_pairs": 0}
    generator = random.Random(ANALYSIS_SEED)
    count = len(values)
    draws = sorted(
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "estimate": statistics.fmean(values),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "layout_pairs": count,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": ANALYSIS_SEED,
        "resampling_unit": "complete independent base-layout pair with four condition means nested",
    }


def _cell_summary(
    roster_row: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observable = [row for row in metrics if row["primary_observable"]]
    selected_count = len(metrics)
    wins = sum(bool(row.get("strict_forecast_win")) for row in observable)
    unresolved = selected_count - len(observable)
    if selected_count:
        win_bounds = [wins / selected_count, (wins + unresolved) / selected_count]
    else:
        win_bounds = [0.0, 1.0]
    means = {}
    for key in (
        "forecast_error",
        "persistence_error",
        "skill",
        "constant_velocity_error",
        "skill_vs_constant_velocity",
        "actual_relative_motion",
        "predicted_relative_motion",
        "actual_cube_motion",
        "actual_bowl_motion",
        "predicted_cube_motion",
        "predicted_bowl_motion",
        "cube_forecast_error",
        "bowl_forecast_error",
        "early_forecast_error",
        "early_persistence_error",
        "early_skill",
        "primary_skill_paired_with_early",
        "primary_minus_early_skill",
    ):
        means[key] = mean_or_none(row[key] for row in metrics if key in row)
    return {
        "cell_id": roster_row["cell_id"],
        "model_id": roster_row["model_id"],
        "layout_pair_id": roster_row["layout_pair_id"],
        "condition_id": roster_row["condition_id"],
        "recording_status": roster_row["recording_status"],
        "selected_requests": selected_count,
        "primary_observable_requests": len(observable),
        "unresolved_selected_requests": unresolved,
        "strict_forecast_wins": wins,
        "full_design_strict_win_bounds": win_bounds,
        "means": means,
    }


def _complete_layout_values(
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    model: str,
    metric: str,
    request_stratum: str | None = None,
    request_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if request_stratum is None:
        lookup = cells
    else:
        require(request_rows is not None, "stratified layout analysis lacks request rows")
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in request_rows:
            if row["model_id"] == model and row.get("movement_stratum") == request_stratum and metric in row:
                grouped[(model, row["layout_pair_id"], row["condition_id"])].append(row[metric])
        lookup = {
            key: {"means": {metric: statistics.fmean(values)}}
            for key, values in grouped.items()
        }
    layouts = [f"C{index:02d}" for index in range(1, 25)]
    complete, missing = [], []
    for layout in layouts:
        values = []
        for condition in CONDITIONS:
            cell = lookup.get((model, layout, condition))
            value = cell.get("means", {}).get(metric) if cell else None
            if value is None:
                break
            values.append(value)
        if len(values) == len(CONDITIONS):
            complete.append({"layout_pair_id": layout, "value": statistics.fmean(values)})
        else:
            missing.append(layout)
    return complete, missing


def _metric_report(
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    model: str,
    metric: str,
) -> dict[str, Any]:
    complete, missing = _complete_layout_values(cells, model=model, metric=metric)
    result = bootstrap_layout_mean([row["value"] for row in complete])
    result["layout_pair_values"] = complete
    result["complete_layout_ids"] = [row["layout_pair_id"] for row in complete]
    result["excluded_incomplete_layout_ids"] = missing
    result["conditional_on_four_condition_observability"] = True
    return result


def reflection_report(
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    model: str,
) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    per_layout_command: dict[tuple[str, str], dict[str, float]] = {}
    for command in ("left", "right"):
        records = []
        for layout_index in range(1, 25):
            layout = f"C{layout_index:02d}"
            original = cells.get((model, layout, f"original_{command}"))
            reflected = cells.get((model, layout, f"reflected_{command}"))
            if not original or not reflected:
                continue
            values = {}
            for metric in ("actual_relative_motion", "predicted_relative_motion"):
                left = original["means"].get(metric)
                right = reflected["means"].get(metric)
                if left is None or right is None:
                    break
                values[metric] = right - left
            if len(values) != 2:
                continue
            values["discrepancy"] = values["predicted_relative_motion"] - values["actual_relative_motion"]
            per_layout_command[(layout, command)] = values
            records.append({"layout_pair_id": layout, **values})
        commands[command] = {
            "actual_reflected_minus_original_motion": bootstrap_layout_mean(
                [row["actual_relative_motion"] for row in records]
            ),
            "predicted_reflected_minus_original_motion": bootstrap_layout_mean(
                [row["predicted_relative_motion"] for row in records]
            ),
            "predicted_minus_actual_contrast_discrepancy": bootstrap_layout_mean(
                [row["discrepancy"] for row in records]
            ),
            "eligible_layout_ids": [row["layout_pair_id"] for row in records],
            "paired_layout_values": records,
        }
    combined = []
    for layout_index in range(1, 25):
        layout = f"C{layout_index:02d}"
        rows = [per_layout_command.get((layout, command)) for command in ("left", "right")]
        if all(rows):
            combined.append(
                {
                    "layout_pair_id": layout,
                    "discrepancy": statistics.fmean(
                        row["discrepancy"] for row in rows if row is not None
                    ),
                }
            )
    return {
        "definition": "For each command and layout, reflected-minus-original qhat minus reflected-minus-original q; either sign is retained.",
        "by_command": commands,
        "combined_command_discrepancy": {
            **bootstrap_layout_mean([row["discrepancy"] for row in combined]),
            "paired_layout_values": combined,
        },
    }


def stopping_report(
    roster: Mapping[str, Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]],
    model: str,
) -> dict[str, Any]:
    cell_offsets = {}
    for cell_id, endpoint in endpoints.items():
        row = roster[cell_id]
        if row["model_id"] != model or endpoint["action_450"] is None or endpoint["first_success_or_action_450"] is None:
            continue
        values = {}
        for name, key in (("action_450", "action_450"), ("first_success", "first_success_or_action_450")):
            point = endpoint[key]
            values[name] = point["cube_robot_xyz"][1] - point["bowl_robot_xyz"][1]
        cell_offsets[(row["layout_pair_id"], row["condition_id"])] = values
    arms = {}
    per_layout_arm: dict[tuple[str, str], dict[str, float]] = {}
    for arm in ("original", "reflected"):
        records = []
        for layout_index in range(1, 25):
            layout = f"C{layout_index:02d}"
            left = cell_offsets.get((layout, f"{arm}_left"))
            right = cell_offsets.get((layout, f"{arm}_right"))
            if left is None or right is None:
                continue
            common = left["action_450"] - right["action_450"]
            first = left["first_success"] - right["first_success"]
            records.append({"layout_pair_id": layout, "action_450": common, "first_success": first, "first_minus_common": first - common})
            per_layout_arm[(layout, arm)] = {
                key: records[-1][key]
                for key in ("action_450", "first_success", "first_minus_common")
            }
        arms[arm] = {
            key: bootstrap_layout_mean([row[key] for row in records])
            for key in ("action_450", "first_success", "first_minus_common")
        }
        arms[arm]["eligible_layout_ids"] = [row["layout_pair_id"] for row in records]
        arms[arm]["paired_layout_values"] = records
    pooled_by_layout: dict[str, list[float]] = {
        "action_450": [],
        "first_success": [],
        "first_minus_common": [],
    }
    for layout_index in range(1, 25):
        layout = f"C{layout_index:02d}"
        values = [per_layout_arm.get((layout, arm)) for arm in ("original", "reflected")]
        if all(values):
            for key in pooled_by_layout:
                pooled_by_layout[key].append(
                    statistics.fmean(value[key] for value in values if value is not None)
                )
    censored = sum(
        row["model_id"] == model and row["recording_status"] == "valid_censored"
        for row in roster.values()
    )
    return {
        "coordinate_definition": "robot-frame (cube_y - bowl_y); instruction response is LEFT minus RIGHT within arm/layout",
        "arms": arms,
        "pooled_arm_descriptive": {
            key: bootstrap_layout_mean(values) for key, values in pooled_by_layout.items()
        },
        "censored_cells_excluded_without_carry_forward": censored,
    }


def _coverage_report(
    selection: Mapping[str, Any],
    roster: Mapping[str, Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    model: str,
) -> dict[str, Any]:
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in request_rows:
        if row["model_id"] == model:
            by_cell[row["cell_id"]].append(row)
    episode_inventory = {
        row["cell_id"]: row
        for row in selection.get("episodes", [])
    }
    require(
        set(episode_inventory) == set(roster),
        "selection episode coverage differs from its scientific roster",
    )
    request_inventory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for wrapper in selection.get("requests", []):
        source = wrapper.get("source", {})
        if source.get("model_id") == model:
            request_inventory[source.get("condition_id")].append(wrapper)
    condition_reports = {}
    for condition in CONDITIONS:
        rows = [row for row in roster.values() if row["model_id"] == model and row["condition_id"] == condition]
        requests = [item for row in rows for item in by_cell.get(row["cell_id"], [])]
        inventory = request_inventory.get(condition, [])
        episode_rows = [episode_inventory[row["cell_id"]] for row in rows]
        ambiguity = Counter(
            f"{role}:{code}"
            for request in requests
            for role, codes in request["ambiguity_codes_by_role"].items()
            for code in codes
            if code != "none"
        )
        condition_reports[condition] = {
            "planned_cells": len(rows),
            "recording_status_counts": {
                status: sum(row["recording_status"] == status for row in rows)
                for status in STATUS_VALUES
            },
            "censor_reason_counts": dict(
                sorted(Counter(row["censor_reason"] for row in rows if row.get("censor_reason")).items())
            ),
            "request_inventory_count": len(inventory),
            "timing_camera_action_eligible_requests": sum(
                wrapper.get("timing_camera_action_eligible") is True for wrapper in inventory
            ),
            "timing_camera_action_ineligibility_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for wrapper in inventory
                        for reason in wrapper.get("eligibility_reasons", [])
                    ).items()
                )
            ),
            "zero_eligible_episodes": sum(row.get("zero_eligible") is True for row in episode_rows),
            "selected_requests": len(requests),
            "primary_observable_requests": sum(row["primary_observable"] for row in requests),
            "constant_velocity_observable_requests": sum(row["cv_observable"] for row in requests),
            "early_horizon_observable_requests": sum(row["early_observable"] for row in requests),
            "unresolved_primary_role_counts": dict(
                sorted(Counter(role for row in requests for role in row["missing_primary_roles"]).items())
            ),
            "unresolved_generated_prediction_requests": sum(
                "predicted" in row["missing_primary_roles"] for row in requests
            ),
            "nontrivial_ambiguity_code_counts_by_role": dict(sorted(ambiguity.items())),
        }
    return condition_reports


def _win_bounds(cells: Mapping[tuple[str, str, str], Mapping[str, Any]], model: str) -> dict[str, Any]:
    layout_bounds = []
    for layout_index in range(1, 25):
        layout = f"C{layout_index:02d}"
        condition_bounds = [cells[(model, layout, condition)]["full_design_strict_win_bounds"] for condition in CONDITIONS]
        layout_bounds.append(
            {
                "layout_pair_id": layout,
                "lower": statistics.fmean(bound[0] for bound in condition_bounds),
                "upper": statistics.fmean(bound[1] for bound in condition_bounds),
            }
        )
    by_condition = {}
    for condition in CONDITIONS:
        bounds = [
            cells[(model, f"C{layout_index:02d}", condition)]["full_design_strict_win_bounds"]
            for layout_index in range(1, 25)
        ]
        by_condition[condition] = {
            "lower": statistics.fmean(bound[0] for bound in bounds),
            "upper": statistics.fmean(bound[1] for bound in bounds),
            "planned_cells": len(bounds),
        }
    return {
        "lower": statistics.fmean(row["lower"] for row in layout_bounds),
        "upper": statistics.fmean(row["upper"] for row in layout_bounds),
        "strict_win_definition": "forecast_error < persistence_error; observed ties are non-wins",
        "missingness_rule": "Each unresolved selected comparison is first a non-win then a win; each zero-eligible, invalid, censored-with-no-selected, or unrun episode contributes [0,1].",
        "weighting": "requests within episode, then four conditions, then 24 independent layouts",
        "by_condition": by_condition,
        "layout_bounds": layout_bounds,
    }


def _stratum_report(
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    model: str,
    stratum: str,
) -> dict[str, Any]:
    complete, missing = _complete_layout_values(
        cells,
        model=model,
        metric="skill",
        request_stratum=stratum,
        request_rows=request_rows,
    )
    result = bootstrap_layout_mean([row["value"] for row in complete])
    result.update(
        {
            "observable_requests": sum(
                row["model_id"] == model and row.get("movement_stratum") == stratum
                for row in request_rows
            ),
            "complete_layout_ids": [row["layout_pair_id"] for row in complete],
            "layout_pair_values": complete,
            "excluded_incomplete_layout_ids": missing,
            "conditional_on_all_four_conditions_having_this_stratum": True,
        }
    )
    return result


def _example_videos(roster: Mapping[str, Mapping[str, Any]], model: str) -> dict[str, Any]:
    selected = {}
    for condition in CONDITIONS:
        candidates = [
            row
            for row in roster.values()
            if row["model_id"] == model and row["condition_id"] == condition and row["recording_status"] == "valid_complete"
        ]
        if not candidates:
            selected[condition] = None
            continue
        def rank(row: Mapping[str, Any]) -> str:
            payload = f"{ANALYSIS_SEED}\0{model}\0{condition}\0{row['cell_id']}".encode()
            return hashlib.sha256(payload).hexdigest()
        winner = min(candidates, key=lambda row: (rank(row), row["cell_id"]))
        selected[condition] = {
            "cell_id": winner["cell_id"],
            "source_video_id": winner["source_video_id"],
            "source_video_sha256": winner["source_video_sha256"],
            "recording_receipt_sha256": winner["recording_receipt_sha256"],
            "selection_rank_sha256": rank(winner),
        }
    return {
        "declared_rule": "For each model and condition, among valid-complete confirmation recordings choose the minimum SHA256(UTF8(analysis_seed) || NUL || model || NUL || condition || NUL || cell_id), tie-breaking by cell_id. No label, error, success, or visual outcome enters the rule.",
        "videos": selected,
    }


def build_report(context: Mapping[str, Any]) -> dict[str, Any]:
    require(
        context.get("evidence_gate")
        == {
            "recording_and_action_chain": "VALIDATED",
            "development_release_freeze": "VALIDATED",
            "confirmation_annotation_freeze": "VALIDATED",
            "blind_human_consensus": "REPRODUCED",
            "technical_statuses_preserved": "VALIDATED",
        },
        "forecast analysis refuses to run without validated hash-bound recordings, the development release freeze, the confirmation annotation freeze, and reproduced human consensus",
    )
    branch = context["branch"]
    require(branch in MODEL_BRANCHES, "analysis branch is invalid")
    threshold = finite_number(context["movement_threshold"], "movement threshold")
    require(threshold >= 0, "movement threshold must be nonnegative")
    selection = context["selection"]
    model_specification = context.get("model_specification")
    require(isinstance(model_specification, Mapping), "analysis context lacks the hash-bound model specification")
    alignments = {
        row["model_id"]: row
        for row in selection.get("alignment_contracts", [])
    }
    roster = {row["cell_id"]: row for row in selection["episode_roster"]}
    expected = {
        cell_id: values
        for cell_id, values in load_annotation_module()._planned_cells("confirmation", branch).items()
    }
    require(set(roster) == set(expected), "analysis roster is not the complete planned branch")
    selected = context["selected"]
    labels = context["labels"]
    require(set(labels) == set(selected), "validated human consensus must cover every selected request exactly")
    annotation_quality = validate_annotation_quality_context(
        context.get("annotation_quality"),
        decisions=context.get("annotation_quality_decisions"),
        labels=labels,
        sources=context.get("sources"),
        consensus_bytes=context.get("annotation_quality_consensus_bytes"),
    )
    histories = context["histories"]
    request_rows = [
        request_metrics(request, labels[request_id], histories.get(request_id), threshold)
        for request_id, request in sorted(selected.items())
    ]
    requests_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        requests_by_cell[row["cell_id"]].append(row)
    cell_summaries = {
        (row["model_id"], row["layout_pair_id"], row["condition_id"]): _cell_summary(
            row,
            requests_by_cell.get(row["cell_id"], []),
        )
        for row in roster.values()
    }
    qualified_models = MODEL_BRANCHES[branch]
    models = {}
    sample_table = []
    for model in MODELS:
        if model not in qualified_models:
            configuration = model_specification.get(model, {})
            sample_table.append(
                {
                    "model_id": model,
                    "branch_status": "unqualified_branch_not_run",
                    "model_configuration": dict(configuration),
                    "checkpoint_revision": configuration.get("checkpoint_revision"),
                    "executed_prefix_cap": configuration.get("executed_prefix_cap"),
                    "primary_horizon_s": None,
                    "generated_frame_index": None,
                    "target_executed_action_offset": None,
                    "camera_id": None,
                    "timestamp_tolerance_s": None,
                    "full_design_planned_confirmation_cells": 96,
                    "scientific_roster_cells": 0,
                    "valid_complete": 0,
                    "valid_censored": 0,
                    "technical_invalid": 0,
                    "unrun_due_unqualified_branch": 96,
                    "selected_requests": 0,
                    "primary_observable_requests": 0,
                    "continuous_complete_layout_pairs": 0,
                }
            )
            continue
        require(model in alignments, f"analysis selection lacks qualified {model} alignment contract")
        configuration = model_specification.get(model)
        require(isinstance(configuration, Mapping), f"analysis specification lacks {model} configuration")
        alignment = alignments[model]
        model_roster = {cell_id: row for cell_id, row in roster.items() if row["model_id"] == model}
        status_counts = Counter(row["recording_status"] for row in model_roster.values())
        primary = _metric_report(cell_summaries, model=model, metric="skill")
        baseline = {
            "forecast_error": _metric_report(cell_summaries, model=model, metric="forecast_error"),
            "persistence_error": _metric_report(cell_summaries, model=model, metric="persistence_error"),
            "constant_velocity_error": _metric_report(cell_summaries, model=model, metric="constant_velocity_error"),
            "forecast_skill_vs_persistence": primary,
            "forecast_skill_vs_constant_velocity": _metric_report(
                cell_summaries,
                model=model,
                metric="skill_vs_constant_velocity",
            ),
        }
        decomposition = {
            key: _metric_report(cell_summaries, model=model, metric=key)
            for key in (
                "actual_relative_motion",
                "predicted_relative_motion",
                "actual_cube_motion",
                "predicted_cube_motion",
                "actual_bowl_motion",
                "predicted_bowl_motion",
                "cube_forecast_error",
                "bowl_forecast_error",
            )
        }
        jointly_observable_early_count = sum(
            row["early_observable"] for row in request_rows if row["model_id"] == model
        )
        early_horizon = alignment.get("early_horizon")
        if early_horizon is None:
            earlier_horizon_report: dict[str, Any] = {
                "status": "unsupported_no_earlier_qualified_exposed_target",
                "qualified_target": None,
                "observable_requests": 0,
            }
        elif not jointly_observable_early_count:
            earlier_horizon_report = {
                "status": "qualified_but_unobservable_in_consensus",
                "qualified_target": dict(early_horizon),
                "observable_requests": 0,
            }
        else:
            earlier_horizon_report = {
                "status": "supported_and_observed",
                "qualified_target": dict(early_horizon),
                "observable_requests": jointly_observable_early_count,
                "paired_skill_comparison": {
                    "definition": (
                        "primary_skill_at_H minus early_skill for the same request; "
                        "request pairs are averaged within episode, equally across four "
                        "conditions, then bootstrapped across independent layouts"
                    ),
                    "primary_skill_at_H": _metric_report(
                        cell_summaries,
                        model=model,
                        metric="primary_skill_paired_with_early",
                    ),
                    "early_skill": _metric_report(
                        cell_summaries,
                        model=model,
                        metric="early_skill",
                    ),
                    "primary_minus_early_skill": _metric_report(
                        cell_summaries,
                        model=model,
                        metric="primary_minus_early_skill",
                    ),
                },
            }
        models[model] = {
            "branch_status": "qualified_and_included",
            "baseline_errors_and_skill": baseline,
            "coverage_by_condition": _coverage_report(selection, roster, request_rows, model),
            "full_design_strict_win_missingness_bounds": _win_bounds(cell_summaries, model),
            "reflection_contrasts": reflection_report(cell_summaries, model),
            "stopping_control": stopping_report(roster, context["endpoints"], model),
            "movement_strata": {
                "threshold_relative_image_diagonal": threshold,
                "threshold_source": "validated confirmation freeze copied exactly from development duplicate-label q95",
                "stationary": _stratum_report(cell_summaries, request_rows, model, "stationary"),
                "moving": _stratum_report(cell_summaries, request_rows, model, "moving"),
            },
            "movement_decomposition": decomposition,
            "earlier_horizon": earlier_horizon_report,
            "declared_rule_examples": _example_videos(roster, model),
        }
        model_requests = [row for row in request_rows if row["model_id"] == model]
        sample_table.append(
            {
                "model_id": model,
                "branch_status": "qualified_and_included",
                "model_configuration": dict(configuration),
                "checkpoint_revision": configuration.get("checkpoint_revision"),
                "executed_prefix_cap": configuration.get("executed_prefix_cap"),
                "primary_horizon_s": alignment["primary_horizon_s"],
                "generated_frame_index": alignment["generated_frame_index"],
                "target_executed_action_offset": alignment["target_executed_action_offset"],
                "camera_id": alignment["camera_id"],
                "timestamp_tolerance_s": alignment["timestamp_tolerance_s"],
                "full_design_planned_confirmation_cells": 96,
                "scientific_roster_cells": len(model_roster),
                "valid_complete": status_counts["valid_complete"],
                "valid_censored": status_counts["valid_censored"],
                "technical_invalid": status_counts["technical_invalid"],
                "unrun": status_counts["not_run"],
                "selected_requests": len(model_requests),
                "primary_observable_requests": sum(row["primary_observable"] for row in model_requests),
                "continuous_complete_layout_pairs": primary["layout_pairs"],
            }
        )
    reduced = len(qualified_models) == 1
    report = {
        "schema_version": OUTPUT_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "cohort_branch": branch,
        "study_scope": (
            f"REDUCED_ONE_MODEL_{qualified_models[0]}_BRANCH; the other primary model is unqualified and its 96 confirmation cells were not substituted"
            if reduced
            else "FULL_TWO_MODEL_BRANCH_REPORTED_AS_TWO_SEPARATE_WITHIN_MODEL_STUDIES"
        ),
        "analysis_contract": {
            "primary": "persistence error minus forecast error in normalized image-plane cube-minus-bowl coordinates; positive favors forecast",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "analysis_seed": ANALYSIS_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "continuous_missingness": "Only layouts with all four observable condition means contribute; contributing layout IDs are explicit.",
            "full_design_missingness": "Strict-win bounds retain every planned condition and layout.",
            "models_pooled": False,
        },
        "model_sample_size_table": sample_table,
        "frozen_resource_budget": context.get("frozen_resource_budget"),
        "annotation_quality": annotation_quality,
        "models": models,
        "claim_boundaries": {
            "forecast_accuracy_claim_gate": "PASSED_HASH_BOUND_RECORDINGS_VALIDATED_DEVELOPMENT_RELEASE_VALIDATED_ANNOTATION_FREEZE_AND_REPRODUCED_HUMAN_CONSENSUS",
            "either_sign_reported": True,
            "between_model_accuracy_ranking_allowed": False,
            "between_model_reason": "The specification requires separate within-model reports; physical horizons may differ and are not silently equated.",
            "causal_world_model_benefit_claim_allowed": False,
            "behavioral_success_from_image_centroids_allowed": False,
            "technical_invalid_is_model_failure": False,
            "censored_last_state_carried_to_action_450": False,
        },
        "request_level_audit": request_rows,
        "cell_level_audit": [cell_summaries[key] for key in sorted(cell_summaries)],
        "source_evidence": context.get("sources", {}),
        "receipt_evidence": context.get("receipt_evidence", {}),
        "analysis_evidence_manifest": context.get("manifest"),
    }
    return sign_document(report)


def analyze_manifest(manifest_path: Path) -> dict[str, Any]:
    return build_report(load_evidence(manifest_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = analyze_manifest(arguments.evidence_manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "cohort_branch": result["cohort_branch"],
                "models": sorted(result["models"]),
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
