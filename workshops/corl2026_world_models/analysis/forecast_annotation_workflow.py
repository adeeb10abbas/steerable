#!/usr/bin/env python3
"""Build blinded two-rater packets for WMF-ABLATION-001.

This module deliberately does not localize objects or manufacture labels.  It
selects requests from timing/camera/action-prefix metadata, verifies a separate
human pixel-blindness receipt for exact rendered hashes, copies those PNGs behind
independently randomized opaque identifiers, validates locked human responses,
and derives the development duplicate-label noise quantile required by the
confirmation freeze. PNG parsing rejects metadata but does not claim to detect
identifying text drawn into image pixels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import tempfile
from typing import Any, Mapping, Sequence
import zlib


STUDY_ID = "WMF-ABLATION-001"
REQUEST_INVENTORY_SCHEMA = "wmf-forecast-request-inventory-v1"
RECORDING_RECEIPT_SCHEMA = "wmf-forecast-recording-receipt-v1"
ACTION_MANIFEST_SCHEMA = "wmf-forecast-action-manifest-v1"
SELECTION_SCHEMA = "wmf-forecast-request-selection-v1"
IMAGE_INVENTORY_SCHEMA = "wmf-forecast-annotation-image-inventory-v1"
RENDER_RECEIPT_SCHEMA = "wmf-forecast-annotation-render-receipt-v1"
FREEZE_SCHEMA = "wmf-forecast-annotation-freeze-v1"
RUBRIC_SCHEMA = "wmf-forecast-annotation-rubric-v1"
EXAMPLE_MANIFEST_SCHEMA = "wmf-forecast-illustrated-example-manifest-v1"
PIXEL_BLINDNESS_RECEIPT_SCHEMA = "wmf-forecast-pixel-blindness-review-v1"
RESTRICTED_MAP_SCHEMA = "wmf-forecast-annotation-restricted-map-v1"
ADJUDICATION_MAP_SCHEMA = "wmf-forecast-adjudication-map-v1"
RATER_PACKET_SCHEMA = "wmf-forecast-rater-packet-v1"
RATER_RESPONSE_SCHEMA = "wmf-forecast-rater-response-v1"
DEVELOPMENT_SUMMARY_SCHEMA = "wmf-forecast-development-label-noise-v1"
FINAL_CONSENSUS_SCHEMA = "wmf-forecast-final-consensus-v2"

REQUEST_SAMPLING_SEED = 2026091302
REQUEST_SAMPLE_CAP = 4
REQUEST_SAMPLING_ALGORITHM = (
    "Within each episode, sort eligible source_request_id values by "
    "SHA256(UTF8(seed) || NUL || UTF8(episode_id) || NUL || "
    "UTF8(source_request_id)), breaking digest ties by source_request_id; "
    "select the first min(4, eligible_count)."
)
MOVEMENT_DISAGREEMENT_DEFINITION = (
    "Euclidean distance between the two raters' cube-minus-bowl image-plane "
    "vectors, with each vector divided by that image's pixel diagonal."
)
MOVEMENT_QUANTILE_METHOD = "linear interpolation at p=0.95 (Hyndman-Fan type 7)"

CONDITIONS = {
    "original_left",
    "original_right",
    "reflected_left",
    "reflected_right",
}
IMAGE_ROLES = {
    "preceding",
    "current",
    "predicted",
    "executed",
    "early_predicted",
    "early_executed",
}
REQUEST_KEYS = {
    "cell_id",
    "source_request_id",
    "source_video_id",
    "source_video_sha256",
    "model_id",
    "layout_pair_id",
    "condition_id",
    "episode_id",
    "request_index",
    "request_start_action_index",
    "action_manifest_sha256",
    "camera_id",
    "camera_crop_id",
    "camera_crop_sha256",
    "alignment_contract_id",
    "alignment_contract_sha256",
    "technical_valid",
    "technical_invalid_reason",
    "camera_identity_match",
    "target_within_executed_prefix",
    "executed_prefix_actions",
    "target_executed_action_offset",
    "generated_frame_index",
    "target_physical_time_s",
    "timestamp_error_s",
    "timestamp_tolerance_s",
    "history_mode",
    "early_horizon_supported",
    "alignment_receipt_id",
    "alignment_receipt_sha256",
}
ALIGNMENT_CONTRACT_KEYS = {
    "contract_id",
    "contract_sha256",
    "model_id",
    "mapping_receipt_id",
    "mapping_receipt_sha256",
    "primary_horizon_s",
    "generated_frame_index",
    "target_executed_action_offset",
    "control_step_s",
    "captured_frame_interval_s",
    "timestamp_tolerance_s",
    "camera_id",
    "camera_crop_id",
    "camera_crop_sha256",
    "image_width_px",
    "image_height_px",
    "early_horizon",
}
EARLY_HORIZON_KEYS = {
    "horizon_s",
    "generated_frame_index",
    "target_executed_action_offset",
}
ROSTER_KEYS = {
    "cell_id",
    "recording_id",
    "model_id",
    "layout_pair_id",
    "condition_id",
    "recording_status",
    "executed_action_count",
    "censor_reason",
    "recording_receipt_path",
    "recording_receipt_sha256",
    "action_manifest_path",
    "action_manifest_sha256",
    "source_video_id",
    "source_video_sha256",
}
ACTION_ROW_KEYS = {
    "action_index",
    "request_index",
    "executed_action_sha256",
    "control_timestamp",
    "physics_step_id",
    "camera_frame_id",
}
ACTION_MANIFEST_KEYS = {
    "schema_version",
    "study_id",
    "cell_id",
    "recording_id",
    "model_id",
    "executed_action_count",
    "actions",
    "payload_sha256",
}
RECORDING_RECEIPT_KEYS = {
    "schema_version",
    "study_id",
    "receipt_id",
    "stage",
    "cell_id",
    "recording_id",
    "model_id",
    "layout_pair_id",
    "condition_id",
    "recording_status",
    "executed_action_count",
    "censor_reason",
    "source_video_id",
    "source_video_sha256",
    "action_manifest_path",
    "action_manifest_sha256",
    "payload_sha256",
}
IMAGE_KEYS = {
    "source_image_id",
    "source_request_id",
    "source_video_id",
    "source_video_sha256",
    "image_role",
    "source_image_path",
    "source_image_sha256",
    "annotation_media_path",
    "annotation_media_sha256",
    "width_px",
    "height_px",
    "camera_id",
    "camera_crop_id",
    "camera_crop_sha256",
    "alignment_receipt_id",
    "alignment_receipt_sha256",
    "render_receipt_id",
    "render_receipt_path",
    "render_receipt_sha256",
    "presentation_sanitized",
    "contains_overlay",
}
RENDER_RECEIPT_KEYS = {
    "schema_version",
    "study_id",
    "receipt_id",
    "source_image_id",
    "source_request_id",
    "source_video_id",
    "source_video_sha256",
    "image_role",
    "source_image_path",
    "source_image_sha256",
    "annotation_media_path",
    "annotation_media_sha256",
    "width_px",
    "height_px",
    "camera_id",
    "camera_crop_id",
    "camera_crop_sha256",
    "alignment_receipt_id",
    "alignment_receipt_sha256",
    "presentation_sanitized",
    "contains_overlay",
    "payload_sha256",
}
ANNOTATION_KEYS = {
    "opaque_image_id",
    "cube_resolvability",
    "bowl_resolvability",
    "cube_identity",
    "bowl_identity",
    "cube_center_px",
    "bowl_center_px",
    "bowl_width_px",
    "ambiguity_codes",
    "source_guess",
    "annotation_seconds",
    "notes",
}
CONSENSUS_ANNOTATION_KEYS = {
    "cube_resolvability",
    "bowl_resolvability",
    "cube_identity",
    "bowl_identity",
    "cube_center_px",
    "bowl_center_px",
    "bowl_width_px",
    "ambiguity_codes",
}
RESPONSE_ATTESTATIONS = {
    "worked_independently",
    "no_identified_or_simultaneous_counterpart_access",
    "no_model_condition_instruction_success_access",
    "uninvolved_in_scorer_construction",
}
ADJUDICATOR_ATTESTATIONS = RESPONSE_ATTESTATIONS | {"no_first_pass_label_access"}
RESPONSE_KEYS = {
    "schema_version",
    "study_id",
    "packet_id",
    "packet_manifest_sha256",
    "rater_slot",
    "rater_code",
    "attestations",
    "started_at",
    "completed_at",
    "locked",
    "locked_at",
    "annotations",
}
RATER_PACKET_KEYS = {
    "schema_version",
    "study_id",
    "packet_id",
    "rater_slot",
    "instructions",
    "rubric_file",
    "rubric_sha256",
    "illustrated_examples_file",
    "illustrated_examples_sha256",
    "item_count",
    "items",
}
RATER_PACKET_ITEM_KEYS = {
    "presentation_index",
    "opaque_image_id",
    "media_file",
    "media_sha256",
    "width_px",
    "height_px",
}
TECHNICAL_INVALID_REASON_CODES = {
    "request_transport_failure",
    "response_contract_invalid",
    "decoded_future_contract_missing",
    "forecast_timing_unavailable",
    "recording_integrity_failure",
    "alignment_receipt_invalid",
    "camera_record_missing",
    "executed_prefix_missing",
    "safety_abort_before_target",
}
TECHNICAL_RECORDING_REASON_CODES = {
    "request_transport_failure",
    "model_runtime_failure",
    "recording_integrity_failure",
    "reset_contract_failure",
    "action_manifest_invalid",
    "camera_record_missing",
}
EXAMPLE_EXCLUSION_KEYS = {
    "model_identity_absent",
    "condition_identity_absent",
    "instruction_text_absent",
    "success_outcome_absent",
    "source_paths_absent",
    "counterpart_identity_absent",
    "overlays_absent",
}
PIXEL_BLINDNESS_RECEIPT_KEYS = {
    "schema_version",
    "study_id",
    "stage",
    "artifact_scope",
    "status",
    "reviewer_code",
    "reviewed_at",
    "review_method",
    "content_exclusions",
    "assets",
    "payload_sha256",
}
ADJUDICATION_MAP_KEYS = {
    "schema_version",
    "study_id",
    "stage",
    "visibility",
    "source_restricted_map",
    "source_responses",
    "first_pass_rater_code_sha256_by_slot",
    "adjudicator_seed",
    "rubric_sha256",
    "rubric_ambiguity_codes",
    "source_guess_options",
    "packet_root",
    "packets",
    "asset_count",
    "assets",
    "payload_sha256",
}
ARTIFACT_SHA_REFERENCE_KEYS = {"path", "artifact_sha256"}
PACKET_STREAM_KEYS = {"delivery_rule", "batch_count", "batches"}
PACKET_BATCH_KEYS = {
    "packet_id",
    "packet_manifest_sha256",
    "packet_relative_path",
    "restricted_asset_ids",
}
ADJUDICATION_ASSET_KEYS = {
    "restricted_asset_id",
    "annotation_media_sha256",
    "width_px",
    "height_px",
    "source_request_ids",
    "rater_opaque_ids",
    "rater_packet_ids",
}


class ContractError(ValueError):
    """Raised when an annotation artifact violates a fail-closed contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    """Hash a file, or a directory as a canonical relative-path/file-hash map."""

    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    require(path.is_dir(), f"artifact does not exist: {path}")
    files = {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    }
    require(files, f"artifact directory is empty: {path}")
    return sha256_bytes(canonical_bytes(files))


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def require_nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty")
    return value


def require_finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    result = float(value)
    if minimum is not None:
        require(result >= minimum, f"{label} must be at least {minimum}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def payload_hash(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("payload_sha256", None)
    return sha256_bytes(canonical_bytes(unsigned))


def sign_document(document: dict[str, Any]) -> dict[str, Any]:
    document = dict(document)
    document["payload_sha256"] = payload_hash(document)
    return document


def verify_signed(document: Mapping[str, Any], label: str) -> None:
    observed = require_sha256(document.get("payload_sha256"), f"{label} payload_sha256")
    require(payload_hash(document) == observed, f"{label} payload hash mismatch")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    require(not missing, f"{label} missing keys: {sorted(missing)}")
    require(not extra, f"{label} has disallowed keys: {sorted(extra)}")


def require_rfc3339_utc(value: Any, label: str) -> datetime:
    text = require_nonempty_string(value, label)
    require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", text) is not None,
        f"{label} must be an RFC3339 UTC timestamp ending in Z",
    )
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ContractError(f"{label} is not a valid timestamp") from error


def _validate_alignment_contracts(
    contracts: Any,
    *,
    stage: str,
    cohort_branch: str,
) -> dict[str, dict[str, Any]]:
    require(isinstance(contracts, list), "alignment_contracts must be a list")
    expected_models = {model for model, _, _ in _planned_cells(stage, cohort_branch).values()}
    by_model: dict[str, dict[str, Any]] = {}
    for raw in contracts:
        require(isinstance(raw, dict), "alignment contract must be an object")
        _exact_keys(raw, ALIGNMENT_CONTRACT_KEYS, f"alignment contract {raw.get('model_id', '<unknown>')}")
        model = raw.get("model_id")
        require(model in expected_models, f"alignment contract model is outside cohort branch: {model}")
        require(model not in by_model, f"duplicate alignment contract for {model}")
        require_nonempty_string(raw["contract_id"], f"{model} alignment contract_id")
        contract_sha = require_sha256(raw["contract_sha256"], f"{model} alignment contract_sha256")
        unsigned = dict(raw)
        unsigned.pop("contract_sha256")
        require(sha256_bytes(canonical_bytes(unsigned)) == contract_sha, f"{model} alignment contract hash mismatch")
        require_nonempty_string(raw["mapping_receipt_id"], f"{model} mapping_receipt_id")
        require_sha256(raw["mapping_receipt_sha256"], f"{model} mapping_receipt_sha256")
        horizon = require_finite(raw["primary_horizon_s"], f"{model} primary_horizon_s", minimum=0)
        control = require_finite(raw["control_step_s"], f"{model} control_step_s", minimum=0)
        capture = require_finite(raw["captured_frame_interval_s"], f"{model} captured_frame_interval_s", minimum=0)
        tolerance = require_finite(raw["timestamp_tolerance_s"], f"{model} timestamp_tolerance_s", minimum=0)
        require(horizon > 0 and control > 0 and capture > 0 and tolerance > 0, f"{model} alignment times must be positive")
        required_tolerance = min(control / 2.0, capture / 2.0)
        require(
            math.isclose(tolerance, required_tolerance, rel_tol=0, abs_tol=1e-12),
            f"{model} timestamp tolerance is not min(half control step, half capture interval)",
        )
        require(type(raw["generated_frame_index"]) is int and raw["generated_frame_index"] >= 0, f"{model} generated_frame_index is invalid")
        require(
            type(raw["target_executed_action_offset"]) is int and raw["target_executed_action_offset"] > 0,
            f"{model} target_executed_action_offset is invalid",
        )
        prefix_cap = 32 if model == "N3" else 8
        require(raw["target_executed_action_offset"] <= prefix_cap, f"{model} target lies outside unchanged executed prefix")
        require_nonempty_string(raw["camera_id"], f"{model} camera_id")
        require_nonempty_string(raw["camera_crop_id"], f"{model} camera_crop_id")
        require_sha256(raw["camera_crop_sha256"], f"{model} camera_crop_sha256")
        require(type(raw["image_width_px"]) is int and raw["image_width_px"] > 0, f"{model} image_width_px is invalid")
        require(type(raw["image_height_px"]) is int and raw["image_height_px"] > 0, f"{model} image_height_px is invalid")
        early = raw["early_horizon"]
        if early is not None:
            require(isinstance(early, dict), f"{model} early_horizon must be null or an object")
            _exact_keys(early, EARLY_HORIZON_KEYS, f"{model} early_horizon")
            early_time = require_finite(early["horizon_s"], f"{model} early horizon_s", minimum=0)
            require(0 < early_time < horizon, f"{model} early horizon is not strictly between zero and H")
            require(type(early["generated_frame_index"]) is int and early["generated_frame_index"] >= 0, f"{model} early frame index is invalid")
            require(early["generated_frame_index"] < raw["generated_frame_index"], f"{model} early frame is not earlier")
            require(
                type(early["target_executed_action_offset"]) is int
                and 0 < early["target_executed_action_offset"] < raw["target_executed_action_offset"],
                f"{model} early action offset is not earlier",
            )
        by_model[model] = dict(raw)
    require(set(by_model) == expected_models, "alignment contracts do not cover the cohort branch models")
    return by_model


def _eligibility_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not row["technical_valid"]:
        reasons.append("technical_invalid")
    if not row["camera_identity_match"]:
        reasons.append("camera_identity_mismatch")
    if not row["target_within_executed_prefix"]:
        reasons.append("target_outside_executed_prefix")
    if (
        row["technical_valid"]
        and row["target_within_executed_prefix"]
        and row["timestamp_error_s"] > row["timestamp_tolerance_s"]
    ):
        reasons.append("timestamp_outside_tolerance")
    return reasons


def _planned_cells(stage: str, cohort_branch: str) -> dict[str, tuple[str, str, str]]:
    require(stage in {"development", "confirmation"}, "planned-cell stage is invalid")
    branch_models = {
        "full_two_model": ("N3", "D1"),
        "reduced_n3": ("N3",),
        "reduced_d1": ("D1",),
    }
    require(cohort_branch in branch_models, "cohort_branch is invalid")
    layouts = (
        [f"D{index:02d}" for index in range(1, 5)]
        if stage == "development"
        else [f"C{index:02d}" for index in range(1, 25)]
    )
    result = {}
    for layout in layouts:
        for model in branch_models[cohort_branch]:
            for condition in sorted(CONDITIONS):
                arm, command = condition.split("_", maxsplit=1)
                cell_id = f"wmf1__{stage}__{layout}__{model}__{arm}__{command}"
                result[cell_id] = (model, layout, condition)
    return result


def _qualified_models(cohort_branch: str) -> tuple[str, ...]:
    branches = {
        "full_two_model": ("N3", "D1"),
        "reduced_n3": ("N3",),
        "reduced_d1": ("D1",),
    }
    require(cohort_branch in branches, "cohort_branch is invalid")
    return branches[cohort_branch]


def _validate_action_manifest(
    path: Path,
    *,
    expected_sha256: str,
    roster_row: Mapping[str, Any],
) -> dict[str, Any]:
    require(path.is_file(), f"action manifest is missing: {path}")
    require(sha256_file(path) == expected_sha256, "action manifest file hash mismatch")
    manifest = load_json(path)
    _exact_keys(manifest, ACTION_MANIFEST_KEYS, f"action manifest {roster_row['cell_id']}")
    require(manifest.get("schema_version") == ACTION_MANIFEST_SCHEMA, "action manifest schema mismatch")
    require(manifest.get("study_id") == STUDY_ID, "action manifest study mismatch")
    verify_signed(manifest, "action manifest")
    for key in ("cell_id", "recording_id", "model_id", "executed_action_count"):
        require(manifest.get(key) == roster_row.get(key), f"action manifest {key} differs from roster")
    actions = manifest.get("actions")
    require(isinstance(actions, list), "action manifest actions must be a list")
    require(len(actions) == roster_row["executed_action_count"], "action manifest length differs from executed_action_count")
    physics_ids = set()
    camera_ids = set()
    for expected_index, action in enumerate(actions):
        require(isinstance(action, dict), "action manifest row must be an object")
        _exact_keys(action, ACTION_ROW_KEYS, f"action manifest row {expected_index}")
        require(action.get("action_index") == expected_index, "action manifest indices are not contiguous")
        require(type(action.get("request_index")) is int and action["request_index"] >= 0, "action request_index is invalid")
        require_sha256(action.get("executed_action_sha256"), "executed action sha256")
        require_rfc3339_utc(action.get("control_timestamp"), "action control_timestamp")
        physics_id = require_nonempty_string(action.get("physics_step_id"), "action physics_step_id")
        camera_id = require_nonempty_string(action.get("camera_frame_id"), "action camera_frame_id")
        require(physics_id not in physics_ids, "action manifest repeats a physics step")
        require(camera_id not in camera_ids, "action manifest repeats a camera frame")
        physics_ids.add(physics_id)
        camera_ids.add(camera_id)
    return manifest


def _validate_recording_artifacts(
    raw: Mapping[str, Any],
    *,
    base: Path,
    stage: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    receipt_path = _resolve_reference(base, raw.get("recording_receipt_path"), "recording receipt path")
    receipt_sha = require_sha256(raw.get("recording_receipt_sha256"), "recording receipt sha256")
    require(receipt_path.is_file() and sha256_file(receipt_path) == receipt_sha, "recording receipt file hash mismatch")
    receipt = load_json(receipt_path)
    _exact_keys(receipt, RECORDING_RECEIPT_KEYS, f"recording receipt {raw['cell_id']}")
    require(receipt.get("schema_version") == RECORDING_RECEIPT_SCHEMA, "recording receipt schema mismatch")
    require(receipt.get("study_id") == STUDY_ID, "recording receipt study mismatch")
    verify_signed(receipt, "recording receipt")
    expected = {
        "stage": stage,
        "cell_id": raw["cell_id"],
        "recording_id": raw["recording_id"],
        "model_id": raw["model_id"],
        "layout_pair_id": raw["layout_pair_id"],
        "condition_id": raw["condition_id"],
        "recording_status": raw["recording_status"],
        "executed_action_count": raw["executed_action_count"],
        "censor_reason": raw["censor_reason"],
        "source_video_id": raw["source_video_id"],
        "source_video_sha256": raw["source_video_sha256"],
        "action_manifest_sha256": raw["action_manifest_sha256"],
    }
    require_nonempty_string(receipt.get("receipt_id"), "recording receipt_id")
    for key, value in expected.items():
        require(receipt.get(key) == value, f"recording receipt {key} differs from roster")
    action_path = _resolve_reference(base, raw.get("action_manifest_path"), "action manifest path")
    receipt_action_path = _resolve_reference(receipt_path.parent, receipt.get("action_manifest_path"), "receipt action manifest path")
    require(receipt_action_path == action_path, "recording receipt action-manifest path differs from roster")
    action_sha = require_sha256(raw.get("action_manifest_sha256"), "action manifest sha256")
    action_manifest = _validate_action_manifest(action_path, expected_sha256=action_sha, roster_row=raw)
    return dict(receipt), action_path, action_manifest


def _validate_episode_roster(
    roster: Any,
    *,
    stage: str,
    cohort_branch: str,
    base: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    require(isinstance(roster, list), "episode_roster must be a list")
    expected = _planned_cells(stage, cohort_branch)
    by_cell: dict[str, dict[str, Any]] = {}
    actions_by_cell: dict[str, dict[str, Any]] = {}
    for raw in roster:
        require(isinstance(raw, dict), "episode roster row must be an object")
        _exact_keys(raw, ROSTER_KEYS, f"episode roster {raw.get('cell_id', '<unknown>')}")
        cell_id = require_nonempty_string(raw["cell_id"], "episode roster cell_id")
        require(cell_id in expected, f"episode roster contains an unplanned cell: {cell_id}")
        require(cell_id not in by_cell, f"episode roster duplicates cell: {cell_id}")
        model, layout, condition = expected[cell_id]
        require(raw["model_id"] == model, f"episode roster model mismatch for {cell_id}")
        require(raw["layout_pair_id"] == layout, f"episode roster layout mismatch for {cell_id}")
        require(raw["condition_id"] == condition, f"episode roster condition mismatch for {cell_id}")
        status = raw["recording_status"]
        require(
            status in {"valid_complete", "valid_censored", "technical_invalid", "not_run"},
            f"episode roster status is invalid for {cell_id}",
        )
        if status in {"valid_complete", "valid_censored"}:
            require_nonempty_string(raw["recording_id"], f"episode roster recording_id {cell_id}")
            require_nonempty_string(raw["source_video_id"], f"episode roster source_video_id {cell_id}")
            require_sha256(raw["source_video_sha256"], f"episode roster source_video_sha256 {cell_id}")
            require(
                type(raw["executed_action_count"]) is int,
                f"episode roster executed_action_count is invalid for {cell_id}",
            )
            if status == "valid_complete":
                require(raw["executed_action_count"] == 450, f"valid-complete recording must contain exactly 450 actions: {cell_id}")
                require(raw["censor_reason"] is None, f"valid-complete recording has a censor reason: {cell_id}")
            else:
                require(0 < raw["executed_action_count"] < 450, f"valid-censored action count must be in [1,449]: {cell_id}")
                require(raw["censor_reason"] == "safety_abort", f"valid-censored recording must identify a safety abort: {cell_id}")
            _, action_path, action_manifest = _validate_recording_artifacts(raw, base=base, stage=stage)
            actions_by_cell[cell_id] = action_manifest
        elif status == "technical_invalid":
            require_nonempty_string(raw["recording_id"], f"technical-invalid recording_id {cell_id}")
            if raw["source_video_id"] is None:
                require(raw["source_video_sha256"] is None, f"technical-invalid video hash lacks an id for {cell_id}")
            else:
                require_nonempty_string(raw["source_video_id"], f"technical-invalid source_video_id {cell_id}")
                require_sha256(raw["source_video_sha256"], f"technical-invalid source_video_sha256 {cell_id}")
            require(
                type(raw["executed_action_count"]) is int and 0 <= raw["executed_action_count"] < 450,
                f"technical-invalid executed_action_count is invalid for {cell_id}",
            )
            require_nonempty_string(raw["censor_reason"], f"technical-invalid censor_reason {cell_id}")
            require(
                raw["censor_reason"] in TECHNICAL_RECORDING_REASON_CODES,
                f"technical-invalid censor_reason is not a frozen objective code: {cell_id}",
            )
            _, action_path, action_manifest = _validate_recording_artifacts(raw, base=base, stage=stage)
            actions_by_cell[cell_id] = action_manifest
        else:
            require(raw["recording_id"] is None, f"not-run cell has a recording_id: {cell_id}")
            require(raw["source_video_id"] is None, f"not-run cell has a source_video_id: {cell_id}")
            require(raw["source_video_sha256"] is None, f"not-run cell has a source_video_sha256: {cell_id}")
            require(raw["executed_action_count"] is None, f"not-run cell has an executed_action_count: {cell_id}")
            require(raw["censor_reason"] is None, f"not-run cell has a censor_reason: {cell_id}")
            for artifact_key in (
                "recording_receipt_path",
                "recording_receipt_sha256",
                "action_manifest_path",
                "action_manifest_sha256",
            ):
                require(raw[artifact_key] is None, f"not-run cell has {artifact_key}: {cell_id}")
        normalized = dict(raw)
        if status != "not_run":
            normalized["recording_receipt_path"] = str(_resolve_reference(base, raw["recording_receipt_path"], "recording receipt path"))
            normalized["action_manifest_path"] = str(action_path)
        by_cell[cell_id] = normalized
    missing = set(expected) - set(by_cell)
    require(not missing, f"episode roster is incomplete; missing planned cells: {sorted(missing)}")
    require(len(by_cell) == len(expected), "episode roster size differs from the planned cohort branch")
    return by_cell, actions_by_cell


def _validate_request(row: Mapping[str, Any], stage: str) -> None:
    _exact_keys(row, REQUEST_KEYS, f"request {row.get('source_request_id', '<unknown>')}")
    for key in (
        "source_request_id",
        "cell_id",
        "source_video_id",
        "layout_pair_id",
        "episode_id",
        "camera_id",
        "alignment_receipt_id",
    ):
        require_nonempty_string(row[key], f"request {key}")
    for key in ("source_video_sha256", "alignment_receipt_sha256"):
        require_sha256(row[key], f"request {key}")
    require(row["model_id"] in {"N3", "D1"}, "request model_id must be N3 or D1")
    require(row["condition_id"] in CONDITIONS, "request condition_id is invalid")
    allowed_layouts = (
        {f"D{index:02d}" for index in range(1, 5)}
        if stage == "development"
        else {f"C{index:02d}" for index in range(1, 25)}
    )
    require(row["layout_pair_id"] in allowed_layouts, "request layout_pair_id does not match the planned stage")
    require(type(row["request_index"]) is int and row["request_index"] >= 0, "request_index is invalid")
    require(
        type(row["request_start_action_index"]) is int and row["request_start_action_index"] >= 0,
        "request_start_action_index is invalid",
    )
    require_sha256(row["action_manifest_sha256"], "request action_manifest_sha256")
    for key in (
        "technical_valid",
        "camera_identity_match",
        "target_within_executed_prefix",
        "early_horizon_supported",
    ):
        require(type(row[key]) is bool, f"request {key} must be boolean")
    require(
        type(row["executed_prefix_actions"]) is int and row["executed_prefix_actions"] > 0,
        "executed_prefix_actions must be a positive integer",
    )
    model_prefix_cap = 32 if row["model_id"] == "N3" else 8
    require(
        row["executed_prefix_actions"] <= model_prefix_cap,
        f"executed_prefix_actions exceeds the {row['model_id']} unchanged executed-prefix cap",
    )
    require(
        type(row["target_executed_action_offset"]) is int and row["target_executed_action_offset"] > 0,
        "target_executed_action_offset must be a positive integer",
    )
    require(
        type(row["generated_frame_index"]) is int and row["generated_frame_index"] >= 0,
        "generated_frame_index must be a nonnegative integer",
    )
    require_finite(row["target_physical_time_s"], "target_physical_time_s", minimum=0)
    require(row["target_physical_time_s"] > 0, "target_physical_time_s must be strictly positive")
    within_prefix = row["target_executed_action_offset"] <= row["executed_prefix_actions"]
    require(
        row["target_within_executed_prefix"] is within_prefix,
        "target_within_executed_prefix disagrees with the recorded action counts",
    )
    if row["technical_valid"]:
        require(row["technical_invalid_reason"] is None, "valid request has technical_invalid_reason")
    else:
        require_nonempty_string(row["technical_invalid_reason"], "technical_invalid_reason")
        require(
            row["technical_invalid_reason"] in TECHNICAL_INVALID_REASON_CODES,
            "technical_invalid_reason is not an objective frozen reason code",
        )
    timing_available = row["technical_valid"] and within_prefix
    if timing_available:
        error = require_finite(row["timestamp_error_s"], "timestamp_error_s", minimum=0)
    else:
        require(
            row["timestamp_error_s"] is None,
            "timestamp_error_s must be null when forecast timing is unavailable",
        )
        error = None
    tolerance = require_finite(row["timestamp_tolerance_s"], "timestamp_tolerance_s", minimum=0)
    require(tolerance > 0, "timestamp_tolerance_s must be positive")
    require(row["history_mode"] in {"preceding_observation", "persistence_at_initial_request"}, "history_mode is invalid")
    if row["history_mode"] == "persistence_at_initial_request":
        require(row["request_index"] == 0, "persistence history fallback is allowed only at request zero")
    else:
        require(row["request_index"] > 0, "request zero must use the persistence history fallback")
    # Evaluate once here so a malformed numeric comparison cannot be hidden.
    if error is not None:
        _ = error <= tolerance


def select_requests(
    inventory: Mapping[str, Any],
    *,
    seed: int = REQUEST_SAMPLING_SEED,
    cap_per_episode: int = REQUEST_SAMPLE_CAP,
    inventory_sha256: str | None = None,
    inventory_path: Path | None = None,
) -> dict[str, Any]:
    """Return a complete, metadata-only request inventory with selection flags."""

    _exact_keys(
        inventory,
        {
            "schema_version",
            "study_id",
            "stage",
            "cohort_branch",
            "inventory_complete",
            "inventory_finalized_at",
            "annotation_state",
            "episode_roster",
            "alignment_contracts",
            "requests",
        },
        "request inventory",
    )
    require(inventory.get("schema_version") == REQUEST_INVENTORY_SCHEMA, "request inventory schema mismatch")
    require(inventory.get("study_id") == STUDY_ID, "request inventory study mismatch")
    stage = inventory.get("stage")
    require(stage in {"development", "confirmation"}, "request inventory stage is invalid")
    cohort_branch = inventory.get("cohort_branch")
    inventory_base = inventory_path.resolve().parent if inventory_path is not None else Path.cwd().resolve()
    roster, action_manifests = _validate_episode_roster(
        inventory.get("episode_roster"),
        stage=stage,
        cohort_branch=cohort_branch,
        base=inventory_base,
    )
    alignment_contracts = _validate_alignment_contracts(
        inventory.get("alignment_contracts"),
        stage=stage,
        cohort_branch=cohort_branch,
    )
    require(inventory.get("inventory_complete") is True, "request inventory is not marked complete")
    require_rfc3339_utc(inventory.get("inventory_finalized_at"), "request inventory_finalized_at")
    require(inventory.get("annotation_state") == "not_started", "request selection must precede image labeling")
    require(seed == REQUEST_SAMPLING_SEED, "request sampling seed differs from the machine-readable specification")
    require(cap_per_episode == REQUEST_SAMPLE_CAP, "request sample cap differs from the specification")
    rows = inventory.get("requests")
    require(isinstance(rows, list), "request inventory requests must be a list")
    require(rows, "request inventory is empty")

    seen: set[str] = set()
    episode_identity: dict[str, tuple[Any, ...]] = {}
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in rows:
        require(isinstance(raw_row, dict), "each request must be an object")
        _validate_request(raw_row, stage)
        request_id = raw_row["source_request_id"]
        require(request_id not in seen, f"duplicate source_request_id: {request_id}")
        seen.add(request_id)
        cell_id = raw_row["cell_id"]
        require(cell_id in roster, f"request references a cell outside the episode roster: {cell_id}")
        roster_row = roster[cell_id]
        require(
            roster_row["recording_status"] in {"valid_complete", "valid_censored"},
            f"request belongs to nonbehavioral roster status {roster_row['recording_status']}: {cell_id}",
        )
        require(raw_row["model_id"] == roster_row["model_id"], f"request model differs from roster: {cell_id}")
        require(raw_row["layout_pair_id"] == roster_row["layout_pair_id"], f"request layout differs from roster: {cell_id}")
        require(raw_row["condition_id"] == roster_row["condition_id"], f"request condition differs from roster: {cell_id}")
        require(raw_row["episode_id"] == roster_row["recording_id"], f"request recording identity differs from roster: {cell_id}")
        require(raw_row["source_video_id"] == roster_row["source_video_id"], f"request video identity differs from roster: {cell_id}")
        require(raw_row["source_video_sha256"] == roster_row["source_video_sha256"], f"request video hash differs from roster: {cell_id}")
        require(
            raw_row["action_manifest_sha256"] == roster_row["action_manifest_sha256"],
            f"request action-manifest hash differs from roster: {cell_id}",
        )
        alignment = alignment_contracts[raw_row["model_id"]]
        require(raw_row["alignment_contract_id"] == alignment["contract_id"], f"request alignment contract id mismatch: {cell_id}")
        require(raw_row["alignment_contract_sha256"] == alignment["contract_sha256"], f"request alignment contract hash mismatch: {cell_id}")
        require(raw_row["generated_frame_index"] == alignment["generated_frame_index"], f"request generated frame differs from frozen H: {cell_id}")
        require(
            raw_row["target_executed_action_offset"] == alignment["target_executed_action_offset"],
            f"request target action differs from frozen H: {cell_id}",
        )
        require(
            math.isclose(raw_row["target_physical_time_s"], alignment["primary_horizon_s"], rel_tol=0, abs_tol=1e-12),
            f"request physical horizon differs from frozen H: {cell_id}",
        )
        require(
            math.isclose(raw_row["timestamp_tolerance_s"], alignment["timestamp_tolerance_s"], rel_tol=0, abs_tol=1e-12),
            f"request timestamp tolerance differs from frozen mapping: {cell_id}",
        )
        require(raw_row["camera_id"] == alignment["camera_id"], f"request camera differs from frozen mapping: {cell_id}")
        require(raw_row["camera_crop_id"] == alignment["camera_crop_id"], f"request crop id differs from frozen mapping: {cell_id}")
        require(raw_row["camera_crop_sha256"] == alignment["camera_crop_sha256"], f"request crop hash differs from frozen mapping: {cell_id}")
        require(
            raw_row["early_horizon_supported"] is (alignment["early_horizon"] is not None),
            f"request early-horizon support differs from frozen mapping: {cell_id}",
        )
        identity = (
            raw_row["model_id"],
            raw_row["layout_pair_id"],
            raw_row["condition_id"],
            raw_row["source_video_id"],
            raw_row["source_video_sha256"],
            raw_row["camera_id"],
        )
        existing = episode_identity.setdefault(cell_id, identity)
        require(existing == identity, f"episode identity changes within {cell_id}")
        by_cell[cell_id].append(dict(raw_row))

    for cell_id, roster_row in roster.items():
        if roster_row["recording_status"] in {"valid_complete", "valid_censored"}:
            require(by_cell[cell_id], f"valid roster cell has no request inventory: {cell_id}")
            indexes = sorted(row["request_index"] for row in by_cell[cell_id])
            require(indexes == list(range(len(indexes))), f"request indices are not unique and contiguous for {cell_id}")
            chunk_size = 32 if roster_row["model_id"] == "N3" else 8
            action_count = roster_row["executed_action_count"]
            expected_count = math.ceil(action_count / chunk_size)
            require(
                len(indexes) == expected_count,
                f"{roster_row['recording_status']} {roster_row['model_id']} cell has {len(indexes)} requests, expected {expected_count} for {action_count} actions",
            )
            ordered = sorted(by_cell[cell_id], key=lambda row: row["request_index"])
            expected_prefixes = [chunk_size] * expected_count
            expected_prefixes[-1] = action_count - chunk_size * (expected_count - 1)
            require(
                [row["executed_prefix_actions"] for row in ordered] == expected_prefixes,
                f"request executed-prefix bounds do not reconstruct the recorded actions for {cell_id}",
            )
            expected_starts = []
            cursor = 0
            for prefix in expected_prefixes:
                expected_starts.append(cursor)
                cursor += prefix
            require(
                [row["request_start_action_index"] for row in ordered] == expected_starts,
                f"request start-action indices do not reconstruct the recording for {cell_id}",
            )
            action_rows = action_manifests[cell_id]["actions"]
            require(
                [row["request_index"] for row in action_rows]
                == [request_index for request_index, prefix in enumerate(expected_prefixes) for _ in range(prefix)],
                f"action manifest request ownership differs from request inventory for {cell_id}",
            )
        else:
            require(not by_cell[cell_id], f"nonbehavioral roster cell has request inventory: {cell_id}")

    output_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for cell_id in sorted(roster):
        roster_row = roster[cell_id]
        episode_id = roster_row["recording_id"] or cell_id
        candidates = by_cell[cell_id]
        eligible = [row for row in candidates if not _eligibility_reasons(row)]

        def rank(row: Mapping[str, Any]) -> tuple[str, str]:
            payload = f"{seed}\0{episode_id}\0{row['source_request_id']}".encode("utf-8")
            return sha256_bytes(payload), row["source_request_id"]

        selected_ids = {
            row["source_request_id"]
            for row in sorted(eligible, key=rank)[: min(cap_per_episode, len(eligible))]
        }
        n = len(eligible)
        selected_count = len(selected_ids)
        probability = None if n == 0 else selected_count / n
        probability_exact = None if n == 0 else f"{selected_count}/{n}"
        episode_rows.append(
            {
                "cell_id": cell_id,
                "episode_id": episode_id,
                "recording_status": roster_row["recording_status"],
                "request_count": len(candidates),
                "eligible_count": n,
                "selected_count": selected_count,
                "zero_eligible": n == 0,
                "eligible_request_inclusion_probability": probability,
                "eligible_request_inclusion_probability_exact": probability_exact,
            }
        )
        for row in sorted(candidates, key=lambda item: item["source_request_id"]):
            reasons = _eligibility_reasons(row)
            output_rows.append(
                {
                    "source": row,
                    "timing_camera_action_eligible": not reasons,
                    "eligibility_reasons": reasons,
                    "selected": row["source_request_id"] in selected_ids,
                    "eligible_request_inclusion_probability": probability if not reasons else None,
                    "eligible_request_inclusion_probability_exact": probability_exact if not reasons else None,
                }
            )

    canonical_inventory = dict(inventory)
    canonical_inventory["requests"] = sorted(rows, key=lambda row: row["source_request_id"])
    canonical_inventory["episode_roster"] = sorted(roster.values(), key=lambda row: row["cell_id"])
    canonical_inventory["alignment_contracts"] = sorted(inventory["alignment_contracts"], key=lambda row: row["model_id"])
    document = {
        "schema_version": SELECTION_SCHEMA,
        "study_id": STUDY_ID,
        "stage": stage,
        "cohort_branch": cohort_branch,
        "inventory_complete": True,
        "inventory_finalized_at": inventory["inventory_finalized_at"],
        "annotation_state_at_selection": inventory["annotation_state"],
        "episode_roster": canonical_inventory["episode_roster"],
        "alignment_contracts": canonical_inventory["alignment_contracts"],
        "visibility": "RESTRICTED ANALYST MANIFEST; NEVER DISTRIBUTE TO RATERS",
        "selection_uses_object_visibility_or_forecast_quality": False,
        "sampling": {
            "seed": seed,
            "cap_per_episode": cap_per_episode,
            "algorithm": REQUEST_SAMPLING_ALGORITHM,
        },
        "provenance": {
            "request_inventory_sha256": inventory_sha256,
            "canonical_request_inventory_sha256": sha256_bytes(canonical_bytes(canonical_inventory)),
        },
        "counts": {
            "requests": len(rows),
            "eligible": sum(row["timing_camera_action_eligible"] for row in output_rows),
            "selected": sum(row["selected"] for row in output_rows),
            "episodes": len(roster),
            "zero_eligible_episodes": sum(row["zero_eligible"] for row in episode_rows),
        },
        "episodes": episode_rows,
        "requests": output_rows,
    }
    return sign_document(document)


def _resolve_reference(base: Path, value: Any, label: str) -> Path:
    text = require_nonempty_string(value, label)
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_rubric(rubric: Mapping[str, Any], *, stage: str) -> None:
    _exact_keys(
        rubric,
        {
            "schema_version",
            "study_id",
            "status",
            "freeze_scope",
            "landmarks",
            "never_infer_hidden_centers",
            "ambiguity_codes",
            "source_guess_options",
            "illustrated_examples",
        },
        "annotation rubric",
    )
    require(rubric.get("schema_version") == RUBRIC_SCHEMA, "annotation rubric schema mismatch")
    require(rubric.get("study_id") == STUDY_ID, "annotation rubric study mismatch")
    require(rubric.get("status") == "frozen", "annotation rubric is not frozen")
    landmarks = rubric.get("landmarks")
    require(isinstance(landmarks, dict), "rubric landmarks must be an object")
    _exact_keys(
        landmarks,
        {"cube_center", "bowl_center", "bowl_width", "partial_occlusion", "identity_uncertainty"},
        "rubric landmarks",
    )
    for key in ("cube_center", "bowl_center", "bowl_width", "partial_occlusion", "identity_uncertainty"):
        text = require_nonempty_string(landmarks.get(key), f"rubric landmark {key}")
        _require_identity_blind_text(text, f"rubric landmark {key}")
    require(rubric.get("never_infer_hidden_centers") is True, "rubric must forbid hidden-center inference")
    ambiguity = rubric.get("ambiguity_codes")
    require(isinstance(ambiguity, list) and ambiguity, "rubric ambiguity_codes must be nonempty")
    codes: list[str] = []
    for item in ambiguity:
        require(isinstance(item, dict), "rubric ambiguity code must be an object")
        _exact_keys(item, {"code", "description"}, "rubric ambiguity code")
        code = require_nonempty_string(item.get("code"), "ambiguity code")
        _require_identity_blind_text(code, f"ambiguity code {code}")
        description = require_nonempty_string(item.get("description"), f"ambiguity description {code}")
        _require_identity_blind_text(description, f"ambiguity description {code}")
        codes.append(code)
    require(len(codes) == len(set(codes)), "rubric ambiguity codes are duplicated")
    require("none" in codes, "rubric ambiguity codes must include none")
    guesses = rubric.get("source_guess_options")
    require(
        guesses == ["camera_observation", "generated_future", "unsure"],
        "rubric source_guess_options must contain camera_observation, generated_future, and unsure",
    )
    examples = rubric.get("illustrated_examples")
    require(isinstance(examples, dict), "rubric illustrated_examples must be an object")
    _exact_keys(
        examples,
        {"manifest_path", "manifest_sha256"},
        "rubric illustrated_examples",
    )
    require_sha256(examples.get("manifest_sha256"), "rubric example manifest_sha256")
    example_name = require_nonempty_string(examples.get("manifest_path"), "rubric example manifest_path")
    require(
        Path(example_name).name == example_name and "/" not in example_name and "\\" not in example_name,
        "rubric example manifest_path must be a source-free basename",
    )
    require(
        example_name == "illustrated_examples.json",
        "rubric example manifest_path must be illustrated_examples.json",
    )
    _require_identity_blind_text(example_name, "rubric example manifest_path")
    freeze_scope = rubric.get("freeze_scope")
    require(
        freeze_scope == "development_duplicate_validation_and_confirmation",
        "rubric freeze_scope must bind development duplicate validation and confirmation",
    )
    require(stage in {"development", "confirmation"}, "rubric validation stage is invalid")


def _require_identity_blind_text(value: str, label: str) -> None:
    lowered = value.casefold()
    forbidden_substrings = (
        "cosmos",
        "dreamzero",
        "original_left",
        "original_right",
        "reflected_left",
        "reflected_right",
        "put the rubik's cube to the left",
        "put the rubik's cube to the right",
        "success",
        "successful",
        "failure",
        "reward",
        "/data/",
        "source_video",
        "source_request",
    )
    require(not any(token in lowered for token in forbidden_substrings), f"{label} exposes model or condition identity")
    words = {word.strip(".,:;_-/()[]{}") for word in lowered.split()}
    require("n3" not in words and "d1" not in words, f"{label} exposes model identity")


def _validate_pixel_blindness_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    stage: str,
    artifact_scope: str,
    expected_assets: Sequence[tuple[str, int, int]],
) -> None:
    _exact_keys(receipt, PIXEL_BLINDNESS_RECEIPT_KEYS, "pixel-blindness review receipt")
    require(receipt.get("schema_version") == PIXEL_BLINDNESS_RECEIPT_SCHEMA, "pixel-blindness receipt schema mismatch")
    require(receipt.get("study_id") == STUDY_ID, "pixel-blindness receipt study mismatch")
    verify_signed(receipt, "pixel-blindness review receipt")
    require(receipt.get("stage") == stage, "pixel-blindness receipt stage mismatch")
    require(receipt.get("artifact_scope") == artifact_scope, "pixel-blindness receipt artifact scope mismatch")
    require(receipt.get("status") == "human_reviewed_source_blind", "pixel content lacks completed human blindness review")
    reviewer = require_nonempty_string(receipt.get("reviewer_code"), "pixel-blindness reviewer_code")
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", reviewer) is not None, "pixel-blindness reviewer_code is not canonical")
    require_rfc3339_utc(receipt.get("reviewed_at"), "pixel-blindness reviewed_at")
    require(
        receipt.get("review_method") == "human_visual_inspection_of_rendered_pixels_and_presentation",
        "pixel-blindness review method is not the required human visual inspection",
    )
    exclusions = receipt.get("content_exclusions")
    require(isinstance(exclusions, dict) and set(exclusions) == EXAMPLE_EXCLUSION_KEYS, "pixel-blindness exclusions changed")
    require(all(value is True for value in exclusions.values()), "human pixel-blindness review found prohibited content")
    assets = receipt.get("assets")
    require(isinstance(assets, list), "pixel-blindness receipt assets must be a list")
    observed = []
    for item in assets:
        require(isinstance(item, dict), "pixel-blindness asset must be an object")
        _exact_keys(item, {"media_sha256", "width_px", "height_px"}, "pixel-blindness asset")
        width = item.get("width_px")
        height = item.get("height_px")
        require(type(width) is int and width > 0, "pixel-blindness asset width_px must be a positive integer")
        require(type(height) is int and height > 0, "pixel-blindness asset height_px must be a positive integer")
        observed.append(
            (
                require_sha256(item.get("media_sha256"), "pixel-blindness media sha256"),
                width,
                height,
            )
        )
    require(len(observed) == len(set(observed)), "pixel-blindness receipt duplicates an asset")
    require(sorted(observed) == sorted(set(expected_assets)), "pixel-blindness receipt does not bind the exact rendered media population")


def _validate_example_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[list[Path], Path]:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "study_id",
            "status",
            "source_scope",
            "reviewer_code",
            "reviewed_at",
            "content_exclusions",
            "pixel_blindness_receipt_path",
            "pixel_blindness_receipt_sha256",
            "files",
            "payload_sha256",
        },
        "illustrated-example manifest",
    )
    require(manifest.get("schema_version") == EXAMPLE_MANIFEST_SCHEMA, "illustrated-example manifest schema mismatch")
    require(manifest.get("study_id") == STUDY_ID, "illustrated-example manifest study mismatch")
    verify_signed(manifest, "illustrated-example manifest")
    require(manifest.get("status") == "approved_source_free", "illustrated examples lack source-free approval")
    require(
        manifest.get("source_scope") == "synthetic_or_model_blind_development_calibration",
        "illustrated examples use a forbidden source scope",
    )
    reviewer = require_nonempty_string(manifest.get("reviewer_code"), "illustrated-example reviewer_code")
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", reviewer) is not None, "example reviewer_code is not canonical")
    require_rfc3339_utc(manifest.get("reviewed_at"), "illustrated-example reviewed_at")
    exclusions = manifest.get("content_exclusions")
    require(isinstance(exclusions, dict) and set(exclusions) == EXAMPLE_EXCLUSION_KEYS, "example content exclusions changed")
    require(all(value is True for value in exclusions.values()), "illustrated examples failed an identity/outcome exclusion")
    files = manifest.get("files")
    require(isinstance(files, list) and files, "illustrated-example manifest has no files")
    paths = []
    seen = set()
    for item in files:
        require(isinstance(item, dict), "illustrated-example file row is invalid")
        _exact_keys(item, {"file", "sha256", "width_px", "height_px", "caption", "purpose"}, "illustrated-example file")
        relative = require_nonempty_string(item.get("file"), "illustrated-example file path")
        candidate = Path(relative)
        require(
            len(candidate.parts) == 2
            and candidate.parts[0] == "examples"
            and re.fullmatch(r"example_[0-9]{3}\.png", candidate.name) is not None,
            "illustrated-example file path is not source-free",
        )
        require(relative not in seen, "illustrated-example file is duplicated")
        seen.add(relative)
        _require_identity_blind_text(require_nonempty_string(item.get("caption"), "example caption"), "example caption")
        _require_identity_blind_text(require_nonempty_string(item.get("purpose"), "example purpose"), "example purpose")
        path = _resolve_reference(manifest_path.parent, relative, "illustrated-example image path")
        require(path.is_file(), f"illustrated-example image is missing: {path}")
        require(sha256_file(path) == require_sha256(item.get("sha256"), "example image sha256"), "example image hash mismatch")
        width = item.get("width_px")
        height = item.get("height_px")
        require(type(width) is int and width > 0 and type(height) is int and height > 0, "example image dimensions are invalid")
        require(_png_dimensions(path) == (width, height), "example image dimensions or metadata are invalid")
        paths.append(path)
    blindness_path = _resolve_reference(
        manifest_path.parent,
        manifest.get("pixel_blindness_receipt_path"),
        "illustrated-example pixel-blindness receipt path",
    )
    require(blindness_path.is_file(), "illustrated-example pixel-blindness receipt is missing")
    blindness_sha = require_sha256(
        manifest.get("pixel_blindness_receipt_sha256"),
        "illustrated-example pixel-blindness receipt sha256",
    )
    require(sha256_file(blindness_path) == blindness_sha, "illustrated-example pixel-blindness receipt hash mismatch")
    _validate_pixel_blindness_receipt(
        load_json(blindness_path),
        receipt_path=blindness_path,
        stage="development",
        artifact_scope="illustrated_examples",
        expected_assets=[
            (item["sha256"], item["width_px"], item["height_px"])
            for item in files
        ],
    )
    return paths, blindness_path


def validate_freeze(freeze: Mapping[str, Any], *, freeze_path: Path, stage: str) -> dict[str, Any]:
    _exact_keys(
        freeze,
        {
            "schema_version",
            "study_id",
            "stage",
            "status",
            "cohort_branch",
            "qualified_model_ids",
            "request_sampling",
            "packet_randomization",
            "rubric",
            "qualified_alignment_contract_sha256_by_model",
            "movement_resolution",
            "development_validation_decision",
            "development_final_consensus",
        },
        "annotation freeze",
    )
    require(freeze.get("schema_version") == FREEZE_SCHEMA, "annotation freeze schema mismatch")
    require(freeze.get("study_id") == STUDY_ID, "annotation freeze study mismatch")
    require(freeze.get("stage") == stage, "annotation freeze stage mismatch")
    required_status = "frozen_for_development_validation" if stage == "development" else "frozen_for_confirmation"
    require(freeze.get("status") == required_status, "annotation freeze status mismatch")
    cohort_branch = freeze.get("cohort_branch")
    qualified_models = _qualified_models(cohort_branch)
    require(
        freeze.get("qualified_model_ids") == list(qualified_models),
        "freeze qualified_model_ids do not exactly match the cohort branch",
    )
    sampling = freeze.get("request_sampling")
    require(isinstance(sampling, dict), "freeze request_sampling must be an object")
    _exact_keys(sampling, {"seed", "cap_per_episode", "algorithm"}, "freeze request_sampling")
    require(sampling.get("seed") == REQUEST_SAMPLING_SEED, "freeze request sampling seed mismatch")
    require(sampling.get("cap_per_episode") == REQUEST_SAMPLE_CAP, "freeze request sample cap mismatch")
    require(sampling.get("algorithm") == REQUEST_SAMPLING_ALGORITHM, "freeze request sampling algorithm mismatch")
    randomization = freeze.get("packet_randomization")
    require(isinstance(randomization, dict), "freeze packet_randomization must be an object")
    _exact_keys(
        randomization,
        {"rater_seeds", "adjudicator_seed", "distinct_rater_seed_values"},
        "freeze packet_randomization",
    )
    seeds = randomization.get("rater_seeds")
    require(isinstance(seeds, dict) and set(seeds) == {"rater_a", "rater_b"}, "freeze must provide two rater seeds")
    require(all(type(value) is int and value >= 0 for value in seeds.values()), "rater seeds must be nonnegative integers")
    require(seeds["rater_a"] != seeds["rater_b"], "rater seeds must be distinct")
    adjudicator_seed = randomization.get("adjudicator_seed")
    require(type(adjudicator_seed) is int and adjudicator_seed >= 0, "adjudicator seed must be nonnegative")
    require(adjudicator_seed not in set(seeds.values()), "adjudicator seed must differ from both rater seeds")
    require(
        randomization.get("distinct_rater_seed_values")
        == [seeds["rater_a"], seeds["rater_b"], adjudicator_seed],
        "distinct_rater_seed_values must mirror rater_a, rater_b, then adjudicator",
    )
    alignment_hashes = freeze.get("qualified_alignment_contract_sha256_by_model")
    require(isinstance(alignment_hashes, dict) and alignment_hashes, "freeze lacks qualified alignment contract hashes")
    require(set(alignment_hashes) == set(qualified_models), "freeze alignment hashes do not exactly cover qualified models")
    for model, digest in alignment_hashes.items():
        require_sha256(digest, f"freeze {model} alignment contract sha256")

    rubric_ref = freeze.get("rubric")
    require(isinstance(rubric_ref, dict), "freeze rubric reference must be an object")
    _exact_keys(rubric_ref, {"path", "sha256"}, "freeze rubric reference")
    base = freeze_path.resolve().parent
    rubric_path = _resolve_reference(base, rubric_ref.get("path"), "rubric path")
    require(rubric_path.is_file(), "rubric file is missing")
    rubric_sha = require_sha256(rubric_ref.get("sha256"), "rubric sha256")
    require(sha256_file(rubric_path) == rubric_sha, "rubric file hash mismatch")
    rubric = load_json(rubric_path)
    _validate_rubric(rubric, stage=stage)
    examples_path = _resolve_reference(rubric_path.parent, rubric["illustrated_examples"]["manifest_path"], "rubric examples manifest path")
    require(examples_path.is_file(), "rubric illustrated examples manifest is missing")
    require(
        sha256_file(examples_path) == rubric["illustrated_examples"]["manifest_sha256"],
        "rubric illustrated examples manifest hash mismatch",
    )
    examples_manifest = load_json(examples_path)
    example_files, examples_blindness_path = _validate_example_manifest(examples_manifest, manifest_path=examples_path)

    movement = freeze.get("movement_resolution")
    require(isinstance(movement, dict), "freeze movement_resolution must be an object")
    _exact_keys(
        movement,
        {"status", "threshold_relative_image_diagonal", "development_summary_path", "development_summary_sha256"},
        "freeze movement_resolution",
    )
    summary: dict[str, Any] | None = None
    if stage == "development":
        require(freeze.get("development_validation_decision") is None, "development freeze cannot pre-approve measurement usability")
        require(movement.get("status") == "pending_duplicate_development_labels", "development threshold status must remain pending")
        require(movement.get("threshold_relative_image_diagonal") is None, "development freeze must not invent a movement threshold")
        require(movement.get("development_summary_path") is None, "development freeze cannot cite a nonexistent summary")
        require(movement.get("development_summary_sha256") is None, "development freeze cannot cite a nonexistent summary hash")
        require(freeze.get("development_final_consensus") is None, "development freeze cannot predeclare final consensus")
    else:
        require(movement.get("status") == "frozen_from_duplicate_development_labels", "confirmation movement threshold is not development-derived")
        threshold = require_finite(
            movement.get("threshold_relative_image_diagonal"),
            "confirmation movement threshold",
            minimum=0,
        )
        summary_path = _resolve_reference(base, movement.get("development_summary_path"), "development summary path")
        require(summary_path.is_file(), "development summary is missing")
        summary_sha = require_sha256(movement.get("development_summary_sha256"), "development summary sha256")
        require(sha256_file(summary_path) == summary_sha, "development summary file hash mismatch")
        summary = load_json(summary_path)
        require(summary.get("schema_version") == DEVELOPMENT_SUMMARY_SCHEMA, "development summary schema mismatch")
        verify_signed(summary, "development summary")
        require(summary.get("study_id") == STUDY_ID, "development summary study mismatch")
        source_files = summary.get("source_files")
        require(isinstance(source_files, dict), "development summary lacks revalidatable source files")
        _exact_keys(
            source_files,
            {"restricted_map", "rater_a_responses", "rater_b_responses"},
            "development summary source_files",
        )
        resolved_sources = {}
        for source_name, reference in source_files.items():
            require(isinstance(reference, dict), f"development summary source {source_name} is invalid")
            _exact_keys(reference, {"path", "artifact_sha256"}, f"development summary source {source_name}")
            source_path = _resolve_reference(summary_path.parent, reference.get("path"), f"development source {source_name} path")
            require_sha256(reference.get("artifact_sha256"), f"development source {source_name} hash")
            require(
                artifact_sha256(source_path) == reference["artifact_sha256"],
                f"development source {source_name} artifact hash mismatch",
            )
            resolved_sources[source_name] = source_path
        require(
            source_files["restricted_map"]["artifact_sha256"] == summary.get("restricted_map_sha256"),
            "development restricted-map hashes disagree",
        )
        require(
            source_files["rater_a_responses"]["artifact_sha256"] == summary.get("rater_a_response_sha256"),
            "development rater-a hashes disagree",
        )
        require(
            source_files["rater_b_responses"]["artifact_sha256"] == summary.get("rater_b_response_sha256"),
            "development rater-b hashes disagree",
        )
        recomputed_summary = summarize_development_labels(
            restricted_map_path=resolved_sources["restricted_map"],
            rater_a_response_path=resolved_sources["rater_a_responses"],
            rater_b_response_path=resolved_sources["rater_b_responses"],
        )
        require(recomputed_summary == summary, "development summary does not reproduce from its locked source artifacts")
        require(summary.get("cohort_branch") == cohort_branch, "confirmation cohort branch differs from development")
        require(summary.get("qualified_model_ids") == list(qualified_models), "confirmation qualified model set differs from development")
        require(
            summary.get("qualified_alignment_contract_sha256_by_model") == alignment_hashes,
            "confirmation alignment contracts differ from development",
        )
        require(summary.get("rubric_sha256") == rubric_sha, "confirmation rubric differs from validated development rubric")
        require(summary.get("eligible_duplicate_count", 0) > 0, "development summary has no usable duplicate labels")
        require(
            summary.get("movement_resolution_threshold_relative_image_diagonal") == threshold,
            "confirmation threshold differs from the development 95th percentile",
        )
        require(summary.get("quantile_probability") == 0.95, "development summary did not use the required 95th percentile")
        require(summary.get("quantile_method") == MOVEMENT_QUANTILE_METHOD, "development summary quantile method mismatch")
        decision = freeze.get("development_validation_decision")
        require(isinstance(decision, dict), "confirmation freeze lacks a development validation decision")
        _exact_keys(
            decision,
            {"measurement_usable", "decided_by", "decided_at", "basis", "development_summary_sha256"},
            "development validation decision",
        )
        require(decision.get("measurement_usable") is True, "development measurement was not approved as usable")
        require_nonempty_string(decision.get("decided_by"), "development decision decided_by")
        require_rfc3339_utc(decision.get("decided_at"), "development decision decided_at")
        require_nonempty_string(decision.get("basis"), "development decision basis")
        require(decision.get("development_summary_sha256") == summary_sha, "development decision summary hash mismatch")
        consensus_ref = freeze.get("development_final_consensus")
        require(isinstance(consensus_ref, dict), "confirmation freeze lacks final adjudicated development consensus")
        _exact_keys(consensus_ref, {"path", "sha256"}, "development final consensus reference")
        consensus_path = _resolve_reference(base, consensus_ref.get("path"), "development final consensus path")
        require(consensus_path.is_file(), "development final consensus is missing")
        consensus_sha = require_sha256(consensus_ref.get("sha256"), "development final consensus sha256")
        require(sha256_file(consensus_path) == consensus_sha, "development final consensus file hash mismatch")
        restricted_mapping = load_json(resolved_sources["restricted_map"])
        verify_signed(restricted_mapping, "restricted mapping")
        validate_final_consensus(
            load_json(consensus_path),
            consensus_path=consensus_path,
            development_summary=summary,
            development_summary_sha256=summary_sha,
            restricted_mapping=restricted_mapping,
        )

    return {
        "rubric": rubric,
        "rubric_path": rubric_path,
        "rubric_sha256": rubric_sha,
        "examples_path": examples_path,
        "examples_sha256": rubric["illustrated_examples"]["manifest_sha256"],
        "examples_manifest": examples_manifest,
        "example_files": example_files,
        "examples_blindness_path": examples_blindness_path,
        "rater_seeds": dict(seeds),
        "adjudicator_seed": adjudicator_seed,
        "cohort_branch": cohort_branch,
        "qualified_model_ids": list(qualified_models),
        "alignment_contract_sha256_by_model": dict(alignment_hashes),
        "development_summary": summary,
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    allowed_chunks = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"sRGB", b"gAMA", b"cHRM", b"pHYs"}
    width = height = None
    observed_chunks: list[bytes] = []
    with path.open("rb") as handle:
        require(handle.read(8) == b"\x89PNG\r\n\x1a\n", f"annotation media is not PNG: {path}")
        while True:
            length_bytes = handle.read(4)
            require(len(length_bytes) == 4, f"annotation media is truncated: {path}")
            length = struct.unpack(">I", length_bytes)[0]
            kind = handle.read(4)
            require(len(kind) == 4, f"annotation media is truncated: {path}")
            payload = handle.read(length)
            checksum = handle.read(4)
            require(len(payload) == length and len(checksum) == 4, f"annotation media is truncated: {path}")
            expected_crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
            observed_crc = struct.unpack(">I", checksum)[0]
            require(observed_crc == expected_crc, f"annotation media PNG CRC mismatch: {path}")
            require(kind in allowed_chunks, f"annotation media contains a disallowed metadata chunk {kind!r}: {path}")
            observed_chunks.append(kind)
            if kind == b"IHDR":
                require(len(observed_chunks) == 1 and len(payload) == 13, f"annotation media has invalid PNG IHDR: {path}")
                width, height = struct.unpack(">II", payload[:8])
            if kind == b"IEND":
                require(length == 0 and not handle.read(1), f"annotation media has invalid trailing PNG data: {path}")
                break
    require(observed_chunks and observed_chunks[0] == b"IHDR", f"annotation media has no leading PNG IHDR: {path}")
    require(b"IDAT" in observed_chunks and observed_chunks[-1] == b"IEND", f"annotation media has incomplete PNG chunks: {path}")
    require(width is not None and height is not None, f"annotation media has no dimensions: {path}")
    require(width > 0 and height > 0, f"annotation media has invalid dimensions: {path}")
    return width, height


def _selected_requests(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(selection.get("schema_version") == SELECTION_SCHEMA, "selection schema mismatch")
    verify_signed(selection, "selection")
    wrappers = selection.get("requests")
    require(isinstance(wrappers, list), "selection requests must be a list")
    sources = []
    for wrapper in wrappers:
        require(isinstance(wrapper, dict), "selection request wrapper is invalid")
        source = wrapper.get("source")
        require(isinstance(source, dict), "selection source request is invalid")
        sources.append(source)
    reconstructed_inventory = {
        "schema_version": REQUEST_INVENTORY_SCHEMA,
        "study_id": selection.get("study_id"),
        "stage": selection.get("stage"),
        "cohort_branch": selection.get("cohort_branch"),
        "inventory_complete": selection.get("inventory_complete"),
        "inventory_finalized_at": selection.get("inventory_finalized_at"),
        "annotation_state": selection.get("annotation_state_at_selection"),
        "episode_roster": selection.get("episode_roster"),
        "alignment_contracts": selection.get("alignment_contracts"),
        "requests": sources,
    }
    provenance = selection.get("provenance")
    require(isinstance(provenance, dict), "selection provenance is invalid")
    expected = select_requests(
        reconstructed_inventory,
        seed=selection.get("sampling", {}).get("seed"),
        cap_per_episode=selection.get("sampling", {}).get("cap_per_episode"),
        inventory_sha256=provenance.get("request_inventory_sha256"),
    )
    require(expected == selection, "selection manifest does not reproduce from its complete inventory")
    rows: dict[str, dict[str, Any]] = {}
    for wrapper in wrappers:
        source = wrapper["source"]
        if wrapper.get("selected"):
            request_id = source.get("source_request_id")
            require(request_id not in rows, "selected request is duplicated")
            rows[request_id] = source
    require(len(rows) == selection.get("counts", {}).get("selected"), "selection count mismatch")
    return rows


def _expected_roles(request: Mapping[str, Any]) -> set[str]:
    roles = {"current", "predicted", "executed"}
    if request["history_mode"] == "preceding_observation":
        roles.add("preceding")
    if request["early_horizon_supported"]:
        roles.update({"early_predicted", "early_executed"})
    return roles


def _validate_render_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    row: Mapping[str, Any],
    source_path: Path,
    media_path: Path,
) -> None:
    _exact_keys(receipt, RENDER_RECEIPT_KEYS, f"render receipt {row['render_receipt_id']}")
    require(receipt.get("schema_version") == RENDER_RECEIPT_SCHEMA, "render receipt schema mismatch")
    require(receipt.get("study_id") == STUDY_ID, "render receipt study mismatch")
    verify_signed(receipt, "render receipt")
    exact_fields = {
        "receipt_id": "render_receipt_id",
        "source_image_id": "source_image_id",
        "source_request_id": "source_request_id",
        "source_video_id": "source_video_id",
        "source_video_sha256": "source_video_sha256",
        "image_role": "image_role",
        "source_image_sha256": "source_image_sha256",
        "annotation_media_sha256": "annotation_media_sha256",
        "width_px": "width_px",
        "height_px": "height_px",
        "camera_id": "camera_id",
        "camera_crop_id": "camera_crop_id",
        "camera_crop_sha256": "camera_crop_sha256",
        "alignment_receipt_id": "alignment_receipt_id",
        "alignment_receipt_sha256": "alignment_receipt_sha256",
        "presentation_sanitized": "presentation_sanitized",
        "contains_overlay": "contains_overlay",
    }
    for receipt_key, row_key in exact_fields.items():
        require(
            receipt.get(receipt_key) == row.get(row_key),
            f"render receipt {receipt_key} differs from image inventory",
        )
    require(
        _resolve_reference(receipt_path.parent, receipt.get("source_image_path"), "render source path")
        == source_path,
        "render receipt source path differs from image inventory",
    )
    require(
        _resolve_reference(receipt_path.parent, receipt.get("annotation_media_path"), "render media path")
        == media_path,
        "render receipt media path differs from image inventory",
    )


def _validate_image_inventory(
    inventory: Mapping[str, Any],
    *,
    inventory_path: Path,
    selection: Mapping[str, Any],
    selection_path: Path,
) -> list[dict[str, Any]]:
    _exact_keys(
        inventory,
        {
            "schema_version",
            "study_id",
            "stage",
            "selection_manifest_sha256",
            "pixel_blindness_receipt_path",
            "pixel_blindness_receipt_sha256",
            "images",
        },
        "image inventory",
    )
    require(inventory.get("schema_version") == IMAGE_INVENTORY_SCHEMA, "image inventory schema mismatch")
    require(inventory.get("study_id") == STUDY_ID, "image inventory study mismatch")
    require(inventory.get("stage") == selection.get("stage"), "image inventory stage mismatch")
    require(
        inventory.get("selection_manifest_sha256") == sha256_file(selection_path),
        "image inventory selection file hash mismatch",
    )
    selected = _selected_requests(selection)
    alignment_by_model = {item["model_id"]: item for item in selection["alignment_contracts"]}
    rows = inventory.get("images")
    require(isinstance(rows, list), "image inventory images must be a list")
    by_request: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seen_images: set[str] = set()
    validated: list[dict[str, Any]] = []
    base = inventory_path.resolve().parent
    for raw in rows:
        require(isinstance(raw, dict), "image inventory row must be an object")
        _exact_keys(raw, IMAGE_KEYS, f"image {raw.get('source_image_id', '<unknown>')}")
        for key in (
            "source_image_id",
            "source_request_id",
            "source_video_id",
            "source_image_path",
            "camera_id",
            "camera_crop_id",
            "alignment_receipt_id",
            "render_receipt_id",
            "render_receipt_path",
        ):
            require_nonempty_string(raw[key], f"image {key}")
        for key in (
            "source_video_sha256",
            "source_image_sha256",
            "annotation_media_sha256",
            "alignment_receipt_sha256",
            "camera_crop_sha256",
            "render_receipt_sha256",
        ):
            require_sha256(raw[key], f"image {key}")
        image_id = raw["source_image_id"]
        require(image_id not in seen_images, f"duplicate source_image_id: {image_id}")
        seen_images.add(image_id)
        request_id = raw["source_request_id"]
        require(request_id in selected, f"image belongs to an unselected request: {request_id}")
        request = selected[request_id]
        require(raw["source_video_id"] == request["source_video_id"], "image source_video_id changed from request inventory")
        require(raw["source_video_sha256"] == request["source_video_sha256"], "image source_video_sha256 changed from request inventory")
        require(raw["camera_id"] == request["camera_id"], "image camera_id changed from request inventory")
        require(raw["camera_crop_id"] == request["camera_crop_id"], "image camera_crop_id changed from request inventory")
        require(raw["camera_crop_sha256"] == request["camera_crop_sha256"], "image camera_crop_sha256 changed from request inventory")
        require(raw["alignment_receipt_id"] == request["alignment_receipt_id"], "image alignment receipt id changed")
        require(raw["alignment_receipt_sha256"] == request["alignment_receipt_sha256"], "image alignment receipt hash changed")
        role = raw["image_role"]
        require(role in IMAGE_ROLES, f"invalid image_role: {role}")
        require(role not in by_request[request_id], f"duplicate image role {role} for {request_id}")
        require(raw["presentation_sanitized"] is True, "annotation image was not attested presentation-sanitized")
        require(raw["contains_overlay"] is False, "annotation image contains an identifying overlay")
        require(type(raw["width_px"]) is int and raw["width_px"] > 0, "image width_px is invalid")
        require(type(raw["height_px"]) is int and raw["height_px"] > 0, "image height_px is invalid")
        alignment = alignment_by_model[request["model_id"]]
        require(
            (raw["width_px"], raw["height_px"])
            == (alignment["image_width_px"], alignment["image_height_px"]),
            "annotation image dimensions differ from the frozen camera/crop contract",
        )
        source_path = _resolve_reference(base, raw["source_image_path"], "source image path")
        require(source_path.is_file(), f"source image is missing: {source_path}")
        require(sha256_file(source_path) == raw["source_image_sha256"], "source image hash mismatch")
        media_path = _resolve_reference(base, raw["annotation_media_path"], "annotation media path")
        require(media_path.is_file(), f"annotation media is missing: {media_path}")
        require(media_path.suffix.lower() == ".png", "annotation media must be uniformly lossless PNG")
        require(sha256_file(media_path) == raw["annotation_media_sha256"], "annotation media hash mismatch")
        require(_png_dimensions(media_path) == (raw["width_px"], raw["height_px"]), "annotation media dimensions mismatch")
        receipt_path = _resolve_reference(base, raw["render_receipt_path"], "render receipt path")
        require(receipt_path.is_file(), f"render receipt is missing: {receipt_path}")
        require(sha256_file(receipt_path) == raw["render_receipt_sha256"], "render receipt file hash mismatch")
        _validate_render_receipt(
            load_json(receipt_path),
            receipt_path=receipt_path,
            row=raw,
            source_path=source_path,
            media_path=media_path,
        )
        row = dict(raw)
        row["source_image_path"] = str(source_path)
        row["annotation_media_path"] = str(media_path)
        row["render_receipt_path"] = str(receipt_path)
        by_request[request_id][role] = row
        validated.append(row)

    require(set(by_request) == set(selected), "one or more selected requests have no annotation images")
    for request_id, request in selected.items():
        observed = set(by_request[request_id])
        expected = _expected_roles(request)
        require(observed == expected, f"image roles for {request_id} are {sorted(observed)}, expected {sorted(expected)}")
    blindness_path = _resolve_reference(base, inventory.get("pixel_blindness_receipt_path"), "annotation pixel-blindness receipt path")
    require(blindness_path.is_file(), "annotation pixel-blindness receipt is missing")
    blindness_sha = require_sha256(inventory.get("pixel_blindness_receipt_sha256"), "annotation pixel-blindness receipt sha256")
    require(sha256_file(blindness_path) == blindness_sha, "annotation pixel-blindness receipt hash mismatch")
    _validate_pixel_blindness_receipt(
        load_json(blindness_path),
        receipt_path=blindness_path,
        stage=inventory["stage"],
        artifact_scope="annotation_media",
        expected_assets=[
            (row["annotation_media_sha256"], row["width_px"], row["height_px"])
            for row in validated
        ],
    )
    return validated


def _opaque_id(seed: int, slot: str, media_sha256: str) -> str:
    payload = f"{seed}\0{slot}\0{media_sha256}".encode("utf-8")
    return "img_" + sha256_bytes(payload)[:24]


def _blind_batches(assets: Sequence[dict[str, Any]], seed: int) -> list[list[dict[str, Any]]]:
    """Partition images so a released batch never contains a counterpart image."""

    require(assets, "at least one distinct image is required for a blinded packet")
    rng = random.Random(seed)
    ordered = list(assets)
    rng.shuffle(ordered)
    batches: list[list[dict[str, Any]]] = []
    request_ids: list[set[str]] = []
    for asset in ordered:
        candidates = [
            index
            for index, occupied in enumerate(request_ids)
            if occupied.isdisjoint(asset["source_request_ids"])
        ]
        if not candidates:
            batches.append([])
            request_ids.append(set())
            candidates = [len(batches) - 1]
        smallest = min(len(batches[index]) for index in candidates)
        choices = [index for index in candidates if len(batches[index]) == smallest]
        chosen = rng.choice(choices)
        batches[chosen].append(asset)
        request_ids[chosen].update(asset["source_request_ids"])
    rng.shuffle(batches)
    for batch in batches:
        rng.shuffle(batch)
        flattened = [request_id for asset in batch for request_id in asset["source_request_ids"]]
        require(len(flattened) == len(set(flattened)), "blind batch contains counterpart images")
    return batches


def _response_template(packet: Mapping[str, Any]) -> dict[str, Any]:
    annotations = []
    for item in packet["items"]:
        annotations.append(
            {
                "opaque_image_id": item["opaque_image_id"],
                "cube_resolvability": None,
                "bowl_resolvability": None,
                "cube_identity": None,
                "bowl_identity": None,
                "cube_center_px": None,
                "bowl_center_px": None,
                "bowl_width_px": None,
                "ambiguity_codes": [],
                "source_guess": None,
                "annotation_seconds": None,
                "notes": "",
            }
        )
    attestations = ADJUDICATOR_ATTESTATIONS if packet["rater_slot"] == "adjudicator" else RESPONSE_ATTESTATIONS
    return {
        "schema_version": RATER_RESPONSE_SCHEMA,
        "study_id": STUDY_ID,
        "packet_id": packet["packet_id"],
        "packet_manifest_sha256": None,
        "rater_slot": packet["rater_slot"],
        "rater_code": None,
        "attestations": {key: False for key in sorted(attestations)},
        "started_at": None,
        "completed_at": None,
        "locked": False,
        "locked_at": None,
        "annotations": annotations,
    }


def package_packets(
    *,
    selection_path: Path,
    image_inventory_path: Path,
    freeze_path: Path,
    packet_root: Path,
    restricted_map_path: Path,
) -> dict[str, Any]:
    """Create two source-free, sequentially released batch streams and a restricted map."""

    require(not packet_root.exists(), "packet output already exists")
    require(not restricted_map_path.exists(), "restricted mapping output already exists")
    packet_resolved = packet_root.resolve()
    restricted_resolved = restricted_map_path.resolve()
    restricted_parent = restricted_resolved.parent
    require(packet_resolved not in restricted_resolved.parents, "restricted map cannot be inside a rater packet")
    require(restricted_parent not in packet_resolved.parents, "rater packets cannot be inside the restricted-map directory")
    selection = load_json(selection_path)
    selected = _selected_requests(selection)
    stage = selection.get("stage")
    freeze = load_json(freeze_path)
    freeze_info = validate_freeze(freeze, freeze_path=freeze_path, stage=stage)
    require(
        set(freeze_info["alignment_contract_sha256_by_model"])
        == {contract["model_id"] for contract in selection["alignment_contracts"]},
        "annotation freeze alignment-contract models differ from the selected cohort branch",
    )
    for contract in selection["alignment_contracts"]:
        require(
            freeze_info["alignment_contract_sha256_by_model"].get(contract["model_id"])
            == contract["contract_sha256"],
            f"{contract['model_id']} selection alignment contract is not the one bound by the annotation freeze",
        )
    inventory = load_json(image_inventory_path)
    image_rows = _validate_image_inventory(
        inventory,
        inventory_path=image_inventory_path,
        selection=selection,
        selection_path=selection_path,
    )

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        grouped[(row["annotation_media_sha256"], row["width_px"], row["height_px"])].append(row)
    assets: list[dict[str, Any]] = []
    for index, ((media_sha, width, height), source_rows) in enumerate(sorted(grouped.items())):
        paths = {row["annotation_media_path"] for row in source_rows}
        # Same bytes may originate at different paths; use one verified representative.
        media_path = sorted(paths)[0]
        source_rows = sorted(source_rows, key=lambda row: (row["source_request_id"], row["image_role"], row["source_image_id"]))
        assets.append(
            {
                "restricted_asset_id": f"asset_{index:06d}",
                "media_sha256": media_sha,
                "media_path": media_path,
                "width_px": width,
                "height_px": height,
                "source_request_ids": sorted({row["source_request_id"] for row in source_rows}),
                "source_records": source_rows,
            }
        )
    require(assets, "no annotation assets were produced")

    packet_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{packet_root.name}.", dir=packet_root.parent))
    packet_metadata: dict[str, dict[str, Any]] = {}
    presentations: dict[str, dict[str, dict[str, str]]] = {}
    restricted_written = False
    try:
        source_orders: dict[str, list[str]] = {}
        for slot in ("rater_a", "rater_b"):
            seed = freeze_info["rater_seeds"][slot]
            batches = _blind_batches(assets, seed)
            source_orders[slot] = [asset["restricted_asset_id"] for batch in batches for asset in batch]
            slot_dir = temporary_root / slot
            slot_dir.mkdir(parents=True)
            examples_name = "illustrated_examples.json"
            require(
                freeze_info["rubric"]["illustrated_examples"]["manifest_path"] == examples_name,
                "rubric example manifest name changes when normalized",
            )
            presentations[slot] = {}
            batch_metadata = []
            for batch_index, ordered in enumerate(batches, start=1):
                batch_token = sha256_bytes(f"{STUDY_ID}\0{stage}\0{slot}\0{seed}\0{batch_index}".encode("utf-8"))[:16]
                batch_dir = slot_dir / f"batch_{batch_token}"
                media_dir = batch_dir / "media"
                media_dir.mkdir(parents=True)
                shutil.copyfile(freeze_info["rubric_path"], batch_dir / "rubric.json")
                shutil.copyfile(freeze_info["examples_path"], batch_dir / examples_name)
                blindness_relative = freeze_info["examples_blindness_path"].relative_to(
                    freeze_info["examples_path"].parent
                )
                blindness_destination = batch_dir / blindness_relative
                blindness_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(freeze_info["examples_blindness_path"], blindness_destination)
                for source_example in freeze_info["example_files"]:
                    relative_example = source_example.relative_to(freeze_info["examples_path"].parent)
                    destination_example = batch_dir / relative_example
                    destination_example.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_example, destination_example)
                packet_id = f"packet_{batch_token}"
                items = []
                for presentation_index, asset in enumerate(ordered, start=1):
                    opaque = _opaque_id(seed, slot, asset["media_sha256"])
                    media_name = f"{opaque}.png"
                    shutil.copyfile(asset["media_path"], media_dir / media_name)
                    items.append(
                        {
                            "presentation_index": presentation_index,
                            "opaque_image_id": opaque,
                            "media_file": f"media/{media_name}",
                            "media_sha256": asset["media_sha256"],
                            "width_px": asset["width_px"],
                            "height_px": asset["height_px"],
                        }
                    )
                    presentations[slot][asset["restricted_asset_id"]] = {
                        "opaque_image_id": opaque,
                        "packet_id": packet_id,
                    }
                require(len({item["opaque_image_id"] for item in items}) == len(items), "opaque image ID collision")
                packet = {
                    "schema_version": RATER_PACKET_SCHEMA,
                    "study_id": STUDY_ID,
                    "packet_id": packet_id,
                    "rater_slot": slot,
                    "instructions": (
                        "Annotate this isolated batch using rubric.json. Do not retain or request any prior or "
                        "later batch, counterpart, model, condition, instruction, success, source-path, or video identity information."
                    ),
                    "rubric_file": "rubric.json",
                    "rubric_sha256": freeze_info["rubric_sha256"],
                    "illustrated_examples_file": examples_name,
                    "illustrated_examples_sha256": freeze_info["examples_sha256"],
                    "item_count": len(items),
                    "items": items,
                }
                packet_path = batch_dir / "packet.json"
                atomic_write_json(packet_path, packet)
                packet_sha = sha256_file(packet_path)
                response = _response_template(packet)
                response["packet_manifest_sha256"] = packet_sha
                atomic_write_json(batch_dir / "response_template.json", response)
                batch_metadata.append(
                    {
                        "packet_id": packet_id,
                        "packet_manifest_sha256": packet_sha,
                        "packet_relative_path": f"{slot}/{batch_dir.name}/packet.json",
                        "restricted_asset_ids": [asset["restricted_asset_id"] for asset in ordered],
                    }
                )
            packet_metadata[slot] = {
                "delivery_rule": "ONE_BATCH_AT_A_TIME_WITH_PRIOR_BATCH_COLLECTED_AND_REVOKED",
                "batch_count": len(batch_metadata),
                "batches": batch_metadata,
            }
        require(source_orders["rater_a"] != source_orders["rater_b"], "two rater orders are identical; freeze different seeds")

        mapping_assets = []
        for asset in assets:
            source_records = []
            for row in asset["source_records"]:
                request = selected[row["source_request_id"]]
                source_records.append(
                    {
                        **row,
                        "model_id": request["model_id"],
                        "layout_pair_id": request["layout_pair_id"],
                        "condition_id": request["condition_id"],
                        "episode_id": request["episode_id"],
                        "request_index": request["request_index"],
                    }
                )
            mapping_assets.append(
                {
                    "restricted_asset_id": asset["restricted_asset_id"],
                    "annotation_media_sha256": asset["media_sha256"],
                    "width_px": asset["width_px"],
                    "height_px": asset["height_px"],
                    "source_records": source_records,
                    "rater_opaque_ids": {
                        slot: presentations[slot][asset["restricted_asset_id"]]["opaque_image_id"]
                        for slot in ("rater_a", "rater_b")
                    },
                    "rater_packet_ids": {
                        slot: presentations[slot][asset["restricted_asset_id"]]["packet_id"]
                        for slot in ("rater_a", "rater_b")
                    },
                }
            )
        restricted = {
            "schema_version": RESTRICTED_MAP_SCHEMA,
            "study_id": STUDY_ID,
            "stage": stage,
            "cohort_branch": selection["cohort_branch"],
            "qualified_model_ids": list(_qualified_models(selection["cohort_branch"])),
            "qualified_alignment_contract_sha256_by_model": {
                contract["model_id"]: contract["contract_sha256"]
                for contract in selection["alignment_contracts"]
            },
            "visibility": "RESTRICTED ANALYST IDENTITY MAP; NEVER DISTRIBUTE TO RATERS OR BLIND ADJUDICATORS",
            "selection_manifest_sha256": sha256_file(selection_path),
            "image_inventory_sha256": sha256_file(image_inventory_path),
            "freeze_sha256": sha256_file(freeze_path),
            "rubric_sha256": freeze_info["rubric_sha256"],
            "rubric_ambiguity_codes": [item["code"] for item in freeze_info["rubric"]["ambiguity_codes"]],
            "source_guess_options": freeze_info["rubric"]["source_guess_options"],
            "adjudicator_seed": freeze_info["adjudicator_seed"],
            "deduplication": "Exact annotation PNG SHA256 plus pixel dimensions; all source identities retained.",
            "packet_root": str(packet_resolved),
            "asset_count": len(mapping_assets),
            "source_record_count": len(image_rows),
            "packets": packet_metadata,
            "assets": mapping_assets,
        }
        restricted = sign_document(restricted)
        restricted_map_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(restricted_map_path, restricted, mode=0o600)
        restricted_written = True
        os.replace(temporary_root, packet_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        if restricted_written and not packet_root.exists():
            restricted_map_path.unlink(missing_ok=True)
        raise

    return {
        "stage": stage,
        "selected_requests": len(selected),
        "source_image_records": len(image_rows),
        "unique_packet_images": len(assets),
        "rater_packets": 2,
        "rater_batches": {slot: metadata["batch_count"] for slot, metadata in packet_metadata.items()},
        "restricted_map_sha256": sha256_file(restricted_map_path),
    }


def _validate_point(value: Any, width: int, height: int, label: str) -> tuple[float, float]:
    require(isinstance(value, list) and len(value) == 2, f"{label} must be [x,y]")
    x = require_finite(value[0], f"{label} x", minimum=0)
    y = require_finite(value[1], f"{label} y", minimum=0)
    require(x <= width - 1 and y <= height - 1, f"{label} lies outside the image")
    return x, y


def _validate_response(
    response: Mapping[str, Any],
    *,
    mapping: Mapping[str, Any],
    slot: str,
    packet: Mapping[str, Any],
    dimensions: Mapping[str, tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    _exact_keys(response, RESPONSE_KEYS, f"{slot} response")
    require(response.get("schema_version") == RATER_RESPONSE_SCHEMA, f"{slot} response schema mismatch")
    require(response.get("study_id") == STUDY_ID, f"{slot} response study mismatch")
    require(response.get("packet_id") == packet["packet_id"], f"{slot} packet id mismatch")
    require(response.get("packet_manifest_sha256") == packet["packet_manifest_sha256"], f"{slot} packet hash mismatch")
    require(response.get("rater_slot") == slot, f"{slot} response slot mismatch")
    require(slot in {"rater_a", "rater_b", "adjudicator"}, "response slot is invalid")
    rater_code = require_nonempty_string(response.get("rater_code"), f"{slot} rater_code")
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", rater_code) is not None,
        f"{slot} rater_code must be a canonical lowercase identifier",
    )
    attestations = response.get("attestations")
    expected_attestations = ADJUDICATOR_ATTESTATIONS if slot == "adjudicator" else RESPONSE_ATTESTATIONS
    require(isinstance(attestations, dict) and set(attestations) == expected_attestations, f"{slot} attestations changed")
    require(all(value is True for value in attestations.values()), f"{slot} independence/blinding attestations are incomplete")
    started_at = require_rfc3339_utc(response.get("started_at"), f"{slot} started_at")
    completed_at = require_rfc3339_utc(response.get("completed_at"), f"{slot} completed_at")
    require(response.get("locked") is True, f"{slot} response is not locked")
    locked_at = require_rfc3339_utc(response.get("locked_at"), f"{slot} locked_at")
    require(started_at <= completed_at <= locked_at, f"{slot} response timestamps are out of order")
    annotations = response.get("annotations")
    require(isinstance(annotations, list), f"{slot} annotations must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    allowed_ambiguity = set(mapping["rubric_ambiguity_codes"])
    allowed_guesses = set(mapping["source_guess_options"])
    for annotation in annotations:
        require(isinstance(annotation, dict), f"{slot} annotation must be an object")
        _exact_keys(annotation, ANNOTATION_KEYS, f"{slot} annotation")
        opaque = annotation.get("opaque_image_id")
        require(opaque in dimensions, f"{slot} response has an unknown opaque image id")
        require(opaque not in by_id, f"{slot} response duplicates an opaque image id")
        width, height = dimensions[opaque]
        cube_resolvability = annotation.get("cube_resolvability")
        bowl_resolvability = annotation.get("bowl_resolvability")
        require(cube_resolvability in {"resolvable", "unknown"}, f"{slot} cube_resolvability is invalid")
        require(bowl_resolvability in {"resolvable", "unknown"}, f"{slot} bowl_resolvability is invalid")
        require(annotation.get("cube_identity") in {"rubiks_cube", "unknown"}, f"{slot} cube_identity is invalid")
        require(annotation.get("bowl_identity") in {"bowl", "unknown"}, f"{slot} bowl_identity is invalid")
        ambiguity = annotation.get("ambiguity_codes")
        require(isinstance(ambiguity, list) and ambiguity, f"{slot} ambiguity_codes must be nonempty")
        require(len(ambiguity) == len(set(ambiguity)), f"{slot} ambiguity_codes are duplicated")
        require(set(ambiguity) <= allowed_ambiguity, f"{slot} ambiguity code is outside the frozen rubric")
        if cube_resolvability == "resolvable":
            require(annotation["cube_identity"] == "rubiks_cube", f"{slot} resolvable cube identity is not confirmed")
            _validate_point(annotation["cube_center_px"], width, height, f"{slot} cube_center_px")
        else:
            require(annotation["cube_identity"] == "unknown", f"{slot} unresolved cube identity must be unknown")
            require(annotation["cube_center_px"] is None, f"{slot} unresolved cube must not have coordinates")
        if bowl_resolvability == "resolvable":
            require(annotation["bowl_identity"] == "bowl", f"{slot} resolvable bowl identity is not confirmed")
            _validate_point(annotation["bowl_center_px"], width, height, f"{slot} bowl_center_px")
            bowl_width = require_finite(annotation["bowl_width_px"], f"{slot} bowl_width_px", minimum=0)
            require(0 < bowl_width <= width, f"{slot} bowl_width_px is outside the image")
        else:
            require(annotation["bowl_identity"] == "unknown", f"{slot} unresolved bowl identity must be unknown")
            require(annotation["bowl_center_px"] is None, f"{slot} unresolved bowl must not have coordinates")
            require(annotation["bowl_width_px"] is None, f"{slot} unresolved bowl must not have bowl width")
        if cube_resolvability == bowl_resolvability == "resolvable":
            require(ambiguity == ["none"], f"{slot} resolvable annotation must use only ambiguity code none")
        else:
            require("none" not in ambiguity, f"{slot} partially unresolved annotation cannot use ambiguity code none")
        require(annotation.get("source_guess") in allowed_guesses, f"{slot} source_guess is invalid")
        require_finite(annotation.get("annotation_seconds"), f"{slot} annotation_seconds", minimum=0)
        require(isinstance(annotation.get("notes"), str), f"{slot} notes must be a string")
        by_id[opaque] = dict(annotation)
    require(set(by_id) == set(dimensions), f"{slot} response is incomplete")
    return by_id


def _validate_packet_artifacts(
    mapping: Mapping[str, Any],
    *,
    slot: str,
    batch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    packet_root = Path(require_nonempty_string(mapping.get("packet_root"), "restricted packet_root")).resolve()
    require(packet_root.is_dir(), "packet root is missing")
    packet_path = _resolve_reference(packet_root, batch.get("packet_relative_path"), "packet path")
    require(packet_path.is_file() and packet_root in packet_path.parents, "packet path escapes or is missing from packet root")
    require(sha256_file(packet_path) == batch.get("packet_manifest_sha256"), "packet manifest hash mismatch")
    packet = load_json(packet_path)
    _exact_keys(packet, RATER_PACKET_KEYS, f"{slot} packet")
    require(packet.get("schema_version") == RATER_PACKET_SCHEMA, f"{slot} packet schema mismatch")
    require(packet.get("study_id") == STUDY_ID, f"{slot} packet study mismatch")
    require(packet.get("packet_id") == batch.get("packet_id"), f"{slot} packet id changed")
    require(packet.get("rater_slot") == slot, f"{slot} packet slot changed")
    require(packet.get("rubric_sha256") == mapping.get("rubric_sha256"), f"{slot} packet rubric changed")
    rubric_path = _resolve_reference(packet_path.parent, packet.get("rubric_file"), "packet rubric path")
    require(rubric_path.is_file() and sha256_file(rubric_path) == packet["rubric_sha256"], f"{slot} packet rubric file hash mismatch")
    examples_path = _resolve_reference(packet_path.parent, packet.get("illustrated_examples_file"), "packet examples path")
    require(
        examples_path.is_file() and sha256_file(examples_path) == packet.get("illustrated_examples_sha256"),
        f"{slot} packet illustrated examples hash mismatch",
    )
    _validate_example_manifest(load_json(examples_path), manifest_path=examples_path)
    items = packet.get("items")
    require(isinstance(items, list) and len(items) == packet.get("item_count"), f"{slot} packet item count mismatch")
    asset_by_id = {asset["restricted_asset_id"]: asset for asset in mapping["assets"]}
    expected_assets = batch.get("restricted_asset_ids")
    require(isinstance(expected_assets, list) and expected_assets, f"{slot} batch has no restricted assets")
    require(len(expected_assets) == len(set(expected_assets)), f"{slot} batch repeats a restricted asset")
    require(all(asset_id in asset_by_id for asset_id in expected_assets), f"{slot} batch contains an unknown restricted asset")
    expected_by_opaque = {
        asset_by_id[asset_id]["rater_opaque_ids"][slot]: asset_by_id[asset_id]
        for asset_id in expected_assets
    }
    dimensions: dict[str, tuple[int, int]] = {}
    indexes = []
    for item in items:
        require(isinstance(item, dict), f"{slot} packet item is invalid")
        _exact_keys(item, RATER_PACKET_ITEM_KEYS, f"{slot} packet item")
        opaque = item.get("opaque_image_id")
        require(opaque in expected_by_opaque and opaque not in dimensions, f"{slot} packet opaque id changed or repeats")
        asset = expected_by_opaque[opaque]
        require(asset["rater_packet_ids"][slot] == packet["packet_id"], f"{slot} asset packet binding changed")
        require(item.get("media_sha256") == asset["annotation_media_sha256"], f"{slot} packet media hash differs from restricted map")
        require((item.get("width_px"), item.get("height_px")) == (asset["width_px"], asset["height_px"]), f"{slot} packet dimensions changed")
        media_path = _resolve_reference(packet_path.parent, item.get("media_file"), "packet media path")
        require(media_path.is_file() and packet_path.parent in media_path.parents, f"{slot} packet media escapes or is missing")
        require(sha256_file(media_path) == item["media_sha256"], f"{slot} packet media hash mismatch")
        require(_png_dimensions(media_path) == (asset["width_px"], asset["height_px"]), f"{slot} packet media dimensions mismatch")
        indexes.append(item.get("presentation_index"))
        dimensions[opaque] = (asset["width_px"], asset["height_px"])
    require(set(dimensions) == set(expected_by_opaque), f"{slot} packet omits a restricted asset")
    require(indexes == list(range(1, len(items) + 1)), f"{slot} packet presentation indexes changed")
    return packet, dimensions


def _validate_response_set(
    response_directory: Path,
    *,
    mapping: Mapping[str, Any],
    slot: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    response_directory = response_directory.resolve()
    require(response_directory.is_dir(), f"{slot} response path must be a directory of locked batch responses")
    response_files = sorted(response_directory.glob("*.json"))
    batches = mapping.get("packets", {}).get(slot, {}).get("batches")
    require(isinstance(batches, list) and batches, f"{slot} restricted packet list is invalid")
    require(len(response_files) == len(batches), f"{slot} response directory must contain exactly one JSON response per batch")
    response_by_packet = {}
    for response_path in response_files:
        response = load_json(response_path)
        packet_id = response.get("packet_id")
        require(packet_id not in response_by_packet, f"{slot} response directory repeats a packet")
        response_by_packet[packet_id] = response
    combined: dict[str, dict[str, Any]] = {}
    rater_codes: set[str] = set()
    for batch in batches:
        packet, dimensions = _validate_packet_artifacts(mapping, slot=slot, batch=batch)
        require(packet["packet_id"] in response_by_packet, f"{slot} response directory omits a packet")
        response = response_by_packet[packet["packet_id"]]
        validated = _validate_response(
            response,
            mapping=mapping,
            slot=slot,
            packet=batch,
            dimensions=dimensions,
        )
        require(not (set(combined) & set(validated)), f"{slot} response repeats an image across batches")
        combined.update(validated)
        rater_codes.add(response["rater_code"])
    require(len(rater_codes) == 1, f"{slot} batch responses must all identify the same rater")
    return combined, next(iter(rater_codes))


def quantile_type7(values: Sequence[float], probability: float) -> float:
    require(values, "cannot compute a quantile from no values")
    require(0 <= probability <= 1, "quantile probability is invalid")
    ordered = sorted(require_finite(value, "quantile value") for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _consensus_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: annotation[key] for key in CONSENSUS_ANNOTATION_KEYS}
    result["ambiguity_codes"] = sorted(result["ambiguity_codes"])
    return result


def _validate_consensus_annotation(
    annotation: Mapping[str, Any],
    *,
    width: int,
    height: int,
    allowed_ambiguity: set[str],
) -> dict[str, Any]:
    require(isinstance(annotation, dict), "final consensus annotation must be an object")
    _exact_keys(annotation, CONSENSUS_ANNOTATION_KEYS, "final consensus annotation")
    cube_status = annotation.get("cube_resolvability")
    bowl_status = annotation.get("bowl_resolvability")
    require(cube_status in {"resolvable", "unknown"}, "final cube_resolvability is invalid")
    require(bowl_status in {"resolvable", "unknown"}, "final bowl_resolvability is invalid")
    require(annotation.get("cube_identity") in {"rubiks_cube", "unknown"}, "final cube_identity is invalid")
    require(annotation.get("bowl_identity") in {"bowl", "unknown"}, "final bowl_identity is invalid")
    ambiguity = annotation.get("ambiguity_codes")
    require(isinstance(ambiguity, list) and ambiguity, "final ambiguity_codes must be nonempty")
    require(len(ambiguity) == len(set(ambiguity)) and set(ambiguity) <= allowed_ambiguity, "final ambiguity_codes are invalid")
    if cube_status == "resolvable":
        require(annotation["cube_identity"] == "rubiks_cube", "final resolvable cube identity is unconfirmed")
        _validate_point(annotation["cube_center_px"], width, height, "final cube_center_px")
    else:
        require(annotation["cube_identity"] == "unknown", "final unresolved cube identity must be unknown")
        require(annotation["cube_center_px"] is None, "final unresolved cube has coordinates")
    if bowl_status == "resolvable":
        require(annotation["bowl_identity"] == "bowl", "final resolvable bowl identity is unconfirmed")
        _validate_point(annotation["bowl_center_px"], width, height, "final bowl_center_px")
        bowl_width = require_finite(annotation["bowl_width_px"], "final bowl_width_px", minimum=0)
        require(0 < bowl_width <= width, "final bowl_width_px is invalid")
    else:
        require(annotation["bowl_identity"] == "unknown", "final unresolved bowl identity must be unknown")
        require(annotation["bowl_center_px"] is None and annotation["bowl_width_px"] is None, "final unresolved bowl has geometry")
    if cube_status == bowl_status == "resolvable":
        require(sorted(ambiguity) == ["none"], "final resolvable annotation must use only none")
    else:
        require("none" not in ambiguity, "final unresolved annotation cannot use none")
    return _consensus_annotation(annotation)


def _load_first_pass_context(
    *,
    restricted_map_path: Path,
    rater_a_response_path: Path,
    rater_b_response_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], str, str]:
    mapping = load_json(restricted_map_path)
    require(mapping.get("schema_version") == RESTRICTED_MAP_SCHEMA, "restricted mapping schema mismatch")
    verify_signed(mapping, "restricted mapping")
    require(mapping.get("study_id") == STUDY_ID, "restricted mapping study mismatch")
    require(mapping.get("stage") in {"development", "confirmation"}, "restricted mapping stage is invalid")
    annotations_a, rater_a_code = _validate_response_set(rater_a_response_path, mapping=mapping, slot="rater_a")
    annotations_b, rater_b_code = _validate_response_set(rater_b_response_path, mapping=mapping, slot="rater_b")
    require(rater_a_code != rater_b_code, "the two response sets identify the same rater")
    return mapping, annotations_a, annotations_b, rater_a_code, rater_b_code


def package_adjudication(
    *,
    restricted_map_path: Path,
    rater_a_response_path: Path,
    rater_b_response_path: Path,
    packet_root: Path,
    adjudication_map_path: Path,
) -> dict[str, Any]:
    """Build source-free packets containing exactly the first-pass disagreements."""

    require(not packet_root.exists(), "adjudication packet output already exists")
    require(not adjudication_map_path.exists(), "adjudication map output already exists")
    packet_resolved = packet_root.resolve()
    map_resolved = adjudication_map_path.resolve()
    require(packet_resolved not in map_resolved.parents, "adjudication map cannot be inside its packet root")
    require(
        map_resolved.parent not in packet_resolved.parents,
        "adjudication packets cannot be inside the restricted-map directory",
    )
    mapping, annotations_a, annotations_b, code_a, code_b = _load_first_pass_context(
        restricted_map_path=restricted_map_path,
        rater_a_response_path=rater_a_response_path,
        rater_b_response_path=rater_b_response_path,
    )
    source_assets = {asset["restricted_asset_id"]: asset for asset in mapping["assets"]}
    disagreements = []
    for asset in mapping["assets"]:
        left = _consensus_annotation(annotations_a[asset["rater_opaque_ids"]["rater_a"]])
        right = _consensus_annotation(annotations_b[asset["rater_opaque_ids"]["rater_b"]])
        if left != right:
            media_path = Path(asset["source_records"][0]["annotation_media_path"]).resolve()
            require(media_path.is_file(), "adjudication source media is missing")
            require(sha256_file(media_path) == asset["annotation_media_sha256"], "adjudication source media hash mismatch")
            require(_png_dimensions(media_path) == (asset["width_px"], asset["height_px"]), "adjudication source media dimensions mismatch")
            disagreements.append(
                {
                    "restricted_asset_id": asset["restricted_asset_id"],
                    "media_path": str(media_path),
                    "media_sha256": asset["annotation_media_sha256"],
                    "width_px": asset["width_px"],
                    "height_px": asset["height_px"],
                    "source_request_ids": sorted(
                        {record["source_request_id"] for record in asset["source_records"]}
                    ),
                }
            )

    source_batch = mapping["packets"]["rater_a"]["batches"][0]
    source_packet, _ = _validate_packet_artifacts(mapping, slot="rater_a", batch=source_batch)
    source_packet_path = _resolve_reference(Path(mapping["packet_root"]), source_batch["packet_relative_path"], "source packet path")
    rubric_path = _resolve_reference(source_packet_path.parent, source_packet["rubric_file"], "source packet rubric")
    examples_path = _resolve_reference(source_packet_path.parent, source_packet["illustrated_examples_file"], "source packet examples")
    example_files, examples_blindness_path = _validate_example_manifest(load_json(examples_path), manifest_path=examples_path)

    packet_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{packet_root.name}.", dir=packet_root.parent))
    map_written = False
    try:
        seed = mapping.get("adjudicator_seed")
        require(type(seed) is int and seed >= 0, "restricted map lacks a frozen adjudicator seed")
        batches = _blind_batches(disagreements, seed) if disagreements else []
        batch_metadata = []
        presentations = {}
        for batch_index, ordered in enumerate(batches, start=1):
            token = sha256_bytes(f"{STUDY_ID}\0{mapping['stage']}\0adjudicator\0{seed}\0{batch_index}".encode())[:16]
            batch_dir = temporary_root / f"batch_{token}"
            media_dir = batch_dir / "media"
            media_dir.mkdir(parents=True)
            shutil.copyfile(rubric_path, batch_dir / "rubric.json")
            shutil.copyfile(examples_path, batch_dir / "illustrated_examples.json")
            for source in [*example_files, examples_blindness_path]:
                relative = source.relative_to(examples_path.parent)
                destination = batch_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            packet_id = f"adjudication_packet_{token}"
            items = []
            for presentation_index, asset in enumerate(ordered, start=1):
                opaque = _opaque_id(seed, "adjudicator", asset["media_sha256"])
                media_file = f"media/{opaque}.png"
                shutil.copyfile(asset["media_path"], batch_dir / media_file)
                items.append(
                    {
                        "presentation_index": presentation_index,
                        "opaque_image_id": opaque,
                        "media_file": media_file,
                        "media_sha256": asset["media_sha256"],
                        "width_px": asset["width_px"],
                        "height_px": asset["height_px"],
                    }
                )
                presentations[asset["restricted_asset_id"]] = {
                    "opaque_image_id": opaque,
                    "packet_id": packet_id,
                }
            packet = {
                "schema_version": RATER_PACKET_SCHEMA,
                "study_id": STUDY_ID,
                "packet_id": packet_id,
                "rater_slot": "adjudicator",
                "instructions": (
                    "Independently annotate each isolated image using rubric.json. Do not request first-pass labels, "
                    "counterparts, model, condition, instruction, outcome, source path, or video identity."
                ),
                "rubric_file": "rubric.json",
                "rubric_sha256": mapping["rubric_sha256"],
                "illustrated_examples_file": "illustrated_examples.json",
                "illustrated_examples_sha256": sha256_file(examples_path),
                "item_count": len(items),
                "items": items,
            }
            packet_path = batch_dir / "packet.json"
            atomic_write_json(packet_path, packet)
            packet_sha = sha256_file(packet_path)
            response = _response_template(packet)
            response["packet_manifest_sha256"] = packet_sha
            atomic_write_json(batch_dir / "response_template.json", response)
            batch_metadata.append(
                {
                    "packet_id": packet_id,
                    "packet_manifest_sha256": packet_sha,
                    "packet_relative_path": f"batch_{token}/packet.json",
                    "restricted_asset_ids": [asset["restricted_asset_id"] for asset in ordered],
                }
            )
        adjudication_assets = []
        for asset in disagreements:
            presentation = presentations[asset["restricted_asset_id"]]
            adjudication_assets.append(
                {
                    "restricted_asset_id": asset["restricted_asset_id"],
                    "annotation_media_sha256": asset["media_sha256"],
                    "width_px": asset["width_px"],
                    "height_px": asset["height_px"],
                    "source_request_ids": asset["source_request_ids"],
                    "rater_opaque_ids": {"adjudicator": presentation["opaque_image_id"]},
                    "rater_packet_ids": {"adjudicator": presentation["packet_id"]},
                }
            )
        adjudication_map = sign_document(
            {
                "schema_version": ADJUDICATION_MAP_SCHEMA,
                "study_id": STUDY_ID,
                "stage": mapping["stage"],
                "visibility": "RESTRICTED ADJUDICATION MAP; NEVER DISTRIBUTE TO THE ADJUDICATOR",
                "source_restricted_map": {
                    "path": str(restricted_map_path.resolve()),
                    "artifact_sha256": artifact_sha256(restricted_map_path),
                },
                "source_responses": {
                    "rater_a": {"path": str(rater_a_response_path.resolve()), "artifact_sha256": artifact_sha256(rater_a_response_path)},
                    "rater_b": {"path": str(rater_b_response_path.resolve()), "artifact_sha256": artifact_sha256(rater_b_response_path)},
                },
                "first_pass_rater_code_sha256_by_slot": {
                    "rater_a": sha256_bytes(code_a.encode()),
                    "rater_b": sha256_bytes(code_b.encode()),
                },
                "adjudicator_seed": seed,
                "rubric_sha256": mapping["rubric_sha256"],
                "rubric_ambiguity_codes": mapping["rubric_ambiguity_codes"],
                "source_guess_options": mapping["source_guess_options"],
                "packet_root": str(packet_resolved),
                "packets": {
                    "adjudicator": {
                        "delivery_rule": "ONE_BATCH_AT_A_TIME_WITH_PRIOR_BATCH_COLLECTED_AND_REVOKED",
                        "batch_count": len(batch_metadata),
                        "batches": batch_metadata,
                    }
                },
                "asset_count": len(adjudication_assets),
                "assets": adjudication_assets,
            }
        )
        adjudication_map_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(adjudication_map_path, adjudication_map, mode=0o600)
        map_written = True
        os.replace(temporary_root, packet_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        if map_written and not packet_root.exists():
            adjudication_map_path.unlink(missing_ok=True)
        raise
    return {
        "stage": mapping["stage"],
        "first_pass_image_count": len(mapping["assets"]),
        "adjudication_required_count": len(disagreements),
        "adjudication_batch_count": len(batch_metadata),
        "adjudication_map_sha256": sha256_file(adjudication_map_path),
    }


def _load_adjudication_context(adjudication_map_path: Path) -> tuple[Any, ...]:
    adjudication = load_json(adjudication_map_path)
    _exact_keys(adjudication, ADJUDICATION_MAP_KEYS, "adjudication map")
    require(adjudication.get("schema_version") == ADJUDICATION_MAP_SCHEMA, "adjudication map schema mismatch")
    require(adjudication.get("study_id") == STUDY_ID, "adjudication map study mismatch")
    require(adjudication.get("stage") in {"development", "confirmation"}, "adjudication map stage is invalid")
    verify_signed(adjudication, "adjudication map")
    base = adjudication_map_path.resolve().parent
    restricted_ref = adjudication.get("source_restricted_map")
    require(isinstance(restricted_ref, dict), "adjudication restricted-map reference is invalid")
    _exact_keys(restricted_ref, ARTIFACT_SHA_REFERENCE_KEYS, "adjudication restricted-map reference")
    require_sha256(restricted_ref.get("artifact_sha256"), "adjudication restricted-map artifact hash")
    restricted_path = _resolve_reference(base, restricted_ref["path"], "adjudication restricted-map path")
    require(artifact_sha256(restricted_path) == restricted_ref["artifact_sha256"], "adjudication restricted-map hash mismatch")
    responses = adjudication.get("source_responses")
    require(isinstance(responses, dict) and set(responses) == {"rater_a", "rater_b"}, "adjudication source responses are invalid")
    response_paths = {}
    for slot, reference in responses.items():
        require(isinstance(reference, dict), f"adjudication {slot} response reference is invalid")
        _exact_keys(reference, ARTIFACT_SHA_REFERENCE_KEYS, f"adjudication {slot} response reference")
        require_sha256(reference.get("artifact_sha256"), f"adjudication {slot} response artifact hash")
        path = _resolve_reference(base, reference["path"], f"adjudication {slot} response path")
        require(artifact_sha256(path) == reference["artifact_sha256"], f"adjudication {slot} response hash mismatch")
        response_paths[slot] = path
    mapping, annotations_a, annotations_b, code_a, code_b = _load_first_pass_context(
        restricted_map_path=restricted_path,
        rater_a_response_path=response_paths["rater_a"],
        rater_b_response_path=response_paths["rater_b"],
    )
    require(mapping["stage"] == adjudication["stage"], "adjudication stage differs from source mapping")
    require(mapping["rubric_sha256"] == adjudication["rubric_sha256"], "adjudication rubric differs from source mapping")
    require(mapping["adjudicator_seed"] == adjudication["adjudicator_seed"], "adjudication seed differs from source mapping")
    require_sha256(adjudication.get("rubric_sha256"), "adjudication rubric sha256")
    require(
        adjudication.get("visibility") == "RESTRICTED ADJUDICATION MAP; NEVER DISTRIBUTE TO THE ADJUDICATOR",
        "adjudication visibility warning changed",
    )
    ambiguity_codes = adjudication.get("rubric_ambiguity_codes")
    require(
        isinstance(ambiguity_codes, list)
        and ambiguity_codes
        and len(ambiguity_codes) == len(set(ambiguity_codes))
        and all(isinstance(code, str) and code for code in ambiguity_codes),
        "adjudication ambiguity codes are invalid",
    )
    require(ambiguity_codes == mapping["rubric_ambiguity_codes"], "adjudication ambiguity codes differ from source mapping")
    source_guess_options = adjudication.get("source_guess_options")
    require(
        source_guess_options == mapping["source_guess_options"],
        "adjudication source-guess options differ from source mapping",
    )
    expected_code_hashes = {"rater_a": sha256_bytes(code_a.encode()), "rater_b": sha256_bytes(code_b.encode())}
    code_hashes = adjudication.get("first_pass_rater_code_sha256_by_slot")
    require(isinstance(code_hashes, dict), "adjudication first-pass identity hashes are invalid")
    _exact_keys(code_hashes, {"rater_a", "rater_b"}, "adjudication first-pass identity hashes")
    for slot, digest in code_hashes.items():
        require_sha256(digest, f"adjudication {slot} identity hash")
    require(adjudication["first_pass_rater_code_sha256_by_slot"] == expected_code_hashes, "adjudication first-pass identities changed")

    packet_root = Path(require_nonempty_string(adjudication.get("packet_root"), "adjudication packet_root")).resolve()
    require(packet_root.is_dir(), "adjudication packet root is missing")
    packet_streams = adjudication.get("packets")
    require(isinstance(packet_streams, dict), "adjudication packet streams are invalid")
    _exact_keys(packet_streams, {"adjudicator"}, "adjudication packet streams")
    stream = packet_streams["adjudicator"]
    require(isinstance(stream, dict), "adjudication packet stream is invalid")
    _exact_keys(stream, PACKET_STREAM_KEYS, "adjudication packet stream")
    require(
        stream.get("delivery_rule") == "ONE_BATCH_AT_A_TIME_WITH_PRIOR_BATCH_COLLECTED_AND_REVOKED",
        "adjudication packet delivery rule changed",
    )
    batches = stream.get("batches")
    require(isinstance(batches, list), "adjudication packet batches must be a list")
    require(stream.get("batch_count") == len(batches), "adjudication packet batch count mismatch")
    batch_asset_ids: list[str] = []
    packet_ids: set[str] = set()
    for batch_index, batch in enumerate(batches):
        require(isinstance(batch, dict), "adjudication packet batch is invalid")
        _exact_keys(batch, PACKET_BATCH_KEYS, f"adjudication packet batch {batch_index}")
        packet_id = require_nonempty_string(batch.get("packet_id"), "adjudication packet_id")
        require(packet_id not in packet_ids, "adjudication packet_id repeats")
        packet_ids.add(packet_id)
        require_sha256(batch.get("packet_manifest_sha256"), "adjudication packet manifest sha256")
        require_nonempty_string(batch.get("packet_relative_path"), "adjudication packet relative path")
        restricted_ids = batch.get("restricted_asset_ids")
        require(
            isinstance(restricted_ids, list)
            and restricted_ids
            and len(restricted_ids) == len(set(restricted_ids))
            and all(isinstance(asset_id, str) and re.fullmatch(r"asset_[0-9]{6}", asset_id) for asset_id in restricted_ids),
            "adjudication packet restricted assets are invalid",
        )
        batch_asset_ids.extend(restricted_ids)

    expected_disputes = []
    for asset in mapping["assets"]:
        left = _consensus_annotation(annotations_a[asset["rater_opaque_ids"]["rater_a"]])
        right = _consensus_annotation(annotations_b[asset["rater_opaque_ids"]["rater_b"]])
        if left != right:
            expected_disputes.append(asset["restricted_asset_id"])
    adjudication_assets = adjudication.get("assets")
    require(isinstance(adjudication_assets, list), "adjudication assets must be a list")
    observed_disputes = []
    for asset in adjudication_assets:
        require(isinstance(asset, dict), "adjudication asset is invalid")
        _exact_keys(asset, ADJUDICATION_ASSET_KEYS, "adjudication asset")
        asset_id = require_nonempty_string(asset.get("restricted_asset_id"), "adjudication restricted_asset_id")
        require(re.fullmatch(r"asset_[0-9]{6}", asset_id) is not None, "adjudication restricted_asset_id is invalid")
        require_sha256(asset.get("annotation_media_sha256"), "adjudication media sha256")
        require(type(asset.get("width_px")) is int and asset["width_px"] > 0, "adjudication width is invalid")
        require(type(asset.get("height_px")) is int and asset["height_px"] > 0, "adjudication height is invalid")
        request_ids = asset.get("source_request_ids")
        require(
            isinstance(request_ids, list)
            and request_ids
            and len(request_ids) == len(set(request_ids))
            and all(isinstance(request_id, str) and request_id for request_id in request_ids),
            "adjudication source request ids are invalid",
        )
        opaque_ids = asset.get("rater_opaque_ids")
        require(isinstance(opaque_ids, dict), "adjudication opaque-id map is invalid")
        _exact_keys(opaque_ids, {"adjudicator"}, "adjudication opaque-id map")
        require(
            isinstance(opaque_ids["adjudicator"], str)
            and re.fullmatch(r"img_[0-9a-f]{24}", opaque_ids["adjudicator"]) is not None,
            "adjudication opaque image id is invalid",
        )
        mapped_packet_ids = asset.get("rater_packet_ids")
        require(isinstance(mapped_packet_ids, dict), "adjudication packet-id map is invalid")
        _exact_keys(mapped_packet_ids, {"adjudicator"}, "adjudication packet-id map")
        require(mapped_packet_ids["adjudicator"] in packet_ids, "adjudication asset references an unknown packet")
        observed_disputes.append(asset_id)
    require(observed_disputes == expected_disputes, "adjudication map does not contain exactly the first-pass disagreements")
    require(adjudication.get("asset_count") == len(expected_disputes), "adjudication asset count mismatch")
    require(
        len(batch_asset_ids) == len(set(batch_asset_ids)) and set(batch_asset_ids) == set(expected_disputes),
        "adjudication packets do not cover each disagreement exactly once",
    )
    source_by_id = {asset["restricted_asset_id"]: asset for asset in mapping["assets"]}
    for asset in adjudication["assets"]:
        source = source_by_id[asset["restricted_asset_id"]]
        require(asset["annotation_media_sha256"] == source["annotation_media_sha256"], "adjudication media differs from source")
        require((asset["width_px"], asset["height_px"]) == (source["width_px"], source["height_px"]), "adjudication dimensions differ from source")
        require(
            asset["source_request_ids"]
            == sorted({record["source_request_id"] for record in source["source_records"]}),
            "adjudication source-request provenance differs from source mapping",
        )
        require(
            asset["rater_packet_ids"]["adjudicator"]
            == next(
                batch["packet_id"]
                for batch in batches
                if asset["restricted_asset_id"] in batch["restricted_asset_ids"]
            ),
            "adjudication asset packet binding changed",
        )
    for batch in batches:
        _validate_packet_artifacts(adjudication, slot="adjudicator", batch=batch)
    return adjudication, mapping, annotations_a, annotations_b, code_a, code_b, restricted_path, response_paths


def merge_adjudication(
    *,
    adjudication_map_path: Path,
    adjudicator_response_path: Path | None,
) -> dict[str, Any]:
    """Mechanically merge exact agreements with a locked blind third-rater response."""

    (
        adjudication,
        mapping,
        annotations_a,
        annotations_b,
        code_a,
        code_b,
        restricted_path,
        response_paths,
    ) = _load_adjudication_context(adjudication_map_path)
    adjudicated_by_asset = {}
    adjudicator_code = None
    adjudicator_response_sha = None
    if adjudication["asset_count"]:
        require(adjudicator_response_path is not None, "first-pass disagreements require a locked adjudicator response directory")
        adjudicator_annotations, adjudicator_code = _validate_response_set(
            adjudicator_response_path,
            mapping=adjudication,
            slot="adjudicator",
        )
        require(
            sha256_bytes(adjudicator_code.encode())
            not in set(adjudication["first_pass_rater_code_sha256_by_slot"].values()),
            "adjudicator is not independent of both first-pass raters",
        )
        adjudicator_response_sha = artifact_sha256(adjudicator_response_path)
        for asset in adjudication["assets"]:
            adjudicated_by_asset[asset["restricted_asset_id"]] = adjudicator_annotations[
                asset["rater_opaque_ids"]["adjudicator"]
            ]
    else:
        require(adjudicator_response_path is None, "no-disagreement cohort must not introduce adjudicator labels")

    labels = []
    adjudicated_count = 0
    allowed_ambiguity = set(mapping["rubric_ambiguity_codes"])
    for asset in mapping["assets"]:
        left = _consensus_annotation(annotations_a[asset["rater_opaque_ids"]["rater_a"]])
        right = _consensus_annotation(annotations_b[asset["rater_opaque_ids"]["rater_b"]])
        if left == right:
            decision_source = "first_pass_exact_agreement"
            annotation = left
        else:
            decision_source = "independent_adjudicator"
            annotation = _consensus_annotation(adjudicated_by_asset[asset["restricted_asset_id"]])
            adjudicated_count += 1
        annotation = _validate_consensus_annotation(
            annotation,
            width=asset["width_px"],
            height=asset["height_px"],
            allowed_ambiguity=allowed_ambiguity,
        )
        labels.append(
            {
                "restricted_asset_id": asset["restricted_asset_id"],
                "decision_source": decision_source,
                "annotation": annotation,
            }
        )
    return sign_document(
        {
            "schema_version": FINAL_CONSENSUS_SCHEMA,
            "study_id": STUDY_ID,
            "stage": mapping["stage"],
            "status": "mechanically_merged_from_locked_blind_responses",
            "adjudication_map_path": str(adjudication_map_path.resolve()),
            "adjudication_map_sha256": sha256_file(adjudication_map_path),
            "source_restricted_map_sha256": sha256_file(restricted_path),
            "first_pass_response_sha256_by_slot": {
                "rater_a": artifact_sha256(response_paths["rater_a"]),
                "rater_b": artifact_sha256(response_paths["rater_b"]),
            },
            "rater_code_sha256_by_slot": {
                "rater_a": sha256_bytes(code_a.encode()),
                "rater_b": sha256_bytes(code_b.encode()),
                "adjudicator": sha256_bytes(adjudicator_code.encode()) if adjudicator_code else None,
            },
            "adjudicator_response_path": str(adjudicator_response_path.resolve()) if adjudicator_response_path else None,
            "adjudicator_response_sha256": adjudicator_response_sha,
            "counts": {
                "images": len(labels),
                "first_pass_exact_agreements": len(labels) - adjudicated_count,
                "independently_adjudicated": adjudicated_count,
            },
            "labels": labels,
        }
    )


def validate_final_consensus(
    consensus: Mapping[str, Any],
    *,
    consensus_path: Path,
    development_summary: Mapping[str, Any],
    development_summary_sha256: str,
    restricted_mapping: Mapping[str, Any],
) -> None:
    require(consensus.get("schema_version") == FINAL_CONSENSUS_SCHEMA, "final consensus schema mismatch")
    verify_signed(consensus, "final consensus")
    require(consensus.get("stage") == "development", "confirmation requires development-stage final consensus")
    adjudication_map_path = _resolve_reference(
        consensus_path.parent,
        consensus.get("adjudication_map_path"),
        "final consensus adjudication-map path",
    )
    adjudicator_response_value = consensus.get("adjudicator_response_path")
    adjudicator_response_path = (
        _resolve_reference(consensus_path.parent, adjudicator_response_value, "final consensus adjudicator-response path")
        if adjudicator_response_value is not None
        else None
    )
    reproduced = merge_adjudication(
        adjudication_map_path=adjudication_map_path,
        adjudicator_response_path=adjudicator_response_path,
    )
    require(reproduced == consensus, "final consensus does not reproduce from blind packet responses")
    require(consensus.get("source_restricted_map_sha256") == development_summary.get("restricted_map_sha256"), "final consensus restricted-map hash mismatch")
    require(
        consensus.get("first_pass_response_sha256_by_slot")
        == {
            "rater_a": development_summary.get("rater_a_response_sha256"),
            "rater_b": development_summary.get("rater_b_response_sha256"),
        },
        "final consensus first-pass response hashes differ from development summary",
    )
    verify_signed(restricted_mapping, "development restricted mapping")
    require(restricted_mapping.get("stage") == "development", "development restricted mapping stage mismatch")
    require_sha256(development_summary_sha256, "development summary sha256")


def _relative_vector(annotation: Mapping[str, Any], width: int, height: int) -> tuple[float, float]:
    cube = annotation["cube_center_px"]
    bowl = annotation["bowl_center_px"]
    diagonal = math.hypot(width, height)
    return (cube[0] - bowl[0]) / diagonal, (cube[1] - bowl[1]) / diagonal


def summarize_development_labels(
    *,
    restricted_map_path: Path,
    rater_a_response_path: Path,
    rater_b_response_path: Path,
) -> dict[str, Any]:
    """Derive (but do not judge adequacy of) the development q95 threshold."""

    mapping = load_json(restricted_map_path)
    require(mapping.get("schema_version") == RESTRICTED_MAP_SCHEMA, "restricted mapping schema mismatch")
    verify_signed(mapping, "restricted mapping")
    require(mapping.get("study_id") == STUDY_ID, "restricted mapping study mismatch")
    require(mapping.get("stage") == "development", "movement threshold must come from development labels")
    annotations_a, rater_a_code = _validate_response_set(
        rater_a_response_path,
        mapping=mapping,
        slot="rater_a",
    )
    annotations_b, rater_b_code = _validate_response_set(
        rater_b_response_path,
        mapping=mapping,
        slot="rater_b",
    )
    require(rater_a_code != rater_b_code, "the two response sets identify the same rater")

    disagreements: list[float] = []
    role_observations: dict[str, list[tuple[bool, bool, float | None]]] = defaultdict(list)
    category_agreement = 0
    exact_agreement = 0
    unknown_either = 0
    rows = []
    for asset in mapping["assets"]:
        left = annotations_a[asset["rater_opaque_ids"]["rater_a"]]
        right = annotations_b[asset["rater_opaque_ids"]["rater_b"]]
        category_same = (
            left["cube_resolvability"],
            left["bowl_resolvability"],
            left["cube_identity"],
            left["bowl_identity"],
            sorted(left["ambiguity_codes"]),
        ) == (
            right["cube_resolvability"],
            right["bowl_resolvability"],
            right["cube_identity"],
            right["bowl_identity"],
            sorted(right["ambiguity_codes"]),
        )
        category_agreement += int(category_same)
        left_consensus = _consensus_annotation(left)
        right_consensus = _consensus_annotation(right)
        exact_same = left_consensus == right_consensus
        exact_agreement += int(exact_same)
        usable = all(
            annotation[entity] == "resolvable"
            for annotation in (left, right)
            for entity in ("cube_resolvability", "bowl_resolvability")
        )
        disagreement = None
        if usable:
            vector_a = _relative_vector(left, asset["width_px"], asset["height_px"])
            vector_b = _relative_vector(right, asset["width_px"], asset["height_px"])
            disagreement = math.dist(vector_a, vector_b)
            disagreements.append(disagreement)
        else:
            unknown_either += 1
        rows.append(
            {
                "restricted_asset_id": asset["restricted_asset_id"],
                "source_image_ids": sorted(
                    {record["source_image_id"] for record in asset["source_records"]}
                ),
                "both_resolvable": usable,
                "categorical_agreement": category_same,
                "first_pass_exact_agreement": exact_same,
                "requires_adjudication": not exact_same,
                "agreed_annotation": left_consensus if exact_same else None,
                "relative_vector_disagreement": disagreement,
            }
        )
        for role in sorted({record["image_role"] for record in asset["source_records"]}):
            role_observations[role].append((usable, category_same, disagreement))
    require(disagreements, "no duplicate development image was resolvable by both raters")
    threshold = quantile_type7(disagreements, 0.95)
    by_image_role = {}
    for role, observations in sorted(role_observations.items()):
        usable_values = [value for usable, _, value in observations if usable and value is not None]
        by_image_role[role] = {
            "unique_image_count": len(observations),
            "eligible_duplicate_count": len(usable_values),
            "unknown_to_either_rater_count": sum(not usable for usable, _, _ in observations),
            "categorical_agreement_count": sum(agreement for _, agreement, _ in observations),
            "relative_vector_disagreement_q95": (
                quantile_type7(usable_values, 0.95) if usable_values else None
            ),
        }
    summary = {
        "schema_version": DEVELOPMENT_SUMMARY_SCHEMA,
        "study_id": STUDY_ID,
        "cohort_branch": mapping["cohort_branch"],
        "qualified_model_ids": mapping["qualified_model_ids"],
        "qualified_alignment_contract_sha256_by_model": mapping[
            "qualified_alignment_contract_sha256_by_model"
        ],
        "status": "EMPIRICAL_DEVELOPMENT_RESULT_REQUIRES_EXPLICIT_USABILITY_DECISION",
        "source_files": {
            "restricted_map": {
                "path": str(restricted_map_path.resolve()),
                "artifact_sha256": artifact_sha256(restricted_map_path),
            },
            "rater_a_responses": {
                "path": str(rater_a_response_path.resolve()),
                "artifact_sha256": artifact_sha256(rater_a_response_path),
            },
            "rater_b_responses": {
                "path": str(rater_b_response_path.resolve()),
                "artifact_sha256": artifact_sha256(rater_b_response_path),
            },
        },
        "restricted_map_sha256": sha256_file(restricted_map_path),
        "rater_a_response_sha256": artifact_sha256(rater_a_response_path),
        "rater_b_response_sha256": artifact_sha256(rater_b_response_path),
        "rubric_sha256": mapping["rubric_sha256"],
        "distinct_raters_attested": True,
        "first_pass_rater_code_sha256_by_slot": {
            "rater_a": sha256_bytes(rater_a_code.encode("utf-8")),
            "rater_b": sha256_bytes(rater_b_code.encode("utf-8")),
        },
        "movement_disagreement_definition": MOVEMENT_DISAGREEMENT_DEFINITION,
        "quantile_probability": 0.95,
        "quantile_method": MOVEMENT_QUANTILE_METHOD,
        "movement_resolution_threshold_relative_image_diagonal": threshold,
        "unique_duplicate_image_count": len(mapping["assets"]),
        "eligible_duplicate_count": len(disagreements),
        "unknown_to_either_rater_count": unknown_either,
        "categorical_agreement_count": category_agreement,
        "categorical_disagreement_count": len(mapping["assets"]) - category_agreement,
        "first_pass_exact_agreement_count": exact_agreement,
        "adjudication_required_count": len(mapping["assets"]) - exact_agreement,
        "final_consensus_status": "REQUIRED_BEFORE_CONFIRMATION",
        "by_image_role": by_image_role,
        "rater_source_guess_counts": {
            "rater_a": dict(sorted(Counter(item["source_guess"] for item in annotations_a.values()).items())),
            "rater_b": dict(sorted(Counter(item["source_guess"] for item in annotations_b.values()).items())),
        },
        "annotation_seconds": {
            "rater_a_total": math.fsum(item["annotation_seconds"] for item in annotations_a.values()),
            "rater_b_total": math.fsum(item["annotation_seconds"] for item in annotations_b.values()),
        },
        "adequacy_is_not_inferred_by_code": True,
        "required_confirmation_input": (
            "An authorized scientific reviewer must inspect development coverage, disagreement, examples, "
            "and movement scale and record measurement_usable=true with a reason; this script cannot supply it."
        ),
        "assets": rows,
    }
    return sign_document(summary)


def _write_selection(args: argparse.Namespace) -> None:
    inventory = load_json(args.inventory)
    result = select_requests(
        inventory,
        seed=args.seed,
        cap_per_episode=args.cap_per_episode,
        inventory_sha256=sha256_file(args.inventory),
        inventory_path=args.inventory,
    )
    atomic_write_json(args.output, result, mode=0o600)
    print(json.dumps(result["counts"], indent=2, sort_keys=True))


def _write_packages(args: argparse.Namespace) -> None:
    result = package_packets(
        selection_path=args.selection,
        image_inventory_path=args.image_inventory,
        freeze_path=args.freeze,
        packet_root=args.packet_root,
        restricted_map_path=args.restricted_map,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _write_adjudication_packages(args: argparse.Namespace) -> None:
    result = package_adjudication(
        restricted_map_path=args.restricted_map,
        rater_a_response_path=args.rater_a_response,
        rater_b_response_path=args.rater_b_response,
        packet_root=args.packet_root,
        adjudication_map_path=args.adjudication_map,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _write_final_consensus(args: argparse.Namespace) -> None:
    result = merge_adjudication(
        adjudication_map_path=args.adjudication_map,
        adjudicator_response_path=args.adjudicator_response,
    )
    atomic_write_json(args.output, result, mode=0o600)
    print(json.dumps({"stage": result["stage"], **result["counts"], "payload_sha256": result["payload_sha256"]}, indent=2, sort_keys=True))


def _validate_freeze_cli(args: argparse.Namespace) -> None:
    freeze = load_json(args.freeze)
    info = validate_freeze(freeze, freeze_path=args.freeze, stage=args.stage)
    movement = freeze["movement_resolution"]
    print(
        json.dumps(
            {
                "freeze_sha256": sha256_file(args.freeze),
                "movement_resolution_status": movement["status"],
                "movement_resolution_threshold_relative_image_diagonal": movement[
                    "threshold_relative_image_diagonal"
                ],
                "rubric_sha256": info["rubric_sha256"],
                "stage": args.stage,
                "status": freeze["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_development_summary(args: argparse.Namespace) -> None:
    summary = summarize_development_labels(
        restricted_map_path=args.restricted_map,
        rater_a_response_path=args.rater_a_response,
        rater_b_response_path=args.rater_b_response,
    )
    atomic_write_json(args.output, summary, mode=0o600)
    print(
        json.dumps(
            {
                "eligible_duplicate_count": summary["eligible_duplicate_count"],
                "movement_resolution_threshold_relative_image_diagonal": summary[
                    "movement_resolution_threshold_relative_image_diagonal"
                ],
                "payload_sha256": summary["payload_sha256"],
                "status": summary["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-requests", help="make the metadata-only per-episode draw")
    select.add_argument("--inventory", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=REQUEST_SAMPLING_SEED)
    select.add_argument("--cap-per-episode", type=int, default=REQUEST_SAMPLE_CAP)
    select.set_defaults(function=_write_selection)

    package = subparsers.add_parser("package", help="make two blinded batch streams and a restricted identity map")
    package.add_argument("--selection", type=Path, required=True)
    package.add_argument("--image-inventory", type=Path, required=True)
    package.add_argument("--freeze", type=Path, required=True)
    package.add_argument("--packet-root", type=Path, required=True)
    package.add_argument("--restricted-map", type=Path, required=True)
    package.set_defaults(function=_write_packages)

    adjudication = subparsers.add_parser(
        "package-adjudication",
        help="make a source-free third-rater packet stream from exact first-pass disagreements",
    )
    adjudication.add_argument("--restricted-map", type=Path, required=True)
    adjudication.add_argument("--rater-a-response", type=Path, required=True)
    adjudication.add_argument("--rater-b-response", type=Path, required=True)
    adjudication.add_argument("--packet-root", type=Path, required=True)
    adjudication.add_argument("--adjudication-map", type=Path, required=True)
    adjudication.set_defaults(function=_write_adjudication_packages)

    merge = subparsers.add_parser(
        "merge-adjudication",
        help="mechanically merge exact agreements and a locked independent adjudicator response",
    )
    merge.add_argument("--adjudication-map", type=Path, required=True)
    merge.add_argument("--adjudicator-response", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(function=_write_final_consensus)

    validate = subparsers.add_parser(
        "validate-freeze",
        help="fail closed unless a development or confirmation annotation freeze is complete",
    )
    validate.add_argument("--freeze", type=Path, required=True)
    validate.add_argument("--stage", choices=("development", "confirmation"), required=True)
    validate.set_defaults(function=_validate_freeze_cli)

    summarize = subparsers.add_parser(
        "summarize-development", help="derive q95 duplicate-label disagreement from locked development responses"
    )
    summarize.add_argument("--restricted-map", type=Path, required=True)
    summarize.add_argument("--rater-a-response", type=Path, required=True)
    summarize.add_argument("--rater-b-response", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(function=_write_development_summary)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
