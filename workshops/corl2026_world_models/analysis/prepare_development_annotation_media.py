#!/usr/bin/env python3
"""Build the fail-closed WMF development annotation media preparation.

This CPU-only bridge joins the formal compiler provenance, immutable timing
sidecars, physical-alignment receipts, and model-specific camera crop replay
contracts.  It emits the complete metadata-only request inventory, the frozen
request draw, exact source-extraction lineage, and metadata-free PNG renders.

It deliberately does *not* emit a distributable rater packet.  A named human
must visually review the exact rendered PNG hashes for pixel-level identity or
outcome leakage before ``forecast_annotation_workflow.py package`` can run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Mapping, Sequence
import zipfile
import zlib

import numpy as np


sys.dont_write_bytecode = True

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
COMPILER_PATH = Path(__file__).with_name("compile_development_evidence.py")
TIMING_PATH = Path(__file__).with_name("qualify_forecast_timing.py")
ANNOTATION_PATH = Path(__file__).with_name("forecast_annotation_workflow.py")
FREEZE_PATH = Path(__file__).with_name("freeze_development_release.py")
CAMERA_REPLAY_PATH = Path(__file__).with_name("camera_crop_replay_witness.py")


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load workshop module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


compiler = _load_module(COMPILER_PATH, "wmf_annotation_bridge_compiler")
timing = _load_module(TIMING_PATH, "wmf_annotation_bridge_timing")
annotation = _load_module(ANNOTATION_PATH, "wmf_annotation_bridge_workflow")
freeze = _load_module(FREEZE_PATH, "wmf_annotation_bridge_freeze")
camera_replay = _load_module(CAMERA_REPLAY_PATH, "wmf_annotation_bridge_camera_replay")

STUDY_ID = "WMF-ABLATION-001"
INPUT_SCHEMA = "wmf-development-annotation-media-input-v1"
SOURCE_LINEAGE_SCHEMA = "wmf-forecast-source-extraction-lineage-v1"
PNG_MANIFEST_SCHEMA = "wmf-forecast-rendered-png-manifest-v1"
PRE_REVIEW_SCHEMA = "wmf-forecast-annotation-image-inventory-pre-review-v1"
PIXEL_REVIEW_CHECKLIST_SCHEMA = "wmf-forecast-pixel-blindness-review-checklist-v1"
PREPARATION_SCHEMA = "wmf-development-annotation-media-preparation-v1"
PUBLISH_TRANCHE_INDEX_SCHEMA = "wmf-development-annotation-media-tranche-index-v1"
COMPILER_JOB_SCHEMA = "wmf-development-evidence-compiler-queue-job-v1"
COMPILER_RECEIPT_SCHEMA = "wmf-development-evidence-compiler-receipt-v1"
TIMING_JOB_SCHEMA = "wmf-forecast-timing-queue-job-v1"
CROP_SCHEMA = "wmf-camera-crop-contract-v1"
MAPPING_SCHEMA = "wmf-forecast-physical-alignment-receipt-v1"
REQUEST_INVENTORY_SCHEMA = "wmf-forecast-request-inventory-v1"
SELECTION_SCHEMA = "wmf-forecast-request-selection-v1"
RENDER_RECEIPT_SCHEMA = "wmf-forecast-annotation-render-receipt-v1"
IMAGE_INVENTORY_SCHEMA = "wmf-forecast-annotation-image-inventory-v1"
REPLAY_IPC_SCHEMA = "wmf-camera-replay-exact-runtime-ipc-v1"
REPLAY_SESSION_SCHEMA = "wmf-camera-replay-exact-runtime-session-v1"
CANONICAL_RAW_ROOT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912"
)

MODELS = ("N3", "D1")
EXPECTED_EPISODES = 32
EXPECTED_REQUESTS = {"N3": 240, "D1": 912}
EXPECTED_TOTAL_REQUESTS = 1152
EXPECTED_TIMING_CAPABLE = {"N3": 240, "D1": 240}
EXPECTED_ELIGIBLE = {"N3": 224, "D1": 224}
EXPECTED_SELECTED = {"N3": 64, "D1": 64}
EXPECTED_MAPPING_TARGET_COUNTS = {
    "N3": {
        "primary": {"eligible": 224, "full_prefix": 224},
        "early": {"eligible": 240, "full_prefix": 224},
    },
    "D1": {
        "primary": {"eligible": 224, "full_prefix": 224},
        "early": {"eligible": 224, "full_prefix": 224},
    },
}
EXPECTED_ALIGNMENT_TARGET_IDENTITIES = {
    "N3": {"primary": (32, 32), "early": (1, 1)},
    "D1": {"primary": (2, 6), "early": (1, 3)},
}
MAPPING_RECEIPT_KEYS = {
    "schema_version", "receipt_id", "study_id", "status", "model_id",
    "development_cell_ids", "development_cell_count", "development_request_count",
    "request_semantics", "timing_claim_boundary", "camera",
    "measured_clock_intervals", "frame_to_physical_time", "primary_target",
    "early_target", "source_receipts", "payload_sha256",
}
EXPECTED_D1_UNMAPPED = 672
EXPECTED_TOTAL_ELIGIBLE = 448
EXPECTED_TOTAL_SELECTED = 128
EXPECTED_INCLUSION_PROBABILITY = 2 / 7
EXPECTED_INCLUSION_PROBABILITY_EXACT = "4/14"
PUBLISH_FILE_LIMIT_BYTES = 16 * 1024 * 1024
PUBLISH_JOB_LIMIT_BYTES = 64 * 1024 * 1024
# Leave room for a signed alias manifest, queue receipt, and Git metadata.  PNG
# bytes are already compressed, so archive compression would not justify
# violating the publisher's 16 MiB per-file boundary.
PUBLISH_TRANCHE_ASSET_BUDGET_BYTES = 48 * 1024 * 1024
EXPECTED_IMAGE_DIMENSIONS = {"N3": (320, 168), "D1": (320, 176)}
EXPECTED_CROP_IDS = {
    "N3": "n3-over-shoulder-left-168x320-v1",
    "D1": "d1-over-shoulder-left-176x320-v1",
}
EXPECTED_CROP_OPERATIONS = {
    "N3": (
        "raw_selected_camera_replay_and_generated_uint8_thwc_"
        "crop_half_open_y360_528_x0_320"
    ),
    "D1": (
        "raw_selected_camera_replay_and_generated_uint8_thwc_"
        "crop_half_open_y176_352_x0_320"
    ),
}
EXPECTED_GENERATED_CANVAS_SHAPES = {
    "N3": [33, 528, 640, 3],
    "D1": [9, 352, 640, 3],
}
EXPECTED_GENERATED_CROPS = {
    "N3": {"y": [360, 528], "x": [0, 320]},
    "D1": {"y": [176, 352], "x": [0, 320]},
}
EXPECTED_D1_COVERAGE = dict(freeze.D1_FORMAL_TIMING_COVERAGE)
EXPECTED_COMPILER_JOB_SCIENCE_COUNTS = {
    "model_runtime_loads": 0,
    "model_servers_started": 0,
    "model_requests_issued_by_job": 0,
    "simulator_processes_started": 0,
    "physical_resets": 0,
    "robot_episodes": 0,
    "behavioral_actions_executed_by_job": 0,
    "behavioral_cells_launched_by_job": 0,
    "labels_created_by_job": 0,
}
EXPECTED_COMPILER_SCIENCE_ACTIVITY = {
    "model_loads": 0,
    "model_requests_issued": 0,
    "simulator_processes_started": 0,
    "physical_resets": 0,
    "behavioral_actions_executed": 0,
    "labels_created": 0,
}
EXPECTED_TIMING_JOB_SCIENCE_COUNTS = {
    "model_runtime_loads": 0,
    "model_servers_started": 0,
    "model_requests_issued_by_job": 0,
    "physical_resets": 0,
    "robot_episodes": 0,
    "behavioral_actions_executed_by_job": 0,
    "behavioral_cells_launched_by_job": 0,
}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")
IPC_HEADER_LIMIT_BYTES = 64 * 1024
IPC_READY_TIMEOUT_S = 300.0
IPC_REPLAY_TIMEOUT_S = 120.0
IPC_CLOSE_TIMEOUT_S = 30.0
IPC_MAX_REQUESTS = 2048

PROVENANCE_KEYS = {
    "schema_version", "study_id", "mode", "model_id", "stage", "status",
    "safe_for_formal_release", "annotation_state", "camera_id",
    "camera_crop_contract", "resource_measurements", "labels",
    "episode_roster", "requests", "payload_sha256",
}
PROVENANCE_REQUEST_KEYS = {
    "source_request_id", "cell_id", "recording_id", "model_id",
    "layout_pair_id", "condition_id", "request_index", "action_step_start",
    "executed_prefix_actions", "current_observation_id",
    "preceding_observation_id", "history_mode", "current_observation",
    "preceding_observation", "official_request_receipt",
    "recorder_model_request", "recorder_response_payload_sha256",
    "adapter_completion", "adapter_journal", "source_video_id",
    "source_video", "model_output_or_action_modified", "model_identity",
    "model_context", "action_manifest", "recording_receipt",
}
FREEZE_FRAGMENT_KEYS = {
    "schema_version", "study_id", "mode", "model_id", "status",
    "safe_for_alignment_input", "resource_receipts_synthesized",
    "development_cells", "payload_sha256",
}
FREEZE_CELL_KEYS = {"cell_receipt", "server_request_receipts", "resource_receipt"}
OBSERVATION_KEYS = {
    "observation_id", "control_step", "physics_step", "physics_time_s",
    "camera_frame_native_id", "camera_frame_id", "camera_capture_time_ns",
    "camera_timestamp_source", "payload_sha256", "payload_artifact",
}


class AnnotationMediaBridgeError(RuntimeError):
    """An immutable provenance, timing, crop, or rendering gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationMediaBridgeError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AnnotationMediaBridgeError("value is not finite canonical JSON") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AnnotationMediaBridgeError("value is not finite JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AnnotationMediaBridgeError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def sign_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    digest = value.get("payload_sha256")
    require(isinstance(digest, str) and SHA_RE.fullmatch(digest) is not None,
            f"{label} payload signature is invalid")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(canonical_bytes(unsigned)) == digest,
            f"{label} payload signature changed")


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} is not a JSON object")
    require(set(value) == expected, f"{label} fields changed")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnnotationMediaBridgeError(f"{label} is not readable JSON: {path}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _lexical_absolute(path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _reject_symlinks(path: Path, label: str, *, stop: Path | None = None) -> None:
    lexical = _lexical_absolute(path)
    boundary = None if stop is None else _lexical_absolute(stop)
    if boundary is not None:
        require(lexical.is_relative_to(boundary), f"{label} lexically escapes raw root")
    cursor = lexical
    while True:
        require(not cursor.is_symlink(), f"{label} contains a symlink: {cursor}")
        if boundary is not None and cursor == boundary:
            return
        parent = cursor.parent
        if parent == cursor:
            require(boundary is None, f"{label} did not reach raw root")
            return
        cursor = parent


def _under(path: Path, root: Path, label: str) -> Path:
    _reject_symlinks(path, label, stop=root)
    try:
        resolved = Path(path).resolve(strict=True)
        resolved_root = Path(root).resolve(strict=True)
    except OSError as error:
        raise AnnotationMediaBridgeError(f"{label} is unavailable") from error
    require(resolved.is_relative_to(resolved_root), f"{label} escapes raw root")
    require(resolved.is_file(), f"{label} is not an immutable regular file")
    return resolved


def descriptor(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(),
            f"descriptor target is not a regular file: {path}")
    return {
        "path": str(resolved) if display_path is None else display_path,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_descriptor(
    value: Any,
    *,
    base: Path,
    raw_root: Path,
    label: str,
) -> tuple[dict[str, Any], Path]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    keys = set(value)
    require(keys in (
        {"path", "sha256", "bytes"},
        {"path", "sha256", "bytes", "event_count", "tail_sha256"},
    ), f"{label} descriptor fields changed")
    raw_path = value.get("path")
    require(isinstance(raw_path, str) and raw_path, f"{label} path is missing")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    path = _under(candidate, raw_root, label)
    expected_sha = value.get("sha256")
    expected_bytes = value.get("bytes")
    require(isinstance(expected_sha, str) and SHA_RE.fullmatch(expected_sha) is not None,
            f"{label} SHA-256 is invalid")
    require(type(expected_bytes) is int and expected_bytes > 0,
            f"{label} byte count is invalid")
    observed = descriptor(path)
    require(observed["sha256"] == expected_sha and observed["bytes"] == expected_bytes,
            f"{label} bytes or SHA-256 changed")
    if "event_count" in value:
        require(type(value.get("event_count")) is int and value["event_count"] > 0
                and isinstance(value.get("tail_sha256"), str)
                and SHA_RE.fullmatch(value["tail_sha256"]) is not None,
                f"{label} journal summary is invalid")
    return dict(value), path


def _same_descriptor(left: Any, right: Any, label: str) -> None:
    require(isinstance(left, Mapping) and isinstance(right, Mapping),
            f"{label} descriptor is missing")
    for key in ("sha256", "bytes"):
        require(left.get(key) == right.get(key), f"{label} descriptor changed: {key}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _zero_science(
    receipt: Mapping[str, Any], label: str, expected: Mapping[str, int]
) -> None:
    counts = receipt.get("science_counts")
    require(counts == expected,
            f"{label} science counts differ from the exact zero-activity schema")


def _exact_compiler_science_activity(value: Any) -> None:
    require(value == EXPECTED_COMPILER_SCIENCE_ACTIVITY,
            "compiler receipt science activity differs from the exact zero schema")


def _validate_compiler(
    manifest: Mapping[str, Any], raw_root: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    raw = manifest.get("compiler")
    _exact_keys(raw, {"job_receipt", "compiler_receipt"}, "compiler inputs")
    _, job_path = _verify_descriptor(
        raw["job_receipt"], base=raw_root, raw_root=raw_root,
        label="compiler queue job receipt",
    )
    job = load_json(job_path, "compiler queue job receipt")
    verify_signed(job, "compiler queue job receipt")
    require(job.get("schema_version") == COMPILER_JOB_SCHEMA
            and job.get("study_id") == STUDY_ID
            and job.get("mode") == "formal_full"
            and job.get("status") == "passed"
            and job.get("decision") == "go"
            and job.get("formal_cohort_complete") is True
            and job.get("safe_for_timing_binding") is True
            and job.get("safe_to_release_confirmation") is False
            and job.get("confirmation_released") is False,
            "compiler queue job did not pass the formal zero-release gate")
    require(job.get("counts") == {
        "compiled_cells": EXPECTED_EPISODES,
        "compiled_source_behavioral_requests": EXPECTED_TOTAL_REQUESTS,
        "compiled_source_behavioral_actions": 14400,
    }, "compiler queue job counts changed")
    _zero_science(
        job, "compiler queue job", EXPECTED_COMPILER_JOB_SCIENCE_COUNTS
    )
    _, receipt_path = _verify_descriptor(
        raw["compiler_receipt"], base=raw_root, raw_root=raw_root,
        label="compiler receipt",
    )
    outputs = job.get("outputs")
    require(isinstance(outputs, Mapping), "compiler queue outputs are missing")
    _same_descriptor(outputs.get("compiler_receipt"), raw["compiler_receipt"],
                     "compiler receipt/job")
    receipt = load_json(receipt_path, "compiler receipt")
    verify_signed(receipt, "compiler receipt")
    require(receipt.get("schema_version") == COMPILER_RECEIPT_SCHEMA
            and receipt.get("study_id") == STUDY_ID
            and receipt.get("mode") == "formal_full"
            and receipt.get("status") == "compiled_complete"
            and receipt.get("formal_cohort_complete") is True
            and receipt.get("safe_for_timing_binding") is True
            and receipt.get("safe_for_confirmation_release") is False
            and receipt.get("counts") == job["counts"],
            "compiler receipt formal gate changed")
    _exact_compiler_science_activity(receipt.get("compiler_science_activity"))
    models = receipt.get("models")
    require(isinstance(models, Mapping) and set(models) == set(MODELS),
            "compiler receipt model coverage changed")
    for model in MODELS:
        output = models[model]
        require(isinstance(output, Mapping)
                and output.get("complete") is True
                and output.get("cell_count") == 16
                and output.get("request_count") == EXPECTED_REQUESTS[model]
                and output.get("action_count") == 7200,
                f"compiler {model} coverage changed")
    return receipt, receipt_path, job


def _validate_provenance(
    *,
    model: str,
    value: Mapping[str, Any],
    path: Path,
    compiler_receipt: Mapping[str, Any],
    compiler_receipt_path: Path,
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _exact_keys(value, PROVENANCE_KEYS, f"{model} compiler provenance")
    verify_signed(value, f"{model} compiler provenance")
    require(value.get("schema_version") == compiler.PROVENANCE_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("mode") == "formal_full"
            and value.get("model_id") == model
            and value.get("stage") == "development"
            and value.get("status") == "complete"
            and value.get("safe_for_formal_release") is True
            and value.get("annotation_state") == "not_started"
            and value.get("camera_id") == "over_shoulder_left_camera"
            and value.get("camera_crop_contract") is None
            and value.get("resource_measurements") is None
            and value.get("labels") is None,
            f"{model} compiler provenance state changed")
    model_output = compiler_receipt["models"][model]
    expected_descriptor = model_output.get("request_provenance")
    require(isinstance(expected_descriptor, Mapping), f"{model} compiler provenance output missing")
    expected_path = compiler_receipt_path.parent / str(expected_descriptor.get("path"))
    require(path == expected_path.resolve(strict=True), f"{model} provenance path differs from compiler receipt")
    require(sha256_file(path) == expected_descriptor.get("sha256")
            and path.stat().st_size == expected_descriptor.get("bytes"),
            f"{model} provenance differs from compiler receipt")
    roster = value.get("episode_roster")
    requests = value.get("requests")
    require(isinstance(roster, list) and len(roster) == 16,
            f"{model} compiler roster is incomplete")
    require(isinstance(requests, list) and len(requests) == EXPECTED_REQUESTS[model],
            f"{model} compiler request provenance is incomplete")
    seen: set[tuple[str, str, int]] = set()
    request_hashes: set[str] = set()
    by_cell: dict[str, list[int]] = defaultdict(list)
    for ordinal, row in enumerate(requests):
        _exact_keys(row, PROVENANCE_REQUEST_KEYS, f"{model} provenance request {ordinal}")
        require(row.get("model_id") == model, f"{model} provenance request model changed")
        cell_id = row.get("cell_id")
        request_index = row.get("request_index")
        require(isinstance(cell_id, str) and type(request_index) is int and request_index >= 0,
                f"{model} provenance request identity is invalid")
        key = (model, cell_id, request_index)
        require(key not in seen, f"{model} provenance request identity is duplicated")
        seen.add(key)
        by_cell[cell_id].append(request_index)
        request_descriptor = row.get("official_request_receipt")
        _, request_path = _verify_descriptor(
            request_descriptor, base=path.parent, raw_root=raw_root,
            label=f"{model} {cell_id} request {request_index} official receipt",
        )
        request_hash = request_descriptor["sha256"]
        require(request_hash not in request_hashes, f"{model} official request receipt is duplicated")
        request_hashes.add(request_hash)
        require(row.get("model_output_or_action_modified") is False,
                f"{model} provenance says model output/action changed")
        require(row.get("history_mode") == (
            "persistence_at_initial_request" if request_index == 0
            else "preceding_observation"
        ), f"{model} provenance history mode changed")
        current = row.get("current_observation")
        _exact_keys(current, OBSERVATION_KEYS, f"{model} request current observation")
        require(current.get("observation_id") == row.get("current_observation_id")
                and current.get("control_step") == row.get("action_step_start"),
                f"{model} request current observation binding changed")
        preceding = row.get("preceding_observation")
        if request_index == 0:
            require(row.get("preceding_observation_id") is None and preceding is None,
                    f"{model} request zero acquired synthetic history")
        else:
            _exact_keys(preceding, OBSERVATION_KEYS, f"{model} request preceding observation")
            require(preceding.get("observation_id") == row.get("preceding_observation_id")
                    and preceding.get("control_step") == row.get("action_step_start") - 1,
                    f"{model} request preceding observation binding changed")
        # These descriptors are reopened again by the annotation workflow and
        # by source extraction.  Verify them here so no count-only provenance
        # can enter the join.
        for descriptor_key in ("adapter_completion", "adapter_journal"):
            _verify_descriptor(
                row.get(descriptor_key), base=path.parent, raw_root=raw_root,
                label=f"{model} {cell_id} {descriptor_key}",
            )
    expected_per_cell = 15 if model == "N3" else 57
    require(len(by_cell) == 16, f"{model} request provenance cell coverage changed")
    for cell_id, indexes in by_cell.items():
        require(sorted(indexes) == list(range(expected_per_cell)),
                f"{model} {cell_id} request schedule changed")
    return [dict(row) for row in roster], [dict(row) for row in requests]


def _load_compiler_freeze_fragment(
    *, model: str, compiler_receipt: Mapping[str, Any],
    compiler_receipt_path: Path, raw_root: Path,
) -> dict[str, Any]:
    model_output = compiler_receipt.get("models", {}).get(model)
    require(isinstance(model_output, Mapping),
            f"{model} compiler model output is missing")
    expected = model_output.get("freeze_cell_evidence")
    _, path = _verify_descriptor(
        expected, base=compiler_receipt_path.parent, raw_root=raw_root,
        label=f"{model} compiler freeze fragment",
    )
    value = load_json(path, f"{model} compiler freeze fragment")
    _exact_keys(value, FREEZE_FRAGMENT_KEYS, f"{model} compiler freeze fragment")
    verify_signed(value, f"{model} compiler freeze fragment")
    require(value.get("schema_version") == compiler.FREEZE_FRAGMENT_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("mode") == "formal_full"
            and value.get("model_id") == model
            and value.get("status") == "complete"
            and value.get("safe_for_alignment_input") is True
            and value.get("resource_receipts_synthesized") is False,
            f"{model} compiler freeze fragment state changed")
    cells = value.get("development_cells")
    require(isinstance(cells, list) and len(cells) == 16,
            f"{model} compiler freeze cell inventory changed")
    request_count = 0
    for ordinal, cell in enumerate(cells):
        _exact_keys(cell, FREEZE_CELL_KEYS,
                    f"{model} compiler freeze cell {ordinal}")
        require(isinstance(cell.get("cell_receipt"), Mapping)
                and isinstance(cell.get("server_request_receipts"), list)
                and len(cell["server_request_receipts"])
                == freeze.MODEL_LIMITS[model]["request_count"]
                and cell.get("resource_receipt") is None,
                f"{model} compiler freeze cell {ordinal} authority changed")
        request_count += len(cell["server_request_receipts"])
    require(request_count == EXPECTED_REQUESTS[model],
            f"{model} compiler freeze request coverage changed")
    return dict(value)


def _rederive_authoritative_physical_alignment(
    *, model: str, fragment: Mapping[str, Any],
    timing_descriptor: Mapping[str, Any], crop_descriptor: Mapping[str, Any],
    evidence_base: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the frozen physical mapping from its already validated sources."""

    expected_cells = {
        cell_id
        for cell_id, row in annotation._planned_cells(
            "development", "full_two_model"
        ).items()
        if row[0] == model
    }
    model_evidence = {
        "model_id": model,
        "camera_crop_contract": dict(crop_descriptor),
        "generated_target_timing_receipt": dict(timing_descriptor),
        "development_cells": fragment["development_cells"],
    }
    try:
        mapping, alignment_unsigned, _ = freeze._derive_model_alignment(
            model_evidence,
            model=model,
            expected_cells=expected_cells,
            evidence_base=evidence_base,
            require_resources=False,
        )
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"{model} authoritative physical alignment replay failed: {error}"
        ) from error
    mapping_sha256 = sha256_bytes(pretty_bytes(mapping))
    alignment = dict(alignment_unsigned)
    alignment["mapping_receipt_sha256"] = mapping_sha256
    alignment["contract_sha256"] = sha256_bytes(canonical_bytes(alignment))
    return dict(mapping), alignment


def _validate_timing_job(
    *, model: str, value: Mapping[str, Any], sidecar_descriptor: Mapping[str, Any]
) -> None:
    verify_signed(value, f"{model} timing queue job receipt")
    require(value.get("schema_version") == TIMING_JOB_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("model_id") == model
            and value.get("status") == "passed"
            and value.get("decision") == "go"
            and value.get("development_timing_sidecar_valid") is True
            and value.get("safe_to_release_confirmation") is False,
            f"{model} timing queue job did not pass")
    require(value.get("referenced_behavioral_cells") == 16
            and value.get("referenced_behavioral_model_requests") == EXPECTED_REQUESTS[model],
            f"{model} timing queue coverage changed")
    _zero_science(
        value, f"{model} timing queue job", EXPECTED_TIMING_JOB_SCIENCE_COUNTS
    )
    outputs = value.get("outputs")
    require(isinstance(outputs, Mapping), f"{model} timing outputs are missing")
    _same_descriptor(outputs.get("raw_development_timing_sidecar"), sidecar_descriptor,
                     f"{model} timing sidecar/job")


def _alignment_contract_hash(value: Mapping[str, Any], label: str) -> None:
    _exact_keys(value, annotation.ALIGNMENT_CONTRACT_KEYS, label)
    supplied = value.get("contract_sha256")
    require(isinstance(supplied, str) and SHA_RE.fullmatch(supplied) is not None,
            f"{label} contract SHA-256 is invalid")
    unsigned = dict(value)
    unsigned.pop("contract_sha256")
    require(sha256_bytes(canonical_bytes(unsigned)) == supplied,
            f"{label} contract hash changed")


def _validate_crop_contract(value: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Delegate the whole signed transform contract to the witness module."""

    try:
        camera_replay.validate_camera_crop_contract(value, expected_model=model)
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"{model} camera crop contract failed executable validation: {error}"
        ) from error
    width, height = EXPECTED_IMAGE_DIMENSIONS[model]
    require(value.get("schema_version") == CROP_SCHEMA
            and value.get("camera_crop_id") == EXPECTED_CROP_IDS[model]
            and value.get("crop_operation") == EXPECTED_CROP_OPERATIONS[model]
            and value.get("image_width_px") == width
            and value.get("image_height_px") == height
            and value["generated_decoded_crop"].get("canvas_shape")
            == EXPECTED_GENERATED_CANVAS_SHAPES[model]
            and value["generated_decoded_crop"].get("y")
            == EXPECTED_GENERATED_CROPS[model]["y"]
            and value["generated_decoded_crop"].get("x")
            == EXPECTED_GENERATED_CROPS[model]["x"]
            and value["generated_decoded_crop"].get("output_shape")
            == [EXPECTED_GENERATED_CANVAS_SHAPES[model][0], height, width, 3]
            and value["original_camera_replay"].get("output_shape")
            == [height, width, 3],
            f"{model} crop contract identity or geometry changed")
    return dict(value)


def _runtime_python_identity(
    crop: Mapping[str, Any], model: str
) -> dict[str, Any]:
    """Reopen the exact interpreter signed by the model's crop witness."""

    runtime = crop.get("runtime_dependencies")
    require(isinstance(runtime, Mapping), f"{model} signed runtime dependencies are missing")
    value = runtime.get("python")
    require(isinstance(value, Mapping)
            and set(value) == {"path", "sha256", "bytes"},
            f"{model} signed Python identity fields changed")
    raw_path = value.get("path")
    require(isinstance(raw_path, str) and Path(raw_path).is_absolute(),
            f"{model} signed Python path is invalid")
    path = Path(raw_path)
    _reject_symlinks(path, f"{model} signed Python executable")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AnnotationMediaBridgeError(
            f"{model} signed Python executable is unavailable"
        ) from error
    require(resolved == path and resolved.is_file()
            and os.access(resolved, os.X_OK),
            f"{model} signed Python executable is not an exact executable file")
    observed = descriptor(resolved)
    require(observed == dict(value), f"{model} signed Python executable changed")
    return observed


def _write_all(stream: Any, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = stream.write(view)
        require(type(written) is int and written > 0, "camera replay IPC write failed")
        view = view[written:]
    stream.flush()


def _read_exact(
    stream: Any, size: int, *, timeout_s: float | None, label: str
) -> bytes:
    require(type(size) is int and size >= 0, f"{label} byte count is invalid")
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        if deadline is not None:
            wait = deadline - time.monotonic()
            require(wait > 0, f"{label} timed out")
            try:
                ready, _, _ = select.select([stream], [], [], wait)
            except (OSError, ValueError) as error:
                raise AnnotationMediaBridgeError(f"{label} readiness check failed") from error
            require(bool(ready), f"{label} timed out")
        try:
            block = stream.read(remaining)
        except OSError as error:
            raise AnnotationMediaBridgeError(f"{label} read failed") from error
        require(isinstance(block, bytes) and block, f"{label} ended early")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _write_ipc_frame(stream: Any, header: Mapping[str, Any], payload: bytes = b"") -> dict[str, Any]:
    require("body_bytes" not in header and "body_sha256" not in header,
            "camera replay IPC header already contains payload identity")
    signed = sign_document({
        **dict(header),
        "body_bytes": len(payload),
        "body_sha256": sha256_bytes(payload),
    })
    encoded = canonical_bytes(signed)
    require(0 < len(encoded) <= IPC_HEADER_LIMIT_BYTES,
            "camera replay IPC header exceeds its bound")
    _write_all(stream, struct.pack(">Q", len(encoded)) + encoded + payload)
    return signed


def _read_ipc_frame(
    stream: Any, *, maximum_payload_bytes: int, timeout_s: float | None,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    prefix = _read_exact(stream, 8, timeout_s=timeout_s, label=f"{label} prefix")
    header_size = struct.unpack(">Q", prefix)[0]
    require(0 < header_size <= IPC_HEADER_LIMIT_BYTES,
            f"{label} header size is invalid")
    encoded = _read_exact(
        stream, header_size, timeout_s=timeout_s, label=f"{label} header"
    )
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnnotationMediaBridgeError(f"{label} header is invalid JSON") from error
    require(isinstance(header, dict), f"{label} header is not an object")
    verify_signed(header, f"{label} header")
    payload_size = header.get("body_bytes")
    require(type(payload_size) is int and 0 <= payload_size <= maximum_payload_bytes,
            f"{label} payload size is invalid")
    payload = _read_exact(
        stream, payload_size, timeout_s=timeout_s, label=f"{label} payload"
    )
    require(header.get("body_sha256") == sha256_bytes(payload),
            f"{label} payload hash changed")
    return header, payload


def _sequence_sha256(values: Sequence[str]) -> str:
    require(all(isinstance(value, str) and SHA_RE.fullmatch(value) is not None
                for value in values), "camera replay sequence contains an invalid hash")
    return sha256_bytes("".join(f"{value}\n" for value in values).encode("ascii"))


def _replay_worker_command(
    *, model: str, crop: Mapping[str, Any], contract_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Build a helper command whose argv[0] is the signed model interpreter."""

    require(model in MODELS, "camera replay worker model changed")
    python_identity = _runtime_python_identity(crop, model)
    contract_identity = descriptor(contract_path)
    require(load_json(contract_path, f"{model} camera crop contract") == dict(crop),
            f"{model} camera crop contract path/value changed")
    bridge_path = Path(__file__).resolve(strict=True)
    camera_path = CAMERA_REPLAY_PATH.resolve(strict=True)
    command = [
        python_identity["path"],
        str(bridge_path),
        "camera-replay-worker",
        "--model", model,
        "--crop-contract", str(contract_path),
        "--crop-contract-sha256", contract_identity["sha256"],
        "--crop-contract-bytes", str(contract_identity["bytes"]),
        "--bridge-sha256", sha256_file(bridge_path),
        "--camera-replay-sha256", sha256_file(camera_path),
    ]
    return command, python_identity


class _ExactRuntimeReplayClient:
    """One persistent exact-interpreter replay helper for a single model."""

    def __init__(
        self, *, model: str, crop: Mapping[str, Any], contract_path: Path,
        evidence_root: Path,
    ) -> None:
        self.model = model
        self.crop = dict(crop)
        self.contract_path = Path(contract_path).resolve(strict=True)
        self.evidence_root = Path(evidence_root)
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.evidence_root / f"{model.lower()}_stderr.log"
        self.receipt_path = self.evidence_root / f"{model.lower()}_session_receipt.json"
        self._request_headers: list[dict[str, Any]] = []
        self._response_headers: list[dict[str, Any]] = []
        self._input_hashes: list[str] = []
        self._output_hashes: list[str] = []
        self._closed = False
        command, python_identity = _replay_worker_command(
            model=model, crop=crop, contract_path=self.contract_path
        )
        self.python_identity = python_identity
        self.contract_identity = descriptor(self.contract_path)
        self.log = self.log_path.open("xb")
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        try:
            self.process = subprocess.Popen(
                command,
                cwd=WORKSHOP_ROOT,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
                start_new_session=True,
            )
            require(self.process.stdin is not None and self.process.stdout is not None,
                    f"{model} camera replay IPC pipes are unavailable")
            ready, payload = _read_ipc_frame(
                self.process.stdout, maximum_payload_bytes=0,
                timeout_s=IPC_READY_TIMEOUT_S, label=f"{model} replay ready",
            )
            require(payload == b""
                    and ready.get("schema_version") == REPLAY_IPC_SCHEMA
                    and ready.get("message_type") == "ready"
                    and ready.get("model_id") == model
                    and ready.get("status") == "exact_runtime_validated"
                    and ready.get("contract_payload_sha256") == crop.get("payload_sha256")
                    and ready.get("python") == python_identity
                    and ready.get("bridge_source", {}).get("sha256")
                    == sha256_file(Path(__file__).resolve(strict=True))
                    and ready.get("camera_replay_source", {}).get("sha256")
                    == sha256_file(CAMERA_REPLAY_PATH.resolve(strict=True)),
                    f"{model} replay helper ready identity changed")
            self.session_id = ready.get("session_id")
            require(isinstance(self.session_id, str) and SHA_RE.fullmatch(self.session_id),
                    f"{model} replay helper session identity is invalid")
            self.ready_header = ready
        except BaseException:
            self.abort()
            raise

    def replay(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        require(not self._closed and self.process.poll() is None,
                f"{self.model} replay helper is not live")
        image = _rgb_u8(image, f"{self.model} replay helper input")
        require(list(image.shape) == [720, 1280, 3],
                f"{self.model} replay helper input shape changed")
        ordinal = len(self._request_headers)
        require(ordinal < IPC_MAX_REQUESTS, "camera replay helper request bound exceeded")
        payload = image.tobytes(order="C")
        request = _write_ipc_frame(self.process.stdin, {
            "schema_version": REPLAY_IPC_SCHEMA,
            "message_type": "replay_request",
            "model_id": self.model,
            "session_id": self.session_id,
            "ordinal": ordinal,
            "input_shape": [720, 1280, 3],
            "input_dtype": "uint8",
            "input_value_sha256": sha256_bytes(payload),
        }, payload)
        width, height = EXPECTED_IMAGE_DIMENSIONS[self.model]
        response, output_bytes = _read_ipc_frame(
            self.process.stdout, maximum_payload_bytes=height * width * 3,
            timeout_s=IPC_REPLAY_TIMEOUT_S,
            label=f"{self.model} replay response {ordinal}",
        )
        require(response.get("schema_version") == REPLAY_IPC_SCHEMA
                and response.get("message_type") == "replay_response"
                and response.get("status") == "passed"
                and response.get("model_id") == self.model
                and response.get("session_id") == self.session_id
                and response.get("ordinal") == ordinal
                and response.get("input_value_sha256") == request["input_value_sha256"]
                and response.get("output_shape") == [height, width, 3]
                and response.get("output_dtype") == "uint8"
                and response.get("output_value_sha256") == sha256_bytes(output_bytes),
                f"{self.model} replay helper response binding changed")
        output = np.frombuffer(output_bytes, dtype=np.uint8).copy().reshape(height, width, 3)
        self._request_headers.append(request)
        self._response_headers.append(response)
        self._input_hashes.append(request["input_value_sha256"])
        self._output_hashes.append(response["output_value_sha256"])
        return output, {
            "schema_version": REPLAY_IPC_SCHEMA,
            "session_id": self.session_id,
            "ordinal": ordinal,
            "input_value_sha256": request["input_value_sha256"],
            "output_value_sha256": response["output_value_sha256"],
            "exact_signed_python_subprocess": True,
        }

    def close(self) -> dict[str, Any]:
        require(not self._closed, f"{self.model} replay helper already closed")
        request_count = len(self._request_headers)
        require(request_count > 0, f"{self.model} replay helper processed no frames")
        transcript = sha256_bytes(b"".join(
            canonical_bytes(row)
            for pair in zip(self._request_headers, self._response_headers)
            for row in pair
        ))
        close_header = _write_ipc_frame(self.process.stdin, {
            "schema_version": REPLAY_IPC_SCHEMA,
            "message_type": "close_request",
            "model_id": self.model,
            "session_id": self.session_id,
            "request_count": request_count,
            "input_sequence_sha256": _sequence_sha256(self._input_hashes),
            "output_sequence_sha256": _sequence_sha256(self._output_hashes),
            "transcript_sha256": transcript,
        })
        response, payload = _read_ipc_frame(
            self.process.stdout, maximum_payload_bytes=0,
            timeout_s=IPC_CLOSE_TIMEOUT_S, label=f"{self.model} replay close",
        )
        require(payload == b""
                and response.get("schema_version") == REPLAY_IPC_SCHEMA
                and response.get("message_type") == "close_response"
                and response.get("status") == "passed"
                and response.get("model_id") == self.model
                and response.get("session_id") == self.session_id
                and response.get("request_count") == request_count
                and response.get("input_sequence_sha256")
                == close_header["input_sequence_sha256"]
                and response.get("output_sequence_sha256")
                == close_header["output_sequence_sha256"]
                and response.get("transcript_sha256") == transcript,
                f"{self.model} replay helper close receipt changed")
        self.process.stdin.close()
        self.process.stdout.close()
        try:
            exit_code = self.process.wait(timeout=IPC_CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            self.abort()
            raise AnnotationMediaBridgeError(
                f"{self.model} replay helper did not exit"
            ) from error
        self.log.close()
        self._closed = True
        require(exit_code == 0, f"{self.model} replay helper exit changed: {exit_code}")
        receipt = sign_document({
            "schema_version": REPLAY_SESSION_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "passed_exact_signed_runtime_replay",
            "model_id": self.model,
            "session_id": self.session_id,
            "camera_crop_contract": self.contract_identity,
            "camera_crop_contract_payload_sha256": self.crop["payload_sha256"],
            "python": self.python_identity,
            "bridge_source": descriptor(Path(__file__).resolve(strict=True)),
            "camera_replay_source": descriptor(CAMERA_REPLAY_PATH.resolve(strict=True)),
            "stderr_log": _relative_descriptor(self.log_path, self.evidence_root.parent),
            "request_count": request_count,
            "input_sequence_sha256": close_header["input_sequence_sha256"],
            "output_sequence_sha256": close_header["output_sequence_sha256"],
            "transcript_sha256": transcript,
            "camera_api": "replay_original_camera_frame",
            "runtime_dependency_validation": "exact_signed_contract_match",
            "cuda_visible_devices": "",
            "model_loaded": False,
            "simulator_state_render_used": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
        })
        _write_exclusive(self.receipt_path, pretty_bytes(receipt))
        return receipt

    def abort(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except BaseException:
                    pass
        for name in ("stdin", "stdout"):
            stream = getattr(process, name, None) if process is not None else None
            if stream is not None:
                try:
                    stream.close()
                except BaseException:
                    pass
        log = getattr(self, "log", None)
        if log is not None and not log.closed:
            log.close()
        self._closed = True


def _run_camera_replay_worker(args: argparse.Namespace) -> None:
    """Run the camera API in the exact interpreter named by its signed contract."""

    model = args.model
    require(model in MODELS, "camera replay worker model changed")
    contract_path = Path(args.crop_contract)
    identity = descriptor(contract_path)
    require(identity["sha256"] == args.crop_contract_sha256
            and identity["bytes"] == args.crop_contract_bytes,
            "camera replay worker crop contract bytes changed")
    require(sha256_file(Path(__file__).resolve(strict=True)) == args.bridge_sha256,
            "camera replay worker bridge source changed")
    require(sha256_file(CAMERA_REPLAY_PATH.resolve(strict=True))
            == args.camera_replay_sha256,
            "camera replay worker camera API source changed")
    crop = _validate_crop_contract(load_json(contract_path, "camera crop contract"), model)
    python_identity = _runtime_python_identity(crop, model)
    observed_python = descriptor(Path(sys.executable).resolve(strict=True))
    require(observed_python == python_identity,
            "camera replay worker is running under the wrong interpreter")
    try:
        camera_replay._verify_runtime_dependency_chain(model, crop)
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"{model} exact runtime dependency preflight failed: {error}"
        ) from error
    session_id = sha256_bytes(canonical_bytes({
        "model_id": model,
        "contract_payload_sha256": crop["payload_sha256"],
        "python": python_identity,
        "bridge_sha256": args.bridge_sha256,
        "camera_replay_sha256": args.camera_replay_sha256,
    }))
    _write_ipc_frame(sys.stdout.buffer, {
        "schema_version": REPLAY_IPC_SCHEMA,
        "message_type": "ready",
        "status": "exact_runtime_validated",
        "model_id": model,
        "session_id": session_id,
        "contract_payload_sha256": crop["payload_sha256"],
        "python": python_identity,
        "bridge_source": descriptor(Path(__file__).resolve(strict=True)),
        "camera_replay_source": descriptor(CAMERA_REPLAY_PATH.resolve(strict=True)),
    })
    request_headers: list[dict[str, Any]] = []
    response_headers: list[dict[str, Any]] = []
    input_hashes: list[str] = []
    output_hashes: list[str] = []
    while True:
        header, payload = _read_ipc_frame(
            sys.stdin.buffer, maximum_payload_bytes=720 * 1280 * 3,
            timeout_s=None, label=f"{model} replay worker request",
        )
        require(header.get("schema_version") == REPLAY_IPC_SCHEMA
                and header.get("model_id") == model
                and header.get("session_id") == session_id,
                "camera replay worker request identity changed")
        if header.get("message_type") == "close_request":
            require(payload == b""
                    and header.get("request_count") == len(request_headers)
                    and header.get("input_sequence_sha256")
                    == _sequence_sha256(input_hashes)
                    and header.get("output_sequence_sha256")
                    == _sequence_sha256(output_hashes),
                    "camera replay worker close sequence changed")
            transcript = sha256_bytes(b"".join(
                canonical_bytes(row)
                for pair in zip(request_headers, response_headers)
                for row in pair
            ))
            require(header.get("transcript_sha256") == transcript,
                    "camera replay worker transcript changed")
            _write_ipc_frame(sys.stdout.buffer, {
                "schema_version": REPLAY_IPC_SCHEMA,
                "message_type": "close_response",
                "status": "passed",
                "model_id": model,
                "session_id": session_id,
                "request_count": len(request_headers),
                "input_sequence_sha256": _sequence_sha256(input_hashes),
                "output_sequence_sha256": _sequence_sha256(output_hashes),
                "transcript_sha256": transcript,
            })
            return
        ordinal = len(request_headers)
        require(header.get("message_type") == "replay_request"
                and header.get("ordinal") == ordinal
                and ordinal < IPC_MAX_REQUESTS
                and header.get("input_shape") == [720, 1280, 3]
                and header.get("input_dtype") == "uint8"
                and len(payload) == 720 * 1280 * 3
                and header.get("input_value_sha256") == sha256_bytes(payload),
                "camera replay worker input binding changed")
        image = np.frombuffer(payload, dtype=np.uint8).copy().reshape(720, 1280, 3)
        try:
            output = camera_replay.replay_original_camera_frame(image, crop)
        except BaseException as error:
            raise AnnotationMediaBridgeError(
                f"{model} exact camera API replay failed: {error}"
            ) from error
        output = _rgb_u8(np.asarray(output), f"{model} replay worker output")
        width, height = EXPECTED_IMAGE_DIMENSIONS[model]
        require(list(output.shape) == [height, width, 3],
                "camera replay worker output shape changed")
        output_payload = output.tobytes(order="C")
        response = _write_ipc_frame(sys.stdout.buffer, {
            "schema_version": REPLAY_IPC_SCHEMA,
            "message_type": "replay_response",
            "status": "passed",
            "model_id": model,
            "session_id": session_id,
            "ordinal": ordinal,
            "input_value_sha256": header["input_value_sha256"],
            "output_shape": [height, width, 3],
            "output_dtype": "uint8",
            "output_value_sha256": sha256_bytes(output_payload),
        }, output_payload)
        request_headers.append(header)
        response_headers.append(response)
        input_hashes.append(header["input_value_sha256"])
        output_hashes.append(response["output_value_sha256"])


def _validate_mapping_and_alignment(
    *,
    model: str,
    mapping: Mapping[str, Any],
    mapping_path: Path,
    alignment: Mapping[str, Any],
    crop: Mapping[str, Any],
    timing_descriptor: Mapping[str, Any],
    timing_sidecar: Mapping[str, Any],
    authoritative_mapping: Mapping[str, Any],
    authoritative_alignment: Mapping[str, Any],
) -> None:
    _exact_keys(mapping, MAPPING_RECEIPT_KEYS,
                f"{model} physical alignment receipt")
    verify_signed(mapping, f"{model} physical alignment receipt")
    expected_cells = sorted(
        cell_id
        for cell_id, row in annotation._planned_cells(
            "development", "full_two_model"
        ).items()
        if row[0] == model
    )
    require(mapping.get("schema_version") == MAPPING_SCHEMA
            and mapping.get("study_id") == STUDY_ID
            and mapping.get("model_id") == model
            and mapping.get("status") == "qualified_from_complete_development_native_timing"
            and mapping.get("development_cell_ids") == expected_cells
            and mapping.get("development_cell_count") == 16
            and mapping.get("development_request_count") == EXPECTED_REQUESTS[model],
            f"{model} physical alignment receipt changed")
    require(mapping.get("request_semantics") == {
        key: freeze.MODEL_LIMITS[model][key]
        for key in (
            "returned_action_horizon", "unchanged_executed_prefix_horizon",
            "action_space", "seed_semantics", "temporal_context",
        )
    }, f"{model} physical mapping request semantics changed")
    _alignment_contract_hash(alignment, f"{model} alignment contract")
    require(alignment.get("model_id") == model
            and alignment.get("mapping_receipt_id") == mapping.get("receipt_id")
            and alignment.get("mapping_receipt_sha256") == sha256_file(mapping_path)
            and alignment.get("camera_id") == crop.get("camera_id")
            and alignment.get("camera_crop_id") == crop.get("camera_crop_id")
            and alignment.get("camera_crop_sha256") == crop.get("payload_sha256")
            and (alignment.get("image_width_px"), alignment.get("image_height_px"))
            == EXPECTED_IMAGE_DIMENSIONS[model]
            and isinstance(alignment.get("early_horizon"), Mapping),
            f"{model} alignment/crop/mapping binding changed")
    primary = mapping.get("primary_target")
    early = mapping.get("early_target")
    require(isinstance(primary, Mapping) and isinstance(early, Mapping),
            f"{model} alignment lacks primary or early target")
    target_keys = {
        "generated_frame_index", "target_physical_time_s", "status",
        "target_executed_action_offset", "eligible_request_count",
        "full_prefix_request_count", "max_camera_timestamp_residual_s",
        "max_physics_timestamp_residual_s",
    }
    clocks = mapping.get("measured_clock_intervals")
    require(isinstance(clocks, Mapping)
            and type(clocks.get("timestamp_tolerance_s")) in (int, float)
            and math.isfinite(float(clocks["timestamp_tolerance_s"]))
            and float(clocks["timestamp_tolerance_s"]) > 0,
            f"{model} measured clock tolerance is invalid")
    for name, target in (("primary", primary), ("early", early)):
        _exact_keys(target, target_keys, f"{model} {name} mapping target")
        residuals = (
            target.get("max_camera_timestamp_residual_s"),
            target.get("max_physics_timestamp_residual_s"),
        )
        expected_target_counts = EXPECTED_MAPPING_TARGET_COUNTS[model][name]
        require(target.get("status") == "qualified"
                and type(target.get("generated_frame_index")) is int
                and target["generated_frame_index"] >= 0
                and type(target.get("target_executed_action_offset")) is int
                and target["target_executed_action_offset"] > 0
                and type(target.get("target_physical_time_s")) in (int, float)
                and math.isfinite(float(target["target_physical_time_s"]))
                and float(target["target_physical_time_s"]) > 0
                and target.get("eligible_request_count")
                == expected_target_counts["eligible"]
                and target.get("full_prefix_request_count")
                == expected_target_counts["full_prefix"]
                and all(type(value) in (int, float)
                        and math.isfinite(float(value))
                        and 0 <= float(value) <= float(clocks["timestamp_tolerance_s"])
                        for value in residuals),
                f"{model} {name} mapping target does not match exact qualified coverage")
        expected_frame, expected_boundary = EXPECTED_ALIGNMENT_TARGET_IDENTITIES[
            model
        ][name]
        require(
            (
                target.get("generated_frame_index"),
                target.get("target_executed_action_offset"),
            )
            == (expected_frame, expected_boundary),
            f"{model} {name} mapping target identity changed",
        )
    mapping_rows = mapping.get("frame_to_physical_time")
    require(isinstance(mapping_rows, list) and mapping_rows
            and all(isinstance(row, Mapping) for row in mapping_rows)
            and sum(row == primary for row in mapping_rows) == 1
            and sum(row == early for row in mapping_rows) == 1,
            f"{model} primary/early targets are not uniquely bound to the mapping table")
    expected_frame_indices = (
        set(range(1, 33)) if model == "N3" else set(range(1, 3))
    )
    require(len(mapping_rows) == len(expected_frame_indices)
            and {row.get("generated_frame_index") for row in mapping_rows}
            == expected_frame_indices,
            f"{model} physical mapping target table coverage changed")
    timing_targets = timing_sidecar.get("generated_targets")
    require(isinstance(timing_targets, list)
            and len(timing_targets) == len(mapping_rows)
            and all(isinstance(row, Mapping) for row in timing_targets),
            f"{model} validated timing target inventory changed")
    timing_by_frame = {
        row.get("generated_frame_index"): row for row in timing_targets
    }
    require(len(timing_by_frame) == len(timing_targets)
            and set(timing_by_frame) == expected_frame_indices,
            f"{model} validated timing target coverage changed")
    for row in mapping_rows:
        _exact_keys(row, target_keys, f"{model} physical mapping target row")
        source_target = timing_by_frame[row["generated_frame_index"]]
        require(
            row.get("target_physical_time_s")
            == source_target.get("target_physical_time_s"),
            f"{model} physical mapping target time differs from timing sidecar",
        )
        bindings = timing_sidecar.get("request_timing_bindings")
        require(
            isinstance(bindings, list)
            and len(bindings) == EXPECTED_REQUESTS[model],
            f"{model} timing request binding inventory changed",
        )
        matched_rows: list[Mapping[str, Any]] = []
        applicable_count = 0
        full_prefix_count = 0
        prefix = freeze.MODEL_LIMITS[model]["unchanged_executed_prefix_horizon"]
        for ordinal, binding in enumerate(bindings):
            require(isinstance(binding, Mapping),
                    f"{model} timing binding {ordinal} changed")
            decoded = binding.get("decoded_output_timing")
            applies = model == "N3" or (
                isinstance(decoded, Mapping)
                and decoded.get("source_timing_mapping_applies") is True
            )
            targets = binding.get("target_bindings")
            require(isinstance(targets, list),
                    f"{model} timing binding {ordinal} target inventory changed")
            if not applies:
                require(targets == [],
                        f"{model} timing-unmapped binding acquired targets")
                continue
            applicable_count += 1
            if binding.get("executed_actions") == prefix:
                full_prefix_count += 1
            matches = [
                target for target in targets
                if isinstance(target, Mapping)
                and target.get("generated_frame_index")
                == row["generated_frame_index"]
            ]
            require(len(matches) == 1,
                    f"{model} timing binding {ordinal} target coverage changed")
            target = matches[0]
            require(
                target.get("executed_control_boundary")
                == row["target_executed_action_offset"]
                and target.get("authority_target_physical_time_s")
                == row["target_physical_time_s"]
                and target.get("status") in {
                    "matched_native_request_clocks",
                    "not_executed_in_truncated_prefix",
                },
                f"{model} timing binding {ordinal} target identity/status changed",
            )
            if target["status"] == "matched_native_request_clocks":
                matched_rows.append(target)
        expected_eligible = (
            240
            if model == "N3" and row["generated_frame_index"] <= 2
            else 224
        )
        camera_residuals = [
            target.get("camera_authority_residual_s") for target in matched_rows
        ]
        physics_residuals = [
            target.get("physics_authority_residual_s") for target in matched_rows
        ]
        require(
            applicable_count == EXPECTED_TIMING_CAPABLE[model]
            and len(matched_rows) == expected_eligible
            and full_prefix_count == 224
            and row.get("status") == "qualified"
            and row.get("eligible_request_count") == len(matched_rows)
            and row.get("full_prefix_request_count") == full_prefix_count
            and all(
                type(value) in (int, float)
                and math.isfinite(float(value))
                and 0 <= float(value) <= float(clocks["timestamp_tolerance_s"])
                for value in camera_residuals + physics_residuals
            )
            and row.get("max_camera_timestamp_residual_s")
            == max(camera_residuals)
            and row.get("max_physics_timestamp_residual_s")
            == max(physics_residuals),
            f"{model} physical mapping target row coverage/residuals changed",
        )
    qualified_rows = [row for row in mapping_rows if row.get("status") == "qualified"]
    require(qualified_rows, f"{model} physical mapping has no qualified targets")
    require(all(type(row.get("generated_frame_index")) is int
                and row["generated_frame_index"] >= 0
                and type(row.get("target_physical_time_s")) in (int, float)
                and math.isfinite(float(row["target_physical_time_s"]))
                and float(row["target_physical_time_s"]) > 0
                for row in qualified_rows),
            f"{model} qualified mapping table rows are invalid")
    expected_qualified_pairs = (
        {(index, index) for index in range(1, 33)}
        if model == "N3" else {(1, 3), (2, 6)}
    )
    require({
        (row.get("generated_frame_index"), row.get("target_executed_action_offset"))
        for row in qualified_rows
    } == expected_qualified_pairs,
            f"{model} qualified frame/action-boundary coverage changed")
    canonical_primary = max(
        qualified_rows,
        key=lambda row: (
            float(row.get("target_physical_time_s", -1)),
            int(row.get("generated_frame_index", -1)),
        ),
    )
    earlier_rows = [
        row for row in qualified_rows
        if float(row.get("target_physical_time_s", math.inf))
        < float(canonical_primary.get("target_physical_time_s", -math.inf))
    ]
    require(earlier_rows, f"{model} physical mapping lacks an earlier qualified target")
    canonical_early = min(
        earlier_rows,
        key=lambda row: (
            float(row.get("target_physical_time_s", math.inf)),
            int(row.get("generated_frame_index", sys.maxsize)),
        ),
    )
    require(primary == canonical_primary and early == canonical_early,
            f"{model} mapping primary/early targets are not canonically derived")
    boundary = mapping.get("timing_claim_boundary")
    require(isinstance(boundary, Mapping)
            and boundary.get("time_source_kind")
            == "native_runtime_exposed_target_offsets"
            and boundary.get("clock_bridge")
            == "elapsed physical seconds from request current original-camera capture"
            and boundary.get("presentation_video_fps_used") is False
            and boundary.get("conditioning_fps_used_as_target_timing") is False
            and boundary.get("generated_frame_index_interpreted_as_action_index") is False,
            f"{model} physical mapping timing authority changed")
    if model == "D1":
        require(boundary.get("generated_targets_scope")
                == freeze.D1_GENERATED_TARGETS_SCOPE
                and boundary.get("request_timing_coverage") == EXPECTED_D1_COVERAGE
                and boundary.get("incremental_standalone_decodes_assigned_target_times")
                is False
                and boundary.get("timing_unmapped_requests_eligible") is False,
                "D1 physical mapping missingness boundary changed")
    require(alignment.get("generated_frame_index") == primary.get("generated_frame_index")
            and alignment.get("target_executed_action_offset")
            == primary.get("target_executed_action_offset")
            and alignment.get("primary_horizon_s") == primary.get("target_physical_time_s"),
            f"{model} primary alignment target changed")
    early_contract = alignment["early_horizon"]
    require(early_contract == {
        "horizon_s": early.get("target_physical_time_s"),
        "generated_frame_index": early.get("generated_frame_index"),
        "target_executed_action_offset": early.get("target_executed_action_offset"),
    }, f"{model} early alignment target changed")
    require(alignment.get("control_step_s") == clocks.get("control_step_s_min")
            and alignment.get("captured_frame_interval_s")
            == clocks.get("captured_frame_interval_s_min")
            and alignment.get("timestamp_tolerance_s")
            == clocks["timestamp_tolerance_s"],
            f"{model} alignment/measured-clock contract changed")
    source_receipts = mapping.get("source_receipts")
    require(isinstance(source_receipts, Mapping), f"{model} mapping source receipts missing")
    _same_descriptor(source_receipts.get("generated_target_timing"), timing_descriptor,
                     f"{model} mapping/timing sidecar")
    camera = mapping.get("camera")
    require(isinstance(camera, Mapping)
            and camera.get("camera_id") == crop.get("camera_id")
            and camera.get("camera_crop_id") == crop.get("camera_crop_id")
            and camera.get("camera_crop_sha256") == crop.get("payload_sha256")
            and (camera.get("image_width_px"), camera.get("image_height_px"))
            == EXPECTED_IMAGE_DIMENSIONS[model]
            and camera.get("crop_operation") == crop.get("crop_operation"),
            f"{model} mapping camera contract changed")
    require(mapping == authoritative_mapping,
            f"{model} physical mapping differs from authoritative source replay")
    require(sha256_file(mapping_path) == sha256_bytes(pretty_bytes(authoritative_mapping)),
            f"{model} physical mapping bytes differ from authoritative source replay")
    require(alignment == authoritative_alignment,
            f"{model} alignment contract differs from authoritative source replay")


def _target_binding(
    binding: Mapping[str, Any], target: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    rows = binding.get("target_bindings")
    require(isinstance(rows, list), "timing target binding inventory is invalid")
    matches = [
        row for row in rows
        if isinstance(row, Mapping)
        and row.get("generated_frame_index") == target.get("generated_frame_index")
        and row.get("executed_control_boundary")
        == target.get("target_executed_action_offset")
        and math.isclose(
            float(row.get("authority_target_physical_time_s", -1)),
            float(target.get("horizon_s", target.get("primary_horizon_s", -2))),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ]
    require(len(matches) <= 1, "timing sidecar duplicates an alignment target")
    return matches[0] if matches else None


def _build_request_inventory(
    *,
    roster_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    timing_by_model: Mapping[str, Mapping[str, Any]],
    alignment_by_model: Mapping[str, Mapping[str, Any]],
    mapping_sha_by_model: Mapping[str, str],
    provenance_base_by_model: Mapping[str, Path],
    inventory_finalized_at: str,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    combined_roster = [dict(row) for model in MODELS for row in roster_by_model[model]]
    rows: list[dict[str, Any]] = []
    rich: dict[tuple[str, str, int], dict[str, Any]] = {}
    for model in MODELS:
        provenance_rows = list(provenance_by_model[model])
        sidecar = timing_by_model[model]
        bindings = sidecar.get("request_timing_bindings")
        require(isinstance(bindings, list) and len(bindings) == len(provenance_rows),
                f"{model} timing/provenance request counts differ")
        by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
        for binding in bindings:
            require(isinstance(binding, Mapping), f"{model} timing binding is invalid")
            key = (model, binding.get("cell_id"), binding.get("request_index"))
            require(isinstance(key[1], str) and type(key[2]) is int and key not in by_key,
                    f"{model} timing binding identity is invalid or duplicated")
            by_key[key] = binding
        alignment = alignment_by_model[model]
        primary_target = {
            "generated_frame_index": alignment["generated_frame_index"],
            "target_executed_action_offset": alignment["target_executed_action_offset"],
            "horizon_s": alignment["primary_horizon_s"],
        }
        for source in provenance_rows:
            key = (model, source["cell_id"], source["request_index"])
            require(key in by_key, f"{model} provenance request lacks timing binding: {key}")
            binding = by_key[key]
            _same_descriptor(binding.get("source_request_receipt"),
                             source.get("official_request_receipt"),
                             f"{model} {key[1]} request receipt")
            _same_descriptor(binding.get("adapter_completion"), source.get("adapter_completion"),
                             f"{model} {key[1]} completion")
            _same_descriptor(binding.get("adapter_journal"), source.get("adapter_journal"),
                             f"{model} {key[1]} journal")
            require(binding.get("action_step_start") == source.get("action_step_start")
                    and binding.get("executed_actions") == source.get("executed_prefix_actions")
                    and binding.get("request_current_observation_id")
                    == source.get("current_observation_id"),
                    f"{model} {key[1]} request schedule differs across timing/compiler")
            response_binding = binding.get("recorder_response_request_binding")
            require(isinstance(response_binding, Mapping)
                    and response_binding.get("payload_sha256")
                    == source.get("recorder_response_payload_sha256"),
                    f"{model} {key[1]} recorder response binding changed")
            decoded = binding.get("decoded_output_timing") if model == "D1" else None
            mapping_applies = model == "N3" or (
                isinstance(decoded, Mapping)
                and decoded.get("source_timing_mapping_applies") is True
            )
            if model == "D1" and not mapping_applies:
                require(isinstance(decoded, Mapping)
                        and decoded.get("source_timing_status")
                        == "unmapped_incremental_standalone_decode"
                        and binding.get("target_bindings") == []
                        and binding.get("timing_eligible_target_count") == 0
                        and binding.get("eligible_for_timed_target_sampling") is False,
                        f"D1 {key[1]} unmapped request acquired a target")
            target = _target_binding(binding, primary_target) if mapping_applies else None
            executed_prefix = source["executed_prefix_actions"]
            within_prefix = alignment["target_executed_action_offset"] <= executed_prefix
            if mapping_applies:
                require(target is not None, f"{model} {key[1]} mapped request lacks primary target")
                expected_status = (
                    "matched_native_request_clocks" if within_prefix
                    else "not_executed_in_truncated_prefix"
                )
                require(target.get("status") == expected_status,
                        f"{model} {key[1]} primary target status changed")
            technical_valid = bool(mapping_applies)
            if technical_valid and within_prefix:
                residuals = [
                    target.get("camera_authority_residual_s"),
                    target.get("physics_authority_residual_s"),
                ]
                require(all(type(item) in (int, float) and math.isfinite(float(item))
                            and float(item) >= 0 for item in residuals),
                        f"{model} {key[1]} primary residual is invalid")
                timestamp_error = max(float(item) for item in residuals)
            else:
                timestamp_error = None
            action_manifest = source.get("action_manifest")
            recording_receipt = source.get("recording_receipt")
            require(isinstance(action_manifest, Mapping)
                    and isinstance(recording_receipt, Mapping),
                    f"{model} {key[1]} compiler recording descriptors are missing")
            row = {
                "source_request_id": source["source_request_id"],
                "cell_id": source["cell_id"],
                "source_video_id": source["source_video_id"],
                "source_video_sha256": source["source_video"]["sha256"],
                "model_id": model,
                "layout_pair_id": source["layout_pair_id"],
                "condition_id": source["condition_id"],
                "episode_id": source["recording_id"],
                "request_index": source["request_index"],
                "request_start_action_index": source["action_step_start"],
                "action_manifest_sha256": action_manifest["sha256"],
                "camera_id": alignment["camera_id"],
                "camera_crop_id": alignment["camera_crop_id"],
                "camera_crop_sha256": alignment["camera_crop_sha256"],
                "alignment_contract_id": alignment["contract_id"],
                "alignment_contract_sha256": alignment["contract_sha256"],
                "technical_valid": technical_valid,
                "technical_invalid_reason": (
                    None if technical_valid else "forecast_timing_unavailable"
                ),
                "camera_identity_match": binding.get("camera_id") == alignment["camera_id"],
                "target_within_executed_prefix": within_prefix,
                "executed_prefix_actions": executed_prefix,
                "target_executed_action_offset": alignment["target_executed_action_offset"],
                "generated_frame_index": alignment["generated_frame_index"],
                "target_physical_time_s": alignment["primary_horizon_s"],
                "timestamp_error_s": timestamp_error,
                "timestamp_tolerance_s": alignment["timestamp_tolerance_s"],
                "history_mode": source["history_mode"],
                "early_horizon_supported": alignment["early_horizon"] is not None,
                "alignment_receipt_id": alignment["mapping_receipt_id"],
                "alignment_receipt_sha256": mapping_sha_by_model[model],
            }
            rows.append(row)
            rich[key] = {
                "source": source,
                "binding": binding,
                "inventory": row,
                "primary_target_binding": target,
                "provenance_base": provenance_base_by_model[model],
            }
        require(set(by_key) == {key for key in rich if key[0] == model},
                f"{model} timing sidecar contains extra requests")
    inventory = {
        "schema_version": REQUEST_INVENTORY_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "development",
        "cohort_branch": "full_two_model",
        "inventory_complete": True,
        "inventory_finalized_at": inventory_finalized_at,
        "annotation_state": "not_started",
        "episode_roster": combined_roster,
        "alignment_contracts": [dict(alignment_by_model[model]) for model in MODELS],
        "requests": rows,
    }
    return inventory, rich


def _assert_inventory_and_selection(
    inventory: Mapping[str, Any], selection: Mapping[str, Any]
) -> None:
    try:
        annotation.verify_signed(selection, "request selection")
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"request selection signature failed: {error}"
        ) from error
    requests = inventory.get("requests")
    roster = inventory.get("episode_roster")
    require(isinstance(requests, list) and len(requests) == EXPECTED_TOTAL_REQUESTS,
            "request inventory is not exactly 1,152 rows")
    require(isinstance(roster, list) and len(roster) == EXPECTED_EPISODES,
            "request inventory is not exactly 32 episodes")
    wrappers = selection.get("requests")
    episodes = selection.get("episodes")
    require(isinstance(wrappers, list) and len(wrappers) == EXPECTED_TOTAL_REQUESTS,
            "request selection does not contain exactly 1,152 wrappers")
    require(isinstance(episodes, list) and len(episodes) == EXPECTED_EPISODES,
            "request selection does not contain exactly 32 episode rows")
    selection_roster = selection.get("episode_roster")
    selection_alignments = selection.get("alignment_contracts")
    require(isinstance(selection_roster, list)
            and len(selection_roster) == EXPECTED_EPISODES
            and {row.get("cell_id"): row for row in selection_roster
                 if isinstance(row, Mapping)}
            == {row.get("cell_id"): row for row in roster
                if isinstance(row, Mapping)}
            and isinstance(selection_alignments, list)
            and {row.get("model_id"): row for row in selection_alignments
                 if isinstance(row, Mapping)}
            == {row.get("model_id"): row
                for row in inventory.get("alignment_contracts", [])
                if isinstance(row, Mapping)},
            "request selection changed the source roster or alignment contracts")
    inventory_by_id: dict[str, Mapping[str, Any]] = {}
    for row in requests:
        source_id = row.get("source_request_id") if isinstance(row, Mapping) else None
        require(isinstance(source_id, str) and source_id not in inventory_by_id,
                "request inventory source identity is invalid or duplicated")
        inventory_by_id[source_id] = row
    wrapper_by_id: dict[str, Mapping[str, Any]] = {}
    for wrapper in wrappers:
        source = wrapper.get("source") if isinstance(wrapper, Mapping) else None
        source_id = source.get("source_request_id") if isinstance(source, Mapping) else None
        require(isinstance(source_id, str) and source_id not in wrapper_by_id
                and source_id in inventory_by_id
                and source == inventory_by_id[source_id],
                "request selection source coverage changed")
        wrapper_by_id[source_id] = wrapper
    require(set(wrapper_by_id) == set(inventory_by_id),
            "request selection does not cover the inventory one-to-one")
    roster_cells = {
        row.get("cell_id"): row for row in roster if isinstance(row, Mapping)
    }
    require(len(roster_cells) == EXPECTED_EPISODES,
            "request inventory episode identity is duplicated")
    episode_by_cell: dict[str, Mapping[str, Any]] = {}
    for episode in episodes:
        cell_id = episode.get("cell_id") if isinstance(episode, Mapping) else None
        require(isinstance(cell_id, str) and cell_id in roster_cells
                and cell_id not in episode_by_cell
                and episode.get("episode_id") == roster_cells[cell_id].get("recording_id")
                and episode.get("recording_status") == "valid_complete",
                "request selection episode coverage changed")
        episode_by_cell[cell_id] = episode
    require(set(episode_by_cell) == set(roster_cells),
            "request selection episode rows do not cover the roster")
    request_counts = Counter(row["model_id"] for row in requests)
    require(dict(request_counts) == EXPECTED_REQUESTS, "per-model request counts changed")
    timing_capable = Counter(
        row["model_id"] for row in requests if row["technical_valid"]
    )
    require(dict(timing_capable) == EXPECTED_TIMING_CAPABLE,
            "per-model source-timing-capable counts changed")
    d1_unmapped = [
        row for row in requests
        if row["model_id"] == "D1" and not row["technical_valid"]
    ]
    require(len(d1_unmapped) == EXPECTED_D1_UNMAPPED
            and all(row["technical_invalid_reason"] == "forecast_timing_unavailable"
                    and row["timestamp_error_s"] is None for row in d1_unmapped),
            "D1 timing-unavailable rows were omitted or altered")
    counts = selection.get("counts")
    require(isinstance(counts, Mapping)
            and counts.get("requests") == EXPECTED_TOTAL_REQUESTS
            and counts.get("eligible") == EXPECTED_TOTAL_ELIGIBLE
            and counts.get("selected") == EXPECTED_TOTAL_SELECTED
            and counts.get("episodes") == EXPECTED_EPISODES
            and counts.get("zero_eligible_episodes") == 0,
            "request selection exact counts changed")
    by_model_eligible: Counter[str] = Counter()
    by_model_selected: Counter[str] = Counter()
    by_cell_wrappers: Counter[str] = Counter()
    by_cell_eligible: Counter[str] = Counter()
    by_cell_selected: Counter[str] = Counter()
    for wrapper in wrappers:
        require(isinstance(wrapper, Mapping) and isinstance(wrapper.get("source"), Mapping),
                "selection request wrapper is invalid")
        model = wrapper["source"]["model_id"]
        cell_id = wrapper["source"]["cell_id"]
        by_cell_wrappers[cell_id] += 1
        if wrapper.get("timing_camera_action_eligible"):
            by_model_eligible[model] += 1
            by_cell_eligible[cell_id] += 1
            require(wrapper.get("eligible_request_inclusion_probability")
                    == EXPECTED_INCLUSION_PROBABILITY
                    and wrapper.get("eligible_request_inclusion_probability_exact")
                    == EXPECTED_INCLUSION_PROBABILITY_EXACT,
                    "eligible request inclusion probability changed")
        if wrapper.get("selected"):
            by_model_selected[model] += 1
            by_cell_selected[cell_id] += 1
    require(dict(by_model_eligible) == EXPECTED_ELIGIBLE,
            "per-model eligible request counts changed")
    require(dict(by_model_selected) == EXPECTED_SELECTED,
            "per-model selected request counts changed")
    for episode in episodes:
        cell_id = episode["cell_id"]
        require(episode.get("request_count") in {15, 57}
                and episode.get("request_count") == by_cell_wrappers[cell_id]
                and episode.get("eligible_count") == 14
                and episode.get("eligible_count") == by_cell_eligible[cell_id]
                and episode.get("selected_count") == 4
                and episode.get("selected_count") == by_cell_selected[cell_id]
                and episode.get("eligible_request_inclusion_probability")
                == EXPECTED_INCLUSION_PROBABILITY
                and episode.get("eligible_request_inclusion_probability_exact")
                == EXPECTED_INCLUSION_PROBABILITY_EXACT,
                "per-episode deterministic selection contract changed")
    try:
        reproduced = annotation.select_requests(
            inventory,
            seed=annotation.REQUEST_SAMPLING_SEED,
            cap_per_episode=annotation.REQUEST_SAMPLE_CAP,
            inventory_sha256=sha256_bytes(pretty_bytes(inventory)),
        )
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"request selection deterministic reproduction failed: {error}"
        ) from error
    require(selection == reproduced,
            "request selection differs from the exact frozen deterministic draw")


def _array_from_npz_node(
    descriptor_value: Mapping[str, Any],
    node: Mapping[str, Any],
    *,
    base: Path,
    raw_root: Path,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(node.get("__type__") == "ndarray" and isinstance(node.get("key"), str),
            f"{label} is not a recorder ndarray")
    _, archive_path = _verify_descriptor(
        descriptor_value.get("artifact"), base=base, raw_root=raw_root,
        label=f"{label} NPZ",
    )
    member = f"{node['key']}.npy"
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            require(member in archive.namelist(), f"{label} NPZ member is missing")
            require(archive.getinfo(member).compress_type == zipfile.ZIP_STORED,
                    f"{label} NPZ member is compressed")
            with archive.open(member, "r") as handle:
                array = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise AnnotationMediaBridgeError(f"{label} cannot be read") from error
    require(list(array.shape) == node.get("shape") and array.dtype.str == node.get("dtype"),
            f"{label} array metadata changed")
    return array, {
        "container": descriptor(archive_path),
        "member": member,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "value_sha256": sha256_bytes(
            compiler.canonical_bytes(
                {"kind": "numpy", "dtype": array.dtype.str, "shape": list(array.shape)},
                ensure_ascii=True,
            ) + array.tobytes(order="C")
        ),
    }


def _array_from_npy_descriptor(
    value: Mapping[str, Any], *, base: Path, raw_root: Path, label: str
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = {
        "path": value.get("path"),
        "sha256": value.get("file_sha256", value.get("sha256")),
        "bytes": value.get("bytes"),
    }
    _, path = _verify_descriptor(raw, base=base, raw_root=raw_root, label=label)
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise AnnotationMediaBridgeError(f"{label} is not a safe NumPy array") from error
    require(list(array.shape) == value.get("shape"), f"{label} shape changed")
    expected_dtype = value.get("dtype")
    require(expected_dtype in {str(array.dtype), array.dtype.name, array.dtype.str},
            f"{label} dtype changed")
    data_sha = value.get("data_sha256")
    require(isinstance(data_sha, str) and data_sha == sha256_bytes(array.tobytes(order="C")),
            f"{label} data hash changed")
    return array, {
        "container": descriptor(path),
        "member": None,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "value_sha256": data_sha,
    }


def _rgb_u8(array: np.ndarray, label: str) -> np.ndarray:
    require(isinstance(array, np.ndarray) and array.dtype == np.uint8
            and array.ndim == 3 and array.shape[2] in (3, 4),
            f"{label} must be uint8 HWC RGB/RGBA")
    result = np.ascontiguousarray(array[:, :, :3])
    require(result.flags.c_contiguous, f"{label} is not C-contiguous")
    return result


def _apply_original_transform(
    image: np.ndarray, crop: Mapping[str, Any], model: str,
    *, replay_client: _ExactRuntimeReplayClient,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    image = _rgb_u8(image, f"{model} original camera")
    try:
        camera_replay.validate_camera_crop_contract(crop, expected_model=model)
        current, replay_evidence = replay_client.replay(image)
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"{model} exact original-camera replay failed: {error}"
        ) from error
    current = _rgb_u8(np.asarray(current), f"{model} replayed original camera")
    require(list(current.shape) == crop["original_camera_replay"]["output_shape"],
            f"{model} original-camera replay output shape changed")
    return (
        current,
        [dict(step) for step in crop["original_camera_replay"]["transform_chain"]],
        replay_evidence,
    )


def _apply_generated_crop(
    frames: np.ndarray, crop: Mapping[str, Any], model: str
) -> np.ndarray:
    try:
        camera_replay.validate_camera_crop_contract(crop, expected_model=model)
        output = camera_replay.extract_generated_crop(frames, crop)
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"{model} exact generated-camera crop failed: {error}"
        ) from error
    output = np.ascontiguousarray(np.asarray(output))
    require(output.dtype == np.uint8
            and list(output.shape) == crop["generated_decoded_crop"]["output_shape"],
            f"{model} generated crop output shape/dtype changed")
    return output


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF)
    )


def png_bytes(image: np.ndarray) -> bytes:
    image = _rgb_u8(image, "PNG image")
    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + image[row].tobytes(order="C") for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _mapping_item(value: Any, key: str, label: str) -> Any:
    try:
        return compiler._mapping_item(value, key, label)
    except BaseException as error:
        raise AnnotationMediaBridgeError(str(error)) from error


def _observation_from_journal(
    *,
    event_payload: Mapping[str, Any],
    expected_control_step: int,
    camera_id: str,
    completion_base: Path,
    raw_root: Path,
    compact: Mapping[str, Any] | None,
    label: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    require(event_payload.get("observation_id") == f"obs_{expected_control_step:06d}"
            and event_payload.get("control_step") == expected_control_step,
            f"{label} observation schedule changed")
    try:
        payload = compiler._verify_payload_descriptor(
            event_payload.get("artifact"), base=completion_base, raw_root=raw_root,
            label=label, required_role="observation",
        )
    except BaseException as error:
        raise AnnotationMediaBridgeError(f"{label} payload failed authentication: {error}") from error
    clock = event_payload.get("clock")
    require(isinstance(clock, Mapping), f"{label} native clock is missing")
    cameras = clock.get("cameras")
    camera = cameras.get(camera_id) if isinstance(cameras, Mapping) else None
    require(isinstance(camera, Mapping), f"{label} original camera clock is missing")
    images = _mapping_item(payload["structure"], "image_obs", label)
    node = _mapping_item(images, camera_id, label)
    array, parent = _array_from_npz_node(
        payload, node, base=completion_base, raw_root=raw_root, label=label
    )
    normalized_frame_id = (
        f"{camera_id}:native-int:{camera['frame_id']}"
        if type(camera.get("frame_id")) is int
        else f"{camera_id}:native-str:{camera.get('frame_id')}"
    )
    metadata = {
        "observation_id": event_payload["observation_id"],
        "control_step": expected_control_step,
        "physics_step": clock.get("physics_step"),
        "physics_time_s": float(clock.get("physics_time_s")),
        "camera_frame_native_id": camera.get("frame_id"),
        "camera_frame_id": normalized_frame_id,
        "camera_capture_time_ns": camera.get("capture_time_ns"),
        "camera_timestamp_source": camera.get("timestamp_source"),
        "payload_sha256": payload["payload_sha256"],
        "payload_artifact": parent["container"],
    }
    if compact is not None:
        require(dict(compact) == metadata, f"{label} differs from compiler provenance")
    parent.update({
        "payload_sha256": payload["payload_sha256"],
        "observation_id": event_payload["observation_id"],
        "control_step": expected_control_step,
        "camera_frame_id": normalized_frame_id,
    })
    return array, parent, metadata


def _generated_future(
    *,
    model: str,
    source: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    request_path: Path,
    completion_base: Path,
    raw_root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    request = load_json(request_path, f"{model} official request receipt")
    if model == "N3":
        try:
            compiler._verify_n3_payload_reference(
                request.get("official_returned_response"), base=request_path.parent,
                raw_root=raw_root, label="N3 official returned response",
                required_role="official_returned_response",
            )
        except BaseException as error:
            raise AnnotationMediaBridgeError(
                f"N3 official generated future failed authentication: {error}"
            ) from error
        official_ref = request["official_returned_response"]
        manifest_candidate = Path(official_ref["manifest_path"])
        if not manifest_candidate.is_absolute():
            manifest_candidate = request_path.parent / manifest_candidate
        manifest_path = _under(manifest_candidate, raw_root, "N3 official response manifest")
        manifest = load_json(manifest_path, "N3 official response manifest")
        official_node = _mapping_item(manifest.get("structure"), "video", "N3 official response")
        raw_path = manifest_path.parent / str(official_node.get("artifact", {}).get("path"))
        array_descriptor = official_node.get("artifact")
        _, array_path = _verify_descriptor(
            array_descriptor, base=manifest_path.parent, raw_root=raw_root,
            label="N3 official generated video",
        )
        try:
            official_array = np.load(array_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise AnnotationMediaBridgeError("N3 official generated video is unreadable") from error
        require(list(official_array.shape) == official_node.get("shape")
                and official_array.dtype.str == official_node.get("dtype"),
                "N3 official generated video metadata changed")
        raw_response = _mapping_item(response_payload["structure"], "raw_response", "N3 response")
        recorder_node = _mapping_item(raw_response, "video", "N3 response")
        recorder_array, recorder_parent = _array_from_npz_node(
            response_payload, recorder_node, base=completion_base,
            raw_root=raw_root, label="N3 recorder generated video",
        )
        require(np.array_equal(recorder_array, official_array),
                "N3 recorder generated future differs from official response")
        return recorder_array, {
            "kind": "official_and_recorder_generated_future",
            "official_manifest": descriptor(manifest_path),
            "official_array": descriptor(array_path),
            "recorder_array": recorder_parent,
            "frame_axis": 0,
        }

    decoded = request.get("offline_decode", {}).get("decoded_rgb")
    require(isinstance(decoded, Mapping), "D1 official decoded RGB is missing")
    try:
        compiler._verify_d1_array_artifact(
            decoded, base=request_path.parent, raw_root=raw_root,
            label="D1 official decoded RGB", tensor=False,
        )
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"D1 official generated future failed authentication: {error}"
        ) from error
    array, parent = _array_from_npy_descriptor(
        decoded, base=request_path.parent, raw_root=raw_root,
        label="D1 official decoded RGB",
    )
    raw_response = _mapping_item(response_payload["structure"], "raw_response", "D1 response")
    try:
        raw_future = compiler._thaw_scalar(
            _mapping_item(raw_response, "future_evidence", "D1 response"),
            "D1 response future evidence",
        )
    except BaseException as error:
        raise AnnotationMediaBridgeError(f"D1 recorder future descriptor is invalid: {error}") from error
    observed = raw_future.get("decoded", {}).get("rgb") if isinstance(raw_future, Mapping) else None
    require(isinstance(observed, Mapping), "D1 recorder decoded RGB descriptor is missing")
    observed_descriptor = {
        "path": observed.get("path"),
        "sha256": observed.get("sha256", observed.get("file_sha256")),
        "bytes": observed.get("bytes"),
    }
    official_descriptor = {
        "path": decoded.get("path"),
        "sha256": decoded.get("file_sha256"),
        "bytes": decoded.get("bytes"),
    }
    _same_descriptor(observed_descriptor, official_descriptor,
                     "D1 recorder/official decoded RGB")
    return array, {
        "kind": "official_and_recorder_generated_future",
        "official_array": parent["container"],
        "recorder_descriptor": observed_descriptor,
        "frame_axis": 0,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _relative_descriptor(path: Path, root: Path) -> dict[str, Any]:
    return descriptor(path, display_path=str(path.relative_to(root)))


def _safe_component(value: str, label: str) -> str:
    require(SAFE_ID_RE.fullmatch(value) is not None, f"unsafe {label}: {value}")
    return value


def _render_selected_with_clients(
    *,
    temp_root: Path,
    raw_root: Path,
    selection: Mapping[str, Any],
    rich: Mapping[tuple[str, str, int], Mapping[str, Any]],
    crop_by_model: Mapping[str, Mapping[str, Any]],
    replay_clients: Mapping[str, _ExactRuntimeReplayClient],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [
        wrapper["source"] for wrapper in selection["requests"] if wrapper["selected"]
    ]
    image_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    journal_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for request in selected:
        model = request["model_id"]
        cell_id = request["cell_id"]
        request_index = request["request_index"]
        key = (model, cell_id, request_index)
        detail = rich.get(key)
        require(isinstance(detail, Mapping), f"selected request lacks rich provenance: {key}")
        source = detail["source"]
        binding = detail["binding"]
        provenance_base = Path(detail["provenance_base"])
        crop = crop_by_model[model]
        cell_key = (model, cell_id)
        if cell_key not in journal_cache:
            _, completion_path = _verify_descriptor(
                source["adapter_completion"], base=provenance_base, raw_root=raw_root,
                label=f"{model} {cell_id} completion",
            )
            completion = load_json(completion_path, f"{model} {cell_id} completion")
            _, journal_path = _verify_descriptor(
                source["adapter_journal"], base=provenance_base, raw_root=raw_root,
                label=f"{model} {cell_id} journal",
            )
            try:
                journal_rows, tail = compiler._verify_journal(journal_path)
            except BaseException as error:
                raise AnnotationMediaBridgeError(
                    f"{model} {cell_id} journal failed authentication: {error}"
                ) from error
            require(completion.get("event_count") == len(journal_rows)
                    and completion.get("journal_tail_sha256") == tail,
                    f"{model} {cell_id} completion does not bind journal")
            observations = [
                compiler._event_payload(row, "observation")
                for row in journal_rows if row.get("kind") == "observation_captured"
            ]
            responses = [
                compiler._event_payload(row, "response")
                for row in journal_rows if row.get("kind") == "model_response_received"
            ]
            require(len(observations) == 451
                    and len(responses) == EXPECTED_REQUESTS[model] // 16,
                    f"{model} {cell_id} journal coverage changed")
            action_descriptor = source["action_manifest"]
            _, action_path = _verify_descriptor(
                action_descriptor, base=provenance_base, raw_root=raw_root,
                label=f"{model} {cell_id} action manifest",
            )
            action_manifest = load_json(action_path, f"{model} {cell_id} action manifest")
            verify_signed(action_manifest, f"{model} {cell_id} action manifest")
            require(action_manifest.get("schema_version") == annotation.ACTION_MANIFEST_SCHEMA
                    and action_manifest.get("cell_id") == cell_id
                    and action_manifest.get("model_id") == model
                    and action_manifest.get("executed_action_count") == 450
                    and len(action_manifest.get("actions", [])) == 450,
                    f"{model} {cell_id} action manifest changed")
            _, recording_path = _verify_descriptor(
                source["recording_receipt"], base=provenance_base, raw_root=raw_root,
                label=f"{model} {cell_id} recording receipt",
            )
            journal_cache[cell_key] = {
                "completion_base": completion_path.parent,
                "adapter_completion": descriptor(completion_path),
                "adapter_journal": {
                    **descriptor(journal_path),
                    "event_count": len(journal_rows),
                    "tail_sha256": tail,
                },
                "action_manifest": descriptor(action_path),
                "recording_receipt": descriptor(recording_path),
                "observations": observations,
                "responses": responses,
                "actions": action_manifest["actions"],
            }
        cached = journal_cache[cell_key]
        response_event = cached["responses"][request_index]
        try:
            response_payload = compiler._verify_payload_descriptor(
                response_event.get("response_artifact"),
                base=cached["completion_base"], raw_root=raw_root,
                label=f"{model} {cell_id} response {request_index}",
                required_role="model_response",
            )
        except BaseException as error:
            raise AnnotationMediaBridgeError(
                f"{model} {cell_id} response failed authentication: {error}"
            ) from error
        require(response_payload["payload_sha256"]
                == source["recorder_response_payload_sha256"]
                == binding["recorder_response_request_binding"]["payload_sha256"],
                f"{model} {cell_id} response payload binding changed")
        _, request_path = _verify_descriptor(
            source["official_request_receipt"], base=provenance_base, raw_root=raw_root,
            label=f"{model} {cell_id} official request {request_index}",
        )
        generated_array, generated_parent = _generated_future(
            model=model, source=source, response_payload=response_payload,
            request_path=request_path, completion_base=cached["completion_base"],
            raw_root=raw_root,
        )
        cropped_generated = _apply_generated_crop(generated_array, crop, model)

        primary = detail["primary_target_binding"]
        early_contract = next(
            contract["early_horizon"] for contract in selection["alignment_contracts"]
            if contract["model_id"] == model
        )
        early = _target_binding(binding, early_contract)
        require(isinstance(primary, Mapping)
                and primary.get("status") == "matched_native_request_clocks"
                and isinstance(early, Mapping)
                and early.get("status") == "matched_native_request_clocks",
                f"selected {model} request lacks matched primary/early targets")
        observation_roles = {
            "current": (source["current_observation_id"], source["current_observation"]),
            "executed": (primary["target_observation_id"], None),
            "early_executed": (early["target_observation_id"], None),
        }
        if source["preceding_observation_id"] is not None:
            observation_roles["preceding"] = (
                source["preceding_observation_id"], source["preceding_observation"]
            )
        role_pixels: dict[str, tuple[np.ndarray, dict[str, Any], Any]] = {}
        for role, (observation_id, compact) in observation_roles.items():
            require(isinstance(observation_id, str)
                    and re.fullmatch(r"obs_[0-9]{6}", observation_id) is not None,
                    f"{model} {role} observation identity is invalid")
            control_step = int(observation_id.split("_")[1])
            pixels, parent, metadata = _observation_from_journal(
                event_payload=cached["observations"][control_step],
                expected_control_step=control_step,
                camera_id=request["camera_id"],
                completion_base=cached["completion_base"], raw_root=raw_root,
                compact=compact, label=f"{model} {cell_id} {role} observation",
            )
            transformed, chain, exact_runtime_replay = _apply_original_transform(
                pixels, crop, model, replay_client=replay_clients[model]
            )
            if role in {"executed", "early_executed"}:
                target = primary if role == "executed" else early
                require(metadata["camera_frame_native_id"] == target.get("target_camera_frame_id")
                        and metadata["camera_capture_time_ns"]
                        == target.get("target_camera_capture_time_ns")
                        and metadata["physics_step"] == target.get("target_physics_step")
                        and metadata["physics_time_s"] == target.get("target_physics_time_s"),
                        f"{model} {role} native target observation changed")
                action_index = control_step - 1
                action_row = cached["actions"][action_index]
                require(action_row.get("action_index") == action_index
                        and action_row.get("camera_frame_id") == metadata["camera_frame_id"],
                        f"{model} {role} action/resulting-camera binding changed")
            role_pixels[role] = (
                transformed,
                {
                    "kind": "recorder_original_camera_observation",
                    **parent,
                    "transform_chain": chain,
                    "exact_runtime_replay": exact_runtime_replay,
                },
                None,
            )
        role_pixels["predicted"] = (
            cropped_generated[request["generated_frame_index"]],
            generated_parent,
            request["generated_frame_index"],
        )
        role_pixels["early_predicted"] = (
            cropped_generated[early_contract["generated_frame_index"]],
            generated_parent,
            early_contract["generated_frame_index"],
        )
        expected_roles = {"current", "predicted", "executed", "early_predicted", "early_executed"}
        if request_index > 0:
            expected_roles.add("preceding")
        require(set(role_pixels) == expected_roles,
                f"{model} {cell_id} rendered role set changed")

        for role in sorted(role_pixels):
            pixels, parent, frame_index = role_pixels[role]
            width, height = EXPECTED_IMAGE_DIMENSIONS[model]
            require(list(pixels.shape) == [height, width, 3] and pixels.dtype == np.uint8,
                    f"{model} {role} rendered pixels changed shape/dtype")
            token = sha256_bytes(
                f"{source['source_request_id']}\0{role}".encode("utf-8")
            )[:32]
            image_id = "source_" + token
            source_rel = Path("source_images") / f"{image_id}.png"
            media_rel = Path("annotation_media") / f"{image_id}.png"
            receipt_rel = Path("render_receipts") / f"{image_id}.json"
            payload = png_bytes(pixels)
            _write_exclusive(temp_root / source_rel, payload)
            _write_exclusive(temp_root / media_rel, payload)
            image_sha = sha256_bytes(payload)
            render_receipt = sign_document({
                "schema_version": RENDER_RECEIPT_SCHEMA,
                "study_id": STUDY_ID,
                "receipt_id": "render_" + token,
                "source_image_id": image_id,
                "source_request_id": source["source_request_id"],
                "source_video_id": source["source_video_id"],
                "source_video_sha256": source["source_video"]["sha256"],
                "image_role": role,
                "source_image_path": "../" + str(source_rel),
                "source_image_sha256": image_sha,
                "annotation_media_path": "../" + str(media_rel),
                "annotation_media_sha256": image_sha,
                "width_px": width,
                "height_px": height,
                "camera_id": request["camera_id"],
                "camera_crop_id": request["camera_crop_id"],
                "camera_crop_sha256": request["camera_crop_sha256"],
                "alignment_receipt_id": request["alignment_receipt_id"],
                "alignment_receipt_sha256": request["alignment_receipt_sha256"],
                "presentation_sanitized": True,
                "contains_overlay": False,
            })
            _write_exclusive(temp_root / receipt_rel, pretty_bytes(render_receipt))
            receipt_sha = sha256_file(temp_root / receipt_rel)
            image_rows.append({
                "source_image_id": image_id,
                "source_request_id": source["source_request_id"],
                "source_video_id": source["source_video_id"],
                "source_video_sha256": source["source_video"]["sha256"],
                "image_role": role,
                "source_image_path": str(source_rel),
                "source_image_sha256": image_sha,
                "annotation_media_path": str(media_rel),
                "annotation_media_sha256": image_sha,
                "width_px": width,
                "height_px": height,
                "camera_id": request["camera_id"],
                "camera_crop_id": request["camera_crop_id"],
                "camera_crop_sha256": request["camera_crop_sha256"],
                "alignment_receipt_id": request["alignment_receipt_id"],
                "alignment_receipt_sha256": request["alignment_receipt_sha256"],
                "render_receipt_id": render_receipt["receipt_id"],
                "render_receipt_path": str(receipt_rel),
                "render_receipt_sha256": receipt_sha,
                "presentation_sanitized": True,
                "contains_overlay": False,
            })
            lineage_rows.append({
                "source_image_id": image_id,
                "source_request_id": source["source_request_id"],
                "join_key": {
                    "model_id": model,
                    "cell_id": cell_id,
                    "request_index": request_index,
                },
                "image_role": role,
                "parent": parent,
                "parent_frame_index": frame_index,
                "camera_crop_contract_id": crop["contract_id"],
                "camera_crop_contract_payload_sha256": crop["payload_sha256"],
                "camera_transform": (
                    crop["generated_decoded_crop"]
                    if role in {"predicted", "early_predicted"}
                    else crop["original_camera_replay"]["transform_chain"]
                ),
                "authenticated_request_evidence": {
                    "adapter_completion": cached["adapter_completion"],
                    "adapter_journal": cached["adapter_journal"],
                    "action_manifest": cached["action_manifest"],
                    "recording_receipt": cached["recording_receipt"],
                    "official_request_receipt": descriptor(request_path),
                    "recorder_response_payload_sha256": response_payload[
                        "payload_sha256"
                    ],
                },
                "source_image": _relative_descriptor(temp_root / source_rel, temp_root),
                "rendered_png": _relative_descriptor(temp_root / media_rel, temp_root),
                "render_receipt": _relative_descriptor(temp_root / receipt_rel, temp_root),
                "simulator_state_render_used": False,
                "presentation_video_fps_used": False,
                "generated_frame_index_interpreted_as_action_index": False,
            })
    return image_rows, lineage_rows


def _render_selected(
    *,
    temp_root: Path,
    raw_root: Path,
    selection: Mapping[str, Any],
    rich: Mapping[tuple[str, str, int], Mapping[str, Any]],
    crop_by_model: Mapping[str, Mapping[str, Any]],
    crop_path_by_model: Mapping[str, Path],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    """Render through one exact signed-runtime helper per model."""

    require(set(crop_path_by_model) == set(MODELS),
            "camera replay crop-path model coverage changed")
    evidence_root = temp_root / "camera_replay_runtime"
    clients: dict[str, _ExactRuntimeReplayClient] = {}
    try:
        for model in MODELS:
            clients[model] = _ExactRuntimeReplayClient(
                model=model,
                crop=crop_by_model[model],
                contract_path=crop_path_by_model[model],
                evidence_root=evidence_root,
            )
        image_rows, lineage_rows = _render_selected_with_clients(
            temp_root=temp_root,
            raw_root=raw_root,
            selection=selection,
            rich=rich,
            crop_by_model=crop_by_model,
            replay_clients=clients,
        )
        receipts: dict[str, dict[str, Any]] = {}
        receipt_descriptors: dict[str, dict[str, Any]] = {}
        for model in MODELS:
            receipts[model] = clients[model].close()
            receipt_descriptors[model] = _relative_descriptor(
                clients[model].receipt_path, temp_root
            )
        for row in lineage_rows:
            model = row["join_key"]["model_id"]
            if row["image_role"] in {"predicted", "early_predicted"}:
                require("exact_runtime_replay" not in row["parent"],
                        "generated image acquired an original-camera replay receipt")
                row["camera_replay_runtime_receipt"] = None
            else:
                evidence = row["parent"].get("exact_runtime_replay")
                require(isinstance(evidence, Mapping)
                        and evidence.get("session_id") == receipts[model]["session_id"],
                        "original-camera lineage lacks exact-runtime replay binding")
                row["camera_replay_runtime_receipt"] = receipt_descriptors[model]
        return image_rows, lineage_rows, receipts
    except BaseException:
        for client in clients.values():
            if not client._closed:
                client.abort()
        raise


def _build_publish_tranche_index(
    *,
    temp_root: Path,
    image_rows: Sequence[Mapping[str, Any]],
    lineage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Partition unique sanitized PNG bytes for bounded result-return jobs.

    Byte-identical render aliases share one published asset.  Tranches are a
    deterministic greedy partition over sorted SHA-256 values and never exceed
    48 MiB of PNG data, leaving at least 16 MiB below the publisher's job cap
    for the signed alias manifest and job receipt.
    """

    lineage_by_image = {
        row["source_image_id"]: row for row in lineage_rows
        if isinstance(row, Mapping)
    }
    require(len(lineage_by_image) == len(lineage_rows),
            "source-extraction lineage image identities are duplicated")
    by_sha: dict[str, dict[str, Any]] = {}
    for row in image_rows:
        digest = row["annotation_media_sha256"]
        path = temp_root / row["annotation_media_path"]
        identity = _relative_descriptor(path, temp_root)
        require(identity["sha256"] == digest
                and identity["bytes"] < PUBLISH_FILE_LIMIT_BYTES,
                "sanitized PNG violates publisher file/hash boundary")
        lineage = lineage_by_image.get(row["source_image_id"])
        require(isinstance(lineage, Mapping), "sanitized PNG lacks source lineage")
        alias = {
            "source_image_id": row["source_image_id"],
            "source_request_id": row["source_request_id"],
            "image_role": row["image_role"],
            "join_key": lineage["join_key"],
            "camera_id": row["camera_id"],
            "camera_crop_id": row["camera_crop_id"],
            "camera_crop_sha256": row["camera_crop_sha256"],
            "alignment_receipt_id": row["alignment_receipt_id"],
            "alignment_receipt_sha256": row["alignment_receipt_sha256"],
            "render_receipt_id": row["render_receipt_id"],
            "render_receipt_sha256": row["render_receipt_sha256"],
            "source_lineage_record_sha256": sha256_bytes(canonical_bytes(lineage)),
        }
        existing = by_sha.setdefault(digest, {
            "asset_sha256": digest,
            "source_path": identity["path"],
            "bytes": identity["bytes"],
            "width_px": row["width_px"],
            "height_px": row["height_px"],
            "aliases": [],
        })
        require(existing["bytes"] == identity["bytes"]
                and existing["width_px"] == row["width_px"]
                and existing["height_px"] == row["height_px"],
                "byte-identical sanitized asset metadata changed")
        existing["aliases"].append(alias)
    assets = []
    for digest in sorted(by_sha):
        asset = by_sha[digest]
        asset["aliases"] = sorted(
            asset["aliases"],
            key=lambda row: (
                row["source_request_id"], row["image_role"], row["source_image_id"]
            ),
        )
        assets.append(asset)
    require(sum(len(row["aliases"]) for row in assets) == len(image_rows),
            "publish asset aliases do not cover every rendered image")
    tranches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for asset in assets:
        if current and current_bytes + asset["bytes"] > PUBLISH_TRANCHE_ASSET_BUDGET_BYTES:
            tranches.append(current)
            current = []
            current_bytes = 0
        require(asset["bytes"] <= PUBLISH_TRANCHE_ASSET_BUDGET_BYTES,
                "one sanitized PNG exceeds the tranche budget")
        current.append(asset)
        current_bytes += asset["bytes"]
    if current:
        tranches.append(current)
    rows = []
    seen: set[str] = set()
    for ordinal, tranche in enumerate(tranches, 1):
        tranche_id = f"development-annotation-media-tranche-{ordinal:03d}"
        digests = [row["asset_sha256"] for row in tranche]
        require(not seen.intersection(digests), "publish tranches overlap")
        seen.update(digests)
        asset_bytes = sum(row["bytes"] for row in tranche)
        aliases = sum(len(row["aliases"]) for row in tranche)
        require(asset_bytes <= PUBLISH_TRANCHE_ASSET_BUDGET_BYTES,
                "publish tranche exceeds bounded asset budget")
        rows.append({
            "tranche_id": tranche_id,
            "ordinal": ordinal,
            "asset_count": len(tranche),
            "alias_count": aliases,
            "asset_bytes": asset_bytes,
            "asset_budget_bytes": PUBLISH_TRANCHE_ASSET_BUDGET_BYTES,
            "publisher_file_limit_bytes": PUBLISH_FILE_LIMIT_BYTES,
            "publisher_job_limit_bytes": PUBLISH_JOB_LIMIT_BYTES,
            "assets": tranche,
        })
    require(seen == set(by_sha), "publish tranche index omitted a sanitized asset")
    source_lineage = _relative_descriptor(
        temp_root / "source_extraction_lineage.json", temp_root
    )
    rendered_manifest = _relative_descriptor(
        temp_root / "rendered_png_manifest.json", temp_root
    )
    return sign_document({
        "schema_version": PUBLISH_TRANCHE_INDEX_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "development",
        "status": "prepared_for_bounded_result_return_pending_human_review",
        "visibility": "RESTRICTED ANALYST MANIFEST; NEVER DISTRIBUTE TO RATERS",
        "unique_asset_count": len(assets),
        "render_alias_count": len(image_rows),
        "tranche_count": len(rows),
        "total_unique_asset_bytes": sum(row["bytes"] for row in assets),
        "source_extraction_lineage": source_lineage,
        "rendered_png_manifest": rendered_manifest,
        "deduplication_key": "annotation_media_sha256",
        "tranches_non_overlapping": True,
        "each_asset_published_once": True,
        "human_pixel_blindness_review_complete": False,
        "safe_for_rater_distribution": False,
        "safe_to_release_confirmation": False,
        "tranches": rows,
    })


def _build_pixel_review_checklist(
    *, temp_root: Path, image_rows: Sequence[Mapping[str, Any]],
    rendered_png_manifest_sha256: str,
) -> dict[str, Any]:
    """Emit an identity-blind checklist, never a completed human receipt."""

    by_asset: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in image_rows:
        key = (
            row["annotation_media_sha256"], row["width_px"], row["height_px"]
        )
        path = temp_root / row["annotation_media_path"]
        identity = _relative_descriptor(path, temp_root)
        require(identity["sha256"] == key[0],
                "pixel-review candidate asset changed")
        by_asset[key].add(identity["path"])
    assets = [
        {
            "media_sha256": digest,
            "width_px": width,
            "height_px": height,
            "candidate_paths": sorted(paths),
        }
        for (digest, width, height), paths in sorted(by_asset.items())
    ]
    receipt_assets = [
        {
            "media_sha256": row["media_sha256"],
            "width_px": row["width_px"],
            "height_px": row["height_px"],
        }
        for row in assets
    ]
    exclusions = sorted(annotation.EXAMPLE_EXCLUSION_KEYS)
    require(assets and len(assets) == len(receipt_assets),
            "pixel-review checklist has no unique assets")
    return sign_document({
        "schema_version": PIXEL_REVIEW_CHECKLIST_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "development",
        "artifact_scope": "annotation_media",
        "status": "awaiting_named_human_visual_inspection",
        "rendered_png_manifest_sha256": rendered_png_manifest_sha256,
        "unique_asset_count": len(assets),
        "assets": assets,
        "required_review_method": (
            "human_visual_inspection_of_rendered_pixels_and_presentation"
        ),
        "required_content_exclusions": exclusions,
        "review_receipt_template_unsigned": {
            "schema_version": annotation.PIXEL_BLINDNESS_RECEIPT_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "artifact_scope": "annotation_media",
            "status": "human_reviewed_source_blind",
            "reviewer_code": None,
            "reviewed_at": None,
            "review_method": (
                "human_visual_inspection_of_rendered_pixels_and_presentation"
            ),
            "content_exclusions": {key: None for key in exclusions},
            "assets": receipt_assets,
            "payload_sha256": None,
        },
        "completion_instructions": (
            "A named human must inspect every unique asset and its intended isolated "
            "presentation, fill reviewer_code/reviewed_at, attest every exclusion true, "
            "then recompute payload_sha256. Unset/null fields are not a review receipt."
        ),
        "human_pixel_blindness_review_complete": False,
        "safe_for_rater_distribution": False,
        "safe_to_release_confirmation": False,
    })


def _write_atomic_directory(target: Path, files: Mapping[str, bytes], staged: Path) -> None:
    target = Path(target)
    require(not target.exists() and not target.is_symlink(),
            f"refusing to overwrite annotation media output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for relative, payload in files.items():
        _write_exclusive(staged / relative, payload)
    os.replace(staged, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def prepare(manifest_path: Path, manifest_sha256: str, output_dir: Path) -> dict[str, Any]:
    supplied_manifest = Path(manifest_path)
    _reject_symlinks(supplied_manifest, "annotation media input manifest")
    manifest_path = supplied_manifest.resolve(strict=True)
    require(SHA_RE.fullmatch(manifest_sha256) is not None
            and sha256_file(manifest_path) == manifest_sha256,
            "annotation media input manifest hash changed")
    manifest = load_json(manifest_path, "annotation media input manifest")
    _exact_keys(
        manifest,
        {"schema_version", "study_id", "mode", "raw_root", "compiler", "models"},
        "annotation media input manifest",
    )
    require(manifest.get("schema_version") == INPUT_SCHEMA
            and manifest.get("study_id") == STUDY_ID
            and manifest.get("mode") == "formal_full_development",
            "annotation media input manifest identity changed")
    raw_root_text = manifest.get("raw_root")
    require(isinstance(raw_root_text, str) and Path(raw_root_text).is_absolute(),
            "annotation media raw root must be absolute")
    raw_root = Path(raw_root_text)
    _reject_symlinks(raw_root, "annotation media raw root")
    raw_root = raw_root.resolve(strict=True)
    require(raw_root.is_dir() and raw_root == CANONICAL_RAW_ROOT,
            "annotation media raw root is not the canonical task PVC root")
    require(manifest_path.is_relative_to(raw_root),
            "annotation media input manifest escapes raw root")
    output_dir = Path(output_dir)
    require(output_dir.parent.resolve(strict=True).is_relative_to(raw_root),
            "annotation media output parent escapes raw root")
    require(not output_dir.exists() and not output_dir.is_symlink(),
            "annotation media output already exists")
    compiler_receipt, compiler_receipt_path, compiler_job = _validate_compiler(
        manifest, raw_root
    )
    raw_models = manifest.get("models")
    require(isinstance(raw_models, list) and len(raw_models) == 2,
            "annotation media inputs must cover exactly two models")
    model_inputs = {
        item.get("model_id"): item for item in raw_models if isinstance(item, Mapping)
    }
    require(set(model_inputs) == set(MODELS), "annotation media model inputs changed")
    roster_by_model: dict[str, list[dict[str, Any]]] = {}
    provenance_by_model: dict[str, list[dict[str, Any]]] = {}
    provenance_base_by_model: dict[str, Path] = {}
    timing_by_model: dict[str, dict[str, Any]] = {}
    crop_by_model: dict[str, dict[str, Any]] = {}
    crop_path_by_model: dict[str, Path] = {}
    alignment_by_model: dict[str, dict[str, Any]] = {}
    mapping_sha_by_model: dict[str, str] = {}
    input_identities: dict[str, Any] = {
        "compiler_job_receipt": descriptor(
            _verify_descriptor(manifest["compiler"]["job_receipt"], base=raw_root,
                               raw_root=raw_root, label="compiler job")[1]
        ),
        "compiler_receipt": descriptor(compiler_receipt_path),
        "models": {},
    }
    for model in MODELS:
        inputs = model_inputs[model]
        _exact_keys(
            inputs,
            {"model_id", "request_provenance", "timing_sidecar", "timing_job_receipt",
             "camera_crop_contract", "alignment_contract", "physical_alignment_receipt"},
            f"{model} annotation media inputs",
        )
        _, provenance_path = _verify_descriptor(
            inputs["request_provenance"], base=raw_root, raw_root=raw_root,
            label=f"{model} request provenance",
        )
        provenance = load_json(provenance_path, f"{model} request provenance")
        roster, provenance_rows = _validate_provenance(
            model=model, value=provenance, path=provenance_path,
            compiler_receipt=compiler_receipt,
            compiler_receipt_path=compiler_receipt_path, raw_root=raw_root,
        )
        freeze_fragment = _load_compiler_freeze_fragment(
            model=model,
            compiler_receipt=compiler_receipt,
            compiler_receipt_path=compiler_receipt_path,
            raw_root=raw_root,
        )
        _, timing_path = _verify_descriptor(
            inputs["timing_sidecar"], base=raw_root, raw_root=raw_root,
            label=f"{model} timing sidecar",
        )
        _, timing_job_path = _verify_descriptor(
            inputs["timing_job_receipt"], base=raw_root, raw_root=raw_root,
            label=f"{model} timing job receipt",
        )
        timing_job = load_json(timing_job_path, f"{model} timing job receipt")
        _validate_timing_job(
            model=model, value=timing_job,
            sidecar_descriptor=inputs["timing_sidecar"],
        )
        request_hashes = [row["official_request_receipt"]["sha256"] for row in provenance_rows]
        try:
            sidecar = timing.validate_development_timing(
                timing_path, inputs["timing_sidecar"]["sha256"],
                expected_model=model, expected_request_hashes=request_hashes,
            )
        except BaseException as error:
            raise AnnotationMediaBridgeError(
                f"{model} timing sidecar failed deep validation: {error}"
            ) from error
        if model == "D1":
            require(sidecar.get("request_timing_coverage") == EXPECTED_D1_COVERAGE,
                    "D1 timing sidecar exact missingness coverage changed")
        _, crop_path = _verify_descriptor(
            inputs["camera_crop_contract"], base=raw_root, raw_root=raw_root,
            label=f"{model} camera crop contract",
        )
        crop = _validate_crop_contract(
            load_json(crop_path, f"{model} camera crop contract"), model
        )
        _, alignment_path = _verify_descriptor(
            inputs["alignment_contract"], base=raw_root, raw_root=raw_root,
            label=f"{model} alignment contract",
        )
        alignment = load_json(alignment_path, f"{model} alignment contract")
        _, mapping_path = _verify_descriptor(
            inputs["physical_alignment_receipt"], base=raw_root, raw_root=raw_root,
            label=f"{model} physical alignment receipt",
        )
        mapping = load_json(mapping_path, f"{model} physical alignment receipt")
        authoritative_mapping, authoritative_alignment = (
            _rederive_authoritative_physical_alignment(
                model=model,
                fragment=freeze_fragment,
                timing_descriptor=inputs["timing_sidecar"],
                crop_descriptor=inputs["camera_crop_contract"],
                evidence_base=compiler_receipt_path.parent,
            )
        )
        _validate_mapping_and_alignment(
            model=model, mapping=mapping, mapping_path=mapping_path,
            alignment=alignment, crop=crop,
            timing_descriptor=inputs["timing_sidecar"],
            timing_sidecar=sidecar,
            authoritative_mapping=authoritative_mapping,
            authoritative_alignment=authoritative_alignment,
        )
        roster_by_model[model] = roster
        provenance_by_model[model] = provenance_rows
        provenance_base_by_model[model] = provenance_path.parent
        timing_by_model[model] = sidecar
        crop_by_model[model] = crop
        crop_path_by_model[model] = crop_path
        alignment_by_model[model] = alignment
        mapping_sha_by_model[model] = sha256_file(mapping_path)
        input_identities["models"][model] = {
            key: descriptor(_verify_descriptor(value, base=raw_root, raw_root=raw_root,
                                               label=f"{model} {key}")[1])
            for key, value in inputs.items() if key != "model_id"
        }
    finalized_at = _utc_now()
    inventory, rich = _build_request_inventory(
        roster_by_model=roster_by_model,
        provenance_by_model=provenance_by_model,
        timing_by_model=timing_by_model,
        alignment_by_model=alignment_by_model,
        mapping_sha_by_model=mapping_sha_by_model,
        provenance_base_by_model=provenance_base_by_model,
        inventory_finalized_at=finalized_at,
    )
    # First validate from temporary paths so the existing workflow reopens all
    # signed action/recording receipts.  Relative compiler bundle paths are
    # normalized to absolute paths below before this call.
    for model in MODELS:
        for source in provenance_by_model[model]:
            roster_row = next(row for row in inventory["episode_roster"] if row["cell_id"] == source["cell_id"])
            roster_row["recording_receipt_path"] = str(
                _verify_descriptor(source["recording_receipt"], base=provenance_base_by_model[model],
                                   raw_root=raw_root, label="recording receipt")[1]
            )
            roster_row["action_manifest_path"] = str(
                _verify_descriptor(source["action_manifest"], base=provenance_base_by_model[model],
                                   raw_root=raw_root, label="action manifest")[1]
            )
    try:
        selection = annotation.select_requests(
            inventory,
            seed=annotation.REQUEST_SAMPLING_SEED,
            cap_per_episode=annotation.REQUEST_SAMPLE_CAP,
            inventory_sha256=sha256_bytes(pretty_bytes(inventory)),
        )
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"request inventory failed the annotation workflow: {error}"
        ) from error
    _assert_inventory_and_selection(inventory, selection)

    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        inventory_payload = pretty_bytes(inventory)
        selection = annotation.select_requests(
            inventory,
            seed=annotation.REQUEST_SAMPLING_SEED,
            cap_per_episode=annotation.REQUEST_SAMPLE_CAP,
            inventory_sha256=sha256_bytes(inventory_payload),
        )
        _assert_inventory_and_selection(inventory, selection)
        _write_exclusive(staged / "request_inventory.json", inventory_payload)
        _write_exclusive(staged / "request_selection.json", pretty_bytes(selection))
        image_rows, lineage_rows, replay_receipts = _render_selected(
            temp_root=staged, raw_root=raw_root, selection=selection,
            rich=rich, crop_by_model=crop_by_model,
            crop_path_by_model=crop_path_by_model,
        )
        selected_zero = sum(
            wrapper["selected"] and wrapper["source"]["request_index"] == 0
            for wrapper in selection["requests"]
        )
        expected_images = EXPECTED_TOTAL_SELECTED * 6 - selected_zero
        require(len(image_rows) == len(lineage_rows) == expected_images
                and 736 <= expected_images <= 768,
                "rendered image count differs from selected role contract")
        original_replays = expected_images - 2 * EXPECTED_TOTAL_SELECTED
        require(set(replay_receipts) == set(MODELS)
                and sum(row["request_count"] for row in replay_receipts.values())
                == original_replays,
                "exact-runtime original-camera replay coverage changed")
        lineage = sign_document({
            "schema_version": SOURCE_LINEAGE_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "complete_source_extraction",
            "request_selection_sha256": sha256_file(staged / "request_selection.json"),
            "camera_crop_contract_sha256_by_model": {
                model: crop_by_model[model]["payload_sha256"] for model in MODELS
            },
            "record_count": len(lineage_rows),
            "records": lineage_rows,
            "simulator_state_render_used": False,
            "presentation_video_fps_used": False,
            "generated_frame_index_interpreted_as_action_index": False,
        })
        _write_exclusive(staged / "source_extraction_lineage.json", pretty_bytes(lineage))
        png_manifest = sign_document({
            "schema_version": PNG_MANIFEST_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "rendered_pending_human_pixel_blindness_review",
            "source_extraction_lineage_sha256": sha256_file(
                staged / "source_extraction_lineage.json"
            ),
            "image_count": len(image_rows),
            "images": [
                {
                    "source_image_id": row["source_image_id"],
                    "source_request_id": row["source_request_id"],
                    "image_role": row["image_role"],
                    "annotation_media_path": row["annotation_media_path"],
                    "annotation_media_sha256": row["annotation_media_sha256"],
                    "width_px": row["width_px"],
                    "height_px": row["height_px"],
                }
                for row in image_rows
            ],
            "metadata_chunks_removed": True,
            "overlays_added": False,
            "human_pixel_blindness_review_complete": False,
            "safe_for_rater_distribution": False,
        })
        _write_exclusive(staged / "rendered_png_manifest.json", pretty_bytes(png_manifest))
        review_checklist = _build_pixel_review_checklist(
            temp_root=staged,
            image_rows=image_rows,
            rendered_png_manifest_sha256=sha256_file(
                staged / "rendered_png_manifest.json"
            ),
        )
        _write_exclusive(
            staged / "pixel_blindness_review_checklist.json",
            pretty_bytes(review_checklist),
        )
        pre_review = sign_document({
            "schema_version": PRE_REVIEW_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "pending_human_pixel_blindness_review",
            "selection_manifest_sha256": sha256_file(staged / "request_selection.json"),
            "source_extraction_lineage_sha256": sha256_file(
                staged / "source_extraction_lineage.json"
            ),
            "rendered_png_manifest_sha256": sha256_file(
                staged / "rendered_png_manifest.json"
            ),
            "pixel_blindness_review_checklist_sha256": sha256_file(
                staged / "pixel_blindness_review_checklist.json"
            ),
            "pixel_blindness_receipt_path": None,
            "pixel_blindness_receipt_sha256": None,
            "images": image_rows,
            "human_action_required": (
                "A named human must visually review the exact rendered PNG hashes and sign "
                "wmf-forecast-pixel-blindness-review-v1 before final image inventory or packets."
            ),
            "safe_for_rater_distribution": False,
        })
        _write_exclusive(staged / "image_inventory_pre_review.json", pretty_bytes(pre_review))
        tranche_index = _build_publish_tranche_index(
            temp_root=staged, image_rows=image_rows, lineage_rows=lineage_rows
        )
        _write_exclusive(
            staged / "publish_tranche_index.json", pretty_bytes(tranche_index)
        )
        preparation = sign_document({
            "schema_version": PREPARATION_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "prepared_pending_human_pixel_blindness_review",
            "input_manifest": descriptor(manifest_path),
            "inputs": input_identities,
            "outputs": {
                "request_inventory": _relative_descriptor(staged / "request_inventory.json", staged),
                "request_selection": _relative_descriptor(staged / "request_selection.json", staged),
                "source_extraction_lineage": _relative_descriptor(
                    staged / "source_extraction_lineage.json", staged
                ),
                "rendered_png_manifest": _relative_descriptor(
                    staged / "rendered_png_manifest.json", staged
                ),
                "image_inventory_pre_review": _relative_descriptor(
                    staged / "image_inventory_pre_review.json", staged
                ),
                "pixel_blindness_review_checklist": _relative_descriptor(
                    staged / "pixel_blindness_review_checklist.json", staged
                ),
                "publish_tranche_index": _relative_descriptor(
                    staged / "publish_tranche_index.json", staged
                ),
                "camera_replay_runtime_receipts": {
                    model: _relative_descriptor(
                        staged / "camera_replay_runtime"
                        / f"{model.lower()}_session_receipt.json",
                        staged,
                    )
                    for model in MODELS
                },
            },
            "counts": {
                "episodes": EXPECTED_EPISODES,
                "requests": EXPECTED_TOTAL_REQUESTS,
                "requests_by_model": EXPECTED_REQUESTS,
                "source_timing_capable_requests": sum(EXPECTED_TIMING_CAPABLE.values()),
                "d1_forecast_timing_unavailable_requests": EXPECTED_D1_UNMAPPED,
                "timing_camera_action_eligible_requests": EXPECTED_TOTAL_ELIGIBLE,
                "selected_requests": EXPECTED_TOTAL_SELECTED,
                "selected_requests_by_model": EXPECTED_SELECTED,
                "selected_request_zero_count": selected_zero,
                "source_extraction_records": len(lineage_rows),
                "rendered_pngs": len(image_rows),
                "unique_pixel_review_assets": review_checklist["unique_asset_count"],
                "unique_sanitized_png_assets": tranche_index["unique_asset_count"],
                "publish_tranches": tranche_index["tranche_count"],
                "total_unique_sanitized_png_bytes": tranche_index[
                    "total_unique_asset_bytes"
                ],
                "camera_replay_exact_runtime_sessions": len(replay_receipts),
                "original_camera_frames_replayed_exact_runtime": original_replays,
                "human_pixel_blindness_reviews": 0,
                "human_labels": 0,
            },
            "sampling": {
                "seed": annotation.REQUEST_SAMPLING_SEED,
                "cap_per_episode": annotation.REQUEST_SAMPLE_CAP,
                "eligible_requests_per_episode": 14,
                "inclusion_probability": EXPECTED_INCLUSION_PROBABILITY,
                "inclusion_probability_exact": EXPECTED_INCLUSION_PROBABILITY_EXACT,
            },
            "science_counts": {
                "model_runtime_loads": 0,
                "model_servers_started": 0,
                "model_requests_issued_by_job": 0,
                "simulator_processes_started": 0,
                "physical_resets": 0,
                "robot_episodes": 0,
                "behavioral_actions_executed_by_job": 0,
                "behavioral_cells_launched_by_job": 0,
                "labels_created_by_job": 0,
            },
            "human_pixel_blindness_review_complete": False,
            "rater_packets_created": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
            "claim_boundary": (
                "CPU-only provenance join, source extraction, deterministic selection and PNG "
                "rendering. No human visual review, labels, policy skill, or confirmation release."
            ),
            "completed_at_utc": _utc_now(),
        })
        _write_atomic_directory(
            output_dir,
            {"preparation_receipt.json": pretty_bytes(preparation)},
            staged,
        )
        return preparation
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        raise


def finalize_reviewed_inventory(
    *, preparation_dir: Path, review_path: Path, review_sha256: str, output_path: Path
) -> dict[str, Any]:
    """Close only the human pixel-blindness gate; never create rater packets."""

    preparation_dir = Path(preparation_dir).resolve(strict=True)
    pre_path = preparation_dir / "image_inventory_pre_review.json"
    pre = load_json(pre_path, "pre-review image inventory")
    verify_signed(pre, "pre-review image inventory")
    require(pre.get("schema_version") == PRE_REVIEW_SCHEMA
            and pre.get("status") == "pending_human_pixel_blindness_review"
            and pre.get("safe_for_rater_distribution") is False,
            "pre-review image inventory state changed")
    review_path = Path(review_path).resolve(strict=True)
    require(SHA_RE.fullmatch(review_sha256) is not None
            and sha256_file(review_path) == review_sha256,
            "pixel-blindness review file hash changed")
    review = load_json(review_path, "pixel-blindness review")
    images = pre.get("images")
    require(isinstance(images, list) and images, "pre-review image inventory is empty")
    try:
        annotation._validate_pixel_blindness_receipt(
            review,
            receipt_path=review_path,
            stage="development",
            artifact_scope="annotation_media",
            expected_assets=[
                (row["annotation_media_sha256"], row["width_px"], row["height_px"])
                for row in images
            ],
        )
    except BaseException as error:
        raise AnnotationMediaBridgeError(
            f"human pixel-blindness receipt failed validation: {error}"
        ) from error
    require(not output_path.exists(), f"refusing to overwrite image inventory: {output_path}")
    relative_review = os.path.relpath(review_path, output_path.parent.resolve())
    inventory = {
        "schema_version": IMAGE_INVENTORY_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "development",
        "selection_manifest_sha256": pre["selection_manifest_sha256"],
        "pixel_blindness_receipt_path": relative_review,
        "pixel_blindness_receipt_sha256": review_sha256,
        "images": images,
    }
    _write_exclusive(output_path, pretty_bytes(inventory))
    return inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--manifest-sha256", required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    finalize = commands.add_parser("finalize-reviewed-inventory")
    finalize.add_argument("--preparation-dir", type=Path, required=True)
    finalize.add_argument("--review", type=Path, required=True)
    finalize.add_argument("--review-sha256", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("camera-replay-worker")
    worker.add_argument("--model", choices=MODELS, required=True)
    worker.add_argument("--crop-contract", type=Path, required=True)
    worker.add_argument("--crop-contract-sha256", required=True)
    worker.add_argument("--crop-contract-bytes", type=int, required=True)
    worker.add_argument("--bridge-sha256", required=True)
    worker.add_argument("--camera-replay-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "camera-replay-worker":
        _run_camera_replay_worker(args)
        return 0
    if args.command == "prepare":
        result = prepare(args.manifest, args.manifest_sha256, args.output_dir)
    else:
        result = finalize_reviewed_inventory(
            preparation_dir=args.preparation_dir,
            review_path=args.review,
            review_sha256=args.review_sha256,
            output_path=args.output,
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        print(json.dumps({
            "status": "technical_failure",
            "error_type": type(error).__name__,
            "detail": str(error),
        }, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(1)
