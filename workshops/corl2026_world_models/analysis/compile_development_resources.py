#!/usr/bin/env python3
"""Compile authenticated development resource measurements from retained evidence.

This program is deliberately CPU-only.  It reuses the formal development
evidence compiler's deep validators to authenticate the eight aggregate
receipts, 32 cells, 1,152 official request receipts, recorder journals and
completions.  It then measures clocks and storage already present on the PVC.
It never imports a model runtime, starts a simulator, issues a request, creates
a label, or releases confirmation.

The output is intentionally a missingness-aware resource audit, not the legacy
``wmf-development-resource-measurement-v1`` accepted by the confirmation
freeze.  The GM execution site and bounded 58-block core study are already
authorized.  N3 behavioral GPU peaks, D1 simulator peaks, simultaneous all-GPU
peaks, annotation time, adjudication time, and a measured safe execution
topology/concurrency envelope were not retained by the completed development
runs.  They remain explicit nulls instead of fabricated zeros.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_COMPILER_PATH = Path(__file__).with_name(
    "compile_development_evidence.py"
)
TIMING_VALIDATOR_PATH = Path(__file__).with_name("qualify_forecast_timing.py")
CONTRACT_PATH = (
    FORECAST_ROOT
    / "experiments/forecast_layout/development_resource_contract.json"
)

INPUT_SCHEMA = "wmf-development-resource-compiler-input-v1"
CELL_SCHEMA = "wmf-development-resource-cell-measurement-v1"
AGGREGATE_SCHEMA = "wmf-development-resource-aggregate-v1"
FILE_INVENTORY_SCHEMA = "wmf-development-resource-file-inventory-v1"
COMPILER_RECEIPT_SCHEMA = "wmf-development-resource-compiler-receipt-v1"
STUDY_ID = "WMF-ABLATION-001"
MODE = "formal_full"
MODELS = ("N3", "D1")
LAYOUTS = ("D01", "D02", "D03", "D04")
EXPECTED_CELLS = 32
EXPECTED_REQUESTS = 1152
EXPECTED_ACTIONS = 14400
MODEL_REQUESTS_PER_CELL = {"N3": 15, "D1": 57}
MODEL_ACTIONS_PER_CELL = {"N3": 450, "D1": 450}
MODEL_REQUEST_COUNTS = {"N3": 240, "D1": 912}
RAW_ROOT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912"
)
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")

EPISODE_WALL_DEFINITION = "journal_attempt_boundary_monotonic_ns_delta"
REQUEST_ROUNDTRIP_DEFINITION = (
    "recorder_model_request_sent_to_model_response_received_monotonic_ns_delta_sum"
)
N3_SERVICE_DEFINITION = (
    "n3_server_request_started_to_completed_monotonic_ns_delta_sum"
)
D1_RANK0_DEFINITION = "d1_producer_rank0_official_infer_wrapper_wall_seconds_sum"
D1_RANK_PROXY_DEFINITION = (
    "d1_producer_sum_of_per_rank_forward_wall_seconds_proxy_sum"
)
D1_DECODE_DEFINITION = "d1_producer_offline_vae_decode_wall_seconds_sum"
D1_OUTPUT_BYTES_DEFINITION = (
    "d1_producer_action_latent_decoded_tensor_and_decoded_rgb_file_bytes_sum"
)
D1_SOURCE_COST_DEFINITION = (
    "GPU-seconds proxy is the sum of measured distributed-rank forward wall "
    "times; it is not a cloud billing claim"
)
RAW_ATTRIBUTABLE_DEFINITION = (
    "unique_resolved_regular_file_paths_in_cell_and_request_trees_excluding_shared_block_overhead"
)
GPU_PEAK_DEFINITION = (
    "pytorch_cuda_allocator_peak_after_per_request_reset_per_rank_or_rank0_decode"
)

SCIENCE_COUNTS = {
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
EXECUTION_AUTHORIZATION = {
    "status": "authorized_bounded_core_study",
    "selected_execution_host": "GM cluster",
    "selected_kubernetes_namespace": "211247-prod",
    "authorized_block_counts": {
        "recording_pilot_max": 2,
        "development": 8,
        "confirmation": 48,
        "maximum_core": 58,
    },
    "authorized_behavioral_cell_counts": {
        "recording_pilot_max": 8,
        "development": 32,
        "confirmation": 192,
        "maximum_core": 232,
    },
    "confirmation_cells_authorized_by_model": {"N3": 96, "D1": 96},
    "measured_safe_parallel_blocks_by_model": None,
    "measured_gpu_memory_envelope_bytes_per_gpu": None,
    "measurement_freeze_complete": False,
    "user_authorization_pending": False,
    "boundary": (
        "Authorization does not substitute for measured GPU peaks, annotation time, "
        "or the pre-confirmation safe topology/concurrency freeze."
    ),
}
MISSING_RELEASE_REQUIREMENTS = (
    "N3 behavioral model-server allocator peak",
    "N3 simulator GPU peak",
    "D1 simulator GPU peak",
    "simultaneous all-process GPU peak for either model",
    "two independent raters' per-image annotation time",
    "adjudication time",
    "measured confirmation execution topology, GPU-memory envelope, and safe concurrency freeze",
)


class ResourceCompilerError(RuntimeError):
    """Retained resource evidence failed a strict validation gate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceCompilerError(message)


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ResourceCompilerError(f"cannot load workshop module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ResourceCompilerError(f"non-finite JSON token: {value}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    supplied = Path(path)
    require(not supplied.is_symlink(), f"{label} is a symlink")
    try:
        value = json.loads(
            supplied.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ResourceCompilerError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResourceCompilerError(f"{label} is unreadable JSON: {supplied}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def compact_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ResourceCompilerError("value is not finite canonical JSON") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ResourceCompilerError("value is not finite JSON") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ResourceCompilerError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def sign_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    output = dict(value)
    output["payload_sha256"] = sha256_bytes(compact_bytes(output))
    return output


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("payload_sha256")
    require(
        isinstance(observed, str) and SHA_RE.fullmatch(observed) is not None,
        f"{label} signature is invalid",
    )
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(
        sha256_bytes(compact_bytes(unsigned)) == observed,
        f"{label} signature changed",
    )


def _valid_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
        f"{label} is not SHA-256",
    )
    return value


def _finite_number(
    value: Any, label: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    number = float(value)
    if positive:
        require(number > 0, f"{label} is not positive")
    if nonnegative:
        require(number >= 0, f"{label} is negative")
    return number


def _lexical_absolute(path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def reject_symlink_components(
    path: Path, label: str, *, stop: Path | None = None
) -> Path:
    lexical = _lexical_absolute(path)
    boundary = None if stop is None else _lexical_absolute(stop)
    if boundary is not None:
        require(
            lexical.is_relative_to(boundary),
            f"{label} lexically escapes its boundary",
        )
    cursor = lexical
    while True:
        require(not cursor.is_symlink(), f"{label} contains a symlink: {cursor}")
        if boundary is not None and cursor == boundary:
            return lexical
        parent = cursor.parent
        if parent == cursor:
            require(boundary is None, f"{label} did not reach its boundary")
            return lexical
        cursor = parent


def under(path: Path, root: Path, label: str, *, must_exist: bool = True) -> Path:
    lexical_root = reject_symlink_components(root, f"{label} root")
    lexical = reject_symlink_components(path, label, stop=lexical_root)
    resolved_root = lexical_root.resolve()
    try:
        resolved = lexical.resolve(strict=must_exist)
    except OSError as error:
        raise ResourceCompilerError(f"{label} is unavailable: {lexical}") from error
    require(resolved.is_relative_to(resolved_root), f"{label} escapes its root")
    return resolved


def file_descriptor(path: Path) -> dict[str, Any]:
    reject_symlink_components(path, "file descriptor")
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as error:
        raise ResourceCompilerError(f"descriptor file is unavailable: {path}") from error
    require(resolved.is_file(), f"descriptor target is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def bytes_descriptor(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(payload), "bytes": len(payload)}


def validate_descriptor(
    value: Any,
    *,
    base: Path,
    raw_root: Path,
    label: str,
    expected_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    require(set(value) >= {"path", "sha256", "bytes"}, f"{label} descriptor fields changed")
    raw_path = value.get("path")
    size = value.get("bytes")
    digest = _valid_sha(value.get("sha256"), f"{label} digest")
    require(isinstance(raw_path, str) and raw_path, f"{label} path is missing")
    require(type(size) is int and size >= 0, f"{label} byte count is invalid")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    path = under(candidate, raw_root, label)
    if expected_path is not None:
        require(path == expected_path, f"{label} path changed")
    require(path.is_file(), f"{label} is not a regular file")
    require(path.stat().st_size == size, f"{label} byte count changed")
    require(sha256_file(path) == digest, f"{label} file hash changed")
    return {"path": str(path), "sha256": digest, "bytes": size}, path


def _same_descriptor(left: Any, right: Any, label: str) -> None:
    require(isinstance(left, Mapping) and isinstance(right, Mapping), f"{label} is missing")
    require(
        {key: left.get(key) for key in ("path", "sha256", "bytes")}
        == {key: right.get(key) for key in ("path", "sha256", "bytes")},
        f"{label} descriptor changed",
    )


def _validate_contract() -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH, "development resource contract")
    require(
        contract.get("schema_version") == "wmf-development-resource-contract-v1"
        and contract.get("study_id") == STUDY_ID
        and contract.get("mode") == MODE,
        "development resource contract identity changed",
    )
    require(
        contract.get("cohort")
        == {
            "models": list(MODELS),
            "layout_pair_ids": list(LAYOUTS),
            "cells": EXPECTED_CELLS,
            "source_behavioral_requests": EXPECTED_REQUESTS,
            "source_behavioral_actions": EXPECTED_ACTIONS,
            "aggregate_receipts": 8,
            "d1_server_receipts": 4,
            "timing_sidecar_receipts": 2,
            "queue_wrapper_snapshots": 14,
        },
        "development resource contract cohort changed",
    )
    require(
        contract.get("measurement_definitions")
        == {
            "episode_wall": EPISODE_WALL_DEFINITION,
            "policy_request_roundtrip": REQUEST_ROUNDTRIP_DEFINITION,
            "n3_server_request_service": N3_SERVICE_DEFINITION,
            "d1_rank0_inference_wrapper": D1_RANK0_DEFINITION,
            "d1_summed_rank_forward_gpu_seconds_proxy": D1_RANK_PROXY_DEFINITION,
            "d1_source_cost_definition": D1_SOURCE_COST_DEFINITION,
            "d1_offline_decode": D1_DECODE_DEFINITION,
            "d1_retained_output_bytes": D1_OUTPUT_BYTES_DEFINITION,
            "raw_attributable_bytes": RAW_ATTRIBUTABLE_DEFINITION,
            "gpu_allocator_peak": GPU_PEAK_DEFINITION,
            "annotation_time": "sum_of_per_image_annotation_seconds_from_locked_rater_responses",
        },
        "development resource measurement definitions changed",
    )
    require(
        contract.get("execution_authorization") == EXECUTION_AUTHORIZATION,
        "development resource execution authorization changed",
    )
    boundary = contract.get("release_boundary")
    require(
        isinstance(boundary, Mapping)
        and boundary.get("safe_to_release_confirmation") is False
        and boundary.get("legacy_resource_receipts_emitted") is False
        and boundary.get("confirmation_jobs_emitted") is False
        and boundary.get("diagnostic_or_missingness_audit_can_release") is False,
        "development resource release boundary changed",
    )
    require(contract.get("science_counts") == SCIENCE_COUNTS, "resource contract claims science")
    return contract


def _event_payloads(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != kind:
            continue
        payload = row.get("payload")
        require(isinstance(payload, Mapping), f"{kind} journal payload is invalid")
        output.append(dict(payload))
    return output


def _descriptor_list_from_manifest(
    manifest: Mapping[str, Any], key: str, identity_keys: Sequence[str]
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    rows = manifest.get(key)
    require(isinstance(rows, list), f"{key} must be a list")
    output: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"{key} row {index} is invalid")
        identity: list[str] = []
        for name in identity_keys:
            value = row.get(name)
            require(isinstance(value, str) and value, f"{key} row {index} lacks {name}")
            identity.append(value)
        descriptor = row.get("receipt" if key != "queue_wrapper_snapshots" else "snapshot")
        require(isinstance(descriptor, Mapping), f"{key} row {index} descriptor is missing")
        tuple_identity = tuple(identity)
        require(tuple_identity not in output, f"{key} duplicates {tuple_identity}")
        output[tuple_identity] = row
    return output


def _validate_manifest(
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[dict[str, Any], Path]:
    supplied = Path(manifest_path)
    reject_symlink_components(supplied, "resource input manifest")
    try:
        path = supplied.resolve(strict=True)
    except OSError as error:
        raise ResourceCompilerError("resource input manifest is unavailable") from error
    require(path.is_file(), "resource input manifest is not a regular file")
    _valid_sha(manifest_sha256, "resource input manifest digest")
    require(sha256_file(path) == manifest_sha256, "resource input manifest hash changed")
    manifest = load_json(path, "resource input manifest")
    require(
        set(manifest)
        == {
            "schema_version",
            "study_id",
            "mode",
            "raw_root",
            "planned_cells",
            "timing_sidecar_receipts",
            "aggregate_receipts",
            "d1_server_receipts",
            "queue_wrapper_snapshots",
        },
        "resource input manifest fields changed",
    )
    require(manifest.get("schema_version") == INPUT_SCHEMA, "resource input schema changed")
    require(manifest.get("study_id") == STUDY_ID, "resource input study changed")
    require(manifest.get("mode") == MODE, "only formal_full resource compilation is supported")
    require(manifest.get("raw_root") == str(RAW_ROOT), "resource raw root changed")
    raw_root = under(Path(str(manifest["raw_root"])), RAW_ROOT, "resource raw root")
    require(raw_root == RAW_ROOT, "resource raw root did not resolve exactly")
    return manifest, path


def _load_and_validate_aggregates(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    raw_root: Path,
    evidence: ModuleType,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str], dict[str, Any]],
]:
    planned_descriptor, planned_path = validate_descriptor(
        manifest.get("planned_cells"),
        base=manifest_path.parent,
        raw_root=raw_root,
        label="planned-cell CSV",
    )
    require(
        planned_descriptor["sha256"] == evidence.PLANNED_CELLS_SHA256,
        "planned-cell CSV is not authoritative",
    )
    planned = evidence._load_planned_cells(planned_path)
    rows = _descriptor_list_from_manifest(
        manifest, "aggregate_receipts", ("model_id", "layout_pair_id")
    )
    expected = {(model, layout) for model in MODELS for layout in LAYOUTS}
    require(set(rows) == expected, "resource compiler requires exactly eight aggregates")
    descriptors: dict[tuple[str, str], dict[str, Any]] = {}
    compiled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    receipts: dict[tuple[str, str], dict[str, Any]] = {}
    for model in MODELS:
        for layout in LAYOUTS:
            row = rows[(model, layout)]
            descriptor, path = validate_descriptor(
                row.get("receipt"),
                base=manifest_path.parent,
                raw_root=raw_root,
                label=f"{model} {layout} aggregate",
            )
            try:
                validated_descriptor, cells = evidence._validate_aggregate(
                    descriptor=descriptor,
                    path=path,
                    model=model,
                    layout=layout,
                    expected_cells=planned[(model, layout)],
                    camera_id="over_shoulder_left_camera",
                    raw_root=raw_root,
                )
            except BaseException as error:
                raise ResourceCompilerError(
                    f"{model} {layout} failed deep development validation: {error}"
                ) from error
            require(len(cells) == 4, f"{model} {layout} cell count changed")
            descriptors[(model, layout)] = dict(validated_descriptor)
            compiled[(model, layout)] = list(cells)
            receipts[(model, layout)] = load_json(path, f"{model} {layout} aggregate")
    return descriptors, compiled, receipts


def _compiled_cells_by_model(
    compiled: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    output = {model: [] for model in MODELS}
    seen: set[str] = set()
    for model in MODELS:
        for layout in LAYOUTS:
            for cell in compiled[(model, layout)]:
                cell_id = cell.get("cell", {}).get("cell_id")
                require(isinstance(cell_id, str) and cell_id, "compiled cell ID is missing")
                require(cell_id not in seen, f"compiled cell is duplicated: {cell_id}")
                seen.add(cell_id)
                output[model].append(dict(cell))
    require(len(seen) == EXPECTED_CELLS, "compiled cell cohort is incomplete")
    return output


def _validate_timing_receipts(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    raw_root: Path,
    aggregates: Mapping[tuple[str, str], Mapping[str, Any]],
    compiled_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    timing: ModuleType,
) -> dict[str, dict[str, Any]]:
    rows = _descriptor_list_from_manifest(
        manifest, "timing_sidecar_receipts", ("model_id",)
    )
    require(set(rows) == {(model,) for model in MODELS}, "both timing receipts are required")
    expected_job = {
        "N3": "timing-n3-development-sidecar-001",
        "D1": "timing-d1-development-sidecar-002",
    }
    expected_requests = MODEL_REQUEST_COUNTS
    output: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        descriptor, path = validate_descriptor(
            rows[(model,)].get("receipt"),
            base=manifest_path.parent,
            raw_root=raw_root,
            label=f"{model} timing sidecar receipt",
            expected_path=(
                raw_root / "control/jobs" / expected_job[model]
                / "publish/timing_job_receipt.json"
            ),
        )
        receipt = load_json(path, f"{model} timing sidecar receipt")
        verify_signed(receipt, f"{model} timing sidecar receipt")
        exact = {
            "schema_version": "wmf-development-timing-sidecar-queue-job-v1",
            "namespace": "wmf_ablation_001_20260912",
            "study_id": STUDY_ID,
            "model_id": model,
            "job_id": expected_job[model],
            "status": "passed",
            "decision": "go",
            "referenced_behavioral_cells": 16,
            "referenced_behavioral_model_requests": expected_requests[model],
            "development_timing_sidecar_valid": True,
            "physical_time_qualified": True,
            "safe_to_release_confirmation": False,
        }
        for key, wanted in exact.items():
            require(receipt.get(key) == wanted, f"{model} timing receipt changed: {key}")
        require(receipt.get("science_counts") == {
            "model_runtime_loads": 0,
            "model_servers_started": 0,
            "model_requests_issued_by_job": 0,
            "physical_resets": 0,
            "robot_episodes": 0,
            "behavioral_actions_executed_by_job": 0,
            "behavioral_cells_launched_by_job": 0,
        }, f"{model} timing receipt claims science")
        inputs = receipt.get("inputs")
        require(isinstance(inputs, Mapping), f"{model} timing inputs are missing")
        input_aggregates = inputs.get("development_aggregates")
        require(isinstance(input_aggregates, Mapping), f"{model} timing aggregates are missing")
        require(set(input_aggregates) == set(LAYOUTS), f"{model} timing aggregate coverage changed")
        for layout in LAYOUTS:
            value = input_aggregates[layout]
            require(isinstance(value, Mapping), f"{model} {layout} timing aggregate is invalid")
            _same_descriptor(
                value.get("aggregate_receipt"),
                aggregates[(model, layout)],
                f"{model} {layout} timing aggregate",
            )
            listed_cells = value.get("cell_receipts")
            expected_cells = [
                cell["cell_descriptor"] for cell in compiled_by_model[model]
                if cell["cell"]["layout_pair_id"] == layout
            ]
            require(
                isinstance(listed_cells, list) and len(listed_cells) == 4,
                f"{model} {layout} timing cell inventory changed",
            )
            for index, expected_cell in enumerate(expected_cells):
                _same_descriptor(
                    listed_cells[index], expected_cell,
                    f"{model} {layout} timing cell {index}",
                )
        outputs = receipt.get("outputs")
        require(isinstance(outputs, Mapping), f"{model} timing outputs are missing")
        inventory_descriptor, inventory_path = validate_descriptor(
            outputs.get("raw_request_inventory"),
            base=path.parent,
            raw_root=raw_root,
            label=f"{model} timing request inventory",
            expected_path=(
                raw_root / "control/jobs" / expected_job[model] / "raw"
                / f"{model.lower()}_development_request_inventory.json"
            ),
        )
        timing_descriptor, timing_path = validate_descriptor(
            outputs.get("raw_development_timing_sidecar"),
            base=path.parent,
            raw_root=raw_root,
            label=f"{model} development timing sidecar",
            expected_path=(
                raw_root / "control/jobs" / expected_job[model] / "raw"
                / f"{model.lower()}_development_timing_sidecar.json"
            ),
        )
        inventory = load_json(inventory_path, f"{model} timing request inventory")
        verify_signed(inventory, f"{model} timing request inventory")
        require(
            inventory.get("schema_version") == timing.REQUEST_INVENTORY_SCHEMA
            and inventory.get("study_id") == STUDY_ID
            and inventory.get("model_id") == model
            and inventory.get("status") == "complete_exact_development_inventory"
            and inventory.get("old_request_receipts_modified") is False
            and inventory.get("model_requests_issued_by_inventory_job") == 0
            and inventory.get("behavioral_actions_executed_by_inventory_job") == 0,
            f"{model} timing request inventory header changed",
        )
        entries = inventory.get("request_receipts")
        require(
            isinstance(entries, list) and len(entries) == expected_requests[model],
            f"{model} timing request inventory count changed",
        )
        compiled_entries: list[dict[str, Any]] = []
        for cell in compiled_by_model[model]:
            cell_id = cell["cell"]["cell_id"]
            for request_index, request in enumerate(cell["requests"]):
                compiled_entries.append({
                    "cell_id": cell_id,
                    "request_index": request_index,
                    "request_receipt": request["official_request_receipt"],
                    "adapter_completion": request["adapter_completion"],
                    "adapter_journal": request["adapter_journal"],
                })
        require(len(compiled_entries) == len(entries), f"{model} compiled timing projection changed")
        for index, (observed, wanted) in enumerate(zip(entries, compiled_entries)):
            require(
                isinstance(observed, Mapping)
                and observed.get("cell_id") == wanted["cell_id"]
                and observed.get("request_index") == wanted["request_index"],
                f"{model} timing request identity changed at {index}",
            )
            for key in ("request_receipt", "adapter_completion", "adapter_journal"):
                _same_descriptor(
                    observed.get(key), wanted[key], f"{model} timing request {index} {key}"
                )
        hashes = [entry["request_receipt"]["sha256"] for entry in compiled_entries]
        require(
            receipt.get("source_request_receipt_sha256s") == hashes
            and len(set(hashes)) == len(hashes),
            f"{model} timing request hash roster changed",
        )
        try:
            validated_timing = timing.validate_development_timing(
                timing_path,
                timing_descriptor["sha256"],
                expected_model=model,
                expected_request_hashes=hashes,
            )
        except BaseException as error:
            raise ResourceCompilerError(
                f"{model} timing sidecar failed semantic replay: {error}"
            ) from error
        bindings = validated_timing.get("request_timing_bindings")
        require(
            isinstance(bindings, list) and len(bindings) == expected_requests[model],
            f"{model} timing binding count changed",
        )
        output[model] = {
            "receipt": descriptor,
            "request_inventory": inventory_descriptor,
            "timing_sidecar": timing_descriptor,
            "request_receipt_sha256s": hashes,
            "physical_time_qualified": True,
            "physical_time_coverage_complete": model == "N3",
            **(
                {"request_timing_coverage": validated_timing.get("request_timing_coverage")}
                if model == "D1" else {}
            ),
        }
    return output


def _validate_d1_server_receipts(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    raw_root: Path,
    aggregate_receipts: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = _descriptor_list_from_manifest(
        manifest, "d1_server_receipts", ("layout_pair_id",)
    )
    require(set(rows) == {(layout,) for layout in LAYOUTS}, "four D1 server receipts are required")
    output: dict[str, dict[str, Any]] = {}
    for layout in LAYOUTS:
        simulator = aggregate_receipts[("D1", layout)]
        server_job_id = simulator.get("server_job_id")
        run_id = simulator.get("run_id")
        require(
            isinstance(server_job_id, str) and SAFE_ID_RE.fullmatch(server_job_id),
            f"D1 {layout} server job ID is invalid",
        )
        expected_path = raw_root / "control/jobs" / server_job_id / "publish/d1_behavioral_server_receipt.json"
        descriptor, path = validate_descriptor(
            rows[(layout,)].get("receipt"),
            base=manifest_path.parent,
            raw_root=raw_root,
            label=f"D1 {layout} server receipt",
            expected_path=expected_path,
        )
        receipt = load_json(path, f"D1 {layout} server receipt")
        exact = {
            "schema_version": "wmf-d1-behavioral-development-server-job-v1",
            "status": "passed",
            "exit_code": 0,
            "failure": None,
            "phase": "development",
            "layout_pair_id": layout,
            "block_id": f"wmf_ablation_001_20260912__development__{layout}__D1",
            "run_id": run_id,
            "server_job_id": server_job_id,
            "paired_simulator_job_id": simulator.get("simulator_job_id"),
            "study_commit": simulator.get("study_commit"),
            "all_server_children_reaped": True,
        }
        for key, wanted in exact.items():
            require(receipt.get(key) == wanted, f"D1 {layout} server receipt changed: {key}")
        process = receipt.get("server_process_exit")
        require(
            isinstance(process, Mapping)
            and process.get("status") == "reaped"
            and process.get("reaped") is True,
            f"D1 {layout} server process was not reaped",
        )
        _same_descriptor(
            receipt.get("server_ready"), simulator.get("server_ready"),
            f"D1 {layout} server-ready binding",
        )
        attempt_root = raw_root / "behavioral/development/D1" / layout / "server_attempts" / server_job_id
        require(
            Path(str(receipt.get("raw_attempt_root", ""))) == attempt_root,
            f"D1 {layout} server attempt root changed",
        )
        topology_descriptor, topology_path = validate_descriptor(
            receipt.get("topology"),
            base=path.parent,
            raw_root=raw_root,
            label=f"D1 {layout} server topology",
            expected_path=attempt_root / "topology_receipt.json",
        )
        topology = load_json(topology_path, f"D1 {layout} server topology")
        require(
            topology.get("schema_version") == "wmf-d1-two-b200-topology-v1"
            and topology.get("status") == "passed"
            and topology.get("preexisting_compute_process_count") == 0,
            f"D1 {layout} server topology header changed",
        )
        devices = topology.get("nvidia_smi_devices")
        require(
            isinstance(devices, list)
            and len(devices) == 2
            and all(isinstance(row, Mapping) and row.get("name") == "NVIDIA B200" for row in devices)
            and len({row.get("uuid") for row in devices}) == 2,
            f"D1 {layout} server topology device roster changed",
        )
        output[layout] = {
            "receipt": descriptor,
            "raw_attempt_root": str(attempt_root),
            "topology": topology_descriptor,
            "topology_payload": topology,
        }
    return output


def _validate_topologies(
    *,
    raw_root: Path,
    aggregate_receipts: Mapping[tuple[str, str], Mapping[str, Any]],
    d1_servers: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {"N3": {}, "D1": {}}
    for layout in LAYOUTS:
        n3 = aggregate_receipts[("N3", layout)]
        n3_attempt = Path(str(n3.get("raw_attempt_root", "")))
        descriptor, path = validate_descriptor(
            n3.get("topology"),
            base=n3_attempt,
            raw_root=raw_root,
            label=f"N3 {layout} topology",
            expected_path=n3_attempt / "topology.json",
        )
        value = load_json(path, f"N3 {layout} topology")
        devices = value.get("devices")
        require(
            value.get("schema_version") == "wmf-n3-two-b200-topology-v1"
            and value.get("status") == "passed"
            and value.get("preexisting_compute_process_count") == 0
            and isinstance(devices, list)
            and len(devices) == 2
            and {row.get("index") for row in devices if isinstance(row, Mapping)} == {"0", "1"}
            and all(isinstance(row, Mapping) and row.get("name") == "NVIDIA B200" for row in devices)
            and len({row.get("uuid") for row in devices if isinstance(row, Mapping)}) == 2,
            f"N3 {layout} topology changed",
        )
        output["N3"][layout] = {
            "joint_worker_topology": descriptor,
            "model_gpu_count": 1,
            "simulator_gpu_count": 1,
            "devices": devices,
        }

        simulator = aggregate_receipts[("D1", layout)]
        sim_attempt = Path(str(simulator.get("raw_attempt_root", "")))
        sim_descriptor, sim_path = validate_descriptor(
            simulator.get("topology"),
            base=sim_attempt,
            raw_root=raw_root,
            label=f"D1 {layout} simulator topology",
            expected_path=sim_attempt / "topology.json",
        )
        sim_value = load_json(sim_path, f"D1 {layout} simulator topology")
        device = sim_value.get("device")
        require(
            sim_value.get("schema_version") == "wmf-d1-behavioral-simulator-one-b200-v1"
            and sim_value.get("status") == "passed"
            and sim_value.get("preexisting_compute_process_count") == 0
            and isinstance(device, Mapping)
            and device.get("name") == "NVIDIA B200"
            and str(device.get("uuid", "")).startswith("GPU-"),
            f"D1 {layout} simulator topology changed",
        )
        output["D1"][layout] = {
            "simulator_topology": sim_descriptor,
            "server_topology": d1_servers[layout]["topology"],
            "simulator_gpu_count": 1,
            "model_server_gpu_count": 2,
            "simulator_device": dict(device),
            "server_devices": d1_servers[layout]["topology_payload"]["nvidia_smi_devices"],
        }
    return output


def _parse_timestamp(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ResourceCompilerError(f"{label} is not RFC3339") from error
    require(parsed.tzinfo is not None, f"{label} lacks a timezone")
    return parsed


def _validate_queue_snapshots(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    raw_root: Path,
    aggregate_receipts: Mapping[tuple[str, str], Mapping[str, Any]],
    d1_servers: Mapping[str, Mapping[str, Any]],
    timing_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = _descriptor_list_from_manifest(manifest, "queue_wrapper_snapshots", ("job_id",))
    expected_jobs: dict[str, tuple[str, str, str]] = {}
    for model in MODELS:
        for layout in LAYOUTS:
            aggregate = aggregate_receipts[(model, layout)]
            job_id = (
                aggregate.get("queue_descriptor", {}).get("job_id")
                if isinstance(aggregate.get("queue_descriptor"), Mapping) else None
            )
            require(isinstance(job_id, str), f"{model} {layout} queue job ID is missing")
            worker = (
                "wmf-forecast-0912-worker-n3-00"
                if model == "N3" else "wmf-forecast-0912-worker-00"
            )
            source = aggregate.get("source_commit" if model == "N3" else "study_commit")
            require(isinstance(source, str), f"{model} {layout} source commit is missing")
            expected_jobs[job_id] = (worker, source, str(aggregate["queue_descriptor"]["sha256"]))
    for layout in LAYOUTS:
        server = d1_servers[layout]
        receipt = load_json(Path(server["receipt"]["path"]), f"D1 {layout} server receipt")
        queue_descriptor = receipt.get("queue_descriptor")
        require(isinstance(queue_descriptor, Mapping), f"D1 {layout} server queue descriptor is missing")
        expected_jobs[str(receipt["server_job_id"])] = (
            "wmf-forecast-0912-worker-d1-00",
            str(receipt["study_commit"]),
            str(queue_descriptor["sha256"]),
        )
    for model in MODELS:
        timing_receipt = load_json(
            Path(timing_receipts[model]["receipt"]["path"]), f"{model} timing receipt"
        )
        expected_jobs[str(timing_receipt["job_id"])] = (
            str(timing_receipt["worker_id"]),
            str(timing_receipt["study_commit"]),
            str(timing_receipt["queue_descriptor"]["sha256"]),
        )
    require(set(rows) == {(job_id,) for job_id in expected_jobs}, "queue snapshot cohort changed")
    output: dict[str, dict[str, Any]] = {}
    for job_id, (worker_id, source_commit, descriptor_sha) in expected_jobs.items():
        expected_snapshot_path = (
            raw_root / "control/results-git/results/wmf_ablation_001_20260912/jobs"
            / f"{job_id}.json"
        )
        snapshot_descriptor, snapshot_path = validate_descriptor(
            rows[(job_id,)].get("snapshot"),
            base=manifest_path.parent,
            raw_root=raw_root,
            label=f"{job_id} queue snapshot",
            expected_path=expected_snapshot_path,
        )
        snapshot = load_json(snapshot_path, f"{job_id} queue snapshot")
        exact = {
            "job_id": job_id,
            "worker_id": worker_id,
            "source_commit": source_commit,
            "descriptor_sha256": descriptor_sha,
            "status": "succeeded",
            "returncode": 0,
            "error_type": None,
            "child_reaped": True,
        }
        for key, wanted in exact.items():
            require(snapshot.get(key) == wanted, f"{job_id} queue snapshot changed: {key}")
        wall = _finite_number(snapshot.get("wall_seconds"), f"{job_id} wall seconds", positive=True)
        started = _parse_timestamp(snapshot.get("started_at"), f"{job_id} start")
        ended = _parse_timestamp(snapshot.get("ended_at"), f"{job_id} end")
        require(ended >= started, f"{job_id} queue wall clock moved backward")
        require(
            abs((ended - started).total_seconds() - wall) <= 2.0,
            f"{job_id} monotonic and wall-clock windows disagree",
        )
        raw_result_path = raw_root / "control/jobs" / job_id / "result.json"
        raw_result_descriptor = file_descriptor(raw_result_path)
        raw_result = load_json(raw_result_path, f"{job_id} raw queue result")
        require(
            raw_result.get("schema_version") == "wmf-cluster-result-v1"
            and raw_result.get("namespace") == "wmf_ablation_001_20260912"
            and raw_result.get("job_dir") == str(raw_root / "control/jobs" / job_id)
            and isinstance(raw_result.get("argv"), list)
            and bool(raw_result["argv"]),
            f"{job_id} raw queue result header changed",
        )
        for key in (
            "job_id", "worker_id", "source_commit", "descriptor_sha256", "status",
            "returncode", "error_type", "started_at", "ended_at", "wall_seconds",
            "child_pid", "child_reaped", "stdout", "stderr",
        ):
            require(raw_result.get(key) == snapshot.get(key), f"{job_id} raw/snapshot field differs: {key}")
        queue_descriptor_path = raw_root / "control/jobs" / job_id / "descriptor.json"
        queue_descriptor = file_descriptor(queue_descriptor_path)
        require(queue_descriptor["sha256"] == descriptor_sha, f"{job_id} queue descriptor hash changed")
        output[job_id] = {
            "published_snapshot": snapshot_descriptor,
            "raw_queue_result": raw_result_descriptor,
            "queue_descriptor": queue_descriptor,
            "worker_id": worker_id,
            "source_commit": source_commit,
            "started_at": snapshot["started_at"],
            "ended_at": snapshot["ended_at"],
            "wall_seconds": wall,
            "child_reaped": True,
        }
    return output


def _validate_cuda_memory(value: Any, label: str) -> dict[str, int | bool]:
    require(isinstance(value, Mapping), f"{label} CUDA memory record is missing")
    required = {
        "cuda_available",
        "allocated_bytes_before",
        "reserved_bytes_before",
        "allocated_bytes_after",
        "reserved_bytes_after",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    require(set(value) == required, f"{label} CUDA memory fields changed")
    require(value.get("cuda_available") is True, f"{label} CUDA was unavailable")
    output: dict[str, int | bool] = {"cuda_available": True}
    for key in sorted(required - {"cuda_available"}):
        observed = value.get(key)
        require(type(observed) is int and observed >= 0, f"{label} {key} is invalid")
        output[key] = observed
    require(
        int(output["reserved_bytes_before"]) >= int(output["allocated_bytes_before"])
        and int(output["reserved_bytes_after"]) >= int(output["allocated_bytes_after"])
        and int(output["peak_allocated_bytes"]) >= max(
            int(output["allocated_bytes_before"]), int(output["allocated_bytes_after"])
        )
        and int(output["peak_reserved_bytes"]) >= max(
            int(output["reserved_bytes_before"]), int(output["reserved_bytes_after"]),
            int(output["peak_allocated_bytes"]),
        ),
        f"{label} CUDA allocator ordering changed",
    )
    return output


def _measure_d1_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
) -> dict[str, Any]:
    rank0_total: list[float] = []
    rank_proxy_total: list[float] = []
    decode_total: list[float] = []
    output_bytes_total = 0
    rank_peaks: dict[int, dict[str, int]] = {
        rank: {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        for rank in (0, 1)
    }
    decode_peak = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    for request_index, request in enumerate(requests):
        label = f"{cell_id} D1 request {request_index}"
        cost = request.get("cost")
        require(isinstance(cost, Mapping), f"{label} cost is missing")
        require(set(cost) == {
            "rank_count",
            "inference_wall_seconds_rank0_wrapper",
            "summed_rank_forward_gpu_seconds_proxy",
            "offline_decode_wall_seconds",
            "retained_output_bytes",
            "definition",
        }, f"{label} cost fields changed")
        require(cost.get("rank_count") == 2, f"{label} rank count changed")
        require(
            cost.get("definition") == D1_SOURCE_COST_DEFINITION,
            f"{label} cost definition changed",
        )
        rank0 = _finite_number(
            cost.get("inference_wall_seconds_rank0_wrapper"),
            f"{label} rank0 wrapper seconds", positive=True,
        )
        rank_proxy = _finite_number(
            cost.get("summed_rank_forward_gpu_seconds_proxy"),
            f"{label} rank proxy seconds", positive=True,
        )
        decode = _finite_number(
            cost.get("offline_decode_wall_seconds"),
            f"{label} decode seconds", positive=True,
        )
        retained = cost.get("retained_output_bytes")
        require(type(retained) is int and retained > 0, f"{label} retained bytes are invalid")
        metrics = request.get("temporal_and_cache_rank_metrics")
        require(
            isinstance(metrics, list)
            and len(metrics) == 2
            and sorted(row.get("rank") for row in metrics if isinstance(row, Mapping)) == [0, 1],
            f"{label} rank metrics changed",
        )
        metric_seconds: list[float] = []
        rank0_forward: float | None = None
        for row in metrics:
            require(isinstance(row, Mapping), f"{label} rank metric is invalid")
            rank = row.get("rank")
            seconds = _finite_number(row.get("wall_seconds"), f"{label} rank {rank} wall", positive=True)
            memory = _validate_cuda_memory(row.get("cuda_memory"), f"{label} rank {rank}")
            metric_seconds.append(seconds)
            if rank == 0:
                rank0_forward = seconds
            rank_peaks[int(rank)]["peak_allocated_bytes"] = max(
                rank_peaks[int(rank)]["peak_allocated_bytes"], int(memory["peak_allocated_bytes"])
            )
            rank_peaks[int(rank)]["peak_reserved_bytes"] = max(
                rank_peaks[int(rank)]["peak_reserved_bytes"], int(memory["peak_reserved_bytes"])
            )
        require(
            math.isclose(rank_proxy, math.fsum(metric_seconds), rel_tol=1e-12, abs_tol=1e-12),
            f"{label} rank proxy differs from rank metrics",
        )
        require(rank0_forward is not None and rank0 >= rank0_forward, f"{label} rank0 wrapper is shorter than forward")
        offline = request.get("offline_decode")
        require(
            isinstance(offline, Mapping)
            and offline.get("requested") is True
            and offline.get("performed") is True,
            f"{label} offline decode was not performed",
        )
        offline_seconds = _finite_number(
            offline.get("wall_seconds"), f"{label} offline decode wall", positive=True
        )
        require(
            math.isclose(decode, offline_seconds, rel_tol=0.0, abs_tol=0.0),
            f"{label} cost/decode wall differs",
        )
        offline_memory = _validate_cuda_memory(offline.get("cuda_memory"), f"{label} decode")
        decode_peak["peak_allocated_bytes"] = max(
            decode_peak["peak_allocated_bytes"], int(offline_memory["peak_allocated_bytes"])
        )
        decode_peak["peak_reserved_bytes"] = max(
            decode_peak["peak_reserved_bytes"], int(offline_memory["peak_reserved_bytes"])
        )
        expected_retained = 0
        for artifact in (
            request.get("official_returned_action"),
            request.get("latent_video"),
            offline.get("decoded_tensor"),
            offline.get("decoded_rgb"),
        ):
            require(isinstance(artifact, Mapping), f"{label} retained artifact is missing")
            size = artifact.get("bytes")
            require(type(size) is int and size > 0, f"{label} retained artifact bytes are invalid")
            expected_retained += size
        require(retained == expected_retained, f"{label} retained output byte sum changed")
        rank0_total.append(rank0)
        rank_proxy_total.append(rank_proxy)
        decode_total.append(decode)
        output_bytes_total += retained
    rank_rows = [
        {"rank": rank, **rank_peaks[rank]}
        for rank in (0, 1)
    ]
    return {
        "rank0_inference_wrapper_seconds_total": math.fsum(rank0_total),
        "summed_rank_forward_gpu_seconds_proxy_total": math.fsum(rank_proxy_total),
        "offline_decode_wall_seconds_total": math.fsum(decode_total),
        "retained_output_bytes_total": output_bytes_total,
        "rank_forward_allocator_peaks": rank_rows,
        "rank_forward_max_device_peak_allocated_bytes": max(
            row["peak_allocated_bytes"] for row in rank_rows
        ),
        "rank_forward_max_device_peak_reserved_bytes": max(
            row["peak_reserved_bytes"] for row in rank_rows
        ),
        "rank_forward_sum_of_device_maxima_allocated_bytes_upper_bound": sum(
            row["peak_allocated_bytes"] for row in rank_rows
        ),
        "rank_forward_sum_of_device_maxima_reserved_bytes_upper_bound": sum(
            row["peak_reserved_bytes"] for row in rank_rows
        ),
        "offline_decode_rank0_peak_allocated_bytes": decode_peak["peak_allocated_bytes"],
        "offline_decode_rank0_peak_reserved_bytes": decode_peak["peak_reserved_bytes"],
    }


def measure_cell(
    cell: Mapping[str, Any],
    *,
    evidence: ModuleType,
) -> dict[str, Any]:
    source = cell.get("cell")
    require(isinstance(source, Mapping), "compiled cell payload is missing")
    cell_id = source.get("cell_id")
    model = source.get("model_config")
    layout = source.get("layout_pair_id")
    require(isinstance(cell_id, str) and cell_id, "cell ID is missing")
    require(model in MODELS and layout in LAYOUTS, f"{cell_id} model/layout changed")
    journal_descriptor = cell.get("journal_descriptor")
    completion_descriptor = cell.get("completion_descriptor")
    require(isinstance(journal_descriptor, Mapping), f"{cell_id} journal descriptor is missing")
    require(isinstance(completion_descriptor, Mapping), f"{cell_id} completion descriptor is missing")
    rows, tail = evidence._verify_journal(Path(str(journal_descriptor["path"])))
    require(
        len(rows) >= 2
        and rows[0].get("kind") == "attempt_started"
        and rows[-1].get("kind") == "attempt_finalized"
        and journal_descriptor.get("tail_sha256") == tail,
        f"{cell_id} journal attempt boundary changed",
    )
    start_ns = rows[0].get("monotonic_ns")
    end_ns = rows[-1].get("monotonic_ns")
    require(
        type(start_ns) is int and type(end_ns) is int and end_ns > start_ns >= 0,
        f"{cell_id} journal monotonic bounds are invalid",
    )
    sent = _event_payloads(rows, "model_request_sent")
    received = _event_payloads(rows, "model_response_received")
    expected_requests = MODEL_REQUESTS_PER_CELL[str(model)]
    require(
        len(sent) == len(received) == expected_requests,
        f"{cell_id} request event coverage changed",
    )
    roundtrip_ns = 0
    for request_index, (send, receive) in enumerate(zip(sent, received)):
        send_ns = send.get("send_monotonic_ns")
        receive_ns = receive.get("receive_monotonic_ns")
        require(
            send.get("request_index") == receive.get("request_index") == request_index
            and type(send_ns) is int
            and type(receive_ns) is int
            and receive_ns >= send_ns >= 0,
            f"{cell_id} request {request_index} round-trip clocks changed",
        )
        roundtrip_ns += receive_ns - send_ns
    request_descriptors = cell.get("request_descriptors")
    require(
        isinstance(request_descriptors, list)
        and len(request_descriptors) == expected_requests,
        f"{cell_id} official request descriptor count changed",
    )
    request_values = [
        load_json(Path(str(descriptor["path"])), f"{cell_id} request {index}")
        for index, descriptor in enumerate(request_descriptors)
    ]
    if model == "N3":
        service_ns = 0
        for request_index, request in enumerate(request_values):
            start = request.get("started_monotonic_ns")
            end = request.get("completed_monotonic_ns")
            require(
                type(start) is int and type(end) is int and end >= start >= 0,
                f"{cell_id} N3 request {request_index} service clocks changed",
            )
            service_ns += end - start
        inference = {
            "comparable_policy_request_roundtrip": {
                "status": "measured",
                "definition": REQUEST_ROUNDTRIP_DEFINITION,
                "seconds_total": roundtrip_ns / 1_000_000_000,
                "nanoseconds_total": roundtrip_ns,
                "request_count": expected_requests,
                "pure_gpu_forward_time": False,
            },
            "n3_server_request_service": {
                "status": "measured",
                "definition": N3_SERVICE_DEFINITION,
                "seconds_total": service_ns / 1_000_000_000,
                "nanoseconds_total": service_ns,
                "request_count": expected_requests,
                "pure_gpu_forward_time": False,
            },
            "pure_gpu_forward": {
                "status": "missing_not_instrumented",
                "seconds_total": None,
            },
        }
        gpu = {
            "definition": GPU_PEAK_DEFINITION,
            "model_server_forward": {
                "status": "missing_not_instrumented_in_n3_behavioral_runtime",
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            },
            "simulator_process": {
                "status": "missing_not_instrumented",
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            },
            "all_development_gpu_processes_peak": {
                "status": "missing_not_measured",
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            },
            "measurement_complete_for_resource_release": False,
        }
    else:
        d1 = _measure_d1_requests(request_values, cell_id=cell_id)
        inference = {
            "comparable_policy_request_roundtrip": {
                "status": "measured",
                "definition": REQUEST_ROUNDTRIP_DEFINITION,
                "seconds_total": roundtrip_ns / 1_000_000_000,
                "nanoseconds_total": roundtrip_ns,
                "request_count": expected_requests,
                "pure_gpu_forward_time": False,
            },
            "d1_rank0_inference_wrapper": {
                "status": "measured",
                "definition": D1_RANK0_DEFINITION,
                "seconds_total": d1["rank0_inference_wrapper_seconds_total"],
            },
            "d1_summed_rank_forward_gpu_seconds_proxy": {
                "status": "measured_proxy_not_billing",
                "definition": D1_RANK_PROXY_DEFINITION,
                "seconds_total": d1["summed_rank_forward_gpu_seconds_proxy_total"],
            },
            "d1_offline_decode": {
                "status": "measured",
                "definition": D1_DECODE_DEFINITION,
                "seconds_total": d1["offline_decode_wall_seconds_total"],
            },
            "d1_retained_model_output": {
                "status": "measured_partial_storage_scope",
                "definition": D1_OUTPUT_BYTES_DEFINITION,
                "bytes_total": d1["retained_output_bytes_total"],
            },
        }
        gpu = {
            "definition": GPU_PEAK_DEFINITION,
            "model_server_rank_forward": {
                "status": "measured_per_rank_per_request",
                "rank_peaks": d1["rank_forward_allocator_peaks"],
                "max_device_peak_allocated_bytes": d1[
                    "rank_forward_max_device_peak_allocated_bytes"
                ],
                "max_device_peak_reserved_bytes": d1[
                    "rank_forward_max_device_peak_reserved_bytes"
                ],
                "sum_of_device_maxima_allocated_bytes_upper_bound": d1[
                    "rank_forward_sum_of_device_maxima_allocated_bytes_upper_bound"
                ],
                "sum_of_device_maxima_reserved_bytes_upper_bound": d1[
                    "rank_forward_sum_of_device_maxima_reserved_bytes_upper_bound"
                ],
                "simultaneous_cross_device_peak_measured": False,
            },
            "offline_decode_rank0": {
                "status": "measured_per_request",
                "peak_allocated_bytes": d1["offline_decode_rank0_peak_allocated_bytes"],
                "peak_reserved_bytes": d1["offline_decode_rank0_peak_reserved_bytes"],
            },
            "simulator_process": {
                "status": "missing_not_instrumented",
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            },
            "all_development_gpu_processes_peak": {
                "status": "missing_no_simultaneous_cross_process_measurement",
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            },
            "measurement_complete_for_resource_release": False,
        }
    return {
        "cell_id": cell_id,
        "model_id": model,
        "layout_pair_id": layout,
        "source_cell_receipt": dict(cell["cell_descriptor"]),
        "adapter_completion": dict(completion_descriptor),
        "adapter_journal": dict(journal_descriptor),
        "official_request_receipts": [dict(row) for row in request_descriptors],
        "episode_wall": {
            "status": "measured",
            "definition": EPISODE_WALL_DEFINITION,
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "elapsed_nanoseconds": end_ns - start_ns,
            "elapsed_seconds": (end_ns - start_ns) / 1_000_000_000,
        },
        "inference_and_request_time": inference,
        "gpu_memory": gpu,
    }


def _scan_tree(root: Path, raw_root: Path, label: str) -> dict[Path, os.stat_result]:
    resolved_root = under(root, raw_root, label)
    require(resolved_root.is_dir(), f"{label} is not a directory")
    output: dict[Path, os.stat_result] = {}
    pending = [resolved_root]
    while pending:
        directory = pending.pop()
        require(not directory.is_symlink(), f"{label} contains a symlink directory: {directory}")
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ResourceCompilerError(f"cannot scan {label}: {directory}") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ResourceCompilerError(f"cannot stat {label}: {path}") from error
            require(not stat.S_ISLNK(info.st_mode), f"{label} contains a symlink: {path}")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            else:
                require(stat.S_ISREG(info.st_mode), f"{label} contains a non-regular file: {path}")
                require(path not in output, f"{label} duplicates a path: {path}")
                output[path] = info
    return output


def _hash_stable_regular(path: Path, expected: os.stat_result, label: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    count = 0
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            require(stat.S_ISREG(before.st_mode), f"{label} ceased to be regular")
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
                count += len(block)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ResourceCompilerError(f"cannot read {label}: {path}") from error
    wanted = (expected.st_dev, expected.st_ino, expected.st_size, expected.st_mtime_ns)
    require(
        wanted == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and count == expected.st_size,
        f"{label} changed while hashing: {path}",
    )
    return {
        "path": str(path),
        "bytes": count,
        "sha256": digest.hexdigest(),
        "storage_device": int(expected.st_dev),
        "storage_inode": int(expected.st_ino),
    }


def _root_owner(
    path: Path,
    roots: Sequence[tuple[Path, dict[str, Any]]],
    label: str,
) -> dict[str, Any]:
    matches = [(root, owner) for root, owner in roots if path.is_relative_to(root)]
    require(len(matches) == 1, f"{label} has {len(matches)} owning roots: {path}")
    return matches[0][1]


def build_file_inventory(
    *,
    raw_root: Path,
    aggregate_receipts: Mapping[tuple[str, str], Mapping[str, Any]],
    d1_servers: Mapping[str, Mapping[str, Any]],
    compiled_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    block_roots: list[tuple[Path, dict[str, Any]]] = []
    for model in MODELS:
        for layout in LAYOUTS:
            receipt = aggregate_receipts[(model, layout)]
            path = under(Path(str(receipt["raw_attempt_root"])), raw_root, f"{model} {layout} raw attempt")
            block_roots.append((path, {
                "model_id": model,
                "layout_pair_id": layout,
                "block_component": "n3_joint_attempt" if model == "N3" else "d1_simulator_attempt",
            }))
    for layout in LAYOUTS:
        path = under(
            Path(str(d1_servers[layout]["raw_attempt_root"])),
            raw_root,
            f"D1 {layout} server raw attempt",
        )
        block_roots.append((path, {
            "model_id": "D1",
            "layout_pair_id": layout,
            "block_component": "d1_server_attempt",
        }))
    require(len({root for root, _ in block_roots}) == 12, "raw block roots overlap by identity")
    for left_index, (left, _) in enumerate(block_roots):
        for right, _ in block_roots[left_index + 1:]:
            require(
                not left.is_relative_to(right) and not right.is_relative_to(left),
                "raw block roots are nested",
            )

    cell_roots: list[tuple[Path, dict[str, Any]]] = []
    request_roots: list[tuple[Path, dict[str, Any]]] = []
    required_paths: set[Path] = set()
    for model in MODELS:
        for cell in compiled_by_model[model]:
            cell_id = str(cell["cell"]["cell_id"])
            cell_root = Path(str(cell["cell_descriptor"]["path"])).parent.resolve()
            owner = {
                "model_id": model,
                "layout_pair_id": str(cell["cell"]["layout_pair_id"]),
                "cell_id": cell_id,
            }
            cell_roots.append((cell_root, owner))
            for key in ("cell_descriptor", "completion_descriptor", "journal_descriptor"):
                required_paths.add(Path(str(cell[key]["path"])).resolve())
            for descriptor in cell["request_descriptors"]:
                request_path = Path(str(descriptor["path"])).resolve()
                required_paths.add(request_path)
                request_roots.append((request_path.parent, owner))
    require(len(cell_roots) == EXPECTED_CELLS, "cell root inventory is incomplete")
    require(len(request_roots) == EXPECTED_REQUESTS, "request root inventory is incomplete")
    require(len({root for root, _ in request_roots}) == EXPECTED_REQUESTS, "request roots are duplicated")

    initial: dict[Path, os.stat_result] = {}
    for root, owner in block_roots:
        label = f"{owner['model_id']} {owner['layout_pair_id']} {owner['block_component']}"
        scanned = _scan_tree(root, raw_root, label)
        require(not set(initial).intersection(scanned), f"{label} overlaps another block tree")
        initial.update(scanned)
    require(required_paths <= set(initial), "required cell/request evidence escaped raw block inventory")

    rows: list[dict[str, Any]] = []
    per_cell: dict[str, dict[str, int]] = {
        str(cell["cell"]["cell_id"]): {
            "recorder_cell_tree_bytes": 0,
            "model_request_tree_bytes": 0,
            "attributable_deduplicated_bytes": 0,
            "recorder_cell_tree_file_count": 0,
            "model_request_tree_file_count": 0,
        }
        for model in MODELS for cell in compiled_by_model[model]
    }
    seen_storage: set[tuple[int, int]] = set()
    unique_storage_bytes = 0
    for path in sorted(initial, key=str):
        block_owner = _root_owner(path, block_roots, "raw file")
        request_matches = [(root, owner) for root, owner in request_roots if path.is_relative_to(root)]
        cell_matches = [(root, owner) for root, owner in cell_roots if path.is_relative_to(root)]
        require(len(request_matches) <= 1 and len(cell_matches) <= 1, "raw file has ambiguous cell/request ownership")
        if request_matches:
            owner = request_matches[0][1]
            scope = "model_request_tree"
        elif cell_matches:
            owner = cell_matches[0][1]
            scope = "recorder_cell_tree"
        else:
            owner = block_owner
            scope = "shared_block_overhead"
        identity = _hash_stable_regular(path, initial[path], "raw evidence file")
        storage_key = (int(identity.pop("storage_device")), int(identity.pop("storage_inode")))
        counted = storage_key not in seen_storage
        if counted:
            seen_storage.add(storage_key)
            unique_storage_bytes += identity["bytes"]
        row = {
            **identity,
            "scope": scope,
            "model_id": owner["model_id"],
            "layout_pair_id": owner["layout_pair_id"],
            "cell_id": owner.get("cell_id"),
            "block_component": block_owner["block_component"],
            "counted_once_by_storage_object": counted,
        }
        rows.append(row)
        if scope in {"recorder_cell_tree", "model_request_tree"}:
            cell_id = str(owner["cell_id"])
            per_cell[cell_id][scope + "_bytes"] += identity["bytes"]
            per_cell[cell_id][scope + "_file_count"] += 1
            per_cell[cell_id]["attributable_deduplicated_bytes"] += identity["bytes"]

    repeated: dict[Path, os.stat_result] = {}
    for root, owner in block_roots:
        label = f"recheck {owner['model_id']} {owner['layout_pair_id']} {owner['block_component']}"
        repeated.update(_scan_tree(root, raw_root, label))
    require(set(repeated) == set(initial), "raw file inventory changed during audit")
    for path in initial:
        before = initial[path]
        after = repeated[path]
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"raw file metadata changed during audit: {path}",
        )
    scope_totals: dict[str, dict[str, int]] = {}
    for scope in ("recorder_cell_tree", "model_request_tree", "shared_block_overhead"):
        selected = [row for row in rows if row["scope"] == scope]
        scope_totals[scope] = {
            "file_count": len(selected),
            "unique_path_bytes": sum(int(row["bytes"]) for row in selected),
        }
    inventory = sign_document({
        "schema_version": FILE_INVENTORY_SCHEMA,
        "study_id": STUDY_ID,
        "status": "complete_immutable_snapshot",
        "definition": {
            "path_deduplication": "resolved_absolute_regular_file_path",
            "storage_object_deduplication": "st_dev_and_st_ino_within_this_single_pvc_scan",
            "mutation_check": "pre_scan_and_post_scan_metadata_plus_open_fd_hash_stability",
            "symlinks": "rejected",
            "non_regular_files": "rejected",
            "scope_priority": ["model_request_tree", "recorder_cell_tree", "shared_block_overhead"],
        },
        "raw_root": str(raw_root),
        "block_roots": [
            {"path": str(root), **owner} for root, owner in block_roots
        ],
        "file_count": len(rows),
        "unique_resolved_path_bytes": sum(int(row["bytes"]) for row in rows),
        "unique_storage_object_count": len(seen_storage),
        "unique_storage_object_bytes": unique_storage_bytes,
        "scope_totals": scope_totals,
        "files": rows,
    })
    return inventory, per_cell


def _missing_annotation() -> dict[str, Any]:
    return {
        "status": "missing_no_authenticated_completed_rater_responses",
        "definition": "sum_of_per_image_annotation_seconds_from_locked_rater_responses",
        "rater_a_total_seconds": None,
        "rater_b_total_seconds": None,
        "adjudication_total_seconds": None,
        "response_session_wall_span_used_as_substitute": False,
        "measurement_complete_for_resource_release": False,
    }


def _attach_storage(
    measurement: Mapping[str, Any], storage: Mapping[str, int]
) -> dict[str, Any]:
    output = dict(measurement)
    output["raw_storage"] = {
        "status": "measured",
        "attributable_bytes_definition": RAW_ATTRIBUTABLE_DEFINITION,
        **dict(storage),
        "shared_block_overhead_allocated_to_cell": False,
    }
    output["annotation_time"] = _missing_annotation()
    output["measurement_completeness"] = {
        "episode_wall": True,
        "request_roundtrip": True,
        "raw_attributable_storage": True,
        "all_gpu_process_peak_memory": False,
        "annotation_time": False,
        "adjudication_time": False,
        "selected_confirmation_host_and_parallelism": False,
    }
    output["status"] = "measured_with_declared_missingness"
    output["safe_for_legacy_resource_release_gate"] = False
    return sign_document({"schema_version": CELL_SCHEMA, "study_id": STUDY_ID, **output})


def _sum_nested(
    cells: Sequence[Mapping[str, Any]], path: Sequence[str]
) -> float:
    values: list[float] = []
    for cell in cells:
        value: Any = cell
        for key in path:
            require(isinstance(value, Mapping) and key in value, f"aggregate metric path is missing: {path}")
            value = value[key]
        values.append(_finite_number(value, f"aggregate metric {'.'.join(path)}", nonnegative=True))
    return math.fsum(values)


def _aggregate_models(
    cells: Sequence[Mapping[str, Any]],
    queue_windows: Mapping[str, Mapping[str, Any]],
    aggregate_receipts: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODELS:
        selected = [cell for cell in cells if cell["model_id"] == model]
        require(len(selected) == 16, f"{model} aggregate cell count changed")
        job_ids = [
            str(aggregate_receipts[(model, layout)]["queue_descriptor"]["job_id"])
            for layout in LAYOUTS
        ]
        model_value: dict[str, Any] = {
            "cell_count": 16,
            "request_count": MODEL_REQUEST_COUNTS[model],
            "behavioral_action_count": 7200,
            "episode_wall_seconds_total": _sum_nested(selected, ("episode_wall", "elapsed_seconds")),
            "policy_request_roundtrip_seconds_total": _sum_nested(
                selected,
                ("inference_and_request_time", "comparable_policy_request_roundtrip", "seconds_total"),
            ),
            "attributable_raw_bytes_total": sum(
                int(cell["raw_storage"]["attributable_deduplicated_bytes"])
                for cell in selected
            ),
            "queue_job_wall_seconds_total_nonoverlap_only_within_this_model_role": math.fsum(
                float(queue_windows[job_id]["wall_seconds"]) for job_id in job_ids
            ),
            "resource_release_complete": False,
        }
        if model == "N3":
            model_value["n3_server_request_service_seconds_total"] = _sum_nested(
                selected, ("inference_and_request_time", "n3_server_request_service", "seconds_total")
            )
            model_value["gpu_memory_status"] = "missing_behavioral_model_and_simulator_peaks"
        else:
            for output_key, source_key in (
                ("d1_rank0_inference_wrapper_seconds_total", "d1_rank0_inference_wrapper"),
                ("d1_summed_rank_forward_gpu_seconds_proxy_total", "d1_summed_rank_forward_gpu_seconds_proxy"),
                ("d1_offline_decode_wall_seconds_total", "d1_offline_decode"),
            ):
                model_value[output_key] = _sum_nested(
                    selected, ("inference_and_request_time", source_key, "seconds_total")
                )
            model_value["d1_retained_model_output_bytes_total"] = sum(
                int(cell["inference_and_request_time"]["d1_retained_model_output"]["bytes_total"])
                for cell in selected
            )
            model_value["d1_model_server_max_device_peak_allocated_bytes"] = max(
                int(cell["gpu_memory"]["model_server_rank_forward"]["max_device_peak_allocated_bytes"])
                for cell in selected
            )
            model_value["d1_model_server_max_device_peak_reserved_bytes"] = max(
                int(cell["gpu_memory"]["model_server_rank_forward"]["max_device_peak_reserved_bytes"])
                for cell in selected
            )
            model_value["gpu_memory_status"] = "model_server_allocator_peaks_measured_simulator_and_all_process_peak_missing"
        output[model] = model_value
    return output


def compile_resources(
    manifest_path: Path,
    manifest_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_contract()
    manifest, manifest_path = _validate_manifest(manifest_path, manifest_sha256)
    raw_root = RAW_ROOT
    output_dir = Path(output_dir)
    require(not output_dir.exists() and not output_dir.is_symlink(), "refusing to overwrite resource output")
    output_parent = under(output_dir.parent, raw_root, "resource output parent")
    require(output_parent.is_dir(), "resource output parent is unavailable")
    require(
        not output_dir.is_relative_to(raw_root / "behavioral/development"),
        "resource output may not be inside inventoried behavioral evidence",
    )

    evidence = _load_module(
        EVIDENCE_COMPILER_PATH, "wmf_resource_deep_evidence_validator"
    )
    timing = _load_module(
        TIMING_VALIDATOR_PATH, "wmf_resource_timing_validator"
    )
    require(
        getattr(evidence, "STUDY_ID", None) == STUDY_ID
        and getattr(evidence, "MODELS", None) == MODELS
        and getattr(evidence, "LAYOUT_IDS", None) == LAYOUTS,
        "development evidence compiler interface changed",
    )
    require(getattr(timing, "STUDY_ID", None) == STUDY_ID, "timing validator study changed")
    aggregates, compiled, aggregate_receipts = _load_and_validate_aggregates(
        manifest=manifest,
        manifest_path=manifest_path,
        raw_root=raw_root,
        evidence=evidence,
    )
    compiled_by_model = _compiled_cells_by_model(compiled)
    timing_receipts = _validate_timing_receipts(
        manifest=manifest,
        manifest_path=manifest_path,
        raw_root=raw_root,
        aggregates=aggregates,
        compiled_by_model=compiled_by_model,
        timing=timing,
    )
    d1_servers = _validate_d1_server_receipts(
        manifest=manifest,
        manifest_path=manifest_path,
        raw_root=raw_root,
        aggregate_receipts=aggregate_receipts,
    )
    topologies = _validate_topologies(
        raw_root=raw_root,
        aggregate_receipts=aggregate_receipts,
        d1_servers=d1_servers,
    )
    queue_windows = _validate_queue_snapshots(
        manifest=manifest,
        manifest_path=manifest_path,
        raw_root=raw_root,
        aggregate_receipts=aggregate_receipts,
        d1_servers=d1_servers,
        timing_receipts=timing_receipts,
    )
    measured = [
        measure_cell(cell, evidence=evidence)
        for model in MODELS for cell in compiled_by_model[model]
    ]
    file_inventory, per_cell_storage = build_file_inventory(
        raw_root=raw_root,
        aggregate_receipts=aggregate_receipts,
        d1_servers=d1_servers,
        compiled_by_model=compiled_by_model,
    )
    complete_cells = [
        _attach_storage(row, per_cell_storage[str(row["cell_id"])])
        for row in measured
    ]

    files: dict[str, bytes] = {}
    cell_descriptors: list[dict[str, Any]] = []
    for cell in complete_cells:
        relative = f"cells/{str(cell['model_id']).lower()}/{cell['cell_id']}.json"
        payload = pretty_bytes(cell)
        files[relative] = payload
        cell_descriptors.append({
            "model_id": cell["model_id"],
            "layout_pair_id": cell["layout_pair_id"],
            "cell_id": cell["cell_id"],
            "measurement": bytes_descriptor(relative, payload),
        })
    inventory_payload = pretty_bytes(file_inventory)
    files["resource_file_inventory.json"] = inventory_payload
    inventory_descriptor = bytes_descriptor("resource_file_inventory.json", inventory_payload)
    aggregate = sign_document({
        "schema_version": AGGREGATE_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "status": "complete_resource_audit_with_declared_missingness",
        "formal_development_cohort_authenticated": True,
        "resource_release_gate_complete": False,
        "safe_to_release_confirmation": False,
        "counts": {
            "cells": EXPECTED_CELLS,
            "source_behavioral_requests": EXPECTED_REQUESTS,
            "source_behavioral_actions": EXPECTED_ACTIONS,
            "new_model_requests": 0,
            "new_behavioral_actions": 0,
        },
        "definitions": {
            "episode_wall": EPISODE_WALL_DEFINITION,
            "request_roundtrip": REQUEST_ROUNDTRIP_DEFINITION,
            "n3_server_service": N3_SERVICE_DEFINITION,
            "d1_rank0_wrapper": D1_RANK0_DEFINITION,
            "d1_summed_rank_proxy": D1_RANK_PROXY_DEFINITION,
            "d1_offline_decode": D1_DECODE_DEFINITION,
            "raw_attributable_bytes": RAW_ATTRIBUTABLE_DEFINITION,
            "gpu_allocator_peak": GPU_PEAK_DEFINITION,
        },
        "inputs": {
            "manifest": file_descriptor(manifest_path),
            "timing_sidecars": timing_receipts,
            "aggregate_receipts": [
                {"model_id": model, "layout_pair_id": layout, "receipt": aggregates[(model, layout)]}
                for model in MODELS for layout in LAYOUTS
            ],
            "d1_server_receipts": [
                {"layout_pair_id": layout, "receipt": d1_servers[layout]["receipt"]}
                for layout in LAYOUTS
            ],
            "topologies": topologies,
            "queue_job_windows": queue_windows,
        },
        "cell_measurements": cell_descriptors,
        "resource_file_inventory": inventory_descriptor,
        "models": _aggregate_models(complete_cells, queue_windows, aggregate_receipts),
        "annotation_time": _missing_annotation(),
        "operator_release_fields": EXECUTION_AUTHORIZATION,
        "missing_release_requirements": list(MISSING_RELEASE_REQUIREMENTS),
        "science_activity": SCIENCE_COUNTS,
        "claim_boundary": (
            "Authenticated resource derivation over retained completed development evidence only. "
            "Declared missing fields are not zeros, and this document cannot release confirmation."
        ),
    })
    aggregate_payload = pretty_bytes(aggregate)
    files["resource_aggregate.json"] = aggregate_payload
    aggregate_descriptor = bytes_descriptor("resource_aggregate.json", aggregate_payload)
    compiler_receipt = sign_document({
        "schema_version": COMPILER_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "status": "compiled_complete_with_declared_missingness",
        "input_manifest": file_descriptor(manifest_path),
        "resource_compiler": {
            "path": "workshops/corl2026_world_models/analysis/compile_development_resources.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "bytes": Path(__file__).resolve().stat().st_size,
        },
        "resource_contract": {
            "path": "workshops/corl2026_world_models/experiments/forecast_layout/development_resource_contract.json",
            "sha256": sha256_file(CONTRACT_PATH.resolve()),
            "bytes": CONTRACT_PATH.resolve().stat().st_size,
        },
        "outputs": {
            "resource_aggregate": aggregate_descriptor,
            "resource_file_inventory": inventory_descriptor,
            "cell_measurement_count": len(cell_descriptors),
            "cell_measurements": cell_descriptors,
        },
        "counts": {
            "authenticated_cells": EXPECTED_CELLS,
            "authenticated_source_behavioral_requests": EXPECTED_REQUESTS,
            "authenticated_source_behavioral_actions": EXPECTED_ACTIONS,
            "inventoried_raw_files": file_inventory["file_count"],
            "inventoried_unique_resolved_path_bytes": file_inventory["unique_resolved_path_bytes"],
        },
        "science_activity": SCIENCE_COUNTS,
        "resource_release_gate_complete": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "behavioral_policy_skill_evaluated": False,
        "claim_boundary": (
            "CPU-only measurement of immutable completed evidence. Missing resource and "
            "annotation dimensions remain explicit and confirmation stays held."
        ),
    })
    receipt_payload = pretty_bytes(compiler_receipt)
    files["resource_compiler_receipt.json"] = receipt_payload
    _write_atomic_directory(output_dir, files)
    validate_resource_bundle(output_dir)
    return compiler_receipt


def _write_atomic_directory(target: Path, files: Mapping[str, bytes]) -> None:
    require(not target.exists() and not target.is_symlink(), f"refusing to overwrite output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for relative, payload in sorted(files.items()):
            path = temporary / relative
            require(path.resolve().is_relative_to(temporary.resolve()), "output path escaped bundle")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_inventory_output(inventory: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "study_id",
        "status",
        "definition",
        "raw_root",
        "block_roots",
        "file_count",
        "unique_resolved_path_bytes",
        "unique_storage_object_count",
        "unique_storage_object_bytes",
        "scope_totals",
        "files",
        "payload_sha256",
    }
    require(set(inventory) == expected_keys, "resource file inventory fields changed")
    require(
        inventory.get("status") == "complete_immutable_snapshot"
        and inventory.get("raw_root") == str(RAW_ROOT),
        "resource file inventory boundary changed",
    )
    require(
        inventory.get("definition")
        == {
            "path_deduplication": "resolved_absolute_regular_file_path",
            "storage_object_deduplication": "st_dev_and_st_ino_within_this_single_pvc_scan",
            "mutation_check": "pre_scan_and_post_scan_metadata_plus_open_fd_hash_stability",
            "symlinks": "rejected",
            "non_regular_files": "rejected",
            "scope_priority": [
                "model_request_tree",
                "recorder_cell_tree",
                "shared_block_overhead",
            ],
        },
        "resource file inventory definition changed",
    )
    roots = inventory.get("block_roots")
    require(isinstance(roots, list) and len(roots) == 12, "resource block-root roster changed")
    expected_roots = {
        (model, layout, component)
        for model, components in (
            ("N3", ("n3_joint_attempt",)),
            ("D1", ("d1_simulator_attempt", "d1_server_attempt")),
        )
        for layout in LAYOUTS
        for component in components
    }
    observed_roots: set[tuple[str, str, str]] = set()
    root_paths: set[str] = set()
    for row in roots:
        require(isinstance(row, Mapping), "resource block-root row is invalid")
        require(
            set(row) == {"path", "model_id", "layout_pair_id", "block_component"},
            "resource block-root fields changed",
        )
        identity = (
            str(row.get("model_id")),
            str(row.get("layout_pair_id")),
            str(row.get("block_component")),
        )
        path = row.get("path")
        require(
            identity in expected_roots
            and identity not in observed_roots
            and isinstance(path, str)
            and Path(path).is_absolute()
            and Path(path).is_relative_to(RAW_ROOT)
            and path not in root_paths,
            "resource block-root identity changed",
        )
        observed_roots.add(identity)
        root_paths.add(path)
    require(observed_roots == expected_roots, "resource block-root coverage changed")

    files = inventory.get("files")
    file_count = inventory.get("file_count")
    require(
        isinstance(files, list)
        and type(file_count) is int
        and file_count == len(files)
        and file_count >= EXPECTED_REQUESTS + EXPECTED_CELLS * 3,
        "resource file inventory count changed",
    )
    scope_names = {
        "model_request_tree",
        "recorder_cell_tree",
        "shared_block_overhead",
    }
    paths_seen: set[str] = set()
    scope_totals = {
        scope: {"file_count": 0, "unique_path_bytes": 0}
        for scope in scope_names
    }
    counted_objects = 0
    counted_bytes = 0
    all_bytes = 0
    for row in files:
        require(isinstance(row, Mapping), "resource file row is invalid")
        require(
            set(row)
            == {
                "path",
                "bytes",
                "sha256",
                "scope",
                "model_id",
                "layout_pair_id",
                "cell_id",
                "block_component",
                "counted_once_by_storage_object",
            },
            "resource file row fields changed",
        )
        path = row.get("path")
        size = row.get("bytes")
        scope = row.get("scope")
        counted = row.get("counted_once_by_storage_object")
        require(
            isinstance(path, str)
            and path not in paths_seen
            and Path(path).is_absolute()
            and Path(path).is_relative_to(RAW_ROOT),
            "resource file path changed",
        )
        require(type(size) is int and size >= 0, "resource file byte count changed")
        _valid_sha(row.get("sha256"), "resource file digest")
        require(scope in scope_names and type(counted) is bool, "resource file scope changed")
        require(
            row.get("model_id") in MODELS and row.get("layout_pair_id") in LAYOUTS,
            "resource file model/layout changed",
        )
        if scope == "shared_block_overhead":
            require(row.get("cell_id") is None, "shared resource file acquired a cell")
        else:
            require(
                isinstance(row.get("cell_id"), str) and row.get("cell_id"),
                "attributable resource file lacks a cell",
            )
        paths_seen.add(path)
        scope_totals[str(scope)]["file_count"] += 1
        scope_totals[str(scope)]["unique_path_bytes"] += size
        all_bytes += size
        if counted:
            counted_objects += 1
            counted_bytes += size
    require(
        inventory.get("scope_totals") == scope_totals
        and inventory.get("unique_resolved_path_bytes") == all_bytes
        and inventory.get("unique_storage_object_count") == counted_objects
        and inventory.get("unique_storage_object_bytes") == counted_bytes
        and 0 < counted_objects <= file_count,
        "resource inventory totals changed",
    )


def _validate_cell_output(cell: Mapping[str, Any], *, model: str, cell_id: str) -> None:
    require(
        set(cell)
        == {
            "schema_version",
            "study_id",
            "cell_id",
            "model_id",
            "layout_pair_id",
            "source_cell_receipt",
            "adapter_completion",
            "adapter_journal",
            "official_request_receipts",
            "episode_wall",
            "inference_and_request_time",
            "gpu_memory",
            "raw_storage",
            "annotation_time",
            "measurement_completeness",
            "status",
            "safe_for_legacy_resource_release_gate",
            "payload_sha256",
        },
        f"{cell_id} resource measurement fields changed",
    )
    require(
        cell.get("cell_id") == cell_id
        and cell.get("model_id") == model
        and cell.get("layout_pair_id") in LAYOUTS,
        f"{cell_id} resource measurement identity changed",
    )
    requests = cell.get("official_request_receipts")
    require(
        isinstance(requests, list)
        and len(requests) == MODEL_REQUESTS_PER_CELL[model]
        and all(
            isinstance(row, Mapping)
            and set(row) >= {"path", "sha256", "bytes"}
            for row in requests
        ),
        f"{cell_id} source request roster changed",
    )
    episode = cell.get("episode_wall")
    require(
        isinstance(episode, Mapping)
        and episode.get("status") == "measured"
        and episode.get("definition") == EPISODE_WALL_DEFINITION
        and type(episode.get("start_monotonic_ns")) is int
        and type(episode.get("end_monotonic_ns")) is int
        and type(episode.get("elapsed_nanoseconds")) is int
        and episode["end_monotonic_ns"] - episode["start_monotonic_ns"]
        == episode["elapsed_nanoseconds"]
        and episode["elapsed_nanoseconds"] > 0
        and math.isclose(
            float(episode.get("elapsed_seconds")),
            episode["elapsed_nanoseconds"] / 1_000_000_000,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        f"{cell_id} episode-wall measurement changed",
    )
    inference = cell.get("inference_and_request_time")
    comparable = (
        inference.get("comparable_policy_request_roundtrip")
        if isinstance(inference, Mapping) else None
    )
    require(
        isinstance(comparable, Mapping)
        and comparable.get("status") == "measured"
        and comparable.get("definition") == REQUEST_ROUNDTRIP_DEFINITION
        and comparable.get("request_count") == MODEL_REQUESTS_PER_CELL[model]
        and comparable.get("pure_gpu_forward_time") is False
        and type(comparable.get("nanoseconds_total")) is int
        and comparable["nanoseconds_total"] >= 0
        and comparable.get("seconds_total")
        == comparable["nanoseconds_total"] / 1_000_000_000,
        f"{cell_id} request-roundtrip measurement changed",
    )
    memory = cell.get("gpu_memory")
    require(
        isinstance(memory, Mapping)
        and memory.get("definition") == GPU_PEAK_DEFINITION
        and memory.get("measurement_complete_for_resource_release") is False,
        f"{cell_id} GPU-memory boundary changed",
    )
    simulator = memory.get("simulator_process")
    all_process = memory.get("all_development_gpu_processes_peak")
    require(
        isinstance(simulator, Mapping)
        and simulator.get("peak_allocated_bytes") is None
        and simulator.get("peak_reserved_bytes") is None
        and isinstance(all_process, Mapping)
        and all_process.get("peak_allocated_bytes") is None
        and all_process.get("peak_reserved_bytes") is None,
        f"{cell_id} unavailable GPU peak was not null",
    )
    if model == "N3":
        service = inference.get("n3_server_request_service")
        server_memory = memory.get("model_server_forward")
        require(
            isinstance(service, Mapping)
            and service.get("definition") == N3_SERVICE_DEFINITION
            and service.get("request_count") == MODEL_REQUESTS_PER_CELL[model]
            and isinstance(server_memory, Mapping)
            and server_memory.get("peak_allocated_bytes") is None
            and server_memory.get("peak_reserved_bytes") is None,
            f"{cell_id} N3 resource boundary changed",
        )
    else:
        require(
            isinstance(inference.get("d1_rank0_inference_wrapper"), Mapping)
            and inference["d1_rank0_inference_wrapper"].get("definition")
            == D1_RANK0_DEFINITION
            and isinstance(
                inference.get("d1_summed_rank_forward_gpu_seconds_proxy"), Mapping
            )
            and inference["d1_summed_rank_forward_gpu_seconds_proxy"].get("definition")
            == D1_RANK_PROXY_DEFINITION
            and isinstance(memory.get("model_server_rank_forward"), Mapping)
            and memory["model_server_rank_forward"].get("simultaneous_cross_device_peak_measured")
            is False,
            f"{cell_id} D1 resource boundary changed",
        )
    require(
        cell.get("annotation_time") == _missing_annotation()
        and cell.get("measurement_completeness")
        == {
            "episode_wall": True,
            "request_roundtrip": True,
            "raw_attributable_storage": True,
            "all_gpu_process_peak_memory": False,
            "annotation_time": False,
            "adjudication_time": False,
            "selected_confirmation_host_and_parallelism": False,
        },
        f"{cell_id} missingness map changed",
    )


def validate_resource_bundle(bundle: Path) -> dict[str, Any]:
    supplied = Path(bundle)
    reject_symlink_components(supplied, "resource bundle")
    try:
        root = supplied.resolve(strict=True)
    except OSError as error:
        raise ResourceCompilerError("resource bundle is unavailable") from error
    require(root.is_dir(), "resource bundle is not a directory")
    paths: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ResourceCompilerError(
                f"cannot scan resource bundle: {directory}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ResourceCompilerError(
                    f"cannot stat resource bundle entry: {path}"
                ) from error
            require(
                not stat.S_ISLNK(info.st_mode),
                f"resource bundle contains a symlink: {path}",
            )
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            else:
                require(
                    stat.S_ISREG(info.st_mode),
                    f"resource bundle contains a non-regular file: {path}",
                )
                paths.append(path)
    paths.sort()
    expected = {
        "resource_compiler_receipt.json",
        "resource_aggregate.json",
        "resource_file_inventory.json",
    }
    expected.update(
        f"cells/{model.lower()}/wmf1__development__{layout}__{model}__{arm}__{command}.json"
        for model in MODELS
        for layout in LAYOUTS
        for arm in ("original", "reflected")
        for command in ("left", "right")
    )
    observed = {str(path.relative_to(root)) for path in paths}
    require(observed == expected, "resource bundle file inventory changed")
    receipt = load_json(root / "resource_compiler_receipt.json", "resource compiler receipt")
    aggregate = load_json(root / "resource_aggregate.json", "resource aggregate")
    inventory = load_json(root / "resource_file_inventory.json", "resource file inventory")
    for value, schema, label in (
        (receipt, COMPILER_RECEIPT_SCHEMA, "resource compiler receipt"),
        (aggregate, AGGREGATE_SCHEMA, "resource aggregate"),
        (inventory, FILE_INVENTORY_SCHEMA, "resource file inventory"),
    ):
        verify_signed(value, label)
        require(value.get("schema_version") == schema, f"{label} schema changed")
        require(value.get("study_id") == STUDY_ID, f"{label} study changed")
    _validate_inventory_output(inventory)
    require(
        set(receipt)
        == {
            "schema_version",
            "study_id",
            "mode",
            "status",
            "input_manifest",
            "resource_compiler",
            "resource_contract",
            "outputs",
            "counts",
            "science_activity",
            "resource_release_gate_complete",
            "safe_to_release_confirmation",
            "confirmation_released",
            "behavioral_policy_skill_evaluated",
            "claim_boundary",
            "payload_sha256",
        }
        and receipt.get("mode") == MODE,
        "resource compiler receipt fields changed",
    )
    require(
        set(aggregate)
        == {
            "schema_version",
            "study_id",
            "mode",
            "status",
            "formal_development_cohort_authenticated",
            "resource_release_gate_complete",
            "safe_to_release_confirmation",
            "counts",
            "definitions",
            "inputs",
            "cell_measurements",
            "resource_file_inventory",
            "models",
            "annotation_time",
            "operator_release_fields",
            "missing_release_requirements",
            "science_activity",
            "claim_boundary",
            "payload_sha256",
        }
        and aggregate.get("mode") == MODE
        and aggregate.get("formal_development_cohort_authenticated") is True,
        "resource aggregate fields changed",
    )
    require(
        receipt.get("status") == "compiled_complete_with_declared_missingness"
        and receipt.get("resource_release_gate_complete") is False
        and receipt.get("safe_to_release_confirmation") is False
        and receipt.get("confirmation_released") is False
        and receipt.get("science_activity") == SCIENCE_COUNTS,
        "resource compiler receipt release boundary changed",
    )
    require(
        aggregate.get("status") == "complete_resource_audit_with_declared_missingness"
        and aggregate.get("resource_release_gate_complete") is False
        and aggregate.get("safe_to_release_confirmation") is False
        and aggregate.get("science_activity") == SCIENCE_COUNTS,
        "resource aggregate release boundary changed",
    )
    require(
        aggregate.get("counts")
        == {
            "cells": EXPECTED_CELLS,
            "source_behavioral_requests": EXPECTED_REQUESTS,
            "source_behavioral_actions": EXPECTED_ACTIONS,
            "new_model_requests": 0,
            "new_behavioral_actions": 0,
        }
        and aggregate.get("definitions")
        == {
            "episode_wall": EPISODE_WALL_DEFINITION,
            "request_roundtrip": REQUEST_ROUNDTRIP_DEFINITION,
            "n3_server_service": N3_SERVICE_DEFINITION,
            "d1_rank0_wrapper": D1_RANK0_DEFINITION,
            "d1_summed_rank_proxy": D1_RANK_PROXY_DEFINITION,
            "d1_offline_decode": D1_DECODE_DEFINITION,
            "raw_attributable_bytes": RAW_ATTRIBUTABLE_DEFINITION,
            "gpu_allocator_peak": GPU_PEAK_DEFINITION,
        }
        and aggregate.get("annotation_time") == _missing_annotation()
        and aggregate.get("operator_release_fields") == EXECUTION_AUTHORIZATION
        and aggregate.get("missing_release_requirements")
        == list(MISSING_RELEASE_REQUIREMENTS),
        "resource aggregate definitions or missingness changed",
    )
    model_summaries = aggregate.get("models")
    require(
        isinstance(model_summaries, Mapping)
        and set(model_summaries) == set(MODELS),
        "resource model summaries changed",
    )
    for model in MODELS:
        summary = model_summaries[model]
        require(
            isinstance(summary, Mapping)
            and summary.get("cell_count") == 16
            and summary.get("request_count") == MODEL_REQUEST_COUNTS[model]
            and summary.get("behavioral_action_count") == 7200
            and summary.get("resource_release_complete") is False,
            f"{model} resource summary changed",
        )
    def local_descriptor(relative: str) -> dict[str, Any]:
        identity = file_descriptor(root / relative)
        identity["path"] = relative
        return identity

    aggregate_descriptor = receipt.get("outputs", {}).get("resource_aggregate")
    inventory_descriptor = receipt.get("outputs", {}).get("resource_file_inventory")
    _same_descriptor(
        aggregate_descriptor,
        local_descriptor("resource_aggregate.json"),
        "aggregate output",
    )
    _same_descriptor(
        inventory_descriptor,
        local_descriptor("resource_file_inventory.json"),
        "inventory output",
    )
    _same_descriptor(
        aggregate.get("resource_file_inventory"),
        local_descriptor("resource_file_inventory.json"),
        "aggregate inventory output",
    )
    require(
        receipt.get("counts")
        == {
            "authenticated_cells": EXPECTED_CELLS,
            "authenticated_source_behavioral_requests": EXPECTED_REQUESTS,
            "authenticated_source_behavioral_actions": EXPECTED_ACTIONS,
            "inventoried_raw_files": inventory["file_count"],
            "inventoried_unique_resolved_path_bytes": inventory[
                "unique_resolved_path_bytes"
            ],
        },
        "resource compiler receipt counts changed",
    )
    rows = aggregate.get("cell_measurements")
    require(isinstance(rows, list) and len(rows) == EXPECTED_CELLS, "cell output roster changed")
    receipt_outputs = receipt.get("outputs")
    require(
        isinstance(receipt_outputs, Mapping)
        and receipt_outputs.get("cell_measurement_count") == EXPECTED_CELLS
        and receipt_outputs.get("cell_measurements") == rows,
        "compiler/aggregate cell output rosters differ",
    )
    seen: set[str] = set()
    for row in rows:
        require(isinstance(row, Mapping), "cell output row is invalid")
        cell_id = row.get("cell_id")
        model = row.get("model_id")
        require(isinstance(cell_id, str) and cell_id not in seen, "cell output identity changed")
        require(model in MODELS, "cell output model changed")
        seen.add(cell_id)
        path = root / "cells" / str(model).lower() / f"{cell_id}.json"
        _same_descriptor(
            row.get("measurement"),
            local_descriptor(str(path.relative_to(root))),
            f"{cell_id} output",
        )
        cell = load_json(path, f"{cell_id} resource measurement")
        verify_signed(cell, f"{cell_id} resource measurement")
        _validate_cell_output(cell, model=str(model), cell_id=cell_id)
        require(
            cell.get("schema_version") == CELL_SCHEMA
            and cell.get("status") == "measured_with_declared_missingness"
            and cell.get("safe_for_legacy_resource_release_gate") is False
            and cell.get("measurement_completeness", {}).get("all_gpu_process_peak_memory") is False
            and cell.get("measurement_completeness", {}).get("annotation_time") is False
            and cell.get("annotation_time", {}).get("rater_a_total_seconds") is None
            and cell.get("annotation_time", {}).get("rater_b_total_seconds") is None,
            f"{cell_id} missingness boundary changed",
        )
    require(len(seen) == EXPECTED_CELLS, "cell output IDs are incomplete")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = compile_resources(args.manifest, args.manifest_sha256, args.output_dir)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        print(
            json.dumps({
                "status": "technical_failure",
                "error_type": type(error).__name__,
                "detail": str(error),
            }, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        raise
