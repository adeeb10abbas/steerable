#!/usr/bin/env python3
"""Build and run the detached zero-science development alignment job.

The workstation-side builder accepts only explicit, hash-bound terminal
receipts.  It never discovers or mutates the queue.  The cluster runtime
reopens their immutable PVC artifacts, reconstructs the exact
``wmf-development-release-evidence-v1`` full-two-model input, and calls the
reviewed freeze implementation with ``confirmation_release=False``.

Only the two mapping receipts, two alignment contracts, and one terminal job
receipt are published.  All source evidence and the complete lineage manifest
remain recoverable under this job's raw directory.  This job has no model,
simulator, label, resource-freeze, annotation, or confirmation authority.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
QUEUE_MODULE_PATH = Path(__file__).with_name("forecast_timing_queue_jobs.py")


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


queue = _load_module(QUEUE_MODULE_PATH, "wmf_development_alignment_queue_support")

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
RAW_ROOT = CONTROL_ROOT.parent
ROBOLAB_PYTHON = queue.ROBOLAB_PYTHON

THIS_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_physical_alignment_jobs.py"
)
QUEUE_RELATIVE = queue.THIS_RELATIVE
CONTRACT_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_physical_alignment_contract.json"
)
FREEZE_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/freeze_development_release.py"
)
TIMING_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/qualify_forecast_timing.py"
)
CAMERA_REPLAY_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/camera_crop_replay_witness.py"
)
COMPILER_QUEUE_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_evidence_compiler_jobs.py"
)
COMPILER_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/compile_development_evidence.py"
)
AGGREGATE_SUPPORT_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_timing_sidecar_jobs.py"
)
ABLATION_SPEC_RELATIVE = (
    queue.FORECAST_RELATIVE / "experiments/forecast_layout/ablation_spec.json"
)
PLANNED_CELLS_RELATIVE = (
    queue.FORECAST_RELATIVE / "experiments/forecast_layout/planned_cells.csv"
)

WAVE_SCHEMA = "wmf-development-physical-alignment-wave-v1"
CONTRACT_SCHEMA = "wmf-development-physical-alignment-contract-v1"
INPUT_SCHEMA = "wmf-development-physical-alignment-input-v1"
JOB_RECEIPT_SCHEMA = "wmf-development-physical-alignment-queue-job-v1"
RAW_MANIFEST_SCHEMA = "wmf-development-physical-alignment-evidence-manifest-v1"
COMPILER_JOB_SCHEMA = "wmf-development-evidence-compiler-queue-job-v1"
COMPILER_RECEIPT_SCHEMA = "wmf-development-evidence-compiler-receipt-v1"
COMPILER_INVENTORY_SCHEMA = "wmf-development-evidence-compiler-output-inventory-v1"
FREEZE_FRAGMENT_SCHEMA = "wmf-development-freeze-cell-evidence-fragment-v1"
TIMING_JOB_SCHEMA = "wmf-development-timing-sidecar-queue-job-v1"
CAMERA_JOB_SCHEMA = "wmf-camera-crop-witness-queue-job-v1"
CAMERA_CONTRACT_SCHEMA = "wmf-camera-crop-contract-v1"

JOB_ID = "development-physical-alignment-001"
WORKER_ROLE = "wmf-forecast-0912-worker-00"
MODE = "formal_full_two_model_alignment"
MAX_WALL_SECONDS = 21600
PUBLISH_LOG_TAIL_BYTES = 8192
PUBLISH_FILE_LIMIT_BYTES = 16 * 1024 * 1024
PUBLISH_JOB_LIMIT_BYTES = 64 * 1024 * 1024
INPUT_RELATIVE = Path("raw/development_physical_alignment_input.json")
EVIDENCE_RELATIVE = Path("raw/development_release_evidence.json")
BUNDLE_RELATIVE = Path("raw/physical_alignment_bundle")
RAW_MANIFEST_RELATIVE = Path("raw/physical_alignment_evidence_manifest.json")
STAGING_RELATIVE = Path("raw/alignment_publish_staging")
PUBLISH_RECEIPT_NAME = "development_physical_alignment_job_receipt.json"
PUBLISH_FAILURE_NAME = "development_physical_alignment_job_failure.json"

MODELS = ("N3", "D1")
OUTPUT_NAMES = {
    "N3": {
        "mapping": "n3_physical_alignment_receipt.json",
        "alignment": "n3_alignment_contract.json",
    },
    "D1": {
        "mapping": "d1_physical_alignment_receipt.json",
        "alignment": "d1_alignment_contract.json",
    },
}
SUCCESS_PUBLISH_FILES = tuple(
    OUTPUT_NAMES[model][role]
    for model in MODELS
    for role in ("mapping", "alignment")
) + (PUBLISH_RECEIPT_NAME,)

COMPILER_JOB_ID = "development-evidence-compiler-formal-004"
COMPILER_JOB_FILE_SHA256 = (
    "00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434"
)
COMPILER_JOB_FILE_BYTES = 12374
COMPILER_SOURCE_COMMIT = "8124d2b3e64e71e584b45d5707e47d763b77ab93"
COMPILER_RECEIPT_SHA256 = (
    "efc6ca7487fc85e3d8cb8e976c4e77e756d51edcacc31a0dbf017bb73ab5d84a"
)
COMPILER_RECEIPT_BYTES = 52888
COMPILER_INVENTORY_SHA256 = (
    "b1f38b3001337845e27ea205638d14b0e6bc90255498b5e5ed048859a155c537"
)
COMPILER_INVENTORY_BYTES = 36866

TIMING_JOB_IDS = {
    "N3": "timing-n3-development-sidecar-001",
    "D1": "timing-d1-development-sidecar-002",
}
TIMING_JOB_FILES = {
    "N3": {
        "sha256": "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
        "bytes": 29935,
    },
    "D1": {
        "sha256": "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
        "bytes": 79824,
    },
}
TIMING_SIDECARS = {
    "N3": {
        "sha256": "6d01b278f317015911f5d46d6b0ed1a3512dc9d89c68a7cbfc43ae5d4d832e0d",
        "bytes": 5473068,
    },
    "D1": {
        "sha256": "6bd497fce7ed5cbff8d78b6769ca4fcc8545fdefdcdf3b41126e92d01ae06ab3",
        "bytes": 3360158,
    },
}
REQUEST_COUNTS = {"N3": 240, "D1": 912}
EXPECTED_D1_COVERAGE = {
    "total_request_count": 912,
    "source_proven_full_decode_request_count": 240,
    "source_unmapped_incremental_decode_request_count": 672,
    "native_clock_matched_request_count": 224,
    "action_prefix_truncated_request_count": 16,
    "native_clock_matched_target_binding_count": 448,
    "action_prefix_truncated_target_binding_count": 32,
    "unmapped_potential_authority_target_count": 1344,
}
FORBIDDEN_CAMERA_JOB_IDS = {
    "camera-crop-replay-witness-001",
    "camera-crop-replay-witness-002",
}
CAMERA_JOB_RE = re.compile(r"camera-crop-replay-witness-(\d{3})\Z")
CAMERA_OUTPUT_KEY = "published_camera_crop_contracts"
CAMERA_TERMINAL_RECEIPT_KEYS = {
    "schema_version", "namespace", "study_id", "status", "decision", "job_id",
    "job_dir", "science_counts", "simulator_state_render_used",
    "whole_frame_identity", "safe_for_physical_alignment_input",
    "safe_to_release_confirmation", "confirmation_released",
    "behavioral_policy_skill_evaluated", "raw_outputs_recoverable_on_gm_pvc",
    "outputs", "payload_sha256",
}
CAMERA_OUTPUT_ENTRY_KEYS = {
    "path", "sha256", "bytes", "schema_version", "model_id",
    "camera_crop_id", "payload_sha256",
}
EXPECTED_CAMERA_CROP_IDS = {
    "N3": "n3-over-shoulder-left-168x320-v1",
    "D1": "d1-over-shoulder-left-176x320-v1",
}
MAPPING_KEYS = {
    "schema_version", "receipt_id", "study_id", "status", "model_id",
    "development_cell_ids", "development_cell_count", "development_request_count",
    "request_semantics", "timing_claim_boundary", "camera",
    "measured_clock_intervals", "frame_to_physical_time", "primary_target",
    "early_target", "source_receipts", "payload_sha256",
}
MAPPING_TARGET_KEYS = {
    "generated_frame_index", "target_physical_time_s", "status",
    "target_executed_action_offset", "eligible_request_count",
    "full_prefix_request_count", "max_camera_timestamp_residual_s",
    "max_physics_timestamp_residual_s",
}
REQUEST_SEMANTIC_KEYS = {
    "returned_action_horizon", "unchanged_executed_prefix_horizon",
    "action_space", "seed_semantics", "temporal_context",
}
TIMING_BOUNDARY_KEYS = {
    "N3": {
        "generated_target_source", "time_source_kind", "clock_bridge",
        "presentation_video_fps_used", "conditioning_fps_used_as_target_timing",
        "generated_frame_index_interpreted_as_action_index",
    },
    "D1": {
        "generated_target_source", "time_source_kind", "clock_bridge",
        "presentation_video_fps_used", "conditioning_fps_used_as_target_timing",
        "generated_frame_index_interpreted_as_action_index",
        "generated_targets_scope", "request_timing_coverage",
        "incremental_standalone_decodes_assigned_target_times",
        "timing_unmapped_requests_eligible",
    },
}
CAMERA_BINDING_KEYS = {
    "camera_id", "camera_crop_id", "camera_crop_sha256", "image_width_px",
    "image_height_px", "crop_operation",
}
MEASURED_CLOCK_KEYS = {
    "control_step_s_min", "control_step_s_max",
    "captured_frame_interval_s_min", "captured_frame_interval_s_max",
    "timestamp_tolerance_s", "tolerance_rule",
}
SOURCE_RECEIPT_KEYS = {
    "development_cell_receipts", "adapter_completions", "adapter_journals",
    "official_request_receipt_sha256s", "generated_target_timing",
    "resource_receipts", "camera_crop_contract",
}
EARLY_HORIZON_KEYS = {
    "horizon_s", "generated_frame_index", "target_executed_action_offset",
}
ALIGNMENT_KEYS = {
    "contract_id", "model_id", "mapping_receipt_id", "mapping_receipt_sha256",
    "primary_horizon_s", "generated_frame_index", "target_executed_action_offset",
    "control_step_s", "captured_frame_interval_s", "timestamp_tolerance_s",
    "camera_id", "camera_crop_id", "camera_crop_sha256", "image_width_px",
    "image_height_px", "early_horizon", "contract_sha256",
}
PLANNED_CELLS_SHA256 = (
    "7d06120a56419877d1acdfdc498c6dc054bf6860ce6bda5b2a25f55ae3b4166e"
)
EXPECTED_BUNDLE_FILES = 71
EXPECTED_SEMANTIC_FILES = 70
TIMESTAMP_TOLERANCE_RULE = (
    "min(half minimum positive native control interval, half minimum positive "
    "original-camera capture interval)"
)


class DevelopmentAlignmentJobError(RuntimeError):
    """A prerequisite, queue identity, or alignment release failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentAlignmentJobError(message)


def _finite_number(value: Any, label: str, *, strictly_positive: bool = False) -> float:
    require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    result = float(value)
    require(
        result > 0 if strictly_positive else result >= 0,
        f"{label} is outside its nonnegative range",
    )
    return result


def _zero_science_counts() -> dict[str, int]:
    return {
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


def _timing_zero_science_counts() -> dict[str, int]:
    result = _zero_science_counts()
    result.pop("simulator_processes_started")
    result.pop("labels_created_by_job")
    return result


def _compiler_zero_science_activity() -> dict[str, int]:
    return {
        "model_loads": 0,
        "model_requests_issued": 0,
        "simulator_processes_started": 0,
        "physical_resets": 0,
        "behavioral_actions_executed": 0,
        "labels_created": 0,
    }


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "queue_wrapper": root / THIS_RELATIVE,
        "queue_support": root / QUEUE_RELATIVE,
        "queue_contract": root / CONTRACT_RELATIVE,
        "freeze_validator": root / FREEZE_RELATIVE,
        "timing_validator": root / TIMING_RELATIVE,
        "camera_replay": root / CAMERA_REPLAY_RELATIVE,
        "compiler_queue": root / COMPILER_QUEUE_RELATIVE,
        "compiler": root / COMPILER_RELATIVE,
        "aggregate_support": root / AGGREGATE_SUPPORT_RELATIVE,
        "ablation_spec": root / ABLATION_SPEC_RELATIVE,
        "planned_cells": root / PLANNED_CELLS_RELATIVE,
    }


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "job": {
            "id": JOB_ID,
            "role": WORKER_ROLE,
            "mode": MODE,
            "max_wall_seconds": MAX_WALL_SECONDS,
            "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
            "raw_root": str(RAW_ROOT),
            "input_relative": str(INPUT_RELATIVE),
            "evidence_relative": str(EVIDENCE_RELATIVE),
            "bundle_relative": str(BUNDLE_RELATIVE),
            "raw_manifest_relative": str(RAW_MANIFEST_RELATIVE),
        },
        "prerequisites": {
            "compiler_job_id": COMPILER_JOB_ID,
            "compiler_job_file_sha256": COMPILER_JOB_FILE_SHA256,
            "timing_job_ids": TIMING_JOB_IDS,
            "timing_job_file_sha256": {
                model: TIMING_JOB_FILES[model]["sha256"] for model in MODELS
            },
            "camera_terminal_minimum_attempt": 3,
            "forbidden_camera_job_ids": sorted(FORBIDDEN_CAMERA_JOB_IDS),
            "camera_terminal_receipt_schema": CAMERA_JOB_SCHEMA,
            "camera_output_key": CAMERA_OUTPUT_KEY,
            "camera_contracts": 2,
            "camera_terminal_output_declarations": 2,
        },
        "release": {
            "evidence_schema": "wmf-development-release-evidence-v1",
            "cohort_branch": "full_two_model",
            "qualified_model_ids": ["N3", "D1"],
            "confirmation_release_argument": False,
            "authoritative_freeze_source_replay": True,
            "published_outputs_byte_identical_to_raw_bundle": True,
            "published_files": list(SUCCESS_PUBLISH_FILES),
            "raw_bundle_files": 4,
            "mapping_receipts": 2,
            "alignment_contracts": 2,
            "publisher_file_limit_bytes": PUBLISH_FILE_LIMIT_BYTES,
            "publisher_job_limit_bytes": PUBLISH_JOB_LIMIT_BYTES,
        },
        "science_counts": _zero_science_counts(),
        "labels_created": 0,
        "resource_measurements_created": 0,
        "annotation_authority": False,
        "resource_freeze_authority": False,
        "confirmation_authority": False,
        "safe_to_release_confirmation": False,
    }


def _validate_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = queue.file_identity(path)
    require(
        identity["sha256"]
        == queue._verified_sha(expected_sha256, "alignment contract digest"),
        "development alignment contract hash changed",
    )
    require(
        queue.load_json(path, "development alignment contract")
        == _expected_contract(),
        "development alignment contract fields changed",
    )
    return identity


def _local_implementation() -> dict[str, dict[str, Any]]:
    paths = _source_paths(REPOSITORY_ROOT)
    missing = [name for name, path in paths.items() if not path.is_file()]
    require(not missing, f"development alignment implementation missing: {missing}")
    result = {name: queue.file_identity(path) for name, path in paths.items()}
    require(
        result["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256,
        "planned-cell CSV hash changed",
    )
    _validate_contract(
        paths["queue_contract"], result["queue_contract"]["sha256"]
    )
    return result


def _descriptor_shape(value: Any, label: str) -> dict[str, Any]:
    require(
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256", "bytes"},
        f"{label} descriptor fields changed",
    )
    path = Path(value.get("path", ""))
    require(path.is_absolute(), f"{label} path must be absolute")
    require(
        path == Path(os.path.abspath(os.fspath(path))),
        f"{label} path is not lexically normalized",
    )
    require(path.is_relative_to(RAW_ROOT), f"{label} path is outside task raw root")
    digest = queue._verified_sha(value.get("sha256"), f"{label} digest")
    size = value.get("bytes")
    require(type(size) is int and size > 0, f"{label} byte count is invalid")
    return {"path": str(path), "sha256": digest, "bytes": size}


def _load_local(path: Path, digest: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = queue.file_identity(path)
    require(
        identity["sha256"] == queue._verified_sha(digest, f"{label} digest"),
        f"{label} local file hash changed",
    )
    value = queue.load_json(path, label)
    queue.verify_signed_document(value, label)
    return value, identity


def _validate_results_checkout(path: Path, expected_commit: str) -> Path:
    supplied = Path(path)
    require(not supplied.is_symlink(), "results checkout is a symlink")
    root = supplied.resolve(strict=True)
    require(root.is_dir(), "results checkout is unavailable")
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, env=environment,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True, text=True, timeout=30, env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DevelopmentAlignmentJobError("results Git checkout verification failed") from error
    require(
        head.returncode == status.returncode == 0
        and head.stdout.strip() == expected_commit
        and not status.stdout,
        "results checkout is not clean at the exact supplied commit",
    )
    return root


def _validate_compiler_job(
    value: Mapping[str, Any], *, file_identity: Mapping[str, Any]
) -> dict[str, Any]:
    require(
        file_identity.get("sha256") == COMPILER_JOB_FILE_SHA256
        and file_identity.get("bytes") == COMPILER_JOB_FILE_BYTES,
        "compiler004 terminal receipt file identity changed",
    )
    exact = {
        "schema_version": COMPILER_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": "formal_full",
        "status": "passed",
        "decision": "go",
        "job_id": COMPILER_JOB_ID,
        "job_dir": str(CONTROL_ROOT / "jobs" / COMPILER_JOB_ID),
        "study_commit": COMPILER_SOURCE_COMMIT,
        "formal_cohort_complete": True,
        "safe_for_timing_binding": True,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "behavioral_policy_skill_evaluated": False,
        "raw_outputs_recoverable_on_gm_pvc": True,
        "counts": {
            "compiled_cells": 32,
            "compiled_source_behavioral_requests": 1152,
            "compiled_source_behavioral_actions": 14400,
        },
        "science_counts": _zero_science_counts(),
        "compiler_science_activity": _compiler_zero_science_activity(),
        "published_files": ["development_evidence_compiler_job_receipt.json"],
    }
    for key, expected in exact.items():
        require(value.get(key) == expected, f"compiler004 receipt changed: {key}")
    outputs = value.get("outputs")
    require(isinstance(outputs, Mapping), "compiler004 outputs are missing")
    compiler_receipt = _descriptor_shape(
        outputs.get("compiler_receipt"), "compiler004 compiler receipt"
    )
    inventory = _descriptor_shape(
        outputs.get("compiler_output_inventory"), "compiler004 output inventory"
    )
    require(
        compiler_receipt
        == {
            "path": str(
                CONTROL_ROOT / "jobs" / COMPILER_JOB_ID / "raw/compiler_bundle/compiler_receipt.json"
            ),
            "sha256": COMPILER_RECEIPT_SHA256,
            "bytes": COMPILER_RECEIPT_BYTES,
        },
        "compiler004 inner receipt identity changed",
    )
    require(
        inventory
        == {
            "path": str(
                CONTROL_ROOT / "jobs" / COMPILER_JOB_ID / "raw/compiler_output_inventory.json"
            ),
            "sha256": COMPILER_INVENTORY_SHA256,
            "bytes": COMPILER_INVENTORY_BYTES,
        },
        "compiler004 output inventory identity changed",
    )
    primary = outputs.get("primary_compiled_outputs")
    require(isinstance(primary, Mapping), "compiler004 primary output map is missing")
    expected_primary = {
        f"{model.lower()}_development_{suffix}.json"
        for model in MODELS
        for suffix in (
            "timing_request_inventory", "request_provenance", "freeze_cells"
        )
    }
    require(set(primary) == expected_primary, "compiler004 primary output inventory changed")
    freeze_fragments = {
        model: _descriptor_shape(
            primary[f"{model.lower()}_development_freeze_cells.json"],
            f"{model} compiler freeze fragment",
        )
        for model in MODELS
    }
    require(
        outputs.get("semantically_validated_nonreceipt_file_count")
        == EXPECTED_SEMANTIC_FILES,
        "compiler004 semantic output count changed",
    )
    return {
        "job_receipt": _descriptor_shape(
            {
                "path": str(
                    CONTROL_ROOT / "jobs" / COMPILER_JOB_ID
                    / "publish/development_evidence_compiler_job_receipt.json"
                ),
                "sha256": file_identity["sha256"],
                "bytes": file_identity["bytes"],
            },
            "compiler004 PVC job receipt",
        ),
        "bundle_root": str(
            CONTROL_ROOT / "jobs" / COMPILER_JOB_ID / "raw/compiler_bundle"
        ),
        "compiler_receipt": compiler_receipt,
        "output_inventory": inventory,
        "freeze_fragments": freeze_fragments,
    }


def _validate_timing_job(
    model: str, value: Mapping[str, Any], *, file_identity: Mapping[str, Any]
) -> dict[str, Any]:
    job_id = TIMING_JOB_IDS[model]
    expected_file = TIMING_JOB_FILES[model]
    require(
        file_identity.get("sha256") == expected_file["sha256"]
        and file_identity.get("bytes") == expected_file["bytes"],
        f"{model} terminal timing receipt file identity changed",
    )
    exact = {
        "schema_version": TIMING_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "model_id": model,
        "status": "passed",
        "decision": "go",
        "job_id": job_id,
        "job_dir": str(CONTROL_ROOT / "jobs" / job_id),
        "referenced_behavioral_cells": 16,
        "referenced_behavioral_model_requests": REQUEST_COUNTS[model],
        "science_counts": _timing_zero_science_counts(),
        "physical_time_qualified": True,
        "development_timing_sidecar_valid": True,
        "behavioral_policy_skill_evaluated": False,
        "safe_to_release_confirmation": False,
        "raw_outputs_recoverable_on_gm_pvc": True,
    }
    for key, expected in exact.items():
        require(value.get(key) == expected, f"{model} timing job changed: {key}")
    if model == "D1":
        require(
            value.get("request_timing_coverage") == EXPECTED_D1_COVERAGE
            and value.get("physical_time_coverage_complete") is False
            and value.get("physical_time_qualified_scope")
            == "source-proven full conditioning-origin decodes only",
            "D1 timing missingness boundary changed",
        )
    else:
        require(
            "physical_time_coverage_complete" not in value
            and "physical_time_qualified_scope" not in value,
            "N3 terminal timing receipt field boundary changed",
        )
    require(
        isinstance(value.get("source_request_receipt_sha256s"), list)
        and len(value["source_request_receipt_sha256s"]) == REQUEST_COUNTS[model]
        and len(set(value["source_request_receipt_sha256s"]))
        == REQUEST_COUNTS[model],
        f"{model} timing request identity inventory changed",
    )
    outputs = value.get("outputs")
    require(isinstance(outputs, Mapping), f"{model} timing outputs are missing")
    sidecar = _descriptor_shape(
        outputs.get("raw_development_timing_sidecar"), f"{model} raw timing sidecar"
    )
    expected_sidecar = {
        "path": str(
            CONTROL_ROOT / "jobs" / job_id / "raw"
            / f"{model.lower()}_development_timing_sidecar.json"
        ),
        **TIMING_SIDECARS[model],
    }
    require(sidecar == expected_sidecar, f"{model} raw timing sidecar identity changed")
    return {
        "model_id": model,
        "job_receipt": {
            "path": str(
                CONTROL_ROOT / "jobs" / job_id / "publish/timing_job_receipt.json"
            ),
            "sha256": file_identity["sha256"],
            "bytes": file_identity["bytes"],
        },
        "sidecar": sidecar,
    }


def _camera_attempt(job_id: Any) -> int:
    require(isinstance(job_id, str), "camera terminal job ID is missing")
    match = CAMERA_JOB_RE.fullmatch(job_id)
    require(match is not None, "camera terminal job ID is invalid")
    ordinal = int(match.group(1))
    require(
        ordinal >= 3 and job_id not in FORBIDDEN_CAMERA_JOB_IDS,
        "diagnostic/nonterminal camera attempt cannot feed alignment",
    )
    return ordinal


def _validate_camera_job(
    value: Mapping[str, Any], *, file_identity: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(
        isinstance(value, Mapping) and set(value) == CAMERA_TERMINAL_RECEIPT_KEYS,
        "camera terminal receipt fields changed",
    )
    queue.verify_signed_document(value, "camera terminal job receipt")
    job_id = value.get("job_id")
    _camera_attempt(job_id)
    exact = {
        "schema_version": CAMERA_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "passed",
        "decision": "qualified_from_original_camera_pixels",
        "job_dir": str(CONTROL_ROOT / "jobs" / str(job_id)),
        "science_counts": _zero_science_counts(),
        "simulator_state_render_used": False,
        "whole_frame_identity": False,
        "safe_for_physical_alignment_input": True,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "behavioral_policy_skill_evaluated": False,
        "raw_outputs_recoverable_on_gm_pvc": True,
    }
    for key, expected in exact.items():
        require(value.get(key) == expected, f"camera terminal receipt changed: {key}")
    outputs = value.get("outputs")
    require(
        isinstance(outputs, Mapping) and set(outputs) == {CAMERA_OUTPUT_KEY},
        "camera terminal outputs changed",
    )
    contracts = outputs.get(CAMERA_OUTPUT_KEY)
    require(
        isinstance(contracts, Mapping) and set(contracts) == set(MODELS),
        "camera terminal contract output inventory changed",
    )
    normalized: dict[str, dict[str, Any]] = {}
    expected_names = {
        "N3": "n3_camera_crop_contract.json",
        "D1": "d1_camera_crop_contract.json",
    }
    for model in MODELS:
        entry = contracts[model]
        require(
            isinstance(entry, Mapping) and set(entry) == CAMERA_OUTPUT_ENTRY_KEYS,
            f"{model} camera terminal output fields changed",
        )
        descriptor = _descriptor_shape(
            {key: entry[key] for key in ("path", "sha256", "bytes")},
            f"{model} published crop contract",
        )
        require(
            descriptor["path"]
            == str(CONTROL_ROOT / "jobs" / str(job_id) / "publish" / expected_names[model])
            and entry.get("schema_version") == CAMERA_CONTRACT_SCHEMA
            and entry.get("model_id") == model
            and entry.get("camera_crop_id") == EXPECTED_CAMERA_CROP_IDS[model]
            and isinstance(entry.get("payload_sha256"), str)
            and queue.SHA256_RE.fullmatch(entry["payload_sha256"]) is not None,
            f"{model} camera terminal output identity changed",
        )
        normalized[model] = {
            **descriptor,
            "schema_version": entry["schema_version"],
            "model_id": entry["model_id"],
            "camera_crop_id": entry["camera_crop_id"],
            "payload_sha256": entry["payload_sha256"],
        }
    job_descriptor = {
        "path": str(
            CONTROL_ROOT / "jobs" / str(job_id)
            / "publish/camera_crop_witness_job_receipt.json"
        ),
        "sha256": file_identity["sha256"],
        "bytes": file_identity["bytes"],
    }
    return job_descriptor, normalized


def _bind_crop_file_to_camera_declaration(
    *, model: str, value: Mapping[str, Any], file_identity: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        isinstance(declaration, Mapping)
        and set(declaration) == CAMERA_OUTPUT_ENTRY_KEYS,
        f"{model} camera output declaration fields changed",
    )
    require(
        file_identity.get("sha256") == declaration["sha256"]
        and file_identity.get("bytes") == declaration["bytes"]
        and value.get("schema_version") == declaration["schema_version"]
        and value.get("model_id") == declaration["model_id"] == model
        and value.get("camera_crop_id") == declaration["camera_crop_id"]
        == EXPECTED_CAMERA_CROP_IDS[model]
        and value.get("payload_sha256") == declaration["payload_sha256"],
        f"{model} crop file differs from camera terminal declaration",
    )
    return dict(declaration)


def _validate_crop_file(
    *, model: str, path: Path, digest: str, camera_replay: ModuleType
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, identity = _load_local(path, digest, f"{model} camera crop contract")
    try:
        camera_replay.validate_camera_crop_contract(value, expected_model=model)
    except BaseException as error:
        raise DevelopmentAlignmentJobError(
            f"{model} camera crop contract failed exact replay validation: {error}"
        ) from error
    return value, identity


def validate_local_prerequisites(
    *,
    compiler_job_receipt: Path,
    compiler_job_receipt_sha256: str,
    n3_timing_job_receipt: Path,
    n3_timing_job_receipt_sha256: str,
    d1_timing_job_receipt: Path,
    d1_timing_job_receipt_sha256: str,
    camera_job_receipt: Path,
    camera_job_receipt_sha256: str,
    n3_camera_crop_contract: Path,
    n3_camera_crop_contract_sha256: str,
    d1_camera_crop_contract: Path,
    d1_camera_crop_contract_sha256: str,
    results_commit: str,
    results_root: Path,
) -> dict[str, Any]:
    """Authenticate explicit fetched receipts; never discover a replacement."""

    results_commit = queue._verified_commit(results_commit)
    results_root = _validate_results_checkout(results_root, results_commit)
    expected_paths = {
        "compiler": results_root / "results/jobs" / COMPILER_JOB_ID
        / "publish/development_evidence_compiler_job_receipt.json",
        "N3_timing": results_root / "results/jobs" / TIMING_JOB_IDS["N3"]
        / "publish/timing_job_receipt.json",
        "D1_timing": results_root / "results/jobs" / TIMING_JOB_IDS["D1"]
        / "publish/timing_job_receipt.json",
    }
    require(
        Path(compiler_job_receipt).resolve(strict=True) == expected_paths["compiler"]
        and Path(n3_timing_job_receipt).resolve(strict=True) == expected_paths["N3_timing"]
        and Path(d1_timing_job_receipt).resolve(strict=True) == expected_paths["D1_timing"],
        "compiler/timing receipts are not the exact fetched-results artifacts",
    )
    compiler_value, compiler_local = _load_local(
        compiler_job_receipt, compiler_job_receipt_sha256,
        "compiler004 terminal job receipt",
    )
    compiler = _validate_compiler_job(
        compiler_value, file_identity=compiler_local
    )
    timing_rows = []
    for model, path, digest in (
        ("N3", n3_timing_job_receipt, n3_timing_job_receipt_sha256),
        ("D1", d1_timing_job_receipt, d1_timing_job_receipt_sha256),
    ):
        value, identity = _load_local(path, digest, f"{model} timing job receipt")
        timing_rows.append(
            _validate_timing_job(model, value, file_identity=identity)
        )
    camera_value, camera_local = _load_local(
        camera_job_receipt, camera_job_receipt_sha256,
        "camera terminal job receipt",
    )
    camera_job, camera_outputs = _validate_camera_job(
        camera_value, file_identity=camera_local
    )
    camera_job_id = camera_value["job_id"]
    expected_camera_root = results_root / "results/jobs" / camera_job_id / "publish"
    require(
        Path(camera_job_receipt).resolve(strict=True)
        == expected_camera_root / "camera_crop_witness_job_receipt.json"
        and Path(n3_camera_crop_contract).resolve(strict=True)
        == expected_camera_root / "n3_camera_crop_contract.json"
        and Path(d1_camera_crop_contract).resolve(strict=True)
        == expected_camera_root / "d1_camera_crop_contract.json",
        "camera receipt/contracts are not the exact fetched-results artifacts",
    )
    camera_replay = _load_module(
        REPOSITORY_ROOT / CAMERA_REPLAY_RELATIVE,
        "wmf_development_alignment_local_camera_validator",
    )
    crop_rows = []
    for model, path, digest in (
        ("N3", n3_camera_crop_contract, n3_camera_crop_contract_sha256),
        ("D1", d1_camera_crop_contract, d1_camera_crop_contract_sha256),
    ):
        value, local_identity = _validate_crop_file(
            model=model, path=path, digest=digest, camera_replay=camera_replay
        )
        expected = camera_outputs[model]
        declaration = _bind_crop_file_to_camera_declaration(
            model=model,
            value=value,
            file_identity=local_identity,
            declaration=expected,
        )
        crop_rows.append({
            "model_id": model,
            "contract": {
                key: declaration[key] for key in ("path", "sha256", "bytes")
            },
            "contract_payload_sha256": value["payload_sha256"],
            "camera_terminal_output": declaration,
        })
    return validate_input_shape({
        "schema_version": INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "results_commit": results_commit,
        "compiler": compiler,
        "timing": timing_rows,
        "camera": {
            "job_receipt": camera_job,
            "crop_contracts": crop_rows,
        },
    })


def validate_input_shape(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "alignment input must be an object")
    require(
        set(value)
        == {"schema_version", "study_id", "mode", "results_commit", "compiler", "timing", "camera"},
        "alignment input fields changed",
    )
    require(
        value.get("schema_version") == INPUT_SCHEMA
        and value.get("study_id") == STUDY_ID
        and value.get("mode") == MODE,
        "alignment input identity changed",
    )
    results_commit = queue._verified_commit(value.get("results_commit"))
    raw_compiler = value.get("compiler")
    require(
        isinstance(raw_compiler, Mapping)
        and set(raw_compiler)
        == {"job_receipt", "bundle_root", "compiler_receipt", "output_inventory", "freeze_fragments"},
        "alignment compiler input fields changed",
    )
    bundle_root = Path(raw_compiler.get("bundle_root", ""))
    expected_bundle = CONTROL_ROOT / "jobs" / COMPILER_JOB_ID / "raw/compiler_bundle"
    require(
        bundle_root == expected_bundle and bundle_root.is_absolute(),
        "alignment compiler bundle root changed",
    )
    fragments = raw_compiler.get("freeze_fragments")
    require(
        isinstance(fragments, Mapping) and set(fragments) == set(MODELS),
        "alignment freeze-fragment inventory changed",
    )
    compiler = {
        "job_receipt": _descriptor_shape(raw_compiler["job_receipt"], "compiler job receipt"),
        "bundle_root": str(bundle_root),
        "compiler_receipt": _descriptor_shape(raw_compiler["compiler_receipt"], "compiler receipt"),
        "output_inventory": _descriptor_shape(raw_compiler["output_inventory"], "compiler output inventory"),
        "freeze_fragments": {
            model: _descriptor_shape(fragments[model], f"{model} freeze fragment")
            for model in MODELS
        },
    }
    timing = value.get("timing")
    require(isinstance(timing, list) and len(timing) == 2, "alignment timing inputs changed")
    timing_by_model: dict[str, dict[str, Any]] = {}
    for row in timing:
        require(
            isinstance(row, Mapping)
            and set(row) == {"model_id", "job_receipt", "sidecar"}
            and row.get("model_id") in MODELS
            and row["model_id"] not in timing_by_model,
            "alignment timing input row changed",
        )
        model = row["model_id"]
        timing_by_model[model] = {
            "model_id": model,
            "job_receipt": _descriptor_shape(row["job_receipt"], f"{model} timing job receipt"),
            "sidecar": _descriptor_shape(row["sidecar"], f"{model} timing sidecar"),
        }
    require(set(timing_by_model) == set(MODELS), "alignment timing model coverage changed")
    raw_camera = value.get("camera")
    require(
        isinstance(raw_camera, Mapping)
        and set(raw_camera) == {"job_receipt", "crop_contracts"},
        "alignment camera input fields changed",
    )
    camera_job_receipt = _descriptor_shape(
        raw_camera["job_receipt"], "camera terminal job receipt"
    )
    camera_job_path = Path(camera_job_receipt["path"])
    require(
        camera_job_path.name == "camera_crop_witness_job_receipt.json"
        and camera_job_path.parent.name == "publish",
        "camera terminal job receipt path changed",
    )
    camera_job_id = camera_job_path.parents[1].name
    _camera_attempt(camera_job_id)
    require(
        camera_job_path
        == CONTROL_ROOT / "jobs" / camera_job_id / "publish"
        / "camera_crop_witness_job_receipt.json",
        "camera terminal job receipt path is not canonical",
    )
    raw_crops = raw_camera.get("crop_contracts")
    require(isinstance(raw_crops, list) and len(raw_crops) == 2, "alignment crop inputs changed")
    crop_by_model: dict[str, dict[str, Any]] = {}
    for row in raw_crops:
        require(
            isinstance(row, Mapping)
            and set(row) == {
                "model_id", "contract", "contract_payload_sha256",
                "camera_terminal_output",
            }
            and row.get("model_id") in MODELS
            and row["model_id"] not in crop_by_model
            and isinstance(row.get("contract_payload_sha256"), str)
            and queue.SHA256_RE.fullmatch(row["contract_payload_sha256"]) is not None,
            "alignment crop input row changed",
        )
        model = row["model_id"]
        contract = _descriptor_shape(row["contract"], f"{model} crop contract")
        declaration = row.get("camera_terminal_output")
        require(
            isinstance(declaration, Mapping)
            and set(declaration) == CAMERA_OUTPUT_ENTRY_KEYS
            and {key: declaration[key] for key in ("path", "sha256", "bytes")}
            == contract
            and declaration.get("schema_version") == CAMERA_CONTRACT_SCHEMA
            and declaration.get("model_id") == model
            and declaration.get("camera_crop_id") == EXPECTED_CAMERA_CROP_IDS[model]
            and declaration.get("payload_sha256")
            == row["contract_payload_sha256"]
            and contract["path"]
            == str(
                CONTROL_ROOT / "jobs" / camera_job_id / "publish"
                / f"{model.lower()}_camera_crop_contract.json"
            ),
            f"{model} crop input differs from camera terminal declaration",
        )
        crop_by_model[model] = {
            "model_id": model,
            "contract": contract,
            "contract_payload_sha256": row["contract_payload_sha256"],
            "camera_terminal_output": dict(declaration),
        }
    require(set(crop_by_model) == set(MODELS), "alignment crop model coverage changed")
    descriptor_paths = [
        compiler["job_receipt"]["path"], compiler["compiler_receipt"]["path"],
        compiler["output_inventory"]["path"],
        *(compiler["freeze_fragments"][model]["path"] for model in MODELS),
        *(timing_by_model[model][field]["path"] for model in MODELS for field in ("job_receipt", "sidecar")),
        raw_camera["job_receipt"]["path"],
        *(crop_by_model[model]["contract"]["path"] for model in MODELS),
    ]
    require(len(descriptor_paths) == len(set(descriptor_paths)), "alignment input paths are ambiguous")
    return {
        "schema_version": INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "results_commit": results_commit,
        "compiler": compiler,
        "timing": [timing_by_model[model] for model in MODELS],
        "camera": {
            "job_receipt": camera_job_receipt,
            "crop_contracts": [crop_by_model[model] for model in MODELS],
        },
    }


def _implementation_argv(implementation: Mapping[str, Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for name in sorted(implementation):
        result.extend([
            f"--{name.replace('_', '-')}-sha256",
            str(implementation[name]["sha256"]),
        ])
    return result


def _job_descriptor(
    *, study_commit: str, implementation: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(
        set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
        "alignment implementation inventory changed",
    )
    normalized = validate_input_shape(inputs)
    argv = [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        "run",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", JOB_ID,
        "--expected-role", WORKER_ROLE,
        "--inputs-json", json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    ]
    argv.extend(_implementation_argv(implementation))
    return {
        "job_id": JOB_ID,
        "released": True,
        "source_commit": commit,
        "role": WORKER_ROLE,
        "argv": argv,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def _build_wave_from_validated_receipts(
    *, study_commit: str, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    implementation = _local_implementation()
    normalized = validate_input_shape(inputs)
    descriptor = _job_descriptor(
        study_commit=study_commit, implementation=implementation, inputs=normalized
    )
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": MODE,
        "source_commit": descriptor["source_commit"],
        "status": "descriptor_only_not_dispatched_all_terminal_receipt_gates_passed",
        "jobs": [descriptor],
        "implementation": implementation,
        "inputs": normalized,
        "science_counts": _zero_science_counts(),
        "labels_created": 0,
        "annotation_authority": False,
        "resource_freeze_authority": False,
        "confirmation_authority": False,
        "safe_to_release_confirmation": False,
        "claim_boundary": (
            "One CPU-only physical-alignment derivation over immutable completed "
            "development evidence. It creates no science, label, resource, annotation, "
            "or confirmation release."
        ),
    }


def build_wave_from_local_files(args: argparse.Namespace) -> dict[str, Any]:
    inputs = validate_local_prerequisites(
        compiler_job_receipt=args.compiler_job_receipt,
        compiler_job_receipt_sha256=args.compiler_job_receipt_sha256,
        n3_timing_job_receipt=args.n3_timing_job_receipt,
        n3_timing_job_receipt_sha256=args.n3_timing_job_receipt_sha256,
        d1_timing_job_receipt=args.d1_timing_job_receipt,
        d1_timing_job_receipt_sha256=args.d1_timing_job_receipt_sha256,
        camera_job_receipt=args.camera_job_receipt,
        camera_job_receipt_sha256=args.camera_job_receipt_sha256,
        n3_camera_crop_contract=args.n3_camera_crop_contract,
        n3_camera_crop_contract_sha256=args.n3_camera_crop_contract_sha256,
        d1_camera_crop_contract=args.d1_camera_crop_contract,
        d1_camera_crop_contract_sha256=args.d1_camera_crop_contract_sha256,
        results_commit=args.results_commit,
        results_root=args.results_root,
    )
    return _build_wave_from_validated_receipts(
        study_commit=args.study_commit, inputs=inputs
    )


def _runtime_implementation(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": queue._verified_sha(
                getattr(args, name + "_sha256"), f"{name} digest"
            )
        }
        for name in sorted(_source_paths(REPOSITORY_ROOT))
    }


def _runtime_descriptor(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    require(args.command == "run", "alignment runtime mode changed")
    require(
        args.job_id == JOB_ID and args.expected_role == WORKER_ROLE,
        "alignment runtime job/role changed",
    )
    try:
        inputs = json.loads(args.inputs_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise DevelopmentAlignmentJobError("alignment runtime inputs JSON is invalid") from error
    inputs = validate_input_shape(inputs)
    implementation = _runtime_implementation(args)
    descriptor = _job_descriptor(
        study_commit=args.study_commit, implementation=implementation, inputs=inputs
    )
    return descriptor, implementation, inputs


def _validate_staged_implementation(
    source_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], ModuleType, ModuleType, ModuleType, ModuleType]:
    paths = _source_paths(source_root)
    require(set(paths) == set(expected), "staged alignment implementation changed")
    observed: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        identity = queue.file_identity(path)
        require(identity["sha256"] == expected[name]["sha256"], f"staged {name} hash changed")
        observed[name] = identity
    require(
        observed["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256,
        "staged planned-cell CSV hash changed",
    )
    _validate_contract(paths["queue_contract"], expected["queue_contract"]["sha256"])
    freeze = _load_module(paths["freeze_validator"], "wmf_alignment_staged_freeze")
    timing = _load_module(paths["timing_validator"], "wmf_alignment_staged_timing")
    camera = _load_module(paths["camera_replay"], "wmf_alignment_staged_camera")
    compiler_queue = _load_module(paths["compiler_queue"], "wmf_alignment_staged_compiler_queue")
    require(
        getattr(freeze, "EVIDENCE_SCHEMA", None) == "wmf-development-release-evidence-v1"
        and getattr(freeze, "BRANCH_MODELS", {}).get("full_two_model") == MODELS,
        "staged freeze alignment interface changed",
    )
    require(
        getattr(camera, "SCHEMA", None) == CAMERA_CONTRACT_SCHEMA,
        "staged camera contract interface changed",
    )
    require(
        getattr(compiler_queue, "EXPECTED_BUNDLE_FILES", None) == EXPECTED_BUNDLE_FILES,
        "staged compiler bundle interface changed",
    )
    return observed, freeze, timing, camera, compiler_queue


def _verify_runtime_descriptor(value: Mapping[str, Any], label: str) -> Path:
    descriptor = _descriptor_shape(value, label)
    path = Path(descriptor["path"])
    _reject_symlink_components(path, label)
    require(queue.file_identity(path) == descriptor, f"{label} bytes/hash changed on PVC")
    return path


def _validate_runtime_receipts(
    inputs: Mapping[str, Any], *, timing: ModuleType, camera: ModuleType
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized = validate_input_shape(inputs)
    compiler_path = _verify_runtime_descriptor(
        normalized["compiler"]["job_receipt"], "compiler004 job receipt"
    )
    compiler_value = queue.load_json(compiler_path, "compiler004 job receipt")
    queue.verify_signed_document(compiler_value, "compiler004 job receipt")
    compiler = _validate_compiler_job(
        compiler_value, file_identity=queue.file_identity(compiler_path)
    )
    require(compiler == normalized["compiler"], "compiler004 normalized input changed")
    timing_values: dict[str, dict[str, Any]] = {}
    for row in normalized["timing"]:
        model = row["model_id"]
        job_path = _verify_runtime_descriptor(row["job_receipt"], f"{model} timing job receipt")
        job_value = queue.load_json(job_path, f"{model} timing job receipt")
        queue.verify_signed_document(job_value, f"{model} timing job receipt")
        observed = _validate_timing_job(
            model, job_value, file_identity=queue.file_identity(job_path)
        )
        require(observed == row, f"{model} normalized timing input changed")
        sidecar_path = _verify_runtime_descriptor(row["sidecar"], f"{model} timing sidecar")
        try:
            sidecar = timing.validate_development_timing(
                sidecar_path, row["sidecar"]["sha256"], expected_model=model
            )
        except BaseException as error:
            raise DevelopmentAlignmentJobError(
                f"{model} raw timing sidecar failed deep validation: {error}"
            ) from error
        if model == "D1":
            require(
                sidecar.get("request_timing_coverage") == EXPECTED_D1_COVERAGE,
                "D1 raw sidecar missingness coverage changed",
            )
        timing_values[model] = sidecar
    camera_path = _verify_runtime_descriptor(
        normalized["camera"]["job_receipt"], "camera terminal job receipt"
    )
    camera_value = queue.load_json(camera_path, "camera terminal job receipt")
    queue.verify_signed_document(camera_value, "camera terminal job receipt")
    camera_job, camera_outputs = _validate_camera_job(
        camera_value, file_identity=queue.file_identity(camera_path)
    )
    require(
        camera_job == normalized["camera"]["job_receipt"],
        "normalized camera job input changed",
    )
    crop_values: dict[str, dict[str, Any]] = {}
    for row in normalized["camera"]["crop_contracts"]:
        model = row["model_id"]
        require(
            row["camera_terminal_output"] == camera_outputs[model]
            and row["contract"] == {
                key: camera_outputs[model][key]
                for key in ("path", "sha256", "bytes")
            },
            f"{model} camera output binding changed",
        )
        path = _verify_runtime_descriptor(row["contract"], f"{model} camera crop contract")
        value = queue.load_json(path, f"{model} camera crop contract")
        try:
            camera.validate_camera_crop_contract(value, expected_model=model)
        except BaseException as error:
            raise DevelopmentAlignmentJobError(
                f"{model} camera crop contract failed deep validation: {error}"
            ) from error
        require(
            value.get("payload_sha256") == row["contract_payload_sha256"],
            f"{model} camera contract payload identity changed",
        )
        _bind_crop_file_to_camera_declaration(
            model=model,
            value=value,
            file_identity=queue.file_identity(path),
            declaration=camera_outputs[model],
        )
        crop_values[model] = value
    return normalized, timing_values, crop_values


def _reject_symlink_components(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = lexical
    while True:
        require(not cursor.is_symlink(), f"{label} contains a symlink: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return lexical
        cursor = parent


def _reject_tree_symlinks(root: Path, label: str) -> Path:
    supplied = Path(root)
    _reject_symlink_components(supplied, label)
    resolved = supplied.resolve(strict=True)
    require(resolved.is_dir(), f"{label} root is unavailable")
    require(not any(path.is_symlink() for path in resolved.rglob("*")), f"{label} contains a symlink")
    return resolved


def _validate_compiler_bundle(
    compiler_input: Mapping[str, Any], *, compiler_queue: ModuleType
) -> dict[str, dict[str, Any]]:
    bundle = _reject_tree_symlinks(Path(compiler_input["bundle_root"]), "compiler bundle")
    require(
        bundle == CONTROL_ROOT / "jobs" / COMPILER_JOB_ID / "raw/compiler_bundle",
        "compiler bundle resolved root changed",
    )
    expected_paths = set(compiler_queue._expected_bundle_paths())
    require(len(expected_paths) == EXPECTED_BUNDLE_FILES, "compiler expected bundle count changed")
    actual: dict[str, dict[str, Any]] = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_dir():
            continue
        require(path.is_file(), "compiler bundle contains a non-file")
        relative = str(path.relative_to(bundle))
        actual[relative] = queue.file_identity(path)
    require(set(actual) == expected_paths, "compiler bundle exact file inventory changed")
    inventory_path = _verify_runtime_descriptor(
        compiler_input["output_inventory"], "compiler output inventory"
    )
    inventory = queue.load_json(inventory_path, "compiler output inventory")
    queue.verify_signed_document(inventory, "compiler output inventory")
    require(
        inventory.get("schema_version") == COMPILER_INVENTORY_SCHEMA
        and inventory.get("namespace") == NAMESPACE
        and inventory.get("study_id") == STUDY_ID
        and inventory.get("mode") == "formal_full"
        and inventory.get("job_id") == COMPILER_JOB_ID
        and inventory.get("bundle_root") == str(bundle)
        and inventory.get("file_count") == EXPECTED_BUNDLE_FILES
        and inventory.get("formal_cohort_complete") is True
        and inventory.get("safe_for_timing_binding") is True
        and inventory.get("safe_to_release_confirmation") is False,
        "compiler output inventory authority changed",
    )
    rows = inventory.get("files")
    require(isinstance(rows, list) and len(rows) == EXPECTED_BUNDLE_FILES, "compiler output inventory rows changed")
    by_relative: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "sha256", "bytes", "relative_path"}
            and isinstance(row.get("relative_path"), str)
            and row["relative_path"] not in by_relative,
            "compiler output inventory row changed",
        )
        by_relative[row["relative_path"]] = row
    require(set(by_relative) == expected_paths, "compiler output inventory coverage changed")
    for relative, observed in actual.items():
        expected = by_relative[relative]
        require(
            expected.get("path") == observed["path"]
            and expected.get("sha256") == observed["sha256"]
            and expected.get("bytes") == observed["bytes"],
            f"compiler output inventory differs at {relative}",
        )
    semantic = inventory.get("semantically_validated_nonreceipt_files")
    require(
        isinstance(semantic, list)
        and len(semantic) == EXPECTED_SEMANTIC_FILES,
        "compiler semantic inventory count changed",
    )
    require(
        all(isinstance(relative, str) for relative in semantic)
        and len(set(semantic)) == len(semantic),
        "compiler semantic inventory rows changed",
    )
    semantic_paths = set(semantic)
    require(
        semantic_paths == expected_paths - {"compiler_receipt.json"},
        "compiler semantic inventory coverage changed",
    )
    receipt_path = _verify_runtime_descriptor(
        compiler_input["compiler_receipt"], "inner compiler receipt"
    )
    require(receipt_path == bundle / "compiler_receipt.json", "inner compiler receipt escaped bundle")
    receipt = queue.load_json(receipt_path, "inner compiler receipt")
    queue.verify_signed_document(receipt, "inner compiler receipt")
    for key, expected in {
        "schema_version": COMPILER_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": "formal_full",
        "status": "compiled_complete",
        "formal_cohort_complete": True,
        "safe_for_timing_binding": True,
        "safe_for_confirmation_release": False,
        "raw_root": str(RAW_ROOT),
        "counts": {
            "compiled_cells": 32,
            "compiled_source_behavioral_requests": 1152,
            "compiled_source_behavioral_actions": 14400,
        },
        "compiler_science_activity": _compiler_zero_science_activity(),
    }.items():
        require(receipt.get(key) == expected, f"inner compiler receipt changed: {key}")
    fragments: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        expected_descriptor = compiler_input["freeze_fragments"][model]
        path = _verify_runtime_descriptor(expected_descriptor, f"{model} freeze fragment")
        relative = f"{model.lower()}_development_freeze_cells.json"
        require(path == bundle / relative and queue.file_identity(path) == actual[relative], f"{model} freeze fragment bundle binding changed")
        value = queue.load_json(path, f"{model} freeze fragment")
        queue.verify_signed_document(value, f"{model} freeze fragment")
        require(
            value.get("schema_version") == FREEZE_FRAGMENT_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("mode") == "formal_full"
            and value.get("model_id") == model
            and value.get("status") == "complete"
            and value.get("safe_for_alignment_input") is True
            and value.get("resource_receipts_synthesized") is False
            and isinstance(value.get("development_cells"), list)
            and len(value["development_cells"]) == 16,
            f"{model} freeze fragment changed",
        )
        fragments[model] = value
    return fragments


def _file_descriptor(path: Path) -> dict[str, Any]:
    return queue.file_identity(path)


def _build_release_evidence(
    *, source_root: Path, fragments: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    timing_by_model = {row["model_id"]: row for row in inputs["timing"]}
    crop_by_model = {row["model_id"]: row for row in inputs["camera"]["crop_contracts"]}
    evidence = {
        "schema_version": "wmf-development-release-evidence-v1",
        "study_id": STUDY_ID,
        "cohort_branch": "full_two_model",
        "qualified_model_ids": ["N3", "D1"],
        "ablation_spec": _file_descriptor(source_root / ABLATION_SPEC_RELATIVE),
        "planned_cells": _file_descriptor(source_root / PLANNED_CELLS_RELATIVE),
        "model_evidence": [
            {
                "model_id": model,
                "camera_crop_contract": crop_by_model[model]["contract"],
                "generated_target_timing_receipt": timing_by_model[model]["sidecar"],
                "development_cells": fragments[model]["development_cells"],
            }
            for model in MODELS
        ],
        "annotation": None,
        "resource_budget_policy": None,
    }
    require(set(evidence) == {
        "schema_version", "study_id", "cohort_branch", "qualified_model_ids",
        "ablation_spec", "planned_cells", "model_evidence", "annotation",
        "resource_budget_policy",
    }, "release evidence fields changed")
    return evidence


def _write_alignment_bundle(
    freeze: ModuleType, evidence_path: Path, bundle_path: Path
) -> dict[str, Any]:
    return freeze.write_bundle(
        evidence_path, bundle_path, confirmation_release=False
    )


def _validate_alignment_bundle(
    bundle: Path, *, freeze: ModuleType, inputs: Mapping[str, Any],
    evidence_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bundle = _reject_tree_symlinks(bundle, "physical alignment bundle")
    expected = set(SUCCESS_PUBLISH_FILES) - {PUBLISH_RECEIPT_NAME}
    actual_paths = {
        str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()
    }
    require(actual_paths == expected, "physical alignment bundle file inventory changed")
    require(
        "confirmation_release_freeze.json" not in actual_paths,
        "alignment-only bundle unexpectedly released confirmation",
    )
    try:
        expected_mappings, expected_alignment_unsigned, release_context = (
            freeze.derive_bundle(evidence_path, require_annotation=False)
        )
    except BaseException as error:
        raise DevelopmentAlignmentJobError(
            f"authoritative freeze-source replay failed: {error}"
        ) from error
    require(
        release_context is None
        and set(expected_mappings) == set(MODELS)
        and set(expected_alignment_unsigned) == set(MODELS),
        "authoritative freeze-source replay model coverage changed",
    )
    mappings: dict[str, dict[str, Any]] = {}
    alignments: dict[str, dict[str, Any]] = {}
    target_identities = {
        "N3": {"primary": (32, 32), "early": (1, 1)},
        "D1": {"primary": (2, 6), "early": (1, 3)},
    }
    target_counts = {
        "N3": {"primary": (224, 224), "early": (240, 224)},
        "D1": {"primary": (224, 224), "early": (224, 224)},
    }
    timing_inputs = {row["model_id"]: row for row in inputs["timing"]}
    crop_inputs = {
        row["model_id"]: row for row in inputs["camera"]["crop_contracts"]
    }
    for model in MODELS:
        mapping_path = bundle / OUTPUT_NAMES[model]["mapping"]
        alignment_path = bundle / OUTPUT_NAMES[model]["alignment"]
        mapping = queue.load_json(mapping_path, f"{model} physical alignment mapping")
        freeze.verify_signed(mapping, f"{model} physical alignment mapping")
        expected_mapping = expected_mappings[model]
        expected_mapping_bytes = freeze.pretty_json_bytes(expected_mapping)
        require(
            mapping == expected_mapping
            and mapping_path.read_bytes() == expected_mapping_bytes,
            f"{model} mapping differs from authoritative freeze-source replay",
        )
        require(
            set(mapping) == MAPPING_KEYS,
            f"{model} physical alignment mapping fields changed",
        )
        cell_ids = mapping.get("development_cell_ids")
        require(
            mapping.get("schema_version") == freeze.MAPPING_SCHEMA
            and mapping.get("receipt_id")
            == f"wmf1-development-{model.lower()}-physical-alignment-v1"
            and mapping.get("study_id") == STUDY_ID
            and mapping.get("model_id") == model
            and mapping.get("status")
            == "qualified_from_complete_development_native_timing"
            and isinstance(cell_ids, list)
            and len(cell_ids) == 16
            and cell_ids == sorted(set(cell_ids))
            and all(isinstance(cell_id, str) and cell_id for cell_id in cell_ids)
            and mapping.get("development_cell_count") == 16
            and mapping.get("development_request_count") == REQUEST_COUNTS[model],
            f"{model} physical alignment mapping changed",
        )
        model_limits = getattr(freeze, "MODEL_LIMITS", {}).get(model)
        require(isinstance(model_limits, Mapping), f"{model} freeze model limits are missing")
        expected_semantics = {
            key: model_limits[key] for key in REQUEST_SEMANTIC_KEYS
        }
        request_semantics = mapping.get("request_semantics")
        require(
            isinstance(request_semantics, Mapping)
            and set(request_semantics) == REQUEST_SEMANTIC_KEYS
            and request_semantics == expected_semantics,
            f"{model} physical alignment request semantics changed",
        )

        boundary = mapping.get("timing_claim_boundary")
        require(
            isinstance(boundary, Mapping)
            and set(boundary) == TIMING_BOUNDARY_KEYS[model]
            and boundary.get("generated_target_source")
            == "request_timing_sidecar.generated_targets"
            and boundary.get("time_source_kind")
            == "native_runtime_exposed_target_offsets"
            and boundary.get("clock_bridge")
            == "elapsed physical seconds from request current original-camera capture"
            and boundary.get("presentation_video_fps_used") is False
            and boundary.get("conditioning_fps_used_as_target_timing") is False
            and boundary.get("generated_frame_index_interpreted_as_action_index") is False,
            f"{model} physical alignment timing claim boundary changed",
        )
        if model == "D1":
            require(
                boundary.get("generated_targets_scope")
                == getattr(freeze, "D1_GENERATED_TARGETS_SCOPE", None)
                and boundary.get("request_timing_coverage") == EXPECTED_D1_COVERAGE
                and boundary.get(
                    "incremental_standalone_decodes_assigned_target_times"
                ) is False
                and boundary.get("timing_unmapped_requests_eligible") is False,
                "D1 frozen mapping missingness boundary changed",
            )

        camera_binding = mapping.get("camera")
        expected_dimensions = (320, 168 if model == "N3" else 176)
        require(
            isinstance(camera_binding, Mapping)
            and set(camera_binding) == CAMERA_BINDING_KEYS
            and camera_binding.get("camera_id") == "over_shoulder_left_camera"
            and camera_binding.get("camera_crop_id")
            == EXPECTED_CAMERA_CROP_IDS[model]
            and camera_binding.get("camera_crop_sha256")
            == crop_inputs[model]["contract_payload_sha256"]
            and (
                camera_binding.get("image_width_px"),
                camera_binding.get("image_height_px"),
            ) == expected_dimensions
            and isinstance(camera_binding.get("crop_operation"), str)
            and bool(camera_binding["crop_operation"]),
            f"{model} frozen mapping camera binding changed",
        )

        clocks = mapping.get("measured_clock_intervals")
        require(
            isinstance(clocks, Mapping) and set(clocks) == MEASURED_CLOCK_KEYS,
            f"{model} measured clock fields changed",
        )
        control_min = _finite_number(
            clocks.get("control_step_s_min"),
            f"{model} minimum control interval", strictly_positive=True,
        )
        control_max = _finite_number(
            clocks.get("control_step_s_max"),
            f"{model} maximum control interval", strictly_positive=True,
        )
        capture_min = _finite_number(
            clocks.get("captured_frame_interval_s_min"),
            f"{model} minimum camera interval", strictly_positive=True,
        )
        capture_max = _finite_number(
            clocks.get("captured_frame_interval_s_max"),
            f"{model} maximum camera interval", strictly_positive=True,
        )
        tolerance = _finite_number(
            clocks.get("timestamp_tolerance_s"),
            f"{model} timestamp tolerance", strictly_positive=True,
        )
        require(
            control_min <= control_max
            and capture_min <= capture_max
            and tolerance == min(control_min / 2.0, capture_min / 2.0)
            and clocks.get("tolerance_rule") == TIMESTAMP_TOLERANCE_RULE,
            f"{model} measured clock contract changed",
        )

        rows = mapping.get("frame_to_physical_time")
        expected_pairs = (
            {index: index for index in range(1, 33)}
            if model == "N3" else {1: 3, 2: 6}
        )
        require(
            isinstance(rows, list)
            and len(rows) == len(expected_pairs)
            and all(isinstance(row, Mapping) and set(row) == MAPPING_TARGET_KEYS
                    for row in rows),
            f"{model} frozen mapping row schema changed",
        )
        row_by_frame = {
            row.get("generated_frame_index"): row for row in rows
        }
        require(
            len(row_by_frame) == len(rows)
            and set(row_by_frame) == set(expected_pairs),
            f"{model} frozen mapping table coverage changed",
        )
        for frame_index, action_offset in expected_pairs.items():
            row = row_by_frame[frame_index]
            expected_eligible = (
                240 if model == "N3" and frame_index <= 2 else 224
            )
            camera_residual = _finite_number(
                row.get("max_camera_timestamp_residual_s"),
                f"{model} frame {frame_index} camera residual",
            )
            physics_residual = _finite_number(
                row.get("max_physics_timestamp_residual_s"),
                f"{model} frame {frame_index} physics residual",
            )
            _finite_number(
                row.get("target_physical_time_s"),
                f"{model} frame {frame_index} target time",
                strictly_positive=True,
            )
            require(
                row.get("status") == "qualified"
                and row.get("target_executed_action_offset") == action_offset
                and row.get("eligible_request_count") == expected_eligible
                and row.get("full_prefix_request_count") == 224
                and camera_residual <= tolerance
                and physics_residual <= tolerance,
                f"{model} frame {frame_index} coverage/residual contract changed",
            )

        primary = mapping.get("primary_target")
        early = mapping.get("early_target")
        require(
            isinstance(primary, Mapping)
            and isinstance(early, Mapping)
            and set(primary) == MAPPING_TARGET_KEYS
            and set(early) == MAPPING_TARGET_KEYS
            and primary
            == row_by_frame[target_identities[model]["primary"][0]]
            and early == row_by_frame[target_identities[model]["early"][0]]
            and (
                primary.get("generated_frame_index"),
                primary.get("target_executed_action_offset"),
            ) == target_identities[model]["primary"]
            and (
                early.get("generated_frame_index"),
                early.get("target_executed_action_offset"),
            ) == target_identities[model]["early"]
            and (
                primary.get("eligible_request_count"),
                primary.get("full_prefix_request_count"),
            ) == target_counts[model]["primary"]
            and (
                early.get("eligible_request_count"),
                early.get("full_prefix_request_count"),
            ) == target_counts[model]["early"],
            f"{model} frozen alignment target identity changed",
        )

        source_receipts = mapping.get("source_receipts")
        require(
            isinstance(source_receipts, Mapping)
            and set(source_receipts) == SOURCE_RECEIPT_KEYS,
            f"{model} frozen mapping source receipt fields changed",
        )
        source_paths: list[str] = []
        for field in (
            "development_cell_receipts", "adapter_completions", "adapter_journals"
        ):
            descriptors = source_receipts.get(field)
            require(
                isinstance(descriptors, list) and len(descriptors) == 16,
                f"{model} {field} inventory changed",
            )
            for ordinal, descriptor in enumerate(descriptors):
                normalized_descriptor = _descriptor_shape(
                    descriptor, f"{model} {field} {ordinal}"
                )
                source_paths.append(normalized_descriptor["path"])
        request_hashes = source_receipts.get("official_request_receipt_sha256s")
        require(
            isinstance(request_hashes, list)
            and len(request_hashes) == REQUEST_COUNTS[model]
            and len(set(request_hashes)) == REQUEST_COUNTS[model]
            and all(
                isinstance(digest, str)
                and queue.SHA256_RE.fullmatch(digest) is not None
                for digest in request_hashes
            )
            and len(source_paths) == len(set(source_paths))
            and source_receipts.get("resource_receipts") == [None] * 16
            and source_receipts.get("generated_target_timing")
            == timing_inputs[model]["sidecar"]
            and source_receipts.get("camera_crop_contract")
            == crop_inputs[model]["contract"],
            f"{model} frozen mapping source receipt binding changed",
        )
        alignment = queue.load_json(alignment_path, f"{model} alignment contract")
        expected_alignment = dict(expected_alignment_unsigned[model])
        expected_alignment["mapping_receipt_sha256"] = queue.sha256_bytes(
            expected_mapping_bytes
        )
        expected_alignment["contract_sha256"] = freeze.sha256_bytes(
            freeze.canonical_bytes(expected_alignment)
        )
        require(
            alignment == expected_alignment
            and alignment_path.read_bytes()
            == freeze.pretty_json_bytes(expected_alignment),
            f"{model} alignment differs from authoritative freeze-source replay",
        )
        require(
            set(alignment) == ALIGNMENT_KEYS,
            f"{model} alignment contract fields changed",
        )
        contract_hash = alignment.get("contract_sha256")
        require(
            isinstance(contract_hash, str)
            and queue.SHA256_RE.fullmatch(contract_hash) is not None,
            f"{model} alignment contract hash is invalid",
        )
        unsigned = dict(alignment)
        unsigned.pop("contract_sha256")
        early_contract = alignment.get("early_horizon")
        require(
            isinstance(early_contract, Mapping)
            and set(early_contract) == EARLY_HORIZON_KEYS,
            f"{model} early alignment fields changed",
        )
        for field in (
            "primary_horizon_s", "control_step_s", "captured_frame_interval_s",
            "timestamp_tolerance_s",
        ):
            _finite_number(
                alignment.get(field), f"{model} alignment {field}",
                strictly_positive=True,
            )
        require(
            freeze.sha256_bytes(freeze.canonical_bytes(unsigned)) == contract_hash
            and alignment.get("contract_id") == f"wmf1-{model.lower()}-alignment-v1"
            and alignment.get("model_id") == model
            and alignment.get("mapping_receipt_id") == mapping.get("receipt_id")
            and alignment.get("mapping_receipt_sha256") == queue.sha256_file(mapping_path)
            and alignment.get("primary_horizon_s")
            == primary["target_physical_time_s"]
            and alignment.get("generated_frame_index") == primary["generated_frame_index"]
            and alignment.get("target_executed_action_offset")
            == primary["target_executed_action_offset"]
            and alignment.get("control_step_s") == control_min
            and alignment.get("captured_frame_interval_s") == capture_min
            and alignment.get("timestamp_tolerance_s") == tolerance
            and alignment.get("camera_id") == camera_binding["camera_id"]
            and alignment.get("camera_crop_id") == camera_binding["camera_crop_id"]
            and alignment.get("camera_crop_sha256")
            == camera_binding["camera_crop_sha256"]
            and (
                alignment.get("image_width_px"),
                alignment.get("image_height_px"),
            ) == expected_dimensions
            and dict(early_contract) == {
                "horizon_s": early["target_physical_time_s"],
                "generated_frame_index": early["generated_frame_index"],
                "target_executed_action_offset": early[
                    "target_executed_action_offset"
                ],
            },
            f"{model} alignment/mapping binding changed",
        )
        mappings[model] = mapping
        alignments[model] = alignment
    return mappings, alignments


def _write_raw_manifest(
    *, path: Path, inputs: Mapping[str, Any], input_identity: Mapping[str, Any],
    evidence_path: Path, bundle: Path,
) -> dict[str, Any]:
    outputs = {
        model: {
            role: queue.file_identity(bundle / OUTPUT_NAMES[model][role])
            for role in ("mapping", "alignment")
        }
        for model in MODELS
    }
    manifest = queue.signed_document({
        "schema_version": RAW_MANIFEST_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "job_id": JOB_ID,
        "mode": MODE,
        "status": "alignment_qualified_raw_evidence_complete",
        "input_manifest": dict(input_identity),
        "source_results_commit": inputs["results_commit"],
        "source_evidence": dict(inputs),
        "development_release_evidence": queue.file_identity(evidence_path),
        "raw_alignment_outputs": outputs,
        "counts": {
            "models": 2,
            "development_cells": 32,
            "source_requests": 1152,
            "mapping_receipts": 2,
            "alignment_contracts": 2,
            "labels_created": 0,
        },
        "science_counts": _zero_science_counts(),
        "annotation_authority": False,
        "resource_freeze_authority": False,
        "confirmation_authority": False,
        "safe_to_release_confirmation": False,
        "raw_outputs_recoverable_on_gm_pvc": True,
    })
    queue.immutable_json(path, manifest, maximum_bytes=2 * 1024 * 1024)
    return manifest


def _publish_transactionally(
    *, raw: Path, publish: Path, bundle: Path, context: Any,
    implementation: Mapping[str, Mapping[str, Any]], inputs: Mapping[str, Any],
    input_identity: Mapping[str, Any], evidence_identity: Mapping[str, Any],
    raw_manifest_identity: Mapping[str, Any],
    receipt_writer: Callable[..., None] | None = None,
) -> dict[str, Any]:
    require(publish.is_dir() and not publish.is_symlink() and not any(publish.iterdir()), "alignment publish directory is not empty")
    staging = raw.parent / STAGING_RELATIVE
    require(not staging.exists() and not staging.is_symlink(), "alignment publish staging already exists")
    staging.mkdir()
    writer = queue.immutable_json if receipt_writer is None else receipt_writer
    published_outputs: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        published_outputs[model] = {}
        for role in ("mapping", "alignment"):
            name = OUTPUT_NAMES[model][role]
            source = bundle / name
            payload = source.read_bytes()
            queue.immutable_bytes(
                staging / name, payload, maximum_bytes=PUBLISH_FILE_LIMIT_BYTES
            )
            staged = queue.file_identity(staging / name)
            original = queue.file_identity(source)
            require(
                staged["sha256"] == original["sha256"]
                and staged["bytes"] == original["bytes"],
                f"published {model} {role} differs from raw bundle",
            )
            published_outputs[model][role] = {
                "path": str(publish / name),
                "sha256": staged["sha256"],
                "bytes": staged["bytes"],
            }
    receipt = queue.signed_document({
        "schema_version": JOB_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": MODE,
        "status": "passed",
        "decision": "alignment_qualified_for_annotation_inventory_only",
        "job_id": context.job_id,
        "job_dir": str(context.job_dir),
        "study_commit": context.study_commit,
        "queue_role": context.role,
        "worker_id": context.worker_id,
        "runtime_identity": {
            "hostname": context.hostname,
            "pod_uid": context.pod_uid,
            "pid": os.getpid(),
        },
        "queue_descriptor": context.descriptor_identity,
        "queue_claim": context.claim_identity,
        "implementation": dict(implementation),
        "inputs": dict(inputs),
        "input_manifest": dict(input_identity),
        "development_release_evidence": dict(evidence_identity),
        "raw_evidence_manifest": dict(raw_manifest_identity),
        "outputs": published_outputs,
        "counts": {
            "models": 2,
            "development_cells": 32,
            "source_requests": 1152,
            "mapping_receipts": 2,
            "alignment_contracts": 2,
            "labels_created": 0,
        },
        "science_counts": _zero_science_counts(),
        "alignment_qualified": True,
        "safe_for_annotation_inventory_input": True,
        "resource_measurements_complete": False,
        "annotation_complete": False,
        "safe_for_rater_distribution": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "behavioral_policy_skill_evaluated": False,
        "annotation_authority": False,
        "resource_freeze_authority": False,
        "confirmation_authority": False,
        "raw_outputs_recoverable_on_gm_pvc": True,
        "published_files": list(SUCCESS_PUBLISH_FILES),
        "claim_boundary": (
            "Alignment-only CPU derivation. Outputs may feed the development "
            "annotation inventory; no label, resource freeze, rater packet, "
            "behavioral result, or confirmation release is authorized."
        ),
        "completed_at_utc": queue.utc_now(),
    })
    writer(staging / PUBLISH_RECEIPT_NAME, receipt, maximum_bytes=2 * 1024 * 1024)
    require(
        {path.name for path in staging.iterdir()} == set(SUCCESS_PUBLISH_FILES),
        "alignment staged publish inventory changed",
    )
    require(
        all(
            path.is_file() and not path.is_symlink()
            and path.stat().st_size <= PUBLISH_FILE_LIMIT_BYTES
            for path in staging.iterdir()
        )
        and sum(path.stat().st_size for path in staging.iterdir())
        <= PUBLISH_JOB_LIMIT_BYTES,
        "alignment staged publication exceeds publisher limits",
    )
    publish.rmdir()
    os.replace(staging, publish)
    queue._fsync_directory(context.job_dir)
    return receipt


def _write_failure(job_dir: Path, context: Any | None, error: BaseException) -> None:
    expected = CONTROL_ROOT / "jobs" / JOB_ID
    try:
        if Path(job_dir) != expected or Path(job_dir).resolve() != expected:
            return
        publish = expected / "publish"
        if publish.exists() and not publish.is_dir():
            return
        if publish.exists() and (publish / PUBLISH_RECEIPT_NAME).exists():
            return
        if not publish.exists():
            publish.mkdir()
        if any(publish.iterdir()):
            return
        failure = queue.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": MODE,
            "status": "technical_invalid",
            "decision": "no_go",
            "job_id": JOB_ID,
            "study_commit": context.study_commit if context is not None else None,
            "queue_role": context.role if context is not None else WORKER_ROLE,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(limit=20)[-12000:],
            },
            "science_counts": _zero_science_counts(),
            "labels_created": 0,
            "annotation_authority": False,
            "resource_freeze_authority": False,
            "confirmation_authority": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(publish / PUBLISH_FAILURE_NAME, failure)
    except BaseException:
        return


def _expected_context_paths(context: Any) -> None:
    require(
        context.job_dir == CONTROL_ROOT / "jobs" / JOB_ID,
        "alignment job directory changed",
    )
    require(
        context.source_root == CONTROL_ROOT / "sources" / context.study_commit,
        "alignment source checkout changed",
    )


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    descriptor, expected_implementation, expected_inputs = _runtime_descriptor(args)
    context = None
    try:
        context = queue.validate_queue_context(
            source_root=args.source_root,
            job_dir=args.job_dir,
            study_commit=args.study_commit,
            job_id=args.job_id,
            expected_role=args.expected_role,
            expected_descriptor=descriptor,
        )
        _expected_context_paths(context)
        implementation, freeze, timing, camera, compiler_queue = (
            _validate_staged_implementation(context.source_root, expected_implementation)
        )
        inputs, _, _ = _validate_runtime_receipts(
            expected_inputs, timing=timing, camera=camera
        )
        fragments = _validate_compiler_bundle(
            inputs["compiler"], compiler_queue=compiler_queue
        )
        raw, publish = queue._prepare_output_directories(context.job_dir)
        input_path = context.job_dir / INPUT_RELATIVE
        evidence_path = context.job_dir / EVIDENCE_RELATIVE
        bundle_path = context.job_dir / BUNDLE_RELATIVE
        manifest_path = context.job_dir / RAW_MANIFEST_RELATIVE
        require(
            input_path.parent == evidence_path.parent == bundle_path.parent == manifest_path.parent == raw,
            "alignment raw output paths changed",
        )
        queue.immutable_json(input_path, inputs, maximum_bytes=2 * 1024 * 1024)
        input_identity = queue.file_identity(input_path)
        evidence = _build_release_evidence(
            source_root=context.source_root, fragments=fragments, inputs=inputs
        )
        queue.immutable_json(evidence_path, evidence, maximum_bytes=4 * 1024 * 1024)
        evidence_identity = queue.file_identity(evidence_path)
        returned = _write_alignment_bundle(freeze, evidence_path, bundle_path)
        require(
            returned.get("status") == "alignment_qualified"
            and returned.get("models") == ["N3", "D1"]
            and "release_freeze" not in returned,
            "freeze alignment-only return value changed",
        )
        _validate_alignment_bundle(
            bundle_path, freeze=freeze, inputs=inputs,
            evidence_path=evidence_path,
        )
        _write_raw_manifest(
            path=manifest_path,
            inputs=inputs,
            input_identity=input_identity,
            evidence_path=evidence_path,
            bundle=bundle_path,
        )
        raw_manifest_identity = queue.file_identity(manifest_path)
        return _publish_transactionally(
            raw=raw,
            publish=publish,
            bundle=bundle_path,
            context=context,
            implementation=implementation,
            inputs=inputs,
            input_identity=input_identity,
            evidence_identity=evidence_identity,
            raw_manifest_identity=raw_manifest_identity,
        )
    except BaseException as error:
        _write_failure(Path(args.job_dir), context, error)
        raise


def _add_local_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-commit", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    for stem in (
        "compiler-job-receipt", "n3-timing-job-receipt", "d1-timing-job-receipt",
        "camera-job-receipt", "n3-camera-crop-contract", "d1-camera-crop-contract",
    ):
        parser.add_argument(f"--{stem}", type=Path, required=True)
        parser.add_argument(f"--{stem}-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-wave", help="emit one receipt-gated descriptor; never dispatch")
    build.add_argument("--study-commit", required=True)
    build.add_argument("--output", type=Path)
    _add_local_inputs(build)
    run = sub.add_parser("run", help="execute in one claimed detached CPU worker")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--study-commit", required=True)
    run.add_argument("--job-dir", type=Path, required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--expected-role", required=True)
    run.add_argument("--inputs-json", required=True)
    for name in sorted(_source_paths(REPOSITORY_ROOT)):
        run.add_argument(f"--{name.replace('_', '-')}-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run_job(args)
        return 0
    wave = build_wave_from_local_files(args)
    payload = queue.canonical_bytes(wave)
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        target = Path(args.output)
        require(not target.exists() and not target.is_symlink(), "refusing to replace wave output")
        queue.immutable_bytes(target, payload, maximum_bytes=4 * 1024 * 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
