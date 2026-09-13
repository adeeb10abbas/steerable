#!/usr/bin/env python3
"""Build and run the detached formal development-evidence compiler job.

The descriptor builder accepts eight explicit, hash-bound aggregate receipts
from the fetched results branch.  It emits one queue descriptor only after the
four N3 and four D1 aggregates independently pass the existing strict
aggregate validator.  It never edits the active queue.

The runtime wrapper repeats every gate from the immutable PVC copies, verifies
the queue descriptor, claim, worker role, POD_UID, clean staged source commit,
and all implementation/contract hashes, then writes the compiler manifest and
bundle only below its unique queue job directory.  Only a compact signed job
receipt is publishable.  This wrapper supports formal_full only and can never
release confirmation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
QUEUE_MODULE_PATH = Path(__file__).with_name("forecast_timing_queue_jobs.py")
AGGREGATE_SUPPORT_PATH = Path(__file__).with_name(
    "development_timing_sidecar_jobs.py"
)


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


queue = _load_module(QUEUE_MODULE_PATH, "wmf_development_compiler_queue_support")
aggregate_support = _load_module(
    AGGREGATE_SUPPORT_PATH, "wmf_development_compiler_aggregate_support"
)

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
RAW_ROOT = CONTROL_ROOT.parent
ROBOLAB_PYTHON = queue.ROBOLAB_PYTHON

THIS_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_evidence_compiler_jobs.py"
)
QUEUE_RELATIVE = queue.THIS_RELATIVE
AGGREGATE_SUPPORT_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_timing_sidecar_jobs.py"
)
CONTRACT_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_evidence_compiler_queue_contract.json"
)
COMPILER_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/compile_development_evidence.py"
)
FREEZE_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/freeze_development_release.py"
)
TIMING_VALIDATOR_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/qualify_forecast_timing.py"
)
ANNOTATION_VALIDATOR_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/forecast_annotation_workflow.py"
)
PLANNED_CELLS_RELATIVE = (
    queue.FORECAST_RELATIVE / "experiments/forecast_layout/planned_cells.csv"
)

WAVE_SCHEMA = "wmf-development-evidence-compiler-wave-v1"
CONTRACT_SCHEMA = "wmf-development-evidence-compiler-queue-contract-v1"
JOB_RECEIPT_SCHEMA = "wmf-development-evidence-compiler-queue-job-v1"
OUTPUT_INVENTORY_SCHEMA = "wmf-development-evidence-compiler-output-inventory-v1"
COMPILER_INPUT_SCHEMA = "wmf-development-evidence-compiler-input-v1"
COMPILER_RECEIPT_SCHEMA = "wmf-development-evidence-compiler-receipt-v1"

JOB_ID = "development-evidence-compiler-formal-001"
WORKER_ROLE = "wmf-forecast-0912-worker-05"
MODE = "formal_full"
CAMERA_ID = "over_shoulder_left_camera"
MAX_WALL_SECONDS = 21600
PUBLISH_LOG_TAIL_BYTES = 8192
MANIFEST_RELATIVE = Path("raw/compiler_input_manifest.json")
BUNDLE_RELATIVE = Path("raw/compiler_bundle")
PUBLISH_RECEIPT_NAME = "development_evidence_compiler_job_receipt.json"
PUBLISH_FAILURE_NAME = "development_evidence_compiler_job_failure.json"
PLANNED_CELLS_SHA256 = (
    "7d06120a56419877d1acdfdc498c6dc054bf6860ce6bda5b2a25f55ae3b4166e"
)
MODELS = ("N3", "D1")
LAYOUTS = ("D01", "D02", "D03", "D04")
EXPECTED_CELLS = 32
EXPECTED_REQUESTS = 1152
EXPECTED_ACTIONS = 14400
EXPECTED_BUNDLE_FILES = 71
MODEL_REQUESTS = {"N3": 240, "D1": 912}
MODEL_ACTIONS = {"N3": 7200, "D1": 7200}

TIMING_INVENTORY_KEYS = {
    "schema_version",
    "study_id",
    "model_id",
    "request_receipts",
}
TIMING_ENTRY_KEYS = {
    "cell_id",
    "request_index",
    "request_receipt",
    "adapter_completion",
    "adapter_journal",
}
PROVENANCE_KEYS = {
    "schema_version",
    "study_id",
    "mode",
    "model_id",
    "stage",
    "status",
    "safe_for_formal_release",
    "annotation_state",
    "camera_id",
    "camera_crop_contract",
    "resource_measurements",
    "labels",
    "episode_roster",
    "requests",
    "payload_sha256",
}
PROVENANCE_REQUEST_KEYS = {
    "source_request_id",
    "cell_id",
    "recording_id",
    "model_id",
    "layout_pair_id",
    "condition_id",
    "request_index",
    "action_step_start",
    "executed_prefix_actions",
    "current_observation_id",
    "preceding_observation_id",
    "history_mode",
    "current_observation",
    "preceding_observation",
    "official_request_receipt",
    "recorder_model_request",
    "recorder_response_payload_sha256",
    "adapter_completion",
    "adapter_journal",
    "source_video_id",
    "source_video",
    "model_output_or_action_modified",
    "model_identity",
    "model_context",
    "action_manifest",
    "recording_receipt",
}
OBSERVATION_KEYS = {
    "observation_id",
    "control_step",
    "physics_step",
    "physics_time_s",
    "camera_frame_native_id",
    "camera_frame_id",
    "camera_capture_time_ns",
    "camera_timestamp_source",
    "payload_sha256",
    "payload_artifact",
}
FREEZE_FRAGMENT_KEYS = {
    "schema_version",
    "study_id",
    "mode",
    "model_id",
    "status",
    "safe_for_alignment_input",
    "resource_receipts_synthesized",
    "development_cells",
    "payload_sha256",
}
MODEL_OUTPUT_CELL_KEYS = {
    "cell_id",
    "source_cell_receipt",
    "action_manifest",
    "recording_receipt",
    "official_request_count",
    "source_video",
}


class DevelopmentCompilerQueueError(RuntimeError):
    """A detached compiler release or runtime evidence gate failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentCompilerQueueError(message)


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


def _compiler_zero_science() -> dict[str, int]:
    return {
        "model_loads": 0,
        "model_requests_issued": 0,
        "simulator_processes_started": 0,
        "physical_resets": 0,
        "behavioral_actions_executed": 0,
        "labels_created": 0,
    }


def _aggregate_job_ids() -> dict[str, dict[str, str]]:
    return {
        model: {
            layout: aggregate_support.AGGREGATES[model][layout].job_id
            for layout in LAYOUTS
        }
        for model in MODELS
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
            "input_manifest_relative": str(MANIFEST_RELATIVE),
            "compiler_output_relative": str(BUNDLE_RELATIVE),
        },
        "compiler": {
            "input_schema": COMPILER_INPUT_SCHEMA,
            "receipt_schema": COMPILER_RECEIPT_SCHEMA,
            "camera_id": CAMERA_ID,
            "planned_cells_sha256": PLANNED_CELLS_SHA256,
            "expected_compiled_cells": EXPECTED_CELLS,
            "expected_source_behavioral_requests": EXPECTED_REQUESTS,
            "expected_source_behavioral_actions": EXPECTED_ACTIONS,
            "supported_production_modes": [MODE],
            "semantic_validator_paths": {
                "freeze": str(FREEZE_RELATIVE),
                "timing": str(TIMING_VALIDATOR_RELATIVE),
                "annotation": str(ANNOTATION_VALIDATOR_RELATIVE),
            },
        },
        "aggregate_jobs": _aggregate_job_ids(),
        "output_policy": {
            "success_publish_files": [PUBLISH_RECEIPT_NAME],
            "failure_publish_files": [PUBLISH_FAILURE_NAME],
            "mutually_exclusive_terminal_receipts": True,
            "raw_bundle_file_count": EXPECTED_BUNDLE_FILES,
            "semantically_validated_nonreceipt_file_count": (
                EXPECTED_BUNDLE_FILES - 1
            ),
            "diagnostic_release_capable": False,
        },
        "science_counts": _zero_science_counts(),
        "safe_to_release_confirmation": False,
    }


def _verified_sha(value: Any, label: str) -> str:
    try:
        return queue._verified_sha(value, label)
    except BaseException as error:
        raise DevelopmentCompilerQueueError(str(error)) from error


def _validate_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = _verified_sha(expected_sha256, "compiler queue contract digest")
    identity = queue.file_identity(path)
    require(identity["sha256"] == expected, "compiler queue contract hash changed")
    contract = queue.load_json(path, "compiler queue contract")
    require(contract == _expected_contract(), "compiler queue contract fields changed")
    return identity


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "queue_wrapper": root / THIS_RELATIVE,
        "queue_support": root / QUEUE_RELATIVE,
        "aggregate_support": root / AGGREGATE_SUPPORT_RELATIVE,
        "queue_contract": root / CONTRACT_RELATIVE,
        "compiler": root / COMPILER_RELATIVE,
        "freeze_validator": root / FREEZE_RELATIVE,
        "timing_validator": root / TIMING_VALIDATOR_RELATIVE,
        "annotation_validator": root / ANNOTATION_VALIDATOR_RELATIVE,
        "planned_cells": root / PLANNED_CELLS_RELATIVE,
    }


def _local_implementation() -> dict[str, dict[str, Any]]:
    identities = {
        name: queue.file_identity(path)
        for name, path in _source_paths(REPOSITORY_ROOT).items()
    }
    require(
        identities["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256,
        "planned-cell CSV hash changed",
    )
    _validate_contract(
        Path(identities["queue_contract"]["path"]),
        identities["queue_contract"]["sha256"],
    )
    return identities


def _aggregate_key(model: str, layout: str) -> tuple[str, str]:
    require(model in MODELS and layout in LAYOUTS, "aggregate identity is unsupported")
    return model, layout


def _expected_aggregate_keys() -> set[tuple[str, str]]:
    return {(model, layout) for model in MODELS for layout in LAYOUTS}


def validate_local_aggregate_inputs(
    inputs: Mapping[tuple[str, str], tuple[Path, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate eight explicitly selected fetched receipts without discovering one."""

    require(
        set(inputs) == _expected_aggregate_keys(),
        "formal compiler requires exactly eight explicit N3/D1 aggregate receipts",
    )
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model in MODELS:
        for layout in LAYOUTS:
            key = _aggregate_key(model, layout)
            supplied_path, supplied_sha256 = inputs[key]
            _verified_sha(supplied_sha256, f"{model} {layout} aggregate digest")
            spec = aggregate_support.AGGREGATES[model][layout]
            try:
                evidence = aggregate_support.validate_local_aggregate_receipt(
                    Path(supplied_path), supplied_sha256, spec
                )
            except BaseException as error:
                raise DevelopmentCompilerQueueError(
                    f"{model} {layout} aggregate gate failed: {error}"
                ) from error
            descriptor = evidence.get("aggregate_receipt")
            require(
                isinstance(descriptor, Mapping),
                f"{model} {layout} aggregate descriptor is missing",
            )
            result[key] = {
                "aggregate_receipt": dict(descriptor),
                "cell_receipts": list(evidence.get("cell_receipts", [])),
            }
    paths = [
        value["aggregate_receipt"]["path"] for value in result.values()
    ]
    require(len(paths) == len(set(paths)), "aggregate receipt paths are ambiguous")
    return result


def _hash_argv(implementation: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        "--queue-wrapper-sha256",
        str(implementation["queue_wrapper"]["sha256"]),
        "--queue-support-sha256",
        str(implementation["queue_support"]["sha256"]),
        "--aggregate-support-sha256",
        str(implementation["aggregate_support"]["sha256"]),
        "--queue-contract-sha256",
        str(implementation["queue_contract"]["sha256"]),
        "--compiler-sha256",
        str(implementation["compiler"]["sha256"]),
        "--freeze-validator-sha256",
        str(implementation["freeze_validator"]["sha256"]),
        "--timing-validator-sha256",
        str(implementation["timing_validator"]["sha256"]),
        "--annotation-validator-sha256",
        str(implementation["annotation_validator"]["sha256"]),
        "--planned-cells-sha256",
        str(implementation["planned_cells"]["sha256"]),
    ]


def _aggregate_argv(
    aggregates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[str]:
    argv: list[str] = []
    for model in MODELS:
        for layout in LAYOUTS:
            descriptor = aggregates[(model, layout)]["aggregate_receipt"]
            prefix = f"--{model.lower()}-{layout.lower()}-aggregate"
            argv.extend(
                [
                    prefix,
                    str(descriptor["path"]),
                    prefix + "-sha256",
                    str(descriptor["sha256"]),
                    prefix + "-bytes",
                    str(descriptor["bytes"]),
                ]
            )
    return argv


def _job_descriptor(
    *,
    study_commit: str,
    implementation: Mapping[str, Mapping[str, Any]],
    aggregates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(
        set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
        "compiler implementation identity inventory changed",
    )
    require(
        set(aggregates) == _expected_aggregate_keys(),
        "compiler aggregate identity inventory changed",
    )
    argv = [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        "formal-full",
        "--source-root",
        "{source_root}",
        "--study-commit",
        commit,
        "--job-dir",
        "{job_dir}",
        "--job-id",
        JOB_ID,
        "--expected-role",
        WORKER_ROLE,
    ]
    argv.extend(_hash_argv(implementation))
    argv.extend(_aggregate_argv(aggregates))
    return {
        "job_id": JOB_ID,
        "released": True,
        "source_commit": commit,
        "role": WORKER_ROLE,
        "argv": argv,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def build_formal_wave(
    *,
    study_commit: str,
    aggregate_inputs: Mapping[tuple[str, str], tuple[Path, str]],
) -> dict[str, Any]:
    """Return one formal descriptor after all eight receipts pass; never dispatch."""

    implementation = _local_implementation()
    aggregates = validate_local_aggregate_inputs(aggregate_inputs)
    descriptor = _job_descriptor(
        study_commit=study_commit,
        implementation=implementation,
        aggregates=aggregates,
    )
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": MODE,
        "source_commit": descriptor["source_commit"],
        "status": "descriptor_only_not_dispatched_all_eight_aggregate_gates_passed",
        "jobs": [descriptor],
        "implementation": implementation,
        "aggregate_receipts": [
            {
                "model_id": model,
                "layout_pair_id": layout,
                "receipt": aggregates[(model, layout)]["aggregate_receipt"],
            }
            for model in MODELS
            for layout in LAYOUTS
        ],
        "expected_compiled_cells": EXPECTED_CELLS,
        "expected_source_behavioral_requests": EXPECTED_REQUESTS,
        "expected_source_behavioral_actions": EXPECTED_ACTIONS,
        "science_counts": _zero_science_counts(),
        "safe_for_timing_binding_after_pass": True,
        "safe_to_release_confirmation": False,
        "diagnostic_release_capable": False,
        "claim_boundary": (
            "One CPU-only formal compiler job over eight explicit immutable passed "
            "development aggregates. It cannot issue science or release confirmation."
        ),
    }


def _runtime_implementation(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    values = {
        "queue_wrapper": args.queue_wrapper_sha256,
        "queue_support": args.queue_support_sha256,
        "aggregate_support": args.aggregate_support_sha256,
        "queue_contract": args.queue_contract_sha256,
        "compiler": args.compiler_sha256,
        "freeze_validator": args.freeze_validator_sha256,
        "timing_validator": args.timing_validator_sha256,
        "annotation_validator": args.annotation_validator_sha256,
        "planned_cells": args.planned_cells_sha256,
    }
    return {
        name: {"sha256": _verified_sha(digest, f"{name} digest")}
        for name, digest in values.items()
    }


def _runtime_aggregates(
    args: argparse.Namespace,
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model in MODELS:
        for layout in LAYOUTS:
            stem = f"{model.lower()}_{layout.lower()}_aggregate"
            path = Path(getattr(args, stem))
            digest = _verified_sha(
                getattr(args, stem + "_sha256"), f"{model} {layout} aggregate digest"
            )
            size = getattr(args, stem + "_bytes")
            require(type(size) is int and size > 0, f"{model} {layout} byte count is invalid")
            spec = aggregate_support.AGGREGATES[model][layout]
            require(path == spec.receipt_path, f"{model} {layout} aggregate path changed")
            result[(model, layout)] = {
                "aggregate_receipt": {
                    "path": str(path),
                    "sha256": digest,
                    "bytes": size,
                }
            }
    paths = [row["aggregate_receipt"]["path"] for row in result.values()]
    require(len(paths) == len(set(paths)), "runtime aggregate receipt paths are ambiguous")
    return result


def _runtime_descriptor(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    require(args.command == "formal-full", "runtime compiler mode is not formal_full")
    require(args.job_id == JOB_ID, "runtime compiler job ID changed")
    require(args.expected_role == WORKER_ROLE, "runtime compiler worker role changed")
    implementation = _runtime_implementation(args)
    require(
        implementation["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256,
        "runtime planned-cell hash changed",
    )
    aggregates = _runtime_aggregates(args)
    descriptor = _job_descriptor(
        study_commit=args.study_commit,
        implementation=implementation,
        aggregates=aggregates,
    )
    return descriptor, implementation, aggregates


def _validate_staged_implementation(
    *,
    source_root: Path,
    expected: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    ModuleType,
    ModuleType,
    ModuleType,
]:
    paths = _source_paths(source_root)
    require(set(expected) == set(paths), "runtime implementation inventory changed")
    observed: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        identity = queue.file_identity(path)
        require(
            identity["sha256"] == expected[name]["sha256"],
            f"staged {name} hash changed",
        )
        observed[name] = identity
    require(
        observed["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256,
        "staged planned-cell CSV is not authoritative",
    )
    _validate_contract(
        Path(observed["queue_contract"]["path"]),
        expected["queue_contract"]["sha256"],
    )
    compiler = _load_module(
        Path(observed["compiler"]["path"]),
        "wmf_detached_development_compiler",
    )
    exact_interface = {
        "INPUT_SCHEMA": COMPILER_INPUT_SCHEMA,
        "COMPILER_SCHEMA": COMPILER_RECEIPT_SCHEMA,
        "PLANNED_CELLS_SHA256": PLANNED_CELLS_SHA256,
        "STUDY_ID": STUDY_ID,
        "MODELS": MODELS,
        "LAYOUT_IDS": LAYOUTS,
    }
    for name, wanted in exact_interface.items():
        require(getattr(compiler, name, None) == wanted, f"compiler interface changed: {name}")
    require(
        CAMERA_ID in getattr(compiler, "PRIMARY_CAMERA_CHOICES", ()),
        "compiler camera interface changed",
    )
    require(
        Path(getattr(compiler, "FREEZE_PATH", Path("missing"))).resolve()
        == Path(observed["freeze_validator"]["path"]),
        "compiler freeze-validator path changed",
    )
    freeze_module_path = Path(
        getattr(getattr(compiler, "freeze", None), "__file__", "missing")
    ).resolve()
    require(
        freeze_module_path == Path(observed["freeze_validator"]["path"]),
        "compiler loaded another freeze validator",
    )
    timing = _load_module(
        Path(observed["timing_validator"]["path"]),
        "wmf_detached_development_timing_validator",
    )
    annotation = _load_module(
        Path(observed["annotation_validator"]["path"]),
        "wmf_detached_development_annotation_validator",
    )
    require(
        getattr(timing, "STUDY_ID", None) == STUDY_ID
        and getattr(timing, "REQUEST_INVENTORY_SCHEMA", None)
        == getattr(compiler, "TIMING_INVENTORY_SCHEMA", None),
        "timing-validator interface changed",
    )
    require(
        getattr(annotation, "STUDY_ID", None) == STUDY_ID
        and getattr(annotation, "ACTION_MANIFEST_SCHEMA", None)
        == getattr(compiler, "ACTION_MANIFEST_SCHEMA", None)
        and getattr(annotation, "RECORDING_RECEIPT_SCHEMA", None)
        == getattr(compiler, "RECORDING_RECEIPT_SCHEMA", None),
        "annotation-validator interface changed",
    )
    require(
        getattr(compiler, "PROVENANCE_SCHEMA", None)
        == "wmf-development-request-provenance-v1"
        and getattr(compiler, "FREEZE_FRAGMENT_SCHEMA", None)
        == "wmf-development-freeze-cell-evidence-fragment-v1",
        "compiler semantic-output interface changed",
    )
    return observed, compiler, timing, annotation


def _validate_runtime_aggregates(
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model in MODELS:
        for layout in LAYOUTS:
            key = (model, layout)
            descriptor = expected[key]["aggregate_receipt"]
            path = Path(descriptor["path"])
            identity = queue.file_identity(path)
            require(identity == descriptor, f"{model} {layout} aggregate bytes/hash changed")
            spec = aggregate_support.AGGREGATES[model][layout]
            try:
                evidence = aggregate_support.validate_local_aggregate_receipt(
                    path, descriptor["sha256"], spec
                )
            except BaseException as error:
                raise DevelopmentCompilerQueueError(
                    f"runtime {model} {layout} aggregate gate failed: {error}"
                ) from error
            require(
                evidence.get("aggregate_receipt") == descriptor,
                f"{model} {layout} aggregate descriptor changed after validation",
            )
            cells = evidence.get("cell_receipts")
            require(
                isinstance(cells, list) and len(cells) == 4,
                f"{model} {layout} cell descriptor inventory changed",
            )
            result[key] = {
                "aggregate_receipt": descriptor,
                "cell_receipts": list(cells),
            }
    return result


def _compiler_manifest(
    *,
    source_root: Path,
    aggregates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    planned = queue.file_identity(source_root / PLANNED_CELLS_RELATIVE)
    require(planned["sha256"] == PLANNED_CELLS_SHA256, "manifest planned-cell hash changed")
    return {
        "schema_version": COMPILER_INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "raw_root": str(RAW_ROOT),
        "camera_id": CAMERA_ID,
        "planned_cells": planned,
        "aggregate_receipts": [
            {
                "model_id": model,
                "layout_pair_id": layout,
                "receipt": aggregates[(model, layout)]["aggregate_receipt"],
            }
            for model in MODELS
            for layout in LAYOUTS
        ],
    }


def _expected_bundle_paths() -> set[str]:
    paths = {"compiler_receipt.json"}
    for model in MODELS:
        lower = model.lower()
        paths.update(
            {
                f"{lower}_development_timing_request_inventory.json",
                f"{lower}_development_request_provenance.json",
                f"{lower}_development_freeze_cells.json",
            }
        )
        for layout in LAYOUTS:
            for cell_id in aggregate_support.AGGREGATES[model][layout].cell_ids:
                base = f"cells/{lower}/{cell_id}"
                paths.add(f"{base}/action_manifest.json")
                paths.add(f"{base}/recording_receipt.json")
    require(len(paths) == EXPECTED_BUNDLE_FILES, "expected compiler bundle inventory changed")
    return paths


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, label: str) -> Path:
    """Reject lexical symlinks before resolution erases their identity."""

    lexical = _lexical_absolute(path)
    cursor = lexical
    while True:
        require(not cursor.is_symlink(), f"{label} contains a symlink: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return lexical
        cursor = parent


def _exact_context_paths(context: Any) -> None:
    expected_job = CONTROL_ROOT / "jobs" / JOB_ID
    expected_source = CONTROL_ROOT / "sources" / context.study_commit
    require(
        context.job_dir == expected_job,
        "queue context job directory is not the exact control-root job",
    )
    require(
        context.source_root == expected_source,
        "queue context source root is not the exact control-root checkout",
    )


def _inventory_bundle(
    bundle: Path,
    *,
    expected_bundle: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    lexical = _reject_symlink_components(bundle, "compiler bundle path")
    expected_lexical = _reject_symlink_components(
        expected_bundle, "expected compiler bundle path"
    )
    require(lexical == expected_lexical, "compiler bundle lexical root changed")
    try:
        bundle = lexical.resolve(strict=True)
        expected_resolved = expected_lexical.resolve(strict=True)
        expected_raw = expected_lexical.parent.resolve(strict=True)
    except OSError as error:
        raise DevelopmentCompilerQueueError("compiler bundle is unavailable") from error
    require(bundle == expected_resolved, "compiler bundle resolved root changed")
    require(
        bundle.parent == expected_raw,
        "compiler bundle escaped the exact job raw directory",
    )
    require(bundle.is_dir(), "compiler bundle is unavailable")
    observed_paths: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    by_relative: dict[str, dict[str, Any]] = {}
    for entry in sorted(bundle.rglob("*")):
        require(not entry.is_symlink(), f"compiler bundle contains a symlink: {entry}")
        if entry.is_dir():
            continue
        require(entry.is_file(), f"compiler bundle contains a non-file: {entry}")
        relative = str(entry.relative_to(bundle))
        require(relative not in observed_paths, "compiler bundle duplicates an output path")
        observed_paths.add(relative)
        identity = queue.file_identity(entry)
        descriptor = {**identity, "relative_path": relative}
        descriptors.append(descriptor)
        by_relative[relative] = identity
    require(observed_paths == _expected_bundle_paths(), "compiler bundle file inventory changed")
    return descriptors, by_relative


def _relative_descriptor_matches(
    value: Any,
    *,
    relative_path: str,
    bundle_files: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    actual = bundle_files.get(relative_path)
    require(actual is not None, f"{label} output file is missing")
    require(
        value.get("path") == relative_path
        and value.get("sha256") == actual["sha256"]
        and value.get("bytes") == actual["bytes"],
        f"{label} descriptor differs from compiler output",
    )


def _validate_compiler_receipt(
    *,
    compiler: ModuleType,
    returned: Any,
    bundle: Path,
    expected_bundle: Path,
    manifest_identity: Mapping[str, Any],
    implementation: Mapping[str, Mapping[str, Any]],
    aggregates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    receipt_path = bundle / "compiler_receipt.json"
    receipt_identity = queue.file_identity(receipt_path)
    receipt = queue.load_json(receipt_path, "compiler receipt")
    require(receipt == returned, "compiler return value differs from its immutable receipt")
    queue.verify_signed_document(receipt, "compiler receipt")
    require(
        getattr(compiler, "COMPILER_SCHEMA", None) == COMPILER_RECEIPT_SCHEMA
        and getattr(compiler, "INPUT_SCHEMA", None) == COMPILER_INPUT_SCHEMA,
        "loaded compiler receipt interface changed",
    )
    try:
        compiler.freeze.verify_signed(receipt, "compiler receipt")
    except BaseException as error:
        raise DevelopmentCompilerQueueError(
            f"compiler-native receipt validation failed: {error}"
        ) from error
    exact = {
        "schema_version": compiler.COMPILER_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "status": "compiled_complete",
        "formal_cohort_complete": True,
        "safe_for_timing_binding": True,
        "safe_for_confirmation_release": False,
        "raw_root": str(RAW_ROOT),
        "camera_id": CAMERA_ID,
        "counts": {
            "compiled_cells": EXPECTED_CELLS,
            "compiled_source_behavioral_requests": EXPECTED_REQUESTS,
            "compiled_source_behavioral_actions": EXPECTED_ACTIONS,
        },
        "compiler_science_activity": _compiler_zero_science(),
    }
    for key, wanted in exact.items():
        require(receipt.get(key) == wanted, f"compiler receipt changed: {key}")
    require(receipt.get("input_manifest") == manifest_identity, "compiler manifest binding changed")
    planned = receipt.get("planned_cells")
    require(
        isinstance(planned, Mapping)
        and planned.get("sha256") == PLANNED_CELLS_SHA256,
        "compiler receipt planned-cell binding changed",
    )
    compiler_source = receipt.get("compiler_source")
    require(
        isinstance(compiler_source, Mapping)
        and compiler_source.get("path") == str(COMPILER_RELATIVE)
        and compiler_source.get("sha256") == implementation["compiler"]["sha256"]
        and compiler_source.get("bytes") == implementation["compiler"]["bytes"],
        "compiler receipt source binding changed",
    )
    require(
        receipt.get("freeze_validator_dependency")
        == {
            "path": str(FREEZE_RELATIVE),
            "sha256": implementation["freeze_validator"]["sha256"],
            "bytes": implementation["freeze_validator"]["bytes"],
        },
        "compiler receipt freeze-validator dependency changed",
    )
    unsupported = receipt.get("unsupported_outputs")
    require(
        isinstance(unsupported, Mapping)
        and dict(unsupported)
        == {
            "resource_metrics": "not_synthesized",
            "camera_crop_contract": "not_created",
            "human_pixel_blindness_receipt": "not_created",
            "labels": "not_created",
            "movement_threshold": "not_created",
            "confirmation_release": "not_created",
        },
        "compiler receipt unsupported-output boundary changed",
    )
    bundle_inventory, bundle_files = _inventory_bundle(
        bundle, expected_bundle=expected_bundle
    )
    models = receipt.get("models")
    require(isinstance(models, Mapping) and set(models) == set(MODELS), "compiler model inventory changed")
    for model in MODELS:
        output = models[model]
        require(isinstance(output, Mapping), f"compiler {model} output is missing")
        expected_aggregates = [
            aggregates[(model, layout)]["aggregate_receipt"] for layout in LAYOUTS
        ]
        exact_model = {
            "complete": True,
            "cell_count": 16,
            "request_count": MODEL_REQUESTS[model],
            "action_count": MODEL_ACTIONS[model],
            "aggregate_receipts": expected_aggregates,
        }
        for key, wanted in exact_model.items():
            require(output.get(key) == wanted, f"compiler {model} output changed: {key}")
        lower = model.lower()
        _relative_descriptor_matches(
            output.get("timing_request_inventory"),
            relative_path=f"{lower}_development_timing_request_inventory.json",
            bundle_files=bundle_files,
            label=f"{model} timing inventory",
        )
        _relative_descriptor_matches(
            output.get("request_provenance"),
            relative_path=f"{lower}_development_request_provenance.json",
            bundle_files=bundle_files,
            label=f"{model} provenance",
        )
        _relative_descriptor_matches(
            output.get("freeze_cell_evidence"),
            relative_path=f"{lower}_development_freeze_cells.json",
            bundle_files=bundle_files,
            label=f"{model} freeze fragment",
        )
        cells = output.get("cells")
        require(isinstance(cells, list) and len(cells) == 16, f"compiler {model} cell inventory changed")
        expected_ids = {
            cell_id
            for layout in LAYOUTS
            for cell_id in aggregate_support.AGGREGATES[model][layout].cell_ids
        }
        require(
            {row.get("cell_id") for row in cells if isinstance(row, Mapping)} == expected_ids,
            f"compiler {model} cell identities changed",
        )
        for row in cells:
            require(isinstance(row, Mapping), f"compiler {model} cell row is invalid")
            cell_id = row["cell_id"]
            require(
                row.get("official_request_count")
                == aggregate_support.REQUESTS_PER_CELL[model],
                f"compiler {model} cell request count changed",
            )
            _relative_descriptor_matches(
                row.get("action_manifest"),
                relative_path=f"cells/{lower}/{cell_id}/action_manifest.json",
                bundle_files=bundle_files,
                label=f"{model} {cell_id} action manifest",
            )
            _relative_descriptor_matches(
                row.get("recording_receipt"),
                relative_path=f"cells/{lower}/{cell_id}/recording_receipt.json",
                bundle_files=bundle_files,
                label=f"{model} {cell_id} recording receipt",
            )
    return receipt, receipt_identity, bundle_inventory, bundle_files


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} is not a JSON object")
    require(set(value) == expected, f"{label} fields changed")


def _same_descriptor(left: Any, right: Any, label: str) -> None:
    require(
        isinstance(left, Mapping) and isinstance(right, Mapping),
        f"{label} descriptor is missing",
    )
    for key in ("path", "sha256", "bytes"):
        require(left.get(key) == right.get(key), f"{label} descriptor changed: {key}")


def _valid_sha_with_compiler(compiler: ModuleType, value: Any, label: str) -> None:
    require(
        bool(getattr(compiler, "_valid_sha")(value)),
        f"{label} is not SHA-256",
    )


def _validate_compiled_observation(
    compiler: ModuleType,
    value: Any,
    *,
    expected_control_step: int,
    label: str,
) -> None:
    _exact_keys(value, OBSERVATION_KEYS, label)
    assert isinstance(value, Mapping)
    require(
        value.get("observation_id") == f"obs_{expected_control_step:06d}"
        and value.get("control_step") == expected_control_step,
        f"{label} control identity changed",
    )
    require(
        type(value.get("physics_step")) is int and value["physics_step"] >= 0,
        f"{label} physics step is invalid",
    )
    require(
        type(value.get("physics_time_s")) in (int, float)
        and math.isfinite(float(value["physics_time_s"])),
        f"{label} physics time is invalid",
    )
    require(
        isinstance(value.get("camera_frame_id"), str)
        and value["camera_frame_id"].startswith(CAMERA_ID + ":native-")
        and value.get("camera_frame_native_id") is not None,
        f"{label} camera frame identity changed",
    )
    require(
        type(value.get("camera_capture_time_ns")) is int
        and value["camera_capture_time_ns"] >= 0
        and isinstance(value.get("camera_timestamp_source"), str)
        and bool(value["camera_timestamp_source"]),
        f"{label} camera clock is invalid",
    )
    _valid_sha_with_compiler(compiler, value.get("payload_sha256"), f"{label} payload")
    artifact = value.get("payload_artifact")
    require(isinstance(artifact, Mapping), f"{label} pixel artifact is missing")
    require(
        isinstance(artifact.get("path"), str)
        and bool(artifact["path"])
        and type(artifact.get("bytes")) is int
        and artifact["bytes"] > 0,
        f"{label} pixel artifact identity is invalid",
    )
    _valid_sha_with_compiler(
        compiler, artifact.get("sha256"), f"{label} pixel artifact"
    )


def _validate_model_identity(compiler: ModuleType, value: Any, model: str) -> None:
    _exact_keys(
        value,
        {
            "source_pins",
            "checkpoint_pin",
            "source_identity",
            "checkpoint_identity",
            "pose_manifest_sha256",
        },
        f"{model} compiled model identity",
    )
    assert isinstance(value, Mapping)
    pins = compiler.MODEL_PINS[model]
    source = value.get("source_pins")
    require(isinstance(source, Mapping), f"{model} source pins are missing")
    require(
        isinstance(source.get("study_commit"), str)
        and compiler.COMMIT_RE.fullmatch(source["study_commit"]) is not None,
        f"{model} source study commit is invalid",
    )
    expected_source = {
        "study_commit": source["study_commit"],
        "robolab_commit": pins["robolab_commit"],
    }
    if model == "N3":
        expected_source["cosmos_commit"] = pins["cosmos_commit"]
    else:
        expected_source.update(
            {
                "dreamzero_commit": pins["dreamzero_commit"],
                "dreamzero_tree": pins["dreamzero_tree"],
            }
        )
    require(dict(source) == expected_source, f"{model} compiled source pins changed")
    checkpoint = {
        "revision": pins["checkpoint_revision"],
        "aggregate_sha256": pins["checkpoint_aggregate_sha256"],
    }
    require(value.get("checkpoint_pin") == checkpoint, f"{model} checkpoint pin changed")
    _valid_sha_with_compiler(
        compiler, value.get("pose_manifest_sha256"), f"{model} pose manifest"
    )
    require(
        isinstance(value.get("source_identity"), str)
        and value["source_identity"].endswith(
            ";pose:" + value["pose_manifest_sha256"]
        )
        and value.get("checkpoint_identity")
        == (
            f"revision:{checkpoint['revision']};"
            f"aggregate:{checkpoint['aggregate_sha256']}"
        ),
        f"{model} compiled source/checkpoint identity changed",
    )


def _validate_model_context(value: Any, model: str, label: str) -> None:
    expected = (
        {
            "server_context_id",
            "server_begin_receipt",
            "server_end_receipt",
            "recorder_context_reset",
        }
        if model == "N3"
        else {
            "server_ready",
            "runtime_identity",
            "server_contract",
            "simulator_claim",
            "server_context_id",
            "server_reset_receipt",
            "server_episode_manifest",
            "recorder_context_reset",
        }
    )
    _exact_keys(value, expected, label)
    assert isinstance(value, Mapping)
    require(
        isinstance(value.get("server_context_id"), str)
        and bool(value["server_context_id"]),
        f"{label} context identity is missing",
    )
    reset = value.get("recorder_context_reset")
    require(
        isinstance(reset, Mapping)
        and reset.get("role") == "context_reset"
        and isinstance(reset.get("artifact_base"), str)
        and Path(reset["artifact_base"]).is_absolute(),
        f"{label} recorder reset binding is invalid",
    )


def _validate_semantic_bundle_outputs(
    *,
    compiler: ModuleType,
    timing: ModuleType,
    annotation: ModuleType,
    bundle: Path,
    bundle_files: Mapping[str, Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> list[str]:
    """Deeply validate every generated JSON file other than the compiler receipt."""

    semantic_paths: set[str] = set()
    per_model: dict[str, dict[str, Any]] = {}
    combined_roster: list[dict[str, Any]] = []
    all_source_request_ids: set[str] = set()

    for model in MODELS:
        lower = model.lower()
        timing_relative = f"{lower}_development_timing_request_inventory.json"
        provenance_relative = f"{lower}_development_request_provenance.json"
        freeze_relative = f"{lower}_development_freeze_cells.json"
        try:
            timing_inventory = timing.load_json(
                bundle / timing_relative, f"{model} compiled timing inventory"
            )
            provenance = compiler.load_json(
                bundle / provenance_relative, f"{model} compiled provenance"
            )
            freeze_fragment = compiler.freeze.load_json(
                bundle / freeze_relative, f"{model} compiled freeze fragment"
            )
        except BaseException as error:
            raise DevelopmentCompilerQueueError(
                f"{model} compiler semantic output is invalid: {error}"
            ) from error
        semantic_paths.update(
            {timing_relative, provenance_relative, freeze_relative}
        )

        _exact_keys(timing_inventory, TIMING_INVENTORY_KEYS, f"{model} timing inventory")
        require(
            timing_inventory.get("schema_version")
            == compiler.TIMING_INVENTORY_SCHEMA
            == timing.REQUEST_INVENTORY_SCHEMA
            and timing_inventory.get("study_id") == STUDY_ID
            and timing_inventory.get("model_id") == model,
            f"{model} timing inventory identity changed",
        )
        timing_entries = timing_inventory.get("request_receipts")
        require(
            isinstance(timing_entries, list)
            and len(timing_entries) == MODEL_REQUESTS[model],
            f"{model} timing request inventory count changed",
        )
        for index, entry in enumerate(timing_entries):
            _exact_keys(entry, TIMING_ENTRY_KEYS, f"{model} timing entry {index}")

        _exact_keys(provenance, PROVENANCE_KEYS, f"{model} provenance")
        try:
            annotation.verify_signed(provenance, f"{model} compiled provenance")
        except BaseException as error:
            raise DevelopmentCompilerQueueError(
                f"{model} provenance signature is invalid: {error}"
            ) from error
        exact_provenance = {
            "schema_version": compiler.PROVENANCE_SCHEMA,
            "study_id": STUDY_ID,
            "mode": MODE,
            "model_id": model,
            "stage": "development",
            "status": "complete",
            "safe_for_formal_release": True,
            "annotation_state": "not_started",
            "camera_id": CAMERA_ID,
            "camera_crop_contract": None,
            "resource_measurements": None,
            "labels": None,
        }
        for key, wanted in exact_provenance.items():
            require(provenance.get(key) == wanted, f"{model} provenance changed: {key}")
        roster = provenance.get("episode_roster")
        requests = provenance.get("requests")
        require(
            isinstance(roster, list) and len(roster) == 16,
            f"{model} provenance roster count changed",
        )
        require(
            isinstance(requests, list) and len(requests) == MODEL_REQUESTS[model],
            f"{model} provenance request count changed",
        )
        combined_roster.extend(dict(row) for row in roster if isinstance(row, Mapping))

        _exact_keys(freeze_fragment, FREEZE_FRAGMENT_KEYS, f"{model} freeze fragment")
        try:
            compiler.freeze.verify_signed(
                freeze_fragment, f"{model} compiled freeze fragment"
            )
        except BaseException as error:
            raise DevelopmentCompilerQueueError(
                f"{model} freeze-fragment signature is invalid: {error}"
            ) from error
        require(
            freeze_fragment.get("schema_version") == compiler.FREEZE_FRAGMENT_SCHEMA
            and freeze_fragment.get("study_id") == STUDY_ID
            and freeze_fragment.get("mode") == MODE
            and freeze_fragment.get("model_id") == model
            and freeze_fragment.get("status") == "complete"
            and freeze_fragment.get("safe_for_alignment_input") is True
            and freeze_fragment.get("resource_receipts_synthesized") is False,
            f"{model} freeze fragment identity changed",
        )
        freeze_entries = freeze_fragment.get("development_cells")
        require(
            isinstance(freeze_entries, list) and len(freeze_entries) == 16,
            f"{model} freeze cell inventory changed",
        )

        model_output = receipt["models"][model]
        cell_outputs_raw = model_output.get("cells")
        require(isinstance(cell_outputs_raw, list), f"{model} compiler cells are missing")
        cell_outputs: dict[str, Mapping[str, Any]] = {}
        for item in cell_outputs_raw:
            _exact_keys(item, MODEL_OUTPUT_CELL_KEYS, f"{model} compiler cell output")
            assert isinstance(item, Mapping)
            cell_id = item.get("cell_id")
            require(
                isinstance(cell_id, str) and cell_id not in cell_outputs,
                f"{model} compiler cell identity is invalid or duplicated",
            )
            require(
                item.get("official_request_count")
                == compiler.MODEL_LIMITS[model]["request_count"],
                f"{cell_id} compiler request count changed",
            )
            cell_outputs[cell_id] = item
            for role in ("action_manifest", "recording_receipt"):
                descriptor = item.get(role)
                require(
                    isinstance(descriptor, Mapping)
                    and set(descriptor) == {"path", "sha256", "bytes"},
                    f"{cell_id} {role} descriptor fields changed",
                )
                semantic_paths.add(str(descriptor["path"]))
        require(len(cell_outputs) == 16, f"{model} compiler cell count changed")

        freeze_by_cell: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        for entry in freeze_entries:
            require(isinstance(entry, Mapping), f"{model} freeze cell row is invalid")
            matches = [
                cell_id
                for cell_id, output in cell_outputs.items()
                if entry.get("cell_receipt") == output.get("source_cell_receipt")
            ]
            require(
                len(matches) == 1,
                f"{model} freeze cell receipt is absent or ambiguous in compiler output",
            )
            cell_id = matches[0]
            require(cell_id not in freeze_by_cell, f"{cell_id} freeze cell is duplicated")
            try:
                validated = compiler.freeze._validate_cell(
                    entry,
                    model=model,
                    expected_cell_id=cell_id,
                    evidence_base=bundle,
                    require_resource=False,
                )
            except BaseException as error:
                raise DevelopmentCompilerQueueError(
                    f"{cell_id} failed detached freeze-validator replay: {error}"
                ) from error
            require(isinstance(validated, Mapping), f"{cell_id} freeze replay is invalid")
            freeze_by_cell[cell_id] = (entry, validated)
        require(
            set(freeze_by_cell) == set(cell_outputs),
            f"{model} freeze/compiler cell identities differ",
        )

        per_model[model] = {
            "timing_entries": timing_entries,
            "provenance_requests": requests,
            "roster": roster,
            "cell_outputs": cell_outputs,
            "freeze_by_cell": freeze_by_cell,
        }

    require(
        len(combined_roster) == EXPECTED_CELLS,
        "combined compiler annotation roster count changed",
    )
    try:
        normalized_roster, action_manifests = annotation._validate_episode_roster(
            combined_roster,
            stage="development",
            cohort_branch="full_two_model",
            base=bundle,
        )
    except BaseException as error:
        raise DevelopmentCompilerQueueError(
            f"compiled annotation roster/artifacts failed validation: {error}"
        ) from error
    require(
        len(normalized_roster) == EXPECTED_CELLS
        and len(action_manifests) == EXPECTED_CELLS,
        "compiled annotation artifact inventory changed",
    )

    for model in MODELS:
        state = per_model[model]
        requests_by_cell: dict[str, list[Mapping[str, Any]]] = {
            cell_id: [] for cell_id in state["cell_outputs"]
        }
        for index, request in enumerate(state["provenance_requests"]):
            _exact_keys(
                request, PROVENANCE_REQUEST_KEYS, f"{model} provenance request {index}"
            )
            assert isinstance(request, Mapping)
            cell_id = request.get("cell_id")
            require(
                isinstance(cell_id, str) and cell_id in requests_by_cell,
                f"{model} provenance request references an unknown cell",
            )
            requests_by_cell[cell_id].append(request)

        ordered_projection: list[dict[str, Any]] = []
        for cell_id, cell_requests in requests_by_cell.items():
            output = state["cell_outputs"][cell_id]
            freeze_entry, validated = state["freeze_by_cell"][cell_id]
            roster = normalized_roster[cell_id]
            require(
                len(cell_requests) == compiler.MODEL_LIMITS[model]["request_count"],
                f"{cell_id} provenance request inventory changed",
            )
            cell_requests.sort(key=lambda row: row.get("request_index", -1))
            require(
                [row.get("request_index") for row in cell_requests]
                == list(range(len(cell_requests))),
                f"{cell_id} provenance request indices changed",
            )
            require(
                Path(roster["action_manifest_path"])
                == (bundle / output["action_manifest"]["path"]).resolve()
                and roster["action_manifest_sha256"]
                == output["action_manifest"]["sha256"],
                f"{cell_id} annotation action-manifest binding changed",
            )
            require(
                Path(roster["recording_receipt_path"])
                == (bundle / output["recording_receipt"]["path"]).resolve()
                and roster["recording_receipt_sha256"]
                == output["recording_receipt"]["sha256"],
                f"{cell_id} annotation recording-receipt binding changed",
            )
            source_video = output.get("source_video")
            require(
                isinstance(source_video, Mapping)
                and source_video.get("sha256") == roster["source_video_sha256"],
                f"{cell_id} source video binding changed",
            )
            request_descriptors = freeze_entry.get("server_request_receipts")
            require(
                isinstance(request_descriptors, list)
                and len(request_descriptors) == len(cell_requests)
                and validated.get("request_receipt_sha256s")
                == [item.get("sha256") for item in request_descriptors],
                f"{cell_id} freeze request identity inventory changed",
            )
            prefix = compiler.MODEL_LIMITS[model]["executed_prefix"]
            for request_index, request in enumerate(cell_requests):
                start = request_index * prefix
                executed = min(prefix, 450 - start)
                require(
                    request.get("model_id") == model
                    and request.get("recording_id") == roster["recording_id"]
                    and request.get("layout_pair_id") == roster["layout_pair_id"]
                    and request.get("condition_id") == roster["condition_id"]
                    and request.get("action_step_start") == start
                    and request.get("executed_prefix_actions") == executed
                    and request.get("current_observation_id") == f"obs_{start:06d}"
                    and request.get("source_video_id") == roster["source_video_id"]
                    and request.get("model_output_or_action_modified") is False,
                    f"{cell_id} provenance request {request_index} identity changed",
                )
                preceding_step = None if request_index == 0 else start - 1
                require(
                    request.get("preceding_observation_id")
                    == (
                        None
                        if preceding_step is None
                        else f"obs_{preceding_step:06d}"
                    )
                    and request.get("history_mode")
                    == (
                        "persistence_at_initial_request"
                        if request_index == 0
                        else "preceding_observation"
                    ),
                    f"{cell_id} provenance history changed at request {request_index}",
                )
                _validate_compiled_observation(
                    compiler,
                    request.get("current_observation"),
                    expected_control_step=start,
                    label=f"{cell_id} current observation {request_index}",
                )
                if preceding_step is None:
                    require(
                        request.get("preceding_observation") is None,
                        f"{cell_id} initial request invents a preceding observation",
                    )
                else:
                    _validate_compiled_observation(
                        compiler,
                        request.get("preceding_observation"),
                        expected_control_step=preceding_step,
                        label=f"{cell_id} preceding observation {request_index}",
                    )
                expected_request = request_descriptors[request_index]
                _same_descriptor(
                    request.get("official_request_receipt"),
                    expected_request,
                    f"{cell_id} official request {request_index}",
                )
                _same_descriptor(
                    request.get("adapter_completion"),
                    validated.get("adapter_completion"),
                    f"{cell_id} adapter completion {request_index}",
                )
                _same_descriptor(
                    request.get("adapter_journal"),
                    validated.get("adapter_journal"),
                    f"{cell_id} adapter journal {request_index}",
                )
                _same_descriptor(
                    request.get("action_manifest"),
                    output["action_manifest"],
                    f"{cell_id} action manifest {request_index}",
                )
                _same_descriptor(
                    request.get("recording_receipt"),
                    output["recording_receipt"],
                    f"{cell_id} recording receipt {request_index}",
                )
                _same_descriptor(
                    request.get("source_video"),
                    source_video,
                    f"{cell_id} source video {request_index}",
                )
                require(
                    request.get("source_request_id")
                    == "request_" + expected_request["sha256"][:32]
                    and request["source_request_id"] not in all_source_request_ids,
                    f"{cell_id} source request identity changed or duplicated",
                )
                all_source_request_ids.add(request["source_request_id"])
                packed = request.get("recorder_model_request")
                _exact_keys(
                    packed,
                    {
                        "role",
                        "payload_sha256",
                        "array_count",
                        "structure_sha256",
                        "artifact",
                        "artifact_base",
                    },
                    f"{cell_id} packed request {request_index}",
                )
                assert isinstance(packed, Mapping)
                require(
                    packed.get("role") == "model_request"
                    and type(packed.get("array_count")) is int
                    and packed["array_count"] >= 0
                    and isinstance(packed.get("artifact_base"), str)
                    and Path(packed["artifact_base"]).is_absolute(),
                    f"{cell_id} packed request {request_index} identity changed",
                )
                _valid_sha_with_compiler(
                    compiler,
                    packed.get("payload_sha256"),
                    f"{cell_id} packed request payload {request_index}",
                )
                _valid_sha_with_compiler(
                    compiler,
                    packed.get("structure_sha256"),
                    f"{cell_id} packed request structure {request_index}",
                )
                _valid_sha_with_compiler(
                    compiler,
                    request.get("recorder_response_payload_sha256"),
                    f"{cell_id} response payload {request_index}",
                )
                _validate_model_identity(compiler, request.get("model_identity"), model)
                _validate_model_context(
                    request.get("model_context"),
                    model,
                    f"{cell_id} model context {request_index}",
                )
                ordered_projection.append(
                    {
                        "cell_id": cell_id,
                        "request_index": request_index,
                        "request_receipt": request["official_request_receipt"],
                        "adapter_completion": request["adapter_completion"],
                        "adapter_journal": request["adapter_journal"],
                    }
                )
        require(
            state["timing_entries"] == ordered_projection,
            f"{model} timing inventory differs from compiled provenance order",
        )

    expected_semantic = _expected_bundle_paths() - {"compiler_receipt.json"}
    require(
        semantic_paths == expected_semantic,
        "not every non-receipt compiler output passed semantic validation",
    )
    require(
        set(bundle_files) == _expected_bundle_paths()
        and all(
            isinstance(bundle_files[path].get("sha256"), str)
            and type(bundle_files[path].get("bytes")) is int
            and bundle_files[path]["bytes"] > 2
            for path in semantic_paths
        ),
        "semantic output identities are incomplete or empty",
    )
    require(
        len(all_source_request_ids) == EXPECTED_REQUESTS,
        "compiled source request identities are incomplete",
    )
    return sorted(semantic_paths)


def _write_failure_receipt(
    *, context: Any | None, job_dir: Path, error: BaseException
) -> None:
    expected_job_dir = CONTROL_ROOT / "jobs" / JOB_ID
    try:
        supplied = Path(job_dir)
        if supplied != expected_job_dir or supplied.resolve() != expected_job_dir:
            return
        publish = supplied / "publish"
        if publish.exists() or publish.is_symlink():
            require(publish.is_dir() and not publish.is_symlink(), "failure publish path is invalid")
        else:
            publish.mkdir()
        success_target = publish / PUBLISH_RECEIPT_NAME
        target = publish / PUBLISH_FAILURE_NAME
        entries = {path.name for path in publish.iterdir()}
        # A passing receipt is terminal.  Never append a contradictory failure,
        # and never add a receipt alongside an unknown publication artifact.
        if success_target.exists() or success_target.is_symlink():
            return
        if entries == {PUBLISH_FAILURE_NAME} and target.is_file() and not target.is_symlink():
            return
        require(not entries, "failure publication inventory is not empty")
        receipt = queue.signed_document(
            {
                "schema_version": JOB_RECEIPT_SCHEMA,
                "namespace": NAMESPACE,
                "study_id": STUDY_ID,
                "mode": MODE,
                "status": "technical_invalid",
                "decision": "no_go",
                "job_id": context.job_id if context is not None else JOB_ID,
                "study_commit": context.study_commit if context is not None else None,
                "queue_role": context.role if context is not None else None,
                "queue_descriptor": context.descriptor_identity if context is not None else None,
                "queue_claim": context.claim_identity if context is not None else None,
                "failure": {
                    "error_type": type(error).__name__,
                    "detail": str(error)[:2000],
                    "traceback": traceback.format_exc(limit=20)[-12000:],
                },
                "science_counts": _zero_science_counts(),
                "formal_cohort_complete": False,
                "safe_for_timing_binding": False,
                "safe_to_release_confirmation": False,
                "confirmation_released": False,
                "behavioral_policy_skill_evaluated": False,
                "claim_boundary": (
                    "Failed CPU-only evidence validation or compilation. No model, simulator, "
                    "reset, request, action, episode, label, or confirmation job was started."
                ),
                "completed_at_utc": queue.utc_now(),
            }
        )
        queue.immutable_json(target, receipt, maximum_bytes=512 * 1024)
    except BaseException:
        return


def run_formal_job(args: argparse.Namespace) -> dict[str, Any]:
    descriptor, expected_implementation, expected_aggregates = _runtime_descriptor(args)
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
        _exact_context_paths(context)
        implementation, compiler, timing, annotation = _validate_staged_implementation(
            source_root=context.source_root,
            expected=expected_implementation,
        )
        aggregates = _validate_runtime_aggregates(expected_aggregates)
        raw, publish = queue._prepare_output_directories(context.job_dir)
        manifest_path = context.job_dir / MANIFEST_RELATIVE
        bundle_path = context.job_dir / BUNDLE_RELATIVE
        require(manifest_path.parent == raw, "compiler input manifest path changed")
        require(bundle_path.parent == raw, "compiler output directory path changed")
        require(
            _reject_symlink_components(bundle_path, "compiler output bundle")
            == context.job_dir / BUNDLE_RELATIVE,
            "compiler output bundle lexical path changed",
        )
        manifest = _compiler_manifest(
            source_root=context.source_root,
            aggregates=aggregates,
        )
        queue.immutable_json(manifest_path, manifest, maximum_bytes=2 * 1024 * 1024)
        manifest_identity = queue.file_identity(manifest_path)
        returned = compiler.compile_manifest(
            manifest_path, manifest_identity["sha256"], bundle_path
        )
        (
            compiler_receipt,
            compiler_receipt_identity,
            bundle_inventory,
            bundle_files,
        ) = _validate_compiler_receipt(
            compiler=compiler,
            returned=returned,
            bundle=bundle_path,
            expected_bundle=context.job_dir / BUNDLE_RELATIVE,
            manifest_identity=manifest_identity,
            implementation=implementation,
            aggregates=aggregates,
        )
        semantic_outputs = _validate_semantic_bundle_outputs(
            compiler=compiler,
            timing=timing,
            annotation=annotation,
            bundle=bundle_path,
            bundle_files=bundle_files,
            receipt=compiler_receipt,
        )
        repeated_inventory, repeated_bundle_files = _inventory_bundle(
            bundle_path, expected_bundle=context.job_dir / BUNDLE_RELATIVE
        )
        require(
            repeated_inventory == bundle_inventory
            and repeated_bundle_files == bundle_files,
            "compiler bundle changed during semantic validation",
        )
        inventory = queue.signed_document(
            {
                "schema_version": OUTPUT_INVENTORY_SCHEMA,
                "namespace": NAMESPACE,
                "study_id": STUDY_ID,
                "mode": MODE,
                "job_id": context.job_id,
                "study_commit": context.study_commit,
                "bundle_root": str(bundle_path),
                "file_count": len(bundle_inventory),
                "files": bundle_inventory,
                "semantically_validated_nonreceipt_files": semantic_outputs,
                "formal_cohort_complete": True,
                "safe_for_timing_binding": True,
                "safe_to_release_confirmation": False,
            }
        )
        inventory_path = raw / "compiler_output_inventory.json"
        queue.immutable_json(inventory_path, inventory, maximum_bytes=2 * 1024 * 1024)
        inventory_identity = queue.file_identity(inventory_path)
        primary_outputs = {
            relative: bundle_files[relative]
            for relative in (
                "n3_development_timing_request_inventory.json",
                "d1_development_timing_request_inventory.json",
                "n3_development_request_provenance.json",
                "d1_development_request_provenance.json",
                "n3_development_freeze_cells.json",
                "d1_development_freeze_cells.json",
            )
        }
        receipt = queue.signed_document(
            {
                "schema_version": JOB_RECEIPT_SCHEMA,
                "namespace": NAMESPACE,
                "study_id": STUDY_ID,
                "mode": MODE,
                "status": "passed",
                "decision": "go",
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
                "implementation": implementation,
                "inputs": {
                    "compiler_manifest": manifest_identity,
                    "planned_cells": manifest["planned_cells"],
                    "aggregate_receipts": manifest["aggregate_receipts"],
                },
                "outputs": {
                    "compiler_receipt": compiler_receipt_identity,
                    "compiler_output_inventory": inventory_identity,
                    "primary_compiled_outputs": primary_outputs,
                    "semantically_validated_nonreceipt_file_count": len(
                        semantic_outputs
                    ),
                },
                "counts": compiler_receipt["counts"],
                "science_counts": _zero_science_counts(),
                "compiler_science_activity": compiler_receipt[
                    "compiler_science_activity"
                ],
                "formal_cohort_complete": True,
                "safe_for_timing_binding": True,
                "safe_to_release_confirmation": False,
                "confirmation_released": False,
                "behavioral_policy_skill_evaluated": False,
                "raw_outputs_recoverable_on_gm_pvc": True,
                "published_files": [PUBLISH_RECEIPT_NAME],
                "claim_boundary": (
                    "CPU-only compilation of 32 immutable completed development cells. "
                    "The result may feed timing binding only; annotation, resource freeze, "
                    "and confirmation release remain separately gated."
                ),
                "completed_at_utc": queue.utc_now(),
            }
        )
        require(
            publish.is_dir()
            and not publish.is_symlink()
            and not any(publish.iterdir()),
            "compiler success publish inventory is not exactly empty",
        )
        # This immutable success write is deliberately the final fallible
        # operation.  The exception handler cannot add a failure beside it.
        queue.immutable_json(
            publish / PUBLISH_RECEIPT_NAME,
            receipt,
            maximum_bytes=512 * 1024,
        )
        return receipt
    except BaseException as error:
        _write_failure_receipt(context=context, job_dir=Path(args.job_dir), error=error)
        raise


def _argument_stem(model: str, layout: str) -> str:
    return f"{model.lower()}_{layout.lower()}_aggregate"


def _add_builder_aggregate_inputs(parser: argparse.ArgumentParser) -> None:
    for model in MODELS:
        for layout in LAYOUTS:
            option = f"--{model.lower()}-{layout.lower()}-aggregate"
            parser.add_argument(option, type=Path, required=True)
            parser.add_argument(option + "-sha256", required=True)


def _add_runtime_aggregate_inputs(parser: argparse.ArgumentParser) -> None:
    for model in MODELS:
        for layout in LAYOUTS:
            option = f"--{model.lower()}-{layout.lower()}-aggregate"
            parser.add_argument(option, type=Path, required=True)
            parser.add_argument(option + "-sha256", required=True)
            parser.add_argument(option + "-bytes", type=int, required=True)


def _builder_aggregate_inputs(
    args: argparse.Namespace,
) -> dict[tuple[str, str], tuple[Path, str]]:
    return {
        (model, layout): (
            Path(getattr(args, _argument_stem(model, layout))),
            getattr(args, _argument_stem(model, layout) + "_sha256"),
        )
        for model in MODELS
        for layout in LAYOUTS
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    emit = commands.add_parser("emit-formal")
    emit.add_argument("--study-commit", required=True)
    _add_builder_aggregate_inputs(emit)
    emit.add_argument("--output", type=Path)

    runtime = commands.add_parser("formal-full")
    runtime.add_argument("--source-root", type=Path, required=True)
    runtime.add_argument("--study-commit", required=True)
    runtime.add_argument("--job-dir", type=Path, required=True)
    runtime.add_argument("--job-id", required=True)
    runtime.add_argument("--expected-role", required=True)
    runtime.add_argument("--queue-wrapper-sha256", required=True)
    runtime.add_argument("--queue-support-sha256", required=True)
    runtime.add_argument("--aggregate-support-sha256", required=True)
    runtime.add_argument("--queue-contract-sha256", required=True)
    runtime.add_argument("--compiler-sha256", required=True)
    runtime.add_argument("--freeze-validator-sha256", required=True)
    runtime.add_argument("--timing-validator-sha256", required=True)
    runtime.add_argument("--annotation-validator-sha256", required=True)
    runtime.add_argument("--planned-cells-sha256", required=True)
    _add_runtime_aggregate_inputs(runtime)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "emit-formal":
        wave = build_formal_wave(
            study_commit=args.study_commit,
            aggregate_inputs=_builder_aggregate_inputs(args),
        )
        queue._write_descriptor_output(args.output, wave)
        return 0
    receipt = run_formal_job(args)
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
            json.dumps(
                {
                    "status": "technical_failure",
                    "error_type": type(error).__name__,
                    "detail": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
