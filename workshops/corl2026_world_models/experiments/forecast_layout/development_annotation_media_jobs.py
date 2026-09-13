#!/usr/bin/env python3
"""Emit and run the detached CPU development annotation-media bridge.

The emitter never edits the active queue.  It freezes one descriptor from a
hash-bound input JSON whose paths name immutable PVC evidence.  The runtime
reconstructs that exact input under its claimed job directory, authenticates
the staged implementation and every input descriptor, then invokes the
fail-closed annotation-media bridge.  Only a compact receipt and the bounded
tranche index are published by the main job; rendered images and restricted
provenance remain on the PVC.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


queue = _load_module(QUEUE_MODULE_PATH, "wmf_annotation_media_queue_support")

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
RAW_ROOT = CONTROL_ROOT.parent
ROBOLAB_PYTHON = queue.ROBOLAB_PYTHON

THIS_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_annotation_media_jobs.py"
)
QUEUE_RELATIVE = queue.THIS_RELATIVE
CONTRACT_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_annotation_media_bridge_contract.json"
)
BRIDGE_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/prepare_development_annotation_media.py"
)
COMPILER_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/compile_development_evidence.py"
)
TIMING_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/qualify_forecast_timing.py"
)
ANNOTATION_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/forecast_annotation_workflow.py"
)
FREEZE_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/freeze_development_release.py"
)
CAMERA_REPLAY_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/camera_crop_replay_witness.py"
)

WAVE_SCHEMA = "wmf-development-annotation-media-wave-v1"
CONTRACT_SCHEMA = "wmf-development-annotation-media-bridge-contract-v1"
JOB_RECEIPT_SCHEMA = "wmf-development-annotation-media-queue-job-v1"
INPUT_SCHEMA = "wmf-development-annotation-media-input-v1"
PREPARATION_SCHEMA = "wmf-development-annotation-media-preparation-v1"
JOB_ID = "development-annotation-media-bridge-001"
WORKER_ROLE = "wmf-forecast-0912-worker-05"
MODE = "formal_full_development"
MAX_WALL_SECONDS = 21600
PUBLISH_LOG_TAIL_BYTES = 8192
MANIFEST_RELATIVE = Path("raw/annotation_media_input_manifest.json")
OUTPUT_RELATIVE = Path("raw/annotation_media_preparation")
PUBLISH_RECEIPT_NAME = "development_annotation_media_job_receipt.json"
PUBLISH_INDEX_NAME = "publish_tranche_index.json"
PUBLISH_FAILURE_NAME = "development_annotation_media_job_failure.json"
TRANCHE_MANIFEST_NAME = "tranche_manifest.json"
TRANCHE_RECEIPT_NAME = "tranche_job_receipt.json"
TRANCHE_INDEX_SCHEMA = "wmf-development-annotation-media-tranche-index-v1"
TRANCHE_MANIFEST_SCHEMA = "wmf-development-annotation-media-published-tranche-v1"
TRANCHE_JOB_RECEIPT_SCHEMA = "wmf-development-annotation-media-tranche-job-v1"
TRANCHE_WAVE_SCHEMA = "wmf-development-annotation-media-tranche-wave-v1"
PUBLISH_FILE_LIMIT_BYTES = 16 * 1024 * 1024
PUBLISH_JOB_LIMIT_BYTES = 64 * 1024 * 1024
PUBLISH_TRANCHE_ASSET_BUDGET_BYTES = 48 * 1024 * 1024
TRANCHE_WORKER_ROLES = (
    "wmf-forecast-0912-worker-00",
    "wmf-forecast-0912-worker-05",
    "wmf-forecast-0912-worker-06",
    "wmf-forecast-0912-worker-09",
)
EXPECTED_EPISODES = 32
EXPECTED_REQUESTS = 1152
EXPECTED_ELIGIBLE = 448
EXPECTED_SELECTED = 128
EXPECTED_D1_UNMAPPED = 672
MODELS = ("N3", "D1")
PREPARATION_CLAIM_BOUNDARY = (
    "Prepared restricted development annotation images and deterministic request "
    "selection only. A legitimate human pixel-blindness review is still required "
    "before final image inventory or rater packet creation."
)


class AnnotationMediaQueueError(RuntimeError):
    """The annotation media queue descriptor or runtime failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationMediaQueueError(message)


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


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "queue_wrapper": root / THIS_RELATIVE,
        "queue_support": root / QUEUE_RELATIVE,
        "bridge": root / BRIDGE_RELATIVE,
        "compiler": root / COMPILER_RELATIVE,
        "timing_validator": root / TIMING_RELATIVE,
        "annotation_workflow": root / ANNOTATION_RELATIVE,
        "freeze_validator": root / FREEZE_RELATIVE,
        "camera_replay": root / CAMERA_REPLAY_RELATIVE,
        "queue_contract": root / CONTRACT_RELATIVE,
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
            "output_relative": str(OUTPUT_RELATIVE),
        },
        "input_policy": {
            "compiler_job_receipts": 1,
            "compiler_receipts": 1,
            "model_request_provenance_files": 2,
            "passed_timing_sidecars": 2,
            "passed_timing_job_receipts": 2,
            "signed_camera_crop_contracts": 2,
            "alignment_contracts": 2,
            "signed_physical_alignment_receipts": 2,
            "join_key": ["model_id", "cell_id", "request_index"],
            "physical_alignment_rederived_from_validated_sources": True,
        },
        "exact_output_counts": {
            "episodes": EXPECTED_EPISODES,
            "requests": EXPECTED_REQUESTS,
            "requests_by_model": {"N3": 240, "D1": 912},
            "source_timing_capable_requests": 480,
            "d1_forecast_timing_unavailable_requests": EXPECTED_D1_UNMAPPED,
            "timing_camera_action_eligible_requests": EXPECTED_ELIGIBLE,
            "selected_requests": EXPECTED_SELECTED,
            "selected_requests_by_model": {"N3": 64, "D1": 64},
            "eligible_requests_per_episode": 14,
            "request_inclusion_probability_exact": "4/14",
            "early_horizon_required_for_both_models": True,
        },
        "output_policy": {
            "success_publish_files": [PUBLISH_RECEIPT_NAME, PUBLISH_INDEX_NAME],
            "failure_publish_files": [PUBLISH_FAILURE_NAME],
            "mutually_exclusive_terminal_receipts": True,
            "restricted_media_remains_on_pvc": True,
            "human_pixel_blindness_review_required": True,
            "pixel_review_checklist_emitted": True,
            "rater_packets_created_by_job": False,
            "publisher_file_limit_bytes": PUBLISH_FILE_LIMIT_BYTES,
            "publisher_job_limit_bytes": PUBLISH_JOB_LIMIT_BYTES,
            "publish_tranche_asset_budget_bytes": PUBLISH_TRANCHE_ASSET_BUDGET_BYTES,
            "tranche_worker_roles": list(TRANCHE_WORKER_ROLES),
            "deduplicate_assets_by": "annotation_media_sha256",
            "tranche_release_requires_exact_terminal_preparation_receipt": True,
            "tranche_index_bound_to_preparation_receipt_output_descriptor": True,
        },
        "runtime_replay_policy": {
            "original_camera_api": "replay_original_camera_frame",
            "original_camera_execution": (
                "one_persistent_isolated_subprocess_per_model"
            ),
            "interpreter_authority": (
                "camera_crop_contract.runtime_dependencies.python"
            ),
            "ipc_schema": "wmf-camera-replay-exact-runtime-ipc-v1",
            "runtime_session_receipts": 2,
            "generated_crop_api": "extract_generated_crop",
            "generated_crop_execution": "in_process_exact_uint8_half_open_slice",
            "cuda_visible_devices": "",
        },
        "science_counts": _zero_science_counts(),
        "safe_for_rater_distribution": False,
        "safe_to_release_confirmation": False,
    }


def _validate_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = queue.file_identity(path)
    require(identity["sha256"] == queue._verified_sha(
        expected_sha256, "annotation media contract digest"
    ), "annotation media queue contract hash changed")
    require(queue.load_json(path, "annotation media queue contract") == _expected_contract(),
            "annotation media queue contract fields changed")
    return identity


def _local_implementation() -> dict[str, dict[str, Any]]:
    paths = _source_paths(REPOSITORY_ROOT)
    missing = [name for name, path in paths.items() if not path.is_file()]
    require(not missing, f"annotation media implementation is incomplete: {missing}")
    identities = {name: queue.file_identity(path) for name, path in paths.items()}
    _validate_contract(paths["queue_contract"], identities["queue_contract"]["sha256"])
    return identities


def _descriptor_shape(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256", "bytes"},
            f"{label} descriptor fields changed")
    require(isinstance(value.get("path"), str) and Path(value["path"]).is_absolute(),
            f"{label} path must be absolute")
    digest = queue._verified_sha(value.get("sha256"), f"{label} digest")
    size = value.get("bytes")
    require(type(size) is int and size > 0, f"{label} byte count is invalid")
    path = Path(value["path"])
    require(path == Path(os.path.abspath(os.fspath(path)))
            and ".." not in path.parts,
            f"{label} path is not lexically normalized")
    require(path.is_relative_to(RAW_ROOT), f"{label} path is outside the task raw root")
    return {"path": str(path), "sha256": digest, "bytes": size}


def validate_input_shape(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "annotation media inputs must be an object")
    require(set(value) == {
        "schema_version", "study_id", "mode", "raw_root", "compiler", "models"
    }, "annotation media input fields changed")
    require(value.get("schema_version") == INPUT_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("mode") == MODE
            and value.get("raw_root") == str(RAW_ROOT),
            "annotation media input identity changed")
    compiler_inputs = value.get("compiler")
    require(isinstance(compiler_inputs, Mapping)
            and set(compiler_inputs) == {"job_receipt", "compiler_receipt"},
            "annotation media compiler inputs changed")
    normalized_compiler = {
        key: _descriptor_shape(raw, f"compiler {key}")
        for key, raw in compiler_inputs.items()
    }
    models = value.get("models")
    require(isinstance(models, list) and len(models) == 2,
            "annotation media inputs require exactly two models")
    normalized_models = []
    seen_models: set[str] = set()
    descriptor_paths = [item["path"] for item in normalized_compiler.values()]
    fields = {
        "request_provenance", "timing_sidecar", "timing_job_receipt",
        "camera_crop_contract", "alignment_contract", "physical_alignment_receipt",
    }
    for item in models:
        require(isinstance(item, Mapping) and set(item) == {"model_id", *fields},
                "annotation media model input fields changed")
        model = item.get("model_id")
        require(model in MODELS and model not in seen_models,
                "annotation media model identity is invalid or duplicated")
        seen_models.add(model)
        normalized = {"model_id": model}
        for field in sorted(fields):
            normalized[field] = _descriptor_shape(item[field], f"{model} {field}")
            descriptor_paths.append(normalized[field]["path"])
        normalized_models.append(normalized)
    require(seen_models == set(MODELS), "annotation media model coverage changed")
    require(len(descriptor_paths) == len(set(descriptor_paths)),
            "annotation media input paths are ambiguous")
    return {
        "schema_version": INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "raw_root": str(RAW_ROOT),
        "compiler": normalized_compiler,
        "models": sorted(normalized_models, key=lambda row: row["model_id"]),
    }


def _implementation_hash_arguments(
    implementation: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    result: list[str] = []
    for name in sorted(implementation):
        result.extend([f"--{name.replace('_', '-')}-sha256", str(implementation[name]["sha256"])])
    return result


def _job_descriptor(
    *, study_commit: str, implementation: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
            "annotation media implementation identity inventory changed")
    normalized_inputs = validate_input_shape(inputs)
    argv = [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        "formal-full",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", JOB_ID,
        "--expected-role", WORKER_ROLE,
        "--inputs-json", json.dumps(
            normalized_inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    ]
    argv.extend(_implementation_hash_arguments(implementation))
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
        "status": "descriptor_only_not_dispatched_inputs_shape_validated",
        "jobs": [descriptor],
        "implementation": implementation,
        "inputs": normalized,
        "expected_counts": _expected_contract()["exact_output_counts"],
        "science_counts": _zero_science_counts(),
        "human_pixel_blindness_review_complete": False,
        "rater_packets_created": False,
        "safe_for_rater_distribution": False,
        "safe_to_release_confirmation": False,
        "claim_boundary": (
            "One detached CPU-only media-preparation job. Input descriptors are shape-"
            "validated at emission and byte-authenticated on the PVC at runtime."
        ),
    }


def _runtime_implementation(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": queue._verified_sha(
            getattr(args, name + "_sha256"), f"{name} digest"
        )}
        for name in sorted(_source_paths(REPOSITORY_ROOT))
    }


def _runtime_descriptor(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    require(args.command == "formal-full", "annotation media runtime mode changed")
    require(args.job_id == JOB_ID and args.expected_role == WORKER_ROLE,
            "annotation media runtime job/role changed")
    try:
        inputs = json.loads(args.inputs_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise AnnotationMediaQueueError("annotation media runtime inputs JSON is invalid") from error
    inputs = validate_input_shape(inputs)
    implementation = _runtime_implementation(args)
    descriptor = _job_descriptor(
        study_commit=args.study_commit, implementation=implementation, inputs=inputs
    )
    return descriptor, implementation, inputs


def _validate_staged_implementation(
    source_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], ModuleType]:
    paths = _source_paths(source_root)
    require(set(paths) == set(expected), "staged annotation media implementation changed")
    observed = {}
    for name, path in paths.items():
        identity = queue.file_identity(path)
        require(identity["sha256"] == expected[name]["sha256"],
                f"staged {name} hash changed")
        observed[name] = identity
    _validate_contract(paths["queue_contract"], expected["queue_contract"]["sha256"])
    bridge = _load_module(paths["bridge"], "wmf_detached_annotation_media_bridge")
    require(getattr(bridge, "INPUT_SCHEMA", None) == INPUT_SCHEMA
            and getattr(bridge, "PREPARATION_SCHEMA", None) == PREPARATION_SCHEMA
            and getattr(bridge, "EXPECTED_TOTAL_REQUESTS", None) == EXPECTED_REQUESTS
            and getattr(bridge, "EXPECTED_TOTAL_ELIGIBLE", None) == EXPECTED_ELIGIBLE
            and getattr(bridge, "EXPECTED_TOTAL_SELECTED", None) == EXPECTED_SELECTED,
            "staged annotation media bridge interface changed")
    loaded_paths = {
        "compiler": Path(bridge.compiler.__file__).resolve(),
        "timing_validator": Path(bridge.timing.__file__).resolve(),
        "annotation_workflow": Path(bridge.annotation.__file__).resolve(),
        "freeze_validator": Path(bridge.freeze.__file__).resolve(),
        "camera_replay": Path(bridge.camera_replay.__file__).resolve(),
    }
    for name, loaded in loaded_paths.items():
        require(loaded == paths[name].resolve(),
                f"bridge loaded a different {name} implementation")
    return observed, bridge


def _validate_runtime_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_input_shape(inputs)
    for label, value in (
        ("compiler job receipt", normalized["compiler"]["job_receipt"]),
        ("compiler receipt", normalized["compiler"]["compiler_receipt"]),
    ):
        require(queue.file_identity(Path(value["path"])) == value,
                f"{label} bytes/hash changed on PVC")
    for row in normalized["models"]:
        for field, value in row.items():
            if field == "model_id":
                continue
            require(queue.file_identity(Path(value["path"])) == value,
                    f"{row['model_id']} {field} bytes/hash changed on PVC")
    return normalized


def _expected_context_paths(context: Any) -> None:
    require(context.job_dir == CONTROL_ROOT / "jobs" / JOB_ID,
            "annotation media job directory changed")
    require(context.source_root == CONTROL_ROOT / "sources" / context.study_commit,
            "annotation media source checkout changed")


def _validate_preparation(
    bridge: ModuleType, output: Path, returned: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = output / "preparation_receipt.json"
    receipt = queue.load_json(receipt_path, "annotation media preparation receipt")
    require(receipt == returned, "bridge return value differs from preparation receipt")
    bridge.verify_signed(receipt, "annotation media preparation receipt")
    require(receipt.get("schema_version") == PREPARATION_SCHEMA
            and receipt.get("study_id") == STUDY_ID
            and receipt.get("stage") == "development"
            and receipt.get("status") == "prepared_pending_human_pixel_blindness_review"
            and receipt.get("human_pixel_blindness_review_complete") is False
            and receipt.get("rater_packets_created") is False
            and receipt.get("safe_for_rater_distribution") is False
            and receipt.get("safe_to_release_confirmation") is False,
            "annotation media preparation state changed")
    expected_counts = {
        "episodes": EXPECTED_EPISODES,
        "requests": EXPECTED_REQUESTS,
        "requests_by_model": {"N3": 240, "D1": 912},
        "source_timing_capable_requests": 480,
        "d1_forecast_timing_unavailable_requests": EXPECTED_D1_UNMAPPED,
        "timing_camera_action_eligible_requests": EXPECTED_ELIGIBLE,
        "selected_requests": EXPECTED_SELECTED,
        "selected_requests_by_model": {"N3": 64, "D1": 64},
        "camera_replay_exact_runtime_sessions": 2,
        "human_pixel_blindness_reviews": 0,
        "human_labels": 0,
    }
    counts = receipt.get("counts")
    require(isinstance(counts, Mapping), "annotation media preparation counts are missing")
    for key, wanted in expected_counts.items():
        require(counts.get(key) == wanted, f"annotation media preparation count changed: {key}")
    image_count = counts.get("rendered_pngs")
    require(type(image_count) is int and 736 <= image_count <= 768
            and counts.get("source_extraction_records") == image_count,
            "annotation media rendered/source-lineage count changed")
    require(counts.get("original_camera_frames_replayed_exact_runtime")
            == image_count - 2 * EXPECTED_SELECTED,
            "annotation media exact-runtime original-camera coverage changed")
    unique_review_count = counts.get("unique_pixel_review_assets")
    require(type(unique_review_count) is int and 0 < unique_review_count <= image_count,
            "annotation media pixel-review asset count changed")
    require(receipt.get("science_counts") == _zero_science_counts(),
            "annotation media preparation reports science activity")
    files: list[dict[str, Any]] = []
    observed: set[str] = set()
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), f"annotation media output contains a symlink: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"annotation media output contains a non-file: {path}")
        relative = str(path.relative_to(output))
        require(relative not in observed, "annotation media output path duplicated")
        observed.add(relative)
        files.append({**queue.file_identity(path), "relative_path": relative})
    expected_fixed = {
        "request_inventory.json", "request_selection.json",
        "source_extraction_lineage.json", "rendered_png_manifest.json",
        "image_inventory_pre_review.json", "pixel_blindness_review_checklist.json",
        "publish_tranche_index.json",
        "preparation_receipt.json",
        "camera_replay_runtime/n3_session_receipt.json",
        "camera_replay_runtime/d1_session_receipt.json",
        "camera_replay_runtime/n3_stderr.log",
        "camera_replay_runtime/d1_stderr.log",
    }
    require(expected_fixed <= observed, "annotation media output fixed files are incomplete")
    require(len(observed) == len(expected_fixed) + 3 * image_count,
            "annotation media output file count changed")
    require(sum(path.startswith("source_images/") for path in observed) == image_count
            and sum(path.startswith("annotation_media/") for path in observed) == image_count
            and sum(path.startswith("render_receipts/") for path in observed) == image_count,
            "annotation media per-image output inventory changed")
    replay_outputs = receipt.get("outputs", {}).get("camera_replay_runtime_receipts")
    require(isinstance(replay_outputs, Mapping) and set(replay_outputs) == set(MODELS),
            "annotation media exact-runtime replay receipt inventory changed")
    replay_count = 0
    for model in MODELS:
        relative = replay_outputs[model]
        require(isinstance(relative, Mapping)
                and set(relative) == {"path", "sha256", "bytes"}
                and relative.get("path")
                == f"camera_replay_runtime/{model.lower()}_session_receipt.json",
                f"{model} replay receipt descriptor changed")
        replay_path = output / relative["path"]
        require(_relative_identity_matches(replay_path, relative),
                f"{model} replay receipt bytes changed")
        replay = queue.load_json(replay_path, f"{model} replay receipt")
        bridge.verify_signed(replay, f"{model} replay receipt")
        crop_descriptor = receipt.get("inputs", {}).get("models", {}).get(
            model, {}
        ).get("camera_crop_contract")
        require(isinstance(crop_descriptor, Mapping),
                f"{model} preparation crop input is missing")
        crop_path = Path(str(crop_descriptor.get("path", "")))
        require(queue.file_identity(crop_path) == crop_descriptor,
                f"{model} preparation crop input changed")
        crop = bridge.load_json(crop_path, f"{model} camera crop contract")
        expected_python = bridge._runtime_python_identity(crop, model)
        stderr = replay.get("stderr_log")
        require(isinstance(stderr, Mapping)
                and set(stderr) == {"path", "sha256", "bytes"}
                and stderr.get("path")
                == f"camera_replay_runtime/{model.lower()}_stderr.log"
                and _relative_identity_matches(output / stderr["path"], stderr),
                f"{model} replay stderr identity changed")
        require(replay.get("schema_version") == bridge.REPLAY_SESSION_SCHEMA
                and replay.get("study_id") == STUDY_ID
                and replay.get("stage") == "development"
                and replay.get("status") == "passed_exact_signed_runtime_replay"
                and replay.get("model_id") == model
                and replay.get("camera_api") == "replay_original_camera_frame"
                and replay.get("runtime_dependency_validation")
                == "exact_signed_contract_match"
                and replay.get("camera_crop_contract") == crop_descriptor
                and replay.get("camera_crop_contract_payload_sha256")
                == crop.get("payload_sha256")
                and replay.get("python") == expected_python
                and replay.get("bridge_source", {}).get("sha256")
                == queue.sha256_file(Path(bridge.__file__).resolve())
                and replay.get("camera_replay_source", {}).get("sha256")
                == queue.sha256_file(Path(bridge.camera_replay.__file__).resolve())
                and replay.get("cuda_visible_devices") == ""
                and replay.get("model_loaded") is False
                and replay.get("simulator_state_render_used") is False
                and replay.get("safe_for_rater_distribution") is False
                and replay.get("safe_to_release_confirmation") is False,
                f"{model} exact-runtime replay receipt state changed")
        replay_count += replay.get("request_count", -1)
    require(replay_count == counts["original_camera_frames_replayed_exact_runtime"],
            "exact-runtime replay receipts do not cover every original frame")
    selection = queue.load_json(output / "request_selection.json", "request selection")
    inventory = queue.load_json(output / "request_inventory.json", "request inventory")
    checklist = queue.load_json(
        output / "pixel_blindness_review_checklist.json", "pixel-review checklist"
    )
    bridge.verify_signed(checklist, "pixel-review checklist")
    require(checklist.get("schema_version") == bridge.PIXEL_REVIEW_CHECKLIST_SCHEMA
            and checklist.get("status") == "awaiting_named_human_visual_inspection"
            and checklist.get("unique_asset_count") == unique_review_count
            and checklist.get("human_pixel_blindness_review_complete") is False
            and checklist.get("safe_for_rater_distribution") is False,
            "pixel-review checklist state changed")
    bridge.annotation.verify_signed(selection, "request selection")
    bridge._assert_inventory_and_selection(inventory, selection)
    selected_zero = sum(
        wrapper.get("selected") is True
        and wrapper.get("source", {}).get("request_index") == 0
        for wrapper in selection["requests"]
    )
    require(counts.get("selected_request_zero_count") == selected_zero
            and image_count == EXPECTED_SELECTED * 6 - selected_zero,
            "annotation media selected-request-zero/render accounting changed")
    tranche_path = output / PUBLISH_INDEX_NAME
    tranche_index = _validate_tranche_index(
        queue.load_json(tranche_path, "publish tranche index")
    )
    tranche_descriptor = receipt.get("outputs", {}).get("publish_tranche_index")
    require(isinstance(tranche_descriptor, Mapping)
            and _relative_identity_matches(tranche_path, tranche_descriptor)
            and tranche_index.get("render_alias_count") == image_count
            and counts.get("unique_sanitized_png_assets")
            == tranche_index.get("unique_asset_count")
            and counts.get("publish_tranches") == tranche_index.get("tranche_count")
            and counts.get("total_unique_sanitized_png_bytes")
            == tranche_index.get("total_unique_asset_bytes"),
            "annotation media preparation/tranche accounting changed")
    return receipt, files


def _validate_tranche_index(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "publish tranche index must be an object")
    require(set(value) == {
        "schema_version", "study_id", "stage", "status", "visibility",
        "unique_asset_count", "render_alias_count", "tranche_count",
        "total_unique_asset_bytes", "source_extraction_lineage",
        "rendered_png_manifest", "deduplication_key", "tranches_non_overlapping",
        "each_asset_published_once", "human_pixel_blindness_review_complete",
        "safe_for_rater_distribution", "safe_to_release_confirmation", "tranches",
        "payload_sha256",
    }, "publish tranche index fields changed")
    queue.verify_signed_document(value, "publish tranche index")
    require(value.get("schema_version") == TRANCHE_INDEX_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("stage") == "development"
            and value.get("status")
            == "prepared_for_bounded_result_return_pending_human_review"
            and value.get("visibility")
            == "RESTRICTED ANALYST MANIFEST; NEVER DISTRIBUTE TO RATERS"
            and value.get("deduplication_key") == "annotation_media_sha256"
            and value.get("tranches_non_overlapping") is True
            and value.get("each_asset_published_once") is True
            and value.get("human_pixel_blindness_review_complete") is False
            and value.get("safe_for_rater_distribution") is False
            and value.get("safe_to_release_confirmation") is False,
            "publish tranche index state changed")
    tranches = value.get("tranches")
    require(isinstance(tranches, list) and tranches,
            "publish tranche inventory is empty")
    require(value.get("tranche_count") == len(tranches),
            "publish tranche count changed")
    for name in ("source_extraction_lineage", "rendered_png_manifest"):
        evidence = value.get(name)
        require(isinstance(evidence, Mapping)
                and set(evidence) == {"path", "sha256", "bytes"}
                and isinstance(evidence.get("path"), str)
                and evidence["path"] and not Path(evidence["path"]).is_absolute()
                and isinstance(evidence.get("sha256"), str)
                and queue._verified_sha(evidence["sha256"], f"{name} digest")
                == evidence["sha256"]
                and type(evidence.get("bytes")) is int and evidence["bytes"] > 0,
                f"publish tranche {name} descriptor changed")
    seen_ids: set[str] = set()
    seen_assets: set[str] = set()
    seen_source_images: set[str] = set()
    alias_count = 0
    total_bytes = 0
    for ordinal, row in enumerate(tranches, 1):
        require(isinstance(row, Mapping), "publish tranche row is invalid")
        require(set(row) == {
            "tranche_id", "ordinal", "asset_count", "alias_count", "asset_bytes",
            "asset_budget_bytes", "publisher_file_limit_bytes",
            "publisher_job_limit_bytes", "assets",
        }, "publish tranche row fields changed")
        tranche_id = row.get("tranche_id")
        require(tranche_id == f"development-annotation-media-tranche-{ordinal:03d}"
                and tranche_id not in seen_ids and row.get("ordinal") == ordinal,
                "publish tranche identity/order changed")
        seen_ids.add(tranche_id)
        assets = row.get("assets")
        require(isinstance(assets, list) and assets,
                f"{tranche_id} asset inventory is empty")
        require(row.get("asset_count") == len(assets)
                and row.get("asset_budget_bytes") == PUBLISH_TRANCHE_ASSET_BUDGET_BYTES
                and row.get("publisher_file_limit_bytes") == PUBLISH_FILE_LIMIT_BYTES
                and row.get("publisher_job_limit_bytes") == PUBLISH_JOB_LIMIT_BYTES,
                f"{tranche_id} publisher boundary changed")
        observed_bytes = 0
        observed_aliases = 0
        digests = []
        for asset in assets:
            require(isinstance(asset, Mapping), f"{tranche_id} asset is invalid")
            require(set(asset) == {
                "asset_sha256", "source_path", "bytes", "width_px", "height_px",
                "aliases",
            }, f"{tranche_id} asset fields changed")
            digest = queue._verified_sha(asset.get("asset_sha256"), "tranche asset digest")
            require(digest not in seen_assets, "sanitized PNG is assigned to multiple tranches")
            seen_assets.add(digest)
            digests.append(digest)
            size = asset.get("bytes")
            require(type(size) is int and 0 < size < PUBLISH_FILE_LIMIT_BYTES,
                    "sanitized PNG violates per-file publisher limit")
            require(isinstance(asset.get("source_path"), str)
                    and not Path(asset["source_path"]).is_absolute(),
                    "tranche source path must be preparation-relative")
            aliases = asset.get("aliases")
            require(isinstance(aliases, list) and aliases,
                    "tranche asset has no provenance aliases")
            asset_source_ids: set[str] = set()
            for alias in aliases:
                require(isinstance(alias, Mapping) and set(alias) == {
                    "source_image_id", "source_request_id", "image_role", "join_key",
                    "camera_id", "camera_crop_id", "camera_crop_sha256",
                    "alignment_receipt_id", "alignment_receipt_sha256",
                    "render_receipt_id", "render_receipt_sha256",
                    "source_lineage_record_sha256",
                }, "tranche provenance alias fields changed")
                join = alias.get("join_key")
                require(isinstance(join, Mapping)
                        and set(join) == {"model_id", "cell_id", "request_index"}
                        and join.get("model_id") in MODELS
                        and isinstance(join.get("cell_id"), str) and join["cell_id"]
                        and type(join.get("request_index")) is int
                        and join["request_index"] >= 0,
                        "tranche provenance join key changed")
                source_image_id = alias.get("source_image_id")
                require(isinstance(source_image_id, str) and source_image_id
                        and source_image_id not in seen_source_images,
                        "tranche source image identity is invalid or duplicated")
                seen_source_images.add(source_image_id)
                asset_source_ids.add(source_image_id)
                require(alias.get("image_role") in {
                    "current", "preceding", "predicted", "executed",
                    "early_predicted", "early_executed",
                } and alias.get("camera_id") == "over_shoulder_left_camera",
                        "tranche image role/camera changed")
                expected_height = 168 if join["model_id"] == "N3" else 176
                expected_crop = (
                    "n3-over-shoulder-left-168x320-v1"
                    if join["model_id"] == "N3"
                    else "d1-over-shoulder-left-176x320-v1"
                )
                require(asset.get("width_px") == 320
                        and asset.get("height_px") == expected_height
                        and alias.get("camera_crop_id") == expected_crop,
                        "tranche model-specific image geometry changed")
                for field in (
                    "camera_crop_sha256", "alignment_receipt_sha256",
                    "render_receipt_sha256", "source_lineage_record_sha256",
                ):
                    queue._verified_sha(alias.get(field), f"tranche alias {field}")
            require(Path(asset["source_path"]).parent == Path("annotation_media")
                    and Path(asset["source_path"]).suffix == ".png"
                    and Path(asset["source_path"]).stem in asset_source_ids,
                    "tranche canonical source path is not one of its aliases")
            observed_bytes += size
            observed_aliases += len(aliases)
        require(digests == sorted(digests), f"{tranche_id} assets are not hash-sorted")
        require(observed_bytes == row.get("asset_bytes")
                and observed_bytes <= PUBLISH_TRANCHE_ASSET_BUDGET_BYTES
                and observed_aliases == row.get("alias_count"),
                f"{tranche_id} asset/alias accounting changed")
        total_bytes += observed_bytes
        alias_count += observed_aliases
    require([digest for row in tranches for digest in [
        asset["asset_sha256"] for asset in row["assets"]
    ]] == sorted(seen_assets), "publish tranche assets are not globally hash-sorted")
    require(value.get("unique_asset_count") == len(seen_assets)
            and value.get("render_alias_count") == alias_count
            and value.get("total_unique_asset_bytes") == total_bytes,
            "publish tranche global accounting changed")
    return dict(value)


def _tranche_descriptor(
    *, study_commit: str, implementation: Mapping[str, Mapping[str, Any]],
    preparation_receipt: Mapping[str, Any], index_descriptor: Mapping[str, Any],
    tranche_id: str, role: str,
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(role in TRANCHE_WORKER_ROLES, "tranche worker role is unsupported")
    argv = [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        "publish-tranche",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", tranche_id,
        "--expected-role", role,
        "--preparation-job-receipt", str(preparation_receipt["path"]),
        "--preparation-job-receipt-sha256", str(preparation_receipt["sha256"]),
        "--preparation-job-receipt-bytes", str(preparation_receipt["bytes"]),
        "--tranche-index", str(index_descriptor["path"]),
        "--tranche-index-sha256", str(index_descriptor["sha256"]),
        "--tranche-index-bytes", str(index_descriptor["bytes"]),
        "--tranche-id", tranche_id,
    ]
    argv.extend(_implementation_hash_arguments(implementation))
    return {
        "job_id": tranche_id,
        "released": True,
        "source_commit": commit,
        "role": role,
        "argv": argv,
        "max_wall_seconds": 3600,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def build_tranche_wave(
    *, study_commit: str, preparation_job_receipt: Mapping[str, Any],
    tranche_index_descriptor: Mapping[str, Any], tranche_index: Mapping[str, Any],
) -> dict[str, Any]:
    implementation = _local_implementation()
    receipt_descriptor = _descriptor_shape(
        preparation_job_receipt, "preparation job receipt"
    )
    index_descriptor = _descriptor_shape(
        tranche_index_descriptor, "publish tranche index"
    )
    index = _validate_tranche_index(tranche_index)
    jobs = []
    for ordinal, tranche in enumerate(index["tranches"]):
        jobs.append(_tranche_descriptor(
            study_commit=study_commit,
            implementation=implementation,
            preparation_receipt=receipt_descriptor,
            index_descriptor=index_descriptor,
            tranche_id=tranche["tranche_id"],
            role=TRANCHE_WORKER_ROLES[ordinal % len(TRANCHE_WORKER_ROLES)],
        ))
    return {
        "schema_version": TRANCHE_WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": "bounded_sanitized_media_result_return",
        "source_commit": queue._verified_commit(study_commit),
        "status": "descriptor_only_not_dispatched_all_tranches_non_overlapping",
        "jobs": jobs,
        "implementation": implementation,
        "preparation_job_receipt": receipt_descriptor,
        "publish_tranche_index": index_descriptor,
        "tranche_count": len(jobs),
        "unique_asset_count": index["unique_asset_count"],
        "total_unique_asset_bytes": index["total_unique_asset_bytes"],
        "each_asset_published_once": True,
        "science_counts": _zero_science_counts(),
        "human_pixel_blindness_review_complete": False,
        "safe_for_rater_distribution": False,
        "safe_to_release_confirmation": False,
    }


def _runtime_file_descriptor(path: Path, digest: str, size: int, label: str) -> dict[str, Any]:
    return _descriptor_shape(
        {"path": str(path), "sha256": digest, "bytes": size}, label
    )


def _runtime_tranche_descriptor(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    require(args.command == "publish-tranche", "tranche runtime mode changed")
    require(args.job_id == args.tranche_id, "tranche runtime job identity changed")
    implementation = _runtime_implementation(args)
    receipt = _runtime_file_descriptor(
        args.preparation_job_receipt,
        args.preparation_job_receipt_sha256,
        args.preparation_job_receipt_bytes,
        "preparation job receipt",
    )
    index_descriptor = _runtime_file_descriptor(
        args.tranche_index,
        args.tranche_index_sha256,
        args.tranche_index_bytes,
        "publish tranche index",
    )
    descriptor = _tranche_descriptor(
        study_commit=args.study_commit,
        implementation=implementation,
        preparation_receipt=receipt,
        index_descriptor=index_descriptor,
        tranche_id=args.tranche_id,
        role=args.expected_role,
    )
    return descriptor, implementation, receipt, index_descriptor


def _copy_exclusive(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as read, target.open("xb") as write:
        for block in iter(lambda: read.read(1024 * 1024), b""):
            write.write(block)
        write.flush()
        os.fsync(write.fileno())


def _secure_preparation_file(root: Path, relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative
            and not Path(relative).is_absolute(), f"{label} path is invalid")
    root = root.resolve(strict=True)
    lexical = root / relative
    cursor = lexical
    while cursor != root:
        require(cursor != cursor.parent and not cursor.is_symlink(),
                f"{label} path contains a symlink or escapes preparation root")
        cursor = cursor.parent
    resolved = lexical.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file()
            and not resolved.is_symlink(), f"{label} escapes preparation root")
    return resolved


def _relative_identity_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    observed = queue.file_identity(path)
    return (observed["sha256"], observed["bytes"]) == (
        expected.get("sha256"), expected.get("bytes")
    )


def _validate_terminal_preparation_job(
    value: Any,
    *,
    receipt_descriptor: Mapping[str, Any],
    index_descriptor: Mapping[str, Any],
    study_commit: str,
    expected_implementation: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Authenticate the sole preparation receipt allowed to release tranches.

    The queue document hash is an integrity checksum, not an authorization
    signature.  Consequently every authority-relevant field is checked here,
    and the tranche index supplied to a follow-on worker must be the exact raw
    index descriptor emitted by this terminal preparation receipt.
    """

    require(isinstance(value, Mapping), "preparation job receipt must be an object")
    require(set(value) == {
        "schema_version", "namespace", "study_id", "mode", "status", "decision",
        "job_id", "job_dir", "study_commit", "queue_role", "worker_id",
        "runtime_identity", "queue_descriptor", "queue_claim", "implementation",
        "inputs", "outputs", "counts", "science_counts",
        "human_pixel_blindness_review_complete", "rater_packets_created",
        "safe_for_rater_distribution", "safe_to_release_confirmation",
        "raw_outputs_recoverable_on_gm_pvc", "published_files", "claim_boundary",
        "completed_at_utc", "payload_sha256",
    }, "preparation job receipt fields changed")
    queue.verify_signed_document(value, "preparation job receipt")
    commit = queue._verified_commit(study_commit)
    job_dir = CONTROL_ROOT / "jobs" / JOB_ID
    expected_receipt = _descriptor_shape(
        receipt_descriptor, "preparation job receipt"
    )
    expected_index = _descriptor_shape(index_descriptor, "publish tranche index")
    require(Path(expected_receipt["path"])
            == job_dir / "publish" / PUBLISH_RECEIPT_NAME,
            "preparation job receipt path is not the canonical terminal publication")
    require(Path(expected_index["path"])
            == job_dir / OUTPUT_RELATIVE / PUBLISH_INDEX_NAME,
            "publish tranche index path is not the canonical raw preparation output")
    require(value.get("schema_version") == JOB_RECEIPT_SCHEMA
            and value.get("namespace") == NAMESPACE
            and value.get("study_id") == STUDY_ID
            and value.get("mode") == MODE
            and value.get("status") == "passed"
            and value.get("decision")
            == "go_for_human_pixel_blindness_review_only"
            and value.get("job_id") == JOB_ID
            and value.get("job_dir") == str(job_dir)
            and value.get("study_commit") == commit
            and value.get("queue_role") == WORKER_ROLE
            and value.get("worker_id") == WORKER_ROLE
            and value.get("science_counts") == _zero_science_counts()
            and value.get("human_pixel_blindness_review_complete") is False
            and value.get("rater_packets_created") is False
            and value.get("safe_for_rater_distribution") is False
            and value.get("safe_to_release_confirmation") is False
            and value.get("raw_outputs_recoverable_on_gm_pvc") is True
            and value.get("published_files")
            == [PUBLISH_RECEIPT_NAME, PUBLISH_INDEX_NAME]
            and value.get("claim_boundary") == PREPARATION_CLAIM_BOUNDARY,
            "preparation job receipt did not pass its exact restricted gate")
    completed = value.get("completed_at_utc")
    require(isinstance(completed, str) and completed,
            "preparation job completion timestamp is missing")
    runtime = value.get("runtime_identity")
    require(isinstance(runtime, Mapping)
            and set(runtime) == {"hostname", "pod_uid", "pid"}
            and isinstance(runtime.get("hostname"), str)
            and runtime["hostname"].startswith(WORKER_ROLE + "-")
            and isinstance(runtime.get("pod_uid"), str) and runtime["pod_uid"]
            and type(runtime.get("pid")) is int and runtime["pid"] > 0,
            "preparation job runtime identity changed")
    for name, relative in (
        ("queue_descriptor", Path("descriptor.json")),
        ("queue_claim", Path("claim/owner.json")),
    ):
        descriptor = _descriptor_shape(value.get(name), f"preparation {name}")
        require(Path(descriptor["path"]) == job_dir / relative,
                f"preparation {name} path changed")

    implementation = value.get("implementation")
    require(isinstance(implementation, Mapping)
            and set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
            "preparation implementation inventory changed")
    source_root = CONTROL_ROOT / "sources" / commit
    normalized_implementation: dict[str, dict[str, Any]] = {}
    for name, relative in (
        (name, path.relative_to(REPOSITORY_ROOT))
        for name, path in _source_paths(REPOSITORY_ROOT).items()
    ):
        descriptor = _descriptor_shape(
            implementation.get(name), f"preparation implementation {name}"
        )
        require(Path(descriptor["path"]) == source_root / relative,
                f"preparation implementation {name} path changed")
        normalized_implementation[name] = descriptor
    if expected_implementation is not None:
        require(normalized_implementation == dict(expected_implementation),
                "preparation implementation differs from tranche runtime")

    inputs = value.get("inputs")
    require(validate_input_shape(inputs) == inputs,
            "preparation input manifest fields/order changed")
    outputs = value.get("outputs")
    require(isinstance(outputs, Mapping) and set(outputs) == {
        "input_manifest", "preparation_receipt", "output_root", "output_file_count",
        "output_files_sha256", "publish_tranche_index",
    }, "preparation job output inventory changed")
    output_root = job_dir / OUTPUT_RELATIVE
    require(outputs.get("output_root") == str(output_root),
            "preparation output root changed")
    for name, expected_path in (
        ("input_manifest", job_dir / MANIFEST_RELATIVE),
        ("preparation_receipt", output_root / "preparation_receipt.json"),
    ):
        descriptor = _descriptor_shape(outputs.get(name), f"preparation output {name}")
        require(Path(descriptor["path"]) == expected_path,
                f"preparation output {name} path changed")
    require(outputs.get("publish_tranche_index") == expected_index,
            "supplied tranche index is not the exact preparation output descriptor")
    require(queue._verified_sha(
        outputs.get("output_files_sha256"), "preparation output inventory digest"
    ) == outputs.get("output_files_sha256"),
            "preparation output inventory digest changed")

    counts = value.get("counts")
    require(isinstance(counts, Mapping) and set(counts) == {
        "episodes", "requests", "requests_by_model", "source_timing_capable_requests",
        "d1_forecast_timing_unavailable_requests",
        "timing_camera_action_eligible_requests", "selected_requests",
        "selected_requests_by_model", "selected_request_zero_count",
        "source_extraction_records", "rendered_pngs", "unique_pixel_review_assets",
        "unique_sanitized_png_assets", "publish_tranches",
        "total_unique_sanitized_png_bytes", "camera_replay_exact_runtime_sessions",
        "original_camera_frames_replayed_exact_runtime",
        "human_pixel_blindness_reviews", "human_labels",
    }, "preparation job count fields changed")
    require(counts.get("episodes") == EXPECTED_EPISODES
            and counts.get("requests") == EXPECTED_REQUESTS
            and counts.get("requests_by_model") == {"N3": 240, "D1": 912}
            and counts.get("source_timing_capable_requests") == 480
            and counts.get("d1_forecast_timing_unavailable_requests")
            == EXPECTED_D1_UNMAPPED
            and counts.get("timing_camera_action_eligible_requests")
            == EXPECTED_ELIGIBLE
            and counts.get("selected_requests") == EXPECTED_SELECTED
            and counts.get("selected_requests_by_model") == {"N3": 64, "D1": 64}
            and counts.get("camera_replay_exact_runtime_sessions") == 2
            and counts.get("human_pixel_blindness_reviews") == 0
            and counts.get("human_labels") == 0,
            "preparation job exact counts changed")
    selected_zero = counts.get("selected_request_zero_count")
    rendered = counts.get("rendered_pngs")
    require(type(selected_zero) is int and 0 <= selected_zero <= EXPECTED_SELECTED
            and type(rendered) is int
            and rendered == EXPECTED_SELECTED * 6 - selected_zero
            and counts.get("source_extraction_records") == rendered
            and counts.get("original_camera_frames_replayed_exact_runtime")
            == rendered - 2 * EXPECTED_SELECTED,
            "preparation job render/replay counts changed")
    require(type(counts.get("unique_pixel_review_assets")) is int
            and 0 < counts["unique_pixel_review_assets"] <= rendered
            and type(counts.get("unique_sanitized_png_assets")) is int
            and 0 < counts["unique_sanitized_png_assets"] <= rendered
            and type(counts.get("publish_tranches")) is int
            and counts["publish_tranches"] > 0
            and type(counts.get("total_unique_sanitized_png_bytes")) is int
            and counts["total_unique_sanitized_png_bytes"] > 0,
            "preparation job media/tranche counts changed")
    require(outputs.get("output_file_count") == 12 + 3 * rendered,
            "preparation output file count changed")
    return dict(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_main_transactionally(
    *, raw: Path, publish: Path, index_source: Path,
    receipt: Mapping[str, Any],
) -> None:
    """Expose the main index and terminal receipt as one directory rename."""

    require(raw.is_dir() and publish.is_dir() and not publish.is_symlink()
            and not any(publish.iterdir()),
            "annotation media main publish directory is not exactly empty")
    staging = raw / "main_publish_staging"
    require(not staging.exists() and not staging.is_symlink(),
            "annotation media main publish staging path already exists")
    staging.mkdir()
    _copy_exclusive(index_source, staging / PUBLISH_INDEX_NAME)
    require(queue.file_identity(staging / PUBLISH_INDEX_NAME)["bytes"]
            < PUBLISH_FILE_LIMIT_BYTES,
            "publish tranche index exceeds the per-file result limit")
    queue.immutable_json(
        staging / PUBLISH_RECEIPT_NAME, receipt, maximum_bytes=2 * 1024 * 1024
    )
    observed = sorted(path.name for path in staging.iterdir())
    require(observed == sorted([PUBLISH_INDEX_NAME, PUBLISH_RECEIPT_NAME]),
            "annotation media main staged publication inventory changed")
    total = sum(path.stat().st_size for path in staging.iterdir() if path.is_file())
    require(total < PUBLISH_JOB_LIMIT_BYTES,
            "annotation media main publication exceeds the job result limit")
    _fsync_directory(staging)
    publish.rmdir()
    os.replace(staging, publish)
    _fsync_directory(publish.parent)


def run_tranche_job(args: argparse.Namespace) -> dict[str, Any]:
    descriptor, expected_implementation, receipt_descriptor, index_descriptor = (
        _runtime_tranche_descriptor(args)
    )
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
        require(context.job_dir == CONTROL_ROOT / "jobs" / args.tranche_id,
                "tranche job directory changed")
        implementation, _ = _validate_staged_implementation(
            context.source_root, expected_implementation
        )
        require(queue.file_identity(Path(receipt_descriptor["path"])) == receipt_descriptor,
                "preparation job receipt changed on PVC")
        preparation_job = queue.load_json(
            Path(receipt_descriptor["path"]), "preparation job receipt"
        )
        _validate_terminal_preparation_job(
            preparation_job,
            receipt_descriptor=receipt_descriptor,
            index_descriptor=index_descriptor,
            study_commit=context.study_commit,
            expected_implementation=implementation,
        )
        require(queue.file_identity(Path(index_descriptor["path"])) == index_descriptor,
                "publish tranche index changed on PVC")
        index = _validate_tranche_index(queue.load_json(
            Path(index_descriptor["path"]), "publish tranche index"
        ))
        matches = [row for row in index["tranches"] if row["tranche_id"] == args.tranche_id]
        require(len(matches) == 1, "requested publish tranche is absent or duplicated")
        tranche = matches[0]
        expected_role = TRANCHE_WORKER_ROLES[(tranche["ordinal"] - 1) % len(TRANCHE_WORKER_ROLES)]
        require(context.role == expected_role, "publish tranche worker assignment changed")
        raw, publish = queue._prepare_output_directories(context.job_dir)
        require(not any(publish.iterdir()), "publish tranche directory is not empty")
        staging = raw / "publish_staging"
        require(not staging.exists() and not staging.is_symlink(),
                "publish tranche staging path already exists")
        staging.mkdir()
        preparation_root = Path(index_descriptor["path"]).parent
        for name in ("source_extraction_lineage", "rendered_png_manifest"):
            evidence = index[name]
            evidence_path = _secure_preparation_file(
                preparation_root, evidence["path"], name
            )
            require(_relative_identity_matches(evidence_path, evidence),
                    f"canonical {name} changed on PVC")
        published_assets = []
        for asset in tranche["assets"]:
            source = _secure_preparation_file(
                preparation_root, asset["source_path"], "publish tranche asset"
            )
            identity = queue.file_identity(source)
            require(identity["sha256"] == asset["asset_sha256"]
                    and identity["bytes"] == asset["bytes"],
                    "publish tranche source asset changed")
            target = staging / "assets" / f"{asset['asset_sha256']}.png"
            _copy_exclusive(source, target)
            copied = queue.file_identity(target)
            require(copied["sha256"] == asset["asset_sha256"]
                    and copied["bytes"] == asset["bytes"],
                    "published sanitized PNG differs from canonical raw asset")
            published_assets.append({
                "asset_sha256": asset["asset_sha256"],
                "published_path": f"assets/{asset['asset_sha256']}.png",
                "bytes": asset["bytes"],
                "width_px": asset["width_px"],
                "height_px": asset["height_px"],
                "aliases": asset["aliases"],
            })
        manifest = queue.signed_document({
            "schema_version": TRANCHE_MANIFEST_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "sanitized_assets_returned_pending_human_pixel_review",
            "visibility": "RESTRICTED ANALYST MANIFEST; NEVER DISTRIBUTE TO RATERS",
            "tranche_id": args.tranche_id,
            "tranche_ordinal": tranche["ordinal"],
            "publish_tranche_index": index_descriptor,
            "source_extraction_lineage": index["source_extraction_lineage"],
            "rendered_png_manifest": index["rendered_png_manifest"],
            "asset_count": len(published_assets),
            "alias_count": sum(len(row["aliases"]) for row in published_assets),
            "asset_bytes": sum(row["bytes"] for row in published_assets),
            "assets": published_assets,
            "assets_deduplicated_by_sha256": True,
            "human_pixel_blindness_review_complete": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
        })
        queue.immutable_json(
            staging / TRANCHE_MANIFEST_NAME, manifest,
            maximum_bytes=PUBLISH_FILE_LIMIT_BYTES - 1,
        )
        staged_manifest_identity = queue.file_identity(staging / TRANCHE_MANIFEST_NAME)
        manifest_identity = {
            **staged_manifest_identity,
            "path": str(publish / TRANCHE_MANIFEST_NAME),
        }
        require(
            tranche["asset_bytes"] + manifest_identity["bytes"] + 2 * 1024 * 1024
            < PUBLISH_JOB_LIMIT_BYTES,
            "publish tranche leaves insufficient room for terminal receipt",
        )
        receipt = queue.signed_document({
            "schema_version": TRANCHE_JOB_RECEIPT_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "passed",
            "decision": "sanitized_assets_returned_for_human_review_only",
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
            "preparation_job_receipt": receipt_descriptor,
            "publish_tranche_index": index_descriptor,
            "tranche_manifest": manifest_identity,
            "asset_count": len(published_assets),
            "alias_count": manifest["alias_count"],
            "asset_bytes": manifest["asset_bytes"],
            "published_files": [
                *[row["published_path"] for row in published_assets],
                TRANCHE_MANIFEST_NAME,
                TRANCHE_RECEIPT_NAME,
            ],
            "each_asset_published_once_in_global_index": True,
            "science_counts": _zero_science_counts(),
            "human_pixel_blindness_review_complete": False,
            "rater_packets_created": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(
            staging / TRANCHE_RECEIPT_NAME, receipt,
            maximum_bytes=PUBLISH_FILE_LIMIT_BYTES - 1,
        )
        total = sum(
            path.stat().st_size for path in staging.rglob("*") if path.is_file()
        )
        require(total < PUBLISH_JOB_LIMIT_BYTES,
                "publish tranche exceeds the per-job result limit")
        require(
            queue.file_identity(staging / TRANCHE_RECEIPT_NAME)["sha256"]
            == queue.sha256_file(staging / TRANCHE_RECEIPT_NAME),
            "publish tranche terminal receipt changed during staging",
        )
        if (staging / "assets").is_dir():
            _fsync_directory(staging / "assets")
        _fsync_directory(staging)
        # Publish becomes visible only after every byte, manifest, and terminal
        # receipt passed its bound. A technical failure before this rename can
        # expose only the failure receipt, never a partial asset population.
        publish.rmdir()
        os.replace(staging, publish)
        _fsync_directory(context.job_dir)
        return receipt
    except BaseException as error:
        _write_failure_receipt(context, Path(args.job_dir), error)
        raise


def _write_failure_receipt(context: Any, job_dir: Path, error: BaseException) -> None:
    try:
        publish = Path(job_dir) / "publish"
        publish.mkdir(parents=True, exist_ok=True)
        success = any(
            (publish / name).exists()
            for name in (PUBLISH_RECEIPT_NAME, TRANCHE_RECEIPT_NAME)
        )
        target = publish / PUBLISH_FAILURE_NAME
        if success or target.exists():
            return
        receipt = queue.signed_document({
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
            "human_pixel_blindness_review_complete": False,
            "rater_packets_created": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(target, receipt, maximum_bytes=512 * 1024)
    except BaseException:
        return


def run_formal_job(args: argparse.Namespace) -> dict[str, Any]:
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
        implementation, bridge = _validate_staged_implementation(
            context.source_root, expected_implementation
        )
        inputs = _validate_runtime_inputs(expected_inputs)
        raw, publish = queue._prepare_output_directories(context.job_dir)
        manifest_path = context.job_dir / MANIFEST_RELATIVE
        output_path = context.job_dir / OUTPUT_RELATIVE
        require(manifest_path.parent == raw and output_path.parent == raw,
                "annotation media output paths changed")
        queue.immutable_json(manifest_path, inputs, maximum_bytes=2 * 1024 * 1024)
        manifest_identity = queue.file_identity(manifest_path)
        returned = bridge.prepare(
            manifest_path, manifest_identity["sha256"], output_path
        )
        preparation, output_files = _validate_preparation(bridge, output_path, returned)
        receipt = queue.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": MODE,
            "status": "passed",
            "decision": "go_for_human_pixel_blindness_review_only",
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
            "inputs": inputs,
            "outputs": {
                "input_manifest": manifest_identity,
                "preparation_receipt": queue.file_identity(
                    output_path / "preparation_receipt.json"
                ),
                "output_root": str(output_path),
                "output_file_count": len(output_files),
                "output_files_sha256": queue.sha256_bytes(
                    queue.compact_bytes(output_files)
                ),
                "publish_tranche_index": queue.file_identity(
                    output_path / "publish_tranche_index.json"
                ),
            },
            "counts": preparation["counts"],
            "science_counts": _zero_science_counts(),
            "human_pixel_blindness_review_complete": False,
            "rater_packets_created": False,
            "safe_for_rater_distribution": False,
            "safe_to_release_confirmation": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "published_files": [PUBLISH_RECEIPT_NAME, PUBLISH_INDEX_NAME],
            "claim_boundary": PREPARATION_CLAIM_BOUNDARY,
            "completed_at_utc": queue.utc_now(),
        })
        index_source = output_path / "publish_tranche_index.json"
        # The compact index is required on the results branch so the workstation
        # can emit follow-on result-return descriptors. Both files become
        # visible together; a receipt-write failure leaves publish empty.
        _publish_main_transactionally(
            raw=raw, publish=publish, index_source=index_source, receipt=receipt
        )
        return receipt
    except BaseException as error:
        _write_failure_receipt(context, Path(args.job_dir), error)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    emit = commands.add_parser("emit-formal")
    emit.add_argument("--study-commit", required=True)
    emit.add_argument("--inputs", type=Path, required=True)
    emit.add_argument("--inputs-sha256", required=True)
    emit.add_argument("--output", type=Path)

    emit_tranches = commands.add_parser("emit-tranches")
    emit_tranches.add_argument("--study-commit", required=True)
    emit_tranches.add_argument("--preparation-job-receipt", type=Path, required=True)
    emit_tranches.add_argument("--preparation-job-receipt-sha256", required=True)
    emit_tranches.add_argument("--tranche-index", type=Path, required=True)
    emit_tranches.add_argument("--tranche-index-sha256", required=True)
    emit_tranches.add_argument("--output", type=Path)

    runtime = commands.add_parser("formal-full")
    runtime.add_argument("--source-root", type=Path, required=True)
    runtime.add_argument("--study-commit", required=True)
    runtime.add_argument("--job-dir", type=Path, required=True)
    runtime.add_argument("--job-id", required=True)
    runtime.add_argument("--expected-role", required=True)
    runtime.add_argument("--inputs-json", required=True)
    for name in sorted(_source_paths(REPOSITORY_ROOT)):
        runtime.add_argument(f"--{name.replace('_', '-')}-sha256", required=True)

    tranche = commands.add_parser("publish-tranche")
    tranche.add_argument("--source-root", type=Path, required=True)
    tranche.add_argument("--study-commit", required=True)
    tranche.add_argument("--job-dir", type=Path, required=True)
    tranche.add_argument("--job-id", required=True)
    tranche.add_argument("--expected-role", required=True)
    tranche.add_argument("--preparation-job-receipt", type=Path, required=True)
    tranche.add_argument("--preparation-job-receipt-sha256", required=True)
    tranche.add_argument("--preparation-job-receipt-bytes", type=int, required=True)
    tranche.add_argument("--tranche-index", type=Path, required=True)
    tranche.add_argument("--tranche-index-sha256", required=True)
    tranche.add_argument("--tranche-index-bytes", type=int, required=True)
    tranche.add_argument("--tranche-id", required=True)
    for name in sorted(_source_paths(REPOSITORY_ROOT)):
        tranche.add_argument(f"--{name.replace('_', '-')}-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "emit-formal":
        identity = queue.file_identity(args.inputs)
        require(identity["sha256"] == queue._verified_sha(
            args.inputs_sha256, "annotation media input hash"
        ), "annotation media input file hash changed")
        wave = build_formal_wave(
            study_commit=args.study_commit,
            inputs=queue.load_json(args.inputs, "annotation media input"),
        )
        queue._write_descriptor_output(args.output, wave)
        return 0
    if args.command == "emit-tranches":
        receipt_identity = queue.file_identity(args.preparation_job_receipt)
        require(receipt_identity["sha256"] == queue._verified_sha(
            args.preparation_job_receipt_sha256, "preparation job receipt digest"
        ), "preparation job receipt file hash changed")
        receipt = queue.load_json(
            args.preparation_job_receipt, "preparation job receipt"
        )
        queue.verify_signed_document(receipt, "preparation job receipt")
        require(receipt.get("schema_version") == JOB_RECEIPT_SCHEMA
                and receipt.get("status") == "passed",
                "preparation job receipt did not pass")
        raw_receipt_path = Path(receipt.get("job_dir", "")) / "publish" / PUBLISH_RECEIPT_NAME
        raw_receipt_descriptor = {
            "path": str(raw_receipt_path),
            "sha256": receipt_identity["sha256"],
            "bytes": receipt_identity["bytes"],
        }
        index_identity = queue.file_identity(args.tranche_index)
        require(index_identity["sha256"] == queue._verified_sha(
            args.tranche_index_sha256, "publish tranche index digest"
        ), "publish tranche index file hash changed")
        index = queue.load_json(args.tranche_index, "publish tranche index")
        raw_index = receipt.get("outputs", {}).get("publish_tranche_index")
        require(isinstance(raw_index, Mapping)
                and raw_index.get("sha256") == index_identity["sha256"]
                and raw_index.get("bytes") == index_identity["bytes"],
                "published tranche index differs from canonical raw index")
        _validate_terminal_preparation_job(
            receipt,
            receipt_descriptor=raw_receipt_descriptor,
            index_descriptor=raw_index,
            study_commit=args.study_commit,
        )
        wave = build_tranche_wave(
            study_commit=args.study_commit,
            preparation_job_receipt=raw_receipt_descriptor,
            tranche_index_descriptor=raw_index,
            tranche_index=index,
        )
        queue._write_descriptor_output(args.output, wave)
        return 0
    result = run_formal_job(args) if args.command == "formal-full" else run_tranche_job(args)
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
