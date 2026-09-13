#!/usr/bin/env python3
"""Queue entrypoint for one exact, six-request DreamZero D1 qualification.

The outer process intentionally runs with the worker image's system Python.  It
checks the immutable queue/source boundary, snapshots and validates the fixed
observation, and proves that exactly two idle B200s are visible through both
``nvidia-smi`` and the pinned DreamZero Python.  Only then does it launch the
official-conditional, two-rank instrumented server and the six-request probe.

Full inputs, model outputs, server logs, and failure tracebacks remain below
``job_dir/raw`` on the PVC.  ``job_dir/publish`` contains one bounded receipt
with hashes and counts.  Every server process group is terminated and reaped in
a ``finally`` block; unexpected orchestration failures are technical-invalid,
not model evidence.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence
import uuid


NAMESPACE = "wmf_ablation_001_20260912"
RECEIPT_SCHEMA = "wmf-d1-qualification-job-receipt-v1"
FAILURE_SCHEMA = "wmf-d1-qualification-job-failure-v1"
PROCESS_SCHEMA = "wmf-d1-server-process-v1"
EXIT_SCHEMA = "wmf-d1-server-exit-v1"
FIXTURE_SCHEMA = "wmf-d1-fixed-observation-validation-v1"
TOPOLOGY_SCHEMA = "wmf-d1-two-b200-topology-v1"
CAPTURE_SCHEMA = "wmf-forecast-layout-fixed-observation-capture-v1"
SETTLED_RESET_SCHEMA = "wmf-forecast-layout-settled-reset-receipt-v1"

D1_PYTHON = Path("/data/users/ali/vla_wam/envs/dreamzero-ab790c1-py311/bin/python")
D1_SOURCE = Path("/data/users/ali/vla_wam/external/DreamZero-v3e004-clean-ab790c1")
D1_CHECKPOINT = Path("/data/users/ali/vla_wam/checkpoints/DreamZero-DROID-96ad344")
D1_TOKENIZER = Path("/data/users/ali/vla_wam/checkpoints/umt5-xxl-tokenizer-66cb9e7")
D1_CUDA_HOME = Path("/data/users/ali/vla_wam/envs/cuda-12.8-toolkit")
DEFAULT_PORT = 18021
EXPECTED_SOURCE_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
EXPECTED_SOURCE_TREE = "6b7ba27f1af81e963a6507f1204c05c65a94098c"
EXPECTED_CHECKPOINT_REVISION = "96ad344138c66e82536422432ad742f015784942"
EXPECTED_TOKENIZER_REVISION = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
EXPECTED_GPU_NAME = "NVIDIA B200"
EXPECTED_GPU_COUNT = 2
EXPECTED_REQUEST_COUNT = 6
EXPECTED_ACTION_SHAPE = [24, 8]
MAX_FIXTURE_BYTES = 64 * 1024 * 1024
MAX_PUBLISH_BYTES = 512 * 1024
MAX_CAPTURED_DIAGNOSTIC_BYTES = 1024 * 1024

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")

RAW_ARRAY_KEYS = (
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/cartesian_position",
    "observation/gripper_position",
)
EXPECTED_ARRAYS = {
    "observation/exterior_image_0_left": ([180, 320, 3], "uint8"),
    "observation/exterior_image_1_left": ([180, 320, 3], "uint8"),
    "observation/wrist_image_left": ([180, 320, 3], "uint8"),
    "observation/joint_position": ([7], "float64"),
    "observation/cartesian_position": ([6], "float64"),
    "observation/gripper_position": ([1], "float64"),
}


class D1JobError(RuntimeError):
    """Fail-closed error with a bounded publish-safe reason code."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        if SAFE_ID_RE.fullmatch(reason) is None:
            reason = "internal_contract_error"
        super().__init__(reason if detail is None else f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class RuntimePaths:
    python: Path = D1_PYTHON
    source: Path = D1_SOURCE
    checkpoint: Path = D1_CHECKPOINT
    tokenizer: Path = D1_TOKENIZER


@dataclass(frozen=True)
class QueueContext:
    source_root: Path
    job_dir: Path
    source_commit: str
    job_id: str


_ACTIVE_SERVER: subprocess.Popen[bytes] | None = None
_INTERRUPTED_SIGNAL: int | None = None
_SIGNAL_SERVER_REAP: dict[str, Any] | None = None
_SIGNAL_HANDLER_ACTIVE = False


def require(condition: bool, reason: str, detail: str | None = None) -> None:
    if not condition:
        raise D1JobError(reason, detail)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D1JobError("noncanonical_json_value") from error


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> bytes:
    path = Path(path)
    require(not path.is_symlink(), "json_target_is_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.parent.is_symlink(), "json_parent_is_symlink")
    payload = canonical_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".d1-job-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise D1JobError("immutable_receipt_exists") from error
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path, reason: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D1JobError(reason) from error


def _run_git(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *argv],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise D1JobError("git_verification_unavailable") from error
    require(result.returncode == 0, "git_verification_failed")
    return result


def validate_queue_context(source_root: Path, job_dir: Path) -> QueueContext:
    try:
        source = Path(source_root).resolve(strict=True)
        job = Path(job_dir).resolve(strict=True)
    except OSError as error:
        raise D1JobError("queue_path_missing") from error
    require(source.is_dir() and job.is_dir(), "queue_path_not_directory")
    state = job.parent.parent
    require(job.parent == state / "jobs", "job_dir_outside_queue_state")
    require(source.parent == state / "sources", "source_root_outside_queue_state")
    require(SAFE_ID_RE.fullmatch(job.name) is not None, "job_id_unsafe")
    descriptor = load_json(job / "descriptor.json", "job_descriptor_unreadable")
    require(isinstance(descriptor, dict), "job_descriptor_invalid")
    require(descriptor.get("schema_version") == "wmf-cluster-job-v1", "job_descriptor_schema_mismatch")
    require(descriptor.get("namespace") == NAMESPACE, "job_descriptor_namespace_mismatch")
    require(descriptor.get("job_id") == job.name, "job_descriptor_id_mismatch")
    source_commit = descriptor.get("source_commit")
    require(isinstance(source_commit, str) and COMMIT_RE.fullmatch(source_commit) is not None, "job_source_commit_invalid")
    require(source.name == source_commit, "source_root_commit_path_mismatch")
    require(_run_git(source, "rev-parse", "HEAD").stdout.strip() == source_commit, "study_commit_mismatch")
    require(not _run_git(source, "status", "--porcelain=v1", "--untracked-files=all").stdout, "study_checkout_dirty")
    return QueueContext(source, job, source_commit, job.name)


def validate_runtime_paths(paths: RuntimePaths, *, enforce_pinned_locations: bool = True) -> None:
    # Keep the lexical venv Python path for execution.  Following its symlink
    # selects the base interpreter and silently drops the pinned environment.
    resolved = RuntimePaths(
        python=Path(os.path.abspath(paths.python)),
        source=paths.source.resolve(),
        checkpoint=paths.checkpoint.resolve(),
        tokenizer=paths.tokenizer.resolve(),
    )
    if enforce_pinned_locations:
        pinned = RuntimePaths(
            python=Path(os.path.abspath(D1_PYTHON)),
            source=D1_SOURCE.resolve(),
            checkpoint=D1_CHECKPOINT.resolve(),
            tokenizer=D1_TOKENIZER.resolve(),
        )
        require(resolved == pinned, "d1_runtime_location_changed")
    require(resolved.python.is_file(), "d1_python_missing")
    require(os.access(resolved.python, os.X_OK), "d1_python_not_executable")
    require(resolved.source.is_dir(), "d1_source_missing")
    require(resolved.checkpoint.is_dir(), "d1_checkpoint_missing")
    require(resolved.tokenizer.is_dir(), "d1_tokenizer_missing")


def snapshot_regular_file(source: Path, target: Path, expected_sha256: str) -> dict[str, Any]:
    require(SHA256_RE.fullmatch(expected_sha256) is not None, "fixture_sha256_invalid")
    source = Path(source)
    target = Path(target)
    try:
        source_info = source.lstat()
    except OSError as error:
        raise D1JobError("fixture_missing") from error
    require(not source.is_symlink() and source.is_file(), "fixture_not_regular_file")
    require(0 < source_info.st_size <= MAX_FIXTURE_BYTES, "fixture_size_invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.parent.is_symlink(), "fixture_snapshot_parent_is_symlink")
    require(not target.exists() and not target.is_symlink(), "fixture_snapshot_exists")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    count = 0
    try:
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as input_stream, target.open("xb") as output_stream:
            before = os.fstat(input_stream.fileno())
            for block in iter(lambda: input_stream.read(4 * 1024 * 1024), b""):
                count += len(block)
                require(count <= MAX_FIXTURE_BYTES, "fixture_size_invalid")
                digest.update(block)
                output_stream.write(block)
            after = os.fstat(input_stream.fileno())
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except D1JobError:
        raise
    except OSError as error:
        raise D1JobError("fixture_snapshot_failed") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "fixture_changed_during_snapshot",
    )
    observed = digest.hexdigest()
    require(observed == expected_sha256, "fixture_sha256_mismatch")
    require(sha256_file(target) == expected_sha256, "fixture_snapshot_sha256_mismatch")
    fsync_directory(target.parent)
    return {
        "source": {"path": str(source.resolve()), "bytes": count, "sha256": observed},
        "snapshot": file_identity(target),
    }


def _verified_capture_artifact(
    descriptor: Any,
    *,
    artifact_root: Path,
    label: str,
) -> dict[str, Any]:
    require(isinstance(descriptor, Mapping), "capture_artifact_descriptor_invalid", label)
    raw_path = descriptor.get("path")
    require(isinstance(raw_path, str) and raw_path, "capture_artifact_path_missing", label)
    path = Path(raw_path).resolve()
    root = Path(artifact_root).resolve()
    require(path.is_relative_to(root), "capture_artifact_path_escape", label)
    require(path.parent == root, "capture_artifact_not_direct_child", label)
    require(not path.is_symlink() and path.is_file(), "capture_artifact_not_regular", label)
    identity = file_identity(path)
    require(identity["sha256"] == descriptor.get("sha256"), "capture_artifact_sha256_mismatch", label)
    require(identity["bytes"] == descriptor.get("bytes"), "capture_artifact_size_mismatch", label)
    return identity


def validate_capture_manifest(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    artifact_root: Path,
    fixture_path: Path,
    fixture_sha256: str,
) -> dict[str, Any]:
    """Validate the accepted-fixture -> settled capture -> D1 preprocessing chain."""

    require(SHA256_RE.fullmatch(manifest_sha256) is not None, "capture_manifest_sha256_invalid")
    manifest_path = Path(manifest_path)
    require(not manifest_path.is_symlink() and manifest_path.is_file(), "capture_manifest_not_regular")
    require(sha256_file(manifest_path) == manifest_sha256, "capture_manifest_sha256_mismatch")
    capture = load_json(manifest_path, "capture_manifest_unreadable")
    require(isinstance(capture, Mapping), "capture_manifest_invalid")
    checks = {
        "schema": capture.get("schema_version") == CAPTURE_SCHEMA,
        "namespace": capture.get("study_namespace") == NAMESPACE,
        "status": capture.get("status") == "passed",
        "layout": capture.get("layout_pair_id") == "P00",
        "arm": capture.get("layout_arm") == "original",
        "reset_command": capture.get("command_task_used_for_reset") == "left",
        "no_model_requests": capture.get("model_request_count") == 0,
        "no_behavioral_actions": capture.get("behavioral_action_count") == 0,
    }
    require(all(checks.values()), "capture_manifest_gate_failed", str(sorted(key for key, value in checks.items() if not value)))
    capture_id = capture.get("capture_id")
    candidate_id = capture.get("candidate_id")
    require(isinstance(capture_id, str) and capture_id, "capture_id_missing")
    require(isinstance(candidate_id, str) and candidate_id.startswith("P00__candidate_"), "capture_candidate_invalid")
    for field in (
        "candidate_payload_sha256",
        "accepted_gate_record_sha256",
    ):
        require(SHA256_RE.fullmatch(str(capture.get(field, ""))) is not None, "capture_gate_binding_invalid", field)
    for field in ("gate_receipt", "pose_manifest", "gate_ledger", "gate_attempt_receipt"):
        descriptor = capture.get(field)
        require(isinstance(descriptor, Mapping), "capture_gate_descriptor_missing", field)
        require(SHA256_RE.fullmatch(str(descriptor.get("sha256", ""))) is not None, "capture_gate_descriptor_invalid", field)
        require(type(descriptor.get("bytes")) is int and descriptor["bytes"] > 0, "capture_gate_descriptor_invalid", field)

    reset = capture.get("settled_reset_receipt")
    require(isinstance(reset, Mapping), "settled_reset_receipt_missing")
    reset_checks = {
        "schema": reset.get("schema_version") == SETTLED_RESET_SCHEMA,
        "passed": reset.get("passed") is True,
        "settled": reset.get("settled") is True,
        "left_not_success": reset.get("left_success") is False,
        "right_not_success": reset.get("right_success") is False,
        "returned": reset.get("settled_observation_returned") is True,
        "no_model": reset.get("model_request_count_during_settle") == 0,
        "counter_zero": reset.get("episode_length_buf_reset_to_zero") is True,
        "collision": isinstance(reset.get("collision_evidence"), Mapping)
        and reset["collision_evidence"].get("passed") is True,
        "visibility": isinstance(reset.get("visibility_evidence"), Mapping)
        and reset["visibility_evidence"].get("passed") is True,
    }
    require(all(reset_checks.values()), "settled_reset_gate_failed", str(sorted(key for key, value in reset_checks.items() if not value)))
    require(reset.get("reset_identity") == capture.get("settled_reset_identity"), "settled_reset_identity_mismatch")
    require(reset.get("pose_manifest_sha256") == capture["pose_manifest"].get("sha256"), "settled_reset_pose_manifest_mismatch")
    settle = reset.get("settle_evidence")
    require(isinstance(settle, Mapping), "settled_reset_evidence_missing")
    require(type(settle.get("settle_steps")) is int and settle["settle_steps"] > 0, "settle_steps_invalid")
    require(type(settle.get("stability_window_steps")) is int and settle["stability_window_steps"] > 0, "stability_window_invalid")

    fresh = capture.get("fresh_physical_checks")
    require(isinstance(fresh, Mapping), "fresh_physical_checks_missing")
    require(isinstance(fresh.get("collision"), Mapping) and fresh["collision"].get("passed") is True, "fresh_collision_check_failed")
    require(isinstance(fresh.get("visibility"), Mapping) and fresh["visibility"].get("passed") is True, "fresh_visibility_check_failed")

    native = capture.get("native_clock")
    source_capture = capture.get("source_capture")
    require(isinstance(native, Mapping) and isinstance(source_capture, Mapping), "capture_timing_missing")
    require(type(native.get("physics_step")) is int and native["physics_step"] >= 0, "capture_physics_step_invalid")
    require(type(native.get("control_step_since_physical_reset")) is int and native["control_step_since_physical_reset"] > 0, "capture_control_step_invalid")
    require(native.get("behavioral_episode_step") == 0, "capture_behavioral_counter_not_zero")
    cameras = native.get("camera_counters")
    frame_ids = source_capture.get("camera_frame_ids")
    timestamps = source_capture.get("camera_capture_time_ns")
    timestamp_sources = source_capture.get("camera_timestamp_source")
    expected_cameras = {"head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"}
    require(isinstance(cameras, Mapping) and set(cameras) == expected_cameras, "capture_camera_inventory_mismatch")
    require(isinstance(frame_ids, Mapping) and set(frame_ids) == expected_cameras, "capture_frame_id_inventory_mismatch")
    require(isinstance(timestamps, Mapping) and set(timestamps) == expected_cameras, "capture_timestamp_inventory_mismatch")
    require(isinstance(timestamp_sources, Mapping) and set(timestamp_sources) == expected_cameras, "capture_timestamp_source_inventory_mismatch")
    for name in sorted(expected_cameras):
        row = cameras[name]
        require(isinstance(row, Mapping), "capture_camera_counter_invalid", name)
        frame_id = row.get("frame_id")
        native_counter = type(frame_id) is int and frame_id >= 0
        rgb_identity = isinstance(frame_id, str) and re.fullmatch(r"rgb-sha256:[0-9a-f]{64}", frame_id) is not None
        require(native_counter or rgb_identity, "capture_frame_id_invalid", name)
        if rgb_identity:
            # Pinned Isaac SensorBase exposes native sensor time but no frame
            # counter.  The producer therefore binds the returned RGB bytes as
            # the frame identity rather than inventing an ordinal.
            array_identity = row.get("rgb_array_identity")
            require(
                row.get("frame_identity_source") == "exact returned RGB array value identity",
                "capture_frame_identity_source_invalid",
                name,
            )
            require(row.get("native_frame_counter") is None, "capture_native_frame_counter_conflict", name)
            require(
                row.get("native_frame_counter_status") == "unavailable_in_pinned_isaaclab_sensorbase",
                "capture_native_frame_counter_status_invalid",
                name,
            )
            require(isinstance(array_identity, Mapping), "capture_rgb_identity_missing", name)
            require(
                array_identity.get("value_sha256") == frame_id.removeprefix("rgb-sha256:"),
                "capture_rgb_identity_binding_mismatch",
                name,
            )
        require(type(row.get("capture_time_ns")) is int and row["capture_time_ns"] >= 0, "capture_timestamp_invalid", name)
        require(isinstance(row.get("native_capture_time_source"), str) and row["native_capture_time_source"], "capture_timestamp_source_invalid", name)
        require(frame_ids[name] == row["frame_id"], "capture_frame_id_binding_mismatch", name)
        require(timestamps[name] == row["capture_time_ns"], "capture_timestamp_binding_mismatch", name)
        require(timestamp_sources[name] == row["native_capture_time_source"], "capture_timestamp_source_binding_mismatch", name)

    preprocessing = capture.get("preprocessing")
    require(isinstance(preprocessing, Mapping), "capture_preprocessing_missing")
    require(preprocessing.get("model_request_count") == 0, "capture_preprocessing_contacted_model")
    d1_preprocessing = preprocessing.get("D1")
    require(isinstance(d1_preprocessing, Mapping), "capture_d1_preprocessing_missing")
    require(d1_preprocessing.get("extraction_method") == "_extract_observation", "capture_d1_extraction_changed")
    require(d1_preprocessing.get("packing_method") == "_pack_request", "capture_d1_packing_changed")
    require(d1_preprocessing.get("wire_array_keys") == list(RAW_ARRAY_KEYS), "capture_d1_wire_keys_changed")
    require(
        d1_preprocessing.get("configuration")
        == {"cam2_source": "right", "resize": "pad", "image_height": 180, "image_width": 320},
        "capture_d1_preprocessing_config_changed",
    )
    for field in ("overlay_source", "official_client_source"):
        source_descriptor = d1_preprocessing.get(field)
        require(isinstance(source_descriptor, Mapping), "capture_d1_preprocessing_source_missing", field)
        require(SHA256_RE.fullmatch(str(source_descriptor.get("sha256", ""))) is not None, "capture_d1_preprocessing_source_invalid", field)

    artifacts = capture.get("artifacts")
    require(isinstance(artifacts, Mapping), "capture_artifacts_missing")
    required_artifacts = {"raw_settled_observation", "simulator_state", "preprocessing_intermediates", "N3", "D1"}
    require(set(artifacts) == required_artifacts, "capture_artifact_inventory_mismatch")
    artifact_identities = {
        name: _verified_capture_artifact(artifacts[name], artifact_root=artifact_root, label=name)
        for name in sorted(required_artifacts)
    }
    d1_descriptor = artifacts["D1"]
    require(isinstance(d1_descriptor.get("arrays"), Mapping), "capture_d1_array_descriptors_missing")
    require(set(d1_descriptor["arrays"]) == set(RAW_ARRAY_KEYS), "capture_d1_array_descriptor_keys_changed")
    model_fixtures = capture.get("model_fixtures")
    require(isinstance(model_fixtures, Mapping) and isinstance(model_fixtures.get("D1"), Mapping), "capture_d1_fixture_missing")
    model_d1 = model_fixtures["D1"]
    require(model_d1.get("array_keys") == list(RAW_ARRAY_KEYS), "capture_d1_fixture_keys_changed")
    model_descriptor = model_d1.get("fixture")
    require(isinstance(model_descriptor, Mapping), "capture_d1_fixture_descriptor_missing")
    require(model_descriptor.get("path") == d1_descriptor.get("path"), "capture_d1_fixture_path_mismatch")
    require(model_descriptor.get("sha256") == d1_descriptor.get("sha256"), "capture_d1_fixture_sha256_mismatch")
    require(model_descriptor.get("bytes") == d1_descriptor.get("bytes"), "capture_d1_fixture_size_mismatch")
    require(Path(str(model_descriptor.get("path"))).resolve() == Path(fixture_path).resolve(), "caller_fixture_not_capture_d1")
    require(model_descriptor.get("sha256") == fixture_sha256, "caller_fixture_sha256_not_capture_d1")
    return {
        "schema_version": "wmf-d1-capture-manifest-validation-v1",
        "status": "passed",
        "manifest": file_identity(manifest_path),
        "capture_id": capture_id,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": capture["candidate_payload_sha256"],
        "accepted_gate_record_sha256": capture["accepted_gate_record_sha256"],
        "settled_reset_identity": capture["settled_reset_identity"],
        "pose_manifest_sha256": capture["pose_manifest"]["sha256"],
        "camera_frame_ids": dict(frame_ids),
        "camera_capture_time_ns": dict(timestamps),
        "physics_step": native["physics_step"],
        "control_step_since_physical_reset": native["control_step_since_physical_reset"],
        "artifact_identities": artifact_identities,
        "d1_preprocessing": dict(d1_preprocessing),
    }


def validate_fixture_arrays(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Heavy fixture validation, called under the pinned DreamZero Python."""

    import numpy as np

    require(sha256_file(path) == expected_sha256, "fixture_sha256_mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(RAW_ARRAY_KEYS), "fixture_array_keys_mismatch")
            records = []
            for key in RAW_ARRAY_KEYS:
                value = np.ascontiguousarray(archive[key])
                shape, dtype = EXPECTED_ARRAYS[key]
                require(list(value.shape) == shape, "fixture_array_shape_mismatch", key)
                require(str(value.dtype) == dtype, "fixture_array_dtype_mismatch", key)
                require(bool(np.isfinite(value).all()), "fixture_array_nonfinite", key)
                records.append(
                    {
                        "key": key,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "data_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                    }
                )
    except D1JobError:
        raise
    except BaseException as error:
        raise D1JobError("fixture_npz_invalid") from error
    return {
        "schema_version": FIXTURE_SCHEMA,
        "status": "passed",
        "validated_at_utc": utc_now(),
        "fixture": file_identity(path),
        "array_count": len(records),
        "arrays": records,
    }


def torch_topology() -> dict[str, Any]:
    """Heavy CUDA validation, called under the pinned DreamZero Python."""

    import torch

    require(torch.cuda.is_available(), "torch_cuda_unavailable")
    require(torch.cuda.device_count() == EXPECTED_GPU_COUNT, "torch_visible_gpu_count_not_two")
    devices = []
    for index in range(EXPECTED_GPU_COUNT):
        name = torch.cuda.get_device_name(index)
        require(name == EXPECTED_GPU_NAME, "torch_visible_gpu_not_b200")
        properties = torch.cuda.get_device_properties(index)
        device_uuid = getattr(properties, "uuid", None)
        if isinstance(device_uuid, bytes):
            device_uuid = device_uuid.decode("ascii", "strict")
        elif device_uuid is not None:
            device_uuid = str(device_uuid)
        devices.append(
            {
                "logical_index": index,
                "name": name,
                "uuid": device_uuid,
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    if all(row["uuid"] for row in devices):
        require(len({row["uuid"] for row in devices}) == 2, "torch_gpu_uuid_not_unique")
    return {
        "schema_version": "wmf-d1-torch-topology-v1",
        "status": "passed",
        "validated_at_utc": utc_now(),
        "cuda_available": True,
        "device_count": len(devices),
        "devices": devices,
    }


def _query_nvidia(nvidia_smi: Path | str, columns: Sequence[str], *, compute: bool) -> list[dict[str, str]]:
    prefix = "--query-compute-apps=" if compute else "--query-gpu="
    try:
        result = subprocess.run(
            [str(nvidia_smi), prefix + ",".join(columns), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise D1JobError("nvidia_query_unavailable") from error
    require(result.returncode == 0, "nvidia_query_failed")
    require(len(result.stdout.encode("utf-8")) <= 65536, "nvidia_query_oversize")
    rows = []
    for values in csv.reader(io.StringIO(result.stdout)):
        if not values or not any(value.strip() for value in values):
            continue
        require(len(values) == len(columns), "nvidia_query_shape_mismatch")
        rows.append({key: value.strip() for key, value in zip(columns, values)})
    return rows


def reconcile_topology(
    *,
    nvidia_devices: Sequence[Mapping[str, str]],
    compute_processes: Sequence[Mapping[str, str]],
    torch_receipt: Mapping[str, Any],
    visible_devices: str | None,
) -> dict[str, Any]:
    require(len(nvidia_devices) == EXPECTED_GPU_COUNT, "nvidia_visible_gpu_count_not_two")
    require(not compute_processes, "preexisting_compute_processes")
    uuids = [row.get("uuid") for row in nvidia_devices]
    indices = [row.get("index") for row in nvidia_devices]
    require(all(row.get("name") == EXPECTED_GPU_NAME for row in nvidia_devices), "nvidia_visible_gpu_not_b200")
    require(all(isinstance(value, str) and value.startswith("GPU-") for value in uuids), "nvidia_gpu_uuid_invalid")
    require(len(set(uuids)) == 2 and len(set(indices)) == 2, "nvidia_gpu_identity_not_unique")
    require(isinstance(visible_devices, str), "nvidia_visible_devices_missing")
    tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    require(len(tokens) == 2 and len(set(tokens)) == 2, "nvidia_visible_devices_not_exact_two")
    require(set(tokens) == set(uuids) or set(tokens) == set(indices), "nvidia_visible_devices_identity_mismatch")
    require(torch_receipt.get("schema_version") == "wmf-d1-torch-topology-v1", "torch_topology_schema_mismatch")
    require(torch_receipt.get("status") == "passed", "torch_topology_failed")
    require(torch_receipt.get("device_count") == 2, "torch_visible_gpu_count_not_two")
    torch_devices = torch_receipt.get("devices")
    require(isinstance(torch_devices, list) and len(torch_devices) == 2, "torch_topology_shape_mismatch")
    require([row.get("logical_index") for row in torch_devices] == [0, 1], "torch_logical_gpu_indices_changed")
    require(all(row.get("name") == EXPECTED_GPU_NAME for row in torch_devices), "torch_visible_gpu_not_b200")
    torch_uuids = [row.get("uuid") for row in torch_devices]
    if all(torch_uuids):
        require(set(torch_uuids) == set(uuids), "torch_nvidia_gpu_uuid_mismatch")
    return {
        "schema_version": TOPOLOGY_SCHEMA,
        "status": "passed",
        "validated_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "pod_uid": os.environ.get("POD_UID"),
        "nvidia_visible_devices": tokens,
        "nvidia_smi_devices": list(nvidia_devices),
        "torch_devices": list(torch_devices),
        "preexisting_compute_process_count": 0,
    }


def run_logged(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[Any]:
    require(timeout > 0, "child_timeout_invalid")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            result = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise D1JobError("child_timed_out") from error
        except OSError as error:
            raise D1JobError("child_launch_failed") from error
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    return result


def run_internal_helper(
    *,
    python: Path,
    module_path: Path,
    mode: str,
    output_path: Path,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    extra: Sequence[str] = (),
) -> dict[str, Any]:
    stdout_path = output_path.with_suffix(".stdout.log")
    stderr_path = output_path.with_suffix(".stderr.log")
    command = [str(python), str(module_path), mode, "--output", str(output_path), *extra]
    result = run_logged(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cwd=cwd,
        env=env,
        timeout=timeout,
    )
    require(result.returncode == 0, "internal_validation_failed")
    value = load_json(output_path, "internal_validation_receipt_unreadable")
    require(isinstance(value, dict) and value.get("status") == "passed", "internal_validation_failed")
    return value


def verify_gpu_topology(
    *,
    runtime: RuntimePaths,
    module_path: Path,
    raw_dir: Path,
    cwd: Path,
    env: Mapping[str, str],
    nvidia_smi: Path | str,
) -> dict[str, Any]:
    devices = _query_nvidia(
        nvidia_smi,
        ("index", "uuid", "name", "driver_version", "memory.total"),
        compute=False,
    )
    processes = _query_nvidia(
        nvidia_smi,
        ("gpu_uuid", "pid", "process_name", "used_memory"),
        compute=True,
    )
    torch_receipt = run_internal_helper(
        python=runtime.python,
        module_path=module_path,
        mode="internal-torch-topology",
        output_path=raw_dir / "torch_topology.json",
        cwd=cwd,
        env=env,
        timeout=300,
    )
    # The helper's CUDA context must have disappeared before model launch.
    residual = _query_nvidia(
        nvidia_smi,
        ("gpu_uuid", "pid", "process_name", "used_memory"),
        compute=True,
    )
    require(not residual, "topology_probe_left_compute_process")
    receipt = reconcile_topology(
        nvidia_devices=devices,
        compute_processes=processes,
        torch_receipt=torch_receipt,
        visible_devices=env.get("NVIDIA_VISIBLE_DEVICES"),
    )
    atomic_json(raw_dir / "topology_receipt.json", receipt, exclusive=True)
    return receipt


def assert_port_available(port: int) -> None:
    require(0 < port < 65536 and port != 5000, "d1_port_invalid")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise D1JobError("d1_port_unavailable") from error
    finally:
        probe.close()


def build_server_command(
    *,
    runtime: RuntimePaths,
    source_root: Path,
    future_root: Path,
    port: int,
    timeout_seconds: int,
) -> list[str]:
    forecast = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
    return [
        str(runtime.python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        str(forecast / "d1_instrumented_server.py"),
        "--port",
        str(port),
        "--future-root",
        str(future_root),
        "--source-root",
        str(runtime.source),
        "--checkpoint-root",
        str(runtime.checkpoint),
        "--tokenizer-root",
        str(runtime.tokenizer),
        "--identity-contract",
        str(forecast / "d1_identity_contract.json"),
        "--study-root",
        str(source_root),
        "--timeout-seconds",
        str(timeout_seconds),
        "--enable-dit-cache",
    ]


def build_probe_command(
    *,
    runtime: RuntimePaths,
    source_root: Path,
    fixture: Path,
    fixture_sha256: str,
    future_root: Path,
    output_dir: Path,
    port: int,
) -> list[str]:
    forecast = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
    return [
        str(runtime.python),
        str(forecast / "d1_probe.py"),
        "--fixture",
        str(fixture),
        "--fixture-sha256",
        fixture_sha256,
        "--future-root",
        str(future_root),
        "--output-dir",
        str(output_dir),
        "--remote-host",
        "127.0.0.1",
        "--remote-port",
        str(port),
        "--plan",
        str(forecast / "d1_probe_plan.json"),
    ]


def _proc_start_identity(pid: int) -> dict[str, Any]:
    identity: dict[str, Any] = {"pid": pid, "pgid": os.getpgid(pid), "hostname": socket.gethostname()}
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text().split()
        identity["linux_proc_start_ticks"] = int(stat_fields[21])
        identity["linux_boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        identity["linux_proc_start_ticks"] = None
        identity["linux_boot_id"] = None
    return identity


def launch_server(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
    env: Mapping[str, str],
) -> tuple[subprocess.Popen[bytes], Any, Any]:
    stdout = stdout_path.open("xb")
    stderr = stderr_path.open("xb")
    try:
        parent_pid = os.getpid()

        def set_parent_death_signal() -> None:
            # Linux resets PR_SET_PDEATHSIG across fork.  Set it in the future
            # torchrun process immediately before exec, and close the small race
            # in which the wrapper could die between fork and prctl.
            libc = ctypes.CDLL(None, use_errno=True)
            result = libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
            if result != 0:
                os._exit(126)
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGTERM)

        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=set_parent_death_signal,
        )
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    return process, stdout, stderr


def validate_server_contract(
    path: Path,
    *,
    runtime: RuntimePaths,
    source_root: Path,
    topology: Mapping[str, Any],
    port: int,
) -> dict[str, Any]:
    contract = load_json(path, "server_contract_unreadable")
    require(isinstance(contract, dict), "server_contract_invalid")
    checks = {
        "schema": contract.get("schema_version") == "wmf-d1-instrumented-server-v1",
        "status": contract.get("status") == "passed",
        "configuration": contract.get("configuration_id") == "D1",
        "official_path": contract.get("official_action_path") == "GrootSimPolicy.lazy_joint_forward_causal",
        "no_s2": contract.get("custom_s2_used") is False,
        "no_patched_s1": contract.get("patched_s1_used") is False,
        "source_commit": contract.get("official_repository_commit") == EXPECTED_SOURCE_COMMIT,
        "source_tree": contract.get("official_repository_tree") == EXPECTED_SOURCE_TREE,
        "source_path": Path(str(contract.get("source_root", ""))).resolve() == runtime.source.resolve(),
        "checkpoint_path": Path(str(contract.get("checkpoint_root", ""))).resolve() == runtime.checkpoint.resolve(),
        "tokenizer_path": Path(str(contract.get("tokenizer_root", ""))).resolve() == runtime.tokenizer.resolve(),
        "world_size": contract.get("world_size") == 2,
        "port": contract.get("port") == port,
        "action_shape": contract.get("returned_action_shape") == EXPECTED_ACTION_SHAPE,
        "noise_seed": contract.get("effective_official_model_noise_seed") == 1140,
        "cache": contract.get("enable_dit_cache") is True,
        "no_dynamic_cache": contract.get("dynamic_cache_schedule") is False,
        "no_trt": contract.get("tensorrt_engine_active") is False,
    }
    require(all(checks.values()), "server_contract_gate_failed", str(sorted(key for key, value in checks.items() if not value)))
    server_topology = contract.get("topology")
    require(isinstance(server_topology, list) and len(server_topology) == 2, "server_topology_invalid")
    require(sorted(row.get("rank") for row in server_topology) == [0, 1], "server_topology_rank_mismatch")
    require(len({row.get("hostname") for row in server_topology}) == 1, "server_topology_multinode")
    require(len({row.get("cuda_device_index") for row in server_topology}) == 2, "server_topology_device_alias")
    require(all(row.get("cuda_device_name") == EXPECTED_GPU_NAME for row in server_topology), "server_topology_not_b200")
    server_uuids = {row.get("cuda_device_uuid") for row in server_topology}
    expected_uuids = {row.get("uuid") for row in topology["nvidia_smi_devices"]}
    require(server_uuids == expected_uuids, "server_topology_uuid_mismatch")

    identity_path = Path(str(contract.get("identity_receipt", ""))).resolve()
    require(identity_path == (path.parent / "identity_receipt.json").resolve(), "identity_receipt_path_mismatch")
    require(identity_path.is_file(), "identity_receipt_missing")
    require(sha256_file(identity_path) == contract.get("identity_receipt_sha256"), "identity_receipt_sha256_mismatch")
    identity = load_json(identity_path, "identity_receipt_unreadable")
    require(identity.get("schema_version") == "wmf-d1-runtime-identity-receipt-v1", "identity_receipt_schema_mismatch")
    require(identity.get("status") == "passed", "identity_receipt_failed")
    require(identity.get("source", {}).get("commit") == EXPECTED_SOURCE_COMMIT, "identity_source_commit_mismatch")
    require(identity.get("source", {}).get("git_tree") == EXPECTED_SOURCE_TREE, "identity_source_tree_mismatch")
    require(identity.get("checkpoint", {}).get("revision") == EXPECTED_CHECKPOINT_REVISION, "identity_checkpoint_mismatch")
    require(identity.get("tokenizer", {}).get("revision") == EXPECTED_TOKENIZER_REVISION, "identity_tokenizer_mismatch")
    loaders = contract.get("bounded_loader_receipts")
    require(isinstance(loaders, list) and sorted(row.get("rank") for row in loaders) == [0, 1], "bounded_loader_receipts_invalid")
    require(all(row.get("receipt", {}).get("passed") is True for row in loaders), "bounded_loader_failed")
    require(all(row.get("receipt", {}).get("forward_path_modified") is False for row in loaders), "bounded_loader_modified_forward")
    overlay = contract.get("instrumentation_overlay", {})
    require(overlay.get("returned_action_modified") is False, "instrumentation_modified_action")
    require(Path(str(overlay.get("path", ""))).resolve() == (source_root / "workshops/corl2026_world_models/experiments/forecast_layout/d1_instrumented_server.py").resolve(), "instrumentation_path_mismatch")
    return contract


def wait_for_server_contract(
    process: subprocess.Popen[bytes],
    contract_path: Path,
    *,
    runtime: RuntimePaths,
    source_root: Path,
    topology: Mapping[str, Any],
    port: int,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    require(timeout_seconds > 0 and poll_seconds > 0, "readiness_timeout_invalid")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        code = process.poll()
        require(code is None, "server_exited_before_ready", str(code))
        if contract_path.is_file():
            contract = validate_server_contract(
                contract_path,
                runtime=runtime,
                source_root=source_root,
                topology=topology,
                port=port,
            )
            require(process.poll() is None, "server_exited_after_contract")
            return contract
        time.sleep(min(poll_seconds, max(0.001, deadline - time.monotonic())))
    raise D1JobError("server_readiness_timed_out")


def terminate_server(
    process: subprocess.Popen[bytes],
    *,
    stdout_handle: Any,
    stderr_handle: Any,
    stdout_path: Path,
    stderr_path: Path,
    grace_seconds: float,
    prior_signal_reap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = utc_now()
    initial_code = process.poll()
    signal_sent: str | None = None
    killed = False
    if initial_code is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            signal_sent = "SIGTERM"
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                signal_sent = "SIGKILL"
                killed = True
            except ProcessLookupError:
                pass
            process.wait(timeout=min(2.0, max(1.0, grace_seconds)))
    else:
        process.wait()
    # Kill any surviving descendants in this exclusively owned server group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for handle in (stdout_handle, stderr_handle):
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
    return {
        "schema_version": EXIT_SCHEMA,
        "status": "reaped",
        "terminated_at_utc": utc_now(),
        "termination_started_at_utc": started,
        "initial_returncode": initial_code,
        "signal_sent": signal_sent,
        "sigkill_required": killed,
        "returncode": process.returncode,
        "reaped": process.returncode is not None,
        "parent_death_signal": "SIGTERM",
        "signal_handler_reap": dict(prior_signal_reap) if prior_signal_reap else None,
        "stdout": file_identity(stdout_path),
        "stderr": file_identity(stderr_path),
    }


def _fast_signal_reap(process: subprocess.Popen[bytes], *, signum: int) -> dict[str, Any]:
    """Bounded cleanup used inside the outer queue job's signal handler."""

    started = time.monotonic()
    term_sent = False
    kill_sent = False
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 1.5
        while process.poll() is None and time.monotonic() < deadline:
            try:
                process.wait(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                kill_sent = True
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
    return {
        "received_signal": signal.Signals(signum).name,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "returncode": process.poll(),
        "reaped": process.poll() is not None,
        "wall_seconds": time.monotonic() - started,
    }


def _signal_handler(signum: int, _frame: Any) -> None:
    global _INTERRUPTED_SIGNAL, _SIGNAL_SERVER_REAP, _SIGNAL_HANDLER_ACTIVE
    _INTERRUPTED_SIGNAL = signum
    if _SIGNAL_HANDLER_ACTIVE:
        return
    _SIGNAL_HANDLER_ACTIVE = True
    process = _ACTIVE_SERVER
    try:
        if process is not None:
            _SIGNAL_SERVER_REAP = _fast_signal_reap(process, signum=signum)
    finally:
        _SIGNAL_HANDLER_ACTIVE = False
    raise D1JobError("job_interrupted")


def _child_environment(job_dir: Path) -> dict[str, str]:
    # DeepSpeed discovers and validates its CUDA extension toolchain during
    # import.  GM's pinned D1 image does not expose a /usr/local/cuda symlink;
    # this task-owned toolkit is the verified native CUDA 12.x installation.
    cuda_home = D1_CUDA_HOME
    require((cuda_home / "bin" / "nvcc").is_file(), "d1_cuda_toolkit_missing")
    runtime_root = job_dir / "raw" / "runtime"
    directories = {
        "tmp": runtime_root / "tmp",
        "xdg": runtime_root / "xdg",
        "torch": runtime_root / "torch",
        "hf": runtime_root / "huggingface",
        "pycache": runtime_root / "pycache",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        CUDA_HOME=str(cuda_home),
        PATH=f"{cuda_home / 'bin'}:{env.get('PATH', '')}",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TMPDIR=str(directories["tmp"]),
        XDG_CACHE_HOME=str(directories["xdg"]),
        TORCH_HOME=str(directories["torch"]),
        HF_HOME=str(directories["hf"]),
        PYTHONPYCACHEPREFIX=str(directories["pycache"]),
    )
    return env


def _compact_contract(contract: Mapping[str, Any], path: Path) -> dict[str, Any]:
    identity_path = Path(str(contract["identity_receipt"]))
    identity = load_json(identity_path, "identity_receipt_unreadable")
    return {
        "artifact": file_identity(path),
        "official_repository_commit": contract["official_repository_commit"],
        "official_repository_tree": contract["official_repository_tree"],
        "official_action_path": contract["official_action_path"],
        "custom_s2_used": contract["custom_s2_used"],
        "patched_s1_used": contract["patched_s1_used"],
        "world_size": contract["world_size"],
        "source_aggregate_sha256": identity["source"]["aggregate_sha256"],
        "checkpoint_aggregate_sha256": identity["checkpoint"]["aggregate_sha256"],
        "tokenizer_aggregate_sha256": identity["tokenizer"]["aggregate_sha256"],
    }


def _publish_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    payload = canonical_bytes(receipt)
    require(len(payload) <= MAX_PUBLISH_BYTES, "publish_receipt_oversize")
    atomic_json(path, receipt, exclusive=True)


def execute_job(
    *,
    source_root: Path,
    job_dir: Path,
    fixture: Path,
    fixture_sha256: str,
    capture_manifest: Path,
    capture_manifest_sha256: str,
    port: int = DEFAULT_PORT,
    readiness_timeout_seconds: float = 7200,
    probe_timeout_seconds: float = 43200,
    server_group_timeout_seconds: int = 50000,
    terminate_grace_seconds: float = 5,
    runtime: RuntimePaths = RuntimePaths(),
    nvidia_smi: Path | str = "nvidia-smi",
    enforce_pinned_locations: bool = True,
) -> dict[str, Any]:
    global _ACTIVE_SERVER, _INTERRUPTED_SIGNAL, _SIGNAL_SERVER_REAP
    context = validate_queue_context(source_root, job_dir)
    raw_dir = context.job_dir / "raw"
    publish_dir = context.job_dir / "publish"
    require(not raw_dir.exists() and not raw_dir.is_symlink(), "raw_attempt_exists")
    require(not publish_dir.exists() and not publish_dir.is_symlink(), "publish_attempt_exists")
    raw_dir.mkdir()
    publish_dir.mkdir()
    _INTERRUPTED_SIGNAL = None
    _SIGNAL_SERVER_REAP = None
    env: dict[str, str] = {}
    module_path = Path(__file__).resolve()
    started = utc_now()
    process: subprocess.Popen[bytes] | None = None
    server_stdout = None
    server_stderr = None
    server_exit: dict[str, Any] | None = None
    topology: dict[str, Any] | None = None
    fixture_receipt: dict[str, Any] | None = None
    capture_receipt: dict[str, Any] | None = None
    server_contract: dict[str, Any] | None = None
    probe_report: dict[str, Any] | None = None
    probe_exit: int | None = None
    decision = "technical_invalid"
    reason: str | None = None
    exit_code = 3

    old_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in old_handlers:
        signal.signal(signum, _signal_handler)
    try:
        require(SHA256_RE.fullmatch(fixture_sha256) is not None, "fixture_sha256_invalid")
        require(SHA256_RE.fullmatch(capture_manifest_sha256) is not None, "capture_manifest_sha256_invalid")
        require(readiness_timeout_seconds > 0, "readiness_timeout_invalid")
        require(probe_timeout_seconds > 0, "probe_timeout_invalid")
        require(server_group_timeout_seconds > 0, "server_timeout_invalid")
        require(terminate_grace_seconds > 0, "termination_grace_invalid")
        assert_port_available(port)
        validate_runtime_paths(runtime, enforce_pinned_locations=enforce_pinned_locations)
        env = _child_environment(context.job_dir)
        capture_copy = raw_dir / "fixed_observation" / "capture_receipt.json"
        capture_snapshot = snapshot_regular_file(
            Path(capture_manifest), capture_copy, capture_manifest_sha256
        )
        capture_receipt = validate_capture_manifest(
            capture_copy,
            capture_manifest_sha256,
            artifact_root=Path(capture_manifest).resolve().parent,
            fixture_path=Path(fixture),
            fixture_sha256=fixture_sha256,
        )
        capture_receipt["source_manifest"] = capture_snapshot["source"]
        capture_receipt["snapshot_manifest"] = capture_snapshot["snapshot"]
        atomic_json(
            raw_dir / "capture_manifest_validation.json", capture_receipt, exclusive=True
        )
        fixture_copy = raw_dir / "fixed_observation" / "d1_fixed_observation.npz"
        fixture_receipt = snapshot_regular_file(Path(fixture), fixture_copy, fixture_sha256)
        validation = run_internal_helper(
            python=runtime.python,
            module_path=module_path,
            mode="internal-validate-fixture",
            output_path=raw_dir / "fixture_validation.json",
            cwd=context.source_root,
            env=env,
            timeout=300,
            extra=("--fixture", str(fixture_copy), "--fixture-sha256", fixture_sha256),
        )
        fixture_receipt["validation"] = {
            "artifact": file_identity(raw_dir / "fixture_validation.json"),
            "array_count": validation["array_count"],
            "arrays": validation["arrays"],
        }
        topology = verify_gpu_topology(
            runtime=runtime,
            module_path=module_path,
            raw_dir=raw_dir,
            cwd=context.source_root,
            env=env,
            nvidia_smi=nvidia_smi,
        )
        assert_port_available(port)
        future_root = raw_dir / "d1_future"
        probe_output = raw_dir / "d1_probe"
        server_command = build_server_command(
            runtime=runtime,
            source_root=context.source_root,
            future_root=future_root,
            port=port,
            timeout_seconds=server_group_timeout_seconds,
        )
        command_receipt = {
            "schema_version": PROCESS_SCHEMA,
            "status": "launching",
            "created_at_utc": utc_now(),
            "argv": server_command,
            "cwd": str(context.source_root),
            "nvidia_visible_devices": env.get("NVIDIA_VISIBLE_DEVICES"),
        }
        atomic_json(raw_dir / "server_command.json", command_receipt, exclusive=True)
        server_stdout_path = raw_dir / "server.stdout.log"
        server_stderr_path = raw_dir / "server.stderr.log"
        process, server_stdout, server_stderr = launch_server(
            server_command,
            stdout_path=server_stdout_path,
            stderr_path=server_stderr_path,
            cwd=context.source_root,
            env=env,
        )
        _ACTIVE_SERVER = process
        process_receipt = {
            **command_receipt,
            "status": "started",
            "started_at_utc": utc_now(),
            "process": _proc_start_identity(process.pid),
        }
        atomic_json(raw_dir / "server_process.json", process_receipt, exclusive=True)
        server_contract_path = future_root / "server_contract.json"
        server_contract = wait_for_server_contract(
            process,
            server_contract_path,
            runtime=runtime,
            source_root=context.source_root,
            topology=topology,
            port=port,
            timeout_seconds=readiness_timeout_seconds,
        )
        readiness = {
            "schema_version": "wmf-d1-server-readiness-v1",
            "status": "passed",
            "ready_at_utc": utc_now(),
            "server_process": _proc_start_identity(process.pid),
            "server_contract": file_identity(server_contract_path),
            "identity_receipt": file_identity(future_root / "identity_receipt.json"),
        }
        atomic_json(raw_dir / "server_readiness.json", readiness, exclusive=True)
        probe_command = build_probe_command(
            runtime=runtime,
            source_root=context.source_root,
            fixture=fixture_copy,
            fixture_sha256=fixture_sha256,
            future_root=future_root,
            output_dir=probe_output,
            port=port,
        )
        atomic_json(
            raw_dir / "probe_command.json",
            {"argv": probe_command, "cwd": str(context.source_root), "started_at_utc": utc_now()},
            exclusive=True,
        )
        probe_result = run_logged(
            probe_command,
            stdout_path=raw_dir / "probe.stdout.log",
            stderr_path=raw_dir / "probe.stderr.log",
            cwd=context.source_root,
            env=env,
            timeout=probe_timeout_seconds,
        )
        probe_exit = probe_result.returncode
        report_path = probe_output / "d1_probe_qualification.json"
        if report_path.is_file():
            probe_report = load_json(report_path, "probe_report_unreadable")
        if probe_exit == 0:
            require(isinstance(probe_report, dict), "probe_report_missing")
            require(probe_report.get("schema_version") == "wmf-d1-six-request-qualification-v1", "probe_report_schema_mismatch")
            require(probe_report.get("status") == "passed" and probe_report.get("passed") is True, "probe_report_not_passed")
            require(probe_report.get("generation_request_count") == 6, "probe_request_count_mismatch")
            require(probe_report.get("behavioral_episode_count") == 0, "probe_behavioral_count_nonzero")
            decision = "qualified"
            exit_code = 0
        elif probe_exit == 20 and isinstance(probe_report, dict):
            require(probe_report.get("schema_version") == "wmf-d1-six-request-qualification-v1", "probe_report_schema_mismatch")
            require(probe_report.get("status") == "failed" and probe_report.get("passed") is False, "probe_failure_report_invalid")
            require(probe_report.get("generation_request_count") == 6, "probe_request_count_mismatch")
            decision = "qualification_failed"
            reason = "declared_probe_checks_failed"
            exit_code = 20
        else:
            raise D1JobError("probe_technical_failure", str(probe_exit))
    except D1JobError as error:
        decision = "technical_invalid"
        reason = error.reason
        exit_code = 3
        atomic_json(
            raw_dir / "technical_failure.json",
            {
                "schema_version": FAILURE_SCHEMA,
                "status": "technical_invalid",
                "reason": reason,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "recorded_at_utc": utc_now(),
                "interrupted_signal": _INTERRUPTED_SIGNAL,
            },
        )
    except BaseException as error:
        decision = "technical_invalid"
        reason = "unexpected_orchestration_failure"
        exit_code = 3
        atomic_json(
            raw_dir / "technical_failure.json",
            {
                "schema_version": FAILURE_SCHEMA,
                "status": "technical_invalid",
                "reason": reason,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "recorded_at_utc": utc_now(),
                "interrupted_signal": _INTERRUPTED_SIGNAL,
            },
        )
    finally:
        if process is not None and server_stdout is not None and server_stderr is not None:
            try:
                server_exit = terminate_server(
                    process,
                    stdout_handle=server_stdout,
                    stderr_handle=server_stderr,
                    stdout_path=raw_dir / "server.stdout.log",
                    stderr_path=raw_dir / "server.stderr.log",
                    grace_seconds=terminate_grace_seconds,
                    prior_signal_reap=_SIGNAL_SERVER_REAP,
                )
                atomic_json(raw_dir / "server_exit.json", server_exit, exclusive=True)
            except BaseException as error:
                decision = "technical_invalid"
                reason = "server_reap_failed"
                exit_code = 3
                atomic_json(
                    raw_dir / "server_reap_failure.json",
                    {
                        "schema_version": FAILURE_SCHEMA,
                        "status": "technical_invalid",
                        "reason": reason,
                        "exception_type": type(error).__name__,
                        "exception": str(error),
                        "traceback": traceback.format_exc(),
                        "recorded_at_utc": utc_now(),
                    },
                )
            finally:
                _ACTIVE_SERVER = None
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    report_identity = None
    compact_probe = None
    if probe_report is not None:
        report_path = raw_dir / "d1_probe" / "d1_probe_qualification.json"
        report_identity = file_identity(report_path)
        compact_probe = {
            "status": probe_report.get("status"),
            "passed": probe_report.get("passed"),
            "generation_request_count": probe_report.get("generation_request_count"),
            "behavioral_episode_count": probe_report.get("behavioral_episode_count"),
            "failed_checks": probe_report.get("failed_checks", []),
            "comparisons": probe_report.get("comparisons", {}),
            "sensitivity_measurements": probe_report.get("sensitivity_measurements", {}),
            "artifact": report_identity,
        }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "study_namespace": NAMESPACE,
        "status": "finished",
        "decision": decision,
        "exit_code": exit_code,
        "reason": reason,
        "job_id": context.job_id,
        "study_commit": context.source_commit,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "configuration_id": "D1",
        "generation_request_count": compact_probe.get("generation_request_count", 0) if compact_probe else 0,
        "behavioral_episode_count": 0,
        "fixture": fixture_receipt,
        "capture_manifest": capture_receipt,
        "topology": topology,
        "server_contract": _compact_contract(server_contract, raw_dir / "d1_future/server_contract.json") if server_contract else None,
        "probe": compact_probe,
        "probe_exit_code": probe_exit,
        "server_process_receipt": file_identity(raw_dir / "server_process.json") if (raw_dir / "server_process.json").is_file() else None,
        "server_exit": server_exit,
        "raw_attempt_directory": str(raw_dir.resolve()),
        "raw_failure": file_identity(raw_dir / "technical_failure.json") if (raw_dir / "technical_failure.json").is_file() else None,
        "claim_boundary": (
            "This is a six-request fixed-observation D1 generation qualification. "
            "It contains zero robot episodes and is not physical forecast/action/camera alignment evidence."
        ),
    }
    _publish_receipt(publish_dir / "d1_qualification_job_receipt.json", receipt)
    atomic_json(raw_dir / "job_receipt.json", receipt, exclusive=True)
    return receipt


def _internal_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=("internal-validate-fixture", "internal-torch-topology"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-sha256")
    args = parser.parse_args(argv)
    try:
        if args.mode == "internal-validate-fixture":
            require(args.fixture is not None, "fixture_missing")
            require(isinstance(args.fixture_sha256, str), "fixture_sha256_invalid")
            receipt = validate_fixture_arrays(args.fixture.resolve(), args.fixture_sha256)
        else:
            receipt = torch_topology()
        atomic_json(args.output, receipt, exclusive=True)
        return 0
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "status": "technical_invalid",
            "reason": error.reason if isinstance(error, D1JobError) else "internal_helper_failure",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "recorded_at_utc": utc_now(),
        }
        try:
            atomic_json(args.output.with_suffix(".failure.json"), failure, exclusive=True)
        except BaseException:
            pass
        return 3


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].startswith("internal-"):
        return _internal_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=7200)
    parser.add_argument("--probe-timeout-seconds", type=float, default=43200)
    parser.add_argument("--server-group-timeout-seconds", type=int, default=50000)
    parser.add_argument("--terminate-grace-seconds", type=float, default=5)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    args = parser.parse_args(arguments)
    try:
        receipt = execute_job(
            source_root=args.source_root,
            job_dir=args.job_dir,
            fixture=args.fixture,
            fixture_sha256=args.fixture_sha256,
            capture_manifest=args.capture_manifest,
            capture_manifest_sha256=args.capture_manifest_sha256,
            port=args.port,
            readiness_timeout_seconds=args.readiness_timeout_seconds,
            probe_timeout_seconds=args.probe_timeout_seconds,
            server_group_timeout_seconds=args.server_group_timeout_seconds,
            terminate_grace_seconds=args.terminate_grace_seconds,
            nvidia_smi=args.nvidia_smi,
        )
    except D1JobError as error:
        print(json.dumps({"decision": "technical_invalid", "exit_code": 3, "reason": error.reason}, sort_keys=True))
        return 3
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "exit_code": receipt["exit_code"],
                "generation_request_count": receipt["generation_request_count"],
                "behavioral_episode_count": 0,
                "job_id": receipt["job_id"],
                "published": True,
            },
            sort_keys=True,
        )
    )
    return int(receipt["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
