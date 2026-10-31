#!/usr/bin/env python3
"""Build and run the prospective terminal camera-crop replay witness.

``build-wave`` is descriptor-only.  It authenticates the five immutable replay
prerequisites and a sixth, separately supplied terminal receipt proving that
diagnostic attempt 002 completed both model replays with zero child exits.  It
never edits the queue.  When ``--queue-template`` is supplied with an exact
clean control checkout and commit, the same command authenticates that tracked
empty, non-shutdown queue and emits a directly consumable queue candidate to
stdout; it still does not edit or publish that queue.  ``run`` is the future
detached queue entry point: it revalidates every
staged source and PVC prerequisite, replays N3 and D1 on CPU, validates both raw
signed crop contracts and their witness arrays, and then atomically publishes
exactly those two contracts plus a terminal receipt.

This source does not authorize building or releasing a descriptor while the
current GM/PVC state and attempt-002 receipt identity are unreconciled.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
QUEUE_PATH = Path(__file__).with_name("forecast_timing_queue_jobs.py")
REPLAY_PATH = FORECAST_ROOT / "analysis/camera_crop_replay_witness.py"
DIAGNOSTIC_PATH = Path(__file__).with_name("camera_crop_witness_jobs.py")


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


queue = _load_module(QUEUE_PATH, "wmf_camera_terminal_queue_support")
replay = _load_module(REPLAY_PATH, "wmf_camera_terminal_replay_support")
diagnostic = _load_module(DIAGNOSTIC_PATH, "wmf_camera_diagnostic_predecessor_support")

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
FORECAST_RELATIVE = queue.FORECAST_RELATIVE

THIS_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/camera_crop_witness_terminal_jobs.py"
)
QUEUE_RELATIVE = queue.THIS_RELATIVE
REPLAY_RELATIVE = FORECAST_RELATIVE / "analysis/camera_crop_replay_witness.py"
DIAGNOSTIC_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/camera_crop_witness_jobs.py"
)
TERMINAL_CONTRACT_RELATIVE = (
    FORECAST_RELATIVE
    / "experiments/forecast_layout/camera_crop_witness_terminal_contract.json"
)
REPLAY_CONTRACT_RELATIVE = diagnostic.CONTRACT_RELATIVE
N3_CONTRACT_RELATIVE = diagnostic.N3_CONTRACT_RELATIVE
D1_IDENTITY_RELATIVE = diagnostic.D1_IDENTITY_RELATIVE
D1_CAPTURE_OVERLAY_RELATIVE = diagnostic.D1_CAPTURE_OVERLAY_RELATIVE

TERMINAL_CONTRACT_SCHEMA = "wmf-camera-crop-witness-terminal-runtime-contract-v1"
WAVE_SCHEMA = "wmf-camera-crop-witness-terminal-wave-v1"
QUEUE_MANIFEST_SCHEMA = "wmf-cluster-queue-v1"
JOB_RECEIPT_SCHEMA = "wmf-camera-crop-witness-queue-job-v1"
FAILURE_SCHEMA = "wmf-camera-crop-witness-queue-job-v1"
JOB_ID = "camera-crop-replay-witness-003"
PRIOR_JOB_ID = "camera-crop-replay-witness-002"
PRIOR_PRIOR_JOB_ID = "camera-crop-replay-witness-001"
WORKER_ROLE = "wmf-forecast-0912-worker-06"
MAX_WALL_SECONDS = 10800
PUBLISH_LOG_TAIL_BYTES = 0
CHILD_LOG_TAIL_BYTES = 8192
RAW_SUBDIR = Path("raw/camera_crop_witness_terminal")
SUCCESS_RECEIPT = "camera_crop_witness_job_receipt.json"
FAILURE_RECEIPT = "camera_crop_witness_job_failure.json"
MODEL_CONTRACT_NAMES = {
    "N3": "n3_camera_crop_contract.json",
    "D1": "d1_camera_crop_contract.json",
}
PRIOR_DIAGNOSTIC_CLUSTER_PATH = (
    CONTROL_ROOT / "jobs" / PRIOR_JOB_ID / "publish" / FAILURE_RECEIPT
)
PRIOR_DIAGNOSTIC_RESULTS_RELATIVE = (
    Path("results/jobs") / PRIOR_JOB_ID / "publish" / FAILURE_RECEIPT
)
PREREQUISITE_CLUSTER_PATHS = dict(diagnostic.PREREQUISITE_CLUSTER_PATHS)
CONTROL_QUEUE_RELATIVE = (
    FORECAST_RELATIVE / "execution/20260912/autonomy/cluster_queue.json"
)
PREREQUISITE_RESULTS_RELATIVES = {
    "n3_source_audit": Path(
        "results/jobs/timing-n3-source-audit-001/publish/n3_source_audit.json"
    ),
    "d1_source_audit": Path(
        "results/jobs/timing-d1-source-audit-001/publish/d1_source_audit.json"
    ),
    "n3_generation": Path(
        "results/jobs/timing-n3-live-generation-p00-002/publish/n3_qualification.json"
    ),
    "d1_generation": Path(
        "results/jobs/timing-d1-normalize-generation-001/publish/d1_generation_probe.json"
    ),
    "d1_qualification": Path(
        "results/jobs/d1-first-live-005/publish/d1_qualification_job_receipt.json"
    ),
}
AUTHENTICATED_JSON_LIMIT_BYTES = 2 * 1024 * 1024
GIT_INDEX_LIMIT_BYTES = 256 * 1024 * 1024
GIT_POINTER_LIMIT_BYTES = 16 * 1024
GIT_LIST_LIMIT_BYTES = 64 * 1024 * 1024
GIT_TRACKED_FILE_LIMIT_BYTES = 512 * 1024 * 1024
GIT_TRACKED_TOTAL_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
GIT_TRACKED_ENTRY_LIMIT = 100_000
GIT_EXECUTABLE = Path("/usr/bin/git")

PRIOR_SOURCE_PINS = {
    DIAGNOSTIC_RELATIVE: "6613de67a170d1719d5b4e1a5be1e2b9305e1a4b6087e8f1d2006f3e8b0f3145",
    REPLAY_CONTRACT_RELATIVE: "31313288aac8cd917707d52eea8c4fca9ee77de1d3c0b42ade466aa78449706a",
    REPLAY_RELATIVE: "30d13920383ea36d26a8a3208685ab057fb070a39e520fcdc6caa4e8449a9f4f",
    QUEUE_RELATIVE: "d448a02ad627bc65c143a3932ae2a4378be642e36a49645f5dc8c3727f00c353",
    N3_CONTRACT_RELATIVE: "a9503011b8f7236dd3e8eb9c2e4328f2ab95021b4f72e5900c01b51cee743e61",
    D1_IDENTITY_RELATIVE: "6c11a3158c944f0deb50116b071511c6bee267e890619746f993a8e8d043cc0b",
    D1_CAPTURE_OVERLAY_RELATIVE: (
        "3f3fe50f4b7747c52a995efb1e07bf00af8a5e38620829d8b0c991b43fd9a7ec"
    ),
}

SUCCESS_RECEIPT_KEYS = {
    "schema_version", "namespace", "study_id", "status", "decision", "job_id",
    "job_dir", "science_counts", "simulator_state_render_used",
    "whole_frame_identity", "safe_for_physical_alignment_input",
    "safe_to_release_confirmation", "confirmation_released",
    "behavioral_policy_skill_evaluated", "raw_outputs_recoverable_on_gm_pvc",
    "outputs", "payload_sha256",
}
OUTPUT_ENTRY_KEYS = {
    "path", "sha256", "bytes", "schema_version", "model_id",
    "camera_crop_id", "payload_sha256",
}
EXPECTED_CAMERA_CROP_IDS = {
    "N3": "n3-over-shoulder-left-168x320-v1",
    "D1": "d1-over-shoulder-left-176x320-v1",
}
PRIOR_RECEIPT_KEYS = {
    "schema_version", "namespace", "study_id", "status", "decision", "job_id",
    "study_commit", "queue_role", "failure", "diagnostic_only", "prior_attempt",
    "science_counts", "simulator_state_render_used", "whole_frame_identity",
    "safe_to_release_confirmation", "confirmation_released", "claim_boundary",
    "completed_at_utc", "payload_sha256",
}
PRIOR_FAILURE_KEYS = {
    "error_type", "message_bytes", "message_sha256", "traceback_frames",
    "traceback_frame_count", "exception_message_published",
    "traceback_source_text_published", "child_exits", "child_diagnostics",
    "child_orchestration_failure",
}
WAVE_KEYS = {
    "schema_version", "namespace", "study_id", "source_commit", "status",
    "jobs", "implementation", "replay_prerequisite_receipts",
    "prior_diagnostic", "expected_outputs", "science_counts",
    "simulator_state_render_used", "whole_frame_identity",
    "safe_for_physical_alignment_input_only_after_runtime_success",
    "safe_to_release_confirmation", "confirmation_released",
    "descriptor_dispatched", "claim_boundary",
}
QUEUE_MANIFEST_KEYS = {"schema_version", "namespace", "shutdown", "jobs"}
WAVE_STATUS = (
    "terminal_descriptor_not_dispatched_six_external_receipt_gates_passed"
)
WAVE_CLAIM_BOUNDARY = (
    "A descriptor-only CPU replay candidate. Building this value does not edit or "
    "release the queue, run attempt 003, qualify a crop, create annotation evidence, "
    "estimate forecast accuracy, or authorize confirmation."
)


class CameraCropTerminalError(RuntimeError):
    """A terminal descriptor, prerequisite, replay, or publication gate failed."""


class CameraCropTerminalChildError(CameraCropTerminalError):
    """One or both terminal replay children failed or could not be reaped."""

    def __init__(
        self,
        exits: Mapping[str, int | None],
        diagnostics: Mapping[str, Any],
        *,
        orchestration_failure: Mapping[str, Any] | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(detail or f"terminal camera replay child exits changed: {dict(exits)}")
        self.exits = dict(exits)
        self.diagnostics = dict(diagnostics)
        self.orchestration_failure = (
            None if orchestration_failure is None else dict(orchestration_failure)
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CameraCropTerminalError(message)


def _lexical_absolute_path(path: Path, label: str) -> Path:
    """Return an already-normalized absolute path without following links."""

    try:
        raw = os.fspath(path)
    except (TypeError, ValueError) as error:
        raise CameraCropTerminalError(f"{label} path is invalid") from error
    require(isinstance(raw, str) and raw and "\x00" not in raw,
            f"{label} path is invalid")
    candidate = Path(raw)
    require(candidate.is_absolute(), f"{label} path is not absolute")
    require(".." not in candidate.parts and str(candidate) == os.path.normpath(raw),
            f"{label} path is not lexically normalized")
    return candidate


def _directory_flags() -> int:
    require(
        hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"),
        "no-follow descriptor traversal is unavailable",
    )
    return (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_nofollow(path: Path, label: str) -> tuple[int, Path]:
    """Open an absolute directory component by component without following links."""

    absolute = _lexical_absolute_path(path, label)
    flags = _directory_flags()
    try:
        descriptor = os.open(os.sep, flags)
    except OSError as error:
        raise CameraCropTerminalError(
            f"cannot open filesystem root for {label}"
        ) from error
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise CameraCropTerminalError(
                    f"{label} is missing, not a directory, or contains a symlink"
                ) from error
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_same_open_directory(path: Path, descriptor: int, label: str) -> None:
    comparison, _ = _open_directory_nofollow(path, label)
    try:
        first = os.fstat(descriptor)
        second = os.fstat(comparison)
        require(
            (first.st_dev, first.st_ino, first.st_mode)
            == (second.st_dev, second.st_ino, second.st_mode),
            f"{label} changed after it was opened",
        )
    finally:
        os.close(comparison)


def _read_regular_file_once(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> tuple[bytes, dict[str, Any], tuple[int, ...]]:
    """Read bounded bytes once from a held parent and retain that exact identity."""

    require(type(maximum_bytes) is int and maximum_bytes > 0,
            f"{label} byte limit is invalid")
    absolute = _lexical_absolute_path(path, label)
    require(absolute.name not in {"", ".", ".."}, f"{label} path is unsafe")
    parent_fd, parent = _open_directory_nofollow(
        absolute.parent, f"{label} parent"
    )
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            named_before = os.stat(
                absolute.name, dir_fd=parent_fd, follow_symlinks=False
            )
            descriptor = os.open(absolute.name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise CameraCropTerminalError(
                f"{label} is missing, not regular, or is a symlink"
            ) from error
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(named_before.st_mode)
            and stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and _stat_identity(named_before) == _stat_identity(before),
            f"{label} is not the expected singly linked regular file",
        )
        require(0 <= before.st_size <= maximum_bytes,
                f"{label} exceeds the byte limit")
        if expected_bytes is not None:
            require(
                type(expected_bytes) is int
                and expected_bytes >= 0
                and before.st_size == expected_bytes,
                f"{label} byte count changed before read",
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            try:
                block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            except InterruptedError:
                continue
            except OSError as error:
                raise CameraCropTerminalError(f"cannot read {label}") from error
            if not block:
                break
            observed += len(block)
            require(observed <= maximum_bytes, f"{label} exceeds the byte limit")
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        require(
            _stat_identity(before) == _stat_identity(after)
            and after.st_nlink == 1
            and len(payload) == after.st_size,
            f"{label} changed while it was read",
        )
        _require_same_open_directory(parent, parent_fd, f"{label} parent")
        try:
            named_after = os.stat(
                absolute.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except OSError as error:
            raise CameraCropTerminalError(f"{label} path changed after read") from error
        require(
            stat.S_ISREG(named_after.st_mode)
            and named_after.st_nlink == 1
            and _stat_identity(named_after) == _stat_identity(after),
            f"{label} path changed after read",
        )
        digest = hashlib.sha256(payload).hexdigest()
        return payload, {
            "path": str(absolute), "bytes": len(payload), "sha256": digest,
        }, _stat_identity(after)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _require_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{label} contains a nonfinite number")
    elif isinstance(value, list):
        for item in value:
            _require_finite_json(item, label)
    elif isinstance(value, Mapping):
        for item in value.values():
            _require_finite_json(item, label)


def _parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CameraCropTerminalError(f"{label} is not strict JSON") from error
    require(isinstance(value, dict), f"{label} is not a JSON object")
    _require_finite_json(value, label)
    return value


def _exact_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int/float equality aliases."""

    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_exact_json_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _exact_json_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def zero_science_counts() -> dict[str, int]:
    return dict(replay.ZERO_SCIENCE_COUNTS)


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "queue_wrapper": root / THIS_RELATIVE,
        "queue_support": root / QUEUE_RELATIVE,
        "replay": root / REPLAY_RELATIVE,
        "diagnostic_predecessor": root / DIAGNOSTIC_RELATIVE,
        "terminal_contract": root / TERMINAL_CONTRACT_RELATIVE,
        "replay_runtime_contract": root / REPLAY_CONTRACT_RELATIVE,
        "n3_runtime_contract": root / N3_CONTRACT_RELATIVE,
        "d1_identity_contract": root / D1_IDENTITY_RELATIVE,
        "d1_capture_overlay": root / D1_CAPTURE_OVERLAY_RELATIVE,
    }


def _local_implementation() -> dict[str, dict[str, Any]]:
    paths = _source_paths(REPOSITORY_ROOT)
    missing = [name for name, path in paths.items() if not path.is_file()]
    require(not missing, f"terminal camera implementation is missing: {missing}")
    return {name: queue.file_identity(path) for name, path in paths.items()}


def _runtime_contract(root: Path) -> dict[str, Any]:
    path = root / TERMINAL_CONTRACT_RELATIVE
    value = replay.load_json(path, "terminal camera witness runtime contract")
    require(set(value) == {
        "schema_version", "study_id", "job", "prior_diagnostic", "replay_runtime",
        "child_runtime", "expected_outputs", "science_counts",
        "simulator_state_render_used", "whole_frame_identity",
        "success_receipt_safe_for_physical_alignment_input",
        "safe_to_release_confirmation", "confirmation_released",
        "behavioral_policy_skill_evaluated", "claim_boundary",
    }, "terminal runtime contract fields changed")
    require(value.get("schema_version") == TERMINAL_CONTRACT_SCHEMA,
            "terminal runtime contract schema changed")
    require(value.get("study_id") == STUDY_ID, "terminal runtime contract study changed")
    require(_exact_json_equal(value.get("job"), {
        "job_id": JOB_ID,
        "worker_role": WORKER_ROLE,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "cpu_only": True,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }), "terminal runtime contract queue identity changed")
    require(_exact_json_equal(value.get("prior_diagnostic"), {
        "job_id": PRIOR_JOB_ID,
        "receipt_name": FAILURE_RECEIPT,
        "schema_version": FAILURE_SCHEMA,
        "status": "technical_invalid",
        "decision": "no_go",
        "diagnostic_only": True,
        "required_child_exits": {"N3": 0, "D1": 0},
        "prior_attempt": PRIOR_PRIOR_JOB_ID,
        "preserved": True,
    }), "terminal runtime prior-diagnostic gate changed")
    replay_descriptor = value.get("replay_runtime")
    require(_exact_json_equal(replay_descriptor, {
        "relative_path": str(REPLAY_CONTRACT_RELATIVE),
        "bytes": 9862,
        "sha256": PRIOR_SOURCE_PINS[REPLAY_CONTRACT_RELATIVE],
        "job_id": PRIOR_JOB_ID,
        "explicit_prerequisite_receipt_count": 5,
        "total_external_receipt_gate_count": 6,
        "authority_role": "replay_inputs_only_not_terminal_authority",
    }), "terminal replay-runtime descriptor changed")
    replay.file_identity(
        root / REPLAY_CONTRACT_RELATIVE,
        label="terminal pinned replay runtime",
        expected_sha256=replay_descriptor["sha256"],
        expected_bytes=replay_descriptor["bytes"],
        within=root,
    )
    diagnostic._runtime_contract(root)
    require(_exact_json_equal(value.get("child_runtime"), {
        "mode": "terminal_replay_after_authenticated_attempt_002_diagnostic",
        "diagnostic_only": False,
        "inherit_queue_environment": True,
        "overrides": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "failure_log_tail_bytes": CHILD_LOG_TAIL_BYTES,
        "publish_crop_contracts": True,
        "terminal_receipt_written_last": True,
        "atomic_directory_publication": True,
    }), "terminal child runtime changed")
    require(_exact_json_equal(value.get("expected_outputs"), {
        "publish_file_count": 3,
        "published_camera_crop_contracts": MODEL_CONTRACT_NAMES,
        "success_receipt": SUCCESS_RECEIPT,
        "failure_receipt": FAILURE_RECEIPT,
    }), "terminal output inventory changed")
    require(_exact_json_equal(value.get("science_counts"), zero_science_counts()),
            "terminal runtime science counts changed")
    require(value.get("simulator_state_render_used") is False
            and value.get("whole_frame_identity") is False
            and value.get("success_receipt_safe_for_physical_alignment_input") is True
            and value.get("safe_to_release_confirmation") is False
            and value.get("confirmation_released") is False
            and value.get("behavioral_policy_skill_evaluated") is False,
            "terminal runtime authority boundary changed")
    return value


def _descriptor_shape(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256", "bytes"},
            f"{label} descriptor fields changed")
    path = Path(value["path"])
    require(path.is_absolute(), f"{label} path is not absolute")
    digest = queue._verified_sha(value["sha256"], f"{label} digest")
    size = value["bytes"]
    require(type(size) is int and size > 0, f"{label} byte count is invalid")
    return {"path": str(path), "sha256": digest, "bytes": size}


def _validate_safe_path_projection(value: Any, label: str) -> None:
    require(isinstance(value, Mapping), f"{label} safe path projection is missing")
    if set(value) == {"absolute_path"}:
        path = value["absolute_path"]
        require(isinstance(path, str)
                and Path(path).is_absolute()
                and str(Path(path)) == path
                and "." not in Path(path).parts
                and ".." not in Path(path).parts,
                f"{label} safe absolute path is invalid")
        require(diagnostic._safe_diagnostic_path(path) == dict(value),
                f"{label} absolute path should have been redacted")
        return
    require(set(value) == {"redacted_value_bytes", "redacted_value_sha256"}
            and type(value["redacted_value_bytes"]) is int
            and value["redacted_value_bytes"] >= 0
            and queue.SHA256_RE.fullmatch(str(value["redacted_value_sha256"])) is not None,
            f"{label} safe redacted path changed")


def _validate_safe_identifier(value: Any, label: str) -> None:
    if isinstance(value, str):
        require(re.fullmatch(r"[A-Za-z0-9_.<>-]+", value) is not None,
                f"{label} safe identifier changed")
        require(diagnostic._safe_identifier(value) == value,
                f"{label} identifier should have been redacted")
        return
    require(isinstance(value, Mapping)
            and set(value) == {"redacted_value_bytes", "redacted_value_sha256"}
            and type(value["redacted_value_bytes"]) is int
            and value["redacted_value_bytes"] >= 0
            and queue.SHA256_RE.fullmatch(str(value["redacted_value_sha256"])) is not None,
            f"{label} safe identifier projection changed")


def _validate_safe_log_summary(value: Any, *, model: str) -> None:
    require(isinstance(value, Mapping) and set(value) == {
        "preflight_events", "traceback_frames", "terminal_errors", "line_count",
        "recognized_line_count", "arbitrary_text_published",
    }, f"attempt 002 {model} safe log summary fields changed")
    require(isinstance(value["preflight_events"], list)
            and isinstance(value["traceback_frames"], list)
            and isinstance(value["terminal_errors"], list)
            and type(value["line_count"]) is int
            and type(value["recognized_line_count"]) is int
            and 0 <= value["recognized_line_count"] <= value["line_count"]
            and value["arbitrary_text_published"] is False,
            f"attempt 002 {model} safe log summary values changed")
    for index, frame in enumerate(value["traceback_frames"]):
        require(isinstance(frame, Mapping) and set(frame) == {"source", "line", "function"}
                and type(frame["line"]) is int and frame["line"] > 0,
                f"attempt 002 {model} traceback frame changed")
        _validate_safe_path_projection(frame["source"], f"{model} traceback {index}")
        _validate_safe_identifier(frame["function"], f"{model} traceback function {index}")
    for index, error in enumerate(value["terminal_errors"]):
        require(isinstance(error, Mapping)
                and set(error).issubset({
                    "error_type", "message_bytes", "message_sha256", "missing_module"
                })
                and {"error_type", "message_bytes", "message_sha256"}.issubset(error)
                and type(error["message_bytes"]) is int and error["message_bytes"] >= 0
                and queue.SHA256_RE.fullmatch(str(error["message_sha256"])) is not None,
                f"attempt 002 {model} terminal error changed")
        _validate_safe_identifier(error["error_type"], f"{model} error type {index}")
        if "missing_module" in error:
            _validate_safe_identifier(
                error["missing_module"], f"{model} missing module {index}"
            )
    for index, event in enumerate(value["preflight_events"]):
        require(isinstance(event, Mapping) and set(event) == {
            "model_id", "phase", "python", "module_specs", "path_environment",
            "cuda_visible_devices_empty", "source_line_bytes", "source_line_sha256",
        }, f"attempt 002 {model} preflight event fields changed")
        require(event["model_id"] == model
                and event["phase"] in {"before_replay", "after_replay_failure"}
                and isinstance(event["python"], Mapping)
                and set(event["python"]) == {"lexical_path", "resolved_path"}
                and isinstance(event["module_specs"], Mapping)
                and isinstance(event["path_environment"], Mapping)
                and event["cuda_visible_devices_empty"] is True
                and type(event["source_line_bytes"]) is int
                and event["source_line_bytes"] > 0
                and queue.SHA256_RE.fullmatch(str(event["source_line_sha256"])) is not None,
                f"attempt 002 {model} preflight event values changed")
        _validate_safe_path_projection(
            event["python"]["lexical_path"], f"{model} preflight lexical path {index}"
        )
        _validate_safe_path_projection(
            event["python"]["resolved_path"], f"{model} preflight resolved path {index}"
        )
        require(set(event["module_specs"]).issubset({
            "numpy", "torch", "torchvision", "PIL", "openpi_client.image_tools"
        }), f"attempt 002 {model} preflight module inventory changed")
        for module_name, module in event["module_specs"].items():
            require(isinstance(module, Mapping)
                    and set(module).issubset({
                        "found", "origin", "search_locations", "lookup_error_type",
                        "lookup_error_detail_bytes", "lookup_error_detail_sha256",
                    })
                    and "found" in module and type(module["found"]) is bool,
                    f"attempt 002 {model} preflight module changed: {module_name}")
            if "origin" in module:
                _validate_safe_path_projection(
                    module["origin"], f"{model} {module_name} origin {index}"
                )
            if "search_locations" in module:
                require(isinstance(module["search_locations"], list),
                        f"attempt 002 {model} module search paths changed")
                for path_index, path in enumerate(module["search_locations"]):
                    _validate_safe_path_projection(
                        path, f"{model} {module_name} search {index}:{path_index}"
                    )
            if "lookup_error_type" in module:
                _validate_safe_identifier(
                    module["lookup_error_type"], f"{model} {module_name} lookup error"
                )
            detail_fields = {
                "lookup_error_detail_bytes", "lookup_error_detail_sha256"
            } & set(module)
            require(not detail_fields or detail_fields == {
                "lookup_error_detail_bytes", "lookup_error_detail_sha256"
            }, f"attempt 002 {model} module lookup detail fields changed")
            if detail_fields:
                require(type(module["lookup_error_detail_bytes"]) is int
                        and module["lookup_error_detail_bytes"] >= 0
                        and queue.SHA256_RE.fullmatch(
                            str(module["lookup_error_detail_sha256"])
                        ) is not None,
                        f"attempt 002 {model} module lookup detail changed")
        require(set(event["path_environment"]).issubset({
            "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"
        }), f"attempt 002 {model} preflight path environment fields changed")
        for name, rows in event["path_environment"].items():
            require(isinstance(rows, list),
                    f"attempt 002 {model} preflight {name} paths changed")
            for path_index, path in enumerate(rows):
                if isinstance(path, Mapping) and set(path) == {
                    "redacted_descriptor_sha256"
                }:
                    require(queue.SHA256_RE.fullmatch(
                        str(path["redacted_descriptor_sha256"])
                    ) is not None,
                            f"attempt 002 {model} redacted descriptor changed")
                else:
                    _validate_safe_path_projection(
                        path, f"{model} preflight {name} {index}:{path_index}"
                    )


def _validate_runtime_preflight(value: Any, *, model: str) -> None:
    require(isinstance(value, Mapping) and set(value) == {
        "python", "path_environment", "cpu_offline_overrides",
        "inherited_queue_environment", "argv_published",
        "arbitrary_environment_values_published",
    }, f"attempt 002 {model} runtime preflight fields changed")
    python = value["python"]
    require(isinstance(python, Mapping) and set(python) == {"lexical_path", "resolved_file"},
            f"attempt 002 {model} runtime Python fields changed")
    expected_python = {
        "N3": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python",
        "D1": "/data/users/ali/vla_wam/envs/dreamzero-ab790c1-py311/bin/python",
    }[model]
    require(python["lexical_path"] == expected_python,
            f"attempt 002 {model} runtime Python path changed")
    resolved = _descriptor_shape(
        python["resolved_file"], f"attempt 002 {model} resolved Python"
    )
    resolved_path = Path(resolved["path"])
    allowed_resolved_parents = {
        Path(expected_python).parent,
        Path("/usr/bin"),
        Path("/usr/local/bin"),
    }
    require(
        resolved_path.parent in allowed_resolved_parents
        and re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", resolved_path.name)
        is not None,
        f"attempt 002 {model} resolved Python path changed",
    )
    paths = value["path_environment"]
    require(isinstance(paths, Mapping)
            and set(paths) == {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
            and all(isinstance(paths[name], list) for name in paths),
            f"attempt 002 {model} runtime path environment changed")
    for name, rows in paths.items():
        for index, row in enumerate(rows):
            _validate_safe_path_projection(row, f"{model} {name} path {index}")
    require(_exact_json_equal(value["cpu_offline_overrides"], {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }), f"attempt 002 {model} CPU/offline overrides changed")
    require(value["inherited_queue_environment"] is True
            and value["argv_published"] is False
            and value["arbitrary_environment_values_published"] is False,
            f"attempt 002 {model} runtime preflight authority changed")


def _validate_prior_child_diagnostic(model: str, value: Any) -> None:
    require(isinstance(value, Mapping), f"attempt 002 {model} diagnostic is missing")
    require(set(value) == {
        "model_id", "exit_code", "full_log", "bounded_tail", "argv_published",
        "environment_values_published", "launched", "launch_receipt",
        "runtime_preflight",
    }, f"attempt 002 {model} diagnostic fields changed")
    require(value.get("model_id") == model
            and type(value.get("exit_code")) is int and value.get("exit_code") == 0,
            f"attempt 002 {model} diagnostic did not exit zero")
    require(value.get("launched") is True, f"attempt 002 {model} was not launched")
    raw_root = CONTROL_ROOT / "jobs" / PRIOR_JOB_ID / "raw/camera_crop_witness"
    full_log = _descriptor_shape(value.get("full_log"), f"attempt 002 {model} full log")
    launch = _descriptor_shape(
        value.get("launch_receipt"), f"attempt 002 {model} launch receipt"
    )
    require(full_log["path"] == str(raw_root / f"{model.lower()}_replay.log"),
            f"attempt 002 {model} full log path changed")
    require(launch["path"] == str(
        raw_root / "child_launch_receipts" / f"{model.lower()}_launch.json"
    ), f"attempt 002 {model} launch receipt path changed")
    bounded = value.get("bounded_tail")
    require(isinstance(bounded, Mapping) and set(bounded) == {
        "maximum_bytes", "source_offset_bytes", "raw_bytes", "raw_sha256",
        "partial_leading_line_dropped_bytes", "safe_structured_summary", "truncated",
    }
            and type(bounded.get("maximum_bytes")) is int
            and bounded.get("maximum_bytes") == CHILD_LOG_TAIL_BYTES
            and type(bounded.get("source_offset_bytes")) is int
            and bounded["source_offset_bytes"] >= 0
            and type(bounded.get("raw_bytes")) is int
            and 0 <= bounded["raw_bytes"] <= CHILD_LOG_TAIL_BYTES
            and type(bounded.get("partial_leading_line_dropped_bytes")) is int
            and 0 <= bounded["partial_leading_line_dropped_bytes"] <= bounded["raw_bytes"]
            and type(bounded.get("truncated")) is bool
            and queue.SHA256_RE.fullmatch(str(bounded.get("raw_sha256", ""))) is not None,
            f"attempt 002 {model} bounded log evidence changed")
    expected_offset = max(0, full_log["bytes"] - CHILD_LOG_TAIL_BYTES)
    expected_raw_bytes = min(full_log["bytes"], CHILD_LOG_TAIL_BYTES)
    require(
        bounded["source_offset_bytes"] == expected_offset
        and bounded["raw_bytes"] == expected_raw_bytes
        and bounded["truncated"] is (expected_offset > 0),
        f"attempt 002 {model} bounded log arithmetic changed",
    )
    dropped = bounded["partial_leading_line_dropped_bytes"]
    require(
        dropped == 0 if expected_offset == 0 else 1 <= dropped <= expected_raw_bytes,
        f"attempt 002 {model} bounded log prefix accounting changed",
    )
    summary = bounded.get("safe_structured_summary")
    _validate_safe_log_summary(summary, model=model)
    require(value.get("argv_published") is False
            and value.get("environment_values_published") is False,
            f"attempt 002 {model} diagnostic authority changed")
    _validate_runtime_preflight(value.get("runtime_preflight"), model=model)


def _validate_prior_diagnostic_value(value: Mapping[str, Any]) -> str:
    require(isinstance(value, Mapping) and set(value) == PRIOR_RECEIPT_KEYS,
            "attempt 002 receipt fields changed")
    queue.verify_signed_document(value, "attempt 002 diagnostic receipt")
    exact = {
        "schema_version": FAILURE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "technical_invalid",
        "decision": "no_go",
        "job_id": PRIOR_JOB_ID,
        "queue_role": WORKER_ROLE,
        "diagnostic_only": True,
        "prior_attempt": PRIOR_PRIOR_JOB_ID,
        "science_counts": zero_science_counts(),
        "simulator_state_render_used": False,
        "whole_frame_identity": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
    }
    for key, expected in exact.items():
        require(_exact_json_equal(value.get(key), expected),
                f"attempt 002 receipt changed: {key}")
    require(value.get("claim_boundary") == (
        "Diagnostic-only technical-invalid CPU replay preserving attempt 001. No model, "
        "simulator, request, action, behavior, label, or confirmation job was started; "
        "no crop contract is published. Bounded signed log evidence is diagnostic only."
    ), "attempt 002 claim boundary changed")
    commit = queue._verified_commit(value.get("study_commit"))
    failure = value.get("failure")
    require(isinstance(failure, Mapping) and set(failure) == PRIOR_FAILURE_KEYS,
            "attempt 002 failure evidence fields changed")
    child_exits = failure.get("child_exits")
    require(failure.get("error_type") == "CameraCropChildError"
            and type(failure.get("message_bytes")) is int
            and failure.get("message_bytes") == 120
            and failure.get("message_sha256")
            == "7730b172cf378e1abfc2d769f743c2be98457c10e6505562fe60e9e6af1ed1d9"
            and isinstance(child_exits, Mapping) and set(child_exits) == {"N3", "D1"}
            and all(type(child_exits[model]) is int and child_exits[model] == 0
                    for model in ("N3", "D1"))
            and failure.get("child_orchestration_failure") is None
            and failure.get("exception_message_published") is False
            and failure.get("traceback_source_text_published") is False,
            "attempt 002 is not a completed two-child diagnostic")
    require(type(failure.get("traceback_frame_count")) is int
            and failure["traceback_frame_count"] == len(failure.get("traceback_frames", [])),
            "attempt 002 traceback evidence changed")
    require(isinstance(failure.get("traceback_frames"), list),
            "attempt 002 traceback frames changed")
    for index, frame in enumerate(failure["traceback_frames"]):
        require(isinstance(frame, Mapping) and set(frame) == {"source", "line", "function"}
                and type(frame["line"]) is int and frame["line"] > 0,
                "attempt 002 traceback frame changed")
        _validate_safe_path_projection(frame["source"], f"attempt 002 traceback {index}")
        _validate_safe_identifier(frame["function"], f"attempt 002 function {index}")
    completed_at = value.get("completed_at_utc")
    require(isinstance(completed_at, str)
            and re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|\+00:00)",
                completed_at,
            ) is not None,
            "attempt 002 completion timestamp changed")
    try:
        parsed_completed_at = datetime.fromisoformat(
            completed_at[:-1] + "+00:00" if completed_at.endswith("Z") else completed_at
        )
    except ValueError as error:
        raise CameraCropTerminalError(
            "attempt 002 completion timestamp changed"
        ) from error
    require(
        parsed_completed_at.utcoffset() == timedelta(0),
        "attempt 002 completion timestamp changed",
    )
    diagnostics = failure.get("child_diagnostics")
    require(isinstance(diagnostics, Mapping) and set(diagnostics) == {"N3", "D1"},
            "attempt 002 child diagnostic inventory changed")
    for model in ("N3", "D1"):
        _validate_prior_child_diagnostic(model, diagnostics[model])
    return commit


def _pread_exact_file(
    descriptor: int, metadata: os.stat_result, maximum_bytes: int, label: str
) -> bytes:
    require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and 0 <= metadata.st_size <= maximum_bytes,
        f"{label} is not the expected singly linked regular file",
    )
    chunks: list[bytes] = []
    offset = 0
    while offset < metadata.st_size:
        try:
            block = os.pread(
                descriptor,
                min(1024 * 1024, metadata.st_size - offset),
                offset,
            )
        except InterruptedError:
            continue
        except OSError as error:
            raise CameraCropTerminalError(f"cannot read {label}") from error
        require(bool(block), f"{label} was truncated while it was read")
        chunks.append(block)
        offset += len(block)
    try:
        extra = os.pread(descriptor, 1, metadata.st_size)
    except OSError as error:
        raise CameraCropTerminalError(f"cannot finish reading {label}") from error
    require(not extra, f"{label} grew while it was read")
    payload = b"".join(chunks)
    require(
        _stat_identity(os.fstat(descriptor)) == _stat_identity(metadata),
        f"{label} changed while it was read",
    )
    return payload


def _open_held_regular_at(
    parent_fd: int,
    name: str,
    label: str,
    *,
    maximum_bytes: int,
) -> tuple[int, bytes, tuple[int, ...]]:
    require(
        name not in {"", ".", ".."} and "/" not in name,
        f"{label} name is unsafe",
    )
    descriptor: int | None = None
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        require(
            _stat_identity(named) == _stat_identity(opened),
            f"{label} changed while it was opened",
        )
        payload = _pread_exact_file(descriptor, opened, maximum_bytes, label)
        return descriptor, payload, _stat_identity(opened)
    except (OSError, CameraCropTerminalError) as error:
        if descriptor is not None:
            os.close(descriptor)
        if isinstance(error, CameraCropTerminalError):
            raise
        raise CameraCropTerminalError(
            f"{label} is missing, not regular, or is a symlink"
        ) from error


def _verify_held_regular_at(
    parent_fd: int,
    name: str,
    descriptor: int,
    payload: bytes,
    identity: tuple[int, ...],
    label: str,
    *,
    maximum_bytes: int,
) -> None:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
    except OSError as error:
        raise CameraCropTerminalError(f"{label} path changed") from error
    require(
        _stat_identity(named) == identity == _stat_identity(opened)
        and _pread_exact_file(descriptor, opened, maximum_bytes, label) == payload,
        f"{label} bytes or metadata changed during read-only validation",
    )


def _one_line_path(payload: bytes, base: Path, label: str) -> Path:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CameraCropTerminalError(f"{label} is not UTF-8") from error
    require(
        text.endswith("\n") and text.count("\n") == 1,
        f"{label} is malformed",
    )
    raw = text[:-1]
    require(bool(raw) and "\x00" not in raw, f"{label} is malformed")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(os.fspath(base / candidate)))
    return _lexical_absolute_path(candidate, label)


def _git_directory_pointer(payload: bytes, root: Path) -> Path:
    prefix = b"gitdir: "
    require(payload.startswith(prefix), "Git administrative pointer is malformed")
    return _one_line_path(
        payload[len(prefix):], root, "Git administrative directory"
    )


def _require_trusted_git_directory(path: Path) -> None:
    current = Path(os.sep)
    directories = [current]
    for component in path.parts[1:]:
        current = current / component
        directories.append(current)
    for current in directories:
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise CameraCropTerminalError(
                "trusted Git executable directory is unavailable"
            ) from error
        require(
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == 0
            and metadata.st_mode & 0o022 == 0,
            "trusted Git executable directory is not root-owned and protected",
        )


@dataclass
class _HeldGitCheckout:
    label: str
    root: Path
    root_fd: int
    dot_git_kind: str
    dot_git_fd: int
    dot_git_identity: tuple[int, ...]
    dot_git_payload: bytes | None
    admin_path: Path
    admin_fd: int
    common_path: Path
    common_fd: int
    commondir_fd: int | None
    commondir_identity: tuple[int, ...] | None
    commondir_payload: bytes | None
    backlink_fd: int | None
    backlink_identity: tuple[int, ...] | None
    backlink_payload: bytes | None
    index_fd: int
    index_identity: tuple[int, ...]
    index_payload: bytes
    executable_parent_fd: int
    executable_fd: int
    executable_identity: tuple[int, ...]

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return tuple(sorted({
            self.root_fd,
            self.admin_fd,
            self.common_fd,
            self.index_fd,
            self.executable_fd,
        }))

    def index_snapshot(self) -> dict[str, Any]:
        return {
            "path": str(self.admin_path / "index"),
            "bytes": len(self.index_payload),
            "sha256": hashlib.sha256(self.index_payload).hexdigest(),
            "metadata": list(self.index_identity),
        }

    def verify(self) -> None:
        _require_same_open_directory(self.root, self.root_fd, self.label)
        try:
            named_dot_git = os.stat(
                ".git", dir_fd=self.root_fd, follow_symlinks=False
            )
        except OSError as error:
            raise CameraCropTerminalError(
                "Git administrative path changed"
            ) from error
        require(
            _stat_identity(named_dot_git) == self.dot_git_identity,
            "Git administrative path changed",
        )
        if self.dot_git_kind == "pointer":
            assert self.dot_git_payload is not None
            _verify_held_regular_at(
                self.root_fd,
                ".git",
                self.dot_git_fd,
                self.dot_git_payload,
                self.dot_git_identity,
                "Git administrative pointer",
                maximum_bytes=GIT_POINTER_LIMIT_BYTES,
            )
        else:
            require(
                _stat_identity(os.fstat(self.dot_git_fd))
                == self.dot_git_identity,
                "Git administrative directory changed",
            )
        _require_same_open_directory(
            self.admin_path, self.admin_fd, "Git administrative directory"
        )
        _require_same_open_directory(
            self.common_path, self.common_fd, "Git common directory"
        )
        if self.commondir_fd is not None:
            assert self.commondir_identity is not None
            assert self.commondir_payload is not None
            _verify_held_regular_at(
                self.admin_fd,
                "commondir",
                self.commondir_fd,
                self.commondir_payload,
                self.commondir_identity,
                "Git common-directory pointer",
                maximum_bytes=GIT_POINTER_LIMIT_BYTES,
            )
        if self.backlink_fd is not None:
            assert self.backlink_identity is not None
            assert self.backlink_payload is not None
            _verify_held_regular_at(
                self.admin_fd,
                "gitdir",
                self.backlink_fd,
                self.backlink_payload,
                self.backlink_identity,
                "Git worktree backlink",
                maximum_bytes=GIT_POINTER_LIMIT_BYTES,
            )
        _verify_held_regular_at(
            self.admin_fd,
            "index",
            self.index_fd,
            self.index_payload,
            self.index_identity,
            "Git index",
            maximum_bytes=GIT_INDEX_LIMIT_BYTES,
        )
        _require_same_open_directory(
            GIT_EXECUTABLE.parent,
            self.executable_parent_fd,
            "trusted Git executable directory",
        )
        try:
            named_executable = os.stat(
                GIT_EXECUTABLE.name,
                dir_fd=self.executable_parent_fd,
                follow_symlinks=False,
            )
            opened_executable = os.fstat(self.executable_fd)
        except OSError as error:
            raise CameraCropTerminalError(
                "trusted Git executable identity changed"
            ) from error
        require(
            _stat_identity(named_executable)
            == self.executable_identity
            == _stat_identity(opened_executable),
            "trusted Git executable identity changed",
        )

    def close(self) -> None:
        for descriptor in {
            self.root_fd,
            self.dot_git_fd,
            self.admin_fd,
            self.common_fd,
            self.commondir_fd,
            self.backlink_fd,
            self.index_fd,
            self.executable_parent_fd,
            self.executable_fd,
        }:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _open_git_checkout(root: Path, label: str) -> _HeldGitCheckout:
    root_fd, root = _open_directory_nofollow(root, label)
    opened: list[int] = [root_fd]
    try:
        try:
            dot_git = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        except OSError as error:
            raise CameraCropTerminalError(
                "Git administrative path is unavailable"
            ) from error
        dot_git_payload: bytes | None = None
        if stat.S_ISDIR(dot_git.st_mode):
            try:
                dot_git_fd = os.open(".git", _directory_flags(), dir_fd=root_fd)
            except OSError as error:
                raise CameraCropTerminalError(
                    "Git administrative directory is unavailable"
                ) from error
            opened.append(dot_git_fd)
            require(
                _stat_identity(os.fstat(dot_git_fd)) == _stat_identity(dot_git),
                "Git administrative directory changed while it was opened",
            )
            dot_git_kind = "directory"
            admin_path = root / ".git"
            admin_fd = dot_git_fd
        elif stat.S_ISREG(dot_git.st_mode):
            dot_git_fd, dot_git_payload, dot_git_identity = _open_held_regular_at(
                root_fd,
                ".git",
                "Git administrative pointer",
                maximum_bytes=GIT_POINTER_LIMIT_BYTES,
            )
            opened.append(dot_git_fd)
            require(
                dot_git_identity == _stat_identity(dot_git),
                "Git administrative pointer changed while it was opened",
            )
            dot_git_kind = "pointer"
            admin_path = _git_directory_pointer(dot_git_payload, root)
            admin_fd, admin_path = _open_directory_nofollow(
                admin_path, "Git administrative directory"
            )
            opened.append(admin_fd)
        else:
            raise CameraCropTerminalError(
                "Git administrative path is not a regular pointer or directory"
            )

        commondir_fd: int | None = None
        commondir_identity: tuple[int, ...] | None = None
        commondir_payload: bytes | None = None
        backlink_fd: int | None = None
        backlink_identity: tuple[int, ...] | None = None
        backlink_payload: bytes | None = None
        try:
            commondir_stat = os.stat(
                "commondir", dir_fd=admin_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            commondir_stat = None
        except OSError as error:
            raise CameraCropTerminalError(
                "Git common-directory pointer is unavailable"
            ) from error
        if commondir_stat is None:
            require(
                dot_git_kind == "directory",
                "linked Git worktree is missing its common-directory pointer",
            )
            common_path = admin_path
            common_fd = admin_fd
        else:
            require(
                stat.S_ISREG(commondir_stat.st_mode),
                "Git common-directory pointer is not regular",
            )
            (
                commondir_fd,
                commondir_payload,
                commondir_identity,
            ) = _open_held_regular_at(
                admin_fd,
                "commondir",
                "Git common-directory pointer",
                maximum_bytes=GIT_POINTER_LIMIT_BYTES,
            )
            opened.append(commondir_fd)
            common_path = _one_line_path(
                commondir_payload, admin_path, "Git common directory"
            )
            common_fd, common_path = _open_directory_nofollow(
                common_path, "Git common directory"
            )
            opened.append(common_fd)
            try:
                admin_relative = admin_path.relative_to(common_path)
            except ValueError as error:
                raise CameraCropTerminalError(
                    "Git worktree administrative directory escaped its common directory"
                ) from error
            require(
                len(admin_relative.parts) == 2
                and admin_relative.parts[0] == "worktrees"
                and admin_relative.parts[1] not in {"", ".", ".."},
                "Git worktree administrative directory shape changed",
            )
            (
                backlink_fd,
                backlink_payload,
                backlink_identity,
            ) = _open_held_regular_at(
                admin_fd,
                "gitdir",
                "Git worktree backlink",
                maximum_bytes=GIT_POINTER_LIMIT_BYTES,
            )
            opened.append(backlink_fd)
            require(
                _one_line_path(
                    backlink_payload, admin_path, "Git worktree backlink target"
                )
                == root / ".git",
                "Git worktree backlink does not name the supplied checkout",
            )

        index_fd, index_payload, index_identity = _open_held_regular_at(
            admin_fd,
            "index",
            "Git index",
            maximum_bytes=GIT_INDEX_LIMIT_BYTES,
        )
        opened.append(index_fd)

        require(
            GIT_EXECUTABLE == Path("/usr/bin/git"),
            "trusted Git executable path changed",
        )
        _require_trusted_git_directory(GIT_EXECUTABLE.parent)
        executable_parent_fd, _ = _open_directory_nofollow(
            GIT_EXECUTABLE.parent, "trusted Git executable directory"
        )
        opened.append(executable_parent_fd)
        try:
            named_executable = os.stat(
                GIT_EXECUTABLE.name,
                dir_fd=executable_parent_fd,
                follow_symlinks=False,
            )
            executable_fd = os.open(
                GIT_EXECUTABLE.name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=executable_parent_fd,
            )
        except OSError as error:
            raise CameraCropTerminalError(
                "trusted Git executable is unavailable"
            ) from error
        opened.append(executable_fd)
        opened_executable = os.fstat(executable_fd)
        require(
            stat.S_ISREG(opened_executable.st_mode)
            and opened_executable.st_uid == 0
            and opened_executable.st_mode & 0o022 == 0
            and opened_executable.st_mode & 0o111 != 0
            and _stat_identity(opened_executable) == _stat_identity(named_executable),
            "trusted Git executable is not a protected root-owned regular file",
        )
        checkout = _HeldGitCheckout(
            label=label,
            root=root,
            root_fd=root_fd,
            dot_git_kind=dot_git_kind,
            dot_git_fd=dot_git_fd,
            dot_git_identity=_stat_identity(dot_git),
            dot_git_payload=dot_git_payload,
            admin_path=admin_path,
            admin_fd=admin_fd,
            common_path=common_path,
            common_fd=common_fd,
            commondir_fd=commondir_fd,
            commondir_identity=commondir_identity,
            commondir_payload=commondir_payload,
            backlink_fd=backlink_fd,
            backlink_identity=backlink_identity,
            backlink_payload=backlink_payload,
            index_fd=index_fd,
            index_identity=index_identity,
            index_payload=index_payload,
            executable_parent_fd=executable_parent_fd,
            executable_fd=executable_fd,
            executable_identity=_stat_identity(opened_executable),
        )
        opened.clear()
        checkout.verify()
        return checkout
    except BaseException:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


@contextmanager
def _held_git_checkout(root: Path, label: str) -> Iterator[_HeldGitCheckout]:
    checkout = _open_git_checkout(root, label)
    try:
        shared_index = _run_git(checkout, "rev-parse", "--shared-index-path")
        assert isinstance(shared_index, str)
        require(not shared_index.strip(), "split Git indexes are not supported")
        yield checkout
    finally:
        try:
            checkout.verify()
        finally:
            checkout.close()


def _safe_git_environment(checkout: _HeldGitCheckout) -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GCM_INTERACTIVE": "never",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_EXEC_PATH": "/nonexistent",
        "GIT_CEILING_DIRECTORIES": "/",
        "GIT_COMMON_DIR": f"/proc/self/fd/{checkout.common_fd}",
        "GIT_INDEX_FILE": f"/proc/self/fd/{checkout.index_fd}",
    }


def _run_git(
    checkout: _HeldGitCheckout, *arguments: str, binary: bool = False
) -> bytes | str:
    require(bool(arguments), "Git provenance command is empty")
    require(
        arguments[0] in {"rev-parse", "show", "ls-files", "ls-tree"},
        "Git provenance command is not allowlisted",
    )
    checkout.verify()
    command = [
        f"/proc/self/fd/{checkout.executable_fd}",
        "--no-pager",
        "--no-replace-objects",
        f"--git-dir=/proc/self/fd/{checkout.admin_fd}",
        f"--work-tree=/proc/self/fd/{checkout.root_fd}",
        "-c", "credential.helper=",
        "-c", "credential.interactive=never",
        "-c", "core.askPass=/bin/false",
        "-c", "core.sshCommand=/bin/false",
        "-c", "core.fsmonitor=false",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.attributesFile=/dev/null",
        "-c", "diff.external=",
        "-c", "protocol.allow=never",
        "-c", "submodule.recurse=false",
        "-c", "fetch.recurseSubmodules=false",
        "-c", "gc.auto=0",
        "-c", "maintenance.auto=false",
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            env=_safe_git_environment(checkout),
            pass_fds=checkout.pass_fds,
            close_fds=True,
            cwd=os.sep,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CameraCropTerminalError(
            "Git provenance verification is unavailable"
        ) from error
    finally:
        checkout.verify()
    require(
        result.returncode == 0,
        f"Git provenance verification failed: {arguments[0]}",
    )
    if binary:
        return result.stdout
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CameraCropTerminalError("Git provenance output is not UTF-8") from error


def _git_index_snapshot(root: Path) -> dict[str, Any]:
    with _held_git_checkout(root, "Git checkout") as checkout:
        return checkout.index_snapshot()


def _require_git_index_unchanged(root: Path, before: Mapping[str, Any]) -> None:
    after = _git_index_snapshot(root)
    require(
        _exact_json_equal(after, before),
        "Git index bytes or metadata changed during read-only validation",
    )


def _safe_checkout_root(path: Path, label: str) -> Path:
    descriptor, root = _open_directory_nofollow(path, label)
    try:
        require(stat.S_ISDIR(os.fstat(descriptor).st_mode),
                f"{label} is not a directory")
    finally:
        os.close(descriptor)
    return root


def _git_blob(
    checkout: _HeldGitCheckout, commit: str, relative: Path, label: str
) -> bytes:
    require(
        not relative.is_absolute()
        and bool(relative.parts)
        and "." not in relative.parts
        and ".." not in relative.parts,
        f"{label} Git path is unsafe",
    )
    payload = _run_git(
        checkout, "show", f"{commit}:{relative.as_posix()}", binary=True
    )
    assert isinstance(payload, bytes)
    return payload


def _nul_git_records(payload: bytes, label: str) -> list[bytes]:
    require(len(payload) <= GIT_LIST_LIMIT_BYTES, f"{label} output is too large")
    if not payload:
        return []
    require(payload.endswith(b"\0"), f"{label} output is malformed")
    records = payload[:-1].split(b"\0")
    require(
        len(records) <= GIT_TRACKED_ENTRY_LIMIT and all(records),
        f"{label} output is malformed",
    )
    return records


def _git_relative_from_output(payload: bytes, label: str) -> Path:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CameraCropTerminalError(f"{label} path is not UTF-8") from error
    relative = Path(text)
    require(
        bool(text)
        and "\x00" not in text
        and not relative.is_absolute()
        and str(relative) == text
        and "." not in relative.parts
        and ".." not in relative.parts,
        f"{label} path is unsafe",
    )
    return relative


def _parse_index_inventory(payload: bytes) -> dict[Path, tuple[str, str]]:
    result: dict[Path, tuple[str, str]] = {}
    for record in _nul_git_records(payload, "Git index inventory"):
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = header.split(b" ")
            mode_text = mode.decode("ascii")
            object_text = object_id.decode("ascii")
        except (ValueError, UnicodeDecodeError) as error:
            raise CameraCropTerminalError("Git index inventory is malformed") from error
        require(stage == b"0", "Git index contains an unresolved entry")
        require(
            mode_text in {"100644", "100755", "120000"}
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_text)
            is not None,
            "Git index entry type or object identity is unsupported",
        )
        path = _git_relative_from_output(raw_path, "Git index")
        require(path not in result, "Git index contains a duplicate path")
        result[path] = (mode_text, object_text)
    return result


def _parse_tree_inventory(payload: bytes) -> dict[Path, tuple[str, str]]:
    result: dict[Path, tuple[str, str]] = {}
    for record in _nul_git_records(payload, "Git commit tree inventory"):
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = header.split(b" ")
            mode_text = mode.decode("ascii")
            type_text = object_type.decode("ascii")
            object_text = object_id.decode("ascii")
        except (ValueError, UnicodeDecodeError) as error:
            raise CameraCropTerminalError(
                "Git commit tree inventory is malformed"
            ) from error
        require(
            mode_text in {"100644", "100755", "120000"}
            and type_text == "blob"
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_text)
            is not None,
            "Git commit tree entry type or object identity is unsupported",
        )
        path = _git_relative_from_output(raw_path, "Git commit tree")
        require(path not in result, "Git commit tree contains a duplicate path")
        result[path] = (mode_text, object_text)
    return result


def _read_checkout_entry(
    checkout: _HeldGitCheckout, relative: Path, mode: str, label: str
) -> bytes:
    require(
        not relative.is_absolute()
        and bool(relative.parts)
        and "." not in relative.parts
        and ".." not in relative.parts,
        f"{label} path is unsafe",
    )
    parent_fd = os.dup(checkout.root_fd)
    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=parent_fd)
            except OSError as error:
                raise CameraCropTerminalError(
                    f"{label} parent is missing, not a directory, or is a symlink"
                ) from error
            os.close(parent_fd)
            parent_fd = child
        name = relative.parts[-1]
        if mode == "120000":
            try:
                before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                target = os.readlink(name, dir_fd=parent_fd)
                after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                target_after = os.readlink(name, dir_fd=parent_fd)
            except OSError as error:
                raise CameraCropTerminalError(
                    f"{label} symlink is unavailable"
                ) from error
            require(
                stat.S_ISLNK(before.st_mode)
                and before.st_nlink == 1
                and _stat_identity(before) == _stat_identity(after)
                and target == target_after,
                f"{label} symlink changed while it was read",
            )
            return os.fsencode(target)
        descriptor, payload, identity = _open_held_regular_at(
            parent_fd,
            name,
            label,
            maximum_bytes=GIT_TRACKED_FILE_LIMIT_BYTES,
        )
        metadata = os.fstat(descriptor)
        require(
            (mode == "100755") == bool(metadata.st_mode & stat.S_IXUSR),
            f"{label} executable mode differs from the Git index",
        )
        _verify_held_regular_at(
            parent_fd,
            name,
            descriptor,
            payload,
            identity,
            label,
            maximum_bytes=GIT_TRACKED_FILE_LIMIT_BYTES,
        )
        return payload
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _checkout_parent_fd(
    checkout: _HeldGitCheckout, relative: Path, label: str
) -> tuple[int, str, Path]:
    """Hold the parent of a relative checkout path without named-root lookup."""

    candidate = Path(relative)
    require(
        not candidate.is_absolute()
        and bool(candidate.parts)
        and "." not in candidate.parts
        and ".." not in candidate.parts,
        f"{label} path is unsafe",
    )
    try:
        parent_fd = os.dup(checkout.root_fd)
    except OSError as error:
        raise CameraCropTerminalError(
            f"cannot hold checkout root for {label}"
        ) from error
    try:
        for component in candidate.parts[:-1]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=parent_fd)
            except OSError as error:
                raise CameraCropTerminalError(
                    f"{label} parent is missing, not a directory, or is a symlink"
                ) from error
            os.close(parent_fd)
            parent_fd = child
        return parent_fd, candidate.parts[-1], candidate
    except BaseException:
        os.close(parent_fd)
        raise


def _read_checkout_regular_file_once(
    checkout: _HeldGitCheckout,
    relative: Path,
    label: str,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> tuple[bytes, dict[str, Any], tuple[int, ...]]:
    """Read one reserved blob through the already-held checkout root."""

    require(type(maximum_bytes) is int and maximum_bytes > 0,
            f"{label} byte limit is invalid")
    parent_fd, name, relative = _checkout_parent_fd(checkout, relative, label)
    descriptor: int | None = None
    try:
        try:
            named_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(
                name,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise CameraCropTerminalError(
                f"{label} is missing, not regular, or is a symlink"
            ) from error
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(named_before.st_mode)
            and stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and _stat_identity(named_before) == _stat_identity(before),
            f"{label} is not the expected singly linked regular file",
        )
        require(0 <= before.st_size <= maximum_bytes,
                f"{label} exceeds the byte limit")
        if expected_bytes is not None:
            require(
                type(expected_bytes) is int
                and expected_bytes >= 0
                and before.st_size == expected_bytes,
                f"{label} byte count changed before read",
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            try:
                block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            except InterruptedError:
                continue
            except OSError as error:
                raise CameraCropTerminalError(f"cannot read {label}") from error
            if not block:
                break
            observed += len(block)
            require(observed <= maximum_bytes, f"{label} exceeds the byte limit")
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        try:
            named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise CameraCropTerminalError(f"{label} path changed after read") from error
        require(
            _stat_identity(before) == _stat_identity(after)
            == _stat_identity(named_after)
            and after.st_nlink == 1
            and len(payload) == after.st_size,
            f"{label} changed while it was read",
        )
        return payload, {
            "path": str(checkout.root / relative),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }, _stat_identity(after)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _require_checkout_regular_file_identity(
    checkout: _HeldGitCheckout,
    relative: Path,
    expected_identity: tuple[int, ...],
    label: str,
) -> None:
    """Recheck reserved metadata from the held root without reopening bytes."""

    parent_fd, name, _relative = _checkout_parent_fd(checkout, relative, label)
    try:
        try:
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise CameraCropTerminalError(
                f"{label} path changed after authentication"
            ) from error
        require(
            stat.S_ISREG(named.st_mode)
            and named.st_nlink == 1
            and _stat_identity(named) == expected_identity,
            f"{label} path changed after authentication",
        )
    finally:
        os.close(parent_fd)


def _git_blob_digest(payload: bytes, object_format: str) -> str:
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    if object_format == "sha1":
        return hashlib.sha1(framed).hexdigest()
    require(object_format == "sha256", "Git object format is unsupported")
    return hashlib.sha256(framed).hexdigest()


def _require_clean_checkout(
    checkout: _HeldGitCheckout,
    expected_commit: str,
    label: str,
    *,
    single_read_paths: Sequence[Path] = (),
) -> None:
    """Prove a checkout clean while reserving named blobs for one held read.

    ``single_read_paths`` remain fully bound by the exact index/commit-tree
    comparison here. Their worktree bytes are authenticated separately by the
    caller's one held read plus an exact Git-blob comparison, so this scan must
    not reopen them.
    """

    commit = queue._verified_commit(expected_commit)
    head_before = _run_git(checkout, "rev-parse", "--verify", "HEAD^{commit}")
    object_format = _run_git(checkout, "rev-parse", "--show-object-format")
    raw_index = _run_git(checkout, "ls-files", "--stage", "-z", binary=True)
    raw_tree = _run_git(
        checkout, "ls-tree", "-r", "-z", "--full-tree", commit, binary=True
    )
    assert isinstance(head_before, str) and isinstance(object_format, str)
    assert isinstance(raw_index, bytes) and isinstance(raw_tree, bytes)
    index = _parse_index_inventory(raw_index)
    tree = _parse_tree_inventory(raw_tree)
    reserved: set[Path] = set()
    for supplied in single_read_paths:
        relative = Path(supplied)
        require(
            not relative.is_absolute()
            and bool(relative.parts)
            and "." not in relative.parts
            and ".." not in relative.parts
            and relative not in reserved,
            f"{label} single-read path inventory is invalid",
        )
        reserved.add(relative)
    require(
        head_before.strip() == commit
        and _exact_json_equal(index, tree)
        and reserved <= set(index),
        f"{label} is not clean at the exact supplied commit",
    )
    require(
        all(index[path][0] == "100644" for path in reserved),
        f"{label} single-read path is not a nonexecutable regular file; "
        "symlinks are forbidden",
    )
    format_name = object_format.strip()
    total_bytes = 0
    for relative, (mode, object_id) in index.items():
        if relative in reserved:
            continue
        payload = _read_checkout_entry(
            checkout, relative, mode, f"{label} tracked file {relative}"
        )
        total_bytes += len(payload)
        require(
            total_bytes <= GIT_TRACKED_TOTAL_LIMIT_BYTES
            and _git_blob_digest(payload, format_name) == object_id,
            f"{label} tracked bytes differ from the exact supplied commit",
        )
    raw_untracked = _run_git(
        checkout,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        binary=True,
    )
    head_after = _run_git(checkout, "rev-parse", "--verify", "HEAD^{commit}")
    assert isinstance(raw_untracked, bytes) and isinstance(head_after, str)
    require(
        not _nul_git_records(raw_untracked, "Git untracked inventory")
        and head_after.strip() == commit,
        f"{label} is not clean at the exact supplied commit",
    )


def _validate_prior_source_commit_in_checkout(
    checkout: _HeldGitCheckout, commit: str
) -> dict[str, dict[str, Any]]:
    commit = queue._verified_commit(commit)
    try:
        resolved = _run_git(
            checkout, "rev-parse", "--verify", f"{commit}^{{commit}}"
        )
    except CameraCropTerminalError as error:
        raise CameraCropTerminalError(
            "attempt 002 study_commit is not a commit object"
        ) from error
    assert isinstance(resolved, str)
    require(
        resolved.strip() == commit,
        "attempt 002 study_commit did not resolve to the exact supplied commit",
    )
    result: dict[str, dict[str, Any]] = {}
    for path, expected_digest in PRIOR_SOURCE_PINS.items():
        payload = _git_blob(checkout, commit, path, "attempt 002 source")
        digest = queue.sha256_bytes(payload)
        require(digest == expected_digest,
                f"attempt 002 source blob changed: {path}")
        result[str(path)] = {"sha256": digest, "bytes": len(payload)}
    return result


def _validate_prior_source_commit(root: Path, commit: str) -> dict[str, dict[str, Any]]:
    root = _safe_checkout_root(root, "source checkout")
    with _held_git_checkout(root, "source checkout") as checkout:
        return _validate_prior_source_commit_in_checkout(checkout, commit)


def _authenticate_runtime_contract_values_in_checkout(
    *,
    checkout: _HeldGitCheckout,
    control_commit: str,
    terminal_runtime: Mapping[str, Any],
    replay_runtime: Mapping[str, Any],
) -> None:
    commit = queue._verified_commit(control_commit)
    for label, relative, parsed in (
        (
            "terminal runtime contract",
            TERMINAL_CONTRACT_RELATIVE,
            terminal_runtime,
        ),
        (
            "replay runtime contract",
            REPLAY_CONTRACT_RELATIVE,
            replay_runtime,
        ),
    ):
        require(isinstance(parsed, Mapping), f"{label} parsed value changed")
        committed = _parse_json_bytes(
            _git_blob(checkout, commit, relative, label),
            f"committed {label}",
        )
        require(
            _exact_json_equal(parsed, committed),
            f"{label} semantics differ from the control commit",
        )


def _authenticate_control_state_in_checkout(
    *,
    checkout: _HeldGitCheckout,
    control_commit: str,
    queue_path: Path,
    implementation: Mapping[str, Any],
    terminal_runtime: Mapping[str, Any] | None = None,
    replay_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate one clean control checkout and its exact tracked queue bytes."""

    commit = queue._verified_commit(control_commit)
    root = checkout.root
    expected_queue_path = root / CONTROL_QUEUE_RELATIVE
    supplied_queue_path = _lexical_absolute_path(queue_path, "control queue")
    require(
        supplied_queue_path == expected_queue_path,
        "control queue is not the exact tracked cluster_queue.json path",
    )
    require(
        isinstance(implementation, Mapping)
        and set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
        "terminal wave implementation inventory changed",
    )
    require(
        (terminal_runtime is None) == (replay_runtime is None),
        "control runtime contract authentication is incomplete",
    )
    if terminal_runtime is not None and replay_runtime is not None:
        _authenticate_runtime_contract_values_in_checkout(
            checkout=checkout,
            control_commit=commit,
            terminal_runtime=terminal_runtime,
            replay_runtime=replay_runtime,
        )
    _require_clean_checkout(
        checkout,
        commit,
        "control checkout",
        single_read_paths=(CONTROL_QUEUE_RELATIVE,),
    )
    queue_payload, _identity, queue_metadata = _read_checkout_regular_file_once(
        checkout,
        CONTROL_QUEUE_RELATIVE,
        "tracked control queue",
        maximum_bytes=AUTHENTICATED_JSON_LIMIT_BYTES,
    )
    require(
        queue_metadata[2] & stat.S_IXUSR == 0,
        "tracked control queue executable mode differs from the Git index",
    )
    require(
        queue_payload
        == _git_blob(checkout, commit, CONTROL_QUEUE_RELATIVE, "control queue"),
        "control queue bytes differ from the exact control Git object",
    )
    for name, local_path in _source_paths(REPOSITORY_ROOT).items():
        descriptor = _descriptor_shape(
            implementation[name], f"terminal {name} implementation"
        )
        try:
            relative = local_path.relative_to(REPOSITORY_ROOT)
        except ValueError as error:
            raise CameraCropTerminalError(
                f"terminal {name} implementation path escaped the repository"
            ) from error
        blob = _git_blob(
            checkout, commit, relative, f"terminal {name} implementation"
        )
        require(
            descriptor["sha256"] == hashlib.sha256(blob).hexdigest()
            and descriptor["bytes"] == len(blob),
            f"terminal {name} implementation differs from the control commit",
        )
    _require_clean_checkout(
        checkout,
        commit,
        "control checkout",
        single_read_paths=(CONTROL_QUEUE_RELATIVE,),
    )
    _require_checkout_regular_file_identity(
        checkout,
        CONTROL_QUEUE_RELATIVE,
        queue_metadata,
        "tracked control queue",
    )
    return _parse_json_bytes(queue_payload, "tracked control queue")


def _authenticate_control_state(
    *,
    control_root: Path,
    control_commit: str,
    queue_path: Path,
    implementation: Mapping[str, Any],
    terminal_runtime: Mapping[str, Any] | None = None,
    replay_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _safe_checkout_root(control_root, "control checkout")
    with _held_git_checkout(root, "control checkout") as checkout:
        return _authenticate_control_state_in_checkout(
            checkout=checkout,
            control_commit=control_commit,
            queue_path=queue_path,
            implementation=implementation,
            terminal_runtime=terminal_runtime,
            replay_runtime=replay_runtime,
        )


def _authenticate_results_claims_in_checkouts(
    *,
    results_checkout: _HeldGitCheckout,
    results_commit: str,
    prerequisite_inputs: Mapping[str, tuple[Path, str]],
    prior_diagnostic_path: Path,
    prior_diagnostic_sha256: str,
    expected_prerequisites: Mapping[str, Any],
    source_checkout: _HeldGitCheckout,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Authenticate all six gates from one held read of each exact file."""

    commit = queue._verified_commit(results_commit)
    root = results_checkout.root
    require(
        isinstance(prerequisite_inputs, Mapping)
        and set(PREREQUISITE_RESULTS_RELATIVES) == set(PREREQUISITE_CLUSTER_PATHS)
        == set(expected_prerequisites) == set(prerequisite_inputs),
        "results prerequisite inventory changed",
    )
    authenticated: dict[str, dict[str, Any]] = {}
    single_read_relatives = tuple(PREREQUISITE_RESULTS_RELATIVES.values()) + (
        PRIOR_DIAGNOSTIC_RESULTS_RELATIVE,
    )
    _require_clean_checkout(
        results_checkout,
        commit,
        "results checkout",
        single_read_paths=single_read_relatives,
    )
    authenticated_metadata: dict[Path, tuple[int, ...]] = {}
    for name in sorted(expected_prerequisites):
        wanted = expected_prerequisites[name]
        claim = prerequisite_inputs[name]
        require(
            isinstance(claim, tuple) and len(claim) == 2,
            f"terminal {name} prerequisite claim changed",
        )
        supplied_path, supplied_sha256 = claim
        expected_path = root / PREREQUISITE_RESULTS_RELATIVES[name]
        require(
            _lexical_absolute_path(
                Path(supplied_path), f"terminal {name} prerequisite"
            ) == expected_path,
            f"terminal {name} prerequisite is not the exact results artifact",
        )
        supplied_digest = queue._verified_sha(
            supplied_sha256, f"terminal {name} prerequisite digest"
        )
        require(
            supplied_digest == wanted["sha256"],
            f"terminal {name} prerequisite supplied SHA-256 changed",
        )
        payload, identity, metadata = _read_checkout_regular_file_once(
            results_checkout,
            PREREQUISITE_RESULTS_RELATIVES[name],
            f"fetched prerequisite {name}",
            maximum_bytes=AUTHENTICATED_JSON_LIMIT_BYTES,
            expected_bytes=wanted["bytes"],
        )
        require(
            metadata[2] & stat.S_IXUSR == 0,
            f"terminal {name} prerequisite executable mode differs from the Git index",
        )
        require(
            identity["sha256"] == wanted["sha256"],
            f"terminal {name} prerequisite identity changed",
        )
        require(
            payload == _git_blob(
                results_checkout,
                commit,
                PREREQUISITE_RESULTS_RELATIVES[name],
                f"fetched prerequisite {name}",
            ),
            f"terminal {name} prerequisite differs from its results Git object",
        )
        value = _parse_json_bytes(payload, f"fetched prerequisite {name}")
        diagnostic._validate_prerequisite_value(name, value, wanted)
        authenticated[name] = identity
        authenticated_metadata[PREREQUISITE_RESULTS_RELATIVES[name]] = metadata

    expected_prior_path = root / PRIOR_DIAGNOSTIC_RESULTS_RELATIVE
    require(
        _lexical_absolute_path(
            prior_diagnostic_path, "terminal attempt 002 receipt"
        ) == expected_prior_path,
        "terminal attempt 002 receipt is not the exact results artifact",
    )
    supplied_prior_digest = queue._verified_sha(
        prior_diagnostic_sha256, "terminal attempt 002 receipt digest"
    )
    payload, identity, metadata = _read_checkout_regular_file_once(
        results_checkout,
        PRIOR_DIAGNOSTIC_RESULTS_RELATIVE,
        "fetched attempt 002 diagnostic receipt",
        maximum_bytes=AUTHENTICATED_JSON_LIMIT_BYTES,
    )
    require(
        metadata[2] & stat.S_IXUSR == 0,
        "terminal attempt 002 receipt executable mode differs from the Git index",
    )
    require(
        identity["sha256"] == supplied_prior_digest,
        "terminal wave attempt 002 diagnostic identity changed",
    )
    require(
        payload == _git_blob(
            results_checkout,
            commit,
            PRIOR_DIAGNOSTIC_RESULTS_RELATIVE,
            "attempt 002 receipt",
        ),
        "attempt 002 receipt differs from its results Git object",
    )
    value = _parse_json_bytes(payload, "fetched attempt 002 diagnostic receipt")
    study_commit = _validate_prior_diagnostic_value(value)
    source_blobs = _validate_prior_source_commit_in_checkout(
        source_checkout, study_commit
    )
    authenticated_prior = {
        "receipt": identity,
        "results_commit": commit,
        "study_commit": study_commit,
        "authenticated_source_blobs": source_blobs,
    }
    authenticated_metadata[PRIOR_DIAGNOSTIC_RESULTS_RELATIVE] = metadata
    _require_clean_checkout(
        results_checkout,
        commit,
        "results checkout",
        single_read_paths=single_read_relatives,
    )
    for relative, observed_metadata in authenticated_metadata.items():
        _require_checkout_regular_file_identity(
            results_checkout,
            relative,
            observed_metadata,
            f"authenticated results artifact {relative}",
        )
    return authenticated, authenticated_prior


def _authenticate_results_claims(
    *,
    results_root: Path,
    results_commit: str,
    prerequisite_inputs: Mapping[str, tuple[Path, str]],
    prior_diagnostic_path: Path,
    prior_diagnostic_sha256: str,
    expected_prerequisites: Mapping[str, Any],
    source_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results_root = _safe_checkout_root(results_root, "results checkout")
    source_root = _safe_checkout_root(source_root, "source checkout")
    with _held_git_checkout(
        results_root, "results checkout"
    ) as results_checkout, _held_git_checkout(
        source_root, "source checkout"
    ) as source_checkout:
        return _authenticate_results_claims_in_checkouts(
            results_checkout=results_checkout,
            results_commit=results_commit,
            prerequisite_inputs=prerequisite_inputs,
            prior_diagnostic_path=prior_diagnostic_path,
            prior_diagnostic_sha256=prior_diagnostic_sha256,
            expected_prerequisites=expected_prerequisites,
            source_checkout=source_checkout,
        )


def _authenticate_results_inputs(
    *,
    results_root: Path,
    results_commit: str,
    prerequisites: Mapping[str, Any],
    prior: Mapping[str, Any],
    expected_prerequisites: Mapping[str, Any],
    source_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reauthenticate a shaped wave through the exact six held-file reads."""

    require(
        isinstance(prerequisites, Mapping)
        and set(prerequisites) == set(expected_prerequisites),
        "results prerequisite inventory changed",
    )
    prerequisite_inputs: dict[str, tuple[Path, str]] = {}
    for name in sorted(prerequisites):
        descriptor = _descriptor_shape(
            prerequisites[name], f"terminal {name} prerequisite"
        )
        prerequisite_inputs[name] = (
            Path(descriptor["path"]), descriptor["sha256"]
        )
    prior_receipt = _descriptor_shape(
        prior.get("receipt"), "terminal attempt 002 receipt"
    )
    authenticated, authenticated_prior = _authenticate_results_claims(
        results_root=results_root,
        results_commit=results_commit,
        prerequisite_inputs=prerequisite_inputs,
        prior_diagnostic_path=Path(prior_receipt["path"]),
        prior_diagnostic_sha256=prior_receipt["sha256"],
        expected_prerequisites=expected_prerequisites,
        source_root=source_root,
    )
    require(
        _exact_json_equal(prerequisites, authenticated),
        "terminal wave replay prerequisite identities changed",
    )
    require(
        _exact_json_equal(prior, authenticated_prior),
        "terminal wave attempt 002 diagnostic identity changed",
    )
    return authenticated, authenticated_prior


def _validate_results_checkout(path: Path, expected_commit: str) -> Path:
    root = _safe_checkout_root(path, "results checkout")
    with _held_git_checkout(root, "results checkout") as checkout:
        _require_clean_checkout(checkout, expected_commit, "results checkout")
        return root


def validate_prior_diagnostic(
    *,
    path: Path,
    supplied_sha256: str,
    results_root: Path,
    results_commit: str,
) -> dict[str, Any]:
    root = _safe_checkout_root(results_root, "results checkout")
    commit = queue._verified_commit(results_commit)
    source_root = _safe_checkout_root(REPOSITORY_ROOT, "source checkout")
    with _held_git_checkout(
        root, "results checkout"
    ) as results_checkout, _held_git_checkout(
        source_root, "source checkout"
    ) as source_checkout:
        _require_clean_checkout(results_checkout, commit, "results checkout")
        expected_path = root / PRIOR_DIAGNOSTIC_RESULTS_RELATIVE
        require(_lexical_absolute_path(path, "attempt 002 receipt") == expected_path,
                "attempt 002 receipt is not the exact fetched-results artifact")
        digest = queue._verified_sha(supplied_sha256, "attempt 002 receipt digest")
        payload, identity, _metadata = _read_regular_file_once(
            expected_path,
            "fetched attempt 002 diagnostic receipt",
            maximum_bytes=AUTHENTICATED_JSON_LIMIT_BYTES,
        )
        require(identity["sha256"] == digest, "attempt 002 receipt digest changed")
        require(
            payload == _git_blob(
                results_checkout, commit, PRIOR_DIAGNOSTIC_RESULTS_RELATIVE,
                "attempt 002 receipt",
            ),
            "attempt 002 receipt differs from its results Git object",
        )
        value = _parse_json_bytes(payload, "fetched attempt 002 diagnostic receipt")
        study_commit = _validate_prior_diagnostic_value(value)
        source_blobs = _validate_prior_source_commit_in_checkout(
            source_checkout, study_commit
        )
        _require_clean_checkout(results_checkout, commit, "results checkout")
        return {
            "receipt": identity,
            "results_commit": commit,
            "study_commit": study_commit,
            "authenticated_source_blobs": source_blobs,
        }


def _hash_argv(implementation: Mapping[str, Mapping[str, Any]]) -> list[str]:
    arguments: list[str] = []
    for name in sorted(implementation):
        arguments.extend(["--implementation", name, str(implementation[name]["sha256"])])
    return arguments


def _prior_runtime_argv(prior: Mapping[str, Any]) -> list[str]:
    receipt = prior["receipt"]
    return [
        "--prior-diagnostic-sha256", str(receipt["sha256"]),
        "--prior-diagnostic-bytes", str(receipt["bytes"]),
        "--prior-diagnostic-study-commit", str(prior["study_commit"]),
    ]


def _job_descriptor(
    *,
    study_commit: str,
    implementation: Mapping[str, Mapping[str, Any]],
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
            "terminal implementation inventory changed")
    _descriptor_shape(prior.get("receipt"), "attempt 002 receipt")
    queue._verified_commit(prior.get("study_commit"))
    arguments = [
        "/usr/bin/python3",
        "{source_root}/" + str(THIS_RELATIVE),
        "run",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", JOB_ID,
        "--expected-role", WORKER_ROLE,
    ]
    arguments.extend(_prior_runtime_argv(prior))
    arguments.extend(_hash_argv(implementation))
    return {
        "job_id": JOB_ID,
        "released": True,
        "source_commit": commit,
        "role": WORKER_ROLE,
        "argv": arguments,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def _wave_from_authenticated(
    *,
    study_commit: str,
    implementation: Mapping[str, Mapping[str, Any]],
    prerequisites: Mapping[str, Mapping[str, Any]],
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    require(len(prerequisites) == 5, "exactly five replay prerequisites are required")
    descriptor = _job_descriptor(
        study_commit=study_commit, implementation=implementation, prior=prior
    )
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "source_commit": descriptor["source_commit"],
        "status": WAVE_STATUS,
        "jobs": [descriptor],
        "implementation": dict(implementation),
        "replay_prerequisite_receipts": dict(prerequisites),
        "prior_diagnostic": dict(prior),
        "expected_outputs": {
            "published_camera_crop_contracts": MODEL_CONTRACT_NAMES,
            "success_receipt": SUCCESS_RECEIPT,
            "publish_file_count": 3,
        },
        "science_counts": zero_science_counts(),
        "simulator_state_render_used": False,
        "whole_frame_identity": False,
        "safe_for_physical_alignment_input_only_after_runtime_success": True,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "descriptor_dispatched": False,
        "claim_boundary": WAVE_CLAIM_BOUNDARY,
    }


def build_wave(
    *,
    study_commit: str,
    prerequisite_inputs: Mapping[str, tuple[Path, str]],
    prior_diagnostic_path: Path,
    prior_diagnostic_sha256: str,
    results_root: Path,
    results_commit: str,
) -> dict[str, Any]:
    """Return one fully gated descriptor; never dispatch or edit queue state."""

    implementation = _local_implementation()
    _runtime_contract(REPOSITORY_ROOT)
    prerequisites = diagnostic.validate_prerequisites(prerequisite_inputs)
    prior = validate_prior_diagnostic(
        path=prior_diagnostic_path,
        supplied_sha256=prior_diagnostic_sha256,
        results_root=results_root,
        results_commit=results_commit,
    )
    return _wave_from_authenticated(
        study_commit=study_commit,
        implementation=implementation,
        prerequisites=prerequisites,
        prior=prior,
    )


def _validate_empty_queue_template(current_queue: Any) -> None:
    require(
        isinstance(current_queue, Mapping)
        and set(current_queue) == QUEUE_MANIFEST_KEYS,
        "queue template fields changed",
    )
    require(
        current_queue.get("schema_version") == QUEUE_MANIFEST_SCHEMA
        and current_queue.get("namespace") == NAMESPACE,
        "queue template identity changed",
    )
    require(current_queue.get("shutdown") is False, "queue template is shutdown")
    require(
        isinstance(current_queue.get("jobs"), list)
        and current_queue["jobs"] == [],
        "queue template is not exactly empty",
    )


def _queue_manifest_from_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": QUEUE_MANIFEST_SCHEMA,
        "namespace": NAMESPACE,
        "shutdown": False,
        "jobs": [{
            "job_id": descriptor["job_id"],
            "released": descriptor["released"],
            "source_commit": descriptor["source_commit"],
            "role": descriptor["role"],
            "argv": list(descriptor["argv"]),
            "max_wall_seconds": descriptor["max_wall_seconds"],
            "publish_log_tail_bytes": descriptor["publish_log_tail_bytes"],
        }],
    }


def build_queue_candidate(
    *,
    study_commit: str,
    prerequisite_inputs: Mapping[str, tuple[Path, str]],
    prior_diagnostic_path: Path,
    prior_diagnostic_sha256: str,
    results_root: Path,
    results_commit: str,
    control_root: Path,
    control_commit: str,
    queue_path: Path,
) -> dict[str, Any]:
    """Authenticate and compose stdout-only queue bytes without legacy reopens."""

    commit = queue._verified_commit(study_commit)
    supplied_control_commit = queue._verified_commit(control_commit)
    require(
        supplied_control_commit == commit,
        "control commit differs from the intended study commit",
    )
    queue._verified_commit(results_commit)
    implementation = _local_implementation()
    terminal_runtime = _runtime_contract(REPOSITORY_ROOT)
    replay_runtime = diagnostic._runtime_contract(REPOSITORY_ROOT)
    control_root = _safe_checkout_root(control_root, "control checkout")
    results_root = _safe_checkout_root(results_root, "results checkout")
    with _held_git_checkout(
        control_root, "control checkout"
    ) as control_checkout, _held_git_checkout(
        results_root, "results checkout"
    ) as results_checkout:
        current_queue = _authenticate_control_state_in_checkout(
            checkout=control_checkout,
            control_commit=supplied_control_commit,
            queue_path=queue_path,
            implementation=implementation,
            terminal_runtime=terminal_runtime,
            replay_runtime=replay_runtime,
        )
        _validate_empty_queue_template(current_queue)
        prerequisites, prior = _authenticate_results_claims_in_checkouts(
            results_checkout=results_checkout,
            results_commit=results_commit,
            prerequisite_inputs=prerequisite_inputs,
            prior_diagnostic_path=prior_diagnostic_path,
            prior_diagnostic_sha256=prior_diagnostic_sha256,
            expected_prerequisites=replay_runtime["prerequisite_receipts"],
            source_checkout=control_checkout,
        )
        wave = _wave_from_authenticated(
            study_commit=commit,
            implementation=implementation,
            prerequisites=prerequisites,
            prior=prior,
        )
        return _queue_manifest_from_descriptor(wave["jobs"][0])


def compose_queue_candidate(
    *,
    wave: Mapping[str, Any],
    expected_study_commit: str,
    results_root: Path,
    control_root: Path,
    control_commit: str,
    queue_path: Path,
) -> dict[str, Any]:
    """Return an exact queue manifest without mutating files, refs, or queue state.

    The returned value is intentionally shaped for the existing cluster queue
    publisher.  It becomes release authority only if a later, separately
    authorized control-branch publication installs it.  This function performs
    no publication and accepts neither a stale wave nor a nonempty queue base.
    """

    commit = queue._verified_commit(expected_study_commit)
    supplied_control_commit = queue._verified_commit(control_commit)
    require(
        supplied_control_commit == commit,
        "control commit differs from the intended study commit",
    )
    require(
        isinstance(wave, Mapping) and set(wave) == WAVE_KEYS,
        "terminal wave fields changed",
    )
    require(
        wave.get("schema_version") == WAVE_SCHEMA
        and wave.get("namespace") == NAMESPACE
        and wave.get("study_id") == STUDY_ID
        and wave.get("source_commit") == commit,
        "terminal wave is stale or has the wrong identity",
    )
    require(wave.get("status") == WAVE_STATUS, "terminal wave status changed")
    require(
        _exact_json_equal(wave.get("science_counts"), zero_science_counts())
        and wave.get("simulator_state_render_used") is False
        and wave.get("whole_frame_identity") is False
        and wave.get("safe_for_physical_alignment_input_only_after_runtime_success")
        is True
        and wave.get("safe_to_release_confirmation") is False
        and wave.get("confirmation_released") is False
        and wave.get("descriptor_dispatched") is False
        and wave.get("claim_boundary") == WAVE_CLAIM_BOUNDARY,
        "terminal wave authority boundary changed",
    )
    require(
        _exact_json_equal(
            wave.get("expected_outputs"),
            {
                "published_camera_crop_contracts": MODEL_CONTRACT_NAMES,
                "success_receipt": SUCCESS_RECEIPT,
                "publish_file_count": 3,
            },
        ),
        "terminal wave expected outputs changed",
    )

    implementation = wave.get("implementation")
    require(
        isinstance(implementation, Mapping)
        and set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
        "terminal wave implementation inventory changed",
    )
    terminal_runtime = _runtime_contract(REPOSITORY_ROOT)
    replay_runtime = diagnostic._runtime_contract(REPOSITORY_ROOT)
    current_implementation = _local_implementation()
    require(
        _exact_json_equal(implementation, current_implementation),
        "terminal wave implementation bytes differ from the current source",
    )
    normalized_implementation = {
        name: _descriptor_shape(implementation[name], f"terminal {name} implementation")
        for name in sorted(implementation)
    }
    current_queue = _authenticate_control_state(
        control_root=control_root,
        control_commit=supplied_control_commit,
        queue_path=queue_path,
        implementation=normalized_implementation,
        terminal_runtime=terminal_runtime,
        replay_runtime=replay_runtime,
    )
    _validate_empty_queue_template(current_queue)

    prerequisites = wave.get("replay_prerequisite_receipts")
    require(
        isinstance(prerequisites, Mapping)
        and set(prerequisites) == set(PREREQUISITE_CLUSTER_PATHS)
        and len(prerequisites) == 5,
        "terminal wave does not contain all five replay prerequisite receipts",
    )
    for name in sorted(prerequisites):
        _descriptor_shape(
            prerequisites[name], f"terminal {name} prerequisite"
        )

    prior = wave.get("prior_diagnostic")
    require(
        isinstance(prior, Mapping)
        and set(prior)
        == {
            "receipt", "results_commit", "study_commit",
            "authenticated_source_blobs",
        },
        "terminal wave attempt 002 diagnostic fields changed",
    )
    _descriptor_shape(prior.get("receipt"), "terminal attempt 002 receipt")
    results_commit = queue._verified_commit(prior.get("results_commit"))
    queue._verified_commit(prior.get("study_commit"))
    authenticated_prerequisites, authenticated_prior = _authenticate_results_inputs(
        results_root=results_root,
        results_commit=results_commit,
        prerequisites=prerequisites,
        prior=prior,
        expected_prerequisites=replay_runtime["prerequisite_receipts"],
        source_root=control_root,
    )
    require(
        _exact_json_equal(prerequisites, authenticated_prerequisites),
        "terminal wave replay prerequisite identities changed",
    )
    require(
        _exact_json_equal(prior, authenticated_prior),
        "terminal wave attempt 002 diagnostic identity changed",
    )
    source_blobs = prior.get("authenticated_source_blobs")
    expected_source_keys = {str(path) for path in PRIOR_SOURCE_PINS}
    require(
        isinstance(source_blobs, Mapping) and set(source_blobs) == expected_source_keys,
        "terminal wave attempt 002 source inventory changed",
    )
    for path, expected_sha256 in PRIOR_SOURCE_PINS.items():
        row = source_blobs[str(path)]
        require(
            isinstance(row, Mapping)
            and set(row) == {"sha256", "bytes"}
            and row.get("sha256") == expected_sha256
            and type(row.get("bytes")) is int
            and row["bytes"] > 0,
            f"terminal wave attempt 002 source identity changed: {path}",
        )

    expected_descriptor = _job_descriptor(
        study_commit=commit,
        implementation=normalized_implementation,
        prior=prior,
    )
    descriptors = wave.get("jobs")
    require(
        isinstance(descriptors, list)
        and len(descriptors) == 1
        and _exact_json_equal(descriptors[0], expected_descriptor),
        "terminal wave executable descriptor changed",
    )

    return _queue_manifest_from_descriptor(expected_descriptor)


def _runtime_implementation(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    rows = args.implementation or []
    result: dict[str, dict[str, str]] = {}
    for name, digest in rows:
        require(name not in result, f"duplicate implementation argument: {name}")
        queue._verified_sha(digest, f"{name} implementation digest")
        result[name] = {"sha256": digest}
    require(set(result) == set(_source_paths(REPOSITORY_ROOT)),
            "runtime terminal implementation arguments changed")
    return result


def _runtime_prior(args: argparse.Namespace) -> dict[str, Any]:
    digest = queue._verified_sha(
        args.prior_diagnostic_sha256, "runtime attempt 002 receipt digest"
    )
    require(type(args.prior_diagnostic_bytes) is int and args.prior_diagnostic_bytes > 0,
            "runtime attempt 002 receipt byte count is invalid")
    return {
        "receipt": {
            "path": str(PRIOR_DIAGNOSTIC_CLUSTER_PATH),
            "sha256": digest,
            "bytes": args.prior_diagnostic_bytes,
        },
        "study_commit": queue._verified_commit(args.prior_diagnostic_study_commit),
    }


def _runtime_descriptor(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    implementation = _runtime_implementation(args)
    prior = _runtime_prior(args)
    require(args.job_id == JOB_ID and args.expected_role == WORKER_ROLE,
            "runtime terminal queue identity changed")
    descriptor = _job_descriptor(
        study_commit=args.study_commit, implementation=implementation, prior=prior
    )
    return descriptor, implementation, prior


def _validate_staged_implementation(
    source_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    paths = _source_paths(source_root)
    require(set(paths) == set(expected), "staged terminal implementation inventory changed")
    result = {}
    for name, path in paths.items():
        identity = replay.file_identity(path, label=f"staged {name}", within=source_root)
        require(identity["sha256"] == expected[name]["sha256"],
                f"staged {name} hash changed")
        result[name] = identity
    _runtime_contract(source_root)
    return result


def _validate_cluster_prior(
    expected: Mapping[str, Any], *, source_root: Path
) -> dict[str, Any]:
    descriptor = _descriptor_shape(expected.get("receipt"), "runtime attempt 002 receipt")
    require(descriptor["path"] == str(PRIOR_DIAGNOSTIC_CLUSTER_PATH),
            "runtime attempt 002 receipt path changed")
    identity = replay.file_identity(
        PRIOR_DIAGNOSTIC_CLUSTER_PATH,
        label="PVC attempt 002 diagnostic receipt",
        expected_path=PRIOR_DIAGNOSTIC_CLUSTER_PATH,
        expected_sha256=descriptor["sha256"],
        expected_bytes=descriptor["bytes"],
    )
    value = replay.load_json(
        PRIOR_DIAGNOSTIC_CLUSTER_PATH, "PVC attempt 002 diagnostic receipt"
    )
    commit = _validate_prior_diagnostic_value(value)
    require(commit == expected.get("study_commit"),
            "runtime attempt 002 study commit changed")
    source_blobs = _validate_prior_source_commit(source_root, commit)
    return {"receipt": identity, "study_commit": commit, "source_blobs": source_blobs}


def _child_command(
    *, model: str, context: Any, implementation: Mapping[str, Mapping[str, Any]], raw: Path
) -> tuple[list[str], Path, Path, Path]:
    require(model in {"N3", "D1"}, "unknown terminal replay model")
    runtime = replay.load_json(
        context.source_root / REPLAY_CONTRACT_RELATIVE,
        "terminal pinned replay runtime contract",
    )
    python_key = "robolab_python" if model == "N3" else "d1_python"
    executable = Path(runtime["paths"][python_key])
    replay.file_identity(executable.resolve(), label=f"{model} Python executable")
    model_root = raw / model
    contract_path = raw / MODEL_CONTRACT_NAMES[model]
    log_path = raw / f"{model.lower()}_replay.log"
    command = [
        str(executable),
        str(context.source_root / REPLAY_RELATIVE),
        "witness",
        "--model", model,
        "--runtime-contract", str(context.source_root / REPLAY_CONTRACT_RELATIVE),
        "--output-root", str(model_root),
        "--output-contract", str(contract_path),
    ]
    require(implementation["replay"]["sha256"]
            == replay.sha256_file(context.source_root / REPLAY_RELATIVE),
            "terminal child replay implementation changed")
    return command, model_root, contract_path, log_path


def _launch_receipt(
    *,
    model: str,
    process: subprocess.Popen[bytes],
    contract_path: Path,
    log_path: Path,
    runtime_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    process_group = os.getpgrp()
    require(process_group == os.getpid(),
            "queue wrapper is not its controller-owned process-group leader")
    return queue.signed_document({
        "schema_version": "wmf-camera-crop-witness-child-launch-v1",
        "job_id": JOB_ID,
        "model_id": model,
        "pid": process.pid,
        "wrapper_pid": os.getpid(),
        "process_group_id": process_group,
        "start_new_session": False,
        "inherits_queue_wrapper_process_group": True,
        "queue_controller_killpg_contains_child": True,
        "contract_path": str(contract_path),
        "log_path": str(log_path),
        "runtime_preflight": dict(runtime_diagnostic),
        "science_counts": zero_science_counts(),
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
    })


def _validate_witness_artifacts(model: str, value: Mapping[str, Any], root: Path) -> None:
    expected = {
        "original_camera_replay": root / "original_camera_crop.npy",
        "generated_decoded_crop": root / "generated_decoded_crop.npy",
    }
    for field, expected_path in expected.items():
        row = value.get(field)
        require(isinstance(row, Mapping), f"{model} {field} evidence is missing")
        descriptor = _descriptor_shape(row.get("witness_artifact"), f"{model} {field} witness")
        require(descriptor["path"] == str(expected_path),
                f"{model} {field} witness path changed")
        replay.file_identity(
            expected_path,
            label=f"{model} {field} witness",
            expected_path=expected_path,
            expected_sha256=descriptor["sha256"],
            expected_bytes=descriptor["bytes"],
            within=root,
        )


def _launch_children(
    *, context: Any, implementation: Mapping[str, Mapping[str, Any]], raw: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    processes: dict[str, subprocess.Popen[bytes]] = {}
    launches: dict[str, Any] = {}
    runtime_diagnostics: dict[str, dict[str, Any]] = {}
    plans: dict[str, tuple[list[str], Path, Path, Path]] = {}
    logs: dict[str, Any] = {}
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    require(os.getpgrp() == os.getpid(),
            "queue wrapper is not its controller-owned process-group leader")
    for model in ("N3", "D1"):
        plans[model] = _child_command(
            model=model, context=context, implementation=implementation, raw=raw
        )
        runtime_diagnostics[model] = diagnostic._runtime_environment_diagnostic(
            Path(plans[model][0][0]), environment
        )
    launch_root = raw / "child_launch_receipts"
    launch_root.mkdir()
    require(launch_root.is_dir() and not launch_root.is_symlink(),
            "terminal child launch receipt directory changed")
    try:
        for model in ("N3", "D1"):
            logs[model] = plans[model][3].open("xb")
    except BaseException:
        diagnostic._close_child_logs(logs)
        raise

    stage = "before_child_launch"
    exits: dict[str, int | None] = {"N3": None, "D1": None}
    try:
        for model in ("N3", "D1"):
            command, _, contract_path, log_path = plans[model]
            stage = f"launch_{model}"
            process = subprocess.Popen(
                command,
                cwd=context.source_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=logs[model],
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            processes[model] = process
            stage = f"persist_launch_{model}"
            receipt_path = launch_root / f"{model.lower()}_launch.json"
            receipt = _launch_receipt(
                model=model,
                process=process,
                contract_path=contract_path,
                log_path=log_path,
                runtime_diagnostic=runtime_diagnostics[model],
            )
            queue.immutable_json(receipt_path, receipt)
            launches[model] = {
                "pid": process.pid,
                "process_group_id": receipt["process_group_id"],
                "start_new_session": False,
                "inherits_queue_wrapper_process_group": True,
                "contract_path": str(contract_path),
                "log_path": str(log_path),
                "launch_receipt_path": str(receipt_path),
                "launch_receipt_persisted": True,
                "launch_receipt": replay.file_identity(
                    receipt_path, label=f"{model} terminal launch receipt"
                ),
            }
        stage = "persist_launch_aggregate"
        queue.immutable_json(raw / "child_launches.json", queue.signed_document({
            "schema_version": "wmf-camera-crop-witness-child-launches-v1",
            "job_id": JOB_ID,
            "children": launches,
            "science_counts": zero_science_counts(),
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
        }))
        deadline = time.monotonic() + MAX_WALL_SECONDS - 60
        stage = "wait_for_children"
        timed_out: set[str] = set()
        for model, process in processes.items():
            try:
                exits[model] = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out.add(model)
        if timed_out:
            cleanup_exits, cleanup = diagnostic._terminate_and_reap_started(processes)
            for model in processes:
                exits[model] = 124 if model in timed_out else cleanup_exits.get(model)
            log_close = diagnostic._close_child_logs(logs)
            raise CameraCropTerminalChildError(
                exits,
                diagnostic._child_diagnostics(
                    log_paths={name: plan[3] for name, plan in plans.items()},
                    exits=exits,
                    runtime_diagnostics=runtime_diagnostics,
                    launches=launches,
                    started=set(processes),
                ),
                orchestration_failure={
                    "stage": "child_wall_timeout",
                    "timed_out_models": sorted(timed_out),
                    "cleanup": cleanup,
                    "log_close": log_close,
                },
                detail="terminal camera replay child wall timeout",
            )
        stage = "finalize_child_logs"
        log_close = diagnostic._close_child_logs(logs)
        require(all(row.get("closed") is True for row in log_close.values()),
                "one or more terminal child logs could not be finalized")
    except CameraCropTerminalChildError:
        raise
    except BaseException as error:
        cleanup_exits, cleanup = diagnostic._terminate_and_reap_started(processes)
        for model in processes:
            exits[model] = cleanup_exits.get(model)
        log_close = diagnostic._close_child_logs(logs)
        diagnostics = diagnostic._child_diagnostics(
            log_paths={name: plan[3] for name, plan in plans.items()},
            exits=exits,
            runtime_diagnostics=runtime_diagnostics,
            launches=launches,
            started=set(processes),
        )
        raise CameraCropTerminalChildError(
            exits,
            diagnostics,
            orchestration_failure={
                "stage": stage,
                "cause": diagnostic._safe_exception_summary(error),
                "cleanup": cleanup,
                "log_close": log_close,
            },
            detail=f"terminal camera replay orchestration failed at {stage}",
        ) from error
    finally:
        diagnostic._close_child_logs(logs)

    diagnostics = diagnostic._child_diagnostics(
        log_paths={name: plan[3] for name, plan in plans.items()},
        exits=exits,
        runtime_diagnostics=runtime_diagnostics,
        launches=launches,
        started=set(processes),
    )
    if exits != {"N3": 0, "D1": 0}:
        raise CameraCropTerminalChildError(exits, diagnostics)
    contracts: dict[str, dict[str, Any]] = {}
    try:
        for model in ("N3", "D1"):
            _, model_root, contract_path, log_path = plans[model]
            value = replay.load_json(contract_path, f"raw {model} terminal camera contract")
            replay.validate_camera_crop_contract(value, expected_model=model)
            _validate_witness_artifacts(model, value, model_root)
            contracts[model] = {
                "value": value,
                "identity": replay.file_identity(
                    contract_path, label=f"raw {model} terminal camera contract"
                ),
                "log": replay.file_identity(log_path, label=f"{model} terminal replay log"),
            }
    except BaseException as error:
        raise CameraCropTerminalChildError(
            exits,
            diagnostics,
            orchestration_failure={
                "stage": "validate_raw_contracts_and_witnesses",
                "cause": diagnostic._safe_exception_summary(error),
                "cleanup": {model: {"reaped": True} for model in ("N3", "D1")},
            },
            detail="terminal camera raw contract validation failed",
        ) from error
    return contracts, launches, diagnostics


def _success_receipt(
    *, job_dir: Path, outputs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    require(set(outputs) == {"N3", "D1"}, "terminal output model inventory changed")
    return queue.signed_document({
        "schema_version": JOB_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "passed",
        "decision": "qualified_from_original_camera_pixels",
        "job_id": JOB_ID,
        "job_dir": str(job_dir),
        "science_counts": zero_science_counts(),
        "simulator_state_render_used": False,
        "whole_frame_identity": False,
        "safe_for_physical_alignment_input": True,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "behavioral_policy_skill_evaluated": False,
        "raw_outputs_recoverable_on_gm_pvc": True,
        "outputs": {"published_camera_crop_contracts": dict(outputs)},
    })


def _validate_success_receipt(value: Mapping[str, Any], *, job_dir: Path) -> None:
    require(set(value) == SUCCESS_RECEIPT_KEYS, "terminal success receipt fields changed")
    queue.verify_signed_document(value, "terminal camera witness receipt")
    require(value.get("schema_version") == JOB_RECEIPT_SCHEMA
            and value.get("namespace") == NAMESPACE
            and value.get("study_id") == STUDY_ID
            and value.get("status") == "passed"
            and value.get("decision") == "qualified_from_original_camera_pixels"
            and value.get("job_id") == JOB_ID
            and value.get("job_dir") == str(job_dir)
            and _exact_json_equal(value.get("science_counts"), zero_science_counts())
            and value.get("simulator_state_render_used") is False
            and value.get("whole_frame_identity") is False
            and value.get("safe_for_physical_alignment_input") is True
            and value.get("safe_to_release_confirmation") is False
            and value.get("confirmation_released") is False
            and value.get("behavioral_policy_skill_evaluated") is False
            and value.get("raw_outputs_recoverable_on_gm_pvc") is True,
            "terminal success receipt authority changed")
    outputs = value.get("outputs")
    require(isinstance(outputs, Mapping)
            and set(outputs) == {"published_camera_crop_contracts"},
            "terminal success output namespace changed")
    rows = outputs["published_camera_crop_contracts"]
    require(isinstance(rows, Mapping) and set(rows) == {"N3", "D1"},
            "terminal success contract inventory changed")
    for model in ("N3", "D1"):
        row = rows[model]
        require(isinstance(row, Mapping) and set(row) == OUTPUT_ENTRY_KEYS,
                f"terminal {model} output fields changed")
        descriptor = _descriptor_shape(
            {key: row[key] for key in ("path", "sha256", "bytes")},
            f"terminal {model} published contract",
        )
        require(descriptor["path"] == str(job_dir / "publish" / MODEL_CONTRACT_NAMES[model])
                and row.get("schema_version") == replay.SCHEMA
                and row.get("model_id") == model
                and row.get("camera_crop_id") == EXPECTED_CAMERA_CROP_IDS[model]
                and queue.SHA256_RE.fullmatch(str(row.get("payload_sha256", ""))) is not None,
                f"terminal {model} published contract identity changed")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _post_commit_fsync(*paths: Path) -> None:
    """Attempt durability syncs without turning an atomic success into failure.

    Every semantic and byte check occurs before the directory rename.  After
    that commit point a published success receipt must not coexist with a
    nonzero wrapper merely because a redundant parent-directory fsync reports
    an error.  Both source and destination parents are nevertheless synced.
    """

    for path in paths:
        try:
            _fsync_directory(path)
        except BaseException:
            pass


def _publish_success(
    *, job_dir: Path, raw: Path, publish: Path,
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    require(set(contracts) == {"N3", "D1"}, "terminal raw contract inventory changed")
    require(publish.is_dir() and not publish.is_symlink() and not any(publish.iterdir()),
            "terminal publish directory is not empty")
    staging = raw / "terminal_publish_staging"
    require(not staging.exists() and not staging.is_symlink(),
            "terminal publication staging already exists")
    staging.mkdir()
    outputs: dict[str, dict[str, Any]] = {}
    for model in ("N3", "D1"):
        raw_identity = contracts[model]["identity"]
        _validate_witness_artifacts(model, contracts[model]["value"], raw / model)
        target = staging / MODEL_CONTRACT_NAMES[model]
        queue.immutable_bytes(
            target,
            Path(raw_identity["path"]).read_bytes(),
            maximum_bytes=2 * 1024 * 1024,
        )
        identity = replay.file_identity(target, label=f"staged {model} terminal contract")
        require(identity["sha256"] == raw_identity["sha256"]
                and identity["bytes"] == raw_identity["bytes"],
                f"staged {model} terminal contract differs from raw contract")
        value = replay.load_json(target, f"staged {model} terminal contract")
        replay.validate_camera_crop_contract(value, expected_model=model)
        outputs[model] = {
            "path": str(job_dir / "publish" / MODEL_CONTRACT_NAMES[model]),
            "sha256": identity["sha256"],
            "bytes": identity["bytes"],
            "schema_version": replay.SCHEMA,
            "model_id": model,
            "camera_crop_id": value["camera_crop_id"],
            "payload_sha256": value["payload_sha256"],
        }
    receipt = _success_receipt(job_dir=job_dir, outputs=outputs)
    _validate_success_receipt(receipt, job_dir=job_dir)
    queue.immutable_json(staging / SUCCESS_RECEIPT, receipt)
    require({path.name for path in staging.iterdir()}
            == {SUCCESS_RECEIPT, *MODEL_CONTRACT_NAMES.values()},
            "terminal staging publication inventory changed")
    _fsync_directory(staging)
    _fsync_directory(raw)
    _fsync_directory(job_dir)
    os.replace(staging, publish)
    _post_commit_fsync(raw, job_dir)
    return receipt


def _write_failure(job_dir: Path, context: Any | None, error: BaseException) -> None:
    expected = CONTROL_ROOT / "jobs" / JOB_ID
    try:
        if Path(job_dir) != expected or Path(job_dir).resolve() != expected:
            return
        publish = expected / "publish"
        if not publish.exists():
            publish.mkdir()
        require(publish.is_dir() and not publish.is_symlink(),
                "terminal failure publish path changed")
        if (publish / SUCCESS_RECEIPT).exists() or (publish / SUCCESS_RECEIPT).is_symlink():
            return
        existing = publish / FAILURE_RECEIPT
        if existing.is_file() and not existing.is_symlink():
            return
        diagnostic._cleanup_partial_publication(publish)
        require(not any(publish.iterdir()), "terminal failure publication inventory is not empty")
        failure = diagnostic._safe_exception_summary(error)
        if isinstance(error, CameraCropTerminalChildError):
            failure["child_exits"] = error.exits
            failure["child_diagnostics"] = error.diagnostics
            failure["child_orchestration_failure"] = error.orchestration_failure
        value = queue.signed_document({
            "schema_version": FAILURE_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "status": "technical_invalid",
            "decision": "no_go",
            "job_id": JOB_ID,
            "study_commit": context.study_commit if context is not None else None,
            "queue_role": context.role if context is not None else WORKER_ROLE,
            "failure": failure,
            "prior_attempt": PRIOR_JOB_ID,
            "science_counts": zero_science_counts(),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_for_physical_alignment_input": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "claim_boundary": (
                "Technical-invalid CPU replay attempt. No model request, simulator action, "
                "behavior, label, physical-alignment authority, or confirmation release was "
                "created; partial derived publish copies were removed while raw evidence was "
                "preserved."
            ),
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(publish / FAILURE_RECEIPT, value)
    except BaseException:
        return


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    descriptor, expected_implementation, expected_prior = _runtime_descriptor(args)
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
        implementation = _validate_staged_implementation(
            context.source_root, expected_implementation
        )
        runtime = _runtime_contract(context.source_root)
        replay_runtime = diagnostic._runtime_contract(context.source_root)
        prerequisites = diagnostic._validate_cluster_prerequisites(replay_runtime)
        require(len(prerequisites) == 5, "PVC replay prerequisite count changed")
        prior = _validate_cluster_prior(expected_prior, source_root=context.source_root)
        raw_root, publish = queue._prepare_output_directories(context.job_dir)
        require(context.job_dir / RAW_SUBDIR == raw_root / "camera_crop_witness_terminal",
                "terminal camera witness raw path changed")
        raw = raw_root / "camera_crop_witness_terminal"
        raw.mkdir()
        queue.immutable_json(raw / "terminal_input.json", queue.signed_document({
            "schema_version": "wmf-camera-crop-witness-terminal-input-v1",
            "job_id": JOB_ID,
            "study_commit": context.study_commit,
            "implementation": implementation,
            "terminal_contract": replay.file_identity(
                context.source_root / TERMINAL_CONTRACT_RELATIVE,
                label="staged terminal contract input",
                within=context.source_root,
            ),
            "prior_diagnostic": prior,
            "replay_prerequisites": prerequisites,
            "science_counts": zero_science_counts(),
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
        }))
        contracts, _, _ = _launch_children(
            context=context, implementation=implementation, raw=raw
        )
        require(runtime["child_runtime"]["diagnostic_only"] is False
                and runtime["child_runtime"]["publish_crop_contracts"] is True,
                "terminal publication boundary changed")
        return _publish_success(
            job_dir=context.job_dir, raw=raw, publish=publish, contracts=contracts
        )
    except BaseException as error:
        _write_failure(Path(args.job_dir), context, error)
        raise


def _add_prerequisite_arguments(parser: argparse.ArgumentParser) -> None:
    for name in sorted(PREREQUISITE_CLUSTER_PATHS):
        flag = name.replace("_", "-")
        parser.add_argument(f"--{flag}", type=Path, required=True)
        parser.add_argument(f"--{flag}-sha256", required=True)


def _builder_inputs(args: argparse.Namespace) -> dict[str, tuple[Path, str]]:
    return {
        name: (getattr(args, name), getattr(args, name + "_sha256"))
        for name in PREREQUISITE_CLUSTER_PATHS
    }


def _validate_build_mode_options(args: argparse.Namespace) -> bool:
    """Validate queue-mode relationships before any checkout or artifact access."""

    queue_mode = args.queue_template is not None
    if not queue_mode:
        require(
            args.control_root is None and args.control_commit is None,
            "control checkout arguments require --queue-template",
        )
        return False

    require(
        args.output is None,
        "queue candidate emission is stdout-only and cannot use --output",
    )
    require(
        args.control_root is not None and args.control_commit is not None,
        "queue candidate emission requires --control-root and --control-commit",
    )
    study_commit = queue._verified_commit(args.study_commit)
    control_commit = queue._verified_commit(args.control_commit)
    queue._verified_commit(args.results_commit)
    require(
        control_commit == study_commit,
        "control commit differs from the intended study commit",
    )
    control_root = _lexical_absolute_path(args.control_root, "control checkout")
    results_root = _lexical_absolute_path(args.results_root, "results checkout")
    require(
        _lexical_absolute_path(args.queue_template, "control queue")
        == control_root / CONTROL_QUEUE_RELATIVE,
        "control queue is not the exact tracked cluster_queue.json path",
    )
    require(
        _lexical_absolute_path(
            args.prior_diagnostic_receipt, "terminal attempt 002 receipt"
        ) == results_root / PRIOR_DIAGNOSTIC_RESULTS_RELATIVE,
        "terminal attempt 002 receipt is not the exact results artifact",
    )
    queue._verified_sha(
        args.prior_diagnostic_receipt_sha256,
        "terminal attempt 002 receipt digest",
    )
    for name, relative in PREREQUISITE_RESULTS_RELATIVES.items():
        require(
            _lexical_absolute_path(
                getattr(args, name), f"terminal {name} prerequisite"
            ) == results_root / relative,
            f"terminal {name} prerequisite is not the exact results artifact",
        )
        queue._verified_sha(
            getattr(args, name + "_sha256"),
            f"terminal {name} prerequisite digest",
        )
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-wave", help="emit one gated descriptor without dispatch")
    build.add_argument("--study-commit", required=True)
    build.add_argument("--results-root", type=Path, required=True)
    build.add_argument("--results-commit", required=True)
    build.add_argument("--prior-diagnostic-receipt", type=Path, required=True)
    build.add_argument("--prior-diagnostic-receipt-sha256", required=True)
    build.add_argument(
        "--queue-template",
        type=Path,
        help=(
            "authenticate the exact tracked cluster_queue.json in --control-root "
            "and emit a directly consumable candidate to stdout"
        ),
    )
    build.add_argument(
        "--control-root", type=Path,
        help="exact clean control-branch checkout containing --queue-template",
    )
    build.add_argument(
        "--control-commit",
        help="exact control checkout commit; must equal --study-commit",
    )
    build.add_argument("--output", type=Path)
    _add_prerequisite_arguments(build)
    run = sub.add_parser("run", help="run inside the claimed detached queue job")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--study-commit", required=True)
    run.add_argument("--job-dir", type=Path, required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--expected-role", required=True)
    run.add_argument("--prior-diagnostic-sha256", required=True)
    run.add_argument("--prior-diagnostic-bytes", type=int, required=True)
    run.add_argument("--prior-diagnostic-study-commit", required=True)
    run.add_argument("--implementation", nargs=2, action="append", metavar=("NAME", "SHA256"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run_job(args)
        return 0
    queue_mode = _validate_build_mode_options(args)
    if queue_mode:
        result = build_queue_candidate(
            study_commit=args.study_commit,
            prerequisite_inputs=_builder_inputs(args),
            prior_diagnostic_path=args.prior_diagnostic_receipt,
            prior_diagnostic_sha256=args.prior_diagnostic_receipt_sha256,
            results_root=args.results_root,
            results_commit=args.results_commit,
            control_root=args.control_root,
            control_commit=args.control_commit,
            queue_path=args.queue_template,
        )
    else:
        result = build_wave(
            study_commit=args.study_commit,
            prerequisite_inputs=_builder_inputs(args),
            prior_diagnostic_path=args.prior_diagnostic_receipt,
            prior_diagnostic_sha256=args.prior_diagnostic_receipt_sha256,
            results_root=args.results_root,
            results_commit=args.results_commit,
        )
    payload = queue.canonical_bytes(result)
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        target = Path(args.output)
        require(not target.exists() and not target.is_symlink(),
                "refusing to replace terminal wave output")
        queue.immutable_bytes(target, payload, maximum_bytes=2 * 1024 * 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
