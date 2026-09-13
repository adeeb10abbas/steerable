#!/usr/bin/env python3
"""Build and run the detached, CPU-only camera-crop replay witness.

``build-wave`` is descriptor-only.  It requires five explicit compact passed
receipts fetched from the results branch and never edits the active queue.
``run`` is the queue entry point: it revalidates the immutable descriptor,
claim, staged source, prerequisite PVC receipts and implementation hashes,
then launches the N3 and D1 byte replays as children of the queue wrapper's
controller-owned process group. Attempt 002 is diagnostic-only: it publishes
one signed zero-science technical-invalid receipt with bounded child-log
evidence and never publishes a crop contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
QUEUE_PATH = Path(__file__).with_name("forecast_timing_queue_jobs.py")
REPLAY_PATH = FORECAST_ROOT / "analysis/camera_crop_replay_witness.py"


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


queue = _load_module(QUEUE_PATH, "wmf_camera_crop_queue_support")
replay = _load_module(REPLAY_PATH, "wmf_camera_crop_replay_support")

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
FORECAST_RELATIVE = queue.FORECAST_RELATIVE

THIS_RELATIVE = FORECAST_RELATIVE / "experiments/forecast_layout/camera_crop_witness_jobs.py"
QUEUE_RELATIVE = queue.THIS_RELATIVE
REPLAY_RELATIVE = FORECAST_RELATIVE / "analysis/camera_crop_replay_witness.py"
CONTRACT_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/camera_crop_witness_contract.json"
)
N3_CONTRACT_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/n3_first_live_contract.json"
)
D1_IDENTITY_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/d1_identity_contract.json"
)
D1_CAPTURE_OVERLAY_RELATIVE = Path("experiments/dreamzero_droid/v2_robolab_client.py")

WAVE_SCHEMA = "wmf-camera-crop-witness-wave-v1"
JOB_RECEIPT_SCHEMA = "wmf-camera-crop-witness-queue-job-v1"
FAILURE_SCHEMA = "wmf-camera-crop-witness-queue-job-v1"
JOB_ID = "camera-crop-replay-witness-002"
PRIOR_JOB_ID = "camera-crop-replay-witness-001"
WORKER_ROLE = "wmf-forecast-0912-worker-06"
MAX_WALL_SECONDS = 10800
PUBLISH_LOG_TAIL_BYTES = 0
CHILD_LOG_TAIL_BYTES = 8192
RAW_SUBDIR = Path("raw/camera_crop_witness")
SUCCESS_RECEIPT = "camera_crop_witness_job_receipt.json"
FAILURE_RECEIPT = "camera_crop_witness_job_failure.json"
MODEL_CONTRACT_NAMES = {
    "N3": "n3_camera_crop_contract.json",
    "D1": "d1_camera_crop_contract.json",
}

PREREQUISITE_CLUSTER_PATHS = {
    "n3_source_audit": CONTROL_ROOT / "jobs/timing-n3-source-audit-001/raw/n3_source_audit.json",
    "d1_source_audit": CONTROL_ROOT / "jobs/timing-d1-source-audit-001/raw/d1_source_audit.json",
    "n3_generation": CONTROL_ROOT / (
        "jobs/timing-n3-live-generation-p00-002/raw/n3_generation_publish/n3_qualification.json"
    ),
    "d1_generation": CONTROL_ROOT / (
        "jobs/timing-d1-normalize-generation-001/raw/d1_generation_probe.json"
    ),
    "d1_qualification": CONTROL_ROOT / (
        "jobs/d1-first-live-005/publish/d1_qualification_job_receipt.json"
    ),
}


class CameraCropQueueError(RuntimeError):
    """A descriptor, prerequisite, detached child, or result gate failed."""


class CameraCropChildError(CameraCropQueueError):
    """A detached replay child failed with signed, bounded log evidence."""

    def __init__(
        self,
        exits: Mapping[str, int | None],
        diagnostics: Mapping[str, Any],
        *,
        orchestration_failure: Mapping[str, Any] | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(detail or f"camera replay child exits changed: {dict(exits)}")
        self.exits = dict(exits)
        self.diagnostics = dict(diagnostics)
        self.orchestration_failure = (
            None if orchestration_failure is None else dict(orchestration_failure)
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CameraCropQueueError(message)


def zero_science_counts() -> dict[str, int]:
    return dict(replay.ZERO_SCIENCE_COUNTS)


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "queue_wrapper": root / THIS_RELATIVE,
        "queue_support": root / QUEUE_RELATIVE,
        "replay": root / REPLAY_RELATIVE,
        "runtime_contract": root / CONTRACT_RELATIVE,
        "n3_runtime_contract": root / N3_CONTRACT_RELATIVE,
        "d1_identity_contract": root / D1_IDENTITY_RELATIVE,
        "d1_capture_overlay": root / D1_CAPTURE_OVERLAY_RELATIVE,
    }


def _local_implementation() -> dict[str, dict[str, Any]]:
    return {
        name: queue.file_identity(path)
        for name, path in _source_paths(REPOSITORY_ROOT).items()
    }


def _runtime_contract(root: Path) -> dict[str, Any]:
    value = replay.load_json(root / CONTRACT_RELATIVE, "camera witness runtime contract")
    require(value.get("schema_version") == replay.RUNTIME_SCHEMA, "runtime contract schema changed")
    require(value.get("study_id") == STUDY_ID, "runtime contract study changed")
    require(value.get("job") == {
        "job_id": JOB_ID,
        "worker_role": WORKER_ROLE,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "cpu_only": True,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }, "runtime contract queue identity changed")
    require(value.get("prior_attempt") == {
        "job_id": PRIOR_JOB_ID,
        "status": "technical_invalid",
        "failure_receipt_sha256": (
            "b8ba6bb884f7c2daaf6695bafd7fed97675d8bc9bc11a125e4353eb52df42550"
        ),
        "wrapper_stderr_sha256": (
            "a2bf9ff0707832e9d1b4968ad6e954cd23290f791b449c29cf72834b39e85922"
        ),
        "preserved": True,
    }, "runtime contract prior-attempt lineage changed")
    require(value.get("child_runtime") == {
        "mode": "diagnostic_replay_of_attempt_001_environment",
        "diagnostic_only": True,
        "inherit_queue_environment": True,
        "overrides": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "failure_log_tail_bytes": CHILD_LOG_TAIL_BYTES,
        "publish_crop_contracts": False,
    }, "runtime contract child environment changed")
    require(value.get("science_counts") == zero_science_counts(),
            "runtime contract science counts changed")
    require(value.get("simulator_state_render_used") is False
            and value.get("whole_frame_identity") is False
            and value.get("safe_to_release_confirmation") is False
            and value.get("confirmation_released") is False,
            "runtime contract authority changed")
    return value


def _validate_prerequisite_value(
    name: str,
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    require(value.get("schema_version") == expected["schema_version"],
            f"{name} schema changed")
    if expected.get("model_id") is not None:
        require(value.get("model_id") == expected["model_id"], f"{name} model changed")
    if expected.get("status") is not None:
        require(value.get("status") == expected["status"], f"{name} status changed")
    if expected.get("decision") is not None:
        require(value.get("decision") == expected["decision"], f"{name} decision changed")
    require(value.get("study_id", STUDY_ID) == STUDY_ID, f"{name} study changed")
    if name == "n3_source_audit":
        require(value.get("source", {}).get("commit")
                == "411d25b2e35bc441126f48c44a4b93e1c0564274", "N3 audit source changed")
    elif name == "d1_source_audit":
        require(value.get("source", {}).get("commit")
                == "ab790c198fbce33503358efbbd4187ce9a89adf3", "D1 audit source changed")
    elif name == "n3_generation":
        require(value.get("qualified") is True
                and value.get("generation_request_count") == 6
                and value.get("robot_episode_count") == 0,
                "N3 generation prerequisite is not the passed zero-policy probe")
    elif name == "d1_generation":
        require(value.get("model_returned_action_executed") is False
                and value.get("selected_request", {}).get("decoded_rgb_shape") == [9, 352, 640, 3],
                "D1 generation prerequisite changed")
    elif name == "d1_qualification":
        require(value.get("behavioral_episode_count") == 0
                and value.get("generation_request_count") == 6,
                "D1 qualification is not the passed zero-policy probe")


def validate_prerequisites(
    paths: Mapping[str, tuple[Path, str]], *, root: Path = REPOSITORY_ROOT
) -> dict[str, dict[str, Any]]:
    runtime = _runtime_contract(root)
    expected = runtime["prerequisite_receipts"]
    require(set(paths) == set(expected), "exactly five explicit prerequisite receipts are required")
    result: dict[str, dict[str, Any]] = {}
    seen: set[Path] = set()
    for name in sorted(expected):
        path, supplied_hash = paths[name]
        require(path not in seen, "prerequisite receipt paths are ambiguous")
        seen.add(path)
        wanted = expected[name]
        require(supplied_hash == wanted["sha256"], f"{name} supplied SHA-256 changed")
        identity = replay.file_identity(
            Path(path),
            label=f"builder prerequisite {name}",
            expected_sha256=wanted["sha256"],
            expected_bytes=wanted["bytes"],
        )
        value = replay.load_json(Path(path), f"builder prerequisite {name}")
        _validate_prerequisite_value(name, value, wanted)
        result[name] = identity
    return result


def _hash_argv(implementation: Mapping[str, Mapping[str, Any]]) -> list[str]:
    argv: list[str] = []
    for name in sorted(implementation):
        argv.extend(["--implementation", name, str(implementation[name]["sha256"])])
    return argv


def _job_descriptor(
    *, study_commit: str, implementation: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
            "implementation inventory changed")
    argv = [
        "/usr/bin/python3",
        "{source_root}/" + str(THIS_RELATIVE),
        "run",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", JOB_ID,
        "--expected-role", WORKER_ROLE,
    ]
    argv.extend(_hash_argv(implementation))
    return {
        "job_id": JOB_ID,
        "released": True,
        "source_commit": commit,
        "role": WORKER_ROLE,
        "argv": argv,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def build_wave(
    *,
    study_commit: str,
    prerequisite_inputs: Mapping[str, tuple[Path, str]],
) -> dict[str, Any]:
    """Return one receipt-gated descriptor without dispatching it."""

    implementation = _local_implementation()
    prerequisites = validate_prerequisites(prerequisite_inputs)
    descriptor = _job_descriptor(study_commit=study_commit, implementation=implementation)
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "source_commit": descriptor["source_commit"],
        "status": "diagnostic_only_not_dispatched_all_five_receipt_gates_passed",
        "jobs": [descriptor],
        "implementation": implementation,
        "prerequisite_receipts": prerequisites,
        "expected_outputs": {
            "technical_invalid_diagnostic_receipt": FAILURE_RECEIPT,
        },
        "diagnostic_only": True,
        "science_counts": zero_science_counts(),
        "simulator_state_render_used": False,
        "whole_frame_identity": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "claim_boundary": (
            "One CPU-only replay diagnostic preserving attempt 001. It may only publish a "
            "bounded technical-invalid diagnostic receipt, never crop contracts, and issues "
            "no model request, simulator action, behavior, label, or confirmation release."
        ),
    }


def _runtime_implementation(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    rows = args.implementation or []
    result: dict[str, dict[str, str]] = {}
    for name, digest in rows:
        require(name not in result, f"duplicate implementation argument: {name}")
        queue._verified_sha(digest, f"{name} implementation digest")
        result[name] = {"sha256": digest}
    require(set(result) == set(_source_paths(REPOSITORY_ROOT)),
            "runtime implementation arguments changed")
    return result


def _runtime_descriptor(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation = _runtime_implementation(args)
    require(args.job_id == JOB_ID and args.expected_role == WORKER_ROLE,
            "runtime queue identity changed")
    return _job_descriptor(
        study_commit=args.study_commit, implementation=implementation
    ), implementation


def _validate_staged_implementation(
    source_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    paths = _source_paths(source_root)
    require(set(paths) == set(expected), "staged implementation inventory changed")
    result = {}
    for name, path in paths.items():
        identity = replay.file_identity(path, label=f"staged {name}", within=source_root)
        require(identity["sha256"] == expected[name]["sha256"],
                f"staged {name} hash changed")
        result[name] = identity
    _runtime_contract(source_root)
    return result


def _validate_cluster_prerequisites(runtime: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected = runtime["prerequisite_receipts"]
    require(set(expected) == set(PREREQUISITE_CLUSTER_PATHS),
            "cluster prerequisite inventory changed")
    result = {}
    for name, path in PREREQUISITE_CLUSTER_PATHS.items():
        wanted = expected[name]
        identity = replay.file_identity(
            path,
            label=f"PVC prerequisite {name}",
            expected_path=path,
            expected_sha256=wanted["sha256"],
            expected_bytes=wanted["bytes"],
        )
        value = replay.load_json(path, f"PVC prerequisite {name}")
        _validate_prerequisite_value(name, value, wanted)
        result[name] = identity
    return result


def _child_command(
    *, model: str, context: Any, implementation: Mapping[str, Mapping[str, Any]], raw: Path
) -> tuple[list[str], Path, Path, Path]:
    runtime = replay.load_json(
        context.source_root / CONTRACT_RELATIVE, "camera witness runtime contract"
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
        "--runtime-contract", str(context.source_root / CONTRACT_RELATIVE),
        "--output-root", str(model_root),
        "--output-contract", str(contract_path),
    ]
    require(implementation["replay"]["sha256"]
            == replay.sha256_file(context.source_root / REPLAY_RELATIVE),
            "child replay implementation changed")
    return command, model_root, contract_path, log_path


_TRACEBACK_FRAME_RE = re.compile(
    r'^\s*File "([^"]+)", line ([1-9][0-9]*), in ([A-Za-z0-9_.<>-]+)\s*$'
)
_TERMINAL_ERROR_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::(.*))?$"
)
_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]")
_SAFE_DIAGNOSTIC_ID_RE = re.compile(r"^[A-Za-z0-9_.<>-]+$")
_SENSITIVE_PATH_COMPONENT_RE = re.compile(
    r"(?:^|[_-])(auth|credential|key|password|private|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_FRAGMENT_RE = re.compile(
    r"(?:sk-(?:proj-)?[A-Za-z0-9]|github_pat_|gh[pousr]_|hf_[A-Za-z0-9]|"
    r"xox[baprs]-|AKIA[0-9A-Z])",
    re.IGNORECASE,
)
_SAFE_ABSOLUTE_PATH_PREFIXES = (
    "/bin/",
    "/data/users/ali/vla_wam/envs/",
    "/data/users/ali/vla_wam/external/",
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/",
    "/lib/",
    "/lib64/",
    "/opt/",
    "/usr/",
    str(REPOSITORY_ROOT) + "/",
)


def _safe_diagnostic_path(value: Any) -> dict[str, Any]:
    text = value if isinstance(value, str) else ""
    sensitive_component = any(
        _SENSITIVE_PATH_COMPONENT_RE.search(component) is not None
        for component in Path(text).parts
    )
    if (
        text
        and Path(text).is_absolute()
        and not any(char in text for char in "\r\n\0")
        and not sensitive_component
        and _SECRET_FRAGMENT_RE.search(text) is None
        and text.startswith(_SAFE_ABSOLUTE_PATH_PREFIXES)
    ):
        return {"absolute_path": text}
    payload = str(value).encode("utf-8", errors="surrogateescape")
    return {
        "redacted_value_bytes": len(payload),
        "redacted_value_sha256": queue.sha256_bytes(payload),
    }


def _safe_identifier(value: Any) -> dict[str, Any] | str:
    if (
        isinstance(value, str)
        and _SAFE_DIAGNOSTIC_ID_RE.fullmatch(value)
        and _SENSITIVE_PATH_COMPONENT_RE.search(value) is None
        and _SECRET_FRAGMENT_RE.search(value) is None
    ):
        return value
    payload = str(value).encode("utf-8", errors="surrogateescape")
    return {
        "redacted_value_bytes": len(payload),
        "redacted_value_sha256": queue.sha256_bytes(payload),
    }


def _safe_exception_summary(error: BaseException) -> dict[str, Any]:
    """Describe an exception without publishing its message or source text."""

    try:
        message = str(error).encode("utf-8", errors="surrogateescape")
    except BaseException as stringify_error:
        fallback = type(stringify_error).__name__.encode("ascii", errors="replace")
        message = b""
        stringify_failure: dict[str, Any] | None = {
            "error_type": _safe_identifier(type(stringify_error).__name__),
            "message_unavailable_marker_sha256": queue.sha256_bytes(fallback),
        }
    else:
        stringify_failure = None
    frames = []
    try:
        extracted = traceback.extract_tb(error.__traceback__, limit=20)
    except BaseException:
        extracted = []
    for frame in extracted:
        frames.append({
            "source": _safe_diagnostic_path(frame.filename),
            "line": int(frame.lineno),
            "function": _safe_identifier(frame.name),
        })
    result: dict[str, Any] = {
        "error_type": _safe_identifier(type(error).__name__),
        "message_bytes": len(message),
        "message_sha256": queue.sha256_bytes(message),
        "traceback_frames": frames,
        "traceback_frame_count": len(frames),
        "exception_message_published": False,
        "traceback_source_text_published": False,
    }
    if stringify_failure is not None:
        result["stringify_failure"] = stringify_failure
    if isinstance(error, OSError):
        if isinstance(error.errno, int):
            result["errno"] = error.errno
        if error.filename is not None:
            result["filename"] = _safe_diagnostic_path(error.filename)
    return result


def _safe_preflight_projection(value: Any, raw_line: bytes) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if (
        value.get("schema_version") != "wmf-camera-crop-child-runtime-preflight-v1"
        or value.get("event") != "runtime_preflight"
        or value.get("model_id") not in {"N3", "D1"}
        or value.get("phase") not in {"before_replay", "after_replay_failure"}
    ):
        return None
    python = value.get("python")
    modules = value.get("module_specs")
    if not isinstance(python, Mapping) or not isinstance(modules, Mapping):
        return None
    projected_modules: dict[str, Any] = {}
    for name in ("numpy", "torch", "torchvision", "PIL", "openpi_client.image_tools"):
        row = modules.get(name)
        if not isinstance(row, Mapping):
            continue
        projected: dict[str, Any] = {"found": row.get("found") is True}
        if row.get("origin") is not None:
            projected["origin"] = _safe_diagnostic_path(row.get("origin"))
        locations = row.get("search_locations")
        if isinstance(locations, list):
            projected["search_locations"] = [
                _safe_diagnostic_path(item) for item in locations
            ]
        error_type = row.get("lookup_error_type")
        if isinstance(error_type, str) and _SAFE_DIAGNOSTIC_ID_RE.fullmatch(error_type):
            projected["lookup_error_type"] = _safe_identifier(error_type)
        detail = row.get("lookup_error_detail")
        if isinstance(detail, str):
            detail_bytes = detail.encode("utf-8", errors="surrogateescape")
            projected["lookup_error_detail_bytes"] = len(detail_bytes)
            projected["lookup_error_detail_sha256"] = queue.sha256_bytes(detail_bytes)
        projected_modules[name] = projected
    path_environment: dict[str, Any] = {}
    raw_environment = value.get("path_environment")
    if isinstance(raw_environment, Mapping):
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            entries = raw_environment.get(name)
            if isinstance(entries, list):
                path_environment[name] = [
                    _safe_diagnostic_path(item.get("absolute_path"))
                    if isinstance(item, Mapping) and "absolute_path" in item
                    else {
                        "redacted_descriptor_sha256": queue.sha256_bytes(
                            queue.compact_bytes(item)
                        )
                    }
                    for item in entries
                ]
    return {
        "model_id": value["model_id"],
        "phase": value["phase"],
        "python": {
            "lexical_path": _safe_diagnostic_path(python.get("lexical_path")),
            "resolved_path": _safe_diagnostic_path(python.get("resolved_path")),
        },
        "module_specs": projected_modules,
        "path_environment": path_environment,
        "cuda_visible_devices_empty": value.get("cuda_visible_devices_empty") is True,
        "source_line_bytes": len(raw_line),
        "source_line_sha256": queue.sha256_bytes(raw_line),
    }


def _safe_log_tail_summary(raw: bytes) -> dict[str, Any]:
    """Project only structural diagnostics; arbitrary log text is never published."""

    lines = raw.splitlines()
    preflights: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    recognized = 0
    for raw_line in lines:
        try:
            line = raw_line.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if line.startswith("{"):
            try:
                projected = _safe_preflight_projection(json.loads(line), raw_line)
            except BaseException:
                projected = None
            if projected is not None:
                preflights.append(projected)
                recognized += 1
                continue
        frame = _TRACEBACK_FRAME_RE.fullmatch(line)
        if frame is not None:
            frames.append({
                "source": _safe_diagnostic_path(frame.group(1)),
                "line": int(frame.group(2)),
                "function": _safe_identifier(frame.group(3)),
            })
            recognized += 1
            continue
        terminal = _TERMINAL_ERROR_RE.fullmatch(line)
        if terminal is not None:
            message = (terminal.group(2) or "").encode("utf-8", errors="surrogateescape")
            row: dict[str, Any] = {
                "error_type": _safe_identifier(terminal.group(1)),
                "message_bytes": len(message),
                "message_sha256": queue.sha256_bytes(message),
            }
            missing = _MISSING_MODULE_RE.search(terminal.group(2) or "")
            if missing is not None:
                row["missing_module"] = _safe_identifier(missing.group(1))
            errors.append(row)
            recognized += 1
    return {
        "preflight_events": preflights,
        "traceback_frames": frames,
        "terminal_errors": errors,
        "line_count": len(lines),
        "recognized_line_count": recognized,
        "arbitrary_text_published": False,
    }


def _bounded_child_log(path: Path, *, model: str, exit_code: int | None) -> dict[str, Any]:
    """Bind a full retained log and a publish-safe structural tail summary."""

    identity = replay.file_identity(Path(path), label=f"{model} retained replay log")
    with Path(identity["path"]).open("rb") as stream:
        size = int(identity["bytes"])
        offset = max(0, size - CHILD_LOG_TAIL_BYTES)
        stream.seek(offset)
        tail = stream.read(CHILD_LOG_TAIL_BYTES + 1)
    require(
        replay.file_identity(Path(identity["path"]), label=f"{model} retained replay log reread")
        == identity,
        f"{model} retained replay log changed while reading its tail",
    )
    require(len(tail) <= CHILD_LOG_TAIL_BYTES, f"{model} log tail bound changed")
    text_tail = tail
    dropped_leading_bytes = 0
    if offset > 0:
        newline = tail.find(b"\n")
        dropped_leading_bytes = len(tail) if newline < 0 else newline + 1
        text_tail = tail[dropped_leading_bytes:]
    safe_summary = _safe_log_tail_summary(text_tail)
    return {
        "model_id": model,
        "exit_code": exit_code,
        "full_log": identity,
        "bounded_tail": {
            "maximum_bytes": CHILD_LOG_TAIL_BYTES,
            "source_offset_bytes": offset,
            "raw_bytes": len(tail),
            "raw_sha256": queue.sha256_bytes(tail),
            "partial_leading_line_dropped_bytes": dropped_leading_bytes,
            "safe_structured_summary": safe_summary,
            "truncated": offset > 0,
        },
        "argv_published": False,
        "environment_values_published": False,
    }


def _safe_environment_paths(value: str | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    rows = []
    for item in value.split(os.pathsep):
        rows.append(_safe_diagnostic_path(item))
    return rows


def _runtime_environment_diagnostic(
    executable: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    lexical = Path(executable)
    require(lexical.is_absolute() and lexical.is_file() and os.access(lexical, os.X_OK),
            "child Python executable is invalid")
    resolved = lexical.resolve(strict=True)
    return {
        "python": {
            "lexical_path": str(lexical),
            "resolved_file": replay.file_identity(resolved, label="resolved child Python"),
        },
        "path_environment": {
            name: _safe_environment_paths(environment.get(name))
            for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        },
        "cpu_offline_overrides": {
            name: environment.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONUNBUFFERED",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
        "inherited_queue_environment": True,
        "argv_published": False,
        "arbitrary_environment_values_published": False,
    }


def _child_diagnostics(
    *,
    log_paths: Mapping[str, Path],
    exits: Mapping[str, int | None],
    runtime_diagnostics: Mapping[str, Mapping[str, Any]],
    launches: Mapping[str, Mapping[str, Any]],
    started: set[str],
) -> dict[str, Any]:
    require(
        set(log_paths) == {"N3", "D1"}
        and set(exits) == {"N3", "D1"}
        and set(runtime_diagnostics) == {"N3", "D1"}
        and set(launches).issubset(started)
        and started.issubset({"N3", "D1"}),
            "child diagnostic inventory changed")
    return {
        model: {
            **_bounded_child_log(log_paths[model], model=model, exit_code=exits[model]),
            "launched": model in started,
            "launch_receipt": (
                launches[model].get("launch_receipt") if model in launches else None
            ),
            "runtime_preflight": dict(runtime_diagnostics[model]),
        }
        for model in ("N3", "D1")
    }


def _terminate_and_reap_started(
    processes: Mapping[str, subprocess.Popen[bytes]],
) -> tuple[dict[str, int | None], dict[str, Any]]:
    """Best-effort local cleanup; queue PGID containment is the final backstop."""

    exits: dict[str, int | None] = {}
    cleanup: dict[str, Any] = {}
    for model, process in processes.items():
        try:
            initial = process.poll()
        except BaseException as error:
            initial = None
            cleanup[model] = {
                "poll_failure": _safe_exception_summary(error),
                "reaped": False,
            }
        if initial is not None:
            try:
                exits[model] = process.wait(timeout=0)
                cleanup[model] = {"signal": "none_already_exited", "reaped": True}
            except BaseException as error:
                exits[model] = initial
                cleanup[model] = {
                    "signal": "none_already_exited",
                    "reaped": False,
                    "wait_failure": _safe_exception_summary(error),
                }
            continue
        try:
            process.terminate()
            signal_name = "SIGTERM_to_child_pid"
        except ProcessLookupError:
            signal_name = "none_process_disappeared"
        except BaseException as error:
            signal_name = "SIGTERM_to_child_pid_failed"
            cleanup.setdefault(model, {})["terminate_failure"] = _safe_exception_summary(error)
        try:
            exits[model] = process.wait(timeout=10)
            cleanup.setdefault(model, {}).update({"signal": signal_name, "reaped": True})
            continue
        except subprocess.TimeoutExpired:
            pass
        except BaseException as error:
            cleanup.setdefault(model, {})["term_wait_failure"] = _safe_exception_summary(error)
        try:
            process.kill()
            kill_name = "SIGKILL_to_child_pid"
        except ProcessLookupError:
            kill_name = "none_process_disappeared"
        except BaseException as error:
            kill_name = "SIGKILL_to_child_pid_failed"
            cleanup.setdefault(model, {})["kill_failure"] = _safe_exception_summary(error)
        try:
            exits[model] = process.wait(timeout=10)
            cleanup.setdefault(model, {}).update({"signal": kill_name, "reaped": True})
        except BaseException as error:
            try:
                exits[model] = process.poll()
            except BaseException:
                exits[model] = None
            cleanup.setdefault(model, {}).update({
                "signal": kill_name,
                "reaped": False,
                "kill_wait_failure": _safe_exception_summary(error),
            })
    return exits, cleanup


def _close_child_logs(logs: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for model, log in logs.items():
        if log.closed:
            results[model] = {"closed": True, "already_closed": True}
            continue
        try:
            log.flush()
            os.fsync(log.fileno())
            log.close()
            results[model] = {"closed": True, "already_closed": False}
        except BaseException as error:
            try:
                log.close()
            except BaseException:
                pass
            results[model] = {
                "closed": log.closed,
                "already_closed": False,
                "failure": _safe_exception_summary(error),
            }
    return results


def _launch_receipt(
    *,
    model: str,
    process: subprocess.Popen[bytes],
    contract_path: Path,
    log_path: Path,
    runtime_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    wrapper_process_group_id = os.getpgrp()
    require(wrapper_process_group_id == os.getpid(),
            "queue wrapper is not its controller-owned process-group leader")
    return queue.signed_document({
        "schema_version": "wmf-camera-crop-witness-child-launch-v1",
        "job_id": JOB_ID,
        "model_id": model,
        "pid": process.pid,
        "wrapper_pid": os.getpid(),
        "process_group_id": wrapper_process_group_id,
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
    # Resolve every executable/source dependency before starting either child.
    for model in ("N3", "D1"):
        plans[model] = _child_command(
            model=model, context=context, implementation=implementation, raw=raw
        )
        command, _, _, _ = plans[model]
        runtime_diagnostics[model] = _runtime_environment_diagnostic(
            Path(command[0]), environment
        )
    launch_receipt_root = raw / "child_launch_receipts"
    launch_receipt_root.mkdir()
    require(launch_receipt_root.is_dir() and not launch_receipt_root.is_symlink(),
            "child launch receipt directory changed")
    try:
        # Precreate both logs so a filesystem/setup error cannot occur after one launch.
        for model in ("N3", "D1"):
            logs[model] = plans[model][3].open("xb")
    except BaseException:
        for log in logs.values():
            if not log.closed:
                log.close()
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
            child_receipt_path = launch_receipt_root / f"{model.lower()}_launch.json"
            child_receipt = _launch_receipt(
                model=model,
                process=process,
                contract_path=contract_path,
                log_path=log_path,
                runtime_diagnostic=runtime_diagnostics[model],
            )
            queue.immutable_json(child_receipt_path, child_receipt)
            launches[model] = {
                "pid": process.pid,
                "process_group_id": child_receipt["process_group_id"],
                "start_new_session": False,
                "inherits_queue_wrapper_process_group": True,
                "contract_path": str(contract_path),
                "log_path": str(log_path),
                "launch_receipt_path": str(child_receipt_path),
                "launch_receipt_persisted": True,
                "launch_receipt": None,
            }
            launches[model]["launch_receipt"] = replay.file_identity(
                child_receipt_path, label=f"{model} child launch receipt"
            )
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
            remaining = max(0.0, deadline - time.monotonic())
            try:
                exits[model] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out.add(model)
        if timed_out:
            cleanup_exits, cleanup = _terminate_and_reap_started(processes)
            for model in processes:
                exits[model] = 124 if model in timed_out else cleanup_exits.get(model)
            log_close = _close_child_logs(logs)
            raise CameraCropChildError(
                exits,
                _child_diagnostics(
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
                detail="camera replay child wall timeout",
            )
        stage = "finalize_child_logs"
        log_close = _close_child_logs(logs)
        require(all(row.get("closed") is True for row in log_close.values()),
                "one or more child logs could not be finalized")
    except CameraCropChildError:
        raise
    except BaseException as error:
        cleanup_exits, cleanup = _terminate_and_reap_started(processes)
        for model in processes:
            exits[model] = cleanup_exits.get(model)
        log_close = _close_child_logs(logs)
        diagnostics = _child_diagnostics(
            log_paths={name: plan[3] for name, plan in plans.items()},
            exits=exits,
            runtime_diagnostics=runtime_diagnostics,
            launches=launches,
            started=set(processes),
        )
        raise CameraCropChildError(
            exits,
            diagnostics,
            orchestration_failure={
                "stage": stage,
                "cause": _safe_exception_summary(error),
                "cleanup": cleanup,
                "log_close": log_close,
            },
            detail=f"camera replay orchestration failed at {stage}",
        ) from error
    finally:
        _close_child_logs(logs)
    diagnostics = _child_diagnostics(
        log_paths={name: plan[3] for name, plan in plans.items()},
        exits=exits,
        runtime_diagnostics=runtime_diagnostics,
        launches=launches,
        started=set(processes),
    )
    if exits != {"N3": 0, "D1": 0}:
        raise CameraCropChildError(exits, diagnostics)
    contracts = {}
    try:
        for model in ("N3", "D1"):
            _, _, contract_path, log_path = plans[model]
            value = replay.load_json(contract_path, f"{model} camera crop contract")
            replay.validate_camera_crop_contract(value, model)
            contracts[model] = {
                "value": value,
                "identity": replay.file_identity(contract_path, label=f"raw {model} contract"),
                "log": replay.file_identity(log_path, label=f"{model} replay log"),
            }
    except BaseException as error:
        raise CameraCropChildError(
            exits,
            diagnostics,
            orchestration_failure={
                "stage": "validate_raw_contracts",
                "cause": _safe_exception_summary(error),
                "cleanup": {model: {"reaped": True} for model in ("N3", "D1")},
            },
            detail="camera replay raw contract validation failed",
        ) from error
    return contracts, launches, diagnostics


def _publish_contracts(
    publish: Path, contracts: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    outputs = {}
    created: list[Path] = []
    try:
        for model in ("N3", "D1"):
            raw_identity = contracts[model]["identity"]
            target = publish / MODEL_CONTRACT_NAMES[model]
            created.append(target)
            queue.immutable_bytes(
                target,
                Path(raw_identity["path"]).read_bytes(),
                maximum_bytes=2 * 1024 * 1024,
            )
            identity = replay.file_identity(target, label=f"published {model} camera contract")
            require(identity["sha256"] == raw_identity["sha256"]
                    and identity["bytes"] == raw_identity["bytes"],
                    f"published {model} contract differs from raw contract")
            value = contracts[model]["value"]
            outputs[model] = {
                **identity,
                "schema_version": replay.SCHEMA,
                "model_id": model,
                "camera_crop_id": value["camera_crop_id"],
                "payload_sha256": value["payload_sha256"],
            }
    except BaseException:
        _cleanup_partial_publication(publish, expected=created)
        raise
    return outputs


def _cleanup_partial_publication(
    publish: Path, *, expected: Sequence[Path] | None = None
) -> bool:
    """Remove only this job's derived contract copies before a terminal receipt.

    Raw witnesses are never touched.  Unknown entries or a terminal success
    receipt make cleanup fail closed instead of broadening the deletion scope.
    """

    directory = Path(publish)
    require(directory.is_dir() and not directory.is_symlink(),
            "partial publication directory changed")
    success = directory / SUCCESS_RECEIPT
    if success.exists() or success.is_symlink():
        return False
    allowed = {directory / name for name in MODEL_CONTRACT_NAMES.values()}
    entries = set(directory.iterdir())
    if expected is not None:
        expected_set = set(expected)
        require(expected_set.issubset(allowed), "partial publication cleanup target changed")
        require(entries.issubset(expected_set), "unexpected partial publication artifact")
    else:
        require(entries.issubset(allowed), "unexpected partial publication artifact")
    for path in sorted(entries):
        require(path in allowed and path.is_file() and not path.is_symlink(),
                "partial publication artifact is not a regular derived contract")
        path.unlink()
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _write_failure(job_dir: Path, context: Any | None, error: BaseException) -> None:
    expected = CONTROL_ROOT / "jobs" / JOB_ID
    try:
        if Path(job_dir) != expected or Path(job_dir).resolve() != expected:
            return
        publish = expected / "publish"
        if not publish.exists():
            publish.mkdir()
        require(publish.is_dir() and not publish.is_symlink(), "failure publish path changed")
        if (publish / SUCCESS_RECEIPT).exists() or (publish / SUCCESS_RECEIPT).is_symlink():
            return
        existing_failure = publish / FAILURE_RECEIPT
        if existing_failure.is_file() and not existing_failure.is_symlink():
            return
        _cleanup_partial_publication(publish)
        require(not any(publish.iterdir()), "failure publication inventory is not empty")
        failure = _safe_exception_summary(error)
        if isinstance(error, CameraCropChildError):
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
            "diagnostic_only": True,
            "prior_attempt": PRIOR_JOB_ID,
            "science_counts": zero_science_counts(),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "claim_boundary": (
                "Diagnostic-only technical-invalid CPU replay preserving attempt 001. No model, "
                "simulator, request, action, behavior, label, or confirmation job was started; "
                "no crop contract is published. Bounded signed log evidence is diagnostic only."
            ),
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(publish / FAILURE_RECEIPT, value)
    except BaseException:
        return


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    descriptor, expected_implementation = _runtime_descriptor(args)
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
        prerequisites = _validate_cluster_prerequisites(runtime)
        raw, publish = queue._prepare_output_directories(context.job_dir)
        require(context.job_dir / RAW_SUBDIR == raw / "camera_crop_witness",
                "camera witness raw path changed")
        witness_raw = raw / "camera_crop_witness"
        witness_raw.mkdir()
        contracts, launches, diagnostics = _launch_children(
            context=context, implementation=implementation, raw=witness_raw
        )
        require(runtime["child_runtime"]["diagnostic_only"] is True
                and runtime["child_runtime"]["publish_crop_contracts"] is False,
                "attempt 002 diagnostic-only boundary changed")
        require(contracts.keys() == {"N3", "D1"} and launches.keys() == {"N3", "D1"},
                "diagnostic replay completion inventory changed")
        raise CameraCropChildError(
            {"N3": 0, "D1": 0},
            diagnostics,
            detail=(
                "diagnostic-only replay unexpectedly completed both children; raw contracts "
                "were retained but intentionally not published"
            ),
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
        name: (
            getattr(args, name),
            getattr(args, name + "_sha256"),
        )
        for name in PREREQUISITE_CLUSTER_PATHS
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-wave", help="emit one gated descriptor without dispatch")
    build.add_argument("--study-commit", required=True)
    build.add_argument("--output", type=Path)
    _add_prerequisite_arguments(build)
    run = sub.add_parser("run", help="run inside the claimed detached queue job")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--study-commit", required=True)
    run.add_argument("--job-dir", type=Path, required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--expected-role", required=True)
    run.add_argument("--implementation", nargs=2, action="append", metavar=("NAME", "SHA256"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run_job(args)
        return 0
    result = build_wave(
        study_commit=args.study_commit, prerequisite_inputs=_builder_inputs(args)
    )
    payload = queue.canonical_bytes(result)
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        target = Path(args.output)
        require(not target.exists() and not target.is_symlink(), "refusing to replace wave output")
        queue.immutable_bytes(target, payload, maximum_bytes=2 * 1024 * 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
