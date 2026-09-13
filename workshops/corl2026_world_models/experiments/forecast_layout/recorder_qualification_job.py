#!/usr/bin/env python3
"""Durable P00 RECORDER_ONLY qualification for the forecast-layout study.

The ``queue`` entry point runs under the worker image's system Python.  It
validates the immutable queue checkout, a complete accepted-fixture/frozen-pose
evidence chain, the clean pinned RoboLab checkout, and exactly one idle B200.
It then starts a new process group under the pinned RoboLab Python and records
PID, stdout, stderr, exit, and compact publication receipts around the raw PVC
attempt.

The ``record`` entry point constructs a timeout-only workshop task and executes
450 joint-position hold actions through :class:`ForecastRecordingAdapter`.
This is deliberately a recorder qualification, not a policy episode: no model
is imported, loaded, or queried.  The reset and every post-action observation
are retained losslessly.  Camera times and frame identities come only from
native Isaac sensor counters.  If those counters are unavailable, the attempt
is technical-invalid and an explicit timing-support receipt is retained; this
module never derives physical time from action or video-frame numbers.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


NAMESPACE = "wmf_ablation_001_20260912"
LAYOUT_PAIR_ID = "P00"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
ROBOLAB_ROOT = Path("/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241")
ROBOLAB_PYTHON = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
ROBOLAB_ENV_ROOT = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50")
RAW_ROOT = Path("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/recording_qualification")
SOURCE_CONTRACT_SHA256 = "88a1268ae7f27776fd5246a5808c069b2906399e99bb0e7d8b6f09020a6b85d3"
CANDIDATE_POOL_SHA256 = "ec80f4adc5272ec666c94d753241b84c954ef382bd30e1f2907459bff1cdb2b1"
RECORDING_CONTRACT_SHA256 = "6244ef873da6f50080d6d3e6e4146cb44c05b751cda38a3047970cf5fbd1dd44"
GATE_RECEIPT_SCHEMA = "wmf-forecast-layout-fixture-job-receipt-v1"
GATE_RECORD_SCHEMA = "wmf-forecast-layout-gate-ledger-record-v1"
GATE_ATTEMPT_SCHEMA = "wmf-forecast-layout-gate-attempt-receipt-v1"
POSE_MANIFEST_SCHEMA = "wmf-forecast-layout-frozen-pose-manifest-v1"
QUEUE_RECEIPT_SCHEMA = "wmf-forecast-recorder-qualification-job-v1"
CHILD_RECEIPT_SCHEMA = "wmf-forecast-recorder-qualification-child-v1"
TIMING_SCHEMA = "wmf-forecast-native-timing-support-v1"
PROCESS_SCHEMA = "wmf-forecast-recorder-child-process-v1"
EXIT_SCHEMA = "wmf-forecast-recorder-child-exit-v1"
FAILURE_SCHEMA = "wmf-forecast-recorder-qualification-failure-v1"
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
MAX_PUBLISH_BYTES = 128 * 1024
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    ("original", "left"): ("original_left.py", "WMFForecastOriginalLeftTask"),
    ("original", "right"): ("original_right.py", "WMFForecastOriginalRightTask"),
    ("reflected", "left"): ("reflected_left.py", "WMFForecastReflectedLeftTask"),
    ("reflected", "right"): ("reflected_right.py", "WMFForecastReflectedRightTask"),
}
REQUIRED_CAMERAS = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
ISAACLAB_SOURCE_ROOTS = tuple(
    ROBOLAB_ENV_ROOT / "lib/python3.11/site-packages/isaaclab/source" / name
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_rl")
)
NATIVE_LIBRARY_PATH = os.pathsep.join(
    (
        "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu",
        "/data/users/ali/glvnd/lib",
        "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib",
        "/usr/lib/x86_64-linux-gnu",
    )
)


class RecorderQualificationError(RuntimeError):
    """Fail-closed qualification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecorderQualificationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecorderQualificationError("value is not finite canonical JSON") from error


def compact_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise RecorderQualificationError("value is not finite compact JSON") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RecorderQualificationError(f"cannot hash evidence file: {path}") from error
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    require(resolved.is_file(), f"evidence file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _immutable_write(path: Path, payload: bytes, *, maximum_bytes: int | None = None) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing to replace evidence: {path}")
    if maximum_bytes is not None:
        require(len(payload) <= maximum_bytes, f"evidence exceeds {maximum_bytes} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.parent.is_symlink(), f"evidence directory is a symlink: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RecorderQualificationError(f"refusing to replace evidence: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any], *, publish: bool = False) -> None:
    _immutable_write(
        path,
        canonical_bytes(value),
        maximum_bytes=MAX_PUBLISH_BYTES if publish else None,
    )


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise RecorderQualificationError(f"non-finite JSON token: {value}")


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except RecorderQualificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecorderQualificationError(f"cannot read JSON evidence: {path}") from error
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value, payload


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def verify_file_descriptor(value: Any, *, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    path_value = value.get("path")
    require(isinstance(path_value, str) and Path(path_value).is_absolute(), f"{label} path must be absolute")
    observed = file_identity(Path(path_value))
    require(observed["sha256"] == value.get("sha256"), f"{label} hash mismatch")
    if "bytes" in value:
        require(observed["bytes"] == value.get("bytes"), f"{label} size mismatch")
    return observed


def _read_gate_ledger(path: Path) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_bytes().splitlines()
    except OSError as error:
        raise RecorderQualificationError("accepted gate ledger is unreadable") from error
    require(bool(lines), "accepted gate ledger is empty")
    output: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, line in enumerate(lines):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_duplicate_safe_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecorderQualificationError("accepted gate ledger is invalid JSONL") from error
        require(isinstance(row, dict), "accepted gate ledger row is not an object")
        require(row.get("schema_version") == GATE_RECORD_SCHEMA, "accepted gate ledger schema changed")
        require(row.get("study_namespace") == NAMESPACE, "accepted gate ledger namespace changed")
        require(row.get("sequence") == sequence, "accepted gate ledger sequence changed")
        require(row.get("previous_record_sha256") == previous, "accepted gate ledger chain broke")
        observed = require_sha256(row.get("record_sha256"), "accepted gate record digest")
        core = dict(row)
        core.pop("record_sha256")
        require(sha256_bytes(canonical_bytes(core)) == observed, "accepted gate record hash mismatch")
        previous = observed
        output.append(row)
    return output


def verify_fixture_release(
    *,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
    layout_arm: str,
) -> dict[str, Any]:
    """Verify accepted P00 evidence through the selected immutable pose row."""

    require(layout_arm in {"original", "reflected"}, "layout arm is invalid")
    require_sha256(gate_receipt_sha256, "gate receipt argument")
    require_sha256(pose_manifest_sha256, "pose manifest argument")
    gate_path = Path(gate_receipt_path).resolve()
    pose_path = Path(pose_manifest_path).resolve()
    require(not gate_path.is_symlink() and not pose_path.is_symlink(), "release evidence cannot be a symlink")
    gate, gate_payload = load_json(gate_path)
    pose, pose_payload = load_json(pose_path)
    require(sha256_bytes(gate_payload) == gate_receipt_sha256, "gate receipt argument hash mismatch")
    require(sha256_bytes(pose_payload) == pose_manifest_sha256, "pose manifest argument hash mismatch")

    require(gate.get("schema_version") == GATE_RECEIPT_SCHEMA, "fixture gate receipt schema changed")
    require(gate.get("study_namespace") == NAMESPACE, "fixture gate namespace changed")
    require(gate.get("status") == "finished", "fixture gate job is unfinished")
    require(gate.get("layout_pair_id") == LAYOUT_PAIR_ID, "fixture gate is not P00")
    require(gate.get("decision") == "accepted" and gate.get("exit_code") == 0, "P00 fixture was not accepted")
    require(gate.get("model_request_count") == 0, "fixture gate contacted a model")
    require(gate.get("behavioral_action_count") == 0, "fixture gate contains behavioral work")
    require(gate.get("source_contract_sha256") == SOURCE_CONTRACT_SHA256, "fixture source contract changed")
    require(gate.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "fixture candidate pool changed")
    candidate_id = gate.get("candidate_id")
    candidate_sha = require_sha256(gate.get("candidate_payload_sha256"), "accepted candidate digest")
    require(isinstance(candidate_id, str) and candidate_id.startswith("P00__candidate_"), "accepted candidate is not P00")

    require(pose.get("schema_version") == POSE_MANIFEST_SCHEMA, "pose manifest schema changed")
    require(pose.get("study_namespace") == NAMESPACE, "pose manifest namespace changed")
    require(
        pose.get("status") == "LIVE_GATE_QUALIFIED_POSES_FROZEN_MODEL_EXECUTION_NOT_RELEASED",
        "pose manifest is not live-qualified and frozen",
    )
    require(pose.get("physical_layout_gate_passed") is True, "pose manifest lacks passed physical gate")
    require(pose.get("released_for_model_inference") is False, "pose manifest improperly releases inference")
    require(pose.get("model_request_count") == 0 and pose.get("behavioral_episode_count") == 0, "pose manifest contains model evidence")
    require(pose.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "pose manifest candidate pool changed")
    task = pose.get("task_contract")
    require(isinstance(task, Mapping), "pose manifest lacks task contract")
    require(task.get("action_cap") == 450, "pose manifest action cap changed")
    require(task.get("termination_terms") == ["time_out"], "pose manifest is not timeout-only")
    require(task.get("success_is_measurement_only") is True, "pose manifest success contract changed")
    require(task.get("success_termination_present") is False, "pose manifest retains success termination")
    rows = pose.get("layout_pairs")
    require(isinstance(rows, Mapping), "pose manifest layout rows are invalid")
    row = rows.get(LAYOUT_PAIR_ID)
    require(isinstance(row, Mapping), "pose manifest lacks P00")
    require(row.get("layout_pair_id") == LAYOUT_PAIR_ID, "P00 pose row ID changed")
    require(row.get("candidate_id") == candidate_id, "pose manifest selected another candidate")
    require(row.get("candidate_payload_sha256") == candidate_sha, "pose manifest candidate digest changed")
    layouts = row.get("layouts")
    require(isinstance(layouts, Mapping) and layout_arm in layouts, "pose manifest lacks selected arm")

    gate_evidence = gate.get("gate_evidence")
    require(isinstance(gate_evidence, Mapping), "fixture receipt lacks gate evidence")
    record_sha = require_sha256(gate_evidence.get("gate_record_sha256"), "accepted gate record digest")
    require(row.get("accepted_gate_record_sha256") == record_sha, "pose manifest gate record changed")
    ledger_identity = verify_file_descriptor(gate.get("gate_ledger"), label="accepted gate ledger")
    require(pose.get("gate_ledger_sha256") == ledger_identity["sha256"], "pose manifest gate-ledger hash changed")
    records = _read_gate_ledger(Path(ledger_identity["path"]))
    matches = [
        record
        for record in records
        if record.get("record_sha256") == record_sha
        and record.get("layout_pair_id") == LAYOUT_PAIR_ID
    ]
    require(len(matches) == 1, "accepted P00 gate record is absent or duplicated")
    record = matches[0]
    require(record.get("decision") == "accepted" and record.get("passed") is True, "P00 gate record did not pass")
    require(record.get("candidate_id") == candidate_id, "gate record candidate changed")
    require(record.get("candidate_payload_sha256") == candidate_sha, "gate record candidate digest changed")
    require(record.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "gate record candidate pool changed")
    require(record.get("model_request_count") == 0 and record.get("behavioral_action_count") == 0, "gate record contains model evidence")
    attempt_identity = verify_file_descriptor(record.get("attempt_receipt"), label="accepted gate attempt")
    gate_attempt_identity = verify_file_descriptor(gate_evidence.get("gate_attempt_receipt"), label="fixture receipt gate attempt")
    pose_attempt_identity = verify_file_descriptor(row.get("accepted_gate_attempt_receipt"), label="pose manifest gate attempt")
    require(attempt_identity == gate_attempt_identity == pose_attempt_identity, "release chain references different gate attempts")
    attempt, _ = load_json(Path(attempt_identity["path"]))
    require(attempt.get("schema_version") == GATE_ATTEMPT_SCHEMA, "gate attempt schema changed")
    require(attempt.get("candidate_id") == candidate_id, "gate attempt candidate changed")
    require(attempt.get("candidate_payload_sha256") == candidate_sha, "gate attempt candidate digest changed")
    require(attempt.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "gate attempt candidate pool changed")
    require(attempt.get("decision") == "accepted" and attempt.get("passed") is True, "gate attempt is not accepted")
    require(attempt.get("model_request_count") == 0 and attempt.get("behavioral_action_count") == 0, "gate attempt contains model evidence")
    evaluation = attempt.get("evaluation")
    require(isinstance(evaluation, Mapping) and evaluation.get("passed") is True, "gate attempt evaluation failed")
    return {
        "gate_receipt": file_identity(gate_path),
        "pose_manifest": file_identity(pose_path),
        "gate_ledger": ledger_identity,
        "gate_attempt_receipt": attempt_identity,
        "accepted_gate_record_sha256": record_sha,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": candidate_sha,
        "pose_row": dict(row),
    }


def _run_git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RecorderQualificationError("Git verification is unavailable") from error
    require(result.returncode == 0, f"Git verification failed: {' '.join(arguments)}")
    return result.stdout


def verify_clean_git(root: Path, expected_commit: str, label: str) -> None:
    require(COMMIT_RE.fullmatch(expected_commit) is not None, f"{label} expected commit is invalid")
    root = Path(root).resolve()
    require(root.is_dir(), f"{label} checkout is missing")
    require(_run_git(root, "rev-parse", "HEAD").strip() == expected_commit, f"{label} commit mismatch")
    require(not _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"), f"{label} checkout is dirty")


def validate_queue_paths(source_root: Path, state_dir: Path, job_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    try:
        source = Path(source_root).resolve(strict=True)
        state = Path(state_dir).resolve(strict=True)
        job = Path(job_dir).resolve(strict=True)
    except OSError as error:
        raise RecorderQualificationError("queue path is missing") from error
    require(source.is_dir() and state.is_dir() and job.is_dir(), "queue paths must be directories")
    require(source.parent == state / "sources", "source root is outside queue state")
    require(job.parent == state / "jobs", "job directory is outside queue state")
    require(SAFE_ID_RE.fullmatch(job.name) is not None, "queue job ID is unsafe")
    descriptor, _ = load_json(job / "descriptor.json")
    require(descriptor.get("schema_version") == "wmf-cluster-job-v1", "queue descriptor schema changed")
    require(descriptor.get("namespace") == NAMESPACE, "queue descriptor namespace changed")
    require(descriptor.get("job_id") == job.name, "queue descriptor job ID changed")
    require(descriptor.get("released") is True, "queue descriptor is not released")
    commit = descriptor.get("source_commit")
    require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None, "queue source commit is invalid")
    require(source.name == commit, "queue source path does not match descriptor commit")
    return source, state, job, descriptor


def _query_nvidia(columns: Sequence[str], *, compute: bool, executable: Path | str = "nvidia-smi") -> list[dict[str, str]]:
    prefix = "--query-compute-apps=" if compute else "--query-gpu="
    try:
        result = subprocess.run(
            [str(executable), prefix + ",".join(columns), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RecorderQualificationError("nvidia-smi query is unavailable") from error
    require(result.returncode == 0, "nvidia-smi query failed")
    rows: list[dict[str, str]] = []
    for values in csv.reader(io.StringIO(result.stdout)):
        if not values or not any(item.strip() for item in values):
            continue
        require(len(values) == len(columns), "nvidia-smi query shape changed")
        rows.append({key: value.strip() for key, value in zip(columns, values)})
    return rows


def verify_idle_b200(executable: Path | str = "nvidia-smi") -> dict[str, Any]:
    devices = _query_nvidia(("index", "uuid", "name", "driver_version"), compute=False, executable=executable)
    require(len(devices) == 1, "recorder worker must expose exactly one assigned GPU")
    device = devices[0]
    require(device["index"] == "0" and device["name"] == "NVIDIA B200", "assigned device is not one index-0 B200")
    require(device["uuid"].startswith("GPU-"), "assigned GPU UUID is invalid")
    processes = _query_nvidia(
        ("gpu_uuid", "pid", "process_name", "used_memory"), compute=True, executable=executable
    )
    require(not processes, "assigned B200 has a pre-existing compute process")
    return {**device, "preexisting_compute_process_count": 0}


def _runtime_directories(state_parent: Path, hostname: str) -> dict[str, Path]:
    safe_host = hostname if SAFE_ID_RE.fullmatch(hostname) else sha256_bytes(hostname.encode())[:24]
    root = (state_parent / "worker_runtime" / "recorder_qualification" / safe_host).resolve()
    require(root.is_relative_to(state_parent.resolve()), "runtime directory escaped queue state")
    output = {
        "tmp": root / "tmp",
        "xdg": root / "xdg",
        "torch": root / "torch",
        "hf": root / "huggingface",
        "numba": root / "numba",
        "mpl": root / "matplotlib",
        "cuda": root / "cuda",
        "warp": root / "warp",
        "pycache": root / "pycache",
    }
    for directory in output.values():
        directory.mkdir(parents=True, exist_ok=True)
        require(directory.is_dir() and os.access(directory, os.W_OK), "runtime cache directory is not writable")
    return output


def build_child_environment(
    *,
    source_root: Path,
    state_parent: Path,
    robolab_root: Path = ROBOLAB_ROOT,
    hostname: str | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the exact headless Isaac environment observed to load on GM."""

    environment = dict(os.environ if base is None else base)
    for name in ("DISPLAY", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES"):
        environment.pop(name, None)
    directories = _runtime_directories(Path(state_parent).resolve(), hostname or socket.gethostname())
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        OMNI_KIT_ACCEPT_EULA="YES",
        VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
        LD_LIBRARY_PATH=NATIVE_LIBRARY_PATH,
        TMPDIR=str(directories["tmp"]),
        XDG_CACHE_HOME=str(directories["xdg"]),
        TORCH_HOME=str(directories["torch"]),
        HF_HOME=str(directories["hf"]),
        NUMBA_CACHE_DIR=str(directories["numba"]),
        MPLCONFIGDIR=str(directories["mpl"]),
        CUDA_CACHE_PATH=str(directories["cuda"]),
        WARP_CACHE_PATH=str(directories["warp"]),
        PYTHONPYCACHEPREFIX=str(directories["pycache"]),
    )
    forecast = Path(source_root).resolve() / "workshops/corl2026_world_models/experiments/forecast_layout"
    roots = [str(forecast), str(Path(source_root).resolve()), str(Path(robolab_root).resolve())]
    roots.extend(str(path) for path in ISAACLAB_SOURCE_ROOTS)
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def build_child_command(
    *,
    source_root: Path,
    output_dir: Path,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
    study_commit: str,
    layout_arm: str,
    command: str,
    environment_seed: int,
    attempt_id: str,
    robolab_root: Path = ROBOLAB_ROOT,
    robolab_python: Path = ROBOLAB_PYTHON,
) -> list[str]:
    script = Path(source_root) / "workshops/corl2026_world_models/experiments/forecast_layout/recorder_qualification_job.py"
    return [
        # Do not resolve this lexical venv path.  ``bin/python`` is normally a
        # symlink to base CPython; executing its resolved target loses the
        # venv's site-packages and has already failed live on GM.
        os.path.abspath(os.fspath(robolab_python)),
        str(script.resolve()),
        "record",
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--robolab-root", str(Path(robolab_root).resolve()),
        "--robolab-commit", ROBOLAB_COMMIT,
        "--gate-receipt", str(Path(gate_receipt_path).resolve()),
        "--gate-receipt-sha256", gate_receipt_sha256,
        "--pose-manifest", str(Path(pose_manifest_path).resolve()),
        "--pose-manifest-sha256", pose_manifest_sha256,
        "--layout-pair-id", LAYOUT_PAIR_ID,
        "--layout-arm", layout_arm,
        "--command", command,
        "--environment-seed", str(environment_seed),
        "--attempt-id", attempt_id,
        "--output-dir", str(Path(output_dir).resolve()),
    ]


def _existing_queue_receipt(path: Path, *, job_id: str, source_commit: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value, payload = load_json(path)
    require(len(payload) <= MAX_PUBLISH_BYTES, "existing publish receipt is too large")
    require(value.get("schema_version") == QUEUE_RECEIPT_SCHEMA, "existing publish receipt schema changed")
    require(value.get("job_id") == job_id and value.get("study_commit") == source_commit, "existing publish receipt identity changed")
    require(value.get("exit_code") in (0, 3), "existing publish receipt exit code changed")
    return value


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _run_detached_child(
    *,
    command_argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    attempt_root: Path,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    stdout_path = attempt_root / "child.stdout.log"
    stderr_path = attempt_root / "child.stderr.log"
    launch_path = attempt_root / "launcher_identity.json"
    immutable_json(
        launch_path,
        {
            "schema_version": "wmf-forecast-recorder-launcher-v1",
            "recorded_at_utc": utc_now(),
            "hostname": socket.gethostname(),
            "launcher_pid": os.getpid(),
            "command_argv": list(command_argv),
            "cwd": str(Path(cwd).resolve()),
            "start_new_session": True,
            "durability_boundary": "Child has an independent process group; queue worker remains its durable monitor.",
        },
    )
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        child = subprocess.Popen(
            list(command_argv),
            cwd=Path(cwd).resolve(),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        process = {
            "schema_version": PROCESS_SCHEMA,
            "recorded_at_utc": utc_now(),
            "hostname": socket.gethostname(),
            "pid": child.pid,
            "process_group_id": child.pid,
            "proc_start_ticks": _process_start_ticks(child.pid),
            "start_new_session": True,
            "launcher_identity": file_identity(launch_path),
        }
        process_path = attempt_root / "child_process.json"
        immutable_json(process_path, process)
        return_code = child.wait()
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    logs = {"stdout": file_identity(stdout_path), "stderr": file_identity(stderr_path)}
    exit_receipt = {
        "schema_version": EXIT_SCHEMA,
        "recorded_at_utc": utc_now(),
        "pid": process["pid"],
        "process_group_id": process["process_group_id"],
        "proc_start_ticks": process["proc_start_ticks"],
        "return_code": return_code,
        "normal_exit": return_code >= 0,
        "terminating_signal": -return_code if return_code < 0 else None,
        "process_identity": file_identity(process_path),
        "logs": logs,
    }
    exit_path = attempt_root / "child_exit.json"
    immutable_json(exit_path, exit_receipt)
    return return_code, process, {**exit_receipt, "identity": file_identity(exit_path)}


def execute_queue_job(
    *,
    source_root: Path,
    state_dir: Path,
    job_dir: Path,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
    layout_arm: str,
    command: str,
    environment_seed: int = 2026091000,
    raw_root: Path = RAW_ROOT,
    robolab_root: Path = ROBOLAB_ROOT,
    robolab_python: Path = ROBOLAB_PYTHON,
    nvidia_smi: Path | str = "nvidia-smi",
) -> dict[str, Any]:
    source, state, job, descriptor = validate_queue_paths(source_root, state_dir, job_dir)
    source_commit = descriptor["source_commit"]
    publish = job / "publish"
    publish.mkdir(exist_ok=True)
    publish_target = publish / "recorder_qualification_receipt.json"
    existing = _existing_queue_receipt(publish_target, job_id=job.name, source_commit=source_commit)
    if existing is not None:
        return dict(existing, idempotent_replay=True)
    context: dict[str, Any] = {
        "gpu": None,
        "release": None,
        "attempt_root": None,
        "child_started": False,
        "child_exit": None,
    }
    try:
        require(layout_arm in {"original", "reflected"}, "layout arm is invalid")
        require(command in PROMPTS, "command is invalid")
        require(type(environment_seed) is int and environment_seed >= 0, "environment seed is invalid")
        verify_clean_git(source, source_commit, "study")
        verify_clean_git(Path(robolab_root), ROBOLAB_COMMIT, "RoboLab")
        require(Path(robolab_python).resolve().is_file(), "pinned RoboLab Python is missing")
        contract_path = source / "workshops/corl2026_world_models/experiments/forecast_layout/recording_contract.json"
        require(sha256_file(contract_path) == RECORDING_CONTRACT_SHA256, "recording contract hash changed")
        context["release"] = verify_fixture_release(
            gate_receipt_path=gate_receipt_path,
            gate_receipt_sha256=gate_receipt_sha256,
            pose_manifest_path=pose_manifest_path,
            pose_manifest_sha256=pose_manifest_sha256,
            layout_arm=layout_arm,
        )
        context["gpu"] = verify_idle_b200(nvidia_smi)
        raw_root = Path(raw_root).resolve()
        raw_root.mkdir(parents=True, exist_ok=True)
        require(raw_root.is_dir() and not raw_root.is_symlink(), "recorder raw root is invalid")
        attempt_root = (raw_root / job.name).resolve()
        require(attempt_root.parent == raw_root, "recorder attempt escaped raw root")
        context["attempt_root"] = str(attempt_root)
        lock_root = raw_root / ".locks"
        lock_root.mkdir(exist_ok=True)
        lock_path = lock_root / f"{LAYOUT_PAIR_ID}-{layout_arm}-{command}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not attempt_root.exists(), "immutable recorder attempt already exists without publish receipt")
            attempt_root.mkdir()
            attempt_id = f"{job.name}-{source_commit[:12]}"
            child_output = attempt_root / "adapter_attempt"
            command_argv = build_child_command(
                source_root=source,
                output_dir=child_output,
                gate_receipt_path=gate_receipt_path,
                gate_receipt_sha256=gate_receipt_sha256,
                pose_manifest_path=pose_manifest_path,
                pose_manifest_sha256=pose_manifest_sha256,
                study_commit=source_commit,
                layout_arm=layout_arm,
                command=command,
                environment_seed=environment_seed,
                attempt_id=attempt_id,
                robolab_root=robolab_root,
                robolab_python=robolab_python,
            )
            child_environment = build_child_environment(
                source_root=source,
                state_parent=state.parent,
                robolab_root=robolab_root,
            )
            context["child_started"] = True
            return_code, process, exit_receipt = _run_detached_child(
                command_argv=command_argv,
                cwd=Path(robolab_root),
                environment=child_environment,
                attempt_root=attempt_root,
            )
            context["child_exit"] = exit_receipt
        require(return_code == 0, "RoboLab recorder child failed")
        child_receipt_path = attempt_root / "child_qualification_receipt.json"
        child_receipt, _ = load_json(child_receipt_path)
        require(child_receipt.get("schema_version") == CHILD_RECEIPT_SCHEMA, "child receipt schema changed")
        require(child_receipt.get("status") == "passed", "child recorder qualification did not pass")
        require(child_receipt.get("study_commit") == source_commit, "child used another study commit")
        require(child_receipt.get("actions_executed") == 450, "child did not execute 450 actions")
        require(child_receipt.get("observation_count") == 451, "child did not retain 451 observations")
        require(child_receipt.get("model_request_count") == 0, "recorder child contacted a model")
        require(child_receipt.get("behavioral_episode_count") == 0, "recorder-only child claimed a behavioral episode")
        require(child_receipt.get("recording_qualification_count") == 1, "recorder child count changed")
        require(child_receipt.get("native_timing_supported") is True, "recorder child lacks native timing")
        receipt = {
            "schema_version": QUEUE_RECEIPT_SCHEMA,
            "study_namespace": NAMESPACE,
            "status": "passed",
            "exit_code": 0,
            "job_id": job.name,
            "study_commit": source_commit,
            "layout_pair_id": LAYOUT_PAIR_ID,
            "layout_arm": layout_arm,
            "command": command,
            "environment_seed": environment_seed,
            "candidate_id": context["release"]["candidate_id"],
            "candidate_payload_sha256": context["release"]["candidate_payload_sha256"],
            "accepted_gate_record_sha256": context["release"]["accepted_gate_record_sha256"],
            "gate_receipt_sha256": gate_receipt_sha256,
            "pose_manifest_sha256": pose_manifest_sha256,
            "gpu_identity": context["gpu"],
            "child_process": process,
            "child_exit_receipt": exit_receipt["identity"],
            "child_logs": exit_receipt["logs"],
            "raw_attempt_directory": str(attempt_root),
            "raw_child_receipt": file_identity(child_receipt_path),
            "adapter_completion": child_receipt["adapter_completion"],
            "adapter_journal": child_receipt["adapter_journal"],
            "timing_support": child_receipt["timing_support"],
            "actions_executed": 450,
            "observation_count": 451,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "recording_qualification_count": 1,
            "finished_at_utc": utc_now(),
            "claim_boundary": "Passed recorder-only simulator qualification using joint-position holds. This is not a learned-policy episode and contains no model request or forecast evidence.",
        }
    except Exception as error:
        timing_identity = None
        if context["attempt_root"] is not None:
            timing_path = Path(context["attempt_root"]) / "adapter_attempt" / "timing_support.json"
            if timing_path.is_file():
                timing_identity = file_identity(timing_path)
        receipt = {
            "schema_version": QUEUE_RECEIPT_SCHEMA,
            "study_namespace": NAMESPACE,
            "status": "technical_invalid",
            "exit_code": 3,
            "job_id": job.name,
            "study_commit": source_commit,
            "layout_pair_id": LAYOUT_PAIR_ID,
            "layout_arm": layout_arm,
            "command": command,
            "environment_seed": environment_seed,
            "reason": type(error).__name__,
            "detail": str(error)[:2000],
            "gpu_identity": context["gpu"],
            "child_started": context["child_started"],
            "child_exit": context["child_exit"],
            "raw_attempt_directory": context["attempt_root"],
            "timing_support": timing_identity,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "recording_qualification_count": 0,
            "finished_at_utc": utc_now(),
            "claim_boundary": "Technical-invalid recorder-only attempt. No learned-policy or forecast claim is made.",
        }
    immutable_json(publish_target, receipt, publish=True)
    return receipt


def _to_numpy(value: Any, *, env_id: int | None = None) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if env_id is not None:
        require(array.ndim >= 1 and array.shape[0] > env_id, "simulator value lacks requested environment")
        array = array[env_id]
    array = np.ascontiguousarray(array)
    require(not array.dtype.hasobject, "simulator value has object dtype")
    if array.dtype.kind in "fc":
        require(bool(np.isfinite(array).all()), "simulator value contains non-finite data")
    return array.copy()


def _native_scalar(value: Any, *, index: int | None = None) -> int | float:
    import numpy as np

    if callable(value):
        value = value()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if index is not None and array.ndim:
        require(array.shape[0] > index, "native counter lacks requested environment")
        array = array[index]
    require(array.size == 1, "native counter is not scalar")
    result = array.reshape(-1)[0].item()
    require(isinstance(result, (int, float)) and math.isfinite(float(result)), "native counter is non-finite")
    return result


def _native_value(
    root: Any,
    names: Sequence[str],
    *,
    index: int | None,
    label: str,
) -> tuple[int | float, str]:
    for name in names:
        current = root
        try:
            for component in name.split("."):
                current = getattr(current, component)
            return _native_scalar(current, index=index), name
        except (AttributeError, RecorderQualificationError, TypeError, ValueError):
            continue
    raise RecorderQualificationError(f"native runtime does not expose {label}")


def _seconds_to_ns(value: int | float) -> int:
    try:
        decimal = Decimal(str(value)) * Decimal(1_000_000_000)
    except (InvalidOperation, ValueError) as error:
        raise RecorderQualificationError("native timestamp cannot be converted to nanoseconds") from error
    result = int(decimal.to_integral_value())
    require(result >= 0, "native timestamp is negative")
    return result


class NativeClockSampler:
    """Strict native-clock adapter; never invents a missing camera identity."""

    def __init__(self, timing_receipt_path: Path, env_cfg: Any, *, env_id: int = 0) -> None:
        self.path = Path(timing_receipt_path)
        self.env_cfg = env_cfg
        self.env_id = env_id
        self.control_baseline: int | None = None
        self.sources: dict[str, Any] | None = None
        self.previous: dict[str, tuple[int, int]] = {}

    def _write_once(self, value: Mapping[str, Any]) -> None:
        if not self.path.exists():
            immutable_json(self.path, value)

    def _unsupported(self, error: BaseException, phase: str, expected_step: int) -> None:
        self._write_once(
            {
                "schema_version": TIMING_SCHEMA,
                "status": "unsupported",
                "supported": False,
                "recorded_at_utc": utc_now(),
                "phase": phase,
                "expected_control_step": expected_step,
                "reason": type(error).__name__,
                "detail": str(error),
                "attempted_native_sources": {
                    "physics_step": ["sim.current_time_step_index", "sim.frame_count", "sim._frame_count"],
                    "physics_time": ["sim.current_time", "sim._current_time"],
                    "control": ["env.common_step_counter", "env._common_step_counter"],
                    "camera_frame": ["sensor.data.frame", "sensor.frame", "sensor._frame"],
                    "camera_timestamp": ["sensor.data.timestamp", "sensor.timestamp", "sensor._timestamp"],
                },
                "forecast_alignment_inferred": False,
                "action_number_used_as_camera_time": False,
                "video_frame_number_used_as_action_number": False,
                "qualification_boundary": "Native camera timing is unsupported; the recorder cannot qualify physical forecast alignment.",
            }
        )

    def __call__(self, env: Any, phase: str, expected_step: int) -> dict[str, Any]:
        try:
            physics_step, physics_step_source = _native_value(
                env.sim, ("current_time_step_index", "frame_count", "_frame_count"), index=None, label="physics frame counter"
            )
            physics_time, physics_time_source = _native_value(
                env.sim, ("current_time", "_current_time"), index=None, label="physics time"
            )
            control, control_source = _native_value(
                env, ("common_step_counter", "_common_step_counter"), index=None, label="control counter"
            )
            require(float(physics_step).is_integer() and physics_step >= 0, "physics frame counter is invalid")
            require(float(physics_time) >= 0, "physics time is negative")
            require(float(control).is_integer() and control >= 0, "control counter is invalid")
            control = int(control)
            if self.control_baseline is None:
                require(expected_step == 0, "native control baseline was not sampled at settled reset")
                self.control_baseline = control
            relative_control = control - self.control_baseline
            require(relative_control == expected_step, "native control counter disagrees with executed-action schedule")
            episode_step, episode_source = _native_value(
                env, ("episode_length_buf",), index=self.env_id, label="episode counter"
            )
            require(float(episode_step).is_integer() and episode_step >= 0, "native episode counter is invalid")
            cameras: dict[str, Any] = {}
            source_rows: dict[str, Any] = {}
            for name in REQUIRED_CAMERAS:
                sensor = env.scene[name]
                frame, frame_source = _native_value(
                    sensor, ("data.frame", "frame", "_frame"), index=self.env_id,
                    label=f"{name} frame counter",
                )
                timestamp, timestamp_source = _native_value(
                    sensor, ("data.timestamp", "timestamp", "_timestamp"), index=self.env_id,
                    label=f"{name} capture timestamp",
                )
                require(float(frame).is_integer() and frame >= 0, f"{name} frame counter is invalid")
                require(float(timestamp) >= 0, f"{name} timestamp is invalid")
                frame_int = int(frame)
                timestamp_ns = _seconds_to_ns(timestamp)
                if name in self.previous:
                    previous_frame, previous_time = self.previous[name]
                    require(frame_int >= previous_frame, f"{name} native frame counter moved backward")
                    require(timestamp_ns >= previous_time, f"{name} native timestamp moved backward")
                self.previous[name] = (frame_int, timestamp_ns)
                cameras[name] = {
                    "frame_id": frame_int,
                    "capture_time_ns": timestamp_ns,
                    "timestamp_source": f"Isaac sensor native {timestamp_source}",
                    "native_capture_time_s": float(timestamp),
                    "frame_id_source": frame_source,
                    "timestamp_domain": "Isaac sensor native simulation-time counter",
                    "capture_time_ns_derivation": "Decimal(str(native_capture_time_s))*1e9 rounded to nearest integer",
                }
                source_rows[name] = {"frame_id_source": frame_source, "timestamp_source": timestamp_source}
            sources = {
                "physics_step": physics_step_source,
                "physics_time": physics_time_source,
                "control": control_source,
                "episode": episode_source,
                "cameras": source_rows,
            }
            if self.sources is None:
                self.sources = sources
                self._write_once(
                    {
                        "schema_version": TIMING_SCHEMA,
                        "status": "supported",
                        "supported": True,
                        "recorded_at_utc": utc_now(),
                        "native_sources": sources,
                        "required_cameras": list(REQUIRED_CAMERAS),
                        "control_baseline": self.control_baseline,
                        "forecast_alignment_inferred": False,
                        "action_number_used_as_camera_time": False,
                        "video_frame_number_used_as_action_number": False,
                        "qualification_boundary": "Native capture identities are available; generated-future-to-physical-time mapping remains a separate qualification.",
                    }
                )
            else:
                require(sources == self.sources, "native timing source identity changed during episode")
            return {
                "physics_step": int(physics_step),
                "physics_time_s": float(physics_time),
                "control_step": relative_control,
                "cameras": cameras,
                "phase": phase,
                "native_control_counter": control,
                "native_control_baseline": self.control_baseline,
                "native_episode_step": int(episode_step),
                "native_sources": sources,
                "forecast_alignment_inferred": False,
            }
        except BaseException as error:
            self._unsupported(error, phase, expected_step)
            raise


def build_joint_position_hold(observation: Mapping[str, Any], device: str) -> Any:
    """Return one [1, 8] joint-position hold action, with no policy input."""

    import torch

    proprio = observation.get("proprio_obs")
    require(isinstance(proprio, Mapping), "joint hold requires proprio_obs")
    arm = proprio.get("arm_joint_pos")
    gripper = proprio.get("gripper_pos")
    require(arm is not None and gripper is not None, "joint hold proprio fields are missing")
    arm = arm.detach().to(device)
    gripper = gripper.detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    require(tuple(action.shape) == (1, 8), f"joint hold action shape changed: {tuple(action.shape)}")
    require(bool(torch.isfinite(action).all()), "joint hold action is non-finite")
    return action


def execute_recorder_only_actions(
    *,
    proxy: Any,
    recorder: Any,
    observation: Mapping[str, Any],
    hold_action_builder: Callable[[Mapping[str, Any], str], Any] = build_joint_position_hold,
) -> dict[str, Any]:
    """Exercise exactly 450 environment steps through the RECORDER_ONLY API."""

    current = observation
    for expected_step in range(1, 451):
        require(not proxy.all_terminated, f"timeout-only environment ended before action {expected_step}")
        action = hold_action_builder(current, proxy.device)
        proposed = _to_numpy(action, env_id=0)
        recorder.record_scripted_action(
            proposed,
            action_source={
                "kind": "joint_position_hold",
                "policy_model": None,
                "model_request": False,
                "scientific_claim": "recorder_qualification_only_nonbehavioral",
                "source_observation_id": recorder.current_observation["observation_id"],
                "action_step": expected_step,
            },
        )
        current, _reward, _terminated, _truncated, _info = proxy.step(action)
        require(recorder.actions_executed == expected_step, "recorder action counter changed")
    require(proxy.all_terminated, "action 450 did not trigger sole timeout")
    receipt = recorder._final_receipt
    require(isinstance(receipt, Mapping), "adapter did not write final receipt")
    require(receipt.get("recording_qualification_valid") is True, "adapter rejected recorder qualification")
    require(receipt.get("behavioral_result_valid") is False, "recorder qualification became behavioral evidence")
    require(receipt.get("actions_executed") == 450, "adapter action count changed")
    require(receipt.get("observation_count") == 451, "adapter observation count changed")
    require(receipt.get("request_count") == 0, "recorder qualification issued a model request")
    require(receipt.get("model_attached") is False, "recorder qualification attached a model")
    return dict(receipt)


def sample_simulator_state(env: Any) -> dict[str, Any]:
    """Retain post-action simulator state as measurement-only evidence."""

    from robolab.core.world.world_state import get_world

    world = get_world(env)
    objects: dict[str, Any] = {}
    for name in ("banana", "bowl", "rubiks_cube"):
        position, quaternion = world.get_pose(name, env_id=0)
        velocity = world.get_velocity(name, env_id=0)
        objects[name] = {
            "position_robot_base_m": _to_numpy(position),
            "quaternion_wxyz": _to_numpy(quaternion),
            "velocity_linear_angular": _to_numpy(velocity),
        }
    robot = env.scene["robot"]
    robot_state: dict[str, Any] = {}
    for attribute in ("root_pos_w", "root_quat_w", "joint_pos", "joint_vel"):
        value = getattr(robot.data, attribute, None)
        require(value is not None, f"robot state lacks {attribute}")
        robot_state[attribute] = _to_numpy(value, env_id=0)
    return {
        "simulator_state_sample_only_not_policy_input": True,
        "objects": objects,
        "robot": robot_state,
        "native_episode_length_buf": _to_numpy(env.episode_length_buf, env_id=0),
    }


def _fresh_physical_callbacks(evidence_root: Path, source_contract: Mapping[str, Any]):
    from robolab_fixture_gate_adapter import RoboLabFixtureGateAdapter
    from robolab.core.world.world_state import get_world

    evidence: dict[str, Any] = {}
    gate = source_contract["live_gate"]

    def collision_sampler(env: Any, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        del observation
        probe = object.__new__(RoboLabFixtureGateAdapter)
        rows = probe._collision_rows(get_world(env))
        passed = all(row.get("clear") is True for row in rows["forbidden_pairs"].values())
        result = {"passed": passed, **rows}
        evidence["collision"] = result
        return result

    def visibility_sampler(env: Any, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        target = Path(evidence_root) / "fresh_physical_camera_evidence"
        target.mkdir(parents=True, exist_ok=False)
        probe = object.__new__(RoboLabFixtureGateAdapter)
        probe.cv2 = cv2
        probe.np = np
        rows = probe._camera_rows(env, observation, get_world(env), target)
        nonblank = all(
            row.get("pixel_range", -1) >= int(gate["minimum_rgb_pixel_range"])
            for row in rows.values()
        )
        visible = all(
            rows[camera]["projected_unoccluded_by_object"].get(name) is True
            for camera in gate["visibility_cameras"]
            for name in ("banana", "bowl", "rubiks_cube")
        )
        result = {
            "passed": bool(nonblank and visible),
            "method": "accepted-gate calibrated projection and OBB occlusion implementation",
            "required_cameras": list(gate["required_cameras"]),
            "visibility_cameras": list(gate["visibility_cameras"]),
            "cameras": rows,
        }
        evidence["visibility"] = result
        return result

    return collision_sampler, visibility_sampler, evidence


def _pose_check(env: Any, configured: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    import numpy as np
    from robolab.core.world.world_state import get_world

    world = get_world(env)
    rows: dict[str, Any] = {}
    for name in ("banana", "bowl", "rubiks_cube"):
        position, quaternion = world.get_pose(name, env_id=0)
        observed_position = _to_numpy(position).reshape(-1).astype(np.float64)
        observed_quaternion = _to_numpy(quaternion).reshape(-1).astype(np.float64)
        expected = np.asarray(configured["positions_robot_base_m"][name], dtype=np.float64)
        require(observed_position.shape == (3,), f"settled {name} position shape changed")
        require(observed_quaternion.shape == (4,), f"settled {name} quaternion shape changed")
        maximum_error = float(np.max(np.abs(observed_position - expected)))
        require(maximum_error <= tolerance, f"settled {name} pose left P00 tolerance")
        rows[name] = {
            "configured_position_robot_base_m": expected.tolist(),
            "observed_position_robot_base_m": observed_position.tolist(),
            "observed_quaternion_wxyz": observed_quaternion.tolist(),
            "maximum_absolute_position_error_m": maximum_error,
        }
    return {"passed": True, "pose_tolerance_m": tolerance, "objects": rows}


def run_live_qualification(args: argparse.Namespace) -> dict[str, Any]:
    """Run inside the pinned RoboLab Python after launching Isaac headlessly."""

    source_root = Path(args.source_root).resolve()
    robolab_root = Path(args.robolab_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    forecast_root = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
    require(args.layout_pair_id == LAYOUT_PAIR_ID, "recorder qualification is restricted to P00")
    require((args.layout_arm, args.command) in TASKS, "recorder task identity is invalid")
    require(not output_dir.exists(), "recorder output directory already exists")
    verify_clean_git(source_root, args.study_commit, "study")
    require(args.robolab_commit == ROBOLAB_COMMIT, "RoboLab expected commit changed")
    verify_clean_git(robolab_root, args.robolab_commit, "RoboLab")
    require(sha256_file(forecast_root / "recording_contract.json") == RECORDING_CONTRACT_SHA256, "recording contract changed")
    release = verify_fixture_release(
        gate_receipt_path=args.gate_receipt,
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=args.pose_manifest,
        pose_manifest_sha256=args.pose_manifest_sha256,
        layout_arm=args.layout_arm,
    )
    if str(forecast_root) not in sys.path:
        sys.path.insert(0, str(forecast_root))
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    os.environ.update(
        WMF_FORECAST_POSE_MANIFEST=str(Path(args.pose_manifest).resolve()),
        WMF_FORECAST_POSE_MANIFEST_SHA256=args.pose_manifest_sha256,
        WMF_FORECAST_LAYOUT_PAIR_ID=LAYOUT_PAIR_ID,
    )

    import cv2  # noqa: F401 - required before Isaac/RoboLab imports in this environment.
    from isaaclab.app import AppLauncher

    launch_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launch_parser)
    launch_args, _ = launch_parser.parse_known_args(["--headless"])
    launch_args.enable_cameras = True
    launcher = AppLauncher(launch_args)
    simulation_app = launcher.app
    env = None
    recorder = None
    try:
        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        from fixture_tasks import settle_for_recording_reset, success_measurements
        from recording_adapter import (
            CONTEXT_RESET_SCOPE,
            RECORDER_ONLY_MODEL_CONFIG,
            FixedDurationEnvProxy,
            ForecastRecordingAdapter,
            assert_fixed_duration_environment,
            verify_journal,
        )

        require(Path(robolab.__file__).resolve().is_relative_to(robolab_root), "effective RoboLab import is outside pinned checkout")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        simulator_output = output_dir.parent / "native_simulator"
        require(not simulator_output.exists(), "native simulator output already exists")
        simulator_output.mkdir()
        set_output_dir(str(simulator_output))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = False
        task_file, task_name = TASKS[(args.layout_arm, args.command)]
        task_path = forecast_root / "task_files" / task_file
        auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        identity = {
            "attempt_id": args.attempt_id,
            "cell_id": f"wmf1__recording_qualification__P00__RECORDER_ONLY__{args.layout_arm}__{args.command}",
            "stage": "recording_qualification",
            "layout_pair_id": LAYOUT_PAIR_ID,
            "layout_arm": args.layout_arm,
            "command": args.command,
            "prompt": PROMPTS[args.command],
            "model_config": RECORDER_ONLY_MODEL_CONFIG,
            "effective_seed": args.environment_seed,
            "source_identity": f"study:{args.study_commit};robolab:{args.robolab_commit};pose:{args.pose_manifest_sha256}",
            "checkpoint_identity": "none_no_model_import_load_or_request",
        }
        recorder = ForecastRecordingAdapter(output_dir, identity)
        recorder.record_context_reset(
            {
                "passed": True,
                "reset_scope": CONTEXT_RESET_SCOPE,
                "server_context_id": f"recorder-only-no-server:{args.attempt_id}",
                "cache_reset_evidence": {
                    "no_model_attached": True,
                    "model_server_started": False,
                    "model_process_count": 0,
                    "model_request_count": 0,
                },
            }
        )
        env, env_cfg = create_env(
            task_name,
            device="cuda:0",
            seed=args.environment_seed,
            num_envs=1,
            instruction_type="default",
            policy="wmf_recorder_only_joint_position_hold_no_policy",
            renderer="realtime",
            rendering_mode="balanced",
        )
        # RoboLab copies runtime fields into env_cfg, not workshop task markers.
        # Validate the constructed timeout and action limit directly.
        assert_fixed_duration_environment(env, env_cfg)
        require(env_cfg.instruction == PROMPTS[args.command], "constructed task prompt changed")
        import torch

        require(torch.cuda.device_count() == 1, "RoboLab child does not see exactly one assigned CUDA device")
        require(torch.cuda.get_device_name(0) == "NVIDIA B200", "RoboLab child CUDA device is not B200")
        source_contract, source_payload = load_json(forecast_root / "layout_source_contract.json")
        require(sha256_bytes(source_payload) == SOURCE_CONTRACT_SHA256, "source contract changed")
        collision_sampler, visibility_sampler, physical_evidence = _fresh_physical_callbacks(
            output_dir.parent, source_contract
        )
        configured = release["pose_row"]["layouts"][args.layout_arm]
        tolerance = float(source_contract["live_gate"]["pose_tolerance_m"])

        def reset_attestor(live_env: Any, observation: Mapping[str, Any], info: Any):
            settled_observation, settled_info, attestation = settle_for_recording_reset(
                live_env,
                observation,
                info,
                pose_manifest_sha256=args.pose_manifest_sha256,
                reset_identity=f"recorder:{args.attempt_id}:{LAYOUT_PAIR_ID}:{args.layout_arm}:{args.command}",
                collision_sampler=collision_sampler,
                visibility_sampler=visibility_sampler,
                settle_steps=int(source_contract["live_gate"]["settle_steps"]),
                stability_window_steps=int(source_contract["live_gate"]["stability_window_steps"]),
                linear_speed_tolerance_m_s=float(source_contract["live_gate"]["linear_speed_tolerance_m_s"]),
                angular_speed_tolerance_rad_s=float(source_contract["live_gate"]["angular_speed_tolerance_rad_s"]),
            )
            attestation = dict(attestation)
            attestation["fresh_pose_check"] = _pose_check(live_env, configured, tolerance)
            attestation["accepted_gate_record_sha256"] = release["accepted_gate_record_sha256"]
            attestation["candidate_id"] = release["candidate_id"]
            return settled_observation, settled_info, attestation

        timing_path = output_dir / "timing_support.json"
        native_clock = NativeClockSampler(timing_path, env_cfg)
        proxy = FixedDurationEnvProxy(
            env,
            env_cfg,
            recorder,
            state_sampler=sample_simulator_state,
            clock_sampler=native_clock,
            success_sampler=success_measurements,
            reset_attestor=reset_attestor,
            env_id=0,
        )
        observation, _info = proxy.reset()
        duplicate, _duplicate_info = proxy.reset()
        require(duplicate is observation, "second runner reset did not return cached settled observation")
        adapter_receipt = execute_recorder_only_actions(
            proxy=proxy,
            recorder=recorder,
            observation=observation,
        )
        timing, _ = load_json(timing_path)
        require(timing.get("schema_version") == TIMING_SCHEMA and timing.get("supported") is True, "native timing did not qualify")
        journal = verify_journal(recorder.journal_path)
        require(journal["event_count"] == adapter_receipt["event_count"], "adapter journal event count changed")
        require(journal["tail_sha256"] == adapter_receipt["journal_tail_sha256"], "adapter journal tail changed")
        child_receipt = {
            "schema_version": CHILD_RECEIPT_SCHEMA,
            "study_namespace": NAMESPACE,
            "status": "passed",
            "completed_at_utc": utc_now(),
            "study_commit": args.study_commit,
            "robolab_commit": args.robolab_commit,
            "layout_pair_id": LAYOUT_PAIR_ID,
            "layout_arm": args.layout_arm,
            "command": args.command,
            "environment_seed": args.environment_seed,
            "attempt_id": args.attempt_id,
            "candidate_id": release["candidate_id"],
            "candidate_payload_sha256": release["candidate_payload_sha256"],
            "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
            "gate_receipt": release["gate_receipt"],
            "pose_manifest": release["pose_manifest"],
            "adapter_completion": file_identity(recorder.completion_path),
            "adapter_journal": {**file_identity(recorder.journal_path), **journal},
            "timing_support": file_identity(timing_path),
            "native_timing_supported": True,
            "actions_executed": adapter_receipt["actions_executed"],
            "observation_count": adapter_receipt["observation_count"],
            "model_request_count": adapter_receipt["request_count"],
            "behavioral_episode_count": 0,
            "recording_qualification_count": 1,
            "action_source": "joint_position_hold_from_each_preceding_original_proprioception",
            "model_attached": False,
            "fresh_physical_checks": physical_evidence,
            "runtime_identity": {
                "pod": socket.gethostname(),
                "pod_uid": os.environ.get("POD_UID"),
                "python": sys.version,
                "python_executable": sys.executable,
                "robolab_module": file_identity(Path(robolab.__file__)),
                "task_file": file_identity(task_path),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "renderer": "realtime",
                "rendering_mode": "balanced",
                "device": "cuda:0",
            },
            "claim_boundary": "This validates the fixed-duration recorder with scripted joint holds only. It is not a learned-policy episode and contains no generated future.",
        }
        immutable_json(output_dir.parent / "child_qualification_receipt.json", child_receipt)
        return child_receipt
    except BaseException as error:
        if recorder is not None and not recorder.finalized:
            recorder.mark_technical_failure("recorder_qualification_runtime", error)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _record_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    try:
        receipt = run_live_qualification(args)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "actions_executed": receipt["actions_executed"],
                    "observation_count": receipt["observation_count"],
                    "model_request_count": 0,
                    "behavioral_episode_count": 0,
                    "child_receipt": str(output_dir.parent / "child_qualification_receipt.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir.parent / "technical_failure.json"
        if not failure_path.exists():
            timing_path = output_dir / "timing_support.json"
            immutable_json(
                failure_path,
                {
                    "schema_version": FAILURE_SCHEMA,
                    "status": "technical_invalid",
                    "failed_at_utc": utc_now(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                    "timing_support": file_identity(timing_path) if timing_path.is_file() else None,
                    "adapter_completion": file_identity(output_dir / "completion.json") if (output_dir / "completion.json").is_file() else None,
                    "adapter_journal": file_identity(output_dir / "events.partial.jsonl") if (output_dir / "events.partial.jsonl").is_file() else None,
                    "model_request_count": 0,
                    "behavioral_episode_count": 0,
                    "recording_qualification_count": 0,
                    "claim_boundary": "Technical-invalid scripted recorder attempt; no learned-policy or forecast claim.",
                },
            )
        traceback.print_exc()
        return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command_name", required=True)
    queue = commands.add_parser("queue", help="run from the durable cluster queue worker")
    queue.add_argument("--source-root", type=Path, required=True)
    queue.add_argument("--state-dir", type=Path, required=True)
    queue.add_argument("--job-dir", type=Path, required=True)
    queue.add_argument("--gate-receipt", type=Path, required=True)
    queue.add_argument("--gate-receipt-sha256", required=True)
    queue.add_argument("--pose-manifest", type=Path, required=True)
    queue.add_argument("--pose-manifest-sha256", required=True)
    queue.add_argument("--layout-arm", choices=("original", "reflected"), required=True)
    queue.add_argument("--command", choices=("left", "right"), required=True)
    queue.add_argument("--environment-seed", type=int, default=2026091000)
    queue.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    queue.add_argument("--robolab-root", type=Path, default=ROBOLAB_ROOT)
    queue.add_argument("--robolab-python", type=Path, default=ROBOLAB_PYTHON)
    queue.add_argument("--nvidia-smi", default="nvidia-smi")

    record = commands.add_parser("record", help="run under the pinned RoboLab Python")
    record.add_argument("--source-root", type=Path, required=True)
    record.add_argument("--study-commit", required=True)
    record.add_argument("--robolab-root", type=Path, required=True)
    record.add_argument("--robolab-commit", required=True)
    record.add_argument("--gate-receipt", type=Path, required=True)
    record.add_argument("--gate-receipt-sha256", required=True)
    record.add_argument("--pose-manifest", type=Path, required=True)
    record.add_argument("--pose-manifest-sha256", required=True)
    record.add_argument("--layout-pair-id", required=True)
    record.add_argument("--layout-arm", choices=("original", "reflected"), required=True)
    record.add_argument("--command", choices=("left", "right"), required=True)
    record.add_argument("--environment-seed", type=int, required=True)
    record.add_argument("--attempt-id", required=True)
    record.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_name == "record":
        return _record_main(args)
    receipt = execute_queue_job(
        source_root=args.source_root,
        state_dir=args.state_dir,
        job_dir=args.job_dir,
        gate_receipt_path=args.gate_receipt,
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=args.pose_manifest,
        pose_manifest_sha256=args.pose_manifest_sha256,
        layout_arm=args.layout_arm,
        command=args.command,
        environment_seed=args.environment_seed,
        raw_root=args.raw_root,
        robolab_root=args.robolab_root,
        robolab_python=args.robolab_python,
        nvidia_smi=args.nvidia_smi,
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return int(receipt["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
