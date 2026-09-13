#!/usr/bin/env python3
"""Capture one hash-bound P00 observation without contacting either model.

The ``queue`` command is intentionally compatible with the cluster worker's
system Python.  It validates the accepted fixture evidence, the frozen pose
manifest, the immutable queue checkout, and the assigned idle B200 before
re-executing this file's ``capture`` command with the pinned RoboLab Python.

The child constructs the workshop timeout-only task, performs a fresh physical
settle, retains the original simulator arrays and native clock counters, and
uses the pinned RoboLab client implementations themselves to create the exact
Cosmos N3 and official-conditional DreamZero D1 wire arrays.  It performs zero
model requests and zero behavioral actions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import importlib
import importlib.util
import inspect
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
from typing import Any, Mapping, Sequence


# Queue source checkouts are immutable evidence.  Imports performed by the
# system-Python wrapper must never create untracked __pycache__ paths in them.
sys.dont_write_bytecode = True


NAMESPACE = "wmf_ablation_001_20260912"
LAYOUT_PAIR_ID = "P00"
LAYOUT_ARM = "original"
COMMAND = "left"
PROMPT = "Put the Rubik's cube to the left of the bowl."
SOURCE_CONTRACT_SHA256 = "88a1268ae7f27776fd5246a5808c069b2906399e99bb0e7d8b6f09020a6b85d3"
CANDIDATE_POOL_SHA256 = "ec80f4adc5272ec666c94d753241b84c954ef382bd30e1f2907459bff1cdb2b1"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
ROBOLAB_ROOT = Path("/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241")
ROBOLAB_PYTHON = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
RAW_ROOT = Path("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/fixed_observations")
ISAACLAB_SOURCE_ROOT = ROBOLAB_PYTHON.parent.parent / "lib/python3.11/site-packages/isaaclab/source"
ISAACLAB_PYTHON_ROOTS = tuple(
    ISAACLAB_SOURCE_ROOT / name
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_rl")
)
QUALIFIED_NATIVE_LIBRARY_PATH = ":".join(
    (
        "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu",
        "/data/users/ali/glvnd/lib",
        "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib",
        "/usr/lib/x86_64-linux-gnu",
    )
)
CAPTURE_SCHEMA = "wmf-forecast-layout-fixed-observation-capture-v1"
QUEUE_RECEIPT_SCHEMA = "wmf-forecast-layout-fixed-observation-job-v1"
GATE_RECEIPT_SCHEMA = "wmf-forecast-layout-fixture-job-receipt-v1"
GATE_RECORD_SCHEMA = "wmf-forecast-layout-gate-ledger-record-v1"
GATE_ATTEMPT_SCHEMA = "wmf-forecast-layout-gate-attempt-receipt-v1"
POSE_MANIFEST_SCHEMA = "wmf-forecast-layout-frozen-pose-manifest-v1"
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
MAX_PUBLISH_BYTES = 128 * 1024
RAW_CAMERAS = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
N3_ARRAY_KEYS = (
    "observation/image",
    "observation/joint_position",
    "observation/gripper_position",
)
D1_ARRAY_KEYS = (
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/cartesian_position",
    "observation/gripper_position",
)


class FixedObservationError(RuntimeError):
    """Fail-closed fixed-observation contract error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixedObservationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FixedObservationError("value is not finite canonical JSON") from error


def compact_canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FixedObservationError("value is not finite canonical JSON") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FixedObservationError(f"cannot hash evidence file: {path}") from error
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file(), f"evidence file is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FixedObservationError(f"non-finite JSON token: {value}")


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except FixedObservationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixedObservationError(f"cannot read JSON evidence: {path}") from error
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value, payload


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def verify_file_descriptor(descriptor: Any, *, label: str) -> dict[str, Any]:
    require(isinstance(descriptor, Mapping), f"{label} descriptor is missing")
    path_value = descriptor.get("path")
    require(isinstance(path_value, str) and Path(path_value).is_absolute(), f"{label} path must be absolute")
    path = Path(path_value).resolve()
    observed = file_identity(path)
    require(observed["sha256"] == descriptor.get("sha256"), f"{label} hash mismatch")
    if "bytes" in descriptor:
        require(observed["bytes"] == descriptor.get("bytes"), f"{label} size mismatch")
    return observed


def _read_gate_ledger(path: Path) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_bytes().splitlines()
    except OSError as error:
        raise FixedObservationError("accepted gate ledger is unreadable") from error
    require(bool(lines), "accepted gate ledger is empty")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, line in enumerate(lines):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_duplicate_safe_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FixedObservationError("accepted gate ledger is invalid JSONL") from error
        require(isinstance(row, dict), "accepted gate ledger row is not an object")
        require(row.get("schema_version") == GATE_RECORD_SCHEMA, "accepted gate ledger schema changed")
        require(row.get("study_namespace") == NAMESPACE, "accepted gate ledger namespace changed")
        require(row.get("sequence") == sequence, "accepted gate ledger sequence changed")
        require(row.get("previous_record_sha256") == previous, "accepted gate ledger chain broke")
        observed = require_sha256(row.get("record_sha256"), "gate record digest")
        core = dict(row)
        core.pop("record_sha256")
        require(sha256_bytes(canonical_bytes(core)) == observed, "accepted gate ledger record hash mismatch")
        previous = observed
        records.append(row)
    return records


def verify_gate_and_pose_manifest(
    *,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify the entire accepted-gate -> P00 frozen-manifest evidence chain."""

    require_sha256(gate_receipt_sha256, "gate receipt argument")
    require_sha256(pose_manifest_sha256, "pose manifest argument")
    gate_receipt_path = Path(gate_receipt_path).resolve()
    pose_manifest_path = Path(pose_manifest_path).resolve()
    require(not gate_receipt_path.is_symlink(), "gate receipt cannot be a symlink")
    require(not pose_manifest_path.is_symlink(), "pose manifest cannot be a symlink")
    gate, gate_payload = load_json(gate_receipt_path)
    manifest, manifest_payload = load_json(pose_manifest_path)
    require(sha256_bytes(gate_payload) == gate_receipt_sha256, "gate receipt argument hash mismatch")
    require(sha256_bytes(manifest_payload) == pose_manifest_sha256, "pose manifest argument hash mismatch")

    require(gate.get("schema_version") == GATE_RECEIPT_SCHEMA, "fixture gate receipt schema changed")
    require(gate.get("study_namespace") == NAMESPACE, "fixture gate receipt namespace changed")
    require(gate.get("status") == "finished", "fixture gate job is not finished")
    require(gate.get("layout_pair_id") == LAYOUT_PAIR_ID, "fixture gate receipt is not P00")
    require(gate.get("decision") == "accepted" and gate.get("exit_code") == 0, "P00 fixture was not accepted")
    require(gate.get("model_request_count") == 0, "fixture gate contacted a model")
    require(gate.get("behavioral_action_count") == 0, "fixture gate contains behavioral actions")
    require(gate.get("source_contract_sha256") == SOURCE_CONTRACT_SHA256, "fixture source contract changed")
    require(gate.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "fixture candidate pool changed")
    candidate_id = gate.get("candidate_id")
    candidate_sha = require_sha256(gate.get("candidate_payload_sha256"), "accepted candidate digest")
    require(isinstance(candidate_id, str) and candidate_id.startswith("P00__candidate_"), "accepted candidate ID is not P00")

    require(manifest.get("schema_version") == POSE_MANIFEST_SCHEMA, "pose manifest schema changed")
    require(manifest.get("study_namespace") == NAMESPACE, "pose manifest namespace changed")
    require(
        manifest.get("status")
        == "LIVE_GATE_QUALIFIED_POSES_FROZEN_MODEL_EXECUTION_NOT_RELEASED",
        "pose manifest is not live-qualified and frozen",
    )
    require(manifest.get("physical_layout_gate_passed") is True, "pose manifest lacks a passed physical gate")
    require(manifest.get("released_for_model_inference") is False, "pose manifest improperly releases inference")
    require(manifest.get("model_request_count") == 0 and manifest.get("behavioral_episode_count") == 0, "pose manifest contains model evidence")
    require(manifest.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "pose manifest candidate pool changed")
    task_contract = manifest.get("task_contract")
    require(isinstance(task_contract, Mapping), "pose manifest lacks task contract")
    require(task_contract.get("action_cap") == 450, "pose manifest action cap changed")
    require(task_contract.get("termination_terms") == ["time_out"], "pose manifest is not timeout-only")
    require(task_contract.get("success_is_measurement_only") is True, "pose manifest success contract changed")
    require(task_contract.get("success_termination_present") is False, "pose manifest retains success termination")
    layouts = manifest.get("layout_pairs")
    require(isinstance(layouts, Mapping) and LAYOUT_PAIR_ID in layouts, "pose manifest lacks P00")
    row = layouts[LAYOUT_PAIR_ID]
    require(isinstance(row, Mapping), "P00 pose row is invalid")
    require(row.get("layout_pair_id") == LAYOUT_PAIR_ID, "P00 pose row ID changed")
    require(row.get("candidate_id") == candidate_id, "pose manifest selected another candidate")
    require(row.get("candidate_payload_sha256") == candidate_sha, "pose manifest candidate hash changed")

    gate_evidence = gate.get("gate_evidence")
    require(isinstance(gate_evidence, Mapping), "fixture receipt lacks gate evidence")
    accepted_record_sha = require_sha256(
        gate_evidence.get("gate_record_sha256"), "accepted gate record digest"
    )
    require(row.get("accepted_gate_record_sha256") == accepted_record_sha, "pose manifest gate record changed")
    gate_ledger_descriptor = gate.get("gate_ledger")
    gate_ledger_identity = verify_file_descriptor(gate_ledger_descriptor, label="accepted gate ledger")
    require(manifest.get("gate_ledger_sha256") == gate_ledger_identity["sha256"], "pose manifest gate-ledger hash changed")
    records = _read_gate_ledger(Path(gate_ledger_identity["path"]))
    accepted = [
        record
        for record in records
        if record.get("record_sha256") == accepted_record_sha
        and record.get("layout_pair_id") == LAYOUT_PAIR_ID
    ]
    require(len(accepted) == 1, "accepted P00 gate record is absent or duplicated")
    record = accepted[0]
    require(record.get("decision") == "accepted" and record.get("passed") is True, "P00 gate record did not pass")
    require(record.get("candidate_id") == candidate_id, "gate record candidate ID changed")
    require(record.get("candidate_payload_sha256") == candidate_sha, "gate record candidate hash changed")
    require(record.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "gate record candidate pool changed")
    require(record.get("model_request_count") == 0 and record.get("behavioral_action_count") == 0, "gate record contains model evidence")

    attempt_identity = verify_file_descriptor(record.get("attempt_receipt"), label="accepted gate attempt")
    receipt_attempt_identity = verify_file_descriptor(
        gate_evidence.get("gate_attempt_receipt"), label="fixture receipt gate attempt"
    )
    require(attempt_identity == receipt_attempt_identity, "gate receipt and ledger reference different attempts")
    manifest_attempt = row.get("accepted_gate_attempt_receipt")
    manifest_attempt_identity = verify_file_descriptor(manifest_attempt, label="pose manifest gate attempt")
    require(attempt_identity == manifest_attempt_identity, "pose manifest references another gate attempt")
    attempt, _ = load_json(Path(attempt_identity["path"]))
    require(attempt.get("schema_version") == GATE_ATTEMPT_SCHEMA, "gate attempt receipt schema changed")
    require(attempt.get("candidate_id") == candidate_id, "gate attempt candidate changed")
    require(attempt.get("candidate_payload_sha256") == candidate_sha, "gate attempt candidate hash changed")
    require(attempt.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "gate attempt candidate pool changed")
    require(attempt.get("decision") == "accepted" and attempt.get("passed") is True, "gate attempt is not accepted")
    require(attempt.get("model_request_count") == 0 and attempt.get("behavioral_action_count") == 0, "gate attempt contains model evidence")
    evaluation = attempt.get("evaluation")
    require(isinstance(evaluation, Mapping) and evaluation.get("passed") is True, "gate attempt evaluation did not pass")

    return {
        "gate_receipt": file_identity(gate_receipt_path),
        "pose_manifest": file_identity(pose_manifest_path),
        "gate_ledger": gate_ledger_identity,
        "gate_attempt_receipt": attempt_identity,
        "accepted_gate_record_sha256": accepted_record_sha,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": candidate_sha,
        "pose_row": dict(row),
    }


def freeze_p00_pose_manifest(
    *,
    source_root: Path,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    expected_pose_manifest_sha256: str | None = None,
) -> tuple[Path, str]:
    """Create the P00-only frozen manifest once, or verify an existing one.

    This is deliberately downstream of a hash-pinned accepted gate receipt.  It
    uses the study's existing freeze builder and then re-opens the complete
    gate/ledger/attempt/manifest chain through :func:`verify_gate_and_pose_manifest`.
    """

    source_root = Path(source_root).resolve()
    gate_receipt_path = Path(gate_receipt_path).resolve()
    pose_manifest_path = Path(pose_manifest_path).resolve()
    require_sha256(gate_receipt_sha256, "gate receipt argument")
    if expected_pose_manifest_sha256 is not None:
        require_sha256(expected_pose_manifest_sha256, "pose manifest argument")
    require(pose_manifest_path.name.endswith(".json"), "pose manifest target must be JSON")
    pose_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = pose_manifest_path.with_name(pose_manifest_path.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if pose_manifest_path.exists():
            observed = sha256_file(pose_manifest_path)
            require(
                expected_pose_manifest_sha256 is not None,
                "existing pose manifest requires its expected SHA-256",
            )
            require(observed == expected_pose_manifest_sha256, "existing pose manifest hash mismatch")
            verify_gate_and_pose_manifest(
                gate_receipt_path=gate_receipt_path,
                gate_receipt_sha256=gate_receipt_sha256,
                pose_manifest_path=pose_manifest_path,
                pose_manifest_sha256=observed,
            )
            return pose_manifest_path, observed

        require(
            expected_pose_manifest_sha256 is None,
            "cannot create a missing pose manifest with a predeclared unknown digest",
        )
        gate, gate_payload = load_json(gate_receipt_path)
        require(sha256_bytes(gate_payload) == gate_receipt_sha256, "gate receipt argument hash mismatch")
        require(gate.get("schema_version") == GATE_RECEIPT_SCHEMA, "fixture gate receipt schema changed")
        require(gate.get("study_namespace") == NAMESPACE, "fixture gate receipt namespace changed")
        require(gate.get("status") == "finished", "fixture gate job is not finished")
        require(gate.get("layout_pair_id") == LAYOUT_PAIR_ID, "fixture gate receipt is not P00")
        require(gate.get("decision") == "accepted" and gate.get("exit_code") == 0, "P00 fixture was not accepted")
        require(gate.get("model_request_count") == 0 and gate.get("behavioral_action_count") == 0, "fixture gate contains model evidence")
        require(gate.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "fixture candidate pool changed")
        gate_evidence = gate.get("gate_evidence")
        require(isinstance(gate_evidence, Mapping), "fixture receipt lacks gate evidence")
        accepted_record_sha = require_sha256(
            gate_evidence.get("gate_record_sha256"), "accepted gate record digest"
        )
        ledger_identity = verify_file_descriptor(gate.get("gate_ledger"), label="accepted gate ledger")
        records = _read_gate_ledger(Path(ledger_identity["path"]))
        matching = [row for row in records if row.get("record_sha256") == accepted_record_sha]
        require(len(matching) == 1, "accepted record is absent from gate ledger")
        require(
            matching[0].get("layout_pair_id") == LAYOUT_PAIR_ID
            and matching[0].get("decision") == "accepted"
            and matching[0].get("passed") is True,
            "accepted gate record is not a passed P00 row",
        )
        attempt_identity = verify_file_descriptor(
            matching[0].get("attempt_receipt"), label="accepted gate attempt"
        )
        receipt_attempt_identity = verify_file_descriptor(
            gate_evidence.get("gate_attempt_receipt"), label="fixture receipt gate attempt"
        )
        require(attempt_identity == receipt_attempt_identity, "gate receipt and ledger reference different attempts")

        candidate_pool_path = (
            source_root
            / "workshops/corl2026_world_models/experiments/forecast_layout/layout_candidate_pool.json"
        )
        pool, pool_payload = load_json(candidate_pool_path)
        require(sha256_bytes(pool_payload) == CANDIDATE_POOL_SHA256, "candidate pool file changed")
        forecast_root = candidate_pool_path.parent
        if str(forecast_root) not in sys.path:
            sys.path.insert(0, str(forecast_root))
        from fixture_layouts import validate_candidate_pool
        from model_blind_fixture_gate import build_frozen_pose_manifest

        validate_candidate_pool(pool, source_sha256=SOURCE_CONTRACT_SHA256)
        manifest = build_frozen_pose_manifest(
            pool=pool,
            candidate_pool_sha256=CANDIDATE_POOL_SHA256,
            records=records,
            gate_ledger_sha256=ledger_identity["sha256"],
            layout_pair_ids=[LAYOUT_PAIR_ID],
        )
        _immutable_json(pose_manifest_path, manifest)
        observed = sha256_file(pose_manifest_path)
        verify_gate_and_pose_manifest(
            gate_receipt_path=gate_receipt_path,
            gate_receipt_sha256=gate_receipt_sha256,
            pose_manifest_path=pose_manifest_path,
            pose_manifest_sha256=observed,
        )
        return pose_manifest_path, observed


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
        raise FixedObservationError("Git source verification is unavailable") from error
    require(result.returncode == 0, f"Git source verification failed: {' '.join(arguments)}")
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
        raise FixedObservationError("queue path is missing") from error
    require(source.is_dir() and state.is_dir() and job.is_dir(), "queue paths must be directories")
    require(source.parent == state / "sources", "source root is outside queue state")
    require(job.parent == state / "jobs", "job directory is outside queue state")
    require(SAFE_ID_RE.fullmatch(job.name) is not None, "queue job ID is unsafe")
    descriptor, _ = load_json(job / "descriptor.json")
    require(descriptor.get("schema_version") == "wmf-cluster-job-v1", "queue descriptor schema changed")
    require(descriptor.get("namespace") == NAMESPACE, "queue descriptor namespace changed")
    require(descriptor.get("job_id") == job.name, "queue descriptor job ID changed")
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
        raise FixedObservationError("nvidia-smi query is unavailable") from error
    require(result.returncode == 0, "nvidia-smi query failed")
    rows = []
    for values in csv.reader(io.StringIO(result.stdout)):
        if not values or not any(item.strip() for item in values):
            continue
        require(len(values) == len(columns), "nvidia-smi query shape changed")
        rows.append({key: value.strip() for key, value in zip(columns, values)})
    return rows


def verify_idle_b200(executable: Path | str = "nvidia-smi") -> dict[str, Any]:
    devices = _query_nvidia(("index", "uuid", "name", "driver_version"), compute=False, executable=executable)
    require(len(devices) == 1, "fixed observation worker must expose exactly one assigned GPU")
    device = devices[0]
    require(device["index"] == "0" and device["name"] == "NVIDIA B200", "assigned device is not one index-0 B200")
    require(device["uuid"].startswith("GPU-"), "assigned GPU UUID is invalid")
    processes = _query_nvidia(
        ("gpu_uuid", "pid", "process_name", "used_memory"),
        compute=True,
        executable=executable,
    )
    require(not processes, "assigned GPU has a pre-existing compute process")
    return {**device, "preexisting_compute_process_count": 0}


def _immutable_write(path: Path, payload: bytes, *, maximum_bytes: int | None = None) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing to replace evidence: {path}")
    if maximum_bytes is not None:
        require(len(payload) <= maximum_bytes, f"evidence exceeds {maximum_bytes} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FixedObservationError(f"refusing to replace evidence: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_json(path: Path, value: Mapping[str, Any], *, publish: bool = False) -> None:
    _immutable_write(path, canonical_bytes(value), maximum_bytes=MAX_PUBLISH_BYTES if publish else None)


def _runtime_directories(state_parent: Path, hostname: str) -> dict[str, Path]:
    safe_host = hostname if SAFE_ID_RE.fullmatch(hostname) else sha256_bytes(hostname.encode())[:24]
    root = (state_parent / "worker_runtime" / "fixed_observation" / safe_host).resolve()
    require(root.is_relative_to(state_parent.resolve()), "runtime directory escaped queue state")
    directories = {
        "tmp": root / "tmp",
        "xdg": root / "xdg",
        "torch": root / "torch",
        "hf": root / "huggingface",
        "numba": root / "numba",
        "mpl": root / "matplotlib",
        "cuda": root / "cuda",
        "pycache": root / "pycache",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
        require(directory.is_dir() and os.access(directory, os.W_OK), "runtime cache directory is not writable")
    return directories


def build_child_environment(
    *,
    base: Mapping[str, str],
    directories: Mapping[str, Path],
    forecast_root: Path,
    source_root: Path,
    robolab_root: Path,
) -> dict[str, str]:
    env = dict(base)
    for inherited_name in ("DISPLAY", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES"):
        env.pop(inherited_name, None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TMPDIR=str(directories["tmp"]),
        XDG_CACHE_HOME=str(directories["xdg"]),
        TORCH_HOME=str(directories["torch"]),
        HF_HOME=str(directories["hf"]),
        NUMBA_CACHE_DIR=str(directories["numba"]),
        MPLCONFIGDIR=str(directories["mpl"]),
        CUDA_CACHE_PATH=str(directories["cuda"]),
        PYTHONPYCACHEPREFIX=str(directories["pycache"]),
        OMNI_KIT_ACCEPT_EULA="YES",
        PYTHONNOUSERSITE="1",
        VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
        LD_LIBRARY_PATH=QUALIFIED_NATIVE_LIBRARY_PATH,
    )
    roots = [
        str(Path(forecast_root).resolve()),
        str(Path(source_root).resolve()),
        str(Path(robolab_root).resolve()),
        *(str(path) for path in ISAACLAB_PYTHON_ROOTS),
    ]
    if base.get("PYTHONPATH"):
        roots.append(base["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env


def build_child_command(
    *,
    source_root: Path,
    output_dir: Path,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    study_commit: str,
    environment_seed: int,
    robolab_root: Path = ROBOLAB_ROOT,
    robolab_python: Path = ROBOLAB_PYTHON,
) -> list[str]:
    script = Path(source_root) / "workshops/corl2026_world_models/experiments/forecast_layout/fixed_observation_job.py"
    return [
        str(Path(robolab_python).resolve()),
        str(script.resolve()),
        "capture",
        "--source-root",
        str(Path(source_root).resolve()),
        "--study-commit",
        study_commit,
        "--robolab-root",
        str(Path(robolab_root).resolve()),
        "--robolab-commit",
        ROBOLAB_COMMIT,
        "--pose-manifest",
        str(Path(pose_manifest_path).resolve()),
        "--pose-manifest-sha256",
        pose_manifest_sha256,
        "--gate-receipt",
        str(Path(gate_receipt_path).resolve()),
        "--gate-receipt-sha256",
        gate_receipt_sha256,
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--environment-seed",
        str(environment_seed),
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


def execute_queue_job(
    *,
    source_root: Path,
    state_dir: Path,
    job_dir: Path,
    pose_manifest_path: Path,
    pose_manifest_sha256: str | None,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
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
    publish_target = publish / "fixed_observation_job_receipt.json"
    existing = _existing_queue_receipt(publish_target, job_id=job.name, source_commit=source_commit)
    if existing is not None:
        return dict(existing, idempotent_replay=True)

    context: dict[str, Any] = {"gpu": None, "release": None, "child_started": False, "child_exit_code": None, "logs": None}
    try:
        require(type(environment_seed) is int and environment_seed >= 0, "environment seed is invalid")
        verify_clean_git(source, source_commit, "study")
        verify_clean_git(Path(robolab_root), ROBOLAB_COMMIT, "RoboLab")
        require(Path(robolab_python).resolve().is_file(), "pinned RoboLab Python is missing")
        pose_manifest_path, resolved_pose_manifest_sha256 = freeze_p00_pose_manifest(
            source_root=source,
            gate_receipt_path=gate_receipt_path,
            gate_receipt_sha256=gate_receipt_sha256,
            pose_manifest_path=pose_manifest_path,
            expected_pose_manifest_sha256=pose_manifest_sha256,
        )
        context["release"] = verify_gate_and_pose_manifest(
            gate_receipt_path=gate_receipt_path,
            gate_receipt_sha256=gate_receipt_sha256,
            pose_manifest_path=pose_manifest_path,
            pose_manifest_sha256=resolved_pose_manifest_sha256,
        )
        context["gpu"] = verify_idle_b200(nvidia_smi)
        raw_root = Path(raw_root).resolve()
        raw_root.mkdir(parents=True, exist_ok=True)
        require(raw_root.is_dir() and not raw_root.is_symlink(), "fixed-observation raw root is invalid")
        attempt_root = (raw_root / job.name).resolve()
        require(attempt_root.parent == raw_root, "fixed-observation attempt escaped raw root")
        require(not attempt_root.exists(), "immutable fixed-observation attempt already exists without a queue receipt")
        attempt_root.mkdir()
        output_dir = attempt_root / "capture"
        stdout_path = attempt_root / "child.stdout.log"
        stderr_path = attempt_root / "child.stderr.log"
        directories = _runtime_directories(state.parent.resolve(), socket.gethostname())
        forecast_root = source / "workshops/corl2026_world_models/experiments/forecast_layout"
        child_env = build_child_environment(
            base=os.environ,
            directories=directories,
            forecast_root=forecast_root,
            source_root=source,
            robolab_root=robolab_root,
        )
        command = build_child_command(
            source_root=source,
            output_dir=output_dir,
            pose_manifest_path=pose_manifest_path,
            pose_manifest_sha256=resolved_pose_manifest_sha256,
            gate_receipt_path=gate_receipt_path,
            gate_receipt_sha256=gate_receipt_sha256,
            study_commit=source_commit,
            environment_seed=environment_seed,
            robolab_root=robolab_root,
            robolab_python=robolab_python,
        )
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            context["child_started"] = True
            child = subprocess.run(
                command,
                cwd=Path(robolab_root).resolve(),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        context["child_exit_code"] = child.returncode
        context["logs"] = {"stdout": file_identity(stdout_path), "stderr": file_identity(stderr_path)}
        require(child.returncode == 0, "RoboLab fixed-observation child failed")
        capture_path = output_dir / "capture_receipt.json"
        capture, _ = load_json(capture_path)
        capture_identity = file_identity(capture_path)
        require(capture.get("schema_version") == CAPTURE_SCHEMA and capture.get("status") == "passed", "child capture receipt did not pass")
        require(capture.get("study_commit") == source_commit, "child capture used another study commit")
        require(
            capture.get("pose_manifest", {}).get("sha256")
            == resolved_pose_manifest_sha256,
            "child capture used another pose manifest",
        )
        require(capture.get("gate_receipt", {}).get("sha256") == gate_receipt_sha256, "child capture used another gate receipt")
        require(capture.get("model_request_count") == 0 and capture.get("behavioral_action_count") == 0, "child capture contains model or behavioral work")
        receipt = {
            "schema_version": QUEUE_RECEIPT_SCHEMA,
            "study_namespace": NAMESPACE,
            "status": "passed",
            "exit_code": 0,
            "job_id": job.name,
            "study_commit": source_commit,
            "layout_pair_id": LAYOUT_PAIR_ID,
            "candidate_id": context["release"]["candidate_id"],
            "environment_seed": environment_seed,
            "gate_receipt_sha256": gate_receipt_sha256,
            "pose_manifest_sha256": resolved_pose_manifest_sha256,
            "gpu_identity": context["gpu"],
            "child_started": True,
            "child_exit_code": 0,
            "child_logs": context["logs"],
            "raw_capture_receipt": capture_identity,
            "model_fixtures": capture["model_fixtures"],
            "model_request_count": 0,
            "behavioral_action_count": 0,
            "finished_at_utc": utc_now(),
            "claim_boundary": "One current P00 physical observation and exact preprocessing only; no model request or behavioral episode.",
        }
    except Exception as error:
        receipt = {
            "schema_version": QUEUE_RECEIPT_SCHEMA,
            "study_namespace": NAMESPACE,
            "status": "technical_invalid",
            "exit_code": 3,
            "job_id": job.name,
            "study_commit": source_commit,
            "layout_pair_id": LAYOUT_PAIR_ID,
            "environment_seed": environment_seed,
            "reason": type(error).__name__,
            "detail": str(error)[:2000],
            "gpu_identity": context["gpu"],
            "child_started": context["child_started"],
            "child_exit_code": context["child_exit_code"],
            "child_logs": context["logs"],
            "model_request_count": 0,
            "behavioral_action_count": 0,
            "finished_at_utc": utc_now(),
            "claim_boundary": "Technical-invalid fixed-observation attempt; no model request or behavioral episode.",
        }
    _immutable_json(publish_target, receipt, publish=True)
    return receipt


def _to_numpy(value: Any, *, env_id: int | None = None) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if env_id is not None:
        require(array.ndim >= 1 and array.shape[0] > env_id, "simulator array lacks requested environment")
        array = array[env_id]
    array = np.ascontiguousarray(array)
    require(not array.dtype.hasobject, "simulator array has object dtype")
    if array.dtype.kind in "fc":
        require(bool(np.isfinite(array).all()), "simulator array contains non-finite values")
    return array.copy()


def _array_identity(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    header = {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"}
    digest = hashlib.sha256()
    digest.update(compact_canonical_bytes(header))
    digest.update(array.tobytes(order="C"))
    return {**header, "value_sha256": digest.hexdigest()}


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    require(bool(arrays), f"NPZ payload is empty: {path.name}")
    frozen = {str(key): _to_numpy(value) for key, value in arrays.items()}
    require(len(frozen) == len(arrays), f"duplicate NPZ keys: {path.name}")
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite NPZ evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **frozen)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **file_identity(path),
        "arrays": {key: _array_identity(value) for key, value in sorted(frozen.items())},
    }


def _scalar(value: Any, *, index: int | None = None) -> int | float:
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


def _native_value(root: Any, names: Sequence[str], *, index: int | None, label: str) -> tuple[int | float, str]:
    for name in names:
        current = root
        try:
            for component in name.split("."):
                current = getattr(current, component)
            return _scalar(current, index=index), name
        except (AttributeError, FixedObservationError, TypeError, ValueError):
            continue
    raise FixedObservationError(f"native runtime does not expose {label}")


def _seconds_to_ns(value: int | float) -> int:
    try:
        decimal = Decimal(str(value)) * Decimal(1_000_000_000)
    except (InvalidOperation, ValueError) as error:
        raise FixedObservationError("native timestamp cannot be converted to nanoseconds") from error
    result = int(decimal.to_integral_value())
    require(result >= 0, "native timestamp is negative")
    return result


def capture_native_clock(
    env: Any,
    env_cfg: Any,
    observation: Mapping[str, Any],
    *,
    env_id: int = 0,
) -> dict[str, Any]:
    """Read only native counters; no video-frame/action-number inference."""

    started_wall = time.time_ns()
    started_monotonic = time.monotonic_ns()
    physics_step, physics_step_source = _native_value(
        env.sim,
        ("current_time_step_index", "frame_count", "_frame_count"),
        index=None,
        label="physics frame counter",
    )
    require(float(physics_step).is_integer() and physics_step >= 0, "physics frame counter is not a nonnegative integer")
    physics_time, physics_time_source = _native_value(
        env.sim, ("current_time", "_current_time"), index=None, label="physics time"
    )
    require(float(physics_time) >= 0, "physics time is negative")
    control_step, control_source = _native_value(
        env, ("common_step_counter", "_common_step_counter"), index=None, label="control step counter"
    )
    require(float(control_step).is_integer() and control_step >= 0, "control step counter is invalid")
    episode_step, episode_source = _native_value(
        env, ("episode_length_buf",), index=env_id, label="episode length counter"
    )
    require(float(episode_step).is_integer() and episode_step == 0, "behavioral episode counter was not reset to zero")
    cameras: dict[str, Any] = {}
    for camera_name in RAW_CAMERAS:
        sensor = env.scene[camera_name]
        sensor_time, sensor_time_source = _native_value(
            sensor,
            ("data.timestamp", "timestamp", "_timestamp"),
            index=env_id,
            label=f"{camera_name} sensor timestamp",
        )
        capture_time, capture_time_source = _native_value(
            sensor,
            ("data.timestamp_last_update", "timestamp_last_update", "_timestamp_last_update"),
            index=env_id,
            label=f"{camera_name} capture timestamp",
        )
        require(float(sensor_time) >= 0, f"{camera_name} sensor timestamp is invalid")
        require(float(capture_time) >= 0, f"{camera_name} capture timestamp is invalid")
        image_group = observation.get("image_obs")
        require(isinstance(image_group, Mapping) and camera_name in image_group, f"{camera_name} observation is missing")
        rgb_identity = _array_identity(_to_numpy(image_group[camera_name], env_id=env_id))
        cameras[camera_name] = {
            "frame_id": f"rgb-sha256:{rgb_identity['value_sha256']}",
            "frame_identity_source": "exact returned RGB array value identity",
            "rgb_array_identity": rgb_identity,
            "native_frame_counter": None,
            "native_frame_counter_status": "unavailable_in_pinned_isaaclab_sensorbase",
            "native_sensor_time_s": float(sensor_time),
            "native_sensor_time_source": sensor_time_source,
            "native_capture_time_s": float(capture_time),
            "native_capture_time_source": capture_time_source,
            "capture_time_ns": _seconds_to_ns(capture_time),
            "capture_time_ns_derivation": "Decimal(str(native_capture_time_s))*1e9 rounded to nearest integer",
            "timestamp_domain": "Isaac SensorBase native last-update simulation-time counter",
        }
    finished_monotonic = time.monotonic_ns()
    finished_wall = time.time_ns()
    sim_cfg = env_cfg.sim
    return {
        "physics_step": int(physics_step),
        "physics_step_source": physics_step_source,
        "physics_time_s": float(physics_time),
        "physics_time_source": physics_time_source,
        "control_step_since_physical_reset": int(control_step),
        "control_step_source": control_source,
        "behavioral_episode_step": int(episode_step),
        "behavioral_episode_step_source": episode_source,
        "camera_counters": cameras,
        "host_read_window": {
            "started_wall_time_ns": started_wall,
            "finished_wall_time_ns": finished_wall,
            "started_monotonic_ns": started_monotonic,
            "finished_monotonic_ns": finished_monotonic,
        },
        "runtime_configuration": {
            "physics_dt_s": float(sim_cfg.dt),
            "decimation": int(env_cfg.decimation),
            "render_interval": int(sim_cfg.render_interval),
        },
        "timing_claim_boundary": "All counters are read from the live runtime. No generated-frame time or action alignment is inferred.",
    }


def _load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, f"cannot load preprocessing module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _source_identity_for_object(value: Any) -> dict[str, Any]:
    path = inspect.getsourcefile(value)
    require(path is not None, f"cannot identify preprocessing source for {value}")
    return file_identity(Path(path))


def extract_exact_model_inputs(
    observation: Mapping[str, Any],
    *,
    study_root: Path,
    env_id: int = 0,
    n3_client_class: Any | None = None,
    d1_client_class: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Invoke the pinned clients' extraction/packing paths without constructing sockets."""

    import numpy as np
    d1_overlay_path = Path(study_root) / "experiments/dreamzero_droid/v2_robolab_client.py"
    if n3_client_class is None:
        from policies.cosmos3.client import Cosmos3Client
    else:
        Cosmos3Client = n3_client_class
    if d1_client_class is None:
        require(d1_overlay_path.is_file(), "frozen DreamZero preprocessing overlay is missing")
        d1_module = _load_module(d1_overlay_path, "wmf_fixed_observation_d1_overlay")
        D1Client = d1_module.V2DreamZeroDroidClient
        d1_overlay_identity = file_identity(d1_overlay_path)
    else:
        D1Client = d1_client_class
        d1_overlay_identity = _source_identity_for_object(D1Client)

    n3_helper = object.__new__(Cosmos3Client)
    n3_helper._image_h = 360
    n3_helper._image_w = 640
    n3_helper._env_session_id = {}
    n3_extracted = n3_helper._extract_observation(observation, env_id=env_id)
    n3_request = n3_helper._pack_request(n3_extracted, PROMPT)
    n3_arrays = {key: np.ascontiguousarray(np.asarray(n3_request[key])).copy() for key in N3_ARRAY_KEYS}
    require(set(N3_ARRAY_KEYS).issubset(n3_request), "official N3 request lacks exact array keys")
    image = n3_arrays["observation/image"]
    require(image.shape == (540, 640, 3) and image.dtype == np.uint8, f"official N3 packed image changed: {image.shape}/{image.dtype}")
    joint = n3_arrays["observation/joint_position"]
    gripper = n3_arrays["observation/gripper_position"]
    require(joint.shape == (7,) and joint.dtype == np.float32, f"official N3 joint input changed: {joint.shape}/{joint.dtype}")
    require(gripper.shape == (1,) and gripper.dtype == np.float32, f"official N3 gripper input changed: {gripper.shape}/{gripper.dtype}")

    d1_helper = object.__new__(D1Client)
    d1_helper.cam2_source = "right"
    d1_helper.resize = "pad"
    d1_helper.image_height = 180
    d1_helper.image_width = 320
    d1_helper._env_session_id = {}
    d1_extracted = d1_helper._extract_observation(observation, env_id=env_id)
    d1_request = d1_helper._pack_request(d1_extracted, PROMPT)
    d1_arrays = {key: np.ascontiguousarray(np.asarray(d1_request[key])).copy() for key in D1_ARRAY_KEYS}
    expected = {
        "observation/exterior_image_0_left": ((180, 320, 3), np.dtype("uint8")),
        "observation/exterior_image_1_left": ((180, 320, 3), np.dtype("uint8")),
        "observation/wrist_image_left": ((180, 320, 3), np.dtype("uint8")),
        "observation/joint_position": ((7,), np.dtype("float64")),
        "observation/cartesian_position": ((6,), np.dtype("float64")),
        "observation/gripper_position": ((1,), np.dtype("float64")),
    }
    for key, (shape, dtype) in expected.items():
        value = d1_arrays[key]
        require(value.shape == shape and value.dtype == dtype, f"official D1 input changed for {key}: {value.shape}/{value.dtype}")
        require(bool(np.isfinite(value).all()), f"official D1 input is non-finite: {key}")

    def numeric_intermediates(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in sorted(value.items()):
            try:
                result[f"{prefix}/{key}"] = _to_numpy(child)
            except (FixedObservationError, TypeError, ValueError):
                continue
        return result

    intermediate_arrays = {
        **numeric_intermediates("N3", n3_extracted),
        **numeric_intermediates("D1", d1_extracted),
    }
    provenance = {
        "N3": {
            "client_class": f"{Cosmos3Client.__module__}.{Cosmos3Client.__qualname__}",
            "client_source": _source_identity_for_object(Cosmos3Client),
            "extraction_method": "_extract_observation",
            "packing_method": "_pack_request",
            "configuration": {"image_height": 360, "image_width": 640},
            "wire_array_keys": list(N3_ARRAY_KEYS),
            "packed_image_layout": "official client output; wrist above left/right exterior views",
        },
        "D1": {
            "client_class": f"{D1Client.__module__}.{D1Client.__qualname__}",
            "overlay_source": d1_overlay_identity,
            "official_client_source": _source_identity_for_object(D1Client.__mro__[1]),
            "extraction_method": "_extract_observation",
            "packing_method": "_pack_request",
            "configuration": {
                "cam2_source": "right",
                "resize": "pad",
                "image_height": 180,
                "image_width": 320,
            },
            "wire_array_keys": list(D1_ARRAY_KEYS),
        },
        "prompt_used_only_for_preprocessing": PROMPT,
        "model_request_count": 0,
    }
    return n3_arrays, d1_arrays, {"arrays": intermediate_arrays, "provenance": provenance}


def write_capture_artifacts(
    *,
    output_dir: Path,
    raw_observation: Mapping[str, Mapping[str, Any]],
    physical_state: Mapping[str, Any],
    native_clock: Mapping[str, Any],
    settled_reset_receipt: Mapping[str, Any],
    fresh_physical_checks: Mapping[str, Any],
    release: Mapping[str, Any],
    n3_arrays: Mapping[str, Any],
    d1_arrays: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    study_commit: str,
    robolab_commit: str,
    environment_seed: int,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    require(not output_dir.exists(), f"refusing to overwrite capture directory: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_arrays = {
        f"image_obs/{key}": value
        for key, value in raw_observation["image_obs"].items()
    } | {
        f"proprio_obs/{key}": value
        for key, value in raw_observation["proprio_obs"].items()
    }
    state_arrays = {str(key): value for key, value in physical_state["arrays"].items()}
    raw_descriptor = _write_npz(output_dir / "raw_settled_observation.npz", raw_arrays)
    state_descriptor = _write_npz(output_dir / "simulator_state.npz", state_arrays)
    n3_descriptor = _write_npz(output_dir / "n3_official_wire_observation.npz", n3_arrays)
    d1_descriptor = _write_npz(output_dir / "d1_official_wire_observation.npz", d1_arrays)
    intermediate_descriptor = _write_npz(
        output_dir / "preprocessing_intermediates.npz", preprocessing["arrays"]
    )
    artifacts = {
        "raw_settled_observation": raw_descriptor,
        "simulator_state": state_descriptor,
        "preprocessing_intermediates": intermediate_descriptor,
        "N3": n3_descriptor,
        "D1": d1_descriptor,
    }
    artifact_logical = {
        key: {
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "arrays": row["arrays"],
        }
        for key, row in artifacts.items()
    }
    camera_counters = native_clock["camera_counters"]
    receipt = {
        "schema_version": CAPTURE_SCHEMA,
        "study_namespace": NAMESPACE,
        "status": "passed",
        "capture_id": f"{LAYOUT_PAIR_ID}-{environment_seed}-{sha256_bytes(compact_canonical_bytes(artifact_logical))[:16]}",
        "captured_at_utc": utc_now(),
        "study_commit": study_commit,
        "robolab_commit": robolab_commit,
        "layout_pair_id": LAYOUT_PAIR_ID,
        "layout_arm": LAYOUT_ARM,
        "command_task_used_for_reset": COMMAND,
        "environment_seed": environment_seed,
        "candidate_id": release["candidate_id"],
        "candidate_payload_sha256": release["candidate_payload_sha256"],
        "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
        "gate_receipt": release["gate_receipt"],
        "pose_manifest": release["pose_manifest"],
        "gate_ledger": release["gate_ledger"],
        "gate_attempt_receipt": release["gate_attempt_receipt"],
        "settled_reset_identity": settled_reset_receipt["reset_identity"],
        "settled_reset_receipt": dict(settled_reset_receipt),
        "fresh_physical_checks": dict(fresh_physical_checks),
        "native_clock": dict(native_clock),
        "source_capture": {
            "simulator_observation_id": "P00_original_settled_observation_000000",
            "camera_frame_ids": {
                name: camera_counters[name]["frame_id"] for name in RAW_CAMERAS
            },
            "camera_capture_time_ns": {
                name: camera_counters[name]["capture_time_ns"] for name in RAW_CAMERAS
            },
            "camera_timestamp_source": {
                name: camera_counters[name]["native_capture_time_source"] for name in RAW_CAMERAS
            },
        },
        "artifacts": artifacts,
        "artifact_logical_sha256": sha256_bytes(compact_canonical_bytes(artifact_logical)),
        "preprocessing": preprocessing["provenance"],
        "model_fixtures": {
            "N3": {
                "interface": "official packed observation/image plus joint/gripper arrays",
                "fixture": {key: n3_descriptor[key] for key in ("path", "bytes", "sha256")},
                "array_keys": list(N3_ARRAY_KEYS),
            },
            "D1": {
                "interface": "official conditional DreamZero six-array request fixture",
                "fixture": {key: d1_descriptor[key] for key in ("path", "bytes", "sha256")},
                "array_keys": list(D1_ARRAY_KEYS),
            },
        },
        "runtime_identity": dict(runtime_identity),
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "settling_hold_action_count": int(
            settled_reset_receipt["settle_evidence"]["settle_steps"]
            + settled_reset_receipt["settle_evidence"]["stability_window_steps"]
        ),
        "claim_boundary": "Current physical P00 reset and client preprocessing evidence only. No model was loaded or queried; no behavioral action was executed; no generated-frame timing is claimed.",
    }
    _immutable_json(output_dir / "capture_receipt.json", receipt)
    return receipt


def _fresh_physical_callbacks(output_dir: Path, source_contract: Mapping[str, Any]):
    from robolab_fixture_gate_adapter import RoboLabFixtureGateAdapter
    from robolab.core.world.world_state import get_world

    evidence: dict[str, Any] = {}
    gate = source_contract["live_gate"]

    def collision_sampler(env: Any, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        del observation
        probe = object.__new__(RoboLabFixtureGateAdapter)
        rows = probe._collision_rows(get_world(env))
        passed = all(item.get("clear") is True for item in rows["forbidden_pairs"].values())
        result = {"passed": passed, **rows}
        evidence["collision"] = result
        return result

    def visibility_sampler(env: Any, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        target = Path(output_dir) / "fresh_physical_camera_evidence"
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
            "method": "same calibrated projection and OBB occlusion implementation as accepted gate",
            "required_cameras": list(gate["required_cameras"]),
            "visibility_cameras": list(gate["visibility_cameras"]),
            "cameras": rows,
        }
        evidence["visibility"] = result
        return result

    return collision_sampler, visibility_sampler, evidence


def run_live_capture(args: argparse.Namespace) -> dict[str, Any]:
    """Run inside the pinned RoboLab Python after Isaac application startup."""

    source_root = Path(args.source_root).resolve()
    robolab_root = Path(args.robolab_root).resolve()
    forecast_root = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
    require(args.layout_pair_id == LAYOUT_PAIR_ID, "fixed observation is restricted to P00")
    require(args.layout_arm == LAYOUT_ARM and args.command == COMMAND, "fixed observation task identity changed")
    verify_clean_git(source_root, args.study_commit, "study")
    require(args.robolab_commit == ROBOLAB_COMMIT, "RoboLab expected commit changed")
    verify_clean_git(robolab_root, args.robolab_commit, "RoboLab")
    release = verify_gate_and_pose_manifest(
        gate_receipt_path=args.gate_receipt,
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=args.pose_manifest,
        pose_manifest_sha256=args.pose_manifest_sha256,
    )
    require(not Path(args.output_dir).exists(), "fixed-observation output directory already exists")

    if str(forecast_root) not in sys.path:
        sys.path.insert(0, str(forecast_root))
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    os.environ.update(
        WMF_FORECAST_POSE_MANIFEST=str(Path(args.pose_manifest).resolve()),
        WMF_FORECAST_POSE_MANIFEST_SHA256=args.pose_manifest_sha256,
        WMF_FORECAST_LAYOUT_PAIR_ID=LAYOUT_PAIR_ID,
    )

    import cv2  # noqa: F401 - RoboLab/Isaac import order requires OpenCV first.
    from isaaclab.app import AppLauncher

    launch_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launch_parser)
    launch_args, _ = launch_parser.parse_known_args(["--headless"])
    launch_args.enable_cameras = True
    launcher = AppLauncher(launch_args)
    simulation_app = launcher.app
    env = None
    try:
        import numpy as np
        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        from fixture_tasks import settle_for_recording_reset

        require(Path(robolab.__file__).resolve().is_relative_to(robolab_root), "effective RoboLab import is outside pinned checkout")
        output_dir = Path(args.output_dir).resolve()
        simulator_output = output_dir.parent / "native_simulator"
        require(not simulator_output.exists(), "native simulator output already exists")
        simulator_output.mkdir()
        set_output_dir(str(simulator_output))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = False
        task_path = forecast_root / "task_files/original_left.py"
        auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        env, env_cfg = create_env(
            "WMFForecastOriginalLeftTask",
            device="cuda:0",
            seed=args.environment_seed,
            num_envs=1,
            instruction_type="default",
            policy="wmf_fixed_observation_no_policy",
            renderer="realtime",
            rendering_mode="balanced",
        )
        require(not hasattr(env_cfg.terminations, "success"), "constructed task retains success termination")
        require(hasattr(env_cfg.terminations, "time_out"), "constructed task lacks timeout termination")
        public_terms = sorted(
            name
            for name, value in vars(env_cfg.terminations).items()
            if not name.startswith("_") and value is not None
        )
        require(public_terms == ["time_out"], f"constructed task has non-timeout termination terms: {public_terms}")
        require(float(env_cfg.episode_length_s) == 30.0, "constructed task duration changed")
        require(env_cfg.instruction == PROMPT, "constructed task prompt changed")
        observation, info = env.reset()
        source_contract, source_payload = load_json(forecast_root / "layout_source_contract.json")
        require(sha256_bytes(source_payload) == SOURCE_CONTRACT_SHA256, "source contract file changed")
        collision_sampler, visibility_sampler, physical_evidence = _fresh_physical_callbacks(
            output_dir.parent, source_contract
        )
        reset_identity = f"fixed-observation:{args.study_commit}:{LAYOUT_PAIR_ID}:{args.environment_seed}"
        observation, info, settled_receipt = settle_for_recording_reset(
            env,
            observation,
            info,
            pose_manifest_sha256=args.pose_manifest_sha256,
            reset_identity=reset_identity,
            collision_sampler=collision_sampler,
            visibility_sampler=visibility_sampler,
            settle_steps=int(source_contract["live_gate"]["settle_steps"]),
            stability_window_steps=int(source_contract["live_gate"]["stability_window_steps"]),
            linear_speed_tolerance_m_s=float(source_contract["live_gate"]["linear_speed_tolerance_m_s"]),
            angular_speed_tolerance_rad_s=float(source_contract["live_gate"]["angular_speed_tolerance_rad_s"]),
        )
        raw_observation = {
            "image_obs": {
                name: _to_numpy(value, env_id=0)
                for name, value in observation["image_obs"].items()
            },
            "proprio_obs": {
                name: _to_numpy(value, env_id=0)
                for name, value in observation["proprio_obs"].items()
            },
        }
        require(set(raw_observation["image_obs"]).issuperset(RAW_CAMERAS), "settled observation lacks original cameras")
        for name in RAW_CAMERAS:
            frame = raw_observation["image_obs"][name]
            require(frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8, f"raw camera changed: {name}")
        native_clock = capture_native_clock(env, env_cfg, observation)

        world = get_world(env)
        state_arrays: dict[str, Any] = {}
        configured = release["pose_row"]["layouts"][LAYOUT_ARM]
        tolerance = float(source_contract["live_gate"]["pose_tolerance_m"])
        for name in ("banana", "bowl", "rubiks_cube"):
            position, quaternion = world.get_pose(name, env_id=0)
            velocity = world.get_velocity(name, env_id=0)
            state_arrays[f"objects/{name}/position_robot_base_m"] = _to_numpy(position).reshape(-1)
            state_arrays[f"objects/{name}/quaternion_wxyz"] = _to_numpy(quaternion).reshape(-1)
            state_arrays[f"objects/{name}/velocity_linear_angular"] = _to_numpy(velocity).reshape(-1)
            expected_position = np.asarray(configured["positions_robot_base_m"][name], dtype=np.float64)
            observed_position = np.asarray(state_arrays[f"objects/{name}/position_robot_base_m"], dtype=np.float64)
            require(observed_position.shape == (3,), f"settled {name} position shape changed")
            require(float(np.max(np.abs(observed_position - expected_position))) <= tolerance, f"settled {name} pose left P00 tolerance")
        robot = env.scene["robot"]
        for attribute in (
            "root_pos_w",
            "root_quat_w",
            "joint_pos",
            "joint_vel",
        ):
            value = getattr(robot.data, attribute, None)
            require(value is not None, f"robot state lacks {attribute}")
            state_arrays[f"robot/{attribute}"] = _to_numpy(value, env_id=0)

        n3_arrays, d1_arrays, preprocessing = extract_exact_model_inputs(
            observation, study_root=source_root
        )
        runtime_identity = {
            "pod": socket.gethostname(),
            "pod_uid": os.environ.get("POD_UID"),
            "python": sys.version,
            "python_executable": sys.executable,
            "robolab_module": file_identity(Path(robolab.__file__)),
            "task_file": file_identity(task_path),
            "renderer": "realtime",
            "rendering_type": "balanced",
            "device": "cuda:0",
        }
        return write_capture_artifacts(
            output_dir=output_dir,
            raw_observation=raw_observation,
            physical_state={"arrays": state_arrays},
            native_clock=native_clock,
            settled_reset_receipt=settled_receipt,
            fresh_physical_checks=physical_evidence,
            release=release,
            n3_arrays=n3_arrays,
            d1_arrays=d1_arrays,
            preprocessing=preprocessing,
            study_commit=args.study_commit,
            robolab_commit=args.robolab_commit,
            environment_seed=args.environment_seed,
            runtime_identity=runtime_identity,
        )
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _capture_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    try:
        receipt = run_live_capture(args)
        print(json.dumps({
            "status": receipt["status"],
            "capture_id": receipt["capture_id"],
            "capture_receipt": str(output_dir / "capture_receipt.json"),
            "model_request_count": 0,
            "behavioral_action_count": 0,
        }, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "technical_failure.json"
        if not failure_path.exists():
            _immutable_json(
                failure_path,
                {
                    "schema_version": "wmf-forecast-layout-fixed-observation-failure-v1",
                    "status": "technical_invalid",
                    "failed_at_utc": utc_now(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                    "model_request_count": 0,
                    "behavioral_action_count": 0,
                    "claim_boundary": "No model request or behavioral episode was attempted.",
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
    queue.add_argument("--pose-manifest", type=Path, required=True)
    queue.add_argument(
        "--pose-manifest-sha256",
        help="required when --pose-manifest already exists; omit to freeze a missing P00 manifest",
    )
    queue.add_argument("--gate-receipt", type=Path, required=True)
    queue.add_argument("--gate-receipt-sha256", required=True)
    queue.add_argument("--environment-seed", type=int, default=2026091000)
    queue.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    queue.add_argument("--robolab-root", type=Path, default=ROBOLAB_ROOT)
    queue.add_argument("--robolab-python", type=Path, default=ROBOLAB_PYTHON)
    queue.add_argument("--nvidia-smi", default="nvidia-smi")

    capture = commands.add_parser("capture", help="internal pinned-RoboLab child")
    capture.add_argument("--source-root", type=Path, required=True)
    capture.add_argument("--study-commit", required=True)
    capture.add_argument("--robolab-root", type=Path, required=True)
    capture.add_argument("--robolab-commit", required=True)
    capture.add_argument("--pose-manifest", type=Path, required=True)
    capture.add_argument("--pose-manifest-sha256", required=True)
    capture.add_argument("--gate-receipt", type=Path, required=True)
    capture.add_argument("--gate-receipt-sha256", required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--environment-seed", type=int, required=True)
    capture.add_argument("--layout-pair-id", default=LAYOUT_PAIR_ID, choices=[LAYOUT_PAIR_ID])
    capture.add_argument("--layout-arm", default=LAYOUT_ARM, choices=[LAYOUT_ARM])
    capture.add_argument("--command", default=COMMAND, choices=[COMMAND])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_name == "capture":
        return _capture_main(args)
    receipt = execute_queue_job(
        source_root=args.source_root,
        state_dir=args.state_dir,
        job_dir=args.job_dir,
        pose_manifest_path=args.pose_manifest,
        pose_manifest_sha256=args.pose_manifest_sha256,
        gate_receipt_path=args.gate_receipt,
        gate_receipt_sha256=args.gate_receipt_sha256,
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
