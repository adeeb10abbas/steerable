#!/usr/bin/env python3
"""Run the exact four-cell P00 N3 behavioral pilot through the durable queue.

The module has deliberately lazy runtime imports because its three entry points
run under three different Python environments:

``queue``
    Runs under the worker image's system Python.  It validates immutable study
    and prerequisite evidence, requires exactly two idle allocated B200s, then
    supervises the policy server on logical GPU 0 and four fresh RoboLab
    processes on logical GPU 1.

``server``
    Runs under the pinned Cosmos environment.  It leaves official joint
    generation and decoding unchanged, binds the prospectively registered seed
    to every request, retains exact transformed inputs/actions/vision latents,
    and enforces one reset-scoped episode at a time.

``cell``
    Runs under the pinned RoboLab/Isaac environment.  It constructs a new
    timeout-only task, wraps the official packed-image client with the workshop
    recorder, and executes exactly 450 policy actions unless a separately
    recorded technical failure or safety abort occurs.

All large evidence remains below the GM PVC raw root.  Only bounded receipts
are written to ``job_dir/publish``.  The six-request N3 qualification is a
prerequisite and is never repeated by this launcher.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Mapping, Sequence
import uuid


NAMESPACE = "wmf_ablation_001_20260912"
STUDY_ID = "WMF-ABLATION-001"
MODEL_CONFIG = "N3"
PHASE = "pilot"
LAYOUT_PAIR_ID = "P00"
BLOCK_ID = "wmf_ablation_001_20260912__pilot__P00__N3"
QUEUE_ROLE = "n3"
EFFECTIVE_SEED = 2026091000
ACTION_CAP = 450
ACTION_HORIZON = 32
ACTION_DIM = 8
REQUEST_COUNT = 15
OBSERVATION_COUNT = 451
FINAL_EXECUTED_ACTIONS = 2

CONDITIONS = (
    ("reflected", "right", "WMFForecastReflectedRightTask"),
    ("reflected", "left", "WMFForecastReflectedLeftTask"),
    ("original", "left", "WMFForecastOriginalLeftTask"),
    ("original", "right", "WMFForecastOriginalRightTask"),
)
CELL_IDS = tuple(
    f"wmf1__pilot__P00__N3__{layout_arm}__{command}"
    for layout_arm, command, _task_name in CONDITIONS
)
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASK_FILES = {
    ("original", "left"): "original_left.py",
    ("original", "right"): "original_right.py",
    ("reflected", "left"): "reflected_left.py",
    ("reflected", "right"): "reflected_right.py",
}

ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
COSMOS_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
CHECKPOINT_AGGREGATE_SHA256 = "55895125805b12635ece5dd2e88453a1aca6a684bac556f114337ae448ffdd83"
N3_QUALIFICATION_RAW_SHA256 = "6dc6519f45e4974e88851137e30f4199c0d96e374216694c72a8924334d91a96"

ROBOLAB_ROOT = Path("/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241")
ROBOLAB_PYTHON = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
COSMOS_ROOT = Path("/data/users/ali/vla_wam/external/v3-clean/cosmos-nano-411d25b")
COSMOS_PYTHON = Path("/data/users/ali/vla_wam/envs/cosmos-nano-411d25b-v3-exact/bin/python")
CHECKPOINT_ROOT = Path("/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid")
RAW_ROOT = Path("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/pilot/N3/P00")
DEFAULT_PORT = 18011
MAX_PUBLISH_BYTES = 512 * 1024
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

QUEUE_RECEIPT_SCHEMA = "wmf-n3-behavioral-pilot-job-v1"
CELL_RECEIPT_SCHEMA = "wmf-n3-behavioral-pilot-cell-v1"
SERVER_READY_SCHEMA = "wmf-n3-behavioral-server-ready-v1"
SERVER_REQUEST_SCHEMA = "wmf-n3-behavioral-server-request-v1"
SERVER_EXIT_SCHEMA = "wmf-n3-behavioral-server-exit-v1"
TOPOLOGY_SCHEMA = "wmf-n3-two-b200-topology-v1"
CAPTURE_RECEIPT_SCHEMA = "wmf-forecast-layout-fixed-observation-capture-v1"
RECORDER_RECEIPT_SCHEMA = "wmf-forecast-recorder-qualification-job-v1"
N3_QUALIFICATION_SCHEMA = "wmf-n3-runtime-qualification-v1"
CONTEXT_RESET_SCOPE = "full_episode_temporal_and_cache_context"


class N3BehavioralPilotError(RuntimeError):
    """Fail-closed launcher error with a bounded stable reason code."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        if SAFE_ID_RE.fullmatch(reason) is None:
            reason = "internal_contract_error"
        super().__init__(reason if detail is None else f"{reason}: {detail}")
        self.reason = reason


def require(condition: bool, reason: str, detail: str | None = None) -> None:
    if not condition:
        raise N3BehavioralPilotError(reason, detail)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise N3BehavioralPilotError("noncanonical_json_value") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    require(resolved.is_file(), "evidence_file_missing", str(resolved))
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def immutable_json(path: Path, value: Mapping[str, Any], *, publish: bool = False) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "immutable_evidence_exists", str(path))
    payload = canonical_bytes(value)
    if publish:
        require(len(payload) <= MAX_PUBLISH_BYTES, "publish_receipt_too_large")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.parent.is_symlink(), "evidence_parent_is_symlink")
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
            raise N3BehavioralPilotError("immutable_evidence_exists", str(path)) from error
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, reason: str = "json_evidence_unreadable") -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise N3BehavioralPilotError(reason, str(path)) from error
    require(isinstance(value, dict), reason, "top-level JSON value is not an object")
    return value


def verify_exact_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(SHA256_RE.fullmatch(expected_sha256) is not None, "invalid_sha256", label)
    supplied = Path(path)
    resolved = supplied.resolve()
    require(supplied.is_absolute(), "evidence_path_not_absolute", label)
    require(not supplied.is_symlink(), "evidence_path_is_symlink", label)
    identity = file_identity(resolved)
    require(identity["sha256"] == expected_sha256, "evidence_sha256_mismatch", label)
    return identity


def _run_git(root: Path, *argv: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *argv],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise N3BehavioralPilotError("git_verification_unavailable") from error
    require(result.returncode == 0, "git_verification_failed", argv[0] if argv else "git")
    return result.stdout


def verify_clean_git(root: Path, commit: str, label: str) -> None:
    require(COMMIT_RE.fullmatch(commit) is not None, "invalid_commit", label)
    resolved = Path(root).resolve()
    require(resolved.is_dir(), "source_checkout_missing", label)
    require(_run_git(resolved, "rev-parse", "HEAD").strip() == commit, "source_commit_mismatch", label)
    require(
        not _run_git(resolved, "status", "--porcelain=v1", "--untracked-files=all"),
        "source_checkout_dirty",
        label,
    )


def safe_cell_component(cell_id: str) -> str:
    return cell_id.replace("__", "-").replace("_", "-")


class ServerProtocol:
    """Pure request-order/reset state machine used by the live server."""

    def __init__(self, *, start_cell_index: int = 0) -> None:
        require(0 <= start_cell_index <= len(CELL_IDS), "invalid_start_cell_index")
        self.next_cell_index = start_cell_index
        self.active: dict[str, Any] | None = None
        self.completed_cells: list[str] = list(CELL_IDS[:start_cell_index])
        self.used_context_ids: set[str] = set()

    def begin(
        self,
        request: Mapping[str, Any],
        *,
        server_context_id: str,
        temporal_reset_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        require(self.active is None, "server_episode_context_overlap")
        require(self.next_cell_index < len(CELL_IDS), "server_block_already_complete")
        require(
            isinstance(server_context_id, str) and server_context_id,
            "server_context_id_missing",
        )
        require(server_context_id not in self.used_context_ids, "server_context_id_reused")
        require(
            isinstance(temporal_reset_evidence, Mapping)
            and temporal_reset_evidence.get("passed") is True,
            "server_temporal_reset_evidence_missing",
        )
        require(
            temporal_reset_evidence.get("unresolved_mutable_temporal_fields") == [],
            "server_temporal_state_unresolved",
        )
        cell_id = CELL_IDS[self.next_cell_index]
        layout_arm, command, _task = CONDITIONS[self.next_cell_index]
        expected = {
            "study_id": STUDY_ID,
            "block_id": BLOCK_ID,
            "cell_id": cell_id,
            "condition_index": self.next_cell_index,
            "layout_arm": layout_arm,
            "command": command,
            "prompt": PROMPTS[command],
            "effective_seed": EFFECTIVE_SEED,
            "expected_actions": ACTION_CAP,
            "expected_requests": REQUEST_COUNT,
        }
        for key, wanted in expected.items():
            require(request.get(key) == wanted, "server_begin_mismatch", key)
        self.used_context_ids.add(server_context_id)
        self.active = {
            **expected,
            "server_context_id": server_context_id,
            "request_count": 0,
            "started_at_utc": utc_now(),
        }
        return {
            "passed": True,
            "reset_scope": CONTEXT_RESET_SCOPE,
            "server_context_id": server_context_id,
            "cache_reset_evidence": {
                "exclusive_active_episode": cell_id,
                "wrapper_request_index_reset_to_zero": True,
                "official_history_length": 1,
                "request_bound_seed": EFFECTIVE_SEED,
                "model_process_reused_but_episode_state_not_reused": True,
                **dict(temporal_reset_evidence),
            },
            "cell_id": cell_id,
            "condition_index": self.next_cell_index,
            "effective_seed": EFFECTIVE_SEED,
        }

    def validate_behavioral(self, request: Mapping[str, Any]) -> dict[str, Any]:
        require(self.active is not None, "behavioral_request_without_episode_reset")
        active = self.active
        request_index = active["request_count"]
        expected = {
            "wmf_request_type": "behavioral",
            "study_id": STUDY_ID,
            "block_id": BLOCK_ID,
            "cell_id": active["cell_id"],
            "condition_index": active["condition_index"],
            "layout_arm": active["layout_arm"],
            "command": active["command"],
            "prompt": active["prompt"],
            "sampling_seed": EFFECTIVE_SEED,
            "effective_seed": EFFECTIVE_SEED,
            "request_index": request_index,
            "action_step_start": request_index * ACTION_HORIZON,
            "server_context_id": active["server_context_id"],
        }
        for key, wanted in expected.items():
            require(request.get(key) == wanted, "server_behavioral_request_mismatch", key)
        require(request_index < REQUEST_COUNT, "server_request_count_exceeded")
        require(request_index * ACTION_HORIZON < ACTION_CAP, "server_request_after_action_cap")
        for key in (
            "observation/image",
            "observation/joint_position",
            "observation/gripper_position",
        ):
            require(key in request, "server_wire_input_missing", key)
        return expected

    def complete_behavioral(self) -> int:
        require(self.active is not None, "server_completion_without_active_episode")
        self.active["request_count"] += 1
        return int(self.active["request_count"])

    def end(self, request: Mapping[str, Any]) -> dict[str, Any]:
        require(self.active is not None, "server_end_without_active_episode")
        active = self.active
        for key in ("study_id", "block_id", "cell_id", "condition_index"):
            require(request.get(key) == active[key], "server_end_mismatch", key)
        require(
            request.get("server_context_id") == active["server_context_id"],
            "server_end_mismatch",
            "server_context_id",
        )
        actions = request.get("actions_executed")
        requests = request.get("request_count")
        completed = (
            request.get("status") == "completed"
            and actions == ACTION_CAP
            and requests == REQUEST_COUNT
            and active["request_count"] == REQUEST_COUNT
            and request.get("final_chunk_executed_actions") == FINAL_EXECUTED_ACTIONS
        )
        response = {
            "passed": completed,
            "server_context_id": request.get("server_context_id"),
            "cell_id": active["cell_id"],
            "condition_index": active["condition_index"],
            "server_request_count": active["request_count"],
            "actions_executed": actions,
            "client_request_count": requests,
            "status": "completed" if completed else "technical_invalid",
            "ended_at_utc": utc_now(),
        }
        self.active = None
        if completed:
            self.completed_cells.append(response["cell_id"])
            self.next_cell_index += 1
        return response


def validate_schedule(source_root: Path) -> dict[str, Any]:
    path = (
        Path(source_root)
        / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
    )
    schedule = load_json(path, "parallel_schedule_unreadable")
    rows = [row for row in schedule.get("jobs", []) if row.get("job_id") == BLOCK_ID]
    require(len(rows) == 1, "n3_p00_schedule_row_missing_or_duplicate")
    row = rows[0]
    require(row.get("phase") == PHASE, "n3_p00_schedule_mismatch", "phase")
    require(row.get("layout_pair_id") == LAYOUT_PAIR_ID, "n3_p00_schedule_mismatch", "layout")
    require(row.get("model_config") == MODEL_CONFIG, "n3_p00_schedule_mismatch", "model")
    require(row.get("candidate_effective_policy_seed") == EFFECTIVE_SEED, "n3_p00_schedule_mismatch", "seed")
    require(row.get("indivisible") is True, "n3_p00_schedule_not_indivisible")
    require(row.get("condition_order") == [f"{a}-{c}" for a, c, _ in CONDITIONS], "n3_p00_order_changed")
    require(row.get("ordered_cell_ids") == list(CELL_IDS), "n3_p00_cell_inventory_changed")
    contract = row.get("execution_contract", {})
    require(contract.get("sequential_conditions") is True, "n3_p00_not_sequential")
    require(contract.get("full_model_and_simulator_reset_before_each_condition") is True, "n3_p00_reset_contract_changed")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "row": row}


def _verify_descriptor(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), "file_descriptor_missing", label)
    path = value.get("path")
    digest = value.get("sha256")
    require(isinstance(path, str) and Path(path).is_absolute(), "file_descriptor_path_invalid", label)
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "file_descriptor_sha_invalid", label)
    observed = file_identity(Path(path))
    require(observed["sha256"] == digest, "file_descriptor_sha_mismatch", label)
    if "bytes" in value:
        require(observed["bytes"] == value["bytes"], "file_descriptor_size_mismatch", label)
    return observed


def validate_prerequisites(
    *,
    source_root: Path,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
    capture_receipt_path: Path,
    capture_receipt_sha256: str,
    recorder_receipt_path: Path,
    recorder_receipt_sha256: str,
    n3_qualification_receipt_path: Path,
    n3_qualification_receipt_sha256: str,
) -> dict[str, Any]:
    forecast = Path(source_root) / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    gate = recorder_job.verify_fixture_release(
        gate_receipt_path=gate_receipt_path,
        gate_receipt_sha256=gate_receipt_sha256,
        pose_manifest_path=pose_manifest_path,
        pose_manifest_sha256=pose_manifest_sha256,
        layout_arm="original",
    )
    reflected = recorder_job.verify_fixture_release(
        gate_receipt_path=gate_receipt_path,
        gate_receipt_sha256=gate_receipt_sha256,
        pose_manifest_path=pose_manifest_path,
        pose_manifest_sha256=pose_manifest_sha256,
        layout_arm="reflected",
    )
    require(gate["candidate_id"] == reflected["candidate_id"], "pose_arms_bind_different_candidates")

    capture_identity = verify_exact_file(capture_receipt_path, capture_receipt_sha256, "capture_receipt")
    capture = load_json(capture_receipt_path, "capture_receipt_unreadable")
    require(capture.get("schema_version") == CAPTURE_RECEIPT_SCHEMA, "capture_receipt_schema_changed")
    require(capture.get("status") == "passed", "capture_not_qualified")
    require(capture.get("layout_pair_id") == LAYOUT_PAIR_ID, "capture_layout_mismatch")
    require(
        isinstance(capture.get("pose_manifest"), Mapping)
        and capture["pose_manifest"].get("sha256") == pose_manifest_sha256,
        "capture_pose_mismatch",
    )
    require(
        isinstance(capture.get("gate_receipt"), Mapping)
        and capture["gate_receipt"].get("sha256") == gate_receipt_sha256,
        "capture_gate_mismatch",
    )
    require(capture.get("environment_seed") == EFFECTIVE_SEED, "capture_environment_seed_mismatch")
    require(capture.get("candidate_id") == gate["candidate_id"], "capture_candidate_mismatch")
    require(capture.get("model_request_count") == 0, "capture_contains_model_requests")
    require(capture.get("behavioral_action_count") == 0, "capture_contains_behavioral_actions")

    recorder_identity = verify_exact_file(recorder_receipt_path, recorder_receipt_sha256, "recorder_receipt")
    recorder = load_json(recorder_receipt_path, "recorder_receipt_unreadable")
    require(recorder.get("schema_version") == RECORDER_RECEIPT_SCHEMA, "recorder_receipt_schema_changed")
    require(recorder.get("status") == "passed" and recorder.get("exit_code") == 0, "recorder_not_qualified")
    require(recorder.get("layout_pair_id") == LAYOUT_PAIR_ID, "recorder_layout_mismatch")
    require(recorder.get("pose_manifest_sha256") == pose_manifest_sha256, "recorder_pose_mismatch")
    require(recorder.get("gate_receipt_sha256") == gate_receipt_sha256, "recorder_gate_mismatch")
    require(recorder.get("environment_seed") == EFFECTIVE_SEED, "recorder_environment_seed_mismatch")
    require(recorder.get("actions_executed") == ACTION_CAP, "recorder_action_count_changed")
    require(recorder.get("observation_count") == OBSERVATION_COUNT, "recorder_observation_count_changed")
    require(recorder.get("model_request_count") == 0, "recorder_contains_model_requests")
    require(recorder.get("behavioral_episode_count") == 0, "recorder_claims_behavioral_episode")
    require(recorder.get("recording_qualification_count") == 1, "recorder_qualification_count_changed")
    child_identity = _verify_descriptor(recorder.get("raw_child_receipt"), "recorder_child_receipt")
    child = load_json(Path(child_identity["path"]), "recorder_child_receipt_unreadable")
    require(child.get("native_timing_supported") is True, "recorder_native_timing_unqualified")
    timing_identity = _verify_descriptor(child.get("timing_support"), "recorder_timing_support")
    timing = load_json(Path(timing_identity["path"]), "recorder_timing_unreadable")
    require(timing.get("supported") is True and timing.get("status") == "supported", "recorder_native_timing_unqualified")

    n3_identity = verify_exact_file(
        n3_qualification_receipt_path,
        n3_qualification_receipt_sha256,
        "n3_qualification_receipt",
    )
    require(
        n3_qualification_receipt_sha256 == N3_QUALIFICATION_RAW_SHA256,
        "n3_qualification_receipt_is_not_released_receipt",
    )
    n3 = load_json(n3_qualification_receipt_path, "n3_qualification_receipt_unreadable")
    require(n3.get("schema_version") == N3_QUALIFICATION_SCHEMA, "n3_qualification_schema_changed")
    require(n3.get("status") == "passed" and n3.get("qualified") is True, "n3_not_qualified")
    require(n3.get("model_config") == MODEL_CONFIG, "n3_qualification_model_mismatch")
    require(n3.get("effective_seed") == EFFECTIVE_SEED, "n3_qualification_seed_mismatch")
    require(n3.get("generation_request_count") == 6, "n3_qualification_request_count_changed")
    require(n3.get("robot_episode_count") == 0, "n3_qualification_claims_robot_episode")
    require(n3.get("source", {}).get("commit") == COSMOS_COMMIT, "n3_source_identity_changed")
    require(n3.get("checkpoint", {}).get("revision") == CHECKPOINT_REVISION, "n3_checkpoint_revision_changed")
    require(n3.get("checkpoint", {}).get("payload_aggregate_sha256") == CHECKPOINT_AGGREGATE_SHA256, "n3_checkpoint_hash_changed")

    return {
        "gate_receipt": gate["gate_receipt"],
        "pose_manifest": gate["pose_manifest"],
        "candidate_id": gate["candidate_id"],
        "candidate_payload_sha256": gate["candidate_payload_sha256"],
        "accepted_gate_record_sha256": gate["accepted_gate_record_sha256"],
        "capture_receipt": capture_identity,
        "recorder_receipt": recorder_identity,
        "recorder_child_receipt": child_identity,
        "native_timing_support": timing_identity,
        "n3_qualification_receipt": n3_identity,
        "generation_qualification_requests_reused_not_rerun": 6,
    }


def _query_nvidia(
    columns: Sequence[str], *, compute: bool, executable: Path | str = "nvidia-smi"
) -> list[dict[str, str]]:
    prefix = "--query-compute-apps=" if compute else "--query-gpu="
    try:
        result = subprocess.run(
            [str(executable), prefix + ",".join(columns), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise N3BehavioralPilotError("nvidia_smi_unavailable") from error
    require(result.returncode == 0, "nvidia_smi_failed")
    rows: list[dict[str, str]] = []
    for values in csv.reader(io.StringIO(result.stdout)):
        if not values or not any(item.strip() for item in values):
            continue
        require(len(values) == len(columns), "nvidia_smi_output_shape_changed")
        rows.append({key: value.strip() for key, value in zip(columns, values)})
    return rows


def verify_two_idle_b200s(executable: Path | str = "nvidia-smi") -> dict[str, Any]:
    devices = _query_nvidia(
        ("index", "uuid", "name", "driver_version", "memory.total"),
        compute=False,
        executable=executable,
    )
    require(len(devices) == 2, "n3_worker_does_not_expose_exactly_two_gpus")
    require({row["index"] for row in devices} == {"0", "1"}, "n3_gpu_logical_indices_changed")
    require(all(row["name"] == "NVIDIA B200" for row in devices), "n3_worker_gpu_is_not_b200")
    require(len({row["uuid"] for row in devices}) == 2, "n3_worker_gpu_uuid_not_distinct")
    require(all(row["uuid"].startswith("GPU-") for row in devices), "n3_worker_gpu_uuid_invalid")
    processes = _query_nvidia(
        ("gpu_uuid", "pid", "process_name", "used_memory"),
        compute=True,
        executable=executable,
    )
    require(not processes, "n3_worker_has_preexisting_compute_process")
    return {
        "schema_version": TOPOLOGY_SCHEMA,
        "status": "passed",
        "devices": devices,
        "model_gpu": devices[0],
        "simulator_gpu": devices[1],
        "preexisting_compute_process_count": 0,
        "assignment": {
            "model_child_cuda_visible_devices": "0",
            "simulator_child_cuda_visible_devices": "1",
            "indices_are_allocation_relative": True,
        },
        "outer_nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "recorded_at_utc": utc_now(),
    }


def _proc_identity(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    try:
        process_group_id: int | None = os.getpgid(process.pid)
    except ProcessLookupError:
        process_group_id = None
    value: dict[str, Any] = {
        "pid": process.pid,
        "process_group_id": process_group_id,
        "hostname": socket.gethostname(),
        "start_new_session": True,
        "returncode_at_identity_capture": process.poll(),
    }
    try:
        fields = Path(f"/proc/{process.pid}/stat").read_text(encoding="utf-8").split()
        value["proc_start_ticks"] = int(fields[21])
        value["linux_boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        value["proc_start_ticks"] = None
        value["linux_boot_id"] = None
    return value


def terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


_ACTIVE_CHILDREN: list[subprocess.Popen[bytes]] = []
_SIGNAL_ACTIVE = False


def _termination_handler(signum: int, _frame: Any) -> None:
    global _SIGNAL_ACTIVE
    if _SIGNAL_ACTIVE:
        return
    _SIGNAL_ACTIVE = True
    for child in reversed(list(_ACTIVE_CHILDREN)):
        try:
            terminate_process_group(child, grace_seconds=3.0)
        except BaseException:
            pass
    raise SystemExit(128 + signum)


@contextmanager
def installed_signal_handlers():
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _termination_handler)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def validate_queue_invocation(
    *, source_root: Path, job_dir: Path, study_commit: str, job_id: str
) -> dict[str, Any]:
    """Bind this process to the exact immutable two-GPU queue descriptor."""

    require(SAFE_ID_RE.fullmatch(job_id) is not None, "invalid_job_id")
    require(COMMIT_RE.fullmatch(study_commit) is not None, "invalid_study_commit")
    source_root = Path(source_root).resolve()
    job_dir = Path(job_dir).resolve()
    verify_clean_git(source_root, study_commit, "study")
    descriptor_path = job_dir / "descriptor.json"
    require(not descriptor_path.is_symlink(), "queue_descriptor_is_symlink")
    descriptor = load_json(descriptor_path, "queue_descriptor_unreadable")
    require(descriptor.get("schema_version") == "wmf-cluster-job-v1", "queue_descriptor_schema_changed")
    require(descriptor.get("namespace") == NAMESPACE, "queue_descriptor_namespace_mismatch")
    require(descriptor.get("job_id") == job_id, "queue_descriptor_job_id_mismatch")
    require(descriptor.get("source_commit") == study_commit, "queue_descriptor_source_commit_mismatch")
    require(descriptor.get("role") == QUEUE_ROLE, "n3_behavioral_job_requires_dedicated_worker_role")
    require(descriptor.get("released") is True, "queue_descriptor_not_released")
    return {**file_identity(descriptor_path), "role": QUEUE_ROLE, "job_id": job_id}


def build_model_environment(
    *, source_root: Path, attempt_root: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    for name in ("DISPLAY", "LD_PRELOAD"):
        environment.pop(name, None)
    runtime = Path(attempt_root) / "server" / "runtime"
    directories = {
        "tmp": runtime / "tmp",
        "torchinductor": runtime / "torchinductor",
        "pycache": runtime / "pycache",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    original_pythonpath = environment.get("PYTHONPATH")
    python_roots = [str(COSMOS_ROOT), str(Path(source_root).resolve())]
    if original_pythonpath:
        python_roots.append(original_pythonpath)
    environment.update(
        CUDA_VISIBLE_DEVICES="0",
        DS_IGNORE_CUDA_DETECTION="1",
        HF_HOME="/data/users/ali/vla_wam/cache/huggingface-cosmos",
        PATH=(
            "/data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/compat-bin:"
            "/data/users/ali/vla_wam/envs/cosmos-nano-411d25b-v3-exact/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        LD_LIBRARY_PATH=(
            "/data/users/ali/vla_wam/envs/cosmos-nano-411d25b-v3-exact/lib/"
            "python3.13/site-packages/nvidia/cudnn/lib:"
            "/data/users/ali/vla_wam/envs/isaac-system-libs/lib:"
            "/data/users/jsalfity/glvnd/lib"
        ),
        PYTHONPATH=os.pathsep.join(python_roots),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPYCACHEPREFIX=str(directories["pycache"]),
        TORCHINDUCTOR_CACHE_DIR=str(directories["torchinductor"]),
        TMPDIR=str(directories["tmp"]),
    )
    return environment


def build_simulator_environment(
    *, source_root: Path, state_parent: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    forecast = Path(source_root).resolve() / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    environment = recorder_job.build_child_environment(
        source_root=Path(source_root).resolve(),
        state_parent=Path(state_parent).resolve(),
        robolab_root=ROBOLAB_ROOT,
        base=base,
    )
    environment["CUDA_VISIBLE_DEVICES"] = "1"
    return environment


def build_server_command(
    *, source_root: Path, attempt_root: Path, port: int, study_commit: str
) -> list[str]:
    script = Path(source_root) / "workshops/corl2026_world_models/experiments/forecast_layout/n3_pilot_live.py"
    return [
        os.path.abspath(os.fspath(COSMOS_PYTHON)),
        str(script.resolve()),
        "server",
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--attempt-root", str(Path(attempt_root).resolve()),
        "--port", str(port),
    ]


def build_cell_command(
    *,
    source_root: Path,
    attempt_root: Path,
    study_commit: str,
    gate_receipt: Path,
    gate_receipt_sha256: str,
    pose_manifest: Path,
    pose_manifest_sha256: str,
    port: int,
    condition_index: int,
) -> list[str]:
    script = Path(source_root) / "workshops/corl2026_world_models/experiments/forecast_layout/n3_pilot_live.py"
    layout_arm, command, _task = CONDITIONS[condition_index]
    return [
        # Preserve this lexical venv path; resolving it loses venv packages.
        os.path.abspath(os.fspath(ROBOLAB_PYTHON)),
        str(script.resolve()),
        "cell",
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--attempt-root", str(Path(attempt_root).resolve()),
        "--gate-receipt", str(Path(gate_receipt).resolve()),
        "--gate-receipt-sha256", gate_receipt_sha256,
        "--pose-manifest", str(Path(pose_manifest).resolve()),
        "--pose-manifest-sha256", pose_manifest_sha256,
        "--layout-arm", layout_arm,
        "--command", command,
        "--condition-index", str(condition_index),
        "--remote-host", "127.0.0.1",
        "--remote-port", str(port),
    ]


@dataclass
class _BehaviorCapture:
    writer: Any
    generator_calls: int = 0
    decode_calls: int = 0
    model_input_artifact: dict[str, Any] | None = None
    generated_action_artifact: dict[str, Any] | None = None
    latent_artifact: dict[str, Any] | None = None
    latent_identity: dict[str, Any] | None = None
    decoder_input_artifact: dict[str, Any] | None = None
    decoder_output_artifact: dict[str, Any] | None = None


class _BehaviorModelProxy:
    """Retain the official joint generation and online decoded future."""

    def __init__(self, delegate: Any, qualification: Any) -> None:
        self._delegate = delegate
        self._qualification = qualification
        self.active: _BehaviorCapture | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def generate_samples_from_batch(self, data_batch: Any, *args: Any, **kwargs: Any) -> Any:
        capture = self.active
        require(capture is not None, "generation_outside_authenticated_behavioral_request")
        capture.generator_calls += 1
        require(capture.generator_calls == 1, "behavioral_request_generated_more_than_once")
        capture.model_input_artifact = capture.writer.write("exact_transformed_model_input", data_batch)
        samples = self._delegate.generate_samples_from_batch(data_batch, *args, **kwargs)
        require(isinstance(samples, Mapping), "joint_generation_result_not_mapping")
        require("action" in samples and "vision" in samples, "joint_generation_missing_action_or_vision")
        action = samples["action"][0]
        latent = samples["vision"][0]
        capture.generated_action_artifact = capture.writer.write("raw_generated_action", action)
        capture.latent_artifact = capture.writer.write("retained_vision_latent", latent)
        capture.latent_identity = self._qualification.value_identity(latent)
        return samples

    def decode(self, latent: Any) -> Any:
        capture = self.active
        require(capture is not None, "decode_outside_authenticated_behavioral_request")
        require(capture.generator_calls == 1, "decode_preceded_joint_generation")
        require(
            self._qualification.value_identity(latent) == capture.latent_identity,
            "decoder_input_differs_from_retained_latent",
        )
        capture.decode_calls += 1
        require(capture.decode_calls == 1, "behavioral_request_decoded_more_than_once")
        capture.decoder_input_artifact = capture.writer.write("exact_decoder_input_latent", latent)
        decoded = self._delegate.decode(latent)
        capture.decoder_output_artifact = capture.writer.write("raw_decoder_output", decoded)
        return decoded


_TEMPORAL_FIELD_MARKERS = (
    "cache",
    "history",
    "buffer",
    "session",
    "context",
    "temporal",
    "past_key",
    "episode",
)
_STATIC_FRAMEWORK_FIELDS = {"_buffers", "_non_persistent_buffers_set"}


def _temporal_field_inventory(value: Any, *, component: str) -> dict[str, Any]:
    """Inventory top-level state that could carry one episode into another.

    PyTorch's registered-buffer containers are architecture/model state, not a
    request cache, so they are recorded but not classified as unresolved.  Any
    other nonempty built-in mutable container with a temporal-looking name
    fails the reset gate.  Scalar configuration such as ``history_length`` is
    retained as evidence but is not mutable episode state.
    """

    rows: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for name, field_value in sorted(vars(value).items()):
        if not any(marker in name.lower() for marker in _TEMPORAL_FIELD_MARKERS):
            continue
        if name in _STATIC_FRAMEWORK_FIELDS:
            classification = "pytorch_static_registry"
            size = len(field_value) if isinstance(field_value, (dict, list, set, tuple)) else None
        elif isinstance(field_value, (dict, list, set)):
            size = len(field_value)
            classification = "empty_mutable_container" if size == 0 else "unresolved_mutable_container"
            if size:
                unresolved.append(f"{component}.{name}")
        elif isinstance(field_value, tuple):
            size = len(field_value)
            classification = "immutable_tuple"
        elif field_value is None or isinstance(field_value, (str, int, float, bool, Path)):
            size = None
            classification = "scalar_or_path_configuration"
        else:
            size = None
            classification = "object_identity_only"
        rows.append(
            {
                "name": name,
                "type": f"{type(field_value).__module__}.{type(field_value).__qualname__}",
                "classification": classification,
                "size": size,
            }
        )
    return {
        "component": component,
        "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "temporal_fields": rows,
        "unresolved_mutable_temporal_fields": unresolved,
    }


def _temporal_state_scan(service: Any, model_proxy: _BehaviorModelProxy) -> dict[str, Any]:
    components = [
        _temporal_field_inventory(service, component="official_service"),
        _temporal_field_inventory(service.pipe, component="official_inference_pipe"),
        _temporal_field_inventory(model_proxy._delegate, component="official_model"),
        _temporal_field_inventory(model_proxy, component="behavioral_model_proxy"),
    ]
    unresolved = sorted(
        field
        for row in components
        for field in row["unresolved_mutable_temporal_fields"]
    )
    return {
        "components": components,
        "unresolved_mutable_temporal_fields": unresolved,
        "model_proxy_active": model_proxy.active is not None,
    }


def _reset_temporal_context(
    *,
    service: Any,
    model_proxy: _BehaviorModelProxy,
    protocol: ServerProtocol,
    torch_module: Any,
    numpy_module: Any,
    server_process_context_id: str,
) -> tuple[str, dict[str, Any]]:
    """Reset the mutable wrapper/RNG state and emit one episode identity."""

    require(protocol.active is None, "server_episode_context_overlap")
    require(model_proxy.active is None, "server_capture_not_clear_at_episode_begin")
    before = _temporal_state_scan(service, model_proxy)
    require(
        before["unresolved_mutable_temporal_fields"] == [],
        "official_model_temporal_state_detected",
        ",".join(before["unresolved_mutable_temporal_fields"]),
    )
    torch_module.manual_seed(EFFECTIVE_SEED)
    torch_module.cuda.manual_seed_all(EFFECTIVE_SEED)
    service._rng = numpy_module.random.default_rng(EFFECTIVE_SEED)
    after = _temporal_state_scan(service, model_proxy)
    require(
        after["unresolved_mutable_temporal_fields"] == [],
        "official_model_temporal_state_detected_after_reset",
        ",".join(after["unresolved_mutable_temporal_fields"]),
    )
    context_id = (
        f"{server_process_context_id}:cell-{protocol.next_cell_index:02d}:"
        f"{uuid.uuid4().hex}"
    )
    return context_id, {
        "passed": True,
        "reset_scope": CONTEXT_RESET_SCOPE,
        "server_process_context_id": server_process_context_id,
        "episode_context_id": context_id,
        "wrapper_protocol_active_before_reset": False,
        "model_proxy_active_before_reset": False,
        "request_index_reset_to_zero": True,
        "torch_cpu_rng_reseeded": True,
        "torch_cuda_rngs_reseeded": True,
        "official_numpy_generator_reinitialized": True,
        "effective_seed": EFFECTIVE_SEED,
        "official_history_length": int(service.cfg.history_length),
        "model_process_reused": True,
        "model_reinstantiated": False,
        "scan_before": before,
        "scan_after": after,
        "unresolved_mutable_temporal_fields": [],
    }


def _validate_behavioral_runtime(service: Any) -> dict[str, Any]:
    cfg = service.cfg
    observed = {
        "checkpoint_path": str(Path(cfg.checkpoint_path).resolve()),
        "domain_name": cfg.domain_name,
        "decode_video": bool(cfg.decode_video),
        "guidance": float(cfg.guidance),
        "denoising_steps": int(cfg.num_steps),
        "shift": float(cfg.shift),
        "conditioning_fps": float(cfg.conditioning_fps),
        "resolution_setting": str(cfg.resolution),
        "action_chunk_size": int(cfg.action_chunk_size),
        "action_dimensions": int(cfg.action_dim),
        "action_space": cfg.action_space,
        "use_state": bool(cfg.use_state),
        "history_length": int(cfg.history_length),
        "deterministic_seed": bool(cfg.deterministic_seed),
    }
    wanted = {
        "checkpoint_path": str(CHECKPOINT_ROOT.resolve()),
        "domain_name": "droid_lerobot",
        "decode_video": True,
        "guidance": 3.0,
        "denoising_steps": 4,
        "shift": 5.0,
        "conditioning_fps": 15.0,
        "resolution_setting": "480",
        "action_chunk_size": ACTION_HORIZON,
        "action_dimensions": ACTION_DIM,
        "action_space": "joint_pos",
        "use_state": True,
        "history_length": 1,
        "deterministic_seed": True,
    }
    changed = [key for key, expected in wanted.items() if observed.get(key) != expected]
    require(not changed, "n3_behavioral_runtime_mismatch", ",".join(changed))
    return observed


def _server_args(server: Any, *, output_dir: Path, host: str, port: int) -> Any:
    return server.RobolabServerArgs(
        checkpoint_path=str(CHECKPOINT_ROOT.resolve()),
        hf_revision=CHECKPOINT_REVISION,
        allow_dcp_checkpoint=False,
        output_dir=Path(output_dir).resolve(),
        domain_name="droid_lerobot",
        decode_video=True,
        seed=EFFECTIVE_SEED,
        deterministic_seed=True,
        guidance=3.0,
        num_steps=4,
        shift=5.0,
        resolution="480",
        conditioning_fps=15.0,
        action_chunk_size=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        action_space="joint_pos",
        use_state=True,
        history_length=1,
        host=host,
        port=port,
    )


def run_server(args: argparse.Namespace) -> int:
    """Run the official decoded N3 service with behavioral-only instrumentation."""

    source_root = Path(args.source_root).resolve()
    attempt_root = Path(args.attempt_root).resolve()
    server_root = attempt_root / "server"
    ready_path = server_root / "ready.json"
    failure_path = server_root / "failure.json"
    require(not ready_path.exists(), "server_ready_receipt_already_exists")
    verify_clean_git(source_root, args.study_commit, "study")
    forecast = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import n3_first_live as qualification

    try:
        contract = qualification.load_contract(forecast / "n3_first_live_contract.json")
        source_identity = qualification.validate_external_source(COSMOS_ROOT, contract)
        checkpoint_identity = qualification.validate_checkpoint(CHECKPOINT_ROOT, contract)
        import torch

        require(torch.cuda.is_available(), "n3_server_cuda_unavailable")
        require(torch.cuda.device_count() == 1, "n3_server_must_see_one_logical_gpu")
        require(torch.cuda.get_device_name(0) == "NVIDIA B200", "n3_server_gpu_is_not_b200")
        gpu = qualification._collect_gpu_evidence(torch, "NVIDIA B200")
        server = qualification._import_official_server(
            COSMOS_ROOT,
            pycache_root=server_root / "runtime" / "official_pycache",
        )
        protocol = ServerProtocol(start_cell_index=args.start_cell_index)
        protocol_lock = threading.Lock()
        server_process_context_id = (
            f"n3-behavioral:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        )

        class BehavioralService(server.RobolabPolicyService):
            def _build_setup_args(self, values: Any) -> Any:
                setup = super()._build_setup_args(values)
                # Proven local-checkpoint compatibility patch; model transforms,
                # denoising, action extraction, and VAE decoding remain official.
                return setup.model_copy(update={"guardrails": False})

            def _next_seed(self) -> int:
                require(protocol.active is not None, "official_seed_requested_without_active_episode")
                return EFFECTIVE_SEED

            def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
                with protocol_lock:
                    control = observation.get("wmf_control")
                    if control == "begin_episode":
                        episode_context_id, reset_evidence = _reset_temporal_context(
                            service=self,
                            model_proxy=model_proxy,
                            protocol=protocol,
                            torch_module=torch,
                            numpy_module=qualification.np,
                            server_process_context_id=server_process_context_id,
                        )
                        return protocol.begin(
                            observation,
                            server_context_id=episode_context_id,
                            temporal_reset_evidence=reset_evidence,
                        )
                    if control == "end_episode":
                        require(model_proxy.active is None, "server_capture_not_clear_at_episode_end")
                        return protocol.end(observation)
                    if control == "ping":
                        return {
                            "passed": True,
                            "server_process_context_id": server_process_context_id,
                        }
                    expected = protocol.validate_behavioral(observation)
                    cell_id = expected["cell_id"]
                    request_index = expected["request_index"]
                    request_root = (
                        attempt_root / "server" / "requests" / safe_cell_component(cell_id)
                        / f"request-{request_index:02d}"
                    )
                    writer = qualification._ArtifactWriter(request_root / "payloads")
                    capture = _BehaviorCapture(writer=writer)
                    model_proxy.active = capture
                    started_wall_ns = time.time_ns()
                    started_monotonic_ns = time.monotonic_ns()
                    wire_artifact = writer.write("exact_wire_request", observation)
                    try:
                        output = super().infer(observation)
                        torch.cuda.synchronize()
                        require(isinstance(output, Mapping), "official_response_not_mapping")
                        action = qualification.np.asarray(output.get("action"))
                        video = qualification.np.asarray(output.get("video"))
                        require(action.shape == (ACTION_HORIZON, ACTION_DIM), "official_action_shape_changed")
                        require(qualification.np.isfinite(action).all(), "official_action_nonfinite")
                        require(
                            video.ndim == 4 and video.shape[0] == ACTION_HORIZON + 1
                            and video.shape[-1] == 3 and video.dtype == qualification.np.uint8,
                            "official_decoded_future_shape_changed",
                        )
                        require(capture.generator_calls == 1, "joint_generation_call_count_changed")
                        require(capture.decode_calls == 1, "online_decode_call_count_changed")
                        returned_artifact = writer.write("official_returned_response", output)
                        receipt = {
                            "schema_version": SERVER_REQUEST_SCHEMA,
                            "status": "passed",
                            "study_id": STUDY_ID,
                            "block_id": BLOCK_ID,
                            "cell_id": cell_id,
                            "condition_index": expected["condition_index"],
                            "request_index": request_index,
                            "action_step_start": expected["action_step_start"],
                            "sampling_seed": EFFECTIVE_SEED,
                            "server_context_id": expected["server_context_id"],
                            "started_wall_time_ns": started_wall_ns,
                            "started_monotonic_ns": started_monotonic_ns,
                            "completed_wall_time_ns": time.time_ns(),
                            "completed_monotonic_ns": time.monotonic_ns(),
                            "wire_request": wire_artifact,
                            "exact_transformed_model_input": capture.model_input_artifact,
                            "raw_generated_action": capture.generated_action_artifact,
                            "retained_vision_latent": capture.latent_artifact,
                            "exact_decoder_input_latent": capture.decoder_input_artifact,
                            "raw_decoder_output": capture.decoder_output_artifact,
                            "official_returned_response": returned_artifact,
                            "returned_action_shape": list(action.shape),
                            "decoded_future_shape": list(video.shape),
                            "joint_generation_calls": capture.generator_calls,
                            "decode_calls": capture.decode_calls,
                            "generation_qualification_request": False,
                            "behavioral_model_request": True,
                        }
                        receipt_path = request_root / "request_receipt.json"
                        immutable_json(receipt_path, receipt)
                        protocol.complete_behavioral()
                        return {
                            **dict(output),
                            "wmf_behavioral_request": True,
                            "wmf_cell_id": cell_id,
                            "wmf_condition_index": expected["condition_index"],
                            "wmf_request_index": request_index,
                            "wmf_action_step_start": expected["action_step_start"],
                            "wmf_sampling_seed": EFFECTIVE_SEED,
                            "wmf_server_context_id": expected["server_context_id"],
                            "wmf_server_request_receipt": file_identity(receipt_path),
                        }
                    except BaseException as error:
                        if not (request_root / "technical_failure.json").exists():
                            immutable_json(
                                request_root / "technical_failure.json",
                                {
                                    "schema_version": SERVER_REQUEST_SCHEMA,
                                    "status": "technical_failure",
                                    "cell_id": cell_id,
                                    "request_index": request_index,
                                    "error_type": type(error).__name__,
                                    "reason": getattr(error, "reason", None),
                                    "traceback": traceback.format_exc(),
                                    "recorded_at_utc": utc_now(),
                                },
                            )
                        raise
                    finally:
                        model_proxy.active = None

        service = BehavioralService(
            _server_args(server, output_dir=server_root / "model_output", host=args.host, port=args.port)
        )
        runtime = _validate_behavioral_runtime(service)
        model_proxy = _BehaviorModelProxy(service.model, qualification)
        service.model = model_proxy
        ready = {
            "schema_version": SERVER_READY_SCHEMA,
            "status": "ready",
            "study_id": STUDY_ID,
            "block_id": BLOCK_ID,
            "model_config": MODEL_CONFIG,
            "server_process_context_id": server_process_context_id,
            "host": args.host,
            "port": args.port,
            "source_commit": args.study_commit,
            "cosmos_source": source_identity,
            "checkpoint": checkpoint_identity,
            "runtime": runtime,
            "gpu": gpu,
            "process": {"pid": os.getpid(), "hostname": socket.gethostname()},
            "expected_cell_order": list(CELL_IDS),
            "completed_prefix_cell_ids": list(CELL_IDS[: args.start_cell_index]),
            "start_cell_index": args.start_cell_index,
            "expected_behavioral_request_count": len(CELL_IDS) * REQUEST_COUNT,
            "generation_qualification_requests_rerun": 0,
            "ready_at_utc": utc_now(),
        }
        immutable_json(ready_path, ready)
        server_cls = server._load_openpi_websocket_policy_server()
        server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()
        return 0
    except BaseException as error:
        server_root.mkdir(parents=True, exist_ok=True)
        if not failure_path.exists():
            immutable_json(
                failure_path,
                {
                    "schema_version": SERVER_EXIT_SCHEMA,
                    "status": "technical_failure",
                    "error_type": type(error).__name__,
                    "reason": getattr(error, "reason", None),
                    "traceback": traceback.format_exc(),
                    "recorded_at_utc": utc_now(),
                },
            )
        raise


def _validate_server_behavioral_response(
    response: Mapping[str, Any], *, cell_id: str, condition_index: int, request_index: int
) -> None:
    expected = {
        "wmf_behavioral_request": True,
        "wmf_cell_id": cell_id,
        "wmf_condition_index": condition_index,
        "wmf_request_index": request_index,
        "wmf_action_step_start": request_index * ACTION_HORIZON,
        "wmf_sampling_seed": EFFECTIVE_SEED,
    }
    for key, value in expected.items():
        require(response.get(key) == value, "n3_server_response_metadata_mismatch", key)
    context = response.get("wmf_server_context_id")
    require(isinstance(context, str) and context, "n3_server_response_context_missing")
    _verify_descriptor(response.get("wmf_server_request_receipt"), "server_request_receipt")


def run_cell(args: argparse.Namespace) -> int:
    """Run exactly one fresh RoboLab process for one ordered P00 condition."""

    source_root = Path(args.source_root).resolve()
    attempt_root = Path(args.attempt_root).resolve()
    condition_index = args.condition_index
    require(0 <= condition_index < len(CONDITIONS), "cell_condition_index_invalid")
    layout_arm, command, task_name = CONDITIONS[condition_index]
    require(args.layout_arm == layout_arm and args.command == command, "cell_condition_identity_mismatch")
    cell_id = CELL_IDS[condition_index]
    cell_root = attempt_root / "cells" / f"{condition_index:02d}-{safe_cell_component(cell_id)}"
    receipt_path = cell_root / "cell_receipt.json"
    failure_path = cell_root / "technical_failure.json"
    require(not cell_root.exists(), "cell_attempt_directory_already_exists")
    cell_root.mkdir(parents=True)
    verify_clean_git(source_root, args.study_commit, "study")
    verify_clean_git(ROBOLAB_ROOT, ROBOLAB_COMMIT, "RoboLab")
    forecast = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import recorder_qualification_job as recorder_job

    release = recorder_job.verify_fixture_release(
        gate_receipt_path=Path(args.gate_receipt),
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=Path(args.pose_manifest),
        pose_manifest_sha256=args.pose_manifest_sha256,
        layout_arm=layout_arm,
    )
    os.environ.update(
        WMF_FORECAST_POSE_MANIFEST=str(Path(args.pose_manifest).resolve()),
        WMF_FORECAST_POSE_MANIFEST_SHA256=args.pose_manifest_sha256,
        WMF_FORECAST_LAYOUT_PAIR_ID=LAYOUT_PAIR_ID,
    )

    simulation_app = None
    env = None
    client = None
    recorder = None
    try:
        import cv2  # noqa: F401 - required before Isaac/RoboLab imports.
        from isaaclab.app import AppLauncher

        launch_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(launch_parser)
        launch_args, _ = launch_parser.parse_known_args(["--headless"])
        launch_args.enable_cameras = True
        launcher = AppLauncher(launch_args)
        simulation_app = launcher.app

        import numpy as np
        import robolab
        import robolab.constants
        from policies.cosmos3.client import Cosmos3Client
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.eval.episode import run_episode
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        from fixture_tasks import settle_for_recording_reset, success_measurements
        from recording_adapter import (
            FixedDurationEnvProxy,
            ForecastRecordingAdapter,
            RecordingClientMixin,
            assert_fixed_duration_environment,
            verify_journal,
        )

        require(Path(robolab.__file__).resolve().is_relative_to(ROBOLAB_ROOT), "effective_robolab_import_unpinned")
        simulator_output = cell_root / "native_simulator"
        simulator_output.mkdir()
        set_output_dir(str(simulator_output))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = False
        robolab.constants.DEBUG = False
        task_path = forecast / "task_files" / TASK_FILES[(layout_arm, command)]
        auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)

        identity = {
            "attempt_id": f"{attempt_root.name}:{condition_index:02d}",
            "cell_id": cell_id,
            "stage": PHASE,
            "layout_pair_id": LAYOUT_PAIR_ID,
            "layout_arm": layout_arm,
            "command": command,
            "prompt": PROMPTS[command],
            "model_config": MODEL_CONFIG,
            "effective_seed": EFFECTIVE_SEED,
            "source_identity": (
                f"study:{args.study_commit};robolab:{ROBOLAB_COMMIT};"
                f"cosmos:{COSMOS_COMMIT};pose:{args.pose_manifest_sha256}"
            ),
            "checkpoint_identity": (
                f"revision:{CHECKPOINT_REVISION};aggregate:{CHECKPOINT_AGGREGATE_SHA256}"
            ),
        }
        recorder = ForecastRecordingAdapter(cell_root / "recording", identity)

        class BoundN3Client(Cosmos3Client):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.wmf_request_index = 0
                self.wmf_server_context_id: str | None = None
                self.wmf_episode_active = False
                self.wmf_end_receipt: dict[str, Any] | None = None
                self.wmf_end_error: str | None = None

            def begin_episode(self) -> Mapping[str, Any]:
                require(not self.wmf_episode_active, "client_episode_context_overlap")
                request = {
                    "wmf_control": "begin_episode",
                    "study_id": STUDY_ID,
                    "block_id": BLOCK_ID,
                    "cell_id": cell_id,
                    "condition_index": condition_index,
                    "layout_arm": layout_arm,
                    "command": command,
                    "prompt": PROMPTS[command],
                    "effective_seed": EFFECTIVE_SEED,
                    "expected_actions": ACTION_CAP,
                    "expected_requests": REQUEST_COUNT,
                }
                response = self.client.infer(request)
                require(isinstance(response, Mapping), "server_begin_response_not_mapping")
                require(response.get("passed") is True, "server_context_reset_failed")
                require(response.get("reset_scope") == CONTEXT_RESET_SCOPE, "server_context_reset_scope_changed")
                require(response.get("cell_id") == cell_id, "server_begin_cell_mismatch")
                context = response.get("server_context_id")
                require(isinstance(context, str) and context, "server_context_id_missing")
                self.wmf_server_context_id = context
                self.wmf_episode_active = True
                return dict(response)

            def _pack_request(self, extracted_obs: dict[str, Any], instruction: str) -> dict[str, Any]:
                require(self.wmf_episode_active, "client_request_without_server_context")
                require(instruction == PROMPTS[command], "client_prompt_changed")
                require(self.wmf_request_index < REQUEST_COUNT, "client_request_count_exceeded")
                request = super()._pack_request(extracted_obs, instruction)
                request.update(
                    wmf_request_type="behavioral",
                    study_id=STUDY_ID,
                    block_id=BLOCK_ID,
                    cell_id=cell_id,
                    condition_index=condition_index,
                    layout_arm=layout_arm,
                    command=command,
                    sampling_seed=EFFECTIVE_SEED,
                    effective_seed=EFFECTIVE_SEED,
                    request_index=self.wmf_request_index,
                    action_step_start=self.wmf_request_index * ACTION_HORIZON,
                    server_context_id=self.wmf_server_context_id,
                )
                return request

            def _unpack_response(self, response: dict[str, Any]) -> np.ndarray:
                require(isinstance(response, Mapping), "n3_server_response_not_mapping")
                _validate_server_behavioral_response(
                    response,
                    cell_id=cell_id,
                    condition_index=condition_index,
                    request_index=self.wmf_request_index,
                )
                require(
                    response.get("wmf_server_context_id") == self.wmf_server_context_id,
                    "n3_server_context_changed",
                )
                video = np.asarray(response.get("video"))
                require(
                    video.ndim == 4 and video.shape[0] == ACTION_HORIZON + 1
                    and video.shape[-1] == 3 and video.dtype == np.uint8,
                    "client_decoded_future_shape_changed",
                )
                action = np.asarray(super()._unpack_response(response))
                require(action.shape == (ACTION_HORIZON, ACTION_DIM), "client_action_shape_changed")
                require(np.isfinite(action).all(), "client_action_nonfinite")
                self.wmf_request_index += 1
                return action

            def reset(self, *, env_id: int | None = None) -> None:
                try:
                    if env_id is None and self.wmf_episode_active:
                        final = recorder._final_receipt
                        status = "technical_failure" if final is None else final["stop_reason"]
                        request = {
                            "wmf_control": "end_episode",
                            "study_id": STUDY_ID,
                            "block_id": BLOCK_ID,
                            "cell_id": cell_id,
                            "condition_index": condition_index,
                            "server_context_id": self.wmf_server_context_id,
                            "status": "completed" if status == "action_cap" else status,
                            "actions_executed": recorder.actions_executed,
                            "request_count": len(recorder.requests),
                            "final_chunk_executed_actions": (
                                final.get("final_chunk", {}).get("executed_actions")
                                if isinstance(final, Mapping) and isinstance(final.get("final_chunk"), Mapping)
                                else None
                            ),
                        }
                        try:
                            response = self.client.infer(request)
                            require(isinstance(response, Mapping), "server_end_response_not_mapping")
                            self.wmf_end_receipt = dict(response)
                            if status == "action_cap":
                                require(response.get("passed") is True, "server_rejected_completed_cell")
                        except BaseException as error:
                            self.wmf_end_error = f"{type(error).__name__}: {error}"
                            raise
                        finally:
                            self.wmf_episode_active = False
                finally:
                    super().reset(env_id=env_id)

        class RecordedN3Client(RecordingClientMixin, BoundN3Client):
            pass

        client = RecordedN3Client(remote_host=args.remote_host, remote_port=args.remote_port)
        client.attach_forecast_recorder(recorder)
        client.reset_for_recorded_episode(client.begin_episode)

        env, env_cfg = create_env(
            task_name,
            device="cuda:0",
            seed=EFFECTIVE_SEED,
            num_envs=1,
            instruction_type="default",
            policy="wmf_n3_behavioral_pilot",
            renderer="realtime",
            rendering_mode="balanced",
        )
        require(not hasattr(env_cfg.terminations, "success"), "constructed_task_retains_success_termination")
        assert_fixed_duration_environment(env, env_cfg)
        require(env_cfg.instruction == PROMPTS[command], "constructed_task_prompt_changed")
        import torch

        require(torch.cuda.device_count() == 1, "simulator_child_must_see_one_logical_gpu")
        require(torch.cuda.get_device_name(0) == "NVIDIA B200", "simulator_child_gpu_is_not_b200")
        source_contract, source_payload = recorder_job.load_json(forecast / "layout_source_contract.json")
        require(
            recorder_job.sha256_bytes(source_payload) == recorder_job.SOURCE_CONTRACT_SHA256,
            "layout_source_contract_changed",
        )
        collision_sampler, visibility_sampler, physical_evidence = recorder_job._fresh_physical_callbacks(
            cell_root, source_contract
        )
        configured = release["pose_row"]["layouts"][layout_arm]
        tolerance = float(source_contract["live_gate"]["pose_tolerance_m"])

        def reset_attestor(live_env: Any, observation: Mapping[str, Any], info: Any):
            settled_observation, settled_info, attestation = settle_for_recording_reset(
                live_env,
                observation,
                info,
                pose_manifest_sha256=args.pose_manifest_sha256,
                reset_identity=f"n3:{attempt_root.name}:{cell_id}",
                collision_sampler=collision_sampler,
                visibility_sampler=visibility_sampler,
                settle_steps=int(source_contract["live_gate"]["settle_steps"]),
                stability_window_steps=int(source_contract["live_gate"]["stability_window_steps"]),
                linear_speed_tolerance_m_s=float(source_contract["live_gate"]["linear_speed_tolerance_m_s"]),
                angular_speed_tolerance_rad_s=float(source_contract["live_gate"]["angular_speed_tolerance_rad_s"]),
            )
            row = dict(attestation)
            row["fresh_pose_check"] = recorder_job._pose_check(live_env, configured, tolerance)
            row["accepted_gate_record_sha256"] = release["accepted_gate_record_sha256"]
            row["candidate_id"] = release["candidate_id"]
            return settled_observation, settled_info, row

        timing_path = cell_root / "recording" / "timing_support.json"
        native_clock = recorder_job.NativeClockSampler(timing_path, env_cfg)
        proxy = FixedDurationEnvProxy(
            env,
            env_cfg,
            recorder,
            state_sampler=recorder_job.sample_simulator_state,
            clock_sampler=native_clock,
            success_sampler=success_measurements,
            reset_attestor=reset_attestor,
            env_id=0,
        )
        env_results, subtask_status, timing = run_episode(
            proxy,
            env_cfg,
            0,
            client,
            headless=True,
            save_videos=True,
            video_mode="viewport",
        )
        require(client.wmf_end_error is None, "client_server_end_failed", client.wmf_end_error)
        require(isinstance(client.wmf_end_receipt, Mapping), "server_end_receipt_missing")
        completion = recorder._final_receipt
        require(isinstance(completion, Mapping), "adapter_completion_missing")
        require(completion.get("behavioral_result_valid") is True, "adapter_rejected_behavioral_cell")
        require(completion.get("actions_executed") == ACTION_CAP, "behavioral_action_count_changed")
        require(completion.get("observation_count") == OBSERVATION_COUNT, "behavioral_observation_count_changed")
        require(completion.get("request_count") == REQUEST_COUNT, "behavioral_request_count_changed")
        require(completion.get("final_two_action_truncation_recorded") is True, "final_two_action_truncation_missing")
        timing_support = load_json(timing_path, "cell_timing_support_unreadable")
        require(timing_support.get("supported") is True, "cell_native_timing_unsupported")
        journal = verify_journal(recorder.journal_path)
        require(journal["event_count"] == completion["event_count"], "cell_journal_event_count_mismatch")
        require(journal["tail_sha256"] == completion["journal_tail_sha256"], "cell_journal_tail_mismatch")
        viewport = [path for path in simulator_output.rglob("*.mp4") if path.is_file() and path.stat().st_size > 0]
        require(len(viewport) == 1, "cell_viewport_video_missing_or_duplicate")
        receipt = {
            "schema_version": CELL_RECEIPT_SCHEMA,
            "status": "passed",
            "study_id": STUDY_ID,
            "block_id": BLOCK_ID,
            "cell_id": cell_id,
            "condition_index": condition_index,
            "layout_pair_id": LAYOUT_PAIR_ID,
            "layout_arm": layout_arm,
            "command": command,
            "prompt": PROMPTS[command],
            "model_config": MODEL_CONFIG,
            "effective_seed": EFFECTIVE_SEED,
            "actions_executed": ACTION_CAP,
            "observation_count": OBSERVATION_COUNT,
            "behavioral_model_request_count": REQUEST_COUNT,
            "behavioral_episode_count": 1,
            "generation_qualification_request_count": 0,
            "final_chunk_executed_actions": FINAL_EXECUTED_ACTIONS,
            "adapter_completion": file_identity(recorder.completion_path),
            "adapter_journal": {**file_identity(recorder.journal_path), **journal},
            "native_timing_support": file_identity(timing_path),
            "server_end_receipt": dict(client.wmf_end_receipt),
            "viewport_video": file_identity(viewport[0]),
            "fresh_physical_checks": physical_evidence,
            "runner_timing": timing,
            "runner_environment_result_repr": repr(env_results),
            "runner_subtask_sample_count": len(subtask_status),
            "candidate_id": release["candidate_id"],
            "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
            "runtime_identity": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "python": sys.version,
                "python_executable": sys.executable,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "robolab_module": file_identity(Path(robolab.__file__)),
                "task_file": file_identity(task_path),
            },
            "completed_at_utc": utc_now(),
            "claim_boundary": (
                "One valid learned-policy behavioral pilot cell: 450 actual actions, "
                "451 original observations, and 15 jointly generated action/future requests."
            ),
        }
        immutable_json(receipt_path, receipt)
        print(json.dumps({"status": "passed", "cell_id": cell_id, "receipt": str(receipt_path)}), flush=True)
        return 0
    except BaseException as error:
        if recorder is not None and not recorder.finalized:
            recorder.mark_technical_failure("n3_behavioral_cell_runtime", error)
        if client is not None and getattr(client, "wmf_episode_active", False):
            try:
                client.reset()
            except BaseException:
                pass
        if not failure_path.exists():
            stop_reason = (
                recorder._final_receipt.get("stop_reason")
                if recorder is not None and isinstance(recorder._final_receipt, Mapping)
                else "technical_failure"
            )
            immutable_json(
                failure_path,
                {
                    "schema_version": CELL_RECEIPT_SCHEMA,
                    "status": "safety_abort" if stop_reason == "safety_abort" else "technical_failure",
                    "cell_id": cell_id,
                    "condition_index": condition_index,
                    "actions_executed": getattr(recorder, "actions_executed", 0),
                    "request_count": len(getattr(recorder, "requests", [])),
                    "behavioral_episode_count": 0,
                    "error_type": type(error).__name__,
                    "reason": getattr(error, "reason", None),
                    "traceback": traceback.format_exc(),
                    "recorded_at_utc": utc_now(),
                },
            )
        raise
    finally:
        if client is not None:
            try:
                client.close()
            except BaseException:
                pass
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        if simulation_app is not None:
            simulation_app.close()


def _launch_logged(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str], stdout_path: Path, stderr_path: Path
) -> tuple[subprocess.Popen[bytes], Any, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("xb")
    stderr = stderr_path.open("xb")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(Path(cwd).resolve()),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    _ACTIVE_CHILDREN.append(process)
    return process, stdout, stderr


def _close_process_logs(stdout: Any, stderr: Any) -> None:
    for stream in (stdout, stderr):
        try:
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()


def _wait_for_server(
    process: subprocess.Popen[bytes], ready_path: Path, *, port: int, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_connect_error: str | None = None
    while time.monotonic() < deadline:
        code = process.poll()
        require(code is None, "n3_server_exited_before_ready", str(code))
        if ready_path.is_file():
            ready = load_json(ready_path, "n3_server_ready_unreadable")
            require(ready.get("schema_version") == SERVER_READY_SCHEMA, "n3_server_ready_schema_changed")
            require(ready.get("status") == "ready", "n3_server_not_ready")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return ready
            except OSError as error:
                last_connect_error = type(error).__name__
        time.sleep(1)
    raise N3BehavioralPilotError("n3_server_ready_timeout", last_connect_error)


def _wait_for_gpu_cleanup(timeout: float = 60.0) -> list[dict[str, str]]:
    deadline = time.monotonic() + timeout
    rows: list[dict[str, str]] = []
    while time.monotonic() < deadline:
        rows = _query_nvidia(("gpu_uuid", "pid", "process_name", "used_memory"), compute=True)
        if not rows:
            return []
        time.sleep(1)
    return rows


def _receipt_counts(
    *, planned: int, launched: int, completed: Sequence[Mapping[str, Any]], attempt_root: Path
) -> dict[str, Any]:
    completed_ids = {str(row.get("cell_id")) for row in completed}
    invalid = 0
    censored = 0
    actions = sum(int(row.get("actions_executed", 0)) for row in completed)
    requests = sum(int(row.get("behavioral_model_request_count", 0)) for row in completed)
    for index in range(launched):
        cell_id = CELL_IDS[index]
        if cell_id in completed_ids:
            continue
        failure = (
            Path(attempt_root) / "cells" / f"{index:02d}-{safe_cell_component(cell_id)}"
            / "technical_failure.json"
        )
        if failure.is_file():
            value = load_json(failure)
            actions += int(value.get("actions_executed", 0))
            requests += int(value.get("request_count", 0))
            if value.get("status") == "safety_abort":
                censored += 1
            else:
                invalid += 1
        else:
            invalid += 1
    return {
        "planned_behavioral_cells": planned,
        "launched_behavioral_cells": launched,
        "completed_valid_behavioral_cells": len(completed),
        "technically_invalid_behavioral_cells": invalid,
        "right_censored_behavioral_cells": censored,
        "unrun_behavioral_cells": planned - launched,
        "actual_behavioral_actions": actions,
        "actual_behavioral_model_requests": requests,
        "new_generation_qualification_requests": 0,
        "reused_prerequisite_generation_qualification_requests": 6,
        "recorder_only_episodes_counted_as_behavioral": 0,
    }


def run_queue(args: argparse.Namespace) -> int:
    """Validate and supervise the entire indivisible P00/N3 block."""

    source_root = Path(args.source_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    raw_root = Path(args.raw_root).resolve()
    publish_dir = job_dir / "publish"
    publish_path = publish_dir / "n3_behavioral_pilot_receipt.json"
    attempt_root = raw_root / args.job_id
    completed: list[dict[str, Any]] = []
    launched = 0
    server_process: subprocess.Popen[bytes] | None = None
    server_stdout = None
    server_stderr = None
    queue_identity: dict[str, Any] | None = None
    prerequisites: dict[str, Any] | None = None
    schedule: dict[str, Any] | None = None
    topology: dict[str, Any] | None = None
    server_ready: dict[str, Any] | None = None
    failure: BaseException | None = None
    server_exit: dict[str, Any] | None = None

    require(raw_root == RAW_ROOT.resolve(), "n3_behavioral_raw_root_changed")
    raw_root.mkdir(parents=True, exist_ok=True)
    lock_path = raw_root / ".locks" / f"{BLOCK_ID}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise N3BehavioralPilotError("n3_p00_block_lock_is_held") from error
        try:
            prior = []
            for path in raw_root.glob("*/publish/n3_behavioral_pilot_receipt.json"):
                try:
                    value = load_json(path)
                except N3BehavioralPilotError:
                    continue
                if value.get("status") == "passed" and value.get("block_id") == BLOCK_ID:
                    prior.append(str(path.resolve()))
            require(not prior, "valid_n3_p00_block_already_exists", prior[0] if prior else None)
            require(not attempt_root.exists(), "n3_attempt_directory_already_exists")
            attempt_root.mkdir(parents=True)
            (attempt_root / "publish").mkdir()
            with installed_signal_handlers():
                try:
                    queue_identity = validate_queue_invocation(
                        source_root=source_root,
                        job_dir=job_dir,
                        study_commit=args.study_commit,
                        job_id=args.job_id,
                    )
                    schedule = validate_schedule(source_root)
                    prerequisites = validate_prerequisites(
                        source_root=source_root,
                        gate_receipt_path=Path(args.gate_receipt),
                        gate_receipt_sha256=args.gate_receipt_sha256,
                        pose_manifest_path=Path(args.pose_manifest),
                        pose_manifest_sha256=args.pose_manifest_sha256,
                        capture_receipt_path=Path(args.capture_receipt),
                        capture_receipt_sha256=args.capture_receipt_sha256,
                        recorder_receipt_path=Path(args.recorder_receipt),
                        recorder_receipt_sha256=args.recorder_receipt_sha256,
                        n3_qualification_receipt_path=Path(args.n3_qualification_receipt),
                        n3_qualification_receipt_sha256=args.n3_qualification_receipt_sha256,
                    )
                    topology = verify_two_idle_b200s()
                    immutable_json(attempt_root / "topology.json", topology)

                    server_command = build_server_command(
                        source_root=source_root,
                        attempt_root=attempt_root,
                        port=args.port,
                        study_commit=args.study_commit,
                    )
                    server_environment = build_model_environment(
                        source_root=source_root,
                        attempt_root=attempt_root,
                    )
                    server_process, server_stdout, server_stderr = _launch_logged(
                        server_command,
                        cwd=COSMOS_ROOT,
                        environment=server_environment,
                        stdout_path=attempt_root / "server" / "stdout.log",
                        stderr_path=attempt_root / "server" / "stderr.log",
                    )
                    immutable_json(
                        attempt_root / "server" / "process.json",
                        {
                            "schema_version": "wmf-n3-behavioral-server-process-v1",
                            "command": server_command,
                            "environment_contract": {
                                key: server_environment.get(key)
                                for key in (
                                    "CUDA_VISIBLE_DEVICES", "DS_IGNORE_CUDA_DETECTION", "HF_HOME",
                                    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "TORCHINDUCTOR_CACHE_DIR",
                                    "TMPDIR",
                                )
                            },
                            "process": _proc_identity(server_process),
                            "started_at_utc": utc_now(),
                        },
                    )
                    server_ready = _wait_for_server(
                        server_process,
                        attempt_root / "server" / "ready.json",
                        port=args.port,
                        timeout=args.server_ready_timeout,
                    )

                    for condition_index in range(len(CONDITIONS)):
                        require(server_process.poll() is None, "n3_server_exited_between_cells")
                        cell_command = build_cell_command(
                            source_root=source_root,
                            attempt_root=attempt_root,
                            study_commit=args.study_commit,
                            gate_receipt=Path(args.gate_receipt),
                            gate_receipt_sha256=args.gate_receipt_sha256,
                            pose_manifest=Path(args.pose_manifest),
                            pose_manifest_sha256=args.pose_manifest_sha256,
                            port=args.port,
                            condition_index=condition_index,
                        )
                        cell_id = CELL_IDS[condition_index]
                        log_root = attempt_root / "cell_logs" / f"{condition_index:02d}-{safe_cell_component(cell_id)}"
                        simulator_environment = build_simulator_environment(
                            source_root=source_root,
                            state_parent=attempt_root,
                        )
                        child, child_stdout, child_stderr = _launch_logged(
                            cell_command,
                            cwd=ROBOLAB_ROOT,
                            environment=simulator_environment,
                            stdout_path=log_root / "stdout.log",
                            stderr_path=log_root / "stderr.log",
                        )
                        launched += 1
                        immutable_json(
                            log_root / "process.json",
                            {
                                "schema_version": "wmf-n3-behavioral-cell-process-v1",
                                "cell_id": cell_id,
                                "condition_index": condition_index,
                                "command": cell_command,
                                "cuda_visible_devices": simulator_environment.get("CUDA_VISIBLE_DEVICES"),
                                "process": _proc_identity(child),
                                "started_at_utc": utc_now(),
                            },
                        )
                        try:
                            try:
                                code = child.wait(timeout=args.cell_timeout)
                            except subprocess.TimeoutExpired:
                                terminate_process_group(child)
                                raise N3BehavioralPilotError("n3_behavioral_cell_timeout", cell_id)
                            require(code == 0, "n3_behavioral_cell_failed", f"{cell_id}:{code}")
                        finally:
                            if child.poll() is None:
                                terminate_process_group(child)
                            if child in _ACTIVE_CHILDREN:
                                _ACTIVE_CHILDREN.remove(child)
                            _close_process_logs(child_stdout, child_stderr)
                        receipt_path = (
                            attempt_root / "cells" / f"{condition_index:02d}-{safe_cell_component(cell_id)}"
                            / "cell_receipt.json"
                        )
                        receipt = load_json(receipt_path, "cell_receipt_unreadable")
                        require(receipt.get("schema_version") == CELL_RECEIPT_SCHEMA, "cell_receipt_schema_changed")
                        require(receipt.get("status") == "passed", "cell_receipt_not_passed")
                        require(receipt.get("cell_id") == cell_id, "cell_receipt_identity_mismatch")
                        require(receipt.get("actions_executed") == ACTION_CAP, "cell_receipt_actions_changed")
                        require(receipt.get("behavioral_model_request_count") == REQUEST_COUNT, "cell_receipt_requests_changed")
                        completed.append(receipt)

                    require(len(completed) == len(CELL_IDS), "n3_behavioral_block_incomplete")
                except BaseException as error:
                    failure = error
                finally:
                    if server_process is not None:
                        terminate_process_group(server_process)
                        if server_process in _ACTIVE_CHILDREN:
                            _ACTIVE_CHILDREN.remove(server_process)
                        server_exit = {
                            "schema_version": SERVER_EXIT_SCHEMA,
                            "returncode": server_process.returncode,
                            "child_reaped": server_process.poll() is not None,
                            "terminated_by_queue_supervisor": True,
                            "ended_at_utc": utc_now(),
                        }
                        if not (attempt_root / "server" / "supervisor_exit.json").exists():
                            immutable_json(attempt_root / "server" / "supervisor_exit.json", server_exit)
                    if server_stdout is not None and server_stderr is not None:
                        _close_process_logs(server_stdout, server_stderr)
                    remaining_processes = _wait_for_gpu_cleanup()
                    cleanup = {
                        "schema_version": "wmf-n3-behavioral-cleanup-v1",
                        "all_children_reaped": not _ACTIVE_CHILDREN,
                        "remaining_compute_processes": remaining_processes,
                        "completed_at_utc": utc_now(),
                    }
                    immutable_json(attempt_root / "cleanup.json", cleanup)
                    if remaining_processes and failure is None:
                        failure = N3BehavioralPilotError("gpu_process_remained_after_cleanup")

            counts = _receipt_counts(
                planned=len(CELL_IDS), launched=launched, completed=completed, attempt_root=attempt_root
            )
            status = "passed" if failure is None else "technical_failure"
            receipt = {
                "schema_version": QUEUE_RECEIPT_SCHEMA,
                "status": status,
                "exit_code": 0 if failure is None else 1,
                "study_id": STUDY_ID,
                "namespace": NAMESPACE,
                "block_id": BLOCK_ID,
                "phase": PHASE,
                "layout_pair_id": LAYOUT_PAIR_ID,
                "model_config": MODEL_CONFIG,
                "effective_seed": EFFECTIVE_SEED,
                "condition_order": [f"{arm}-{command}" for arm, command, _task in CONDITIONS],
                "cell_ids": list(CELL_IDS),
                "counts": counts,
                "source_commit": args.study_commit,
                "queue_descriptor": queue_identity,
                "schedule": None if schedule is None else {key: schedule[key] for key in ("path", "sha256")},
                "prerequisites": prerequisites,
                "topology": None if topology is None else file_identity(attempt_root / "topology.json"),
                "server_ready": (
                    file_identity(attempt_root / "server" / "ready.json") if server_ready is not None else None
                ),
                "server_exit": server_exit,
                "cell_receipts": [
                    file_identity(
                        attempt_root / "cells" / f"{index:02d}-{safe_cell_component(CELL_IDS[index])}"
                        / "cell_receipt.json"
                    )
                    for index in range(len(completed))
                ],
                "raw_attempt_root": str(attempt_root),
                "raw_attempt_recoverable_on_gm_pvc": True,
                "failure": (
                    None
                    if failure is None
                    else {
                        "error_type": type(failure).__name__,
                        "reason": getattr(failure, "reason", None),
                        "detail": str(failure),
                    }
                ),
                "completed_at_utc": utc_now(),
                "claim_boundary": (
                    "This receipt counts only valid 450-action N3 behavioral cells. The reused six-request "
                    "generation qualification and recorder-only qualification are explicitly nonbehavioral."
                ),
            }
            immutable_json(attempt_root / "publish" / "n3_behavioral_pilot_receipt.json", receipt, publish=True)
            publish_dir.mkdir(parents=True, exist_ok=True)
            immutable_json(publish_path, receipt, publish=True)
            print(json.dumps({"status": status, "counts": counts, "raw_attempt_root": str(attempt_root)}), flush=True)
            return 0 if failure is None else 1
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    queue = subparsers.add_parser("queue", help="supervise the exact four-cell block")
    queue.add_argument("--source-root", type=Path, required=True)
    queue.add_argument("--study-commit", required=True)
    queue.add_argument("--job-dir", type=Path, required=True)
    queue.add_argument("--job-id", required=True)
    queue.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    queue.add_argument("--gate-receipt", type=Path, required=True)
    queue.add_argument("--gate-receipt-sha256", required=True)
    queue.add_argument("--pose-manifest", type=Path, required=True)
    queue.add_argument("--pose-manifest-sha256", required=True)
    queue.add_argument("--capture-receipt", type=Path, required=True)
    queue.add_argument("--capture-receipt-sha256", required=True)
    queue.add_argument("--recorder-receipt", type=Path, required=True)
    queue.add_argument("--recorder-receipt-sha256", required=True)
    queue.add_argument("--n3-qualification-receipt", type=Path, required=True)
    queue.add_argument("--n3-qualification-receipt-sha256", required=True)
    queue.add_argument("--port", type=int, default=DEFAULT_PORT)
    queue.add_argument("--server-ready-timeout", type=float, default=7200.0)
    queue.add_argument("--cell-timeout", type=float, default=10800.0)

    server = subparsers.add_parser("server", help="run the official instrumented N3 server")
    server.add_argument("--source-root", type=Path, required=True)
    server.add_argument("--study-commit", required=True)
    server.add_argument("--attempt-root", type=Path, required=True)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=DEFAULT_PORT)

    cell = subparsers.add_parser("cell", help="run one fresh 450-action simulator cell")
    cell.add_argument("--source-root", type=Path, required=True)
    cell.add_argument("--study-commit", required=True)
    cell.add_argument("--attempt-root", type=Path, required=True)
    cell.add_argument("--gate-receipt", type=Path, required=True)
    cell.add_argument("--gate-receipt-sha256", required=True)
    cell.add_argument("--pose-manifest", type=Path, required=True)
    cell.add_argument("--pose-manifest-sha256", required=True)
    cell.add_argument("--layout-arm", choices=("original", "reflected"), required=True)
    cell.add_argument("--command", choices=("left", "right"), required=True)
    cell.add_argument("--condition-index", type=int, required=True)
    cell.add_argument("--remote-host", default="127.0.0.1")
    cell.add_argument("--remote-port", type=int, default=DEFAULT_PORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(1 <= getattr(args, "port", getattr(args, "remote_port", 0)) <= 65535, "invalid_port")
    if args.mode == "queue":
        require(args.server_ready_timeout > 0 and args.cell_timeout > 0, "invalid_timeout")
        return run_queue(args)
    if args.mode == "server":
        return run_server(args)
    if args.mode == "cell":
        return run_cell(args)
    raise N3BehavioralPilotError("unknown_mode")


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
                    "reason": getattr(error, "reason", None),
                    "detail": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
