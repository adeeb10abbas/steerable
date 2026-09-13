#!/usr/bin/env python3
"""Two-job launcher for the exact four-cell P00 DreamZero D1 pilot.

The D1 model is a non-interleavable, two-rank global-state service.  It runs in
the dedicated ``d1`` queue worker while a distinct ``worker-00`` queue job runs
four fresh RoboLab simulator children.  The jobs authenticate one another with
immutable receipts and renewable leases on the shared PVC.  A service reset is
accepted only when the resulting two-rank reset receipt appears below the
authenticated server attempt, which also proves that Kubernetes service port
18021 reached the intended model process.

Large observations, exact wire/model inputs, all 24x8 returned actions,
retained latents, offline decodes, executed actions, and timing evidence remain
on the GM PVC.  Queue publish directories receive bounded receipts only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
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
MODEL_CONFIG = "D1"
PHASE = "pilot"
LAYOUT_PAIR_ID = "P00"
BLOCK_ID = "wmf_ablation_001_20260912__pilot__P00__D1"
SERVER_QUEUE_ROLE = "d1"
SIMULATOR_QUEUE_ROLE = "wmf-forecast-0912-worker-00"
SERVICE_HOST = "wmf-forecast-0912-d1"
SERVICE_PORT = 18021

EFFECTIVE_MODEL_NOISE_SEED = 1140
ENVIRONMENT_SEED = 2026091000
ACTION_CAP = 450
OBSERVATION_COUNT = 451
RETURNED_ACTION_HORIZON = 24
EXECUTED_PREFIX_HORIZON = 8
ACTION_DIM = 8
REQUEST_COUNT = math.ceil(ACTION_CAP / EXECUTED_PREFIX_HORIZON)
FINAL_EXECUTED_ACTIONS = ACTION_CAP % EXECUTED_PREFIX_HORIZON

CONDITIONS = (
    ("reflected", "right", "WMFForecastReflectedRightTask"),
    ("reflected", "left", "WMFForecastReflectedLeftTask"),
    ("original", "left", "WMFForecastOriginalLeftTask"),
    ("original", "right", "WMFForecastOriginalRightTask"),
)
CELL_IDS = tuple(
    f"wmf1__pilot__P00__D1__{layout_arm}__{command}"
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
D1_SOURCE_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
D1_SOURCE_TREE = "6b7ba27f1af81e963a6507f1204c05c65a94098c"
D1_SOURCE_AGGREGATE_SHA256 = "a39e1ef8b7d8668caf15914637446fe58acb512e6672413d3af8b4b6442eca7f"
CHECKPOINT_REVISION = "96ad344138c66e82536422432ad742f015784942"
CHECKPOINT_AGGREGATE_SHA256 = "b4af0ac93474c3295c1ba841a34a8f2f91a5c3ec3c6aac1431b97689a6618c56"
TOKENIZER_REVISION = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
TOKENIZER_AGGREGATE_SHA256 = "00f9b974f8f0b33c5e284849a0507c308893d01d915377ba88c2f7084ea92434"
D1_QUALIFICATION_RECEIPT_SHA256 = "3c856549999b9145dc07c30853a4c6d2968d09eb31c2db883f5eb1655d31627b"

ROBOLAB_ROOT = Path("/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241")
ROBOLAB_PYTHON = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
RAW_ROOT = Path("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/pilot/D1/P00")

MAX_PUBLISH_BYTES = 512 * 1024
LEASE_TTL_SECONDS = 90.0
LEASE_POLL_SECONDS = 2.0
D1_CONNECT_TIMEOUT_SECONDS = 300.0
D1_CONNECT_RETRIES = 5
D1_CONNECT_BACKOFF_BASE_SECONDS = 2.0
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

SERVER_RECEIPT_SCHEMA = "wmf-d1-behavioral-pilot-server-job-v1"
SIMULATOR_RECEIPT_SCHEMA = "wmf-d1-behavioral-pilot-simulator-job-v1"
CELL_RECEIPT_SCHEMA = "wmf-d1-behavioral-pilot-cell-v1"
SERVER_READY_SCHEMA = "wmf-d1-behavioral-server-ready-v1"
SERVER_LEASE_SCHEMA = "wmf-d1-behavioral-server-lease-v1"
SIMULATOR_CLAIM_SCHEMA = "wmf-d1-behavioral-simulator-claim-v1"
SIMULATOR_LEASE_SCHEMA = "wmf-d1-behavioral-simulator-lease-v1"
SIMULATOR_TERMINAL_SCHEMA = "wmf-d1-behavioral-simulator-terminal-v1"
D1_QUALIFICATION_SCHEMA = "wmf-d1-qualification-job-receipt-v1"
D1_SERVER_CONTRACT_SCHEMA = "wmf-d1-instrumented-server-v1"
D1_RESET_SCHEMA = "wmf-d1-two-rank-reset-receipt-v1"
D1_REQUEST_SCHEMA = "wmf-d1-request-receipt-v1"
D1_EPISODE_SCHEMA = "wmf-d1-episode-manifest-v1"
CAPTURE_RECEIPT_SCHEMA = "wmf-forecast-layout-fixed-observation-capture-v1"
RECORDER_RECEIPT_SCHEMA = "wmf-forecast-recorder-qualification-job-v1"

RESET_FIELDS_TO_NONE = (
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
    "clip_feas",
    "ys",
    "language",
)
CACHE_FIELDS = (
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
)
RAW_ARRAY_KEYS = (
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/cartesian_position",
    "observation/gripper_position",
)

PILOT_CONTRACT = {
    "schema_version": "wmf-d1-behavioral-pilot-contract-v1",
    "study_id": STUDY_ID,
    "block_id": BLOCK_ID,
    "configuration_id": MODEL_CONFIG,
    "condition_order": [f"{arm}-{command}" for arm, command, _task in CONDITIONS],
    "cell_ids": list(CELL_IDS),
    "per_cell": {
        "actions": ACTION_CAP,
        "observations": OBSERVATION_COUNT,
        "requests": REQUEST_COUNT,
        "returned_action_shape": [RETURNED_ACTION_HORIZON, ACTION_DIM],
        "executed_prefix": EXECUTED_PREFIX_HORIZON,
        "final_request_executed_actions": FINAL_EXECUTED_ACTIONS,
    },
    "official_path": "GrootSimPolicy.lazy_joint_forward_causal",
    "source_commit": D1_SOURCE_COMMIT,
    "source_tree": D1_SOURCE_TREE,
    "checkpoint_revision": CHECKPOINT_REVISION,
    "effective_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
    "noise_semantics": "fixed; cells and requests are not independent noise draws",
    "video_guidance_scale": 5.0,
    "configured_inference_steps": 16,
    "custom_s2_used": False,
    "patched_s1_used": False,
    "server_world_size": 2,
    "service": {"host": SERVICE_HOST, "port": SERVICE_PORT},
    "transport": {
        "compression": None,
        "max_size": None,
        "ping_interval": None,
        "ping_timeout": None,
        "stateful_request_replay": False,
    },
}


class D1BehavioralPilotError(RuntimeError):
    """Fail-closed launcher error with a bounded stable reason code."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        if SAFE_ID_RE.fullmatch(reason) is None:
            reason = "internal_contract_error"
        super().__init__(reason if detail is None else f"{reason}: {detail}")
        self.reason = reason


def require(condition: bool, reason: str, detail: str | None = None) -> None:
    if not condition:
        raise D1BehavioralPilotError(reason, detail)


def connect_websocket_without_keepalive(
    connect: Any, *, uri: str, auth_headers: Mapping[str, str]
) -> Any:
    """Open one D1 socket without protocol pings during synchronous inference.

    DreamZero can hold the server event loop while generating. A protocol
    keepalive timeout is therefore not evidence that the stateful request was
    unconsumed. Disable both ping fields explicitly; queue and cell wall-time
    bounds remain the liveness mechanism.
    """

    return connect(
        uri,
        additional_headers=dict(auth_headers),
        compression=None,
        max_size=None,
        open_timeout=D1_CONNECT_TIMEOUT_SECONDS,
        ping_interval=None,
        ping_timeout=None,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D1BehavioralPilotError("noncanonical_json_value") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PILOT_CONTRACT_SHA256 = sha256_bytes(canonical_bytes(PILOT_CONTRACT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    require(resolved.is_file(), "evidence_file_missing", str(resolved))
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


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
    require(not path.parent.is_symlink(), "evidence_parent_is_symlink", str(path.parent))
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
            raise D1BehavioralPilotError("immutable_evidence_exists", str(path)) from error
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def replace_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace one explicitly mutable lease file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.parent.is_symlink() and not path.is_symlink(), "lease_path_is_symlink")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, reason: str = "json_evidence_unreadable") -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D1BehavioralPilotError(reason, str(path)) from error
    require(isinstance(value, dict), reason, "top-level JSON is not an object")
    return value


def verify_exact_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(SHA256_RE.fullmatch(expected_sha256) is not None, "invalid_sha256", label)
    supplied = Path(path)
    require(supplied.is_absolute(), "evidence_path_not_absolute", label)
    require(not supplied.is_symlink(), "evidence_path_is_symlink", label)
    observed = file_identity(supplied.resolve())
    require(observed["sha256"] == expected_sha256, "evidence_sha256_mismatch", label)
    return observed


def _verify_descriptor(value: Any, label: str, *, allowed_root: Path | None = None) -> dict[str, Any]:
    require(isinstance(value, Mapping), "file_descriptor_missing", label)
    raw_path = value.get("path")
    digest = value.get("sha256", value.get("file_sha256"))
    require(isinstance(raw_path, str) and Path(raw_path).is_absolute(), "file_descriptor_path_invalid", label)
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "file_descriptor_sha_invalid", label)
    lexical = Path(os.path.abspath(raw_path))
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        require(lexical.is_relative_to(root), "artifact_escaped_allowed_root", label)
        cursor = lexical
        while cursor != root:
            require(not cursor.is_symlink(), "artifact_path_contains_symlink", label)
            cursor = cursor.parent
    observed = file_identity(lexical)
    require(observed["sha256"] == digest, "file_descriptor_sha_mismatch", label)
    if "bytes" in value:
        require(observed["bytes"] == value["bytes"], "file_descriptor_size_mismatch", label)
    return observed


def _run_git(root: Path, *argv: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *argv], capture_output=True, text=True, timeout=60,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise D1BehavioralPilotError("git_verification_unavailable") from error
    require(result.returncode == 0, "git_verification_failed", argv[0] if argv else "git")
    return result.stdout


def verify_clean_git(root: Path, commit: str, label: str) -> None:
    require(COMMIT_RE.fullmatch(commit) is not None, "invalid_commit", label)
    root = Path(root).resolve()
    require(root.is_dir(), "source_checkout_missing", label)
    require(_run_git(root, "rev-parse", "HEAD").strip() == commit, "source_commit_mismatch", label)
    require(not _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"), "source_checkout_dirty", label)


def safe_component(value: str) -> str:
    require(SAFE_ID_RE.fullmatch(value) is not None, "unsafe_identifier", value)
    return value.replace("__", "-").replace("_", "-")


def validate_schedule(source_root: Path) -> dict[str, Any]:
    path = Path(source_root) / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
    schedule = load_json(path, "parallel_schedule_unreadable")
    rows = [row for row in schedule.get("jobs", []) if row.get("job_id") == BLOCK_ID]
    require(len(rows) == 1, "d1_p00_schedule_row_missing_or_duplicate")
    row = rows[0]
    expected = {
        "phase": PHASE,
        "layout_pair_id": LAYOUT_PAIR_ID,
        "model_config": MODEL_CONFIG,
        "candidate_effective_policy_seed": EFFECTIVE_MODEL_NOISE_SEED,
        "indivisible": True,
        "condition_order": [f"{arm}-{command}" for arm, command, _task in CONDITIONS],
        "ordered_cell_ids": list(CELL_IDS),
    }
    for key, wanted in expected.items():
        require(row.get(key) == wanted, "d1_p00_schedule_mismatch", key)
    contract = row.get("execution_contract", {})
    require(contract.get("sequential_conditions") is True, "d1_p00_not_sequential")
    require(contract.get("no_global_DreamZero_context_interleaving") is True, "d1_p00_interleaving_contract_changed")
    require(contract.get("full_model_and_simulator_reset_before_each_condition") is True, "d1_p00_reset_contract_changed")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "row": row}


def validate_queue_invocation(
    *, source_root: Path, job_dir: Path, study_commit: str, job_id: str, expected_role: str
) -> dict[str, Any]:
    require(SAFE_ID_RE.fullmatch(job_id) is not None, "invalid_job_id")
    require(COMMIT_RE.fullmatch(study_commit) is not None, "invalid_study_commit")
    source_root = Path(source_root).resolve()
    job_dir = Path(job_dir).resolve()
    verify_clean_git(source_root, study_commit, "study")
    require(job_dir.name == job_id, "queue_job_directory_mismatch")
    descriptor_path = job_dir / "descriptor.json"
    require(not descriptor_path.is_symlink(), "queue_descriptor_is_symlink")
    descriptor = load_json(descriptor_path, "queue_descriptor_unreadable")
    expected = {
        "schema_version": "wmf-cluster-job-v1",
        "namespace": NAMESPACE,
        "job_id": job_id,
        "source_commit": study_commit,
        "role": expected_role,
        "released": True,
    }
    for key, wanted in expected.items():
        require(descriptor.get(key) == wanted, "queue_descriptor_mismatch", key)
    return {**file_identity(descriptor_path), "role": expected_role, "job_id": job_id}


def validate_d1_qualification_payload(receipt: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": D1_QUALIFICATION_SCHEMA,
        "status": "finished",
        "decision": "qualified",
        "exit_code": 0,
        "reason": None,
        "configuration_id": MODEL_CONFIG,
        "generation_request_count": 6,
        "behavioral_episode_count": 0,
    }
    for key, wanted in expected.items():
        require(receipt.get(key) == wanted, "d1_qualification_not_passed", key)
    probe = receipt.get("probe")
    require(isinstance(probe, Mapping), "d1_qualification_probe_missing")
    require(probe.get("status") == "passed" and probe.get("passed") is True, "d1_qualification_probe_failed")
    require(probe.get("generation_request_count") == 6, "d1_qualification_probe_count_changed")
    require(probe.get("behavioral_episode_count") == 0, "d1_qualification_claims_behavioral_episode")
    require(probe.get("failed_checks") == [], "d1_qualification_probe_checks_failed")
    contract = receipt.get("server_contract")
    require(isinstance(contract, Mapping), "d1_qualification_server_contract_missing")
    exact = {
        "official_repository_commit": D1_SOURCE_COMMIT,
        "official_repository_tree": D1_SOURCE_TREE,
        "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
        "custom_s2_used": False,
        "patched_s1_used": False,
        "world_size": 2,
        "source_aggregate_sha256": D1_SOURCE_AGGREGATE_SHA256,
        "checkpoint_aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256,
        "tokenizer_aggregate_sha256": TOKENIZER_AGGREGATE_SHA256,
    }
    for key, wanted in exact.items():
        require(contract.get(key) == wanted, "d1_qualification_identity_changed", key)


def validate_d1_qualification(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(expected_sha256 == D1_QUALIFICATION_RECEIPT_SHA256, "d1_qualification_is_not_released_receipt")
    identity = verify_exact_file(path, expected_sha256, "d1_qualification_receipt")
    receipt = load_json(path, "d1_qualification_receipt_unreadable")
    validate_d1_qualification_payload(receipt)
    _verify_descriptor(receipt["probe"].get("artifact"), "d1_qualification_probe_artifact")
    _verify_descriptor(receipt["server_contract"].get("artifact"), "d1_qualification_contract_artifact")
    return identity


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
    d1_qualification_receipt_path: Path,
    d1_qualification_receipt_sha256: str,
) -> dict[str, Any]:
    forecast = Path(source_root) / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    original = recorder_job.verify_fixture_release(
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
    require(original["candidate_id"] == reflected["candidate_id"], "pose_arms_bind_different_candidates")

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
    require(capture.get("environment_seed") == ENVIRONMENT_SEED, "capture_environment_seed_mismatch")
    require(capture.get("candidate_id") == original["candidate_id"], "capture_candidate_mismatch")
    require(capture.get("model_request_count") == 0, "capture_contains_model_requests")
    require(capture.get("behavioral_action_count") == 0, "capture_contains_behavioral_actions")

    recorder_identity = verify_exact_file(recorder_receipt_path, recorder_receipt_sha256, "recorder_receipt")
    recorder = load_json(recorder_receipt_path, "recorder_receipt_unreadable")
    require(recorder.get("schema_version") == RECORDER_RECEIPT_SCHEMA, "recorder_receipt_schema_changed")
    require(recorder.get("status") == "passed" and recorder.get("exit_code") == 0, "recorder_not_qualified")
    require(recorder.get("layout_pair_id") == LAYOUT_PAIR_ID, "recorder_layout_mismatch")
    require(recorder.get("pose_manifest_sha256") == pose_manifest_sha256, "recorder_pose_mismatch")
    require(recorder.get("gate_receipt_sha256") == gate_receipt_sha256, "recorder_gate_mismatch")
    require(recorder.get("environment_seed") == ENVIRONMENT_SEED, "recorder_environment_seed_mismatch")
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

    d1_identity = validate_d1_qualification(
        d1_qualification_receipt_path, d1_qualification_receipt_sha256
    )
    return {
        "gate_receipt": original["gate_receipt"],
        "pose_manifest": original["pose_manifest"],
        "candidate_id": original["candidate_id"],
        "candidate_payload_sha256": original["candidate_payload_sha256"],
        "accepted_gate_record_sha256": original["accepted_gate_record_sha256"],
        "capture_receipt": capture_identity,
        "recorder_receipt": recorder_identity,
        "recorder_child_receipt": child_identity,
        "native_timing_support": timing_identity,
        "d1_qualification_receipt": d1_identity,
        "generation_qualification_requests_reused_not_rerun": 6,
    }


def _query_nvidia(
    columns: Sequence[str], *, compute: bool, executable: Path | str = "nvidia-smi"
) -> list[dict[str, str]]:
    prefix = "--query-compute-apps=" if compute else "--query-gpu="
    try:
        result = subprocess.run(
            [str(executable), prefix + ",".join(columns), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise D1BehavioralPilotError("nvidia_smi_unavailable") from error
    require(result.returncode == 0, "nvidia_smi_failed")
    rows: list[dict[str, str]] = []
    for values in csv.reader(io.StringIO(result.stdout)):
        if not values or not any(item.strip() for item in values):
            continue
        require(len(values) == len(columns), "nvidia_smi_output_shape_changed")
        rows.append({key: value.strip() for key, value in zip(columns, values)})
    return rows


def verify_one_idle_b200(executable: Path | str = "nvidia-smi") -> dict[str, Any]:
    devices = _query_nvidia(
        ("index", "uuid", "name", "driver_version", "memory.total"),
        compute=False, executable=executable,
    )
    require(len(devices) == 1, "d1_simulator_worker_does_not_expose_exactly_one_gpu")
    require(devices[0].get("name") == "NVIDIA B200", "d1_simulator_gpu_is_not_b200")
    require(str(devices[0].get("uuid", "")).startswith("GPU-"), "d1_simulator_gpu_uuid_invalid")
    processes = _query_nvidia(
        ("gpu_uuid", "pid", "process_name", "used_memory"), compute=True, executable=executable
    )
    require(not processes, "d1_simulator_worker_has_preexisting_compute_process")
    return {
        "schema_version": "wmf-d1-behavioral-simulator-one-b200-v1",
        "status": "passed",
        "device": devices[0],
        "preexisting_compute_process_count": 0,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "recorded_at_utc": utc_now(),
    }


def _proc_identity(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    try:
        pgid: int | None = os.getpgid(process.pid)
    except ProcessLookupError:
        pgid = None
    result: dict[str, Any] = {
        "pid": process.pid,
        "process_group_id": pgid,
        "hostname": socket.gethostname(),
        "pod_uid": os.environ.get("POD_UID"),
        "start_new_session": True,
        "returncode_at_capture": process.poll(),
    }
    try:
        fields = Path(f"/proc/{process.pid}/stat").read_text(encoding="utf-8").split()
        result["proc_start_ticks"] = int(fields[21])
        result["linux_boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        result["proc_start_ticks"] = None
        result["linux_boot_id"] = None
    return result


def terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    if process.poll() is None:
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
            process.wait(timeout=max(1.0, min(grace_seconds, 5.0)))
    else:
        process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


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
    global _SIGNAL_ACTIVE
    _SIGNAL_ACTIVE = False
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    for signum in previous:
        signal.signal(signum, _termination_handler)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _SIGNAL_ACTIVE = False


def _launch_logged(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str],
    stdout_path: Path, stderr_path: Path
) -> tuple[subprocess.Popen[bytes], Any, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("xb")
    stderr = stderr_path.open("xb")
    try:
        process = subprocess.Popen(
            list(command), cwd=str(Path(cwd).resolve()), env=dict(environment),
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True,
        )
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    _ACTIVE_CHILDREN.append(process)
    return process, stdout, stderr


def _close_logs(stdout: Any, stderr: Any) -> None:
    for stream in (stdout, stderr):
        try:
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()


@contextmanager
def supervised_logged_child(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str],
    stdout_path: Path, stderr_path: Path
):
    """Unconditionally reap the entire child group, including receipt failures."""

    process, stdout, stderr = _launch_logged(
        command, cwd=cwd, environment=environment,
        stdout_path=stdout_path, stderr_path=stderr_path,
    )
    try:
        yield process, stdout, stderr
    finally:
        try:
            terminate_process_group(process)
        finally:
            if process in _ACTIVE_CHILDREN:
                _ACTIVE_CHILDREN.remove(process)
            _close_logs(stdout, stderr)


class LeaseWriter:
    """Renew a PVC lease without converting it into scientific evidence."""

    def __init__(
        self, path: Path, *, schema: str, owner_job_id: str, run_id: str,
        server_ready_sha256: str, simulator_claim_sha256: str | None = None,
        lease_token: str | None = None, ttl_seconds: float = LEASE_TTL_SECONDS,
    ) -> None:
        require(ttl_seconds > 4 * LEASE_POLL_SECONDS, "lease_ttl_too_short")
        self.path = Path(path)
        self.schema = schema
        self.owner_job_id = owner_job_id
        self.run_id = run_id
        self.ready_sha = server_ready_sha256
        self.claim_sha = simulator_claim_sha256
        self.lease_token = lease_token
        self.ttl_seconds = ttl_seconds
        self.stop_event = threading.Event()
        self.failure: BaseException | None = None
        self.generation = 0
        self.thread = threading.Thread(target=self._run, name=f"lease-{owner_job_id}", daemon=True)

    def _payload(self) -> dict[str, Any]:
        now = time.time_ns()
        return {
            "schema_version": self.schema,
            "status": "live",
            "run_id": self.run_id,
            "owner_job_id": self.owner_job_id,
            "server_ready_sha256": self.ready_sha,
            "simulator_claim_sha256": self.claim_sha,
            "lease_token": self.lease_token,
            "generation": self.generation,
            "heartbeat_unix_ns": now,
            "expires_unix_ns": now + int(self.ttl_seconds * 1e9),
            "hostname": socket.gethostname(),
            "pod_uid": os.environ.get("POD_UID"),
        }

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.generation += 1
                replace_json(self.path, self._payload())
                self.stop_event.wait(min(self.ttl_seconds / 3.0, 15.0))
        except BaseException as error:
            self.failure = error
            self.stop_event.set()

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.path.is_file() and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.raise_if_failed()
        require(self.path.is_file(), "lease_initial_write_timeout")

    def raise_if_failed(self) -> None:
        if self.failure is not None:
            raise D1BehavioralPilotError("lease_writer_failed", str(self.failure))

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=20)
        require(not self.thread.is_alive(), "lease_writer_did_not_stop")
        self.raise_if_failed()


def validate_live_lease(
    path: Path, *, schema: str, owner_job_id: str, run_id: str,
    server_ready_sha256: str, simulator_claim_sha256: str | None = None,
    lease_token: str | None = None, now_ns: int | None = None,
) -> dict[str, Any]:
    lease = load_json(path, "lease_unreadable")
    expected = {
        "schema_version": schema,
        "status": "live",
        "run_id": run_id,
        "owner_job_id": owner_job_id,
        "server_ready_sha256": server_ready_sha256,
        "simulator_claim_sha256": simulator_claim_sha256,
        "lease_token": lease_token,
    }
    for key, wanted in expected.items():
        require(lease.get(key) == wanted, "lease_identity_mismatch", key)
    require(type(lease.get("generation")) is int and lease["generation"] > 0, "lease_generation_invalid")
    require(type(lease.get("heartbeat_unix_ns")) is int, "lease_heartbeat_invalid")
    require(type(lease.get("expires_unix_ns")) is int, "lease_expiry_invalid")
    require(lease["expires_unix_ns"] > (time.time_ns() if now_ns is None else now_ns), "lease_expired")
    return lease


def coordination_paths(raw_root: Path, run_id: str) -> dict[str, Path]:
    require(SAFE_ID_RE.fullmatch(run_id) is not None, "invalid_run_id")
    root = Path(raw_root).resolve() / "coordination" / run_id
    return {
        "root": root,
        "server_ready": root / "server_ready.json",
        "server_lease": root / "server_lease.json",
        "simulator_claim": root / "simulator_claim.json",
        "simulator_lease": root / "simulator_lease.json",
        "simulator_terminal": root / "simulator_terminal.json",
    }


def validate_live_server_contract_payload(
    contract: Mapping[str, Any], *, source_root: Path, future_root: Path
) -> None:
    expected = {
        "schema_version": D1_SERVER_CONTRACT_SCHEMA,
        "status": "passed",
        "configuration_id": MODEL_CONFIG,
        "official_repository_commit": D1_SOURCE_COMMIT,
        "official_repository_tree": D1_SOURCE_TREE,
        "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
        "custom_s2_used": False,
        "patched_s1_used": False,
        "world_size": 2,
        "port": SERVICE_PORT,
        "returned_action_shape": [RETURNED_ACTION_HORIZON, ACTION_DIM],
        "executed_action_prefix": EXECUTED_PREFIX_HORIZON,
        "effective_official_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
        "video_guidance_scale": 5.0,
        "configured_inference_steps": 16,
        "evaluated_dit_step_count": 8,
        "dynamic_cache_schedule": False,
        "tensorrt_engine_active": False,
        "enable_dit_cache": True,
        "future_root": str(Path(future_root).resolve()),
    }
    for key, wanted in expected.items():
        require(contract.get(key) == wanted, "live_d1_server_contract_mismatch", key)
    require(
        contract.get("noise_semantics") == "fixed; no request is an independent noise draw",
        "live_d1_noise_semantics_changed",
    )
    overlay = contract.get("instrumentation_overlay")
    require(isinstance(overlay, Mapping), "live_d1_instrumentation_overlay_missing")
    expected_overlay = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/experiments/forecast_layout/d1_instrumented_server.py"
    )
    require(Path(str(overlay.get("path", ""))).resolve() == expected_overlay, "live_d1_instrumentation_path_changed")
    require(overlay.get("sha256") == sha256_file(expected_overlay), "live_d1_instrumentation_hash_changed")
    require(overlay.get("returned_action_modified") is False, "live_d1_instrumentation_modified_action")
    topology = contract.get("topology")
    require(isinstance(topology, list) and len(topology) == 2, "live_d1_server_topology_invalid")
    require(sorted(row.get("rank") for row in topology) == [0, 1], "live_d1_server_rank_inventory_changed")
    require(len({row.get("cuda_device_index") for row in topology}) == 2, "live_d1_server_gpu_alias")
    require(all(row.get("cuda_device_name") == "NVIDIA B200" for row in topology), "live_d1_server_gpu_changed")
    heads = contract.get("head_contracts")
    require(isinstance(heads, list) and sorted(row.get("rank") for row in heads) == [0, 1], "live_d1_head_contracts_invalid")
    require(all(row.get("status") == "passed" for row in heads), "live_d1_head_contract_failed")
    loaders = contract.get("bounded_loader_receipts")
    require(isinstance(loaders, list) and sorted(row.get("rank") for row in loaders) == [0, 1], "live_d1_loader_receipts_invalid")
    require(all(row.get("receipt", {}).get("passed") is True for row in loaders), "live_d1_bounded_loader_failed")
    require(all(row.get("receipt", {}).get("forward_path_modified") is False for row in loaders), "live_d1_loader_modified_forward")
    identity_path = Path(str(contract.get("identity_receipt", "")))
    require(identity_path.is_absolute(), "live_d1_identity_path_invalid")
    identity = load_json(identity_path, "live_d1_identity_unreadable")
    require(sha256_file(identity_path) == contract.get("identity_receipt_sha256"), "live_d1_identity_hash_changed")
    require(identity.get("status") == "passed", "live_d1_identity_failed")
    require(identity.get("source", {}).get("commit") == D1_SOURCE_COMMIT, "live_d1_source_commit_changed")
    require(identity.get("source", {}).get("git_tree") == D1_SOURCE_TREE, "live_d1_source_tree_changed")
    require(identity.get("source", {}).get("aggregate_sha256") == D1_SOURCE_AGGREGATE_SHA256, "live_d1_source_hash_changed")
    require(identity.get("checkpoint", {}).get("revision") == CHECKPOINT_REVISION, "live_d1_checkpoint_revision_changed")
    require(identity.get("checkpoint", {}).get("aggregate_sha256") == CHECKPOINT_AGGREGATE_SHA256, "live_d1_checkpoint_hash_changed")
    require(identity.get("tokenizer", {}).get("revision") == TOKENIZER_REVISION, "live_d1_tokenizer_revision_changed")
    require(identity.get("tokenizer", {}).get("aggregate_sha256") == TOKENIZER_AGGREGATE_SHA256, "live_d1_tokenizer_hash_changed")


def validate_server_ready(
    path: Path, expected_sha256: str, *, run_id: str, server_job_id: str,
    study_commit: str, source_root: Path
) -> dict[str, Any]:
    identity = verify_exact_file(path, expected_sha256, "server_ready")
    ready = load_json(path, "server_ready_unreadable")
    expected = {
        "schema_version": SERVER_READY_SCHEMA,
        "status": "ready",
        "run_id": run_id,
        "server_job_id": server_job_id,
        "study_commit": study_commit,
        "study_id": STUDY_ID,
        "block_id": BLOCK_ID,
        "model_config": MODEL_CONFIG,
        "service_host": SERVICE_HOST,
        "service_port": SERVICE_PORT,
        "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
        "expected_cell_ids": list(CELL_IDS),
        "returned_action_shape": [RETURNED_ACTION_HORIZON, ACTION_DIM],
        "executed_prefix_horizon": EXECUTED_PREFIX_HORIZON,
        "effective_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
        "global_state_noninterleaving": True,
    }
    for key, wanted in expected.items():
        require(ready.get(key) == wanted, "server_ready_mismatch", key)
    future_root = Path(str(ready.get("future_root", "")))
    require(future_root.is_absolute(), "server_ready_future_root_invalid")
    contract_identity = _verify_descriptor(
        ready.get("server_contract"), "server_ready_contract", allowed_root=future_root
    )
    contract = load_json(Path(contract_identity["path"]), "server_ready_contract_unreadable")
    validate_live_server_contract_payload(contract, source_root=source_root, future_root=future_root)
    identity_receipt = _verify_descriptor(
        ready.get("runtime_identity"), "server_ready_runtime_identity", allowed_root=future_root
    )
    require(identity_receipt["sha256"] == contract.get("identity_receipt_sha256"), "server_ready_identity_contract_mismatch")
    require(
        ready.get("server_contract_sha256") == contract_identity["sha256"],
        "server_ready_contract_hash_binding_changed",
    )
    return {"ready": ready, "identity": identity, "contract": contract}


def validate_simulator_claim(
    path: Path, *, run_id: str, simulator_job_id: str, server_job_id: str,
    server_ready_sha256: str, study_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    claim = load_json(path, "simulator_claim_unreadable")
    expected = {
        "schema_version": SIMULATOR_CLAIM_SCHEMA,
        "status": "claimed",
        "run_id": run_id,
        "simulator_job_id": simulator_job_id,
        "server_job_id": server_job_id,
        "server_ready_sha256": server_ready_sha256,
        "study_commit": study_commit,
        "block_id": BLOCK_ID,
        "worker_role": SIMULATOR_QUEUE_ROLE,
        "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
    }
    for key, wanted in expected.items():
        require(claim.get(key) == wanted, "simulator_claim_mismatch", key)
    token = claim.get("lease_token")
    require(isinstance(token, str) and SAFE_ID_RE.fullmatch(token) is not None, "simulator_claim_token_invalid")
    start = claim.get("start_cell_index")
    require(type(start) is int and 0 <= start < len(CELL_IDS), "simulator_claim_start_index_invalid")
    return claim, file_identity(path)


def _resolve_server_artifact(value: Any, label: str, *, future_root: Path) -> dict[str, Any]:
    return _verify_descriptor(value, label, allowed_root=future_root)


def _validate_mapping_artifacts(value: Any, label: str, *, future_root: Path) -> None:
    require(isinstance(value, Mapping), "d1_mapping_evidence_missing", label)
    entries = value.get("entries")
    require(isinstance(entries, list) and value.get("entry_count") == len(entries), "d1_mapping_entry_count_changed", label)
    keys = [row.get("key") for row in entries if isinstance(row, Mapping)]
    require(len(keys) == len(entries) and len(set(keys)) == len(keys), "d1_mapping_keys_invalid", label)
    for offset, entry in enumerate(entries):
        _resolve_server_artifact(entry, f"{label}_{offset}", future_root=future_root)


def _validate_temporal_reset(reset: Mapping[str, Any], *, expected_control: Mapping[str, Any]) -> dict[str, Any]:
    require(reset.get("schema_version") == D1_RESET_SCHEMA, "d1_reset_schema_changed")
    require(reset.get("status") == "passed" and reset.get("world_size") == 2, "d1_reset_not_passed")
    require(reset.get("control") == dict(expected_control), "d1_reset_control_changed")
    ranks = reset.get("rank_receipts")
    require(isinstance(ranks, list) and sorted(row.get("rank") for row in ranks) == [0, 1], "d1_reset_rank_inventory_changed")
    reset_ids = {row.get("reset_id") for row in ranks}
    require(reset_ids == {reset.get("reset_id")}, "d1_reset_id_disagrees_across_ranks")
    scan = []
    for row in ranks:
        require(row.get("schema_version") == D1_RESET_SCHEMA and row.get("status") == "passed", "d1_rank_reset_failed")
        require(row.get("failures") == [], "d1_rank_reset_has_failures")
        after = row.get("after")
        before = row.get("before")
        require(isinstance(before, Mapping) and isinstance(after, Mapping), "d1_temporal_scan_missing")
        require(after.get("current_start_frame") == 0, "d1_reset_frame_not_zero")
        fields = after.get("fields")
        require(isinstance(fields, Mapping), "d1_reset_field_scan_missing")
        for field in RESET_FIELDS_TO_NONE:
            require(fields.get(field, {}).get("is_none") is True, "d1_temporal_field_not_cleared", field)
        if row.get("rank") == 0:
            wrapper = row.get("wrapper_after")
            require(isinstance(wrapper, Mapping), "d1_wrapper_reset_scan_missing")
            require(all(value == 0 for value in wrapper.get("frame_buffer_lengths", {}).values()), "d1_wrapper_frames_not_cleared")
            require(wrapper.get("call_count") == 0, "d1_wrapper_call_count_not_zero")
            require(wrapper.get("is_first_call") is True, "d1_wrapper_first_call_not_reset")
            require(wrapper.get("video_across_time_count") == 0, "d1_wrapper_video_history_not_cleared")
            require(wrapper.get("current_session_id") is None, "d1_wrapper_session_not_cleared")
        scan.append({
            "rank": row["rank"],
            "before": before,
            "after": after,
            "fields_cleared": row.get("fields_cleared"),
        })
    return {
        "passed": True,
        "reset_id": reset.get("reset_id"),
        "world_size": 2,
        "rank_temporal_state_scan": scan,
        "unresolved_mutable_temporal_fields": [],
    }


def validate_reset_receipt(
    path: Path, *, expected_control: Mapping[str, Any], future_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_path = Path(future_root).resolve() / "episodes" / str(expected_control["episode_id"]) / "reset_receipt.json"
    require(Path(path).resolve() == expected_path, "d1_reset_receipt_path_changed")
    reset = load_json(path, "d1_reset_receipt_unreadable")
    scan = _validate_temporal_reset(reset, expected_control=expected_control)
    return reset, file_identity(path), scan


def _validate_first_request_temporal_metrics(value: Any) -> dict[str, Any]:
    require(isinstance(value, list) and sorted(row.get("rank") for row in value) == [0, 1], "d1_request_rank_metrics_invalid")
    scans = []
    for row in value:
        pre = row.get("temporal_before")
        post = row.get("temporal_after")
        require(isinstance(pre, Mapping) and isinstance(post, Mapping), "d1_request_temporal_scan_missing")
        require(pre.get("current_start_frame") == 0, "d1_first_request_did_not_start_at_zero")
        require(type(post.get("current_start_frame")) is int and post["current_start_frame"] > 0, "d1_first_request_did_not_advance")
        for field in CACHE_FIELDS:
            require(pre.get("fields", {}).get(field, {}).get("is_none") is True, "d1_first_request_cache_not_empty", field)
            require(post.get("fields", {}).get(field, {}).get("is_none") is False, "d1_first_request_cache_not_initialized", field)
        events = row.get("cache_reinitialization", {})
        require(len(events.get("_create_kv_caches", [])) == 1, "d1_kv_cache_not_recreated_once")
        require(len(events.get("_create_crossattn_caches", [])) == 1, "d1_crossattn_cache_not_recreated_once")
        scans.append({"rank": row["rank"], "before": pre, "after": post, "cache_reinitialization": events})
    return {"passed": True, "rank_temporal_state_scan": scans, "unresolved_mutable_temporal_fields": []}


def validate_request_receipt(
    path: Path,
    *,
    future_root: Path,
    episode_id: str,
    session_id: str,
    prompt: str,
    request_index: int,
    server_contract_sha256: str,
    returned_action: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    expected_path = (
        Path(future_root).resolve() / "episodes" / episode_id
        / f"request_{request_index:04d}" / "request_receipt.json"
    )
    require(Path(path).resolve() == expected_path, "d1_request_receipt_path_changed")
    receipt = load_json(path, "d1_request_receipt_unreadable")
    expected = {
        "schema_version": D1_REQUEST_SCHEMA,
        "configuration_id": MODEL_CONFIG,
        "episode_id": episode_id,
        "request_index": request_index,
        "probe_id": None,
        "prompt": prompt,
        "session_id": session_id,
        "effective_official_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
        "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
        "custom_s2_used": False,
        "patched_s1_used": False,
        "official_forward_call_count": 1,
        "measurement_control": {
            "probe_id": None,
            "offline_decode": True,
            "probe_plan_sha256": PILOT_CONTRACT_SHA256,
        },
    }
    for key, wanted in expected.items():
        require(receipt.get(key) == wanted, "d1_request_receipt_mismatch", key)
    require(receipt.get("noise_semantics") == "fixed; this request is not an independent noise draw", "d1_request_noise_semantics_changed")
    for key in ("raw_inputs", "converted_inputs", "normalized_model_inputs"):
        _validate_mapping_artifacts(receipt.get(key), f"request_{request_index}_{key}", future_root=future_root)
    action_entry = receipt.get("official_returned_action")
    action_identity = _resolve_server_artifact(
        action_entry, f"request_{request_index}_returned_action", future_root=future_root
    )
    require(action_entry.get("shape") == [RETURNED_ACTION_HORIZON, ACTION_DIM], "d1_returned_action_shape_changed")
    require(action_entry.get("dtype") == "float32", "d1_returned_action_dtype_changed")
    persisted_action = np.load(action_identity["path"], allow_pickle=False)
    require(persisted_action.shape == (RETURNED_ACTION_HORIZON, ACTION_DIM), "d1_persisted_action_shape_changed")
    require(persisted_action.dtype == np.float32 and np.isfinite(persisted_action).all(), "d1_persisted_action_invalid")
    if returned_action is not None:
        observed = np.asarray(returned_action)
        require(observed.shape == persisted_action.shape and observed.dtype == np.float32, "d1_wire_action_shape_or_dtype_changed")
        require(np.array_equal(observed, persisted_action, equal_nan=False), "d1_wire_action_differs_from_server_artifact")
    latent = receipt.get("latent_video")
    _resolve_server_artifact(latent, f"request_{request_index}_latent", future_root=future_root)
    require(isinstance(latent.get("shape"), list) and latent["shape"], "d1_latent_shape_missing")
    require(isinstance(latent.get("data_sha256"), str), "d1_latent_data_hash_missing")
    decode = receipt.get("offline_decode")
    require(isinstance(decode, Mapping), "d1_offline_decode_missing")
    require(decode.get("requested") is True and decode.get("performed") is True, "d1_offline_decode_not_performed")
    require(decode.get("latent_data_sha256_before") == latent.get("data_sha256"), "d1_decode_input_latent_changed")
    require(decode.get("latent_data_sha256_after") == latent.get("data_sha256"), "d1_decode_mutated_latent")
    for key in ("decoded_tensor", "decoded_rgb"):
        _resolve_server_artifact(decode.get(key), f"request_{request_index}_{key}", future_root=future_root)
    metrics = receipt.get("temporal_and_cache_rank_metrics")
    require(isinstance(metrics, list) and sorted(row.get("rank") for row in metrics) == [0, 1], "d1_request_rank_metrics_invalid")
    if request_index == 0:
        _validate_first_request_temporal_metrics(metrics)
    cost = receipt.get("cost")
    require(isinstance(cost, Mapping) and cost.get("rank_count") == 2, "d1_request_cost_rank_count_changed")
    require(type(cost.get("inference_wall_seconds_rank0_wrapper")) in (int, float), "d1_request_wall_time_missing")
    return receipt, file_identity(path)


def validate_episode_manifest(
    path: Path,
    *,
    future_root: Path,
    episode_id: str,
    session_id: str,
    prompt: str,
    expected_control: Mapping[str, Any],
    server_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    expected_path = Path(future_root).resolve() / "episodes" / episode_id / "episode_manifest.json"
    require(Path(path).resolve() == expected_path, "d1_episode_manifest_path_changed")
    manifest = load_json(path, "d1_episode_manifest_unreadable")
    expected = {
        "schema_version": D1_EPISODE_SCHEMA,
        "configuration_id": MODEL_CONFIG,
        "episode_id": episode_id,
        "status": "complete",
        "official_repository_commit": D1_SOURCE_COMMIT,
        "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
        "custom_s2_used": False,
        "patched_s1_used": False,
        "effective_official_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
        "request_count": REQUEST_COUNT,
        "server_contract_sha256": server_contract_sha256,
    }
    for key, wanted in expected.items():
        require(manifest.get(key) == wanted, "d1_episode_manifest_mismatch", key)
    require(manifest.get("noise_semantics") == "fixed; not an independent draw", "d1_episode_noise_semantics_changed")
    reset = manifest.get("two_rank_reset")
    require(isinstance(reset, Mapping), "d1_episode_reset_missing")
    _validate_temporal_reset(reset, expected_control=expected_control)
    requests = manifest.get("requests")
    require(isinstance(requests, list) and len(requests) == REQUEST_COUNT, "d1_episode_request_inventory_changed")
    request_identities: list[dict[str, Any]] = []
    for index, embedded in enumerate(requests):
        request_path = expected_path.parent / f"request_{index:04d}" / "request_receipt.json"
        request, identity = validate_request_receipt(
            request_path,
            future_root=future_root,
            episode_id=episode_id,
            session_id=session_id,
            prompt=prompt,
            request_index=index,
            server_contract_sha256=server_contract_sha256,
        )
        require(request == embedded, "d1_episode_embedded_request_changed", str(index))
        request_identities.append(identity)
    return manifest, file_identity(path), request_identities


def _wait_for_file(path: Path, *, timeout: float, process: subprocess.Popen[bytes] | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process is not None:
            require(process.poll() is None, "child_exited_while_waiting_for_evidence", str(process.poll()))
        time.sleep(min(0.2, max(0.001, deadline - time.monotonic())))
    raise D1BehavioralPilotError("evidence_wait_timeout", str(path))


def _wait_for_server_ready_file(path: Path, *, timeout: float) -> str:
    """Wait on the shared PVC, then return the immutable ready receipt hash."""

    _wait_for_file(path, timeout=timeout)
    require(not Path(path).is_symlink(), "server_ready_is_symlink")
    return sha256_file(path)


def _wait_for_port(host: str, port: int, *, timeout: float, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        require(process.poll() is None, "d1_server_exited_before_socket_ready", str(process.poll()))
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as error:
            last_error = f"{type(error).__name__}:{error}"
        time.sleep(1)
    raise D1BehavioralPilotError("d1_server_socket_ready_timeout", last_error)


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
    # worker-00 is a dedicated one-GPU pod.  Keep the allocation-provided
    # NVIDIA_VISIBLE_DEVICES and do not remap it to a GPU from another pod.
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def build_cell_command(
    *, source_root: Path, attempt_root: Path, study_commit: str,
    gate_receipt: Path, gate_receipt_sha256: str,
    pose_manifest: Path, pose_manifest_sha256: str,
    run_id: str, server_job_id: str, server_ready_sha256: str,
    simulator_claim_sha256: str, lease_token: str,
    future_root: Path, condition_index: int,
) -> list[str]:
    require(0 <= condition_index < len(CONDITIONS), "cell_condition_index_invalid")
    script = Path(source_root) / "workshops/corl2026_world_models/experiments/forecast_layout/d1_behavioral_pilot_jobs.py"
    layout_arm, command, _task = CONDITIONS[condition_index]
    return [
        os.path.abspath(os.fspath(ROBOLAB_PYTHON)), str(script.resolve()), "cell",
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--attempt-root", str(Path(attempt_root).resolve()),
        "--gate-receipt", str(Path(gate_receipt).resolve()),
        "--gate-receipt-sha256", gate_receipt_sha256,
        "--pose-manifest", str(Path(pose_manifest).resolve()),
        "--pose-manifest-sha256", pose_manifest_sha256,
        "--run-id", run_id,
        "--server-job-id", server_job_id,
        "--server-ready-sha256", server_ready_sha256,
        "--simulator-claim-sha256", simulator_claim_sha256,
        "--lease-token", lease_token,
        "--future-root", str(Path(future_root).resolve()),
        "--layout-arm", layout_arm,
        "--command", command,
        "--condition-index", str(condition_index),
        "--remote-host", SERVICE_HOST,
        "--remote-port", str(SERVICE_PORT),
    ]


def run_cell(args: argparse.Namespace) -> int:
    """Run one fresh 450-action RoboLab process against the authenticated D1 service."""

    source_root = Path(args.source_root).resolve()
    attempt_root = Path(args.attempt_root).resolve()
    require(attempt_root.is_relative_to(RAW_ROOT.resolve() / "simulator_attempts"), "cell_attempt_root_outside_d1_raw_root")
    condition_index = args.condition_index
    require(0 <= condition_index < len(CONDITIONS), "cell_condition_index_invalid")
    layout_arm, command, task_name = CONDITIONS[condition_index]
    require(args.layout_arm == layout_arm and args.command == command, "cell_condition_identity_mismatch")
    cell_id = CELL_IDS[condition_index]
    cell_root = attempt_root / "cells" / f"{condition_index:02d}-{safe_component(cell_id)}"
    receipt_path = cell_root / "cell_receipt.json"
    failure_path = cell_root / "technical_failure.json"
    require(not cell_root.exists(), "cell_attempt_directory_already_exists")
    cell_root.mkdir(parents=True)
    verify_clean_git(source_root, args.study_commit, "study")
    verify_clean_git(ROBOLAB_ROOT, ROBOLAB_COMMIT, "RoboLab")

    paths = coordination_paths(RAW_ROOT, args.run_id)
    ready_bundle = validate_server_ready(
        paths["server_ready"], args.server_ready_sha256,
        run_id=args.run_id, server_job_id=args.server_job_id,
        study_commit=args.study_commit, source_root=source_root,
    )
    ready = ready_bundle["ready"]
    future_root = Path(args.future_root).resolve()
    require(future_root == Path(ready["future_root"]).resolve(), "cell_future_root_mismatch")
    claim, claim_identity = validate_simulator_claim(
        paths["simulator_claim"], run_id=args.run_id,
        simulator_job_id=attempt_root.name, server_job_id=args.server_job_id,
        server_ready_sha256=args.server_ready_sha256, study_commit=args.study_commit,
    )
    require(claim_identity["sha256"] == args.simulator_claim_sha256, "cell_simulator_claim_hash_mismatch")
    require(claim.get("lease_token") == args.lease_token, "cell_simulator_lease_token_mismatch")
    require(condition_index >= claim["start_cell_index"], "cell_precedes_resumed_prefix")
    validate_live_lease(
        paths["server_lease"], schema=SERVER_LEASE_SCHEMA,
        owner_job_id=args.server_job_id, run_id=args.run_id,
        server_ready_sha256=args.server_ready_sha256,
    )
    validate_live_lease(
        paths["simulator_lease"], schema=SIMULATOR_LEASE_SCHEMA,
        owner_job_id=attempt_root.name, run_id=args.run_id,
        server_ready_sha256=args.server_ready_sha256,
        simulator_claim_sha256=args.simulator_claim_sha256,
        lease_token=args.lease_token,
    )

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
        import cv2  # noqa: F401 - must precede Isaac/RoboLab imports.
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
        from policies.dreamzero.client import DreamZeroClient
        from robolab.eval.base_client import InferenceClient
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
        import torch
        require(torch.cuda.device_count() == 1, "simulator_child_must_see_one_logical_gpu")
        require(torch.cuda.get_device_name(0) == "NVIDIA B200", "simulator_child_gpu_is_not_b200")

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
            "effective_seed": EFFECTIVE_MODEL_NOISE_SEED,
            "source_identity": (
                f"study:{args.study_commit};robolab:{ROBOLAB_COMMIT};"
                f"dreamzero:{D1_SOURCE_COMMIT};pose:{args.pose_manifest_sha256}"
            ),
            "checkpoint_identity": (
                f"revision:{CHECKPOINT_REVISION};aggregate:{CHECKPOINT_AGGREGATE_SHA256}"
            ),
        }
        recorder = ForecastRecordingAdapter(cell_root / "recording", identity)

        class BoundD1Client(DreamZeroClient):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.wmf_request_index = 0
                self.wmf_episode_id: str | None = None
                self.wmf_session_id: str | None = None
                self.wmf_episode_active = False
                self.wmf_begin_receipt: dict[str, Any] | None = None
                self.wmf_reset_receipt: dict[str, Any] | None = None
                self.wmf_reset_identity: dict[str, Any] | None = None
                self.wmf_temporal_reset_scan: dict[str, Any] | None = None
                self.wmf_episode_manifest: dict[str, Any] | None = None
                self.wmf_episode_manifest_identity: dict[str, Any] | None = None
                self.wmf_request_identities: list[dict[str, Any]] = []
                self.wmf_finalize_error: str | None = None

            def _connect_with_retries(self) -> None:
                """Retry only connection setup; never enable protocol pings."""

                import websockets.sync.client

                last_error: BaseException | None = None
                for attempt in range(1, D1_CONNECT_RETRIES + 1):
                    try:
                        self._ws = connect_websocket_without_keepalive(
                            websockets.sync.client.connect,
                            uri=self._uri,
                            auth_headers=self._auth_headers,
                        )
                        self._server_metadata = self._packer.unpack(
                            self._ws.recv(timeout=D1_CONNECT_TIMEOUT_SECONDS)
                        )
                        return
                    except BaseException as error:
                        last_error = error
                        if self._ws is not None:
                            try:
                                self._ws.close()
                            except BaseException:
                                pass
                            self._ws = None
                        if attempt < D1_CONNECT_RETRIES:
                            time.sleep(D1_CONNECT_BACKOFF_BASE_SECONDS * attempt)
                raise D1BehavioralPilotError(
                    "d1_websocket_connection_failed", type(last_error).__name__
                ) from last_error

            def _one_shot_send_recv(self, payload: bytes, *, timeout: float) -> Any:
                """Never replay a stateful D1 reset or inference after ambiguity."""

                self._ensure_connected()
                require(self._ws is not None, "d1_websocket_not_connected")
                try:
                    self._ws.send(payload)
                    return self._ws.recv(timeout=timeout)
                except BaseException as error:
                    # The server may have consumed the request.  Retrying would
                    # silently advance its global temporal state a second time.
                    try:
                        self._ws.close()
                    except BaseException:
                        pass
                    self._ws = None
                    raise D1BehavioralPilotError(
                        "d1_stateful_transport_ambiguous_no_retry", type(error).__name__
                    ) from error

            def _send_instrumented_reset(self, control: Mapping[str, Any]) -> None:
                raw = self._one_shot_send_recv(
                    self._packer.pack(
                        {
                            "endpoint": "reset",
                            "session_ids": [control["expected_session_id"]]
                            if control.get("expected_session_id") else None,
                            "wmf_d1_reset": dict(control),
                        }
                    ),
                    timeout=args.reset_timeout,
                )
                reply = raw if isinstance(raw, str) else self._packer.unpack(raw)
                require(reply == "reset successful", "d1_service_reset_failed", repr(reply))

            def _reset_control(self, episode_id: str, session_id: str) -> dict[str, Any]:
                return {
                    "episode_id": episode_id,
                    "expected_session_id": session_id,
                    "purpose": "d1_behavioral_pilot",
                    "study_id": STUDY_ID,
                    "block_id": BLOCK_ID,
                    "cell_id": cell_id,
                    "condition_index": condition_index,
                    "layout_arm": layout_arm,
                    "command": command,
                    "server_ready_sha256": args.server_ready_sha256,
                    "simulator_claim_sha256": args.simulator_claim_sha256,
                    "simulator_lease_token": args.lease_token,
                    "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
                }

            def begin_episode(self) -> Mapping[str, Any]:
                require(not self.wmf_episode_active, "d1_client_episode_context_overlap")
                episode_id = f"d1p00-c{condition_index:02d}-{uuid.uuid4().hex}"
                session_id = str(uuid.uuid4())
                require(episode_id != session_id, "d1_context_identifiers_alias")
                control = self._reset_control(episode_id, session_id)
                self._env_session_id[0] = session_id
                self._send_instrumented_reset(control)
                # _send_recv may reconnect and clear the client's local session
                # map.  Rebind only the same authenticated value after success.
                self._env_session_id[0] = session_id
                reset_path = future_root / "episodes" / episode_id / "reset_receipt.json"
                _wait_for_file(reset_path, timeout=args.evidence_timeout)
                reset, reset_identity, reset_scan = validate_reset_receipt(
                    reset_path, expected_control=control, future_root=future_root
                )
                self.wmf_episode_id = episode_id
                self.wmf_session_id = session_id
                self.wmf_episode_active = True
                self.wmf_reset_receipt = reset
                self.wmf_reset_identity = reset_identity
                self.wmf_temporal_reset_scan = reset_scan
                self.wmf_begin_receipt = {
                    "passed": True,
                    "service_route_proved_by_reset_artifact": True,
                    "service_host": args.remote_host,
                    "service_port": args.remote_port,
                    "server_ready_sha256": args.server_ready_sha256,
                    "simulator_claim_sha256": args.simulator_claim_sha256,
                    "episode_context_id": episode_id,
                    "client_session_id": session_id,
                    "server_reset_receipt": reset_identity,
                    "temporal_state_scan": reset_scan,
                    "server_metadata": self._server_metadata,
                }
                return dict(self.wmf_begin_receipt)

            def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict[str, Any]:
                require(env_id == 0 and self.wmf_episode_active, "d1_observation_without_active_context")
                extracted = super()._extract_observation(raw_obs, env_id=env_id)
                require(extracted.get("session_id") == self.wmf_session_id, "d1_client_session_changed")
                return extracted

            def _pack_request(self, extracted_obs: dict[str, Any], instruction: str) -> dict[str, Any]:
                require(self.wmf_episode_active, "d1_request_without_active_context")
                require(instruction == PROMPTS[command], "d1_client_prompt_changed")
                require(self.wmf_request_index < REQUEST_COUNT, "d1_client_request_count_exceeded")
                request = super()._pack_request(extracted_obs, instruction)
                request["observation/joint_position"] = np.asarray(
                    extracted_obs["joint_position"], dtype=np.float64
                )
                request["observation/cartesian_position"] = np.zeros(6, dtype=np.float64)
                request["observation/gripper_position"] = np.asarray(
                    extracted_obs["gripper_position"], dtype=np.float64
                )
                for key in RAW_ARRAY_KEYS[:3]:
                    value = np.asarray(request[key])
                    require(value.shape == (180, 320, 3) and value.dtype == np.uint8, "d1_wire_image_contract_changed", key)
                require(np.asarray(request["observation/joint_position"]).dtype == np.float64, "d1_wire_joint_dtype_changed")
                require(np.asarray(request["observation/cartesian_position"]).dtype == np.float64, "d1_wire_cartesian_dtype_changed")
                require(np.asarray(request["observation/gripper_position"]).dtype == np.float64, "d1_wire_gripper_dtype_changed")
                request["wmf_d1_measurement"] = {
                    "probe_id": None,
                    "offline_decode": True,
                    "probe_plan_sha256": PILOT_CONTRACT_SHA256,
                }
                return request

            def _query_server(self, request: Any) -> dict[str, Any]:
                require(self.wmf_episode_id is not None and self.wmf_session_id is not None, "d1_query_context_missing")
                index = self.wmf_request_index
                packed_response = self._one_shot_send_recv(
                    self._packer.pack(request), timeout=args.inference_timeout
                )
                if isinstance(packed_response, str):
                    raise D1BehavioralPilotError("d1_server_returned_error", packed_response[:1000])
                raw = self._packer.unpack(packed_response)
                action = raw.get("actions", raw) if isinstance(raw, Mapping) else raw
                request_path = (
                    future_root / "episodes" / self.wmf_episode_id
                    / f"request_{index:04d}" / "request_receipt.json"
                )
                _wait_for_file(request_path, timeout=args.evidence_timeout)
                request_receipt, request_identity = validate_request_receipt(
                    request_path,
                    future_root=future_root,
                    episode_id=self.wmf_episode_id,
                    session_id=self.wmf_session_id,
                    prompt=PROMPTS[command],
                    request_index=index,
                    server_contract_sha256=ready["server_contract_sha256"],
                    returned_action=action,
                )
                latent = request_receipt["latent_video"]
                decoded_tensor = request_receipt["offline_decode"]["decoded_tensor"]
                decoded_rgb = request_receipt["offline_decode"]["decoded_rgb"]
                self.wmf_request_identities.append(request_identity)
                return {
                    "actions": action,
                    "future_evidence": {
                        "latent": {
                            "path": latent["path"], "sha256": latent["file_sha256"],
                            "bytes": latent["bytes"],
                        },
                        "decoded": {
                            "tensor": {
                                "path": decoded_tensor["path"],
                                "sha256": decoded_tensor["file_sha256"],
                                "bytes": decoded_tensor["bytes"],
                            },
                            "rgb": {
                                "path": decoded_rgb["path"],
                                "sha256": decoded_rgb["file_sha256"],
                                "bytes": decoded_rgb["bytes"],
                            },
                        },
                    },
                    "wmf_server_request_receipt": request_identity,
                    "wmf_episode_context_id": self.wmf_episode_id,
                    "wmf_request_index": index,
                    "wmf_effective_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
                }

            def _unpack_response(self, response: Any) -> np.ndarray:
                require(isinstance(response, Mapping), "d1_augmented_response_missing")
                require(response.get("wmf_episode_context_id") == self.wmf_episode_id, "d1_response_context_changed")
                require(response.get("wmf_request_index") == self.wmf_request_index, "d1_response_request_order_changed")
                require(response.get("wmf_effective_model_noise_seed") == EFFECTIVE_MODEL_NOISE_SEED, "d1_response_noise_seed_changed")
                action = np.asarray(super()._unpack_response(response))
                require(action.shape == (RETURNED_ACTION_HORIZON, ACTION_DIM), "d1_client_returned_action_shape_changed")
                require(action.dtype == np.float32 and np.isfinite(action).all(), "d1_client_returned_action_invalid")
                self.wmf_request_index += 1
                return action

            def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
                executable = np.asarray(super()._postprocess_chunk(chunk))
                require(executable.shape == (RETURNED_ACTION_HORIZON, ACTION_DIM), "d1_client_executable_chunk_shape_changed")
                require(np.isin(executable[:, -1], [0.0, 1.0]).all(), "d1_gripper_binarization_changed")
                return executable

            def reset(self, *, env_id: int | None = None) -> None:
                try:
                    if env_id is None and self.wmf_episode_active:
                        assert self.wmf_episode_id is not None
                        assert self.wmf_session_id is not None
                        control = self._reset_control(self.wmf_episode_id, self.wmf_session_id)
                        finalize = {
                            "finalize_only": True,
                            "purpose": "d1_behavioral_pilot_finalize",
                            "previous_episode_id": self.wmf_episode_id,
                            "server_ready_sha256": args.server_ready_sha256,
                            "simulator_claim_sha256": args.simulator_claim_sha256,
                            "simulator_lease_token": args.lease_token,
                            "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
                        }
                        try:
                            self._send_instrumented_reset(finalize)
                            manifest_path = future_root / "episodes" / self.wmf_episode_id / "episode_manifest.json"
                            _wait_for_file(manifest_path, timeout=args.evidence_timeout)
                            completion = recorder._final_receipt
                            if isinstance(completion, Mapping) and completion.get("stop_reason") == "action_cap":
                                manifest, manifest_identity, request_identities = validate_episode_manifest(
                                    manifest_path,
                                    future_root=future_root,
                                    episode_id=self.wmf_episode_id,
                                    session_id=self.wmf_session_id,
                                    prompt=PROMPTS[command],
                                    expected_control=control,
                                    server_contract_sha256=ready["server_contract_sha256"],
                                )
                                self.wmf_episode_manifest = manifest
                                self.wmf_episode_manifest_identity = manifest_identity
                                self.wmf_request_identities = request_identities
                            else:
                                self.wmf_episode_manifest_identity = file_identity(manifest_path)
                        except BaseException as error:
                            self.wmf_finalize_error = f"{type(error).__name__}: {error}"
                            raise
                        finally:
                            self.wmf_episode_active = False
                finally:
                    if env_id is None:
                        self._env_session_id.clear()
                    else:
                        self._env_session_id.pop(env_id, None)
                    # Calling DreamZeroClient.reset would issue an uninstrumented
                    # server reset, which the official D1 overlay correctly rejects.
                    InferenceClient.reset(self, env_id=env_id)

        class RecordedD1Client(RecordingClientMixin, BoundD1Client):
            pass

        client = RecordedD1Client(
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            open_loop_horizon=EXECUTED_PREFIX_HORIZON,
            image_height=180,
            image_width=320,
            binarize_gripper=True,
            resize="pad",
            cam2_source="right",
        )
        client.attach_forecast_recorder(recorder)
        client.reset_for_recorded_episode(client.begin_episode)

        env, env_cfg = create_env(
            task_name,
            device="cuda:0",
            seed=ENVIRONMENT_SEED,
            num_envs=1,
            instruction_type="default",
            policy="wmf_d1_behavioral_pilot",
            renderer="realtime",
            rendering_mode="balanced",
        )
        require(not hasattr(env_cfg.terminations, "success"), "constructed_task_retains_success_termination")
        assert_fixed_duration_environment(env, env_cfg)
        require(env_cfg.instruction == PROMPTS[command], "constructed_task_prompt_changed")

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
                reset_identity=f"d1:{attempt_root.name}:{cell_id}",
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
            proxy, env_cfg, 0, client,
            headless=True, save_videos=True, video_mode="viewport",
        )
        if client.wmf_episode_active:
            client.reset()
        require(client.wmf_finalize_error is None, "d1_client_finalize_failed", client.wmf_finalize_error)
        require(isinstance(client.wmf_begin_receipt, Mapping), "d1_server_begin_receipt_missing")
        require(isinstance(client.wmf_temporal_reset_scan, Mapping), "d1_temporal_reset_scan_missing")
        require(isinstance(client.wmf_episode_manifest_identity, Mapping), "d1_episode_manifest_missing")
        require(len(client.wmf_request_identities) == REQUEST_COUNT, "d1_server_request_receipt_count_changed")
        completion = recorder._final_receipt
        require(isinstance(completion, Mapping), "adapter_completion_missing")
        require(completion.get("behavioral_result_valid") is True, "adapter_rejected_behavioral_cell")
        require(completion.get("actions_executed") == ACTION_CAP, "behavioral_action_count_changed")
        require(completion.get("observation_count") == OBSERVATION_COUNT, "behavioral_observation_count_changed")
        require(completion.get("request_count") == REQUEST_COUNT, "behavioral_request_count_changed")
        require(completion.get("final_two_action_truncation_recorded") is True, "final_two_action_truncation_missing")
        final_chunk = completion.get("final_chunk")
        require(
            isinstance(final_chunk, Mapping)
            and final_chunk.get("returned_actions") == RETURNED_ACTION_HORIZON
            and final_chunk.get("eligible_executable_prefix_actions") == EXECUTED_PREFIX_HORIZON
            and final_chunk.get("executed_actions") == FINAL_EXECUTED_ACTIONS
            and final_chunk.get("unused_executable_prefix_actions") == 6
            and final_chunk.get("returned_actions_outside_executable_prefix") == 16,
            "d1_final_request_accounting_changed",
        )
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
            "effective_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
            "noise_semantics": "fixed; this cell is not an independent noise draw",
            "environment_seed": ENVIRONMENT_SEED,
            "actions_executed": ACTION_CAP,
            "observation_count": OBSERVATION_COUNT,
            "behavioral_model_request_count": REQUEST_COUNT,
            "behavioral_episode_count": 1,
            "generation_qualification_request_count": 0,
            "returned_action_shape": [RETURNED_ACTION_HORIZON, ACTION_DIM],
            "executed_prefix_horizon": EXECUTED_PREFIX_HORIZON,
            "final_chunk_executed_actions": FINAL_EXECUTED_ACTIONS,
            "source_pins": {
                "study_commit": args.study_commit,
                "robolab_commit": ROBOLAB_COMMIT,
                "dreamzero_commit": D1_SOURCE_COMMIT,
                "dreamzero_tree": D1_SOURCE_TREE,
            },
            "checkpoint_pin": {
                "revision": CHECKPOINT_REVISION,
                "aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256,
            },
            "server_ready": ready_bundle["identity"],
            "simulator_claim": claim_identity,
            "server_begin_receipt": dict(client.wmf_begin_receipt),
            "server_reset_receipt": client.wmf_reset_identity,
            "server_temporal_reset_scan": client.wmf_temporal_reset_scan,
            "server_episode_manifest": client.wmf_episode_manifest_identity,
            "server_request_receipts": client.wmf_request_identities,
            "adapter_completion": file_identity(recorder.completion_path),
            "adapter_journal": {**file_identity(recorder.journal_path), **journal},
            "native_timing_support": file_identity(timing_path),
            "viewport_video": file_identity(viewport[0]),
            "fresh_physical_checks": physical_evidence,
            "runner_timing": timing,
            "runner_environment_result_repr": repr(env_results),
            "runner_subtask_sample_count": len(subtask_status),
            "candidate_id": release["candidate_id"],
            "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
            "runtime_identity": {
                "hostname": socket.gethostname(),
                "pod_uid": os.environ.get("POD_UID"),
                "pid": os.getpid(),
                "python": sys.version,
                "python_executable": sys.executable,
                "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "robolab_module": file_identity(Path(robolab.__file__)),
                "task_file": file_identity(task_path),
            },
            "completed_at_utc": utc_now(),
            "claim_boundary": (
                "One valid learned-policy D1 behavioral pilot cell: 450 actual actions, "
                "451 original observations, and 57 official conditional action/future requests."
            ),
        }
        immutable_json(receipt_path, receipt)
        print(json.dumps({"status": "passed", "cell_id": cell_id, "receipt": str(receipt_path)}), flush=True)
        return 0
    except BaseException as error:
        if recorder is not None and not recorder.finalized:
            recorder.mark_technical_failure("d1_behavioral_cell_runtime", error)
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
                    "server_episode_manifest": getattr(client, "wmf_episode_manifest_identity", None),
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


def validate_passed_cell_receipt(
    path: Path, *, condition_index: int, study_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash and deeply validate one immutable cell before it may be resumed."""

    require(0 <= condition_index < len(CELL_IDS), "resume_cell_index_invalid")
    require(COMMIT_RE.fullmatch(study_commit) is not None, "invalid_study_commit")
    path = Path(path).resolve()
    receipt = load_json(path, "resume_cell_receipt_unreadable")
    layout_arm, command, _task = CONDITIONS[condition_index]
    expected = {
        "schema_version": CELL_RECEIPT_SCHEMA,
        "status": "passed",
        "study_id": STUDY_ID,
        "block_id": BLOCK_ID,
        "cell_id": CELL_IDS[condition_index],
        "condition_index": condition_index,
        "layout_pair_id": LAYOUT_PAIR_ID,
        "layout_arm": layout_arm,
        "command": command,
        "prompt": PROMPTS[command],
        "model_config": MODEL_CONFIG,
        "effective_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
        "noise_semantics": "fixed; this cell is not an independent noise draw",
        "environment_seed": ENVIRONMENT_SEED,
        "actions_executed": ACTION_CAP,
        "observation_count": OBSERVATION_COUNT,
        "behavioral_model_request_count": REQUEST_COUNT,
        "behavioral_episode_count": 1,
        "generation_qualification_request_count": 0,
        "returned_action_shape": [RETURNED_ACTION_HORIZON, ACTION_DIM],
        "executed_prefix_horizon": EXECUTED_PREFIX_HORIZON,
        "final_chunk_executed_actions": FINAL_EXECUTED_ACTIONS,
    }
    for key, wanted in expected.items():
        require(receipt.get(key) == wanted, "resume_cell_receipt_mismatch", key)
    source_pins = receipt.get("source_pins")
    require(isinstance(source_pins, Mapping), "resume_cell_source_pin_mismatch")
    prior_study_commit = source_pins.get("study_commit")
    require(isinstance(prior_study_commit, str) and COMMIT_RE.fullmatch(prior_study_commit) is not None, "resume_cell_study_commit_invalid")
    require(
        source_pins.get("robolab_commit") == ROBOLAB_COMMIT
        and source_pins.get("dreamzero_commit") == D1_SOURCE_COMMIT
        and source_pins.get("dreamzero_tree") == D1_SOURCE_TREE,
        "resume_cell_source_pin_mismatch",
    )
    require(
        receipt.get("checkpoint_pin")
        == {"revision": CHECKPOINT_REVISION, "aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256},
        "resume_cell_checkpoint_pin_mismatch",
    )
    for key in ("adapter_completion", "adapter_journal", "native_timing_support", "viewport_video"):
        _verify_descriptor(receipt.get(key), f"resume_{key}")
    completion = load_json(Path(receipt["adapter_completion"]["path"]), "resume_adapter_completion_unreadable")
    require(
        completion.get("behavioral_result_valid") is True
        and completion.get("actions_executed") == ACTION_CAP
        and completion.get("observation_count") == OBSERVATION_COUNT
        and completion.get("request_count") == REQUEST_COUNT
        and completion.get("final_two_action_truncation_recorded") is True,
        "resume_adapter_completion_invalid",
    )
    timing = load_json(Path(receipt["native_timing_support"]["path"]), "resume_timing_unreadable")
    require(timing.get("supported") is True, "resume_native_timing_invalid")

    ready_identity = _verify_descriptor(receipt.get("server_ready"), "resume_server_ready")
    ready = load_json(Path(ready_identity["path"]), "resume_server_ready_unreadable")
    require(ready.get("schema_version") == SERVER_READY_SCHEMA and ready.get("status") == "ready", "resume_server_ready_invalid")
    require(ready.get("study_commit") == prior_study_commit, "resume_server_ready_commit_mismatch")
    require(ready.get("block_id") == BLOCK_ID and ready.get("model_config") == MODEL_CONFIG, "resume_server_ready_block_mismatch")
    require(ready.get("pilot_contract_sha256") == PILOT_CONTRACT_SHA256, "resume_pilot_contract_changed")
    require(ready.get("service_host") == SERVICE_HOST and ready.get("service_port") == SERVICE_PORT, "resume_service_endpoint_changed")
    server_contract_identity = _verify_descriptor(ready.get("server_contract"), "resume_server_contract")
    require(server_contract_identity["sha256"] == ready.get("server_contract_sha256"), "resume_server_contract_binding_changed")
    server_contract = load_json(Path(server_contract_identity["path"]), "resume_server_contract_unreadable")
    for key, wanted in {
        "schema_version": D1_SERVER_CONTRACT_SCHEMA,
        "status": "passed",
        "official_repository_commit": D1_SOURCE_COMMIT,
        "official_repository_tree": D1_SOURCE_TREE,
        "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
        "custom_s2_used": False,
        "patched_s1_used": False,
        "world_size": 2,
        "returned_action_shape": [RETURNED_ACTION_HORIZON, ACTION_DIM],
        "executed_action_prefix": EXECUTED_PREFIX_HORIZON,
        "effective_official_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
    }.items():
        require(server_contract.get(key) == wanted, "resume_server_contract_mismatch", key)

    claim_identity = _verify_descriptor(receipt.get("simulator_claim"), "resume_simulator_claim")
    claim = load_json(Path(claim_identity["path"]), "resume_simulator_claim_unreadable")
    require(claim.get("schema_version") == SIMULATOR_CLAIM_SCHEMA and claim.get("status") == "claimed", "resume_simulator_claim_invalid")
    require(claim.get("server_ready_sha256") == ready_identity["sha256"], "resume_claim_ready_binding_changed")
    require(claim.get("study_commit") == prior_study_commit, "resume_claim_commit_mismatch")

    begin = receipt.get("server_begin_receipt")
    require(isinstance(begin, Mapping) and begin.get("passed") is True, "resume_server_begin_invalid")
    require(begin.get("service_route_proved_by_reset_artifact") is True, "resume_service_route_not_proved")
    episode_id = begin.get("episode_context_id")
    session_id = begin.get("client_session_id")
    require(isinstance(episode_id, str) and SAFE_ID_RE.fullmatch(episode_id) is not None, "resume_episode_context_invalid")
    require(isinstance(session_id, str) and session_id and session_id != episode_id, "resume_client_session_invalid")
    require(begin.get("server_ready_sha256") == ready_identity["sha256"], "resume_begin_ready_binding_changed")
    require(begin.get("simulator_claim_sha256") == claim_identity["sha256"], "resume_begin_claim_binding_changed")
    expected_control = {
        "episode_id": episode_id,
        "expected_session_id": session_id,
        "purpose": "d1_behavioral_pilot",
        "study_id": STUDY_ID,
        "block_id": BLOCK_ID,
        "cell_id": CELL_IDS[condition_index],
        "condition_index": condition_index,
        "layout_arm": layout_arm,
        "command": command,
        "server_ready_sha256": ready_identity["sha256"],
        "simulator_claim_sha256": claim_identity["sha256"],
        "simulator_lease_token": claim.get("lease_token"),
        "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
    }
    manifest_identity = _verify_descriptor(receipt.get("server_episode_manifest"), "resume_episode_manifest")
    manifest_path = Path(manifest_identity["path"])
    future_root = manifest_path.parents[2]
    require(Path(ready.get("future_root", "")).resolve() == future_root.resolve(), "resume_manifest_future_root_changed")
    reset_path = future_root / "episodes" / episode_id / "reset_receipt.json"
    _reset, reset_identity, reset_scan = validate_reset_receipt(
        reset_path, expected_control=expected_control, future_root=future_root
    )
    require(receipt.get("server_reset_receipt") == reset_identity, "resume_reset_descriptor_changed")
    require(receipt.get("server_temporal_reset_scan") == reset_scan, "resume_temporal_reset_scan_changed")
    _manifest, observed_manifest_identity, request_identities = validate_episode_manifest(
        manifest_path,
        future_root=future_root,
        episode_id=episode_id,
        session_id=session_id,
        prompt=PROMPTS[command],
        expected_control=expected_control,
        server_contract_sha256=server_contract_identity["sha256"],
    )
    require(observed_manifest_identity == manifest_identity, "resume_manifest_descriptor_changed")
    require(receipt.get("server_request_receipts") == request_identities, "resume_request_descriptor_inventory_changed")
    return receipt, file_identity(path)


def discover_completed_prefix(
    raw_root: Path, *, study_commit: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return the sole hash-verified contiguous P00 prefix, refusing contradictions."""

    raw_root = Path(raw_root).resolve()
    found: dict[int, tuple[dict[str, Any], dict[str, Any], Path]] = {}
    for path in sorted(raw_root.glob("simulator_attempts/*/cells/*/cell_receipt.json")):
        attempt_root = path.parents[2].resolve()
        require(not attempt_root.is_symlink(), "resume_attempt_is_symlink", str(attempt_root))
        value = load_json(path, "resume_cell_receipt_unreadable")
        index = value.get("condition_index")
        require(type(index) is int, "resume_cell_index_invalid", str(path))
        receipt, identity = validate_passed_cell_receipt(
            path, condition_index=index, study_commit=study_commit
        )
        require(index not in found, "resume_duplicate_passed_cell", CELL_IDS[index])
        require(not (path.parent / "technical_failure.json").exists(), "resume_contradictory_cell_receipts", CELL_IDS[index])
        found[index] = (receipt, identity, attempt_root)
    indices = sorted(found)
    require(indices == list(range(len(indices))), "resume_passed_cells_not_contiguous")
    receipts = [found[index][0] for index in indices]
    identities = [found[index][1] for index in indices]
    episode_ids = [row["server_begin_receipt"]["episode_context_id"] for row in receipts]
    session_ids = [row["server_begin_receipt"]["client_session_id"] for row in receipts]
    require(len(episode_ids) == len(set(episode_ids)), "resume_episode_context_id_reused")
    require(len(session_ids) == len(set(session_ids)), "resume_client_session_id_reused")
    require(not set(episode_ids).intersection(session_ids), "resume_context_id_alias")
    provenance = {
        "schema_version": "wmf-d1-behavioral-resume-v1",
        "status": "passed",
        "block_id": BLOCK_ID,
        "source_commit": study_commit,
        "completed_prefix_cell_ids": list(CELL_IDS[: len(indices)]),
        "start_cell_index": len(indices),
        "prior_cell_receipts": identities,
        "prior_attempt_roots": sorted({str(found[index][2]) for index in indices}),
        "prior_study_commits": sorted({row["source_pins"]["study_commit"] for row in receipts}),
        "scan_completed_at_utc": utc_now(),
    }
    return receipts, identities, provenance


def validate_simulator_terminal(
    path: Path, *, run_id: str, simulator_job_id: str, server_job_id: str,
    server_ready_sha256: str, claim_sha256: str | None = None,
    lease_token: str | None = None,
) -> dict[str, Any]:
    terminal = load_json(path, "simulator_terminal_unreadable")
    expected = {
        "schema_version": SIMULATOR_TERMINAL_SCHEMA,
        "run_id": run_id,
        "simulator_job_id": simulator_job_id,
        "server_job_id": server_job_id,
        "server_ready_sha256": server_ready_sha256,
        "simulator_claim_sha256": claim_sha256,
        "lease_token": lease_token,
        "block_id": BLOCK_ID,
    }
    for key, wanted in expected.items():
        require(terminal.get(key) == wanted, "simulator_terminal_mismatch", key)
    require(terminal.get("status") in {"passed", "technical_failure"}, "simulator_terminal_status_invalid")
    require(terminal.get("safe_for_server_shutdown") is True, "simulator_terminal_not_safe_for_shutdown")
    return terminal


def _minimal_terminal(
    *, args: argparse.Namespace, status: str, claim_identity: Mapping[str, Any] | None,
    lease_token: str | None, simulator_receipt: Mapping[str, Any] | None,
    failure: BaseException | None,
) -> dict[str, Any]:
    return {
        "schema_version": SIMULATOR_TERMINAL_SCHEMA,
        "status": status,
        "run_id": args.run_id,
        "simulator_job_id": args.job_id,
        "server_job_id": args.server_job_id,
        "server_ready_sha256": args.server_ready_sha256,
        "simulator_claim_sha256": None if claim_identity is None else claim_identity.get("sha256"),
        "lease_token": lease_token,
        "block_id": BLOCK_ID,
        "simulator_receipt": simulator_receipt,
        "failure": (
            None if failure is None else {
                "error_type": type(failure).__name__,
                "reason": getattr(failure, "reason", None),
                "detail": str(failure)[:2000],
            }
        ),
        "all_simulator_children_reaped": not _ACTIVE_CHILDREN,
        "safe_for_server_shutdown": not _ACTIVE_CHILDREN,
        "completed_at_utc": utc_now(),
    }


def _wait_for_claim_or_terminal(
    *, paths: Mapping[str, Path], process: subprocess.Popen[bytes],
    run_id: str, simulator_job_id: str, server_job_id: str,
    study_commit: str, server_ready_sha256: str, timeout: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, "d1_server_exited_before_simulator_claim", str(process.poll()))
        if paths["simulator_claim"].is_file():
            claim, identity = validate_simulator_claim(
                paths["simulator_claim"], run_id=run_id,
                simulator_job_id=simulator_job_id, server_job_id=server_job_id,
                server_ready_sha256=server_ready_sha256, study_commit=study_commit,
            )
            return claim, identity
        if paths["simulator_terminal"].is_file():
            terminal = validate_simulator_terminal(
                paths["simulator_terminal"], run_id=run_id,
                simulator_job_id=simulator_job_id, server_job_id=server_job_id,
                server_ready_sha256=server_ready_sha256,
            )
            return None, terminal
        time.sleep(LEASE_POLL_SECONDS)
    raise D1BehavioralPilotError("simulator_claim_timeout")


def _wait_for_simulator_lease(
    *, paths: Mapping[str, Path], process: subprocess.Popen[bytes],
    run_id: str, simulator_job_id: str, server_ready_sha256: str,
    simulator_claim_sha256: str, lease_token: str, timeout: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        require(process.poll() is None, "d1_server_exited_before_simulator_lease")
        if paths["simulator_terminal"].is_file():
            raise D1BehavioralPilotError("paired_simulator_terminal_before_live_lease")
        if paths["simulator_lease"].is_file():
            try:
                return validate_live_lease(
                    paths["simulator_lease"], schema=SIMULATOR_LEASE_SCHEMA,
                    owner_job_id=simulator_job_id, run_id=run_id,
                    server_ready_sha256=server_ready_sha256,
                    simulator_claim_sha256=simulator_claim_sha256,
                    lease_token=lease_token,
                )
            except BaseException as error:
                last_error = error
        time.sleep(0.2)
    raise D1BehavioralPilotError("simulator_initial_lease_timeout", str(last_error))


def run_server_job(args: argparse.Namespace) -> int:
    """Own the exact two-rank D1 server until the paired simulator is terminal."""

    source_root = Path(args.source_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    raw_root = Path(args.raw_root).resolve()
    require(raw_root == RAW_ROOT.resolve(), "d1_behavioral_raw_root_changed")
    require(args.port == SERVICE_PORT, "d1_behavioral_service_port_changed")
    paths = coordination_paths(raw_root, args.run_id)
    attempt_root = raw_root / "server_attempts" / args.job_id
    publish_path = job_dir / "publish" / "d1_behavioral_server_receipt.json"
    failure: BaseException | None = None
    process: subprocess.Popen[bytes] | None = None
    stdout = None
    stderr = None
    lease: LeaseWriter | None = None
    ready_identity: dict[str, Any] | None = None
    claim: dict[str, Any] | None = None
    claim_identity: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    queue_identity: dict[str, Any] | None = None
    d1_qualification: dict[str, Any] | None = None
    topology: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    process_exit: dict[str, Any] | None = None

    raw_root.mkdir(parents=True, exist_ok=True)
    lock_path = raw_root / ".locks" / "d1-global-server.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise D1BehavioralPilotError("d1_global_server_lock_is_held") from error
        try:
            require(not attempt_root.exists(), "d1_server_attempt_already_exists")
            require(not paths["root"].exists(), "d1_coordination_run_already_exists")
            attempt_root.mkdir(parents=True)
            paths["root"].mkdir(parents=True)
            (attempt_root / "publish").mkdir()
            with installed_signal_handlers():
                try:
                    queue_identity = validate_queue_invocation(
                        source_root=source_root, job_dir=job_dir,
                        study_commit=args.study_commit, job_id=args.job_id,
                        expected_role=SERVER_QUEUE_ROLE,
                    )
                    d1_qualification = validate_d1_qualification(
                        Path(args.d1_qualification_receipt), args.d1_qualification_receipt_sha256
                    )
                    forecast = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
                    if str(forecast) not in sys.path:
                        sys.path.insert(0, str(forecast))
                    import d1_qualification_job as d1_job

                    runtime = d1_job.RuntimePaths()
                    d1_job.validate_runtime_paths(runtime)
                    environment = d1_job._child_environment(job_dir)
                    topology = d1_job.verify_gpu_topology(
                        runtime=runtime,
                        module_path=forecast / "d1_qualification_job.py",
                        raw_dir=attempt_root,
                        cwd=source_root,
                        env=environment,
                        nvidia_smi="nvidia-smi",
                    )
                    d1_job.assert_port_available(args.port)
                    future_root = attempt_root / "future"
                    command = d1_job.build_server_command(
                        runtime=runtime,
                        source_root=source_root,
                        future_root=future_root,
                        port=args.port,
                        timeout_seconds=args.server_group_timeout_seconds,
                    )
                    stdout_path = attempt_root / "server_stdout.log"
                    stderr_path = attempt_root / "server_stderr.log"
                    process, stdout, stderr = d1_job.launch_server(
                        command, stdout_path=stdout_path, stderr_path=stderr_path,
                        cwd=source_root, env=environment,
                    )
                    _ACTIVE_CHILDREN.append(process)
                    immutable_json(
                        attempt_root / "server_process.json",
                        {
                            "schema_version": "wmf-d1-behavioral-server-process-v1",
                            "command": command,
                            "process": _proc_identity(process),
                            "environment_contract": {
                                key: environment.get(key)
                                for key in ("NVIDIA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH", "PATH", "PYTHONPYCACHEPREFIX")
                            },
                            "started_at_utc": utc_now(),
                        },
                    )
                    contract_path = future_root / "server_contract.json"
                    contract = d1_job.wait_for_server_contract(
                        process, contract_path,
                        runtime=runtime, source_root=source_root, topology=topology,
                        port=args.port, timeout_seconds=args.server_ready_timeout,
                    )
                    validate_live_server_contract_payload(
                        contract, source_root=source_root, future_root=future_root
                    )
                    _wait_for_port("127.0.0.1", args.port, timeout=120, process=process)
                    ready = {
                        "schema_version": SERVER_READY_SCHEMA,
                        "status": "ready",
                        "run_id": args.run_id,
                        "server_job_id": args.job_id,
                        "paired_simulator_job_id": args.simulator_job_id,
                        "study_commit": args.study_commit,
                        "study_id": STUDY_ID,
                        "block_id": BLOCK_ID,
                        "model_config": MODEL_CONFIG,
                        "service_host": SERVICE_HOST,
                        "service_port": SERVICE_PORT,
                        "future_root": str(future_root.resolve()),
                        "server_contract": file_identity(contract_path),
                        "server_contract_sha256": sha256_file(contract_path),
                        "runtime_identity": file_identity(future_root / "identity_receipt.json"),
                        "topology": file_identity(attempt_root / "topology_receipt.json"),
                        "d1_qualification_receipt": d1_qualification,
                        "pilot_contract": PILOT_CONTRACT,
                        "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
                        "expected_cell_ids": list(CELL_IDS),
                        "returned_action_shape": [RETURNED_ACTION_HORIZON, ACTION_DIM],
                        "executed_prefix_horizon": EXECUTED_PREFIX_HORIZON,
                        "effective_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
                        "noise_semantics": "fixed; requests and cells are not independent draws",
                        "global_state_noninterleaving": True,
                        "process": _proc_identity(process),
                        "server_instance_nonce": uuid.uuid4().hex,
                        "ready_at_utc": utc_now(),
                    }
                    immutable_json(paths["server_ready"], ready)
                    ready_identity = file_identity(paths["server_ready"])
                    lease = LeaseWriter(
                        paths["server_lease"], schema=SERVER_LEASE_SCHEMA,
                        owner_job_id=args.job_id, run_id=args.run_id,
                        server_ready_sha256=ready_identity["sha256"],
                    )
                    lease.start()
                    claim, other = _wait_for_claim_or_terminal(
                        paths=paths, process=process, run_id=args.run_id,
                        simulator_job_id=args.simulator_job_id, server_job_id=args.job_id,
                        study_commit=args.study_commit,
                        server_ready_sha256=ready_identity["sha256"],
                        timeout=args.simulator_claim_timeout,
                    )
                    if claim is None:
                        terminal = other
                        raise D1BehavioralPilotError("paired_simulator_declined_before_claim")
                    claim_identity = other
                    assert claim_identity is not None
                    _wait_for_simulator_lease(
                        paths=paths, process=process, run_id=args.run_id,
                        simulator_job_id=args.simulator_job_id,
                        server_ready_sha256=ready_identity["sha256"],
                        simulator_claim_sha256=claim_identity["sha256"],
                        lease_token=claim["lease_token"],
                    )
                    while True:
                        lease.raise_if_failed()
                        require(process.poll() is None, "d1_server_exited_while_simulator_live", str(process.poll()))
                        if paths["simulator_terminal"].is_file():
                            terminal = validate_simulator_terminal(
                                paths["simulator_terminal"], run_id=args.run_id,
                                simulator_job_id=args.simulator_job_id, server_job_id=args.job_id,
                                server_ready_sha256=ready_identity["sha256"],
                                claim_sha256=claim_identity["sha256"],
                                lease_token=claim["lease_token"],
                            )
                            break
                        validate_live_lease(
                            paths["simulator_lease"], schema=SIMULATOR_LEASE_SCHEMA,
                            owner_job_id=args.simulator_job_id, run_id=args.run_id,
                            server_ready_sha256=ready_identity["sha256"],
                            simulator_claim_sha256=claim_identity["sha256"],
                            lease_token=claim["lease_token"],
                        )
                        time.sleep(LEASE_POLL_SECONDS)
                    require(terminal.get("status") == "passed", "paired_simulator_failed")
                except BaseException as error:
                    failure = error
                finally:
                    if lease is not None:
                        try:
                            lease.close()
                        except BaseException as error:
                            if failure is None:
                                failure = error
                    if process is not None:
                        try:
                            terminate_process_group(process, grace_seconds=args.terminate_grace_seconds)
                            process_exit = {
                                "status": "reaped",
                                "returncode": process.returncode,
                                "reaped": process.poll() is not None,
                                "ended_at_utc": utc_now(),
                            }
                        except BaseException as error:
                            if failure is None:
                                failure = D1BehavioralPilotError("d1_server_reap_failed", str(error))
                        finally:
                            if process in _ACTIVE_CHILDREN:
                                _ACTIVE_CHILDREN.remove(process)
                            if stdout is not None and stderr is not None:
                                try:
                                    _close_logs(stdout, stderr)
                                except BaseException as error:
                                    if failure is None:
                                        failure = D1BehavioralPilotError("d1_server_log_close_failed", str(error))
            receipt = {
                "schema_version": SERVER_RECEIPT_SCHEMA,
                "status": "passed" if failure is None else "technical_failure",
                "exit_code": 0 if failure is None else 1,
                "run_id": args.run_id,
                "server_job_id": args.job_id,
                "paired_simulator_job_id": args.simulator_job_id,
                "study_commit": args.study_commit,
                "block_id": BLOCK_ID,
                "queue_descriptor": queue_identity,
                "d1_qualification_receipt": d1_qualification,
                "topology": None if topology is None else file_identity(attempt_root / "topology_receipt.json"),
                "server_ready": ready_identity,
                "simulator_claim": claim_identity,
                "simulator_terminal": (
                    file_identity(paths["simulator_terminal"]) if paths["simulator_terminal"].is_file() else None
                ),
                "server_process_exit": process_exit,
                "all_server_children_reaped": not _ACTIVE_CHILDREN,
                "raw_attempt_root": str(attempt_root),
                "failure": None if failure is None else {
                    "error_type": type(failure).__name__,
                    "reason": getattr(failure, "reason", None),
                    "detail": str(failure)[:2000],
                },
                "completed_at_utc": utc_now(),
            }
            immutable_json(attempt_root / "publish" / "d1_behavioral_server_receipt.json", receipt, publish=True)
            publish_path.parent.mkdir(parents=True, exist_ok=True)
            immutable_json(publish_path, receipt, publish=True)
            print(json.dumps({"status": receipt["status"], "server_ready": ready_identity}), flush=True)
            return receipt["exit_code"]
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _receipt_counts(
    *, start_cell_index: int, launched: int,
    completed: Sequence[Mapping[str, Any]], attempt_root: Path,
) -> dict[str, Any]:
    completed_ids = {str(row.get("cell_id")) for row in completed}
    invalid = 0
    censored = 0
    actions = sum(int(row.get("actions_executed", 0)) for row in completed)
    requests = sum(int(row.get("behavioral_model_request_count", 0)) for row in completed)
    for index in range(start_cell_index, start_cell_index + launched):
        if CELL_IDS[index] in completed_ids:
            continue
        failure_path = (
            Path(attempt_root) / "cells" / f"{index:02d}-{safe_component(CELL_IDS[index])}"
            / "technical_failure.json"
        )
        if failure_path.is_file():
            value = load_json(failure_path)
            actions += int(value.get("actions_executed", 0))
            requests += int(value.get("request_count", 0))
            if value.get("status") == "safety_abort":
                censored += 1
            else:
                invalid += 1
        else:
            invalid += 1
    return {
        "planned_behavioral_cells": len(CELL_IDS),
        "launched_behavioral_cells": start_cell_index + launched,
        "resumed_valid_behavioral_cells": start_cell_index,
        "newly_launched_behavioral_cells": launched,
        "completed_valid_behavioral_cells": len(completed),
        "technically_invalid_behavioral_cells": invalid,
        "right_censored_behavioral_cells": censored,
        "unrun_behavioral_cells": len(CELL_IDS) - start_cell_index - launched,
        "actual_behavioral_actions": actions,
        "actual_behavioral_model_requests": requests,
        "new_generation_qualification_requests": 0,
        "reused_prerequisite_generation_qualification_requests": 6,
        "recorder_only_episodes_counted_as_behavioral": 0,
    }


def run_simulator_job(args: argparse.Namespace) -> int:
    """Claim the exact ready server and supervise four fresh simulator cells."""

    source_root = Path(args.source_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    raw_root = Path(args.raw_root).resolve()
    require(raw_root == RAW_ROOT.resolve(), "d1_behavioral_raw_root_changed")
    require(args.remote_host == SERVICE_HOST and args.remote_port == SERVICE_PORT, "d1_service_endpoint_changed")
    paths = coordination_paths(raw_root, args.run_id)
    attempt_root = raw_root / "simulator_attempts" / args.job_id
    publish_path = job_dir / "publish" / "d1_behavioral_pilot_receipt.json"
    failure: BaseException | None = None
    queue_identity: dict[str, Any] | None = None
    schedule: dict[str, Any] | None = None
    prerequisites: dict[str, Any] | None = None
    topology: dict[str, Any] | None = None
    ready_bundle: dict[str, Any] | None = None
    claim_identity: dict[str, Any] | None = None
    lease_token: str | None = None
    lease: LeaseWriter | None = None
    resume: dict[str, Any] | None = None
    resumed_identities: list[dict[str, Any]] = []
    new_identities: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    start_cell_index = 0
    launched = 0
    receipt: dict[str, Any] | None = None

    raw_root.mkdir(parents=True, exist_ok=True)
    lock_path = raw_root / ".locks" / f"{safe_component(BLOCK_ID)}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise D1BehavioralPilotError("d1_p00_block_lock_is_held") from error
        try:
            try:
                completed, resumed_identities, resume = discover_completed_prefix(
                    raw_root, study_commit=args.study_commit
                )
                start_cell_index = len(completed)
                require(start_cell_index < len(CELL_IDS), "valid_d1_p00_block_already_exists")
                require(not attempt_root.exists(), "d1_simulator_attempt_already_exists")
                attempt_root.mkdir(parents=True)
                (attempt_root / "publish").mkdir()
                immutable_json(attempt_root / "resume.json", resume)
                with installed_signal_handlers():
                    try:
                        queue_identity = validate_queue_invocation(
                            source_root=source_root, job_dir=job_dir,
                            study_commit=args.study_commit, job_id=args.job_id,
                            expected_role=SIMULATOR_QUEUE_ROLE,
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
                            d1_qualification_receipt_path=Path(args.d1_qualification_receipt),
                            d1_qualification_receipt_sha256=args.d1_qualification_receipt_sha256,
                        )
                        observed_ready_sha256 = _wait_for_server_ready_file(
                            paths["server_ready"], timeout=args.server_ready_timeout
                        )
                        if args.server_ready_sha256 is not None:
                            require(
                                args.server_ready_sha256 == observed_ready_sha256,
                                "supplied_server_ready_sha256_mismatch",
                            )
                        args.server_ready_sha256 = observed_ready_sha256
                        ready_bundle = validate_server_ready(
                            paths["server_ready"], args.server_ready_sha256,
                            run_id=args.run_id, server_job_id=args.server_job_id,
                            study_commit=args.study_commit, source_root=source_root,
                        )
                        validate_live_lease(
                            paths["server_lease"], schema=SERVER_LEASE_SCHEMA,
                            owner_job_id=args.server_job_id, run_id=args.run_id,
                            server_ready_sha256=args.server_ready_sha256,
                        )
                        topology = verify_one_idle_b200()
                        immutable_json(attempt_root / "topology.json", topology)
                        lease_token = f"lease-{uuid.uuid4().hex}"
                        claim = {
                            "schema_version": SIMULATOR_CLAIM_SCHEMA,
                            "status": "claimed",
                            "run_id": args.run_id,
                            "simulator_job_id": args.job_id,
                            "server_job_id": args.server_job_id,
                            "server_ready_sha256": args.server_ready_sha256,
                            "study_commit": args.study_commit,
                            "block_id": BLOCK_ID,
                            "worker_role": SIMULATOR_QUEUE_ROLE,
                            "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
                            "start_cell_index": start_cell_index,
                            "completed_prefix_cell_ids": list(CELL_IDS[:start_cell_index]),
                            "lease_token": lease_token,
                            "process": {
                                "pid": os.getpid(), "hostname": socket.gethostname(),
                                "pod_uid": os.environ.get("POD_UID"),
                            },
                            "claimed_at_utc": utc_now(),
                        }
                        immutable_json(paths["simulator_claim"], claim)
                        claim_identity = file_identity(paths["simulator_claim"])
                        # Self-validate the immutable claim before advertising a lease.
                        validate_simulator_claim(
                            paths["simulator_claim"], run_id=args.run_id,
                            simulator_job_id=args.job_id, server_job_id=args.server_job_id,
                            server_ready_sha256=args.server_ready_sha256,
                            study_commit=args.study_commit,
                        )
                        lease = LeaseWriter(
                            paths["simulator_lease"], schema=SIMULATOR_LEASE_SCHEMA,
                            owner_job_id=args.job_id, run_id=args.run_id,
                            server_ready_sha256=args.server_ready_sha256,
                            simulator_claim_sha256=claim_identity["sha256"],
                            lease_token=lease_token,
                        )
                        lease.start()

                        for condition_index in range(start_cell_index, len(CONDITIONS)):
                            lease.raise_if_failed()
                            validate_live_lease(
                                paths["server_lease"], schema=SERVER_LEASE_SCHEMA,
                                owner_job_id=args.server_job_id, run_id=args.run_id,
                                server_ready_sha256=args.server_ready_sha256,
                            )
                            command = build_cell_command(
                                source_root=source_root,
                                attempt_root=attempt_root,
                                study_commit=args.study_commit,
                                gate_receipt=Path(args.gate_receipt),
                                gate_receipt_sha256=args.gate_receipt_sha256,
                                pose_manifest=Path(args.pose_manifest),
                                pose_manifest_sha256=args.pose_manifest_sha256,
                                run_id=args.run_id,
                                server_job_id=args.server_job_id,
                                server_ready_sha256=args.server_ready_sha256,
                                simulator_claim_sha256=claim_identity["sha256"],
                                lease_token=lease_token,
                                future_root=Path(ready_bundle["ready"]["future_root"]),
                                condition_index=condition_index,
                            )
                            cell_id = CELL_IDS[condition_index]
                            log_root = attempt_root / "cell_logs" / f"{condition_index:02d}-{safe_component(cell_id)}"
                            environment = build_simulator_environment(
                                source_root=source_root, state_parent=attempt_root
                            )
                            with supervised_logged_child(
                                command,
                                cwd=ROBOLAB_ROOT,
                                environment=environment,
                                stdout_path=log_root / "stdout.log",
                                stderr_path=log_root / "stderr.log",
                            ) as (child, _stdout, _stderr):
                                launched += 1
                                immutable_json(
                                    log_root / "process.json",
                                    {
                                        "schema_version": "wmf-d1-behavioral-cell-process-v1",
                                        "cell_id": cell_id,
                                        "condition_index": condition_index,
                                        "command": command,
                                        "nvidia_visible_devices": environment.get("NVIDIA_VISIBLE_DEVICES"),
                                        "process": _proc_identity(child),
                                        "started_at_utc": utc_now(),
                                    },
                                )
                                deadline = time.monotonic() + args.cell_timeout
                                while child.poll() is None:
                                    require(time.monotonic() < deadline, "d1_behavioral_cell_timeout", cell_id)
                                    lease.raise_if_failed()
                                    validate_live_lease(
                                        paths["server_lease"], schema=SERVER_LEASE_SCHEMA,
                                        owner_job_id=args.server_job_id, run_id=args.run_id,
                                        server_ready_sha256=args.server_ready_sha256,
                                    )
                                    time.sleep(LEASE_POLL_SECONDS)
                                code = child.wait()
                                require(code == 0, "d1_behavioral_cell_failed", f"{cell_id}:{code}")
                            receipt_path = (
                                attempt_root / "cells" / f"{condition_index:02d}-{safe_component(cell_id)}"
                                / "cell_receipt.json"
                            )
                            cell_receipt, cell_identity = validate_passed_cell_receipt(
                                receipt_path,
                                condition_index=condition_index,
                                study_commit=args.study_commit,
                            )
                            completed.append(cell_receipt)
                            new_identities.append(cell_identity)
                        require(len(completed) == len(CELL_IDS), "d1_behavioral_block_incomplete")
                    except BaseException as error:
                        failure = error
                    finally:
                        for residual in reversed(list(_ACTIVE_CHILDREN)):
                            try:
                                terminate_process_group(residual)
                            finally:
                                if residual in _ACTIVE_CHILDREN:
                                    _ACTIVE_CHILDREN.remove(residual)
                        if lease is not None:
                            try:
                                lease.close()
                            except BaseException as error:
                                if failure is None:
                                    failure = error
                counts = _receipt_counts(
                    start_cell_index=start_cell_index, launched=launched,
                    completed=completed, attempt_root=attempt_root,
                )
                receipt = {
                    "schema_version": SIMULATOR_RECEIPT_SCHEMA,
                    "status": "passed" if failure is None else "technical_failure",
                    "exit_code": 0 if failure is None else 1,
                    "study_id": STUDY_ID,
                    "namespace": NAMESPACE,
                    "run_id": args.run_id,
                    "server_job_id": args.server_job_id,
                    "simulator_job_id": args.job_id,
                    "study_commit": args.study_commit,
                    "block_id": BLOCK_ID,
                    "phase": PHASE,
                    "layout_pair_id": LAYOUT_PAIR_ID,
                    "model_config": MODEL_CONFIG,
                    "effective_model_noise_seed": EFFECTIVE_MODEL_NOISE_SEED,
                    "noise_semantics": "fixed; cells and requests are not independent noise draws",
                    "condition_order": [f"{arm}-{command}" for arm, command, _task in CONDITIONS],
                    "cell_ids": list(CELL_IDS),
                    "counts": counts,
                    "queue_descriptor": queue_identity,
                    "schedule": None if schedule is None else {key: schedule[key] for key in ("path", "sha256")},
                    "prerequisites": prerequisites,
                    "topology": None if topology is None else file_identity(attempt_root / "topology.json"),
                    "server_ready": None if ready_bundle is None else ready_bundle["identity"],
                    "simulator_claim": claim_identity,
                    "resume": file_identity(attempt_root / "resume.json"),
                    "resumed_cell_receipts": resumed_identities,
                    "new_cell_receipts": new_identities,
                    "cell_receipts": resumed_identities + new_identities,
                    "all_simulator_children_reaped": not _ACTIVE_CHILDREN,
                    "raw_attempt_root": str(attempt_root),
                    "raw_attempt_recoverable_on_gm_pvc": True,
                    "failure": None if failure is None else {
                        "error_type": type(failure).__name__,
                        "reason": getattr(failure, "reason", None),
                        "detail": str(failure)[:2000],
                    },
                    "completed_at_utc": utc_now(),
                    "claim_boundary": (
                        "This receipt counts only valid 450-action D1 behavioral cells. The reused "
                        "six-request generation and recorder-only qualifications remain nonbehavioral."
                    ),
                }
                immutable_json(attempt_root / "publish" / "d1_behavioral_pilot_receipt.json", receipt, publish=True)
                publish_path.parent.mkdir(parents=True, exist_ok=True)
                immutable_json(publish_path, receipt, publish=True)
            except BaseException as outer_error:
                if failure is None:
                    failure = outer_error
                if receipt is None and attempt_root.is_dir():
                    # Preserve a bounded receipt even for setup/receipt failures.
                    receipt = {
                        "schema_version": SIMULATOR_RECEIPT_SCHEMA,
                        "status": "technical_failure",
                        "exit_code": 1,
                        "run_id": args.run_id,
                        "server_job_id": args.server_job_id,
                        "simulator_job_id": args.job_id,
                        "study_commit": args.study_commit,
                        "block_id": BLOCK_ID,
                        "failure": {
                            "error_type": type(failure).__name__,
                            "reason": getattr(failure, "reason", None),
                            "detail": str(failure)[:2000],
                        },
                        "all_simulator_children_reaped": not _ACTIVE_CHILDREN,
                        "raw_attempt_root": str(attempt_root),
                        "completed_at_utc": utc_now(),
                    }
                    fallback = attempt_root / "publish" / "d1_behavioral_pilot_receipt.json"
                    if not fallback.exists():
                        immutable_json(fallback, receipt, publish=True)
                    if not publish_path.exists():
                        publish_path.parent.mkdir(parents=True, exist_ok=True)
                        immutable_json(publish_path, receipt, publish=True)
            finally:
                # Terminal is written only after every simulator child has been
                # reaped and after the aggregate receipt write was attempted.
                terminal_status = "passed" if receipt is not None and receipt.get("status") == "passed" else "technical_failure"
                terminal_receipt = (
                    file_identity(attempt_root / "publish" / "d1_behavioral_pilot_receipt.json")
                    if (attempt_root / "publish" / "d1_behavioral_pilot_receipt.json").is_file()
                    else None
                )
                terminal = _minimal_terminal(
                    args=args, status=terminal_status,
                    claim_identity=claim_identity, lease_token=lease_token,
                    simulator_receipt=terminal_receipt, failure=failure,
                )
                if paths["root"].is_dir() and not paths["simulator_terminal"].exists():
                    immutable_json(paths["simulator_terminal"], terminal)
            print(
                json.dumps(
                    {
                        "status": "passed" if failure is None else "technical_failure",
                        "counts": None if receipt is None else receipt.get("counts"),
                        "raw_attempt_root": str(attempt_root),
                    }
                ),
                flush=True,
            )
            return 0 if failure is None else 1
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server = subparsers.add_parser("server-job", help="run the two-rank D1 service on role d1")
    server.add_argument("--source-root", type=Path, required=True)
    server.add_argument("--study-commit", required=True)
    server.add_argument("--job-dir", type=Path, required=True)
    server.add_argument("--job-id", required=True)
    server.add_argument("--simulator-job-id", required=True)
    server.add_argument("--run-id", required=True)
    server.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    server.add_argument("--d1-qualification-receipt", type=Path, required=True)
    server.add_argument("--d1-qualification-receipt-sha256", required=True)
    server.add_argument("--port", type=int, default=SERVICE_PORT)
    server.add_argument("--server-ready-timeout", type=float, default=7200.0)
    server.add_argument("--simulator-claim-timeout", type=float, default=7200.0)
    server.add_argument("--server-group-timeout-seconds", type=int, default=100000)
    server.add_argument("--terminate-grace-seconds", type=float, default=10.0)

    simulator = subparsers.add_parser(
        "simulator-job", help="run the four-cell block on worker-00 against an exact ready receipt"
    )
    simulator.add_argument("--source-root", type=Path, required=True)
    simulator.add_argument("--study-commit", required=True)
    simulator.add_argument("--job-dir", type=Path, required=True)
    simulator.add_argument("--job-id", required=True)
    simulator.add_argument("--server-job-id", required=True)
    simulator.add_argument("--run-id", required=True)
    simulator.add_argument(
        "--server-ready-sha256",
        help="optional prospective binding; otherwise hash the immutable ready receipt on the PVC",
    )
    simulator.add_argument("--server-ready-timeout", type=float, default=7200.0)
    simulator.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    simulator.add_argument("--gate-receipt", type=Path, required=True)
    simulator.add_argument("--gate-receipt-sha256", required=True)
    simulator.add_argument("--pose-manifest", type=Path, required=True)
    simulator.add_argument("--pose-manifest-sha256", required=True)
    simulator.add_argument("--capture-receipt", type=Path, required=True)
    simulator.add_argument("--capture-receipt-sha256", required=True)
    simulator.add_argument("--recorder-receipt", type=Path, required=True)
    simulator.add_argument("--recorder-receipt-sha256", required=True)
    simulator.add_argument("--d1-qualification-receipt", type=Path, required=True)
    simulator.add_argument("--d1-qualification-receipt-sha256", required=True)
    simulator.add_argument("--remote-host", default=SERVICE_HOST)
    simulator.add_argument("--remote-port", type=int, default=SERVICE_PORT)
    simulator.add_argument("--cell-timeout", type=float, default=43200.0)

    cell = subparsers.add_parser("cell", help="run one fresh fixed-duration simulator cell")
    cell.add_argument("--source-root", type=Path, required=True)
    cell.add_argument("--study-commit", required=True)
    cell.add_argument("--attempt-root", type=Path, required=True)
    cell.add_argument("--gate-receipt", type=Path, required=True)
    cell.add_argument("--gate-receipt-sha256", required=True)
    cell.add_argument("--pose-manifest", type=Path, required=True)
    cell.add_argument("--pose-manifest-sha256", required=True)
    cell.add_argument("--run-id", required=True)
    cell.add_argument("--server-job-id", required=True)
    cell.add_argument("--server-ready-sha256", required=True)
    cell.add_argument("--simulator-claim-sha256", required=True)
    cell.add_argument("--lease-token", required=True)
    cell.add_argument("--future-root", type=Path, required=True)
    cell.add_argument("--layout-arm", choices=("original", "reflected"), required=True)
    cell.add_argument("--command", choices=("left", "right"), required=True)
    cell.add_argument("--condition-index", type=int, required=True)
    cell.add_argument("--remote-host", default=SERVICE_HOST)
    cell.add_argument("--remote-port", type=int, default=SERVICE_PORT)
    cell.add_argument("--evidence-timeout", type=float, default=120.0)
    cell.add_argument("--reset-timeout", type=float, default=1200.0)
    cell.add_argument("--inference-timeout", type=float, default=1200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(SAFE_ID_RE.fullmatch(args.run_id) is not None, "invalid_run_id")
    require(COMMIT_RE.fullmatch(args.study_commit) is not None, "invalid_study_commit")
    if args.mode == "server-job":
        require(SAFE_ID_RE.fullmatch(args.simulator_job_id) is not None, "invalid_simulator_job_id")
        require(
            args.server_ready_timeout > 0
            and args.simulator_claim_timeout > 0
            and args.server_group_timeout_seconds > 0
            and args.terminate_grace_seconds > 0,
            "invalid_timeout",
        )
        return run_server_job(args)
    if args.mode == "simulator-job":
        if args.server_ready_sha256 is not None:
            require(SHA256_RE.fullmatch(args.server_ready_sha256) is not None, "invalid_server_ready_sha256")
        require(args.cell_timeout > 0 and args.server_ready_timeout > 0, "invalid_timeout")
        return run_simulator_job(args)
    if args.mode == "cell":
        for value, label in (
            (args.server_ready_sha256, "server_ready"),
            (args.simulator_claim_sha256, "simulator_claim"),
        ):
            require(SHA256_RE.fullmatch(value) is not None, "invalid_sha256", label)
        require(args.remote_host == SERVICE_HOST and args.remote_port == SERVICE_PORT, "d1_service_endpoint_changed")
        require(
            args.evidence_timeout > 0 and args.reset_timeout > 0 and args.inference_timeout > 0,
            "invalid_timeout",
        )
        return run_cell(args)
    raise D1BehavioralPilotError("unknown_mode")


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
