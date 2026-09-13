#!/usr/bin/env python3
"""Queue entrypoint for one model-blind, live RoboLab fixture gate.

This outer process intentionally runs with the cluster worker's system Python.
It validates the immutable Git/PVC admission boundary and idle B200 before it
starts the pinned RoboLab Python.  Full Isaac output and camera evidence remain
on the PVC; only a bounded provenance/result receipt is placed in ``publish``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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
import tempfile
from typing import Any, Mapping, Sequence


NAMESPACE = "wmf_ablation_001_20260912"
RECEIPT_SCHEMA = "wmf-forecast-layout-fixture-job-receipt-v1"
OUTER_LEDGER_SCHEMA = "wmf-forecast-layout-fixture-job-ledger-record-v1"
SOURCE_CONTRACT_SHA256 = "88a1268ae7f27776fd5246a5808c069b2906399e99bb0e7d8b6f09020a6b85d3"
CANDIDATE_POOL_SHA256 = "ec80f4adc5272ec666c94d753241b84c954ef382bd30e1f2907459bff1cdb2b1"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
WORKER_POOL_TRANSITION_SHA256 = "b14db71d3b6dc90de8761e66ea938010f5a3193a072ef4f5c600b73be6c86ce9"
ROBOLAB_ROOT = Path("/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241")
ROBOLAB_PYTHON = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
ISAACLAB_SOURCE_ROOT = Path(
    "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaaclab/source"
)
ISAACLAB_PACKAGES = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_mimic",
    "isaaclab_rl",
)
NATIVE_LIBRARY_PATH = ":".join(
    (
        "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu",
        "/data/users/ali/glvnd/lib",
        "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib",
        "/usr/lib/x86_64-linux-gnu",
    )
)
PLANNED_LAYOUT_IDS = (
    "P00",
    *(f"D{index:02d}" for index in range(1, 5)),
    *(f"C{index:02d}" for index in range(1, 25)),
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
DECISION_EXIT = {"accepted": 0, "physical_rejection": 2, "technical_invalid": 3}
MAX_PUBLISH_BYTES = 128 * 1024


class FixtureJobError(RuntimeError):
    """Fail-closed error carrying a bounded, non-secret receipt reason."""

    def __init__(self, reason: str) -> None:
        if not SAFE_ID_RE.fullmatch(reason):
            reason = "internal_contract_error"
        super().__init__(reason)
        self.reason = reason


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise FixtureJobError(reason)


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following a venv entrypoint symlink."""

    return Path(os.path.abspath(path))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FixtureJobError("noncanonical_json_value") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                count += len(block)
    except OSError as error:
        raise FixtureJobError("evidence_file_unreadable") from error
    return {"path": str(Path(path).resolve()), "bytes": count, "sha256": digest.hexdigest()}


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, "duplicate_json_key")
        output[key] = value
    return output


def _reject_constant(_value: str) -> None:
    raise FixtureJobError("nonfinite_json_value")


def load_json(path: Path, reason: str) -> tuple[Any, bytes]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except FixtureJobError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureJobError(reason) from error
    return value, payload


def _run_git(root: Path, *argv: str, accepted: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *argv],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FixtureJobError("git_verification_unavailable") from error
    require(result.returncode in accepted, "git_verification_failed")
    return result


def verify_clean_git(root: Path, expected_commit: str, label: str) -> None:
    require(COMMIT_RE.fullmatch(expected_commit) is not None, f"{label}_commit_invalid")
    require(root.is_dir(), f"{label}_checkout_missing")
    actual = _run_git(root, "rev-parse", "HEAD").stdout.strip()
    require(actual == expected_commit, f"{label}_commit_mismatch")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    require(not status, f"{label}_checkout_dirty")


def immutable_json(path: Path, value: Mapping[str, Any]) -> bytes:
    path = Path(path)
    require(not path.is_symlink(), "publish_target_is_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.parent.is_symlink(), "publish_directory_is_symlink")
    payload = canonical_bytes(value)
    require(len(payload) <= MAX_PUBLISH_BYTES, "publish_receipt_oversize")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".fixture-receipt-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FixtureJobError("immutable_publish_receipt_exists") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload


def _append_outer_ledger(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o664)
    with os.fdopen(descriptor, "r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        lines = stream.read().splitlines()
        previous: str | None = None
        for sequence, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FixtureJobError("fixture_job_ledger_invalid") from error
            require(row.get("schema_version") == OUTER_LEDGER_SCHEMA, "fixture_job_ledger_schema_mismatch")
            require(row.get("sequence") == sequence, "fixture_job_ledger_sequence_mismatch")
            require(row.get("previous_record_sha256") == previous, "fixture_job_ledger_chain_mismatch")
            observed = row.get("record_sha256")
            require(isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None, "fixture_job_ledger_hash_invalid")
            core = dict(row)
            core.pop("record_sha256", None)
            require(sha256_bytes(canonical_bytes(core)) == observed, "fixture_job_ledger_hash_mismatch")
            previous = observed
        row = dict(record)
        row.update(
            schema_version=OUTER_LEDGER_SCHEMA,
            study_namespace=NAMESPACE,
            sequence=len(lines),
            previous_record_sha256=previous,
            recorded_at_utc=utc_now(),
        )
        row["record_sha256"] = sha256_bytes(canonical_bytes(row))
        payload = json.dumps(
            row, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8") + b"\n"
        stream.seek(0, os.SEEK_END)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        return row


def _read_gate_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise FixtureJobError("gate_ledger_unreadable") from error
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FixtureJobError("gate_ledger_invalid") from error
        require(isinstance(row, dict), "gate_ledger_invalid")
        require(row.get("schema_version") == "wmf-forecast-layout-gate-ledger-record-v1", "gate_ledger_schema_mismatch")
        require(row.get("sequence") == sequence, "gate_ledger_sequence_mismatch")
        require(row.get("previous_record_sha256") == previous, "gate_ledger_chain_mismatch")
        observed = row.get("record_sha256")
        require(isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None, "gate_ledger_hash_invalid")
        core = dict(row)
        core.pop("record_sha256", None)
        require(sha256_bytes(canonical_bytes(core)) == observed, "gate_ledger_hash_mismatch")
        require(row.get("decision") in DECISION_EXIT, "gate_ledger_decision_invalid")
        previous = observed
        records.append(row)
    return records


def _canonical_candidate_sha(candidate: Mapping[str, Any]) -> str:
    core = dict(candidate)
    core.pop("candidate_payload_sha256", None)
    return sha256_bytes(canonical_bytes(core))


def load_layout_inputs(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    forecast_root = source_root / "workshops/corl2026_world_models/experiments/forecast_layout"
    source_path = forecast_root / "layout_source_contract.json"
    pool_path = forecast_root / "layout_candidate_pool.json"
    source, source_payload = load_json(source_path, "source_contract_unreadable")
    pool, pool_payload = load_json(pool_path, "candidate_pool_unreadable")
    require(sha256_bytes(source_payload) == SOURCE_CONTRACT_SHA256, "source_contract_hash_mismatch")
    require(sha256_bytes(pool_payload) == CANDIDATE_POOL_SHA256, "candidate_pool_hash_mismatch")
    require(isinstance(source, dict) and source.get("study_namespace") == NAMESPACE, "source_contract_namespace_mismatch")
    require(source.get("schema_version") == "wmf-forecast-layout-source-contract-v1", "source_contract_schema_mismatch")
    require(source.get("scene", {}).get("robolab_commit") == ROBOLAB_COMMIT, "source_robolab_commit_mismatch")
    require(isinstance(pool, dict) and pool.get("study_namespace") == NAMESPACE, "candidate_pool_namespace_mismatch")
    require(pool.get("schema_version") == "wmf-forecast-layout-candidate-pool-v1", "candidate_pool_schema_mismatch")
    require(pool.get("source_contract_sha256") == SOURCE_CONTRACT_SHA256, "candidate_pool_source_hash_mismatch")
    require(pool.get("released") is False and pool.get("launch_ready") is False, "candidate_pool_release_boundary_bypassed")
    rows = pool.get("candidates")
    require(isinstance(rows, list) and len(rows) == 116, "candidate_pool_inventory_mismatch")
    seen: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), "candidate_row_invalid")
        candidate_id = row.get("candidate_id")
        require(isinstance(candidate_id, str) and SAFE_ID_RE.fullmatch(candidate_id) is not None, "candidate_id_invalid")
        require(candidate_id not in seen, "candidate_id_duplicate")
        seen.add(candidate_id)
        observed = row.get("candidate_payload_sha256")
        require(isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None, "candidate_hash_invalid")
        require(_canonical_candidate_sha(row) == observed, "candidate_hash_mismatch")
        require(row.get("source_contract_sha256") == SOURCE_CONTRACT_SHA256, "candidate_source_hash_mismatch")
    return source, pool


def select_candidate(
    pool: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    layout_pair_id: str,
    requested_candidate: str | None,
) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None]:
    accepted = [
        row for row in records
        if row.get("layout_pair_id") == layout_pair_id and row.get("decision") == "accepted"
    ]
    require(len(accepted) <= 1, "multiple_accepted_candidates")
    candidates = sorted(
        (row for row in pool["candidates"] if row.get("layout_pair_id") == layout_pair_id),
        key=lambda row: row.get("candidate_rank", -1),
    )
    require(len(candidates) == 4, "layout_candidate_inventory_mismatch")
    if requested_candidate is not None:
        candidate = next((row for row in candidates if row.get("candidate_id") == requested_candidate), None)
        require(candidate is not None, "candidate_not_in_layout")
        terminal = next(
            (
                row for row in reversed(records)
                if row.get("candidate_id") == requested_candidate
                and row.get("decision") in {"accepted", "physical_rejection"}
            ),
            None,
        )
        if terminal is not None:
            return None, terminal
        require(not accepted, "layout_already_accepted")
        return dict(candidate), None
    if accepted:
        return None, accepted[0]
    for candidate in candidates:
        rejected = any(
            row.get("candidate_id") == candidate["candidate_id"]
            and row.get("decision") == "physical_rejection"
            for row in records
        )
        if not rejected:
            return dict(candidate), None
    return None, None


def verify_deployment_identity(
    source_root: Path, hostname: str, pod_uid: str | None
) -> dict[str, Any]:
    receipt_path = source_root / "workshops/corl2026_world_models/execution/20260912/autonomy/deployment_receipt.json"
    receipt, _payload = load_json(receipt_path, "deployment_receipt_unreadable")
    require(isinstance(receipt, dict), "deployment_receipt_invalid")
    require(receipt.get("schema_version") == "wmf-deployment-receipt-v1", "deployment_receipt_schema_mismatch")
    require(
        receipt.get("status") == "deployed_and_execution_handed_to_durable_workstation_coordinator",
        "deployment_not_handed_off",
    )
    matches = [row for row in receipt.get("pods", []) if row.get("pod") == hostname]
    receipt_kind = "initial_deployment"
    if not matches:
        transition_path = source_root / "workshops/corl2026_world_models/execution/20260912/autonomy/worker_pool_transition.json"
        transition, transition_payload = load_json(
            transition_path, "worker_pool_transition_unreadable"
        )
        require(
            sha256_bytes(transition_payload) == WORKER_POOL_TRANSITION_SHA256,
            "worker_pool_transition_hash_mismatch",
        )
        require(isinstance(transition, dict), "worker_pool_transition_invalid")
        require(
            transition.get("schema_version") == "wmf-worker-pool-transition-v1",
            "worker_pool_transition_schema_mismatch",
        )
        require(
            transition.get("study_namespace") == NAMESPACE,
            "worker_pool_transition_namespace_mismatch",
        )
        matches = [
            row for row in transition.get("running_pods", [])
            if row.get("pod") == hostname
        ]
        receipt_path = transition_path
        receipt_kind = "corrected_worker_pool_transition"
    require(len(matches) == 1, "hostname_absent_from_deployment_receipt")
    row = matches[0]
    expected_uid = row.get("uid")
    require(
        isinstance(expected_uid, str) and SAFE_ID_RE.fullmatch(expected_uid) is not None,
        "receipt_pod_uid_invalid",
    )
    if pod_uid:
        require(expected_uid == pod_uid, "pod_uid_mismatch")
    require(row.get("role") == "worker" and row.get("phase") == "Running" and row.get("ready") is True, "pod_not_ready_worker")
    if receipt_kind == "corrected_worker_pool_transition":
        require(
            row.get("gpu_request") == 1
            and row.get("gpu_limit") == 1
            and row.get("nvidia_smi_gpu_count") == 1
            and row.get("torch_cuda_device_count") == 1,
            "transition_gpu_isolation_not_qualified",
        )
    return {
        "receipt": file_identity(receipt_path),
        "receipt_kind": receipt_kind,
        "pod": hostname,
        "pod_uid": expected_uid,
        "pod_uid_attestation_source": (
            "downward_api_environment_and_hash_bound_receipt"
            if pod_uid
            else "exact_kernel_hostname_and_hash_bound_receipt"
        ),
        "deployment_job": row.get("job"),
        "node": row.get("node"),
    }


def _query_csv(executable: Path | str, columns: Sequence[str], *, compute: bool) -> list[dict[str, str]]:
    prefix = "--query-compute-apps=" if compute else "--query-gpu="
    try:
        result = subprocess.run(
            [str(executable), prefix + ",".join(columns), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FixtureJobError("nvidia_query_unavailable") from error
    require(result.returncode == 0, "nvidia_query_failed")
    require(len(result.stdout.encode("utf-8")) <= 65536, "nvidia_query_oversize")
    rows: list[dict[str, str]] = []
    for values in csv.reader(io.StringIO(result.stdout)):
        if not values or not any(value.strip() for value in values):
            continue
        require(len(values) == len(columns), "nvidia_query_shape_mismatch")
        rows.append({key: value.strip() for key, value in zip(columns, values)})
    return rows


def verify_idle_b200(nvidia_smi: Path | str) -> dict[str, Any]:
    devices = _query_csv(nvidia_smi, ("index", "uuid", "name", "driver_version"), compute=False)
    require(len(devices) == 1, "visible_gpu_count_not_one")
    device = devices[0]
    require(device["index"] == "0", "visible_gpu_index_not_zero")
    require(device["name"] == "NVIDIA B200", "visible_gpu_not_b200")
    require(device["uuid"].startswith("GPU-") and len(device["uuid"]) <= 80, "visible_gpu_uuid_invalid")
    processes = _query_csv(
        nvidia_smi,
        ("gpu_uuid", "pid", "process_name", "used_memory"),
        compute=True,
    )
    require(not processes, "preexisting_compute_processes")
    return {
        "index": 0,
        "uuid": device["uuid"],
        "name": device["name"],
        "driver_version": device["driver_version"],
        "preexisting_compute_process_count": 0,
    }


def _validate_queue_paths(source_root: Path, state_dir: Path, job_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    try:
        source = Path(source_root).resolve(strict=True)
        state = Path(state_dir).resolve(strict=True)
        job = Path(job_dir).resolve(strict=True)
    except OSError as error:
        raise FixtureJobError("queue_path_missing") from error
    require(state.is_dir() and job.is_dir() and source.is_dir(), "queue_path_not_directory")
    require(job.parent == state / "jobs", "job_dir_outside_queue_state")
    require(source.parent == state / "sources", "source_root_outside_queue_state")
    require(SAFE_ID_RE.fullmatch(job.name) is not None, "job_id_unsafe")
    descriptor, _ = load_json(job / "descriptor.json", "job_descriptor_unreadable")
    require(isinstance(descriptor, dict), "job_descriptor_invalid")
    require(descriptor.get("schema_version") == "wmf-cluster-job-v1", "job_descriptor_schema_mismatch")
    require(descriptor.get("namespace") == NAMESPACE, "job_descriptor_namespace_mismatch")
    require(descriptor.get("job_id") == job.name, "job_descriptor_id_mismatch")
    commit = descriptor.get("source_commit")
    require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None, "job_source_commit_invalid")
    require(source.name == commit, "source_root_commit_path_mismatch")
    return source, state, job, descriptor


def _existing_publish_receipt(path: Path, job_id: str, layout_pair_id: str, source_commit: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    require(not path.is_symlink(), "publish_receipt_is_symlink")
    value, payload = load_json(path, "publish_receipt_unreadable")
    require(len(payload) <= MAX_PUBLISH_BYTES, "publish_receipt_oversize")
    require(isinstance(value, dict) and value.get("schema_version") == RECEIPT_SCHEMA, "publish_receipt_schema_mismatch")
    require(value.get("job_id") == job_id, "publish_receipt_job_mismatch")
    require(value.get("layout_pair_id") == layout_pair_id, "publish_receipt_layout_mismatch")
    require(value.get("study_commit") == source_commit, "publish_receipt_commit_mismatch")
    require(value.get("exit_code") in (0, 2, 3), "publish_receipt_exit_invalid")
    return value


def _runtime_directories(state_parent: Path, hostname: str) -> dict[str, Path]:
    safe_hostname = hostname if SAFE_ID_RE.fullmatch(hostname) else sha256_bytes(hostname.encode("utf-8"))[:24]
    root = (state_parent / "worker_runtime" / "fixture_gate" / safe_hostname).resolve()
    require(root.is_relative_to(state_parent.resolve()), "runtime_root_escape")
    directories = {
        "root": root,
        "tmp": root / "tmp",
        "xdg": root / "cache" / "xdg",
        "torch": root / "cache" / "torch",
        "huggingface": root / "cache" / "huggingface",
        "numba": root / "cache" / "numba",
        "matplotlib": root / "cache" / "matplotlib",
        "cuda": root / "cache" / "cuda",
        "warp": root / "cache" / "warp",
        "pycache": root / "cache" / "pycache",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
        require(directory.is_dir() and os.access(directory, os.W_OK), "runtime_directory_not_writable")
    return directories


def _build_child_env(
    directories: Mapping[str, Path], forecast_root: Path, robolab_root: Path
) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("DISPLAY", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    env.update(
        OMNI_KIT_ACCEPT_EULA="YES",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
        LD_LIBRARY_PATH=NATIVE_LIBRARY_PATH,
        TMPDIR=str(directories["tmp"]),
        XDG_CACHE_HOME=str(directories["xdg"]),
        TORCH_HOME=str(directories["torch"]),
        HF_HOME=str(directories["huggingface"]),
        NUMBA_CACHE_DIR=str(directories["numba"]),
        MPLCONFIGDIR=str(directories["matplotlib"]),
        CUDA_CACHE_PATH=str(directories["cuda"]),
        WARP_CACHE_PATH=str(directories["warp"]),
        PYTHONPYCACHEPREFIX=str(directories["pycache"]),
    )
    old_pythonpath = env.get("PYTHONPATH")
    python_roots = [
        str(forecast_root),
        str(Path(robolab_root).resolve()),
        *(str(ISAACLAB_SOURCE_ROOT / package) for package in ISAACLAB_PACKAGES),
    ]
    if old_pythonpath:
        python_roots.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_roots)
    return env


def _gate_evidence_from_record(record: Mapping[str, Any] | None, layout_root: Path) -> dict[str, Any]:
    if record is None:
        return {"gate_record_count": 0}
    attempt = record.get("attempt_receipt")
    require(isinstance(attempt, dict), "gate_attempt_receipt_missing")
    attempt_path = Path(str(attempt.get("path", ""))).resolve()
    require(attempt_path.is_relative_to((layout_root / "attempts").resolve()), "gate_attempt_path_escape")
    identity = file_identity(attempt_path)
    require(identity["sha256"] == attempt.get("sha256") and identity["bytes"] == attempt.get("bytes"), "gate_attempt_identity_mismatch")
    return {
        "gate_record_sha256": record.get("record_sha256"),
        "gate_attempt_receipt": identity,
        "failure_count": int(record.get("failure_count", 0)),
        "raw_attempt_directory": str(attempt_path.parent),
    }


def _publish_result(
    *,
    target: Path,
    job_id: str,
    layout_pair_id: str,
    source_commit: str,
    candidate: Mapping[str, Any] | None,
    decision: str,
    reason: str | None,
    child_exit: int | None,
    child_started: bool,
    deployment: Mapping[str, Any] | None,
    gpu: Mapping[str, Any] | None,
    gate_ledger: Path | None,
    gate_records: Sequence[Mapping[str, Any]],
    gate_record: Mapping[str, Any] | None,
    outer_record: Mapping[str, Any] | None,
    child_logs: Mapping[str, Any] | None,
    layout_root: Path | None,
) -> dict[str, Any]:
    exit_code = DECISION_EXIT[decision]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "study_namespace": NAMESPACE,
        "status": "finished",
        "job_id": job_id,
        "layout_pair_id": layout_pair_id,
        "candidate_id": candidate.get("candidate_id") if candidate else (gate_record or {}).get("candidate_id"),
        "candidate_payload_sha256": candidate.get("candidate_payload_sha256") if candidate else (gate_record or {}).get("candidate_payload_sha256"),
        "decision": decision,
        "exit_code": exit_code,
        "reason": reason,
        "child_started": child_started,
        "child_exit_code": child_exit,
        "study_commit": source_commit,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
        "robolab_commit": ROBOLAB_COMMIT,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "finished_at_utc": utc_now(),
        "deployment_identity": deployment,
        "gpu_identity": gpu,
        "fixture_job_ledger_record_sha256": outer_record.get("record_sha256") if outer_record else None,
        "gate_record_count": len(gate_records),
        "child_logs": child_logs,
        "claim_boundary": "Model-blind physical fixture qualification only; this receipt releases no model inference or behavioral episode.",
    }
    if outer_record is not None and layout_root is not None:
        outer_ledger = layout_root / "fixture_job_ledger.jsonl"
        receipt["fixture_job_ledger"] = {
            **file_identity(outer_ledger),
            "record_count": int(outer_record["sequence"]) + 1,
            "last_record_sha256": outer_record["record_sha256"],
        }
    if gate_ledger is not None and gate_ledger.exists():
        receipt["gate_ledger"] = {
            **file_identity(gate_ledger),
            "record_count": len(gate_records),
            "last_record_sha256": gate_records[-1]["record_sha256"] if gate_records else None,
        }
    if gate_record is not None and layout_root is not None:
        receipt["gate_evidence"] = _gate_evidence_from_record(gate_record, layout_root)
    immutable_json(target, receipt)
    return receipt


def execute_job(
    *,
    source_root: Path,
    state_dir: Path,
    job_dir: Path,
    layout_pair_id: str,
    requested_candidate: str | None = None,
    robolab_root: Path = ROBOLAB_ROOT,
    robolab_python: Path = ROBOLAB_PYTHON,
    expected_robolab_commit: str = ROBOLAB_COMMIT,
    nvidia_smi: Path | str = "nvidia-smi",
    hostname: str | None = None,
    pod_uid: str | None = None,
) -> dict[str, Any]:
    require(layout_pair_id in PLANNED_LAYOUT_IDS, "layout_pair_unplanned")
    if requested_candidate is not None:
        require(SAFE_ID_RE.fullmatch(requested_candidate) is not None, "candidate_id_unsafe")
    source, state, job, descriptor = _validate_queue_paths(source_root, state_dir, job_dir)
    source_commit = descriptor["source_commit"]
    publish_dir = job / "publish"
    require(not publish_dir.is_symlink(), "publish_directory_is_symlink")
    publish_dir.mkdir(exist_ok=True)
    publish_target = publish_dir / "fixture_gate_receipt.json"
    existing = _existing_publish_receipt(publish_target, job.name, layout_pair_id, source_commit)
    if existing is not None:
        return dict(existing, idempotent_replay=True)

    state_parent = state.parent.resolve()
    fixture_root = (state_parent / "fixture_gates").resolve()
    require(fixture_root.is_relative_to(state_parent), "fixture_root_escape")
    layout_root = (fixture_root / layout_pair_id).resolve()
    require(layout_root.is_relative_to(fixture_root), "layout_root_escape")
    layout_root.mkdir(parents=True, exist_ok=True)
    lock_path = layout_root / "gate.lock"
    lock_path.touch(exist_ok=True)

    context: dict[str, Any] = {
        "candidate": None,
        "deployment": None,
        "gpu": None,
        "gate_ledger": layout_root / "gate_ledger.jsonl",
        "gate_records": [],
        "gate_record": None,
        "outer_record": None,
        "child_logs": None,
        "child_started": False,
        "child_exit": None,
    }
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _existing_publish_receipt(publish_target, job.name, layout_pair_id, source_commit)
        if existing is not None:
            return dict(existing, idempotent_replay=True)
        try:
            verify_clean_git(source, source_commit, "study")
            require(expected_robolab_commit == ROBOLAB_COMMIT, "robolab_expected_commit_changed")
            verify_clean_git(Path(robolab_root).resolve(), expected_robolab_commit, "robolab")
            require(Path(robolab_python).resolve().is_file(), "robolab_python_missing")
            source_contract, pool = load_layout_inputs(source)
            del source_contract
            context["gate_records"] = _read_gate_ledger(context["gate_ledger"])
            candidate, terminal = select_candidate(
                pool,
                context["gate_records"],
                layout_pair_id,
                requested_candidate,
            )
            if terminal is not None:
                context["gate_record"] = terminal
                decision = str(terminal["decision"])
                reason = "prior_terminal_gate_record"
            elif candidate is None:
                decision = "physical_rejection"
                reason = "candidate_pool_exhausted"
            else:
                context["candidate"] = candidate
                current_hostname = hostname or socket.gethostname()
                current_pod_uid = pod_uid or os.environ.get("POD_UID")
                context["deployment"] = verify_deployment_identity(source, current_hostname, current_pod_uid)
                context["gpu"] = verify_idle_b200(nvidia_smi)
                directories = _runtime_directories(state_parent, current_hostname)
                invocation_dir = layout_root / "queue_attempts" / job.name
                require(not invocation_dir.exists() and not invocation_dir.is_symlink(), "immutable_queue_attempt_exists")
                invocation_dir.mkdir(parents=True)
                adapter_config = {
                    "study_root": str(source),
                    "robolab_root": str(Path(robolab_root).resolve()),
                    "expected_study_commit": source_commit,
                    "expected_robolab_commit": ROBOLAB_COMMIT,
                    "pod": current_hostname,
                    "pod_uid": context["deployment"]["pod_uid"],
                    "gpu_uuid": context["gpu"]["uuid"],
                    "device": "cuda:0",
                    "renderer": "realtime",
                    "rendering_type": "balanced",
                }
                adapter_path = invocation_dir / "adapter_config.json"
                immutable_json(adapter_path, adapter_config)
                forecast_root = source / "workshops/corl2026_world_models/experiments/forecast_layout"
                command = [
                    # Preserve the lexical venv entrypoint.  Resolving this
                    # symlink executes the base CPython and drops the venv's
                    # site-packages (including IsaacLab's ``toml`` dependency).
                    str(_lexical_absolute(robolab_python)),
                    str(forecast_root / "model_blind_fixture_gate.py"),
                    "qualify",
                    "--source-contract",
                    str(forecast_root / "layout_source_contract.json"),
                    "--source-contract-sha256",
                    SOURCE_CONTRACT_SHA256,
                    "--candidate-pool",
                    str(forecast_root / "layout_candidate_pool.json"),
                    "--candidate-pool-sha256",
                    CANDIDATE_POOL_SHA256,
                    "--layout-pair-id",
                    layout_pair_id,
                    "--candidate-id",
                    candidate["candidate_id"],
                    "--adapter",
                    "robolab_fixture_gate_adapter:make_adapter",
                    "--adapter-config",
                    str(adapter_path),
                    "--ledger",
                    str(context["gate_ledger"]),
                    "--attempt-root",
                    str(layout_root / "attempts"),
                ]
                stdout_path = invocation_dir / "child.stdout.log"
                stderr_path = invocation_dir / "child.stderr.log"
                child_env = _build_child_env(
                    directories, forecast_root, Path(robolab_root).resolve()
                )
                before_count = len(context["gate_records"])
                with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                    context["child_started"] = True
                    child = subprocess.run(
                        command,
                        cwd=forecast_root,
                        env=child_env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    stdout.flush()
                    stderr.flush()
                    os.fsync(stdout.fileno())
                    os.fsync(stderr.fileno())
                context["child_exit"] = child.returncode
                context["child_logs"] = {
                    "stdout": file_identity(stdout_path),
                    "stderr": file_identity(stderr_path),
                }
                context["gate_records"] = _read_gate_ledger(context["gate_ledger"])
                recognized = child.returncode in (0, 2, 3)
                if recognized:
                    require(len(context["gate_records"]) == before_count + 1, "child_gate_record_count_mismatch")
                    gate_record = context["gate_records"][-1]
                    expected_decision = {0: "accepted", 2: "physical_rejection", 3: "technical_invalid"}[child.returncode]
                    require(gate_record.get("decision") == expected_decision, "child_gate_decision_mismatch")
                    require(gate_record.get("layout_pair_id") == layout_pair_id, "child_gate_layout_mismatch")
                    require(gate_record.get("candidate_id") == candidate["candidate_id"], "child_gate_candidate_mismatch")
                    require(gate_record.get("candidate_payload_sha256") == candidate["candidate_payload_sha256"], "child_gate_candidate_hash_mismatch")
                    require(gate_record.get("candidate_pool_sha256") == CANDIDATE_POOL_SHA256, "child_gate_pool_hash_mismatch")
                    context["gate_record"] = gate_record
                    decision = expected_decision
                    reason = None
                else:
                    decision = "technical_invalid"
                    reason = "child_unexpected_exit"

            context["outer_record"] = _append_outer_ledger(
                layout_root / "fixture_job_ledger.jsonl",
                {
                    "job_id": job.name,
                    "layout_pair_id": layout_pair_id,
                    "candidate_id": (context["candidate"] or context["gate_record"] or {}).get("candidate_id"),
                    "candidate_payload_sha256": (context["candidate"] or context["gate_record"] or {}).get("candidate_payload_sha256"),
                    "study_commit": source_commit,
                    "decision": decision,
                    "exit_code": DECISION_EXIT[decision],
                    "child_started": context["child_started"],
                    "child_exit_code": context["child_exit"],
                    "gate_record_sha256": (context["gate_record"] or {}).get("record_sha256"),
                },
            )
            return _publish_result(
                target=publish_target,
                job_id=job.name,
                layout_pair_id=layout_pair_id,
                source_commit=source_commit,
                candidate=context["candidate"],
                decision=decision,
                reason=reason,
                child_exit=context["child_exit"],
                child_started=context["child_started"],
                deployment=context["deployment"],
                gpu=context["gpu"],
                gate_ledger=context["gate_ledger"],
                gate_records=context["gate_records"],
                gate_record=context["gate_record"],
                outer_record=context["outer_record"],
                child_logs=context["child_logs"],
                layout_root=layout_root,
            )
        except FixtureJobError as error:
            context["outer_record"] = _append_outer_ledger(
                layout_root / "fixture_job_ledger.jsonl",
                {
                    "job_id": job.name,
                    "layout_pair_id": layout_pair_id,
                    "candidate_id": (context["candidate"] or {}).get("candidate_id"),
                    "candidate_payload_sha256": (context["candidate"] or {}).get("candidate_payload_sha256"),
                    "study_commit": source_commit,
                    "decision": "technical_invalid",
                    "exit_code": 3,
                    "child_started": context["child_started"],
                    "child_exit_code": context["child_exit"],
                    "gate_record_sha256": (context["gate_record"] or {}).get("record_sha256"),
                    "reason": error.reason,
                },
            )
            return _publish_result(
                target=publish_target,
                job_id=job.name,
                layout_pair_id=layout_pair_id,
                source_commit=source_commit,
                candidate=context["candidate"],
                decision="technical_invalid",
                reason=error.reason,
                child_exit=context["child_exit"],
                child_started=context["child_started"],
                deployment=context["deployment"],
                gpu=context["gpu"],
                gate_ledger=context["gate_ledger"],
                gate_records=context["gate_records"],
                gate_record=context["gate_record"],
                outer_record=context["outer_record"],
                child_logs=context["child_logs"],
                layout_root=layout_root,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--layout-pair-id", choices=PLANNED_LAYOUT_IDS, required=True)
    parser.add_argument("--candidate")
    args = parser.parse_args(argv)
    try:
        receipt = execute_job(
            source_root=args.source_root,
            state_dir=args.state_dir,
            job_dir=args.job_dir,
            layout_pair_id=args.layout_pair_id,
            requested_candidate=args.candidate,
        )
    except FixtureJobError as error:
        print(json.dumps({"decision": "technical_invalid", "exit_code": 3, "reason": error.reason}, sort_keys=True))
        return 3
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "exit_code": receipt["exit_code"],
                "idempotent_replay": receipt.get("idempotent_replay", False),
                "job_id": receipt["job_id"],
                "layout_pair_id": receipt["layout_pair_id"],
                "published": True,
            },
            sort_keys=True,
        )
    )
    return int(receipt["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
