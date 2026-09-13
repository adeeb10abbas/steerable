#!/usr/bin/env python3
"""Detached GM machine-resource qualification without behavioral-cell reruns.

The first wave contains three receipt-gated jobs:

* N3 exact six-request fixed-input runtime plus a settled zero-behavioral-action
  simulator in its existing two-GPU worker;
* D1 exact six-request official runtime in its existing two-GPU worker; and
* one settled zero-behavioral-action D1 simulator on its existing one-GPU
  worker, held concurrently through a hash-bound PVC handshake.

The second wave contains one CPU-safe compiler.  A pass freezes only a
conservative one-block-at-a-time GM machine envelope.  Human annotation and
adjudication time remain unavailable until the real two-rater workflow is
complete, so no mode in this file can release confirmation.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
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
import time
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
FORECAST_RELATIVE = Path("workshops/corl2026_world_models")
LAYOUT_RELATIVE = FORECAST_RELATIVE / "experiments/forecast_layout"
THIS_RELATIVE = LAYOUT_RELATIVE / "development_resource_qualification_jobs.py"
CONTRACT_RELATIVE = LAYOUT_RELATIVE / "development_resource_qualification_contract.json"
SAMPLER_RELATIVE = LAYOUT_RELATIVE / "resource_gpu_sampler.py"
QUEUE_RELATIVE = LAYOUT_RELATIVE / "forecast_timing_queue_jobs.py"
FIXED_RELATIVE = LAYOUT_RELATIVE / "fixed_observation_job.py"
N3_RELATIVE = LAYOUT_RELATIVE / "n3_first_live.py"
D1_JOB_RELATIVE = LAYOUT_RELATIVE / "d1_qualification_job.py"


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


queue = _load_module(Path(__file__).with_name("forecast_timing_queue_jobs.py"), "wmf_resource_qualification_queue")
gpu = _load_module(Path(__file__).with_name("resource_gpu_sampler.py"), "wmf_resource_gpu_sampler")

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
RAW_ROOT = CONTROL_ROOT.parent

CONTRACT_SCHEMA = "wmf-development-resource-qualification-contract-v1"
WAVE_SCHEMA = "wmf-development-resource-qualification-wave-v1"
FINALIZE_WAVE_SCHEMA = "wmf-development-resource-qualification-finalize-wave-v1"
JOB_RECEIPT_SCHEMA = "wmf-development-resource-qualification-job-v1"
FINAL_RECEIPT_SCHEMA = "wmf-development-resource-machine-freeze-v1"
SHARED_SCHEMA = "wmf-development-resource-peer-handshake-v1"
CHILD_LAUNCH_SCHEMA = "wmf-development-resource-child-launch-v1"
CHILD_LAUNCH_ATTEMPT_SCHEMA = "wmf-development-resource-child-launch-attempt-v1"
CHILD_LAUNCH_OUTCOME_SCHEMA = "wmf-development-resource-child-launch-outcome-v1"
SIMULATOR_ACCOUNTING_SCHEMA = "wmf-development-resource-simulator-accounting-event-v1"
QUEUE_JOB_SCHEMA = queue.QUEUE_JOB_SCHEMA

COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")

N3_JOB_ID = "development-resource-qualification-n3-001"
D1_MODEL_JOB_ID = "development-resource-qualification-d1-model-001"
D1_SIM_JOB_ID = "development-resource-qualification-d1-simulator-001"
FINALIZE_JOB_ID = "development-resource-qualification-finalize-001"
D1_SESSION_ID = "development-resource-qualification-d1-001"

N3_ROLE = "n3"
N3_WORKER_ID = "wmf-forecast-0912-worker-n3-00"
D1_MODEL_ROLE = "d1"
D1_MODEL_WORKER_ID = "wmf-forecast-0912-worker-d1-00"
D1_SIM_ROLE = "wmf-forecast-0912-worker-00"
D1_SIM_WORKER_ID = D1_SIM_ROLE
FINALIZE_ROLE = "wmf-forecast-0912-worker-09"

RESOURCE_AUDIT_JOB_ID = "development-resource-compiler-formal-001"
RESOURCE_AUDIT_SCHEMA = "wmf-development-resource-queue-job-v1"
RESOURCE_AUDIT_PVC_RECEIPT = (
    CONTROL_ROOT / "jobs" / RESOURCE_AUDIT_JOB_ID / "publish/development_resource_job_receipt.json"
)

N3_PRIOR_RECEIPT = queue.N3_GENERATION_CLUSTER_JOB_RECEIPT
N3_PRIOR_RECEIPT_SHA256 = "1ead7d0c58b8abd7c27361ef37c42a4e131354212c490cd019f45918f150ea2a"
N3_OBSERVATION_MANIFEST = queue.PREPARATION_RAW_ROOT / "observation_manifest.json"
N3_OBSERVATION_MANIFEST_SHA256 = queue.PREPARATION_MANIFEST_SHA256
N3_SOURCE = queue.N3_SOURCE
N3_CHECKPOINT = queue.N3_CHECKPOINT
N3_PYTHON = queue.COSMOS_PYTHON
N3_COMPAT_BIN = queue.N3_COMPAT_BIN
N3_HF_HOME = queue.N3_HF_HOME
N3_LD_LIBRARY_PATH = queue.N3_LD_LIBRARY_PATH
SYSTEM_PATH = queue.SYSTEM_PATH

D1_PRIOR_RECEIPT = queue.D1_QUALIFICATION_RECEIPT
D1_PRIOR_RECEIPT_SHA256 = queue.D1_QUALIFICATION_RECEIPT_SHA256
FIXED_CAPTURE = queue.CAPTURE_RECEIPT
FIXED_CAPTURE_SHA256 = queue.CAPTURE_RECEIPT_SHA256
ROBOLAB_PYTHON = queue.ROBOLAB_PYTHON
ROBOLAB_ROOT = Path("/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241")

MAX_WALL_SECONDS = 50000
PUBLISH_LOG_TAIL_BYTES = 0
MIB = 1024 * 1024


class ResourceQualificationError(RuntimeError):
    """A descriptor, runtime measurement, or compiler failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceQualificationError(message)


def _verified_commit(value: Any) -> str:
    require(isinstance(value, str) and COMMIT_RE.fullmatch(value) is not None,
            "study commit is invalid")
    return value


def _verified_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} is not SHA-256")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return queue.load_json(path, label)


def _contract(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    value = _load_json(Path(root) / CONTRACT_RELATIVE, "resource qualification contract")
    require(value.get("schema_version") == CONTRACT_SCHEMA, "resource qualification contract schema changed")
    require(value.get("study_id") == STUDY_ID, "resource qualification contract study changed")
    scope = value.get("probe_scope")
    require(isinstance(scope, Mapping), "resource qualification scope is missing")
    require(scope.get("new_nonbehavioral_generation_requests_total") == 12,
            "resource qualification request count changed")
    require(scope.get("robot_episodes") == scope.get("behavioral_actions") == 0,
            "resource qualification contract claims behavior")
    runtime = value.get("runtime")
    require(
        isinstance(runtime, Mapping)
        and runtime.get("peer_failure_poll_interval_seconds") == 0.25,
        "resource qualification peer-failure polling changed",
    )
    failure_accounting = value.get("failure_accounting")
    require(
        isinstance(failure_accounting, Mapping)
        and "attempt/outcome" in str(
            failure_accounting.get("child_launch_count_authority", "")
        )
        and "lower and upper bounds" in str(
            failure_accounting.get("simulator_count_authority", "")
        )
        and "every 0.25 seconds" in str(
            failure_accounting.get("d1_peer_failure_monitor", "")
        )
        and failure_accounting.get("behavioral_effect", "").startswith("none;"),
        "resource qualification failure accounting changed",
    )
    topology = value.get("topology_freeze_policy")
    require(
        isinstance(topology, Mapping)
        and topology.get("global_max_parallel_blocks") == 1
        and topology.get("cross_model_simultaneous_blocks_allowed") is False
        and topology.get("max_parallel_blocks_by_model") == {"N3": 1, "D1": 1},
        "resource qualification topology policy changed",
    )
    boundary = value.get("release_boundary")
    require(
        isinstance(boundary, Mapping)
        and boundary.get("safe_to_release_confirmation") is False
        and boundary.get("annotation_time_may_be_synthesized") is False
        and boundary.get("confirmation_jobs_emitted") is False,
        "resource qualification release boundary changed",
    )
    return value


def _dependency_paths(root: Path) -> dict[str, Path]:
    base = Path(root) / LAYOUT_RELATIVE
    return {
        "fixed_observation_job.py": base / "fixed_observation_job.py",
        "n3_first_live.py": base / "n3_first_live.py",
        "n3_first_live_contract.json": base / "n3_first_live_contract.json",
        "d1_qualification_job.py": base / "d1_qualification_job.py",
        "d1_instrumented_server.py": base / "d1_instrumented_server.py",
        "d1_probe.py": base / "d1_probe.py",
        "d1_probe_plan.json": base / "d1_probe_plan.json",
        "d1_identity_contract.json": base / "d1_identity_contract.json",
        "fixture_tasks.py": base / "fixture_tasks.py",
        "robolab_fixture_gate_adapter.py": base / "robolab_fixture_gate_adapter.py",
        "task_files/original_left.py": base / "task_files/original_left.py",
    }


def _validate_dependencies(root: Path, contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected = contract.get("pinned_workshop_dependencies")
    require(isinstance(expected, Mapping), "pinned workshop dependency inventory is missing")
    paths = _dependency_paths(root)
    require(set(paths) == set(expected), "pinned workshop dependency inventory changed")
    result = {}
    for name, path in paths.items():
        identity = queue.file_identity(path)
        require(identity["sha256"] == expected[name], f"pinned dependency changed: {name}")
        result[name] = identity
    return result


def _implementation(root: Path) -> dict[str, dict[str, Any]]:
    contract = _contract(root)
    result = {
        "wrapper": queue.file_identity(Path(root) / THIS_RELATIVE),
        "contract": queue.file_identity(Path(root) / CONTRACT_RELATIVE),
        "sampler": queue.file_identity(Path(root) / SAMPLER_RELATIVE),
        "queue_support": queue.file_identity(Path(root) / QUEUE_RELATIVE),
    }
    result.update({f"dependency:{name}": value for name, value in _validate_dependencies(root, contract).items()})
    return result


def _zero_behavior_counts() -> dict[str, int]:
    return {
        "robot_episodes": 0,
        "behavioral_cells": 0,
        "behavioral_actions": 0,
        "labels_created": 0,
        "retained_behavioral_cells_rerun": 0,
        "retained_behavioral_cells_relabelled": 0,
    }


def _validate_audit_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = queue.file_identity(path)
    require(identity["sha256"] == _verified_sha(expected_sha256, "resource audit receipt digest"),
            "resource audit receipt hash mismatch")
    value = _load_json(path, "resource audit receipt")
    queue.verify_signed_document(value, "resource audit receipt")
    require(value.get("schema_version") == RESOURCE_AUDIT_SCHEMA, "resource audit schema changed")
    require(value.get("job_id") == RESOURCE_AUDIT_JOB_ID, "resource audit job changed")
    require(value.get("status") == "passed_with_declared_missingness", "resource audit did not pass")
    require(value.get("decision") == "no_go_resource_gate_incomplete", "resource audit decision changed")
    require(value.get("safe_to_release_confirmation") is False, "resource audit incorrectly released confirmation")
    missing = value.get("missing_release_requirements")
    require(isinstance(missing, list) and any("annotation" in str(item) for item in missing),
            "resource audit no longer declares annotation missingness")
    return {"artifact": identity, "receipt": value}


@dataclass(frozen=True)
class ProbeJob:
    mode: str
    job_id: str
    role: str
    worker_id: str


PROBE_JOBS = (
    ProbeJob("n3-profile", N3_JOB_ID, N3_ROLE, N3_WORKER_ID),
    ProbeJob("d1-model-profile", D1_MODEL_JOB_ID, D1_MODEL_ROLE, D1_MODEL_WORKER_ID),
    ProbeJob("d1-simulator-profile", D1_SIM_JOB_ID, D1_SIM_ROLE, D1_SIM_WORKER_ID),
)
PROBE_BY_MODE = {job.mode: job for job in PROBE_JOBS}


def _probe_argv(
    job: ProbeJob,
    study_commit: str,
    *,
    implementation: Mapping[str, Mapping[str, Any]],
    resource_audit_sha256: str,
) -> list[str]:
    return [
        "/usr/bin/python3",
        "{source_root}/" + str(THIS_RELATIVE),
        job.mode,
        "--source-root", "{source_root}",
        "--study-commit", study_commit,
        "--job-dir", "{job_dir}",
        "--job-id", job.job_id,
        "--expected-role", job.role,
        "--expected-worker-id", job.worker_id,
        "--wrapper-sha256", implementation["wrapper"]["sha256"],
        "--contract-sha256", implementation["contract"]["sha256"],
        "--sampler-sha256", implementation["sampler"]["sha256"],
        "--queue-support-sha256", implementation["queue_support"]["sha256"],
        "--resource-audit-receipt", str(RESOURCE_AUDIT_PVC_RECEIPT),
        "--resource-audit-receipt-sha256", resource_audit_sha256,
    ]


def build_probe_descriptor(
    job: ProbeJob,
    study_commit: str,
    *,
    implementation: Mapping[str, Mapping[str, Any]],
    resource_audit_sha256: str,
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "released": True,
        "source_commit": _verified_commit(study_commit),
        "role": job.role,
        "argv": _probe_argv(
            job,
            study_commit,
            implementation=implementation,
            resource_audit_sha256=resource_audit_sha256,
        ),
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def build_probe_wave(
    *, study_commit: str, resource_audit_receipt: Path, resource_audit_receipt_sha256: str
) -> dict[str, Any]:
    commit = _verified_commit(study_commit)
    audit = _validate_audit_receipt(resource_audit_receipt, resource_audit_receipt_sha256)
    implementation = _implementation(REPOSITORY_ROOT)
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "descriptor_only_not_dispatched_audit_gate_passed",
        "source_commit": commit,
        "source_resource_audit": audit["artifact"],
        "new_nonbehavioral_generation_requests_if_all_jobs_run": 12,
        "new_zero_behavioral_action_simulator_resets_if_all_jobs_run": 2,
        **_zero_behavior_counts(),
        "jobs": [
            build_probe_descriptor(
                job,
                commit,
                implementation=implementation,
                resource_audit_sha256=audit["artifact"]["sha256"],
            )
            for job in PROBE_JOBS
        ],
        "safe_to_release_confirmation": False,
        "claim_boundary": (
            "Three detached machine-resource probes only; descriptors are not an active queue. "
            "They add twelve nonbehavioral fixed-input requests and no behavioral cell."
        ),
    }


def _runtime_probe_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    require(args.command in PROBE_BY_MODE, "runtime probe mode changed")
    job = PROBE_BY_MODE[args.command]
    require(args.job_id == job.job_id, "runtime resource job ID changed")
    require(args.expected_role == job.role, "runtime resource role changed")
    require(args.expected_worker_id == job.worker_id, "runtime resource worker changed")
    implementation = {
        "wrapper": {"sha256": args.wrapper_sha256},
        "contract": {"sha256": args.contract_sha256},
        "sampler": {"sha256": args.sampler_sha256},
        "queue_support": {"sha256": args.queue_support_sha256},
    }
    return build_probe_descriptor(
        job,
        args.study_commit,
        implementation=implementation,
        resource_audit_sha256=args.resource_audit_receipt_sha256,
    )


def _validate_staged_implementation(
    source_root: Path, args: argparse.Namespace
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    observed = _implementation(source_root)
    for key, wanted in (
        ("wrapper", args.wrapper_sha256),
        ("contract", args.contract_sha256),
        ("sampler", args.sampler_sha256),
        ("queue_support", args.queue_support_sha256),
    ):
        require(observed[key]["sha256"] == wanted, f"staged resource {key} changed")
    return observed, _contract(source_root)


def _validate_runtime_prerequisites(args: argparse.Namespace) -> dict[str, Any]:
    require(Path(args.resource_audit_receipt) == RESOURCE_AUDIT_PVC_RECEIPT,
            "runtime resource audit path changed")
    audit = _validate_audit_receipt(
        Path(args.resource_audit_receipt), args.resource_audit_receipt_sha256
    )
    n3_identity = queue.file_identity(N3_PRIOR_RECEIPT)
    require(n3_identity["sha256"] == N3_PRIOR_RECEIPT_SHA256,
            "prior N3 generation receipt changed")
    n3_prior = _load_json(N3_PRIOR_RECEIPT, "prior N3 generation receipt")
    queue.verify_signed_document(n3_prior, "prior N3 generation receipt")
    require(n3_prior.get("status") == "passed" and n3_prior.get("decision") == "go",
            "prior N3 generation did not pass")
    require(n3_prior.get("science_counts", {}).get("model_requests_issued_by_job") == 6,
            "prior N3 generation request count changed")
    d1_identity = queue.file_identity(D1_PRIOR_RECEIPT)
    require(d1_identity["sha256"] == D1_PRIOR_RECEIPT_SHA256,
            "prior D1 generation receipt changed")
    d1_prior = _load_json(D1_PRIOR_RECEIPT, "prior D1 generation receipt")
    require(d1_prior.get("decision") == "qualified" and d1_prior.get("generation_request_count") == 6,
            "prior D1 generation did not qualify")
    capture_identity = queue.file_identity(FIXED_CAPTURE)
    require(capture_identity["sha256"] == FIXED_CAPTURE_SHA256, "fixed P00 capture changed")
    manifest_identity = queue.file_identity(N3_OBSERVATION_MANIFEST)
    require(manifest_identity["sha256"] == N3_OBSERVATION_MANIFEST_SHA256,
            "fixed N3 observation manifest changed")
    return {
        "development_resource_audit": audit["artifact"],
        "prior_n3_generation": n3_identity,
        "prior_d1_generation": d1_identity,
        "fixed_p00_capture": capture_identity,
        "fixed_n3_observation_manifest": manifest_identity,
    }


def _session_root(study_commit: str) -> Path:
    return CONTROL_ROOT / "resource_sessions" / f"{D1_SESSION_ID}-{_verified_commit(study_commit)}"


def _immutable_signed(path: Path, value: Mapping[str, Any], *, maximum_bytes: int = 1024 * 1024) -> dict[str, Any]:
    signed = gpu.signed_document(value)
    queue.immutable_json(path, signed, maximum_bytes=maximum_bytes)
    return signed


class _SimulatorAccountingJournal:
    """Durable hash chain around reset and every reset-settling action.

    A before-event is fsynced before the corresponding call.  An after-event is
    fsynced only after that call returns.  A technical failure can therefore
    report an exact completed count, or an honest one-call interval when the
    call's completion is unknowable.
    """

    def __init__(
        self,
        path: Path,
        *,
        study_commit: str,
        model_job_id: str,
        simulator_job_id: str,
    ) -> None:
        self.path = Path(path)
        require(not self.path.exists() and not self.path.is_symlink(),
                "simulator accounting journal already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        require(not self.path.parent.is_symlink(),
                "simulator accounting journal parent is a symlink")
        self.study_commit = _verified_commit(study_commit)
        self.model_job_id = model_job_id
        self.simulator_job_id = simulator_job_id
        self.sequence = 0
        self.tail: str | None = None

    def append(self, event: str, **fields: Any) -> None:
        require(
            isinstance(event, str)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", event) is not None,
            "simulator accounting event name is invalid",
        )
        reserved = {
            "schema_version", "study_id", "study_commit", "model_job_id",
            "simulator_job_id", "sequence", "previous_event_sha256", "event",
            "wall_time_ns", "monotonic_time_ns", "event_sha256",
        }
        require(not (reserved & set(fields)),
                "simulator accounting event uses a reserved field")
        base = {
            "schema_version": SIMULATOR_ACCOUNTING_SCHEMA,
            "study_id": STUDY_ID,
            "study_commit": self.study_commit,
            "model_job_id": self.model_job_id,
            "simulator_job_id": self.simulator_job_id,
            "sequence": self.sequence,
            "previous_event_sha256": self.tail,
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_time_ns": time.monotonic_ns(),
            **fields,
        }
        digest = gpu.sha256_bytes(gpu.compact_bytes(base))
        row = {**base, "event_sha256": digest}
        mode = "xb" if self.sequence == 0 else "ab"
        with self.path.open(mode) as stream:
            stream.write(gpu.compact_bytes(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.sequence == 0:
            queue._fsync_directory(self.path.parent)
        self.sequence += 1
        self.tail = digest


def _load_simulator_accounting_events(
    path: Path,
    *,
    study_commit: str,
    model_job_id: str,
    simulator_job_id: str,
) -> list[dict[str, Any]]:
    supplied = Path(path)
    require(supplied.is_file() and not supplied.is_symlink(),
            "simulator accounting journal is missing or invalid")
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, raw in enumerate(supplied.read_bytes().splitlines()):
        require(bool(raw), "simulator accounting journal contains a blank row")
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResourceQualificationError(
                "simulator accounting journal contains invalid JSON"
            ) from error
        require(isinstance(row, dict), "simulator accounting event is not an object")
        require(row.get("schema_version") == SIMULATOR_ACCOUNTING_SCHEMA,
                "simulator accounting schema changed")
        require(row.get("study_id") == STUDY_ID,
                "simulator accounting study changed")
        require(row.get("study_commit") == study_commit,
                "simulator accounting commit changed")
        require(row.get("model_job_id") == model_job_id,
                "simulator accounting model job changed")
        require(row.get("simulator_job_id") == simulator_job_id,
                "simulator accounting simulator job changed")
        require(row.get("sequence") == sequence,
                "simulator accounting sequence changed")
        require(row.get("previous_event_sha256") == previous,
                "simulator accounting chain changed")
        observed = row.get("event_sha256")
        require(isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None,
                "simulator accounting event signature is invalid")
        unsigned = dict(row)
        unsigned.pop("event_sha256")
        require(gpu.sha256_bytes(gpu.compact_bytes(unsigned)) == observed,
                "simulator accounting event signature changed")
        for key in ("wall_time_ns", "monotonic_time_ns"):
            require(type(row.get(key)) is int and row[key] > 0,
                    f"simulator accounting {key} is invalid")
        previous = observed
        rows.append(row)
    require(rows, "simulator accounting journal is empty")
    return rows


def _summarize_simulator_accounting(
    path: Path,
    *,
    study_commit: str,
    model_job_id: str,
    simulator_job_id: str,
) -> dict[str, Any]:
    """Authenticate lifecycle events and return exact values or tight bounds."""

    rows = _load_simulator_accounting_events(
        path,
        study_commit=study_commit,
        model_job_id=model_job_id,
        simulator_job_id=simulator_job_id,
    )
    allowed = {
        "simulator_child_entered",
        "simulation_app_launch_started",
        "simulation_app_launch_completed",
        "environment_creation_started",
        "environment_creation_completed",
        "physical_reset_started",
        "physical_reset_completed",
        "physical_reset_failed",
        "settling_hold_started",
        "settling_hold_completed",
        "settling_hold_failed",
        "settling_validation_completed",
        "simulator_ready_persist_started",
        "simulator_ready_persist_completed",
    }
    require(all(row.get("event") in allowed for row in rows),
            "simulator accounting contains an unknown event")
    names = [str(row["event"]) for row in rows]
    require(names[0] == "simulator_child_entered",
            "simulator accounting does not begin at child entry")

    reset_started = [row for row in rows if row["event"] == "physical_reset_started"]
    reset_completed = [row for row in rows if row["event"] == "physical_reset_completed"]
    reset_failed = [row for row in rows if row["event"] == "physical_reset_failed"]
    require(
        len(reset_started) <= 1
        and len(reset_completed) <= 1
        and len(reset_failed) <= 1
        and len(reset_completed) + len(reset_failed) <= 1,
            "simulator accounting repeats the physical reset")
    reset_terminal = reset_completed + reset_failed
    require(not reset_terminal or reset_started,
            "simulator accounting closes a reset that never started")
    if reset_terminal:
        require(reset_started[0]["sequence"] < reset_terminal[0]["sequence"],
                "simulator accounting reset order changed")

    started_holds: dict[int, dict[str, Any]] = {}
    completed_holds: set[int] = set()
    terminal_holds: set[int] = set()
    failed_holds: set[int] = set()
    for row in rows:
        if row["event"] not in {"settling_hold_started", "settling_hold_completed", "settling_hold_failed"}:
            continue
        ordinal = row.get("hold_ordinal")
        phase = row.get("settling_phase")
        require(type(ordinal) is int and ordinal >= 0,
                "simulator accounting hold ordinal is invalid")
        require(phase in {"settle", "stability"},
                "simulator accounting settling phase is invalid")
        if row["event"] == "settling_hold_started":
            require(ordinal == len(started_holds) and ordinal not in started_holds,
                    "simulator accounting hold starts are not contiguous")
            require(not failed_holds,
                    "simulator accounting continues after a failed hold")
            require(ordinal == 0 or ordinal - 1 in completed_holds,
                    "simulator accounting overlaps settling holds")
            require(reset_completed and row["sequence"] > reset_completed[0]["sequence"],
                    "simulator accounting hold precedes completed reset")
            started_holds[ordinal] = row
        else:
            require(ordinal in started_holds,
                    "simulator accounting closes a hold that never started")
            require(started_holds[ordinal]["settling_phase"] == phase,
                    "simulator accounting hold phase changed")
            require(ordinal not in terminal_holds,
                    "simulator accounting hold closes more than once")
            require(started_holds[ordinal]["sequence"] < row["sequence"],
                    "simulator accounting hold order changed")
            terminal_holds.add(ordinal)
            if row["event"] == "settling_hold_completed":
                completed_holds.add(ordinal)
            else:
                failed_holds.add(ordinal)

    validations = [row for row in rows if row["event"] == "settling_validation_completed"]
    require(len(validations) <= 1, "simulator accounting repeats settling validation")
    if validations:
        hold_terminal_sequences = [
            row["sequence"]
            for row in rows
            if row["event"] in {"settling_hold_completed", "settling_hold_failed"}
        ]
        require(
            reset_completed
            and len(started_holds) == len(completed_holds)
            and not failed_holds
            and validations[0].get("settling_hold_action_count") == len(completed_holds)
            and validations[0]["sequence"] > max(
                [reset_completed[0]["sequence"]]
                + hold_terminal_sequences
            ),
            "simulator accounting validates incomplete settling",
        )

    ready_started = [row for row in rows if row["event"] == "simulator_ready_persist_started"]
    ready_completed = [row for row in rows if row["event"] == "simulator_ready_persist_completed"]
    require(len(ready_started) <= 1 and len(ready_completed) <= 1,
            "simulator accounting repeats ready persistence")
    require(not ready_completed or ready_started,
            "simulator accounting completes ready persistence without starting it")
    if ready_started:
        require(validations and ready_started[0]["sequence"] > validations[0]["sequence"],
                "simulator accounting starts readiness before settling validation")
    if ready_completed:
        require(ready_started[0]["sequence"] < ready_completed[0]["sequence"],
                "simulator accounting ready-persistence order changed")

    reset_lower = len(reset_completed)
    reset_upper = len(reset_started)
    holds_lower = len(completed_holds)
    holds_upper = len(started_holds)
    physical_resets = reset_lower if reset_lower == reset_upper else None
    settling_holds = holds_lower if holds_lower == holds_upper else None
    return {
        "status": "authenticated",
        "journal": queue.file_identity(path),
        "event_count": len(rows),
        "tail_event_sha256": rows[-1]["event_sha256"],
        "last_event": rows[-1]["event"],
        "physical_resets": physical_resets,
        "physical_resets_lower_bound": reset_lower,
        "physical_resets_upper_bound": reset_upper,
        "settling_hold_actions": settling_holds,
        "settling_hold_actions_lower_bound": holds_lower,
        "settling_hold_actions_upper_bound": holds_upper,
        "reset_completion_ambiguous": reset_lower != reset_upper,
        "settling_completion_ambiguous": holds_lower != holds_upper,
    }


def _verify_shared(path: Path, *, kind: str, study_commit: str) -> dict[str, Any]:
    value = _load_json(path, f"D1 peer {kind}")
    gpu.verify_signed_document(value, f"D1 peer {kind}")
    require(value.get("schema_version") == SHARED_SCHEMA, f"D1 peer {kind} schema changed")
    require(value.get("kind") == kind, f"D1 peer {kind} kind changed")
    require(value.get("study_commit") == study_commit, f"D1 peer {kind} commit changed")
    require(value.get("session_id") == D1_SESSION_ID, f"D1 peer {kind} session changed")
    require(value.get("model_job_id") == D1_MODEL_JOB_ID, f"D1 peer {kind} model job changed")
    require(value.get("simulator_job_id") == D1_SIM_JOB_ID, f"D1 peer {kind} simulator job changed")
    return value


def _raise_if_d1_simulator_failed(path: Path, *, study_commit: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    require(path.is_file() and not path.is_symlink(), "D1 simulator failure marker is invalid")
    failure = _verify_shared(path, kind="simulator_failed", study_commit=study_commit)
    require(failure.get("status") == "technical_invalid",
            "D1 simulator failure marker status changed")
    raise ResourceQualificationError(
        "D1 simulator peer became technical-invalid: "
        + str(failure.get("error_type", "unknown"))
    )


def _wait_for_d1_simulator_ready(
    ready_path: Path,
    failure_path: Path,
    *,
    study_commit: str,
    timeout_seconds: float,
) -> None:
    require(math.isfinite(timeout_seconds) and timeout_seconds > 0,
            "D1 simulator wait timeout is invalid")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _raise_if_d1_simulator_failed(failure_path, study_commit=study_commit)
        if ready_path.is_file() and not ready_path.is_symlink():
            return
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
    _raise_if_d1_simulator_failed(failure_path, study_commit=study_commit)
    raise ResourceQualificationError("timed out waiting for D1 simulator readiness")


def _wait_for_file(
    path: Path,
    *,
    timeout_seconds: float,
    child: subprocess.Popen[bytes] | None = None,
) -> None:
    require(math.isfinite(timeout_seconds) and timeout_seconds > 0, "file wait timeout is invalid")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file() and not path.is_symlink():
            return
        if child is not None and child.poll() is not None:
            raise ResourceQualificationError(f"child exited before evidence appeared: {path.name}")
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
    raise ResourceQualificationError(f"timed out waiting for evidence: {path.name}")


def _preflight_gpu_count(count: int, name: str) -> dict[str, Any]:
    snapshot = gpu.query_nvidia_smi()
    require(len(snapshot["devices"]) == count, "visible GPU count changed")
    require(all(row["name"] == name for row in snapshot["devices"]), "visible GPU model changed")
    require(not snapshot["compute_processes"], "assigned GPU has a pre-existing compute process")
    return snapshot


def _proc_identity(
    process: subprocess.Popen[bytes], *, fallback_pgid: int | None = None
) -> dict[str, Any]:
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        require(type(fallback_pgid) is int and fallback_pgid > 1,
                "exited child lacks a known process group")
        pgid = fallback_pgid
    identity: dict[str, Any] = {
        "pid": process.pid,
        "pgid": pgid,
        "hostname": socket.gethostname(),
        "started_wall_time_ns": time.time_ns(),
        "started_monotonic_ns": time.monotonic_ns(),
    }
    try:
        stat_fields = Path(f"/proc/{process.pid}/stat").read_text(encoding="utf-8").split()
        identity["linux_proc_start_ticks"] = int(stat_fields[21])
        identity["linux_boot_id"] = Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="utf-8").strip()
    except (OSError, ValueError, IndexError):
        identity["linux_proc_start_ticks"] = None
        identity["linux_boot_id"] = None
    return identity


def _launch_evidence_path(path: Path, kind: str) -> Path:
    require(kind in {"attempt", "outcome"}, "child launch evidence kind is invalid")
    supplied = Path(path)
    require(supplied.name.endswith(".json"), "child launch receipt name is invalid")
    return supplied.with_name(f"{supplied.name[:-5]}.{kind}.json")


def _launch_common(
    *, launch_kind: str, study_commit: str, job_id: str,
    command: Sequence[str], cwd: Path,
) -> dict[str, Any]:
    require(
        launch_kind in {"simulator", "n3_model", "d1_server", "d1_probe"},
        "child launch kind is invalid",
    )
    require(job_id in {N3_JOB_ID, D1_MODEL_JOB_ID, D1_SIM_JOB_ID},
            "child launch job ID is invalid")
    return {
        "study_id": STUDY_ID,
        "study_commit": _verified_commit(study_commit),
        "job_id": job_id,
        "launch_kind": launch_kind,
        "command_sha256": gpu.sha256_bytes(gpu.compact_bytes(list(command))),
        "cwd": str(Path(cwd).resolve()),
    }


def _record_child_launch_attempt(
    launch_receipt_path: Path,
    *, launch_kind: str, study_commit: str, job_id: str,
    command: Sequence[str], cwd: Path,
) -> dict[str, Any]:
    """Fsync an attempt marker immediately before entering ``Popen``."""

    return _immutable_signed(_launch_evidence_path(launch_receipt_path, "attempt"), {
        "schema_version": CHILD_LAUNCH_ATTEMPT_SCHEMA,
        "status": "popen_about_to_be_called",
        **_launch_common(
            launch_kind=launch_kind,
            study_commit=study_commit,
            job_id=job_id,
            command=command,
            cwd=cwd,
        ),
        "wrapper": {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "hostname": socket.gethostname(),
        },
        "attempt_wall_time_ns": time.time_ns(),
        "attempt_monotonic_time_ns": time.monotonic_ns(),
    })


def _record_child_launch_outcome(
    launch_receipt_path: Path,
    *, attempt: Mapping[str, Any], process: subprocess.Popen[bytes] | None,
    launch_kind: str, study_commit: str, job_id: str,
    command: Sequence[str], cwd: Path, error: BaseException | None = None,
) -> dict[str, Any]:
    """Fsync whether ``Popen`` returned; this is the count authority."""

    process_started = process is not None
    require(process_started == (error is None), "child launch outcome is inconsistent")
    value: dict[str, Any] = {
        "schema_version": CHILD_LAUNCH_OUTCOME_SCHEMA,
        "status": "popen_succeeded" if process_started else "popen_failed",
        **_launch_common(
            launch_kind=launch_kind,
            study_commit=study_commit,
            job_id=job_id,
            command=command,
            cwd=cwd,
        ),
        "attempt_payload_sha256": _verified_sha(
            attempt.get("payload_sha256"), "child launch attempt payload"
        ),
        "process_started": process_started,
        "outcome_wall_time_ns": time.time_ns(),
        "outcome_monotonic_time_ns": time.monotonic_ns(),
    }
    if process_started:
        assert process is not None
        value["process"] = _proc_identity(
            process,
            fallback_pgid=(process.pid if launch_kind == "d1_server" else os.getpgrp()),
        )
        value["error_type"] = None
    else:
        assert error is not None
        value["process"] = None
        value["error_type"] = type(error).__name__
    return _immutable_signed(_launch_evidence_path(launch_receipt_path, "outcome"), value)


def _verify_child_launch_attempt(
    launch_receipt_path: Path,
    *, launch_kind: str, study_commit: str, job_id: str,
) -> dict[str, Any]:
    path = _launch_evidence_path(launch_receipt_path, "attempt")
    value = _load_json(path, f"{launch_kind} launch attempt")
    gpu.verify_signed_document(value, f"{launch_kind} launch attempt")
    expected = {
        "schema_version": CHILD_LAUNCH_ATTEMPT_SCHEMA,
        "status": "popen_about_to_be_called",
        "study_id": STUDY_ID,
        "study_commit": study_commit,
        "job_id": job_id,
        "launch_kind": launch_kind,
    }
    for key, wanted in expected.items():
        require(value.get(key) == wanted, f"{launch_kind} launch attempt changed: {key}")
    _verified_sha(value.get("command_sha256"), f"{launch_kind} launch attempt command")
    require(isinstance(value.get("cwd"), str) and Path(value["cwd"]).is_absolute(),
            f"{launch_kind} launch attempt cwd is invalid")
    wrapper = value.get("wrapper")
    require(
        isinstance(wrapper, Mapping)
        and type(wrapper.get("pid")) is int and wrapper["pid"] > 1
        and type(wrapper.get("pgid")) is int and wrapper["pgid"] > 1
        and isinstance(wrapper.get("hostname"), str) and wrapper["hostname"],
        f"{launch_kind} launch attempt wrapper is invalid",
    )
    return value


def _verify_child_launch_outcome(
    launch_receipt_path: Path,
    *, launch_kind: str, study_commit: str, job_id: str,
) -> dict[str, Any]:
    attempt = _verify_child_launch_attempt(
        launch_receipt_path,
        launch_kind=launch_kind,
        study_commit=study_commit,
        job_id=job_id,
    )
    path = _launch_evidence_path(launch_receipt_path, "outcome")
    value = _load_json(path, f"{launch_kind} launch outcome")
    gpu.verify_signed_document(value, f"{launch_kind} launch outcome")
    expected = {
        "schema_version": CHILD_LAUNCH_OUTCOME_SCHEMA,
        "study_id": STUDY_ID,
        "study_commit": study_commit,
        "job_id": job_id,
        "launch_kind": launch_kind,
        "attempt_payload_sha256": attempt["payload_sha256"],
    }
    for key, wanted in expected.items():
        require(value.get(key) == wanted, f"{launch_kind} launch outcome changed: {key}")
    started = value.get("process_started")
    require(type(started) is bool, f"{launch_kind} launch outcome flag is invalid")
    require(
        value.get("status") == ("popen_succeeded" if started else "popen_failed"),
        f"{launch_kind} launch outcome status changed",
    )
    if started:
        process = value.get("process")
        require(
            isinstance(process, Mapping)
            and type(process.get("pid")) is int and process["pid"] > 1
            and type(process.get("pgid")) is int and process["pgid"] > 1,
            f"{launch_kind} launch outcome process is invalid",
        )
        require(value.get("error_type") is None,
                f"{launch_kind} successful launch carries an error")
    else:
        require(value.get("process") is None,
                f"{launch_kind} failed launch carries a process")
        require(isinstance(value.get("error_type"), str) and value["error_type"],
                f"{launch_kind} failed launch lacks an error type")
    return value


def _record_child_launch(
    path: Path,
    *,
    process: subprocess.Popen[bytes],
    launch_kind: str,
    study_commit: str,
    job_id: str,
    command: Sequence[str],
    cwd: Path,
    attempt: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist launch success only after ``Popen`` returned a real child."""

    require(process.pid > 1, "child launch PID is invalid")
    common = _launch_common(
        launch_kind=launch_kind,
        study_commit=study_commit,
        job_id=job_id,
        command=command,
        cwd=cwd,
    )
    require(outcome.get("process_started") is True,
            "child launch success lacks a successful Popen outcome")
    require(attempt.get("payload_sha256") == outcome.get("attempt_payload_sha256"),
            "child launch attempt/outcome binding changed")
    identity = dict(outcome["process"])
    require(identity.get("pid") == process.pid, "child launch PID changed")
    receipt = _immutable_signed(Path(path), {
        "schema_version": CHILD_LAUNCH_SCHEMA,
        "status": "launched",
        **common,
        "process": identity,
        "attempt_receipt": queue.file_identity(
            _launch_evidence_path(path, "attempt")
        ),
        "outcome_receipt": queue.file_identity(
            _launch_evidence_path(path, "outcome")
        ),
    })
    return receipt


def _verify_child_launch(
    path: Path,
    *,
    launch_kind: str,
    study_commit: str,
    job_id: str,
) -> dict[str, Any]:
    attempt = _verify_child_launch_attempt(
        path,
        launch_kind=launch_kind,
        study_commit=study_commit,
        job_id=job_id,
    )
    outcome = _verify_child_launch_outcome(
        path,
        launch_kind=launch_kind,
        study_commit=study_commit,
        job_id=job_id,
    )
    require(outcome.get("process_started") is True,
            f"{launch_kind} launch receipt has a failed Popen outcome")
    value = _load_json(Path(path), f"{launch_kind} launch receipt")
    gpu.verify_signed_document(value, f"{launch_kind} launch receipt")
    expected = {
        "schema_version": CHILD_LAUNCH_SCHEMA,
        "status": "launched",
        "study_id": STUDY_ID,
        "study_commit": study_commit,
        "job_id": job_id,
        "launch_kind": launch_kind,
    }
    for key, wanted in expected.items():
        require(value.get(key) == wanted, f"{launch_kind} launch receipt changed: {key}")
    process = value.get("process")
    require(
        isinstance(process, Mapping)
        and type(process.get("pid")) is int and process["pid"] > 1
        and type(process.get("pgid")) is int and process["pgid"] > 1
        and isinstance(process.get("hostname"), str) and process["hostname"]
        and type(process.get("started_wall_time_ns")) is int
        and process["started_wall_time_ns"] > 0
        and type(process.get("started_monotonic_ns")) is int
        and process["started_monotonic_ns"] > 0,
        f"{launch_kind} launch process identity is invalid",
    )
    _verified_sha(value.get("command_sha256"), f"{launch_kind} launch command digest")
    require(isinstance(value.get("cwd"), str) and Path(value["cwd"]).is_absolute(),
            f"{launch_kind} launch cwd is invalid")
    require(value.get("attempt_receipt") == queue.file_identity(
        _launch_evidence_path(path, "attempt")
    ), f"{launch_kind} launch attempt identity changed")
    require(value.get("outcome_receipt") == queue.file_identity(
        _launch_evidence_path(path, "outcome")
    ), f"{launch_kind} launch outcome identity changed")
    require(attempt["payload_sha256"] == outcome["attempt_payload_sha256"],
            f"{launch_kind} launch attempt/outcome binding changed")
    require(dict(process) == dict(outcome["process"]),
            f"{launch_kind} launch process differs from Popen outcome")
    return value


def _launch_child(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    launch_receipt_path: Path,
    launch_kind: str,
    study_commit: str,
    job_id: str,
    termination_grace_seconds: float,
) -> tuple[subprocess.Popen[bytes], Any, Any, dict[str, Any]]:
    stdout = stdout_path.open("xb")
    stderr = stderr_path.open("xb")
    process: subprocess.Popen[bytes] | None = None
    attempt: dict[str, Any] | None = None
    try:
        attempt = _record_child_launch_attempt(
            launch_receipt_path,
            launch_kind=launch_kind,
            study_commit=study_commit,
            job_id=job_id,
            command=command,
            cwd=cwd,
        )
        try:
            process = subprocess.Popen(
                list(command), cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr,
            )
        except BaseException as error:
            _record_child_launch_outcome(
                launch_receipt_path,
                attempt=attempt,
                process=None,
                launch_kind=launch_kind,
                study_commit=study_commit,
                job_id=job_id,
                command=command,
                cwd=cwd,
                error=error,
            )
            raise
        outcome = _record_child_launch_outcome(
            launch_receipt_path,
            attempt=attempt,
            process=process,
            launch_kind=launch_kind,
            study_commit=study_commit,
            job_id=job_id,
            command=command,
            cwd=cwd,
        )
        launch = _record_child_launch(
            launch_receipt_path,
            process=process,
            launch_kind=launch_kind,
            study_commit=study_commit,
            job_id=job_id,
            command=command,
            cwd=cwd,
            attempt=attempt,
            outcome=outcome,
        )
    except BaseException as error:
        if process is not None:
            with contextlib.suppress(BaseException):
                _terminate_child(process, grace_seconds=termination_grace_seconds)
        stdout.close()
        stderr.close()
        if process is not None:
            raise ResourceQualificationError(
                "child launched but its durable launch receipt could not be written"
            ) from error
        raise
    return process, stdout, stderr, launch


def _finish_child_handles(stdout: Any, stderr: Any) -> None:
    for handle in (stdout, stderr):
        with contextlib.suppress(BaseException):
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(BaseException):
            handle.close()


def _terminate_child(process: subprocess.Popen[bytes], *, grace_seconds: float) -> dict[str, Any]:
    initial = process.poll()
    signal_sent = None
    killed = False
    if initial is None:
        process.terminate()
        signal_sent = "SIGTERM"
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            signal_sent = "SIGKILL"
            killed = True
            process.wait(timeout=5)
    else:
        process.wait()
    return {
        "initial_returncode": initial,
        "returncode": process.returncode,
        "signal_sent": signal_sent,
        "sigkill_required": killed,
        "reaped": process.returncode is not None,
    }


def _run_peer_monitored_child(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
    poll_interval_seconds: float,
    peer_failure_path: Path,
    study_commit: str,
    job_id: str,
    launch_receipt_path: Path,
    termination_receipt_path: Path,
    termination_grace_seconds: float,
) -> tuple[subprocess.CompletedProcess[Any], dict[str, Any]]:
    """Run one owned child while polling the signed D1 peer-failure marker."""

    require(math.isfinite(timeout_seconds) and timeout_seconds > 0,
            "monitored child timeout is invalid")
    require(math.isfinite(poll_interval_seconds) and 0 < poll_interval_seconds <= 1,
            "peer failure poll interval is invalid")
    process, stdout, stderr, launch = _launch_child(
        command,
        cwd=cwd,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        launch_receipt_path=launch_receipt_path,
        launch_kind="d1_probe",
        study_commit=study_commit,
        job_id=job_id,
        termination_grace_seconds=termination_grace_seconds,
    )
    deadline = time.monotonic() + timeout_seconds
    termination_reason: str | None = None
    observed_error: BaseException | None = None
    handles_closed = False
    try:
        require(
            launch["process"]["pgid"] == os.getpgrp(),
            "monitored D1 probe escaped the queue wrapper process group",
        )
        while True:
            returncode = process.poll()
            if returncode is not None:
                process.wait()
                _finish_child_handles(stdout, stderr)
                handles_closed = True
                return subprocess.CompletedProcess(list(command), returncode), launch
            try:
                _raise_if_d1_simulator_failed(
                    peer_failure_path, study_commit=study_commit
                )
            except BaseException as error:
                termination_reason = "simulator_peer_failure"
                observed_error = error
                break
            if time.monotonic() >= deadline:
                termination_reason = "probe_timeout"
                observed_error = ResourceQualificationError(
                    "D1 resource probe timed out"
                )
                break
            time.sleep(
                min(
                    poll_interval_seconds,
                    max(0.01, deadline - time.monotonic()),
                )
            )

        termination = _terminate_child(
            process, grace_seconds=termination_grace_seconds
        )
        _finish_child_handles(stdout, stderr)
        handles_closed = True
        require(termination.get("reaped") is True,
                "monitored D1 probe child was not reaped")
        termination_receipt = _immutable_signed(termination_receipt_path, {
            "schema_version": "wmf-development-resource-monitored-child-termination-v1",
            "status": "terminated_and_reaped",
            "study_id": STUDY_ID,
            "study_commit": study_commit,
            "job_id": job_id,
            "launch_kind": "d1_probe",
            "reason": termination_reason,
            "peer_failure_path": str(Path(peer_failure_path).resolve()),
            "launch_receipt": queue.file_identity(launch_receipt_path),
            "process": launch["process"],
            "same_process_group_as_wrapper": (
                launch["process"]["pgid"] == os.getpgrp()
            ),
            "termination": termination,
            "terminated_wall_time_ns": time.time_ns(),
        })
        require(termination_receipt.get("status") == "terminated_and_reaped",
                "monitored D1 probe termination receipt changed")
        assert observed_error is not None
        raise ResourceQualificationError(
            f"D1 monitored probe stopped for {termination_reason}; child reaped"
        ) from observed_error
    finally:
        if not handles_closed:
            if process.poll() is None:
                with contextlib.suppress(BaseException):
                    _terminate_child(
                        process, grace_seconds=termination_grace_seconds
                    )
            _finish_child_handles(stdout, stderr)


def _capture_release(source_root: Path) -> tuple[ModuleType, dict[str, Any], dict[str, Any]]:
    fixed = _load_module(
        Path(source_root) / FIXED_RELATIVE,
        f"wmf_resource_fixed_{os.getpid()}_{time.monotonic_ns()}",
    )
    capture, payload = fixed.load_json(FIXED_CAPTURE)
    require(fixed.sha256_bytes(payload) == FIXED_CAPTURE_SHA256, "fixed capture hash changed")
    require(
        capture.get("schema_version") == fixed.CAPTURE_SCHEMA
        and capture.get("status") == "passed"
        and capture.get("layout_pair_id") == "P00"
        and capture.get("layout_arm") == "original"
        and capture.get("command_task_used_for_reset") == "left"
        and capture.get("model_request_count") == 0
        and capture.get("behavioral_action_count") == 0,
        "fixed capture semantics changed",
    )
    gate = capture.get("gate_receipt")
    pose = capture.get("pose_manifest")
    require(isinstance(gate, Mapping) and isinstance(pose, Mapping),
            "fixed capture gate/pose descriptors are missing")
    release = fixed.verify_gate_and_pose_manifest(
        gate_receipt_path=Path(str(gate["path"])),
        gate_receipt_sha256=str(gate["sha256"]),
        pose_manifest_path=Path(str(pose["path"])),
        pose_manifest_sha256=str(pose["sha256"]),
        layout_pair_id="P00",
        candidate_id=str(capture["candidate_id"]),
    )
    return fixed, capture, release


def _simulator_child_environment(
    *, source_root: Path, job_dir: Path, logical_gpu: str
) -> dict[str, str]:
    fixed = _load_module(
        Path(source_root) / FIXED_RELATIVE,
        f"wmf_resource_fixed_env_{os.getpid()}_{time.monotonic_ns()}",
    )
    directories = fixed._runtime_directories(
        CONTROL_ROOT.resolve(), f"{socket.gethostname()}-{Path(job_dir).name}"
    )
    environment = fixed.build_child_environment(
        base=os.environ,
        directories=directories,
        forecast_root=Path(source_root) / LAYOUT_RELATIVE,
        source_root=source_root,
        robolab_root=ROBOLAB_ROOT,
    )
    environment["CUDA_VISIBLE_DEVICES"] = logical_gpu
    return environment


def _simulator_hold_command(
    *,
    source_root: Path,
    study_commit: str,
    output_dir: Path,
    ready_file: Path,
    stop_file: Path,
    model_job_id: str,
    simulator_job_id: str,
    timeout_seconds: int,
) -> list[str]:
    return [
        str(ROBOLAB_PYTHON),
        str(Path(source_root) / THIS_RELATIVE),
        "internal-simulator-hold",
        "--source-root", str(Path(source_root)),
        "--study-commit", study_commit,
        "--output-dir", str(output_dir),
        "--ready-file", str(ready_file),
        "--stop-file", str(stop_file),
        "--model-job-id", model_job_id,
        "--simulator-job-id", simulator_job_id,
        "--hold-timeout-seconds", str(timeout_seconds),
    ]


class _AccountingEnvProxy:
    """Delegate an environment while journaling every reset-settling step."""

    def __init__(
        self,
        target: Any,
        journal: _SimulatorAccountingJournal,
        *,
        settle_steps: int,
        stability_window_steps: int,
    ) -> None:
        self._target = target
        self._journal = journal
        self._settle_steps = settle_steps
        self._total_steps = settle_steps + stability_window_steps
        self.started = 0
        self.completed = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def step(self, action: Any) -> Any:
        ordinal = self.started
        require(ordinal < self._total_steps,
                "simulator settling issued more holds than frozen")
        phase = "settle" if ordinal < self._settle_steps else "stability"
        self._journal.append(
            "settling_hold_started",
            hold_ordinal=ordinal,
            settling_phase=phase,
        )
        self.started += 1
        try:
            result = self._target.step(action)
        except BaseException as error:
            with contextlib.suppress(BaseException):
                self._journal.append(
                    "settling_hold_failed",
                    hold_ordinal=ordinal,
                    settling_phase=phase,
                    error_type=type(error).__name__,
                )
            raise
        self._journal.append(
            "settling_hold_completed",
            hold_ordinal=ordinal,
            settling_phase=phase,
        )
        self.completed += 1
        return result


def run_internal_simulator_hold(args: argparse.Namespace) -> dict[str, Any]:
    """Construct and hold the exact P00 scene without a behavioral action."""

    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    ready_file = Path(args.ready_file).resolve()
    stop_file = Path(args.stop_file).resolve()
    require(not output_dir.exists() and not output_dir.is_symlink(),
            "simulator hold output already exists")
    output_dir.mkdir(parents=True)
    accounting_path = output_dir / "simulator_accounting.jsonl"
    accounting = _SimulatorAccountingJournal(
        accounting_path,
        study_commit=args.study_commit,
        model_job_id=args.model_job_id,
        simulator_job_id=args.simulator_job_id,
    )
    accounting.append("simulator_child_entered", pid=os.getpid())
    simulation_app = None
    env = None
    started_wall_ns = time.time_ns()
    try:
        fixed, capture, release = _capture_release(source_root)
        fixed.verify_clean_git(source_root, args.study_commit, "study")
        fixed.verify_clean_git(ROBOLAB_ROOT, fixed.ROBOLAB_COMMIT, "RoboLab")
        pose = capture["pose_manifest"]
        gate = capture["gate_receipt"]
        os.environ.update(
            WMF_FORECAST_POSE_MANIFEST=str(Path(str(pose["path"])).resolve()),
            WMF_FORECAST_POSE_MANIFEST_SHA256=str(pose["sha256"]),
            WMF_FORECAST_LAYOUT_PAIR_ID="P00",
        )
        forecast_root = source_root / LAYOUT_RELATIVE
        for path in (forecast_root, source_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        import cv2  # noqa: F401 -- required before Isaac imports.
        from isaaclab.app import AppLauncher

        launch_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(launch_parser)
        launch_args, _ = launch_parser.parse_known_args(["--headless"])
        launch_args.enable_cameras = True
        accounting.append("simulation_app_launch_started")
        launcher = AppLauncher(launch_args)
        simulation_app = launcher.app
        accounting.append("simulation_app_launch_completed")

        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        from fixture_tasks import settle_for_recording_reset

        require(Path(robolab.__file__).resolve().is_relative_to(ROBOLAB_ROOT),
                "effective RoboLab import escaped pinned source")
        native_output = output_dir / "native_simulator"
        native_output.mkdir()
        set_output_dir(str(native_output))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = False
        task_path = forecast_root / "task_files/original_left.py"
        auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        accounting.append("environment_creation_started")
        env, env_cfg = create_env(
            "WMFForecastOriginalLeftTask",
            device="cuda:0",
            seed=2026091000,
            num_envs=1,
            instruction_type="default",
            policy="wmf_resource_zero_behavioral_action",
            renderer="realtime",
            rendering_mode="balanced",
        )
        accounting.append("environment_creation_completed")
        require(not hasattr(env_cfg.terminations, "success"),
                "resource simulator retained success termination")
        require(hasattr(env_cfg.terminations, "time_out"),
                "resource simulator lacks timeout termination")
        terms = sorted(
            name for name, value in vars(env_cfg.terminations).items()
            if not name.startswith("_") and value is not None
        )
        require(terms == ["time_out"], "resource simulator termination set changed")
        require(float(env_cfg.episode_length_s) == 30.0, "resource simulator duration changed")
        source_contract, source_payload = fixed.load_json(forecast_root / "layout_source_contract.json")
        require(fixed.sha256_bytes(source_payload) == fixed.SOURCE_CONTRACT_SHA256,
                "layout source contract changed")
        settle_steps = int(source_contract["live_gate"]["settle_steps"])
        stability_steps = int(source_contract["live_gate"]["stability_window_steps"])
        accounting.append("physical_reset_started")
        try:
            observation, info = env.reset()
        except BaseException as error:
            with contextlib.suppress(BaseException):
                accounting.append("physical_reset_failed", error_type=type(error).__name__)
            raise
        accounting.append("physical_reset_completed")
        collision, visibility, physical = fixed._fresh_physical_callbacks(
            output_dir, source_contract
        )
        accounted_env = _AccountingEnvProxy(
            env,
            accounting,
            settle_steps=settle_steps,
            stability_window_steps=stability_steps,
        )
        observation, info, settled = settle_for_recording_reset(
            accounted_env,
            observation,
            info,
            pose_manifest_sha256=str(pose["sha256"]),
            reset_identity=f"resource-qualification:{args.study_commit}:{args.model_job_id}",
            collision_sampler=collision,
            visibility_sampler=visibility,
            settle_steps=settle_steps,
            stability_window_steps=stability_steps,
            linear_speed_tolerance_m_s=float(source_contract["live_gate"]["linear_speed_tolerance_m_s"]),
            angular_speed_tolerance_rad_s=float(source_contract["live_gate"]["angular_speed_tolerance_rad_s"]),
        )
        require(
            accounted_env.started == accounted_env.completed == settle_steps + stability_steps,
            "simulator settling accounting differs from frozen hold count",
        )
        accounting.append(
            "settling_validation_completed",
            settling_hold_action_count=accounted_env.completed,
        )
        native = fixed.capture_native_clock(env, env_cfg, observation)
        require(native["behavioral_episode_step"] == 0,
                "zero-action simulator behavioral counter is nonzero")
        require(settled.get("left_success") is False and settled.get("right_success") is False,
                "zero-action simulator reset is already successful")
        require(
            physical.get("collision", {}).get("passed") is True
            and physical.get("visibility", {}).get("passed") is True,
            "zero-action simulator physical checks failed",
        )
        ready_wall_ns = time.time_ns()
        accounting.append("simulator_ready_persist_started")
        accounting_before_ready = _summarize_simulator_accounting(
            accounting_path,
            study_commit=args.study_commit,
            model_job_id=args.model_job_id,
            simulator_job_id=args.simulator_job_id,
        )
        require(
            accounting_before_ready["physical_resets"] == 1
            and accounting_before_ready["settling_hold_actions"]
            == settle_steps + stability_steps,
            "simulator readiness accounting is incomplete",
        )
        common = {
            "schema_version": SHARED_SCHEMA,
            "kind": "simulator_ready",
            "study_id": STUDY_ID,
            "study_commit": args.study_commit,
            "session_id": D1_SESSION_ID if args.model_job_id == D1_MODEL_JOB_ID else N3_JOB_ID,
            "model_job_id": args.model_job_id,
            "simulator_job_id": args.simulator_job_id,
            "hostname": socket.gethostname(),
            "pod_uid": os.environ.get("POD_UID"),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started_wall_time_ns": started_wall_ns,
            "ready_wall_time_ns": ready_wall_ns,
            "candidate_id": release["candidate_id"],
            "gate_receipt_sha256": gate["sha256"],
            "pose_manifest_sha256": pose["sha256"],
            "settled_reset": settled,
            "native_clock": native,
            "settling_hold_action_count": int(
                settled["settle_evidence"]["settle_steps"]
                + settled["settle_evidence"]["stability_window_steps"]
            ),
            "physical_reset_count": 1,
            "simulator_accounting_tail_before_ready": accounting_before_ready[
                "tail_event_sha256"
            ],
            "model_request_count": 0,
            "robot_episode_count": 0,
            "behavioral_action_count": 0,
        }
        ready = _immutable_signed(output_dir / "simulator_ready.json", common)
        queue.immutable_json(ready_file, ready, maximum_bytes=2 * 1024 * 1024)
        accounting.append("simulator_ready_persist_completed")
        _wait_for_file(stop_file, timeout_seconds=float(args.hold_timeout_seconds))
        stop = _load_json(stop_file, "simulator peer stop")
        gpu.verify_signed_document(stop, "simulator peer stop")
        require(stop.get("schema_version") == SHARED_SCHEMA and stop.get("kind") == "model_done",
                "simulator peer stop changed")
        require(stop.get("study_commit") == args.study_commit, "simulator peer stop commit changed")
        require(stop.get("model_job_id") == args.model_job_id, "simulator peer model job changed")
        require(stop.get("simulator_job_id") == args.simulator_job_id,
                "simulator peer simulator job changed")
        finished_wall_ns = time.time_ns()
        final_accounting = _summarize_simulator_accounting(
            accounting_path,
            study_commit=args.study_commit,
            model_job_id=args.model_job_id,
            simulator_job_id=args.simulator_job_id,
        )
        require(
            final_accounting["physical_resets"] == 1
            and final_accounting["settling_hold_actions"]
            == common["settling_hold_action_count"],
            "final simulator accounting differs from ready receipt",
        )
        receipt = gpu.signed_document({
            "schema_version": "wmf-development-resource-zero-action-simulator-v1",
            "status": "passed",
            "study_id": STUDY_ID,
            "study_commit": args.study_commit,
            "model_job_id": args.model_job_id,
            "simulator_job_id": args.simulator_job_id,
            "ready_receipt": queue.file_identity(output_dir / "simulator_ready.json"),
            "shared_ready_receipt": queue.file_identity(ready_file),
            "peer_stop_receipt": queue.file_identity(stop_file),
            "started_wall_time_ns": started_wall_ns,
            "ready_wall_time_ns": ready_wall_ns,
            "finished_wall_time_ns": finished_wall_ns,
            "physical_reset_count": 1,
            "settling_hold_action_count": common["settling_hold_action_count"],
            "simulator_accounting": final_accounting,
            "model_request_count": 0,
            "robot_episode_count": 0,
            "behavioral_action_count": 0,
            "claim_boundary": (
                "Settled P00 simulator residency and rendering only; settling holds are not "
                "behavioral actions and no policy request was issued."
            ),
        })
        queue.immutable_json(
            output_dir / "simulator_hold_receipt.json", receipt, maximum_bytes=2 * 1024 * 1024
        )
        return receipt
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def _internal_simulator_main(args: argparse.Namespace) -> int:
    try:
        receipt = run_internal_simulator_hold(args)
        print(json.dumps({
            "status": receipt["status"],
            "model_request_count": 0,
            "robot_episode_count": 0,
            "behavioral_action_count": 0,
        }, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        try:
            phase_accounting = _summarize_simulator_accounting(
                output / "simulator_accounting.jsonl",
                study_commit=args.study_commit,
                model_job_id=args.model_job_id,
                simulator_job_id=args.simulator_job_id,
            )
        except BaseException as accounting_error:
            phase_accounting = {
                "status": "unavailable_or_invalid",
                "error_type": type(accounting_error).__name__,
                "physical_resets": None,
                "physical_resets_lower_bound": 0,
                "physical_resets_upper_bound": 1,
                "settling_hold_actions": None,
                "settling_hold_actions_lower_bound": 0,
                "settling_hold_actions_upper_bound": None,
                "reset_completion_ambiguous": True,
                "settling_completion_ambiguous": True,
            }
        with contextlib.suppress(BaseException):
            _immutable_signed(output / "technical_failure.json", {
                "schema_version": "wmf-development-resource-zero-action-simulator-failure-v1",
                "status": "technical_invalid",
                "error_type": type(error).__name__,
                "detail": str(error)[:2000],
                "traceback": traceback.format_exc(limit=30)[-16000:],
                "phase_accounting": phase_accounting,
                "physical_reset_count": phase_accounting["physical_resets"],
                "physical_reset_count_bounds": {
                    "lower": phase_accounting["physical_resets_lower_bound"],
                    "upper": phase_accounting["physical_resets_upper_bound"],
                },
                "settling_hold_action_count": phase_accounting["settling_hold_actions"],
                "settling_hold_action_count_bounds": {
                    "lower": phase_accounting["settling_hold_actions_lower_bound"],
                    "upper": phase_accounting["settling_hold_actions_upper_bound"],
                },
                "model_request_count": 0,
                "robot_episode_count": 0,
                "behavioral_action_count": 0,
            })
        traceback.print_exc()
        return 3


def _simulator_science_counts(settling_hold_actions: int) -> dict[str, int]:
    require(type(settling_hold_actions) is int and settling_hold_actions > 0,
            "simulator settling-hold count is invalid")
    return {
        "model_runtime_loads": 0,
        "model_servers_started": 0,
        "model_requests_issued": 0,
        "model_requests_completed": 0,
        "simulator_processes_started": 1,
        "physical_resets": 1,
        "settling_hold_actions": settling_hold_actions,
        **_zero_behavior_counts(),
    }


def run_d1_simulator_profile(
    *, context: Any, contract: Mapping[str, Any], implementation: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    raw, publish = queue._prepare_output_directories(context.job_dir)
    runtime = contract["runtime"]
    preflight = _preflight_gpu_count(1, runtime["required_gpu_name"])
    session = _session_root(context.study_commit)
    require(not session.exists() and not session.is_symlink(), "D1 resource session already exists")
    session.parent.mkdir(parents=True, exist_ok=True)
    session.mkdir()
    ready_file = session / "simulator_ready.json"
    stop_file = session / "model_done.json"
    output = raw / "simulator_hold"
    command = _simulator_hold_command(
        source_root=context.source_root,
        study_commit=context.study_commit,
        output_dir=output,
        ready_file=ready_file,
        stop_file=stop_file,
        model_job_id=D1_MODEL_JOB_ID,
        simulator_job_id=D1_SIM_JOB_ID,
        timeout_seconds=int(runtime["simulator_hold_timeout_seconds"]),
    )
    env = _simulator_child_environment(
        source_root=context.source_root, job_dir=context.job_dir, logical_gpu="0"
    )
    process = None
    stdout = stderr = None
    sampler = None
    started_wall_ns = time.time_ns()
    try:
        process, stdout, stderr, process_launch = _launch_child(
            command,
            cwd=ROBOLAB_ROOT,
            env=env,
            stdout_path=raw / "simulator.stdout.log",
            stderr_path=raw / "simulator.stderr.log",
            launch_receipt_path=raw / "simulator.launch.json",
            launch_kind="simulator",
            study_commit=context.study_commit,
            job_id=context.job_id,
            termination_grace_seconds=float(runtime["child_termination_grace_seconds"]),
        )
        process_identity = process_launch["process"]
        sampler = gpu.GpuSampler(
            raw / "gpu_samples.jsonl",
            interval_seconds=float(runtime["nvidia_smi_sample_interval_seconds"]),
        )
        sampler.set_root("simulator", process.pid)
        sampler.set_phase("simulator_starting")
        sampler.start()
        _wait_for_file(
            ready_file,
            timeout_seconds=float(runtime["peer_start_timeout_seconds"]),
            child=process,
        )
        ready = _verify_shared(ready_file, kind="simulator_ready", study_commit=context.study_commit)
        started_file = session / "model_started.json"
        _wait_for_file(
            started_file,
            timeout_seconds=float(runtime["peer_start_timeout_seconds"]),
            child=process,
        )
        started = _verify_shared(
            started_file, kind="model_started", study_commit=context.study_commit
        )
        local_model_started_observed_wall_ns = time.time_ns()
        sampler.set_phase("holding_model_runtime")
        _wait_for_file(
            stop_file,
            timeout_seconds=float(runtime["simulator_hold_timeout_seconds"]),
            child=process,
        )
        local_model_done_observed_wall_ns = time.time_ns()
        try:
            process.wait(timeout=600)
        except subprocess.TimeoutExpired as error:
            raise ResourceQualificationError(
                "D1 simulator child did not exit after peer stop"
            ) from error
        finished_wall_ns = time.time_ns()
        _finish_child_handles(stdout, stderr)
        stdout = stderr = None
        sample_run = sampler.stop()
        sampler = None
        require(process.returncode == 0, "D1 zero-action simulator child failed")
        child_receipt_path = output / "simulator_hold_receipt.json"
        child_receipt = _load_json(child_receipt_path, "D1 simulator hold receipt")
        gpu.verify_signed_document(child_receipt, "D1 simulator hold receipt")
        require(child_receipt.get("status") == "passed", "D1 simulator hold did not pass")
        child_accounting = _summarize_simulator_accounting(
            output / "simulator_accounting.jsonl",
            study_commit=context.study_commit,
            model_job_id=D1_MODEL_JOB_ID,
            simulator_job_id=D1_SIM_JOB_ID,
        )
        require(
            child_receipt.get("simulator_accounting") == child_accounting
            and child_accounting["physical_resets"] == 1
            and type(child_accounting["settling_hold_actions"]) is int
            and child_accounting["settling_hold_actions"] > 0,
            "D1 simulator accounting does not reproduce",
        )
        done = _verify_shared(stop_file, kind="model_done", study_commit=context.study_commit)
        require(done.get("model_status") == "passed", "D1 peer model did not pass")
        intervals = {
            "simulator_lifetime": (started_wall_ns, finished_wall_ns),
            "simultaneous_model_runtime": (
                local_model_started_observed_wall_ns,
                local_model_done_observed_wall_ns,
            ),
        }
        summary = gpu.summarize_samples(
            Path(sample_run["artifact"]["path"]),
            expected_gpu_count=1,
            required_gpu_name=runtime["required_gpu_name"],
            maximum_gap_seconds=float(runtime["maximum_accepted_sample_gap_seconds"]),
            intervals=intervals,
            minimum_samples_by_interval={"simultaneous_model_runtime": int(runtime["minimum_simultaneous_phase_samples"])},
        )
        summary_path = raw / "gpu_summary.json"
        queue.immutable_json(summary_path, summary, maximum_bytes=2 * 1024 * 1024)
        receipt = gpu.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "status": "passed",
            "decision": "machine_measurement_passed_confirmation_held",
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": "d1-simulator-profile",
            "job_id": context.job_id,
            "study_commit": context.study_commit,
            "queue_role": context.role,
            "worker_id": context.worker_id,
            "runtime_identity": {"hostname": context.hostname, "pod_uid": context.pod_uid, "pid": os.getpid()},
            "queue_descriptor": context.descriptor_identity,
            "queue_claim": context.claim_identity,
            "implementation": implementation,
            "prerequisites": prerequisites,
            "preflight": preflight,
            "process": process_identity,
            "process_launch": queue.file_identity(raw / "simulator.launch.json"),
            "child": {"argv": command, "returncode": process.returncode, "reaped": True},
            "child_receipt": queue.file_identity(child_receipt_path),
            "peer_handshake": {
                "session_root": str(session),
                "simulator_ready": queue.file_identity(ready_file),
                "model_started": queue.file_identity(session / "model_started.json"),
                "model_done": queue.file_identity(stop_file),
                "simulator_local_model_started_observed_wall_time_ns": local_model_started_observed_wall_ns,
                "simulator_local_model_done_observed_wall_time_ns": local_model_done_observed_wall_ns,
            },
            "gpu_samples": sample_run,
            "gpu_summary": queue.file_identity(summary_path),
            "science_counts": _simulator_science_counts(
                int(child_receipt["settling_hold_action_count"])
            ),
            "machine_resource_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "claim_boundary": (
                "One settled zero-behavioral-action simulator held during D1 model runtime. "
                "This is machine capacity evidence, not a policy episode or annotation."
            ),
        })
        queue.immutable_json(
            publish / "development_resource_qualification_job_receipt.json",
            receipt,
            maximum_bytes=4 * 1024 * 1024,
        )
        return receipt
    finally:
        if sampler is not None:
            with contextlib.suppress(BaseException):
                sampler.stop()
        if process is not None and process.poll() is None:
            with contextlib.suppress(BaseException):
                _terminate_child(process, grace_seconds=float(runtime["child_termination_grace_seconds"]))
        if stdout is not None and stderr is not None:
            _finish_child_handles(stdout, stderr)


def _n3_child_environment(job_dir: Path, source_root: Path) -> dict[str, str]:
    runtime_root = Path(job_dir) / "raw" / "n3_runtime"
    torchinductor = runtime_root / "torchinductor"
    temporary = runtime_root / "tmp"
    torchinductor.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="0",
        DS_IGNORE_CUDA_DETECTION="1",
        HF_HOME=str(N3_HF_HOME),
        PATH=f"{N3_COMPAT_BIN}:{N3_PYTHON.parent}:{SYSTEM_PATH}",
        LD_LIBRARY_PATH=N3_LD_LIBRARY_PATH,
        PYTHONPATH=f"{N3_SOURCE}:{Path(source_root)}:{environment.get('PYTHONPATH', '')}",
        TORCHINDUCTOR_CACHE_DIR=str(torchinductor),
        TMPDIR=str(temporary),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
    )
    return environment


def _n3_command(source_root: Path, raw: Path) -> list[str]:
    return [
        str(N3_PYTHON),
        str(Path(source_root) / N3_RELATIVE),
        "--source-root", str(N3_SOURCE),
        "--checkpoint-root", str(N3_CHECKPOINT),
        "--observation-manifest", str(N3_OBSERVATION_MANIFEST),
        "--output-dir", str(raw / "n3_profile"),
        "--publish-dir", str(raw / "n3_profile_publish"),
        "--effective-seed", "2026091000",
    ]


def _n3_journal(path: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]]]:
    require(path.is_file() and not path.is_symlink(), "N3 profile journal is missing")
    events: list[dict[str, Any]] = []
    previous = None
    for sequence, line in enumerate(path.read_bytes().splitlines()):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResourceQualificationError("N3 profile journal is invalid JSON") from error
        require(isinstance(row, dict) and row.get("sequence") == sequence,
                "N3 profile journal sequence changed")
        require(row.get("previous_event_sha256") == previous, "N3 profile journal chain changed")
        observed = row.get("event_sha256")
        unsigned = dict(row)
        unsigned.pop("event_sha256", None)
        require(
            isinstance(observed, str)
            and hashlib.sha256(
                json.dumps(
                    unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest() == observed,
            "N3 profile journal signature changed",
        )
        previous = observed
        events.append(row)
    require(bool(events), "N3 profile journal is empty")

    def unique(kind: str) -> dict[str, Any]:
        rows = [row for row in events if row.get("kind") == kind]
        require(len(rows) == 1, f"N3 profile journal {kind} count changed")
        return rows[0]

    gpu_ready = unique("gpu_validated")
    loaded = unique("official_model_loaded")
    starts = [row for row in events if row.get("kind") == "generation_request_started"]
    finishes = [row for row in events if row.get("kind") == "generation_request_completed"]
    require(len(starts) == len(finishes) == 6, "N3 profile request journal count changed")
    intervals = {
        "model_load": (int(gpu_ready["wall_time_ns"]), int(loaded["wall_time_ns"])),
        "model_inference": (int(starts[0]["wall_time_ns"]), int(finishes[-1]["wall_time_ns"])),
    }
    require(all(end > start for start, end in intervals.values()),
            "N3 model phase interval is invalid")
    return events, intervals


def _n3_allocator_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    requests = receipt.get("requests")
    require(isinstance(requests, list) and len(requests) == 6,
            "N3 allocator request inventory changed")
    peak_allocated = []
    peak_reserved = []
    for index, row in enumerate(requests):
        require(isinstance(row, Mapping) and row.get("request_index") == index,
                "N3 allocator request order changed")
        memory = row.get("gpu_memory")
        require(isinstance(memory, Mapping), "N3 allocator memory record is missing")
        after = memory.get("after_request")
        require(isinstance(after, Mapping), "N3 allocator after-request record is missing")
        allocated = after.get("peak_allocated_bytes")
        reserved = after.get("reserved_bytes")
        require(type(allocated) is int and allocated > 0, "N3 allocator peak is invalid")
        require(type(reserved) is int and reserved >= allocated, "N3 allocator reserved peak is invalid")
        peak_allocated.append(allocated)
        peak_reserved.append(reserved)
    return {
        "scope": "exact six-request fixed-input N3 runtime on logical GPU 0",
        "request_count": 6,
        "peak_allocated_bytes_max": max(peak_allocated),
        "reserved_bytes_after_request_max": max(peak_reserved),
        "peak_reserved_bytes_max": None,
        "peak_reserved_bytes_status": "not_exposed_by_pinned_n3_wrapper",
        "measurement_boundary": "PyTorch allocator only; nvidia-smi whole-process/device peaks are separate",
    }


def _model_science_counts(*, model: str, settling_hold_actions: int = 0) -> dict[str, int]:
    require(type(settling_hold_actions) is int and settling_hold_actions >= 0,
            "model settling-hold count is invalid")
    return {
        "model_runtime_loads": 1,
        "model_servers_started": 0 if model == "N3" else 1,
        "model_requests_issued": 6,
        "model_requests_completed": 6,
        "simulator_processes_started": 1 if model == "N3" else 0,
        "physical_resets": 1 if model == "N3" else 0,
        "settling_hold_actions": settling_hold_actions if model == "N3" else 0,
        **_zero_behavior_counts(),
    }


def run_n3_profile(
    *, context: Any, contract: Mapping[str, Any], implementation: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    raw, publish = queue._prepare_output_directories(context.job_dir)
    runtime = contract["runtime"]
    preflight = _preflight_gpu_count(2, runtime["required_gpu_name"])
    session = raw / "simulator_session"
    session.mkdir()
    ready_file = session / "simulator_ready.json"
    stop_file = session / "model_done.json"
    simulator_output = raw / "simulator_hold"
    sim_command = _simulator_hold_command(
        source_root=context.source_root,
        study_commit=context.study_commit,
        output_dir=simulator_output,
        ready_file=ready_file,
        stop_file=stop_file,
        model_job_id=N3_JOB_ID,
        simulator_job_id=N3_JOB_ID,
        timeout_seconds=int(runtime["simulator_hold_timeout_seconds"]),
    )
    sim_env = _simulator_child_environment(
        source_root=context.source_root, job_dir=context.job_dir, logical_gpu="1"
    )
    model_command = _n3_command(context.source_root, raw)
    model_env = _n3_child_environment(context.job_dir, context.source_root)
    sim_process = model_process = None
    sim_stdout = sim_stderr = model_stdout = model_stderr = None
    sampler = None
    sim_started_wall_ns = time.time_ns()
    model_started_wall_ns = model_finished_wall_ns = None
    try:
        sim_process, sim_stdout, sim_stderr, sim_launch = _launch_child(
            sim_command,
            cwd=ROBOLAB_ROOT,
            env=sim_env,
            stdout_path=raw / "simulator.stdout.log",
            stderr_path=raw / "simulator.stderr.log",
            launch_receipt_path=raw / "simulator.launch.json",
            launch_kind="simulator",
            study_commit=context.study_commit,
            job_id=context.job_id,
            termination_grace_seconds=float(runtime["child_termination_grace_seconds"]),
        )
        sim_identity = sim_launch["process"]
        sampler = gpu.GpuSampler(
            raw / "gpu_samples.jsonl",
            interval_seconds=float(runtime["nvidia_smi_sample_interval_seconds"]),
        )
        sampler.set_root("simulator", sim_process.pid)
        sampler.set_phase("simulator_starting")
        sampler.start()
        _wait_for_file(
            ready_file,
            timeout_seconds=float(runtime["peer_start_timeout_seconds"]),
            child=sim_process,
        )
        ready = _load_json(ready_file, "N3 simulator ready")
        gpu.verify_signed_document(ready, "N3 simulator ready")
        require(
            ready.get("kind") == "simulator_ready"
            and ready.get("study_commit") == context.study_commit
            and ready.get("model_job_id") == N3_JOB_ID
            and ready.get("simulator_job_id") == N3_JOB_ID,
            "N3 simulator readiness changed",
        )
        sampler.set_phase("model_running_with_simulator")
        model_started_wall_ns = time.time_ns()
        model_process, model_stdout, model_stderr, model_launch = _launch_child(
            model_command,
            cwd=N3_SOURCE,
            env=model_env,
            stdout_path=raw / "n3.stdout.log",
            stderr_path=raw / "n3.stderr.log",
            launch_receipt_path=raw / "n3_model.launch.json",
            launch_kind="n3_model",
            study_commit=context.study_commit,
            job_id=context.job_id,
            termination_grace_seconds=float(runtime["child_termination_grace_seconds"]),
        )
        model_identity = model_launch["process"]
        sampler.set_root("model", model_process.pid)
        try:
            model_process.wait(timeout=float(MAX_WALL_SECONDS - 600))
        except subprocess.TimeoutExpired as error:
            raise ResourceQualificationError("N3 resource model child timed out") from error
        model_finished_wall_ns = time.time_ns()
        _finish_child_handles(model_stdout, model_stderr)
        model_stdout = model_stderr = None
        require(model_process.returncode == 0, "N3 resource model child failed")
        stop = _immutable_signed(stop_file, {
            "schema_version": SHARED_SCHEMA,
            "kind": "model_done",
            "study_id": STUDY_ID,
            "study_commit": context.study_commit,
            "session_id": N3_JOB_ID,
            "model_job_id": N3_JOB_ID,
            "simulator_job_id": N3_JOB_ID,
            "model_status": "passed",
            "model_started_wall_time_ns": model_started_wall_ns,
            "model_finished_wall_time_ns": model_finished_wall_ns,
        })
        del stop
        try:
            sim_process.wait(timeout=600)
        except subprocess.TimeoutExpired as error:
            raise ResourceQualificationError("N3 resource simulator child did not exit") from error
        sim_finished_wall_ns = time.time_ns()
        _finish_child_handles(sim_stdout, sim_stderr)
        sim_stdout = sim_stderr = None
        require(sim_process.returncode == 0, "N3 resource simulator child failed")
        sample_run = sampler.stop()
        sampler = None

        compact_path = raw / "n3_profile_publish/n3_qualification.json"
        raw_receipt_path = raw / "n3_profile/n3_qualification.json"
        compact = _load_json(compact_path, "N3 resource qualification")
        full = _load_json(raw_receipt_path, "N3 resource raw qualification")
        require(
            compact.get("status") == "passed"
            and compact.get("qualified") is True
            and compact.get("generation_request_count") == 6
            and compact.get("robot_episode_count") == 0,
            "N3 resource qualification did not pass",
        )
        require(full.get("requests") and full.get("generation_request_count") == 6,
                "N3 resource raw qualification changed")
        _, phase_intervals = _n3_journal(raw / "n3_profile/events.partial.jsonl")
        phase_intervals.update({
            "simulator_lifetime": (sim_started_wall_ns, sim_finished_wall_ns),
            "simultaneous_model_runtime": (model_started_wall_ns, model_finished_wall_ns),
        })
        summary = gpu.summarize_samples(
            Path(sample_run["artifact"]["path"]),
            expected_gpu_count=2,
            required_gpu_name=runtime["required_gpu_name"],
            maximum_gap_seconds=float(runtime["maximum_accepted_sample_gap_seconds"]),
            intervals=phase_intervals,
            minimum_samples_by_interval={
                "model_inference": int(runtime["minimum_inference_phase_samples"]),
                "simultaneous_model_runtime": int(runtime["minimum_simultaneous_phase_samples"]),
            },
        )
        summary_path = raw / "gpu_summary.json"
        queue.immutable_json(summary_path, summary, maximum_bytes=2 * 1024 * 1024)
        sim_receipt_path = simulator_output / "simulator_hold_receipt.json"
        sim_receipt = _load_json(sim_receipt_path, "N3 simulator hold receipt")
        gpu.verify_signed_document(sim_receipt, "N3 simulator hold receipt")
        require(sim_receipt.get("status") == "passed", "N3 simulator hold did not pass")
        sim_accounting = _summarize_simulator_accounting(
            simulator_output / "simulator_accounting.jsonl",
            study_commit=context.study_commit,
            model_job_id=N3_JOB_ID,
            simulator_job_id=N3_JOB_ID,
        )
        require(
            sim_receipt.get("simulator_accounting") == sim_accounting
            and sim_accounting["physical_resets"] == 1
            and sim_accounting["settling_hold_actions"]
            == sim_receipt.get("settling_hold_action_count"),
            "N3 simulator accounting does not reproduce",
        )
        receipt = gpu.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "status": "passed",
            "decision": "machine_measurement_passed_confirmation_held",
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": "n3-profile",
            "model_id": "N3",
            "job_id": context.job_id,
            "study_commit": context.study_commit,
            "queue_role": context.role,
            "worker_id": context.worker_id,
            "runtime_identity": {"hostname": context.hostname, "pod_uid": context.pod_uid, "pid": os.getpid()},
            "queue_descriptor": context.descriptor_identity,
            "queue_claim": context.claim_identity,
            "implementation": implementation,
            "prerequisites": prerequisites,
            "preflight": preflight,
            "processes": {"model": model_identity, "simulator": sim_identity},
            "process_launches": {
                "model": queue.file_identity(raw / "n3_model.launch.json"),
                "simulator": queue.file_identity(raw / "simulator.launch.json"),
            },
            "model_output": {
                "qualification": queue.file_identity(compact_path),
                "raw_qualification": queue.file_identity(raw_receipt_path),
                "event_journal": queue.file_identity(raw / "n3_profile/events.partial.jsonl"),
                "allocator": _n3_allocator_summary(full),
            },
            "simulator_output": queue.file_identity(sim_receipt_path),
            "gpu_samples": sample_run,
            "gpu_summary": queue.file_identity(summary_path),
            "science_counts": _model_science_counts(
                model="N3",
                settling_hold_actions=int(sim_receipt["settling_hold_action_count"]),
            ),
            "machine_resource_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "claim_boundary": (
                "Six new fixed-input N3 generation requests plus one settled zero-behavioral-action "
                "simulator. No retained behavioral cell was rerun or relabelled."
            ),
        })
        queue.immutable_json(
            publish / "development_resource_qualification_job_receipt.json",
            receipt,
            maximum_bytes=4 * 1024 * 1024,
        )
        return receipt
    finally:
        if model_started_wall_ns is not None and not stop_file.exists():
            with contextlib.suppress(BaseException):
                _immutable_signed(stop_file, {
                    "schema_version": SHARED_SCHEMA,
                    "kind": "model_done",
                    "study_id": STUDY_ID,
                    "study_commit": context.study_commit,
                    "session_id": N3_JOB_ID,
                    "model_job_id": N3_JOB_ID,
                    "simulator_job_id": N3_JOB_ID,
                    "model_status": "technical_invalid",
                    "model_started_wall_time_ns": model_started_wall_ns,
                    "model_finished_wall_time_ns": time.time_ns(),
                })
        if sampler is not None:
            with contextlib.suppress(BaseException):
                sampler.stop()
        for process in (model_process, sim_process):
            if process is not None and process.poll() is None:
                with contextlib.suppress(BaseException):
                    _terminate_child(process, grace_seconds=float(runtime["child_termination_grace_seconds"]))
        for handles in ((model_stdout, model_stderr), (sim_stdout, sim_stderr)):
            if handles[0] is not None and handles[1] is not None:
                _finish_child_handles(handles[0], handles[1])


def _d1_allocator_summary(future_root: Path) -> dict[str, Any]:
    episodes = Path(future_root) / "episodes"
    require(episodes.is_dir() and not episodes.is_symlink(), "D1 resource episode directory is missing")
    manifests = sorted(episodes.glob("*/episode_manifest.json"))
    require(len(manifests) == 6, "D1 resource episode manifest count changed")
    peak_allocated: dict[int, list[int]] = {0: [], 1: []}
    peak_reserved: dict[int, list[int]] = {0: [], 1: []}
    requests = 0
    for manifest_path in manifests:
        manifest = _load_json(manifest_path, "D1 resource episode manifest")
        rows = manifest.get("requests")
        require(manifest.get("status") == "complete" and isinstance(rows, list) and len(rows) == 1,
                "D1 resource episode manifest changed")
        requests += 1
        metrics = rows[0].get("temporal_and_cache_rank_metrics")
        require(isinstance(metrics, list) and len(metrics) == 2,
                "D1 resource rank metric inventory changed")
        for metric in metrics:
            rank = metric.get("rank")
            memory = metric.get("cuda_memory")
            require(rank in (0, 1) and isinstance(memory, Mapping),
                    "D1 resource allocator record changed")
            allocated = memory.get("peak_allocated_bytes")
            reserved = memory.get("peak_reserved_bytes")
            require(type(allocated) is int and allocated > 0,
                    "D1 resource allocator allocated peak is invalid")
            require(type(reserved) is int and reserved >= allocated,
                    "D1 resource allocator reserved peak is invalid")
            peak_allocated[int(rank)].append(allocated)
            peak_reserved[int(rank)].append(reserved)
    require(requests == 6 and all(len(values) == 6 for values in peak_allocated.values()),
            "D1 resource allocator coverage changed")
    return {
        "scope": "exact six-request official D1 fixed-input runtime across two distributed ranks",
        "request_count": 6,
        "ranks": [
            {
                "rank": rank,
                "peak_allocated_bytes_max": max(peak_allocated[rank]),
                "peak_reserved_bytes_max": max(peak_reserved[rank]),
            }
            for rank in (0, 1)
        ],
        "measurement_boundary": "PyTorch allocator only; nvidia-smi whole-process/device peaks are separate",
    }


def run_d1_model_profile(
    *, context: Any, contract: Mapping[str, Any], implementation: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    raw, publish = queue._prepare_output_directories(context.job_dir)
    runtime_contract = contract["runtime"]
    session = _session_root(context.study_commit)
    ready_file = session / "simulator_ready.json"
    simulator_failed_file = session / "simulator_failed.json"
    started_file = session / "model_started.json"
    done_file = session / "model_done.json"
    _wait_for_d1_simulator_ready(
        ready_file,
        simulator_failed_file,
        study_commit=context.study_commit,
        timeout_seconds=float(runtime_contract["peer_start_timeout_seconds"]),
    )
    ready = _verify_shared(ready_file, kind="simulator_ready", study_commit=context.study_commit)
    require(ready.get("model_request_count") == 0 and ready.get("behavioral_action_count") == 0,
            "D1 simulator peer claims scientific work")

    d1 = _load_module(
        context.source_root / D1_JOB_RELATIVE,
        f"wmf_resource_d1_{context.job_id.replace('-', '_')}",
    )
    d1.validate_runtime_paths(d1.RuntimePaths(), enforce_pinned_locations=True)
    d1_env = d1._child_environment(context.job_dir)
    capture = _load_json(FIXED_CAPTURE, "D1 fixed capture")
    fixture_descriptor = capture.get("model_fixtures", {}).get("D1", {}).get("fixture")
    require(isinstance(fixture_descriptor, Mapping), "D1 fixed fixture descriptor is missing")
    fixture_path = Path(str(fixture_descriptor.get("path")))
    fixture_sha = _verified_sha(fixture_descriptor.get("sha256"), "D1 fixture digest")
    capture_validation = d1.validate_capture_manifest(
        FIXED_CAPTURE,
        FIXED_CAPTURE_SHA256,
        artifact_root=FIXED_CAPTURE.parent,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha,
    )
    fixture_copy = raw / "fixed_observation/d1_fixed_observation.npz"
    fixture_snapshot = d1.snapshot_regular_file(fixture_path, fixture_copy, fixture_sha)
    validation = d1.run_internal_helper(
        python=d1.D1_PYTHON,
        module_path=context.source_root / D1_JOB_RELATIVE,
        mode="internal-validate-fixture",
        output_path=raw / "fixture_validation.json",
        cwd=context.source_root,
        env=d1_env,
        timeout=300,
        extra=("--fixture", str(fixture_copy), "--fixture-sha256", fixture_sha),
    )
    require(validation.get("array_count") == 6, "D1 fixture validation count changed")
    topology = d1.verify_gpu_topology(
        runtime=d1.RuntimePaths(),
        module_path=context.source_root / D1_JOB_RELATIVE,
        raw_dir=raw,
        cwd=context.source_root,
        env=d1_env,
        nvidia_smi="nvidia-smi",
    )
    preflight = _preflight_gpu_count(2, runtime_contract["required_gpu_name"])

    future_root = raw / "d1_future"
    probe_output = raw / "d1_probe"
    port = d1.DEFAULT_PORT
    d1.assert_port_available(port)
    _raise_if_d1_simulator_failed(
        simulator_failed_file, study_commit=context.study_commit
    )
    server_command = d1.build_server_command(
        runtime=d1.RuntimePaths(),
        source_root=context.source_root,
        future_root=future_root,
        port=port,
        timeout_seconds=48000,
    )
    server = None
    server_stdout = server_stderr = None
    server_exit = None
    sampler = gpu.GpuSampler(
        raw / "gpu_samples.jsonl",
        interval_seconds=float(runtime_contract["nvidia_smi_sample_interval_seconds"]),
    )
    model_started_wall_ns = time.time_ns()
    model_finished_wall_ns = None
    done_written = False
    try:
        _immutable_signed(started_file, {
            "schema_version": SHARED_SCHEMA,
            "kind": "model_started",
            "study_id": STUDY_ID,
            "study_commit": context.study_commit,
            "session_id": D1_SESSION_ID,
            "model_job_id": D1_MODEL_JOB_ID,
            "simulator_job_id": D1_SIM_JOB_ID,
            "model_started_wall_time_ns": model_started_wall_ns,
            "simulator_ready_payload_sha256": ready["payload_sha256"],
        })
        sampler.set_phase("model_loading_with_simulator")
        server_stdout_path = raw / "server.stdout.log"
        server_stderr_path = raw / "server.stderr.log"
        server_attempt = _record_child_launch_attempt(
            raw / "server.launch.json",
            launch_kind="d1_server",
            study_commit=context.study_commit,
            job_id=context.job_id,
            command=server_command,
            cwd=context.source_root,
        )
        try:
            server, server_stdout, server_stderr = d1.launch_server(
                server_command,
                stdout_path=server_stdout_path,
                stderr_path=server_stderr_path,
                cwd=context.source_root,
                env=d1_env,
            )
        except BaseException as error:
            _record_child_launch_outcome(
                raw / "server.launch.json",
                attempt=server_attempt,
                process=None,
                launch_kind="d1_server",
                study_commit=context.study_commit,
                job_id=context.job_id,
                command=server_command,
                cwd=context.source_root,
                error=error,
            )
            raise
        server_outcome = _record_child_launch_outcome(
            raw / "server.launch.json",
            attempt=server_attempt,
            process=server,
            launch_kind="d1_server",
            study_commit=context.study_commit,
            job_id=context.job_id,
            command=server_command,
            cwd=context.source_root,
        )
        server_launch = _record_child_launch(
            raw / "server.launch.json",
            process=server,
            launch_kind="d1_server",
            study_commit=context.study_commit,
            job_id=context.job_id,
            command=server_command,
            cwd=context.source_root,
            attempt=server_attempt,
            outcome=server_outcome,
        )
        server_identity = server_launch["process"]
        sampler.set_root("model", server.pid)
        sampler.start()
        server_contract_path = future_root / "server_contract.json"
        server_contract = d1.wait_for_server_contract(
            server,
            server_contract_path,
            runtime=d1.RuntimePaths(),
            source_root=context.source_root,
            topology=topology,
            port=port,
            timeout_seconds=7200,
        )
        _raise_if_d1_simulator_failed(
            simulator_failed_file, study_commit=context.study_commit
        )
        model_loaded_wall_ns = time.time_ns()
        sampler.set_phase("model_inference_with_simulator")
        probe_command = d1.build_probe_command(
            runtime=d1.RuntimePaths(),
            source_root=context.source_root,
            fixture=fixture_copy,
            fixture_sha256=fixture_sha,
            future_root=future_root,
            output_dir=probe_output,
            port=port,
        )
        inference_started_wall_ns = time.time_ns()
        probe_result, probe_launch = _run_peer_monitored_child(
            probe_command,
            stdout_path=raw / "probe.stdout.log",
            stderr_path=raw / "probe.stderr.log",
            cwd=context.source_root,
            env=d1_env,
            timeout_seconds=43200,
            poll_interval_seconds=float(
                runtime_contract["peer_failure_poll_interval_seconds"]
            ),
            peer_failure_path=simulator_failed_file,
            study_commit=context.study_commit,
            job_id=context.job_id,
            launch_receipt_path=raw / "probe.launch.json",
            termination_receipt_path=raw / "probe.termination.json",
            termination_grace_seconds=float(
                runtime_contract["child_termination_grace_seconds"]
            ),
        )
        inference_finished_wall_ns = time.time_ns()
        _raise_if_d1_simulator_failed(
            simulator_failed_file, study_commit=context.study_commit
        )
        require(probe_result.returncode == 0, "D1 resource six-request probe failed")
        report_path = probe_output / "d1_probe_qualification.json"
        report = _load_json(report_path, "D1 resource qualification")
        require(
            report.get("schema_version") == "wmf-d1-six-request-qualification-v1"
            and report.get("status") == "passed"
            and report.get("passed") is True
            and report.get("generation_request_count") == 6
            and report.get("behavioral_episode_count") == 0,
            "D1 resource qualification did not pass",
        )
        sampler.set_phase("model_shutdown_with_simulator")
        server_exit = d1.terminate_server(
            server,
            stdout_handle=server_stdout,
            stderr_handle=server_stderr,
            stdout_path=server_stdout_path,
            stderr_path=server_stderr_path,
            grace_seconds=float(runtime_contract["child_termination_grace_seconds"]),
        )
        server_stdout = server_stderr = None
        require(server_exit.get("reaped") is True, "D1 resource server was not reaped")
        model_finished_wall_ns = time.time_ns()
        sample_run = sampler.stop()
        sampler = None
        intervals = {
            "model_load": (model_started_wall_ns, model_loaded_wall_ns),
            "model_inference": (inference_started_wall_ns, inference_finished_wall_ns),
            "simultaneous_model_runtime": (model_started_wall_ns, model_finished_wall_ns),
        }
        summary = gpu.summarize_samples(
            Path(sample_run["artifact"]["path"]),
            expected_gpu_count=2,
            required_gpu_name=runtime_contract["required_gpu_name"],
            maximum_gap_seconds=float(runtime_contract["maximum_accepted_sample_gap_seconds"]),
            intervals=intervals,
            minimum_samples_by_interval={
                "model_inference": int(runtime_contract["minimum_inference_phase_samples"]),
                "simultaneous_model_runtime": int(runtime_contract["minimum_simultaneous_phase_samples"]),
            },
        )
        summary_path = raw / "gpu_summary.json"
        queue.immutable_json(summary_path, summary, maximum_bytes=2 * 1024 * 1024)
        allocator = _d1_allocator_summary(future_root)
        done = _immutable_signed(done_file, {
            "schema_version": SHARED_SCHEMA,
            "kind": "model_done",
            "study_id": STUDY_ID,
            "study_commit": context.study_commit,
            "session_id": D1_SESSION_ID,
            "model_job_id": D1_MODEL_JOB_ID,
            "simulator_job_id": D1_SIM_JOB_ID,
            "model_status": "passed",
            "model_started_wall_time_ns": model_started_wall_ns,
            "model_finished_wall_time_ns": model_finished_wall_ns,
            "generation_requests_issued": 6,
            "generation_requests_completed": 6,
            "gpu_summary": queue.file_identity(summary_path),
        })
        done_written = True
        receipt = gpu.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "status": "passed",
            "decision": "machine_measurement_passed_confirmation_held",
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": "d1-model-profile",
            "model_id": "D1",
            "job_id": context.job_id,
            "study_commit": context.study_commit,
            "queue_role": context.role,
            "worker_id": context.worker_id,
            "runtime_identity": {"hostname": context.hostname, "pod_uid": context.pod_uid, "pid": os.getpid()},
            "queue_descriptor": context.descriptor_identity,
            "queue_claim": context.claim_identity,
            "implementation": implementation,
            "prerequisites": prerequisites,
            "preflight": preflight,
            "capture_validation": capture_validation,
            "fixture_snapshot": fixture_snapshot,
            "topology": topology,
            "server_process": server_identity,
            "server_launch": queue.file_identity(raw / "server.launch.json"),
            "server_contract": d1._compact_contract(server_contract, server_contract_path),
            "server_exit": server_exit,
            "probe": queue.file_identity(report_path),
            "probe_launch": queue.file_identity(raw / "probe.launch.json"),
            "probe_process": probe_launch["process"],
            "allocator": allocator,
            "peer_handshake": {
                "session_root": str(session),
                "simulator_ready": queue.file_identity(ready_file),
                "model_started": queue.file_identity(started_file),
                "model_done": queue.file_identity(done_file),
                "model_done_payload_sha256": done["payload_sha256"],
            },
            "gpu_samples": sample_run,
            "gpu_summary": queue.file_identity(summary_path),
            "science_counts": _model_science_counts(model="D1"),
            "machine_resource_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "claim_boundary": (
                "Six new fixed-input official D1 generation requests while a separate settled "
                "simulator is resident. No retained behavioral cell was rerun or relabelled."
            ),
        })
        queue.immutable_json(
            publish / "development_resource_qualification_job_receipt.json",
            receipt,
            maximum_bytes=4 * 1024 * 1024,
        )
        return receipt
    finally:
        if server is not None and server.poll() is None:
            with contextlib.suppress(BaseException):
                server_exit = d1.terminate_server(
                    server,
                    stdout_handle=server_stdout,
                    stderr_handle=server_stderr,
                    stdout_path=raw / "server.stdout.log",
                    stderr_path=raw / "server.stderr.log",
                    grace_seconds=float(runtime_contract["child_termination_grace_seconds"]),
                )
                server_stdout = server_stderr = None
        if server_stdout is not None and server_stderr is not None:
            _finish_child_handles(server_stdout, server_stderr)
        if sampler is not None:
            with contextlib.suppress(BaseException):
                sampler.stop()
        if not done_written:
            with contextlib.suppress(BaseException):
                if model_finished_wall_ns is None:
                    model_finished_wall_ns = time.time_ns()
                partial_issued, partial_completed, partial_status = (
                    _d1_partial_request_counts(raw / "d1_future/episodes")
                )
                _immutable_signed(done_file, {
                    "schema_version": SHARED_SCHEMA,
                    "kind": "model_done",
                    "study_id": STUDY_ID,
                    "study_commit": context.study_commit,
                    "session_id": D1_SESSION_ID,
                    "model_job_id": D1_MODEL_JOB_ID,
                    "simulator_job_id": D1_SIM_JOB_ID,
                    "model_status": "technical_invalid",
                    "model_started_wall_time_ns": model_started_wall_ns,
                    "model_finished_wall_time_ns": model_finished_wall_ns,
                    "generation_requests_issued": partial_issued,
                    "generation_requests_completed": partial_completed,
                    "request_count_evidence_status": partial_status,
                })


def _failure_launch_state(
    path: Path,
    *,
    launch_kind: str,
    study_commit: str,
    job_id: str,
) -> tuple[int | None, dict[str, Any]]:
    """Return 1 only from an authenticated post-Popen receipt.

    The signed attempt/outcome pair closes the interval around ``Popen``.  A
    missing attempt proves the launch path was not entered; an authenticated
    failed outcome proves zero; and an authenticated successful outcome proves
    one even if writing the richer final launch receipt later failed.  An
    attempt without a valid outcome is unknown, never silently coerced.
    """

    supplied = Path(path)
    attempt_path = _launch_evidence_path(supplied, "attempt")
    outcome_path = _launch_evidence_path(supplied, "outcome")
    evidence_exists = any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (supplied, attempt_path, outcome_path)
    )
    if not evidence_exists:
        return 0, {
            "status": "not_launched",
            "receipt": None,
            "attempt_receipt": None,
            "outcome_receipt": None,
        }
    try:
        require(attempt_path.is_file() and not attempt_path.is_symlink(),
                f"{launch_kind} launch attempt path is invalid")
        attempt = _verify_child_launch_attempt(
            supplied,
            launch_kind=launch_kind,
            study_commit=study_commit,
            job_id=job_id,
        )
        if not outcome_path.exists() and not outcome_path.is_symlink():
            return None, {
                "status": "authenticated_attempt_outcome_missing",
                "receipt": None,
                "attempt_receipt": queue.file_identity(attempt_path),
                "outcome_receipt": None,
            }
        require(outcome_path.is_file() and not outcome_path.is_symlink(),
                f"{launch_kind} launch outcome path is invalid")
        outcome = _verify_child_launch_outcome(
            supplied,
            launch_kind=launch_kind,
            study_commit=study_commit,
            job_id=job_id,
        )
        started = bool(outcome["process_started"])
        base = {
            "status": (
                "authenticated_launched"
                if started and supplied.is_file() and not supplied.is_symlink()
                else "authenticated_popen_succeeded_final_receipt_missing"
                if started
                else "authenticated_popen_failed"
            ),
            "attempt_receipt": queue.file_identity(attempt_path),
            "outcome_receipt": queue.file_identity(outcome_path),
            "process": dict(outcome["process"]) if started else None,
        }
        if started and supplied.is_file() and not supplied.is_symlink():
            value = _verify_child_launch(
                supplied,
                launch_kind=launch_kind,
                study_commit=study_commit,
                job_id=job_id,
            )
            base["receipt"] = queue.file_identity(supplied)
            base["process"] = dict(value["process"])
        else:
            require(not supplied.exists() and not supplied.is_symlink(),
                    f"{launch_kind} final launch receipt path is invalid")
            base["receipt"] = None
        return int(started), base
    except BaseException as error:
        return None, {
            "status": "invalid_launch_evidence",
            "receipt_path": str(supplied),
            "attempt_receipt_path": str(attempt_path),
            "outcome_receipt_path": str(outcome_path),
            "error_type": type(error).__name__,
        }


def _safe_simulator_failure_accounting(
    path: Path,
    *,
    child_launch_state: int | None,
    study_commit: str,
    model_job_id: str,
    simulator_job_id: str,
) -> dict[str, Any]:
    try:
        return _summarize_simulator_accounting(
            path,
            study_commit=study_commit,
            model_job_id=model_job_id,
            simulator_job_id=simulator_job_id,
        )
    except BaseException as error:
        # A proved Popen failure means the child could not reset anything.  If
        # launch succeeded (or its receipt is corrupt), absence/corruption of
        # the inner journal makes completion unknowable and must stay bounded.
        definitely_not_started = child_launch_state == 0
        return {
            "status": "not_applicable_child_not_launched" if definitely_not_started
            else "unavailable_or_invalid",
            "error_type": None if definitely_not_started else type(error).__name__,
            "physical_resets": 0 if definitely_not_started else None,
            "physical_resets_lower_bound": 0,
            "physical_resets_upper_bound": 0 if definitely_not_started else 1,
            "settling_hold_actions": 0 if definitely_not_started else None,
            "settling_hold_actions_lower_bound": 0,
            "settling_hold_actions_upper_bound": 0 if definitely_not_started else None,
            "reset_completion_ambiguous": not definitely_not_started,
            "settling_completion_ambiguous": not definitely_not_started,
        }


def _n3_partial_request_counts(path: Path) -> tuple[int | None, int | None, str]:
    if not path.exists() and not path.is_symlink():
        return 0, 0, "no_request_journal"
    try:
        require(path.is_file() and not path.is_symlink(), "N3 request journal is invalid")
        issued = completed = 0
        previous: str | None = None
        open_requests: set[int] = set()
        for sequence, line in enumerate(path.read_bytes().splitlines()):
            row = json.loads(line)
            require(isinstance(row, Mapping) and row.get("sequence") == sequence,
                    "N3 partial request journal sequence changed")
            require(row.get("previous_event_sha256") == previous,
                    "N3 partial request journal chain changed")
            observed = row.get("event_sha256")
            unsigned = dict(row)
            unsigned.pop("event_sha256", None)
            require(
                isinstance(observed, str)
                and hashlib.sha256(
                    json.dumps(
                        unsigned, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest() == observed,
                "N3 partial request journal signature changed",
            )
            previous = observed
            kind = row.get("kind")
            if kind not in {"generation_request_started", "generation_request_completed"}:
                continue
            payload = row.get("payload")
            require(isinstance(payload, Mapping),
                    "N3 partial request event payload is invalid")
            request_index = payload.get("request_index")
            require(type(request_index) is int and 0 <= request_index < 6,
                    "N3 partial request index is invalid")
            if kind == "generation_request_started":
                require(request_index == issued and request_index not in open_requests,
                        "N3 partial request starts are not contiguous")
                open_requests.add(request_index)
                issued += 1
            else:
                require(request_index in open_requests and request_index == completed,
                        "N3 partial request completion has no matching start")
                open_requests.remove(request_index)
                completed += 1
        require(0 <= completed <= issued <= 6, "N3 partial request counts are invalid")
        return issued, completed, "authenticated_runtime_journal"
    except BaseException:
        return None, None, "invalid_runtime_journal"


def _d1_partial_request_counts(path: Path) -> tuple[int | None, int | None, str]:
    if not path.exists() and not path.is_symlink():
        return 0, 0, "no_server_request_directory"
    try:
        require(path.is_dir() and not path.is_symlink(),
                "D1 partial episode root is invalid")
        request_dirs: list[Path] = []
        completed = 0
        for episode in sorted(path.iterdir()):
            require(episode.is_dir() and not episode.is_symlink(),
                    "D1 partial episode entry is invalid")
            children = sorted(episode.glob("request_*"))
            require(len(children) <= 1, "D1 partial episode has repeated requests")
            for request in children:
                require(
                    request.is_dir() and not request.is_symlink()
                    and re.fullmatch(r"request_[0-9]{4}", request.name) is not None,
                    "D1 partial request directory is invalid",
                )
                request_dirs.append(request)
                receipt = request / "request_receipt.json"
                if receipt.exists() or receipt.is_symlink():
                    require(receipt.is_file() and not receipt.is_symlink(),
                            "D1 partial request receipt is invalid")
                    value = _load_json(receipt, "D1 partial request receipt")
                    require(value.get("schema_version") == "wmf-d1-request-receipt-v1",
                            "D1 partial request receipt schema changed")
                    completed += 1
        require(0 <= completed <= len(request_dirs) <= 6,
                "D1 partial request counts are outside the six-request protocol")
        return len(request_dirs), completed, "validated_server_artifacts"
    except BaseException:
        return None, None, "invalid_server_request_artifacts"


def _failure_science_counts(
    job: ProbeJob, job_dir: Path, *, study_commit: str
) -> dict[str, Any]:
    raw = Path(job_dir) / "raw"
    issued: int | None = 0
    completed: int | None = 0
    request_evidence_status = "not_applicable"
    launch_evidence: dict[str, Any] = {}
    load_started: int | None = 0
    server_started: int | None = 0
    simulator_started: int | None = 0
    physical_resets: int | None = 0
    settling: int | None = 0
    reset_lower: int | None = 0
    reset_upper: int | None = 0
    settling_lower: int | None = 0
    settling_upper: int | None = 0
    simulator_accounting: dict[str, Any] = {"status": "not_applicable"}
    if job.mode == "n3-profile":
        simulator_started, launch_evidence["simulator"] = _failure_launch_state(
            raw / "simulator.launch.json",
            launch_kind="simulator",
            study_commit=study_commit,
            job_id=job.job_id,
        )
        load_started, launch_evidence["model"] = _failure_launch_state(
            raw / "n3_model.launch.json",
            launch_kind="n3_model",
            study_commit=study_commit,
            job_id=job.job_id,
        )
        issued, completed, request_evidence_status = _n3_partial_request_counts(
            raw / "n3_profile/events.partial.jsonl"
        )
        simulator_accounting = _safe_simulator_failure_accounting(
            raw / "simulator_hold/simulator_accounting.jsonl",
            child_launch_state=simulator_started,
            study_commit=study_commit,
            model_job_id=N3_JOB_ID,
            simulator_job_id=N3_JOB_ID,
        )
    elif job.mode == "d1-model-profile":
        server_started, launch_evidence["server"] = _failure_launch_state(
            raw / "server.launch.json",
            launch_kind="d1_server",
            study_commit=study_commit,
            job_id=job.job_id,
        )
        load_started = server_started
        probe_started, launch_evidence["probe"] = _failure_launch_state(
            raw / "probe.launch.json",
            launch_kind="d1_probe",
            study_commit=study_commit,
            job_id=job.job_id,
        )
        launch_evidence["probe"]["science_count_role"] = (
            "process_lifecycle_only; requests come from server artifacts"
        )
        issued, completed, request_evidence_status = _d1_partial_request_counts(
            raw / "d1_future/episodes"
        )
        del probe_started
    else:
        simulator_started, launch_evidence["simulator"] = _failure_launch_state(
            raw / "simulator.launch.json",
            launch_kind="simulator",
            study_commit=study_commit,
            job_id=job.job_id,
        )
        simulator_accounting = _safe_simulator_failure_accounting(
            raw / "simulator_hold/simulator_accounting.jsonl",
            child_launch_state=simulator_started,
            study_commit=study_commit,
            model_job_id=D1_MODEL_JOB_ID,
            simulator_job_id=D1_SIM_JOB_ID,
        )
    if job.mode in {"n3-profile", "d1-simulator-profile"}:
        if simulator_accounting.get("status") == "authenticated" and simulator_started != 1:
            prior_launch = dict(launch_evidence["simulator"])
            if prior_launch.get("status") == "authenticated_popen_failed":
                simulator_started = None
                launch_evidence["simulator"] = {
                    "status": "conflicting_authenticated_launch_and_child_entry_evidence",
                    "outer_launch_evidence": prior_launch,
                    "inner_child_journal": simulator_accounting["journal"],
                }
            else:
                simulator_started = 1
                launch_evidence["simulator"] = {
                    "status": "authenticated_launched_from_inner_child_entry",
                    "outer_launch_evidence": prior_launch,
                    "inner_child_journal": simulator_accounting["journal"],
                }
        physical_resets = simulator_accounting["physical_resets"]
        settling = simulator_accounting["settling_hold_actions"]
        reset_lower = simulator_accounting["physical_resets_lower_bound"]
        reset_upper = simulator_accounting["physical_resets_upper_bound"]
        settling_lower = simulator_accounting["settling_hold_actions_lower_bound"]
        settling_upper = simulator_accounting["settling_hold_actions_upper_bound"]
    return {
        "model_runtime_loads_started": load_started,
        "model_servers_started": server_started,
        "model_requests_issued": issued,
        "model_requests_completed": completed,
        "request_count_evidence_status": request_evidence_status,
        "simulator_processes_started": simulator_started,
        "physical_resets": physical_resets,
        "physical_resets_lower_bound": reset_lower,
        "physical_resets_upper_bound": reset_upper,
        "settling_hold_actions": settling,
        "settling_hold_actions_lower_bound": settling_lower,
        "settling_hold_actions_upper_bound": settling_upper,
        "launch_evidence": launch_evidence,
        "simulator_accounting": simulator_accounting,
        **_zero_behavior_counts(),
    }


def _signal_d1_peer_failure(
    *, args: argparse.Namespace, context: Any | None, error: BaseException
) -> None:
    """Release the paired D1 job even when failure precedes child launch.

    These markers have no success authority.  They exist solely so a peer can
    fail closed promptly instead of consuming its full six-hour hold timeout.
    """

    command = getattr(args, "command", None)
    if command not in ("d1-model-profile", "d1-simulator-profile"):
        return
    job = PROBE_BY_MODE[command]
    require(getattr(args, "job_id", None) == job.job_id,
            "D1 peer failure signal job ID changed")
    supplied_commit = getattr(context, "study_commit", getattr(args, "study_commit", None))
    study_commit = _verified_commit(supplied_commit)
    session = _session_root(study_commit)
    session.parent.mkdir(parents=True, exist_ok=True)
    try:
        session.mkdir()
    except FileExistsError:
        pass
    require(session.is_dir() and not session.is_symlink(),
            "D1 peer failure session is invalid")
    now = time.time_ns()
    common = {
        "schema_version": SHARED_SCHEMA,
        "study_id": STUDY_ID,
        "study_commit": study_commit,
        "session_id": D1_SESSION_ID,
        "model_job_id": D1_MODEL_JOB_ID,
        "simulator_job_id": D1_SIM_JOB_ID,
        "status": "technical_invalid",
        "error_type": type(error).__name__,
        "error_detail": str(error)[:2000],
        "failure_observed_wall_time_ns": now,
    }
    counts = _failure_science_counts(
        job, Path(args.job_dir), study_commit=study_commit
    )
    if command == "d1-simulator-profile":
        failure_file = session / "simulator_failed.json"
        if failure_file.exists() or failure_file.is_symlink():
            _verify_shared(
                failure_file, kind="simulator_failed", study_commit=study_commit
            )
            return
        _immutable_signed(
            failure_file,
            {
                **common,
                "kind": "simulator_failed",
                "simulator_process_started": counts["simulator_processes_started"],
                "physical_resets": counts["physical_resets"],
                "physical_resets_bounds": {
                    "lower": counts["physical_resets_lower_bound"],
                    "upper": counts["physical_resets_upper_bound"],
                },
                "settling_hold_actions": counts["settling_hold_actions"],
                "settling_hold_actions_bounds": {
                    "lower": counts["settling_hold_actions_lower_bound"],
                    "upper": counts["settling_hold_actions_upper_bound"],
                },
                "launch_evidence": counts["launch_evidence"],
                "simulator_accounting": counts["simulator_accounting"],
            },
        )
        return

    started_file = session / "model_started.json"
    model_started_wall_ns = now
    if started_file.exists() or started_file.is_symlink():
        started = _verify_shared(
            started_file, kind="model_started", study_commit=study_commit
        )
        candidate = started.get("model_started_wall_time_ns")
        if type(candidate) is int and candidate > 0:
            model_started_wall_ns = candidate
    else:
        _immutable_signed(
            started_file,
            {
                **common,
                "kind": "model_started",
                "model_started_wall_time_ns": model_started_wall_ns,
                "model_launch_succeeded": counts["model_runtime_loads_started"],
                "launch_evidence": counts["launch_evidence"],
            },
        )
    done_file = session / "model_done.json"
    if done_file.exists() or done_file.is_symlink():
        _verify_shared(done_file, kind="model_done", study_commit=study_commit)
        return
    _immutable_signed(
        done_file,
        {
            **common,
            "kind": "model_done",
            "model_status": "technical_invalid",
            "model_started_wall_time_ns": model_started_wall_ns,
            "model_finished_wall_time_ns": time.time_ns(),
            "generation_requests_issued": counts["model_requests_issued"],
            "generation_requests_completed": counts["model_requests_completed"],
            "request_count_evidence_status": counts["request_count_evidence_status"],
        },
    )


def _write_probe_failure(
    *, args: argparse.Namespace, context: Any | None, error: BaseException
) -> None:
    try:
        job = PROBE_BY_MODE[args.command]
        job_dir = Path(args.job_dir)
        publish = job_dir / "publish"
        if publish.exists() or publish.is_symlink():
            require(publish.is_dir() and not publish.is_symlink(), "failure publish path is invalid")
        else:
            publish.mkdir()
        success = publish / "development_resource_qualification_job_receipt.json"
        failure = publish / "development_resource_qualification_job_failure.json"
        if success.exists() or failure.exists():
            return
        receipt = gpu.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "status": "technical_invalid",
            "decision": "no_go",
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": job.mode,
            "job_id": job.job_id,
            "study_commit": getattr(context, "study_commit", getattr(args, "study_commit", None)),
            "queue_role": getattr(context, "role", getattr(args, "expected_role", None)),
            "failure": {
                "error_type": type(error).__name__,
                "detail": str(error)[:2000],
                "traceback": traceback.format_exc(limit=30)[-16000:],
            },
            "science_counts": _failure_science_counts(
                job,
                job_dir,
                study_commit=_verified_commit(
                    getattr(context, "study_commit", getattr(args, "study_commit", None))
                ),
            ),
            "machine_resource_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "claim_boundary": (
                "Technical-invalid resource attempt. Any started nonbehavioral requests are "
                "counted separately; no behavioral cell or label was created."
            ),
        })
        queue.immutable_json(failure, receipt, maximum_bytes=2 * 1024 * 1024)
    except BaseException:
        return


def run_probe_job(args: argparse.Namespace) -> dict[str, Any]:
    context = None
    try:
        expected = _runtime_probe_descriptor(args)
        context = queue.validate_queue_context(
            source_root=args.source_root,
            job_dir=args.job_dir,
            study_commit=args.study_commit,
            job_id=args.job_id,
            expected_role=args.expected_role,
            expected_worker_id=args.expected_worker_id,
            expected_descriptor=expected,
        )
        implementation, contract = _validate_staged_implementation(context.source_root, args)
        prerequisites = _validate_runtime_prerequisites(args)
        if args.command == "n3-profile":
            return run_n3_profile(
                context=context, contract=contract, implementation=implementation,
                prerequisites=prerequisites,
            )
        if args.command == "d1-model-profile":
            return run_d1_model_profile(
                context=context, contract=contract, implementation=implementation,
                prerequisites=prerequisites,
            )
        return run_d1_simulator_profile(
            context=context, contract=contract, implementation=implementation,
            prerequisites=prerequisites,
        )
    except BaseException as error:
        with contextlib.suppress(BaseException):
            _signal_d1_peer_failure(args=args, context=context, error=error)
        _write_probe_failure(args=args, context=context, error=error)
        raise


def _probe_receipt_pvc(job_id: str) -> Path:
    return CONTROL_ROOT / "jobs" / job_id / "publish/development_resource_qualification_job_receipt.json"


def _validate_probe_receipt(
    path: Path, expected_sha256: str, *, mode: str, expected_job_id: str
) -> dict[str, Any]:
    identity = queue.file_identity(path)
    require(identity["sha256"] == _verified_sha(expected_sha256, f"{mode} receipt digest"),
            f"{mode} receipt hash mismatch")
    value = _load_json(path, f"{mode} receipt")
    gpu.verify_signed_document(value, f"{mode} receipt")
    exact = {
        "schema_version": JOB_RECEIPT_SCHEMA,
        "status": "passed",
        "decision": "machine_measurement_passed_confirmation_held",
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": mode,
        "job_id": expected_job_id,
        "machine_resource_gate_complete": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
    }
    for key, wanted in exact.items():
        require(value.get(key) == wanted, f"{mode} receipt changed: {key}")
    counts = value.get("science_counts")
    require(isinstance(counts, Mapping), f"{mode} science counts are missing")
    require(all(counts.get(key) == 0 for key in _zero_behavior_counts()),
            f"{mode} receipt claims behavioral or label work")
    return {"identity": identity, "receipt": value}


def _finalize_argv(
    study_commit: str,
    *,
    implementation: Mapping[str, Mapping[str, Any]],
    receipt_hashes: Mapping[str, str],
) -> list[str]:
    argv = [
        "/usr/bin/python3",
        "{source_root}/" + str(THIS_RELATIVE),
        "finalize",
        "--source-root", "{source_root}",
        "--study-commit", study_commit,
        "--job-dir", "{job_dir}",
        "--job-id", FINALIZE_JOB_ID,
        "--expected-role", FINALIZE_ROLE,
        "--expected-worker-id", FINALIZE_ROLE,
        "--wrapper-sha256", implementation["wrapper"]["sha256"],
        "--contract-sha256", implementation["contract"]["sha256"],
        "--sampler-sha256", implementation["sampler"]["sha256"],
        "--queue-support-sha256", implementation["queue_support"]["sha256"],
    ]
    for name, job_id in (
        ("n3", N3_JOB_ID), ("d1-model", D1_MODEL_JOB_ID), ("d1-simulator", D1_SIM_JOB_ID)
    ):
        argv.extend((
            f"--{name}-receipt", str(_probe_receipt_pvc(job_id)),
            f"--{name}-receipt-sha256", receipt_hashes[name],
        ))
    return argv


def build_finalize_descriptor(
    study_commit: str,
    *,
    implementation: Mapping[str, Mapping[str, Any]],
    receipt_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "job_id": FINALIZE_JOB_ID,
        "released": True,
        "source_commit": _verified_commit(study_commit),
        "role": FINALIZE_ROLE,
        "argv": _finalize_argv(
            study_commit, implementation=implementation, receipt_hashes=receipt_hashes
        ),
        "max_wall_seconds": 3600,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def build_finalize_wave(
    *,
    study_commit: str,
    n3_receipt: Path,
    n3_receipt_sha256: str,
    d1_model_receipt: Path,
    d1_model_receipt_sha256: str,
    d1_simulator_receipt: Path,
    d1_simulator_receipt_sha256: str,
) -> dict[str, Any]:
    commit = _verified_commit(study_commit)
    supplied = {
        "n3": _validate_probe_receipt(
            n3_receipt, n3_receipt_sha256, mode="n3-profile", expected_job_id=N3_JOB_ID
        ),
        "d1-model": _validate_probe_receipt(
            d1_model_receipt, d1_model_receipt_sha256,
            mode="d1-model-profile", expected_job_id=D1_MODEL_JOB_ID,
        ),
        "d1-simulator": _validate_probe_receipt(
            d1_simulator_receipt, d1_simulator_receipt_sha256,
            mode="d1-simulator-profile", expected_job_id=D1_SIM_JOB_ID,
        ),
    }
    require(all(row["receipt"].get("study_commit") == commit for row in supplied.values()),
            "resource probe receipts do not share the finalize commit")
    implementation = _implementation(REPOSITORY_ROOT)
    hashes = {name: row["identity"]["sha256"] for name, row in supplied.items()}
    return {
        "schema_version": FINALIZE_WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "descriptor_only_not_dispatched_three_probe_gate_passed",
        "source_commit": commit,
        "inputs": {name: row["identity"] for name, row in supplied.items()},
        "new_model_requests_issued_by_finalize": 0,
        **_zero_behavior_counts(),
        "jobs": [build_finalize_descriptor(
            commit, implementation=implementation, receipt_hashes=hashes
        )],
        "safe_to_release_confirmation": False,
        "claim_boundary": (
            "One CPU-safe compiler descriptor only. It can freeze the serial machine envelope "
            "but cannot invent human annotation time or release confirmation."
        ),
    }


def _runtime_finalize_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    require(args.job_id == FINALIZE_JOB_ID, "finalize job ID changed")
    require(args.expected_role == FINALIZE_ROLE and args.expected_worker_id == FINALIZE_ROLE,
            "finalize worker changed")
    implementation = {
        "wrapper": {"sha256": args.wrapper_sha256},
        "contract": {"sha256": args.contract_sha256},
        "sampler": {"sha256": args.sampler_sha256},
        "queue_support": {"sha256": args.queue_support_sha256},
    }
    hashes = {
        "n3": args.n3_receipt_sha256,
        "d1-model": args.d1_model_receipt_sha256,
        "d1-simulator": args.d1_simulator_receipt_sha256,
    }
    return build_finalize_descriptor(
        args.study_commit, implementation=implementation, receipt_hashes=hashes
    )


def _summary_from_receipt(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    descriptor = value.get("gpu_summary")
    require(isinstance(descriptor, Mapping), f"{label} GPU summary descriptor is missing")
    path = Path(str(descriptor.get("path")))
    identity = queue.file_identity(path)
    require(
        identity["sha256"] == descriptor.get("sha256")
        and identity["bytes"] == descriptor.get("bytes"),
        f"{label} GPU summary identity changed",
    )
    summary = _load_json(path, f"{label} GPU summary")
    gpu.verify_signed_document(summary, f"{label} GPU summary")
    require(summary.get("schema_version") == gpu.SUMMARY_SCHEMA and summary.get("status") == "passed",
            f"{label} GPU summary did not pass")
    sample_descriptor = summary.get("sample_journal")
    require(isinstance(sample_descriptor, Mapping), f"{label} sample descriptor is missing")
    sample_identity = queue.file_identity(Path(str(sample_descriptor.get("path"))))
    require(
        sample_identity["sha256"] == sample_descriptor.get("sha256")
        and sample_identity["bytes"] == sample_descriptor.get("bytes"),
        f"{label} sample journal changed",
    )
    rows = gpu.load_samples(Path(sample_identity["path"]))
    require(len(rows) == summary.get("sample_count"), f"{label} sample count changed")
    return summary


def _headroom_rows(
    summary: Mapping[str, Any], interval: str, *, multiplier: float
) -> list[dict[str, Any]]:
    value = summary.get("intervals", {}).get(interval)
    require(isinstance(value, Mapping), f"resource interval is missing: {interval}")
    total = summary.get("gpu_total_memory_bytes_by_uuid")
    indices = summary.get("gpu_logical_index_by_uuid")
    require(isinstance(total, Mapping) and isinstance(indices, Mapping),
            "resource GPU identity maps are missing")
    devices = value.get("devices")
    require(isinstance(devices, list) and devices, f"resource interval devices are missing: {interval}")
    rows = []
    for device in devices:
        uuid = device.get("gpu_uuid")
        peak = device.get("whole_device_peak_bytes")
        process_peak = device.get("whole_process_peak_bytes")
        require(
            isinstance(uuid, str)
            and type(peak) is int and peak > 0
            and type(process_peak) is int and process_peak > 0,
            "resource device peak is invalid",
        )
        required = math.ceil(peak * multiplier)
        capacity = total.get(uuid)
        require(type(capacity) is int and capacity > 0, "resource GPU capacity is invalid")
        require(required <= capacity, "measured GPU peak plus frozen headroom exceeds capacity")
        rows.append({
            "gpu_uuid": uuid,
            "logical_index": indices.get(uuid),
            "observed_whole_device_peak_bytes": peak,
            "observed_whole_process_peak_bytes": process_peak,
            "headroom_multiplier": multiplier,
            "required_bytes_with_headroom": required,
            "visible_capacity_bytes": capacity,
            "remaining_bytes_after_headroom": capacity - required,
            "task_owner_process_peak_bytes": dict(device.get("task_owner_process_peak_bytes", {})),
        })
    return rows


def _validate_owner_topology(
    *, n3: Sequence[Mapping[str, Any]], d1_model: Sequence[Mapping[str, Any]],
    d1_sim: Sequence[Mapping[str, Any]],
) -> None:
    require(len(n3) == 2 and len(d1_model) == 2 and len(d1_sim) == 1,
            "resource topology GPU count changed")
    n3_model = [row for row in n3 if row["task_owner_process_peak_bytes"].get("model", 0) > 0]
    n3_sim = [row for row in n3 if row["task_owner_process_peak_bytes"].get("simulator", 0) > 0]
    require(len(n3_model) == len(n3_sim) == 1,
            "N3 model/simulator ownership did not map to one GPU each")
    require(n3_model[0]["gpu_uuid"] != n3_sim[0]["gpu_uuid"],
            "N3 model and simulator used the same GPU")
    require(all(row["task_owner_process_peak_bytes"].get("model", 0) > 0 for row in d1_model),
            "D1 model did not occupy both distributed GPUs")
    require(d1_sim[0]["task_owner_process_peak_bytes"].get("simulator", 0) > 0,
            "D1 simulator process peak is absent")
    require(
        not ({row["gpu_uuid"] for row in d1_model} & {row["gpu_uuid"] for row in d1_sim}),
        "D1 model and simulator workers were assigned overlapping GPUs",
    )


def run_finalize_job(args: argparse.Namespace) -> dict[str, Any]:
    expected = _runtime_finalize_descriptor(args)
    context = queue.validate_queue_context(
        source_root=args.source_root,
        job_dir=args.job_dir,
        study_commit=args.study_commit,
        job_id=args.job_id,
        expected_role=args.expected_role,
        expected_worker_id=args.expected_worker_id,
        expected_descriptor=expected,
    )
    implementation, contract = _validate_staged_implementation(context.source_root, args)
    supplied = {
        "n3": (Path(args.n3_receipt), args.n3_receipt_sha256, "n3-profile", N3_JOB_ID),
        "d1-model": (
            Path(args.d1_model_receipt), args.d1_model_receipt_sha256,
            "d1-model-profile", D1_MODEL_JOB_ID,
        ),
        "d1-simulator": (
            Path(args.d1_simulator_receipt), args.d1_simulator_receipt_sha256,
            "d1-simulator-profile", D1_SIM_JOB_ID,
        ),
    }
    expected_paths = {
        "n3": _probe_receipt_pvc(N3_JOB_ID),
        "d1-model": _probe_receipt_pvc(D1_MODEL_JOB_ID),
        "d1-simulator": _probe_receipt_pvc(D1_SIM_JOB_ID),
    }
    receipts = {}
    for name, (path, digest, mode, job_id) in supplied.items():
        require(path == expected_paths[name], f"{name} runtime receipt path changed")
        receipts[name] = _validate_probe_receipt(
            path, digest, mode=mode, expected_job_id=job_id
        )
        require(receipts[name]["receipt"].get("study_commit") == context.study_commit,
                f"{name} runtime receipt commit changed")

    counts = {name: row["receipt"]["science_counts"] for name, row in receipts.items()}
    require(counts["n3"].get("model_requests_issued") == 6,
            "N3 resource request count changed")
    require(counts["d1-model"].get("model_requests_issued") == 6,
            "D1 resource request count changed")
    require(counts["d1-simulator"].get("model_requests_issued") == 0,
            "D1 simulator contacted a model")
    require(
        counts["n3"].get("physical_resets") == 1
        and counts["d1-simulator"].get("physical_resets") == 1,
        "resource simulator reset count changed",
    )

    n3_summary = _summary_from_receipt(receipts["n3"]["receipt"], "N3")
    d1_model_summary = _summary_from_receipt(receipts["d1-model"]["receipt"], "D1 model")
    d1_sim_summary = _summary_from_receipt(receipts["d1-simulator"]["receipt"], "D1 simulator")

    # Both D1 receipts must bind the exact same three immutable handshake files.
    model_peer = receipts["d1-model"]["receipt"].get("peer_handshake")
    sim_peer = receipts["d1-simulator"]["receipt"].get("peer_handshake")
    require(isinstance(model_peer, Mapping) and isinstance(sim_peer, Mapping),
            "D1 peer handshake is missing")
    for key in ("simulator_ready", "model_started", "model_done"):
        left, right = model_peer.get(key), sim_peer.get(key)
        require(
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and left.get("sha256") == right.get("sha256")
            and left.get("bytes") == right.get("bytes")
            and left.get("path") == right.get("path"),
            f"D1 peer handshake differs: {key}",
        )

    multiplier = float(contract["runtime"]["headroom_multiplier"])
    n3_rows = _headroom_rows(n3_summary, "simultaneous_model_runtime", multiplier=multiplier)
    d1_model_rows = _headroom_rows(
        d1_model_summary, "simultaneous_model_runtime", multiplier=multiplier
    )
    d1_sim_rows = _headroom_rows(
        d1_sim_summary, "simultaneous_model_runtime", multiplier=multiplier
    )
    _validate_owner_topology(n3=n3_rows, d1_model=d1_model_rows, d1_sim=d1_sim_rows)

    # Preserve allocator data but never substitute it for driver-visible peaks.
    n3_allocator = receipts["n3"]["receipt"].get("model_output", {}).get("allocator")
    d1_allocator = receipts["d1-model"]["receipt"].get("allocator")
    require(
        isinstance(n3_allocator, Mapping) and n3_allocator.get("request_count") == 6,
        "N3 allocator evidence is incomplete",
    )
    require(
        isinstance(d1_allocator, Mapping) and d1_allocator.get("request_count") == 6,
        "D1 allocator evidence is incomplete",
    )

    raw, publish = queue._prepare_output_directories(context.job_dir)
    freeze = gpu.signed_document({
        "schema_version": FINAL_RECEIPT_SCHEMA,
        "status": "machine_resource_gate_passed_annotation_time_pending",
        "decision": "serial_gm_topology_frozen_confirmation_held",
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "study_commit": context.study_commit,
        "job_id": context.job_id,
        "queue_role": context.role,
        "worker_id": context.worker_id,
        "runtime_identity": {"hostname": context.hostname, "pod_uid": context.pod_uid, "pid": os.getpid()},
        "queue_descriptor": context.descriptor_identity,
        "queue_claim": context.claim_identity,
        "implementation": implementation,
        "inputs": {name: row["identity"] for name, row in receipts.items()},
        "measurement": {
            "sample_interval_seconds": contract["runtime"]["nvidia_smi_sample_interval_seconds"],
            "maximum_accepted_sample_gap_seconds": contract["runtime"]["maximum_accepted_sample_gap_seconds"],
            "whole_device_peak_definition": contract["measured_fields"]["whole_device_peak"],
            "whole_process_peak_definition": contract["measured_fields"]["whole_process_peak"],
            "allocator_metrics_are_not_nvidia_smi_metrics": True,
        },
        "model_envelopes": {
            "N3": {
                "topology": "one two-GPU worker; model and simulator on distinct assigned GPUs",
                "simultaneous_gpu_envelope": n3_rows,
                "model_load": n3_summary["intervals"]["model_load"],
                "model_inference": n3_summary["intervals"]["model_inference"],
                "simulator": n3_summary["intervals"]["simulator_lifetime"],
                "allocator": n3_allocator,
            },
            "D1": {
                "topology": "one two-GPU distributed model worker plus one distinct one-GPU simulator worker",
                "model_worker_simultaneous_gpu_envelope": d1_model_rows,
                "simulator_worker_simultaneous_gpu_envelope": d1_sim_rows,
                "model_load": d1_model_summary["intervals"]["model_load"],
                "model_inference": d1_model_summary["intervals"]["model_inference"],
                "simulator": d1_sim_summary["intervals"]["simulator_lifetime"],
                "allocator": d1_allocator,
            },
        },
        "safe_execution_topology": {
            **contract["topology_freeze_policy"],
            "measurement_status": "passed_at_one_serial_block_per_model",
            "higher_concurrency_qualified": False,
            "scheduling_rule": (
                "Run at most one intact four-condition block globally. Do not overlap N3 and D1 "
                "blocks; reuse only the measured per-model topology."
            ),
        },
        "science_counts": {
            "new_nonbehavioral_generation_requests": 12,
            "new_simulator_resets": 2,
            "new_settling_hold_actions": (
                int(counts["n3"]["settling_hold_actions"])
                + int(counts["d1-simulator"]["settling_hold_actions"])
            ),
            **_zero_behavior_counts(),
        },
        "machine_resource_gate_complete": True,
        "resource_measurement_freeze_complete": True,
        "annotation_time": {
            "status": "pending_completed_two_rater_development_workflow",
            "rater_a_total_seconds": None,
            "rater_b_total_seconds": None,
            "adjudication_total_seconds": None,
            "synthesized": False,
        },
        "remaining_resource_release_requirements": [
            "two independent raters' authenticated development annotation time",
            "authenticated adjudication time when adjudication occurs"
        ],
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "behavioral_policy_skill_evaluated": False,
        "claim_boundary": (
            "Conservative serial GM machine-capacity freeze from twelve new nonbehavioral fixed-input "
            "requests and two zero-behavioral-action simulator resets. It is not per-cell behavioral "
            "memory, prediction skill, annotation evidence, or authority to release confirmation."
        ),
    })
    raw_path = raw / "machine_resource_freeze.json"
    queue.immutable_json(raw_path, freeze, maximum_bytes=8 * 1024 * 1024)
    receipt = dict(freeze)
    receipt["raw_machine_resource_freeze"] = queue.file_identity(raw_path)
    # Re-sign after adding the raw self-contained source descriptor.
    receipt.pop("payload_sha256")
    receipt = gpu.signed_document(receipt)
    queue.immutable_json(
        publish / "development_resource_machine_freeze.json",
        receipt,
        maximum_bytes=8 * 1024 * 1024,
    )
    return receipt


def _write_finalize_failure(args: argparse.Namespace, error: BaseException) -> None:
    try:
        publish = Path(args.job_dir) / "publish"
        if publish.exists() or publish.is_symlink():
            require(publish.is_dir() and not publish.is_symlink(),
                    "finalize failure publish path is invalid")
        else:
            publish.mkdir()
        success = publish / "development_resource_machine_freeze.json"
        failure = publish / "development_resource_machine_freeze_failure.json"
        if success.exists() or failure.exists():
            return
        receipt = gpu.signed_document({
            "schema_version": FINAL_RECEIPT_SCHEMA,
            "status": "technical_invalid",
            "decision": "no_go",
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "study_commit": getattr(args, "study_commit", None),
            "job_id": FINALIZE_JOB_ID,
            "failure": {
                "error_type": type(error).__name__,
                "detail": str(error)[:2000],
                "traceback": traceback.format_exc(limit=30)[-16000:],
            },
            "new_model_requests_issued_by_finalize": 0,
            **_zero_behavior_counts(),
            "machine_resource_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
        })
        queue.immutable_json(failure, receipt, maximum_bytes=2 * 1024 * 1024)
    except BaseException:
        return


def _add_runtime_base(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-role", required=True)
    parser.add_argument("--expected-worker-id", required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--sampler-sha256", required=True)
    parser.add_argument("--queue-support-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    emit = commands.add_parser("emit-probes", help="emit three descriptors; do not dispatch")
    emit.add_argument("--study-commit", required=True)
    emit.add_argument("--resource-audit-receipt", type=Path, required=True)
    emit.add_argument("--resource-audit-receipt-sha256", required=True)
    emit.add_argument("--output", type=Path)

    emit_finalize = commands.add_parser(
        "emit-finalize", help="emit one receipt-gated CPU finalize descriptor"
    )
    emit_finalize.add_argument("--study-commit", required=True)
    for name in ("n3", "d1-model", "d1-simulator"):
        emit_finalize.add_argument(f"--{name}-receipt", type=Path, required=True)
        emit_finalize.add_argument(f"--{name}-receipt-sha256", required=True)
    emit_finalize.add_argument("--output", type=Path)

    for mode in PROBE_BY_MODE:
        runtime = commands.add_parser(mode)
        _add_runtime_base(runtime)
        runtime.add_argument("--resource-audit-receipt", type=Path, required=True)
        runtime.add_argument("--resource-audit-receipt-sha256", required=True)

    simulator = commands.add_parser("internal-simulator-hold")
    simulator.add_argument("--source-root", type=Path, required=True)
    simulator.add_argument("--study-commit", required=True)
    simulator.add_argument("--output-dir", type=Path, required=True)
    simulator.add_argument("--ready-file", type=Path, required=True)
    simulator.add_argument("--stop-file", type=Path, required=True)
    simulator.add_argument("--model-job-id", required=True)
    simulator.add_argument("--simulator-job-id", required=True)
    simulator.add_argument("--hold-timeout-seconds", type=int, required=True)

    finalize = commands.add_parser("finalize")
    _add_runtime_base(finalize)
    for name in ("n3", "d1-model", "d1-simulator"):
        finalize.add_argument(f"--{name}-receipt", type=Path, required=True)
        finalize.add_argument(f"--{name}-receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "emit-probes":
        wave = build_probe_wave(
            study_commit=args.study_commit,
            resource_audit_receipt=args.resource_audit_receipt,
            resource_audit_receipt_sha256=args.resource_audit_receipt_sha256,
        )
        queue._write_descriptor_output(args.output, wave)
        return 0
    if args.command == "emit-finalize":
        wave = build_finalize_wave(
            study_commit=args.study_commit,
            n3_receipt=args.n3_receipt,
            n3_receipt_sha256=args.n3_receipt_sha256,
            d1_model_receipt=args.d1_model_receipt,
            d1_model_receipt_sha256=args.d1_model_receipt_sha256,
            d1_simulator_receipt=args.d1_simulator_receipt,
            d1_simulator_receipt_sha256=args.d1_simulator_receipt_sha256,
        )
        queue._write_descriptor_output(args.output, wave)
        return 0
    if args.command == "internal-simulator-hold":
        return _internal_simulator_main(args)
    if args.command == "finalize":
        try:
            receipt = run_finalize_job(args)
        except BaseException as error:
            _write_finalize_failure(args, error)
            raise
    else:
        receipt = run_probe_job(args)
    print(json.dumps({
        "status": receipt["status"],
        "job_id": receipt.get("job_id"),
        "safe_to_release_confirmation": False,
    }, sort_keys=True), flush=True)
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
        raise
