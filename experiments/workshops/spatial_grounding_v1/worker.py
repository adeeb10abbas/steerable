"""Finite, durable SGW-01 partition worker; production adapters are fail-closed."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .contract import Cell, ContractError, Release, load_release, validate_stage_authorizations, verify_completion_pointer
from .gpu_idle_probe import select_idle
from .recorder import AttemptRecorder, atomic_json, next_attempt_number, utc_now

EXIT_RELEASE_INVALID, EXIT_ATTEMPTS_EXHAUSTED, EXIT_STORAGE_BUDGET_BLOCKED = 42, 43, 44
SOURCE_QUEUE_EPISODE_COUNT = 1044
MAX_IDLE_PROBE_AGE_SECONDS = 300
STOP_REQUESTED = False
STOP_GRACE_EXPIRED = False


class DeadlineExceeded(RuntimeError):
    pass


class ResourceBlocked(OSError):
    """Raised before model construction when a fixed resource gate fails."""


class Adapter(Protocol):
    def reset(self, cell: Cell, recorder: AttemptRecorder) -> dict[str, Any]: ...
    def run_episode(self, cell: Cell, recorder: AttemptRecorder, reset: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


ScoreFn = Callable[[Mapping[str, Any], Cell], Mapping[str, Any]]


def _stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    # Do not kill a completed attempt prematurely, but do not permit a hung
    # runtime to survive Kubernetes' documented 180-second termination grace.
    signal.signal(signal.SIGALRM, _grace_expired)
    signal.setitimer(signal.ITIMER_REAL, 180)


def _grace_expired(_signum: int, _frame: Any) -> None:
    global STOP_GRACE_EXPIRED
    STOP_GRACE_EXPIRED = True
    raise DeadlineExceeded("SIGTERM grace period expired")


def _run_with_deadline(seconds: int, operation: str, callback):
    if seconds <= 0:
        raise ContractError(f"{operation} deadline must be positive")
    previous = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(DeadlineExceeded(f"{operation} deadline exceeded")))
        signal.setitimer(signal.ITIMER_REAL, seconds)
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def model_lock(release: Release, model: str):
    path = release.root.parent / "locks" / f"{model}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"global model lock held: {path}") from exc
        stream.write(json.dumps({"pid": os.getpid(), "release_id": release.release_id, "started_at_utc": utc_now()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_adapter(model: str) -> Adapter:
    """Load only the production adapter supplied by the runtime owner."""
    module_name = f"experiments.workshops.spatial_grounding_v1.adapters"
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, "load_production_adapter")
        return factory(model)
    except (ImportError, AttributeError) as exc:
        raise ContractError("production adapter is unavailable; fake adapters are tests-only") from exc


def _completion(release: Release, cell: Cell) -> bool:
    path = release.root.parent / "cells" / f"{cell.cell_id}.complete.json"
    if not path.exists():
        return False
    verify_completion_pointer(release, path)
    return True


def _receipt(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
        raise ResourceBlocked(f"{label} receipt is absent or unhashed")
    path = Path(value["path"])
    if not path.is_file():
        raise ResourceBlocked(f"{label} receipt is unavailable: {path}")
    from .contract import load_json, sha256_file
    try:
        if sha256_file(path) != value["sha256"]:
            raise ResourceBlocked(f"{label} receipt hash changed: {path}")
        return load_json(path, label)
    except (ContractError, OSError, UnicodeError) as exc:
        raise ResourceBlocked(f"{label} receipt is unreadable or malformed: {exc}") from exc


def _expiry(receipt: Mapping[str, Any], label: str) -> datetime:
    value = receipt.get("expires_at_utc")
    if not isinstance(value, str):
        raise ResourceBlocked(f"{label} receipt lacks expires_at_utc")
    try:
        expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResourceBlocked(f"{label} receipt has invalid expires_at_utc") from exc
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise ResourceBlocked(f"{label} receipt is expired")
    return expiry


def _unexpired(receipt: Mapping[str, Any], label: str) -> None:
    _expiry(receipt, label)


def _source_queue_hash(release: Release) -> str:
    receipt = json.loads((release.root / "release_receipt.json").read_text(encoding="utf-8"))
    value = receipt.get("source_queue_sha256")
    if not isinstance(value, str):
        raise ResourceBlocked("release lacks source queue provenance")
    return value


def _runtime_identity_sha256(release: Release) -> str:
    """Hash the execution identity without receipt references, avoiding a hash cycle."""
    binding = release.binding
    identity = {
        "worker_image_digest": binding["worker_image_digest"],
        "source_commit": binding["source_commit"],
        "simulator_commit": binding["simulator_commit"],
        "model_code_commits": binding["model_code_commits"],
        "checkpoint_hashes": binding["checkpoint_hashes"],
        "node_gpu_type": binding["node_gpu_type"],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _receipt_identity(release: Release, receipt: Mapping[str, Any], label: str) -> None:
    binding = release.binding
    expected = {
        "release_id": release.release_id,
        "source_queue_sha256": _source_queue_hash(release),
        "source_queue_episode_count": SOURCE_QUEUE_EPISODE_COUNT,
        "pvc_name": binding["pvc_name"],
        "pvc_mount_path": binding["pvc_mount_path"],
        "study_root": binding["source_root"],
        "runtime_identity_sha256": _runtime_identity_sha256(release),
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise ResourceBlocked(f"{label} receipt is not bound to this release, PVC, study root, and runtime")


def _positive_finite_number(value: Any) -> bool:
    return type(value) in {int, float} and value > 0 and math.isfinite(value)


def _as_needed_scaling(authorization: Mapping[str, Any], constraints: Mapping[str, Any]) -> bool:
    """Only an explicit approved null-cap receipt can waive legacy aggregate caps."""
    return (
        authorization.get("budget_mode") == "existing_idle_capacity_no_aggregate_hour_cap"
        and constraints.get("allocation_scaling") == "as_needed_verified_idle_capacity"
        and "max_concurrent_model_workers" in constraints
        and "max_total_allocated_gpus" in constraints
        and constraints["max_concurrent_model_workers"] is None
        and constraints["max_total_allocated_gpus"] is None
    )


def _authorization_check(release: Release, *, model: str) -> str:
    authorization = _receipt(release.binding.get("operational_authorization_receipt"), "operational authorization")
    if authorization.get("schema_version") != "sgw-01-operational-authorization-v1" or authorization.get("status") != "approved":
        raise ResourceBlocked("operational authorization receipt is not approved")
    mode = authorization.get("budget_mode")
    if mode not in {"numeric_global_gpu_hour_cap", "existing_idle_capacity_no_aggregate_hour_cap"}:
        raise ResourceBlocked("operational authorization has an unsupported budget mode")
    scope = authorization.get("scope")
    constraints = authorization.get("constraints")
    if not isinstance(scope, Mapping) or not isinstance(constraints, Mapping):
        raise ResourceBlocked("operational authorization lacks bounded scope and constraints")
    models = scope.get("models")
    authorized_workers = constraints.get("max_concurrent_model_workers")
    authorized_gpus = constraints.get("max_total_allocated_gpus")
    as_needed = _as_needed_scaling(authorization, constraints)
    if (not isinstance(authorization.get("owner_approval_reference"), Mapping)
            or authorization.get("source_protocol_sha256") != release.hashes["protocol.json"]
            or authorization.get("source_queue_sha256") != _source_queue_hash(release)
            or scope.get("context") != release.binding["context"]
            or scope.get("namespace") != release.binding["namespace"]
            or scope.get("pvc") != release.binding["pvc_name"]
            or scope.get("persistent_study_root") != release.binding["source_root"]
            or not isinstance(models, list) or model not in models
            or scope.get("maximum_registered_behavioral_episodes") != SOURCE_QUEUE_EPISODE_COUNT
            or scope.get("maximum_attempts_per_behavioral_cell") != 3
            or not (as_needed or (
                type(authorized_workers) is int and type(authorized_gpus) is int
                and 1 <= release.binding["max_concurrent_model_workers"] <= authorized_workers <= 2
                and 1 <= release.binding["max_total_allocated_gpus"] <= authorized_gpus <= 4
            ))
            or constraints.get("existing_authorized_cluster_capacity_only") is not True
            or constraints.get("fresh_idle_allocation_check_required") is not True
            or constraints.get("new_paid_capacity_allowed") is not False
            or constraints.get("new_cluster_provisioning_allowed") is not False
            or constraints.get("preempt_or_stop_unowned_workloads_allowed") is not False
            or constraints.get("bounded_job_deadline_required") is not True):
        raise ResourceBlocked("operational authorization does not cover this bounded existing-capacity launch")
    if mode == "numeric_global_gpu_hour_cap" and not _positive_finite_number(authorization.get("approved_global_gpu_hours")):
        raise ResourceBlocked("numeric operational authorization lacks a finite global GPU-hour cap")
    return mode


def _allocation_check(release: Release, *, model: str, minimum_runtime_seconds: int,
                      verify_idle_probe: bool = True) -> int:
    """Validate the coordinator-owned Job reservation; this is not a local ledger."""
    receipt = _receipt(release.binding.get("external_allocation_receipt"), "external allocation")
    if receipt.get("schema") != "sgw-01-external-allocation-v1" or receipt.get("status") != "approved":
        raise ResourceBlocked("external allocation receipt is not approved")
    mode = _authorization_check(release, model=model)
    _receipt_identity(release, receipt, "external allocation")
    required_strings = ("owner_approval_reference", "reservation_id", "context", "namespace", "job_name", "job_uid", "pod_uid", "model")
    if any(not isinstance(receipt.get(field), str) or not receipt[field] for field in required_strings):
        raise ResourceBlocked("external allocation receipt lacks concrete reservation identity")
    if receipt["context"] != release.binding["context"] or receipt["namespace"] != release.binding["namespace"]:
        raise ResourceBlocked("external allocation receipt has a different Kubernetes scope")
    if receipt["model"] != model or receipt["job_uid"] != os.environ.get("JOB_UID") or receipt["pod_uid"] != os.environ.get("POD_UID"):
        raise ResourceBlocked("external allocation receipt does not bind this Job UID, pod UID, and model")
    gpu_count = receipt.get("allocated_gpu_count")
    active = receipt.get("activeDeadlineSeconds")
    if type(gpu_count) is not int or gpu_count != release.binding["model_gpu_counts"].get(model) or type(active) is not int or active <= 0:
        raise ResourceBlocked("external allocation receipt lacks the exact GPU allocation/deadline")
    start_value = receipt.get("startTime")
    deadline_value = receipt.get("deadline_utc")
    if not isinstance(start_value, str) or not isinstance(deadline_value, str):
        raise ResourceBlocked("external allocation receipt lacks Job timing")
    try:
        start = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(deadline_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResourceBlocked("external allocation receipt has invalid Job timing") from exc
    if (start.tzinfo is None or deadline.tzinfo is None or start > datetime.now(timezone.utc)
            or deadline != start + timedelta(seconds=active)):
        raise ResourceBlocked("external allocation receipt has invalid Job timing")
    reservation = receipt.get("reservation_gpu_hours")
    expected = gpu_count * active / 3600
    if receipt.get("budget_mode") != mode or not _positive_finite_number(reservation) or abs(reservation - expected) > 1e-9:
        raise ResourceBlocked("external allocation receipt has invalid per-Job worst-case GPU-hour accounting")
    if mode == "numeric_global_gpu_hour_cap":
        approved = receipt.get("approved_global_gpu_hours")
        reserved = receipt.get("globally_reserved_gpu_hours")
        authorization = _receipt(release.binding["operational_authorization_receipt"], "operational authorization")
        if (not _positive_finite_number(approved) or not _positive_finite_number(reserved)
                or approved != authorization["approved_global_gpu_hours"]
                or not reservation <= reserved <= approved):
            raise ResourceBlocked("external allocation receipt has invalid capped global reservation accounting")
    if verify_idle_probe:
        idle = _receipt(receipt.get("gpu_idle_probe_receipt"), "GPU idle probe")
        allocated = receipt.get("allocated_gpu_uuids")
        observed = idle.get("observed_at_unix")
        gpus = idle.get("gpus")
        occupied = idle.get("compute_occupied_uuids")
        selected = idle.get("selected_gpu")
        now = datetime.now(timezone.utc).timestamp()
        if (idle.get("study_id") != "SGW-01"
                or idle.get("status") != "passed_idle_snapshot_only"
                or not _positive_finite_number(observed)
                or not start.timestamp() <= observed <= now
                or now - observed > MAX_IDLE_PROBE_AGE_SECONDS
                or type(idle.get("expected_visible_gpu_count")) is not int
                or idle["expected_visible_gpu_count"] != gpu_count
                or type(idle.get("model_requests")) is not int or idle["model_requests"] != 0
                or not isinstance(allocated, list) or len(allocated) != gpu_count
                or not all(isinstance(item, str) and item for item in allocated)
                or len(set(allocated)) != gpu_count
                or not isinstance(gpus, list) or not isinstance(occupied, list) or not isinstance(selected, Mapping)
                or len(gpus) != gpu_count
                or not all(isinstance(item, Mapping) and isinstance(item.get("uuid"), str) for item in gpus)
                or not all(isinstance(item, str) for item in occupied)
                or selected.get("uuid") not in allocated
                or set(allocated) != {item["uuid"] for item in gpus}
                or selected not in gpus
                or set(allocated) & set(occupied)
        ):
            raise ResourceBlocked("GPU idle probe does not prove this exact pre-launch allocation")
        for gpu in gpus:
            if any(type(gpu.get(field)) is not int or gpu[field] < 0 for field in (
                "index", "memory_used_mib", "memory_free_mib", "utilization_percent",
            )):
                raise ResourceBlocked("GPU idle probe has malformed device measurements")
            try:
                select_idle([gpu], set(occupied), 1)
            except (ValueError, RuntimeError) as exc:
                raise ResourceBlocked("not every allocated GPU passed the idle guard") from exc
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining < minimum_runtime_seconds:
        raise ResourceBlocked("external allocation expires before one bounded operation can finish")
    return remaining


def _space_check(release: Release, *, stage: str) -> None:
    floor = 100 * 1024**3
    configured = release.binding.get("minimum_free_bytes")
    if configured is not None and (type(configured) is not int or configured < floor):
        raise ResourceBlocked("minimum_free_bytes cannot lower the immutable 100 GiB floor")
    required = max(floor, configured or floor)
    storage_ref = release.binding.get("storage_budget_receipt")
    if stage in {"D", "C"} and storage_ref is None:
        raise ResourceBlocked(f"{stage} requires a measured pilot storage receipt")
    if storage_ref is not None:
        storage = _receipt(storage_ref, "storage budget")
        _unexpired(storage, "storage budget")
        if storage.get("status") != "approved":
            raise ResourceBlocked("storage budget receipt is not approved")
        _receipt_identity(release, storage, "storage budget")
        p95 = storage.get("pilot_p95_episode_bytes")
        remaining = storage.get("global_remaining_episode_count")
        if (type(p95) is not int or p95 <= 0 or type(remaining) is not int
                or remaining < SOURCE_QUEUE_EPISODE_COUNT):
            raise ResourceBlocked("storage receipt lacks conservative global remaining/pilot-P95 accounting")
        required = max(required, (3 * p95 * remaining + 1) // 2)
    free = shutil.disk_usage(release.root.parent).free
    if free < required:
        raise ResourceBlocked(f"persistent storage blocked: {free} < {required} free bytes")


def _budget_check(release: Release, *, stage: str, model: str, minimum_runtime_seconds: int) -> None:
    budget = _receipt(release.binding.get("resource_budget_receipt"), "resource budget")
    expiry = _expiry(budget, "resource budget")
    if budget.get("status") != "approved":
        raise ResourceBlocked("resource budget receipt is not approved")
    _receipt_identity(release, budget, "resource budget")
    if budget.get("model") not in {model, "all_models"}:
        raise ResourceBlocked("resource budget receipt does not cover this model")
    mode = _authorization_check(release, model=model)
    approved = budget.get("approved_gpu_hours")
    estimated = budget.get("estimated_remaining_gpu_hours")
    if not _positive_finite_number(estimated):
        raise ResourceBlocked("resource budget receipt lacks a finite conservative GPU-hour estimate")
    if mode == "numeric_global_gpu_hour_cap" and (
            not _positive_finite_number(approved) or estimated > approved):
        raise ResourceBlocked("numeric resource budget receipt lacks approved conservative GPU-hour coverage")
    if mode == "numeric_global_gpu_hour_cap":
        authorization = _receipt(release.binding["operational_authorization_receipt"], "operational authorization")
        if approved > authorization["approved_global_gpu_hours"]:
            raise ResourceBlocked("resource budget exceeds the owner-approved global GPU-hour cap")
    if (expiry - datetime.now(timezone.utc)).total_seconds() < minimum_runtime_seconds:
        raise ResourceBlocked("resource budget approval expires before one bounded episode can finish")
    if stage in {"D", "C"}:
        runtime = _receipt(budget.get("measured_runtime_receipt"), "measured runtime")
        _unexpired(runtime, "measured runtime")
        if runtime.get("status") != "measured":
            raise ResourceBlocked(f"{stage} requires a measured pilot runtime receipt")
        _receipt_identity(release, runtime, "measured runtime")
        if (runtime.get("model") not in {model, "all_models"}
                or runtime.get("max_cell_attempts") != 3
                or runtime.get("request_deadline_seconds") != release.binding.get("request_deadline_seconds", 300)
                or runtime.get("episode_deadline_seconds") != release.binding.get("episode_deadline_seconds", 900)
                or not _positive_finite_number(runtime.get("estimated_remaining_gpu_hours"))):
            raise ResourceBlocked(f"{stage} runtime estimate does not cover this model, deadlines, and retry allowance")
        if estimated < runtime["estimated_remaining_gpu_hours"]:
            raise ResourceBlocked(f"{stage} budget estimate is below the measured remaining GPU-hour bound")
    workers = release.binding.get("max_concurrent_model_workers")
    total_gpus = release.binding.get("max_total_allocated_gpus")
    authorization = _receipt(release.binding["operational_authorization_receipt"], "operational authorization")
    constraints = authorization.get("constraints")
    as_needed = isinstance(constraints, Mapping) and _as_needed_scaling(authorization, constraints)
    if (type(workers) is not int or type(total_gpus) is not int or workers < 1 or total_gpus < 1
            or (not as_needed and (workers > 2 or total_gpus > 4))
            or any(type(value) is not int or value < 1 for value in release.binding["model_gpu_counts"].values())
            or sum(release.binding["model_gpu_counts"].values()) > total_gpus):
        raise ResourceBlocked("runtime binding exceeds frozen two-worker/four-GPU ceiling")


def _status(release: Release, worker_id: str, **values: Any) -> None:
    atomic_json(release.root.parent / "status" / f"{worker_id}.json", {
        "schema_version": "sgw-01-worker-status-v1", "release_id": release.release_id,
        "worker_id": worker_id, "updated_at_utc": utc_now(), **values,
    })


class _Heartbeat:
    def __init__(self, release: Release, worker_id: str, seconds: int):
        if seconds <= 0:
            raise ContractError("heartbeat interval must be positive")
        self.release = release
        self.worker_id = worker_id
        self.seconds = seconds
        self.current_cell: str | None = None
        self.valid = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sgw-heartbeat", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.seconds + 1)

    def _run(self) -> None:
        while not self._stop.wait(self.seconds):
            _status(self.release, self.worker_id, state="running", current_cell=self.current_cell,
                    valid=self.valid, last_error=self.last_error, heartbeat=True)


def _stage_authorized(release: Release, stage: str) -> None:
    receipt = json.loads((release.root / "release_receipt.json").read_text(encoding="utf-8"))
    authorizations = validate_stage_authorizations(receipt.get("stage_authorizations"), stage)
    if stage not in authorizations:
        raise ContractError(f"stage {stage} lacks a release authorization receipt")


def _load_scorer() -> ScoreFn:
    """Resolve the scorer lazily so production cannot fall back to a fake."""
    try:
        from .scoring import FrozenScoringConfig, GoalSpec, result_payload, score_episode
    except ImportError as exc:
        raise ContractError("production scorer is unavailable") from exc

    def score(trace: Mapping[str, Any], cell: Cell) -> Mapping[str, Any]:
        goal = GoalSpec(
            family=cell.family, physical_goal_sign=int(cell.row["physical_goal_sign"]),
            form=str(cell.row["form"]),
        )
        scored = score_episode(trace, goal, FrozenScoringConfig())
        return result_payload(
            scored,
            release_id=str(trace["release_id"]),
            cell_id=cell.cell_id,
            attempt_id=str(trace["attempt_id"]),
            completed_at_utc=str(trace["completed_at_utc"]),
        )
    return score


def _canonical_outcome(outcome: Mapping[str, Any], cell: Cell, scorer: ScoreFn | None) -> dict[str, Any]:
    """Scorer, not adapter termination labels, decides every model denominator."""
    value = dict(outcome)
    if value.get("status") == "technical_invalid":
        if not isinstance(value.get("technical_cause"), str):
            raise ContractError("technical invalid execution lacks a technical cause")
        return value
    mapping = value.get("episode_mapping")
    if not isinstance(mapping, list):
        raise ContractError("nontechnical execution lacks the raw episode mapping required for scoring")
    trace = {
        "states": mapping,
        "terminal_observed": value.get("safety_terminated") is not True,
        "termination_reason": value.get("termination_reason"),
        "safety_terminated": value.get("safety_terminated") is True,
        "success_events": value.get("success_events", []),
    }
    trace.update({
        "release_id": cell.row["release_id"],
        "cell_id": cell.cell_id,
        "attempt_id": value.get("attempt_id", "pending"),
        "completed_at_utc": utc_now(),
    })
    scored = dict((scorer or _load_scorer())(trace, cell))
    status = scored.get("status")
    if status not in {"valid_success", "valid_model_failure", "censored", "technical_invalid"}:
        raise ContractError("scorer did not emit a canonical SGW status")
    # A normal 450-action endpoint is scored; censoring is only a physical
    # safety truncation, never an adapter convenience label.
    if status == "censored" and value.get("safety_terminated") is not True:
        raise ContractError("only explicit physical safety truncation may be censored")
    value.update(scored)
    value["status"] = status
    return value


def run_partition(release: Release, *, model: str, family: str, stage: str, max_valid: int,
                  max_attempts: int, worker_id: str, adapter: Adapter | None = None,
                  heartbeat_seconds: int = 60, scorer: ScoreFn | None = None) -> int:
    cells = release.partition(model, family, stage)
    if max_valid != len(cells) or max_attempts != 3:
        raise ContractError("partition limits must equal the frozen stage ceiling and three total attempts")
    _stage_authorized(release, stage)
    request_deadline = int(release.binding.get("request_deadline_seconds", 300))
    episode_deadline = int(release.binding.get("episode_deadline_seconds", 900))
    bounded_operation_seconds = request_deadline + episode_deadline
    valid = 0
    heartbeat = _Heartbeat(release, worker_id, heartbeat_seconds)
    heartbeat.start()
    try:
        with model_lock(release, model):
            pending = [cell for cell in cells if not _completion(release, cell)]
            if not pending:
                _status(release, worker_id, state="complete", valid=len(cells), expected=len(cells))
                return 0
            try:
                _space_check(release, stage=stage)
                _budget_check(release, stage=stage, model=model, minimum_runtime_seconds=bounded_operation_seconds)
                _allocation_check(release, model=model, minimum_runtime_seconds=bounded_operation_seconds)
            except ResourceBlocked as exc:
                _status(release, worker_id, state="blocked", reason=str(exc), valid=valid)
                return EXIT_STORAGE_BUDGET_BLOCKED
            # The global lock must cover model-server/factory construction, not
            # merely requests, so a duplicate Job cannot load a second policy.
            adapter = adapter or load_adapter(model)
            for cell in cells:
                heartbeat.current_cell = cell.cell_id
                heartbeat.valid = valid
                if _completion(release, cell):
                    valid += 1
                    continue
                while not STOP_REQUESTED:
                    try:
                        _space_check(release, stage=stage)
                        _budget_check(release, stage=stage, model=model, minimum_runtime_seconds=bounded_operation_seconds)
                        allocation_remaining = _allocation_check(
                            release, model=model, minimum_runtime_seconds=bounded_operation_seconds,
                            verify_idle_probe=False,
                        )
                    except ResourceBlocked as exc:
                        _status(release, worker_id, state="blocked", reason=str(exc), valid=valid)
                        return EXIT_STORAGE_BUDGET_BLOCKED
                    number = next_attempt_number(release, cell)
                    if number > max_attempts:
                        _status(release, worker_id, state="blocked", cell_id=cell.cell_id,
                                reason="technical attempts exhausted", valid=valid)
                        return EXIT_ATTEMPTS_EXHAUSTED
                    recorder = AttemptRecorder(release, cell, f"attempt-{number:03d}")
                    recorder.begin()
                    try:
                        reset = _run_with_deadline(
                            min(request_deadline, allocation_remaining), "reset",
                            lambda: adapter.reset(cell, recorder),
                        )
                        if not isinstance(reset, dict) or not reset.get("full_reset"):
                            raise ContractError("adapter did not attest a full reset")
                        remaining_after_reset = _allocation_check(
                            release, model=model, minimum_runtime_seconds=episode_deadline,
                            verify_idle_probe=False,
                        )
                        outcome = _run_with_deadline(
                            min(episode_deadline, remaining_after_reset), "episode",
                            lambda: adapter.run_episode(cell, recorder, reset)
                        )
                        outcome = {**outcome, "attempt_id": recorder.attempt_id}
                        outcome = _canonical_outcome(outcome, cell, scorer)
                        published = recorder.complete(outcome)
                    except DeadlineExceeded as exc:
                        heartbeat.last_error = f"{type(exc).__name__}: {exc}"
                        recorder.event("technical_invalid", error_type=type(exc).__name__, error=str(exc))
                        published = recorder.complete({"status": "technical_invalid", "technical_cause": heartbeat.last_error})
                    except ContractError:
                        raise
                    except OSError as exc:
                        heartbeat.last_error = f"{type(exc).__name__}: {exc}"
                        recorder.event("technical_invalid", error_type=type(exc).__name__, error=str(exc))
                        published = recorder.complete({"status": "technical_invalid", "technical_cause": heartbeat.last_error})
                    except Exception as exc:
                        recorder.event("fatal_unclassified_error", error_type=type(exc).__name__, error=str(exc))
                        raise ContractError(f"unclassified adapter failure; attempt preserved: {type(exc).__name__}") from exc
                    _status(release, worker_id, state="running", current_cell=cell.cell_id, valid=valid,
                            attempts=number, bytes_written=sum(p.stat().st_size for p in recorder.path.rglob("*") if p.is_file()))
                    if published or _completion(release, cell):
                        valid += 1
                        break
                if STOP_REQUESTED:
                    _status(release, worker_id, state="interrupted", current_cell=cell.cell_id, valid=valid)
                    return EXIT_ATTEMPTS_EXHAUSTED
        _status(release, worker_id, state="complete", valid=valid, expected=len(cells))
        return 0
    finally:
        heartbeat.close()
        close = getattr(adapter, "close", None) if adapter is not None else None
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--model", choices=("N3", "D1"), required=True)
    parser.add_argument("--family", choices=("LAT", "HEIGHT", "DIST"), required=True)
    parser.add_argument("--stage", choices=("P", "D", "C"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-valid-episodes", type=int, required=True)
    parser.add_argument("--max-cell-attempts", type=int, default=3)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    release = load_release(args.release)
    try:
        code = run_partition(release, model=args.model, family=args.family, stage=args.stage,
                             max_valid=args.max_valid_episodes, max_attempts=args.max_cell_attempts,
                             worker_id=f"{args.model}-{args.family}-{args.stage}-{os.getpid()}",
                             heartbeat_seconds=args.heartbeat_seconds)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        code = EXIT_STORAGE_BUDGET_BLOCKED
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        code = EXIT_RELEASE_INVALID
    raise SystemExit(code)


if __name__ == "__main__":
    main()
