#!/usr/bin/env python3
"""Seal, prepare, and run the CPU-only confirmation evidence compiler.

This helper deliberately has no queue-release command.  It builds a signed
compiler input from an authoritative signed terminal-block index, or runs the
compiler against that immutable input and writes a compact signed job receipt.
Existing queue infrastructure may later invoke ``run`` as an ordinary bounded
CPU job; importing or executing this file cannot append to or mutate a queue.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
COMPILER_PATH = FORECAST_ROOT / "analysis" / "compile_confirmation_evidence.py"
CONTRACT_PATH = Path(__file__).with_name("confirmation_evidence_compiler_contract.json")
QUEUE_MODULE_PATH = Path(__file__).with_name("forecast_timing_queue_jobs.py")

STUDY_ID = "WMF-ABLATION-001"
BLOCK_INDEX_SCHEMA = "wmf-confirmation-terminal-block-index-v1"
SEAL_PLAN_SCHEMA = "wmf-confirmation-cohort-seal-plan-v1"
JOB_RECEIPT_SCHEMA = "wmf-confirmation-evidence-compiler-job-v1"
SEAL_JOB_RECEIPT_SCHEMA = "wmf-confirmation-cohort-seal-job-v1"
MANIFEST_JOB_RECEIPT_SCHEMA = "wmf-confirmation-evidence-manifest-job-v1"
INPUT_SCHEMA = "wmf-confirmation-evidence-compiler-input-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
BRANCH_MODELS = {
    "full_two_model": ("N3", "D1"),
    "reduced_n3": ("N3",),
    "reduced_d1": ("D1",),
}
BLOCK_INDEX_KEYS = {
    "schema_version",
    "study_id",
    "cohort_branch",
    "study_commit",
    "development_release_freeze",
    "prepared_schedule",
    "cohort_close_receipt",
    "block_receipts",
    "payload_sha256",
}
SEAL_PLAN_KEYS = {
    "schema_version",
    "study_id",
    "cohort_branch",
    "study_commit",
    "development_release_freeze",
    "prepared_schedule",
    "selected_blocks",
    "payload_sha256",
}
SEAL_SELECTION_KEYS = {"model_id", "layout_pair_id", "selected_queue_job_ids"}


class ConfirmationCompilerJobError(RuntimeError):
    """A non-releasing compiler preparation/runtime gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfirmationCompilerJobError(message)


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None,
            f"cannot load compiler: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


compiler = _load_module(COMPILER_PATH, "wmf_confirmation_evidence_compiler_job")
queue = _load_module(QUEUE_MODULE_PATH, "wmf_confirmation_evidence_queue_support")
require(queue.NAMESPACE == compiler.NAMESPACE, "queue/compiler namespace changed")
BLOCK_ROW_KEYS = set(compiler.BLOCK_INPUT_KEYS)

THIS_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/confirmation_evidence_compiler_jobs.py"
)
RUN_BUNDLE_RELATIVE = Path("raw/compiler_bundle")
RUN_RECEIPT_NAME = "confirmation_evidence_compiler_job_receipt.json"
MANIFEST_RELATIVE = Path("raw/compiler_input_manifest.json")
MANIFEST_RECEIPT_NAME = "confirmation_evidence_manifest_job_receipt.json"
SEAL_BUNDLE_RELATIVE = Path("raw/cohort_seal")
SEAL_RECEIPT_NAME = "confirmation_cohort_seal_job_receipt.json"
QUEUE_MAX_WALL_SECONDS = 21600
QUEUE_PUBLISH_LOG_TAIL_BYTES = 8192
MAX_SEAL_ADMISSION_WAIT_SECONDS = 900.0
MIN_POST_CUTOFF_COMPILER_WALL_SECONDS = 14400.0


def canonical_bytes(value: Any) -> bytes:
    return compiler.canonical_bytes(value)


def pretty_bytes(value: Any) -> bytes:
    return compiler.pretty_bytes(value)


def sha256_file(path: Path) -> str:
    return compiler.sha256_file(path)


def sign_document(value: Mapping[str, Any]) -> dict[str, Any]:
    return compiler.sign_document(value)


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    try:
        compiler.verify_signed(value, label)
    except Exception as error:
        raise ConfirmationCompilerJobError(str(error)) from error


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    require(not missing and not extra,
            f"{label} fields changed (missing={sorted(missing)}, extra={sorted(extra)})")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(pretty_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_descriptor(path: Path) -> dict[str, Any]:
    return compiler.development.file_descriptor(path)


def _verify_file(path: Path, digest: str, label: str) -> dict[str, Any]:
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{label} SHA-256 is invalid")
    supplied = Path(path)
    compiler.development._reject_symlink_components(supplied, label)
    resolved = supplied.resolve()
    require(resolved.is_file() and sha256_file(resolved) == digest,
            f"{label} is missing or changed")
    return _file_descriptor(resolved)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _confirmation_admission_barrier(
    control: Mapping[str, Any], *, sealed_at: str
) -> dict[str, Any]:
    """Prove the snapshotted control cannot admit another queued claim."""

    sealed_unix = datetime.fromisoformat(sealed_at[:-1] + "+00:00").timestamp()
    deadline = control.get("admission_deadline_unix")
    if control.get("shutdown") is True:
        kind = "global_shutdown"
    else:
        require(
            type(deadline) in {int, float}
            and not isinstance(deadline, bool)
            and math.isfinite(float(deadline))
            and float(deadline) > 0
            and float(deadline) <= sealed_unix,
            "cohort cannot be sealed before shutdown or its absolute admission deadline",
        )
        kind = "expired_absolute_admission_deadline"
    return {
        "kind": kind,
        "control_commit": control["control_commit"],
        "control_generation": control["control_generation"],
        "active_job_ids": list(control["active_job_ids"]),
        "admission_deadline_unix": deadline,
        "verified_at_utc": sealed_at,
    }


def _regular_file_descriptor(path: Path, label: str) -> dict[str, Any]:
    supplied = Path(path)
    compiler.development._reject_symlink_components(supplied, label)
    resolved = supplied.resolve()
    require(resolved.is_file(), f"{label} is missing")
    info = resolved.stat()
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular file")
    return _file_descriptor(resolved)


def _argv_option(argv: Sequence[Any], name: str, label: str) -> str:
    try:
        return compiler._option(argv, name, label)
    except Exception as error:
        raise ConfirmationCompilerJobError(str(error)) from error


def _load_seal_plan(
    path: Path,
    digest: str,
    *,
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _verify_file(path, digest, "confirmation cohort seal plan")
    value = compiler.load_json(Path(identity["path"]), "confirmation cohort seal plan")
    _exact_keys(value, SEAL_PLAN_KEYS, "confirmation cohort seal plan")
    require(value.get("schema_version") == SEAL_PLAN_SCHEMA,
            "confirmation cohort seal-plan schema changed")
    require(value.get("study_id") == STUDY_ID,
            "confirmation cohort seal-plan study changed")
    verify_signed(value, "confirmation cohort seal plan")
    branch = value.get("cohort_branch")
    commit = value.get("study_commit")
    require(branch in BRANCH_MODELS, "confirmation cohort seal-plan branch is invalid")
    require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None,
            "confirmation cohort seal-plan commit is invalid")
    release, _ = compiler._exact_file_reference(
        value.get("development_release_freeze"),
        base=Path(identity["path"]).parent,
        label="seal-plan development release freeze",
    )
    schedule, schedule_path = compiler._exact_file_reference(
        value.get("prepared_schedule"),
        base=Path(identity["path"]).parent,
        label="seal-plan prepared schedule",
    )
    require(value.get("development_release_freeze") == release
            and value.get("prepared_schedule") == schedule,
            "seal-plan file descriptors must be exact absolute identities")
    expected_schedule = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
    )
    require(schedule_path == expected_schedule.resolve(),
            "seal-plan prepared schedule is not the staged authoritative schedule")
    rows = value.get("selected_blocks")
    require(isinstance(rows, list), "confirmation cohort seal selections are invalid")
    expected = {
        (model, f"C{index:02d}")
        for model in BRANCH_MODELS[str(branch)]
        for index in range(1, 25)
    }
    observed: set[tuple[str, str]] = set()
    selections: dict[tuple[str, str], list[str]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"seal selection {index} is invalid")
        _exact_keys(row, SEAL_SELECTION_KEYS, f"seal selection {index}")
        key = (row.get("model_id"), row.get("layout_pair_id"))
        require(key in expected and key not in observed,
                f"seal selection {index} identity is invalid or duplicated")
        ids = row.get("selected_queue_job_ids")
        expected_count = 1 if key[0] == "N3" else 2
        require(
            isinstance(ids, list)
            and len(ids) == expected_count
            and ids == sorted(set(ids))
            and all(isinstance(item, str) and compiler.SAFE_COMPONENT_RE.fullmatch(item)
                    for item in ids),
            f"seal selection {index} queue-job selection is invalid",
        )
        observed.add((str(key[0]), str(key[1])))
        selections[(str(key[0]), str(key[1]))] = list(ids)
    require(observed == expected, "seal selections do not exactly cover the cohort")
    return identity, {
        **value,
        "development_release_freeze": release,
        "prepared_schedule": schedule,
        "selections": selections,
    }


@contextmanager
def _exclusive_lock(path: Path, label: str):
    supplied = Path(path)
    require(not supplied.is_symlink(), f"{label} is a symlink")
    supplied.parent.mkdir(parents=True, exist_ok=True)
    with supplied.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ConfirmationCompilerJobError(f"{label} is held") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _live_process_group_members(process_group_id: int | None) -> list[int]:
    """Return current Linux processes in the queue child's detached PGID."""

    if type(process_group_id) is not int or process_group_id <= 0:
        return []
    output: list[int] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir() if proc_root.is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            end = raw.rfind(")")
            fields = raw[end + 2:].split()
            if len(fields) >= 3 and int(fields[2]) == process_group_id:
                output.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
            continue
    return sorted(output)


def _matching_confirmation_job(
    descriptor: Mapping[str, Any], study_commit: str
) -> tuple[str, str, str, str | None] | None:
    argv = descriptor.get("argv")
    if not isinstance(argv, list) or len(argv) < 3 or not isinstance(argv[1], str):
        return None
    runner_path = argv[1]
    runner = Path(runner_path).name
    mode = argv[2]
    if runner == compiler.n3_confirmation.RUNNER_FILENAME:
        require(mode == "queue", "confirmation N3 descriptor has an unknown mode")
        model = "N3"
        run_id = None
    elif runner == compiler.d1_confirmation.RUNNER_FILENAME:
        require(mode in {"server-job", "simulator-job"},
                "confirmation D1 descriptor has an unknown mode")
        model = "D1"
        run_id = _argv_option(argv, "--run-id", str(descriptor.get("job_id")))
    else:
        return None
    expected_runner_path = (
        "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
        + runner
    )
    require(runner_path == expected_runner_path,
            "confirmation queue descriptor runner path changed")
    require(descriptor.get("source_commit") == study_commit,
            "confirmation queue descriptor exists at a different study commit")
    layout = _argv_option(argv, "--layout-pair-id", str(descriptor.get("job_id")))
    require(layout in compiler.LAYOUTS, "confirmation queue descriptor layout is invalid")
    return model, layout, str(mode), run_id


def _scan_confirmation_queue_jobs(
    *,
    state_dir: Path,
    study_commit: str,
    branch: str,
    schedules: Mapping[tuple[str, str], Any],
    selected: Mapping[tuple[str, str], Sequence[str]],
    raw_root: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    compiler.development._reject_symlink_components(Path(state_dir), "cluster queue state root")
    compiler.development._reject_symlink_components(Path(raw_root), "confirmation raw root")
    jobs_root = Path(state_dir).resolve() / "jobs"
    raw_root = Path(raw_root).resolve()
    require(jobs_root.is_dir() and not jobs_root.is_symlink(),
            "cluster queue jobs root is unavailable or unsafe")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in schedules
    }
    seen_ids: set[str] = set()
    for job_dir in sorted(jobs_root.iterdir(), key=lambda item: item.name):
        if not job_dir.is_dir() or job_dir.is_symlink():
            continue
        descriptor_path = job_dir / "descriptor.json"
        if not descriptor_path.exists():
            continue
        descriptor_identity = _regular_file_descriptor(
            descriptor_path, f"queue descriptor {job_dir.name}"
        )
        descriptor = compiler.load_json(descriptor_path, f"queue descriptor {job_dir.name}")
        matched = _matching_confirmation_job(descriptor, study_commit)
        if matched is None:
            continue
        model, layout, mode, run_id = matched
        key = (model, layout)
        require(model in BRANCH_MODELS[branch] and key in schedules,
                f"out-of-cohort confirmation descriptor exists: {job_dir.name}")
        job_id = descriptor.get("job_id")
        require(
            isinstance(job_id, str)
            and job_id == job_dir.name
            and job_id not in seen_ids
            and _argv_option(descriptor.get("argv", []), "--job-id", job_id) == job_id,
            f"confirmation queue descriptor/job-directory identity changed: {job_dir.name}",
        )
        seen_ids.add(job_id)
        claim_path = job_dir / "claim" / "owner.json"
        result_path = job_dir / "result.json"
        claim_identity = _regular_file_descriptor(claim_path, f"queue claim {job_id}")
        result_identity = _regular_file_descriptor(result_path, f"queue result {job_id}")
        result = compiler.load_json(result_path, f"queue result {job_id}")
        descendants = _live_process_group_members(result.get("child_pid"))
        require(not descendants, f"confirmation queue job retains live descendants: {job_id}")
        if model == "N3":
            runtime_path = job_dir / "publish" / compiler.n3_confirmation.AGGREGATE_FILENAME
            runtime_identity = _regular_file_descriptor(
                runtime_path, f"{job_id} N3 runtime receipt"
            )
            runtime_value = compiler.load_json(runtime_path, f"{job_id} N3 runtime receipt")
            attempt = Path(str(runtime_value.get("raw_attempt_root", "")))
            require(attempt.is_absolute(), f"{job_id} N3 runtime attempt root is invalid")
            attempt = compiler.development._under(
                attempt, raw_root, f"{job_id} N3 runtime attempt root"
            )
            attempt_counts = compiler._validate_count_object(
                runtime_value.get("counts"), model=model, layout=layout
            )
            cleanup_identity = _regular_file_descriptor(
                attempt / "cleanup.json", f"{job_id} N3 cleanup receipt"
            )
            process_terminal_identity = _regular_file_descriptor(
                attempt / "server" / "supervisor_exit.json",
                f"{job_id} N3 server terminal receipt",
            )
            protocol_identity = None
            failure_rows = _seal_failure_rows(
                aggregate={"counts": attempt_counts, "raw_attempt": attempt},
                schedule=schedules[key], model=model, layout=layout,
                simulator_worker_role=None,
            )
        elif mode == "server-job":
            runtime_path = job_dir / "publish" / "d1_behavioral_server_receipt.json"
            runtime_identity = _regular_file_descriptor(
                runtime_path, f"{job_id} D1 server runtime receipt"
            )
            cleanup_identity = None
            process_terminal_identity = None
            protocol_identity = None
            failure_rows = []
        else:
            runtime_path = (
                job_dir / "publish" / compiler.d1_confirmation.SIMULATOR_RECEIPT_FILENAME
            )
            runtime_identity = _regular_file_descriptor(
                runtime_path, f"{job_id} D1 simulator runtime receipt"
            )
            runtime_value = compiler.load_json(
                runtime_path, f"{job_id} D1 simulator runtime receipt"
            )
            protocol_path = (
                Path(raw_root).resolve() / "behavioral" / "confirmation" / "D1"
                / layout / "coordination" / str(run_id) / "simulator_terminal.json"
            )
            protocol_identity = _regular_file_descriptor(
                protocol_path, f"{job_id} D1 simulator protocol terminal"
            )
            cleanup_identity = None
            process_terminal_identity = None
            if isinstance(runtime_value.get("counts"), Mapping):
                attempt_value = runtime_value.get("raw_attempt_root")
                require(isinstance(attempt_value, str) and Path(attempt_value).is_absolute(),
                        f"{job_id} D1 runtime attempt root is invalid")
                attempt = compiler.development._under(
                    Path(attempt_value), raw_root, f"{job_id} D1 runtime attempt root"
                )
                attempt_counts = compiler._validate_count_object(
                    runtime_value.get("counts"), model=model, layout=layout
                )
                failure_rows = _seal_failure_rows(
                    aggregate={"counts": attempt_counts, "raw_attempt": attempt},
                    schedule=schedules[key], model=model, layout=layout,
                    simulator_worker_role=descriptor.get("role"),
                )
            else:
                failure_rows = []
        row = {
            "job_id": job_id,
            "model_id": model,
            "layout_pair_id": layout,
            "block_id": schedules[key].block_id,
            "mode": mode,
            "run_id": run_id,
            "selected_for_block_evidence": job_id in selected[key],
            "descriptor": descriptor_identity,
            "claim_owner": claim_identity,
            "result": result_identity,
            "live_descendant_pids": descendants,
            "runtime_terminal_evidence": {
                "runtime_receipt": runtime_identity,
                "cleanup_receipt": cleanup_identity,
                "nested_process_terminal": process_terminal_identity,
                "protocol_terminal": protocol_identity,
                "failure_cells": failure_rows,
            },
        }
        # Reuse the compiler's exact queue/claim/result gate before any seal is
        # made visible.  Absolute descriptors make the temporary close path
        # irrelevant.
        compiler._validate_queue_job(
            row,
            close_path=Path(state_dir).resolve() / ".cohort-close-preflight.json",
            model=model,
            layout=layout,
            block_id=schedules[key].block_id,
            study_commit=study_commit,
            expected_state_dir=Path(state_dir).resolve(),
        )
        grouped[key].append(row)
    require(seen_ids, "no confirmation queue descriptors exist at the frozen study commit")
    for key, ids in selected.items():
        available = {row["job_id"] for row in grouped[key]}
        require(set(ids) <= available,
                f"selected confirmation queue jobs are missing for {key[0]} {key[1]}")
    return grouped


def _descriptor_if_present(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _regular_file_descriptor(path, label)


def _single_source_video(cell_root: Path, label: str) -> dict[str, Any] | None:
    native = cell_root / "native_simulator"
    if not native.exists() and not native.is_symlink():
        return None
    require(native.is_dir() and not native.is_symlink(), f"{label} native video root is unsafe")
    entries = list(native.rglob("*"))
    require(not any(item.is_symlink() for item in entries), f"{label} native video tree has a symlink")
    videos = sorted(
        item for item in entries
        if item.is_file() and item.suffix.lower() == ".mp4" and item.stat().st_size > 0
    )
    require(len(videos) <= 1, f"{label} has ambiguous source videos")
    return None if not videos else _regular_file_descriptor(videos[0], f"{label} source video")


def _seal_failure_rows(
    *, aggregate: Mapping[str, Any], schedule: Any, model: str, layout: str,
    simulator_worker_role: str | None,
) -> list[dict[str, Any]]:
    counts = aggregate["counts"]
    first_failed = counts["completed_valid_behavioral_cells"]
    failed = counts["launched_behavioral_cells"] - first_failed
    require(failed <= 1, f"{model} {layout} has multiple failed cells in one indivisible block")
    rows: list[dict[str, Any]] = []
    for index in range(first_failed, first_failed + failed):
        cell_id = schedule.cell_ids[index]
        cell_root = compiler._cell_root(aggregate["raw_attempt"], index, cell_id)
        failure_path = cell_root / "technical_failure.json"
        failure_descriptor = _descriptor_if_present(failure_path, f"{cell_id} failure receipt")
        if failure_descriptor is None:
            require(
                counts["technically_invalid_behavioral_cells"] == 1
                and counts["right_censored_behavioral_cells"] == 0
                and counts["actual_behavioral_actions"] == 450 * first_failed
                and counts["actual_behavioral_model_requests"]
                == compiler.MODEL_FULL_REQUESTS[model] * first_failed,
                f"{cell_id} missing failure receipt is not an authenticated zero-evidence crash",
            )
            status = "technical_failure"
            actions = 0
            requests = 0
            artifact_state = "absent_verified_at_cohort_close"
            absence_reason = "child_terminated_before_failure_receipt"
        else:
            failure = compiler.load_json(failure_path, f"{cell_id} failure receipt")
            status = failure.get("status")
            actions = failure.get("actions_executed")
            requests = failure.get("request_count")
            require(status in {"safety_abort", "technical_failure"}
                    and type(actions) is int and type(requests) is int,
                    f"{cell_id} failure receipt status/counts are invalid")
            artifact_state = "present_hash_bound"
            absence_reason = None
        completion = _descriptor_if_present(
            cell_root / "recording" / "completion.json", f"{cell_id} adapter completion"
        )
        journal = _descriptor_if_present(
            cell_root / "recording" / "events.partial.jsonl", f"{cell_id} adapter journal"
        )
        video = _single_source_video(cell_root, cell_id)
        context_terminal = None
        if failure_descriptor is not None:
            context_value = failure.get("server_context_terminal")
            require(
                status != "safety_abort" or isinstance(context_value, Mapping),
                f"{cell_id} safety_abort lacks native server-context terminal evidence",
            )
            if context_value is not None:
                try:
                    if model == "N3":
                        validated_failure = (
                            compiler.n3_confirmation.validate_failed_confirmation_cell(
                                failure_path,
                                condition_index=index,
                                block=schedule,
                            )
                        )
                    else:
                        require(
                            isinstance(simulator_worker_role, str)
                            and bool(simulator_worker_role),
                            f"{cell_id} D1 terminal validation lacks its queue role",
                        )
                        with compiler.d1_confirmation._configured_for_validation(
                            schedule, simulator_worker_role
                        ):
                            validated_failure = (
                                compiler.d1_confirmation.validate_failed_confirmation_cell(
                                    failure_path,
                                    condition_index=index,
                                    block=schedule,
                                )
                            )
                except ConfirmationCompilerJobError:
                    raise
                except Exception as error:
                    raise ConfirmationCompilerJobError(
                        f"{cell_id} native terminal validation failed: {error}"
                    ) from error
                require(
                    validated_failure == failure,
                    f"{cell_id} native terminal validator changed its failure bytes",
                )
                context_path = Path(str(context_value.get("path", "")))
                context_terminal = _verify_file(
                    context_path,
                    str(context_value.get("sha256")),
                    f"{cell_id} model-context terminal",
                )
                require(context_value.get("bytes") == context_terminal["bytes"],
                        f"{cell_id} model-context terminal byte count changed")
        rows.append({
            "cell_id": cell_id,
            "condition_index": index,
            "status": status,
            "actions_executed": actions,
            "request_count": requests,
            "artifact_state": artifact_state,
            "failure_receipt": failure_descriptor,
            "absence_reason": absence_reason,
            "adapter_completion": completion,
            "adapter_journal": journal,
            "source_video": video,
            "context_terminal": context_terminal,
        })
    return rows


def _audit_d1_protocol(
    *, raw_root: Path
) -> tuple[list[str], list[str]]:
    now_ns = time.time_ns()
    unexpired: list[str] = []
    unterminated: list[str] = []
    d1_root = Path(raw_root).resolve() / "behavioral" / "confirmation" / "D1"
    roots = sorted(d1_root.glob("C*/coordination/*")) if d1_root.is_dir() else []
    seen_run_ids: set[str] = set()
    for root in roots:
        require(root.is_dir() and not root.is_symlink(),
                f"D1 coordination root is unsafe: {root}")
        run_id = root.name
        require(run_id not in seen_run_ids,
                f"D1 run ID is reused across confirmation layouts: {run_id}")
        seen_run_ids.add(run_id)
        for name in ("server_lease.json", "simulator_lease.json"):
            path = root / name
            if path.is_file():
                value = compiler.load_json(path, f"D1 {run_id} {name}")
                expiry = value.get("expires_unix_ns")
                require(type(expiry) is int, f"D1 {run_id} lease expiry is invalid")
                if expiry > now_ns:
                    unexpired.append(str(path.resolve()))
        claim = root / "simulator_claim.json"
        terminal = root / "simulator_terminal.json"
        if claim.is_file() and not terminal.is_file():
            unterminated.append(str(claim.resolve()))
            continue
        if claim.is_file():
            claim_identity = _regular_file_descriptor(claim, f"D1 {run_id} simulator claim")
            claim_value = compiler.load_json(claim, f"D1 {run_id} simulator claim")
            require(
                claim_value.get("schema_version") == compiler.d1_confirmation.pilot.SIMULATOR_CLAIM_SCHEMA
                and claim_value.get("status") == "claimed"
                and claim_value.get("run_id") == run_id,
                f"D1 {run_id} simulator claim identity changed",
            )
        else:
            claim_identity = None
        if terminal.is_file():
            terminal_value = compiler.load_json(terminal, f"D1 {run_id} simulator terminal")
            require(
                terminal_value.get("schema_version")
                == compiler.d1_confirmation.pilot.SIMULATOR_TERMINAL_SCHEMA
                and terminal_value.get("run_id") == run_id
                and terminal_value.get("status") in {"passed", "technical_failure"}
                and terminal_value.get("all_simulator_children_reaped") is True
                and terminal_value.get("safe_for_server_shutdown") is True
                and terminal_value.get("simulator_claim_sha256")
                == (None if claim_identity is None else claim_identity["sha256"]),
                f"D1 {run_id} simulator terminal does not close its claim",
            )
    return sorted(unexpired), sorted(unterminated)


def _wait_for_seal_admission_cutoff(context: Any) -> dict[str, Any]:
    """Wait inside an already-authenticated claim until admission is closed.

    The queue cannot claim a sealer after its cutoff, and global shutdown kills
    active children.  Therefore the only executable transition is: release the
    sealer alone with a short future cutoff, claim it before that cutoff, keep
    the control bytes immutable while it waits, then seal after expiration.
    """

    state_dir = context.job_dir.parent.parent
    control_path = state_dir / "control.json"
    control_identity = _regular_file_descriptor(
        control_path, "pre-seal cluster queue control state"
    )
    control_payload = control_path.read_bytes()
    control = compiler.load_json(control_path, "pre-seal cluster queue control state")
    _exact_keys(
        control,
        {
            "namespace", "control_commit", "control_generation",
            "admission_deadline_unix", "shutdown", "active_job_ids",
        },
        "pre-seal cluster queue control state",
    )
    claim = compiler.load_json(
        Path(context.claim_identity["path"]), "pre-seal queue claim owner"
    )
    require(
        _file_descriptor(Path(context.descriptor_identity["path"]))
        == context.descriptor_identity
        and _file_descriptor(Path(context.claim_identity["path"]))
        == context.claim_identity,
        "sealer descriptor or claim changed after queue authentication",
    )
    deadline = control.get("admission_deadline_unix")
    now = time.time()
    require(
        control.get("namespace") == compiler.NAMESPACE
        and control.get("shutdown") is False
        and control.get("active_job_ids") == [context.job_id]
        and control.get("control_commit") == claim.get("control_commit")
        and control.get("control_generation") == claim.get("control_generation")
        and type(deadline) in {int, float}
        and not isinstance(deadline, bool)
        and math.isfinite(float(deadline))
        and float(deadline) > 0
        and type(claim.get("claimed_unix")) in {int, float}
        and float(claim["claimed_unix"]) < float(deadline),
        "sealer must be the sole active job claimed before an immutable admission cutoff",
    )
    remaining = float(deadline) - now
    require(
        remaining <= MAX_SEAL_ADMISSION_WAIT_SECONDS,
        "sealer admission cutoff is not a bounded near-term deadline",
    )
    monotonic_deadline = time.monotonic() + max(0.0, remaining) + 2.0
    while time.time() < float(deadline):
        require(
            time.monotonic() <= monotonic_deadline,
            "wall clock did not reach the immutable admission cutoff in time",
        )
        require(
            control_path.read_bytes() == control_payload,
            "cluster queue control changed while the claimed sealer waited",
        )
        time.sleep(min(0.25, max(0.001, float(deadline) - time.time())))
    require(
        control_path.read_bytes() == control_payload
        and _file_descriptor(control_path) == control_identity,
        "cluster queue control changed at the sealer admission boundary",
    )
    require(
        _file_descriptor(Path(context.descriptor_identity["path"]))
        == context.descriptor_identity
        and _file_descriptor(Path(context.claim_identity["path"]))
        == context.claim_identity,
        "sealer descriptor or claim changed at the admission boundary",
    )
    conservative_remaining = (
        QUEUE_MAX_WALL_SECONDS - max(0.0, time.time() - float(claim["claimed_unix"]))
    )
    require(
        conservative_remaining >= MIN_POST_CUTOFF_COMPILER_WALL_SECONDS,
        "claimed sealer lacks the reserved post-cutoff compiler wall budget",
    )
    return control_identity


def _atomic_directory(output_dir: Path, files: Mapping[str, bytes]) -> None:
    output = Path(output_dir)
    require(not output.exists(), f"refusing to overwrite immutable output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            target = temporary / relative
            require(target.resolve().is_relative_to(temporary.resolve()),
                    "cohort-seal output path escapes its temporary root")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def seal_confirmation_cohort(
    *,
    seal_plan_path: Path,
    seal_plan_sha256: str,
    state_dir: Path,
    source_root: Path,
    raw_root: Path,
    output_dir: Path,
    producer_context: Any,
    expected_control_identity: Mapping[str, Any],
    camera_id: str,
) -> dict[str, Any]:
    """Seal queue/PVC terminal state without releasing or executing work."""

    source_supplied = Path(source_root)
    raw_supplied = Path(raw_root)
    state_supplied = Path(state_dir)
    compiler.development._reject_symlink_components(source_supplied, "cohort-seal source root")
    compiler.development._reject_symlink_components(raw_supplied, "cohort-seal raw root")
    compiler.development._reject_symlink_components(state_supplied, "cohort-seal queue root")
    source_root = source_supplied.resolve()
    raw_root = raw_supplied.resolve()
    state_dir = state_supplied.resolve()
    require(source_root.is_dir() and raw_root.is_dir() and state_dir.is_dir(),
            "cohort-seal source/raw/queue state root is unavailable")
    require(
        producer_context is not None
        and producer_context.source_root == source_root
        and producer_context.job_dir.parent.parent == state_dir
        and producer_context.study_commit,
        "cohort seal lacks an authenticated queue producer context",
    )
    plan_identity, plan = _load_seal_plan(
        seal_plan_path, seal_plan_sha256, source_root=source_root
    )
    branch = str(plan["cohort_branch"])
    study_commit = str(plan["study_commit"])
    require(
        producer_context.study_commit == study_commit
        and Path(output_dir).resolve()
        == producer_context.job_dir / SEAL_BUNDLE_RELATIVE,
        "cohort seal producer source/output identity changed",
    )
    require(camera_id in compiler.development.PRIMARY_CAMERA_CHOICES,
            "cohort seal compiler camera is invalid")
    _verify_clean_source(source_root, study_commit)
    require(
        Path(__file__).resolve()
        == source_root
        / "workshops/corl2026_world_models/experiments/forecast_layout/"
        "confirmation_evidence_compiler_jobs.py",
        "cohort sealer is not executing from the frozen staged source",
    )
    try:
        release = compiler.freeze.validate_release_freeze(
            Path(plan["development_release_freeze"]["path"]),
            plan["development_release_freeze"]["sha256"],
        )
    except Exception as error:
        raise ConfirmationCompilerJobError(
            f"cohort-seal confirmation release failed: {error}"
        ) from error
    require(
        release.get("cohort_branch") == branch
        and release.get("qualified_model_ids") == list(BRANCH_MODELS[branch]),
        "cohort-seal release branch/model set changed",
    )
    schedules: dict[tuple[str, str], Any] = {}
    for model in BRANCH_MODELS[branch]:
        for layout in compiler.LAYOUTS:
            try:
                schedule = (
                    compiler.n3_confirmation.load_confirmation_block(source_root, layout)
                    if model == "N3"
                    else compiler.d1_confirmation.load_confirmation_block(source_root, layout)
                )
            except Exception as error:
                raise ConfirmationCompilerJobError(
                    f"cohort-seal {model} {layout} schedule failed: {error}"
                ) from error
            require(
                compiler.development.file_descriptor(schedule.schedule_path)
                == plan["prepared_schedule"],
                f"cohort-seal {model} {layout} prepared schedule changed",
            )
            schedules[(model, layout)] = schedule

    release_lock = state_dir / "release.lock"
    require(release_lock.exists() and release_lock.is_file(),
            "cluster queue release lock is unavailable")
    with ExitStack() as locks:
        locks.enter_context(_exclusive_lock(release_lock, "cluster queue release lock"))
        require(
            _file_descriptor(Path(producer_context.descriptor_identity["path"]))
            == producer_context.descriptor_identity
            and _file_descriptor(Path(producer_context.claim_identity["path"]))
            == producer_context.claim_identity,
            "cohort-seal producer descriptor/claim changed under release lock",
        )
        if "D1" in BRANCH_MODELS[branch]:
            require(
                compiler.d1_confirmation.GLOBAL_D1_SERVER_LOCK.is_file()
                and not compiler.d1_confirmation.GLOBAL_D1_SERVER_LOCK.is_symlink(),
                "global D1 server lock does not exist as a stable regular file",
            )
            locks.enter_context(_exclusive_lock(
                compiler.d1_confirmation.GLOBAL_D1_SERVER_LOCK,
                "global D1 server lock",
            ))
        control_path = state_dir / "control.json"
        control_identity = _regular_file_descriptor(
            control_path, "cluster queue control state"
        )
        require(
            control_identity == dict(expected_control_identity),
            "cluster queue control changed after the sealer admission cutoff",
        )
        control = compiler.load_json(control_path, "cluster queue control state")
        _exact_keys(
            control,
            {
                "namespace", "control_commit", "control_generation",
                "admission_deadline_unix", "shutdown", "active_job_ids",
            },
            "cluster queue control state",
        )
        active_ids = control.get("active_job_ids")
        control_deadline = control.get("admission_deadline_unix")
        require(control.get("namespace") == compiler.NAMESPACE
                and isinstance(control.get("control_commit"), str)
                and COMMIT_RE.fullmatch(control["control_commit"]) is not None
                and type(control.get("control_generation")) is int
                and control["control_generation"] >= 0
                and type(control.get("shutdown")) is bool
                and isinstance(active_ids, list)
                and active_ids == list(dict.fromkeys(active_ids))
                and all(isinstance(item, str) and compiler.SAFE_COMPONENT_RE.fullmatch(item)
                        for item in active_ids)
                and (
                    control_deadline is None
                    or (
                        type(control_deadline) in {int, float}
                        and not isinstance(control_deadline, bool)
                        and math.isfinite(float(control_deadline))
                        and float(control_deadline) > 0
                    )
                ),
                "cluster queue control state is invalid")
        require(
            active_ids == [producer_context.job_id],
            "cohort seal was not the sole active queue admission",
        )
        grouped = _scan_confirmation_queue_jobs(
            state_dir=state_dir,
            study_commit=study_commit,
            branch=branch,
            schedules=schedules,
            selected=plan["selections"],
            raw_root=raw_root,
        )
        all_ids = sorted(
            row["job_id"] for rows in grouped.values() for row in rows
        )
        unexpired, unterminated = (
            _audit_d1_protocol(raw_root=raw_root)
            if "D1" in BRANCH_MODELS[branch] else ([], [])
        )
        require(not unexpired, "D1 confirmation leases remain unexpired")
        require(not unterminated, "D1 confirmation protocol claims remain unterminated")

        block_rows: list[dict[str, Any]] = []
        aggregates: dict[tuple[str, str], dict[str, Any]] = {}
        for key in sorted(schedules):
            model, layout = key
            schedule = schedules[key]
            queue_rows = sorted(grouped[key], key=lambda row: row["job_id"])
            lifecycles = compiler._validate_block_attempt_lifecycles(
                queue_rows=queue_rows,
                close_path=Path(output_dir).resolve() / "cohort_close_receipt.json",
                model=model,
                layout=layout,
                schedule=schedule,
                study_commit=study_commit,
                source_root=source_root,
                raw_root=raw_root,
                release_descriptor=plan["development_release_freeze"],
                release_freeze=release,
                queue_state_dir=state_dir,
            )
            selected_ids = lifecycles["selected_queue_job_ids"]
            require(
                selected_ids == list(plan["selections"][key]),
                f"{model} {layout} seal-plan selection is not the derived retry tail",
            )
            aggregate = lifecycles["aggregate"]
            aggregates[key] = aggregate
            evidence_form = aggregate["evidence_form"]
            aggregate_descriptor = aggregate["descriptor"]
            selected_aggregate_job = next(
                row for row in queue_rows
                if row["job_id"] in selected_ids and row["mode"] != "server-job"
            )
            failure_rows = selected_aggregate_job["runtime_terminal_evidence"]["failure_cells"]
            d1_pair = lifecycles["d1_pair"]
            block_rows.append({
                "model_id": model,
                "layout_pair_id": layout,
                "block_id": schedule.block_id,
                "schedule_row_sha256": compiler.sha256_bytes(
                    compiler.canonical_bytes(schedule.schedule_row)
                ),
                "evidence_form": evidence_form,
                "receipt": aggregate_descriptor,
                "failure_cells": failure_rows,
                "queue_jobs": queue_rows,
                "selected_queue_job_ids": selected_ids,
                "d1_pair": d1_pair,
            })

        sealed_at = _utc_now()
        admission_barrier = _confirmation_admission_barrier(
            control, sealed_at=sealed_at
        )
        control_payload = control_path.read_bytes()
        require(
            compiler.sha256_bytes(control_payload) == control_identity["sha256"]
            and len(control_payload) == control_identity["bytes"],
            "cluster queue control state changed during cohort seal",
        )
        seal_plan_payload = Path(plan_identity["path"]).read_bytes()
        sealer_payload = Path(__file__).resolve().read_bytes()
        active_producer = _producer_queue_job(producer_context)
        control_descriptor = {
            "path": "control_state.json",
            "sha256": compiler.sha256_bytes(control_payload),
            "bytes": len(control_payload),
        }
        close = sign_document({
            "schema_version": compiler.COHORT_CLOSE_SCHEMA,
            "study_id": STUDY_ID,
            "namespace": compiler.NAMESPACE,
            "status": "cohort_terminal_no_live_claims_or_descendants",
            "cohort_branch": branch,
            "study_commit": study_commit,
            "sealed_at_utc": sealed_at,
            "development_release_freeze": plan["development_release_freeze"],
            "prepared_schedule": plan["prepared_schedule"],
            "seal_plan": {
                "path": "seal_plan.json",
                "sha256": compiler.sha256_bytes(seal_plan_payload),
                "bytes": len(seal_plan_payload),
            },
            "sealer_source": {
                "path": "sealer_source.py",
                "sha256": compiler.sha256_bytes(sealer_payload),
                "bytes": len(sealer_payload),
            },
            "producer_queue_job": active_producer,
            "queue_state_dir": str(state_dir),
            "raw_root": str(raw_root),
            "compiler_camera_id": camera_id,
            "control_state": control_descriptor,
            "confirmation_admission_barrier": admission_barrier,
            "release_lock_held_during_seal": True,
            "all_confirmation_queue_job_ids": all_ids,
            "blocks": block_rows,
            "nonterminal_claim_job_ids": [],
            "live_descendant_processes": [],
            "d1_global_server_lock_path": (
                str(compiler.d1_confirmation.GLOBAL_D1_SERVER_LOCK.resolve())
                if "D1" in BRANCH_MODELS[branch] else None
            ),
            "unexpired_d1_lease_paths": [],
            "unterminated_d1_protocol_claim_paths": [],
        })
        close_payload = pretty_bytes(close)
        close_descriptor = {
            "path": "cohort_close_receipt.json",
            "sha256": compiler.sha256_bytes(close_payload),
            "bytes": len(close_payload),
        }
        index = sign_document({
            "schema_version": BLOCK_INDEX_SCHEMA,
            "study_id": STUDY_ID,
            "cohort_branch": branch,
            "study_commit": study_commit,
            "development_release_freeze": plan["development_release_freeze"],
            "prepared_schedule": plan["prepared_schedule"],
            "cohort_close_receipt": close_descriptor,
            "block_receipts": block_rows,
        })
        _atomic_directory(
            output_dir,
            {
                "control_state.json": control_payload,
                "seal_plan.json": seal_plan_payload,
                "sealer_source.py": sealer_payload,
                "cohort_close_receipt.json": close_payload,
                "terminal_block_index.json": pretty_bytes(index),
            },
        )
        block_index_path = Path(output_dir).resolve() / "terminal_block_index.json"
        manifest_path = producer_context.job_dir / MANIFEST_RELATIVE
        compiler_output = producer_context.job_dir / RUN_BUNDLE_RELATIVE
        manifest = build_compiler_manifest(
            block_index_path=block_index_path,
            block_index_sha256=sha256_file(block_index_path),
            source_root=source_root,
            raw_root=raw_root,
            study_commit=study_commit,
            inventory_finalized_at=sealed_at,
            camera_id=camera_id,
            release_freeze_path=Path(plan["development_release_freeze"]["path"]),
            release_freeze_sha256=plan["development_release_freeze"]["sha256"],
            output=manifest_path,
            active_seal_producer=active_producer,
        )
        compiled = compiler.compile_manifest(
            manifest_path,
            sha256_file(manifest_path),
            compiler_output,
            active_seal_producer=active_producer,
        )
        require(
            control_path.read_bytes() == control_payload
            and _file_descriptor(control_path) == control_identity
            and _file_descriptor(Path(producer_context.descriptor_identity["path"]))
            == producer_context.descriptor_identity
            and _file_descriptor(Path(producer_context.claim_identity["path"]))
            == producer_context.claim_identity,
            "queue control/producer lineage changed during locked compilation",
        )
        _verify_clean_source(source_root, study_commit)
    return {
        "status": "cohort_terminal_no_live_claims_or_descendants",
        "sealed_at_utc": sealed_at,
        "cohort_branch": branch,
        "study_commit": study_commit,
        "queue_jobs": len(all_ids),
        "blocks": len(block_rows),
        "cohort_close_receipt": _file_descriptor(
            Path(output_dir).resolve() / "cohort_close_receipt.json"
        ),
        "terminal_block_index": _file_descriptor(
            Path(output_dir).resolve() / "terminal_block_index.json"
        ),
        "compiler_input_manifest": _file_descriptor(manifest_path),
        "compiler_receipt": _file_descriptor(
            compiler_output / "compiler_receipt.json"
        ),
        "compiler_counts": dict(compiled["counts"]),
        "confirmation_released": False,
        "labels_created": False,
        "scientific_results_computed": False,
    }


def run_seal_job(
    *,
    source_root: Path,
    study_commit: str,
    job_dir: Path,
    job_id: str,
    seal_plan_path: Path,
    seal_plan_sha256: str,
    raw_root: Path,
    camera_id: str,
) -> dict[str, Any]:
    """Seal and compile source evidence inside one authenticated queue claim."""

    context = _queue_context(
        command="seal-cohort",
        source_root=source_root,
        study_commit=study_commit,
        job_dir=job_dir,
        job_id=job_id,
        seal_plan_path=seal_plan_path,
        seal_plan_sha256=seal_plan_sha256,
        raw_root=raw_root,
        camera_id=camera_id,
    )
    control_identity = _wait_for_seal_admission_cutoff(context)
    raw, publish = queue._prepare_output_directories(context.job_dir)
    output_dir = context.job_dir / SEAL_BUNDLE_RELATIVE
    require(output_dir.parent == raw, "cohort seal escaped its job-local raw directory")
    result = seal_confirmation_cohort(
        seal_plan_path=seal_plan_path,
        seal_plan_sha256=seal_plan_sha256,
        state_dir=context.job_dir.parent.parent,
        source_root=context.source_root,
        raw_root=raw_root,
        output_dir=output_dir,
        producer_context=context,
        expected_control_identity=control_identity,
        camera_id=camera_id,
    )
    job_receipt = sign_document({
        "schema_version": SEAL_JOB_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "status": result["status"],
        "job_id": context.job_id,
        "job_dir": str(context.job_dir),
        "study_commit": context.study_commit,
        "queue_role": context.role,
        "worker_id": context.worker_id,
        "runtime_identity": {
            "hostname": context.hostname,
            "pod_uid": context.pod_uid,
            "pid": os.getpid(),
        },
        "queue_descriptor": context.descriptor_identity,
        "queue_claim": context.claim_identity,
        "queue_result_lineage": {
            "path": str(context.job_dir / "result.json"),
            "availability": "written_by_queue_worker_after_wrapper_exit",
            "job_id": context.job_id,
            "source_commit": context.study_commit,
            "descriptor_sha256": context.descriptor_identity["sha256"],
            "worker_id": context.worker_id,
        },
        "seal_plan": _file_descriptor(Path(seal_plan_path).resolve()),
        "cohort_close_receipt": result["cohort_close_receipt"],
        "terminal_block_index": result["terminal_block_index"],
        "compiler_input_manifest": result["compiler_input_manifest"],
        "compiler_receipt": result["compiler_receipt"],
        "compiler_counts": result["compiler_counts"],
        "science_counts": _zero_science_counts(),
        "queue_mutations": 0,
        "confirmation_released": False,
        "labels_created": False,
        "scientific_results_computed": False,
        "claim_boundary": (
            "Authenticated no-release terminal queue/PVC snapshot only; the job "
            "starts no model, simulator, reset, episode, request, action, or label."
        ),
    })
    queue.immutable_json(
        publish / SEAL_RECEIPT_NAME, job_receipt, maximum_bytes=512 * 1024
    )
    return job_receipt


def _validate_block_index(
    path: Path, digest: str, *, source_root: Path, raw_root: Path,
    active_seal_producer: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _verify_file(path, digest, "terminal block index")
    value = compiler.load_json(Path(identity["path"]), "terminal block index")
    _exact_keys(value, BLOCK_INDEX_KEYS, "terminal block index")
    require(value.get("schema_version") == BLOCK_INDEX_SCHEMA,
            "terminal block index schema changed")
    require(value.get("study_id") == STUDY_ID,
            "terminal block index study changed")
    verify_signed(value, "terminal block index")
    branch = value.get("cohort_branch")
    require(branch in BRANCH_MODELS, "terminal block index branch is invalid")
    study_commit = value.get("study_commit")
    require(isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
            "terminal block index study commit is invalid")
    release_identity, _ = compiler._exact_file_reference(
        value.get("development_release_freeze"),
        base=Path(identity["path"]).parent,
        label="terminal-index development release freeze",
    )
    schedule_identity, _ = compiler._exact_file_reference(
        value.get("prepared_schedule"),
        base=Path(identity["path"]).parent,
        label="terminal-index prepared schedule",
    )
    close_identity, close_path = compiler._exact_file_reference(
        value.get("cohort_close_receipt"),
        base=Path(identity["path"]).parent,
        label="terminal-index cohort-close receipt",
    )
    try:
        _close_descriptor, _close_path, close = compiler._load_cohort_close(
            close_identity,
            manifest_base=Path(identity["path"]).parent,
            branch=str(branch),
            study_commit=study_commit,
            release_descriptor=release_identity,
            source_root=Path(source_root).resolve(),
            active_seal_producer=active_seal_producer,
        )
    except Exception as error:
        raise ConfirmationCompilerJobError(
            f"terminal-index cohort-close receipt failed: {error}"
        ) from error
    require(close_path == _close_path, "terminal-index cohort-close path changed")
    require(
        Path(str(close.get("raw_root"))).resolve() == Path(raw_root).resolve(),
        "terminal-index cohort-close raw root differs from compiler input",
    )
    require(close.get("prepared_schedule") == schedule_identity,
            "terminal-index prepared schedule differs from cohort close")
    rows = value.get("block_receipts")
    require(isinstance(rows, list), "terminal block index rows are invalid")
    expected = {
        (model, f"C{index:02d}")
        for model in BRANCH_MODELS[str(branch)]
        for index in range(1, 25)
    }
    observed = set()
    normalized = []
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"terminal block row {index} is invalid")
        _exact_keys(row, BLOCK_ROW_KEYS, f"terminal block row {index}")
        key = (row.get("model_id"), row.get("layout_pair_id"))
        require(key in expected and key not in observed,
                f"terminal block row {index} identity is invalid or duplicated")
        observed.add(key)
        require(row.get("block_id") is not None
                and row.get("evidence_form") in {
                    compiler.FULL_AGGREGATE_FORM, compiler.D1_ZERO_LAUNCH_FORM
                }, f"terminal block row {index} closure metadata is invalid")
        descriptor = row.get("receipt")
        require(isinstance(descriptor, Mapping), f"terminal block row {index} lacks a receipt")
        receipt_identity = _verify_file(
            Path(str(descriptor.get("path"))),
            str(descriptor.get("sha256")),
            f"{key[0]} {key[1]} terminal block receipt",
        )
        require(descriptor.get("bytes") == receipt_identity["bytes"],
                f"{key[0]} {key[1]} terminal block byte count changed")
        normalized.append({**dict(row), "receipt": receipt_identity})
    require(observed == expected, "terminal block index does not exactly cover its cohort")
    require(value.get("block_receipts") == close.get("blocks"),
            "terminal block index differs from its cohort-close ledger")
    return identity, {
        **value,
        "development_release_freeze": release_identity,
        "prepared_schedule": schedule_identity,
        "cohort_close_receipt": close_identity,
        "block_receipts": normalized,
    }


def build_compiler_manifest(
    *,
    block_index_path: Path,
    block_index_sha256: str,
    source_root: Path,
    raw_root: Path,
    study_commit: str,
    inventory_finalized_at: str,
    camera_id: str,
    release_freeze_path: Path,
    release_freeze_sha256: str,
    output: Path,
    active_seal_producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require(COMMIT_RE.fullmatch(study_commit) is not None, "study commit is invalid")
    compiler.development._reject_symlink_components(Path(source_root), "compiler source root")
    compiler.development._reject_symlink_components(Path(raw_root), "compiler raw root")
    source_root = Path(source_root).resolve()
    raw_root = Path(raw_root).resolve()
    require(source_root.is_dir() and raw_root.is_dir(), "source/raw root is unavailable")
    _verify_clean_source(source_root, study_commit)
    index_identity, index = _validate_block_index(
        block_index_path, block_index_sha256,
        source_root=source_root, raw_root=raw_root,
        active_seal_producer=active_seal_producer,
    )
    release_identity = _verify_file(
        release_freeze_path, release_freeze_sha256, "development confirmation-release freeze"
    )
    try:
        release = compiler.freeze.validate_release_freeze(
            Path(release_identity["path"]), release_identity["sha256"]
        )
    except Exception as error:
        raise ConfirmationCompilerJobError(f"confirmation release freeze failed: {error}") from error
    require(release.get("cohort_branch") == index["cohort_branch"],
            "block index branch differs from confirmation release")
    require(index["study_commit"] == study_commit,
            "block index commit differs from requested compiler commit")
    require(index["development_release_freeze"] == release_identity,
            "block index release differs from requested confirmation release")
    require(
        compiler.development.file_descriptor(
            source_root
            / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
        ) == index["prepared_schedule"],
        "block index prepared schedule differs from staged source",
    )
    compiler._validate_rfc3339_utc(inventory_finalized_at, "inventory_finalized_at")
    close = compiler.load_json(
        Path(index["cohort_close_receipt"]["path"]), "compiler cohort-close receipt"
    )
    require(inventory_finalized_at == close.get("sealed_at_utc"),
            "compiler inventory time differs from authoritative cohort close")
    require(camera_id in compiler.development.PRIMARY_CAMERA_CHOICES,
            "camera is not an original recorded camera")
    result = sign_document({
        "schema_version": INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "cohort_branch": index["cohort_branch"],
        "inventory_finalized_at": inventory_finalized_at,
        "source_root": str(source_root),
        "raw_root": str(raw_root),
        "camera_id": camera_id,
        "study_commit": study_commit,
        "development_release_freeze": release_identity,
        "cohort_close_receipt": index["cohort_close_receipt"],
        "block_receipts": index["block_receipts"],
    })
    # The external index remains separately hash-bound in the caller's durable
    # job ledger; the compiler input itself includes and signs every normalized
    # terminal receipt rather than trusting that wrapper transitively.
    _ = index_identity
    _atomic_json(output, result)
    return result


def _verify_clean_source(source_root: Path, study_commit: str) -> None:
    require(COMMIT_RE.fullmatch(study_commit) is not None, "study commit is invalid")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ConfirmationCompilerJobError(f"cannot verify staged source: {error}") from error
    require(head == study_commit, "staged source HEAD differs from compiler input")
    require(not dirty.strip(), "staged source is dirty")


def _zero_science_counts() -> dict[str, int]:
    return {
        "model_runtime_loads": 0,
        "model_servers_started": 0,
        "model_requests_issued_by_job": 0,
        "simulator_processes_started": 0,
        "physical_resets": 0,
        "robot_episodes": 0,
        "behavioral_actions_executed_by_job": 0,
        "behavioral_cells_launched_by_job": 0,
        "labels_created_by_job": 0,
        "confirmation_jobs_released_by_job": 0,
    }


def _queue_path_argument(
    path: Path, *, source_root: Path, job_dir: Path, state_dir: Path
) -> str:
    """Invert the queue worker's three exact path substitutions.

    This is intentionally lexical: resolving an input first could conceal a
    symlink or make a descriptor with a substituted path appear equivalent to
    the bytes the worker actually executed.
    """

    lexical = Path(os.path.abspath(os.fspath(path)))
    roots = (
        (Path(source_root), "{source_root}"),
        (Path(job_dir), "{job_dir}"),
        (Path(state_dir), "{state_dir}"),
    )
    for root, token in roots:
        root = Path(os.path.abspath(os.fspath(root)))
        if lexical == root:
            return token
        if lexical.is_relative_to(root):
            return token + "/" + lexical.relative_to(root).as_posix()
    return str(lexical)


def _queue_argv(
    *,
    command: str,
    source_root: Path,
    study_commit: str,
    job_dir: Path,
    job_id: str,
    manifest_path: Path | None = None,
    manifest_sha256: str | None = None,
    seal_plan_path: Path | None = None,
    seal_plan_sha256: str | None = None,
    raw_root: Path | None = None,
    block_index_path: Path | None = None,
    block_index_sha256: str | None = None,
    inventory_finalized_at: str | None = None,
    camera_id: str | None = None,
    release_freeze_path: Path | None = None,
    release_freeze_sha256: str | None = None,
) -> list[str]:
    state_dir = Path(job_dir).resolve().parent.parent
    argv = [
        "/usr/bin/python3",
        "{source_root}/" + THIS_RELATIVE.as_posix(),
        command,
        "--source-root", "{source_root}",
        "--study-commit", study_commit,
        "--job-dir", "{job_dir}",
        "--job-id", job_id,
    ]
    if command == "run":
        require(manifest_path is not None and manifest_sha256 is not None,
                "compiler queue manifest arguments are incomplete")
        argv.extend([
            "--manifest",
            _queue_path_argument(
                manifest_path, source_root=source_root, job_dir=job_dir,
                state_dir=state_dir,
            ),
            "--manifest-sha256", manifest_sha256,
        ])
    elif command == "build-manifest":
        require(
            block_index_path is not None
            and block_index_sha256 is not None
            and raw_root is not None
            and inventory_finalized_at is not None
            and camera_id is not None
            and release_freeze_path is not None
            and release_freeze_sha256 is not None,
            "manifest-builder queue arguments are incomplete",
        )
        argv.extend([
            "--block-index",
            _queue_path_argument(
                block_index_path, source_root=source_root, job_dir=job_dir,
                state_dir=state_dir,
            ),
            "--block-index-sha256", block_index_sha256,
            "--raw-root",
            _queue_path_argument(
                raw_root, source_root=source_root, job_dir=job_dir,
                state_dir=state_dir,
            ),
            "--inventory-finalized-at", inventory_finalized_at,
            "--camera-id", camera_id,
            "--release-freeze",
            _queue_path_argument(
                release_freeze_path, source_root=source_root, job_dir=job_dir,
                state_dir=state_dir,
            ),
            "--release-freeze-sha256", release_freeze_sha256,
        ])
    elif command == "seal-cohort":
        require(
            seal_plan_path is not None
            and seal_plan_sha256 is not None
            and raw_root is not None
            and camera_id is not None,
            "cohort-seal queue arguments are incomplete",
        )
        argv.extend([
            "--seal-plan",
            _queue_path_argument(
                seal_plan_path, source_root=source_root, job_dir=job_dir,
                state_dir=state_dir,
            ),
            "--seal-plan-sha256", seal_plan_sha256,
            "--raw-root",
            _queue_path_argument(
                raw_root, source_root=source_root, job_dir=job_dir,
                state_dir=state_dir,
            ),
            "--camera-id", camera_id,
        ])
    else:  # pragma: no cover - callers expose a closed command set.
        raise ConfirmationCompilerJobError("unknown authenticated queue command")
    return argv


def _queue_context(
    *,
    command: str,
    source_root: Path,
    study_commit: str,
    job_dir: Path,
    job_id: str,
    manifest_path: Path | None = None,
    manifest_sha256: str | None = None,
    seal_plan_path: Path | None = None,
    seal_plan_sha256: str | None = None,
    raw_root: Path | None = None,
    block_index_path: Path | None = None,
    block_index_sha256: str | None = None,
    inventory_finalized_at: str | None = None,
    camera_id: str | None = None,
    release_freeze_path: Path | None = None,
    release_freeze_sha256: str | None = None,
) -> Any:
    """Authenticate descriptor, claim, worker hostname/POD, and staged source.

    Role and worker identity are deliberately read from the immutable queue
    descriptor/claim.  They are never accepted as command-line assertions.
    Every executable field other than that released role is reconstructed
    independently from this wrapper's closed interface.
    """

    job_supplied = Path(job_dir)
    require(not job_supplied.is_symlink(), "compiler queue job directory is a symlink")
    try:
        job = job_supplied.resolve(strict=True)
    except OSError as error:
        raise ConfirmationCompilerJobError("compiler queue job directory is missing") from error
    descriptor_path = job / "descriptor.json"
    claim_path = job / "claim" / "owner.json"
    try:
        released = queue.load_json(descriptor_path, "compiler queue descriptor")
        claim = queue.load_json(claim_path, "compiler queue claim")
    except Exception as error:
        raise ConfirmationCompilerJobError(str(error)) from error
    role = released.get("role")
    worker_id = claim.get("worker_id")
    require(
        isinstance(role, str)
        and role.startswith("wmf-forecast-0912-worker-")
        and compiler.SAFE_COMPONENT_RE.fullmatch(role) is not None,
        "compiler queue descriptor role is not an owned static worker role",
    )
    require(
        isinstance(worker_id, str)
        and compiler.SAFE_COMPONENT_RE.fullmatch(worker_id) is not None,
        "compiler queue claim worker identity is invalid",
    )
    expected = {
        "schema_version": "wmf-cluster-job-v1",
        "namespace": compiler.NAMESPACE,
        "job_id": job_id,
        "released": True,
        "source_commit": study_commit,
        "role": role,
        "argv": _queue_argv(
            command=command,
            source_root=source_root,
            study_commit=study_commit,
            job_dir=job,
            job_id=job_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            seal_plan_path=seal_plan_path,
            seal_plan_sha256=seal_plan_sha256,
            raw_root=raw_root,
            block_index_path=block_index_path,
            block_index_sha256=block_index_sha256,
            inventory_finalized_at=inventory_finalized_at,
            camera_id=camera_id,
            release_freeze_path=release_freeze_path,
            release_freeze_sha256=release_freeze_sha256,
        ),
        "max_wall_seconds": QUEUE_MAX_WALL_SECONDS,
        "publish_log_tail_bytes": QUEUE_PUBLISH_LOG_TAIL_BYTES,
    }
    try:
        context = queue.validate_queue_context(
            source_root=source_root,
            job_dir=job,
            study_commit=study_commit,
            job_id=job_id,
            expected_role=role,
            expected_descriptor=expected,
            expected_worker_id=worker_id,
        )
    except Exception as error:
        raise ConfirmationCompilerJobError(f"compiler queue context failed: {error}") from error
    expected_state = queue.CONTROL_ROOT.resolve()
    require(
        context.job_dir == expected_state / "jobs" / job_id
        and context.source_root == expected_state / "sources" / study_commit,
        "compiler queue context is outside the authoritative control root",
    )
    expected_wrapper = context.source_root / THIS_RELATIVE
    require(
        Path(__file__).resolve() == expected_wrapper,
        "compiler wrapper is not executing from the staged source root",
    )
    return context


def _producer_queue_job(context: Any) -> dict[str, Any]:
    return {
        "job_id": context.job_id,
        "job_dir": str(context.job_dir),
        "role": context.role,
        "worker_id": context.worker_id,
        "runtime_identity": {
            "hostname": context.hostname,
            "pod_uid": context.pod_uid,
            "pid": os.getpid(),
        },
        "descriptor": context.descriptor_identity,
        "claim_owner": context.claim_identity,
        "queue_result_lineage": {
            "path": str(context.job_dir / "result.json"),
            "availability": "written_by_queue_worker_after_wrapper_exit",
            "job_id": context.job_id,
            "source_commit": context.study_commit,
            "descriptor_sha256": context.descriptor_identity["sha256"],
            "worker_id": context.worker_id,
        },
    }


def run_compiler_job(
    *,
    source_root: Path,
    study_commit: str,
    job_dir: Path,
    job_id: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    context = _queue_context(
        command="run",
        source_root=source_root,
        study_commit=study_commit,
        job_dir=job_dir,
        job_id=job_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    manifest_identity = _verify_file(manifest_path, manifest_sha256, "compiler input manifest")
    manifest = compiler.load_json(Path(manifest_identity["path"]), "compiler input manifest")
    require(manifest.get("schema_version") == INPUT_SCHEMA, "compiler input schema changed")
    compiler.verify_signed(manifest, "compiler input manifest")
    require(
        Path(str(manifest.get("source_root"))).resolve() == context.source_root
        and manifest.get("study_commit") == context.study_commit,
        "compiler manifest source identity differs from its authenticated queue job",
    )
    require(
        COMPILER_PATH.resolve()
        == context.source_root
        / queue.FORECAST_RELATIVE
        / "analysis/compile_confirmation_evidence.py",
        "compiler implementation is not executing from the staged source root",
    )
    raw, publish = queue._prepare_output_directories(context.job_dir)
    output_dir = context.job_dir / RUN_BUNDLE_RELATIVE
    job_receipt = publish / RUN_RECEIPT_NAME
    require(output_dir.parent == raw, "compiler output escaped its job-local raw directory")
    result = compiler.compile_manifest(
        Path(manifest_identity["path"]), manifest_identity["sha256"], output_dir
    )
    compiler_receipt_path = Path(output_dir).resolve() / "compiler_receipt.json"
    zero_science_path = Path(output_dir).resolve() / "zero_science_receipt.json"
    zero_science = compiler.load_json(zero_science_path, "compiler zero-science receipt")
    compiler.verify_signed(zero_science, "compiler zero-science receipt")
    require(all(
        value == 0 for key, value in zero_science.items()
        if key not in {"schema_version", "study_id", "stage", "payload_sha256"}
    ), "compiler zero-science receipt contains activity")
    job = sign_document({
        "schema_version": JOB_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "status": "compiled_complete_roster",
        "job_id": context.job_id,
        "job_dir": str(context.job_dir),
        "worker_role": context.role,
        "worker_id": context.worker_id,
        "runtime_identity": {
            "hostname": context.hostname,
            "pod_uid": context.pod_uid,
            "pid": os.getpid(),
        },
        "queue_descriptor": context.descriptor_identity,
        "queue_claim": context.claim_identity,
        "queue_result_lineage": {
            "path": str(context.job_dir / "result.json"),
            "availability": "written_by_queue_worker_after_wrapper_exit",
            "job_id": context.job_id,
            "source_commit": context.study_commit,
            "descriptor_sha256": context.descriptor_identity["sha256"],
            "worker_id": context.worker_id,
        },
        "study_commit": context.study_commit,
        "input_manifest": manifest_identity,
        "contract": _file_descriptor(CONTRACT_PATH),
        "wrapper_source": _file_descriptor(Path(__file__).resolve()),
        "compiler_source": _file_descriptor(COMPILER_PATH),
        "compiler_receipt": _file_descriptor(compiler_receipt_path),
        "zero_science_receipt": _file_descriptor(zero_science_path),
        "counts": dict(result["counts"]),
        "science_counts": _zero_science_counts(),
        "queue_mutations": 0,
        "confirmation_released": False,
        "labels_created": False,
        "scientific_results_computed": False,
        "claim_boundary": (
            "Detached-compatible CPU evidence compilation only; no model request, "
            "simulator action, label, metric, or confirmation queue release occurred."
        ),
    })
    queue.immutable_json(job_receipt, job, maximum_bytes=512 * 1024)
    return job


def run_manifest_job(
    *,
    source_root: Path,
    study_commit: str,
    job_dir: Path,
    job_id: str,
    block_index_path: Path,
    block_index_sha256: str,
    raw_root: Path,
    inventory_finalized_at: str,
    camera_id: str,
    release_freeze_path: Path,
    release_freeze_sha256: str,
) -> dict[str, Any]:
    """Build the compiler input in an authenticated detached CPU queue job."""

    context = _queue_context(
        command="build-manifest",
        source_root=source_root,
        study_commit=study_commit,
        job_dir=job_dir,
        job_id=job_id,
        block_index_path=block_index_path,
        block_index_sha256=block_index_sha256,
        raw_root=raw_root,
        inventory_finalized_at=inventory_finalized_at,
        camera_id=camera_id,
        release_freeze_path=release_freeze_path,
        release_freeze_sha256=release_freeze_sha256,
    )
    raw, publish = queue._prepare_output_directories(context.job_dir)
    output = context.job_dir / MANIFEST_RELATIVE
    require(output.parent == raw, "compiler manifest escaped its job-local raw directory")
    manifest = build_compiler_manifest(
        block_index_path=block_index_path,
        block_index_sha256=block_index_sha256,
        source_root=context.source_root,
        raw_root=raw_root,
        study_commit=context.study_commit,
        inventory_finalized_at=inventory_finalized_at,
        camera_id=camera_id,
        release_freeze_path=release_freeze_path,
        release_freeze_sha256=release_freeze_sha256,
        output=output,
    )
    receipt = sign_document({
        "schema_version": MANIFEST_JOB_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "status": "compiler_manifest_built",
        "job_id": context.job_id,
        "job_dir": str(context.job_dir),
        "study_commit": context.study_commit,
        "queue_role": context.role,
        "worker_id": context.worker_id,
        "runtime_identity": {
            "hostname": context.hostname,
            "pod_uid": context.pod_uid,
            "pid": os.getpid(),
        },
        "queue_descriptor": context.descriptor_identity,
        "queue_claim": context.claim_identity,
        "queue_result_lineage": {
            "path": str(context.job_dir / "result.json"),
            "availability": "written_by_queue_worker_after_wrapper_exit",
            "job_id": context.job_id,
            "source_commit": context.study_commit,
            "descriptor_sha256": context.descriptor_identity["sha256"],
            "worker_id": context.worker_id,
        },
        "inputs": {
            "terminal_block_index": _file_descriptor(Path(block_index_path).resolve()),
            "development_release_freeze": _file_descriptor(
                Path(release_freeze_path).resolve()
            ),
        },
        "compiler_input_manifest": _file_descriptor(output),
        "cohort_branch": manifest["cohort_branch"],
        "block_receipts": len(manifest["block_receipts"]),
        "science_counts": _zero_science_counts(),
        "queue_mutations": 0,
        "confirmation_released": False,
        "labels_created": False,
        "scientific_results_computed": False,
        "claim_boundary": (
            "Authenticated CPU-only manifest assembly from the already closed cohort; "
            "no model, simulator, action, label, metric, or queue release occurred."
        ),
    })
    queue.immutable_json(
        publish / MANIFEST_RECEIPT_NAME, receipt, maximum_bytes=512 * 1024
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-cohort")
    seal.add_argument("--source-root", type=Path, required=True)
    seal.add_argument("--study-commit", required=True)
    seal.add_argument("--job-dir", type=Path, required=True)
    seal.add_argument("--job-id", required=True)
    seal.add_argument("--seal-plan", type=Path, required=True)
    seal.add_argument("--seal-plan-sha256", required=True)
    seal.add_argument("--raw-root", type=Path, required=True)
    seal.add_argument(
        "--camera-id",
        choices=compiler.development.PRIMARY_CAMERA_CHOICES,
        required=True,
    )
    build = subparsers.add_parser("build-manifest")
    build.add_argument("--block-index", type=Path, required=True)
    build.add_argument("--block-index-sha256", required=True)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--study-commit", required=True)
    build.add_argument("--job-dir", type=Path, required=True)
    build.add_argument("--job-id", required=True)
    build.add_argument("--raw-root", type=Path, required=True)
    build.add_argument("--inventory-finalized-at", required=True)
    build.add_argument("--camera-id", choices=compiler.development.PRIMARY_CAMERA_CHOICES, required=True)
    build.add_argument("--release-freeze", type=Path, required=True)
    build.add_argument("--release-freeze-sha256", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--study-commit", required=True)
    run.add_argument("--job-dir", type=Path, required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "seal-cohort":
            result = run_seal_job(
                seal_plan_path=args.seal_plan,
                seal_plan_sha256=args.seal_plan_sha256,
                source_root=args.source_root,
                study_commit=args.study_commit,
                job_dir=args.job_dir,
                job_id=args.job_id,
                raw_root=args.raw_root,
                camera_id=args.camera_id,
            )
            output = Path(args.job_dir).resolve() / "publish" / SEAL_RECEIPT_NAME
            summary = {
                "status": result["status"],
                "science_counts": result["science_counts"],
                "confirmation_released": False,
                "output": str(output),
                "sha256": sha256_file(output),
            }
        elif args.command == "build-manifest":
            result = run_manifest_job(
                block_index_path=args.block_index,
                block_index_sha256=args.block_index_sha256,
                source_root=args.source_root,
                study_commit=args.study_commit,
                job_dir=args.job_dir,
                job_id=args.job_id,
                raw_root=args.raw_root,
                inventory_finalized_at=args.inventory_finalized_at,
                camera_id=args.camera_id,
                release_freeze_path=args.release_freeze,
                release_freeze_sha256=args.release_freeze_sha256,
            )
            output = Path(args.job_dir).resolve() / "publish" / MANIFEST_RECEIPT_NAME
            summary = {
                "status": result["status"],
                "cohort_branch": result["cohort_branch"],
                "block_receipts": result["block_receipts"],
                "science_counts": result["science_counts"],
                "confirmation_released": False,
                "output": str(output),
                "sha256": sha256_file(output),
            }
        else:
            result = run_compiler_job(
                source_root=args.source_root,
                study_commit=args.study_commit,
                job_dir=args.job_dir,
                job_id=args.job_id,
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
            )
            output = Path(args.job_dir).resolve() / "publish" / RUN_RECEIPT_NAME
            summary = {
                "status": result["status"],
                "science_counts": result["science_counts"],
                "confirmation_released": False,
                "output": str(output),
                "sha256": sha256_file(output),
            }
    except (
        ConfirmationCompilerJobError,
        compiler.ConfirmationCompilerError,
        queue.TimingQueueError,
    ) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
