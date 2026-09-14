#!/usr/bin/env python3
"""Build one authenticated confirmation release wave without dispatching it.

The builder consumes externally produced, hash-signed release, fixture,
resource, terminal-runtime, attempt-ledger, and live pending
cluster-reconciliation evidence. It never infers a clean cluster from
missing files or caller flags, and it never edits ``cluster_queue.json``. Its
only outputs are a deterministic plan, an exact queue fragment, and a receipt
binding the two. This module is deliberately not a producer for the live
ledger/reconciliation inputs.  The resulting descriptor-only wave remains
unauthorized until the cluster coordinator publishes the finalizer artifacts
and the independent post-publication verifier accepts that H1 publication.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


STUDY_ID = "WMF-ABLATION-001"
NAMESPACE = "wmf_ablation_001_20260912"
CONTRACT_FILENAME = "confirmation_release_wave_contract.json"
BUILDER_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_release_wave_jobs.py"
)
CONTRACT_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_release_wave_contract.json"
)
EVIDENCE_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_release_evidence_jobs.py"
)
EVIDENCE_CONTRACT_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_release_evidence_contract.json"
)
RUNTIME_IDENTITIES_PRODUCER_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_terminal_runtime_identities.py"
)
RUNTIME_IDENTITIES_CONTRACT_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_terminal_runtime_identities_contract.json"
)
LEDGER_SCHEMA = "wmf-confirmation-result-attempt-ledger-v1"
RECONCILIATION_SCHEMA = "wmf-confirmation-cluster-reconciliation-v1"
RUNTIME_IDENTITIES_SCHEMA = "wmf-confirmation-terminal-runtime-identities-v1"
PUBLICATION_PENDING_SCHEMA = "wmf-confirmation-release-publication-pending-v1"
PUBLICATION_VERIFICATION_SCHEMA = "wmf-confirmation-release-publication-verification-v1"
PLAN_SCHEMA = "wmf-confirmation-release-wave-plan-v1"
RECEIPT_SCHEMA = "wmf-confirmation-release-wave-receipt-v1"
AUTHORIZATION_SCHEMA = "wmf-confirmation-release-wave-authorization-v1"
RUNTIME_ADMISSION_SCHEMA = "wmf-confirmation-runtime-release-admission-v1"
RUNTIME_ADMISSION_CLAIM_BOUNDARY = (
    "Live read-only admission before any behavioral reset, model request, or action."
)
QUEUE_SCHEMA = "wmf-cluster-queue-v1"
QUEUE_JOB_SCHEMA = "wmf-cluster-job-v1"
SAFE_ID_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

RUNTIME_IDENTITY_FILES = {
    "n3_behavioral_pilot_job.py",
    "n3_confirmation_block_job.py",
    "d1_behavioral_pilot_jobs.py",
    "d1_confirmation_block_jobs.py",
    "d1_instrumented_server.py",
}
PREREQUISITE_NAMES = {
    "n3_pilot_receipt",
    "recorder_receipt",
    "d1_qualification_receipt",
    "d1_pilot_simulator_receipt",
    "d1_pilot_server_receipt",
}
RESULT_KEYS = {
    "schema_version", "namespace", "job_id", "worker_id", "source_commit",
    "descriptor_sha256", "started_at", "argv", "job_dir", "status",
    "returncode", "error_type", "ended_at", "wall_seconds", "child_pid",
    "child_reaped", "stdout", "stderr",
}
FORBIDDEN_ARG_FRAGMENTS = (
    "password", "private-key", "private_key", "secret", "token", "credential",
)


class ConfirmationReleaseWaveError(RuntimeError):
    """A stable fail-closed release-planning error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfirmationReleaseWaveError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfirmationReleaseWaveError("value is not finite canonical JSON") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfirmationReleaseWaveError("value is not finite JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexical_absolute_path(path: Path, label: str) -> Path:
    try:
        candidate = Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as error:
        raise ConfirmationReleaseWaveError(f"{label} path is invalid") from error
    require(candidate.is_absolute(), f"{label} path is not absolute")
    return candidate


def _path_component_snapshot(
    path: Path, label: str, *, allow_missing: bool,
) -> tuple[Path, tuple[tuple[str, int, int, int], ...]]:
    """Resolve no links and retain every existing component's inode identity."""

    candidate = _lexical_absolute_path(path, label)
    current = Path(candidate.anchor)
    paths = [current]
    for part in candidate.parts[1:]:
        current = current / part
        paths.append(current)
    rows: list[tuple[str, int, int, int]] = []
    for index, component in enumerate(paths):
        try:
            metadata = component.lstat()
        except FileNotFoundError as error:
            if allow_missing:
                break
            raise ConfirmationReleaseWaveError(
                f"{label} path component is missing: {component}"
            ) from error
        except OSError as error:
            raise ConfirmationReleaseWaveError(
                f"{label} path component is unreadable: {component}"
            ) from error
        require(
            not stat.S_ISLNK(metadata.st_mode),
            f"{label} path component is a symlink: {component}",
        )
        if index < len(paths) - 1:
            require(
                stat.S_ISDIR(metadata.st_mode),
                f"{label} parent component is not a directory: {component}",
            )
        rows.append(
            (str(component), metadata.st_dev, metadata.st_ino, metadata.st_mode)
        )
    return candidate, tuple(rows)


def _require_component_snapshot(
    path: Path, snapshot: Sequence[tuple[str, int, int, int]], label: str,
    *, exact: bool,
) -> None:
    candidate, current = _path_component_snapshot(
        path, label, allow_missing=not exact,
    )
    del candidate
    expected = tuple(snapshot)
    require(
        (current == expected if exact else current[: len(expected)] == expected),
        f"{label} path components changed",
    )


_STABLE_FILE_FIELDS = (
    "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
    "st_mtime_ns", "st_ctime_ns",
)


def _same_file_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in _STABLE_FILE_FIELDS
    )


def _open_directory_fd(
    path: Path, label: str,
) -> tuple[int, Path, tuple[tuple[str, int, int, int], ...], os.stat_result]:
    candidate = _lexical_absolute_path(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    components = candidate.parts
    require(components and components[0] == candidate.anchor, f"{label} path is invalid")
    descriptor = -1
    rows: list[tuple[str, int, int, int]] = []
    current = Path(candidate.anchor)
    try:
        descriptor = os.open(candidate.anchor, flags)
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISDIR(metadata.st_mode), f"{label} root is not a directory",
        )
        rows.append((str(current), metadata.st_dev, metadata.st_ino, metadata.st_mode))
        for part in components[1:]:
            try:
                entry = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise ConfirmationReleaseWaveError(
                    f"{label} path component is unreadable: {current / part}"
                ) from error
            require(
                stat.S_ISDIR(entry.st_mode) and not stat.S_ISLNK(entry.st_mode),
                f"{label} path component is not a safe directory: {current / part}",
            )
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise ConfirmationReleaseWaveError(
                    f"{label} path component cannot be opened safely: {current / part}"
                ) from error
            try:
                opened = os.fstat(child)
                require(
                    stat.S_ISDIR(opened.st_mode)
                    and (opened.st_dev, opened.st_ino, opened.st_mode)
                    == (entry.st_dev, entry.st_ino, entry.st_mode),
                    f"{label} path component changed during open: {current / part}",
                )
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
            metadata = opened
            current /= part
            rows.append(
                (str(current), metadata.st_dev, metadata.st_ino, metadata.st_mode)
            )
        snapshot = tuple(rows)
        _require_component_snapshot(candidate, snapshot, label, exact=True)
        return descriptor, candidate, snapshot, metadata
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ConfirmationReleaseWaveError(f"{label} directory is unreadable") from error
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _read_regular_at(
    parent_fd: int, name: str, label: str, *, capture_payload: bool,
) -> tuple[bytes | None, str, os.stat_result]:
    require(
        isinstance(name, str) and name not in {"", ".", ".."}
        and "/" not in name and "\x00" not in name,
        f"{label} filename is invalid",
    )
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ConfirmationReleaseWaveError(f"{label} is unreadable") from error
    try:
        opened = os.fstat(descriptor)
        require(stat.S_ISREG(opened.st_mode), f"{label} is not a regular file")
        digest = hashlib.sha256()
        payload_parts: list[bytes] = []
        observed_bytes = 0
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            observed_bytes += len(chunk)
            digest.update(chunk)
            if capture_payload:
                payload_parts.append(chunk)
        completed = os.fstat(descriptor)
        require(
            _same_file_metadata(opened, completed)
            and observed_bytes == completed.st_size,
            f"{label} changed while reading",
        )
        try:
            at_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ConfirmationReleaseWaveError(
                f"{label} pathname disappeared while reading"
            ) from error
        require(
            stat.S_ISREG(at_path.st_mode)
            and _same_file_metadata(completed, at_path),
            f"{label} pathname changed while reading",
        )
        payload = b"".join(payload_parts) if capture_payload else None
        return payload, digest.hexdigest(), completed
    finally:
        os.close(descriptor)


def _read_regular_file(
    path: Path, label: str, *, capture_payload: bool,
) -> tuple[bytes | None, str, os.stat_result, Path]:
    candidate, snapshot = _path_component_snapshot(path, label, allow_missing=False)
    require(candidate.name not in {"", ".", ".."}, f"{label} path is not a file")
    parent_fd, parent, parent_snapshot, _metadata = _open_directory_fd(
        candidate.parent, f"{label} parent",
    )
    try:
        payload, digest, opened = _read_regular_at(
            parent_fd, candidate.name, label, capture_payload=capture_payload,
        )
        _require_component_snapshot(parent, parent_snapshot, f"{label} parent", exact=True)
        _require_component_snapshot(candidate, snapshot, label, exact=True)
        return payload, digest, opened, candidate
    finally:
        os.close(parent_fd)


def sha256_file(path: Path) -> str:
    _payload, digest, _metadata, _candidate = _read_regular_file(
        Path(path), f"file {path}", capture_payload=False,
    )
    return digest


def signed_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document already has a payload hash")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_signed_document(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("payload_sha256")
    require(
        isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None,
        f"{label} payload hash is invalid",
    )
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    require(
        sha256_bytes(canonical_bytes(unsigned)) == observed,
        f"{label} payload hash mismatch",
    )


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} fields changed")


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


def load_json_with_identity(
    path: Path, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, digest, metadata, candidate = _read_regular_file(
        Path(path), label, capture_payload=True,
    )
    assert payload is not None
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ConfirmationReleaseWaveError(f"{label} is unreadable") from error
    require(isinstance(value, dict), f"{label} is not an object")
    _require_finite_json(value, label)
    return {
        "path": str(candidate), "bytes": metadata.st_size, "sha256": digest,
    }, value


def load_json(path: Path, label: str) -> dict[str, Any]:
    _identity, value = load_json_with_identity(path, label)
    return value


def file_identity(path: Path) -> dict[str, Any]:
    _payload, digest, metadata, candidate = _read_regular_file(
        Path(path), f"file {path}", capture_payload=False,
    )
    return {
        "path": str(candidate), "bytes": metadata.st_size, "sha256": digest,
    }


def verify_input(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(
        isinstance(expected_sha256, str) and SHA256_RE.fullmatch(expected_sha256) is not None,
        f"{label} expected SHA-256 is invalid",
    )
    identity = file_identity(path)
    require(identity["sha256"] == expected_sha256, f"{label} SHA-256 mismatch")
    return identity


def verify_json_input(
    path: Path, expected_sha256: str, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        isinstance(expected_sha256, str)
        and SHA256_RE.fullmatch(expected_sha256) is not None,
        f"{label} expected SHA-256 is invalid",
    )
    identity, value = load_json_with_identity(path, label)
    require(identity["sha256"] == expected_sha256, f"{label} SHA-256 mismatch")
    return identity, value


def verify_descriptor(value: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    exact_keys(value, {"path", "bytes", "sha256"}, f"{label} descriptor")
    identity, document = load_json_with_identity(
        Path(str(value.get("path", ""))), label,
    )
    require(identity == dict(value), f"{label} descriptor identity changed")
    return identity, document


def safe_id(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SAFE_ID_RE.fullmatch(value) is not None
        and value not in {".", ".."},
        f"{label} is not a safe immutable ID",
    )
    return value


def parse_utc(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} is missing")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ConfirmationReleaseWaveError(f"{label} is not an ISO-8601 timestamp") from error
    require(parsed.tzinfo is not None, f"{label} is not timezone-aware")
    require(parsed.utcoffset() == timedelta(0), f"{label} is not UTC")
    return parsed.astimezone(timezone.utc)


def _load_path_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    prior_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        raise ConfirmationReleaseWaveError(f"failed to import {path.name}: {type(error).__name__}") from error
    finally:
        sys.dont_write_bytecode = prior_bytecode_setting
    return module


def load_runtime_modules(source_root: Path) -> dict[str, Any]:
    root = Path(source_root).resolve()
    forecast = root / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    # Bare imports are deliberate: the runtime modules themselves use these
    # names and must share one set of module globals during native validation.
    try:
        common = importlib.import_module("confirmation_runtime_common")
        fixture = importlib.import_module("confirmation_fixture_freeze")
        n3 = importlib.import_module("n3_confirmation_block_job")
        d1 = importlib.import_module("d1_confirmation_block_jobs")
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            f"confirmation runtime import failed: {type(error).__name__}"
        ) from error
    for name, module in (
        ("confirmation_runtime_common.py", common),
        ("confirmation_fixture_freeze.py", fixture),
        ("n3_confirmation_block_job.py", n3),
        ("d1_confirmation_block_jobs.py", d1),
    ):
        require(
            Path(str(getattr(module, "__file__", ""))).resolve() == (forecast / name).resolve(),
            f"runtime module was imported from another source root: {name}",
        )
    queue_path = root / "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.py"
    queue = _load_path_module(queue_path, "wmf_confirmation_release_cluster_queue")
    return {"common": common, "fixture": fixture, "n3": n3, "d1": d1, "queue": queue}


def load_release_evidence_module(source_root: Path):
    """Load the exact external-evidence producer used for native replay."""

    root = Path(source_root).resolve()
    path = (root / EVIDENCE_RELATIVE).resolve()
    require(path.is_file() and not path.is_symlink(), "release-evidence producer is missing")
    module = _load_path_module(path, "wmf_confirmation_release_evidence_native")
    require(
        Path(str(getattr(module, "__file__", ""))).resolve() == path
        and getattr(module, "STUDY_ID", None) == STUDY_ID
        and getattr(module, "NAMESPACE", None) == NAMESPACE
        and getattr(module, "LEDGER_SCHEMA", None) == LEDGER_SCHEMA
        and getattr(module, "RECONCILIATION_SCHEMA", None) == RECONCILIATION_SCHEMA
        and getattr(module, "PENDING_SCHEMA", None) == PUBLICATION_PENDING_SCHEMA
        and getattr(module, "VERIFICATION_SCHEMA", None) == PUBLICATION_VERIFICATION_SCHEMA
        and callable(getattr(module, "queue_triplet", None))
        and callable(getattr(module, "validate_queue_claim", None))
        and callable(getattr(module, "validate_result_attempt_ledger", None))
        and callable(getattr(module, "validate_pending_reconciliation", None))
        and callable(getattr(module, "validate_publication_pending", None))
        and callable(getattr(module, "verify_published_finalizer", None))
        and callable(getattr(module, "validate_worker_attestation", None))
        and callable(getattr(module, "gpu_inventory", None))
        and callable(getattr(module, "_pod_uid", None))
        and callable(getattr(module, "deployment_topology", None))
        and callable(getattr(module, "load_contract", None)),
        "release-evidence native validation API changed",
    )
    return module


def load_terminal_runtime_identities_module(source_root: Path):
    """Load the exact external source/publication validator."""

    root = Path(source_root).resolve()
    path = (root / RUNTIME_IDENTITIES_PRODUCER_RELATIVE).resolve()
    require(
        path.is_file() and not path.is_symlink(),
        "terminal-runtime identity producer is missing",
    )
    module = _load_path_module(
        path, "wmf_confirmation_terminal_runtime_identities_native",
    )
    require(
        Path(str(getattr(module, "__file__", ""))).resolve() == path
        and getattr(module, "STUDY_ID", None) == STUDY_ID
        and getattr(module, "NAMESPACE", None) == NAMESPACE
        and getattr(module, "OUTPUT_SCHEMA", None) == RUNTIME_IDENTITIES_SCHEMA
        and callable(getattr(module, "validate_terminal_runtime_identities", None)),
        "terminal-runtime identity native validation API changed",
    )
    return module


def load_contract(source_root: Path) -> dict[str, Any]:
    root = Path(source_root).resolve()
    path = root / "workshops/corl2026_world_models/experiments/forecast_layout" / CONTRACT_FILENAME
    value = load_json(path, "release-wave contract")
    exact_keys(
        value,
        {
            "schema_version", "study_id", "namespace", "status", "execution_readiness",
            "prepared_schedule",
            "branch_models", "lane_priority", "minimum_control_generation",
            "maximum_reconciliation_age_seconds", "maximum_job_wall_seconds",
            "publication_grace_seconds", "publish_log_tail_bytes", "trusted_publication", "runtime",
            "runtime_admission", "external_evidence_schemas", "terminal_context_schemas",
            "source_dependencies", "output",
        },
        "release-wave contract",
    )
    require(
        value.get("schema_version") == "wmf-confirmation-release-wave-contract-v1"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("status") == "source_only_fail_closed",
        "release-wave contract identity changed",
    )
    readiness = value.get("execution_readiness")
    require(isinstance(readiness, Mapping), "release-wave execution boundary missing")
    exact_keys(
        readiness,
        {
            "status", "builder_synthesizes_evidence", "required_external_artifacts",
            "claim_boundary",
        },
        "release-wave execution boundary",
    )
    require(
        readiness.get("status") == "pending_cluster_coordinator_h1_verification"
        and readiness.get("builder_synthesizes_evidence") is False
        and readiness.get("required_external_artifacts")
        == [
            LEDGER_SCHEMA,
            RECONCILIATION_SCHEMA,
        ],
        "release-wave external-producer boundary changed",
    )
    trusted = value.get("trusted_publication")
    require(isinstance(trusted, Mapping), "trusted publication contract missing")
    exact_keys(
        trusted,
        {
            "results_remote_aliases", "results_ref", "allowed_fetch_urls",
            "repository_identity", "git_path_prefix", "cluster_status_schema",
        },
        "trusted publication contract",
    )
    require(
        trusted.get("results_remote_aliases") == ["publish", "origin"]
        and trusted.get("results_ref")
        == "refs/heads/codex/forecast-layout-gm-20260912-results"
        and trusted.get("allowed_fetch_urls")
        == [
            "https://github.com/adeeb10abbas/steerable.git",
            "git@github.com:adeeb10abbas/steerable.git",
            "ssh://git@github.com/adeeb10abbas/steerable.git",
        ]
        and trusted.get("repository_identity")
        == {
            "host": "github.com", "owner": "adeeb10abbas",
            "repository": "steerable",
        },
        "trusted publication repository identity changed",
    )
    require(
        value.get("external_evidence_schemas") == {
            "confirmation_freeze": "wmf-development-confirmation-release-freeze-v1",
            "fixture_freeze": "wmf-confirmation-fixture-freeze-v1",
            "resource_qualification": "wmf-development-resource-machine-freeze-v1",
            "result_attempt_ledger": LEDGER_SCHEMA,
            "cluster_reconciliation": RECONCILIATION_SCHEMA,
            "terminal_runtime_identities": RUNTIME_IDENTITIES_SCHEMA,
            "publication_pending": PUBLICATION_PENDING_SCHEMA,
            "publication_verification": PUBLICATION_VERIFICATION_SCHEMA,
        },
        "release-wave evidence schema inventory changed",
    )
    require(
        value.get("runtime_admission") == {
            "schema": RUNTIME_ADMISSION_SCHEMA,
            "d1_pair_ack_schema": "wmf-d1-confirmation-runtime-admission-ack-v1",
            "fresh_h1_stage": "queue_start",
            "downstream_replay_allows_consume_by_expiry_after_fresh_admission": True,
            "required_live_identity": [
                "published_descriptor", "immutable_queue_claim",
                "selected_worker_role", "hostname", "pod_uid",
                "gpu_uuid_name_memory_count", "zero_compute_processes",
            ],
            "science_must_not_start_before_gate": True,
            "active_queue_mutation": False,
            "jobs_dispatched": 0,
        },
        "runtime admission contract changed",
    )
    output = value.get("output")
    require(isinstance(output, Mapping), "release-wave output contract missing")
    exact_keys(
        output,
        {
            "plan_schema", "queue_schema", "receipt_schema", "authorization_schema",
            "status", "execution_release_authorized", "active_queue_mutation",
            "invented_cluster_evidence",
        },
        "release-wave output contract",
    )
    require(
        output == {
            "plan_schema": PLAN_SCHEMA,
            "queue_schema": QUEUE_SCHEMA,
            "receipt_schema": RECEIPT_SCHEMA,
            "authorization_schema": AUTHORIZATION_SCHEMA,
            "status": "pending_h1_verification",
            "execution_release_authorized": False,
            "active_queue_mutation": False,
            "invented_cluster_evidence": False,
        },
        "release-wave output boundary changed",
    )
    dependencies = value.get("source_dependencies")
    require(isinstance(dependencies, Mapping) and dependencies, "source dependency pins missing")
    require(
        {
            EVIDENCE_RELATIVE,
            EVIDENCE_CONTRACT_RELATIVE,
            RUNTIME_IDENTITIES_PRODUCER_RELATIVE,
            RUNTIME_IDENTITIES_CONTRACT_RELATIVE,
        }.issubset(dependencies),
        "external evidence producer source pins are missing",
    )
    for relative, digest in dependencies.items():
        require(
            isinstance(relative, str) and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "source dependency path is unsafe",
        )
        verify_input(root / relative, str(digest), f"source dependency {relative}")
    return value


def validate_study_commit(
    source_root: Path, study_commit: str, contract: Mapping[str, Any]
) -> None:
    """Bind every executable/validator/deployment pin to the emitted commit."""

    root = Path(source_root).resolve()
    try:
        exists = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{study_commit}^{{commit}}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ConfirmationReleaseWaveError("study commit verification failed") from error
    require(exists.returncode == 0, "study commit is not a local commit object")
    expected_sources = dict(contract["source_dependencies"])
    expected_sources.update({
        BUILDER_RELATIVE: sha256_file(root / BUILDER_RELATIVE),
        CONTRACT_RELATIVE: sha256_file(root / CONTRACT_RELATIVE),
    })
    for relative, expected_sha in expected_sources.items():
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "show", f"{study_commit}:{relative}"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ConfirmationReleaseWaveError(
                f"study commit source verification failed: {relative}"
            ) from error
        require(
            result.returncode == 0 and sha256_bytes(result.stdout) == expected_sha,
            f"study commit does not contain pinned source: {relative}",
        )


def _models_for_branch(contract: Mapping[str, Any], branch: Any) -> tuple[str, ...]:
    branches = contract.get("branch_models")
    require(isinstance(branches, Mapping) and branch in branches, "release branch is invalid")
    models = branches[branch]
    require(
        isinstance(models, list) and models in [["N3", "D1"], ["N3"], ["D1"]],
        "release branch model inventory changed",
    )
    return tuple(models)


def validate_schedule(
    *, source_root: Path, contract: Mapping[str, Any], runtime: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...], dict[tuple[str, str], Any]]:
    schedule_contract = contract.get("prepared_schedule")
    require(isinstance(schedule_contract, Mapping), "prepared-schedule contract missing")
    exact_keys(
        schedule_contract,
        {
            "path", "sha256", "confirmation_row_count", "layout_count_per_model",
            "layout_order_source", "confirmation_blocks_in_hash_order",
        },
        "prepared-schedule contract",
    )
    path = Path(source_root).resolve() / str(schedule_contract["path"])
    identity, value = verify_json_input(
        path, str(schedule_contract["sha256"]), "prepared schedule",
    )
    rows = value.get("jobs")
    require(isinstance(rows, list), "prepared schedule rows missing")
    confirmation_rows = [row for row in rows if isinstance(row, Mapping) and row.get("phase") == "confirmation"]
    require(
        len(confirmation_rows) == schedule_contract["confirmation_row_count"] == 48,
        "prepared confirmation row inventory changed",
    )
    assignment = value.get("order_assignment")
    require(isinstance(assignment, Mapping), "prepared schedule order assignment missing")
    order = assignment.get("confirmation_blocks_in_hash_order")
    require(
        order == schedule_contract["confirmation_blocks_in_hash_order"]
        and schedule_contract["layout_order_source"]
        == "order_assignment.confirmation_blocks_in_hash_order",
        "prepared confirmation hash order changed",
    )
    blocks: dict[tuple[str, str], Any] = {}
    for model in ("N3", "D1"):
        for layout in order:
            schedule_block = runtime["common"].load_confirmation_schedule_block(
                source_root, layout, model
            )
            require(schedule_block.schedule_sha256 == identity["sha256"], "runtime schedule hash changed")
            runtime_block = runtime[model.lower()].load_confirmation_block(source_root, layout)
            require(
                runtime_block.block_id == schedule_block.block_id
                and tuple(runtime_block.condition_order) == tuple(schedule_block.condition_order)
                and tuple(runtime_block.cell_ids) == tuple(schedule_block.cell_ids),
                f"{model} runtime block differs from prepared schedule: {layout}",
            )
            blocks[(model, layout)] = runtime_block
    return identity, tuple(order), blocks


def validate_release_freeze(
    *, descriptor: dict[str, Any], source_root: Path, contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    identity, value = verify_descriptor(descriptor, "confirmation release freeze")
    require(identity == descriptor, "confirmation release freeze identity changed")
    expected_schema = contract["external_evidence_schemas"]["confirmation_freeze"]
    require(value.get("schema_version") == expected_schema, "confirmation release schema changed")
    decision = value.get("release_decision")
    annotation = value.get("annotation")
    budget = value.get("resource_budget")
    usability = annotation.get("usability_decision") if isinstance(annotation, Mapping) else None
    movement = annotation.get("movement_resolution") if isinstance(annotation, Mapping) else None
    require(
        value.get("status") == "frozen_for_confirmation"
        and isinstance(decision, Mapping)
        and decision.get("eligible") is True
        and decision.get("blockers") == []
        and isinstance(annotation, Mapping)
        and isinstance(usability, Mapping)
        and usability.get("measurement_usable") is True
        and isinstance(movement, Mapping)
        and movement.get("status")
        == "frozen_from_duplicate_development_labels"
        and isinstance(budget, Mapping)
        and budget.get("status") == "frozen_from_complete_development_measurements",
        "confirmation release is still held by development evidence",
    )
    branch = value.get("cohort_branch")
    models = _models_for_branch(contract, branch)
    require(value.get("qualified_model_ids") == list(models), "release model branch changed")
    validations = {}
    for model in models:
        try:
            validations[model] = runtime["common"].verify_confirmation_freeze(
                Path(descriptor["path"]), descriptor["sha256"],
                source_root=source_root, model_config=model,
            )
        except BaseException as error:
            raise ConfirmationReleaseWaveError(
                f"confirmation release freeze failed native validation for {model}"
            ) from error
    # A future release freeze may carry the prepared across-block order.  If it
    # does, it must agree exactly; absence does not authorize another order.
    if "confirmation_blocks_in_hash_order" in value:
        require(
            value["confirmation_blocks_in_hash_order"]
            == contract["prepared_schedule"]["confirmation_blocks_in_hash_order"],
            "release freeze across-block order differs from prepared order",
        )
    by_model = budget.get("by_model")
    require(isinstance(by_model, Mapping) and set(by_model) == set(models), "release resource model budget changed")
    for model in models:
        row = by_model[model]
        require(
            isinstance(row, Mapping)
            and type(row.get("max_parallel_blocks")) is int
            and row["max_parallel_blocks"] > 0,
            f"{model} signed release concurrency is invalid",
        )
    return value, validations, models


RESOURCE_KEYS = {
    "schema_version", "status", "decision", "namespace", "study_id",
    "study_commit", "job_id", "queue_role", "worker_id", "runtime_identity",
    "queue_descriptor", "queue_claim", "implementation", "inputs", "measurement",
    "model_envelopes", "safe_execution_topology", "science_counts",
    "machine_resource_gate_complete", "resource_measurement_freeze_complete",
    "annotation_time", "remaining_resource_release_requirements",
    "safe_to_release_confirmation", "confirmation_released",
    "behavioral_policy_skill_evaluated", "claim_boundary",
    "raw_machine_resource_freeze", "payload_sha256",
}


def validate_resource_qualification(
    *, descriptor: dict[str, Any], study_commit: str, models: Sequence[str],
    release: Mapping[str, Any], contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    identity, value = verify_descriptor(descriptor, "resource qualification")
    require(identity == descriptor, "resource qualification identity changed")
    exact_keys(value, RESOURCE_KEYS, "resource qualification")
    verify_signed_document(value, "resource qualification")
    require(
        value.get("schema_version")
        == contract["external_evidence_schemas"]["resource_qualification"]
        and value.get("status") == "machine_resource_gate_passed_annotation_time_pending"
        and value.get("decision") == "serial_gm_topology_frozen_confirmation_held"
        and value.get("namespace") == NAMESPACE
        and value.get("study_id") == STUDY_ID
        and value.get("study_commit") == study_commit
        and value.get("machine_resource_gate_complete") is True
        and value.get("resource_measurement_freeze_complete") is True,
        "resource qualification is not the selected passed machine freeze",
    )
    # This artifact is capacity evidence, never release authority.  Release is
    # separately established by the development confirmation freeze and live
    # reconciliation receipt.
    require(
        value.get("safe_to_release_confirmation") is False
        and value.get("confirmation_released") is False,
        "resource qualification release boundary changed",
    )
    verify_descriptor(value.get("raw_machine_resource_freeze"), "raw resource qualification")
    topology = value.get("safe_execution_topology")
    require(isinstance(topology, Mapping), "resource safe topology missing")
    exact_keys(
        topology,
        {
            "global_max_parallel_blocks", "cross_model_simultaneous_blocks_allowed",
            "max_parallel_blocks_by_model", "N3", "D1", "measurement_status",
            "higher_concurrency_qualified", "scheduling_rule",
        },
        "resource safe topology",
    )
    limits = topology.get("max_parallel_blocks_by_model")
    require(isinstance(limits, Mapping), "resource model concurrency missing")
    qualified: dict[str, int] = {}
    release_by_model = release["resource_budget"]["by_model"]
    for model in models:
        limit = limits.get(model)
        require(type(limit) is int and limit > 0, f"{model} resource concurrency is invalid")
        release_limit = release_by_model[model]["max_parallel_blocks"]
        qualified[model] = min(limit, release_limit)
    global_limit = topology.get("global_max_parallel_blocks")
    require(type(global_limit) is int and global_limit > 0, "resource global concurrency is invalid")
    require(
        topology.get("cross_model_simultaneous_blocks_allowed") is False,
        "cross-model simultaneous blocks were not safely qualified",
    )
    n3_topology = topology.get("N3")
    d1_topology = topology.get("D1")
    require(
        isinstance(n3_topology, Mapping)
        and n3_topology.get("model_worker_role") == contract["runtime"]["N3"]["role"]
        and n3_topology.get("gpu_count") == contract["runtime"]["N3"]["gpu_count_per_worker"],
        "resource N3 role/GPU topology changed",
    )
    require(
        isinstance(d1_topology, Mapping)
        and d1_topology.get("model_worker_role") == contract["runtime"]["D1"]["server_role"]
        and d1_topology.get("model_gpu_count") == contract["runtime"]["D1"]["server_gpu_count"]
        and d1_topology.get("simulator_worker_role") in contract["runtime"]["D1"]["simulator_roles"]
        and d1_topology.get("simulator_gpu_count") == contract["runtime"]["D1"]["simulator_gpu_count"],
        "resource D1 role/GPU topology changed",
    )
    if global_limit > 1 or any(value > 1 for value in limits.values()):
        require(
            topology.get("higher_concurrency_qualified") is True,
            "resource qualification did not explicitly measure higher concurrency",
        )
    else:
        require(
            topology.get("higher_concurrency_qualified") is False,
            "serial resource qualification higher-concurrency flag changed",
        )
    # A model envelope is required for every selected model; it binds this
    # topology to actual measurement rather than a bare caller-supplied cap.
    envelopes = value.get("model_envelopes")
    require(
        isinstance(envelopes, Mapping) and all(model in envelopes for model in models),
        "resource model envelopes are incomplete",
    )
    for model in qualified:
        qualified[model] = min(qualified[model], global_limit)
    return value, qualified


def validate_runtime_identities(
    *, descriptor: dict[str, Any], source_root: Path, study_commit: str,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(isinstance(descriptor, Mapping), "terminal runtime identity descriptor is missing")
    exact_keys(
        descriptor, {"path", "bytes", "sha256"},
        "terminal runtime identity descriptor",
    )
    native = load_terminal_runtime_identities_module(source_root)
    try:
        replay = native.validate_terminal_runtime_identities(
            receipt_path=Path(str(descriptor.get("path", ""))),
            expected_sha256=str(descriptor.get("sha256", "")),
            source_root=source_root,
            source_commit=study_commit,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "terminal runtime identity publication failed native validation"
        ) from error
    require(isinstance(replay, Mapping), "terminal runtime native replay is missing")
    exact_keys(
        replay, {"receipt", "identity", "prerequisite_receipts"},
        "terminal runtime native replay",
    )
    identity = replay.get("identity")
    value = replay.get("receipt")
    require(
        isinstance(identity, Mapping) and dict(identity) == dict(descriptor)
        and isinstance(value, Mapping),
        "terminal runtime identity publication changed",
    )
    exact_keys(
        value,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "published_at_utc", "runtime_files", "terminal_context_schemas",
            "prerequisite_receipts", "claim_boundary", "payload_sha256",
        },
        "terminal runtime identities",
    )
    verify_signed_document(value, "terminal runtime identities")
    require(
        value.get("schema_version")
        == contract["external_evidence_schemas"]["terminal_runtime_identities"]
        and value.get("status") == "published_terminal_context_runtime"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit,
        "terminal runtime publication identity changed",
    )
    parse_utc(value.get("published_at_utc"), "terminal runtime publication time")
    require(
        value.get("terminal_context_schemas") == contract["terminal_context_schemas"],
        "terminal context schema publication changed",
    )
    runtime_files = value.get("runtime_files")
    require(
        isinstance(runtime_files, Mapping) and set(runtime_files) == RUNTIME_IDENTITY_FILES,
        "terminal runtime file inventory changed",
    )
    dependencies = contract["source_dependencies"]
    forecast_prefix = "workshops/corl2026_world_models/experiments/forecast_layout/"
    for name, row in runtime_files.items():
        require(isinstance(row, Mapping), f"runtime identity is invalid: {name}")
        exact_keys(row, {"path", "sha256"}, f"runtime identity {name}")
        expected_path = forecast_prefix + name
        require(
            row.get("path") == expected_path
            and row.get("sha256") == dependencies.get(expected_path),
            f"published runtime identity changed: {name}",
        )
        verify_input(Path(source_root).resolve() / expected_path, row["sha256"], f"runtime {name}")
    prerequisites = value.get("prerequisite_receipts")
    require(
        isinstance(prerequisites, Mapping) and set(prerequisites) == PREREQUISITE_NAMES,
        "published prerequisite receipt inventory changed",
    )
    require(
        isinstance(replay.get("prerequisite_receipts"), Mapping)
        and dict(replay["prerequisite_receipts"]) == dict(prerequisites),
        "terminal runtime native prerequisite replay changed",
    )
    verified: dict[str, dict[str, Any]] = {}
    for name, row in prerequisites.items():
        identity, _receipt = verify_descriptor(row, name)
        verified[name] = identity
    return value, verified


def deployment_topology(
    *, source_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    names = (
        "n3_queue_worker.json", "d1_queue_worker_v2.json",
        "science_queue_workers_v3.json", "science_queue_workers_pod_uid_050609.json",
    )
    observed: dict[str, dict[str, Any]] = {}
    for name in names:
        relative = f"workshops/corl2026_world_models/execution/20260912/autonomy/{name}"
        value = load_json(root / relative, f"deployment {name}")
        items = value.get("items")
        require(
            value.get("apiVersion") == "v1" and value.get("kind") == "List"
            and isinstance(items, list),
            f"deployment list changed: {name}",
        )
        for item in items:
            if not isinstance(item, Mapping) or item.get("kind") != "Job":
                continue
            metadata = item.get("metadata")
            spec = item.get("spec", {}).get("template", {}).get("spec", {})
            containers = spec.get("containers")
            require(
                isinstance(metadata, Mapping) and isinstance(containers, list)
                and len(containers) == 1 and isinstance(containers[0], Mapping),
                f"worker deployment structure changed: {name}",
            )
            container = containers[0]
            argv = container.get("args")
            require(isinstance(argv, list), f"worker args missing: {name}")
            for option in ("--worker-id", "--role", "--admission-deadline-unix"):
                require(argv.count(option) == 1, f"worker option changed: {name}:{option}")
            worker_id = argv[argv.index("--worker-id") + 1]
            role = argv[argv.index("--role") + 1]
            limits = container.get("resources", {}).get("limits", {})
            requests = container.get("resources", {}).get("requests", {})
            gpu_count = limits.get("nvidia.com/gpu")
            require(
                type(gpu_count) is int and gpu_count > 0
                and requests.get("nvidia.com/gpu") == gpu_count,
                f"worker GPU topology changed: {worker_id}",
            )
            deployment_sha = contract["source_dependencies"][relative]
            if worker_id in observed:
                require(
                    observed[worker_id]["role"] == role
                    and observed[worker_id]["gpu_count"] == gpu_count,
                    f"replacement worker topology changed: {worker_id}",
                )
                observed[worker_id]["deployment_sha256s"].append(deployment_sha)
            else:
                observed[worker_id] = {
                    "role": role,
                    "gpu_count": gpu_count,
                    "deployment_sha256": deployment_sha,
                    "deployment_sha256s": [deployment_sha],
                }
    n3 = contract["runtime"]["N3"]
    d1 = contract["runtime"]["D1"]
    require(
        any(row["role"] == n3["role"] and row["gpu_count"] == n3["gpu_count_per_worker"] for row in observed.values()),
        "no deployed N3 two-GPU worker contract",
    )
    require(
        any(row["role"] == d1["server_role"] and row["gpu_count"] == d1["server_gpu_count"] for row in observed.values()),
        "no deployed D1 two-GPU server contract",
    )
    deployed_simulator_roles = {
        row["role"] for row in observed.values() if row["gpu_count"] == d1["simulator_gpu_count"]
    }
    require(
        set(d1["simulator_roles"]).issubset(deployed_simulator_roles),
        "D1 simulator role deployment coverage changed",
    )
    return observed


def _raw_from_normalized(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    exact_keys(
        value,
        {
            "schema_version", "namespace", "job_id", "released", "source_commit",
            "role", "argv", "max_wall_seconds", "publish_log_tail_bytes",
        },
        label,
    )
    require(
        value.get("schema_version") == "wmf-cluster-job-v1"
        and value.get("namespace") == NAMESPACE,
        f"{label} queue identity changed",
    )
    return {
        key: value[key]
        for key in (
            "job_id", "released", "source_commit", "role", "argv",
            "max_wall_seconds", "publish_log_tail_bytes",
        )
    }


def _parse_descriptor(
    descriptor: Mapping[str, Any], *, model: str, layout: str, study_commit: str,
    runtime: Mapping[str, Any], expected_job_ids: Sequence[str],
    expected_attempt_id: str | None = None,
    expected_options: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    argv = descriptor.get("argv")
    require(
        isinstance(argv, list) and len(argv) >= 3
        and all(isinstance(item, str) and item for item in argv),
        "queue descriptor argv missing",
    )
    require(
        not any(
            fragment in item.lower()
            for item in argv for fragment in FORBIDDEN_ARG_FRAGMENTS
        ),
        "queue descriptor argv contains secret-bearing material",
    )
    expected_runner = "{source_root}/" + runtime["contract"]["runtime"][model]["runner"]
    require(
        argv[0] == "/usr/bin/python3" and argv[1] == expected_runner,
        f"{model} queue runner changed",
    )
    options = [item for item in argv if isinstance(item, str) and item.startswith("--")]
    require(len(options) == len(set(options)), f"{model} queue argv contains duplicate options")
    if expected_options is not None:
        for option, wanted in expected_options.items():
            require(
                argv.count(option) == 1 and argv.index(option) + 1 < len(argv)
                and argv[argv.index(option) + 1] == wanted,
                f"{model} queue evidence binding changed: {option}",
            )
    module = runtime[model.lower()]
    try:
        args = module.build_parser().parse_args(argv[2:])
    except SystemExit as error:
        raise ConfirmationReleaseWaveError(f"{model} queue argv failed runtime parsing") from error
    require(
        args.layout_pair_id == layout and args.study_commit == study_commit
        and args.job_id == descriptor["job_id"]
        and args.source_root == Path("{source_root}")
        and args.job_dir == Path("{job_dir}"),
        f"{model} queue argv identity changed",
    )
    for name, value in vars(args).items():
        if name.endswith("_sha256") and value is not None:
            require(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
                f"{model} queue SHA-256 option is invalid: {name}",
            )
    path_names = [
        "gate_receipt", "pose_manifest", "capture_receipt", "confirmation_freeze",
        "fixture_freeze", "pilot_receipt", "recorder_receipt",
        "d1_qualification_receipt", "pilot_simulator_receipt", "pilot_server_receipt",
    ]
    for name in path_names:
        if hasattr(args, name):
            require(Path(getattr(args, name)).is_absolute(), f"{model} evidence path is not absolute: {name}")
    if model == "N3":
        require(args.mode == "queue" and descriptor["role"] == runtime["contract"]["runtime"]["N3"]["role"], "N3 queue mode or role changed")
        require(list(expected_job_ids) == [descriptor["job_id"]], "N3 attempt job inventory changed")
        if expected_attempt_id is not None:
            require(expected_attempt_id == descriptor["job_id"], "N3 attempt/job identity changed")
    else:
        require(args.mode in {"server-job", "simulator-job"}, "D1 queue mode changed")
        server_id, simulator_id = expected_job_ids
        if args.mode == "server-job":
            require(
                descriptor["job_id"] == server_id
                and descriptor["role"] == runtime["contract"]["runtime"]["D1"]["server_role"]
                and args.simulator_job_id == simulator_id,
                "D1 server pairing changed",
            )
        else:
            require(
                descriptor["job_id"] == simulator_id
                and descriptor["role"] == args.simulator_worker_role
                and args.server_job_id == server_id,
                "D1 simulator pairing changed",
            )
        require(
            args.simulator_worker_role in runtime["contract"]["runtime"]["D1"]["simulator_roles"],
            "D1 simulator role changed",
        )
        if expected_attempt_id is not None:
            require(args.run_id == expected_attempt_id, "D1 attempt/run identity changed")
    return args


def _validate_queue_result(
    value: Mapping[str, Any], *, descriptor: Mapping[str, Any], queue: Any,
) -> None:
    exact_keys(value, RESULT_KEYS, "queue result")
    normalized = queue.normalize_job(_raw_from_normalized(descriptor, "queue descriptor"))
    require(normalized == dict(descriptor), "queue descriptor normalization changed")
    descriptor_sha = sha256_bytes(queue.encode(normalized))
    require(
        value.get("schema_version") == "wmf-cluster-result-v1"
        and value.get("namespace") == NAMESPACE
        and value.get("job_id") == descriptor["job_id"]
        and value.get("source_commit") == descriptor["source_commit"]
        and value.get("descriptor_sha256") == descriptor_sha
        and value.get("status") in {"succeeded", "failed", "timed_out", "interrupted"}
        and value.get("child_reaped") is True,
        "queue result does not terminally bind its descriptor",
    )
    safe_id(value.get("worker_id"), "queue result worker ID")
    started = parse_utc(value.get("started_at"), "queue result start time")
    ended = parse_utc(value.get("ended_at"), "queue result end time")
    wall = value.get("wall_seconds")
    require(
        ended >= started
        and not isinstance(wall, bool) and isinstance(wall, (int, float))
        and math.isfinite(wall) and wall >= 0,
        "queue result timing is invalid",
    )
    if value.get("status") == "succeeded":
        require(
            value.get("returncode") == 0 and value.get("error_type") is None,
            "successful queue result has contradictory exit evidence",
        )
    else:
        require(
            value.get("returncode") is None or type(value.get("returncode")) is int,
            "failed queue result return code is invalid",
        )
    child_pid = value.get("child_pid")
    require(child_pid is None or (type(child_pid) is int and child_pid > 1), "queue result child PID is invalid")
    argv = value.get("argv")
    job_dir_text = value.get("job_dir")
    require(
        isinstance(argv, list) and argv
        and all(isinstance(item, str) and item for item in argv)
        and isinstance(job_dir_text, str) and Path(job_dir_text).is_absolute(),
        "queue result runtime identity is invalid",
    )
    job_dir = Path(job_dir_text).resolve()
    require(len(job_dir.parents) >= 2, "queue result job directory is invalid")
    state_dir = job_dir.parents[1]
    expected_argv = [
        item.replace(
            "{source_root}", str(state_dir / "sources" / descriptor["source_commit"])
        ).replace("{job_dir}", str(job_dir)).replace("{state_dir}", str(state_dir))
        for item in descriptor["argv"]
    ]
    require(argv == expected_argv, "queue result argv does not bind the executed descriptor")
    require(
        not any(
            fragment in item.lower()
            for item in argv for fragment in FORBIDDEN_ARG_FRAGMENTS
        ),
        "queue result argv contains secret-bearing material",
    )
    for name in ("stdout", "stderr"):
        row = value.get(name)
        require(
            isinstance(row, Mapping) and set(row) == {"bytes", "sha256"}
            and type(row.get("bytes")) is int and row["bytes"] >= 0
            and SHA256_RE.fullmatch(str(row.get("sha256"))) is not None,
            "queue result stream identity is invalid",
        )


def _validate_queue_claim(
    value: Mapping[str, Any], *, identity: Mapping[str, Any],
    descriptor: Mapping[str, Any], descriptor_identity: Mapping[str, Any],
    result: Mapping[str, Any], result_identity: Mapping[str, Any], queue: Any,
) -> dict[str, Any]:
    exact_keys(
        value,
        {
            "worker_id", "claimed_at", "claimed_unix", "worker_pid",
            "control_commit", "control_generation", "descriptor_sha256",
            "release_boundary",
        },
        "queue claim owner",
    )
    worker_id = safe_id(value.get("worker_id"), "queue claim worker ID")
    claimed_at = parse_utc(value.get("claimed_at"), "queue claim time")
    claimed_unix = value.get("claimed_unix")
    descriptor_sha = sha256_bytes(queue.encode(descriptor))
    require(
        not isinstance(claimed_unix, bool) and isinstance(claimed_unix, (int, float))
        and math.isfinite(claimed_unix) and claimed_unix > 0
        and abs(claimed_at.timestamp() - float(claimed_unix)) <= 5
        and type(value.get("worker_pid")) is int and value["worker_pid"] > 1
        and isinstance(value.get("control_commit"), str)
        and COMMIT_RE.fullmatch(value["control_commit"]) is not None
        and type(value.get("control_generation")) is int
        and value["control_generation"] > 0
        and value.get("descriptor_sha256") == descriptor_sha
        and value.get("release_boundary") == "claim_committed_under_shared_release_lock"
        and worker_id == result.get("worker_id")
        and parse_utc(result.get("started_at"), "queue result start time") >= claimed_at,
        "queue claim owner does not bind the descriptor/result runtime",
    )
    job_id = descriptor["job_id"]
    job_root = Path(str(identity["path"])).resolve().parents[1]
    require(
        Path(str(identity["path"])).name == "owner.json"
        and Path(str(identity["path"])).parent.name == "claim"
        and job_root.name == job_id
        and Path(str(descriptor_identity["path"])).resolve() == job_root / "descriptor.json"
        and Path(str(result_identity["path"])).resolve() == job_root / "result.json"
        and Path(str(result.get("job_dir"))).resolve() == job_root,
        "queue claim/descriptor/result paths do not share one immutable job directory",
    )
    return dict(value)


def _native_cell_state(
    *, runtime: Mapping[str, Any], model: str, block: Any,
    condition_index: int, study_commit: str, receipt_path: Path,
) -> str:
    try:
        if receipt_path.name == "technical_failure.json":
            result = runtime[model.lower()].validate_failed_confirmation_cell(
                receipt_path, condition_index=condition_index, block=block,
            )
            status = result.get("status")
            require(status in {"safety_abort", "technical_failure"}, "native failure state changed")
            return "safety_censored" if status == "safety_abort" else "technical_invalid"
        runtime[model.lower()].validate_passed_confirmation_cell(
            receipt_path, condition_index=condition_index,
            study_commit=study_commit, block=block,
        )
        return "passed"
    except ConfirmationReleaseWaveError:
        raise
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            f"native {model} confirmation validation failed at cell {condition_index}"
        ) from error


def _historical_evidence_options(
    *, model: str, fixture: Mapping[str, Any],
    release_identity: Mapping[str, Any], fixture_identity: Mapping[str, Any],
    prerequisites: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    expected = {
        "--candidate-id": str(fixture["candidate_id"]),
        "--gate-receipt": str(fixture["gate_receipt"]["path"]),
        "--gate-receipt-sha256": str(fixture["gate_receipt"]["sha256"]),
        "--pose-manifest": str(fixture["pose_manifest"]["path"]),
        "--pose-manifest-sha256": str(fixture["pose_manifest"]["sha256"]),
        "--capture-receipt": str(fixture["capture_receipt"]["path"]),
        "--capture-receipt-sha256": str(fixture["capture_receipt"]["sha256"]),
        "--confirmation-freeze": str(release_identity["path"]),
        "--confirmation-freeze-sha256": str(release_identity["sha256"]),
        "--fixture-freeze": str(fixture_identity["path"]),
        "--fixture-freeze-sha256": str(fixture_identity["sha256"]),
    }
    if model == "N3":
        expected.update({
            "--pilot-receipt": prerequisites["n3_pilot_receipt"]["path"],
            "--pilot-receipt-sha256": prerequisites["n3_pilot_receipt"]["sha256"],
        })
    else:
        for option, name in (
            ("recorder-receipt", "recorder_receipt"),
            ("d1-qualification-receipt", "d1_qualification_receipt"),
            ("pilot-simulator-receipt", "d1_pilot_simulator_receipt"),
            ("pilot-server-receipt", "d1_pilot_server_receipt"),
        ):
            expected[f"--{option}"] = prerequisites[name]["path"]
            expected[f"--{option}-sha256"] = prerequisites[name]["sha256"]
    return expected


def validate_result_ledger(
    *, descriptor: dict[str, Any], study_commit: str, branch: str,
    source_root: Path, state_dir: Path, evidence: Any,
    models: Sequence[str], layout_order: Sequence[str], blocks: Mapping[tuple[str, str], Any],
    schedule_identity: Mapping[str, Any], release_identity: Mapping[str, Any],
    fixture_identity: Mapping[str, Any], resource_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any], runtime: Mapping[str, Any],
    fixture_rows: Mapping[str, Mapping[str, Any]],
    prerequisites: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any], dict[tuple[str, str], dict[str, Any]], set[str], set[str],
    dict[str, dict[str, Any]],
]:
    require(set(fixture_rows) == set(layout_order), "ledger fixture binding inventory changed")
    require(set(prerequisites) == PREREQUISITE_NAMES, "ledger prerequisite binding inventory changed")
    identity, value = verify_descriptor(descriptor, "confirmation result ledger")
    require(identity == descriptor, "confirmation result ledger identity changed")
    exact_keys(
        value,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "cohort_branch", "qualified_model_ids", "prepared_schedule",
            "confirmation_freeze", "fixture_freeze", "resource_qualification",
            "terminal_runtime_identities", "layout_order", "blocks",
            "attempt_inventory_complete", "claim_boundary", "payload_sha256",
        },
        "confirmation result ledger",
    )
    verify_signed_document(value, "confirmation result ledger")
    try:
        native = evidence.validate_result_attempt_ledger(
            value,
            source_root=Path(source_root).resolve(),
            state_dir=Path(state_dir).resolve(),
            study_commit=study_commit,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "confirmation result ledger failed native producer validation"
        ) from error
    require(
        isinstance(native, Mapping) and native.get("document") == value,
        "native result-ledger replay returned another document",
    )
    require(
        value.get("schema_version") == LEDGER_SCHEMA
        and value.get("status") == "authenticated_native_attempt_inventory"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit
        and value.get("cohort_branch") == branch
        and value.get("qualified_model_ids") == list(models)
        and value.get("layout_order") == list(layout_order)
        and value.get("attempt_inventory_complete") is True,
        "confirmation result ledger identity changed",
    )
    expected_inputs = {
        "prepared_schedule": dict(schedule_identity),
        "confirmation_freeze": dict(release_identity),
        "fixture_freeze": dict(fixture_identity),
        "resource_qualification": dict(resource_identity),
        "terminal_runtime_identities": dict(runtime_identity),
    }
    for name, expected in expected_inputs.items():
        require(value.get(name) == expected, f"result ledger input binding changed: {name}")
    rows = value.get("blocks")
    expected_order = [(model, layout) for model in models for layout in layout_order]
    require(
        isinstance(rows, list) and len(rows) == 24 * len(models),
        "result ledger does not account for every selected base layout",
    )
    observed_order = [
        (row.get("model_config"), row.get("layout_pair_id"))
        for row in rows if isinstance(row, Mapping)
    ]
    require(observed_order == expected_order, "result ledger block order changed")
    by_block: dict[tuple[str, str], dict[str, Any]] = {}
    all_attempt_ids: set[str] = set()
    all_job_ids: set[str] = set()
    all_claims: dict[str, dict[str, Any]] = {}
    for row, key in zip(rows, expected_order, strict=True):
        require(isinstance(row, Mapping), "result ledger block row is invalid")
        exact_keys(
            row,
            {
                "model_config", "layout_pair_id", "condition_order", "cell_ids",
                "state", "completed_prefix_cells", "terminal_attempt_id", "attempts",
            },
            f"ledger block {key[0]}:{key[1]}",
        )
        model, layout = key
        block = blocks[key]
        require(
            row.get("condition_order") == list(block.condition_order)
            and row.get("cell_ids") == list(block.cell_ids),
            f"ledger block permutation changed: {model}:{layout}",
        )
        attempts = row.get("attempts")
        require(isinstance(attempts, list), f"ledger attempts missing: {model}:{layout}")
        prefix = 0
        terminal_state: str | None = None
        previous_attempt: str | None = None
        last_attempt_state: str | None = None
        for attempt_number, attempt in enumerate(attempts, start=1):
            require(terminal_state is None, f"attempt follows terminal block: {model}:{layout}")
            require(isinstance(attempt, Mapping), "ledger attempt is invalid")
            exact_keys(
                attempt,
                {
                    "attempt_id", "attempt_number", "predecessor_attempt_id",
                    "start_cell_index", "job_ids", "queue_descriptors",
                    "queue_claims", "queue_results", "cells", "runtime_evidence",
                },
                f"ledger attempt {model}:{layout}:{attempt_number}",
            )
            attempt_id = safe_id(attempt.get("attempt_id"), "attempt ID")
            require(attempt_id not in all_attempt_ids, "attempt ID was reused")
            all_attempt_ids.add(attempt_id)
            expected_attempt_id, expected_job_ids = _attempt_identity(
                model, layout, attempt_number
            )
            require(
                attempt.get("attempt_number") == attempt_number
                and attempt.get("predecessor_attempt_id") == previous_attempt
                and attempt.get("start_cell_index") == prefix,
                f"attempt predecessor or retry prefix changed: {attempt_id}",
            )
            previous_attempt = attempt_id
            job_ids = attempt.get("job_ids")
            expected_job_count = 1 if model == "N3" else 2
            require(
                isinstance(job_ids, list) and len(job_ids) == expected_job_count
                and len(set(job_ids)) == expected_job_count,
                f"attempt job inventory changed: {attempt_id}",
            )
            require(
                attempt_id == expected_attempt_id and job_ids == expected_job_ids,
                f"attempt/job immutable naming changed: {attempt_id}",
            )
            for job_id in job_ids:
                safe_id(job_id, "queue job ID")
                require(job_id not in all_job_ids, "queue job ID was reused")
                all_job_ids.add(job_id)
            descriptor_rows = attempt.get("queue_descriptors")
            claim_rows = attempt.get("queue_claims")
            result_rows = attempt.get("queue_results")
            require(
                isinstance(descriptor_rows, list) and len(descriptor_rows) == expected_job_count
                and isinstance(claim_rows, list) and len(claim_rows) == expected_job_count
                and isinstance(result_rows, list) and len(result_rows) == expected_job_count,
                f"attempt queue evidence inventory changed: {attempt_id}",
            )
            loaded_descriptors: list[dict[str, Any]] = []
            parsed_descriptors: list[argparse.Namespace] = []
            descriptor_identities: list[dict[str, Any]] = []
            loaded_results: list[dict[str, Any]] = []
            for descriptor_row in descriptor_rows:
                _identity, queue_descriptor = verify_descriptor(
                    descriptor_row, f"queue descriptor for {attempt_id}"
                )
                normalized = runtime["queue"].normalize_job(
                    _raw_from_normalized(queue_descriptor, "historical queue descriptor")
                )
                require(normalized == queue_descriptor, "historical queue descriptor normalization drifted")
                require(
                    queue_descriptor.get("source_commit") == study_commit
                    and queue_descriptor.get("released") is True
                    and queue_descriptor.get("publish_log_tail_bytes") == 0,
                    "historical queue descriptor release binding changed",
                )
                parsed = _parse_descriptor(
                    queue_descriptor, model=model, layout=layout, study_commit=study_commit,
                    runtime=runtime, expected_job_ids=job_ids,
                    expected_attempt_id=attempt_id,
                    expected_options=_historical_evidence_options(
                        model=model, fixture=fixture_rows[layout],
                        release_identity=release_identity,
                        fixture_identity=fixture_identity,
                        prerequisites=prerequisites,
                    ),
                )
                parsed_descriptors.append(parsed)
                loaded_descriptors.append(queue_descriptor)
                descriptor_identities.append(_identity)
            require(
                [item["job_id"] for item in loaded_descriptors] == job_ids,
                f"queue descriptor order changed: {attempt_id}",
            )
            if model == "D1":
                require(
                    [item.mode for item in parsed_descriptors] == ["server-job", "simulator-job"]
                    and len({item.run_id for item in parsed_descriptors}) == 1
                    and len({item.simulator_worker_role for item in parsed_descriptors}) == 1,
                    f"D1 descriptor pair does not share one run/role: {attempt_id}",
                )
                _expected_attempt, _raw_rows, expected_descriptors = _d1_descriptors(
                    layout=layout, attempt_number=attempt_number,
                    study_commit=study_commit,
                    simulator_role=parsed_descriptors[0].simulator_worker_role,
                        fixture=fixture_rows[layout], release_identity=release_identity,
                        fixture_identity=fixture_identity, prerequisites=prerequisites,
                        finalizer_job_id=parsed_descriptors[0].confirmation_release_finalizer_job_id,
                        consume_by_utc=parsed_descriptors[0].confirmation_release_consume_by_utc,
                        contract=runtime["contract"], queue=runtime["queue"],
                )
            else:
                _expected_attempt, _raw_row, expected_descriptor = _n3_descriptor(
                    layout=layout, attempt_number=attempt_number,
                    study_commit=study_commit, fixture=fixture_rows[layout],
                        release_identity=release_identity,
                        fixture_identity=fixture_identity, prerequisites=prerequisites,
                        finalizer_job_id=parsed_descriptors[0].confirmation_release_finalizer_job_id,
                        consume_by_utc=parsed_descriptors[0].confirmation_release_consume_by_utc,
                        contract=runtime["contract"], queue=runtime["queue"],
                )
                expected_descriptors = [expected_descriptor]
            require(
                loaded_descriptors == expected_descriptors,
                f"queue descriptor literal argv changed: {attempt_id}",
            )
            result_identities: list[dict[str, Any]] = []
            for result_row, queue_descriptor in zip(result_rows, loaded_descriptors, strict=True):
                _identity, queue_result = verify_descriptor(result_row, f"queue result for {attempt_id}")
                _validate_queue_result(queue_result, descriptor=queue_descriptor, queue=runtime["queue"])
                loaded_results.append(queue_result)
                result_identities.append(_identity)
            require(
                [item["job_id"] for item in loaded_results] == job_ids,
                f"queue result order changed: {attempt_id}",
            )
            for claim_row, queue_descriptor, descriptor_identity, queue_result, result_identity in zip(
                claim_rows, loaded_descriptors, descriptor_identities,
                loaded_results, result_identities, strict=True,
            ):
                claim_identity, queue_claim = verify_descriptor(
                    claim_row, f"queue claim for {attempt_id}"
                )
                validated_claim = _validate_queue_claim(
                    queue_claim, identity=claim_identity,
                    descriptor=queue_descriptor, descriptor_identity=descriptor_identity,
                    result=queue_result, result_identity=result_identity,
                    queue=runtime["queue"],
                )
                all_claims[queue_descriptor["job_id"]] = validated_claim
            runtime_evidence = attempt.get("runtime_evidence")
            require(isinstance(runtime_evidence, Mapping), f"attempt runtime evidence missing: {attempt_id}")
            zero_launch = runtime_evidence.get("form") == "zero_launch_technical_failure"
            cell_rows = attempt.get("cells")
            require(
                isinstance(cell_rows, list) and (zero_launch or bool(cell_rows)),
                f"attempt has neither native cell nor authenticated zero-launch outcome: {attempt_id}",
            )
            expected_index = prefix
            last_attempt_state = "technical_invalid" if zero_launch else None
            if zero_launch:
                require(
                    cell_rows == []
                    and runtime_evidence.get("zero_behavioral_cells_launched") is True
                    and any(result["status"] != "succeeded" for result in loaded_results),
                    f"zero-launch technical attempt is contradictory: {attempt_id}",
                )
            for cell_position, cell in enumerate(cell_rows):
                require(isinstance(cell, Mapping), "ledger cell outcome is invalid")
                exact_keys(cell, {"condition_index", "state", "receipt"}, "ledger cell outcome")
                condition_index = cell.get("condition_index")
                require(
                    condition_index == expected_index and 0 <= condition_index < 4,
                    f"ledger cell outcome is not a contiguous prefix: {attempt_id}",
                )
                receipt_identity, _receipt = verify_descriptor(
                    cell.get("receipt"), f"native receipt {attempt_id}:{condition_index}"
                )
                receipt_attempt_root = Path(receipt_identity["path"]).resolve().parents[2]
                expected_receipt_attempt = job_ids[0] if model == "N3" else job_ids[1]
                require(
                    receipt_attempt_root.name == expected_receipt_attempt,
                    f"native cell receipt belongs to another queue attempt: {attempt_id}",
                )
                native_state = _native_cell_state(
                    runtime=runtime, model=model, block=block,
                    condition_index=condition_index, study_commit=study_commit,
                    receipt_path=Path(receipt_identity["path"]),
                )
                require(cell.get("state") == native_state, "ledger promoted a non-native cell state")
                last_attempt_state = native_state
                if native_state == "passed":
                    prefix += 1
                    expected_index += 1
                    if prefix == 4:
                        terminal_state = "passed"
                    require(
                        terminal_state is None or cell_position == len(cell_rows) - 1,
                        "cell follows completed block",
                    )
                elif native_state == "safety_censored":
                    terminal_state = "safety_censored"
                    require(cell_position == len(cell_rows) - 1, "cell follows safety censor")
                else:
                    require(
                        native_state == "technical_invalid" and cell_position == len(cell_rows) - 1,
                        "cell follows technical-invalid outcome",
                    )
            if terminal_state is None and last_attempt_state == "passed":
                # A worker can exit after durably writing one or more passed
                # cells but before it can author the next cell's failure
                # receipt. The authenticated terminal/reaped outer failure is
                # technical attempt evidence; the durable passed prefix is
                # retained and the next retry resumes strictly after it.
                require(
                    any(result["status"] != "succeeded" for result in loaded_results),
                    "partial passed prefix has no terminal outer technical failure",
                )
                last_attempt_state = "technical_invalid"
            require(
                terminal_state is not None or last_attempt_state == "technical_invalid",
                "incomplete attempt lacks a native technical-invalid terminal",
            )
            # Native cell receipts are the scientific authority.  A queue
            # wrapper may fail after the fourth cell is durably written (for
            # example during publication postprocessing); that cleanup failure
            # must be reconciled, but it must not make four valid native cells
            # eligible for another scientific attempt.
            if last_attempt_state in {"technical_invalid", "safety_censored"}:
                require(
                    any(result["status"] != "succeeded" for result in loaded_results),
                    "failed or censored attempt has only success queue results",
                )
        if terminal_state is not None:
            derived_state = terminal_state
        elif attempts:
            derived_state = "technical_invalid"
        else:
            derived_state = "not_run"
        require(
            row.get("state") == derived_state
            and row.get("completed_prefix_cells") == prefix
            and row.get("terminal_attempt_id")
            == (previous_attempt if terminal_state is not None else None),
            f"ledger block derived state changed: {model}:{layout}",
        )
        by_block[key] = dict(row)
    require(set(all_claims) == all_job_ids, "ledger queue claim inventory is incomplete")
    require(
        native.get("blocks") == by_block
        and native.get("attempt_ids") == all_attempt_ids
        and native.get("job_ids") == all_job_ids
        and native.get("queue_claims") == all_claims,
        "native result-ledger summary differs from scientific replay",
    )
    return value, by_block, all_attempt_ids, all_job_ids, all_claims


def validate_pending_reconciliation(
    *, descriptor: Mapping[str, Any], ledger_identity: Mapping[str, Any],
    ledger_validation: Mapping[str, Any], all_attempt_ids: set[str],
    all_job_ids: set[str], source_root: Path, state_dir: Path,
    study_commit: str, contract: Mapping[str, Any],
    runtime: Mapping[str, Any], evidence: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    """Replay producer-authored pending facts; do not predict coordinator H1."""

    identity, value = verify_descriptor(descriptor, "pending cluster reconciliation")
    require(identity == descriptor, "pending cluster reconciliation identity changed")
    active = value.get("control_semantics", {}).get("active_job_ids")
    require(
        isinstance(active, list) and len(active) == 1,
        "pending reconciliation does not identify one finalizer",
    )
    finalizer_id = safe_id(active[0], "pending finalizer job ID")
    try:
        finalizer_triplet = evidence.queue_triplet(
            state_dir=Path(state_dir).resolve(), job_id=finalizer_id,
            source_root=Path(source_root).resolve(), queue=runtime["queue"],
            allow_running_self=True,
        )
        native = evidence.validate_pending_reconciliation(
            value,
            source_root=Path(source_root).resolve(),
            state_dir=Path(state_dir).resolve(),
            study_commit=study_commit,
            ledger_identity=ledger_identity,
            finalizer_triplet=finalizer_triplet,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "pending reconciliation failed native producer validation"
        ) from error
    require(
        isinstance(native, Mapping) and native.get("document") == value,
        "native pending-reconciliation replay returned another document",
    )
    created = parse_utc(value.get("created_at_utc"), "pending reconciliation creation time")
    consume_by = parse_utc(
        value.get("consume_by_utc"), "pending reconciliation consume-by time"
    )
    as_of = datetime.now(timezone.utc)
    require(
        0 <= (as_of - created).total_seconds()
        <= contract["maximum_reconciliation_age_seconds"],
        "pending reconciliation is stale or future-dated",
    )
    require(as_of <= consume_by, "pending reconciliation consume-by time expired")
    generation = value.get("observed_control_generation")
    lower = value.get("observed_control_generation_lower_bound")
    require(
        type(generation) is int and type(lower) is int
        and generation >= lower >= contract["minimum_control_generation"]
        and generation != 998 and lower != 998,
        "pending reconciliation generation is stale or contradictory",
    )
    prior_ids = native.get("all_prior_job_ids")
    workers = native.get("post_exit_workers")
    capacity_ids = native.get("authorized_capacity_worker_ids")
    require(
        isinstance(prior_ids, list) and prior_ids == sorted(set(prior_ids))
        and all(safe_id(item, "prior queue job ID") for item in prior_ids)
        and all_job_ids.issubset(set(prior_ids))
        and isinstance(workers, list)
        and isinstance(capacity_ids, list) and capacity_ids
        and capacity_ids == list(dict.fromkeys(capacity_ids)),
        "pending reconciliation inventory or topology summary changed",
    )
    require(
        native.get("control_semantics") == value.get("control_semantics")
        and native.get("ledger_validation") == dict(ledger_validation)
        and set(native.get("all_queue_jobs", {})) == set(prior_ids)
        and ledger_validation.get("attempt_ids") == all_attempt_ids
        and ledger_validation.get("job_ids") == all_job_ids,
        "pending reconciliation is detached from the validated ledger",
    )
    workers_by_id = {
        row.get("worker_id"): dict(row) for row in workers if isinstance(row, Mapping)
    }
    authorized_workers = [workers_by_id[item] for item in capacity_ids if item in workers_by_id]
    require(
        [row.get("worker_id") for row in authorized_workers] == capacity_ids,
        "pending reconciliation authorized capacity is missing or reordered",
    )
    return value, authorized_workers, set(prior_ids)


def _fixture_bindings(row: Mapping[str, Any]) -> list[str]:
    return [
        "--candidate-id", str(row["candidate_id"]),
        "--gate-receipt", str(row["gate_receipt"]["path"]),
        "--gate-receipt-sha256", str(row["gate_receipt"]["sha256"]),
        "--pose-manifest", str(row["pose_manifest"]["path"]),
        "--pose-manifest-sha256", str(row["pose_manifest"]["sha256"]),
        "--capture-receipt", str(row["capture_receipt"]["path"]),
        "--capture-receipt-sha256", str(row["capture_receipt"]["sha256"]),
    ]


def _freeze_bindings(
    release_identity: Mapping[str, Any], fixture_identity: Mapping[str, Any]
) -> list[str]:
    return [
        "--confirmation-freeze", str(release_identity["path"]),
        "--confirmation-freeze-sha256", str(release_identity["sha256"]),
        "--fixture-freeze", str(fixture_identity["path"]),
        "--fixture-freeze-sha256", str(fixture_identity["sha256"]),
    ]


def _attempt_identity(
    model: str, layout: str, attempt_number: int
) -> tuple[str, list[str]]:
    stem = f"confirmation-{layout.lower()}-{model.lower()}-a{attempt_number:03d}"
    attempt_id = stem
    if model == "N3":
        return attempt_id, [stem]
    return attempt_id, [f"{stem}-server", f"{stem}-simulator"]


def _n3_descriptor(
    *, layout: str, attempt_number: int, study_commit: str,
    fixture: Mapping[str, Any], release_identity: Mapping[str, Any],
    fixture_identity: Mapping[str, Any], prerequisites: Mapping[str, Any],
    finalizer_job_id: str, consume_by_utc: str,
    contract: Mapping[str, Any], queue: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    attempt_id, job_ids = _attempt_identity("N3", layout, attempt_number)
    job_id = job_ids[0]
    argv = [
        "/usr/bin/python3", "{source_root}/" + contract["runtime"]["N3"]["runner"],
        contract["runtime"]["N3"]["mode"],
        "--layout-pair-id", layout,
        "--source-root", "{source_root}",
        "--study-commit", study_commit,
        "--job-dir", "{job_dir}",
        "--job-id", job_id,
        "--confirmation-release-finalizer-job-id", finalizer_job_id,
        "--confirmation-release-consume-by-utc", consume_by_utc,
        *_fixture_bindings(fixture),
        *_freeze_bindings(release_identity, fixture_identity),
        "--pilot-receipt", prerequisites["n3_pilot_receipt"]["path"],
        "--pilot-receipt-sha256", prerequisites["n3_pilot_receipt"]["sha256"],
    ]
    raw = {
        "job_id": job_id,
        "released": True,
        "source_commit": study_commit,
        "role": contract["runtime"]["N3"]["role"],
        "argv": argv,
        "max_wall_seconds": contract["maximum_job_wall_seconds"],
        "publish_log_tail_bytes": contract["publish_log_tail_bytes"],
    }
    return attempt_id, raw, queue.normalize_job(raw)


def _d1_descriptors(
    *, layout: str, attempt_number: int, study_commit: str,
    simulator_role: str, fixture: Mapping[str, Any],
    release_identity: Mapping[str, Any], fixture_identity: Mapping[str, Any],
    prerequisites: Mapping[str, Any], contract: Mapping[str, Any], queue: Any,
    finalizer_job_id: str, consume_by_utc: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    attempt_id, job_ids = _attempt_identity("D1", layout, attempt_number)
    server_id, simulator_id = job_ids
    common = [
        "--layout-pair-id", layout,
        "--simulator-worker-role", simulator_role,
        "--source-root", "{source_root}",
        "--study-commit", study_commit,
        "--confirmation-release-finalizer-job-id", finalizer_job_id,
        "--confirmation-release-consume-by-utc", consume_by_utc,
    ]
    evidence = [
        *_fixture_bindings(fixture),
        *_freeze_bindings(release_identity, fixture_identity),
        "--recorder-receipt", prerequisites["recorder_receipt"]["path"],
        "--recorder-receipt-sha256", prerequisites["recorder_receipt"]["sha256"],
        "--d1-qualification-receipt", prerequisites["d1_qualification_receipt"]["path"],
        "--d1-qualification-receipt-sha256", prerequisites["d1_qualification_receipt"]["sha256"],
        "--pilot-simulator-receipt", prerequisites["d1_pilot_simulator_receipt"]["path"],
        "--pilot-simulator-receipt-sha256", prerequisites["d1_pilot_simulator_receipt"]["sha256"],
        "--pilot-server-receipt", prerequisites["d1_pilot_server_receipt"]["path"],
        "--pilot-server-receipt-sha256", prerequisites["d1_pilot_server_receipt"]["sha256"],
    ]
    runner = "{source_root}/" + contract["runtime"]["D1"]["runner"]
    server_argv = [
        "/usr/bin/python3", runner, contract["runtime"]["D1"]["server_mode"],
        *common,
        "--job-dir", "{job_dir}",
        "--job-id", server_id,
        "--simulator-job-id", simulator_id,
        "--run-id", attempt_id,
        *evidence,
        "--port", str(contract["runtime"]["D1"]["service_port"]),
    ]
    simulator_argv = [
        "/usr/bin/python3", runner, contract["runtime"]["D1"]["simulator_mode"],
        *common,
        "--job-dir", "{job_dir}",
        "--job-id", simulator_id,
        "--server-job-id", server_id,
        "--run-id", attempt_id,
        *evidence,
        "--remote-host", contract["runtime"]["D1"]["service_host"],
        "--remote-port", str(contract["runtime"]["D1"]["service_port"]),
    ]
    raw = [
        {
            "job_id": server_id, "released": True, "source_commit": study_commit,
            "role": contract["runtime"]["D1"]["server_role"], "argv": server_argv,
            "max_wall_seconds": contract["maximum_job_wall_seconds"],
            "publish_log_tail_bytes": contract["publish_log_tail_bytes"],
        },
        {
            "job_id": simulator_id, "released": True, "source_commit": study_commit,
            "role": simulator_role, "argv": simulator_argv,
            "max_wall_seconds": contract["maximum_job_wall_seconds"],
            "publish_log_tail_bytes": contract["publish_log_tail_bytes"],
        },
    ]
    return attempt_id, raw, [queue.normalize_job(row) for row in raw]


def _round_trip_new_jobs(
    *, raw_jobs: Sequence[Mapping[str, Any]], normalized_jobs: Sequence[Mapping[str, Any]],
    model: str, layout: str, study_commit: str, runtime: Mapping[str, Any],
) -> None:
    require(len(raw_jobs) == len(normalized_jobs), "new queue normalization count changed")
    expected_ids = [row["job_id"] for row in normalized_jobs]
    for raw, normalized in zip(raw_jobs, normalized_jobs, strict=True):
        require(
            runtime["queue"].normalize_job(dict(raw)) == dict(normalized),
            "new queue normalization was not deterministic",
        )
        argv = normalized.get("argv")
        options = [item for item in argv if isinstance(item, str) and item.startswith("--")]
        require(len(options) == len(set(options)), "new queue argv contains a duplicate option")
        _parse_descriptor(
            normalized, model=model, layout=layout, study_commit=study_commit,
            runtime=runtime, expected_job_ids=expected_ids,
        )


def _select_lane(
    *, models: Sequence[str], layout_order: Sequence[str],
    ledger_blocks: Mapping[tuple[str, str], Mapping[str, Any]],
    lane_priority: Sequence[str],
) -> tuple[str | None, list[str]]:
    for model in lane_priority:
        if model not in models:
            continue
        remaining = [
            layout for layout in layout_order
            if ledger_blocks[(model, layout)]["state"] in {"not_run", "technical_invalid"}
        ]
        if remaining:
            return model, remaining
    return None, []


def _validate_selected_prerequisites(
    *, model: str, block: Any, fixture: Mapping[str, Any], source_root: Path,
    study_commit: str, release_identity: Mapping[str, Any],
    fixture_identity: Mapping[str, Any], prerequisites: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> None:
    values: dict[str, Any] = {
        "source_root": Path(source_root).resolve(),
        "study_commit": study_commit,
        "candidate_id": fixture["candidate_id"],
        "gate_receipt": Path(fixture["gate_receipt"]["path"]),
        "gate_receipt_sha256": fixture["gate_receipt"]["sha256"],
        "pose_manifest": Path(fixture["pose_manifest"]["path"]),
        "pose_manifest_sha256": fixture["pose_manifest"]["sha256"],
        "capture_receipt": Path(fixture["capture_receipt"]["path"]),
        "capture_receipt_sha256": fixture["capture_receipt"]["sha256"],
        "confirmation_freeze": Path(release_identity["path"]),
        "confirmation_freeze_sha256": release_identity["sha256"],
        "fixture_freeze": Path(fixture_identity["path"]),
        "fixture_freeze_sha256": fixture_identity["sha256"],
    }
    if model == "N3":
        values.update({
            "pilot_receipt": Path(prerequisites["n3_pilot_receipt"]["path"]),
            "pilot_receipt_sha256": prerequisites["n3_pilot_receipt"]["sha256"],
        })
    else:
        for target, name in (
            ("recorder_receipt", "recorder_receipt"),
            ("d1_qualification_receipt", "d1_qualification_receipt"),
            ("pilot_simulator_receipt", "d1_pilot_simulator_receipt"),
            ("pilot_server_receipt", "d1_pilot_server_receipt"),
        ):
            values[target] = Path(prerequisites[name]["path"])
            values[target + "_sha256"] = prerequisites[name]["sha256"]
    try:
        if model == "N3":
            runtime["n3"].validate_prerequisites(
                argparse.Namespace(**values), block, include_pilot=True
            )
        else:
            runtime["d1"].validate_prerequisites(argparse.Namespace(**values), block)
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            f"selected {model} runtime prerequisites failed deep validation: {block.layout_pair_id}"
        ) from error


def build_pending_release_wave(
    *, source_root: Path, state_dir: Path, study_commit: str,
    confirmation_freeze: Path, confirmation_freeze_sha256: str,
    fixture_freeze: Path, fixture_freeze_sha256: str,
    resource_qualification: Path, resource_qualification_sha256: str,
    terminal_runtime_identities: Path, terminal_runtime_identities_sha256: str,
    result_attempt_ledger: Path, result_attempt_ledger_sha256: str,
    cluster_reconciliation: Path, cluster_reconciliation_sha256: str,
) -> dict[str, Any]:
    """Build an inert wave that still requires coordinator H1 verification."""

    root = Path(source_root).resolve()
    require(root.is_dir(), "source root is missing")
    state = Path(state_dir).resolve()
    require(state.is_dir() and not state.is_symlink(), "queue state directory is missing or unsafe")
    require(
        isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
        "study commit is invalid",
    )
    contract = load_contract(root)
    builder_identity = file_identity(root / BUILDER_RELATIVE)
    contract_identity = file_identity(root / CONTRACT_RELATIVE)
    validate_study_commit(root, study_commit, contract)
    runtime = load_runtime_modules(root)
    runtime["contract"] = contract
    evidence = load_release_evidence_module(root)
    release_identity = verify_input(
        confirmation_freeze, confirmation_freeze_sha256, "confirmation release freeze"
    )
    fixture_identity = verify_input(
        fixture_freeze, fixture_freeze_sha256, "confirmation fixture freeze"
    )
    resource_identity = verify_input(
        resource_qualification, resource_qualification_sha256, "resource qualification"
    )
    runtime_identity = verify_input(
        terminal_runtime_identities, terminal_runtime_identities_sha256,
        "terminal runtime identities",
    )
    ledger_identity = verify_input(
        result_attempt_ledger, result_attempt_ledger_sha256, "result attempt ledger"
    )
    reconciliation_identity = verify_input(
        cluster_reconciliation, cluster_reconciliation_sha256, "cluster reconciliation"
    )
    schedule_identity, layout_order, blocks = validate_schedule(
        source_root=root, contract=contract, runtime=runtime
    )
    release, _release_validations, models = validate_release_freeze(
        descriptor=release_identity, source_root=root, contract=contract, runtime=runtime
    )
    branch = str(release["cohort_branch"])
    try:
        fixture_validation = runtime["fixture"].validate_fixture_freeze(
            Path(fixture_identity["path"]), fixture_identity["sha256"],
            source_root=root, expected_study_commit=study_commit,
            deep_validate_selected=False,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError("confirmation fixture freeze failed native validation") from error
    require(
        fixture_validation.get("layout_count") == 24,
        "fixture freeze does not retain all 24 base layouts",
    )
    observed_fixture_identity, fixture_value = load_json_with_identity(
        Path(fixture_identity["path"]), "confirmation fixture freeze",
    )
    require(
        observed_fixture_identity == fixture_identity,
        "confirmation fixture freeze changed after native validation",
    )
    fixture_rows = {
        row["layout_pair_id"]: row
        for row in fixture_value.get("layouts", []) if isinstance(row, Mapping)
    }
    require(set(fixture_rows) == set(layout_order), "fixture layout inventory changed")
    resource, resource_limits = validate_resource_qualification(
        descriptor=resource_identity, study_commit=study_commit, models=models,
        release=release, contract=contract,
    )
    runtime_publication, prerequisites = validate_runtime_identities(
        descriptor=runtime_identity, source_root=root, study_commit=study_commit,
        contract=contract,
    )
    ledger, ledger_blocks, all_attempt_ids, all_job_ids, queue_claims = validate_result_ledger(
        descriptor=ledger_identity, study_commit=study_commit, branch=branch,
        source_root=root, state_dir=state, evidence=evidence,
        models=models, layout_order=layout_order, blocks=blocks,
        schedule_identity=schedule_identity, release_identity=release_identity,
        fixture_identity=fixture_identity, resource_identity=resource_identity,
        runtime_identity=runtime_identity, runtime=runtime,
        fixture_rows=fixture_rows, prerequisites=prerequisites,
    )
    ledger_validation = {
        "document": ledger,
        "blocks": ledger_blocks,
        "attempt_ids": all_attempt_ids,
        "job_ids": all_job_ids,
        "queue_claims": queue_claims,
    }
    reconciliation, workers, all_prior_job_ids = validate_pending_reconciliation(
        descriptor=reconciliation_identity, ledger_identity=ledger_identity,
        ledger_validation=ledger_validation, all_attempt_ids=all_attempt_ids,
        all_job_ids=all_job_ids, source_root=root, state_dir=state,
        study_commit=study_commit, contract=contract, runtime=runtime,
        evidence=evidence,
    )
    as_of_utc = reconciliation["created_at_utc"]
    consume_by_utc = reconciliation["consume_by_utc"]
    parse_utc(as_of_utc, "signed reconciliation creation time")
    parse_utc(consume_by_utc, "signed reconciliation consume-by time")
    finalizer_job_ids = reconciliation["control_semantics"]["active_job_ids"]
    require(
        isinstance(finalizer_job_ids, list) and len(finalizer_job_ids) == 1,
        "signed reconciliation does not identify exactly one finalizer",
    )
    finalizer_job_id = safe_id(finalizer_job_ids[0], "release finalizer job ID")
    lane, remaining = _select_lane(
        models=models, layout_order=layout_order, ledger_blocks=ledger_blocks,
        lane_priority=contract["lane_priority"],
    )
    raw_jobs: list[dict[str, Any]] = []
    normalized_jobs: list[dict[str, Any]] = []
    prospective_attempts: list[dict[str, Any]] = []
    selected_layouts: list[str] = []
    if lane == "N3":
        idle_n3 = [
            worker for worker in workers
            if worker["role"] == contract["runtime"]["N3"]["role"]
            and worker["gpu_count"] == contract["runtime"]["N3"]["gpu_count_per_worker"]
        ]
        cap = min(resource_limits["N3"], len(idle_n3), len(remaining))
        require(cap > 0, "no freshly reconciled isolated N3 worker capacity")
        selected_layouts = remaining[:cap]
        for layout in selected_layouts:
            block_row = ledger_blocks[("N3", layout)]
            _validate_selected_prerequisites(
                model="N3", block=blocks[("N3", layout)], fixture=fixture_rows[layout],
                source_root=root, study_commit=study_commit,
                release_identity=release_identity, fixture_identity=fixture_identity,
                prerequisites=prerequisites, runtime=runtime,
            )
            attempt_number = len(block_row["attempts"]) + 1
            attempt_id, raw, normalized = _n3_descriptor(
                layout=layout, attempt_number=attempt_number, study_commit=study_commit,
                fixture=fixture_rows[layout], release_identity=release_identity,
                fixture_identity=fixture_identity, prerequisites=prerequisites,
                finalizer_job_id=finalizer_job_id, consume_by_utc=consume_by_utc,
                contract=contract, queue=runtime["queue"],
            )
            _round_trip_new_jobs(
                raw_jobs=[raw], normalized_jobs=[normalized], model="N3", layout=layout,
                study_commit=study_commit, runtime=runtime,
            )
            raw_jobs.append(raw)
            normalized_jobs.append(normalized)
            prospective_attempts.append({
                "model_config": "N3", "layout_pair_id": layout,
                "attempt_id": attempt_id, "attempt_number": attempt_number,
                "predecessor_attempt_id": (
                    block_row["attempts"][-1]["attempt_id"] if block_row["attempts"] else None
                ),
                "start_cell_index": block_row["completed_prefix_cells"],
                "job_ids": [raw["job_id"]],
            })
    elif lane == "D1":
        idle_servers = [
            worker for worker in workers
            if worker["role"] == contract["runtime"]["D1"]["server_role"]
            and worker["gpu_count"] == contract["runtime"]["D1"]["server_gpu_count"]
        ]
        by_sim_role = {
            worker["role"]: worker for worker in workers
            if worker["role"] in contract["runtime"]["D1"]["simulator_roles"]
            and worker["gpu_count"] == contract["runtime"]["D1"]["simulator_gpu_count"]
        }
        simulator_roles = [
            role for role in contract["runtime"]["D1"]["simulator_roles"] if role in by_sim_role
        ]
        require(idle_servers and simulator_roles, "no freshly reconciled D1 server/simulator pair")
        require(
            contract["runtime"]["D1"]["global_server_concurrency"] == 1,
            "D1 global server concurrency changed",
        )
        layout = remaining[0]
        selected_layouts = [layout]
        block_row = ledger_blocks[("D1", layout)]
        _validate_selected_prerequisites(
            model="D1", block=blocks[("D1", layout)], fixture=fixture_rows[layout],
            source_root=root, study_commit=study_commit,
            release_identity=release_identity, fixture_identity=fixture_identity,
            prerequisites=prerequisites, runtime=runtime,
        )
        attempt_number = len(block_row["attempts"]) + 1
        attempt_id, raws, normalized = _d1_descriptors(
            layout=layout, attempt_number=attempt_number, study_commit=study_commit,
            simulator_role=simulator_roles[0], fixture=fixture_rows[layout],
            release_identity=release_identity, fixture_identity=fixture_identity,
            prerequisites=prerequisites, contract=contract, queue=runtime["queue"],
            finalizer_job_id=finalizer_job_id, consume_by_utc=consume_by_utc,
        )
        _round_trip_new_jobs(
            raw_jobs=raws, normalized_jobs=normalized, model="D1", layout=layout,
            study_commit=study_commit, runtime=runtime,
        )
        raw_jobs.extend(raws)
        normalized_jobs.extend(normalized)
        prospective_attempts.append({
            "model_config": "D1", "layout_pair_id": layout,
            "attempt_id": attempt_id, "attempt_number": attempt_number,
            "predecessor_attempt_id": (
                block_row["attempts"][-1]["attempt_id"] if block_row["attempts"] else None
            ),
            "start_cell_index": block_row["completed_prefix_cells"],
            "job_ids": [row["job_id"] for row in raws],
        })
    require(
        len(raw_jobs) == len({row["job_id"] for row in raw_jobs})
        and not ({row["job_id"] for row in raw_jobs} & all_prior_job_ids)
        and not ({row["attempt_id"] for row in prospective_attempts} & all_attempt_ids),
        "prospective immutable IDs collide with prior evidence",
    )
    queue_fragment = {
        "schema_version": QUEUE_SCHEMA,
        "namespace": NAMESPACE,
        "shutdown": False,
        "jobs": raw_jobs,
    }
    descriptor_hashes = {
        row["job_id"]: sha256_bytes(runtime["queue"].encode(row)) for row in normalized_jobs
    }
    queue_sha = sha256_bytes(pretty_bytes(queue_fragment))
    input_identities = {
        "release_wave_builder": builder_identity,
        "release_wave_contract": contract_identity,
        "prepared_schedule": schedule_identity,
        "confirmation_freeze": release_identity,
        "fixture_freeze": fixture_identity,
        "resource_qualification": resource_identity,
        "terminal_runtime_identities": runtime_identity,
        "result_attempt_ledger": ledger_identity,
        "cluster_reconciliation": reconciliation_identity,
    }
    state_counts = {
        state: sum(1 for row in ledger_blocks.values() if row["state"] == state)
        for state in ("not_run", "technical_invalid", "passed", "safety_censored")
    }
    layout_accounting = {
        model: [
            {
                "layout_pair_id": layout,
                "state": ledger_blocks[(model, layout)]["state"],
                "completed_prefix_cells": ledger_blocks[(model, layout)]["completed_prefix_cells"],
                "attempt_count": len(ledger_blocks[(model, layout)]["attempts"]),
            }
            for layout in layout_order
        ]
        for model in models
    }
    plan = signed_document({
        "schema_version": PLAN_SCHEMA,
        "status": "pending_h1_verification",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "as_of_utc": as_of_utc,
        "consume_by_utc": consume_by_utc,
        "release_finalizer_job_id": finalizer_job_id,
        "cohort_branch": branch,
        "qualified_model_ids": list(models),
        "layout_order": list(layout_order),
        "input_identities": input_identities,
        "control_generation_lower_bound": reconciliation[
            "observed_control_generation_lower_bound"
        ],
        "observed_control_generation": reconciliation["observed_control_generation"],
        "confirmation_release": False,
        "execution_release_authorized": False,
        "post_publication_verification_required": True,
        "external_evidence_producer_boundary": contract["execution_readiness"],
        "builder_synthesized_external_evidence": False,
        "wave_model": lane,
        "selected_layouts": selected_layouts,
        "prospective_attempts": prospective_attempts,
        "ledger_state_counts": state_counts,
        "layout_accounting_by_model": layout_accounting,
        "descriptor_sha256": descriptor_hashes,
        "queue_fragment_sha256": queue_sha,
        "queue_job_count": len(raw_jobs),
        "d1_global_server_concurrency": 1,
        "outcome_dependent_ordering": False,
        "base_layouts_per_selected_model": 24,
        "publish_log_tail_bytes": 0,
        "claim_boundary": (
            "Source-only plan from externally signed evidence. It does not mutate, stage, "
            "claim, dispatch, or publish the active cluster queue."
        ),
    })
    plan_sha = sha256_bytes(pretty_bytes(plan))
    receipt = signed_document({
        "schema_version": RECEIPT_SCHEMA,
        "status": "source_only_release_wave_built_pending_h1_verification",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "as_of_utc": as_of_utc,
        "consume_by_utc": consume_by_utc,
        "release_finalizer_job_id": finalizer_job_id,
        "cohort_branch": branch,
        "qualified_model_ids": list(models),
        "confirmation_release": False,
        "execution_release_authorized": False,
        "post_publication_verification_required": True,
        "external_evidence_producer_boundary": contract["execution_readiness"],
        "builder_synthesized_external_evidence": False,
        "input_identities": input_identities,
        "plan_sha256": plan_sha,
        "queue_fragment_sha256": queue_sha,
        "descriptor_sha256": descriptor_hashes,
        "queue_job_ids": [row["job_id"] for row in raw_jobs],
        "attempt_ids": [row["attempt_id"] for row in prospective_attempts],
        "queue_mutated": False,
        "jobs_dispatched": 0,
        "published_log_tail_bytes": 0,
        "claim_boundary": (
            "Receipt for deterministic descriptor construction only; cluster state remains "
            "external authority and no reconciliation or ledger fact was synthesized."
        ),
    })
    return {
        "plan": plan,
        "queue_fragment": queue_fragment,
        "receipt": receipt,
    }


def _validate_published_wave_artifacts(
    *, source_root: Path, study_commit: str, contract: Mapping[str, Any],
    runtime: Mapping[str, Any], evidence: Any, verification: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay the inert published wave before granting local release authority."""

    exact_keys(
        verification,
        {
            "document", "publication_commit", "verified_remote_head",
            "artifact_descriptors", "cluster_status", "producer_result", "artifacts",
        },
        "published-finalizer verification result",
    )
    document = verification.get("document")
    artifacts = verification.get("artifacts")
    remote = verification.get("artifact_descriptors")
    require(
        isinstance(document, Mapping) and isinstance(artifacts, Mapping)
        and isinstance(remote, Mapping),
        "published-finalizer verification result is incomplete",
    )
    exact_keys(
        document,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "verified_at_utc", "consume_by_utc", "results_remote", "results_ref",
            "observed_results_ancestor_commit", "pending_publication_commit",
            "verified_remote_head", "intervening_commit_count",
            "later_status_only_commit_count", "finalizer_job_id",
            "artifact_descriptors", "producer_outer_result", "semantic_control",
            "queue_fragment_release_gate_passed", "results_branch_written_by_verifier",
            "study_worktree_mutated", "claim_boundary", "payload_sha256",
        },
        "publication verification",
    )
    verify_signed_document(document, "publication verification")
    trusted = contract["trusted_publication"]
    publication_commit = verification.get("publication_commit")
    verified_head = verification.get("verified_remote_head")
    require(
        document.get("schema_version") == PUBLICATION_VERIFICATION_SCHEMA
        and document.get("status") == "coordinator_publication_verified_read_only"
        and document.get("study_id") == STUDY_ID
        and document.get("namespace") == NAMESPACE
        and document.get("study_commit") == study_commit
        and document.get("results_remote") == trusted["repository_identity"]
        and document.get("results_ref") == trusted["results_ref"]
        and document.get("pending_publication_commit") == publication_commit
        and document.get("verified_remote_head") == verified_head
        and isinstance(publication_commit, str)
        and COMMIT_RE.fullmatch(publication_commit) is not None
        and isinstance(verified_head, str) and COMMIT_RE.fullmatch(verified_head) is not None
        and document.get("artifact_descriptors") == remote
        and document.get("producer_outer_result") == verification.get("producer_result")
        and document.get("queue_fragment_release_gate_passed") is True
        and document.get("results_branch_written_by_verifier") is False
        and document.get("study_worktree_mutated") is False,
        "publication verification does not authorize this source-only wave",
    )
    verified_at = parse_utc(document.get("verified_at_utc"), "publication verification time")
    consume_by = parse_utc(document.get("consume_by_utc"), "publication consume-by time")
    require(verified_at <= consume_by, "publication verification is past consume-by time")
    finalizer_id = safe_id(document.get("finalizer_job_id"), "published finalizer job ID")

    exact_keys(
        artifacts,
        {
            "ledger", "reconciliation", "plan", "queue_fragment",
            "wave_receipt", "publication_pending", "evidence_receipt",
            "worker_attestation",
        },
        "published release artifacts",
    )
    plan = artifacts.get("plan")
    queue_fragment = artifacts.get("queue_fragment")
    receipt = artifacts.get("wave_receipt")
    pending = artifacts.get("publication_pending")
    require(
        all(isinstance(item, Mapping) for item in (plan, queue_fragment, receipt, pending)),
        "published wave artifacts are not JSON objects",
    )
    try:
        validated_pending = evidence.validate_publication_pending(
            pending, study_commit=study_commit, verify_local_artifacts=False,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "published pending receipt failed native producer validation"
        ) from error
    require(
        validated_pending == dict(pending)
        and pending.get("producer_queue_job", {}).get("job_id") == finalizer_id,
        "published pending receipt belongs to another release finalizer",
    )
    expected_h1_paths = pending.get("expected_h1_paths")
    require(
        isinstance(expected_h1_paths, Mapping)
        and set(expected_h1_paths) == {
            "ledger", "reconciliation", "plan", "queue_fragment", "wave_receipt",
            "pending", "evidence_receipt", "worker_attestation", "cluster_status",
            "producer_job_status", "producer_publish_manifest",
        }
        and set(remote) == {
            "ledger", "reconciliation", "plan", "queue_fragment", "wave_receipt",
            "publication_pending", "evidence_receipt", "worker_attestation",
            "cluster_status", "producer_job_status", "producer_publish_manifest",
        },
        "published finalizer path or remote artifact inventory changed",
    )

    exact_keys(
        plan,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "as_of_utc", "consume_by_utc", "release_finalizer_job_id",
            "cohort_branch", "qualified_model_ids", "layout_order",
            "input_identities", "control_generation_lower_bound",
            "observed_control_generation", "confirmation_release",
            "execution_release_authorized", "post_publication_verification_required",
            "external_evidence_producer_boundary", "builder_synthesized_external_evidence",
            "wave_model", "selected_layouts", "prospective_attempts",
            "ledger_state_counts", "layout_accounting_by_model", "descriptor_sha256",
            "queue_fragment_sha256", "queue_job_count", "d1_global_server_concurrency",
            "outcome_dependent_ordering", "base_layouts_per_selected_model",
            "publish_log_tail_bytes", "claim_boundary", "payload_sha256",
        },
        "published release plan",
    )
    exact_keys(
        receipt,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "as_of_utc", "consume_by_utc", "release_finalizer_job_id",
            "cohort_branch", "qualified_model_ids", "confirmation_release",
            "execution_release_authorized", "post_publication_verification_required",
            "external_evidence_producer_boundary", "builder_synthesized_external_evidence",
            "input_identities", "plan_sha256", "queue_fragment_sha256",
            "descriptor_sha256", "queue_job_ids", "attempt_ids", "queue_mutated",
            "jobs_dispatched", "published_log_tail_bytes", "claim_boundary",
            "payload_sha256",
        },
        "published release-wave receipt",
    )
    verify_signed_document(plan, "published release plan")
    verify_signed_document(receipt, "published release-wave receipt")
    require(
        plan.get("schema_version") == PLAN_SCHEMA
        and plan.get("status") == "pending_h1_verification"
        and receipt.get("schema_version") == RECEIPT_SCHEMA
        and receipt.get("status")
        == "source_only_release_wave_built_pending_h1_verification"
        and all(row.get("study_id") == STUDY_ID for row in (plan, receipt))
        and all(row.get("namespace") == NAMESPACE for row in (plan, receipt))
        and all(row.get("study_commit") == study_commit for row in (plan, receipt))
        and all(row.get("confirmation_release") is False for row in (plan, receipt))
        and all(row.get("execution_release_authorized") is False for row in (plan, receipt))
        and all(row.get("post_publication_verification_required") is True for row in (plan, receipt))
        and all(row.get("builder_synthesized_external_evidence") is False for row in (plan, receipt))
        and all(row.get("external_evidence_producer_boundary") == contract["execution_readiness"] for row in (plan, receipt))
        and receipt.get("input_identities") == plan.get("input_identities")
        and receipt.get("plan_sha256") == sha256_bytes(pretty_bytes(plan)),
        "published plan/receipt release boundary changed",
    )
    parse_utc(plan.get("as_of_utc"), "published plan as-of time")
    plan_consume_by = parse_utc(plan.get("consume_by_utc"), "published plan consume-by time")
    require(
        receipt.get("as_of_utc") == plan.get("as_of_utc")
        and receipt.get("consume_by_utc") == plan.get("consume_by_utc")
        and plan.get("consume_by_utc") == document.get("consume_by_utc")
        and plan.get("consume_by_utc") == pending.get("consume_by_utc")
        and artifacts["reconciliation"].get("consume_by_utc") == plan.get("consume_by_utc")
        and plan_consume_by >= verified_at
        and receipt.get("release_finalizer_job_id") == plan.get("release_finalizer_job_id")
        and plan.get("release_finalizer_job_id") == finalizer_id,
        "published plan/receipt/finalizer capacity lifetime differs",
    )

    inputs = plan.get("input_identities")
    require(isinstance(inputs, Mapping), "published plan input identities missing")
    exact_keys(
        inputs,
        {
            "release_wave_builder", "release_wave_contract", "prepared_schedule",
            "confirmation_freeze", "fixture_freeze", "resource_qualification",
            "terminal_runtime_identities", "result_attempt_ledger",
            "cluster_reconciliation",
        },
        "published plan input identities",
    )
    for name, row in inputs.items():
        require(
            isinstance(row, Mapping) and set(row) == {"path", "bytes", "sha256"}
            and isinstance(row.get("path"), str) and Path(row["path"]).is_absolute()
            and type(row.get("bytes")) is int and row["bytes"] >= 0
            and isinstance(row.get("sha256"), str)
            and SHA256_RE.fullmatch(row["sha256"]) is not None,
            f"published plan input identity is invalid: {name}",
        )
    require(
        inputs["result_attempt_ledger"] == pending["artifacts"]["ledger"]
        and inputs["cluster_reconciliation"] == pending["artifacts"]["reconciliation"],
        "published plan is detached from its ledger or reconciliation",
    )
    for name, value in (
        ("release_wave_builder", file_identity(Path(source_root).resolve() / BUILDER_RELATIVE)),
        ("release_wave_contract", file_identity(Path(source_root).resolve() / CONTRACT_RELATIVE)),
    ):
        require(
            inputs[name]["bytes"] == value["bytes"]
            and inputs[name]["sha256"] == value["sha256"],
            f"published plan used another {name}",
        )

    expected_serialized = {
        "ledger": artifacts["ledger"],
        "reconciliation": artifacts["reconciliation"],
        "plan": plan,
        "queue_fragment": queue_fragment,
        "wave_receipt": receipt,
        "publication_pending": pending,
        "evidence_receipt": artifacts["evidence_receipt"],
        "worker_attestation": artifacts["worker_attestation"],
    }
    for name, value in expected_serialized.items():
        blob = pretty_bytes(value)
        local_descriptor = pending["artifacts"].get(name)
        if name in {"ledger", "reconciliation", "plan", "queue_fragment", "wave_receipt"}:
            require(
                isinstance(local_descriptor, Mapping)
                and local_descriptor.get("bytes") == len(blob)
                and local_descriptor.get("sha256") == sha256_bytes(blob),
                f"pending receipt does not bind published artifact: {name}",
            )
        remote_descriptor = remote.get(name)
        expected_path_key = "pending" if name == "publication_pending" else name
        require(
            isinstance(remote_descriptor, Mapping)
            and set(remote_descriptor) == {"commit", "git_path", "bytes", "sha256"}
            and remote_descriptor.get("commit") == publication_commit
            and remote_descriptor.get("git_path") == expected_h1_paths[expected_path_key]
            and remote_descriptor.get("bytes") == len(blob)
            and remote_descriptor.get("sha256") == sha256_bytes(blob),
            f"remote publication does not bind release artifact: {name}",
        )

    exact_keys(queue_fragment, {"schema_version", "namespace", "shutdown", "jobs"}, "published queue fragment")
    jobs = queue_fragment.get("jobs")
    require(
        queue_fragment.get("schema_version") == QUEUE_SCHEMA
        and queue_fragment.get("namespace") == NAMESPACE
        and queue_fragment.get("shutdown") is False
        and isinstance(jobs, list),
        "published queue fragment identity changed",
    )
    queue_sha = sha256_bytes(pretty_bytes(queue_fragment))
    require(
        plan.get("queue_fragment_sha256") == queue_sha
        and receipt.get("queue_fragment_sha256") == queue_sha
        and plan.get("queue_job_count") == len(jobs)
        and plan.get("publish_log_tail_bytes") == 0
        and receipt.get("published_log_tail_bytes") == 0
        and receipt.get("queue_mutated") is False
        and receipt.get("jobs_dispatched") == 0,
        "published queue fragment count/hash/release boundary changed",
    )
    prospective = plan.get("prospective_attempts")
    selected = plan.get("selected_layouts")
    require(
        isinstance(prospective, list) and isinstance(selected, list)
        and selected == [row.get("layout_pair_id") for row in prospective if isinstance(row, Mapping)],
        "published prospective attempt order changed",
    )
    jobs_by_id: dict[str, dict[str, Any]] = {}
    ordered_job_ids: list[str] = []
    attempt_ids: list[str] = []
    for attempt in prospective:
        require(isinstance(attempt, Mapping), "published prospective attempt is invalid")
        exact_keys(
            attempt,
            {
                "model_config", "layout_pair_id", "attempt_id", "attempt_number",
                "predecessor_attempt_id", "start_cell_index", "job_ids",
            },
            "published prospective attempt",
        )
        model = attempt.get("model_config")
        layout = attempt.get("layout_pair_id")
        attempt_id = safe_id(attempt.get("attempt_id"), "published attempt ID")
        attempt_job_ids = attempt.get("job_ids")
        require(
            model in {"N3", "D1"} and model == plan.get("wave_model")
            and isinstance(layout, str) and layout in contract["prepared_schedule"]["confirmation_blocks_in_hash_order"]
            and isinstance(attempt_job_ids, list)
            and len(attempt_job_ids) == (1 if model == "N3" else 2),
            "published prospective attempt topology changed",
        )
        raw_rows: list[dict[str, Any]] = []
        normalized_rows: list[dict[str, Any]] = []
        for job_id in attempt_job_ids:
            safe_id(job_id, "published queue job ID")
            matches = [dict(row) for row in jobs if isinstance(row, Mapping) and row.get("job_id") == job_id]
            require(len(matches) == 1 and job_id not in jobs_by_id, "published queue job is missing or duplicated")
            raw = matches[0]
            normalized = runtime["queue"].normalize_job(raw)
            require(
                normalized.get("released") is True
                and normalized.get("source_commit") == study_commit
                and normalized.get("publish_log_tail_bytes") == 0,
                "published queue job is not an exact releasable zero-tail descriptor",
            )
            jobs_by_id[job_id] = raw
            raw_rows.append(raw)
            normalized_rows.append(normalized)
        _round_trip_new_jobs(
            raw_jobs=raw_rows, normalized_jobs=normalized_rows,
            model=str(model), layout=str(layout), study_commit=study_commit, runtime=runtime,
        )
        ordered_job_ids.extend(attempt_job_ids)
        attempt_ids.append(attempt_id)
    require(
        ordered_job_ids == [row.get("job_id") for row in jobs]
        and len(ordered_job_ids) == len(set(ordered_job_ids))
        and receipt.get("queue_job_ids") == ordered_job_ids
        and receipt.get("attempt_ids") == attempt_ids
        and receipt.get("descriptor_sha256") == plan.get("descriptor_sha256")
        and set(plan.get("descriptor_sha256", {})) == set(ordered_job_ids),
        "published queue/attempt/descriptor inventory changed",
    )
    for job_id, raw in jobs_by_id.items():
        require(
            plan["descriptor_sha256"][job_id]
            == sha256_bytes(runtime["queue"].encode(runtime["queue"].normalize_job(raw))),
            f"published queue descriptor hash changed: {job_id}",
        )
    return dict(plan), dict(queue_fragment), dict(receipt), dict(document)


def verify_published_release_wave(
    *, source_root: Path, finalizer_job_id: str, study_commit: str,
) -> dict[str, Any]:
    """Return release authority only after read-only coordinator-H1 verification."""

    root = Path(source_root).resolve()
    safe_id(finalizer_job_id, "finalizer job ID")
    require(
        isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
        "study commit is invalid",
    )
    contract = load_contract(root)
    validate_study_commit(root, study_commit, contract)
    runtime = load_runtime_modules(root)
    runtime["contract"] = contract
    evidence = load_release_evidence_module(root)
    try:
        verification = evidence.verify_published_finalizer(
            source_root=root, finalizer_job_id=finalizer_job_id,
            study_commit=study_commit,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "coordinator publication failed native read-only verification"
        ) from error
    require(isinstance(verification, Mapping), "native publication verification returned no result")
    plan, queue_fragment, receipt, verification_document = _validate_published_wave_artifacts(
        source_root=root, study_commit=study_commit, contract=contract,
        runtime=runtime, evidence=evidence, verification=verification,
    )
    authorization = signed_document({
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "confirmation_release_authorized_after_coordinator_publication",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "verified_at_utc": verification_document["verified_at_utc"],
        "consume_by_utc": verification_document["consume_by_utc"],
        "finalizer_job_id": finalizer_job_id,
        "cohort_branch": plan["cohort_branch"],
        "qualified_model_ids": plan["qualified_model_ids"],
        "wave_model": plan["wave_model"],
        "selected_layouts": plan["selected_layouts"],
        "queue_job_ids": receipt["queue_job_ids"],
        "confirmation_release": True,
        "execution_release_authorized": True,
        "publication_verification_payload_sha256": verification_document["payload_sha256"],
        "pending_publication_commit": verification_document["pending_publication_commit"],
        "verified_remote_head": verification_document["verified_remote_head"],
        "published_plan": verification["artifact_descriptors"]["plan"],
        "published_queue_fragment": verification["artifact_descriptors"]["queue_fragment"],
        "published_wave_receipt": verification["artifact_descriptors"]["wave_receipt"],
        "plan_sha256": sha256_bytes(pretty_bytes(plan)),
        "queue_fragment_sha256": sha256_bytes(pretty_bytes(queue_fragment)),
        "wave_receipt_sha256": sha256_bytes(pretty_bytes(receipt)),
        "queue_mutated": False,
        "jobs_dispatched": 0,
        "results_branch_written_by_verifier": False,
        "claim_boundary": (
            "This signed result authorizes copying only the exact published queue fragment "
            "through the normal control-branch workflow; it does not mutate or dispatch it."
        ),
    })
    return {
        "authorization": authorization,
        "plan": plan,
        "queue_fragment": queue_fragment,
        "receipt": receipt,
        "publication_verification": verification_document,
    }


def _current_utc() -> datetime:
    """Return the process wall clock; tests patch this private seam only."""

    return datetime.now(timezone.utc)


def _gpu_identity(value: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in value:
        require(isinstance(row, Mapping), f"{label} GPU row is invalid")
        identity = {
            "index": row.get("index"),
            "uuid": row.get("uuid"),
            "name": row.get("name"),
            "memory_total_mib": row.get("memory_total_mib"),
        }
        require(
            type(identity["index"]) is int and identity["index"] >= 0
            and isinstance(identity["uuid"], str) and bool(identity["uuid"])
            and isinstance(identity["name"], str) and bool(identity["name"])
            and type(identity["memory_total_mib"]) is int
            and identity["memory_total_mib"] > 0,
            f"{label} GPU identity is invalid",
        )
        result.append(identity)
    require(
        len({row["index"] for row in result}) == len(result)
        and len({row["uuid"] for row in result}) == len(result),
        f"{label} GPU identity is duplicated",
    )
    return sorted(result, key=lambda row: row["index"])


def validate_runtime_release_admission(
    *, source_root: Path, state_dir: Path, job_dir: Path, job_id: str,
    study_commit: str, expected_role: str, finalizer_job_id: str,
    consume_by_utc: str,
) -> dict[str, Any]:
    """Re-authenticate one H1-published job on its claimed worker.

    This is intentionally a live, read-only gate.  No caller supplies a clock,
    worker identity, pod identity, GPU identity, queue descriptor, or claim.
    The only admitted stage is the outer queue-process start, where the GPUs
    must still be idle. Later model loading cannot mint or refresh authority.
    """

    root = Path(source_root).resolve()
    state = Path(state_dir).resolve()
    job_root = Path(job_dir).resolve()
    safe_id(job_id, "runtime admission queue job")
    safe_id(finalizer_job_id, "runtime admission finalizer")
    safe_id(expected_role, "runtime admission role")
    require(
        isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
        "runtime admission study commit invalid",
    )
    supplied_consume_by = parse_utc(consume_by_utc, "runtime admission consume-by time")
    contract = load_contract(root)
    validate_study_commit(root, study_commit, contract)
    runtime = load_runtime_modules(root)
    runtime["contract"] = contract
    evidence = load_release_evidence_module(root)
    try:
        evidence_contract = evidence.load_contract(root)
        verification = evidence.verify_published_finalizer(
            source_root=root, finalizer_job_id=finalizer_job_id,
            study_commit=study_commit,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "runtime admission could not authenticate coordinator H1"
        ) from error
    require(isinstance(evidence_contract, Mapping), "runtime evidence contract missing")
    trusted_state = Path(str(evidence_contract.get("queue", {}).get("state_dir", ""))).resolve()
    require(state == trusted_state, "runtime admission state directory is not trusted PVC")
    require(
        job_root == state / "jobs" / job_id,
        "runtime admission job directory changed",
    )
    require(isinstance(verification, Mapping), "runtime H1 verifier returned no result")
    plan, queue_fragment, receipt, verification_document = _validate_published_wave_artifacts(
        source_root=root, study_commit=study_commit, contract=contract,
        runtime=runtime, evidence=evidence, verification=verification,
    )
    now = _current_utc()
    verified_at = parse_utc(
        verification_document.get("verified_at_utc"), "runtime H1 verification time"
    )
    verified_consume_by = parse_utc(
        verification_document.get("consume_by_utc"), "runtime H1 consume-by time"
    )
    require(
        verified_at <= now <= supplied_consume_by == verified_consume_by
        and plan.get("consume_by_utc") == consume_by_utc
        and receipt.get("consume_by_utc") == consume_by_utc
        and plan.get("release_finalizer_job_id") == finalizer_job_id
        and receipt.get("release_finalizer_job_id") == finalizer_job_id,
        "runtime release authority is expired or detached from finalizer",
    )
    raw_jobs = queue_fragment.get("jobs")
    require(isinstance(raw_jobs, list) and raw_jobs, "runtime release wave has no jobs")
    matches = [row for row in raw_jobs if isinstance(row, Mapping) and row.get("job_id") == job_id]
    require(len(matches) == 1, "runtime job is absent or duplicated in published fragment")
    raw_job = dict(matches[0])
    normalized_job = runtime["queue"].normalize_job(raw_job)
    require(
        normalized_job.get("role") == expected_role
        and normalized_job.get("source_commit") == study_commit
        and normalized_job.get("released") is True
        and normalized_job.get("publish_log_tail_bytes") == 0,
        "runtime published descriptor identity changed",
    )
    argv = normalized_job.get("argv")
    require(isinstance(argv, list), "runtime published descriptor argv missing")
    for option, expected in (
        ("--confirmation-release-finalizer-job-id", finalizer_job_id),
        ("--confirmation-release-consume-by-utc", consume_by_utc),
    ):
        require(
            argv.count(option) == 1 and argv.index(option) + 1 < len(argv)
            and argv[argv.index(option) + 1] == expected,
            f"runtime published descriptor authority changed: {option}",
        )
    try:
        triplet = evidence.queue_triplet(
            state_dir=state, job_id=job_id, source_root=root,
            queue=runtime["queue"], allow_running_self=True,
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "runtime descriptor/claim failed authenticated control replay"
        ) from error
    require(
        isinstance(triplet, Mapping)
        and triplet.get("descriptor_value") == normalized_job
        and triplet.get("result") is None,
        "runtime descriptor/claim differs from published running job",
    )
    claim = triplet.get("claim_value")
    require(isinstance(claim, Mapping), "runtime immutable queue claim missing")
    worker_id = safe_id(claim.get("worker_id"), "runtime claimed worker")

    control_path = state / "control.json"
    control_identity, control = load_json_with_identity(
        control_path, "runtime queue control",
    )
    exact_keys(
        control,
        {
            "namespace", "control_commit", "control_generation",
            "admission_deadline_unix", "shutdown", "active_job_ids",
        },
        "runtime queue control",
    )
    expected_job_ids = [row.get("job_id") for row in raw_jobs]
    deadline = control.get("admission_deadline_unix")
    maximum_wall = max(int(row.get("max_wall_seconds", 0)) for row in raw_jobs)
    grace = evidence_contract.get("queue", {}).get("publication_grace_seconds")
    require(
        control.get("namespace") == NAMESPACE
        and control.get("control_commit") == claim.get("control_commit")
        and type(control.get("control_generation")) is int
        and control["control_generation"] >= claim.get("control_generation", sys.maxsize)
        and control.get("shutdown") is False
        and control.get("active_job_ids") == expected_job_ids
        and not isinstance(deadline, bool) and isinstance(deadline, (int, float))
        and math.isfinite(float(deadline))
        and type(grace) is int and grace >= 0
        and float(deadline) - now.timestamp() >= maximum_wall + grace,
        "runtime live control is not the exact releasable wave",
    )

    artifacts = verification.get("artifacts")
    require(isinstance(artifacts, Mapping), "runtime H1 artifacts missing")
    reconciliation = artifacts.get("reconciliation")
    pending = artifacts.get("publication_pending")
    require(
        isinstance(reconciliation, Mapping) and isinstance(pending, Mapping),
        "runtime H1 capacity artifacts missing",
    )
    gpu_state = reconciliation.get("task_gpu_state")
    require(isinstance(gpu_state, Mapping), "runtime H1 GPU state missing")
    exact_keys(
        gpu_state,
        {
            "inventory_complete", "fixed_deployment_worker_count",
            "fixed_deployment_worker_ids",
            "deployment_inventory_is_scientific_sample_size",
            "selected_attestation_worker_ids", "workers", "lane",
            "authorized_post_exit_capacity_worker_ids", "finalizer_uses_no_gpu",
            "post_exit_capacity_requires_terminal_reaped_outer_result",
        },
        "runtime H1 GPU state",
    )
    if expected_role == contract["runtime"]["N3"]["role"]:
        expected_lane = "N3"
        expected_gpu_count = contract["runtime"]["N3"]["gpu_count_per_worker"]
    elif expected_role == contract["runtime"]["D1"]["server_role"]:
        expected_lane = "D1"
        expected_gpu_count = contract["runtime"]["D1"]["server_gpu_count"]
    else:
        require(
            expected_role in contract["runtime"]["D1"]["simulator_roles"],
            "runtime role is outside released confirmation topology",
        )
        expected_lane = "D1"
        expected_gpu_count = contract["runtime"]["D1"]["simulator_gpu_count"]
    capacity_ids = gpu_state.get("authorized_post_exit_capacity_worker_ids")
    selected_ids = gpu_state.get("selected_attestation_worker_ids")
    workers = gpu_state.get("workers")
    topology = evidence.deployment_topology(root)
    require(
        gpu_state.get("inventory_complete") is True
        and gpu_state.get("fixed_deployment_worker_count") == len(topology) == 32
        and gpu_state.get("fixed_deployment_worker_ids") == sorted(topology)
        and gpu_state.get("deployment_inventory_is_scientific_sample_size") is False
        and gpu_state.get("lane") == expected_lane
        and isinstance(selected_ids, list)
        and selected_ids == sorted(set(selected_ids))
        and isinstance(capacity_ids, list)
        and capacity_ids == sorted(set(capacity_ids))
        and set(capacity_ids).issubset(selected_ids)
        and worker_id in capacity_ids
        and isinstance(workers, list)
        and selected_ids == sorted(
            row.get("worker_id") for row in workers if isinstance(row, Mapping)
        ),
        "runtime claimed worker is outside published capacity",
    )
    worker_matches = [
        dict(row) for row in workers
        if isinstance(row, Mapping) and row.get("worker_id") == worker_id
    ]
    require(len(worker_matches) == 1, "runtime published worker identity is missing or duplicated")
    worker = worker_matches[0]
    require(
        worker.get("role") == expected_role
        and worker.get("gpu_count") == expected_gpu_count,
        "runtime claimed worker role/topology changed",
    )
    require(
        worker_id in topology and topology[worker_id].get("role") == expected_role
        and topology[worker_id].get("gpu_count") == worker.get("gpu_count"),
        "runtime worker is outside authenticated deployment topology",
    )
    attestation_descriptor = worker.get("attestation")
    require(isinstance(attestation_descriptor, Mapping), "runtime worker attestation descriptor missing")
    attestation_identity, attestation = verify_json_input(
        Path(str(attestation_descriptor.get("path", ""))),
        str(attestation_descriptor.get("sha256", "")),
        "runtime worker attestation",
    )
    require(
        attestation_identity == dict(attestation_descriptor),
        "runtime worker attestation byte identity changed",
    )
    finalizer_worker_id = pending.get("producer_queue_job", {}).get("worker_id")
    if worker_id == finalizer_worker_id:
        require(
            attestation == artifacts.get("worker_attestation"),
            "runtime finalizer-worker attestation differs from H1",
        )
    else:
        attestation_job_id = safe_id(
            attestation.get("queue_job_id"), "runtime worker attestation job"
        )
        try:
            attestation_triplet = evidence.queue_triplet(
                state_dir=state, job_id=attestation_job_id, source_root=root,
                queue=runtime["queue"], allow_running_self=False,
            )
            validated_worker = evidence.validate_worker_attestation(
                attestation, descriptor_identity=attestation_identity,
                source_root=root, state_dir=state, study_commit=study_commit,
                queue_triplet_row=attestation_triplet,
            )
        except BaseException as error:
            raise ConfirmationReleaseWaveError(
                "runtime selected-worker attestation failed native replay"
            ) from error
        require(validated_worker == worker, "runtime selected-worker summary changed")

    hostname = socket.gethostname()
    require(
        isinstance(hostname, str) and bool(hostname)
        and os.environ.get("HOSTNAME") == hostname
        and worker.get("hostname") == hostname,
        "runtime hostname differs from published worker",
    )
    try:
        pod_uid, pod_uid_source = evidence._pod_uid(os.environ, Path("/proc"))
        current_gpus, compute_processes = evidence.gpu_inventory()
    except BaseException as error:
        raise ConfirmationReleaseWaveError("runtime pod/GPU identity unavailable") from error
    attested_gpus = attestation.get("gpu_inventory", {}).get("gpus")
    require(
        pod_uid == worker.get("pod_uid") == attestation.get("pod_uid")
        and _gpu_identity(current_gpus, "current")
        == _gpu_identity(attested_gpus, "attested")
        and len(current_gpus) == worker.get("gpu_count")
        and compute_processes == [],
        "runtime pod/GPU identity or idle boundary changed",
    )
    require(
        file_identity(control_path) == control_identity
        and file_identity(job_root / "descriptor.json") == triplet.get("descriptor")
        and file_identity(job_root / "claim" / "owner.json") == triplet.get("claim"),
        "runtime control, descriptor, or claim changed during admission",
    )
    return signed_document({
        "schema_version": RUNTIME_ADMISSION_SCHEMA,
        "status": "authenticated_before_runtime_start",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "checked_at_utc": now.isoformat().replace("+00:00", "Z"),
        "consume_by_utc": consume_by_utc,
        "stage": "queue_start",
        "release_finalizer_job_id": finalizer_job_id,
        "publication_verification_payload_sha256": verification_document["payload_sha256"],
        "pending_publication_commit": verification_document["pending_publication_commit"],
        "verified_remote_head": verification_document["verified_remote_head"],
        "published_queue_fragment": verification["artifact_descriptors"]["queue_fragment"],
        "published_queue_fragment_sha256": sha256_bytes(pretty_bytes(queue_fragment)),
        "queue_job_ids": expected_job_ids,
        "job_id": job_id,
        "worker_id": worker_id,
        "role": expected_role,
        "queue_descriptor": dict(triplet["descriptor"]),
        "queue_claim": dict(triplet["claim"]),
        "claim_control": dict(triplet["claim_control"]),
        "live_control": control_identity,
        "hostname": hostname,
        "pod_uid": pod_uid,
        "pod_uid_source": pod_uid_source,
        "gpu_identity": _gpu_identity(current_gpus, "current"),
        "idle_gpu_required": True,
        "gpu_compute_processes_observed": bool(compute_processes),
        "science_reset_request_action_started": False,
        "queue_mutated": False,
        "jobs_dispatched": 0,
        "claim_boundary": RUNTIME_ADMISSION_CLAIM_BOUNDARY,
    })


def validate_runtime_admission_receipt(
    *, receipt_path: Path, receipt_sha256: str, source_root: Path,
    study_commit: str, expected_job_id: str, expected_role: str,
    expected_finalizer_job_id: str, expected_consume_by_utc: str,
    verify_local_worker: bool,
) -> dict[str, Any]:
    """Validate an immutable start-time admission receipt before science.

    ``verify_local_worker`` is true only for the current queue job.  A paired
    D1 process validates its peer receipt with it false, then compares the two
    receipts' independently H1-bound publication fields.
    """

    identity, value = verify_json_input(
        receipt_path, receipt_sha256, "runtime admission receipt",
    )
    exact_keys(
        value,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "checked_at_utc", "consume_by_utc", "stage",
            "release_finalizer_job_id", "publication_verification_payload_sha256",
            "pending_publication_commit", "verified_remote_head",
            "published_queue_fragment", "published_queue_fragment_sha256",
            "queue_job_ids", "job_id", "worker_id", "role", "queue_descriptor",
            "queue_claim", "claim_control", "live_control", "hostname", "pod_uid",
            "pod_uid_source", "gpu_identity", "idle_gpu_required",
            "gpu_compute_processes_observed", "science_reset_request_action_started",
            "queue_mutated", "jobs_dispatched", "claim_boundary", "payload_sha256",
        },
        "runtime admission receipt",
    )
    verify_signed_document(value, "runtime admission receipt")
    checked = parse_utc(value.get("checked_at_utc"), "runtime admission receipt time")
    consume_by = parse_utc(
        value.get("consume_by_utc"), "runtime admission receipt consume-by time"
    )
    now = _current_utc()
    require(
        value.get("schema_version") == RUNTIME_ADMISSION_SCHEMA
        and value.get("status") == "authenticated_before_runtime_start"
        and value.get("study_id") == STUDY_ID and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit
        and value.get("stage") == "queue_start"
        and checked <= now and checked <= consume_by
        and value.get("consume_by_utc") == expected_consume_by_utc
        and value.get("release_finalizer_job_id") == expected_finalizer_job_id
        and value.get("job_id") == expected_job_id
        and value.get("role") == expected_role
        and value.get("idle_gpu_required") is True
        and value.get("gpu_compute_processes_observed") is False
        and value.get("science_reset_request_action_started") is False
        and value.get("queue_mutated") is False
        and value.get("jobs_dispatched") == 0
        and value.get("claim_boundary") == RUNTIME_ADMISSION_CLAIM_BOUNDARY,
        "runtime admission receipt identity changed",
    )
    safe_id(value.get("worker_id"), "runtime receipt worker")
    require(
        isinstance(value.get("queue_job_ids"), list)
        and value["queue_job_ids"] == list(dict.fromkeys(value["queue_job_ids"]))
        and expected_job_id in value["queue_job_ids"]
        and all(safe_id(item, "runtime receipt queue job") for item in value["queue_job_ids"]),
        "runtime admission queue inventory changed",
    )
    for name in (
        "publication_verification_payload_sha256", "published_queue_fragment_sha256",
    ):
        require(
            isinstance(value.get(name), str) and SHA256_RE.fullmatch(value[name]) is not None,
            f"runtime admission hash invalid: {name}",
        )
    for name in ("pending_publication_commit", "verified_remote_head"):
        require(
            isinstance(value.get(name), str) and COMMIT_RE.fullmatch(value[name]) is not None,
            f"runtime admission commit invalid: {name}",
        )
    published_fragment = value.get("published_queue_fragment")
    require(
        isinstance(published_fragment, Mapping)
        and set(published_fragment) == {"commit", "git_path", "bytes", "sha256"}
        and published_fragment.get("commit") == value.get("pending_publication_commit")
        and isinstance(published_fragment.get("git_path"), str)
        and not Path(published_fragment["git_path"]).is_absolute()
        and ".." not in Path(published_fragment["git_path"]).parts
        and type(published_fragment.get("bytes")) is int
        and published_fragment["bytes"] >= 0
        and isinstance(published_fragment.get("sha256"), str)
        and SHA256_RE.fullmatch(published_fragment["sha256"]) is not None
        and published_fragment.get("sha256")
        == value.get("published_queue_fragment_sha256"),
        "runtime admission published queue descriptor invalid",
    )
    for name in ("queue_descriptor", "queue_claim", "live_control"):
        row = value.get(name)
        require(
            isinstance(row, Mapping) and set(row) == {"path", "bytes", "sha256"}
            and isinstance(row.get("path"), str) and Path(row["path"]).is_absolute()
            and type(row.get("bytes")) is int and row["bytes"] >= 0
            and isinstance(row.get("sha256"), str)
            and SHA256_RE.fullmatch(row["sha256"]) is not None,
            f"runtime admission descriptor invalid: {name}",
        )
    descriptor_identity, descriptor = verify_descriptor(
        value["queue_descriptor"], "runtime admission queue descriptor"
    )
    claim_identity, claim = verify_descriptor(
        value["queue_claim"], "runtime admission queue claim"
    )
    descriptor_path = Path(descriptor_identity["path"])
    job_root = descriptor_path.parent
    state_dir = job_root.parents[1]
    require(
        descriptor_path == job_root / "descriptor.json"
        and job_root == state_dir / "jobs" / expected_job_id
        and Path(claim_identity["path"]) == job_root / "claim" / "owner.json"
        and Path(str(value["live_control"]["path"])) == state_dir / "control.json",
        "runtime admission immutable queue paths changed",
    )
    root = Path(source_root).resolve()
    contract = load_contract(root)
    validate_study_commit(root, study_commit, contract)
    runtime = load_runtime_modules(root)
    evidence = load_release_evidence_module(root)
    exact_keys(
        descriptor,
        {
            "schema_version", "namespace", "job_id", "released",
            "source_commit", "role", "argv", "max_wall_seconds",
            "publish_log_tail_bytes",
        },
        "runtime admission normalized queue descriptor",
    )
    raw_descriptor = {
        name: descriptor[name]
        for name in (
            "job_id", "released", "source_commit", "role", "argv",
            "max_wall_seconds", "publish_log_tail_bytes",
        )
    }
    require(
        runtime["queue"].normalize_job(raw_descriptor) == descriptor
        and descriptor.get("schema_version") == QUEUE_JOB_SCHEMA
        and descriptor.get("namespace") == NAMESPACE
        and descriptor.get("job_id") == expected_job_id
        and descriptor.get("released") is True
        and descriptor.get("source_commit") == study_commit
        and descriptor.get("role") == expected_role
        and descriptor.get("publish_log_tail_bytes") == 0,
        "runtime admission immutable queue descriptor changed",
    )
    descriptor_argv = descriptor.get("argv")
    require(isinstance(descriptor_argv, list), "runtime admission queue argv missing")
    for option, expected in (
        ("--confirmation-release-finalizer-job-id", expected_finalizer_job_id),
        ("--confirmation-release-consume-by-utc", expected_consume_by_utc),
    ):
        require(
            descriptor_argv.count(option) == 1
            and descriptor_argv.index(option) + 1 < len(descriptor_argv)
            and descriptor_argv[descriptor_argv.index(option) + 1] == expected,
            f"runtime admission immutable queue argv changed: {option}",
        )
    try:
        evidence.validate_queue_claim(
            claim, descriptor=descriptor, result=None, queue=runtime["queue"]
        )
    except BaseException as error:
        raise ConfirmationReleaseWaveError(
            "runtime admission immutable queue claim changed"
        ) from error
    require(
        claim.get("worker_id") == value.get("worker_id"),
        "runtime admission claim worker changed",
    )
    claim_control = value.get("claim_control")
    require(
        isinstance(claim_control, Mapping)
        and set(claim_control) == {
            "job_id", "control_commit", "control_generation",
            "control_queue_blob_sha256", "descriptor_sha256",
        }
        and claim_control.get("job_id") == expected_job_id,
        "runtime admission claim-control changed",
    )
    require(
        isinstance(claim_control.get("control_commit"), str)
        and COMMIT_RE.fullmatch(claim_control["control_commit"]) is not None
        and type(claim_control.get("control_generation")) is int
        and claim_control["control_generation"] >= contract["minimum_control_generation"]
        and isinstance(claim_control.get("control_queue_blob_sha256"), str)
        and SHA256_RE.fullmatch(claim_control["control_queue_blob_sha256"]) is not None
        and isinstance(claim_control.get("descriptor_sha256"), str)
        and SHA256_RE.fullmatch(claim_control["descriptor_sha256"]) is not None
        and claim_control.get("control_commit") == claim.get("control_commit")
        and claim_control.get("control_generation") == claim.get("control_generation")
        and claim_control.get("descriptor_sha256") == claim.get("descriptor_sha256")
        == descriptor_identity["sha256"],
        "runtime admission claim-control identity changed",
    )
    gpu_identity = value.get("gpu_identity")
    require(isinstance(gpu_identity, list), "runtime admission GPU identity missing")
    expected_gpu = _gpu_identity(gpu_identity, "receipt")
    if expected_role == contract["runtime"]["N3"]["role"]:
        receipt_gpu_count = contract["runtime"]["N3"]["gpu_count_per_worker"]
    elif expected_role == contract["runtime"]["D1"]["server_role"]:
        receipt_gpu_count = contract["runtime"]["D1"]["server_gpu_count"]
    else:
        require(
            expected_role in contract["runtime"]["D1"]["simulator_roles"],
            "runtime admission receipt role is outside released topology",
        )
        receipt_gpu_count = contract["runtime"]["D1"]["simulator_gpu_count"]
    require(
        len(expected_gpu) == receipt_gpu_count,
        "runtime admission receipt GPU count changed",
    )
    if verify_local_worker:
        hostname = socket.gethostname()
        try:
            pod_uid, _pod_source = evidence._pod_uid(os.environ, Path("/proc"))
            current_gpus, _compute = evidence.gpu_inventory()
        except BaseException as error:
            raise ConfirmationReleaseWaveError(
                "runtime admission current pod/GPU identity unavailable"
            ) from error
        require(
            hostname == value.get("hostname") and os.environ.get("HOSTNAME") == hostname
            and pod_uid == value.get("pod_uid")
            and _gpu_identity(current_gpus, "current") == expected_gpu,
            "runtime admission current worker identity changed",
        )
    return value


# Compatibility name for source callers.  It is deliberately no more
# permissive than the pending builder: no pre-H1 API emits an authorized wave.
build_release_wave = build_pending_release_wave


def _write_regular_at(
    parent_fd: int, name: str, data: bytes, label: str,
) -> os.stat_result:
    require(
        name not in {"", ".", ".."} and "/" not in name and "\x00" not in name,
        f"{label} filename is invalid",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o644, dir_fd=parent_fd)
    except OSError as error:
        raise ConfirmationReleaseWaveError(f"{label} cannot be created safely") from error
    opened = os.fstat(descriptor)
    try:
        require(
            stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1,
            f"{label} is not a singly linked regular file",
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        at_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        require(
            stat.S_ISREG(at_path.st_mode)
            and (at_path.st_dev, at_path.st_ino) == (opened.st_dev, opened.st_ino)
            and at_path.st_nlink == 1 and at_path.st_size == len(data),
            f"{label} pathname changed during creation",
        )
        return at_path
    except BaseException:
        try:
            at_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (at_path.st_dev, at_path.st_ino) == (opened.st_dev, opened.st_ino):
                os.unlink(name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_output_directory(
    path: Path, label: str,
) -> tuple[
    Path, int, int, tuple[tuple[str, int, int, int], ...],
    tuple[tuple[str, int, int, int], ...],
]:
    output = _lexical_absolute_path(path, label)
    require(output.name not in {"", ".", ".."}, f"{label} path is invalid")
    parent_fd, parent, parent_snapshot, _parent_metadata = _open_directory_fd(
        output.parent, f"{label} parent",
    )
    entry: os.stat_result | None = None
    output_fd = -1
    try:
        try:
            entry = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ConfirmationReleaseWaveError(
                f"{label} must be a pre-existing directory"
            ) from error
        require(stat.S_ISDIR(entry.st_mode), f"{label} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        output_fd = os.open(output.name, flags, dir_fd=parent_fd)
        opened = os.fstat(output_fd)
        require(
            stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (entry.st_dev, entry.st_ino),
            f"{label} directory identity changed",
        )
        _require_component_snapshot(parent, parent_snapshot, f"{label} parent", exact=True)
        _candidate, output_snapshot = _path_component_snapshot(
            output, label, allow_missing=False,
        )
        require(
            output_snapshot
            and (output_snapshot[-1][1], output_snapshot[-1][2])
            == (opened.st_dev, opened.st_ino),
            f"{label} path changed during open",
        )
        return (
            output, output_fd, parent_fd, parent_snapshot, output_snapshot,
        )
    except BaseException:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(parent_fd)
        raise


def _optional_file_identity(path: Path) -> dict[str, Any] | None:
    candidate, snapshot = _path_component_snapshot(
        path, f"optional file {path}", allow_missing=True,
    )
    if len(snapshot) < len(candidate.parts):
        return None
    return file_identity(candidate)


def _write_artifact_batch(
    *, output_dir: Path, artifacts: Mapping[str, Mapping[str, Any]],
    filenames: Mapping[str, str], active_queue: Path | None,
    active_change_reason: str,
) -> dict[str, dict[str, Any]]:
    require(set(artifacts) == set(filenames), "immutable artifact inventory changed")
    (
        output, output_fd, parent_fd, parent_snapshot, output_snapshot,
    ) = _open_output_directory(output_dir, "output directory")
    created_files: dict[str, os.stat_result] = {}
    preexisting: dict[str, os.stat_result] = {}
    data = {name: pretty_bytes(value) for name, value in artifacts.items()}
    active_before = (
        None if active_queue is None else _optional_file_identity(active_queue)
    )
    success = False
    try:
        # Validate every collision before the first write through the same
        # directory descriptor used for creation and rollback.
        for name, filename in filenames.items():
            require(
                filename not in {"", ".", ".."} and "/" not in filename,
                f"immutable output filename is invalid: {filename}",
            )
            try:
                metadata = os.stat(filename, dir_fd=output_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            require(
                stat.S_ISREG(metadata.st_mode),
                f"immutable output path is invalid: {filename}",
            )
            payload, _digest, observed = _read_regular_at(
                output_fd, filename, f"immutable output {filename}",
                capture_payload=True,
            )
            require(
                payload == data[name], f"immutable output differs: {filename}",
            )
            preexisting[filename] = observed
        for name, filename in filenames.items():
            if filename in preexisting:
                continue
            created_files[filename] = _write_regular_at(
                output_fd, filename, data[name], f"immutable output {filename}",
            )
            _require_component_snapshot(
                output, output_snapshot, "output directory", exact=True,
            )
        _require_component_snapshot(
            output.parent, parent_snapshot, "output directory parent", exact=True,
        )
        _require_component_snapshot(
            output, output_snapshot, "output directory", exact=True,
        )
        active_after = (
            None if active_queue is None else _optional_file_identity(active_queue)
        )
        require(active_after == active_before, active_change_reason)
        result: dict[str, dict[str, Any]] = {}
        for name, filename in filenames.items():
            _payload, digest, observed = _read_regular_at(
                output_fd, filename, f"immutable output {filename}",
                capture_payload=False,
            )
            expected_metadata = preexisting.get(filename) or created_files.get(filename)
            require(
                expected_metadata is not None
                and _same_file_metadata(expected_metadata, observed),
                f"immutable output changed after write: {filename}",
            )
            result[name] = {
                "path": str(output / filename),
                "bytes": observed.st_size,
                "sha256": digest,
            }
        _require_component_snapshot(
            output, output_snapshot, "output directory", exact=True,
        )
        success = True
        return result
    finally:
        if not success:
            for filename, identity in reversed(tuple(created_files.items())):
                try:
                    at_path = os.stat(
                        filename, dir_fd=output_fd, follow_symlinks=False,
                    )
                    if (at_path.st_dev, at_path.st_ino) == (
                        identity.st_dev, identity.st_ino,
                    ):
                        os.unlink(filename, dir_fd=output_fd)
                except OSError:
                    pass
        os.close(output_fd)
        os.close(parent_fd)


def _immutable_write(path: Path, value: Mapping[str, Any]) -> None:
    target = _lexical_absolute_path(path, "immutable output")
    _write_artifact_batch(
        output_dir=target.parent, artifacts={"value": value},
        filenames={"value": target.name}, active_queue=None,
        active_change_reason="unexpected active queue change",
    )


def write_release_wave(
    *, output_dir: Path, source_root: Path, result: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Write into a pre-existing directory while preserving the active queue."""

    filenames = {
        "plan": "confirmation_release_plan.json",
        "queue_fragment": "confirmation_release_queue_fragment.json",
        "receipt": "confirmation_release_wave_receipt.json",
    }
    active_queue = (
        _lexical_absolute_path(source_root, "source root")
        / "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json"
    )
    output = _lexical_absolute_path(output_dir, "output directory")
    require(
        all(output / filename != active_queue for filename in filenames.values()),
        "active cluster queue is not an output",
    )
    return _write_artifact_batch(
        output_dir=output, artifacts={name: result[name] for name in filenames},
        filenames=filenames, active_queue=active_queue,
        active_change_reason="active cluster queue changed during source-only write",
    )


def write_authorized_release_wave(
    *, output_dir: Path, source_root: Path, result: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Persist verification in a pre-existing directory, never the active queue."""

    exact_keys(
        result,
        {"authorization", "plan", "queue_fragment", "receipt", "publication_verification"},
        "authorized release-wave result",
    )
    authorization = result.get("authorization")
    plan = result.get("plan")
    queue_fragment = result.get("queue_fragment")
    receipt = result.get("receipt")
    publication_verification = result.get("publication_verification")
    require(
        all(
            isinstance(value, Mapping)
            for value in (
                authorization, plan, queue_fragment, receipt,
                publication_verification,
            )
        ),
        "authorized release-wave artifacts are not JSON objects",
    )
    exact_keys(
        authorization,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "verified_at_utc", "consume_by_utc", "finalizer_job_id", "cohort_branch",
            "qualified_model_ids", "wave_model", "selected_layouts",
            "queue_job_ids", "confirmation_release", "execution_release_authorized",
            "publication_verification_payload_sha256", "pending_publication_commit",
            "verified_remote_head", "published_plan", "published_queue_fragment",
            "published_wave_receipt", "plan_sha256", "queue_fragment_sha256",
            "wave_receipt_sha256", "queue_mutated", "jobs_dispatched",
            "results_branch_written_by_verifier", "claim_boundary", "payload_sha256",
        },
        "release-wave authorization",
    )
    verify_signed_document(authorization, "release-wave authorization")
    verify_signed_document(plan, "authorized release plan")
    verify_signed_document(receipt, "authorized release receipt")
    verify_signed_document(publication_verification, "authorized publication verification")
    jobs = queue_fragment.get("jobs")
    require(
        authorization.get("schema_version") == AUTHORIZATION_SCHEMA
        and authorization.get("status")
        == "confirmation_release_authorized_after_coordinator_publication"
        and authorization.get("study_id") == STUDY_ID
        and authorization.get("namespace") == NAMESPACE
        and authorization.get("study_commit") == plan.get("study_commit")
        == receipt.get("study_commit") == publication_verification.get("study_commit")
        and authorization.get("verified_at_utc")
        == publication_verification.get("verified_at_utc")
        and authorization.get("consume_by_utc")
        == publication_verification.get("consume_by_utc")
        == plan.get("consume_by_utc") == receipt.get("consume_by_utc")
        and authorization.get("finalizer_job_id")
        == publication_verification.get("finalizer_job_id")
        and authorization.get("cohort_branch") == plan.get("cohort_branch")
        and authorization.get("qualified_model_ids") == plan.get("qualified_model_ids")
        and authorization.get("wave_model") == plan.get("wave_model")
        and authorization.get("selected_layouts") == plan.get("selected_layouts")
        and isinstance(jobs, list)
        and authorization.get("queue_job_ids")
        == [row.get("job_id") for row in jobs if isinstance(row, Mapping)]
        == receipt.get("queue_job_ids")
        and authorization.get("confirmation_release") is True
        and authorization.get("execution_release_authorized") is True
        and authorization.get("queue_mutated") is False
        and authorization.get("jobs_dispatched") == 0
        and authorization.get("results_branch_written_by_verifier") is False
        and authorization.get("publication_verification_payload_sha256")
        == publication_verification.get("payload_sha256")
        and authorization.get("pending_publication_commit")
        == publication_verification.get("pending_publication_commit")
        and authorization.get("verified_remote_head")
        == publication_verification.get("verified_remote_head")
        and authorization.get("plan_sha256") == sha256_bytes(pretty_bytes(plan))
        and authorization.get("queue_fragment_sha256")
        == sha256_bytes(pretty_bytes(queue_fragment))
        and authorization.get("wave_receipt_sha256")
        == sha256_bytes(pretty_bytes(receipt))
        and authorization.get("published_plan")
        == publication_verification.get("artifact_descriptors", {}).get("plan")
        and authorization.get("published_queue_fragment")
        == publication_verification.get("artifact_descriptors", {}).get("queue_fragment")
        and authorization.get("published_wave_receipt")
        == publication_verification.get("artifact_descriptors", {}).get("wave_receipt"),
        "release-wave authorization is detached or not executable",
    )
    filenames = {
        "authorization": "confirmation_release_authorization.json",
        "plan": "confirmation_release_plan.json",
        "queue_fragment": "confirmation_release_queue_fragment.json",
        "receipt": "confirmation_release_wave_receipt.json",
        "publication_verification": "confirmation_release_publication_verification.json",
    }
    active_queue = (
        _lexical_absolute_path(source_root, "source root")
        / "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json"
    )
    output = _lexical_absolute_path(output_dir, "output directory")
    require(
        all(output / filename != active_queue for filename in filenames.values()),
        "active cluster queue is not an output",
    )
    return _write_artifact_batch(
        output_dir=output, artifacts={name: result[name] for name in filenames},
        filenames=filenames, active_queue=active_queue,
        active_change_reason="active cluster queue changed during verified write",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    pending = subparsers.add_parser("build-pending")
    pending.add_argument("--source-root", type=Path, required=True)
    pending.add_argument("--state-dir", type=Path, required=True)
    pending.add_argument("--study-commit", required=True)
    pending.add_argument("--confirmation-freeze", type=Path, required=True)
    pending.add_argument("--confirmation-freeze-sha256", required=True)
    pending.add_argument("--fixture-freeze", type=Path, required=True)
    pending.add_argument("--fixture-freeze-sha256", required=True)
    pending.add_argument("--resource-qualification", type=Path, required=True)
    pending.add_argument("--resource-qualification-sha256", required=True)
    pending.add_argument("--terminal-runtime-identities", type=Path, required=True)
    pending.add_argument("--terminal-runtime-identities-sha256", required=True)
    pending.add_argument("--result-attempt-ledger", type=Path, required=True)
    pending.add_argument("--result-attempt-ledger-sha256", required=True)
    pending.add_argument("--cluster-reconciliation", type=Path, required=True)
    pending.add_argument("--cluster-reconciliation-sha256", required=True)
    pending.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-published")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--study-commit", required=True)
    verify.add_argument("--finalizer-job-id", required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "build-pending":
            result = build_pending_release_wave(
                source_root=args.source_root,
                state_dir=args.state_dir,
                study_commit=args.study_commit,
                confirmation_freeze=args.confirmation_freeze,
                confirmation_freeze_sha256=args.confirmation_freeze_sha256,
                fixture_freeze=args.fixture_freeze,
                fixture_freeze_sha256=args.fixture_freeze_sha256,
                resource_qualification=args.resource_qualification,
                resource_qualification_sha256=args.resource_qualification_sha256,
                terminal_runtime_identities=args.terminal_runtime_identities,
                terminal_runtime_identities_sha256=args.terminal_runtime_identities_sha256,
                result_attempt_ledger=args.result_attempt_ledger,
                result_attempt_ledger_sha256=args.result_attempt_ledger_sha256,
                cluster_reconciliation=args.cluster_reconciliation,
                cluster_reconciliation_sha256=args.cluster_reconciliation_sha256,
            )
            identities = write_release_wave(
                output_dir=args.output_dir, source_root=args.source_root, result=result
            )
            status = "built_pending_h1_not_dispatched"
        else:
            result = verify_published_release_wave(
                source_root=args.source_root,
                finalizer_job_id=args.finalizer_job_id,
                study_commit=args.study_commit,
            )
            identities = write_authorized_release_wave(
                output_dir=args.output_dir, source_root=args.source_root, result=result
            )
            status = "verified_release_fragment_not_staged"
    except ConfirmationReleaseWaveError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": status, "outputs": identities}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
