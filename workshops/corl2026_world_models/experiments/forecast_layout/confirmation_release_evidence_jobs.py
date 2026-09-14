#!/usr/bin/env python3
"""Produce and verify fail-closed confirmation release evidence.

This module deliberately separates facts a running queue child can know from
facts created only after that child exits.  A lane-specific GM finalizer writes
immutable *pending* evidence into its queue publish directory.  The existing
cluster coordinator subsequently records the child's actual result, copied
artifacts, and cluster status on the results branch.  ``verify-published-finalizer``
fetches that branch into a temporary bare repository and performs a read-only
round trip check; it never writes the results branch or the study worktree.

No command in this file mutates the active queue, runs a model, resets a
simulator, emits labels, or claims scientific results.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
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
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence


STUDY_ID = "WMF-ABLATION-001"
NAMESPACE = "wmf_ablation_001_20260912"
CONTRACT_FILENAME = "confirmation_release_evidence_contract.json"
CONTRACT_SCHEMA = "wmf-confirmation-release-evidence-contract-v1"
WORKER_ATTESTATION_SCHEMA = "wmf-confirmation-fixed-worker-attestation-v1"
LEDGER_SCHEMA = "wmf-confirmation-result-attempt-ledger-v1"
RECONCILIATION_SCHEMA = "wmf-confirmation-cluster-reconciliation-v1"
PENDING_SCHEMA = "wmf-confirmation-release-publication-pending-v1"
VERIFICATION_SCHEMA = "wmf-confirmation-release-publication-verification-v1"
EVIDENCE_RECEIPT_SCHEMA = "wmf-confirmation-release-evidence-receipt-v1"
INPUTS_SCHEMA = "wmf-confirmation-release-evidence-inputs-v1"
QUEUE_SCHEMA = "wmf-cluster-queue-v1"
RESULT_SCHEMA = "wmf-cluster-result-v1"
STATUS_SCHEMA = "wmf-cluster-status-v1"
TRUSTED_FETCH_URL = "https://github.com/adeeb10abbas/steerable.git"
TRUSTED_GIT_PATHS = ("/usr/local/bin/git", "/usr/bin/git")
MAX_EVIDENCE_AGE_SECONDS = 300
MAX_AUTHENTICATED_GRAPH_BYTES = 192 * 1024 * 1024
ISOLATED_GIT_CONFIG = b"[core]\n\trepositoryformatversion = 0\n\tbare = true\n"
FINALIZER_RE = re.compile(r"confirmation-release-finalizer-(n3|d1)-a([0-9]{3})\Z")
ATTESTATION_RE = re.compile(
    r"confirmation-release-worker-attest-(n3|d1)-a([0-9]{3})-([a-zA-Z0-9_.-]+)\Z"
)
SAFE_ID_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
POD_UID_RE = re.compile(
    r"(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32,64})\Z"
)
TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "interrupted"}
QUEUE_RESULT_KEYS = {
    "schema_version", "namespace", "job_id", "worker_id", "source_commit",
    "descriptor_sha256", "started_at", "argv", "job_dir", "status",
    "returncode", "error_type", "ended_at", "wall_seconds", "child_pid",
    "child_reaped", "stdout", "stderr",
}
DESCRIPTOR_KEYS = {
    "schema_version", "namespace", "job_id", "released", "source_commit",
    "role", "argv", "max_wall_seconds", "publish_log_tail_bytes",
}
CLAIM_KEYS = {
    "worker_id", "claimed_at", "claimed_unix", "worker_pid",
    "control_commit", "control_generation", "descriptor_sha256",
    "release_boundary",
}
CONTROL_KEYS = {
    "namespace", "control_commit", "control_generation",
    "admission_deadline_unix", "shutdown", "active_job_ids",
}
PUBLISHED_FILENAMES = (
    "result_attempt_ledger.json",
    "cluster_reconciliation.json",
    "confirmation_release_plan.json",
    "confirmation_release_queue_fragment.json",
    "confirmation_release_wave_receipt.json",
    "publication_pending.json",
    "release_evidence_receipt.json",
)
WORKER_ATTESTATION_KEYS = {
    "schema_version", "status", "study_id", "namespace", "study_commit",
    "observed_at_utc", "consume_by_utc", "worker_id", "role", "hostname", "pod_uid",
    "pod_uid_source", "queue_job_id", "queue_descriptor", "queue_claim",
    "claim_control", "semantic_control", "deployment", "process_inventory",
    "gpu_inventory", "worker_active_job_ids", "claim_boundary", "payload_sha256",
}
EVIDENCE_RECEIPT_KEYS = {
    "schema_version", "status", "study_id", "namespace", "study_commit",
    "created_at_utc", "consume_by_utc", "inputs", "artifacts", "outer_result_available",
    "cluster_status_available", "publication_commit_available", "queue_mutated",
    "results_branch_written_by_finalizer", "science_or_labels_emitted",
    "claim_boundary", "payload_sha256",
}


class ConfirmationReleaseEvidenceError(RuntimeError):
    """Stable fail-closed producer/verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfirmationReleaseEvidenceError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfirmationReleaseEvidenceError("value is not canonical finite JSON") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfirmationReleaseEvidenceError("value is not finite JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexical_absolute_path(path: Path, label: str) -> Path:
    """Return one normalized absolute path without resolving filesystem links."""

    supplied = Path(path)
    require(".." not in supplied.parts, f"{label} contains parent traversal")
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    return Path(os.path.normpath(os.fspath(supplied)))


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _close_descriptors(descriptors: Sequence[int]) -> None:
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _open_directory_chain(
    path: Path, label: str, *, allow_missing: bool = False,
) -> tuple[Path, list[int], tuple[tuple[str, int, int, int], ...]]:
    """Open every directory component no-follow and retain the fd chain."""

    candidate = _lexical_absolute_path(path, label)
    descriptors: list[int] = []
    records: list[tuple[str, int, int, int]] = []
    current = Path(candidate.anchor)
    try:
        descriptor = os.open(current, _directory_open_flags())
        descriptors.append(descriptor)
        metadata = os.fstat(descriptor)
        require(stat.S_ISDIR(metadata.st_mode), f"{label} root is not a directory")
        records.append((str(current), metadata.st_dev, metadata.st_ino,
                        stat.S_IFMT(metadata.st_mode)))
        for part in candidate.parts[1:]:
            try:
                descriptor = os.open(
                    part, _directory_open_flags(), dir_fd=descriptors[-1],
                )
            except FileNotFoundError as error:
                if allow_missing:
                    _close_descriptors(descriptors)
                    return candidate, [], ()
                raise ConfirmationReleaseEvidenceError(
                    f"{label} component is missing: {current / part}"
                ) from error
            except OSError as error:
                raise ConfirmationReleaseEvidenceError(
                    f"{label} component cannot be opened no-follow: {current / part}"
                ) from error
            descriptors.append(descriptor)
            current = current / part
            metadata = os.fstat(descriptor)
            require(
                stat.S_ISDIR(metadata.st_mode),
                f"{label} component is not a directory: {current}",
            )
            records.append((str(current), metadata.st_dev, metadata.st_ino,
                            stat.S_IFMT(metadata.st_mode)))
    except BaseException:
        _close_descriptors(descriptors)
        raise
    return candidate, descriptors, tuple(records)


def _verify_directory_chain(
    path: Path, expected: tuple[tuple[str, int, int, int], ...], label: str,
) -> None:
    _candidate, descriptors, observed = _open_directory_chain(path, label)
    try:
        require(observed == expected, f"{label} component identity changed during operation")
    finally:
        _close_descriptors(descriptors)


def _entry_metadata(
    parent_descriptor: int, name: str, label: str,
) -> os.stat_result:
    require(name not in {"", ".", ".."} and "/" not in name, f"{label} name is unsafe")
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise ConfirmationReleaseEvidenceError(f"{label} is unreadable") from error
    require(not stat.S_ISLNK(metadata.st_mode), f"{label} is a symlink")
    return metadata


def _fresh_path_metadata(path: Path, label: str) -> os.stat_result:
    candidate = _lexical_absolute_path(path, label)
    _parent, descriptors, _records = _open_directory_chain(candidate.parent, f"{label} parent")
    try:
        return _entry_metadata(descriptors[-1], candidate.name, label)
    finally:
        _close_descriptors(descriptors)


def _optional_path_metadata(path: Path, label: str) -> os.stat_result | None:
    candidate = _lexical_absolute_path(path, label)
    _parent, descriptors, records = _open_directory_chain(
        candidate.parent, f"{label} parent", allow_missing=True,
    )
    if not descriptors:
        return None
    try:
        try:
            metadata = os.stat(
                candidate.name, dir_fd=descriptors[-1], follow_symlinks=False,
            )
        except FileNotFoundError:
            metadata = None
        except OSError as error:
            raise ConfirmationReleaseEvidenceError(f"{label} is unreadable") from error
        if metadata is not None:
            require(not stat.S_ISLNK(metadata.st_mode), f"{label} is a symlink")
        _verify_directory_chain(candidate.parent, records, f"{label} parent")
        return metadata
    finally:
        _close_descriptors(descriptors)


def _directory_state(path: Path, label: str) -> str:
    metadata = _optional_path_metadata(path, label)
    if metadata is None:
        return "absent"
    require(stat.S_ISDIR(metadata.st_mode), f"{label} is not a directory")
    return "present"


def _path_component_snapshot(
    path: Path, label: str, *, allow_missing: bool = True,
) -> tuple[Path, tuple[tuple[str, int, int, int, int], ...]]:
    """Capture every existing component identity without resolving links."""

    candidate = _lexical_absolute_path(path, label)
    current = Path(candidate.anchor)
    records: list[tuple[str, int, int, int, int]] = []
    try:
        root_metadata = current.lstat()
    except OSError as error:
        raise ConfirmationReleaseEvidenceError(
            f"{label} root component is unreadable: {current}"
        ) from error
    require(not stat.S_ISLNK(root_metadata.st_mode), f"{label} symlink component rejected: {current}")
    records.append((str(current), root_metadata.st_dev, root_metadata.st_ino,
                    root_metadata.st_mode, root_metadata.st_nlink))
    parts = candidate.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            require(allow_missing, f"{label} is missing: {current}")
            break
        except OSError as error:
            raise ConfirmationReleaseEvidenceError(
                f"{label} component is unreadable: {current}"
            ) from error
        require(not stat.S_ISLNK(metadata.st_mode), f"{label} symlink component rejected: {current}")
        if index < len(parts) - 1:
            require(
                stat.S_ISDIR(metadata.st_mode),
                f"{label} ancestor is not a directory: {current}",
            )
        records.append((str(current), metadata.st_dev, metadata.st_ino,
                        metadata.st_mode, metadata.st_nlink))
    return candidate, tuple(records)


def _validate_no_symlink_components(
    path: Path, label: str, *, allow_missing: bool = True,
) -> Path:
    """Reject a symlink or non-directory ancestor before any path-based I/O."""

    candidate, _snapshot = _path_component_snapshot(
        path, label, allow_missing=allow_missing,
    )
    return candidate


def _require_component_snapshot(
    path: Path, prior: tuple[tuple[str, int, int, int, int], ...],
    label: str, *, exact: bool, allow_missing: bool = False,
) -> None:
    _candidate, current = _path_component_snapshot(
        path, label, allow_missing=allow_missing,
    )
    require(
        current == prior if exact else current[:len(prior)] == prior,
        f"{label} component identity changed during operation",
    )


def _confined_path(
    path: Path, *, anchor: Path, label: str, allow_missing: bool = True,
) -> Path:
    """Validate symlink-free lexical containment under one trusted root."""

    trusted = _lexical_absolute_path(anchor, f"{label} anchor")
    candidate = _lexical_absolute_path(path, label)
    try:
        candidate.relative_to(trusted)
    except ValueError as error:
        raise ConfirmationReleaseEvidenceError(
            f"{label} escapes its trusted root: {candidate}"
        ) from error
    trusted = _validate_no_symlink_components(
        trusted, f"{label} anchor", allow_missing=allow_missing,
    )
    candidate = _validate_no_symlink_components(
        candidate, label, allow_missing=allow_missing,
    )
    return candidate


def _existing_directory(path: Path, label: str) -> Path:
    candidate, descriptors, records = _open_directory_chain(path, label)
    try:
        require(records, f"{label} directory chain is empty")
        _verify_directory_chain(candidate, records, label)
    finally:
        _close_descriptors(descriptors)
    return candidate


def _ensure_confined_directory(path: Path, *, anchor: Path, label: str) -> Path:
    """Create a contained directory by no-follow fd-relative traversal."""

    trusted = _existing_directory(anchor, f"{label} anchor")
    candidate = _lexical_absolute_path(path, label)
    try:
        relative = candidate.relative_to(trusted)
    except ValueError as error:
        raise ConfirmationReleaseEvidenceError(
            f"{label} escapes its trusted root: {candidate}"
        ) from error
    _anchor, anchor_descriptors, anchor_records = _open_directory_chain(
        trusted, f"{label} anchor",
    )
    traversal_descriptors = list(anchor_descriptors)
    current_descriptor = traversal_descriptors[-1]
    current_path = trusted
    created: list[tuple[int, str, int, int]] = []
    target_records = list(anchor_records)
    try:
        for part in relative.parts:
            require(part not in {"", ".", ".."}, f"{label} component is unsafe")
            try:
                next_descriptor = os.open(
                    part, _directory_open_flags(), dir_fd=current_descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o755, dir_fd=current_descriptor)
                except OSError as error:
                    raise ConfirmationReleaseEvidenceError(
                        f"cannot create {label}: {current_path / part}"
                    ) from error
                created_metadata = _entry_metadata(
                    current_descriptor, part, f"created {label} component",
                )
                require(
                    stat.S_ISDIR(created_metadata.st_mode),
                    f"created {label} component is not a directory",
                )
                created.append((current_descriptor, part,
                                created_metadata.st_dev, created_metadata.st_ino))
                try:
                    next_descriptor = os.open(
                        part, _directory_open_flags(), dir_fd=current_descriptor,
                    )
                except OSError as error:
                    raise ConfirmationReleaseEvidenceError(
                        f"created {label} component cannot be opened safely: {current_path / part}"
                    ) from error
            except OSError as error:
                raise ConfirmationReleaseEvidenceError(
                    f"{label} component cannot be opened no-follow: {current_path / part}"
                ) from error
            metadata = os.fstat(next_descriptor)
            require(stat.S_ISDIR(metadata.st_mode), f"{label} component is not a directory")
            traversal_descriptors.append(next_descriptor)
            current_descriptor = next_descriptor
            current_path = current_path / part
            target_records.append((str(current_path), metadata.st_dev, metadata.st_ino,
                                   stat.S_IFMT(metadata.st_mode)))
        _verify_directory_chain(trusted, anchor_records, f"{label} anchor")
        _verify_directory_chain(candidate, tuple(target_records), label)
    except BaseException:
        for parent_descriptor, name, device, inode in reversed(created):
            try:
                metadata = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False,
                )
                if (
                    stat.S_ISDIR(metadata.st_mode)
                    and (metadata.st_dev, metadata.st_ino) == (device, inode)
                ):
                    os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        _close_descriptors(traversal_descriptors)
    return candidate


def _read_regular_entry(
    *, parent_descriptor: int, parent_records: tuple[tuple[str, int, int, int], ...],
    path: Path, label: str, capture_payload: bool,
) -> tuple[bytes | None, str, os.stat_result, Path]:
    candidate = _lexical_absolute_path(path, label)
    require(candidate.name not in {"", ".", ".."}, f"{label} path is not a file")
    before = _entry_metadata(parent_descriptor, candidate.name, label)
    require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
        f"{label} is not a singly linked regular file",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ConfirmationReleaseEvidenceError(f"{label} is unreadable") from error
    try:
        opened = os.fstat(descriptor)
        require(
            stat.S_ISREG(opened.st_mode)
            and (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            f"{label} inode changed before open",
        )
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
        stable_fd_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        require(
            all(getattr(opened, field) == getattr(completed, field)
                for field in stable_fd_fields)
            and observed_bytes == completed.st_size,
            f"{label} changed while reading",
        )
        at_path = _entry_metadata(parent_descriptor, candidate.name, label)
        require(
            stat.S_ISREG(at_path.st_mode)
            and all(getattr(completed, field) == getattr(at_path, field)
                    for field in stable_fd_fields)
            and observed_bytes == at_path.st_size,
            f"{label} pathname changed while reading",
        )
        _verify_directory_chain(
            candidate.parent, parent_records, f"{label} parent",
        )
        fresh = _fresh_path_metadata(candidate, label)
        require(
            all(getattr(completed, field) == getattr(fresh, field)
                for field in stable_fd_fields),
            f"{label} canonical pathname changed while reading",
        )
        payload = b"".join(payload_parts) if capture_payload else None
        require(
            payload is None or len(payload) == completed.st_size,
            f"{label} payload length changed while reading",
        )
        return payload, digest.hexdigest(), opened, candidate
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular_file(
    path: Path, label: str, *, capture_payload: bool,
) -> tuple[bytes | None, str, os.stat_result, Path]:
    """Read one regular file through a retained no-follow parent fd chain."""

    candidate = _lexical_absolute_path(path, label)
    require(candidate.name not in {"", ".", ".."}, f"{label} path is not a file")
    _parent, parent_descriptors, parent_records = _open_directory_chain(
        candidate.parent, f"{label} parent",
    )
    try:
        return _read_regular_entry(
            parent_descriptor=parent_descriptors[-1], parent_records=parent_records,
            path=candidate, label=label, capture_payload=capture_payload,
        )
    finally:
        _close_descriptors(parent_descriptors)


def sha256_file(path: Path) -> str:
    _payload, digest, _metadata, _candidate = _read_regular_file(
        Path(path), f"file {path}", capture_payload=False,
    )
    return digest


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} fields changed")


def safe_id(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SAFE_ID_RE.fullmatch(value) is not None
        and value not in {".", ".."},
        f"{label} is not a safe ID",
    )
    return value


def parse_utc(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} is missing")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        result = datetime.fromisoformat(text)
    except ValueError as error:
        raise ConfirmationReleaseEvidenceError(f"{label} is not ISO-8601") from error
    require(result.tzinfo is not None, f"{label} is not timezone-aware")
    require(result.utcoffset() == timedelta(0), f"{label} is not UTC")
    return result.astimezone(timezone.utc)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_after(value: str, seconds: int) -> str:
    require(type(seconds) is int and seconds > 0, "UTC expiry interval invalid")
    return (parse_utc(value, "UTC expiry origin") + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def signed_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document already contains payload_sha256")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_signed_document(value: Mapping[str, Any], label: str) -> None:
    digest = value.get("payload_sha256")
    require(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
        f"{label} payload hash is invalid",
    )
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    require(sha256_bytes(canonical_bytes(unsigned)) == digest, f"{label} payload hash mismatch")


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ConfirmationReleaseEvidenceError(
                    f"{label} contains duplicate JSON key: {key}"
                )
            result[key] = item
        return result

    def reject_nonfinite_constant(constant: str) -> Any:
        raise ConfirmationReleaseEvidenceError(
            f"{label} contains non-finite JSON constant: {constant}"
        )

    def parse_finite_float(number: str) -> float:
        value = float(number)
        if not math.isfinite(value):
            raise ConfirmationReleaseEvidenceError(
                f"{label} contains non-finite JSON number: {number}"
            )
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
            parse_float=parse_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmationReleaseEvidenceError(f"{label} is unreadable") from error
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def load_json_with_identity(
    path: Path, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, _digest, _metadata, _candidate = _read_regular_file(
        Path(path), label, capture_payload=True,
    )
    assert payload is not None
    value = _strict_json_object(payload, label)
    identity = {
        "path": str(_candidate),
        "bytes": _metadata.st_size,
        "sha256": _digest,
    }
    return identity, value


def load_json(path: Path, label: str) -> dict[str, Any]:
    _identity, value = load_json_with_identity(path, label)
    return value


def file_identity(path: Path) -> dict[str, Any]:
    _payload, digest, metadata, candidate = _read_regular_file(
        Path(path), f"file {path}", capture_payload=False,
    )
    return {
        "path": str(candidate),
        "bytes": metadata.st_size,
        "sha256": digest,
    }


def verify_descriptor(value: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    exact_keys(value, {"path", "bytes", "sha256"}, f"{label} descriptor")
    identity, document = load_json_with_identity(
        Path(str(value.get("path", ""))), label,
    )
    require(identity == dict(value), f"{label} descriptor changed")
    return identity, document


def immutable_write(
    path: Path, value: Mapping[str, Any], *, anchor: Path,
) -> dict[str, Any]:
    trusted = _existing_directory(Path(anchor), "immutable output anchor")
    target = _lexical_absolute_path(Path(path), "immutable output")
    try:
        target.relative_to(trusted)
    except ValueError as error:
        raise ConfirmationReleaseEvidenceError(
            f"immutable output escapes its trusted root: {target}"
        ) from error
    require(target.name not in {"", ".", ".."}, "immutable output path is invalid")
    _ensure_confined_directory(
        target.parent, anchor=trusted, label="immutable output parent",
    )
    _parent, parent_descriptors, parent_records = _open_directory_chain(
        target.parent, "immutable output parent",
    )
    parent_descriptor = parent_descriptors[-1]
    data = pretty_bytes(value)
    try:
        before = os.stat(
            target.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError:
        before = None
    except OSError as error:
        _close_descriptors(parent_descriptors)
        raise ConfirmationReleaseEvidenceError(
            f"immutable output is unreadable: {target}"
        ) from error
    if before is not None:
        _close_descriptors(parent_descriptors)
        require(
            stat.S_ISREG(before.st_mode),
            f"immutable output is not a regular file: {target}",
        )
        observed, observed_digest, observed_metadata, observed_candidate = _read_regular_file(
            target, "immutable output", capture_payload=True,
        )
        require(observed == data, f"immutable output differs: {target}")
        return {
            "path": str(observed_candidate),
            "bytes": observed_metadata.st_size,
            "sha256": observed_digest,
        }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    opened: os.stat_result | None = None
    try:
        descriptor = os.open(
            target.name, flags, 0o644, dir_fd=parent_descriptor,
        )
    except OSError as error:
        _verify_directory_chain(
            target.parent, parent_records, "immutable output parent",
        )
        _close_descriptors(parent_descriptors)
        raise ConfirmationReleaseEvidenceError(
            f"immutable output cannot be created safely: {target}"
        ) from error
    opened = os.fstat(descriptor)
    if not (stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1):
        os.close(descriptor)
        descriptor = -1
        _close_descriptors(parent_descriptors)
        raise ConfirmationReleaseEvidenceError(
            f"immutable output is not a singly linked regular file: {target}"
        )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            require(written > 0, "immutable output write made no progress")
            offset += written
        os.fsync(descriptor)
        completed = os.fstat(descriptor)
        require(
            stat.S_ISREG(completed.st_mode)
            and (completed.st_dev, completed.st_ino) == (opened.st_dev, opened.st_ino)
            and completed.st_size == len(data),
            f"immutable output changed during creation: {target}",
        )
        at_path = _entry_metadata(parent_descriptor, target.name, "immutable output")
        require(
            stat.S_ISREG(at_path.st_mode)
            and (at_path.st_dev, at_path.st_ino) == (completed.st_dev, completed.st_ino)
            and at_path.st_size == completed.st_size,
            f"immutable output pathname changed during creation: {target}",
        )
        _verify_directory_chain(
            target.parent, parent_records, "immutable output parent",
        )
        fresh = _fresh_path_metadata(target, "immutable output")
        require(
            (fresh.st_dev, fresh.st_ino, fresh.st_size)
            == (completed.st_dev, completed.st_ino, completed.st_size),
            f"immutable output canonical pathname changed during creation: {target}",
        )
    except BaseException:
        try:
            at_path = os.stat(
                target.name, dir_fd=parent_descriptor, follow_symlinks=False,
            )
            if (
                opened is not None
                and (at_path.st_dev, at_path.st_ino) == (opened.st_dev, opened.st_ino)
            ):
                os.unlink(target.name, dir_fd=parent_descriptor)
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _close_descriptors(parent_descriptors)
    return {
        "path": str(target),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def _load_path_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    prior_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        raise ConfirmationReleaseEvidenceError(
            f"failed to import {path.name}: {type(error).__name__}"
        ) from error
    finally:
        sys.dont_write_bytecode = prior_bytecode_setting
    return module


def load_contract(source_root: Path) -> dict[str, Any]:
    root = _existing_directory(Path(source_root), "release-evidence source root")
    path = (
        root
        / "workshops/corl2026_world_models/experiments/forecast_layout"
        / CONTRACT_FILENAME
    )
    value = load_json(path, "release-evidence contract")
    exact_keys(
        value,
        {
            "schema_version", "study_id", "namespace", "status",
            "execution_readiness", "schemas", "transport_authentication", "queue", "producer",
            "worker_attestation", "lock_authentication", "path_authentication",
            "attempt_inventory", "pending_publication",
            "deployment_manifests", "source_paths", "security",
        },
        "release-evidence contract",
    )
    require(
        value.get("schema_version") == CONTRACT_SCHEMA
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("status") == "source_only_not_deployed",
        "release-evidence contract identity changed",
    )
    schemas = value.get("schemas")
    require(
        schemas == {
            "worker_attestation": WORKER_ATTESTATION_SCHEMA,
            "result_attempt_ledger": LEDGER_SCHEMA,
            "cluster_reconciliation": RECONCILIATION_SCHEMA,
            "publication_pending": PENDING_SCHEMA,
            "publication_verification": VERIFICATION_SCHEMA,
            "evidence_receipt": EVIDENCE_RECEIPT_SCHEMA,
        },
        "release-evidence schemas changed",
    )
    transport = value.get("transport_authentication")
    require(isinstance(transport, Mapping), "release-evidence transport policy missing")
    exact_keys(
        transport,
        {
            "git_executable_policy", "environment_policy", "remote_policy",
            "claim_source_ancestry_rule", "coordinator_credentials_rule",
        },
        "release-evidence transport policy",
    )
    require(
        all(isinstance(item, str) and bool(item) for item in transport.values())
        and "descriptor.source_commit" in transport["claim_source_ancestry_rule"]
        and "literal source-pinned HTTPS" in transport["remote_policy"]
        and "never asserts credential" in transport["coordinator_credentials_rule"],
        "release-evidence transport policy changed",
    )
    execution = value.get("execution_readiness")
    require(isinstance(execution, Mapping), "release-evidence execution boundary missing")
    exact_keys(
        execution,
        {"status", "queue_mutation", "results_branch_workstation_write", "claim_boundary"},
        "release-evidence execution boundary",
    )
    require(
        execution.get("status") == "requires_reviewed_consumer_pending_publication_support"
        and execution.get("queue_mutation") is False
        and execution.get("results_branch_workstation_write") is False
        and isinstance(execution.get("claim_boundary"), str)
        and bool(execution["claim_boundary"]),
        "release-evidence execution boundary changed",
    )
    queue_contract = value.get("queue")
    require(isinstance(queue_contract, Mapping), "release-evidence queue contract missing")
    exact_keys(
        queue_contract,
        {
            "state_dir", "queue_path", "control_ref", "control_branch",
            "results_ref", "results_branch", "allowed_results_urls",
            "trusted_fetch_url", "trusted_git_paths", "repository_identity",
            "minimum_control_generation",
            "forbidden_control_generation", "maximum_job_wall_seconds",
            "publication_grace_seconds", "maximum_attestation_age_seconds",
            "maximum_control_history_commits", "publish_log_tail_bytes",
        },
        "release-evidence queue contract",
    )
    producer_contract = value.get("producer")
    require(isinstance(producer_contract, Mapping), "release-evidence producer contract missing")
    exact_keys(
        producer_contract,
        {
            "finalizer_job_id_pattern", "attestation_job_id_prefix", "finalizer_roles",
            "attestation_job_id_pattern",
            "d1_required_simulator_roles", "publication_subdir",
            "minimum_finalizer_remaining_wall_seconds", "published_files",
        },
        "release-evidence producer contract",
    )
    require(
        producer_contract.get("finalizer_job_id_pattern")
        == "confirmation-release-finalizer-(n3|d1)-a[0-9]{3}"
        and producer_contract.get("attestation_job_id_pattern")
        == "confirmation-release-worker-attest-(n3|d1)-a[0-9]{3}-{fixed_worker_suffix}",
        "release-evidence producer job-ID scope changed",
    )
    worker_contract = value.get("worker_attestation")
    require(isinstance(worker_contract, Mapping), "worker-attestation contract missing")
    exact_keys(
        worker_contract,
        {
            "requires_kernel_or_downward_api_pod_uid",
            "requires_exact_parent_controller_argv",
            "requires_complete_visible_process_inventory",
            "requires_zero_gpu_compute_processes",
            "own_attestation_job_is_not_a_gpu_process",
            "requires_attestation_release_strictly_before_finalizer",
            "requires_exact_shared_admission_deadline",
            "requires_attempt_scoped_job_ids",
            "permits_authenticated_capacity_subset",
            "fixed_deployment_inventory_not_sample_size",
            "selection_rule",
            "attestation_id_rule",
        },
        "worker-attestation contract",
    )
    worker_boolean_keys = set(worker_contract) - {"selection_rule", "attestation_id_rule"}
    require(
        all(worker_contract[item] is True for item in worker_boolean_keys)
        and isinstance(worker_contract["selection_rule"], str)
        and "32-worker deployment inventory" in worker_contract["selection_rule"]
        and "selected subset" in worker_contract["selection_rule"]
        and isinstance(worker_contract["attestation_id_rule"], str)
        and "lane and finalizer attempt" in worker_contract["attestation_id_rule"],
        "worker-attestation contract weakened",
    )
    lock_contract = value.get("lock_authentication")
    require(isinstance(lock_contract, Mapping), "lock-authentication contract missing")
    exact_keys(
        lock_contract,
        {
            "canonical_release_lock_creation_only",
            "daemon_locks_must_preexist",
            "nofollow_regular_fd_validation",
        },
        "lock-authentication contract",
    )
    require(all(item is True for item in lock_contract.values()), "lock-authentication contract weakened")
    path_contract = value.get("path_authentication")
    require(isinstance(path_contract, Mapping), "path-authentication contract missing")
    exact_keys(
        path_contract,
        {
            "rejects_symlink_components_before_io",
            "rechecks_existing_component_inode_chain_after_io",
            "runtime_paths_confined_to_state_or_schedule_raw_root",
            "immutable_writes_confined_to_job_output_root",
            "failed_write_cleanup_requires_owned_inode",
            "fd_relative_recursive_inventory",
            "requires_singly_linked_regular_evidence",
            "same_fd_identity_hash_payload_and_parse",
            "strict_json_rejects_duplicate_and_nonfinite_values",
            "containment_precedes_candidate_probe",
        },
        "path-authentication contract",
    )
    require(all(item is True for item in path_contract.values()), "path-authentication contract weakened")
    attempt_contract = value.get("attempt_inventory")
    require(isinstance(attempt_contract, Mapping), "attempt-inventory contract missing")
    exact_keys(
        attempt_contract,
        {
            "models", "layout_order", "normal_form", "zero_launch_form",
            "native_prefix_requires_model_native_release_admission",
            "d1_native_prefix_requires_signed_pair_admission_ack",
            "zero_launch_rule",
        },
        "attempt-inventory contract",
    )
    require(
        attempt_contract["native_prefix_requires_model_native_release_admission"] is True
        and attempt_contract["d1_native_prefix_requires_signed_pair_admission_ack"] is True,
        "attempt-inventory release-admission contract weakened",
    )
    pending_contract = value.get("pending_publication")
    require(isinstance(pending_contract, Mapping), "pending-publication contract missing")
    exact_keys(
        pending_contract,
        {
            "producer_does_not_predict", "sole_active_job_rule", "generation_rule",
            "freshness_rule", "ancestry_rule",
        },
        "pending-publication contract",
    )
    source_paths = value.get("source_paths")
    require(isinstance(source_paths, Mapping), "release-evidence source-path contract missing")
    exact_keys(
        source_paths,
        {
            "producer", "consumer", "cluster_queue", "prepared_schedule",
            "n3_runtime", "d1_runtime",
        },
        "release-evidence source-path contract",
    )
    manifests = value.get("deployment_manifests")
    require(
        isinstance(manifests, list) and len(manifests) == 5
        and len(set(manifests)) == 5
        and all(isinstance(item, str) and item for item in manifests),
        "release-evidence deployment manifest contract changed",
    )
    require(
        queue_contract.get("publish_log_tail_bytes") == 0
        and queue_contract.get("trusted_fetch_url") == TRUSTED_FETCH_URL
        and TRUSTED_FETCH_URL in queue_contract.get("allowed_results_urls", [])
        and queue_contract.get("trusted_git_paths") == list(TRUSTED_GIT_PATHS)
        and queue_contract.get("repository_identity")
        == {"host": "github.com", "owner": "adeeb10abbas", "repository": "steerable"}
        and queue_contract.get("forbidden_control_generation") == 998
        and type(queue_contract.get("minimum_control_generation")) is int
        and queue_contract["minimum_control_generation"] > 998
        and type(queue_contract.get("maximum_control_history_commits")) is int
        and queue_contract["maximum_control_history_commits"] > 0
        and queue_contract.get("maximum_attestation_age_seconds") == MAX_EVIDENCE_AGE_SECONDS,
        "release-evidence queue authority changed",
    )
    security = value.get("security")
    require(isinstance(security, Mapping), "release-evidence security boundary missing")
    exact_keys(
        security,
        {
            "uses_kubernetes_api", "reads_or_logs_secrets", "publishes_log_tails",
            "mutates_queue", "mutates_study_worktree_during_remote_fetch",
            "workstation_pushes_results_branch", "science_or_labels_emitted",
        },
        "release-evidence security boundary",
    )
    require(
        all(security.get(key) is False for key in (
            "uses_kubernetes_api", "reads_or_logs_secrets", "publishes_log_tails",
            "mutates_queue", "mutates_study_worktree_during_remote_fetch",
            "workstation_pushes_results_branch", "science_or_labels_emitted",
        )),
        "release-evidence security boundary changed",
    )
    return value


def load_modules(source_root: Path) -> dict[str, Any]:
    root = _existing_directory(Path(source_root), "release-evidence source root")
    contract = load_contract(root)
    paths = contract["source_paths"]
    modules = {
        "queue": _load_path_module(root / paths["cluster_queue"], "wmf_release_evidence_queue"),
        "wave": _load_path_module(root / paths["consumer"], "wmf_release_evidence_wave"),
    }
    modules["contract"] = contract
    return modules


def _git(
    repo: Path, *args: str, accepted: tuple[int, ...] = (0,), timeout: int = 60,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    repository, repository_descriptors, repository_records = _open_directory_chain(
        Path(repo), "Git repository",
    )
    repository_descriptor = repository_descriptors[-1]
    environment = _safe_git_environment()
    try:
        try:
            result = subprocess.run(
                [
                    _trusted_git_executable(),
                    "-c", "credential.helper=",
                    "-c", "core.askPass=",
                    "-c", "core.sshCommand=/bin/false",
                    "-c", "protocol.ext.allow=never",
                    "-c", "protocol.file.allow=always",
                    "-C", f"/proc/self/fd/{repository_descriptor}",
                    *args,
                ],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=environment, timeout=timeout, check=False, text=text,
                pass_fds=(repository_descriptor,),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ConfirmationReleaseEvidenceError("Git operation failed") from error
        _verify_directory_chain(repository, repository_records, "Git repository")
    finally:
        _close_descriptors(repository_descriptors)
    require(result.returncode in accepted, f"Git {args[0] if args else 'operation'} failed")
    return result


def _trusted_git_executable() -> str:
    """Resolve one root-owned Git binary without consulting caller PATH."""

    allowed = {str(Path(item)) for item in TRUSTED_GIT_PATHS}
    for candidate_text in TRUSTED_GIT_PATHS:
        candidate = Path(candidate_text)
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ConfirmationReleaseEvidenceError("trusted Git executable is unreadable") from error
        current = candidate
        seen: set[str] = set()
        while True:
            current_text = str(current)
            require(current_text in allowed and current_text not in seen, "trusted Git symlink chain escaped or cycled")
            seen.add(current_text)
            try:
                info = current.lstat()
            except OSError as error:
                raise ConfirmationReleaseEvidenceError("trusted Git executable chain is unreadable") from error
            is_link = stat.S_ISLNK(info.st_mode)
            require(
                info.st_uid == 0 and (is_link or info.st_mode & 0o022 == 0),
                "trusted Git executable chain is not root-owned and protected",
            )
            for directory in [current.parent, *current.parent.parents]:
                directory_info = directory.stat()
                require(
                    directory_info.st_uid == 0
                    and stat.S_ISDIR(directory_info.st_mode)
                    and directory_info.st_mode & 0o022 == 0,
                    "trusted Git executable directory is not root-owned and non-writable",
                )
            if stat.S_ISREG(info.st_mode):
                require(os.access(current, os.X_OK), "trusted Git executable is not executable")
                return current_text
            require(is_link, "trusted Git path is not a regular file or symlink")
            try:
                target = os.readlink(current)
            except OSError as error:
                raise ConfirmationReleaseEvidenceError("trusted Git symlink is unreadable") from error
            current = Path(os.path.normpath(
                target if os.path.isabs(target) else str(current.parent / target)
            ))
    raise ConfirmationReleaseEvidenceError("no trusted Git executable is installed")


def _safe_git_environment() -> dict[str, str]:
    """Return a closed Git transport environment with no caller inheritance."""

    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent/wmf-release-evidence",
        "XDG_CONFIG_HOME": "/nonexistent/wmf-release-evidence",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GCM_INTERACTIVE": "never",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }


def _validate_source_files_at_commit(
    source_root: Path, study_commit: str, *, require_head: bool,
) -> None:
    root = _existing_directory(Path(source_root), "release-evidence source root")
    require(COMMIT_RE.fullmatch(study_commit) is not None, "study commit is invalid")
    head = _git(root, "rev-parse", "HEAD^{commit}", text=True).stdout.strip()
    if require_head:
        require(head == study_commit, "staged source HEAD differs from study commit")
        status = _git(
            root, "status", "--porcelain=v1", "--untracked-files=all", text=True
        ).stdout
        require(status == "", "staged source contains tracked or untracked residue")
    contract = load_contract(root)
    contract_relative = (
        "workshops/corl2026_world_models/experiments/forecast_layout/"
        + CONTRACT_FILENAME
    )
    for relative in [
        contract_relative,
        *contract["source_paths"].values(),
        *contract["deployment_manifests"],
    ]:
        path = _confined_path(
            root / relative, anchor=root, label=f"pinned source {relative}",
            allow_missing=False,
        )
        metadata = _fresh_path_metadata(path, f"pinned source {relative}")
        require(stat.S_ISREG(metadata.st_mode), f"pinned source missing: {relative}")
        blob = _git(root, "show", f"{study_commit}:{relative}").stdout
        require(sha256_bytes(blob) == sha256_file(path), f"study commit source differs: {relative}")


def validate_source_context(source_root: Path, study_commit: str) -> None:
    _validate_source_files_at_commit(source_root, study_commit, require_head=True)


def _descriptor_from_normalized(value: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(value, DESCRIPTOR_KEYS, "queue descriptor")
    require(
        value.get("schema_version") == "wmf-cluster-job-v1"
        and value.get("namespace") == NAMESPACE
        and value.get("released") is True,
        "queue descriptor identity changed",
    )
    return {
        key: value[key]
        for key in (
            "job_id", "released", "source_commit", "role", "argv",
            "max_wall_seconds", "publish_log_tail_bytes",
        )
    }


def _expanded_argv(
    descriptor: Mapping[str, Any], *, source_root: Path, job_dir: Path, state_dir: Path
) -> list[str]:
    source = _existing_directory(Path(source_root), "expanded argv source root")
    state = _existing_directory(Path(state_dir), "expanded argv state root")
    job = _confined_path(
        Path(job_dir), anchor=state / "jobs", label="expanded argv job directory",
        allow_missing=False,
    )
    return [
        item.replace("{source_root}", str(source))
        .replace("{job_dir}", str(job))
        .replace("{state_dir}", str(state))
        for item in descriptor["argv"]
    ]


def validate_queue_result(
    result: Mapping[str, Any], *, descriptor: Mapping[str, Any], queue: Any,
    state_dir: Path, job_dir: Path,
) -> None:
    exact_keys(result, QUEUE_RESULT_KEYS, "queue result")
    normalized = queue.normalize_job(_descriptor_from_normalized(descriptor))
    require(normalized == dict(descriptor), "queue descriptor normalization changed")
    require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("namespace") == NAMESPACE
        and result.get("job_id") == descriptor["job_id"]
        and result.get("source_commit") == descriptor["source_commit"]
        and result.get("descriptor_sha256") == sha256_bytes(queue.encode(normalized))
        and result.get("status") in TERMINAL_STATUSES
        and result.get("child_reaped") is True,
        "queue result is not terminal/reaped or detached from descriptor",
    )
    safe_id(result.get("worker_id"), "queue result worker")
    started = parse_utc(result.get("started_at"), "queue result started_at")
    ended = parse_utc(result.get("ended_at"), "queue result ended_at")
    wall = result.get("wall_seconds")
    require(
        ended >= started and not isinstance(wall, bool)
        and isinstance(wall, (int, float)) and math.isfinite(wall) and wall >= 0,
        "queue result timing is invalid",
    )
    if result["status"] == "succeeded":
        require(
            result.get("returncode") == 0 and result.get("error_type") is None,
            "successful queue result has contradictory exit evidence",
        )
    else:
        require(
            result.get("returncode") is None or type(result.get("returncode")) is int,
            "terminal queue failure return code is invalid",
        )
    state_root = _existing_directory(Path(state_dir), "queue state root")
    expected_source = _confined_path(
        state_root / "sources" / descriptor["source_commit"],
        anchor=state_root / "sources", label="staged queue source", allow_missing=False,
    )
    expected_job = _confined_path(
        Path(job_dir), anchor=state_root / "jobs", label="queue result job directory",
        allow_missing=False,
    )
    require(
        _confined_path(
            Path(str(result.get("job_dir", ""))), anchor=state_root / "jobs",
            label="queue result declared job directory", allow_missing=False,
        ) == expected_job
        and result.get("argv") == _expanded_argv(
            descriptor, source_root=expected_source,
            job_dir=expected_job, state_dir=state_root,
        ),
        "queue result argv/job_dir does not bind descriptor substitutions",
    )
    for stream in ("stdout", "stderr"):
        row = result.get(stream)
        require(
            isinstance(row, Mapping) and set(row) == {"bytes", "sha256"}
            and type(row.get("bytes")) is int and row["bytes"] >= 0
            and isinstance(row.get("sha256"), str)
            and SHA256_RE.fullmatch(row["sha256"]) is not None,
            f"queue result {stream} descriptor is invalid",
        )
        stream_path = job_dir / f"{stream}.log"
        identity = file_identity(stream_path)
        require(
            {"bytes": identity["bytes"], "sha256": identity["sha256"]} == dict(row),
            f"queue result {stream} changed after exit",
        )


def _queue_manifest_at_claim(
    *, source_root: Path, claim: Mapping[str, Any], descriptor: Mapping[str, Any],
    queue_path: str, queue: Any,
    control_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    commit = claim.get("control_commit")
    require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None, "claim commit invalid")
    if control_snapshot is None:
        with _fetch_authorized_control_history(source_root=source_root) as snapshot:
            return _queue_manifest_at_claim(
                source_root=source_root, claim=claim, descriptor=descriptor,
                queue_path=queue_path, queue=queue, control_snapshot=snapshot,
            )
    repo = Path(str(control_snapshot.get("repo", "")))
    head = str(control_snapshot.get("head", ""))
    _existing_directory(repo, "authorized control snapshot")
    require(COMMIT_RE.fullmatch(head) is not None, "authorized control snapshot invalid")
    ancestry = _git(repo, "merge-base", "--is-ancestor", commit, head, accepted=(0, 1, 128))
    require(ancestry.returncode == 0, "claim commit is outside authorized control history")
    source_commit = descriptor.get("source_commit")
    require(
        isinstance(source_commit, str) and COMMIT_RE.fullmatch(source_commit) is not None,
        "claim descriptor source commit invalid",
    )
    source_ancestry = _git(
        repo, "merge-base", "--is-ancestor", source_commit, commit,
        accepted=(0, 1, 128),
    )
    require(
        source_ancestry.returncode == 0,
        "descriptor source commit is outside its authorized claim-control history",
    )
    distance_text = _git(repo, "rev-list", "--count", f"{commit}..{head}", text=True).stdout.strip()
    require(distance_text.isdigit(), "authorized control history distance invalid")
    require(
        int(distance_text) <= load_contract(source_root)["queue"]["maximum_control_history_commits"],
        "claim commit is outside bounded authorized control history",
    )
    blob = _git(repo, "show", f"{commit}:{queue_path}").stdout
    manifest = _strict_json_object(blob, "claim control queue blob")
    require(
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == QUEUE_SCHEMA
        and manifest.get("namespace") == NAMESPACE
        and manifest.get("shutdown") is False
        and isinstance(manifest.get("jobs"), list),
        "claim control queue blob is invalid",
    )
    matches = [row for row in manifest["jobs"] if isinstance(row, Mapping) and row.get("job_id") == descriptor["job_id"]]
    require(len(matches) == 1 and matches[0].get("released") is True, "claim commit did not release job")
    normalized_raw = dict(matches[0])
    # Queue manifests carry raw descriptors; comparison to the normalized PVC
    # descriptor is the authorization boundary.
    normalized = queue.normalize_job(normalized_raw)
    require(normalized == dict(descriptor), "claim control queue descriptor differs from PVC")
    return {
        "job_id": descriptor["job_id"],
        "control_commit": commit,
        "control_generation": claim["control_generation"],
        "control_queue_blob_sha256": sha256_bytes(blob),
        "descriptor_sha256": sha256_bytes(queue.encode(normalized)),
    }


def validate_queue_claim(
    claim: Mapping[str, Any], *, descriptor: Mapping[str, Any], result: Mapping[str, Any] | None,
    queue: Any,
) -> None:
    exact_keys(claim, CLAIM_KEYS, "queue claim")
    claimed_at = parse_utc(claim.get("claimed_at"), "queue claim time")
    claimed_unix = claim.get("claimed_unix")
    require(
        safe_id(claim.get("worker_id"), "claim worker")
        and not isinstance(claimed_unix, bool)
        and isinstance(claimed_unix, (int, float)) and math.isfinite(claimed_unix)
        and abs(claimed_at.timestamp() - float(claimed_unix)) <= 5
        and type(claim.get("worker_pid")) is int and claim["worker_pid"] > 0
        and isinstance(claim.get("control_commit"), str)
        and COMMIT_RE.fullmatch(claim["control_commit"]) is not None
        and type(claim.get("control_generation")) is int
        and claim["control_generation"] >= 999 and claim["control_generation"] != 998
        and claim.get("descriptor_sha256") == sha256_bytes(queue.encode(descriptor))
        and claim.get("release_boundary") == "claim_committed_under_shared_release_lock",
        "queue claim does not authenticate its release boundary",
    )
    if result is not None:
        require(
            result.get("worker_id") == claim["worker_id"]
            and parse_utc(result.get("started_at"), "queue result start") >= claimed_at,
            "queue claim and result worker/time differ",
        )


def queue_triplet(
    *, state_dir: Path, job_id: str, source_root: Path, queue: Any,
    allow_running_self: bool = False,
    control_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safe_id(job_id, "queue job")
    state_root = _existing_directory(Path(state_dir), "queue state root")
    jobs_root = _existing_directory(state_root / "jobs", "queue jobs root")
    job_root = _confined_path(
        jobs_root / job_id, anchor=jobs_root, label=f"queue job directory {job_id}",
        allow_missing=False,
    )
    _existing_directory(job_root, f"queue job directory {job_id}")
    descriptor_identity, descriptor = load_json_with_identity(
        job_root / "descriptor.json", f"{job_id} descriptor",
    )
    normalized = queue.normalize_job(_descriptor_from_normalized(descriptor))
    require(normalized == descriptor and descriptor["job_id"] == job_id, f"{job_id} descriptor invalid")
    claim_identity, claim = load_json_with_identity(
        job_root / "claim" / "owner.json", f"{job_id} claim",
    )
    result_path = job_root / "result.json"
    result_identity: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    result_metadata = _optional_path_metadata(result_path, f"{job_id} result")
    if result_metadata is not None:
        require(stat.S_ISREG(result_metadata.st_mode), f"{job_id} result is not regular")
        result_identity, result = load_json_with_identity(
            result_path, f"{job_id} result",
        )
        validate_queue_result(
            result, descriptor=descriptor, queue=queue,
            state_dir=state_dir, job_dir=job_root,
        )
    else:
        require(allow_running_self, f"queue job lacks terminal result: {job_id}")
    validate_queue_claim(claim, descriptor=descriptor, result=result, queue=queue)
    history = _queue_manifest_at_claim(
        source_root=source_root, claim=claim, descriptor=descriptor,
        queue_path=load_contract(source_root)["queue"]["queue_path"], queue=queue,
        control_snapshot=control_snapshot,
    )
    return {
        "job_id": job_id,
        "descriptor": descriptor_identity,
        "descriptor_value": descriptor,
        "claim": claim_identity,
        "claim_value": claim,
        "result": result_identity,
        "result_value": result,
        "claim_control": history,
    }


def deployment_topology(source_root: Path) -> dict[str, dict[str, Any]]:
    """Return every pinned fixed worker and every admissible deployment version."""

    root = _existing_directory(Path(source_root), "release-evidence source root")
    contract = load_contract(root)
    observed: dict[str, dict[str, Any]] = {}
    for relative in contract["deployment_manifests"]:
        path = root / relative
        identity, value = load_json_with_identity(
            path, f"deployment manifest {relative}",
        )
        require(
            value.get("apiVersion") == "v1" and value.get("kind") == "List"
            and isinstance(value.get("items"), list),
            f"deployment manifest structure changed: {relative}",
        )
        for item_index, item in enumerate(value["items"]):
            if not isinstance(item, Mapping) or item.get("kind") != "Job":
                continue
            metadata = item.get("metadata")
            pod_spec = item.get("spec", {}).get("template", {}).get("spec", {})
            containers = pod_spec.get("containers")
            require(
                isinstance(metadata, Mapping) and isinstance(containers, list)
                and len(containers) == 1 and isinstance(containers[0], Mapping),
                f"worker deployment is not single-container: {relative}:{item_index}",
            )
            container = containers[0]
            args = container.get("args")
            command = container.get("command")
            require(
                isinstance(args, list) and all(isinstance(item, str) for item in args)
                and isinstance(command, list) and all(isinstance(item, str) for item in command),
                f"worker deployment argv missing: {relative}:{item_index}",
            )
            for option in ("--worker-id", "--role", "--admission-deadline-unix"):
                require(args.count(option) == 1, f"worker deployment option changed: {option}")
            worker_id = safe_id(args[args.index("--worker-id") + 1], "deployment worker")
            role = safe_id(args[args.index("--role") + 1], "deployment role")
            deadline_text = args[args.index("--admission-deadline-unix") + 1]
            try:
                deadline_unix = float(deadline_text)
            except ValueError as error:
                raise ConfirmationReleaseEvidenceError("deployment admission deadline invalid") from error
            require(math.isfinite(deadline_unix) and deadline_unix > 0, "deployment deadline invalid")
            requests = container.get("resources", {}).get("requests", {})
            limits = container.get("resources", {}).get("limits", {})
            gpu_count = limits.get("nvidia.com/gpu")
            require(
                type(gpu_count) is int and gpu_count > 0
                and requests.get("nvidia.com/gpu") == gpu_count,
                f"deployment GPU request/limit changed: {worker_id}",
            )
            env_rows = container.get("env", [])
            require(isinstance(env_rows, list), f"deployment env invalid: {worker_id}")
            pod_uid_downward = any(
                isinstance(row, Mapping) and row.get("name") == "POD_UID"
                and row.get("valueFrom", {}).get("fieldRef", {}).get("fieldPath") == "metadata.uid"
                for row in env_rows
            )
            record = {
                "worker_id": worker_id,
                "role": role,
                "gpu_count": gpu_count,
                "manifest": identity,
                "manifest_relative_path": relative,
                "item_index": item_index,
                "job_name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "container_name": container.get("name"),
                "image": container.get("image"),
                "command": list(command),
                "args": list(args),
                "admission_deadline_unix": deadline_unix,
                "pod_uid_downward_api": pod_uid_downward,
            }
            require(
                record["namespace"] == "211247-prod"
                and record["job_name"] == worker_id
                and isinstance(record["image"], str) and "@sha256:" in record["image"],
                f"deployment identity changed: {worker_id}",
            )
            if worker_id not in observed:
                observed[worker_id] = {
                    "worker_id": worker_id, "role": role, "gpu_count": gpu_count,
                    "records": [],
                }
            require(
                observed[worker_id]["role"] == role
                and observed[worker_id]["gpu_count"] == gpu_count,
                f"replacement deployment changed worker topology: {worker_id}",
            )
            observed[worker_id]["records"].append(record)
    require(len(observed) == 32, "fixed worker inventory is not exactly 32 workers")
    roles = [row["role"] for row in observed.values()]
    require(len(roles) == len(set(roles)), "fixed worker roles are not unique")
    require(roles.count("n3") == 1 and roles.count("d1") == 1, "model worker inventory changed")
    return observed


def _pod_uid(environ: Mapping[str, str], proc_root: Path) -> tuple[str, str]:
    supplied = environ.get("POD_UID", "")
    if POD_UID_RE.fullmatch(supplied):
        return supplied, "downward_api_metadata_uid"
    try:
        cgroup = (Path(proc_root) / "self" / "cgroup").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConfirmationReleaseEvidenceError("pod UID is unavailable") from error
    matches = re.findall(
        r"(?:^|/)pod([0-9a-f]{8}(?:[-_][0-9a-f]{4}){3}[-_][0-9a-f]{12})(?:/|$)",
        cgroup,
    )
    normalized = sorted({item.replace("_", "-") for item in matches})
    require(len(normalized) == 1 and POD_UID_RE.fullmatch(normalized[0]) is not None, "pod UID is ambiguous")
    return normalized[0], "proc_self_cgroup"


def _proc_cmdline(proc_root: Path, pid: int) -> list[str]:
    try:
        data = (Path(proc_root) / str(pid) / "cmdline").read_bytes()
    except OSError as error:
        raise ConfirmationReleaseEvidenceError(f"process cmdline unreadable: {pid}") from error
    return [item.decode("utf-8", errors="strict") for item in data.split(b"\0") if item]


def _proc_parent(proc_root: Path, pid: int) -> int:
    try:
        rows = (Path(proc_root) / str(pid) / "status").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ConfirmationReleaseEvidenceError(f"process status unreadable: {pid}") from error
    values = [row.split(":", 1)[1].strip() for row in rows if row.startswith("PPid:")]
    require(len(values) == 1 and values[0].isdigit(), f"process parent invalid: {pid}")
    return int(values[0])


def visible_process_inventory(proc_root: Path = Path("/proc")) -> tuple[list[dict[str, Any]], set[int]]:
    current = os.getpid()
    ancestry: set[int] = set()
    cursor = current
    while cursor > 0 and cursor not in ancestry:
        ancestry.add(cursor)
        parent = _proc_parent(proc_root, cursor)
        if parent == cursor:
            break
        cursor = parent
    rows: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            int(path.name) for path in Path(proc_root).iterdir() if path.name.isdigit()
        )
    except OSError as error:
        raise ConfirmationReleaseEvidenceError("process inventory unreadable") from error
    for pid in candidates:
        try:
            command = _proc_cmdline(proc_root, pid)
            parent = _proc_parent(proc_root, pid)
        except ConfirmationReleaseEvidenceError:
            # A process disappearing while scanned is ambiguous, not clean.
            raise
        rows.append({"pid": pid, "ppid": parent, "argv_sha256": sha256_bytes(canonical_bytes(command))})
    return rows, ancestry


def _run_checked(argv: Sequence[str], *, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ConfirmationReleaseEvidenceError("local evidence command failed") from error
    require(result.returncode == 0, "local evidence command failed")
    return result.stdout


def gpu_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = _run_checked([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",")]
        require(len(parts) == 6, "GPU inventory row changed")
        try:
            index, total, used, utilization = int(parts[0]), int(parts[3]), int(parts[4]), int(parts[5])
        except ValueError as error:
            raise ConfirmationReleaseEvidenceError("GPU inventory numeric field invalid") from error
        require(index >= 0 and total > 0 and 0 <= used <= total and 0 <= utilization <= 100, "GPU inventory value invalid")
        gpus.append({
            "index": index, "uuid": parts[1], "name": parts[2],
            "memory_total_mib": total, "memory_used_mib": used,
            "utilization_percent": utilization,
        })
    compute_output = _run_checked([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    compute: list[dict[str, Any]] = []
    for line in compute_output.splitlines():
        if not line.strip() or line.strip().lower().startswith("no running"):
            continue
        parts = [item.strip() for item in line.split(",")]
        require(len(parts) == 4 and parts[0].isdigit(), "GPU compute-process row invalid")
        compute.append({
            "pid": int(parts[0]), "gpu_uuid": parts[1], "process_name": parts[2],
            "used_memory_mib": None if parts[3] in {"N/A", "[N/A]"} else int(parts[3]),
        })
    return gpus, compute


def _expected_controller_argv(record: Mapping[str, Any]) -> list[str]:
    args = record["args"]
    require(
        len(args) >= 4 and args[0].endswith("/cluster_queue.py")
        and SHA256_RE.fullmatch(args[1]) is not None and args[2] == "worker",
        "deployment bootstrap argv changed",
    )
    return [args[0], *args[2:]]


def _select_runtime_deployment(
    records: Sequence[Mapping[str, Any]], *, pod_uid_source: str,
    parent_argv: Sequence[str],
) -> Mapping[str, Any]:
    matches = []
    for record in records:
        expected = _expected_controller_argv(record)
        # argv[0] is the Python executable and may be a resolved equivalent;
        # every controller argument after it is literal deployment authority.
        if list(parent_argv[1:]) == expected:
            if pod_uid_source == "downward_api_metadata_uid" and not record["pod_uid_downward_api"]:
                continue
            matches.append(record)
    require(len(matches) == 1, "runtime controller does not identify one pinned deployment")
    return matches[0]


def validate_running_queue_context(
    *, source_root: Path, state_dir: Path, job_dir: Path, job_id: str,
    study_commit: str, expected_role: str, runtime_argv: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    validate_source_context(source_root, study_commit)
    modules = load_modules(source_root)
    queue = modules["queue"]
    state_root = _existing_directory(Path(state_dir), "running queue state root")
    jobs_root = _existing_directory(state_root / "jobs", "running queue jobs root")
    expected_job = _confined_path(
        jobs_root / safe_id(job_id, "running job"), anchor=jobs_root,
        label="expected running job directory", allow_missing=False,
    )
    supplied_job = _confined_path(
        Path(job_dir), anchor=jobs_root, label="running job directory", allow_missing=False,
    )
    require(supplied_job == expected_job, "running job directory changed")
    triplet = queue_triplet(
        state_dir=state_dir, job_id=job_id, source_root=source_root,
        queue=queue, allow_running_self=True,
    )
    require(triplet["result"] is None, "running queue child already has an outer result")
    descriptor = triplet["descriptor_value"]
    require(
        descriptor["source_commit"] == study_commit
        and descriptor["role"] == expected_role
        and descriptor["publish_log_tail_bytes"] == 0,
        "running queue descriptor source/role/publication changed",
    )
    if runtime_argv is not None:
        expanded = _expanded_argv(
            descriptor, source_root=Path(source_root), job_dir=Path(job_dir),
            state_dir=Path(state_dir),
        )
        require(
            list(runtime_argv) == expanded[1:],
            "running process argv differs from expanded queue descriptor",
        )
    control_path = _confined_path(
        state_root / "control.json", anchor=state_root,
        label="live queue control", allow_missing=False,
    )
    control_identity, control = load_json_with_identity(
        control_path, "live queue control",
    )
    exact_keys(control, CONTROL_KEYS, "live queue control")
    claim = triplet["claim_value"]
    require(
        control.get("namespace") == NAMESPACE
        and control.get("control_commit") == claim["control_commit"]
        and type(control.get("control_generation")) is int
        and control["control_generation"] >= claim["control_generation"]
        and control.get("shutdown") is False
        and job_id in control.get("active_job_ids", []),
        "running queue claim is not admitted by live semantic control",
    )
    return triplet, control, control_identity, modules


def attest_worker(
    *, source_root: Path, state_dir: Path, job_dir: Path, job_id: str,
    study_commit: str, expected_worker_id: str, expected_finalizer_job_id: str,
    _now_utc: str | None = None, runtime_argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None, proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Attest only the current fixed worker from its own process namespace."""

    environ = dict(os.environ if environ is None else environ)
    topology = deployment_topology(source_root)
    require(expected_worker_id in topology, "expected worker is not in fixed deployment inventory")
    if ATTESTATION_RE.fullmatch(job_id) is not None:
        lane, attempt_number, scoped_worker_id = _attestation_identity(
            job_id, topology=topology,
        )
        require(
            scoped_worker_id == expected_worker_id
            and expected_finalizer_job_id
            == f"confirmation-release-finalizer-{lane.lower()}-a{attempt_number:03d}",
            "attestation job is not bound to its exact finalizer attempt",
        )
    else:
        finalizer_match = FINALIZER_RE.fullmatch(job_id)
        require(
            finalizer_match is not None
            and job_id == expected_finalizer_job_id
            and topology[expected_worker_id]["role"]
            == load_contract(source_root)["producer"]["finalizer_roles"][
                finalizer_match.group(1).upper()
            ],
            "running finalizer worker attestation identity changed",
        )
    expected = topology[expected_worker_id]
    triplet, control, _control_identity, _modules = validate_running_queue_context(
        source_root=source_root, state_dir=state_dir, job_dir=job_dir,
        job_id=job_id, study_commit=study_commit, expected_role=expected["role"],
        runtime_argv=runtime_argv,
    )
    claim = triplet["claim_value"]
    require(claim["worker_id"] == expected_worker_id, "attestation was claimed by another worker")
    hostname = socket.gethostname()
    require(
        hostname and environ.get("HOSTNAME") == hostname,
        "kernel hostname and runtime hostname differ",
    )
    pod_uid, pod_uid_source = _pod_uid(environ, proc_root)
    parent_pid = os.getppid()
    parent_argv = _proc_cmdline(proc_root, parent_pid)
    deployment = _select_runtime_deployment(
        expected["records"], pod_uid_source=pod_uid_source, parent_argv=parent_argv,
    )
    processes, ancestry = visible_process_inventory(proc_root)
    unknown = [row for row in processes if row["pid"] not in ancestry]
    require(unknown == [], "worker process namespace contains non-ancestry processes")
    gpus, compute = gpu_inventory()
    require(len(gpus) == expected["gpu_count"], "visible GPU count differs from deployment")
    require(compute == [], "worker has live GPU compute processes")
    observed_at = _now_utc or utc_now()
    observed = parse_utc(observed_at, "worker attestation time")
    deadline = control.get("admission_deadline_unix")
    require(
        not isinstance(deadline, bool) and isinstance(deadline, (int, float))
        and math.isfinite(deadline) and deadline > observed.timestamp(),
        "worker admission deadline is expired or invalid",
    )
    receipt = signed_document({
        "schema_version": WORKER_ATTESTATION_SCHEMA,
        "status": "authenticated_worker_gpu_idle_while_attesting",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "observed_at_utc": observed_at,
        "consume_by_utc": utc_after(observed_at, MAX_EVIDENCE_AGE_SECONDS),
        "worker_id": expected_worker_id,
        "role": expected["role"],
        "hostname": hostname,
        "pod_uid": pod_uid,
        "pod_uid_source": pod_uid_source,
        "queue_job_id": job_id,
        "queue_descriptor": triplet["descriptor"],
        "queue_claim": triplet["claim"],
        "claim_control": triplet["claim_control"],
        "semantic_control": {
            "namespace": control["namespace"],
            "control_commit": control["control_commit"],
            "observed_generation": control["control_generation"],
            "claim_generation": claim["control_generation"],
            "admission_deadline_unix": deadline,
            "shutdown": control["shutdown"],
            "active_job_ids": list(control["active_job_ids"]),
        },
        "deployment": {
            "manifest": deployment["manifest"],
            "manifest_relative_path": deployment["manifest_relative_path"],
            "item_index": deployment["item_index"],
            "job_name": deployment["job_name"],
            "container_name": deployment["container_name"],
            "image": deployment["image"],
            "gpu_count": deployment["gpu_count"],
            "controller_argv_sha256": sha256_bytes(canonical_bytes(parent_argv)),
            "controller_pid": parent_pid,
        },
        "process_inventory": {
            "inventory_complete": True,
            "visible": processes,
            "allowed_ancestry_pids": sorted(ancestry),
            "non_ancestry_processes": [],
            "current_pid": os.getpid(),
            "current_process_is_attestation_only": True,
        },
        "gpu_inventory": {
            "inventory_complete": True,
            "gpus": gpus,
            "compute_processes": [],
            "idle": True,
        },
        "worker_active_job_ids": [job_id],
        "claim_boundary": (
            "This worker attests only its own pod, controller ancestry, visible GPUs, "
            "and process namespace. It does not assert another worker's state."
        ),
    })
    job_root = _confined_path(
        Path(job_dir), anchor=_existing_directory(Path(state_dir), "queue state root") / "jobs",
        label="worker-attestation job directory", allow_missing=False,
    )
    output = job_root / "publish" / "worker_attestation.json"
    immutable_write(output, receipt, anchor=job_root)
    return receipt


def _producer_relative(source_root: Path) -> str:
    return str(load_contract(source_root)["source_paths"]["producer"])


def _raw_queue_job(
    *, job_id: str, role: str, source_commit: str, argv: Sequence[str],
    max_wall_seconds: int,
) -> dict[str, Any]:
    safe_id(job_id, "new queue job")
    safe_id(role, "new queue role")
    require(COMMIT_RE.fullmatch(source_commit) is not None, "new queue source commit invalid")
    require(
        isinstance(argv, Sequence) and not isinstance(argv, (str, bytes))
        and all(isinstance(item, str) and item for item in argv),
        "new queue argv invalid",
    )
    require(type(max_wall_seconds) is int and 0 < max_wall_seconds <= 172800, "new queue wall invalid")
    return {
        "job_id": job_id,
        "released": True,
        "source_commit": source_commit,
        "role": role,
        "argv": list(argv),
        "max_wall_seconds": max_wall_seconds,
        "publish_log_tail_bytes": 0,
    }


def build_attestation_queue_fragment(
    *, source_root: Path, study_commit: str, lane: str, attempt_number: int,
    worker_ids: Sequence[str], max_wall_seconds: int = 600,
) -> dict[str, Any]:
    """Build, but never stage, one selected attempt-scoped attestation wave."""

    topology = deployment_topology(source_root)
    require(
        lane in {"N3", "D1"}
        and type(attempt_number) is int and 1 <= attempt_number <= 999,
        "attestation finalizer scope is invalid",
    )
    require(
        isinstance(worker_ids, Sequence) and not isinstance(worker_ids, (str, bytes)),
        "selected attestation worker inventory is invalid",
    )
    selected = list(worker_ids)
    require(
        all(isinstance(worker_id, str) for worker_id in selected)
        and selected == sorted(set(selected))
        and 1 <= len(selected) <= len(topology)
        and all(worker_id in topology for worker_id in selected),
        "selected attestation worker inventory is invalid",
    )
    expected = _expected_attestation_jobs(
        source_root, study_commit, lane=lane, attempt_number=attempt_number,
        max_wall_seconds=max_wall_seconds,
    )
    jobs = [expected[worker_id] for worker_id in selected]
    return {
        "schema_version": QUEUE_SCHEMA,
        "namespace": NAMESPACE,
        "shutdown": False,
        "jobs": jobs,
    }


def build_finalizer_queue_fragment(
    *, source_root: Path, study_commit: str, lane: str, attempt_number: int,
    inputs_path: Path, inputs_sha256: str, max_wall_seconds: int = 900,
) -> dict[str, Any]:
    """Build, but never stage, the sole lane-specific pending finalizer."""

    contract = load_contract(source_root)
    require(lane in {"N3", "D1"}, "finalizer lane is invalid")
    require(type(attempt_number) is int and 1 <= attempt_number <= 999, "finalizer attempt invalid")
    require(SHA256_RE.fullmatch(inputs_sha256) is not None, "finalizer inputs hash invalid")
    require(Path(inputs_path).is_absolute(), "finalizer inputs path must be absolute")
    canonical_inputs_path = _validate_no_symlink_components(
        Path(inputs_path), "finalizer inputs path", allow_missing=True,
    )
    job_id = f"confirmation-release-finalizer-{lane.lower()}-a{attempt_number:03d}"
    producer = _producer_relative(source_root)
    job = _raw_queue_job(
        job_id=job_id, role=contract["producer"]["finalizer_roles"][lane],
        source_commit=study_commit, max_wall_seconds=max_wall_seconds,
        argv=[
            "/usr/bin/python3", "{source_root}/" + producer, "finalize-pending",
            "--source-root", "{source_root}",
            "--state-dir", "{state_dir}",
            "--job-dir", "{job_dir}",
            "--job-id", job_id,
            "--study-commit", study_commit,
            "--expected-lane", lane,
            "--inputs", str(canonical_inputs_path),
            "--inputs-sha256", inputs_sha256,
        ],
    )
    return {"schema_version": QUEUE_SCHEMA, "namespace": NAMESPACE, "shutdown": False, "jobs": [job]}


def _validate_finalizer_descriptor(
    descriptor: Mapping[str, Any], *, source_root: Path, study_commit: str, queue: Any,
) -> dict[str, Any]:
    """Require the literal descriptor emitted by this producer's builder."""

    normalized = queue.normalize_job(_descriptor_from_normalized(descriptor))
    require(normalized == dict(descriptor), "finalizer descriptor normalization changed")
    job_id = safe_id(descriptor.get("job_id"), "finalizer descriptor job")
    match = FINALIZER_RE.fullmatch(job_id)
    require(match is not None, "finalizer descriptor job ID invalid")
    lane = match.group(1).upper()
    attempt_number = int(match.group(2))
    argv = descriptor.get("argv")
    require(
        isinstance(argv, list) and len(argv) == 19
        and argv[:3] == [
            "/usr/bin/python3",
            "{source_root}/" + _producer_relative(source_root),
            "finalize-pending",
        ],
        "finalizer descriptor runner changed",
    )
    options = argv[3::2]
    values = argv[4::2]
    expected_options = [
        "--source-root", "--state-dir", "--job-dir", "--job-id",
        "--study-commit", "--expected-lane", "--inputs", "--inputs-sha256",
    ]
    require(options == expected_options and len(values) == len(expected_options), "finalizer descriptor options changed")
    parsed = dict(zip(options, values, strict=True))
    inputs_path = Path(parsed["--inputs"])
    canonical_inputs_path = _validate_no_symlink_components(
        inputs_path, "finalizer descriptor inputs path", allow_missing=True,
    )
    require(
        parsed["--source-root"] == "{source_root}"
        and parsed["--state-dir"] == "{state_dir}"
        and parsed["--job-dir"] == "{job_dir}"
        and parsed["--job-id"] == job_id
        and parsed["--study-commit"] == study_commit
        and parsed["--expected-lane"] == lane
        and inputs_path.is_absolute()
        and SHA256_RE.fullmatch(parsed["--inputs-sha256"]) is not None,
        "finalizer descriptor values changed",
    )
    expected_raw = build_finalizer_queue_fragment(
        source_root=source_root,
        study_commit=study_commit,
        lane=lane,
        attempt_number=attempt_number,
        inputs_path=inputs_path,
        inputs_sha256=parsed["--inputs-sha256"],
        max_wall_seconds=descriptor["max_wall_seconds"],
    )["jobs"][0]
    require(
        queue.normalize_job(expected_raw) == dict(descriptor),
        "finalizer descriptor is not the exact producer-built job",
    )
    return {
        "job_id": job_id,
        "lane": lane,
        "attempt_number": attempt_number,
        "inputs_path": str(canonical_inputs_path),
        "inputs_sha256": parsed["--inputs-sha256"],
    }


def validate_inputs(path: Path, expected_sha256: str, *, study_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require(SHA256_RE.fullmatch(expected_sha256) is not None, "inputs SHA-256 invalid")
    identity, value = load_json_with_identity(path, "release evidence inputs")
    require(identity["sha256"] == expected_sha256, "inputs SHA-256 mismatch")
    exact_keys(
        value,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "confirmation_freeze", "fixture_freeze", "resource_qualification",
            "terminal_runtime_identities", "worker_attestation_job_ids",
            "results_remote_alias", "results_ref", "claim_boundary", "payload_sha256",
        },
        "release evidence inputs",
    )
    verify_signed_document(value, "release evidence inputs")
    require(
        value.get("schema_version") == INPUTS_SCHEMA
        and value.get("status") == "candidate_paths_only_native_validation_required"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit,
        "release evidence inputs identity changed",
    )
    for name in (
        "confirmation_freeze", "fixture_freeze", "resource_qualification",
        "terminal_runtime_identities",
    ):
        verify_descriptor(value.get(name), f"inputs {name}")
    job_ids = value.get("worker_attestation_job_ids")
    require(
        isinstance(job_ids, list),
        "worker attestation job inventory changed",
    )
    _attestation_scope(job_ids)
    return identity, value


def _artifact_slot(path: Path, *, anchor: Path) -> dict[str, Any]:
    candidate = _confined_path(
        Path(path), anchor=anchor, label="runtime artifact slot", allow_missing=True,
    )
    metadata = _optional_path_metadata(candidate, "runtime artifact slot")
    if metadata is not None:
        require(
            stat.S_ISREG(metadata.st_mode),
            f"artifact slot is not a regular file: {candidate}",
        )
        return {
            "state": "present", "expected_path": str(candidate),
            "descriptor": file_identity(candidate),
        }
    return {"state": "absent", "expected_path": str(candidate), "descriptor": None}


def _regular_inventory(roots: Sequence[Path], *, anchor: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    trusted_anchor = _lexical_absolute_path(anchor, "runtime inventory anchor")

    def visit(
        directory_descriptor: int, directory_path: Path,
        directory_records: tuple[tuple[str, int, int, int], ...],
    ) -> None:
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as error:
            raise ConfirmationReleaseEvidenceError(
                f"runtime inventory directory is unreadable: {directory_path}"
            ) from error
        require(len(names) == len(set(names)), "runtime inventory names are ambiguous")
        for name in names:
            require(
                isinstance(name, str) and name not in {"", ".", ".."}
                and "/" not in name and "\x00" not in name,
                "runtime inventory entry name is unsafe",
            )
            path = directory_path / name
            metadata = _entry_metadata(
                directory_descriptor, name, "runtime inventory entry",
            )
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_descriptor = os.open(
                        name, _directory_open_flags(), dir_fd=directory_descriptor,
                    )
                except OSError as error:
                    raise ConfirmationReleaseEvidenceError(
                        f"runtime inventory directory cannot be opened safely: {path}"
                    ) from error
                try:
                    child_metadata = os.fstat(child_descriptor)
                    require(
                        stat.S_ISDIR(child_metadata.st_mode)
                        and (child_metadata.st_dev, child_metadata.st_ino)
                        == (metadata.st_dev, metadata.st_ino),
                        f"runtime inventory directory changed before open: {path}",
                    )
                    child_records = directory_records + ((
                        str(path), child_metadata.st_dev, child_metadata.st_ino,
                        stat.S_IFMT(child_metadata.st_mode),
                    ),)
                    visit(child_descriptor, path, child_records)
                    at_parent = _entry_metadata(
                        directory_descriptor, name, "runtime inventory directory",
                    )
                    require(
                        (at_parent.st_dev, at_parent.st_ino)
                        == (child_metadata.st_dev, child_metadata.st_ino),
                        f"runtime inventory directory changed during traversal: {path}",
                    )
                    _verify_directory_chain(
                        path, child_records, "runtime inventory directory",
                    )
                finally:
                    os.close(child_descriptor)
                continue
            require(
                stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                f"runtime inventory contains a nonregular or multiply linked entry: {path}",
            )
            _payload, digest, observed, candidate = _read_regular_entry(
                parent_descriptor=directory_descriptor,
                parent_records=directory_records,
                path=path, label="runtime inventory file", capture_payload=False,
            )
            if candidate not in seen:
                seen.add(candidate)
                files.append({
                    "path": str(candidate), "bytes": observed.st_size,
                    "sha256": digest,
                })

    for supplied in roots:
        root = _confined_path(
            Path(supplied), anchor=trusted_anchor,
            label="runtime inventory root", allow_missing=True,
        )
        root_metadata = _optional_path_metadata(root, "runtime inventory root")
        if root_metadata is None:
            continue
        require(
            stat.S_ISDIR(root_metadata.st_mode),
            f"runtime root is not a directory: {root}",
        )
        _root, root_descriptors, root_records = _open_directory_chain(
            root, "runtime inventory root",
        )
        try:
            visit(root_descriptors[-1], root, root_records)
            _verify_directory_chain(root, root_records, "runtime inventory root")
        finally:
            _close_descriptors(root_descriptors)
    return sorted(files, key=lambda row: row["path"])


def _cell_receipt_paths(
    attempt_root: Path, *, anchor: Path,
) -> tuple[list[Path], list[Path]]:
    root = _confined_path(
        Path(attempt_root), anchor=anchor, label="native attempt root", allow_missing=True,
    )
    root_metadata = _optional_path_metadata(root, "native attempt root")
    if root_metadata is None:
        return [], []
    require(stat.S_ISDIR(root_metadata.st_mode), "native attempt root is not a directory")
    cells = _confined_path(
        root / "cells", anchor=anchor, label="native cells parent", allow_missing=True,
    )
    cells_metadata = _optional_path_metadata(cells, "native cells parent")
    if cells_metadata is None:
        return [], []
    require(stat.S_ISDIR(cells_metadata.st_mode), "native cells parent is not a directory")
    _cells, cells_descriptors, cells_records = _open_directory_chain(
        cells, "native cells parent",
    )
    try:
        passed: list[Path] = []
        failed: list[Path] = []
        try:
            cell_names = sorted(os.listdir(cells_descriptors[-1]))
        except OSError as error:
            raise ConfirmationReleaseEvidenceError("native cells parent is unreadable") from error
        require(len(cell_names) == len(set(cell_names)), "native cell directory names are ambiguous")
        for cell_name in cell_names:
            require(
                cell_name not in {"", ".", ".."} and "/" not in cell_name
                and "\x00" not in cell_name,
                "native cell directory name is unsafe",
            )
            entry = _entry_metadata(
                cells_descriptors[-1], cell_name, "native cell directory",
            )
            require(stat.S_ISDIR(entry.st_mode), "native cells parent contains a non-directory")
            try:
                cell_descriptor = os.open(
                    cell_name, _directory_open_flags(), dir_fd=cells_descriptors[-1],
                )
            except OSError as error:
                raise ConfirmationReleaseEvidenceError(
                    "native cell directory cannot be opened safely"
                ) from error
            try:
                opened = os.fstat(cell_descriptor)
                require(
                    (opened.st_dev, opened.st_ino) == (entry.st_dev, entry.st_ino),
                    "native cell directory changed before open",
                )
                cell_path = cells / cell_name
                cell_records = cells_records + ((
                    str(cell_path), opened.st_dev, opened.st_ino,
                    stat.S_IFMT(opened.st_mode),
                ),)
                for receipt_name, destination in (
                    ("cell_receipt.json", passed),
                    ("technical_failure.json", failed),
                ):
                    try:
                        receipt_metadata = os.stat(
                            receipt_name, dir_fd=cell_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        continue
                    except OSError as error:
                        raise ConfirmationReleaseEvidenceError(
                            "native cell receipt is unreadable"
                        ) from error
                    require(
                        stat.S_ISREG(receipt_metadata.st_mode)
                        and receipt_metadata.st_nlink == 1,
                        "native cell receipt is not a singly linked regular file",
                    )
                    destination.append(cell_path / receipt_name)
                at_parent = _entry_metadata(
                    cells_descriptors[-1], cell_name, "native cell directory",
                )
                require(
                    (at_parent.st_dev, at_parent.st_ino)
                    == (opened.st_dev, opened.st_ino),
                    "native cell directory changed during enumeration",
                )
                _verify_directory_chain(
                    cell_path, cell_records, "native cell directory",
                )
            finally:
                os.close(cell_descriptor)
        passed.sort()
        failed.sort()
        require(
            all(stat.S_ISREG(_fresh_path_metadata(path, "native cell receipt").st_mode)
                for path in passed + failed),
            "native cell inventory contains a nonregular path",
        )
        _verify_directory_chain(cells, cells_records, "native cells parent")
    finally:
        _close_descriptors(cells_descriptors)
    return passed, failed


def _condition_index_from_path(path: Path) -> int:
    prefix = Path(path).parent.name.split("-", 1)[0]
    require(prefix.isdigit(), f"cell directory lacks condition prefix: {path}")
    value = int(prefix)
    require(0 <= value < 4, f"cell condition index is out of range: {path}")
    return value


def _slot_present(slots: Sequence[Mapping[str, Any]]) -> bool:
    return any(row.get("state") == "present" for row in slots)


def _validate_zero_aggregate(slot: Mapping[str, Any], *, model: str) -> None:
    if slot.get("state") == "absent":
        return
    descriptor, value = verify_descriptor(slot.get("descriptor"), "zero-launch aggregate")
    require(descriptor["path"] == slot.get("expected_path"), "zero-launch aggregate path changed")
    counts = value.get("counts")
    expected_schema = {
        "N3": "wmf-n3-behavioral-confirmation-job-v1",
        "D1": "wmf-d1-behavioral-confirmation-simulator-job-v1",
    }.get(model)
    require(
        expected_schema is not None
        and value.get("schema_version") == expected_schema
        and value.get("status") == "technical_failure",
        "zero-launch aggregate model/schema/status changed",
    )
    if isinstance(counts, Mapping):
        require(
            counts.get("launched_behavioral_cells") == 0
            and counts.get("resumed_valid_behavioral_cells") == 0
            and counts.get("newly_launched_behavioral_cells") == 0
            and counts.get("completed_valid_behavioral_cells") == 0
            and counts.get("technically_invalid_behavioral_cells") in {0, 1}
            and counts.get("right_censored_behavioral_cells") == 0
            and counts.get("actual_behavioral_actions") == 0
            and counts.get("actual_behavioral_model_requests") == 0,
            "zero-launch aggregate contains behavioral work",
        )
    else:
        require(
            model == "D1"
            and value.get("exit_code") == 1
            and value.get("all_simulator_children_reaped") is True,
            "count-less aggregate is not the explicit D1 pre-launch fallback",
        )


def _validate_cleanup_slots(slots: Sequence[Mapping[str, Any]], *, model: str) -> None:
    for slot in slots:
        require(
            isinstance(slot, Mapping) and set(slot) == {"state", "expected_path", "descriptor"}
            and slot.get("state") in {"present", "absent"},
            "runtime cleanup slot shape changed",
        )
        if slot["state"] == "absent":
            require(slot.get("descriptor") is None, "absent cleanup slot has a descriptor")
            continue
        identity, value = verify_descriptor(slot.get("descriptor"), "runtime cleanup")
        require(identity["path"] == slot.get("expected_path"), "runtime cleanup path changed")
        schema = value.get("schema_version")
        if schema == "wmf-n3-behavioral-confirmation-cleanup-v1":
            require(
                model == "N3" and value.get("all_children_reaped") is True
                and value.get("remaining_compute_processes") == [],
                "N3 runtime cleanup is not terminal",
            )
        elif schema and "server-exit" in str(schema):
            require(value.get("child_reaped") is True, "N3 server child was not reaped")
        elif schema and "simulator-terminal" in str(schema):
            require(
                model == "D1" and value.get("all_simulator_children_reaped") is True
                and value.get("safe_for_server_shutdown") is True,
                "D1 simulator terminal is not clean",
            )
        elif schema == "wmf-d1-behavioral-confirmation-server-job-v1":
            process_exit = value.get("server_process_exit")
            require(
                model == "D1" and value.get("all_server_children_reaped") is True
                and (
                    process_exit is None
                    or (
                        isinstance(process_exit, Mapping)
                        and process_exit.get("status") == "reaped"
                        and process_exit.get("reaped") is True
                    )
                ),
                "D1 server terminal receipt is not clean",
            )
        elif schema == "wmf-d1-behavioral-confirmation-simulator-job-v1":
            require(
                model == "D1" and value.get("all_simulator_children_reaped") is True,
                "D1 simulator aggregate is not clean",
            )
        else:
            raise ConfirmationReleaseEvidenceError("runtime cleanup schema is unknown")


def _required_argv_option(descriptor: Mapping[str, Any], option: str) -> str:
    argv = descriptor.get("argv")
    require(
        isinstance(argv, list) and argv.count(option) == 1,
        f"confirmation descriptor option missing or duplicated: {option}",
    )
    position = argv.index(option)
    require(
        position + 1 < len(argv) and isinstance(argv[position + 1], str)
        and bool(argv[position + 1]),
        f"confirmation descriptor option value missing: {option}",
    )
    return argv[position + 1]


def _validate_native_release_admission(
    *, source_root: Path, study_commit: str, model: str,
    runtime_evidence: Mapping[str, Any], job_rows: Sequence[Mapping[str, Any]],
    wave: Any,
) -> None:
    """Bind every launched native prefix to its persisted H1 admission receipt."""

    require(
        runtime_evidence.get("form") == "native_cell_prefix",
        "native release admission requires a launched cell prefix",
    )
    aggregate_slots = runtime_evidence.get("aggregate_slots")
    terminal_slots = runtime_evidence.get("terminal_slots")
    require(
        isinstance(aggregate_slots, list) and isinstance(terminal_slots, list),
        "native release admission slots are missing",
    )

    def first_value(slots: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
        present = [slot for slot in slots if slot.get("state") == "present"]
        require(present, f"{label} receipt is missing")
        _identity, value = verify_descriptor(present[0].get("descriptor"), label)
        return value

    bindings: list[tuple[Mapping[str, Any], dict[str, Any], str]]
    if model == "N3":
        require(len(job_rows) == 1, "N3 native admission queue job count changed")
        bindings = [(job_rows[0], first_value(aggregate_slots, "N3 aggregate"), "N3")]
    else:
        require(len(job_rows) == 2, "D1 native admission queue pair changed")
        bindings = [
            (job_rows[0], first_value(terminal_slots[:2], "D1 server terminal"), "D1 server"),
            (job_rows[1], first_value(aggregate_slots, "D1 simulator aggregate"), "D1 simulator"),
        ]
    inventory = runtime_evidence.get("complete_regular_file_inventory")
    require(isinstance(inventory, list), "native admission complete inventory missing")
    inventory_rows = {
        (row.get("path"), row.get("bytes"), row.get("sha256"))
        for row in inventory if isinstance(row, Mapping)
    }
    validated: list[dict[str, Any]] = []
    admission_descriptors: list[dict[str, Any]] = []
    for job_row, receipt, label in bindings:
        descriptor = job_row.get("descriptor_value")
        require(isinstance(descriptor, Mapping), f"{label} queue descriptor missing")
        expected_receipt_schema = {
            "N3": "wmf-n3-behavioral-confirmation-job-v1",
            "D1 server": "wmf-d1-behavioral-confirmation-server-job-v1",
            "D1 simulator": "wmf-d1-behavioral-confirmation-simulator-job-v1",
        }[label]
        require(
            receipt.get("schema_version") == expected_receipt_schema
            and receipt.get("status") in {"passed", "technical_failure"},
            f"{label} aggregate schema/status changed",
        )
        admission_descriptor = receipt.get("release_admission")
        require(isinstance(admission_descriptor, Mapping), f"{label} release admission missing")
        exact_keys(
            admission_descriptor, {"path", "bytes", "sha256"},
            f"{label} release admission descriptor",
        )
        require(
            (
                admission_descriptor.get("path"), admission_descriptor.get("bytes"),
                admission_descriptor.get("sha256"),
            ) in inventory_rows,
            f"{label} release admission is absent from complete runtime inventory",
        )
        value = wave.validate_runtime_admission_receipt(
            receipt_path=Path(str(admission_descriptor["path"])),
            receipt_sha256=str(admission_descriptor["sha256"]),
            source_root=source_root, study_commit=study_commit,
            expected_job_id=str(job_row["job_id"]),
            expected_role=str(descriptor.get("role")),
            expected_finalizer_job_id=_required_argv_option(
                descriptor, "--confirmation-release-finalizer-job-id",
            ),
            expected_consume_by_utc=_required_argv_option(
                descriptor, "--confirmation-release-consume-by-utc",
            ),
            verify_local_worker=False,
        )
        require(isinstance(value, Mapping), f"{label} release admission did not validate")
        validated.append(dict(value))
        admission_descriptors.append(dict(admission_descriptor))
    if model == "D1":
        shared_fields = (
            "release_finalizer_job_id", "consume_by_utc", "pending_publication_commit",
            "published_queue_fragment_sha256", "queue_job_ids",
        )
        require(
            all(validated[0].get(field) == validated[1].get(field) for field in shared_fields),
            "D1 server/simulator release admissions do not share one H1 authority",
        )
        roots = runtime_evidence.get("roots")
        require(isinstance(roots, list) and len(roots) == 3, "D1 native runtime roots changed")
        ack_path = Path(str(roots[2].get("path", ""))) / "release_admission_ack.json"
        matches = [
            row for row in inventory if isinstance(row, Mapping)
            and row.get("path") == str(ack_path)
        ]
        require(len(matches) == 1, "D1 release admission ack is absent from complete inventory")
        ack_identity, ack = load_json_with_identity(ack_path, "D1 release admission ack")
        require(ack_identity == matches[0], "D1 release admission ack changed after inventory")
        exact_keys(
            ack,
            {
                "schema_version", "status", "study_id", "namespace", "study_commit",
                "run_id", "block_id", "layout_pair_id", "release_finalizer_job_id",
                "consume_by_utc", "server_job_id", "simulator_job_id",
                "server_worker_id", "simulator_worker_id", "server_admission",
                "simulator_admission", "server_ready", "simulator_claim",
                "server_publication_verification_payload_sha256",
                "simulator_publication_verification_payload_sha256",
                "pending_publication_commit", "server_verified_remote_head",
                "simulator_verified_remote_head", "published_queue_fragment_sha256",
                "science_reset_request_action_started", "queue_mutated", "jobs_dispatched",
                "completed_at_utc", "claim_boundary", "payload_sha256",
            },
            "D1 release admission ack",
        )
        verify_signed_document(ack, "D1 release admission ack")
        for descriptor_name in ("server_ready", "simulator_claim"):
            sidecar = ack.get(descriptor_name)
            require(
                isinstance(sidecar, Mapping)
                and (
                    sidecar.get("path"), sidecar.get("bytes"), sidecar.get("sha256"),
                ) in inventory_rows,
                f"D1 release ack {descriptor_name} is absent from complete inventory",
            )
        require(
            ack.get("schema_version") == "wmf-d1-confirmation-runtime-admission-ack-v1"
            and ack.get("status") == "distinct_server_simulator_admitted_before_science"
            and ack.get("study_id") == STUDY_ID and ack.get("namespace") == NAMESPACE
            and ack.get("study_commit") == study_commit
            and ack.get("server_job_id") == job_rows[0]["job_id"]
            and ack.get("simulator_job_id") == job_rows[1]["job_id"]
            and ack.get("server_admission") == admission_descriptors[0]
            and ack.get("simulator_admission") == admission_descriptors[1]
            and ack.get("server_worker_id") == validated[0].get("worker_id")
            and ack.get("simulator_worker_id") == validated[1].get("worker_id")
            and ack.get("server_worker_id") != ack.get("simulator_worker_id")
            and ack.get("release_finalizer_job_id") == validated[0].get("release_finalizer_job_id")
            and ack.get("consume_by_utc") == validated[0].get("consume_by_utc")
            and ack.get("pending_publication_commit") == validated[0].get("pending_publication_commit")
            and ack.get("published_queue_fragment_sha256") == validated[0].get("published_queue_fragment_sha256")
            and ack.get("science_reset_request_action_started") is False
            and ack.get("queue_mutated") is False and ack.get("jobs_dispatched") == 0,
            "D1 release admission ack identity or authority changed",
        )


def _confirmation_descriptor_identity(descriptor: Mapping[str, Any]) -> dict[str, Any] | None:
    argv = descriptor.get("argv")
    if not isinstance(argv, list) or len(argv) < 3:
        return None
    runner = argv[1]
    if runner.endswith("/n3_confirmation_block_job.py"):
        model = "N3"
    elif runner.endswith("/d1_confirmation_block_jobs.py"):
        model = "D1"
    else:
        return None
    match = re.fullmatch(
        r"confirmation-(c[0-9]{2})-(n3|d1)-a([0-9]{3})(?:-(server|simulator))?",
        str(descriptor.get("job_id", "")),
    )
    require(match is not None, "confirmation queue job uses a noncanonical ID")
    layout = match.group(1).upper()
    require(match.group(2).upper() == model, "confirmation queue job model/ID differ")
    attempt_number = int(match.group(3))
    mode_suffix = match.group(4)
    mode = argv[2]
    if model == "N3":
        require(mode == "queue" and mode_suffix is None, "N3 confirmation mode/ID changed")
    else:
        require(
            (mode == "server-job" and mode_suffix == "server")
            or (mode == "simulator-job" and mode_suffix == "simulator"),
            "D1 confirmation mode/ID changed",
        )
    return {
        "model": model,
        "layout": layout,
        "attempt_number": attempt_number,
        "attempt_id": f"confirmation-{layout.lower()}-{model.lower()}-a{attempt_number:03d}",
        "mode": mode,
    }


def scan_queue_state(
    *, source_root: Path, state_dir: Path, study_commit: str,
    running_finalizer_job_id: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, int], list[dict[str, Any]]]]:
    """Authenticate every queue job directory, then index confirmation attempts."""

    modules = load_modules(source_root)
    queue = modules["queue"]
    state_root = _existing_directory(Path(state_dir), "queue state root")
    jobs_root = _existing_directory(state_root / "jobs", "queue jobs root")
    all_jobs: dict[str, dict[str, Any]] = {}
    attempts: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    _jobs, jobs_descriptors, jobs_records = _open_directory_chain(
        jobs_root, "queue jobs root",
    )
    try:
        try:
            entry_names = sorted(os.listdir(jobs_descriptors[-1]))
        except OSError as error:
            raise ConfirmationReleaseEvidenceError("queue jobs inventory is unreadable") from error
        require(len(entry_names) == len(set(entry_names)), "queue job inventory is ambiguous")
        with _fetch_authorized_control_history(source_root=source_root) as control_snapshot:
            for entry_name in entry_names:
                job_id = safe_id(entry_name, "queue directory")
                entry_metadata = _entry_metadata(
                    jobs_descriptors[-1], entry_name, f"queue entry {entry_name}",
                )
                require(
                    stat.S_ISDIR(entry_metadata.st_mode),
                    f"orphan/non-directory queue entry: {entry_name}",
                )
                triplet = queue_triplet(
                    state_dir=state_dir, job_id=job_id, source_root=source_root,
                    queue=queue, allow_running_self=(job_id == running_finalizer_job_id),
                    control_snapshot=control_snapshot,
                )
                all_jobs[job_id] = triplet
                identity = _confirmation_descriptor_identity(triplet["descriptor_value"])
                if identity is None:
                    continue
                require(
                    triplet["descriptor_value"]["source_commit"] == study_commit,
                    "confirmation queue attempt uses another study commit",
                )
                attempts.setdefault(
                    (identity["model"], identity["layout"], identity["attempt_number"]), []
                ).append({**triplet, **identity})
        _verify_directory_chain(jobs_root, jobs_records, "queue jobs root")
    finally:
        _close_descriptors(jobs_descriptors)
    return all_jobs, attempts


def _scientific_context(
    *, source_root: Path, study_commit: str, inputs: Mapping[str, Any], modules: Mapping[str, Any],
) -> dict[str, Any]:
    wave = modules["wave"]
    contract = wave.load_contract(source_root)
    runtime = wave.load_runtime_modules(source_root)
    runtime["contract"] = contract
    schedule_identity, layout_order, blocks = wave.validate_schedule(
        source_root=source_root, contract=contract, runtime=runtime,
    )
    identities: dict[str, dict[str, Any]] = {}
    input_documents: dict[str, dict[str, Any]] = {}
    for name in (
        "confirmation_freeze", "fixture_freeze", "resource_qualification",
        "terminal_runtime_identities",
    ):
        identity, input_document = verify_descriptor(inputs[name], name)
        identities[name] = identity
        input_documents[name] = input_document
    release, _release_validation, models = wave.validate_release_freeze(
        descriptor=identities["confirmation_freeze"], source_root=source_root,
        contract=contract, runtime=runtime,
    )
    fixture_validation = runtime["fixture"].validate_fixture_freeze(
        Path(identities["fixture_freeze"]["path"]), identities["fixture_freeze"]["sha256"],
        source_root=source_root, expected_study_commit=study_commit,
        deep_validate_selected=False,
    )
    require(fixture_validation.get("layout_count") == 24, "fixture freeze lacks 24 layouts")
    fixture_value = input_documents["fixture_freeze"]
    fixture_rows = {
        row["layout_pair_id"]: row
        for row in fixture_value.get("layouts", []) if isinstance(row, Mapping)
    }
    require(set(fixture_rows) == set(layout_order), "fixture layout inventory changed")
    wave.validate_resource_qualification(
        descriptor=identities["resource_qualification"], study_commit=study_commit,
        models=models, release=release, contract=contract,
    )
    _runtime_publication, prerequisites = wave.validate_runtime_identities(
        descriptor=identities["terminal_runtime_identities"], source_root=source_root,
        study_commit=study_commit, contract=contract,
    )
    return {
        "wave": wave, "contract": contract, "runtime": runtime,
        "schedule_identity": schedule_identity, "layout_order": layout_order,
        "blocks": blocks, "release": release, "models": models,
        "fixture_rows": fixture_rows, "prerequisites": prerequisites,
        "identities": identities,
    }


def _attempt_roots_and_slots(
    *, model: str, block: Any, attempt_id: str, job_ids: Sequence[str], state_dir: Path,
) -> tuple[list[Path], Path, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    state_root = _existing_directory(Path(state_dir), "attempt queue state root")
    queue_jobs = _existing_directory(state_root / "jobs", "attempt queue jobs root")
    raw_root = _validate_no_symlink_components(
        Path(block.raw_root), "trusted block raw root", allow_missing=True,
    )
    if model == "N3":
        attempt_root = _confined_path(
            raw_root / job_ids[0], anchor=raw_root,
            label="N3 attempt root", allow_missing=True,
        )
        roots = [attempt_root]
        aggregate = [
            _artifact_slot(
                queue_jobs / job_ids[0] / "publish" / "n3_behavioral_confirmation_receipt.json",
                anchor=queue_jobs,
            ),
            _artifact_slot(
                attempt_root / "publish" / "n3_behavioral_confirmation_receipt.json",
                anchor=raw_root,
            ),
        ]
        cleanup = [
            _artifact_slot(attempt_root / "cleanup.json", anchor=raw_root),
            _artifact_slot(
                attempt_root / "server" / "supervisor_exit.json", anchor=raw_root,
            ),
        ]
        terminal: list[dict[str, Any]] = []
        cell_root = attempt_root
    else:
        server_id, simulator_id = job_ids
        server_root = _confined_path(
            raw_root / "server_attempts" / server_id, anchor=raw_root,
            label="D1 server attempt root", allow_missing=True,
        )
        simulator_root = _confined_path(
            raw_root / "simulator_attempts" / simulator_id, anchor=raw_root,
            label="D1 simulator attempt root", allow_missing=True,
        )
        coordination_root = _confined_path(
            raw_root / "coordination" / attempt_id, anchor=raw_root,
            label="D1 coordination root", allow_missing=True,
        )
        roots = [server_root, simulator_root, coordination_root]
        aggregate = [
            _artifact_slot(
                queue_jobs / simulator_id / "publish" / "d1_behavioral_confirmation_receipt.json",
                anchor=queue_jobs,
            ),
            _artifact_slot(
                simulator_root / "publish" / "d1_behavioral_confirmation_receipt.json",
                anchor=raw_root,
            ),
        ]
        cleanup = []
        terminal = [
            _artifact_slot(
                queue_jobs / server_id / "publish" / "d1_behavioral_server_receipt.json",
                anchor=queue_jobs,
            ),
            _artifact_slot(
                server_root / "publish" / "d1_behavioral_server_receipt.json",
                anchor=raw_root,
            ),
            _artifact_slot(
                coordination_root / "simulator_terminal.json", anchor=raw_root,
            ),
        ]
        cell_root = simulator_root
    return roots, cell_root, aggregate, cleanup, terminal


def _build_runtime_evidence(
    *, model: str, block: Any, attempt_id: str, job_rows: Sequence[Mapping[str, Any]],
    state_dir: Path,
) -> tuple[dict[str, Any], list[Path], list[Path]]:
    job_ids = [str(row["job_id"]) for row in job_rows]
    raw_root, raw_root_snapshot = _path_component_snapshot(
        Path(block.raw_root), "trusted block raw root", allow_missing=True,
    )
    roots, cell_root, aggregate_slots, cleanup_slots, terminal_slots = _attempt_roots_and_slots(
        model=model, block=block, attempt_id=attempt_id,
        job_ids=job_ids, state_dir=state_dir,
    )
    passed, failed = _cell_receipt_paths(cell_root, anchor=raw_root)
    require(
        not ({_condition_index_from_path(path) for path in passed}
             & {_condition_index_from_path(path) for path in failed}),
        "cell contains both passed and failure receipts",
    )
    inventory = _regular_inventory(roots, anchor=raw_root)
    queue_results = [row["result_value"] for row in job_rows]
    require(all(result is not None for result in queue_results), "attempt queue result missing")
    zero = not passed and not failed
    if zero:
        require(
            any(result["status"] != "succeeded" for result in queue_results),
            "zero-launch attempt has only successful queue results",
        )
        for slot in aggregate_slots:
            _validate_zero_aggregate(slot, model=model)
        if not _slot_present(aggregate_slots):
            behavioral_result = queue_results[0] if model == "N3" else queue_results[-1]
            require(
                behavioral_result.get("status") != "succeeded"
                and behavioral_result.get("child_pid") is None
                and behavioral_result.get("returncode") is None
                and isinstance(behavioral_result.get("error_type"), str)
                and bool(behavioral_result["error_type"]),
                "aggregate-less zero launch lacks queue proof that the behavioral child never spawned",
            )
            require(
                _regular_inventory([cell_root], anchor=raw_root) == [],
                "aggregate-less zero launch has behavioral runtime artifacts",
            )
        form = "zero_launch_technical_failure"
        reason = "terminal_queue_failure_before_behavioral_launch"
    else:
        require(_slot_present(aggregate_slots), "native-cell attempt lacks aggregate receipt")
        form = "native_cell_prefix"
        reason = None
    _validate_cleanup_slots(cleanup_slots + terminal_slots, model=model)
    if model == "N3" and queue_results[0].get("child_pid") is not None:
        require(
            cleanup_slots and cleanup_slots[0].get("state") == "present",
            "launched N3 queue child lacks terminal cleanup evidence",
        )
    if model == "D1":
        if queue_results[0].get("child_pid") is not None:
            require(
                _slot_present(terminal_slots[:2]),
                "launched D1 server queue child lacks terminal runtime receipt",
            )
        if queue_results[-1].get("child_pid") is not None:
            require(
                terminal_slots and terminal_slots[-1].get("state") == "present",
                "launched D1 simulator queue child lacks simulator-terminal receipt",
            )
    for mirrored in (aggregate_slots, terminal_slots[:2] if model == "D1" else []):
        hashes = {
            row["descriptor"]["sha256"] for row in mirrored
            if row.get("state") == "present" and isinstance(row.get("descriptor"), Mapping)
        }
        require(len(hashes) <= 1, "mirrored runtime receipts differ")
    evidence = {
        "form": form,
        "zero_behavioral_cells_launched": zero,
        "roots": [
            {"path": str(path), "state": _directory_state(path, "runtime evidence root")}
            for path in roots
        ],
        "aggregate_slots": aggregate_slots,
        "cleanup_slots": cleanup_slots,
        "terminal_slots": terminal_slots,
        "cell_receipts": [file_identity(path) for path in passed],
        "failure_receipts": [file_identity(path) for path in failed],
        "complete_regular_file_inventory": inventory,
        "inventory_complete": True,
        "claim_control_history": [dict(row["claim_control"]) for row in job_rows],
        "reason": reason,
    }
    _require_component_snapshot(
        raw_root, raw_root_snapshot, "trusted block raw root",
        exact=True, allow_missing=True,
    )
    return evidence, passed, failed


def assemble_result_attempt_ledger(
    *, source_root: Path, state_dir: Path, study_commit: str,
    inputs: Mapping[str, Any], running_finalizer_job_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Derive the complete retry ledger from immutable queue/PVC evidence."""

    modules = load_modules(source_root)
    context = _scientific_context(
        source_root=source_root, study_commit=study_commit, inputs=inputs, modules=modules,
    )
    all_jobs, indexed = scan_queue_state(
        source_root=source_root, state_dir=state_dir, study_commit=study_commit,
        running_finalizer_job_id=running_finalizer_job_id,
    )
    wave = context["wave"]
    rows: list[dict[str, Any]] = []
    for model in context["models"]:
        for layout in context["layout_order"]:
            block = context["blocks"][(model, layout)]
            ordinals = sorted(
                number for (candidate_model, candidate_layout, number) in indexed
                if (candidate_model, candidate_layout) == (model, layout)
            )
            require(ordinals == list(range(1, len(ordinals) + 1)), f"retry ordinals are not contiguous: {model}:{layout}")
            attempts: list[dict[str, Any]] = []
            prefix = 0
            terminal_state: str | None = None
            previous: str | None = None
            for number in ordinals:
                require(terminal_state is None, f"retry follows terminal block: {model}:{layout}")
                attempt_id, expected_job_ids = wave._attempt_identity(model, layout, number)
                job_rows = indexed[(model, layout, number)]
                order = {job_id: index for index, job_id in enumerate(expected_job_ids)}
                require(
                    {row["job_id"] for row in job_rows} == set(expected_job_ids),
                    f"attempt queue pair is incomplete: {attempt_id}",
                )
                job_rows = sorted(job_rows, key=lambda row: order[row["job_id"]])
                evidence, passed_paths, failure_paths = _build_runtime_evidence(
                    model=model, block=block, attempt_id=attempt_id,
                    job_rows=job_rows, state_dir=state_dir,
                )
                if evidence["form"] == "native_cell_prefix":
                    _validate_native_release_admission(
                        source_root=source_root, study_commit=study_commit,
                        model=model, runtime_evidence=evidence,
                        job_rows=job_rows, wave=wave,
                    )
                cell_paths = sorted(
                    [*passed_paths, *failure_paths], key=_condition_index_from_path
                )
                cells: list[dict[str, Any]] = []
                if evidence["form"] == "zero_launch_technical_failure":
                    last_state = "technical_invalid"
                else:
                    expected_index = prefix
                    last_state = None
                    for position, path in enumerate(cell_paths):
                        index = _condition_index_from_path(path)
                        require(index == expected_index, f"attempt native receipts are not a contiguous retry suffix: {attempt_id}")
                        native = wave._native_cell_state(
                            runtime=context["runtime"], model=model, block=block,
                            condition_index=index, study_commit=study_commit,
                            receipt_path=path,
                        )
                        cells.append({"condition_index": index, "state": native, "receipt": file_identity(path)})
                        last_state = native
                        if native == "passed":
                            prefix += 1
                            expected_index += 1
                            if prefix == 4:
                                terminal_state = "passed"
                        elif native == "safety_censored":
                            require(position == len(cell_paths) - 1, "cell follows safety censor")
                            terminal_state = "safety_censored"
                        else:
                            require(native == "technical_invalid" and position == len(cell_paths) - 1, "cell follows technical failure")
                    require(last_state is not None, "native attempt has no native receipts")
                if terminal_state is None and last_state == "passed":
                    # A durable passed prefix is retryable only when the
                    # reaped outer job proves a technical interruption. Match
                    # the consumer: success cannot explain missing cells.
                    require(
                        any(row["result_value"]["status"] != "succeeded" for row in job_rows),
                        "partial passed prefix has no terminal outer technical failure",
                    )
                    last_state = "technical_invalid"
                if last_state in {"technical_invalid", "safety_censored"}:
                    require(
                        any(row["result_value"]["status"] != "succeeded" for row in job_rows),
                        "failed/censored native attempt has only successful queue results",
                    )
                attempts.append({
                    "attempt_id": attempt_id,
                    "attempt_number": number,
                    "predecessor_attempt_id": previous,
                    "start_cell_index": cells[0]["condition_index"] if cells else prefix,
                    "job_ids": list(expected_job_ids),
                    "queue_descriptors": [row["descriptor"] for row in job_rows],
                    "queue_claims": [row["claim"] for row in job_rows],
                    "queue_results": [row["result"] for row in job_rows],
                    "cells": cells,
                    "runtime_evidence": evidence,
                })
                previous = attempt_id
            derived = terminal_state or ("technical_invalid" if attempts else "not_run")
            rows.append({
                "model_config": model,
                "layout_pair_id": layout,
                "condition_order": list(block.condition_order),
                "cell_ids": list(block.cell_ids),
                "state": derived,
                "completed_prefix_cells": prefix,
                "terminal_attempt_id": previous if terminal_state is not None else None,
                "attempts": attempts,
            })
    identities = context["identities"]
    ledger = signed_document({
        "schema_version": LEDGER_SCHEMA,
        "status": "authenticated_native_attempt_inventory",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "cohort_branch": context["release"]["cohort_branch"],
        "qualified_model_ids": list(context["models"]),
        "prepared_schedule": context["schedule_identity"],
        "confirmation_freeze": identities["confirmation_freeze"],
        "fixture_freeze": identities["fixture_freeze"],
        "resource_qualification": identities["resource_qualification"],
        "terminal_runtime_identities": identities["terminal_runtime_identities"],
        "layout_order": list(context["layout_order"]),
        "blocks": rows,
        "attempt_inventory_complete": True,
        "claim_boundary": (
            "Every confirmation attempt was derived from the complete queue/PVC scan. "
            "Native receipts are model-validated; explicit zero-launch technical attempts "
            "remain non-scientific missingness and never become passed or censored cells."
        ),
    })
    validate_result_attempt_ledger(
        ledger, source_root=source_root, state_dir=state_dir,
        study_commit=study_commit, all_queue_jobs=all_jobs,
    )
    return ledger, all_jobs


def validate_result_attempt_ledger(
    value: Mapping[str, Any], *, source_root: Path, state_dir: Path,
    study_commit: str, all_queue_jobs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate producer-specific retry/absence evidence without weakening native consumers."""

    exact_keys(
        value,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "cohort_branch", "qualified_model_ids", "prepared_schedule",
            "confirmation_freeze", "fixture_freeze", "resource_qualification",
            "terminal_runtime_identities", "layout_order", "blocks",
            "attempt_inventory_complete", "claim_boundary", "payload_sha256",
        },
        "result attempt ledger",
    )
    verify_signed_document(value, "result attempt ledger")
    require(
        value.get("schema_version") == LEDGER_SCHEMA
        and value.get("status") == "authenticated_native_attempt_inventory"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit
        and value.get("attempt_inventory_complete") is True,
        "result attempt ledger identity changed",
    )
    modules = load_modules(source_root)
    queue = modules["queue"]
    context = _scientific_context(
        source_root=source_root,
        study_commit=study_commit,
        inputs={
            name: value[name]
            for name in (
                "confirmation_freeze", "fixture_freeze", "resource_qualification",
                "terminal_runtime_identities",
            )
        },
        modules=modules,
    )
    require(
        value.get("cohort_branch") == context["release"]["cohort_branch"]
        and value.get("qualified_model_ids") == list(context["models"])
        and value.get("prepared_schedule") == context["schedule_identity"]
        and value.get("layout_order") == list(context["layout_order"]),
        "result attempt ledger freeze/schedule binding changed",
    )
    state_root = _existing_directory(Path(state_dir), "ledger queue state root")
    jobs_root = _existing_directory(state_root / "jobs", "ledger queue jobs root")
    if all_queue_jobs is None:
        control = load_json(
            _confined_path(
                state_root / "control.json", anchor=state_root,
                label="ledger validation control", allow_missing=False,
            ),
            "ledger validation control",
        )
        exact_keys(control, CONTROL_KEYS, "ledger validation control")
        active = control.get("active_job_ids")
        require(
            isinstance(active, list) and len(active) == 1
            and isinstance(active[0], str) and FINALIZER_RE.fullmatch(active[0]) is not None,
            "standalone ledger replay requires the sole running finalizer",
        )
        all_queue_jobs, _indexed = scan_queue_state(
            source_root=source_root, state_dir=state_dir, study_commit=study_commit,
            running_finalizer_job_id=active[0],
        )
    blocks = value.get("blocks")
    expected_block_order = [
        (model, layout) for model in context["models"] for layout in context["layout_order"]
    ]
    require(
        isinstance(blocks, list) and len(blocks) == len(expected_block_order),
        "result attempt ledger blocks missing",
    )
    seen_attempts: set[str] = set()
    seen_jobs: set[str] = set()
    claims: dict[str, dict[str, Any]] = {}
    by_block: dict[tuple[str, str], dict[str, Any]] = {}
    for block, expected_block_key in zip(blocks, expected_block_order, strict=True):
        require(isinstance(block, Mapping) and isinstance(block.get("attempts"), list), "ledger block invalid")
        exact_keys(
            block,
            {
                "model_config", "layout_pair_id", "condition_order", "cell_ids",
                "state", "completed_prefix_cells", "terminal_attempt_id", "attempts",
            },
            "result ledger block",
        )
        model, layout = expected_block_key
        prepared_block = context["blocks"][(model, layout)]
        require(
            (block.get("model_config"), block.get("layout_pair_id")) == expected_block_key
            and block.get("condition_order") == list(prepared_block.condition_order)
            and block.get("cell_ids") == list(prepared_block.cell_ids),
            "result ledger block roster/permutation changed",
        )
        prefix = 0
        terminal = False
        previous = None
        for number, attempt in enumerate(block["attempts"], start=1):
            require(not terminal and isinstance(attempt, Mapping), "attempt follows terminal block")
            exact_keys(
                attempt,
                {
                    "attempt_id", "attempt_number", "predecessor_attempt_id", "start_cell_index",
                    "job_ids", "queue_descriptors", "queue_claims", "queue_results", "cells",
                    "runtime_evidence",
                },
                "result ledger attempt",
            )
            attempt_id = safe_id(attempt.get("attempt_id"), "ledger attempt")
            require(
                attempt_id not in seen_attempts and attempt.get("attempt_number") == number
                and attempt.get("predecessor_attempt_id") == previous,
                "attempt retry chain changed",
            )
            seen_attempts.add(attempt_id)
            previous = attempt_id
            job_ids = attempt.get("job_ids")
            require(
                isinstance(job_ids, list) and job_ids
                and all(safe_id(item, "ledger queue job") for item in job_ids)
                and not (set(job_ids) & seen_jobs),
                "ledger queue job inventory invalid",
            )
            seen_jobs.update(job_ids)
            descriptors = attempt.get("queue_descriptors")
            claim_rows = attempt.get("queue_claims")
            result_rows = attempt.get("queue_results")
            require(
                isinstance(descriptors, list) and isinstance(claim_rows, list)
                and isinstance(result_rows, list)
                and len(descriptors) == len(claim_rows) == len(result_rows) == len(job_ids),
                "attempt queue triplet inventory changed",
            )
            for job_id, descriptor_row, claim_row, result_row in zip(
                job_ids, descriptors, claim_rows, result_rows, strict=True,
            ):
                descriptor_identity, descriptor_value = verify_descriptor(
                    descriptor_row, f"ledger descriptor {job_id}"
                )
                claim_identity, claim_value = verify_descriptor(
                    claim_row, f"ledger claim {job_id}"
                )
                result_identity, result_value = verify_descriptor(
                    result_row, f"ledger result {job_id}"
                )
                job_root = _confined_path(
                    jobs_root / job_id, anchor=jobs_root,
                    label=f"ledger queue job {job_id}", allow_missing=False,
                )
                require(
                    descriptor_identity["path"] == str(job_root / "descriptor.json")
                    and claim_identity["path"] == str(job_root / "claim" / "owner.json")
                    and result_identity["path"] == str(job_root / "result.json"),
                    "ledger queue triplet paths changed",
                )
                validate_queue_result(
                    result_value, descriptor=descriptor_value, queue=queue,
                    state_dir=state_dir, job_dir=job_root,
                )
                validate_queue_claim(
                    claim_value, descriptor=descriptor_value,
                    result=result_value, queue=queue,
                )
                require(
                    all_queue_jobs is not None and job_id in all_queue_jobs
                    and all_queue_jobs[job_id]["descriptor"] == descriptor_identity
                    and all_queue_jobs[job_id]["claim"] == claim_identity
                    and all_queue_jobs[job_id]["result"] == result_identity
                    and all_queue_jobs[job_id]["descriptor_value"] == descriptor_value
                    and all_queue_jobs[job_id]["claim_value"] == claim_value
                    and all_queue_jobs[job_id]["result_value"] == result_value,
                    "ledger queue triplet differs from complete queue scan",
                )
                claims[job_id] = claim_value
            runtime = attempt.get("runtime_evidence")
            require(isinstance(runtime, Mapping), "attempt runtime evidence missing")
            exact_keys(
                runtime,
                {
                    "form", "zero_behavioral_cells_launched", "roots", "aggregate_slots",
                    "cleanup_slots", "terminal_slots", "cell_receipts", "failure_receipts",
                    "complete_regular_file_inventory", "inventory_complete",
                    "claim_control_history", "reason",
                },
                "attempt runtime evidence",
            )
            require(runtime.get("inventory_complete") is True, "attempt inventory is incomplete")
            roots = runtime.get("roots")
            require(isinstance(roots, list), "attempt roots missing")
            expected_roots, expected_cell_root, expected_aggregate, expected_cleanup, expected_terminal = (
                _attempt_roots_and_slots(
                    model=model, block=prepared_block, attempt_id=attempt_id,
                    job_ids=job_ids, state_dir=state_dir,
                )
            )
            expected_root_rows = [
                {"path": str(path), "state": _directory_state(path, "runtime evidence root")}
                for path in expected_roots
            ]
            require(roots == expected_root_rows, "attempt runtime roots differ from trusted schedule paths")
            raw_root = _validate_no_symlink_components(
                Path(prepared_block.raw_root), "trusted block raw root", allow_missing=True,
            )
            observed_inventory = _regular_inventory(expected_roots, anchor=raw_root)
            require(observed_inventory == runtime.get("complete_regular_file_inventory"), "attempt artifacts changed after inventory")
            histories = runtime.get("claim_control_history")
            require(
                isinstance(histories, list) and len(histories) == len(job_ids)
                and [row.get("job_id") for row in histories] == job_ids,
                "attempt claim-control history incomplete",
            )
            require(
                all_queue_jobs is not None
                and histories == [all_queue_jobs[job_id]["claim_control"] for job_id in job_ids],
                "attempt claim-control history differs from authenticated queue commits",
            )
            for name in ("aggregate_slots", "cleanup_slots", "terminal_slots"):
                slots = runtime.get(name)
                require(isinstance(slots, list), f"attempt {name} invalid")
                expected_slots = {
                    "aggregate_slots": expected_aggregate,
                    "cleanup_slots": expected_cleanup,
                    "terminal_slots": expected_terminal,
                }[name]
                require(slots == expected_slots, f"attempt {name} differs from trusted runtime paths")
                for slot in slots:
                    require(
                        isinstance(slot, Mapping)
                        and set(slot) == {"state", "expected_path", "descriptor"}
                        and slot.get("state") in {"present", "absent"}
                        and isinstance(slot.get("expected_path"), str),
                        f"attempt {name} slot changed",
                    )
                    queue_jobs_root = _existing_directory(
                        _existing_directory(Path(state_dir), "queue state root") / "jobs",
                        "queue jobs root",
                    )
                    lexical_slot = _lexical_absolute_path(
                        Path(slot["expected_path"]), f"attempt {name} slot path",
                    )
                    slot_anchor = (
                        queue_jobs_root
                        if lexical_slot.is_relative_to(queue_jobs_root)
                        else raw_root
                    )
                    expected_path = _confined_path(
                        lexical_slot, anchor=slot_anchor,
                        label=f"attempt {name} slot path", allow_missing=True,
                    )
                    if slot["state"] == "present":
                        identity = file_identity(expected_path)
                        require(identity == slot.get("descriptor"), f"attempt {name} changed after inventory")
                    else:
                        absent_metadata = _optional_path_metadata(
                            expected_path, f"attempt absent {name}",
                        )
                        require(
                            slot.get("descriptor") is None
                            and absent_metadata is None,
                            f"attempt absent {name} appeared after inventory",
                        )
            _validate_cleanup_slots(
                [*runtime["cleanup_slots"], *runtime["terminal_slots"]], model=model,
            )
            for descriptor in [*runtime["cell_receipts"], *runtime["failure_receipts"]]:
                identity = file_identity(Path(descriptor["path"]))
                require(identity == descriptor, "native runtime descriptor changed")
            zero = runtime.get("form") == "zero_launch_technical_failure"
            if zero:
                require(
                    runtime.get("zero_behavioral_cells_launched") is True
                    and runtime.get("cell_receipts") == []
                    and runtime.get("failure_receipts") == []
                    and attempt.get("cells") == []
                    and runtime.get("reason") == "terminal_queue_failure_before_behavioral_launch",
                    "zero-launch technical evidence changed",
                )
                for slot in runtime["aggregate_slots"]:
                    _validate_zero_aggregate(slot, model=model)
                if not _slot_present(runtime["aggregate_slots"]):
                    behavioral_job_id = job_ids[0] if model == "N3" else job_ids[-1]
                    behavioral_result = all_queue_jobs[behavioral_job_id]["result_value"]
                    require(
                        behavioral_result.get("status") != "succeeded"
                        and behavioral_result.get("child_pid") is None
                        and behavioral_result.get("returncode") is None
                        and isinstance(behavioral_result.get("error_type"), str)
                        and bool(behavioral_result["error_type"]),
                        "aggregate-less zero launch lost no-child-spawn proof",
                    )
                    cell_root = expected_cell_root
                    require(
                        _regular_inventory([cell_root], anchor=raw_root) == [],
                        "aggregate-less zero launch gained behavioral runtime artifacts",
                    )
            else:
                require(
                    runtime.get("form") == "native_cell_prefix"
                    and runtime.get("zero_behavioral_cells_launched") is False
                    and runtime.get("reason") is None
                    and attempt.get("cells"),
                    "native attempt evidence changed",
                )
                _validate_native_release_admission(
                    source_root=source_root, study_commit=study_commit,
                    model=model, runtime_evidence=runtime,
                    job_rows=[all_queue_jobs[job_id] for job_id in job_ids],
                    wave=context["wave"],
                )
            cells = attempt.get("cells")
            require(isinstance(cells, list), "attempt cells invalid")
            last_state = "technical_invalid" if zero else None
            if cells:
                require(cells[0].get("condition_index") == prefix, "retry prefix start changed")
            for index, cell in enumerate(cells):
                exact_keys(cell, {"condition_index", "state", "receipt"}, "ledger cell")
                require(cell.get("condition_index") == prefix, "cell retry prefix is not contiguous")
                receipt_identity, _receipt_value = verify_descriptor(
                    cell.get("receipt"), "ledger native cell receipt"
                )
                state = context["wave"]._native_cell_state(
                    runtime=context["runtime"], model=model, block=prepared_block,
                    condition_index=prefix, study_commit=study_commit,
                    receipt_path=Path(receipt_identity["path"]),
                )
                require(cell.get("state") == state, "ledger promoted a non-native cell state")
                require(state in {"passed", "technical_invalid", "safety_censored"}, "cell state invalid")
                last_state = state
                if state == "passed":
                    prefix += 1
                    if prefix == 4:
                        terminal = True
                else:
                    require(index == len(cells) - 1, "cell follows terminal failure/censor")
                    terminal = state == "safety_censored"
            statuses = [all_queue_jobs[job_id]["result_value"]["status"] for job_id in job_ids]
            if zero:
                require(any(status != "succeeded" for status in statuses), "zero-launch attempt outer results all succeeded")
            else:
                if not terminal and last_state == "passed":
                    require(
                        any(status != "succeeded" for status in statuses),
                        "partial passed prefix has no terminal outer technical failure",
                    )
                    last_state = "technical_invalid"
                if last_state in {"technical_invalid", "safety_censored"}:
                    require(
                        any(status != "succeeded" for status in statuses),
                        "failed/censored native attempt has only successful queue results",
                    )
        expected_state = "passed" if prefix == 4 else (
            "safety_censored" if terminal else ("technical_invalid" if block["attempts"] else "not_run")
        )
        require(
            block.get("completed_prefix_cells") == prefix
            and block.get("state") == expected_state,
            "ledger block summary differs from retry replay",
        )
        require(
            block.get("terminal_attempt_id") == (previous if expected_state in {"passed", "safety_censored"} else None),
            "ledger terminal attempt identity changed",
        )
        key = (str(block.get("model_config")), str(block.get("layout_pair_id")))
        require(key not in by_block, "ledger block duplicated")
        by_block[key] = dict(block)
    confirmation_queue_jobs = {
        job_id for job_id, row in all_queue_jobs.items()
        if _confirmation_descriptor_identity(row["descriptor_value"]) is not None
    }
    require(
        seen_jobs == confirmation_queue_jobs,
        "ledger omits or invents a confirmation queue attempt",
    )
    return {
        "document": dict(value),
        "blocks": by_block,
        "attempt_ids": seen_attempts,
        "job_ids": seen_jobs,
        "queue_claims": claims,
    }


def _attestation_identity(
    job_id: Any, *, topology: Mapping[str, Mapping[str, Any]],
) -> tuple[str, int, str]:
    require(isinstance(job_id, str), "attestation job ID is missing")
    match = ATTESTATION_RE.fullmatch(job_id)
    require(match is not None, "attestation job ID is not attempt-scoped")
    lane = match.group(1).upper()
    attempt_number = int(match.group(2))
    worker_id = "wmf-forecast-0912-worker-" + match.group(3)
    require(worker_id in topology, "attestation job names an unknown fixed worker")
    return lane, attempt_number, worker_id


def _attestation_scope(job_ids: Sequence[str]) -> tuple[str, int]:
    require(
        isinstance(job_ids, Sequence) and not isinstance(job_ids, (str, bytes))
        and 1 <= len(job_ids) <= 32
        and all(isinstance(job_id, str) for job_id in job_ids)
        and list(job_ids) == sorted(set(job_ids)),
        "worker attestation job inventory is not a nonempty ordered subset",
    )
    matches = [ATTESTATION_RE.fullmatch(str(item)) for item in job_ids]
    require(all(match is not None for match in matches), "worker attestation job ID changed")
    scopes = {
        (match.group(1).upper(), int(match.group(2)))
        for match in matches if match is not None
    }
    require(len(scopes) == 1, "worker attestations mix finalizer attempts")
    return next(iter(scopes))


def _expected_attestation_jobs(
    source_root: Path, study_commit: str, *, lane: str, attempt_number: int,
    max_wall_seconds: int = 600,
) -> dict[str, dict[str, Any]]:
    topology = deployment_topology(source_root)
    require(
        lane in {"N3", "D1"}
        and type(attempt_number) is int and 1 <= attempt_number <= 999,
        "attestation finalizer scope is invalid",
    )
    finalizer_job_id = (
        f"confirmation-release-finalizer-{lane.lower()}-a{attempt_number:03d}"
    )
    producer = _producer_relative(source_root)
    result: dict[str, dict[str, Any]] = {}
    for worker_id in sorted(topology):
        suffix = worker_id.removeprefix("wmf-forecast-0912-worker-")
        job_id = (
            f"confirmation-release-worker-attest-{lane.lower()}-"
            f"a{attempt_number:03d}-{suffix}"
        )
        result[worker_id] = _raw_queue_job(
            job_id=job_id, role=topology[worker_id]["role"],
            source_commit=study_commit, max_wall_seconds=max_wall_seconds,
            argv=[
                "/usr/bin/python3", "{source_root}/" + producer, "attest-worker",
                "--source-root", "{source_root}",
                "--state-dir", "{state_dir}",
                "--job-dir", "{job_dir}",
                "--job-id", job_id,
                "--study-commit", study_commit,
                "--expected-worker-id", worker_id,
                "--expected-finalizer-job-id", finalizer_job_id,
            ],
        )
    return result


def _validate_worker_attestation_payload(
    value: Mapping[str, Any], *, source_root: Path, state_dir: Path,
    study_commit: str, expected_triplet: Mapping[str, Any], as_of: datetime,
    expected_semantic_active_job_ids: Sequence[str],
    require_staged_path_exists: bool = True,
) -> dict[str, Any]:
    """Deep-check one server-authored, worker-local attestation payload."""

    exact_keys(value, WORKER_ATTESTATION_KEYS, "worker attestation")
    verify_signed_document(value, "worker attestation")
    require(
        value.get("schema_version") == WORKER_ATTESTATION_SCHEMA
        and value.get("status") == "authenticated_worker_gpu_idle_while_attesting"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit,
        "worker attestation identity changed",
    )
    job_id = safe_id(expected_triplet.get("job_id"), "attestation queue job")
    worker_id = safe_id(value.get("worker_id"), "attested worker")
    topology = deployment_topology(source_root)
    require(worker_id in topology, "attested worker is outside fixed topology")
    expected = topology[worker_id]
    descriptor = expected_triplet.get("descriptor_value")
    claim = expected_triplet.get("claim_value")
    require(
        isinstance(descriptor, Mapping) and isinstance(claim, Mapping)
        and value.get("queue_job_id") == job_id
        and value.get("queue_descriptor") == expected_triplet.get("descriptor")
        and value.get("queue_claim") == expected_triplet.get("claim")
        and value.get("claim_control") == expected_triplet.get("claim_control")
        and value.get("role") == expected["role"]
        and claim.get("worker_id") == worker_id,
        "worker attestation is detached from queue/deployment identity",
    )
    claim_control = value.get("claim_control")
    require(isinstance(claim_control, Mapping), "worker attestation claim-control missing")
    exact_keys(
        claim_control,
        {
            "job_id", "control_commit", "control_generation",
            "control_queue_blob_sha256", "descriptor_sha256",
        },
        "worker attestation claim-control",
    )
    semantic = value.get("semantic_control")
    require(isinstance(semantic, Mapping), "worker attestation semantic control missing")
    exact_keys(
        semantic,
        {
            "namespace", "control_commit", "observed_generation", "claim_generation",
            "admission_deadline_unix", "shutdown", "active_job_ids",
        },
        "worker attestation semantic control",
    )
    deadline = semantic.get("admission_deadline_unix")
    require(
        semantic.get("namespace") == NAMESPACE
        and semantic.get("control_commit") == claim.get("control_commit")
        and semantic.get("claim_generation") == claim.get("control_generation")
        and type(semantic.get("observed_generation")) is int
        and semantic["observed_generation"] >= claim.get("control_generation", sys.maxsize)
        and semantic["observed_generation"] != load_contract(source_root)["queue"]["forbidden_control_generation"]
        and not isinstance(deadline, bool) and isinstance(deadline, (int, float))
        and math.isfinite(deadline) and semantic.get("shutdown") is False
        and semantic.get("active_job_ids") == list(expected_semantic_active_job_ids),
        "worker attestation semantic control changed",
    )
    observed = parse_utc(value.get("observed_at_utc"), "worker attestation time")
    consume_by = parse_utc(value.get("consume_by_utc"), "worker attestation consume-by")
    require(
        0 <= (as_of - observed).total_seconds()
        <= load_contract(source_root)["queue"]["maximum_attestation_age_seconds"]
        and value.get("consume_by_utc")
        == utc_after(value["observed_at_utc"], MAX_EVIDENCE_AGE_SECONDS)
        and observed <= as_of <= consume_by
        and float(deadline) > observed.timestamp(),
        "worker attestation is stale, future-dated, or past admission",
    )
    require(
        isinstance(value.get("hostname"), str) and bool(value["hostname"])
        and isinstance(value.get("pod_uid"), str)
        and POD_UID_RE.fullmatch(value["pod_uid"]) is not None
        and value.get("pod_uid_source") in {"downward_api_metadata_uid", "proc_self_cgroup"},
        "worker pod identity invalid",
    )
    deployment = value.get("deployment")
    require(isinstance(deployment, Mapping), "worker deployment evidence missing")
    exact_keys(
        deployment,
        {
            "manifest", "manifest_relative_path", "item_index", "job_name",
            "container_name", "image", "gpu_count", "controller_argv_sha256",
            "controller_pid",
        },
        "worker deployment evidence",
    )
    manifest = deployment.get("manifest")
    require(isinstance(manifest, Mapping), "worker deployment manifest descriptor missing")
    exact_keys(manifest, {"path", "bytes", "sha256"}, "worker deployment manifest descriptor")
    relative = deployment.get("manifest_relative_path")
    require(isinstance(relative, str) and relative in load_contract(source_root)["deployment_manifests"], "worker deployment manifest path changed")
    source_root_path = _existing_directory(Path(source_root), "worker source root")
    local_manifest_path = _confined_path(
        source_root_path / relative, anchor=source_root_path,
        label="local worker deployment manifest", allow_missing=False,
    )
    local_manifest = file_identity(local_manifest_path)
    if require_staged_path_exists:
        state_root = _existing_directory(Path(state_dir), "worker queue state root")
        staged_sources = _existing_directory(state_root / "sources", "staged source root")
    else:
        state_root = _validate_no_symlink_components(
            Path(state_dir), "published worker queue state root", allow_missing=True,
        )
        staged_sources = _validate_no_symlink_components(
            state_root / "sources", "published staged source root", allow_missing=True,
        )
    staged_manifest_path = _confined_path(
        staged_sources / study_commit / relative, anchor=staged_sources,
        label="staged worker deployment manifest",
        allow_missing=not require_staged_path_exists,
    )
    require(
        manifest.get("path") == str(staged_manifest_path)
        and manifest.get("bytes") == local_manifest["bytes"]
        and manifest.get("sha256") == local_manifest["sha256"],
        "worker deployment manifest is not the staged source blob",
    )
    candidates = [
        row for row in expected["records"]
        if row["manifest_relative_path"] == relative
        and row["item_index"] == deployment.get("item_index")
        and row["job_name"] == deployment.get("job_name")
        and row["container_name"] == deployment.get("container_name")
        and row["image"] == deployment.get("image")
        and row["gpu_count"] == deployment.get("gpu_count")
        and (
            value.get("pod_uid_source") != "downward_api_metadata_uid"
            or row["pod_uid_downward_api"] is True
        )
    ]
    require(
        len(candidates) == 1
        and isinstance(deployment.get("controller_pid"), int)
        and deployment["controller_pid"] > 0
        and isinstance(deployment.get("controller_argv_sha256"), str)
        and SHA256_RE.fullmatch(deployment["controller_argv_sha256"]) is not None,
        "worker attestation deployment is ambiguous or invalid",
    )
    process = value.get("process_inventory")
    require(isinstance(process, Mapping), "worker process inventory missing")
    exact_keys(
        process,
        {
            "inventory_complete", "visible", "allowed_ancestry_pids",
            "non_ancestry_processes", "current_pid",
            "current_process_is_attestation_only",
        },
        "worker process inventory",
    )
    visible = process.get("visible")
    ancestry = process.get("allowed_ancestry_pids")
    require(isinstance(visible, list) and isinstance(ancestry, list), "worker process inventory rows missing")
    process_rows: dict[int, Mapping[str, Any]] = {}
    for row in visible:
        require(isinstance(row, Mapping), "worker visible process row invalid")
        exact_keys(row, {"pid", "ppid", "argv_sha256"}, "worker visible process")
        pid = row.get("pid")
        require(
            type(pid) is int and pid > 0 and pid not in process_rows
            and type(row.get("ppid")) is int and row["ppid"] >= 0
            and isinstance(row.get("argv_sha256"), str)
            and SHA256_RE.fullmatch(row["argv_sha256"]) is not None,
            "worker visible process row changed",
        )
        process_rows[pid] = row
    current_pid = process.get("current_pid")
    controller_pid = deployment["controller_pid"]
    require(
        process.get("inventory_complete") is True
        and process.get("non_ancestry_processes") == []
        and process.get("current_process_is_attestation_only") is True
        and ancestry == sorted(process_rows)
        and type(current_pid) is int and current_pid in process_rows
        and controller_pid in process_rows
        and process_rows[current_pid]["ppid"] == controller_pid
        and process_rows[controller_pid]["argv_sha256"] == deployment["controller_argv_sha256"],
        "worker process/GPU attestation is not clean",
    )
    gpu = value.get("gpu_inventory")
    require(isinstance(gpu, Mapping), "worker GPU inventory missing")
    exact_keys(gpu, {"inventory_complete", "gpus", "compute_processes", "idle"}, "worker GPU inventory")
    gpu_rows = gpu.get("gpus")
    require(isinstance(gpu_rows, list), "worker GPU inventory rows missing")
    gpu_indices: set[int] = set()
    gpu_uuids: set[str] = set()
    for row in gpu_rows:
        require(isinstance(row, Mapping), "worker GPU row invalid")
        exact_keys(
            row,
            {
                "index", "uuid", "name", "memory_total_mib", "memory_used_mib",
                "utilization_percent",
            },
            "worker GPU row",
        )
        require(
            type(row.get("index")) is int and row["index"] >= 0
            and row["index"] not in gpu_indices
            and isinstance(row.get("uuid"), str) and bool(row["uuid"])
            and row["uuid"] not in gpu_uuids
            and isinstance(row.get("name"), str) and bool(row["name"])
            and type(row.get("memory_total_mib")) is int and row["memory_total_mib"] > 0
            and type(row.get("memory_used_mib")) is int
            and 0 <= row["memory_used_mib"] <= row["memory_total_mib"]
            and type(row.get("utilization_percent")) is int
            and 0 <= row["utilization_percent"] <= 100,
            "worker GPU row values invalid",
        )
        gpu_indices.add(row["index"])
        gpu_uuids.add(row["uuid"])
    require(
        gpu.get("inventory_complete") is True and gpu.get("idle") is True
        and gpu.get("compute_processes") == []
        and len(gpu_rows) == expected["gpu_count"]
        and value.get("worker_active_job_ids") == [job_id],
        "worker process/GPU attestation is not clean",
    )
    return {
        "worker_id": worker_id,
        "role": expected["role"],
        "gpu_count": expected["gpu_count"],
        "idle": True,
        "active_job_ids": [],
        "deployment_sha256": manifest["sha256"],
        "admission_deadline_utc": datetime.fromtimestamp(
            float(deadline), timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "pod_uid": value["pod_uid"],
        "hostname": value["hostname"],
        "observed_at_utc": value["observed_at_utc"],
        "consume_by_utc": value["consume_by_utc"],
    }


def validate_worker_attestation(
    value: Mapping[str, Any], *, descriptor_identity: Mapping[str, Any],
    source_root: Path, state_dir: Path, study_commit: str,
    queue_triplet_row: Mapping[str, Any],
) -> dict[str, Any]:
    as_of = parse_utc(utc_now(), "worker-attestation validation time")
    worker_id = safe_id(value.get("worker_id"), "attested worker")
    topology = deployment_topology(source_root)
    lane, attempt_number, scoped_worker_id = _attestation_identity(
        queue_triplet_row.get("job_id"), topology=topology,
    )
    require(scoped_worker_id == worker_id, "attestation worker/job identity changed")
    expected_jobs = _expected_attestation_jobs(
        source_root, study_commit, lane=lane, attempt_number=attempt_number,
        max_wall_seconds=queue_triplet_row["descriptor_value"]["max_wall_seconds"],
    )
    require(worker_id in expected_jobs, "attested worker is outside fixed topology")
    expected_raw = expected_jobs[worker_id]
    modules = load_modules(source_root)
    expected_normalized = modules["queue"].normalize_job(expected_raw)
    triplet = queue_triplet_row
    require(
        triplet["descriptor_value"] == expected_normalized
        and triplet["result_value"] is not None
        and triplet["result_value"]["status"] == "succeeded"
        and triplet["result_value"]["returncode"] == 0
        and value.get("queue_job_id") == triplet["job_id"]
        and value.get("queue_descriptor") == triplet["descriptor"]
        and value.get("queue_claim") == triplet["claim"]
        and value.get("claim_control") == triplet["claim_control"],
        "worker attestation is detached from its successful queue execution",
    )
    observed = parse_utc(value.get("observed_at_utc"), "worker attestation time")
    result = triplet["result_value"]
    require(
        parse_utc(result["started_at"], "attestation result start") <= observed
        <= parse_utc(result["ended_at"], "attestation result end")
        and 0 <= (as_of - observed).total_seconds()
        <= load_contract(source_root)["queue"]["maximum_attestation_age_seconds"],
        "worker attestation is stale or future-dated",
    )
    summary = _validate_worker_attestation_payload(
        value, source_root=source_root, state_dir=state_dir,
        study_commit=study_commit, expected_triplet=triplet, as_of=as_of,
        expected_semantic_active_job_ids=value["semantic_control"]["active_job_ids"],
    )
    summary["attestation"] = dict(descriptor_identity)
    return summary


def collect_worker_attestations(
    *, source_root: Path, state_dir: Path, study_commit: str,
    lane: str, attempt_number: int, job_ids: Sequence[str],
    all_jobs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = _expected_attestation_jobs(
        source_root, study_commit, lane=lane, attempt_number=attempt_number,
    )
    by_job_id = {row["job_id"]: worker_id for worker_id, row in expected.items()}
    selected_ids = list(job_ids)
    require(
        selected_ids == sorted(set(selected_ids))
        and 1 <= len(selected_ids) <= len(expected)
        and all(job_id in by_job_id for job_id in selected_ids),
        "selected worker attestation job set is not exact for the finalizer attempt",
    )
    workers: list[dict[str, Any]] = []
    controls: list[Mapping[str, Any]] = []
    state_root = _existing_directory(Path(state_dir), "attestation queue state root")
    jobs_root = _existing_directory(state_root / "jobs", "attestation queue jobs root")
    for job_id in selected_ids:
        require(job_id in all_jobs, f"worker attestation job missing: {job_id}")
        triplet = all_jobs[job_id]
        receipt_path = _confined_path(
            jobs_root / job_id / "publish" / "worker_attestation.json",
            anchor=jobs_root, label=f"worker attestation {job_id}",
            allow_missing=False,
        )
        identity, receipt = load_json_with_identity(
            receipt_path, f"worker attestation {job_id}",
        )
        worker = validate_worker_attestation(
            receipt, descriptor_identity=identity, source_root=source_root,
            state_dir=state_dir, study_commit=study_commit,
            queue_triplet_row=triplet,
        )
        workers.append(worker)
        controls.append(receipt["semantic_control"])
    require(
        len(workers) == len(selected_ids)
        and {row["worker_id"] for row in workers}
        == {by_job_id[job_id] for job_id in selected_ids},
        "selected worker attestation inventory is incomplete",
    )
    expected_active = selected_ids
    semantic_tuples = {
        (
            row.get("namespace"), row.get("control_commit"),
            row.get("admission_deadline_unix"), row.get("shutdown"),
            tuple(row.get("active_job_ids", [])),
        )
        for row in controls
    }
    require(
        len(semantic_tuples) == 1
        and next(iter(semantic_tuples))[-1] == tuple(expected_active),
        "worker attestations do not share one semantic attestation release",
    )
    return sorted(workers, key=lambda row: row["worker_id"])


def _semantic_control(control: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(control, CONTROL_KEYS, "finalizer live control")
    return {
        "namespace": control["namespace"],
        "control_commit": control["control_commit"],
        "admission_deadline_unix": control["admission_deadline_unix"],
        "shutdown": control["shutdown"],
        "active_job_ids": list(control["active_job_ids"]),
    }


def _require_attestation_wave_precedes_finalizer(
    *, source_root: Path, study_commit: str,
    all_jobs: Mapping[str, Mapping[str, Any]],
    finalizer_triplet: Mapping[str, Any],
) -> list[str]:
    finalizer_id = safe_id(finalizer_triplet.get("job_id"), "finalizer job")
    finalizer_match = FINALIZER_RE.fullmatch(finalizer_id)
    require(finalizer_match is not None, "finalizer ID is invalid")
    lane = finalizer_match.group(1).upper()
    attempt_number = int(finalizer_match.group(2))
    topology = deployment_topology(source_root)
    selected_ids = []
    selected_workers: set[str] = set()
    for job_id in sorted(all_jobs):
        match = ATTESTATION_RE.fullmatch(job_id)
        if match is None:
            continue
        job_lane, job_attempt, worker_id = _attestation_identity(
            job_id, topology=topology,
        )
        if (job_lane, job_attempt) != (lane, attempt_number):
            continue
        require(worker_id not in selected_workers, "attestation wave duplicates a fixed worker")
        selected_workers.add(worker_id)
        selected_ids.append(job_id)
    require(
        selected_ids,
        "finalizer inventory lacks its attempt-scoped worker attestation wave",
    )
    expected = _expected_attestation_jobs(
        source_root, study_commit, lane=lane, attempt_number=attempt_number,
    )
    queue = load_modules(source_root)["queue"]
    require(
        all_jobs.keys() >= set(selected_ids)
        and all(
            all_jobs[job_id]["descriptor_value"]
            == queue.normalize_job(
                expected[_attestation_identity(job_id, topology=topology)[2]]
            )
            for job_id in selected_ids
        ),
        "attempt-scoped attestation descriptor set changed",
    )
    finalizer_claim = finalizer_triplet["claim_value"]
    finalizer_commit = finalizer_claim["control_commit"]
    finalizer_generation = finalizer_claim["control_generation"]
    with _fetch_authorized_control_history(source_root=source_root) as snapshot:
        repository = Path(snapshot["repo"])
        for job_id in selected_ids:
            claim = all_jobs[job_id]["claim_value"]
            require(
                type(claim.get("control_generation")) is int
                and claim["control_generation"] < finalizer_generation,
                "worker attestation release does not precede the finalizer generation",
            )
            ancestry = _git(
                repository, "merge-base", "--is-ancestor",
                claim["control_commit"], finalizer_commit,
                accepted=(0, 1, 128),
            )
            require(
                ancestry.returncode == 0
                and claim["control_commit"] != finalizer_commit,
                "worker attestation control is not before the finalizer control",
            )
    return selected_ids


def _require_worker_shared_admission_deadline(
    *, workers: Sequence[Mapping[str, Any]], semantic_control: Mapping[str, Any],
) -> str:
    deadline = semantic_control.get("admission_deadline_unix")
    require(
        not isinstance(deadline, bool) and isinstance(deadline, (int, float))
        and math.isfinite(deadline),
        "finalizer shared admission deadline is invalid",
    )
    shared_deadline_utc = datetime.fromtimestamp(
        float(deadline), timezone.utc
    ).isoformat().replace("+00:00", "Z")
    require(
        bool(workers)
        and all(row.get("admission_deadline_utc") == shared_deadline_utc for row in workers),
        "worker attestations do not bind the finalizer shared admission deadline",
    )
    return shared_deadline_utc


def _authorized_capacity_workers(
    *, lane: str, workers: Sequence[Mapping[str, Any]], finalizer_worker: str,
    finalizer_job_id: str, d1_simulator_roles: Sequence[str],
) -> list[str]:
    """Select every directly attested N3 worker or one exact D1 pair."""

    by_id = {str(row.get("worker_id")): row for row in workers}
    require(finalizer_worker in by_id, "finalizer worker is absent from capacity inventory")
    own = by_id[finalizer_worker]
    if lane == "N3":
        require(own.get("role") == "n3" and own.get("gpu_count") == 2, "N3 finalizer capacity invalid")
        selected = sorted(
            worker_id for worker_id, row in by_id.items()
            if row.get("role") == "n3" and row.get("gpu_count") == 2
            and row.get("idle") is True
            and row.get("active_job_ids") in ([], [finalizer_job_id])
        )
        require(finalizer_worker in selected, "N3 finalizer is not fresh idle capacity")
        return selected
    require(lane == "D1", "capacity lane invalid")
    require(own.get("role") == "d1" and own.get("gpu_count") == 2, "D1 server finalizer capacity invalid")
    simulators = [
        row for row in workers
        if row.get("role") in d1_simulator_roles and row.get("gpu_count") == 1
        and row.get("active_job_ids") == [] and row.get("idle") is True
    ]
    require(simulators, "D1 release lacks a fresh simulator-worker attestation")
    selected_simulator = min(
        simulators, key=lambda row: d1_simulator_roles.index(str(row["role"])),
    )
    return [finalizer_worker, str(selected_simulator["worker_id"])]


def build_pending_reconciliation(
    *, source_root: Path, state_dir: Path, study_commit: str, lane: str,
    finalizer_triplet: Mapping[str, Any], control: Mapping[str, Any],
    all_jobs: Mapping[str, Mapping[str, Any]], ledger_identity: Mapping[str, Any],
    ledger_validation: Mapping[str, Any], workers: Sequence[Mapping[str, Any]],
    finalizer_attestation_identity: Mapping[str, Any],
    finalizer_attestation: Mapping[str, Any], observed_results_ancestor_commit: str,
    controller_lock_probes: Sequence[Mapping[str, Any]],
    d1_global_lock_probe: Mapping[str, Any],
) -> dict[str, Any]:
    contract = load_contract(source_root)
    state_root = _validate_no_symlink_components(
        Path(state_dir), "pending reconciliation queue state root", allow_missing=True,
    )
    created_at_utc = utc_now()
    as_of = parse_utc(created_at_utc, "pending reconciliation time")
    finalizer_id = finalizer_triplet["job_id"]
    require(FINALIZER_RE.fullmatch(finalizer_id) is not None, "finalizer ID invalid")
    require(finalizer_triplet["result"] is None, "finalizer already has outer result")
    claim = finalizer_triplet["claim_value"]
    semantic = _semantic_control(control)
    generation = control.get("control_generation")
    require(
        semantic == {
            "namespace": NAMESPACE,
            "control_commit": claim["control_commit"],
            "admission_deadline_unix": control["admission_deadline_unix"],
            "shutdown": False,
            "active_job_ids": [finalizer_id],
        }
        and type(generation) is int
        and generation >= claim["control_generation"]
        and generation >= contract["queue"]["minimum_control_generation"]
        and generation != contract["queue"]["forbidden_control_generation"],
        "finalizer semantic control/generation is not releasable",
    )
    deadline = control["admission_deadline_unix"]
    require(
        not isinstance(deadline, bool) and isinstance(deadline, (int, float))
        and math.isfinite(deadline),
        "shared admission deadline invalid",
    )
    remaining = math.floor(float(deadline) - as_of.timestamp())
    require(
        remaining >= contract["queue"]["maximum_job_wall_seconds"]
        + contract["queue"]["publication_grace_seconds"],
        "shared admission wall is insufficient for the next scientific job",
    )
    require(COMMIT_RE.fullmatch(observed_results_ancestor_commit) is not None, "results ancestor invalid")
    all_prior = sorted(all_jobs)
    terminal_ids = sorted(job_id for job_id, row in all_jobs.items() if row["result"] is not None)
    require(
        terminal_ids == sorted(set(all_prior) - {finalizer_id}),
        "queue inventory contains another nonterminal job",
    )
    finalizer_worker = finalizer_attestation.get("worker_id")
    topology = deployment_topology(source_root)
    expected_role = contract["producer"]["finalizer_roles"][lane]
    require(
        finalizer_worker in topology
        and topology[finalizer_worker]["role"] == expected_role
        and finalizer_attestation.get("gpu_inventory", {}).get("idle") is True
        and finalizer_attestation.get("gpu_inventory", {}).get("compute_processes") == [],
        "lane finalizer did not attest its own required GPU worker",
    )
    worker_rows = [dict(row) for row in workers]
    matches = [index for index, row in enumerate(worker_rows) if row["worker_id"] == finalizer_worker]
    require(len(matches) == 1, "finalizer worker is absent/duplicated in complete inventory")
    own = worker_rows[matches[0]]
    own.update({
        "idle": True,
        "active_job_ids": [finalizer_id],
        "attestation": dict(finalizer_attestation_identity),
        "pod_uid": finalizer_attestation["pod_uid"],
        "hostname": finalizer_attestation["hostname"],
        "observed_at_utc": finalizer_attestation["observed_at_utc"],
        "consume_by_utc": finalizer_attestation["consume_by_utc"],
    })
    authorized_capacity = _authorized_capacity_workers(
        lane=lane, workers=worker_rows, finalizer_worker=str(finalizer_worker),
        finalizer_job_id=finalizer_id,
        d1_simulator_roles=contract["producer"]["d1_required_simulator_roles"],
    )
    attempt_ids = sorted(ledger_validation["attempt_ids"])
    ledger_jobs = sorted(ledger_validation["job_ids"])
    reconciliation = signed_document({
        "schema_version": RECONCILIATION_SCHEMA,
        "status": "pending_finalizer_publication",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "confirmation_release": True,
        "created_at_utc": created_at_utc,
        "consume_by_utc": min(
            [utc_after(created_at_utc, MAX_EVIDENCE_AGE_SECONDS)]
            + [str(row["consume_by_utc"]) for row in worker_rows],
            key=lambda item: parse_utc(item, "worker capacity consume-by"),
        ),
        "observed_control_generation_lower_bound": claim["control_generation"],
        "observed_control_generation": generation,
        "control_semantics": semantic,
        "queue_release_lock": {
            "path": str(state_root / "release.lock"),
            "kind": "kernel_advisory_flock",
            "held_by_finalizer": True,
            "legacy_directory_present": False,
            "sole_active_queue_job_id": finalizer_id,
            "controller_lock_probes": [dict(row) for row in controller_lock_probes],
            "d1_global_lock_probe": dict(d1_global_lock_probe),
        },
        "prior_inventory": {
            "inventory_complete": True,
            "ledger_attempt_ids": attempt_ids,
            "ledger_job_ids": ledger_jobs,
            "all_prior_job_ids": all_prior,
            "claim_job_ids": all_prior,
            "terminal_result_job_ids": terminal_ids,
            "pending_outer_result_job_ids": [finalizer_id],
            "semantic_active_job_ids": [finalizer_id],
            "unresolved_claim_job_ids": [],
            "stale_claim_job_ids": [],
            "orphan_job_ids": [],
        },
        "process_cleanup": {
            "inventory_complete": True,
            "active_owned_processes": [os.getpid()],
            "allowed_finalizer_process_id": os.getpid(),
            "active_descendants": [],
            "unreaped_children": [],
            "cleanup_errors": [],
            "post_exit_requires_outer_child_reaped": True,
        },
        "task_gpu_state": {
            "inventory_complete": True,
            "fixed_deployment_worker_count": len(topology),
            "fixed_deployment_worker_ids": sorted(topology),
            "deployment_inventory_is_scientific_sample_size": False,
            "selected_attestation_worker_ids": sorted(
                row["worker_id"] for row in worker_rows
            ),
            "workers": worker_rows,
            "lane": lane,
            "authorized_post_exit_capacity_worker_ids": authorized_capacity,
            "finalizer_uses_no_gpu": True,
            "post_exit_capacity_requires_terminal_reaped_outer_result": True,
        },
        "results_publication": {
            "status": "pending_cluster_coordinator_publication",
            "round_trip_verified": False,
            "observed_results_ancestor_commit": observed_results_ancestor_commit,
            "ledger": dict(ledger_identity),
            "expected_finalizer_job_id": finalizer_id,
        },
        "shared_admission": {
            "deadline_utc": datetime.fromtimestamp(float(deadline), timezone.utc).isoformat().replace("+00:00", "Z"),
            "remaining_wall_seconds_at_observation": remaining,
            "maximum_job_wall_seconds": contract["queue"]["maximum_job_wall_seconds"],
            "publication_grace_seconds": contract["queue"]["publication_grace_seconds"],
            "generation_is_lower_bound": True,
        },
        "claim_boundary": (
            "A running lane-specific finalizer is the sole semantic active job. Its own CPU-only "
            "process and GPU-idle receipt are explicit; release remains pending until Git H1 proves "
            "the outer queue child terminal and reaped."
        ),
    })
    validate_pending_reconciliation(
        reconciliation, source_root=source_root, study_commit=study_commit,
        ledger_identity=ledger_identity, finalizer_triplet=finalizer_triplet,
        state_dir=state_dir,
    )
    return reconciliation


def validate_pending_reconciliation(
    value: Mapping[str, Any], *, source_root: Path, study_commit: str,
    ledger_identity: Mapping[str, Any], finalizer_triplet: Mapping[str, Any],
    state_dir: Path,
) -> dict[str, Any]:
    """Replay every live/PVC attestation behind a pending reconciliation."""

    exact_keys(
        value,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "confirmation_release", "created_at_utc", "consume_by_utc",
            "observed_control_generation_lower_bound", "observed_control_generation",
            "control_semantics", "queue_release_lock", "prior_inventory",
            "process_cleanup", "task_gpu_state", "results_publication",
            "shared_admission", "claim_boundary",
            "payload_sha256",
        },
        "pending cluster reconciliation",
    )
    verify_signed_document(value, "pending cluster reconciliation")
    require(
        value.get("schema_version") == RECONCILIATION_SCHEMA
        and value.get("status") == "pending_finalizer_publication"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit
        and value.get("confirmation_release") is True,
        "pending reconciliation identity changed",
    )
    validation_time_utc = utc_now()
    as_of = parse_utc(validation_time_utc, "pending reconciliation validation time")
    created = parse_utc(value.get("created_at_utc"), "pending reconciliation creation time")
    consume_by = parse_utc(value.get("consume_by_utc"), "pending reconciliation consume-by")
    require(
        created <= as_of <= consume_by
        and (as_of - created).total_seconds() <= MAX_EVIDENCE_AGE_SECONDS,
        "pending reconciliation is stale, future-dated, or expired",
    )
    contract = load_contract(source_root)
    state_root = _existing_directory(Path(state_dir), "pending reconciliation queue state root")
    trusted_state_root = _lexical_absolute_path(
        Path(contract["queue"]["state_dir"]), "configured trusted PVC state root",
    )
    require(
        state_root == trusted_state_root,
        "pending reconciliation state directory is not trusted PVC path",
    )
    generation = value.get("observed_control_generation")
    lower = value.get("observed_control_generation_lower_bound")
    require(
        type(generation) is int and type(lower) is int and generation >= lower
        and lower == finalizer_triplet["claim_value"]["control_generation"]
        and generation >= contract["queue"]["minimum_control_generation"]
        and generation != contract["queue"]["forbidden_control_generation"],
        "pending reconciliation generation invalid",
    )
    semantic = value.get("control_semantics")
    finalizer_id = finalizer_triplet["job_id"]
    current_control_path = _confined_path(
        state_root / "control.json", anchor=state_root,
        label="current finalizer control", allow_missing=False,
    )
    current_control = load_json(current_control_path, "current finalizer control")
    require(
        isinstance(semantic, Mapping)
        and dict(semantic) == _semantic_control(current_control)
        and semantic.get("namespace") == NAMESPACE
        and semantic.get("control_commit") == finalizer_triplet["claim_value"]["control_commit"]
        and semantic.get("shutdown") is False
        and semantic.get("active_job_ids") == [finalizer_id],
        "pending reconciliation semantic control changed",
    )
    all_jobs, _attempts = scan_queue_state(
        source_root=source_root, state_dir=state_dir, study_commit=study_commit,
        running_finalizer_job_id=finalizer_id,
    )
    selected_attestation_job_ids = _require_attestation_wave_precedes_finalizer(
        source_root=source_root,
        study_commit=study_commit,
        all_jobs=all_jobs,
        finalizer_triplet=finalizer_triplet,
    )
    finalizer_match = FINALIZER_RE.fullmatch(finalizer_id)
    require(finalizer_match is not None, "pending finalizer ID invalid")
    selected_wave_workers = collect_worker_attestations(
        source_root=source_root,
        state_dir=state_dir,
        study_commit=study_commit,
        lane=finalizer_match.group(1).upper(),
        attempt_number=int(finalizer_match.group(2)),
        job_ids=selected_attestation_job_ids,
        all_jobs=all_jobs,
    )
    inventory = value.get("prior_inventory")
    require(isinstance(inventory, Mapping), "pending prior inventory missing")
    exact_keys(
        inventory,
        {
            "inventory_complete", "ledger_attempt_ids", "ledger_job_ids",
            "all_prior_job_ids", "claim_job_ids", "terminal_result_job_ids",
            "pending_outer_result_job_ids", "semantic_active_job_ids",
            "unresolved_claim_job_ids", "stale_claim_job_ids", "orphan_job_ids",
        },
        "pending prior inventory",
    )
    all_prior = sorted(all_jobs)
    terminal = sorted(job_id for job_id, row in all_jobs.items() if row["result"] is not None)
    observed_ledger_identity, ledger_value = load_json_with_identity(
        Path(str(ledger_identity["path"])), "pending ledger",
    )
    require(observed_ledger_identity == dict(ledger_identity), "pending ledger changed")
    ledger_validation = validate_result_attempt_ledger(
        ledger_value, source_root=source_root, state_dir=state_dir,
        study_commit=study_commit, all_queue_jobs=all_jobs,
    )
    require(
        inventory.get("inventory_complete") is True
        and inventory.get("ledger_attempt_ids") == sorted(ledger_validation["attempt_ids"])
        and inventory.get("ledger_job_ids") == sorted(ledger_validation["job_ids"])
        and inventory.get("all_prior_job_ids") == all_prior
        and inventory.get("claim_job_ids") == all_prior
        and inventory.get("terminal_result_job_ids") == terminal
        and inventory.get("pending_outer_result_job_ids") == [finalizer_id]
        and inventory.get("semantic_active_job_ids") == [finalizer_id]
        and inventory.get("unresolved_claim_job_ids") == []
        and inventory.get("stale_claim_job_ids") == []
        and inventory.get("orphan_job_ids") == []
        and terminal == sorted(set(all_prior) - {finalizer_id}),
        "pending queue/PVC inventory does not close exactly",
    )
    lock = value.get("queue_release_lock")
    require(isinstance(lock, Mapping), "pending release-lock evidence missing")
    exact_keys(
        lock,
        {
            "path", "kind", "held_by_finalizer", "legacy_directory_present",
            "sole_active_queue_job_id", "controller_lock_probes", "d1_global_lock_probe",
        },
        "pending release lock",
    )
    require(
        lock.get("path") == str(state_root / "release.lock")
        and lock.get("kind") == "kernel_advisory_flock"
        and lock.get("held_by_finalizer") is True
        and lock.get("legacy_directory_present") is False
        and lock.get("sole_active_queue_job_id") == finalizer_id,
        "pending release lock identity changed",
    )
    expected_probes = _probe_controller_locks(
        state_dir, sorted(row["worker_id"] for row in selected_wave_workers),
    )
    require(lock.get("controller_lock_probes") == expected_probes, "controller lock liveness changed")
    d1_probe = lock.get("d1_global_lock_probe")
    require(
        isinstance(d1_probe, Mapping)
        and (
            d1_probe == {"path": None, "observed": "not_required"}
            or (
                d1_probe.get("observed") == "acquired_by_finalizer"
                and isinstance(d1_probe.get("path"), str)
                and Path(d1_probe["path"]).name == "d1-global-server.lock"
            )
        ),
        "D1 global lock proof changed",
    )
    process = value.get("process_cleanup")
    require(isinstance(process, Mapping), "pending process cleanup missing")
    exact_keys(
        process,
        {
            "inventory_complete", "active_owned_processes", "allowed_finalizer_process_id",
            "active_descendants", "unreaped_children", "cleanup_errors",
            "post_exit_requires_outer_child_reaped",
        },
        "pending process cleanup",
    )
    require(
        process.get("inventory_complete") is True
        and process.get("active_owned_processes") == [os.getpid()]
        and process.get("allowed_finalizer_process_id") == os.getpid()
        and process.get("active_descendants") == []
        and process.get("unreaped_children") == []
        and process.get("cleanup_errors") == []
        and process.get("post_exit_requires_outer_child_reaped") is True,
        "pending process cleanup is caller-authored or incomplete",
    )
    publication = value.get("results_publication")
    exact_keys(
        publication,
        {
            "status", "round_trip_verified", "observed_results_ancestor_commit",
            "ledger", "expected_finalizer_job_id",
        },
        "pending results publication",
    )
    require(
        isinstance(publication, Mapping)
        and publication.get("status") == "pending_cluster_coordinator_publication"
        and publication.get("round_trip_verified") is False
        and publication.get("ledger") == dict(ledger_identity)
        and publication.get("expected_finalizer_job_id") == finalizer_id
        and isinstance(publication.get("observed_results_ancestor_commit"), str)
        and COMMIT_RE.fullmatch(publication["observed_results_ancestor_commit"]) is not None,
        "pending reconciliation publication boundary changed",
    )
    gpu = value.get("task_gpu_state")
    require(isinstance(gpu, Mapping), "pending GPU state missing")
    exact_keys(
        gpu,
        {
            "inventory_complete", "fixed_deployment_worker_count",
            "fixed_deployment_worker_ids",
            "deployment_inventory_is_scientific_sample_size",
            "selected_attestation_worker_ids", "workers", "lane",
            "authorized_post_exit_capacity_worker_ids", "finalizer_uses_no_gpu",
            "post_exit_capacity_requires_terminal_reaped_outer_result",
        },
        "pending GPU state",
    )
    workers = gpu.get("workers")
    topology = deployment_topology(source_root)
    selected_wave_worker_ids = sorted(row["worker_id"] for row in selected_wave_workers)
    require(
        gpu.get("inventory_complete") is True and isinstance(workers, list)
        and 1 <= len(workers) <= len(topology)
        and gpu.get("fixed_deployment_worker_count") == len(topology) == 32
        and gpu.get("fixed_deployment_worker_ids") == sorted(topology)
        and gpu.get("deployment_inventory_is_scientific_sample_size") is False
        and gpu.get("selected_attestation_worker_ids") == selected_wave_worker_ids
        and gpu.get("finalizer_uses_no_gpu") is True
        and gpu.get("post_exit_capacity_requires_terminal_reaped_outer_result") is True,
        "pending worker inventory incomplete",
    )
    capacity_ids = gpu.get("authorized_post_exit_capacity_worker_ids")
    require(isinstance(capacity_ids, list) and capacity_ids, "pending release capacity missing")
    seen_workers: set[str] = set()
    finalizer_worker = finalizer_triplet["claim_value"]["worker_id"]
    validated_workers: list[dict[str, Any]] = []
    for worker in workers:
        require(isinstance(worker, Mapping), "pending worker row invalid")
        exact_keys(
            worker,
            {
                "worker_id", "role", "gpu_count", "idle", "active_job_ids",
                "deployment_sha256", "admission_deadline_utc", "attestation",
                "pod_uid", "hostname", "observed_at_utc", "consume_by_utc",
            },
            "pending worker row",
        )
        worker_id = safe_id(worker.get("worker_id"), "pending worker")
        require(worker_id not in seen_workers and worker_id in topology, "pending worker duplicate/unknown")
        seen_workers.add(worker_id)
        attestation_identity, attestation = verify_descriptor(worker.get("attestation"), f"pending worker attestation {worker_id}")
        if worker_id == finalizer_worker:
            validated = _validate_worker_attestation_payload(
                attestation, source_root=source_root, state_dir=state_dir,
                study_commit=study_commit, expected_triplet=finalizer_triplet,
                as_of=as_of, expected_semantic_active_job_ids=[finalizer_id],
            )
            validated["active_job_ids"] = [finalizer_id]
            validated["attestation"] = dict(attestation_identity)
            require(
                validated == dict(worker),
                "running finalizer worker attestation changed",
            )
        else:
            attestation_job_id = attestation.get("queue_job_id")
            require(attestation_job_id in all_jobs, "worker attestation queue job absent")
            validated = validate_worker_attestation(
                attestation, descriptor_identity=attestation_identity,
                source_root=source_root, state_dir=state_dir,
                study_commit=study_commit,
                queue_triplet_row=all_jobs[attestation_job_id],
            )
            require(validated == dict(worker), "pending worker row differs from native attestation")
        require(
            parse_utc(worker["observed_at_utc"], "pending worker time") <= as_of
            and worker["role"] == topology[worker_id]["role"]
            and worker["gpu_count"] == topology[worker_id]["gpu_count"]
            and worker["idle"] is True,
            "pending worker topology/time changed",
        )
        validated_workers.append(validated)
    require(
        seen_workers == set(selected_wave_worker_ids),
        "pending selected worker inventory is not exact",
    )
    _require_worker_shared_admission_deadline(
        workers=workers,
        semantic_control=semantic,
    )
    expected_consume_by = min(
        [utc_after(value["created_at_utc"], MAX_EVIDENCE_AGE_SECONDS)]
        + [str(row["consume_by_utc"]) for row in workers],
        key=lambda item: parse_utc(item, "pending worker consume-by"),
    )
    require(
        value.get("consume_by_utc") == expected_consume_by and as_of <= consume_by,
        "pending reconciliation capacity authority is expired or detached from workers",
    )
    lane = gpu.get("lane")
    by_id = {row["worker_id"]: row for row in workers}
    require(finalizer_worker in capacity_ids and finalizer_worker in by_id, "finalizer capacity ID missing")
    expected_capacity = _authorized_capacity_workers(
        lane=str(lane), workers=workers, finalizer_worker=finalizer_worker,
        finalizer_job_id=finalizer_id,
        d1_simulator_roles=contract["producer"]["d1_required_simulator_roles"],
    )
    require(capacity_ids == expected_capacity, "pending authorized capacity selection changed")
    if lane == "N3":
        require(
            by_id[finalizer_worker]["role"] == "n3"
            and by_id[finalizer_worker]["gpu_count"] == 2,
            "N3 pending capacity is not the attesting two-GPU worker",
        )
    elif lane == "D1":
        require(
            len(capacity_ids) == 2 and by_id[finalizer_worker]["role"] == "d1"
            and by_id[finalizer_worker]["gpu_count"] == 2,
            "D1 pending server capacity invalid",
        )
        simulator = [by_id[item] for item in capacity_ids if item != finalizer_worker]
        require(
            len(simulator) == 1 and simulator[0]["role"] in contract["producer"]["d1_required_simulator_roles"]
            and simulator[0]["gpu_count"] == 1 and simulator[0]["active_job_ids"] == [],
            "D1 pending simulator capacity invalid",
        )
    else:
        raise ConfirmationReleaseEvidenceError("pending lane invalid")
    admission = value.get("shared_admission")
    require(isinstance(admission, Mapping), "pending shared admission missing")
    exact_keys(
        admission,
        {
            "deadline_utc", "remaining_wall_seconds_at_observation",
            "maximum_job_wall_seconds", "publication_grace_seconds",
            "generation_is_lower_bound",
        },
        "pending shared admission",
    )
    deadline = parse_utc(admission.get("deadline_utc"), "pending admission deadline")
    observed_remaining = math.floor((deadline - created).total_seconds())
    current_remaining = math.floor((deadline - as_of).total_seconds())
    require(
        admission.get("remaining_wall_seconds_at_observation") == observed_remaining
        and admission.get("maximum_job_wall_seconds") == contract["queue"]["maximum_job_wall_seconds"]
        and admission.get("publication_grace_seconds") == contract["queue"]["publication_grace_seconds"]
        and admission.get("generation_is_lower_bound") is True
        and abs(deadline.timestamp() - float(semantic["admission_deadline_unix"])) < 1e-6
        and observed_remaining >= contract["queue"]["maximum_job_wall_seconds"] + contract["queue"]["publication_grace_seconds"]
        and current_remaining >= contract["queue"]["maximum_job_wall_seconds"] + contract["queue"]["publication_grace_seconds"],
        "pending shared admission is stale/insufficient/contradictory",
    )
    post_exit = []
    for worker in workers:
        row = dict(worker)
        if row["worker_id"] in capacity_ids:
            row["active_job_ids"] = []
        post_exit.append(row)
    return {
        "document": dict(value),
        "post_exit_workers": post_exit,
        "control_semantics": dict(semantic),
        "all_prior_job_ids": list(value["prior_inventory"]["all_prior_job_ids"]),
        "authorized_capacity_worker_ids": list(capacity_ids),
        "ledger_validation": ledger_validation,
        "all_queue_jobs": all_jobs,
    }


def _configured_results_remote(
    *, source_root: Path, alias: str, contract: Mapping[str, Any]
) -> str:
    safe_id(alias, "results remote alias")
    configured = _git(
        source_root, "config", "--get-all", f"remote.{alias}.url", text=True,
        accepted=(0, 1),
    ).stdout.splitlines()
    resolved = _git(
        source_root, "remote", "get-url", "--all", alias, text=True,
        accepted=(0, 2),
    ).stdout.splitlines()
    allowed = contract["queue"]["allowed_results_urls"]
    require(
        len(configured) == len(resolved) == 1
        and configured[0] == resolved[0] and configured[0] in allowed,
        "results remote is not the authorized repository",
    )
    return configured[0]


def _initialize_bare_repository(path: Path, label: str) -> None:
    initialized = subprocess.run(
        [
            _trusted_git_executable(), "init", "--bare", "--quiet",
            "--template=", str(path),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=30, check=False,
        env=_safe_git_environment(),
    )
    require(initialized.returncode == 0, f"temporary {label} repository initialization failed")
    _write_isolated_git_config(path)
    _validate_isolated_git_repository(path)


def _isolated_git_config_bytes(promisor_url: str | None = None) -> bytes:
    if promisor_url is None:
        return ISOLATED_GIT_CONFIG
    require(
        promisor_url == TRUSTED_FETCH_URL
        and "\n" not in promisor_url and '"' not in promisor_url,
        "isolated promisor URL is not the code-pinned transport",
    )
    return (
        b"[core]\n\trepositoryformatversion = 1\n\tbare = true\n"
        b"[remote \"wmf-trusted-promisor\"]\n\turl = "
        + promisor_url.encode("utf-8")
        + b"\n\tpromisor = true\n\tpartialclonefilter = blob:limit=1048576\n"
    )


def _write_isolated_git_config(
    repository: Path, *, promisor_url: str | None = None,
) -> None:
    root = _existing_directory(Path(repository), "isolated Git repository")
    config = root / "config"
    _root, root_descriptors, root_records = _open_directory_chain(
        root, "isolated Git repository",
    )
    root_descriptor = root_descriptors[-1]
    try:
        before = os.stat("config", dir_fd=root_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        before = None
    except OSError as error:
        _close_descriptors(root_descriptors)
        raise ConfirmationReleaseEvidenceError("isolated Git config is unreadable") from error
    if before is not None:
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            "isolated Git config is not a singly linked regular file",
        )
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    if before is None:
        flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    created = False
    try:
        descriptor = os.open("config", flags, 0o600, dir_fd=root_descriptor)
        opened = os.fstat(descriptor)
        created = before is None
        require(
            stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1
            and (before is None or (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino)),
            "isolated Git config inode changed before open",
        )
        at_path = _entry_metadata(root_descriptor, "config", "isolated Git config")
        require(
            (at_path.st_dev, at_path.st_ino) == (opened.st_dev, opened.st_ino),
            "isolated Git config pathname changed before write",
        )
        _verify_directory_chain(root, root_records, "isolated Git repository")
        payload = _isolated_git_config_bytes(promisor_url)
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            require(written > 0, "isolated Git config write made no progress")
            offset += written
        os.fsync(descriptor)
        completed = os.fstat(descriptor)
        require(
            (completed.st_dev, completed.st_ino) == (opened.st_dev, opened.st_ino)
            and completed.st_size == len(payload),
            "isolated Git config changed while writing",
        )
        at_path = _entry_metadata(root_descriptor, "config", "isolated Git config")
        require(
            (at_path.st_dev, at_path.st_ino) == (opened.st_dev, opened.st_ino),
            "isolated Git config pathname changed after write",
        )
        _verify_directory_chain(root, root_records, "isolated Git repository")
    except ConfirmationReleaseEvidenceError:
        if created:
            try:
                at_path = os.stat(
                    "config", dir_fd=root_descriptor, follow_symlinks=False,
                )
                opened_metadata = os.fstat(descriptor) if descriptor >= 0 else None
                if (
                    opened_metadata is not None
                    and (at_path.st_dev, at_path.st_ino)
                    == (opened_metadata.st_dev, opened_metadata.st_ino)
                ):
                    os.unlink("config", dir_fd=root_descriptor)
            except OSError:
                pass
        raise
    except OSError as error:
        raise ConfirmationReleaseEvidenceError("isolated Git config cannot be fixed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _close_descriptors(root_descriptors)


def _validate_isolated_git_repository(
    repository: Path, *, promisor_url: str | None = None,
) -> int:
    root = _existing_directory(Path(repository), "isolated Git repository")
    config = _confined_path(
        root / "config", anchor=root, label="isolated Git config", allow_missing=False,
    )
    config_payload, _digest, _metadata, _candidate = _read_regular_file(
        config, "isolated Git config", capture_payload=True,
    )
    require(
        config_payload == _isolated_git_config_bytes(promisor_url),
        "isolated Git repository config changed",
    )
    inventory = _regular_inventory([root], anchor=root)
    total = sum(int(row["bytes"]) for row in inventory)
    require(
        total <= MAX_AUTHENTICATED_GRAPH_BYTES,
        "isolated authenticated graph exceeds byte limit",
    )
    return total


def _remote_head(_repo: Path, remote_url: str, ref: str) -> str:
    # Never let source-repository config rewrite the literal trusted endpoint.
    with tempfile.TemporaryDirectory(prefix="wmf-confirmation-remote-head-") as temporary:
        bare = Path(temporary) / "observation.git"
        _initialize_bare_repository(bare, "remote observation")
        rows = _git(
            bare, "ls-remote", "--heads", remote_url, ref, text=True,
        ).stdout.splitlines()
        _validate_isolated_git_repository(bare)
    require(len(rows) == 1, "results remote head is missing or ambiguous")
    fields = rows[0].split()
    require(
        len(fields) == 2 and COMMIT_RE.fullmatch(fields[0]) is not None and fields[1] == ref,
        "results remote head response invalid",
    )
    return fields[0]


def _single_authorized_remote(
    *, source_root: Path, contract: Mapping[str, Any], label: str,
) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for alias in ("publish", "origin"):
        try:
            url = _configured_results_remote(
                source_root=source_root, alias=alias, contract=contract,
            )
        except ConfirmationReleaseEvidenceError:
            continue
        candidates.append((alias, url))
    require(len(candidates) == 1, f"authorized {label} remote alias is missing or ambiguous")
    return candidates[0]


@contextmanager
def _fetch_authorized_control_history(*, source_root: Path) -> Iterator[dict[str, Any]]:
    """Fetch the exact GitHub control ref without trusting local-only objects."""

    contract = load_contract(source_root)
    alias, _reported_url = _single_authorized_remote(
        source_root=source_root, contract=contract, label="control",
    )
    remote_url = contract["queue"]["trusted_fetch_url"]
    require(
        isinstance(remote_url, str) and remote_url in contract["queue"]["allowed_results_urls"],
        "trusted control fetch URL is not source-pinned",
    )
    control_ref = contract["queue"]["control_ref"]
    before = _remote_head(source_root, remote_url, control_ref)
    with tempfile.TemporaryDirectory(prefix="wmf-confirmation-control-auth-") as temporary:
        bare = Path(temporary) / "control.git"
        _initialize_bare_repository(bare, "control")
        _git(
            bare, "fetch", "--no-tags", "--no-recurse-submodules",
            "--filter=blob:limit=1048576", remote_url,
            f"{control_ref}:refs/heads/verified-control", timeout=120,
        )
        # Filtered fetches may install promisor transport state. Required
        # source/queue blobs are below the bound, so erase all fetch-created
        # config before reading objects; a missing blob must fail locally.
        _write_isolated_git_config(bare)
        _validate_isolated_git_repository(bare)
        fetched = _git(
            bare, "rev-parse", "refs/heads/verified-control^{commit}", text=True,
        ).stdout.strip()
        require(fetched == before, "authorized control ref moved during fetch")
        yield {
            "repo": str(bare), "head": before, "ref": control_ref,
            "remote_alias": alias, "remote_url": remote_url,
        }
        _validate_isolated_git_repository(bare)
    require(
        _remote_head(source_root, remote_url, control_ref) == before,
        "authorized control ref moved during validation",
    )


def observe_results_ancestor(
    *, source_root: Path, remote_alias: str, results_ref: str,
) -> tuple[str, str]:
    contract = load_contract(source_root)
    require(results_ref == contract["queue"]["results_ref"], "results ref changed")
    _configured_results_remote(
        source_root=source_root, alias=remote_alias, contract=contract,
    )
    url = contract["queue"]["trusted_fetch_url"]
    require(
        isinstance(url, str) and url in contract["queue"]["allowed_results_urls"],
        "trusted results fetch URL is not source-pinned",
    )
    return url, _remote_head(source_root, url, results_ref)


@contextmanager
def _validated_lock_handle(
    path: Path, *, allow_canonical_release_creation: bool = False,
    state_dir: Path | None = None,
) -> Iterator[Any]:
    """Open a regular lock inode without following or mutating special files."""

    lock = _lexical_absolute_path(Path(path), "lock path")
    require(lock.name not in {"", ".", ".."}, "lock path is invalid")
    parent, parent_descriptors, parent_records = _open_directory_chain(
        lock.parent, "lock parent",
    )
    parent_descriptor = parent_descriptors[-1]
    if allow_canonical_release_creation:
        trusted_state = _existing_directory(Path(state_dir), "queue state root") if state_dir is not None else None
        require(
            trusted_state is not None and lock == trusted_state / "release.lock",
            "only the canonical queue release lock may be created",
        )
    try:
        before = os.stat(
            lock.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError:
        if not allow_canonical_release_creation:
            _close_descriptors(parent_descriptors)
            raise ConfirmationReleaseEvidenceError(
                f"required pre-existing lock is missing: {lock}"
            )
        before = None
    except OSError as error:
        _close_descriptors(parent_descriptors)
        raise ConfirmationReleaseEvidenceError(f"lock inode is unreadable: {lock}") from error
    if before is not None:
        if not (stat.S_ISREG(before.st_mode) and before.st_nlink == 1):
            _close_descriptors(parent_descriptors)
            raise ConfirmationReleaseEvidenceError(
                f"lock inode is not a singly linked regular file: {lock}"
            )
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if allow_canonical_release_creation and before is None:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = -1
    opened: os.stat_result | None = None
    created_here = False
    ready_to_yield = False
    try:
        descriptor = os.open(
            lock.name, flags, 0o644, dir_fd=parent_descriptor,
        )
        created_here = before is None
    except OSError as error:
        _close_descriptors(parent_descriptors)
        raise ConfirmationReleaseEvidenceError(f"lock file open failed safely: {lock}") from error
    try:
        opened = os.fstat(descriptor)
        at_path = _entry_metadata(parent_descriptor, lock.name, "lock inode")
        require(
            stat.S_ISREG(opened.st_mode)
            and opened.st_nlink == 1
            and stat.S_ISREG(at_path.st_mode)
            and at_path.st_nlink == 1
            and (
                before is None
                or (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino)
            )
            and (opened.st_dev, opened.st_ino) == (at_path.st_dev, at_path.st_ino),
            f"lock inode changed or is non-regular: {lock}",
        )
        _verify_directory_chain(lock.parent, parent_records, "lock parent")
        ready_to_yield = True
        with os.fdopen(descriptor, "r+b", buffering=0) as handle:
            descriptor = -1
            yield handle
    finally:
        if created_here and not ready_to_yield and opened is not None:
            try:
                at_parent = os.stat(
                    lock.name, dir_fd=parent_descriptor, follow_symlinks=False,
                )
                if (at_parent.st_dev, at_parent.st_ino) == (opened.st_dev, opened.st_ino):
                    os.unlink(lock.name, dir_fd=parent_descriptor)
            except OSError:
                pass
        if descriptor >= 0:
            os.close(descriptor)
        _close_descriptors(parent_descriptors)


def _require_lock_path_matches_handle(lock: Path, handle: Any) -> None:
    try:
        opened = os.fstat(handle.fileno())
        at_path = _fresh_path_metadata(Path(lock), "locked path")
    except OSError as error:
        raise ConfirmationReleaseEvidenceError(
            f"locked path disappeared or became unreadable: {lock}"
        ) from error
    require(
        stat.S_ISREG(opened.st_mode)
        and opened.st_nlink == 1
        and stat.S_ISREG(at_path.st_mode)
        and at_path.st_nlink == 1
        and (opened.st_dev, opened.st_ino) == (at_path.st_dev, at_path.st_ino),
        f"locked path no longer names the held regular inode: {lock}",
    )


@contextmanager
def _advisory_lock(path: Path, *, expect_available: bool) -> Iterator[dict[str, Any]]:
    lock = Path(path)
    with _validated_lock_handle(lock) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            require(not expect_available, f"lock is unexpectedly held: {lock}")
            _require_lock_path_matches_handle(lock, handle)
            try:
                yield {"path": str(lock), "observed": "held_by_another_process"}
            finally:
                _require_lock_path_matches_handle(lock, handle)
            return
        _require_lock_path_matches_handle(lock, handle)
        require(expect_available, f"required controller lock is not held: {lock}")
        try:
            yield {"path": str(lock), "observed": "acquired_by_finalizer"}
        finally:
            try:
                _require_lock_path_matches_handle(lock, handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _canonical_release_lock(state_dir: Path) -> Iterator[dict[str, Any]]:
    state = _existing_directory(Path(state_dir), "queue state root")
    lock = state / "release.lock"
    with _validated_lock_handle(
        lock, allow_canonical_release_creation=True, state_dir=state,
    ) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ConfirmationReleaseEvidenceError(
                "release lock is held by another process"
            ) from error
        _require_lock_path_matches_handle(lock, handle)
        try:
            yield {"path": str(lock), "observed": "acquired_by_finalizer"}
        finally:
            try:
                _require_lock_path_matches_handle(lock, handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _probe_controller_locks(state_dir: Path, worker_ids: Sequence[str]) -> list[dict[str, Any]]:
    results = []
    state = _existing_directory(Path(state_dir), "controller-lock queue state root")
    for name in ["coordinator.lock", *[f"worker-{worker_id}.lock" for worker_id in worker_ids]]:
        with _advisory_lock(state / name, expect_available=False) as row:
            results.append({"name": name, **row})
    return results


def _next_lane(ledger_validation: Mapping[str, Any], models: Sequence[str], layout_order: Sequence[str]) -> str | None:
    blocks = ledger_validation["blocks"]
    for model in ("N3", "D1"):
        if model not in models:
            continue
        if any(blocks[(model, layout)]["state"] in {"not_run", "technical_invalid"} for layout in layout_order):
            return model
    return None


def build_publication_pending(
    *, study_commit: str, remote_alias: str,
    observed_results_ancestor_commit: str, finalizer_triplet: Mapping[str, Any],
    finalizer_attestation: Mapping[str, Any], reconciliation: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    created_at_utc = reconciliation.get("created_at_utc")
    require(
        isinstance(created_at_utc, str),
        "publication pending lacks the internally observed reconciliation time",
    )
    finalizer_id = finalizer_triplet["job_id"]
    publish_prefix = f"results/jobs/{finalizer_id}/publish/confirmation_release_evidence"
    expected_paths = {
        name: f"{publish_prefix}/{filename}"
        for name, filename in (
            ("ledger", "result_attempt_ledger.json"),
            ("reconciliation", "cluster_reconciliation.json"),
            ("plan", "confirmation_release_plan.json"),
            ("queue_fragment", "confirmation_release_queue_fragment.json"),
            ("wave_receipt", "confirmation_release_wave_receipt.json"),
        )
    }
    expected_paths.update({
        "pending": f"{publish_prefix}/publication_pending.json",
        "evidence_receipt": f"{publish_prefix}/release_evidence_receipt.json",
        "worker_attestation": f"results/jobs/{finalizer_id}/publish/worker_attestation.json",
        "cluster_status": f"results/{NAMESPACE}/status.json",
        "producer_job_status": f"results/{NAMESPACE}/jobs/{finalizer_id}.json",
        "producer_publish_manifest": f"results/jobs/{finalizer_id}/publish_manifest.json",
    })
    require(set(artifacts) == {"ledger", "reconciliation", "plan", "queue_fragment", "wave_receipt"}, "pending artifact set changed")
    pending = signed_document({
        "schema_version": PENDING_SCHEMA,
        "status": "awaiting_cluster_coordinator_publication",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "created_at_utc": created_at_utc,
        "consume_by_utc": reconciliation["consume_by_utc"],
        "results_remote": {
            "alias": remote_alias,
            "identity": {"host": "github.com", "owner": "adeeb10abbas", "repository": "steerable"},
        },
        "results_ref": "refs/heads/codex/forecast-layout-gm-20260912-results",
        "observed_results_ancestor_commit": observed_results_ancestor_commit,
        "producer_queue_job": {
            "job_id": finalizer_id,
            "descriptor": finalizer_triplet["descriptor"],
            "descriptor_value": finalizer_triplet["descriptor_value"],
            "claim": finalizer_triplet["claim"],
            "claim_value": finalizer_triplet["claim_value"],
            "claim_control": finalizer_triplet["claim_control"],
            "worker_id": finalizer_triplet["claim_value"]["worker_id"],
            "hostname": finalizer_attestation["hostname"],
            "pod_uid": finalizer_attestation["pod_uid"],
            "pid": os.getpid(),
        },
        "claim_control": {
            "generation_lower_bound": reconciliation["observed_control_generation_lower_bound"],
            "observed_generation": reconciliation["observed_control_generation"],
            "semantics": reconciliation["control_semantics"],
        },
        "artifacts": {name: dict(row) for name, row in artifacts.items()},
        "expected_h1_paths": expected_paths,
        "post_publication_requirements": {
            "locate_first_descendant_containing_pending": True,
            "intervening_commits_coordinator_only": True,
            "immutable_published_paths_may_not_be_rewritten": True,
            "producer_outer_status": "succeeded",
            "producer_outer_returncode": 0,
            "producer_outer_child_reaped": True,
            "semantic_active_job_ids": [finalizer_id],
            "generation_may_only_increase": True,
            "remote_head_stable_across_verification": True,
            "workstation_results_write_allowed": False,
        },
        "claim_boundary": (
            "This pending receipt deliberately omits the future outer result, stream hashes, "
            "cluster-status hash, and publication commit. Those facts exist only after exit and "
            "must be authenticated by the read-only results-branch verifier."
        ),
    })
    validate_publication_pending(pending, study_commit=study_commit)
    return pending


def validate_publication_pending(
    value: Mapping[str, Any], *, study_commit: str, verify_local_artifacts: bool = True,
) -> dict[str, Any]:
    verify_signed_document(value, "publication pending")
    exact_keys(
        value,
        {
            "schema_version", "status", "study_id", "namespace", "study_commit",
            "created_at_utc", "consume_by_utc", "results_remote", "results_ref",
            "observed_results_ancestor_commit", "producer_queue_job", "claim_control",
            "artifacts", "expected_h1_paths", "post_publication_requirements",
            "claim_boundary", "payload_sha256",
        },
        "publication pending",
    )
    require(
        value.get("schema_version") == PENDING_SCHEMA
        and value.get("status") == "awaiting_cluster_coordinator_publication"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit
        and value.get("results_ref") == "refs/heads/codex/forecast-layout-gm-20260912-results"
        and COMMIT_RE.fullmatch(str(value.get("observed_results_ancestor_commit", ""))) is not None,
        "publication pending identity changed",
    )
    created = parse_utc(value.get("created_at_utc"), "pending creation time")
    consume_by = parse_utc(value.get("consume_by_utc"), "pending consume-by time")
    require(
        created <= consume_by
        and consume_by <= created + timedelta(seconds=MAX_EVIDENCE_AGE_SECONDS),
        "pending consume-by time exceeds capacity evidence lifetime",
    )
    require(
        value.get("results_remote") == {
            "alias": value.get("results_remote", {}).get("alias"),
            "identity": {"host": "github.com", "owner": "adeeb10abbas", "repository": "steerable"},
        }
        and isinstance(value["results_remote"]["alias"], str),
        "pending results repository identity changed",
    )
    producer = value.get("producer_queue_job")
    require(isinstance(producer, Mapping), "pending producer queue evidence missing")
    exact_keys(
        producer,
        {
            "job_id", "descriptor", "descriptor_value", "claim", "claim_value",
            "claim_control", "worker_id", "hostname", "pod_uid", "pid",
        },
        "pending producer queue job",
    )
    finalizer_id = safe_id(producer.get("job_id"), "pending finalizer")
    require(FINALIZER_RE.fullmatch(finalizer_id) is not None, "pending finalizer ID invalid")
    descriptor = producer.get("descriptor_value")
    claim = producer.get("claim_value")
    for name in ("descriptor", "claim"):
        row = producer.get(name)
        require(
            isinstance(row, Mapping) and set(row) == {"path", "bytes", "sha256"}
            and type(row.get("bytes")) is int and row["bytes"] > 0
            and isinstance(row.get("sha256"), str)
            and SHA256_RE.fullmatch(row["sha256"]) is not None,
            f"pending producer {name} descriptor invalid",
        )
    claim_control_row = producer.get("claim_control")
    require(isinstance(claim_control_row, Mapping), "pending producer claim-control missing")
    exact_keys(
        claim_control_row,
        {
            "job_id", "control_commit", "control_generation",
            "control_queue_blob_sha256", "descriptor_sha256",
        },
        "pending producer claim-control",
    )
    require(
        isinstance(descriptor, Mapping) and descriptor.get("job_id") == finalizer_id
        and isinstance(claim, Mapping) and claim.get("worker_id") == producer.get("worker_id")
        and claim.get("control_commit") == value.get("claim_control", {}).get("semantics", {}).get("control_commit")
        and value.get("claim_control", {}).get("semantics", {}).get("active_job_ids") == [finalizer_id]
        and producer.get("claim_control", {}).get("job_id") == finalizer_id
        and producer.get("claim_control", {}).get("control_commit") == claim.get("control_commit")
        and isinstance(producer.get("hostname"), str) and bool(producer["hostname"])
        and isinstance(producer.get("pod_uid"), str)
        and POD_UID_RE.fullmatch(producer["pod_uid"]) is not None
        and type(producer.get("pid")) is int and producer["pid"] > 0,
        "pending producer/claim semantic binding changed",
    )
    claim_control = value.get("claim_control")
    require(isinstance(claim_control, Mapping), "pending claim control missing")
    exact_keys(
        claim_control,
        {"generation_lower_bound", "observed_generation", "semantics"},
        "pending claim control",
    )
    semantics = claim_control.get("semantics")
    require(isinstance(semantics, Mapping), "pending claim-control semantics missing")
    exact_keys(semantics, {"namespace", "control_commit", "admission_deadline_unix", "shutdown", "active_job_ids"}, "pending claim-control semantics")
    require(
        type(claim_control.get("generation_lower_bound")) is int
        and type(claim_control.get("observed_generation")) is int
        and claim_control["observed_generation"] >= claim_control["generation_lower_bound"]
        and claim_control["generation_lower_bound"] == claim.get("control_generation")
        and semantics.get("namespace") == NAMESPACE and semantics.get("shutdown") is False,
        "pending claim-control generations changed",
    )
    artifacts = value.get("artifacts")
    require(
        isinstance(artifacts, Mapping)
        and set(artifacts) == {"ledger", "reconciliation", "plan", "queue_fragment", "wave_receipt"},
        "pending artifact inventory changed",
    )
    for name, descriptor_row in artifacts.items():
        require(
            isinstance(descriptor_row, Mapping)
            and set(descriptor_row) == {"path", "bytes", "sha256"}
            and type(descriptor_row.get("bytes")) is int and descriptor_row["bytes"] >= 0
            and isinstance(descriptor_row.get("sha256"), str)
            and SHA256_RE.fullmatch(descriptor_row["sha256"]) is not None,
            f"pending artifact descriptor invalid: {name}",
        )
        if verify_local_artifacts:
            identity = file_identity(Path(str(descriptor_row.get("path", ""))))
            require(identity == descriptor_row, f"pending artifact changed: {name}")
    requirements = value.get("post_publication_requirements")
    require(isinstance(requirements, Mapping), "pending post-publication requirements missing")
    exact_keys(
        requirements,
        {
            "locate_first_descendant_containing_pending", "intervening_commits_coordinator_only",
            "immutable_published_paths_may_not_be_rewritten", "producer_outer_status",
            "producer_outer_returncode", "producer_outer_child_reaped",
            "semantic_active_job_ids", "generation_may_only_increase",
            "remote_head_stable_across_verification", "workstation_results_write_allowed",
        },
        "pending post-publication requirements",
    )
    require(
        requirements.get("locate_first_descendant_containing_pending") is True
        and requirements.get("intervening_commits_coordinator_only") is True
        and requirements.get("immutable_published_paths_may_not_be_rewritten") is True
        and requirements.get("producer_outer_status") == "succeeded"
        and requirements.get("producer_outer_returncode") == 0
        and requirements.get("producer_outer_child_reaped") is True
        and requirements.get("semantic_active_job_ids") == [finalizer_id]
        and requirements.get("generation_may_only_increase") is True
        and requirements.get("remote_head_stable_across_verification") is True
        and requirements.get("workstation_results_write_allowed") is False,
        "pending post-publication gate changed",
    )
    expected_paths = value.get("expected_h1_paths")
    require(isinstance(expected_paths, Mapping), "pending expected H1 paths missing")
    prefix = f"results/jobs/{finalizer_id}/publish/confirmation_release_evidence"
    expected = {
        "ledger": f"{prefix}/result_attempt_ledger.json",
        "reconciliation": f"{prefix}/cluster_reconciliation.json",
        "plan": f"{prefix}/confirmation_release_plan.json",
        "queue_fragment": f"{prefix}/confirmation_release_queue_fragment.json",
        "wave_receipt": f"{prefix}/confirmation_release_wave_receipt.json",
        "pending": f"{prefix}/publication_pending.json",
        "evidence_receipt": f"{prefix}/release_evidence_receipt.json",
        "worker_attestation": f"results/jobs/{finalizer_id}/publish/worker_attestation.json",
        "cluster_status": f"results/{NAMESPACE}/status.json",
        "producer_job_status": f"results/{NAMESPACE}/jobs/{finalizer_id}.json",
        "producer_publish_manifest": f"results/jobs/{finalizer_id}/publish_manifest.json",
    }
    require(dict(expected_paths) == expected, "pending expected H1 path inventory changed")
    return dict(value)


def _call_pending_wave_builder(
    *, wave: Any, source_root: Path, state_dir: Path, study_commit: str,
    inputs: Mapping[str, Any], ledger_identity: Mapping[str, Any],
    reconciliation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    builder = getattr(wave, "build_pending_release_wave", None)
    require(callable(builder), "release-wave consumer lacks reviewed pending-publication API")
    result = builder(
        source_root=source_root,
        state_dir=state_dir,
        study_commit=study_commit,
        confirmation_freeze=Path(inputs["confirmation_freeze"]["path"]),
        confirmation_freeze_sha256=inputs["confirmation_freeze"]["sha256"],
        fixture_freeze=Path(inputs["fixture_freeze"]["path"]),
        fixture_freeze_sha256=inputs["fixture_freeze"]["sha256"],
        resource_qualification=Path(inputs["resource_qualification"]["path"]),
        resource_qualification_sha256=inputs["resource_qualification"]["sha256"],
        terminal_runtime_identities=Path(inputs["terminal_runtime_identities"]["path"]),
        terminal_runtime_identities_sha256=inputs["terminal_runtime_identities"]["sha256"],
        result_attempt_ledger=Path(ledger_identity["path"]),
        result_attempt_ledger_sha256=ledger_identity["sha256"],
        cluster_reconciliation=Path(reconciliation_identity["path"]),
        cluster_reconciliation_sha256=reconciliation_identity["sha256"],
    )
    require(
        isinstance(result, Mapping) and set(result) == {"plan", "queue_fragment", "receipt"},
        "pending release-wave builder return shape changed",
    )
    return dict(result)


def finalize_pending(
    *, source_root: Path, state_dir: Path, job_dir: Path, job_id: str,
    study_commit: str, expected_lane: str, inputs_path: Path,
    inputs_sha256: str, runtime_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the sole cluster finalizer while holding the shared release lock."""

    contract = load_contract(source_root)
    require(expected_lane in {"N3", "D1"}, "expected finalizer lane invalid")
    match = FINALIZER_RE.fullmatch(job_id)
    require(match is not None and match.group(1).upper() == expected_lane, "finalizer ID/lane differ")
    inputs_identity, inputs = validate_inputs(
        inputs_path, inputs_sha256, study_commit=study_commit,
    )
    require(
        _attestation_scope(inputs["worker_attestation_job_ids"])
        == (expected_lane, int(match.group(2))),
        "worker attestation wave is not bound to this finalizer attempt",
    )
    expected_role = contract["producer"]["finalizer_roles"][expected_lane]
    state_root = _existing_directory(Path(state_dir), "finalizer queue state root")
    trusted_state_root = _existing_directory(
        Path(contract["queue"]["state_dir"]), "trusted GM PVC queue root",
    )
    require(
        state_root == trusted_state_root,
        "finalizer state directory is not the trusted GM PVC queue root",
    )
    jobs_root = _existing_directory(state_root / "jobs", "finalizer queue jobs root")
    expected_job_root = _confined_path(
        jobs_root / job_id, anchor=jobs_root,
        label="finalizer canonical job directory", allow_missing=False,
    )
    supplied_job_root = _confined_path(
        Path(job_dir), anchor=jobs_root,
        label="finalizer supplied job directory", allow_missing=False,
    )
    require(supplied_job_root == expected_job_root, "finalizer job directory changed")
    output_root = _ensure_confined_directory(
        supplied_job_root / "publish" / contract["producer"]["publication_subdir"],
        anchor=supplied_job_root, label="finalizer publication directory",
    )
    with _canonical_release_lock(state_root):
            finalizer_triplet, control, _control_identity, modules = validate_running_queue_context(
                source_root=source_root, state_dir=state_dir, job_dir=job_dir,
                job_id=job_id, study_commit=study_commit, expected_role=expected_role,
                runtime_argv=runtime_argv,
            )
            elapsed_since_claim = max(
                0.0,
                time.time() - float(finalizer_triplet["claim_value"]["claimed_unix"]),
            )
            require(
                finalizer_triplet["descriptor_value"]["max_wall_seconds"]
                - elapsed_since_claim
                >= contract["producer"]["minimum_finalizer_remaining_wall_seconds"],
                "finalizer wrapper wall budget is insufficient to seal and publish pending evidence",
            )
            topology = deployment_topology(source_root)
            selected_worker_ids = sorted({
                _attestation_identity(item, topology=topology)[2]
                for item in inputs["worker_attestation_job_ids"]
            })
            controller_locks = _probe_controller_locks(state_dir, selected_worker_ids)
            lock_context: contextmanager[Any]
            if expected_lane == "D1":
                d1_lock = Path(modules["wave"].load_runtime_modules(source_root)["d1"].GLOBAL_D1_SERVER_LOCK)
                d1_context = _advisory_lock(d1_lock, expect_available=True)
            else:
                d1_context = nullcontext({"path": None, "observed": "not_required"})
            with d1_context as d1_lock_receipt:
                remote_url, ancestor = observe_results_ancestor(
                    source_root=source_root,
                    remote_alias=inputs["results_remote_alias"],
                    results_ref=inputs["results_ref"],
                )
                del remote_url
                all_jobs, _indexed = scan_queue_state(
                    source_root=source_root, state_dir=state_dir,
                    study_commit=study_commit, running_finalizer_job_id=job_id,
                )
                workers = collect_worker_attestations(
                    source_root=source_root, state_dir=state_dir,
                    study_commit=study_commit,
                    lane=expected_lane, attempt_number=int(match.group(2)),
                    job_ids=inputs["worker_attestation_job_ids"],
                    all_jobs=all_jobs,
                )
                finalizer_attestation = attest_worker(
                    source_root=source_root, state_dir=state_dir, job_dir=job_dir,
                    job_id=job_id, study_commit=study_commit,
                    expected_worker_id=finalizer_triplet["claim_value"]["worker_id"],
                    expected_finalizer_job_id=job_id,
                    runtime_argv=runtime_argv,
                )
                finalizer_attestation_identity, observed_finalizer_attestation = load_json_with_identity(
                    supplied_job_root / "publish" / "worker_attestation.json",
                    "running finalizer worker attestation",
                )
                require(
                    observed_finalizer_attestation == finalizer_attestation,
                    "running finalizer attestation changed after immutable write",
                )
                ledger, all_jobs = assemble_result_attempt_ledger(
                    source_root=source_root, state_dir=state_dir,
                    study_commit=study_commit, inputs=inputs,
                    running_finalizer_job_id=job_id,
                )
                ledger_identity = immutable_write(
                    output_root / "result_attempt_ledger.json", ledger,
                    anchor=output_root,
                )
                ledger_validation = validate_result_attempt_ledger(
                    ledger, source_root=source_root, state_dir=state_dir,
                    study_commit=study_commit, all_queue_jobs=all_jobs,
                )
                context = _scientific_context(
                    source_root=source_root, study_commit=study_commit,
                    inputs=inputs, modules=modules,
                )
                lane = _next_lane(
                    ledger_validation, context["models"], context["layout_order"],
                )
                require(lane == expected_lane, "caller-selected finalizer lane differs from ledger")
                reconciliation = build_pending_reconciliation(
                    source_root=source_root, state_dir=state_dir,
                    study_commit=study_commit, lane=expected_lane,
                    finalizer_triplet=finalizer_triplet, control=control,
                    all_jobs=all_jobs, ledger_identity=ledger_identity,
                    ledger_validation=ledger_validation, workers=workers,
                    finalizer_attestation_identity=finalizer_attestation_identity,
                    finalizer_attestation=finalizer_attestation,
                    observed_results_ancestor_commit=ancestor,
                    controller_lock_probes=controller_locks,
                    d1_global_lock_probe=d1_lock_receipt,
                )
                reconciliation_identity = immutable_write(
                    output_root / "cluster_reconciliation.json", reconciliation,
                    anchor=output_root,
                )
                wave_result = _call_pending_wave_builder(
                    wave=modules["wave"], source_root=source_root,
                    state_dir=state_dir,
                    study_commit=study_commit,
                    inputs=inputs, ledger_identity=ledger_identity,
                    reconciliation_identity=reconciliation_identity,
                )
                wave_identities = {
                    "plan": immutable_write(
                        output_root / "confirmation_release_plan.json", wave_result["plan"],
                        anchor=output_root,
                    ),
                    "queue_fragment": immutable_write(
                        output_root / "confirmation_release_queue_fragment.json",
                        wave_result["queue_fragment"], anchor=output_root,
                    ),
                    "wave_receipt": immutable_write(
                        output_root / "confirmation_release_wave_receipt.json",
                        wave_result["receipt"], anchor=output_root,
                    ),
                }
                artifacts = {
                    "ledger": ledger_identity,
                    "reconciliation": reconciliation_identity,
                    **wave_identities,
                }
                pending = build_publication_pending(
                    study_commit=study_commit,
                    remote_alias=inputs["results_remote_alias"],
                    observed_results_ancestor_commit=ancestor,
                    finalizer_triplet=finalizer_triplet,
                    finalizer_attestation=finalizer_attestation,
                    reconciliation=reconciliation, artifacts=artifacts,
                )
                pending_identity = immutable_write(
                    output_root / "publication_pending.json", pending, anchor=output_root,
                )
                receipt = signed_document({
                    "schema_version": EVIDENCE_RECEIPT_SCHEMA,
                    "status": "pending_coordinator_publication_no_science",
                    "study_id": STUDY_ID,
                    "namespace": NAMESPACE,
                    "study_commit": study_commit,
                    "created_at_utc": reconciliation["created_at_utc"],
                    "consume_by_utc": reconciliation["consume_by_utc"],
                    "inputs": inputs_identity,
                    "artifacts": {**artifacts, "publication_pending": pending_identity},
                    "outer_result_available": False,
                    "cluster_status_available": False,
                    "publication_commit_available": False,
                    "queue_mutated": False,
                    "results_branch_written_by_finalizer": False,
                    "science_or_labels_emitted": False,
                    "claim_boundary": (
                        "The queue child has built a descriptor-only wave, but it is not release "
                        "authority until the coordinator publishes the child's terminal outer result "
                        "and the read-only verifier authenticates the remote history."
                    ),
                })
                receipt_identity = immutable_write(
                    output_root / "release_evidence_receipt.json", receipt,
                    anchor=output_root,
                )
                return {
                    "ledger": ledger_identity,
                    "reconciliation": reconciliation_identity,
                    "plan": wave_identities["plan"],
                    "queue_fragment": wave_identities["queue_fragment"],
                    "wave_receipt": wave_identities["wave_receipt"],
                    "publication_pending": pending_identity,
                    "evidence_receipt": receipt_identity,
                }


@contextmanager
def _fetch_remote_history(
    *, remote_url: str, results_ref: str, expected_head: str,
) -> Iterator[Path]:
    """Fetch one exact remote observation without touching the study checkout."""

    with tempfile.TemporaryDirectory(prefix="wmf-confirmation-release-verify-") as temporary:
        bare = Path(temporary) / "results.git"
        _initialize_bare_repository(bare, "results")
        _git(
            bare, "fetch", "--no-tags", "--no-recurse-submodules",
            "--filter=blob:limit=1048576", remote_url,
            f"{results_ref}:refs/heads/verified-results", timeout=120,
        )
        # Keep only a code-generated promisor stanza for bounded on-demand
        # reads of larger, explicitly inventoried compact artifacts. No
        # caller/configured alias or fetch-created URL survives.
        _write_isolated_git_config(bare, promisor_url=remote_url)
        _validate_isolated_git_repository(bare, promisor_url=remote_url)
        fetched = _git(
            bare, "rev-parse", "refs/heads/verified-results^{commit}", text=True
        ).stdout.strip()
        require(fetched == expected_head, "results ref moved during exact fetch")
        yield bare
        _write_isolated_git_config(bare)
        _validate_isolated_git_repository(bare)


def _git_blob(repo: Path, commit: str, path: str, *, required: bool = True) -> bytes | None:
    result = _git(repo, "show", f"{commit}:{path}", accepted=(0, 128))
    if result.returncode:
        require(not required, f"published Git blob is missing: {path}")
        return None
    return result.stdout


def _commit_metadata(repo: Path, commit: str) -> dict[str, str]:
    value = _git(
        repo, "show", "-s", "--format=%an%x00%ae%x00%s%x00%P", commit,
    ).stdout.decode("utf-8", errors="strict")
    fields = value.rstrip("\n").split("\0")
    require(len(fields) == 4, "coordinator commit metadata changed")
    return {"author_name": fields[0], "author_email": fields[1], "subject": fields[2], "parents": fields[3]}


def _commit_changes(repo: Path, commit: str) -> list[tuple[str, str]]:
    output = _git(
        repo, "diff-tree", "--no-commit-id", "--name-status", "-r", commit,
    ).stdout.decode("utf-8", errors="strict")
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        require(len(fields) == 2 and fields[0] in {"A", "M"}, "coordinator commit has delete/rename/complex change")
        rows.append((fields[0], fields[1]))
    return rows


def _validate_coordinator_commit(repo: Path, commit: str) -> list[tuple[str, str]]:
    metadata = _commit_metadata(repo, commit)
    require(
        metadata == {
            "author_name": "WMF cluster coordinator",
            "author_email": "wmf-cluster@users.noreply.github.com",
            "subject": "Record cluster queue receipts",
            "parents": metadata["parents"],
        }
        and len(metadata["parents"].split()) == 1,
        "results history contains a noncoordinator or nonlinear commit",
    )
    return _commit_changes(repo, commit)


def _parse_json_blob(blob: bytes, label: str) -> dict[str, Any]:
    return _strict_json_object(blob, label)


def _remote_blob_descriptor(commit: str, path: str, blob: bytes) -> dict[str, Any]:
    return {"commit": commit, "git_path": path, "bytes": len(blob), "sha256": sha256_bytes(blob)}


def _descriptor_matches_blob(descriptor: Mapping[str, Any], blob: bytes, *, label: str) -> None:
    require(
        set(descriptor) == {"path", "bytes", "sha256"}
        and type(descriptor.get("bytes")) is int
        and descriptor.get("bytes") == len(blob)
        and descriptor.get("sha256") == sha256_bytes(blob),
        f"{label} descriptor differs from published bytes",
    )


def _validate_remote_finalizer_attestation(
    *, value: Mapping[str, Any], source_root: Path, study_commit: str,
    pending: Mapping[str, Any], reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    producer = pending["producer_queue_job"]
    finalizer_id = producer["job_id"]
    state_dir = Path(load_contract(source_root)["queue"]["state_dir"])
    triplet = {
        "job_id": finalizer_id,
        "descriptor": producer["descriptor"],
        "descriptor_value": producer["descriptor_value"],
        "claim": producer["claim"],
        "claim_value": producer["claim_value"],
        "claim_control": producer["claim_control"],
    }
    summary = _validate_worker_attestation_payload(
        value, source_root=source_root, state_dir=state_dir,
        study_commit=study_commit, expected_triplet=triplet,
        as_of=parse_utc(pending["created_at_utc"], "pending creation time"),
        expected_semantic_active_job_ids=[finalizer_id],
        require_staged_path_exists=False,
    )
    require(
        value.get("observed_at_utc") == pending.get("created_at_utc")
        and value.get("hostname") == producer.get("hostname")
        and value.get("pod_uid") == producer.get("pod_uid")
        and value.get("process_inventory", {}).get("current_pid") == producer.get("pid"),
        "published finalizer attestation differs from pending producer identity",
    )
    workers = reconciliation.get("task_gpu_state", {}).get("workers")
    require(isinstance(workers, list), "published reconciliation worker inventory missing")
    matched = [row for row in workers if isinstance(row, Mapping) and row.get("worker_id") == summary["worker_id"]]
    local_attestation = matched[0].get("attestation") if len(matched) == 1 else None
    require(
        len(matched) == 1 and isinstance(local_attestation, Mapping)
        and local_attestation.get("path") == str(
            _lexical_absolute_path(state_dir, "published queue state root")
            / "jobs" / finalizer_id / "publish" / "worker_attestation.json"
        )
        and local_attestation.get("bytes") == len(pretty_bytes(value))
        and local_attestation.get("sha256") == sha256_bytes(pretty_bytes(value))
        and matched[0].get("active_job_ids") == [finalizer_id]
        and matched[0].get("pod_uid") == value.get("pod_uid")
        and matched[0].get("hostname") == value.get("hostname"),
        "published finalizer attestation is detached from reconciliation",
    )
    return dict(value)


def _validate_published_evidence_receipt(
    *, value: Mapping[str, Any], pending: Mapping[str, Any],
    pending_blob: bytes, source_root: Path, study_commit: str,
) -> dict[str, Any]:
    exact_keys(value, EVIDENCE_RECEIPT_KEYS, "release evidence receipt")
    verify_signed_document(value, "release evidence receipt")
    require(
        value.get("schema_version") == EVIDENCE_RECEIPT_SCHEMA
        and value.get("status") == "pending_coordinator_publication_no_science"
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == study_commit
        and value.get("created_at_utc") == pending.get("created_at_utc")
        and value.get("consume_by_utc") == pending.get("consume_by_utc")
        and value.get("outer_result_available") is False
        and value.get("cluster_status_available") is False
        and value.get("publication_commit_available") is False
        and value.get("queue_mutated") is False
        and value.get("results_branch_written_by_finalizer") is False
        and value.get("science_or_labels_emitted") is False
        and isinstance(value.get("claim_boundary"), str) and bool(value["claim_boundary"]),
        "release evidence receipt identity/boundary changed",
    )
    producer = pending["producer_queue_job"]
    queue = load_modules(source_root)["queue"]
    descriptor_info = _validate_finalizer_descriptor(
        producer["descriptor_value"], source_root=source_root,
        study_commit=study_commit, queue=queue,
    )
    inputs = value.get("inputs")
    require(isinstance(inputs, Mapping), "release evidence input descriptor missing")
    exact_keys(inputs, {"path", "bytes", "sha256"}, "release evidence input descriptor")
    require(
        inputs.get("path") == descriptor_info["inputs_path"]
        and inputs.get("sha256") == descriptor_info["inputs_sha256"]
        and type(inputs.get("bytes")) is int and inputs["bytes"] > 0,
        "release evidence receipt input differs from finalizer descriptor",
    )
    artifacts = value.get("artifacts")
    require(
        isinstance(artifacts, Mapping)
        and set(artifacts) == {
            "ledger", "reconciliation", "plan", "queue_fragment", "wave_receipt",
            "publication_pending",
        },
        "release evidence receipt artifact inventory changed",
    )
    for name in ("ledger", "reconciliation", "plan", "queue_fragment", "wave_receipt"):
        require(artifacts.get(name) == pending["artifacts"][name], f"release evidence receipt detached from {name}")
    pending_descriptor = artifacts["publication_pending"]
    require(isinstance(pending_descriptor, Mapping), "release evidence pending descriptor missing")
    _descriptor_matches_blob(pending_descriptor, pending_blob, label="release evidence pending")
    finalizer_id = producer["job_id"]
    configured_state_root = _validate_no_symlink_components(
        Path(load_contract(source_root)["queue"]["state_dir"]),
        "published configured queue state root", allow_missing=True,
    )
    output_root = (
        configured_state_root
        / "jobs" / finalizer_id / "publish" / "confirmation_release_evidence"
    )
    expected_names = {
        "ledger": "result_attempt_ledger.json",
        "reconciliation": "cluster_reconciliation.json",
        "plan": "confirmation_release_plan.json",
        "queue_fragment": "confirmation_release_queue_fragment.json",
        "wave_receipt": "confirmation_release_wave_receipt.json",
        "publication_pending": "publication_pending.json",
    }
    for name, filename in expected_names.items():
        row = artifacts[name]
        require(
            isinstance(row, Mapping) and row.get("path") == str(output_root / filename),
            f"release evidence artifact has noncanonical local path: {name}",
        )
    return dict(value)


def _validate_published_status(
    *, status: Mapping[str, Any], pending: Mapping[str, Any],
    finalizer_id: str, reconciliation: Mapping[str, Any], queue: Any,
    source_root: Path,
) -> dict[str, Any]:
    exact_keys(
        status,
        {
            "schema_version", "namespace", "control", "jobs",
            "automatic_claim_recovery", "default_published_log_tail_bytes",
            "maximum_combined_log_tail_bytes_per_job",
        },
        "published cluster status",
    )
    require(
        status.get("schema_version") == STATUS_SCHEMA
        and status.get("namespace") == NAMESPACE
        and status.get("automatic_claim_recovery") is False
        and status.get("default_published_log_tail_bytes") == 0
        and status.get("maximum_combined_log_tail_bytes_per_job") == 8192,
        "published cluster status identity changed",
    )
    control = status.get("control")
    require(isinstance(control, Mapping), "published cluster status lacks control")
    exact_keys(control, CONTROL_KEYS, "published cluster control")
    semantics = pending["claim_control"]["semantics"]
    require(
        all(control.get(key) == semantics[key] for key in (
            "namespace", "control_commit", "admission_deadline_unix", "shutdown", "active_job_ids",
        ))
        and type(control.get("control_generation")) is int
        and control["control_generation"] >= pending["claim_control"]["generation_lower_bound"]
        and control["control_generation"] != 998,
        "published control changed semantically or regressed generation",
    )
    rows = status.get("jobs")
    require(isinstance(rows, list), "published cluster job status missing")
    prior = reconciliation["prior_inventory"]["all_prior_job_ids"]
    require(
        [row.get("job_id") for row in rows if isinstance(row, Mapping)] == prior
        and len(rows) == len(prior),
        "published cluster status is not the complete finalizer inventory",
    )
    matches = [row for row in rows if row.get("job_id") == finalizer_id]
    require(len(matches) == 1, "published finalizer status missing or duplicated")
    finalizer = matches[0]
    producer = pending["producer_queue_job"]
    descriptor = producer["descriptor_value"]
    _validate_finalizer_descriptor(
        descriptor, source_root=source_root,
        study_commit=pending["study_commit"], queue=queue,
    )
    descriptor_identity = producer["descriptor"]
    claim_identity = producer["claim"]
    claim = producer["claim_value"]
    configured_state_root = _validate_no_symlink_components(
        Path(load_contract(source_root)["queue"]["state_dir"]),
        "published configured queue state root", allow_missing=True,
    )
    job_root = configured_state_root / "jobs" / finalizer_id
    require(
        isinstance(descriptor_identity, Mapping)
        and descriptor_identity.get("path") == str(job_root / "descriptor.json")
        and descriptor_identity.get("bytes") == len(queue.encode(descriptor))
        and descriptor_identity.get("sha256") == sha256_bytes(queue.encode(descriptor))
        and isinstance(claim_identity, Mapping)
        and claim_identity.get("path") == str(job_root / "claim" / "owner.json")
        and claim_identity.get("bytes") == len(queue.encode(claim))
        and claim_identity.get("sha256") == sha256_bytes(queue.encode(claim)),
        "pending producer queue file identities changed",
    )
    validate_queue_claim(claim, descriptor=descriptor, result=None, queue=queue)
    require(
        _queue_manifest_at_claim(
            source_root=source_root, claim=claim, descriptor=descriptor,
            queue_path=load_contract(source_root)["queue"]["queue_path"], queue=queue,
        ) == producer["claim_control"],
        "pending finalizer claim differs from authorized control commit",
    )
    started = parse_utc(finalizer.get("started_at"), "published finalizer start")
    ended = parse_utc(finalizer.get("ended_at"), "published finalizer end")
    created = parse_utc(pending["created_at_utc"], "pending creation time")
    wall = finalizer.get("wall_seconds")
    require(
        finalizer.get("source_commit") == pending["study_commit"]
        and finalizer.get("descriptor_sha256") == sha256_bytes(
            queue.encode(descriptor)
        )
        and finalizer.get("worker_id") == producer["worker_id"]
        and finalizer.get("status") == "succeeded"
        and finalizer.get("returncode") == 0
        and finalizer.get("error_type") is None
        and finalizer.get("child_reaped") is True
        and started <= created <= ended
        and not isinstance(wall, bool) and isinstance(wall, (int, float))
        and math.isfinite(wall) and 0 <= wall <= descriptor["max_wall_seconds"] + 30
        and "log_tails" not in finalizer,
        "published finalizer outer result is not successful/reaped",
    )
    for row in rows:
        require(isinstance(row, Mapping), "published status job row invalid")
        exact_keys(
            row,
            {
                "job_id", "source_commit", "descriptor_sha256", "status", "returncode",
                "worker_id", "started_at", "ended_at", "wall_seconds", "error_type",
                "child_pid", "child_reaped", "stdout", "stderr",
            },
            "published terminal job row",
        )
        require(
            row.get("status") in TERMINAL_STATUSES
            and row.get("child_reaped") is True and "log_tails" not in row,
            "published status contains live/unreaped/log-bearing queue job",
        )
        for stream in ("stdout", "stderr"):
            stream_row = row.get(stream)
            require(
                isinstance(stream_row, Mapping) and set(stream_row) == {"bytes", "sha256"}
                and type(stream_row.get("bytes")) is int and stream_row["bytes"] >= 0
                and isinstance(stream_row.get("sha256"), str)
                and SHA256_RE.fullmatch(stream_row["sha256"]) is not None,
                "published status stream descriptor invalid",
            )
    return dict(finalizer)


def verify_published_finalizer(
    *, source_root: Path, finalizer_job_id: str, study_commit: str,
) -> dict[str, Any]:
    """Authenticate the first coordinator commit containing one pending finalizer.

    The operation is read-only with respect to both the authorized remote and
    the study checkout.  It returns a signed local verification document and
    remote Git-blob descriptors; callers decide where to retain that receipt.
    """

    safe_id(finalizer_job_id, "finalizer job")
    require(FINALIZER_RE.fullmatch(finalizer_job_id) is not None, "finalizer job ID invalid")
    _validate_source_files_at_commit(source_root, study_commit, require_head=False)
    contract = load_contract(source_root)
    queue = load_modules(source_root)["queue"]
    results_ref = contract["queue"]["results_ref"]
    # A pending receipt stores the alias, but lookup must start from an
    # allow-listed configured URL. Exactly one valid alias is accepted.
    alias, _reported_url = _single_authorized_remote(
        source_root=source_root, contract=contract, label="results",
    )
    remote_url = contract["queue"]["trusted_fetch_url"]
    require(
        isinstance(remote_url, str) and remote_url in contract["queue"]["allowed_results_urls"],
        "trusted results fetch URL is not source-pinned",
    )
    initial_head = _remote_head(source_root, remote_url, results_ref)
    before_status = _git(
        source_root, "status", "--porcelain=v1", "--untracked-files=all", text=True
    ).stdout
    pending_path = (
        f"results/jobs/{finalizer_job_id}/publish/confirmation_release_evidence/"
        "publication_pending.json"
    )
    with _fetch_remote_history(
        remote_url=remote_url, results_ref=results_ref, expected_head=initial_head,
    ) as snapshot:
        # Locate pending from the current head first; it supplies the observed
        # ancestor and exact artifact paths without trusting caller input.
        head_pending_blob = _git_blob(snapshot, initial_head, pending_path)
        pending = _parse_json_blob(head_pending_blob, "published pending receipt")  # type: ignore[arg-type]
        validate_publication_pending(
            pending, study_commit=study_commit, verify_local_artifacts=False,
        )
        require(
            pending["producer_queue_job"]["job_id"] == finalizer_job_id
            and pending["results_remote"] == {
                "alias": alias,
                "identity": contract["queue"]["repository_identity"],
            }
            and pending["results_ref"] == results_ref,
            "pending receipt targets another producer/repository",
        )
        ancestor = pending["observed_results_ancestor_commit"]
        require(
            _git_blob(snapshot, ancestor, pending_path, required=False) is None,
            "pending receipt already existed at its claimed results ancestor",
        )
        ancestry = _git(
            snapshot, "merge-base", "--is-ancestor", ancestor, initial_head,
            accepted=(0, 1),
        )
        require(ancestry.returncode == 0 and ancestor != initial_head, "observed results ancestor is not before publication")
        commits = _git(
            snapshot, "rev-list", "--first-parent", "--reverse", f"{ancestor}..{initial_head}",
            text=True,
        ).stdout.splitlines()
        require(commits and len(commits) <= 256, "publication history is empty or unbounded")
        first_pending: str | None = None
        commit_changes: dict[str, list[tuple[str, str]]] = {}
        for commit in commits:
            commit_changes[commit] = _validate_coordinator_commit(snapshot, commit)
            present = _git_blob(snapshot, commit, pending_path, required=False) is not None
            if present and first_pending is None:
                first_pending = commit
        require(first_pending is not None, "no coordinator descendant contains pending receipt")
        publication_commit = first_pending
        publication_index = commits.index(publication_commit)
        status_path = f"results/{NAMESPACE}/status.json"
        for index, commit in enumerate(commits):
            changes = commit_changes[commit]
            if index != publication_index:
                require(
                    all(path == status_path and status in {"A", "M"} for status, path in changes),
                    "intervening/later commit is not a coordinator status-only snapshot",
                )
        expected_paths = pending["expected_h1_paths"]
        artifact_blobs: dict[str, bytes] = {}
        remote_descriptors: dict[str, dict[str, Any]] = {}
        for name, local_descriptor in pending["artifacts"].items():
            git_path = expected_paths[name]
            blob = _git_blob(snapshot, publication_commit, git_path)
            require(
                len(blob) == local_descriptor["bytes"]
                and sha256_bytes(blob) == local_descriptor["sha256"],
                f"coordinator-published artifact changed: {name}",
            )
            artifact_blobs[name] = blob
            remote_descriptors[name] = _remote_blob_descriptor(publication_commit, git_path, blob)
        publication_pending_blob = _git_blob(snapshot, publication_commit, expected_paths["pending"])
        require(publication_pending_blob == head_pending_blob, "pending receipt changed within remote history")
        remote_descriptors["publication_pending"] = _remote_blob_descriptor(
            publication_commit, expected_paths["pending"], publication_pending_blob
        )
        reconciliation = _parse_json_blob(artifact_blobs["reconciliation"], "published reconciliation")
        verify_signed_document(reconciliation, "published reconciliation")
        ledger = _parse_json_blob(artifact_blobs["ledger"], "published ledger")
        verify_signed_document(ledger, "published ledger")
        require(
            reconciliation.get("results_publication", {}).get("observed_results_ancestor_commit") == ancestor
            and reconciliation.get("results_publication", {}).get("ledger", {}).get("sha256")
            == sha256_bytes(artifact_blobs["ledger"]),
            "published ledger/reconciliation/ancestor binding changed",
        )
        require(
            reconciliation.get("created_at_utc") == pending.get("created_at_utc")
            and reconciliation.get("consume_by_utc") == pending.get("consume_by_utc"),
            "published pending/reconciliation capacity lifetime changed",
        )
        parsed_artifacts = {
            "ledger": ledger,
            "reconciliation": reconciliation,
            "plan": _parse_json_blob(artifact_blobs["plan"], "published release plan"),
            "queue_fragment": _parse_json_blob(
                artifact_blobs["queue_fragment"], "published queue fragment"
            ),
            "wave_receipt": _parse_json_blob(
                artifact_blobs["wave_receipt"], "published wave receipt"
            ),
            "publication_pending": pending,
        }
        status_blob = _git_blob(snapshot, publication_commit, expected_paths["cluster_status"])
        status = _parse_json_blob(status_blob, "published cluster status")
        producer_result = _validate_published_status(
            status=status, pending=pending, finalizer_id=finalizer_job_id,
            reconciliation=reconciliation, queue=queue, source_root=source_root,
        )
        remote_descriptors["cluster_status"] = _remote_blob_descriptor(
            publication_commit, expected_paths["cluster_status"], status_blob
        )
        job_blob = _git_blob(snapshot, publication_commit, expected_paths["producer_job_status"])
        require(_parse_json_blob(job_blob, "published finalizer job") == producer_result, "job/status finalizer rows differ")
        remote_descriptors["producer_job_status"] = _remote_blob_descriptor(
            publication_commit, expected_paths["producer_job_status"], job_blob
        )
        manifest_blob = _git_blob(snapshot, publication_commit, expected_paths["producer_publish_manifest"])
        manifest = _parse_json_blob(manifest_blob, "producer publish manifest")
        exact_keys(
            manifest,
            {
                "files", "errors", "per_file_limit_bytes", "total_limit_bytes",
                "copied_bytes",
            },
            "producer publish manifest",
        )
        require(
            manifest.get("errors") == [] and isinstance(manifest.get("files"), list)
            and manifest.get("per_file_limit_bytes") == 16 * 1024 * 1024
            and manifest.get("total_limit_bytes") == 64 * 1024 * 1024,
            "producer artifact publication was incomplete",
        )
        manifest_rows: dict[str, Mapping[str, Any]] = {}
        for row in manifest["files"]:
            require(isinstance(row, Mapping), "producer publish manifest row invalid")
            exact_keys(row, {"path", "bytes", "sha256"}, "producer publish manifest row")
            relative = row.get("path")
            require(
                isinstance(relative, str) and relative not in manifest_rows
                and not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and type(row.get("bytes")) is int and row["bytes"] >= 0
                and isinstance(row.get("sha256"), str)
                and SHA256_RE.fullmatch(row["sha256"]) is not None,
                "producer publish manifest row changed",
            )
            manifest_rows[relative] = row
        required_relative = {
            Path(path).relative_to(f"results/jobs/{finalizer_job_id}/publish").as_posix()
            for name, path in expected_paths.items()
            if name not in {"cluster_status", "producer_job_status", "producer_publish_manifest"}
        }
        require(
            required_relative == set(manifest_rows)
            and manifest.get("copied_bytes") == sum(row["bytes"] for row in manifest_rows.values()),
            "publish manifest omitted, added, or miscounted a finalizer artifact",
        )
        for name in ("evidence_receipt", "worker_attestation"):
            git_path = expected_paths[name]
            blob = _git_blob(snapshot, publication_commit, git_path)
            relative = Path(git_path).relative_to(
                f"results/jobs/{finalizer_job_id}/publish"
            ).as_posix()
            manifest_row = manifest_rows[relative]
            require(
                manifest_row["bytes"] == len(blob)
                and manifest_row["sha256"] == sha256_bytes(blob),
                f"publish manifest differs from {name} bytes",
            )
            artifact_blobs[name] = blob
            remote_descriptors[name] = _remote_blob_descriptor(
                publication_commit, git_path, blob
            )
        for name, git_path in expected_paths.items():
            if name in {"cluster_status", "producer_job_status", "producer_publish_manifest"}:
                continue
            relative = Path(git_path).relative_to(
                f"results/jobs/{finalizer_job_id}/publish"
            ).as_posix()
            blob = (
                publication_pending_blob if name == "pending"
                else artifact_blobs["evidence_receipt"] if name == "evidence_receipt"
                else artifact_blobs["worker_attestation"] if name == "worker_attestation"
                else artifact_blobs[name]
            )
            require(
                manifest_rows[relative]["bytes"] == len(blob)
                and manifest_rows[relative]["sha256"] == sha256_bytes(blob),
                f"publish manifest differs from published artifact: {name}",
            )
        evidence_receipt = _parse_json_blob(
            artifact_blobs["evidence_receipt"], "published release evidence receipt"
        )
        _validate_published_evidence_receipt(
            value=evidence_receipt, pending=pending,
            pending_blob=publication_pending_blob, source_root=source_root,
            study_commit=study_commit,
        )
        worker_attestation = _parse_json_blob(
            artifact_blobs["worker_attestation"], "published finalizer worker attestation"
        )
        _validate_remote_finalizer_attestation(
            value=worker_attestation, source_root=source_root,
            study_commit=study_commit, pending=pending,
            reconciliation=reconciliation,
        )
        parsed_artifacts["evidence_receipt"] = evidence_receipt
        parsed_artifacts["worker_attestation"] = worker_attestation
        remote_descriptors["producer_publish_manifest"] = _remote_blob_descriptor(
            publication_commit, expected_paths["producer_publish_manifest"], manifest_blob
        )
        publication_changes = commit_changes[publication_commit]
        prior_ids = set(reconciliation["prior_inventory"]["all_prior_job_ids"])
        for status_code, path in publication_changes:
            if path == status_path:
                continue
            require(status_code == "A", "coordinator rewrote an immutable published path")
            allowed = path.startswith(f"results/{NAMESPACE}/jobs/") or any(
                path.startswith(f"results/jobs/{job_id}/") for job_id in prior_ids
            )
            require(allowed, "publication commit added a path outside reconciled job inventory")
        latest_status_blob = _git_blob(snapshot, initial_head, expected_paths["cluster_status"])
        latest_status = _parse_json_blob(latest_status_blob, "latest published cluster status")
        latest_producer = _validate_published_status(
            status=latest_status, pending=pending, finalizer_id=finalizer_job_id,
            reconciliation=reconciliation, queue=queue, source_root=source_root,
        )
        require(
            latest_status["control"]["control_generation"]
            >= status["control"]["control_generation"]
            and latest_producer == producer_result,
            "later coordinator status regressed generation or rewrote finalizer result",
        )
        verified_at_utc = utc_now()
        now = parse_utc(verified_at_utc, "publication verification time")
        created = parse_utc(pending["created_at_utc"], "pending creation time")
        consume_by = parse_utc(pending["consume_by_utc"], "pending consume-by time")
        require(
            0 <= (now - created).total_seconds()
            <= contract["queue"]["maximum_attestation_age_seconds"]
            and now <= consume_by,
            "pending release plan is stale or future-dated",
        )
    require(
        _remote_head(source_root, remote_url, results_ref) == initial_head,
        "results remote moved during publication verification",
    )
    after_status = _git(
        source_root, "status", "--porcelain=v1", "--untracked-files=all", text=True
    ).stdout
    require(after_status == before_status, "read-only verifier changed the study worktree")
    document = signed_document({
        "schema_version": VERIFICATION_SCHEMA,
        "status": "coordinator_publication_verified_read_only",
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "study_commit": study_commit,
        "verified_at_utc": verified_at_utc,
        "consume_by_utc": pending["consume_by_utc"],
        "results_remote": contract["queue"]["repository_identity"],
        "results_ref": results_ref,
        "observed_results_ancestor_commit": ancestor,
        "pending_publication_commit": publication_commit,
        "verified_remote_head": initial_head,
        "intervening_commit_count": publication_index,
        "later_status_only_commit_count": len(commits) - publication_index - 1,
        "finalizer_job_id": finalizer_job_id,
        "artifact_descriptors": remote_descriptors,
        "producer_outer_result": producer_result,
        "semantic_control": dict(latest_status["control"]),
        "queue_fragment_release_gate_passed": True,
        "results_branch_written_by_verifier": False,
        "study_worktree_mutated": False,
        "claim_boundary": (
            "Read-only verification of the existing GM coordinator publication. This receipt "
            "does not itself stage the already-built confirmation queue fragment."
        ),
    })
    return {
        "document": document,
        "publication_commit": publication_commit,
        "verified_remote_head": initial_head,
        "artifact_descriptors": remote_descriptors,
        "cluster_status": status,
        "producer_result": producer_result,
        "artifacts": parsed_artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    attestation_queue = subparsers.add_parser("build-attestation-queue")
    attestation_queue.add_argument("--source-root", type=Path, required=True)
    attestation_queue.add_argument("--study-commit", required=True)
    attestation_queue.add_argument("--lane", choices=("N3", "D1"), required=True)
    attestation_queue.add_argument("--attempt-number", type=int, required=True)
    attestation_queue.add_argument("--worker-id", action="append", required=True)
    attestation_queue.add_argument("--output", type=Path, required=True)

    attest = subparsers.add_parser("attest-worker")
    attest.add_argument("--source-root", type=Path, required=True)
    attest.add_argument("--state-dir", type=Path, required=True)
    attest.add_argument("--job-dir", type=Path, required=True)
    attest.add_argument("--job-id", required=True)
    attest.add_argument("--study-commit", required=True)
    attest.add_argument("--expected-worker-id", required=True)
    attest.add_argument("--expected-finalizer-job-id", required=True)

    finalizer_queue = subparsers.add_parser("build-finalizer-job")
    finalizer_queue.add_argument("--source-root", type=Path, required=True)
    finalizer_queue.add_argument("--study-commit", required=True)
    finalizer_queue.add_argument("--lane", choices=("N3", "D1"), required=True)
    finalizer_queue.add_argument("--attempt-number", type=int, required=True)
    finalizer_queue.add_argument("--inputs", type=Path, required=True)
    finalizer_queue.add_argument("--inputs-sha256", required=True)
    finalizer_queue.add_argument("--output", type=Path, required=True)

    finalizer = subparsers.add_parser("finalize-pending")
    finalizer.add_argument("--source-root", type=Path, required=True)
    finalizer.add_argument("--state-dir", type=Path, required=True)
    finalizer.add_argument("--job-dir", type=Path, required=True)
    finalizer.add_argument("--job-id", required=True)
    finalizer.add_argument("--study-commit", required=True)
    finalizer.add_argument("--expected-lane", choices=("N3", "D1"), required=True)
    finalizer.add_argument("--inputs", type=Path, required=True)
    finalizer.add_argument("--inputs-sha256", required=True)

    verify = subparsers.add_parser("verify-published-finalizer")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--finalizer-job-id", required=True)
    verify.add_argument("--study-commit", required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actual_runtime_argv = sys.argv if argv is None else None
    try:
        if args.mode == "build-attestation-queue":
            value = build_attestation_queue_fragment(
                source_root=args.source_root, study_commit=args.study_commit,
                lane=args.lane, attempt_number=args.attempt_number,
                worker_ids=sorted(args.worker_id),
            )
            output_path = _lexical_absolute_path(args.output, "descriptor output")
            identity = immutable_write(
                output_path, value,
                anchor=_existing_directory(output_path.parent, "descriptor output parent"),
            )
            result = {"status": "descriptor_only_not_staged", "output": identity}
        elif args.mode == "attest-worker":
            value = attest_worker(
                source_root=args.source_root, state_dir=args.state_dir,
                job_dir=args.job_dir, job_id=args.job_id,
                study_commit=args.study_commit,
                expected_worker_id=args.expected_worker_id,
                expected_finalizer_job_id=args.expected_finalizer_job_id,
                runtime_argv=actual_runtime_argv,
            )
            attestation_path = (
                _lexical_absolute_path(Path(args.job_dir), "worker job directory")
                / "publish" / "worker_attestation.json"
            )
            attestation_identity, observed_attestation = load_json_with_identity(
                attestation_path, "worker attestation output",
            )
            require(
                observed_attestation == value,
                "worker attestation output changed after immutable write",
            )
            result = {
                "status": "worker_attested_no_science",
                "output": attestation_identity,
                "worker_id": value["worker_id"],
            }
        elif args.mode == "build-finalizer-job":
            value = build_finalizer_queue_fragment(
                source_root=args.source_root, study_commit=args.study_commit,
                lane=args.lane, attempt_number=args.attempt_number,
                inputs_path=args.inputs, inputs_sha256=args.inputs_sha256,
            )
            output_path = _lexical_absolute_path(args.output, "descriptor output")
            identity = immutable_write(
                output_path, value,
                anchor=_existing_directory(output_path.parent, "descriptor output parent"),
            )
            result = {"status": "descriptor_only_not_staged", "output": identity}
        elif args.mode == "finalize-pending":
            outputs = finalize_pending(
                source_root=args.source_root, state_dir=args.state_dir,
                job_dir=args.job_dir, job_id=args.job_id,
                study_commit=args.study_commit, expected_lane=args.expected_lane,
                inputs_path=args.inputs, inputs_sha256=args.inputs_sha256,
                runtime_argv=actual_runtime_argv,
            )
            result = {"status": "pending_coordinator_publication_no_science", "outputs": outputs}
        else:
            verified = verify_published_finalizer(
                source_root=args.source_root,
                finalizer_job_id=args.finalizer_job_id,
                study_commit=args.study_commit,
            )
            output_path = _lexical_absolute_path(args.output, "verification output")
            identity = immutable_write(
                output_path, verified["document"],
                anchor=_existing_directory(output_path.parent, "verification output parent"),
            )
            result = {
                "status": "coordinator_publication_verified_read_only",
                "output": identity,
                "publication_commit": verified["publication_commit"],
                "verified_remote_head": verified["verified_remote_head"],
            }
    except ConfirmationReleaseEvidenceError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
