#!/usr/bin/env python3
"""Publish authenticated confirmation terminal-runtime source identities.

This is a source/provenance producer, not an execution launcher.  Its contract
fixes the only five terminal-context runtime files and the only five historical
prerequisite receipts that may be published.  The program authenticates an
exact, stable head of the authorized public control branch in a disposable Git
graph, compares every local runtime byte with both that Git blob and the frozen
contract hash, and reads every prerequisite through no-follow file descriptors.

The single output is written immutably and atomically.  No caller can provide a
runtime path, runtime digest, prerequisite path, or prerequisite digest.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

STUDY_ID = "WMF-ABLATION-001"
NAMESPACE = "wmf_ablation_001_20260912"
OUTPUT_SCHEMA = "wmf-confirmation-terminal-runtime-identities-v1"
CONTRACT_SCHEMA = "wmf-confirmation-terminal-runtime-identities-contract-v1"
OUTPUT_STATUS = "published_terminal_context_runtime"
CONTRACT_STATUS = "frozen_for_final_published_control_runtime"
AUTHORIZED_REPOSITORY_URL = "https://github.com/adeeb10abbas/steerable.git"
AUTHORIZED_CONTROL_BRANCH = "codex/forecast-layout-gm-20260912"
AUTHORIZED_CONTROL_REF = f"refs/heads/{AUTHORIZED_CONTROL_BRANCH}"
AUTHENTICATED_GRAPH_REF = "refs/wmf-terminal-runtime-auth/control"
GIT_EXECUTABLE_CANDIDATES = ("/usr/bin/git", "/usr/local/bin/git")
AUTHENTICATION_TIMEOUT_SECONDS = 60
AUTHENTICATED_BLOB_LIMIT_BYTES = 1024 * 1024
MAX_AUTHENTICATED_GRAPH_BYTES = 192 * 1024 * 1024
ISOLATED_REPOSITORY_CONFIG = (
    b"[core]\n"
    b"\trepositoryformatversion = 0\n"
    b"\tbare = true\n"
)
ISOLATED_REPOSITORY_CONFIG_SHA256 = hashlib.sha256(
    ISOLATED_REPOSITORY_CONFIG
).hexdigest()
NETWORK_ENVIRONMENT_KEYS = frozenset(
    {
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE",
    }
)

FORECAST_PREFIX = "workshops/corl2026_world_models/experiments/forecast_layout/"
PRODUCER_RELATIVE = FORECAST_PREFIX + "confirmation_terminal_runtime_identities.py"
CONTRACT_RELATIVE = (
    FORECAST_PREFIX + "confirmation_terminal_runtime_identities_contract.json"
)
RUNTIME_FILES = {
    "n3_behavioral_pilot_job.py": FORECAST_PREFIX + "n3_behavioral_pilot_job.py",
    "n3_confirmation_block_job.py": FORECAST_PREFIX + "n3_confirmation_block_job.py",
    "d1_behavioral_pilot_jobs.py": FORECAST_PREFIX + "d1_behavioral_pilot_jobs.py",
    "d1_confirmation_block_jobs.py": FORECAST_PREFIX + "d1_confirmation_block_jobs.py",
    "d1_instrumented_server.py": FORECAST_PREFIX + "d1_instrumented_server.py",
}
PREREQUISITE_NAMES = {
    "n3_pilot_receipt",
    "recorder_receipt",
    "d1_qualification_receipt",
    "d1_pilot_simulator_receipt",
    "d1_pilot_server_receipt",
}
TERMINAL_CONTEXT_SCHEMAS = {
    "N3": "wmf-n3-terminal-context-receipt-v1",
    "D1": "wmf-d1-terminal-context-receipt-v1",
}
CLAIM_BOUNDARY = (
    "Published Git blob identities and immutable prerequisite receipt descriptors only; "
    "no machine runtime identity, execution release, queue mutation, model request, "
    "simulator action, behavioral result, or scientific claim is created."
)

CONTRACT_KEYS = {
    "schema_version", "status", "study_id", "namespace",
    "authorized_control_source", "runtime_files", "terminal_context_schemas",
    "prerequisite_receipts", "output", "claim_boundary", "prohibitions",
}
CONTROL_KEYS = {
    "repository_url", "branch", "ref", "commit_requirement",
    "read_transport", "git_executable_candidates", "blob_limit_bytes",
    "graph_limit_bytes",
}
RUNTIME_CONTRACT_KEYS = {"path", "sha256"}
PREREQUISITE_CONTRACT_KEYS = {"path", "bytes", "sha256", "expected_fields"}
OUTPUT_CONTRACT_KEYS = {"schema_version", "status", "exact_fields"}
OUTPUT_KEYS = {
    "schema_version", "status", "study_id", "namespace", "study_commit",
    "published_at_utc", "runtime_files", "terminal_context_schemas",
    "prerequisite_receipts", "claim_boundary", "payload_sha256",
}
DESCRIPTOR_KEYS = {"path", "bytes", "sha256"}
RUNTIME_OUTPUT_KEYS = {"path", "sha256"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_OUTPUT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
SAFE_STAGING_RE = re.compile(r"\.wmf-terminal-runtime-[0-9a-f]{24}\.tmp\Z")

PACKAGE = Path(__file__).resolve().parents[2]
REPOSITORY = PACKAGE.parents[1]
CONTRACT_PATH = Path(__file__).with_name(
    "confirmation_terminal_runtime_identities_contract.json"
)


class TerminalRuntimeIdentityError(RuntimeError):
    """A stable fail-closed identity-publication error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalRuntimeIdentityError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    require(not missing, f"{label} missing keys: {sorted(missing)}")
    require(not extra, f"{label} has disallowed keys: {sorted(extra)}")


def _reject_constant(value: str) -> None:
    raise TerminalRuntimeIdentityError(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalRuntimeIdentityError(
            f"{label} is not strict UTF-8 JSON: {error}"
        ) from error
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _finite_float(value: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite JSON number is prohibited: {value}")
    return parsed


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TerminalRuntimeIdentityError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TerminalRuntimeIdentityError(f"value is not finite JSON: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def payload_hash(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("payload_sha256", None)
    return sha256_bytes(canonical_bytes(unsigned))


def sign_document(document: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in document, "document is already signed")
    result = dict(document)
    result["payload_sha256"] = payload_hash(result)
    return result


def verify_signed(document: Mapping[str, Any], label: str) -> None:
    digest = document.get("payload_sha256")
    require(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
        f"{label} lacks a lowercase SHA-256 payload signature",
    )
    require(payload_hash(document) == digest, f"{label} payload signature changed")


def _directory_flags() -> int:
    require(
        hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"),
        "no-follow descriptor traversal is unavailable",
    )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_directory_nofollow(path: Path, label: str) -> tuple[int, Path]:
    """Open an existing absolute directory component by component."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = _directory_flags()
    try:
        descriptor = os.open(os.sep, flags)
    except OSError:
        raise TerminalRuntimeIdentityError(
            f"cannot open filesystem root for {label}"
        ) from None
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError:
                raise TerminalRuntimeIdentityError(
                    f"{label} is missing, not a directory, or contains a symlink"
                ) from None
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute
    except Exception:
        os.close(descriptor)
        raise


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_same_open_directory(path: Path, descriptor: int, label: str) -> None:
    comparison, _ = _open_directory_nofollow(path, label)
    try:
        first = os.fstat(descriptor)
        second = os.fstat(comparison)
        require(
            (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino),
            f"{label} changed after it was opened",
        )
    finally:
        os.close(comparison)


def _read_all(descriptor: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            block = os.read(descriptor, 1024 * 1024)
        except OSError:
            raise TerminalRuntimeIdentityError(f"cannot read {label}") from None
        if not block:
            break
        chunks.append(block)
        total += len(block)
        require(total <= AUTHENTICATED_BLOB_LIMIT_BYTES, f"{label} exceeds the byte limit")
    return b"".join(chunks)


def _read_regular_file(
    path: Path, label: str, *, expected_size: int | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Read one path from a held parent and prove the same inode remains named."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    require(absolute.name not in {"", ".", ".."}, f"{label} path is unsafe")
    parent_fd, parent = _open_directory_nofollow(absolute.parent, f"{label} parent")
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(absolute.name, flags, dir_fd=parent_fd)
        except OSError:
            raise TerminalRuntimeIdentityError(
                f"{label} is missing, not regular, or is a symlink"
            ) from None
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"{label} is not a singly linked regular file",
        )
        if expected_size is not None:
            require(
                type(expected_size) is int and expected_size >= 0
                and before.st_size == expected_size,
                f"{label} byte count changed before read",
            )
        payload = _read_all(descriptor, label)
        after = os.fstat(descriptor)
        require(
            _stat_identity(before) == _stat_identity(after)
            and after.st_nlink == 1 and len(payload) == after.st_size,
            f"{label} changed while it was read",
        )
        _require_same_open_directory(parent, parent_fd, f"{label} parent")
        try:
            named = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            raise TerminalRuntimeIdentityError(f"{label} path changed after read") from None
        require(
            stat.S_ISREG(named.st_mode)
            and named.st_nlink == 1
            and (named.st_dev, named.st_ino) == (after.st_dev, after.st_ino),
            f"{label} path changed after read",
        )
        return payload, {
            "path": str(absolute),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _safe_relative_path(value: Any, expected: str, label: str) -> str:
    require(isinstance(value, str) and value == expected, f"{label} path changed")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"{label} path is unsafe")
    return value


def _validate_contract(value: Mapping[str, Any], *, require_frozen: bool = True) -> None:
    exact_keys(value, CONTRACT_KEYS, "terminal runtime identity contract")
    require(
        value.get("schema_version") == CONTRACT_SCHEMA
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE,
        "terminal runtime identity contract identity changed",
    )
    if require_frozen:
        require(
            value.get("status") == CONTRACT_STATUS,
            "terminal runtime identity contract is not frozen for final published runtime bytes",
        )

    control = value.get("authorized_control_source")
    exact_keys(control, CONTROL_KEYS, "authorized control source")
    require(
        control.get("repository_url") == AUTHORIZED_REPOSITORY_URL
        and control.get("branch") == AUTHORIZED_CONTROL_BRANCH
        and control.get("ref") == AUTHORIZED_CONTROL_REF
        and control.get("commit_requirement")
        == "source_commit_is_ancestor_of_stable_remote_control_head"
        and control.get("read_transport") == "literal_public_https_isolated_bare"
        and control.get("git_executable_candidates") == list(GIT_EXECUTABLE_CANDIDATES)
        and control.get("blob_limit_bytes") == AUTHENTICATED_BLOB_LIMIT_BYTES
        and control.get("graph_limit_bytes") == MAX_AUTHENTICATED_GRAPH_BYTES,
        "authorized control source changed",
    )

    runtime = value.get("runtime_files")
    require(
        isinstance(runtime, Mapping) and set(runtime) == set(RUNTIME_FILES),
        "terminal runtime file inventory changed",
    )
    for name, expected_path in RUNTIME_FILES.items():
        row = runtime.get(name)
        exact_keys(row, RUNTIME_CONTRACT_KEYS, f"runtime contract {name}")
        _safe_relative_path(row.get("path"), expected_path, f"runtime contract {name}")
        digest = row.get("sha256")
        require(
            isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"runtime contract {name} is not frozen to a SHA-256",
        )

    require(
        value.get("terminal_context_schemas") == TERMINAL_CONTEXT_SCHEMAS,
        "terminal context schema contract changed",
    )
    prerequisites = value.get("prerequisite_receipts")
    require(
        isinstance(prerequisites, Mapping)
        and set(prerequisites) == PREREQUISITE_NAMES,
        "prerequisite receipt inventory changed",
    )
    paths: set[str] = set()
    for name in sorted(PREREQUISITE_NAMES):
        row = prerequisites.get(name)
        exact_keys(row, PREREQUISITE_CONTRACT_KEYS, f"prerequisite contract {name}")
        path = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        expected = row.get("expected_fields")
        require(
            isinstance(path, str) and Path(path).is_absolute()
            and ".." not in Path(path).parts and path not in paths,
            f"prerequisite contract {name} path is invalid or duplicated",
        )
        paths.add(path)
        require(type(size) is int and size > 0, f"prerequisite contract {name} bytes are invalid")
        require(
            isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"prerequisite contract {name} SHA-256 is invalid",
        )
        require(
            isinstance(expected, Mapping) and expected
            and all(isinstance(key, str) and key for key in expected)
            and all(
                item is None or isinstance(item, (str, int, bool))
                for item in expected.values()
            ),
            f"prerequisite contract {name} expected fields are invalid",
        )

    output = value.get("output")
    exact_keys(output, OUTPUT_CONTRACT_KEYS, "terminal runtime output contract")
    require(
        output.get("schema_version") == OUTPUT_SCHEMA
        and output.get("status") == OUTPUT_STATUS
        and output.get("exact_fields") == sorted(OUTPUT_KEYS),
        "terminal runtime output contract changed",
    )
    require(value.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary changed")
    prohibitions = value.get("prohibitions")
    expected_prohibitions = {
        "caller_selected_runtime_identity": False,
        "caller_selected_prerequisite_identity": False,
        "machine_runtime_identity_created": False,
        "queue_mutated": False,
        "job_released_or_dispatched": False,
        "model_or_simulator_started": False,
        "science_or_labels_created": False,
    }
    require(prohibitions == expected_prohibitions, "terminal runtime prohibitions changed")


def _validate_timestamp(value: str) -> None:
    require(isinstance(value, str) and value.endswith("Z"), "publication time is not UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise TerminalRuntimeIdentityError("publication time is invalid") from error
    require(parsed.utcoffset() == timezone.utc.utcoffset(parsed), "publication time is not UTC")


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key in NETWORK_ENVIRONMENT_KEYS
    }
    environment.update(
        {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def _validate_git_candidate(raw: str) -> dict[str, str]:
    require(raw in GIT_EXECUTABLE_CANDIDATES, "Git executable path is not pinned")
    current = Path(raw)
    invoked = current
    trusted = {Path("/usr/bin"), Path("/usr/local/bin")}
    seen: set[Path] = set()
    for _ in range(8):
        require(current not in seen, "Git executable symlink chain contains a cycle")
        seen.add(current)
        require(current.is_absolute() and current.parent in trusted, "Git executable leaves trusted directories")
        try:
            parent_stat = current.parent.lstat()
            metadata = current.lstat()
        except OSError:
            raise TerminalRuntimeIdentityError("pinned Git executable is unavailable") from None
        require(
            current.parent.is_dir() and not current.parent.is_symlink()
            and parent_stat.st_uid == 0 and parent_stat.st_mode & 0o022 == 0,
            "Git system directory is not root-owned and protected",
        )
        is_link = stat.S_ISLNK(metadata.st_mode)
        require(
            metadata.st_uid == 0 and (is_link or metadata.st_mode & 0o022 == 0),
            "Git executable is not root-owned and protected",
        )
        if not is_link:
            require(stat.S_ISREG(metadata.st_mode), "Git executable is not regular")
            return {"invoked_path": str(invoked), "resolved_path": str(current)}
        try:
            target = Path(os.readlink(current))
        except OSError:
            raise TerminalRuntimeIdentityError("cannot inspect Git executable symlink") from None
        current = target if target.is_absolute() else current.parent / target
        current = Path(os.path.normpath(current))
        require(str(current) in GIT_EXECUTABLE_CANDIDATES, "Git executable symlink target is not pinned")
    raise TerminalRuntimeIdentityError("Git executable symlink chain is too deep")


def _git_executable() -> dict[str, str]:
    for candidate in GIT_EXECUTABLE_CANDIDATES:
        try:
            return _validate_git_candidate(candidate)
        except TerminalRuntimeIdentityError:
            continue
    raise TerminalRuntimeIdentityError("no protected pinned Git executable is available")


def _git_run(
    repository: Path,
    arguments: Sequence[str],
    label: str,
    *,
    protocol: str,
    allowed_returncodes: Sequence[int] = (0,),
) -> subprocess.CompletedProcess[bytes]:
    require(protocol in {"https", "file"}, "isolated Git protocol is not bounded")
    executable = _git_executable()["invoked_path"]
    configuration = [
        "-c", "credential.helper=",
        "-c", "credential.interactive=never",
        "-c", "core.askPass=/bin/false",
        "-c", "core.sshCommand=",
        "-c", "http.extraHeader=",
        "-c", "http.followRedirects=false",
        "-c", "http.sslVerify=true",
        "-c", "protocol.allow=never",
        "-c", f"protocol.{protocol}.allow=always",
    ]
    try:
        completed = subprocess.run(
            [executable, *configuration, *arguments],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=AUTHENTICATION_TIMEOUT_SECONDS,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        raise TerminalRuntimeIdentityError(
            f"cannot authenticate {label}; isolated Git execution failed"
        ) from None
    require(completed.returncode in allowed_returncodes, f"cannot authenticate {label}")
    return completed


def _git_bytes(repository: Path, arguments: Sequence[str], label: str, *, protocol: str) -> bytes:
    return _git_run(repository, arguments, label, protocol=protocol).stdout


def _write_graph_config(repository: Path) -> None:
    config = repository / "config"
    require(
        repository.is_dir() and not repository.is_symlink()
        and config.is_file() and not config.is_symlink(),
        "isolated authentication repository is unsafe",
    )
    try:
        with config.open("wb") as handle:
            handle.write(ISOLATED_REPOSITORY_CONFIG)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        raise TerminalRuntimeIdentityError("cannot fix isolated Git configuration") from None


def _validate_graph_config(repository: Path) -> None:
    config = repository / "config"
    try:
        payload = config.read_bytes()
    except OSError:
        raise TerminalRuntimeIdentityError("cannot inspect isolated Git configuration") from None
    require(
        config.is_file() and not config.is_symlink()
        and payload == ISOLATED_REPOSITORY_CONFIG,
        "isolated Git configuration drifted",
    )


def _graph_size(repository: Path) -> int:
    total = 0
    try:
        for path in repository.rglob("*"):
            require(not path.is_symlink(), "isolated Git graph contains a symlink")
            if path.is_file():
                total += path.stat().st_size
                require(total <= MAX_AUTHENTICATED_GRAPH_BYTES, "isolated Git graph exceeds its byte limit")
    except OSError:
        raise TerminalRuntimeIdentityError("cannot measure isolated Git graph") from None
    return total


def _parse_remote_ref(payload: bytes, expected_ref: str, label: str) -> str:
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        raise TerminalRuntimeIdentityError(f"{label} is not UTF-8") from None
    require(len(lines) == 1, f"{label} did not return exactly one ref")
    fields = lines[0].split("\t")
    require(
        len(fields) == 2 and COMMIT_RE.fullmatch(fields[0]) is not None
        and fields[1] == expected_ref,
        f"{label} returned a different or invalid ref",
    )
    return fields[0]


def _observe_remote(repository: Path, remote_url: str, protocol: str) -> str:
    _validate_graph_config(repository)
    return _parse_remote_ref(
        _git_bytes(
            repository,
            ["ls-remote", "--exit-code", "--refs", remote_url, AUTHORIZED_CONTROL_REF],
            "authorized control ref observation",
            protocol=protocol,
        ),
        AUTHORIZED_CONTROL_REF,
        "authorized control ref observation",
    )


@contextlib.contextmanager
def _authenticated_graph(
    source_commit: str,
    *,
    remote_url: str,
    protocol: str,
) -> Iterable[tuple[Path, dict[str, Any]]]:
    require(
        isinstance(source_commit, str) and COMMIT_RE.fullmatch(source_commit) is not None,
        "source commit is not a full Git object ID",
    )
    if protocol == "https":
        require(remote_url == AUTHORIZED_REPOSITORY_URL, "authorized repository URL changed")
    with tempfile.TemporaryDirectory(prefix="wmf-terminal-runtime-auth-", dir="/tmp") as temporary:
        root = Path(temporary)
        graph = root / "graph.git"
        _git_bytes(
            root,
            ["init", "--quiet", "--bare", "--template=", str(graph)],
            "isolated bare repository initialization",
            protocol=protocol,
        )
        _write_graph_config(graph)
        before = _observe_remote(graph, remote_url, protocol)
        fetch = [
            "fetch", "--quiet", "--no-tags", "--no-recurse-submodules",
            "--refmap=",
        ]
        if protocol == "https":
            fetch.append(f"--filter=blob:limit={AUTHENTICATED_BLOB_LIMIT_BYTES}")
        fetch.extend([remote_url, f"+{AUTHORIZED_CONTROL_REF}:{AUTHENTICATED_GRAPH_REF}"])
        _git_bytes(graph, fetch, "authorized exact-ref fetch", protocol=protocol)
        _write_graph_config(graph)
        _validate_graph_config(graph)
        fetched = _git_bytes(
            graph,
            ["rev-parse", f"{AUTHENTICATED_GRAPH_REF}^{{commit}}"],
            "authenticated control commit",
            protocol=protocol,
        ).decode("ascii", errors="strict").strip()
        after = _observe_remote(graph, remote_url, protocol)
        require(
            COMMIT_RE.fullmatch(fetched) is not None and before == fetched == after,
            "authorized control head changed during authentication",
        )
        _git_bytes(
            graph,
            ["cat-file", "-e", f"{source_commit}^{{commit}}"],
            "published immutable source commit",
            protocol=protocol,
        )
        ancestry = _git_run(
            graph,
            ["merge-base", "--is-ancestor", source_commit, fetched],
            "source-to-control ancestry",
            protocol=protocol,
            allowed_returncodes=(0, 1),
        )
        require(
            ancestry.returncode == 0,
            "immutable source commit is not an ancestor of the stable published control head",
        )
        executable = _git_executable()
        version = _git_bytes(graph, ["--version"], "isolated Git version", protocol=protocol)
        try:
            version_text = version.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError:
            raise TerminalRuntimeIdentityError("isolated Git version is invalid") from None
        require(
            re.fullmatch(r"git version [0-9][0-9A-Za-z.+-]*", version_text) is not None,
            "isolated Git version is invalid",
        )
        graph_bytes = _graph_size(graph)
        yield graph, {
            "repository_url": AUTHORIZED_REPOSITORY_URL if protocol == "https" else remote_url,
            "control_ref": AUTHORIZED_CONTROL_REF,
            "source_commit": source_commit,
            "remote_ref_commit": fetched,
            "read_transport": "literal_public_https_isolated_bare" if protocol == "https" else "test_file_isolated_bare",
            "git_executable_path": executable["invoked_path"],
            "git_executable_resolved_path": executable["resolved_path"],
            "git_version": version_text,
            "isolated_repository_config_sha256": ISOLATED_REPOSITORY_CONFIG_SHA256,
            "authenticated_graph_bytes": graph_bytes,
        }
        require(_observe_remote(graph, remote_url, protocol) == fetched, "control head drifted")
        _validate_graph_config(graph)
        _graph_size(graph)


def _git_blob(graph: Path, commit: str, relative: str, *, protocol: str) -> bytes:
    tree = _git_bytes(
        graph,
        ["ls-tree", "-z", commit, "--", relative],
        f"Git tree entry {relative}",
        protocol=protocol,
    )
    entries = [entry for entry in tree.split(b"\0") if entry]
    require(len(entries) == 1, f"Git tree entry changed: {relative}")
    try:
        prefix, named = entries[0].split(b"\t", 1)
        mode, object_type, object_id = prefix.decode("ascii").split(" ")
        name = named.decode("utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError):
        raise TerminalRuntimeIdentityError(f"Git tree entry is invalid: {relative}") from None
    require(
        mode in {"100644", "100755"} and object_type == "blob"
        and re.fullmatch(r"[0-9a-f]{40,64}", object_id) is not None
        and name == relative,
        f"Git tree entry is not a regular tracked blob: {relative}",
    )
    payload = _git_bytes(
        graph,
        ["show", f"{commit}:{relative}"],
        f"Git blob {relative}",
        protocol=protocol,
    )
    require(len(payload) <= AUTHENTICATED_BLOB_LIMIT_BYTES, f"Git blob is too large: {relative}")
    return payload


def _source_path(source_root: Path, relative: str, label: str) -> Path:
    _safe_relative_path(relative, relative, label)
    root = Path(os.path.abspath(os.fspath(source_root)))
    candidate = root.joinpath(*Path(relative).parts)
    require(candidate != root and root in candidate.parents, f"{label} escapes source root")
    return candidate


def _validate_source_and_evidence(
    *,
    source_root: Path,
    source_commit: str,
    graph: Path,
    protocol: str,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(source_root)))
    root_fd, root_path = _open_directory_nofollow(root, "source root")
    try:
        _require_same_open_directory(root_path, root_fd, "source root")
    finally:
        os.close(root_fd)

    producer_path = _source_path(root, PRODUCER_RELATIVE, "producer source")
    contract_path = _source_path(root, CONTRACT_RELATIVE, "producer contract")
    require(
        Path(__file__).resolve(strict=True) == producer_path.resolve(strict=True)
        and CONTRACT_PATH.resolve(strict=True) == contract_path.resolve(strict=True),
        "executing producer or contract path differs from the authenticated source root",
    )
    producer_git_payload = _git_blob(
        graph, source_commit, PRODUCER_RELATIVE, protocol=protocol
    )
    contract_git_payload = _git_blob(
        graph, source_commit, CONTRACT_RELATIVE, protocol=protocol
    )
    producer_payload, producer_identity = _read_regular_file(
        producer_path, "producer source", expected_size=len(producer_git_payload)
    )
    contract_payload, contract_identity = _read_regular_file(
        contract_path, "producer contract", expected_size=len(contract_git_payload)
    )
    require(
        producer_payload == producer_git_payload,
        "producer source differs from the final published Git blob",
    )
    require(
        contract_payload == contract_git_payload,
        "producer contract differs from the final published Git blob",
    )
    contract = load_json_bytes(contract_payload, "terminal runtime identity contract")
    _validate_contract(contract)

    runtime_output: dict[str, dict[str, str]] = {}
    runtime_state: dict[str, dict[str, Any]] = {}
    for name, relative in RUNTIME_FILES.items():
        git_payload = _git_blob(graph, source_commit, relative, protocol=protocol)
        payload, identity = _read_regular_file(
            _source_path(root, relative, f"runtime {name}"),
            f"runtime {name}",
            expected_size=len(git_payload),
        )
        expected = contract["runtime_files"][name]
        require(payload == git_payload, f"runtime {name} differs from final published Git blob")
        require(identity["sha256"] == expected["sha256"], f"runtime {name} differs from frozen hash")
        runtime_output[name] = {"path": relative, "sha256": identity["sha256"]}
        runtime_state[name] = identity

    prerequisite_output: dict[str, dict[str, Any]] = {}
    prerequisite_state: dict[str, dict[str, Any]] = {}
    for name in sorted(PREREQUISITE_NAMES):
        expected = contract["prerequisite_receipts"][name]
        payload, identity = _read_regular_file(
            Path(expected["path"]),
            f"prerequisite {name}",
            expected_size=expected["bytes"],
        )
        require(
            identity == {key: expected[key] for key in ("path", "bytes", "sha256")},
            f"prerequisite {name} descriptor changed",
        )
        receipt = load_json_bytes(payload, f"prerequisite {name}")
        for key, wanted in expected["expected_fields"].items():
            require(receipt.get(key) == wanted, f"prerequisite {name} field changed: {key}")
        prerequisite_output[name] = dict(identity)
        prerequisite_state[name] = identity

    return {
        "contract": contract,
        "producer_identity": producer_identity,
        "contract_identity": contract_identity,
        "runtime_files": runtime_output,
        "runtime_state": runtime_state,
        "prerequisite_receipts": prerequisite_output,
        "prerequisite_state": prerequisite_state,
    }


def _stable_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": value["contract"],
        "producer_identity": value["producer_identity"],
        "contract_identity": value["contract_identity"],
        "runtime_files": value["runtime_files"],
        "runtime_state": value["runtime_state"],
        "prerequisite_receipts": value["prerequisite_receipts"],
        "prerequisite_state": value["prerequisite_state"],
    }


def _validate_receipt(value: Mapping[str, Any], state: Mapping[str, Any], source_commit: str) -> None:
    exact_keys(value, OUTPUT_KEYS, "terminal runtime identities receipt")
    verify_signed(value, "terminal runtime identities receipt")
    _validate_timestamp(value.get("published_at_utc"))
    require(
        value.get("schema_version") == OUTPUT_SCHEMA
        and value.get("status") == OUTPUT_STATUS
        and value.get("study_id") == STUDY_ID
        and value.get("namespace") == NAMESPACE
        and value.get("study_commit") == source_commit
        and value.get("runtime_files") == state["runtime_files"]
        and value.get("terminal_context_schemas") == TERMINAL_CONTEXT_SCHEMAS
        and value.get("prerequisite_receipts") == state["prerequisite_receipts"]
        and value.get("claim_boundary") == CLAIM_BOUNDARY,
        "terminal runtime identities receipt changed or broadened",
    )
    runtime = value.get("runtime_files")
    require(isinstance(runtime, Mapping) and set(runtime) == set(RUNTIME_FILES), "receipt runtime inventory changed")
    for name, row in runtime.items():
        exact_keys(row, RUNTIME_OUTPUT_KEYS, f"receipt runtime {name}")
    prerequisites = value.get("prerequisite_receipts")
    require(
        isinstance(prerequisites, Mapping) and set(prerequisites) == PREREQUISITE_NAMES,
        "receipt prerequisite inventory changed",
    )
    for name, row in prerequisites.items():
        exact_keys(row, DESCRIPTOR_KEYS, f"receipt prerequisite {name}")


def _require_child_absent(parent_fd: int, name: str) -> None:
    require(SAFE_OUTPUT_RE.fullmatch(name) is not None, "output filename is unsafe")
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise TerminalRuntimeIdentityError("cannot inspect output path") from None
    raise TerminalRuntimeIdentityError("refusing to replace terminal runtime identity output")


def _create_staging_file(parent_fd: int) -> tuple[str, int]:
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(32):
        name = f".wmf-terminal-runtime-{secrets.token_hex(12)}.tmp"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        except OSError:
            raise TerminalRuntimeIdentityError("cannot create immutable staging file") from None
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
        require(
            stat.S_ISREG(named.st_mode)
            and (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino),
            "staging file changed while opening",
        )
        return name, descriptor
    raise TerminalRuntimeIdentityError("cannot allocate a unique staging file")


def _named_matches_fd(parent_fd: int, name: str, descriptor: int, label: str) -> None:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise TerminalRuntimeIdentityError(f"{label} path changed") from None
    opened = os.fstat(descriptor)
    require(
        stat.S_ISREG(named.st_mode)
        and (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino),
        f"{label} path changed",
    )


def _write_and_validate_staging(descriptor: int, payload: bytes) -> None:
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            require(written > 0, "staging write made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = _read_all(descriptor, "staged terminal runtime identities")
    except OSError:
        raise TerminalRuntimeIdentityError("cannot persist terminal runtime identities") from None
    require(observed == payload, "staged terminal runtime identities changed")
    load_json_bytes(observed, "staged terminal runtime identities")


def _rename_noreplace(parent_fd: int, source_name: str, target_name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    operation = getattr(libc, "renameat2", None)
    require(operation is not None, "atomic no-replace publication is unavailable")
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    result = operation(
        parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(target_name), 1
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number == errno.EEXIST:
        raise TerminalRuntimeIdentityError("refusing to replace terminal runtime identity output")
    if number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise TerminalRuntimeIdentityError("atomic no-replace publication is unavailable")
    raise TerminalRuntimeIdentityError("cannot atomically publish terminal runtime identities")


def _remove_if_matches(parent_fd: int, name: str, descriptor: int) -> None:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if stat.S_ISREG(named.st_mode) and (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino):
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
    except FileNotFoundError:
        return
    except OSError:
        return


def _revalidate(
    *,
    expected: Mapping[str, Any],
    source_root: Path,
    source_commit: str,
    graph: Path,
    remote_url: str,
    protocol: str,
    control_head: str,
) -> None:
    observed = _validate_source_and_evidence(
        source_root=source_root,
        source_commit=source_commit,
        graph=graph,
        protocol=protocol,
    )
    require(_stable_state(observed) == _stable_state(expected), "source or prerequisite identity drifted")
    require(
        _observe_remote(graph, remote_url, protocol) == control_head,
        "published control head drifted before identity publication",
    )


def _validate_published_impl(
    *,
    receipt_path: Path,
    expected_sha256: str,
    source_root: Path,
    source_commit: str,
    remote_url: str,
    protocol: str,
) -> dict[str, Any]:
    """Deeply validate one published receipt through its authenticated producer."""

    require(
        isinstance(expected_sha256, str)
        and SHA256_RE.fullmatch(expected_sha256) is not None,
        "terminal runtime identities expected SHA-256 is invalid",
    )
    first_payload, first_identity = _read_regular_file(
        Path(receipt_path), "terminal runtime identities receipt"
    )
    require(
        first_identity["sha256"] == expected_sha256,
        "terminal runtime identities receipt SHA-256 changed",
    )
    receipt = load_json_bytes(first_payload, "terminal runtime identities receipt")
    with _authenticated_graph(
        source_commit, remote_url=remote_url, protocol=protocol
    ) as (graph, authentication):
        state = _validate_source_and_evidence(
            source_root=Path(source_root),
            source_commit=source_commit,
            graph=graph,
            protocol=protocol,
        )
        _validate_receipt(receipt, state, source_commit)
        require(
            _observe_remote(graph, remote_url, protocol)
            == authentication["remote_ref_commit"],
            "published control head drifted during receipt validation",
        )
        second_payload, second_identity = _read_regular_file(
            Path(receipt_path), "terminal runtime identities receipt"
        )
        require(
            second_payload == first_payload and second_identity == first_identity,
            "terminal runtime identities receipt drifted during validation",
        )
        second_state = _validate_source_and_evidence(
            source_root=Path(source_root),
            source_commit=source_commit,
            graph=graph,
            protocol=protocol,
        )
        require(
            _stable_state(second_state) == _stable_state(state),
            "source or prerequisite identity drifted during receipt validation",
        )
    return {
        "receipt": receipt,
        "identity": first_identity,
        "prerequisite_receipts": state["prerequisite_receipts"],
    }


def validate_terminal_runtime_identities(
    *,
    receipt_path: Path,
    expected_sha256: str,
    source_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Production native validator for the confirmation release consumer."""

    require(
        AUTHORIZED_REPOSITORY_URL == "https://github.com/adeeb10abbas/steerable.git"
        and AUTHORIZED_CONTROL_REF == "refs/heads/codex/forecast-layout-gm-20260912",
        "production control authority changed",
    )
    return _validate_published_impl(
        receipt_path=receipt_path,
        expected_sha256=expected_sha256,
        source_root=source_root,
        source_commit=source_commit,
        remote_url=AUTHORIZED_REPOSITORY_URL,
        protocol="https",
    )


def _produce_impl(
    *,
    source_root: Path,
    source_commit: str,
    output: Path,
    remote_url: str,
    protocol: str,
    published_at_utc: str | None = None,
    before_rename: Callable[[], None] | None = None,
    after_rename: Callable[[], None] | None = None,
) -> dict[str, Any]:
    absolute_output = Path(os.path.abspath(os.fspath(output)))
    require(
        absolute_output != Path(os.sep)
        and SAFE_OUTPUT_RE.fullmatch(absolute_output.name) is not None,
        "output path is unsafe",
    )
    absolute_source = Path(os.path.abspath(os.fspath(source_root)))
    require(
        absolute_output.parent != absolute_source
        and absolute_source not in absolute_output.parents,
        "identity output must be outside the immutable source checkout",
    )
    parent_fd, parent = _open_directory_nofollow(absolute_output.parent, "output parent")
    staging_name: str | None = None
    staging_fd: int | None = None
    published = False
    try:
        _require_same_open_directory(parent, parent_fd, "output parent")
        _require_child_absent(parent_fd, absolute_output.name)
        with _authenticated_graph(
            source_commit, remote_url=remote_url, protocol=protocol
        ) as (graph, authentication):
            state = _validate_source_and_evidence(
                source_root=absolute_source,
                source_commit=source_commit,
                graph=graph,
                protocol=protocol,
            )
            timestamp = published_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _validate_timestamp(timestamp)
            receipt = sign_document(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    "status": OUTPUT_STATUS,
                    "study_id": STUDY_ID,
                    "namespace": NAMESPACE,
                    "study_commit": source_commit,
                    "published_at_utc": timestamp,
                    "runtime_files": state["runtime_files"],
                    "terminal_context_schemas": TERMINAL_CONTEXT_SCHEMAS,
                    "prerequisite_receipts": state["prerequisite_receipts"],
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            _validate_receipt(receipt, state, source_commit)
            payload = pretty_bytes(receipt)

            _require_same_open_directory(parent, parent_fd, "output parent")
            _require_child_absent(parent_fd, absolute_output.name)
            staging_name, staging_fd = _create_staging_file(parent_fd)
            _write_and_validate_staging(staging_fd, payload)
            _named_matches_fd(parent_fd, staging_name, staging_fd, "staging file")
            _revalidate(
                expected=state,
                source_root=absolute_source,
                source_commit=source_commit,
                graph=graph,
                remote_url=remote_url,
                protocol=protocol,
                control_head=authentication["remote_ref_commit"],
            )
            if before_rename is not None:
                before_rename()
            _require_same_open_directory(parent, parent_fd, "output parent")
            _require_child_absent(parent_fd, absolute_output.name)
            _named_matches_fd(parent_fd, staging_name, staging_fd, "staging file")
            _rename_noreplace(parent_fd, staging_name, absolute_output.name)
            published = True
            os.fsync(parent_fd)
            _named_matches_fd(parent_fd, absolute_output.name, staging_fd, "published output")
            if after_rename is not None:
                after_rename()
            _require_same_open_directory(parent, parent_fd, "output parent")
            _revalidate(
                expected=state,
                source_root=absolute_source,
                source_commit=source_commit,
                graph=graph,
                remote_url=remote_url,
                protocol=protocol,
                control_head=authentication["remote_ref_commit"],
            )
            _named_matches_fd(parent_fd, absolute_output.name, staging_fd, "published output")
            os.lseek(staging_fd, 0, os.SEEK_SET)
            require(
                _read_all(staging_fd, "published terminal runtime identities") == payload,
                "published terminal runtime identities changed",
            )
            return {
                "path": str(absolute_output),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "payload_sha256": receipt["payload_sha256"],
                "study_commit": source_commit,
            }
    except Exception:
        if staging_fd is not None and staging_name is not None:
            _remove_if_matches(
                parent_fd,
                absolute_output.name if published else staging_name,
                staging_fd,
            )
        raise
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


def produce(source_root: Path, source_commit: str, output: Path) -> dict[str, Any]:
    """Production entry point with no caller-controlled trust anchors."""

    require(
        AUTHORIZED_REPOSITORY_URL == "https://github.com/adeeb10abbas/steerable.git"
        and AUTHORIZED_CONTROL_REF == "refs/heads/codex/forecast-layout-gm-20260912",
        "production control authority changed",
    )
    return _produce_impl(
        source_root=source_root,
        source_commit=source_commit,
        output=output,
        remote_url=AUTHORIZED_REPOSITORY_URL,
        protocol="https",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = produce(args.source_root, args.source_commit, args.output)
    except TerminalRuntimeIdentityError as error:
        print(f"terminal_runtime_identity_error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
