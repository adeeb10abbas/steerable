#!/usr/bin/env python3
"""Compile a validated forecast analysis into bounded publication artifacts.

This program is a fail-closed, source-only formatter.  It replays the final
analyzer, the confirmation evidence compiler, and every selected confirmation
fixture before it writes anything.  It does not create labels, recompute a
scientific estimator, choose examples from outcomes, edit a paper, or launch
model/simulator work.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]
FORECAST = PACKAGE / "experiments" / "forecast_layout"
ANALYZER_PATH = Path(__file__).with_name("forecast_evidence_analysis.py")
CONFIRMATION_COMPILER_PATH = Path(__file__).with_name(
    "compile_confirmation_evidence.py"
)
FIXTURE_VALIDATOR_PATH = FORECAST / "confirmation_fixture_freeze.py"
CONTRACT_PATH = FORECAST / "forecast_publication_contract.json"

STUDY_ID = "WMF-ABLATION-001"
INPUT_SCHEMA = "wmf-forecast-publication-input-v1"
PAPER_EVIDENCE_SCHEMA = "wmf-forecast-paper-evidence-v1"
FORECAST_FIGURE_SCHEMA = "wmf-forecast-skill-layout-effects-figure-input-v1"
SCENE_FIGURE_SCHEMA = "wmf-forecast-actual-scene-timing-figure-input-v1"
VIDEO_MANIFEST_SCHEMA = "wmf-forecast-public-example-video-manifest-v1"
BUILD_RECEIPT_SCHEMA = "wmf-forecast-publication-build-receipt-v1"
FINAL_ANALYSIS_SCHEMA = "wmf-forecast-final-analysis-v1"
CONFIRMATION_COMPILER_SCHEMA = "wmf-confirmation-evidence-compiler-receipt-v1"
PRIVATE_VIDEO_SCHEMA = "wmf-confirmation-private-video-inventory-v1"
FIXTURE_FREEZE_SCHEMA = "wmf-confirmation-fixture-freeze-v1"
CLAIM_GATE = (
    "PASSED_HASH_BOUND_RECORDINGS_VALIDATED_DEVELOPMENT_RELEASE_VALIDATED_"
    "ANNOTATION_FREEZE_AND_REPRODUCED_HUMAN_CONSENSUS"
)
BOOTSTRAP_RESAMPLES = 10_000
ANALYSIS_SEED = 2026091301
MAX_SELECTED_VIDEO_BYTES = 16 * 1024 * 1024
MAX_SELECTED_VIDEO_TOTAL_BYTES = 64 * 1024 * 1024
AUTHORIZED_REPOSITORY_URL = "https://github.com/adeeb10abbas/steerable.git"
AUTHORIZED_REPOSITORY_URLS = frozenset(
    {
        AUTHORIZED_REPOSITORY_URL,
        "git@github.com:adeeb10abbas/steerable.git",
        "ssh://git@github.com/adeeb10abbas/steerable.git",
    }
)
AUTHORIZED_PUBLICATION_BRANCH = "codex/forecast-layout-gm-20260912"
AUTHORIZED_PUBLICATION_REF = f"refs/heads/{AUTHORIZED_PUBLICATION_BRANCH}"
AUTHORIZED_REMOTE_ALIASES = ("publish", "origin")
AUTHENTICATED_GRAPH_REF = "refs/wmf-publication-auth/control"
GIT_EXECUTABLE_CANDIDATES = ("/usr/bin/git", "/usr/local/bin/git")
AUTHENTICATED_BLOB_LIMIT_BYTES = 1024 * 1024
MAX_AUTHENTICATED_GRAPH_BYTES = 192 * 1024 * 1024
AUTHENTICATION_TIMEOUT_SECONDS = 60
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
MODELS = ("N3", "D1")
CONDITIONS = (
    "original_left",
    "original_right",
    "reflected_left",
    "reflected_right",
)
BRANCH_MODELS = {
    "full_two_model": ("N3", "D1"),
    "reduced_n3": ("N3",),
    "reduced_d1": ("D1",),
}
LAYOUTS = tuple(f"C{index:02d}" for index in range(1, 25))
ARMS = ("original", "reflected")
MOVABLE_OBJECTS = ("banana", "bowl", "rubiks_cube")
INPUT_KEYS = {
    "schema_version",
    "study_id",
    "cohort_branch",
    "final_analysis",
    "confirmation_fixture_freeze",
    "confirmation_compiler_receipt",
    "private_video_inventory",
    "analyzer_source",
    "publication_source_commit",
    "copy_selected_videos",
    "payload_sha256",
}
DESCRIPTOR_KEYS = {"path", "bytes", "sha256"}
REFERENCE_KEYS = {"path", "sha256"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")
PRIVATE_PATH_MARKERS = ("/data/", "/home/", "/Users/", "/private/", "/mnt/", "/tmp/")

STATIC_OUTPUTS = (
    "paper_evidence.json",
    "model_results.csv",
    "coverage_by_condition.csv",
    "model_results_table.tex",
    "sample_size_table.tex",
    "forecast_skill_layout_effects.json",
    "actual_scene_timing.json",
    "example_videos.json",
)


class PublicationContractError(ValueError):
    """The supplied evidence is not safe for publication compilation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationContractError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    require(not missing, f"{label} missing keys: {sorted(missing)}")
    require(not extra, f"{label} has disallowed keys: {sorted(extra)}")


def _reject_constant(value: str) -> None:
    raise PublicationContractError(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


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
        raise PublicationContractError(f"value is not finite canonical JSON: {error}") from error


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
        raise PublicationContractError(f"value is not finite JSON: {error}") from error


def payload_hash(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("payload_sha256", None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def sign_document(document: Mapping[str, Any]) -> dict[str, Any]:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except OSError as error:
        raise PublicationContractError(f"cannot read {label}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationContractError(f"{label} is not strict UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    candidate = path if path.is_absolute() else Path.cwd() / path
    cursor = candidate
    while True:
        require(not cursor.is_symlink(), f"{label} contains a symlink component: {cursor}")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent


def file_descriptor(path: Path, *, public_path: str | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file(), f"not a regular file: {resolved}")
    return {
        "path": public_path if public_path is not None else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _resolve_descriptor(
    base: Path, value: Any, label: str
) -> tuple[dict[str, Any], Path]:
    exact_keys(value, DESCRIPTOR_KEYS, label)
    raw = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    require(isinstance(raw, str) and raw.strip(), f"{label} path must be nonempty")
    require(type(size) is int and size >= 0, f"{label} byte count is invalid")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{label} digest is invalid")
    supplied = Path(raw)
    path = supplied if supplied.is_absolute() else Path(base) / supplied
    _reject_symlink_components(path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PublicationContractError(f"{label} is missing: {path}") from error
    require(resolved.is_file() and not resolved.is_symlink(), f"{label} is not a regular file")
    require(resolved.stat().st_size == size, f"{label} byte count changed")
    require(sha256_file(resolved) == digest, f"{label} SHA-256 changed")
    return dict(value), resolved


def _resolve_reference(base: Path, value: Any, label: str) -> Path:
    exact_keys(value, REFERENCE_KEYS, label)
    raw = value.get("path")
    digest = value.get("sha256")
    require(isinstance(raw, str) and raw.strip(), f"{label} path must be nonempty")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{label} digest is invalid")
    supplied = Path(raw)
    path = supplied if supplied.is_absolute() else Path(base) / supplied
    _reject_symlink_components(path, label)
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise PublicationContractError(f"{label} is missing: {path}") from error
    require(path.is_file() and sha256_file(path) == digest, f"{label} SHA-256 changed")
    return path


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None,
            f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise PublicationContractError(f"cannot import {path}: {error}") from error
    return module


def _load_analyzer(path: Path = ANALYZER_PATH) -> ModuleType:
    return _load_module(Path(path), "wmf_publication_final_analyzer")


def _load_confirmation_compiler(path: Path = CONFIRMATION_COMPILER_PATH) -> ModuleType:
    return _load_module(
        Path(path), "wmf_publication_confirmation_compiler"
    )


def _load_fixture_validator(path: Path = FIXTURE_VALIDATOR_PATH) -> ModuleType:
    forecast = Path(path).resolve().parent
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    return _load_module(Path(path), "wmf_publication_fixture_validator")


def _validate_committed_publication_sources(
    commit: str, *, trusted_repository: Path
) -> dict[str, dict[str, Any]]:
    """Require this compiler and its contract to equal tracked blobs at ``commit``."""

    require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None,
            "publication source commit is not a full Git object ID")
    head = _git_bytes(
        REPOSITORY, ["rev-parse", "HEAD"], "publication checkout HEAD"
    ).decode("ascii", errors="strict").strip()
    require(head == commit,
            "publication_source_commit differs from the executing trusted checkout HEAD")
    result = {}
    for label, path in (
        ("publication_compiler_source", Path(__file__).resolve()),
        ("publication_contract", CONTRACT_PATH.resolve()),
    ):
        try:
            relative = path.relative_to(REPOSITORY.resolve()).as_posix()
        except ValueError as error:
            raise PublicationContractError(f"{label} is outside the repository") from error
        tracked = _isolated_git_bytes(
            trusted_repository,
            ["show", f"{commit}:{relative}"],
            f"{label} tracked blob in the isolated remote graph",
        )
        require(
            len(tracked) <= AUTHENTICATED_BLOB_LIMIT_BYTES,
            f"{label} exceeds the authenticated source-blob limit",
        )
        try:
            current = path.read_bytes()
        except OSError as error:
            raise PublicationContractError(f"cannot read {label}: {error}") from error
        require(tracked == current,
                f"{label} is dirty or differs from publication source commit {commit}")
        result[label] = {
            "path": relative,
            "bytes": len(current),
            "sha256": sha256_bytes(current),
        }
    return result


def _git_environment() -> dict[str, str]:
    """Disable interactive credential prompts without exposing credential state."""

    environment = os.environ.copy()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_ASKPASS": "/bin/false",
            "SSH_ASKPASS": "/bin/false",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    for variable in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(variable, None)
    return environment


def _git_result(
    repository: Path,
    arguments: Sequence[str],
    label: str,
    *,
    allowed_returncodes: Sequence[int] = (0,),
    timeout: int = 30,
) -> subprocess.CompletedProcess[bytes]:
    """Run Git while keeping commands, URLs, stderr, and exceptions private."""

    executable = _resolve_git_executable()["invoked_path"]
    try:
        completed = subprocess.run(
            [executable, *arguments],
            cwd=Path(repository),
            check=False,
            capture_output=True,
            timeout=timeout,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        raise PublicationContractError(
            f"cannot authenticate {label}; Git execution failed"
        ) from None
    require(
        completed.returncode in allowed_returncodes,
        f"cannot authenticate {label}",
    )
    return completed


def _git_bytes(repository: Path, arguments: Sequence[str], label: str) -> bytes:
    return _git_result(repository, arguments, label).stdout


def _decode_git_values(payload: bytes, label: str, *, nul: bool = False) -> list[str]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise PublicationContractError(
            f"cannot authenticate {label}; Git output is not UTF-8"
        ) from None
    values = decoded.split("\0") if nul else decoded.splitlines()
    return [value for value in values if value]


def _git_config_values(repository: Path, key: str, label: str) -> list[str]:
    completed = _git_result(
        repository,
        ["config", "--null", "--get-all", key],
        label,
        allowed_returncodes=(0, 1),
    )
    if completed.returncode == 1:
        return []
    return _decode_git_values(completed.stdout, label, nul=True)


def _authorized_url(value: str) -> bool:
    """Accept only explicit, credential-free GitHub spellings of the pinned repo."""

    return value in AUTHORIZED_REPOSITORY_URLS


def _resolve_authorized_remote(repository: Path) -> str:
    """Choose the deployment-specific alias without trusting a redirected URL."""

    repository = Path(repository)
    for alias in AUTHORIZED_REMOTE_ALIASES:
        raw_fetch = _git_config_values(
            repository, f"remote.{alias}.url", f"{alias} fetch URL configuration"
        )
        if not raw_fetch:
            continue
        require(
            len(raw_fetch) == 1 and _authorized_url(raw_fetch[0]),
            f"{alias} does not identify the authorized publication repository",
        )
        raw_push = _git_config_values(
            repository, f"remote.{alias}.pushurl", f"{alias} push URL configuration"
        )
        effective_raw_push = raw_push if raw_push else raw_fetch
        require(
            len(effective_raw_push) == 1 and _authorized_url(effective_raw_push[0]),
            f"{alias} push does not identify the authorized publication repository",
        )
        expanded_fetch = _decode_git_values(
            _git_bytes(
                repository,
                ["remote", "get-url", "--all", alias],
                f"{alias} effective fetch URL",
            ),
            f"{alias} effective fetch URL",
        )
        expanded_push = _decode_git_values(
            _git_bytes(
                repository,
                ["remote", "get-url", "--push", "--all", alias],
                f"{alias} effective push URL",
            ),
            f"{alias} effective push URL",
        )
        require(
            len(expanded_fetch) == 1 and _authorized_url(expanded_fetch[0]),
            f"{alias} effective fetch URL is redirected away from the authorized repository",
        )
        require(
            len(expanded_push) == 1 and _authorized_url(expanded_push[0]),
            f"{alias} effective push URL is redirected away from the authorized repository",
        )
        return alias
    raise PublicationContractError(
        "neither publish nor origin identifies the authorized publication repository"
    )


def _parse_remote_ref(payload: bytes, expected_ref: str, label: str) -> str:
    lines = _decode_git_values(payload, label)
    require(len(lines) == 1, f"{label} did not return exactly one pinned ref")
    fields = lines[0].split("\t")
    require(
        len(fields) == 2
        and COMMIT_RE.fullmatch(fields[0]) is not None
        and fields[1] == expected_ref,
        f"{label} returned an invalid or different ref",
    )
    return fields[0]


def _validate_git_executable_candidate(
    raw: str,
    *,
    allowed_candidates: Sequence[str] = GIT_EXECUTABLE_CANDIDATES,
    trusted_directories: Sequence[str] = ("/usr/bin", "/usr/local/bin"),
    owner_uid: int = 0,
) -> dict[str, str]:
    """Validate a regular Git binary or a fully trusted in-bin symlink chain."""

    require(raw in allowed_candidates, "Git executable path is not pinned")
    trusted = {Path(value) for value in trusted_directories}
    current = Path(raw)
    seen: set[Path] = set()
    invoked = current
    for _ in range(8):
        require(current not in seen, "Git executable symlink chain contains a cycle")
        seen.add(current)
        require(
            current.is_absolute() and current.parent in trusted,
            "Git executable symlink chain leaves trusted system bin directories",
        )
        try:
            directory_metadata = current.parent.lstat()
            metadata = current.lstat()
        except OSError:
            raise PublicationContractError("pinned Git executable is unavailable") from None
        require(
            current.parent.is_dir()
            and not current.parent.is_symlink()
            and directory_metadata.st_uid == owner_uid
            and directory_metadata.st_mode & 0o022 == 0,
            "Git system bin directory is not owned and protected as required",
        )
        is_link = current.is_symlink()
        require(
            metadata.st_uid == owner_uid
            and (is_link or metadata.st_mode & 0o022 == 0),
            "Git executable chain is not owned and protected as required",
        )
        if is_link:
            try:
                target = Path(os.readlink(current))
            except OSError:
                raise PublicationContractError("cannot inspect Git executable symlink") from None
            current = target if target.is_absolute() else current.parent / target
            current = Path(os.path.normpath(current))
            require(
                current.as_posix() in allowed_candidates,
                "Git executable symlink target is not another pinned binary",
            )
            continue
        require(current.is_file(), "pinned Git executable is not a regular file")
        return {
            "invoked_path": invoked.as_posix(),
            "resolved_path": current.as_posix(),
        }
    raise PublicationContractError("Git executable symlink chain is too deep")


def _resolve_git_executable() -> dict[str, str]:
    """Select one trusted system Git without consulting PATH or Git config."""

    for candidate in GIT_EXECUTABLE_CANDIDATES:
        try:
            return _validate_git_executable_candidate(candidate)
        except PublicationContractError:
            continue
    raise PublicationContractError(
        "no root-owned non-writable Git executable exists at an allowed system path"
    )


def _isolated_git_environment() -> dict[str, str]:
    """Retain only network/CA settings and controlled Git safety switches."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key in NETWORK_ENVIRONMENT_KEYS
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


def _write_controlled_isolated_repository_config(repository: Path) -> None:
    """Replace Git's generated local config with one exact inert core section."""

    repository = Path(repository)
    config = repository / "config"
    require(
        repository.is_dir() and not repository.is_symlink(),
        "isolated authentication repository is not a safe directory",
    )
    require(
        config.is_file() and not config.is_symlink(),
        "isolated authentication repository config is not a regular file",
    )
    try:
        with config.open("wb") as stream:
            stream.write(ISOLATED_REPOSITORY_CONFIG)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise PublicationContractError(
            "cannot establish the controlled isolated repository config"
        ) from None


def _validate_controlled_isolated_repository_config(repository: Path) -> None:
    """Reject any local config drift, including helpers, rewrites, or includes."""

    config = Path(repository) / "config"
    try:
        require(
            config.is_file() and not config.is_symlink(),
            "isolated authentication repository config is not a regular file",
        )
        payload = config.read_bytes()
    except OSError:
        raise PublicationContractError(
            "cannot authenticate the isolated repository config"
        ) from None
    require(
        payload == ISOLATED_REPOSITORY_CONFIG,
        "isolated authentication repository config drifted",
    )


def _isolated_git_result(
    repository: Path,
    arguments: Sequence[str],
    label: str,
    *,
    allowed_returncodes: Sequence[int] = (0,),
    protocol: str = "https",
    timeout: int = AUTHENTICATION_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    """Run trusted Git with no inherited executable, config, helper, or SSH state."""

    identity = _resolve_git_executable()
    executable = identity["invoked_path"]
    require(protocol in {"https", "file"}, "isolated Git protocol is not bounded")
    controlled_config = [
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
            [executable, *controlled_config, *arguments],
            cwd=Path(repository),
            check=False,
            capture_output=True,
            timeout=timeout,
            env=_isolated_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        raise PublicationContractError(
            f"cannot authenticate {label}; isolated Git execution failed"
        ) from None
    require(
        completed.returncode in allowed_returncodes,
        f"cannot authenticate {label}",
    )
    return completed


def _isolated_git_bytes(
    repository: Path,
    arguments: Sequence[str],
    label: str,
    *,
    protocol: str = "https",
) -> bytes:
    return _isolated_git_result(
        repository, arguments, label, protocol=protocol
    ).stdout


def _authenticated_graph_size(repository: Path) -> int:
    total = 0
    try:
        for path in repository.rglob("*"):
            require(not path.is_symlink(), "isolated authenticated graph contains a symlink")
            if path.is_file():
                total += path.stat().st_size
                require(
                    total <= MAX_AUTHENTICATED_GRAPH_BYTES,
                    "isolated authenticated graph exceeds its byte limit",
                )
    except OSError:
        raise PublicationContractError(
            "cannot measure the isolated authenticated graph"
        ) from None
    return total


def _remote_ref_round_trip_isolated(
    repository: Path,
    *,
    remote_url: str,
    expected_ref: str,
    protocol: str,
) -> dict[str, str]:
    """Observe and fetch one literal remote/ref without a configured alias."""

    require(
        expected_ref == AUTHORIZED_PUBLICATION_REF,
        "publication remote ref differs from the pinned control ref",
    )
    _validate_controlled_isolated_repository_config(repository)
    ls_remote_arguments = [
        "ls-remote", "--exit-code", "--refs", remote_url, expected_ref
    ]
    before = _parse_remote_ref(
        _isolated_git_bytes(
            repository,
            ls_remote_arguments,
            "authorized publication ref before isolated fetch",
            protocol=protocol,
        ),
        expected_ref,
        "authorized publication ref before isolated fetch",
    )
    _validate_controlled_isolated_repository_config(repository)
    _isolated_git_bytes(
        repository,
        [
            "fetch",
            "--quiet",
            "--no-tags",
            "--no-recurse-submodules",
            f"--filter=blob:limit={AUTHENTICATED_BLOB_LIMIT_BYTES}",
            "--refmap=",
            remote_url,
            f"+{expected_ref}:{AUTHENTICATED_GRAPH_REF}",
        ],
        "authorized publication exact-ref isolated fetch",
        protocol=protocol,
    )
    # A filtered fetch records its promisor URL/filter in local config.  The
    # required trusted blobs are deliberately below the filter limit, so erase
    # that transport-created state before any object or ancestry read.  A
    # missing required blob then fails locally instead of lazily contacting a
    # receipt- or config-selected endpoint.
    _write_controlled_isolated_repository_config(repository)
    _validate_controlled_isolated_repository_config(repository)
    _authenticated_graph_size(repository)
    fetched = _isolated_git_bytes(
        repository,
        ["rev-parse", f"{AUTHENTICATED_GRAPH_REF}^{{commit}}"],
        "isolated fetched authorized publication commit",
        protocol=protocol,
    ).decode("ascii", errors="strict").strip()
    require(
        COMMIT_RE.fullmatch(fetched) is not None,
        "isolated fetched authorized publication commit is invalid",
    )
    _validate_controlled_isolated_repository_config(repository)
    after = _parse_remote_ref(
        _isolated_git_bytes(
            repository,
            ls_remote_arguments,
            "authorized publication ref after isolated fetch",
            protocol=protocol,
        ),
        expected_ref,
        "authorized publication ref after isolated fetch",
    )
    _validate_controlled_isolated_repository_config(repository)
    return {
        "ref": expected_ref,
        "before_fetch": before,
        "fetched": fetched,
        "after_fetch": after,
    }


def _validate_local_publication_checkout(
    commit: str, repository: Path
) -> dict[str, str]:
    local_head = _git_bytes(
        repository, ["rev-parse", "HEAD"], "publication checkout HEAD"
    ).decode("ascii", errors="strict").strip()
    require(
        local_head == commit,
        "publication_source_commit differs from the executing checkout HEAD",
    )
    branch = _git_result(
        repository,
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        "publication checkout branch",
        allowed_returncodes=(0, 1),
    )
    if branch.returncode == 0:
        branch_name = branch.stdout.decode("utf-8", errors="strict").strip()
        require(
            branch_name == AUTHORIZED_PUBLICATION_BRANCH,
            "publication checkout is attached to a different branch",
        )
        checkout_mode = "attached_control_branch"
    else:
        checkout_mode = "detached_immutable_commit"
    return {
        "local_head_commit": commit,
        "checkout_mode": checkout_mode,
        "deployment_remote_alias": _resolve_authorized_remote(repository),
    }


@contextlib.contextmanager
def _authenticated_source_graph_from_url(
    *,
    publication_commit: str,
    study_commit: str,
    remote_url: str,
    protocol: str,
) -> Iterable[tuple[dict[str, Any], Path]]:
    """Materialize and validate a disposable, filtered remote object graph."""

    for value, label in (
        (publication_commit, "publication source commit"),
        (study_commit, "confirmation study commit"),
    ):
        require(
            isinstance(value, str) and COMMIT_RE.fullmatch(value) is not None,
            f"{label} is not a full Git object ID",
        )
    with tempfile.TemporaryDirectory(
        prefix="wmf-publication-auth-", dir="/tmp"
    ) as temporary:
        root = Path(temporary).resolve(strict=True)
        require(
            REPOSITORY.resolve() not in (root, *root.parents),
            "isolated authentication repository overlaps the executing checkout",
        )
        graph = root / "graph.git"
        _isolated_git_bytes(
            root,
            ["init", "--quiet", "--bare", "--template=", str(graph)],
            "isolated bare authentication repository initialization",
            protocol=protocol,
        )
        require(graph.is_dir() and not graph.is_symlink(),
                "isolated bare authentication repository was not created safely")
        _write_controlled_isolated_repository_config(graph)
        _validate_controlled_isolated_repository_config(graph)
        observation = _remote_ref_round_trip_isolated(
            graph,
            remote_url=remote_url,
            expected_ref=AUTHORIZED_PUBLICATION_REF,
            protocol=protocol,
        )
        remote_commit = observation.get("before_fetch")
        require(
            isinstance(remote_commit, str)
            and COMMIT_RE.fullmatch(remote_commit) is not None
            and observation.get("fetched") == remote_commit
            and observation.get("after_fetch") == remote_commit,
            "publication control ref was stale or changed during isolated verification",
        )
        for commit, label in (
            (study_commit, "confirmation study commit in isolated remote graph"),
            (publication_commit, "publication source commit in isolated remote graph"),
            (remote_commit, "control-ref commit in isolated remote graph"),
        ):
            _isolated_git_bytes(
                graph, ["cat-file", "-e", f"{commit}^{{commit}}"], label,
                protocol=protocol,
            )
        for ancestor, descendant, label in (
            (study_commit, publication_commit, "study-to-publication ancestry"),
            (publication_commit, remote_commit, "publication-to-control ancestry"),
        ):
            ancestry = _isolated_git_result(
                graph,
                ["merge-base", "--is-ancestor", ancestor, descendant],
                f"isolated {label}",
                allowed_returncodes=(0, 1),
                protocol=protocol,
            )
            require(
                ancestry.returncode == 0,
                f"isolated remote graph does not prove {label}",
            )
        executable = _resolve_git_executable()
        version = _isolated_git_bytes(
            graph, ["--version"], "isolated Git version", protocol=protocol
        ).decode("ascii", errors="strict").strip()
        require(
            re.fullmatch(r"git version [0-9][0-9A-Za-z.+-]*", version) is not None,
            "isolated Git version output is invalid",
        )
        graph_bytes = _authenticated_graph_size(graph)
        _validate_controlled_isolated_repository_config(graph)
        yield (
            {
                "repository_url": AUTHORIZED_REPOSITORY_URL,
                "control_branch": AUTHORIZED_PUBLICATION_BRANCH,
                "control_ref": AUTHORIZED_PUBLICATION_REF,
                "remote_ref_commit": remote_commit,
                "read_transport": "literal_public_https_isolated_bare",
                "git_executable_path": executable["invoked_path"],
                "git_executable_resolved_path": executable["resolved_path"],
                "git_version": version,
                "blob_filter_limit_bytes": AUTHENTICATED_BLOB_LIMIT_BYTES,
                "authenticated_graph_bytes": graph_bytes,
                "authenticated_graph_max_bytes": MAX_AUTHENTICATED_GRAPH_BYTES,
                "isolated_repository_config_sha256": (
                    ISOLATED_REPOSITORY_CONFIG_SHA256
                ),
                "ls_remote_before_and_after_fetch_match": True,
                "study_to_publication_to_control_ancestry": True,
            },
            graph,
        )
        _validate_controlled_isolated_repository_config(graph)
        _authenticated_graph_size(graph)


@contextlib.contextmanager
def _authenticated_publication_source_graph(
    publication_commit: str,
    study_commit: str,
    *,
    repository: Path | None = None,
) -> Iterable[tuple[dict[str, Any], Path]]:
    """Authenticate local P against a config-independent public HTTPS graph."""

    require(
        AUTHORIZED_REPOSITORY_URL
        == "https://github.com/adeeb10abbas/steerable.git",
        "authorized publication repository constant changed",
    )
    repository = (REPOSITORY if repository is None else Path(repository)).resolve(
        strict=True
    )
    local = _validate_local_publication_checkout(publication_commit, repository)
    with _authenticated_source_graph_from_url(
        publication_commit=publication_commit,
        study_commit=study_commit,
        remote_url=AUTHORIZED_REPOSITORY_URL,
        protocol="https",
    ) as (remote, graph):
        yield ({**remote, **local}, graph)


def _validate_trusted_science_sources(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    supplied_analyzer_descriptor: Mapping[str, Any],
    publication_source_commit: str,
    trusted_repository: Path,
) -> dict[str, Any]:
    """Authenticate historical executable dependencies without importing them.

    The receipt may name files, but it is not trusted to define where executable
    code comes from.  This function fixes three canonical repository paths,
    checks their bytes against Git at the exact confirmation study commit, and
    connects that commit to the already-authenticated publication-source commit.
    Only its returned paths may subsequently be imported.
    """

    study_commit = receipt.get("study_commit")
    require(isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
            "confirmation study source commit is invalid")
    require(
        isinstance(publication_source_commit, str)
        and COMMIT_RE.fullmatch(publication_source_commit) is not None,
        "publication source commit is invalid",
    )
    source_root_value = receipt.get("source_root")
    require(isinstance(source_root_value, str) and Path(source_root_value).is_absolute(),
            "confirmation source root is not an absolute checkout")
    source_root_supplied = Path(source_root_value)
    _reject_symlink_components(source_root_supplied, "confirmation source root")
    try:
        source_root = source_root_supplied.resolve(strict=True)
    except OSError as error:
        raise PublicationContractError("confirmation source root is unavailable") from error
    require(source_root.is_dir(), "confirmation source root is not a directory")

    trusted_repository = Path(trusted_repository).resolve(strict=True)
    _isolated_git_bytes(
        trusted_repository,
        ["cat-file", "-e", f"{study_commit}^{{commit}}"],
        "confirmation study commit in the isolated remote graph",
    )
    _isolated_git_bytes(
        trusted_repository,
        ["cat-file", "-e", f"{publication_source_commit}^{{commit}}"],
        "publication source commit in the isolated remote graph",
    )
    ancestry = _isolated_git_result(
        trusted_repository,
        ["merge-base", "--is-ancestor", study_commit, publication_source_commit],
        "study-to-publication ancestry in the isolated remote graph",
        allowed_returncodes=(0, 1),
    )
    require(
        ancestry.returncode == 0,
        "confirmation study commit is not an ancestor of the authenticated publication source commit",
    )

    checkout_root = _git_bytes(
        source_root, ["rev-parse", "--show-toplevel"], "confirmation checkout root"
    ).decode("utf-8", errors="strict").strip()
    require(Path(checkout_root).resolve() == source_root,
            "confirmation source root is not the exact Git checkout root")
    checkout_head = _git_bytes(
        source_root, ["rev-parse", "HEAD"], "confirmation checkout HEAD"
    ).decode("ascii", errors="strict").strip()
    require(checkout_head == study_commit,
            "confirmation source checkout HEAD differs from the receipt study commit")
    dirty = _git_bytes(
        source_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        "confirmation checkout status",
    )
    require(not dirty.strip(), "confirmation source checkout is dirty")

    canonical = {
        "confirmation_compiler_source": (
            receipt.get("compiler_source"),
            "workshops/corl2026_world_models/analysis/compile_confirmation_evidence.py",
        ),
        "analyzer_source": (
            receipt.get("final_analyzer_dependency"),
            "workshops/corl2026_world_models/analysis/forecast_evidence_analysis.py",
        ),
        "fixture_validator_source": (
            receipt.get("fixture_freeze_dependency"),
            "workshops/corl2026_world_models/experiments/forecast_layout/confirmation_fixture_freeze.py",
        ),
    }
    require(canonical["analyzer_source"][0] == supplied_analyzer_descriptor,
            "supplied analyzer descriptor differs from the compiler receipt")
    descriptors: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for label, (raw_descriptor, relative) in canonical.items():
        descriptor, path = _resolve_descriptor(
            receipt_path.parent, raw_descriptor, label
        )
        expected_path = (source_root / relative).resolve(strict=True)
        require(path == expected_path,
                f"{label} is not at its canonical path in the confirmation checkout")
        tracked = _isolated_git_bytes(
            trusted_repository,
            ["show", f"{study_commit}:{relative}"],
            f"{label} study blob in the isolated remote graph",
        )
        publication_blob = _isolated_git_bytes(
            trusted_repository,
            ["show", f"{publication_source_commit}:{relative}"],
            f"{label} publication blob in the isolated remote graph",
        )
        require(
            len(tracked) <= AUTHENTICATED_BLOB_LIMIT_BYTES
            and len(publication_blob) <= AUTHENTICATED_BLOB_LIMIT_BYTES,
            f"{label} exceeds the authenticated source-blob limit",
        )
        current = path.read_bytes()
        require(
            tracked == publication_blob == current,
            f"{label} differs between the study checkout and authenticated publication HEAD",
        )
        require(
            descriptor["bytes"] == len(tracked)
            and descriptor["sha256"] == sha256_bytes(tracked),
            f"{label} descriptor differs from the trusted study-commit blob",
        )
        descriptors[label] = descriptor
        paths[label] = path
    return {
        "study_commit": study_commit,
        "publication_source_commit": publication_source_commit,
        "study_commit_is_ancestor_of_publication_source_commit": True,
        "descriptors": descriptors,
        "paths": paths,
    }


def _finite(value: Any, label: str) -> float:
    require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length,
            f"{label} must have {length} values")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _validate_metric(value: Any, label: str, *, required: bool) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} is missing")
    estimate = value.get("estimate")
    interval = value.get("ci95")
    count = value.get("layout_pairs")
    if estimate is None:
        require(not required, f"{label} required primary estimate is null")
        require(interval is None and count == 0, f"{label} null state is inconsistent")
        return dict(value)
    _finite(estimate, f"{label} estimate")
    require(isinstance(interval, list) and len(interval) == 2,
            f"{label} interval is invalid")
    low, high = (_finite(item, f"{label} interval") for item in interval)
    require(low <= high, f"{label} interval is reversed")
    require(type(count) is int and 1 <= count <= 24, f"{label} layout count is invalid")
    require(value.get("resamples") == BOOTSTRAP_RESAMPLES,
            f"{label} did not use 10,000 layout bootstraps")
    require(value.get("seed") == ANALYSIS_SEED, f"{label} analysis seed changed")
    rows = value.get("layout_pair_values")
    ids = value.get("complete_layout_ids")
    if rows is not None:
        require(isinstance(rows, list) and len(rows) == count,
                f"{label} layout values are incomplete")
        observed = []
        for row in rows:
            require(isinstance(row, Mapping), f"{label} layout value is invalid")
            layout = row.get("layout_pair_id")
            require(layout in LAYOUTS and layout not in observed,
                    f"{label} layout identity is invalid or duplicated")
            observed.append(layout)
        if ids is not None:
            require(ids == observed, f"{label} complete layout IDs differ from values")
    return dict(value)


def _validate_annotation_quality(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "final analysis lacks annotation quality")
    images = value.get("images")
    agreements = value.get("first_pass_exact_agreements")
    adjudicated = value.get("independently_adjudicated")
    require(
        type(images) is int and images > 0
        and type(agreements) is int and agreements >= 0
        and type(adjudicated) is int and adjudicated >= 0
        and agreements + adjudicated == images,
        "final analysis lacks a complete human consensus inventory",
    )
    require(
        value.get("unit") == "distinct_blinded_annotation_images"
        and isinstance(value.get("final_consensus_sha256"), str)
        and SHA256_RE.fullmatch(value["final_consensus_sha256"]) is not None,
        "final analysis annotation gate is incomplete",
    )
    return dict(value)


def _validate_analysis_report(
    report: Mapping[str, Any], branch: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    verify_signed(report, "final analysis")
    require(
        report.get("schema_version") == FINAL_ANALYSIS_SCHEMA
        and report.get("study_id") == STUDY_ID
        and report.get("stage") == "confirmation"
        and report.get("cohort_branch") == branch,
        "final analysis identity, stage, or cohort changed",
    )
    expected_models = BRANCH_MODELS[branch]
    expected_scope = (
        "FULL_TWO_MODEL_BRANCH_REPORTED_AS_TWO_SEPARATE_WITHIN_MODEL_STUDIES"
        if branch == "full_two_model"
        else f"REDUCED_ONE_MODEL_{expected_models[0]}_BRANCH; the other primary model is unqualified and its 96 confirmation cells were not substituted"
    )
    require(report.get("study_scope") == expected_scope,
            "final analysis full/reduced branch semantics changed")
    contract = report.get("analysis_contract")
    require(
        isinstance(contract, Mapping)
        and contract.get("models_pooled") is False
        and contract.get("bootstrap_resamples") == BOOTSTRAP_RESAMPLES
        and contract.get("analysis_seed") == ANALYSIS_SEED,
        "final analysis pools models or changed the frozen estimator",
    )
    boundaries = report.get("claim_boundaries")
    require(
        isinstance(boundaries, Mapping)
        and boundaries.get("forecast_accuracy_claim_gate") == CLAIM_GATE
        and boundaries.get("either_sign_reported") is True
        and boundaries.get("between_model_accuracy_ranking_allowed") is False
        and boundaries.get("causal_world_model_benefit_claim_allowed") is False,
        "final analysis claim gate is missing or broadened",
    )
    _validate_annotation_quality(report.get("annotation_quality"))
    sources = report.get("source_evidence")
    require(
        isinstance(sources, Mapping)
        and isinstance(sources.get("annotation_freeze"), Mapping)
        and isinstance(sources.get("final_consensus"), Mapping)
        and isinstance(sources.get("request_selection"), Mapping),
        "final analysis lacks hash-bound labels or request selection",
    )
    sample = report.get("model_sample_size_table")
    require(isinstance(sample, list) and len(sample) == 2,
            "final analysis model/sample-size table is incomplete")
    sample_by_model: dict[str, dict[str, Any]] = {}
    for raw in sample:
        require(isinstance(raw, Mapping), "sample-size row is invalid")
        model = raw.get("model_id")
        require(model in MODELS and model not in sample_by_model,
                "sample-size model is invalid or duplicated")
        row = dict(raw)
        sample_by_model[model] = row
        require(row.get("full_design_planned_confirmation_cells") == 96,
                f"{model} full-design denominator changed")
        if model in expected_models:
            require(row.get("branch_status") == "qualified_and_included",
                    f"{model} is not an included qualified branch")
            counts = [
                row.get("valid_complete"), row.get("valid_censored"),
                row.get("technical_invalid"), row.get("unrun"),
            ]
            require(all(type(item) is int and item >= 0 for item in counts),
                    f"{model} status counts are invalid")
            require(row.get("scientific_roster_cells") == 96 and sum(counts) == 96,
                    f"{model} status counts do not preserve the 96-cell cohort")
            for field in (
                "primary_horizon_s", "generated_frame_index",
                "target_executed_action_offset", "timestamp_tolerance_s",
            ):
                require(row.get(field) is not None, f"{model} timing field {field} is null")
            require(isinstance(row.get("camera_id"), str) and row["camera_id"],
                    f"{model} camera identity is missing")
        else:
            require(
                row.get("branch_status") == "unqualified_branch_not_run"
                and row.get("scientific_roster_cells") == 0
                and row.get("valid_complete") == 0
                and row.get("valid_censored") == 0
                and row.get("technical_invalid") == 0
                and row.get("unrun_due_unqualified_branch") == 96,
                f"{model} reduced-branch absence was substituted or relabeled",
            )
    require(set(sample_by_model) == set(MODELS), "sample-size models changed")
    models = report.get("models")
    require(isinstance(models, Mapping) and set(models) == set(expected_models),
            "analysis models are pooled, missing, or outside the branch")
    normalized: dict[str, dict[str, Any]] = {}
    for model in expected_models:
        value = models.get(model)
        require(isinstance(value, Mapping) and value.get("branch_status") == "qualified_and_included",
                f"{model} analysis is not a separate qualified report")
        baseline = value.get("baseline_errors_and_skill")
        require(isinstance(baseline, Mapping), f"{model} baseline analysis is missing")
        forecast = _validate_metric(baseline.get("forecast_error"), f"{model} forecast error", required=True)
        persistence = _validate_metric(baseline.get("persistence_error"), f"{model} persistence error", required=True)
        skill = _validate_metric(
            baseline.get("forecast_skill_vs_persistence"),
            f"{model} forecast skill versus persistence",
            required=True,
        )
        require(
            forecast.get("complete_layout_ids") == persistence.get("complete_layout_ids")
            == skill.get("complete_layout_ids")
            and forecast.get("layout_pairs") == persistence.get("layout_pairs")
            == skill.get("layout_pairs")
            == sample_by_model[model].get("continuous_complete_layout_pairs"),
            f"{model} primary metrics use different layout cohorts",
        )
        _validate_metric(
            baseline.get("constant_velocity_error"),
            f"{model} constant-velocity error",
            required=False,
        )
        _validate_metric(
            baseline.get("forecast_skill_vs_constant_velocity"),
            f"{model} forecast skill versus constant velocity",
            required=False,
        )
        coverage = value.get("coverage_by_condition")
        require(isinstance(coverage, Mapping) and set(coverage) == set(CONDITIONS),
                f"{model} condition coverage is incomplete")
        for condition in CONDITIONS:
            condition_row = coverage[condition]
            require(
                isinstance(condition_row, Mapping)
                and condition_row.get("planned_cells") == 24
                and isinstance(condition_row.get("recording_status_counts"), Mapping)
                and sum(condition_row["recording_status_counts"].values()) == 24,
                f"{model}/{condition} coverage denominator changed",
            )
        bounds = value.get("full_design_strict_win_missingness_bounds")
        require(
            isinstance(bounds, Mapping)
            and len(bounds.get("layout_bounds", [])) == 24
            and set(bounds.get("by_condition", {})) == set(CONDITIONS),
            f"{model} missingness bounds are incomplete",
        )
        for key in ("reflection_contrasts", "stopping_control", "movement_strata",
                    "movement_decomposition", "earlier_horizon", "declared_rule_examples"):
            require(isinstance(value.get(key), Mapping), f"{model} {key} is missing")
        normalized[model] = dict(value)
    return [sample_by_model[model] for model in MODELS], normalized


def _validate_compiler_identity(receipt: Mapping[str, Any], branch: str) -> None:
    require(
        receipt.get("schema_version") == CONFIRMATION_COMPILER_SCHEMA
        and receipt.get("study_id") == STUDY_ID
        and receipt.get("stage") == "confirmation"
        and receipt.get("status") == "compiled_complete_roster"
        and receipt.get("cohort_branch") == branch,
        "confirmation compiler receipt is not a complete matching cohort",
    )
    verify_signed(receipt, "confirmation compiler receipt")
    require(
        receipt.get("safe_for_request_selection") is True
        and receipt.get("safe_for_analysis_manifest_assembly") is False
        and receipt.get("labels_created") is False
        and receipt.get("scientific_results_computed") is False
        and receipt.get("confirmation_released") is False,
        "confirmation compiler source-only boundary changed",
    )


def _validate_compiler_counts(
    receipt: Mapping[str, Any], branch: str, sample: Sequence[Mapping[str, Any]]
) -> None:
    _validate_compiler_identity(receipt, branch)
    included = [row for row in sample if row["model_id"] in BRANCH_MODELS[branch]]
    expected = {
        "planned_cells": 96 * len(included),
        "valid_complete": sum(int(row["valid_complete"]) for row in included),
        "valid_censored": sum(int(row["valid_censored"]) for row in included),
        "technical_invalid": sum(int(row["technical_invalid"]) for row in included),
        "not_run": sum(int(row["unrun"]) for row in included),
    }
    counts = receipt.get("counts")
    require(isinstance(counts, Mapping), "confirmation compiler counts are missing")
    for key, wanted in expected.items():
        require(counts.get(key) == wanted,
                f"confirmation compiler/analyzer status count mismatch: {key}")


def _validate_fixture_binding(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    fixture_descriptor: Mapping[str, Any],
) -> None:
    close_path = _resolve_descriptor(
        receipt_path.parent,
        receipt.get("cohort_close_receipt"),
        "confirmation compiler cohort-close receipt",
    )[1]
    close = load_json(close_path, "confirmation cohort-close receipt")
    rows = close.get("blocks")
    require(isinstance(rows, list) and len(rows) == 24 * len(BRANCH_MODELS[receipt["cohort_branch"]]),
            "confirmation cohort-close block inventory is incomplete")
    observed: set[tuple[str, str]] = set()
    for row in rows:
        require(isinstance(row, Mapping), "confirmation cohort-close block row is invalid")
        model, layout = row.get("model_id"), row.get("layout_pair_id")
        key = (model, layout)
        require(model in BRANCH_MODELS[receipt["cohort_branch"]] and layout in LAYOUTS
                and key not in observed,
                "confirmation cohort-close model/layout is invalid or duplicated")
        observed.add(key)
        aggregate_path = _resolve_descriptor(
            close_path.parent, row.get("receipt"), f"{model}/{layout} aggregate receipt"
        )[1]
        aggregate = load_json(aggregate_path, f"{model}/{layout} aggregate receipt")
        prerequisites = aggregate.get("prerequisites")
        require(
            isinstance(prerequisites, Mapping)
            and prerequisites.get("confirmation_fixture_freeze") == dict(fixture_descriptor),
            f"{model}/{layout} binds a different confirmation fixture freeze",
        )
    expected = {
        (model, layout)
        for model in BRANCH_MODELS[receipt["cohort_branch"]]
        for layout in LAYOUTS
    }
    require(observed == expected, "fixture binding does not cover the exact cohort")


def _validate_private_videos(
    inventory: Mapping[str, Any], *, branch: str, inventory_path: Path,
    compiler_receipt: Mapping[str, Any], compiler_receipt_path: Path,
) -> list[dict[str, Any]]:
    verify_signed(inventory, "private video inventory")
    require(
        inventory.get("schema_version") == PRIVATE_VIDEO_SCHEMA
        and inventory.get("study_id") == STUDY_ID
        and inventory.get("stage") == "confirmation"
        and inventory.get("cohort_branch") == branch
        and inventory.get("visibility") == "private_source_evidence_not_blind_annotation_media",
        "private video inventory identity or visibility changed",
    )
    rows = inventory.get("videos")
    require(isinstance(rows, list) and len(rows) == 96 * len(BRANCH_MODELS[branch]),
            "private video inventory does not cover the exact cohort")
    outputs = compiler_receipt.get("outputs")
    require(isinstance(outputs, Mapping), "confirmation compiler outputs are missing")
    roster_path = _resolve_descriptor(
        compiler_receipt_path.parent,
        outputs.get("request_inventory"),
        "confirmation compiler request inventory",
    )[1]
    request_inventory = load_json(roster_path, "confirmation request inventory")
    roster_rows = request_inventory.get("episode_roster")
    require(isinstance(roster_rows, list) and len(roster_rows) == len(rows),
            "confirmation compiler roster/private-video coverage differs")
    roster = {}
    for item in roster_rows:
        require(isinstance(item, Mapping) and isinstance(item.get("cell_id"), str)
                and item["cell_id"] not in roster,
                "confirmation compiler roster cell is invalid or duplicated")
        roster[item["cell_id"]] = item
    source_rows = compiler_receipt.get("source_confirmation_artifacts")
    require(isinstance(source_rows, list),
            "confirmation compiler source-artifact inventory is missing")
    source_artifacts = {}
    for item in source_rows:
        require(isinstance(item, Mapping) and isinstance(item.get("cell_id"), str)
                and item["cell_id"] not in source_artifacts,
                "confirmation compiler source-artifact cell is invalid or duplicated")
        source_artifacts[item["cell_id"]] = item

    cells: set[str] = set()
    video_ids: set[str] = set()
    normalized = []
    for raw in rows:
        require(isinstance(raw, Mapping), "private video row is invalid")
        row = dict(raw)
        cell = row.get("cell_id")
        require(isinstance(cell, str) and cell not in cells,
                "private video cell identity is missing or duplicated")
        cells.add(cell)
        roster_row = roster.get(cell)
        require(isinstance(roster_row, Mapping),
                f"private video cell is absent from the compiler roster: {cell}")
        require(
            row.get("private_not_for_blind_raters") is True
            and row.get("model_id") in BRANCH_MODELS[branch]
            and row.get("layout_pair_id") in LAYOUTS
            and row.get("condition_id") in CONDITIONS,
            f"private video row identity changed: {cell}",
        )
        for key in (
            "recording_id", "model_id", "layout_pair_id", "condition_id",
            "recording_status", "source_video_id",
        ):
            require(row.get(key) == roster_row.get(key),
                    f"private video/compiler roster mismatch for {cell}: {key}")
        source = row.get("source_video")
        source_id = row.get("source_video_id")
        status = row.get("recording_status")
        require(status in {"valid_complete", "valid_censored", "technical_invalid", "not_run"},
                f"private video status is invalid: {cell}")
        if status == "not_run":
            require(source is None and source_id is None,
                    f"unrun cell {cell} fabricates a source video")
            require(roster_row.get("source_video_sha256") is None,
                    f"unrun compiler roster cell {cell} retains a video hash")
        elif status in {"valid_complete", "valid_censored"}:
            require(isinstance(source_id, str) and source_id not in video_ids,
                    f"source video ID is missing or duplicated: {cell}")
            video_ids.add(source_id)
            descriptor, _ = _resolve_descriptor(
                inventory_path.parent, source, f"private source video {cell}"
            )
            require(roster_row.get("source_video_sha256") == descriptor["sha256"],
                    f"valid compiler roster video hash differs for {cell}")
        elif source is None:
            artifact = source_artifacts.get(cell)
            require(
                source_id is None
                and roster_row.get("source_video_sha256") is None
                and isinstance(artifact, Mapping)
                and artifact.get("recording_status") == "technical_invalid"
                and artifact.get("source_video") is None,
                f"video-less technical-invalid cell {cell} is not compiler-bound zero/no-video evidence",
            )
        else:
            require(isinstance(source_id, str) and source_id not in video_ids,
                    f"technical-invalid source video ID is missing or duplicated: {cell}")
            video_ids.add(source_id)
            descriptor, _ = _resolve_descriptor(
                inventory_path.parent, source, f"private source video {cell}"
            )
            require(roster_row.get("source_video_sha256") == descriptor["sha256"],
                    f"technical-invalid compiler roster video hash differs for {cell}")
        normalized.append(row)
    require(set(roster) == cells, "private video inventory does not exactly cover the compiler roster")
    return normalized


def _selection_rank(model: str, condition: str, cell_id: str) -> str:
    return hashlib.sha256(
        f"{ANALYSIS_SEED}\0{model}\0{condition}\0{cell_id}".encode("utf-8")
    ).hexdigest()


def _join_examples(
    models: Mapping[str, Mapping[str, Any]],
    videos: Sequence[Mapping[str, Any]],
    *,
    inventory_path: Path,
    copy_selected: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    private_sources: list[dict[str, Any]] = []
    copied_total_bytes = 0
    for model in models:
        declared = models[model]["declared_rule_examples"]
        require(
            isinstance(declared.get("declared_rule"), str)
            and "No label, error, success, or visual outcome" in declared["declared_rule"],
            f"{model} example rule is not outcome-blind",
        )
        by_condition = declared.get("videos")
        require(isinstance(by_condition, Mapping) and set(by_condition) == set(CONDITIONS),
                f"{model} declared example inventory changed")
        for condition in CONDITIONS:
            candidates = [
                row for row in videos
                if row.get("model_id") == model
                and row.get("condition_id") == condition
                and row.get("recording_status") == "valid_complete"
            ]
            require(candidates, f"{model}/{condition} has no valid-complete example candidate")
            winner = min(
                candidates,
                key=lambda row: (_selection_rank(model, condition, str(row["cell_id"])), row["cell_id"]),
            )
            expected = {
                "cell_id": winner["cell_id"],
                "source_video_id": winner["source_video_id"],
                "source_video_sha256": winner["source_video"]["sha256"],
                "selection_rank_sha256": _selection_rank(model, condition, winner["cell_id"]),
            }
            observed = by_condition[condition]
            require(isinstance(observed, Mapping),
                    f"{model}/{condition} declared example is absent")
            for key, wanted in expected.items():
                require(observed.get(key) == wanted,
                        f"{model}/{condition} example differs from the outcome-blind rule: {key}")
            matches = [
                row for row in videos
                if row.get("cell_id") == observed.get("cell_id")
                and row.get("source_video_id") == observed.get("source_video_id")
                and isinstance(row.get("source_video"), Mapping)
                and row["source_video"].get("sha256") == observed.get("source_video_sha256")
            ]
            require(len(matches) == 1,
                    f"{model}/{condition} private video join is absent or ambiguous")
            source_descriptor, source_path = _resolve_descriptor(
                inventory_path.parent,
                matches[0]["source_video"],
                f"selected private source video {model}/{condition}",
            )
            require(source_path.suffix.lower() == ".mp4",
                    f"selected private source is not an MP4: {model}/{condition}")
            if copy_selected:
                require(source_descriptor["bytes"] <= MAX_SELECTED_VIDEO_BYTES,
                        f"selected MP4 exceeds the {MAX_SELECTED_VIDEO_BYTES}-byte per-file cap")
                copied_total_bytes += source_descriptor["bytes"]
                require(copied_total_bytes <= MAX_SELECTED_VIDEO_TOTAL_BYTES,
                        f"selected MP4 set exceeds the {MAX_SELECTED_VIDEO_TOTAL_BYTES}-byte total cap")
            relative = f"videos/{model}__{condition}__{winner['cell_id']}.mp4"
            require(SAFE_ID_RE.fullmatch(Path(relative).name) is not None,
                    "derived example video filename is unsafe")
            published = (
                {
                    "path": relative,
                    "bytes": source_descriptor["bytes"],
                    "sha256": source_descriptor["sha256"],
                }
                if copy_selected else None
            )
            selected_rows.append(
                {
                    "model_id": model,
                    "condition_id": condition,
                    "cell_id": winner["cell_id"],
                    "source_video_id": winner["source_video_id"],
                    "source_video_sha256": source_descriptor["sha256"],
                    "source_video_bytes": source_descriptor["bytes"],
                    "selection_rank_sha256": expected["selection_rank_sha256"],
                    "published_copy": published,
                }
            )
            if copy_selected:
                private_sources.append(
                    {
                        "relative_path": relative,
                        "source_path": source_path,
                        "bytes": source_descriptor["bytes"],
                        "sha256": source_descriptor["sha256"],
                    }
                )
    return selected_rows, private_sources


def _extract_scenes(
    freeze: Mapping[str, Any], *, freeze_path: Path
) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    rows = freeze.get("layouts")
    require(isinstance(rows, list) and len(rows) == 24,
            "fixture freeze does not contain 24 layouts")
    scenes = []
    dependencies: list[tuple[Path, str]] = []
    for index, raw in enumerate(rows, start=1):
        require(isinstance(raw, Mapping), "fixture freeze layout row is invalid")
        layout = f"C{index:02d}"
        require(raw.get("layout_pair_id") == layout,
                "fixture freeze layout order changed")
        pose_descriptor, pose_path = _resolve_descriptor(
            freeze_path.parent, raw.get("pose_manifest"), f"{layout} pose manifest"
        )
        pose = load_json(pose_path, f"{layout} pose manifest")
        pair = pose.get("layout_pairs", {}).get(layout)
        require(isinstance(pair, Mapping), f"{layout} pose manifest lacks its layout")
        require(
            pair.get("candidate_payload_sha256") == raw.get("candidate_payload_sha256")
            and pair.get("accepted_gate_record_sha256") == raw.get("accepted_gate_record_sha256"),
            f"{layout} pose manifest differs from the fixture freeze",
        )
        layouts = pair.get("layouts")
        require(isinstance(layouts, Mapping) and set(layouts) == set(ARMS),
                f"{layout} pose arms changed")
        safe_layouts: dict[str, Any] = {}
        for arm in ARMS:
            arm_value = layouts[arm]
            require(isinstance(arm_value, Mapping), f"{layout}/{arm} pose is invalid")
            positions = arm_value.get("positions_robot_base_m")
            quaternions = arm_value.get("quaternions_wxyz")
            require(
                isinstance(positions, Mapping) and set(positions) == set(MOVABLE_OBJECTS)
                and isinstance(quaternions, Mapping) and set(quaternions) == set(MOVABLE_OBJECTS),
                f"{layout}/{arm} object inventory changed",
            )
            safe_layouts[arm] = {
                "positions_robot_base_m": {
                    name: _finite_vector(positions[name], 3, f"{layout}/{arm}/{name} position")
                    for name in MOVABLE_OBJECTS
                },
                "quaternions_wxyz": {
                    name: _finite_vector(quaternions[name], 4, f"{layout}/{arm}/{name} quaternion")
                    for name in MOVABLE_OBJECTS
                },
            }
        scenes.append(
            {
                "layout_pair_id": layout,
                "environment_seed": raw.get("environment_seed"),
                "candidate_id": raw.get("candidate_id"),
                "candidate_payload_sha256": raw.get("candidate_payload_sha256"),
                "accepted_gate_record_sha256": raw.get("accepted_gate_record_sha256"),
                "pose_manifest": {
                    "bytes": pose_descriptor["bytes"],
                    "sha256": pose_descriptor["sha256"],
                },
                "layouts": safe_layouts,
            }
        )
        dependencies.append((pose_path, pose_descriptor["sha256"]))
    return scenes, dependencies


def _public_source_identity(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {"bytes": descriptor["bytes"], "sha256": descriptor["sha256"]}


def _load_and_validate(input_manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(input_manifest_path)
    _reject_symlink_components(manifest_path, "publication input manifest")
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_json(manifest_path, "publication input manifest")
    exact_keys(manifest, INPUT_KEYS, "publication input manifest")
    verify_signed(manifest, "publication input manifest")
    branch = manifest.get("cohort_branch")
    require(
        manifest.get("schema_version") == INPUT_SCHEMA
        and manifest.get("study_id") == STUDY_ID
        and branch in BRANCH_MODELS
        and isinstance(manifest.get("publication_source_commit"), str)
        and COMMIT_RE.fullmatch(manifest["publication_source_commit"]) is not None
        and type(manifest.get("copy_selected_videos")) is bool,
        "publication input manifest identity, branch, or copy policy is invalid",
    )
    resolved: dict[str, tuple[dict[str, Any], Path]] = {}
    for key in (
        "final_analysis", "confirmation_fixture_freeze",
        "confirmation_compiler_receipt", "private_video_inventory", "analyzer_source",
    ):
        resolved[key] = _resolve_descriptor(manifest_path.parent, manifest[key], key)

    compiler_descriptor, compiler_receipt_path = resolved["confirmation_compiler_receipt"]
    receipt = load_json(compiler_receipt_path, "confirmation compiler receipt")
    _validate_compiler_identity(receipt, str(branch))
    study_commit = receipt.get("study_commit")
    require(
        isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
        "confirmation compiler study commit is invalid",
    )
    analyzer_descriptor, analyzer_path = resolved["analyzer_source"]
    require(receipt.get("final_analyzer_dependency") == analyzer_descriptor,
            "confirmation compiler binds a different analyzer source descriptor")
    with _authenticated_publication_source_graph(
        manifest["publication_source_commit"], study_commit
    ) as (authorized_publication_remote, authenticated_graph):
        committed_publication_sources = _validate_committed_publication_sources(
            manifest["publication_source_commit"],
            trusted_repository=authenticated_graph,
        )
        trusted_science_sources = _validate_trusted_science_sources(
            receipt=receipt,
            receipt_path=compiler_receipt_path,
            supplied_analyzer_descriptor=analyzer_descriptor,
            publication_source_commit=manifest["publication_source_commit"],
            trusted_repository=authenticated_graph,
        )
    require(
        trusted_science_sources["paths"]["analyzer_source"] == analyzer_path,
        "trusted analyzer path differs from the supplied analyzer descriptor",
    )

    analyzer = _load_analyzer(trusted_science_sources["paths"]["analyzer_source"])
    require(getattr(analyzer, "OUTPUT_SCHEMA", None) == FINAL_ANALYSIS_SCHEMA,
            "loaded analyzer output schema changed")

    analysis_descriptor, analysis_path = resolved["final_analysis"]
    analysis = load_json(analysis_path, "final analysis")
    sample, models = _validate_analysis_report(analysis, str(branch))
    analysis_manifest_reference = analysis.get("analysis_evidence_manifest")
    require(
        isinstance(analysis_manifest_reference, Mapping)
        and Path(str(analysis_manifest_reference.get("path", ""))).is_absolute(),
        "final analysis evidence manifest must retain an exact absolute source path",
    )
    analysis_manifest_path = _resolve_reference(
        analysis_path.parent, analysis_manifest_reference, "analysis evidence manifest"
    )
    try:
        replayed = analyzer.analyze_manifest(
            Path(str(analysis_manifest_reference["path"]))
        )
    except Exception as error:
        raise PublicationContractError(f"final analyzer replay failed: {error}") from error
    require(replayed == analysis,
            "supplied final analysis does not exactly reproduce from its signed evidence")
    require(sha256_file(analyzer_path) == analyzer_descriptor["sha256"],
            "analyzer source drifted during replay")

    _validate_compiler_counts(receipt, str(branch), sample)
    compiler_source_descriptor = trusted_science_sources["descriptors"]["confirmation_compiler_source"]
    compiler_source_path = trusted_science_sources["paths"]["confirmation_compiler_source"]
    compiler = _load_confirmation_compiler(compiler_source_path)
    require(getattr(compiler, "COMPILER_SCHEMA", None) == CONFIRMATION_COMPILER_SCHEMA,
            "loaded confirmation compiler schema changed")
    try:
        compiler._validate_compiled_bundle(compiler_receipt_path.parent)
    except Exception as error:
        raise PublicationContractError(
            f"confirmation evidence compiler replay failed: {error}"
        ) from error
    require(load_json(compiler_receipt_path, "confirmation compiler receipt after replay") == receipt,
            "confirmation compiler receipt drifted during deep replay")

    private_descriptor, private_path = resolved["private_video_inventory"]
    outputs = receipt.get("outputs")
    require(isinstance(outputs, Mapping), "confirmation compiler output inventory is missing")
    declared_private = outputs.get("private_video_inventory")
    declared_private_descriptor, declared_private_path = _resolve_descriptor(
        compiler_receipt_path.parent,
        declared_private,
        "confirmation compiler private video inventory",
    )
    require(
        declared_private_path == private_path
        and declared_private_descriptor.get("bytes") == private_descriptor["bytes"]
        and declared_private_descriptor.get("sha256") == private_descriptor["sha256"],
        "private video inventory differs from the confirmation compiler receipt",
    )
    private_inventory = load_json(private_path, "private video inventory")
    video_rows = _validate_private_videos(
        private_inventory,
        branch=str(branch),
        inventory_path=private_path,
        compiler_receipt=receipt,
        compiler_receipt_path=compiler_receipt_path,
    )

    request_selection_path = _resolve_reference(
        analysis_path.parent,
        analysis["source_evidence"]["request_selection"],
        "analysis request selection",
    )
    selection = load_json(request_selection_path, "analysis request selection")
    verify_signed(selection, "analysis request selection")
    require(
        selection.get("selection_uses_object_visibility_or_forecast_quality") is False
        and selection.get("provenance", {}).get("request_inventory_sha256")
        == outputs.get("request_inventory", {}).get("sha256"),
        "analysis request selection is outcome-based or detached from compiled evidence",
    )

    fixture_descriptor, fixture_path = resolved["confirmation_fixture_freeze"]
    fixture = load_json(fixture_path, "confirmation fixture freeze")
    verify_signed(fixture, "confirmation fixture freeze")
    study_commit = receipt.get("study_commit")
    require(isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
            "confirmation compiler study commit is invalid")
    require(
        fixture.get("schema_version") == FIXTURE_FREEZE_SCHEMA
        and fixture.get("study_id") == STUDY_ID
        and fixture.get("status") == "frozen_for_confirmation"
        and fixture.get("created_from_study_commit") == study_commit
        and fixture.get("selection_uses_target_model_outcomes") is False
        and fixture.get("layout_count") == 24
        and fixture.get("model_request_count") == 0
        and fixture.get("behavioral_action_count") == 0,
        "confirmation fixture freeze is not the exact model-blind 24-layout cohort",
    )
    fixture_validator_descriptor = trusted_science_sources["descriptors"]["fixture_validator_source"]
    fixture_validator_path = trusted_science_sources["paths"]["fixture_validator_source"]
    fixture_validator = _load_fixture_validator(fixture_validator_path)
    source_root = receipt.get("source_root")
    require(isinstance(source_root, str) and Path(source_root).is_absolute(),
            "confirmation compiler source root is invalid")
    for layout in LAYOUTS:
        try:
            validation = fixture_validator.validate_fixture_freeze(
                fixture_path,
                fixture_descriptor["sha256"],
                source_root=Path(source_root),
                expected_layout_pair_id=layout,
                expected_study_commit=study_commit,
                deep_validate_selected=True,
            )
        except Exception as error:
            raise PublicationContractError(
                f"deep fixture replay failed for {layout}: {error}"
            ) from error
        require(
            isinstance(validation, Mapping)
            and validation.get("fixture_freeze") == fixture_descriptor
            and validation.get("selected_layout", {}).get("layout_pair_id") == layout,
            f"deep fixture replay returned a different {layout} binding",
        )
    _validate_fixture_binding(
        receipt=receipt,
        receipt_path=compiler_receipt_path,
        fixture_descriptor=fixture_descriptor,
    )
    scenes, pose_dependencies = _extract_scenes(fixture, freeze_path=fixture_path)
    examples, video_sources = _join_examples(
        models,
        video_rows,
        inventory_path=private_path,
        copy_selected=manifest["copy_selected_videos"],
    )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "branch": branch,
        "analysis": analysis,
        "sample": sample,
        "models": models,
        "receipt": receipt,
        "receipt_path": compiler_receipt_path,
        "fixture": fixture,
        "scenes": scenes,
        "examples": examples,
        "video_sources": video_sources,
        "source_descriptors": {
            key: resolved[key][0] for key in resolved
        },
        "source_paths": {
            key: resolved[key][1] for key in resolved
        },
        "dependency_descriptors": {
            "confirmation_compiler_source": compiler_source_descriptor,
            "fixture_validator_source": fixture_validator_descriptor,
        },
        "dependency_paths": {
            "confirmation_compiler_source": compiler_source_path,
            "fixture_validator_source": fixture_validator_path,
        },
        "analysis_manifest_path": analysis_manifest_path,
        "analysis_manifest_sha256": analysis_manifest_reference["sha256"],
        "pose_dependencies": pose_dependencies,
        "study_commit": study_commit,
        "publication_source_commit": manifest["publication_source_commit"],
        "authorized_publication_remote": authorized_publication_remote,
        "committed_publication_sources": committed_publication_sources,
        "trusted_science_sources": trusted_science_sources,
    }


def _metric_compact(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "estimate", "ci95", "layout_pairs", "resamples", "seed",
            "resampling_unit", "complete_layout_ids", "excluded_incomplete_layout_ids",
            "layout_pair_values",
        )
        if key in value
    }


def _sample_public(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "model_id", "branch_status", "checkpoint_revision", "executed_prefix_cap",
        "primary_horizon_s", "generated_frame_index", "target_executed_action_offset",
        "camera_id", "timestamp_tolerance_s", "full_design_planned_confirmation_cells",
        "scientific_roster_cells", "valid_complete", "valid_censored",
        "technical_invalid", "unrun", "unrun_due_unqualified_branch",
        "selected_requests", "primary_observable_requests",
        "continuous_complete_layout_pairs",
    )
    return {field: row.get(field) for field in fields}


def _build_public_documents(context: Mapping[str, Any]) -> dict[str, bytes]:
    analysis = context["analysis"]
    branch = context["branch"]
    public_sample = [_sample_public(row) for row in context["sample"]]
    source_bindings = {
        "publication_input_manifest": {
            "bytes": context["manifest_path"].stat().st_size,
            "sha256": sha256_file(context["manifest_path"]),
        },
        **{
            key: _public_source_identity(value)
            for key, value in context["source_descriptors"].items()
        },
        **{
            key: _public_source_identity(value)
            for key, value in context["dependency_descriptors"].items()
        },
        "analysis_evidence_manifest": {
            "bytes": context["analysis_manifest_path"].stat().st_size,
            "sha256": context["analysis_manifest_sha256"],
        },
        "study_commit": context["study_commit"],
        "publication_source_commit": context["publication_source_commit"],
        "authorized_publication_remote": context["authorized_publication_remote"],
        "committed_publication_sources": context["committed_publication_sources"],
        "trusted_science_source_authentication": {
            "study_commit": context["trusted_science_sources"]["study_commit"],
            "publication_source_commit": context["trusted_science_sources"]["publication_source_commit"],
            "study_commit_is_ancestor_of_publication_source_commit": True,
            "canonical_tracked_sources": {
                key: _public_source_identity(value)
                for key, value in context["trusted_science_sources"]["descriptors"].items()
            },
        },
    }
    paper_models = {}
    for model, value in context["models"].items():
        paper_models[model] = {
            "branch_status": value["branch_status"],
            "baseline_errors_and_skill": value["baseline_errors_and_skill"],
            "coverage_by_condition": value["coverage_by_condition"],
            "full_design_strict_win_missingness_bounds": value[
                "full_design_strict_win_missingness_bounds"
            ],
            "reflection_contrasts": value["reflection_contrasts"],
            "stopping_control": value["stopping_control"],
            "movement_strata": value["movement_strata"],
            "movement_decomposition": value["movement_decomposition"],
            "earlier_horizon": value["earlier_horizon"],
        }
    paper = sign_document(
        {
            "schema_version": PAPER_EVIDENCE_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "confirmation",
            "cohort_branch": branch,
            "study_scope": analysis["study_scope"],
            "claim_boundaries": analysis["claim_boundaries"],
            "analysis_contract": analysis["analysis_contract"],
            "annotation_quality": analysis["annotation_quality"],
            "model_sample_size_table": public_sample,
            "models": paper_models,
            "source_bindings": source_bindings,
            "paper_generation_performed": False,
            "labels_created": False,
            "scientific_estimates_recomputed": False,
        }
    )

    forecast_figure = sign_document(
        {
            "schema_version": FORECAST_FIGURE_SCHEMA,
            "study_id": STUDY_ID,
            "cohort_branch": branch,
            "models_pooled": False,
            "analysis_seed": ANALYSIS_SEED,
            "layout_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "models": {
                model: {
                    "forecast_skill_vs_persistence": _metric_compact(
                        value["baseline_errors_and_skill"]["forecast_skill_vs_persistence"]
                    ),
                    "reflection_layout_effects": value["reflection_contrasts"],
                }
                for model, value in context["models"].items()
            },
            "source_final_analysis_sha256": context["source_descriptors"]["final_analysis"]["sha256"],
        }
    )
    timing = {
        row["model_id"]: {
            key: row.get(key)
            for key in (
                "primary_horizon_s", "generated_frame_index",
                "target_executed_action_offset", "executed_prefix_cap",
                "camera_id", "timestamp_tolerance_s",
            )
        }
        for row in context["sample"]
        if row["model_id"] in BRANCH_MODELS[branch]
    }
    scene_figure = sign_document(
        {
            "schema_version": SCENE_FIGURE_SCHEMA,
            "study_id": STUDY_ID,
            "cohort_branch": branch,
            "scene_coordinate_frame": "robot_base_m",
            "scene_source": "model_blind_physically_gated_confirmation_fixture_freeze",
            "layout_count": 24,
            "layouts": context["scenes"],
            "model_timing": timing,
            "source_fixture_freeze_sha256": context["source_descriptors"]["confirmation_fixture_freeze"]["sha256"],
            "source_final_analysis_sha256": context["source_descriptors"]["final_analysis"]["sha256"],
        }
    )
    videos = sign_document(
        {
            "schema_version": VIDEO_MANIFEST_SCHEMA,
            "study_id": STUDY_ID,
            "cohort_branch": branch,
            "selection_rule": (
                "For each included model and condition, choose the valid-complete cell with "
                "minimum SHA256(UTF8(analysis_seed) || NUL || model || NUL || condition || "
                "NUL || cell_id), tie-breaking by cell_id. Labels, errors, success and visual "
                "outcomes are never inputs."
            ),
            "analysis_seed": ANALYSIS_SEED,
            "copy_selected_videos": context["manifest"]["copy_selected_videos"],
            "copy_limits_bytes": {
                "per_file": MAX_SELECTED_VIDEO_BYTES,
                "total": MAX_SELECTED_VIDEO_TOTAL_BYTES,
            },
            "expected_examples": 4 * len(BRANCH_MODELS[branch]),
            "examples": context["examples"],
            "private_source_paths_published": False,
            "source_private_video_inventory": _public_source_identity(
                context["source_descriptors"]["private_video_inventory"]
            ),
        }
    )

    documents: dict[str, bytes] = {
        "paper_evidence.json": pretty_bytes(paper),
        "forecast_skill_layout_effects.json": pretty_bytes(forecast_figure),
        "actual_scene_timing.json": pretty_bytes(scene_figure),
        "example_videos.json": pretty_bytes(videos),
    }
    documents["model_results.csv"] = _model_results_csv(context).encode("utf-8")
    documents["coverage_by_condition.csv"] = _coverage_csv(context).encode("utf-8")
    documents["model_results_table.tex"] = _model_results_tex(context).encode("utf-8")
    documents["sample_size_table.tex"] = _sample_size_tex(context).encode("utf-8")
    require(set(documents) == set(STATIC_OUTPUTS), "static publication output set changed")
    for name, payload in documents.items():
        if name.endswith(".json"):
            _assert_no_private_paths(load_json_bytes(payload, name), name)
        else:
            _assert_text_has_no_private_paths(payload.decode("utf-8"), name)
    return documents


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationContractError(f"generated {label} is invalid JSON: {error}") from error
    require(isinstance(value, dict), f"generated {label} is not an object")
    return value


def _assert_no_private_paths(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_private_paths(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_private_paths(item, f"{label}[{index}]")
    elif isinstance(value, str):
        require(
            (not value.startswith("/") or value in GIT_EXECUTABLE_CANDIDATES)
            and not any(marker in value for marker in PRIVATE_PATH_MARKERS)
            and (
                value == AUTHORIZED_REPOSITORY_URL
                or WINDOWS_ABSOLUTE_RE.search(value) is None
            )
            and "file://" not in value,
            f"public output leaks an absolute private path at {label}",
        )


def _assert_text_has_no_private_paths(text: str, label: str) -> None:
    for token in re.split(r"[\s,&{}]+", text):
        require(
            not any(marker in token for marker in PRIVATE_PATH_MARKERS)
            and WINDOWS_ABSOLUTE_RE.search(token) is None
            and "file://" not in token,
            f"public text output leaks an absolute private path in {label}",
        )


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    number = _finite(value, "table numeric value")
    return f"{number:.6g}"


def _metric_columns(metric: Mapping[str, Any]) -> list[str]:
    interval = metric.get("ci95")
    return [
        _format_number(metric.get("estimate")),
        _format_number(None if interval is None else interval[0]),
        _format_number(None if interval is None else interval[1]),
        str(metric.get("layout_pairs", 0)),
    ]


def _model_results_csv(context: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "model_id", "branch_status", "planned_cells", "valid_complete",
            "valid_censored", "technical_invalid", "unrun",
            "selected_requests", "primary_observable_requests",
            "forecast_error", "forecast_error_ci_low", "forecast_error_ci_high",
            "forecast_error_layout_pairs", "persistence_error",
            "persistence_error_ci_low", "persistence_error_ci_high",
            "persistence_error_layout_pairs", "forecast_skill_vs_persistence",
            "forecast_skill_ci_low", "forecast_skill_ci_high", "forecast_skill_layout_pairs",
            "constant_velocity_error", "constant_velocity_ci_low",
            "constant_velocity_ci_high", "constant_velocity_layout_pairs",
            "strict_win_bound_lower", "strict_win_bound_upper",
            "reflection_discrepancy", "reflection_discrepancy_ci_low",
            "reflection_discrepancy_ci_high", "reflection_discrepancy_layout_pairs",
        ]
    )
    sample = {row["model_id"]: row for row in context["sample"]}
    for model in MODELS:
        row = sample[model]
        if model not in context["models"]:
            writer.writerow(
                [model, row["branch_status"], 96, 0, 0, 0, 96, 0, 0]
                + ["NA"] * 22
            )
            continue
        value = context["models"][model]
        baseline = value["baseline_errors_and_skill"]
        bounds = value["full_design_strict_win_missingness_bounds"]
        reflection = value["reflection_contrasts"]["combined_command_discrepancy"]
        writer.writerow(
            [
                model, row["branch_status"], 96, row["valid_complete"],
                row["valid_censored"], row["technical_invalid"], row["unrun"],
                row["selected_requests"], row["primary_observable_requests"],
            ]
            + _metric_columns(baseline["forecast_error"])
            + _metric_columns(baseline["persistence_error"])
            + _metric_columns(baseline["forecast_skill_vs_persistence"])
            + _metric_columns(baseline["constant_velocity_error"])
            + [_format_number(bounds["lower"]), _format_number(bounds["upper"])]
            + _metric_columns(reflection)
        )
    return stream.getvalue()


def _coverage_csv(context: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "model_id", "condition_id", "planned_cells", "valid_complete",
            "valid_censored", "technical_invalid", "not_run",
            "request_inventory_count", "timing_camera_action_eligible_requests",
            "zero_eligible_episodes", "selected_requests", "primary_observable_requests",
            "constant_velocity_observable_requests", "early_horizon_observable_requests",
            "unresolved_generated_prediction_requests",
        ]
    )
    for model, value in context["models"].items():
        for condition in CONDITIONS:
            row = value["coverage_by_condition"][condition]
            statuses = row["recording_status_counts"]
            writer.writerow(
                [
                    model, condition, row["planned_cells"], statuses["valid_complete"],
                    statuses["valid_censored"], statuses["technical_invalid"],
                    statuses["not_run"], row["request_inventory_count"],
                    row["timing_camera_action_eligible_requests"],
                    row["zero_eligible_episodes"], row["selected_requests"],
                    row["primary_observable_requests"],
                    row["constant_velocity_observable_requests"],
                    row["early_horizon_observable_requests"],
                    row["unresolved_generated_prediction_requests"],
                ]
            )
    return stream.getvalue()


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _metric_tex(metric: Mapping[str, Any]) -> str:
    if metric.get("estimate") is None:
        return "NA"
    interval = metric["ci95"]
    return (
        f"{_format_number(metric['estimate'])} "
        f"[{_format_number(interval[0])}, {_format_number(interval[1])}]"
    )


def _model_results_tex(context: Mapping[str, Any]) -> str:
    lines = [
        "% Generated evidence fragment; do not edit by hand.",
        r"\begin{tabular}{lrrrr}",
        r"Model & Forecast error & Persistence error & Skill (95\% CI) & Layout pairs \\",
        r"\hline",
    ]
    for model, value in context["models"].items():
        baseline = value["baseline_errors_and_skill"]
        skill = baseline["forecast_skill_vs_persistence"]
        lines.append(
            f"{_tex_escape(model)} & {_metric_tex(baseline['forecast_error'])} & "
            f"{_metric_tex(baseline['persistence_error'])} & {_metric_tex(skill)} & "
            f"{skill['layout_pairs']} " + r"\\"
        )
    lines.extend([r"\end{tabular}", ""])
    return "\n".join(lines)


def _sample_size_tex(context: Mapping[str, Any]) -> str:
    lines = [
        "% Generated evidence fragment; do not edit by hand.",
        r"\begin{tabular}{llrrrrrr}",
        r"Model & Branch status & Planned & Complete & Censored & Invalid & Unrun & Selected requests \\",
        r"\hline",
    ]
    for row in context["sample"]:
        unrun = row.get("unrun", row.get("unrun_due_unqualified_branch", 0))
        lines.append(
            f"{_tex_escape(row['model_id'])} & {_tex_escape(row['branch_status'])} & 96 & "
            f"{row.get('valid_complete', 0)} & {row.get('valid_censored', 0)} & "
            f"{row.get('technical_invalid', 0)} & {unrun} & {row.get('selected_requests', 0)} "
            + r"\\"
        )
    lines.extend([r"\end{tabular}", ""])
    return "\n".join(lines)


def _copy_verified(source: Path, target: Path, expected_bytes: int, expected_sha: str) -> None:
    require(not target.exists(), f"refusing to overwrite staged video: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        for block in iter(lambda: input_handle.read(1024 * 1024), b""):
            output_handle.write(block)
            digest.update(block)
            count += len(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    require(count == expected_bytes and digest.hexdigest() == expected_sha,
            f"selected MP4 copy changed: {target.name}")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _public_build_source(path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    return file_descriptor(path, public_path=relative)


def _reverify_sources(context: Mapping[str, Any]) -> None:
    with _authenticated_publication_source_graph(
        context["publication_source_commit"], context["study_commit"]
    ) as (remote, authenticated_graph):
        expected_remote = dict(context["authorized_publication_remote"])
        observed_remote = dict(remote)
        expected_remote.pop("authenticated_graph_bytes", None)
        observed_remote.pop("authenticated_graph_bytes", None)
        require(
            observed_remote == expected_remote,
            "authorized publication remote identity drifted during compilation",
        )
        committed = _validate_committed_publication_sources(
            context["publication_source_commit"],
            trusted_repository=authenticated_graph,
        )
        require(
            committed == context["committed_publication_sources"],
            "committed publication source identities drifted during compilation",
        )
        trusted = _validate_trusted_science_sources(
            receipt=context["receipt"],
            receipt_path=context["receipt_path"],
            supplied_analyzer_descriptor=context["source_descriptors"]["analyzer_source"],
            publication_source_commit=context["publication_source_commit"],
            trusted_repository=authenticated_graph,
        )
    require(
        trusted["descriptors"] == context["trusted_science_sources"]["descriptors"]
        and trusted["paths"] == context["trusted_science_sources"]["paths"],
        "trusted science source identities drifted during publication compilation",
    )
    for key, descriptor in context["source_descriptors"].items():
        path = context["source_paths"][key]
        require(
            path.stat().st_size == descriptor["bytes"]
            and sha256_file(path) == descriptor["sha256"],
            f"{key} drifted during publication compilation",
        )
    for key, descriptor in context["dependency_descriptors"].items():
        path = context["dependency_paths"][key]
        require(
            path.stat().st_size == descriptor["bytes"]
            and sha256_file(path) == descriptor["sha256"],
            f"{key} drifted during publication compilation",
        )
    require(
        sha256_file(context["analysis_manifest_path"])
        == context["analysis_manifest_sha256"],
        "analysis evidence manifest drifted during publication compilation",
    )
    for path, digest in context["pose_dependencies"]:
        require(sha256_file(path) == digest,
                "fixture pose manifest drifted during publication compilation")
    for source in context["video_sources"]:
        require(
            source["source_path"].stat().st_size == source["bytes"]
            and sha256_file(source["source_path"]) == source["sha256"],
            "selected private source video drifted during publication compilation",
        )


def _validate_staged_bundle(bundle: Path, receipt: Mapping[str, Any]) -> None:
    verify_signed(receipt, "publication build receipt")
    require(
        receipt.get("schema_version") == BUILD_RECEIPT_SCHEMA
        and receipt.get("study_id") == STUDY_ID
        and receipt.get("status") == "compiled_source_only_publication_artifacts"
        and receipt.get("paper_generation_performed") is False
        and receipt.get("labels_created") is False
        and receipt.get("scientific_estimates_recomputed") is False,
        "publication build receipt broadens the source-only operation",
    )
    inventory = receipt.get("output_inventory")
    require(isinstance(inventory, Mapping), "publication build output inventory is missing")
    expected_paths = set(STATIC_OUTPUTS)
    expected_paths.update(
        row["path"]
        for row in receipt.get("selected_video_copies", [])
    )
    require(set(inventory) == expected_paths,
            "publication build receipt does not bind every non-receipt output")
    for relative, descriptor in inventory.items():
        require(isinstance(relative, str) and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts,
                "publication output path is unsafe")
        expected_path = bundle / relative
        observed = file_descriptor(expected_path, public_path=relative)
        require(observed == descriptor, f"publication output changed: {relative}")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*") if path.is_file()
    }
    require(actual == expected_paths | {"build_receipt.json"},
            "publication directory contains an undeclared output")
    for name in STATIC_OUTPUTS:
        path = bundle / name
        if name.endswith(".json"):
            document = load_json(path, f"staged {name}")
            verify_signed(document, f"staged {name}")
            _assert_no_private_paths(document, name)
        else:
            _assert_text_has_no_private_paths(path.read_text(encoding="utf-8"), name)


def compile_publication(input_manifest: Path, output_dir: Path) -> dict[str, Any]:
    """Validate every source and atomically create one immutable output directory."""

    target = Path(output_dir)
    require(not target.exists(), f"refusing to replace publication output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent, "publication output parent")
    context = _load_and_validate(Path(input_manifest))
    documents = _build_public_documents(context)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        for name in STATIC_OUTPUTS:
            _write_bytes(temporary / name, documents[name])
        for source in context["video_sources"]:
            _copy_verified(
                source["source_path"],
                temporary / source["relative_path"],
                source["bytes"],
                source["sha256"],
            )
        output_paths = list(STATIC_OUTPUTS) + [
            row["relative_path"] for row in context["video_sources"]
        ]
        output_inventory = {
            relative: file_descriptor(temporary / relative, public_path=relative)
            for relative in sorted(output_paths)
        }
        receipt = sign_document(
            {
                "schema_version": BUILD_RECEIPT_SCHEMA,
                "study_id": STUDY_ID,
                "status": "compiled_source_only_publication_artifacts",
                "cohort_branch": context["branch"],
                "study_commit": context["study_commit"],
                "publication_source_commit": context["publication_source_commit"],
                "authorized_publication_remote": context["authorized_publication_remote"],
                "publication_compiler_source": context["committed_publication_sources"]["publication_compiler_source"],
                "publication_contract": context["committed_publication_sources"]["publication_contract"],
                "trusted_science_source_authentication": {
                    "study_commit": context["trusted_science_sources"]["study_commit"],
                    "publication_source_commit": context["trusted_science_sources"]["publication_source_commit"],
                    "study_commit_is_ancestor_of_publication_source_commit": True,
                    "canonical_tracked_sources": {
                        key: _public_source_identity(value)
                        for key, value in context["trusted_science_sources"]["descriptors"].items()
                    },
                },
                "input_bindings": {
                    "publication_input_manifest": {
                        "bytes": context["manifest_path"].stat().st_size,
                        "sha256": sha256_file(context["manifest_path"]),
                    },
                    **{
                        key: _public_source_identity(value)
                        for key, value in context["source_descriptors"].items()
                    },
                    **{
                        key: _public_source_identity(value)
                        for key, value in context["dependency_descriptors"].items()
                    },
                    "analysis_evidence_manifest": {
                        "bytes": context["analysis_manifest_path"].stat().st_size,
                        "sha256": context["analysis_manifest_sha256"],
                    },
                },
                "model_ids": list(context["models"]),
                "models_pooled": False,
                "layout_count": 24,
                "selected_example_count": len(context["examples"]),
                "selected_video_copy_count": len(context["video_sources"]),
                "selected_video_copy_limits_bytes": {
                    "per_file": MAX_SELECTED_VIDEO_BYTES,
                    "total": MAX_SELECTED_VIDEO_TOTAL_BYTES,
                },
                "selected_video_copies": [
                    {
                        "path": source["relative_path"],
                        "bytes": source["bytes"],
                        "sha256": source["sha256"],
                    }
                    for source in context["video_sources"]
                ],
                "output_count_including_self_signed_receipt": len(output_inventory) + 1,
                "output_inventory": output_inventory,
                "receipt_self_binding": "payload_sha256",
                "paper_generation_performed": False,
                "labels_created": False,
                "scientific_estimates_recomputed": False,
                "claim_boundary": (
                    "Validated source-only publication fragments and optional byte-identical "
                    "declared-rule videos; this receipt does not edit or generate a paper, "
                    "create labels, rank models, or recompute science."
                ),
            }
        )
        _write_bytes(temporary / "build_receipt.json", pretty_bytes(receipt))
        _reverify_sources(context)
        _validate_staged_bundle(temporary, receipt)
        os.replace(temporary, target)
        published = True
        return receipt
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    receipt = compile_publication(arguments.input_manifest, arguments.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(arguments.output_dir),
                "cohort_branch": receipt["cohort_branch"],
                "models": receipt["model_ids"],
                "selected_example_count": receipt["selected_example_count"],
                "payload_sha256": receipt["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
