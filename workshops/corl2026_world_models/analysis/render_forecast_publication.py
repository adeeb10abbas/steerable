#!/usr/bin/env python3
"""Render deterministic, source-only SVGs from validated publication inputs.

The renderer consumes the two signed figure-input documents emitted by
``compile_forecast_publication.py`` and their signed build receipt.  It only
performs presentation transforms: coordinates are projected into SVG space and
already-computed estimates are drawn.  It never recomputes a scientific
estimator, pools models, or infers a frame-to-action timing conversion.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]
FORECAST = PACKAGE / "experiments" / "forecast_layout"
CONTRACT_PATH = FORECAST / "forecast_figure_render_contract.json"
PUBLICATION_COMPILER_PATH = Path(__file__).with_name("compile_forecast_publication.py")
RENDERER_SOURCE_PATH = Path(__file__).resolve()

STUDY_ID = "WMF-ABLATION-001"
FORECAST_INPUT_SCHEMA = "wmf-forecast-skill-layout-effects-figure-input-v1"
SCENE_INPUT_SCHEMA = "wmf-forecast-actual-scene-timing-figure-input-v1"
PUBLICATION_RECEIPT_SCHEMA = "wmf-forecast-publication-build-receipt-v1"
RENDER_RECEIPT_SCHEMA = "wmf-forecast-figure-render-receipt-v1"
AUTHORIZED_REPOSITORY_URL = "https://github.com/adeeb10abbas/steerable.git"
AUTHORIZED_PUBLICATION_BRANCH = "codex/forecast-layout-gm-20260912"
AUTHORIZED_PUBLICATION_REF = f"refs/heads/{AUTHORIZED_PUBLICATION_BRANCH}"
AUTHENTICATED_BLOB_LIMIT_BYTES = 1024 * 1024
AUTHENTICATED_GRAPH_MAX_BYTES = 192 * 1024 * 1024
TRUSTED_PUBLICATION_AUTHENTICATOR_SHA256 = (
    "1a76b0b78eb2a5dec74c220865450946f4a07731376fbeb4d80912884368bec8"
)
ISOLATED_REPOSITORY_CONFIG_SHA256 = hashlib.sha256(
    b"[core]\n\trepositoryformatversion = 0\n\tbare = true\n"
).hexdigest()

ANALYSIS_SEED = 2026091301
BOOTSTRAP_RESAMPLES = 10_000
MODELS = ("N3", "D1")
BRANCH_MODELS = {
    "full_two_model": ("N3", "D1"),
    "reduced_n3": ("N3",),
    "reduced_d1": ("D1",),
}
LAYOUTS = tuple(f"C{index:02d}" for index in range(1, 25))
ARMS = ("original", "reflected")
OBJECTS = ("banana", "bowl", "rubiks_cube")
COMMANDS = ("left", "right")
RESAMPLING_UNIT = (
    "complete independent base-layout pair with four condition means nested"
)
REFLECTION_DEFINITION = (
    "For each command and layout, reflected-minus-original qhat minus "
    "reflected-minus-original q; either sign is retained."
)

FORECAST_FILENAME = "forecast_skill_layout_effects.json"
SCENE_FILENAME = "actual_scene_timing.json"
PUBLICATION_RECEIPT_FILENAME = "build_receipt.json"
BASE_OUTPUTS = ("actual_scene_layouts.svg", "qualified_timing.svg")
PUBLICATION_STATIC_OUTPUTS = {
    "paper_evidence.json",
    "model_results.csv",
    "coverage_by_condition.csv",
    "model_results_table.tex",
    "sample_size_table.tex",
    FORECAST_FILENAME,
    SCENE_FILENAME,
    "example_videos.json",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
SAFE_OUTPUT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
STAGING_NAME_RE = re.compile(r"\.wmf-figure-render-[0-9a-f]{24}\Z")
WINDOWS_ABSOLUTE_RE = re.compile(r"[A-Za-z]:[\\/]")
PRIVATE_PATH_MARKERS = ("/data/", "/home/", "/Users/", "/private/", "/mnt/", "/tmp/")
ALLOWED_ABSOLUTE_PUBLIC_VALUES = {"/usr/bin/git", "/usr/local/bin/git"}
ALLOWED_REPOSITORY_URLS = {
    AUTHORIZED_REPOSITORY_URL,
    "git@github.com:adeeb10abbas/steerable.git",
    "ssh://git@github.com/adeeb10abbas/steerable.git",
}

FORECAST_KEYS = {
    "schema_version", "study_id", "cohort_branch", "models_pooled",
    "analysis_seed", "layout_bootstrap_resamples", "models",
    "source_final_analysis_sha256", "payload_sha256",
}
FORECAST_MODEL_KEYS = {
    "forecast_skill_vs_persistence", "reflection_layout_effects",
}
SKILL_METRIC_KEYS = {
    "estimate", "ci95", "layout_pairs", "resamples", "seed",
    "resampling_unit", "complete_layout_ids", "excluded_incomplete_layout_ids",
    "layout_pair_values",
}
REFLECTION_KEYS = {"definition", "by_command", "combined_command_discrepancy"}
REFLECTION_COMMAND_KEYS = {
    "actual_reflected_minus_original_motion",
    "predicted_reflected_minus_original_motion",
    "predicted_minus_actual_contrast_discrepancy",
    "eligible_layout_ids", "paired_layout_values",
}
BOOTSTRAP_PRESENT_KEYS = {
    "estimate", "ci95", "layout_pairs", "resamples", "seed", "resampling_unit",
}
BOOTSTRAP_NULL_KEYS = {"estimate", "ci95", "layout_pairs"}
SCENE_KEYS = {
    "schema_version", "study_id", "cohort_branch", "scene_coordinate_frame",
    "scene_source", "layout_count", "layouts", "model_timing",
    "source_fixture_freeze_sha256", "source_final_analysis_sha256",
    "payload_sha256",
}
SCENE_ROW_KEYS = {
    "layout_pair_id", "environment_seed", "candidate_id",
    "candidate_payload_sha256", "accepted_gate_record_sha256",
    "pose_manifest", "layouts",
}
ARM_KEYS = {"positions_robot_base_m", "quaternions_wxyz"}
TIMING_KEYS = {
    "primary_horizon_s", "generated_frame_index", "target_executed_action_offset",
    "executed_prefix_cap", "camera_id", "timestamp_tolerance_s",
}
PUBLICATION_RECEIPT_KEYS = {
    "schema_version", "study_id", "status", "cohort_branch", "study_commit",
    "publication_source_commit", "authorized_publication_remote",
    "publication_compiler_source", "publication_contract",
    "trusted_science_source_authentication", "input_bindings", "model_ids",
    "models_pooled", "layout_count", "selected_example_count",
    "selected_video_copy_count", "selected_video_copy_limits_bytes",
    "selected_video_copies", "output_count_including_self_signed_receipt",
    "output_inventory", "receipt_self_binding", "paper_generation_performed",
    "labels_created", "scientific_estimates_recomputed", "claim_boundary",
    "payload_sha256",
}
INPUT_BINDING_KEYS = {
    "publication_input_manifest", "final_analysis", "confirmation_fixture_freeze",
    "confirmation_compiler_receipt", "private_video_inventory", "analyzer_source",
    "confirmation_compiler_source", "fixture_validator_source",
    "analysis_evidence_manifest",
}
DESCRIPTOR_KEYS = {"path", "bytes", "sha256"}
HASH_SIZE_KEYS = {"bytes", "sha256"}
AUTHENTICATED_REMOTE_KEYS = {
    "repository_url", "control_branch", "control_ref", "remote_ref_commit",
    "read_transport", "git_executable_path", "git_executable_resolved_path",
    "git_version", "blob_filter_limit_bytes", "authenticated_graph_bytes",
    "authenticated_graph_max_bytes", "isolated_repository_config_sha256",
    "ls_remote_before_and_after_fetch_match",
    "study_to_publication_to_control_ancestry", "local_head_commit",
    "checkout_mode", "deployment_remote_alias",
}
COMMITTED_RENDERER_SOURCE_KEYS = {"renderer_source", "renderer_contract"}
RENDER_INPUT_BINDING_KEYS = {
    FORECAST_FILENAME, SCENE_FILENAME, PUBLICATION_RECEIPT_FILENAME,
}
RENDER_RECEIPT_KEYS = {
    "schema_version", "study_id", "status", "cohort_branch", "model_ids",
    "models_pooled", "publication_source_commit", "renderer_source_commit",
    "publication_to_renderer_to_control_ancestry", "authorized_renderer_remote",
    "committed_renderer_sources", "source_publication_build_receipt_payload_sha256",
    "source_final_analysis_sha256", "source_fixture_freeze_sha256",
    "input_bindings", "output_inventory",
    "output_count_including_self_signed_receipt", "receipt_self_binding",
    "scientific_estimates_recomputed", "frame_to_action_mapping_inferred",
    "labels_created", "paper_files_edited_or_generated", "claim_boundary",
    "payload_sha256",
}
RENDER_CLAIM_BOUNDARY = (
    "Deterministic SVG presentation of signed publication inputs only; "
    "coordinates are projected for display, models remain separate, missing "
    "values remain missing, and no scientific or timing estimator is derived."
)


class FigureRenderContractError(ValueError):
    """The publication bundle is not safe for deterministic rendering."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureRenderContractError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    require(not missing, f"{label} missing keys: {sorted(missing)}")
    require(not extra, f"{label} has disallowed keys: {sorted(extra)}")


def _reject_constant(value: str) -> None:
    raise FigureRenderContractError(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FigureRenderContractError(f"value is not finite canonical JSON: {error}") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FigureRenderContractError(f"value is not finite JSON: {error}") from error


def payload_hash(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("payload_sha256", None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def sign_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    require("payload_sha256" not in result, "document is already signed")
    result["payload_sha256"] = payload_hash(result)
    return result


def verify_signed(document: Mapping[str, Any], label: str) -> None:
    digest = document.get("payload_sha256")
    require(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
        f"{label} lacks a lowercase SHA-256 payload signature",
    )
    require(payload_hash(document) == digest, f"{label} payload signature changed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, label: str) -> None:
    candidate = path if path.is_absolute() else Path.cwd() / path
    cursor = candidate
    while True:
        require(not cursor.is_symlink(), f"{label} contains a symlink component: {cursor}")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    require(all(hasattr(os, name) for name in required),
            "no-follow directory traversal is unavailable on this platform")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_nofollow(path: Path, label: str) -> tuple[int, Path]:
    """Open an existing directory one component at a time without following links."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = _directory_open_flags()
    try:
        descriptor = os.open(os.sep, flags)
    except OSError:
        raise FigureRenderContractError(f"cannot open the filesystem root for {label}") from None
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError:
                raise FigureRenderContractError(
                    f"{label} is missing, not a directory, or contains a symlink"
                ) from None
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute
    except Exception:
        os.close(descriptor)
        raise


def _require_same_open_directory(path: Path, descriptor: int, label: str) -> None:
    """Reject replacement of the path while retaining the originally opened fd."""

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


def _require_child_absent(parent_fd: int, name: str, label: str) -> None:
    require(SAFE_OUTPUT_NAME_RE.fullmatch(name) is not None,
            f"{label} name is unsafe")
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise FigureRenderContractError(f"cannot inspect {label}") from None
    raise FigureRenderContractError(f"refusing to replace figure output: {name}")


def _require_staging_child_absent(parent_fd: int, name: str) -> None:
    require(STAGING_NAME_RE.fullmatch(name) is not None,
            "figure staging directory name is unsafe")
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise FigureRenderContractError(
            "cannot inspect figure staging directory rollback name"
        ) from None
    raise FigureRenderContractError(
        "figure staging directory rollback name unexpectedly exists"
    )


def _create_staging_directory(parent_fd: int) -> tuple[str, int]:
    for _ in range(32):
        name = f".wmf-figure-render-{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        except OSError:
            raise FigureRenderContractError(
                "cannot create the same-parent figure staging directory"
            ) from None
        descriptor: int | None = None
        try:
            descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
            by_name = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(descriptor)
            require(
                stat.S_ISDIR(by_name.st_mode)
                and (by_name.st_dev, by_name.st_ino) == (opened.st_dev, opened.st_ino),
                "figure staging directory changed while it was opened",
            )
            return name, descriptor
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    raise FigureRenderContractError("cannot allocate a unique figure staging directory")


def _opened_directory_path(descriptor: int) -> Path:
    metadata = os.fstat(descriptor)
    require(stat.S_ISDIR(metadata.st_mode), "opened figure staging object is not a directory")
    path = Path(f"/proc/self/fd/{descriptor}")
    require(path.is_dir(), "opened figure staging directory is unavailable")
    return path


def _require_child_matches_fd(parent_fd: int, name: str, descriptor: int) -> None:
    try:
        by_name = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise FigureRenderContractError("figure staging directory name changed") from None
    opened = os.fstat(descriptor)
    require(
        stat.S_ISDIR(by_name.st_mode)
        and (by_name.st_dev, by_name.st_ino) == (opened.st_dev, opened.st_ino),
        "figure staging directory name changed",
    )


def _remove_staging_directory(parent_fd: int, name: str, staging_fd: int) -> None:
    """Remove the renderer's flat staging directory through held descriptors."""

    for child in os.listdir(staging_fd):
        require("/" not in child and child not in {".", ".."},
                "figure staging cleanup encountered an unsafe name")
        metadata = os.stat(child, dir_fd=staging_fd, follow_symlinks=False)
        require(not stat.S_ISDIR(metadata.st_mode),
                "figure staging cleanup refuses an unexpected subdirectory")
        os.unlink(child, dir_fd=staging_fd)
    _require_child_matches_fd(parent_fd, name, staging_fd)
    os.rmdir(name, dir_fd=parent_fd)


def load_json(
    path: Path, label: str, *, trusted_open_directory_fd_path: bool = False
) -> dict[str, Any]:
    if not trusted_open_directory_fd_path:
        _reject_symlink_components(path, label)
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
    except OSError as error:
        raise FigureRenderContractError(f"cannot read {label}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FigureRenderContractError(f"{label} is not strict UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def file_descriptor(path: Path, *, public_path: str) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"not a regular file: {public_path}")
    return {
        "path": public_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _repository_relative(path: Path, repository: Path, label: str) -> str:
    try:
        return Path(path).resolve(strict=True).relative_to(
            Path(repository).resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as error:
        raise FigureRenderContractError(f"{label} is outside the repository") from error


def _load_publication_compiler(path: Path = PUBLICATION_COMPILER_PATH) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "wmf_figure_renderer_publication_authenticator", Path(path)
    )
    require(specification is not None and specification.loader is not None,
            "cannot load the committed publication authenticator")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    except Exception:
        raise FigureRenderContractError(
            "cannot load the committed publication authenticator"
        ) from None
    require(
        getattr(module, "AUTHORIZED_REPOSITORY_URL", None) == AUTHORIZED_REPOSITORY_URL
        and getattr(module, "AUTHORIZED_PUBLICATION_BRANCH", None)
        == AUTHORIZED_PUBLICATION_BRANCH
        and getattr(module, "AUTHORIZED_PUBLICATION_REF", None)
        == AUTHORIZED_PUBLICATION_REF
        and getattr(module, "BUILD_RECEIPT_SCHEMA", None)
        == PUBLICATION_RECEIPT_SCHEMA,
        "publication authenticator constants differ from the renderer contract",
    )
    return module


def _production_source_graph_factory(
    compiler: ModuleType, renderer_commit: str, publication_commit: str,
    repository: Path,
) -> Any:
    del repository
    return compiler._authenticated_source_graph_from_url(
        publication_commit=renderer_commit,
        study_commit=publication_commit,
        remote_url=AUTHORIZED_REPOSITORY_URL,
        protocol="https",
    )


def _validate_authenticated_remote(value: Any, renderer_commit: str) -> dict[str, Any]:
    exact_keys(value, AUTHENTICATED_REMOTE_KEYS, "authenticated renderer remote")
    require(
        value.get("repository_url") == AUTHORIZED_REPOSITORY_URL
        and value.get("control_branch") == AUTHORIZED_PUBLICATION_BRANCH
        and value.get("control_ref") == AUTHORIZED_PUBLICATION_REF
        and value.get("read_transport") == "literal_public_https_isolated_bare"
        and value.get("local_head_commit") == renderer_commit
        and value.get("checkout_mode") in {
            "attached_control_branch", "detached_immutable_commit",
        }
        and value.get("deployment_remote_alias") in {"publish", "origin"}
        and value.get("ls_remote_before_and_after_fetch_match") is True
        and value.get("study_to_publication_to_control_ancestry") is True,
        "authenticated renderer remote identity or ancestry changed",
    )
    require(
        isinstance(value.get("remote_ref_commit"), str)
        and COMMIT_RE.fullmatch(value["remote_ref_commit"]) is not None,
        "authenticated renderer control commit is invalid",
    )
    _sha(
        value.get("isolated_repository_config_sha256"),
        "authenticated renderer isolated config",
    )
    require(
        value.get("git_executable_path") in ALLOWED_ABSOLUTE_PUBLIC_VALUES
        and value.get("git_executable_resolved_path") in ALLOWED_ABSOLUTE_PUBLIC_VALUES
        and isinstance(value.get("git_version"), str)
        and re.fullmatch(r"git version [0-9][0-9A-Za-z.+-]*", value["git_version"])
        is not None,
        "authenticated renderer Git identity changed",
    )
    graph_bytes = value.get("authenticated_graph_bytes")
    graph_limit = value.get("authenticated_graph_max_bytes")
    require(
        value.get("blob_filter_limit_bytes") == AUTHENTICATED_BLOB_LIMIT_BYTES
        and type(graph_bytes) is int and graph_bytes > 0
        and graph_limit == AUTHENTICATED_GRAPH_MAX_BYTES
        and graph_bytes <= graph_limit,
        "authenticated renderer graph bounds changed",
    )
    require(
        value.get("isolated_repository_config_sha256")
        == ISOLATED_REPOSITORY_CONFIG_SHA256,
        "authenticated renderer isolated configuration identity changed",
    )
    return dict(value)


def _trusted_blob(
    compiler: ModuleType, graph: Path, commit: str, relative: str, label: str,
) -> bytes:
    try:
        payload = compiler._isolated_git_bytes(
            graph, ["show", f"{commit}:{relative}"], label
        )
    except Exception:
        raise FigureRenderContractError(
            f"cannot authenticate {label} in the isolated source graph"
        ) from None
    require(len(payload) <= AUTHENTICATED_BLOB_LIMIT_BYTES,
            f"{label} exceeds the authenticated source-blob limit")
    return payload


def _authenticate_renderer_sources(
    publication_receipt: Mapping[str, Any], *, source_graph_factory: Any = None,
) -> dict[str, Any]:
    """Bind renderer bytes to a published commit through an isolated Git graph."""

    require(
        AUTHORIZED_REPOSITORY_URL
        == "https://github.com/adeeb10abbas/steerable.git",
        "authorized renderer repository constant changed",
    )
    repository = REPOSITORY.resolve(strict=True)
    compiler_relative = _repository_relative(
        PUBLICATION_COMPILER_PATH, repository, "publication authenticator"
    )
    declared_compiler = publication_receipt.get("publication_compiler_source")
    exact_keys(declared_compiler, DESCRIPTOR_KEYS,
               "publication authenticator descriptor")
    require(declared_compiler.get("path") == compiler_relative,
            "publication authenticator is not at its canonical repository path")
    observed_compiler = file_descriptor(
        PUBLICATION_COMPILER_PATH, public_path=compiler_relative
    )
    require(
        observed_compiler == declared_compiler
        and observed_compiler["sha256"] == TRUSTED_PUBLICATION_AUTHENTICATOR_SHA256,
        "publication authenticator differs from its pinned trusted byte identity",
    )
    compiler = _load_publication_compiler(PUBLICATION_COMPILER_PATH)
    try:
        renderer_commit = compiler._git_bytes(
            repository, ["rev-parse", "HEAD"], "renderer checkout HEAD"
        ).decode("ascii", errors="strict").strip()
    except Exception:
        raise FigureRenderContractError("cannot authenticate renderer checkout HEAD") from None
    require(COMMIT_RE.fullmatch(renderer_commit) is not None,
            "renderer checkout HEAD is not a full Git object ID")
    publication_commit = publication_receipt.get("publication_source_commit")
    require(isinstance(publication_commit, str)
            and COMMIT_RE.fullmatch(publication_commit) is not None,
            "publication source commit is invalid")
    try:
        local = compiler._validate_local_publication_checkout(renderer_commit, repository)
    except Exception:
        raise FigureRenderContractError(
            "renderer checkout is not the authorized branch or detached immutable HEAD"
        ) from None
    factory = source_graph_factory or _production_source_graph_factory
    try:
        graph_context = factory(
            compiler, renderer_commit, publication_commit, repository
        )
        with graph_context as (remote, graph):
            authenticated_remote = _validate_authenticated_remote(
                {**dict(remote), **dict(local)}, renderer_commit
            )
            compiler_at_publication = _trusted_blob(
                compiler, graph, publication_commit, compiler_relative,
                "publication authenticator at publication source commit",
            )
            compiler_at_renderer = _trusted_blob(
                compiler, graph, renderer_commit, compiler_relative,
                "publication authenticator at renderer source commit",
            )
            current_compiler = PUBLICATION_COMPILER_PATH.read_bytes()
            require(
                compiler_at_publication == compiler_at_renderer == current_compiler
                and len(current_compiler) == declared_compiler["bytes"]
                and hashlib.sha256(current_compiler).hexdigest()
                == declared_compiler["sha256"],
                "publication authenticator differs across authenticated source commits",
            )
            committed_sources: dict[str, dict[str, Any]] = {}
            for key, path in (
                ("renderer_source", RENDERER_SOURCE_PATH),
                ("renderer_contract", CONTRACT_PATH),
            ):
                relative = _repository_relative(path, repository, key)
                tracked = _trusted_blob(
                    compiler, graph, renderer_commit, relative,
                    f"{key} at renderer source commit",
                )
                try:
                    current = Path(path).read_bytes()
                except OSError:
                    raise FigureRenderContractError(f"cannot read current {key}") from None
                require(tracked == current,
                        f"{key} is dirty or differs from renderer source commit")
                committed_sources[key] = {
                    "path": relative,
                    "bytes": len(current),
                    "sha256": hashlib.sha256(current).hexdigest(),
                }
    except FigureRenderContractError:
        raise
    except Exception:
        raise FigureRenderContractError(
            "cannot authenticate renderer source commit against the stable control ref"
        ) from None
    return {
        "renderer_source_commit": renderer_commit,
        "publication_to_renderer_to_control_ancestry": True,
        "authorized_renderer_remote": authenticated_remote,
        "committed_renderer_sources": committed_sources,
    }


def _hash_size(value: Any, label: str) -> dict[str, Any]:
    exact_keys(value, HASH_SIZE_KEYS, label)
    size, digest = value["bytes"], value["sha256"]
    require(type(size) is int and size >= 0, f"{label} byte count is invalid")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{label} SHA-256 is invalid")
    return dict(value)


def _inventory_descriptor(value: Any, expected_path: str, label: str) -> dict[str, Any]:
    exact_keys(value, DESCRIPTOR_KEYS, label)
    require(value.get("path") == expected_path, f"{label} path differs from its inventory key")
    _hash_size({"bytes": value.get("bytes"), "sha256": value.get("sha256")}, label)
    candidate = Path(expected_path)
    require(
        not candidate.is_absolute() and ".." not in candidate.parts
        and candidate.as_posix() == expected_path,
        f"{label} path is unsafe",
    )
    return dict(value)


def _finite(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length,
            f"{label} must contain {length} values")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} must be a lowercase SHA-256")
    return value


def _ordered_layout_subset(value: Any, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    require(
        all(isinstance(item, str) and item in LAYOUTS for item in value),
        f"{label} contains an invalid layout",
    )
    require(len(set(value)) == len(value), f"{label} contains duplicate layouts")
    expected = [layout for layout in LAYOUTS if layout in set(value)]
    require(value == expected, f"{label} is not in canonical layout order")
    return list(value)


def _assert_no_private_paths(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_private_paths(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_private_paths(item, f"{label}[{index}]")
    elif isinstance(value, str):
        require(
            value in ALLOWED_REPOSITORY_URLS
            or (
                (not value.startswith("/") or value in ALLOWED_ABSOLUTE_PUBLIC_VALUES)
                and not any(marker in value for marker in PRIVATE_PATH_MARKERS)
                and WINDOWS_ABSOLUTE_RE.search(value) is None
                and "file://" not in value
            ),
            f"public figure input leaks an absolute private path at {label}",
        )


def _validate_publication_receipt(receipt: Mapping[str, Any], root: Path) -> str:
    exact_keys(receipt, PUBLICATION_RECEIPT_KEYS, "publication build receipt")
    verify_signed(receipt, "publication build receipt")
    branch = receipt.get("cohort_branch")
    require(
        receipt.get("schema_version") == PUBLICATION_RECEIPT_SCHEMA
        and receipt.get("study_id") == STUDY_ID
        and receipt.get("status") == "compiled_source_only_publication_artifacts"
        and branch in BRANCH_MODELS,
        "publication build receipt identity or cohort is invalid",
    )
    expected_models = list(BRANCH_MODELS[branch])
    require(
        receipt.get("model_ids") == expected_models
        and receipt.get("models_pooled") is False
        and receipt.get("layout_count") == 24
        and receipt.get("selected_example_count") == 4 * len(expected_models)
        and receipt.get("receipt_self_binding") == "payload_sha256"
        and receipt.get("paper_generation_performed") is False
        and receipt.get("labels_created") is False
        and receipt.get("scientific_estimates_recomputed") is False,
        "publication build receipt broadens or changes the source-only cohort",
    )
    for key in ("study_commit", "publication_source_commit"):
        require(
            isinstance(receipt.get(key), str) and COMMIT_RE.fullmatch(receipt[key]) is not None,
            f"publication build receipt {key} is invalid",
        )
    exact_keys(receipt.get("input_bindings"), INPUT_BINDING_KEYS,
               "publication build receipt input bindings")
    for key, descriptor in receipt["input_bindings"].items():
        _hash_size(descriptor, f"publication input binding {key}")
    for key in ("publication_compiler_source", "publication_contract"):
        descriptor = receipt.get(key)
        exact_keys(descriptor, DESCRIPTOR_KEYS, f"publication receipt {key}")
        require(
            isinstance(descriptor.get("path"), str)
            and not Path(descriptor["path"]).is_absolute()
            and ".." not in Path(descriptor["path"]).parts,
            f"publication receipt {key} path is unsafe",
        )
        _hash_size({"bytes": descriptor.get("bytes"), "sha256": descriptor.get("sha256")},
                   f"publication receipt {key}")

    inventory = receipt.get("output_inventory")
    require(isinstance(inventory, Mapping), "publication output inventory is missing")
    copies = receipt.get("selected_video_copies")
    copy_count = receipt.get("selected_video_copy_count")
    require(isinstance(copies, list) and type(copy_count) is int and copy_count == len(copies),
            "publication selected-video count is inconsistent")
    copy_paths: set[str] = set()
    for index, raw in enumerate(copies):
        require(isinstance(raw, Mapping), f"selected video {index} is invalid")
        path = raw.get("path")
        require(isinstance(path, str), f"selected video {index} lacks a path")
        descriptor = _inventory_descriptor(raw, path, f"selected video {index}")
        require(path.startswith("videos/") and path not in copy_paths,
                f"selected video {index} path is invalid or duplicated")
        copy_paths.add(path)
        require(inventory.get(path) == descriptor,
                f"selected video {index} differs from publication inventory")
    limits = receipt.get("selected_video_copy_limits_bytes")
    exact_keys(limits, {"per_file", "total"}, "selected-video copy limits")
    require(limits == {"per_file": 16 * 1024 * 1024, "total": 64 * 1024 * 1024},
            "selected-video copy limits changed")
    require(set(inventory) == PUBLICATION_STATIC_OUTPUTS | copy_paths,
            "publication output inventory set changed")
    require(
        receipt.get("output_count_including_self_signed_receipt") == len(inventory) + 1,
        "publication output count is inconsistent",
    )
    for relative, raw in inventory.items():
        require(isinstance(relative, str), "publication inventory path is not text")
        descriptor = _inventory_descriptor(raw, relative, f"publication output {relative}")
        path = root / relative
        _reject_symlink_components(path, f"publication output {relative}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise FigureRenderContractError(f"publication output is missing: {relative}") from error
        require(resolved.is_file() and not resolved.is_symlink(),
                f"publication output is not a regular file: {relative}")
        observed = file_descriptor(resolved, public_path=relative)
        require(observed == descriptor, f"publication output changed: {relative}")
    all_entries = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in all_entries),
            "publication directory contains a symlink")
    actual = {
        path.relative_to(root).as_posix()
        for path in all_entries if path.is_file()
    }
    require(actual == set(inventory) | {PUBLICATION_RECEIPT_FILENAME},
            "publication directory contains an undeclared or missing file")
    _assert_no_private_paths(receipt, "publication build receipt")
    return branch


def _validate_skill_metric(value: Any, label: str) -> dict[str, Any]:
    exact_keys(value, SKILL_METRIC_KEYS, label)
    estimate = _finite(value.get("estimate"), f"{label} estimate")
    interval = value.get("ci95")
    require(isinstance(interval, list) and len(interval) == 2, f"{label} interval is invalid")
    low, high = (_finite(item, f"{label} interval") for item in interval)
    require(low <= high, f"{label} interval is reversed")
    count = value.get("layout_pairs")
    require(type(count) is int and 1 <= count <= 24, f"{label} layout count is invalid")
    require(
        value.get("resamples") == BOOTSTRAP_RESAMPLES
        and value.get("seed") == ANALYSIS_SEED
        and value.get("resampling_unit") == RESAMPLING_UNIT,
        f"{label} frozen bootstrap identity changed",
    )
    ids = _ordered_layout_subset(value.get("complete_layout_ids"), f"{label} complete layouts")
    excluded = _ordered_layout_subset(
        value.get("excluded_incomplete_layout_ids"), f"{label} excluded layouts"
    )
    require(not set(ids) & set(excluded) and set(ids) | set(excluded) == set(LAYOUTS),
            f"{label} complete/excluded layout coverage changed")
    rows = value.get("layout_pair_values")
    require(isinstance(rows, list) and len(rows) == count == len(ids),
            f"{label} layout values do not match the declared count")
    for index, row in enumerate(rows):
        exact_keys(row, {"layout_pair_id", "value"}, f"{label} layout value {index}")
        require(row.get("layout_pair_id") == ids[index], f"{label} layout value order changed")
        _finite(row.get("value"), f"{label} {ids[index]} value")
    return dict(value)


def _validate_bootstrap_metric(value: Any, label: str, expected_count: int) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} is missing")
    if value.get("estimate") is None:
        exact_keys(value, BOOTSTRAP_NULL_KEYS, label)
        require(value.get("ci95") is None and value.get("layout_pairs") == 0 == expected_count,
                f"{label} null state is inconsistent")
        return dict(value)
    exact_keys(value, BOOTSTRAP_PRESENT_KEYS, label)
    estimate = _finite(value.get("estimate"), f"{label} estimate")
    interval = value.get("ci95")
    require(isinstance(interval, list) and len(interval) == 2, f"{label} interval is invalid")
    low, high = (_finite(item, f"{label} interval") for item in interval)
    require(low <= high, f"{label} interval is reversed")
    require(type(value.get("layout_pairs")) is int
            and value["layout_pairs"] == expected_count > 0,
            f"{label} layout count is inconsistent")
    require(
        value.get("resamples") == BOOTSTRAP_RESAMPLES
        and value.get("seed") == ANALYSIS_SEED
        and value.get("resampling_unit") == RESAMPLING_UNIT,
        f"{label} frozen bootstrap identity changed",
    )
    return dict(value)


def _validate_reflection(value: Any, label: str) -> dict[str, Any]:
    exact_keys(value, REFLECTION_KEYS, label)
    definition = value.get("definition")
    require(isinstance(definition, str) and definition.strip() and len(definition) <= 1000,
            f"{label} definition is missing or unbounded")
    by_command = value.get("by_command")
    require(isinstance(by_command, Mapping) and set(by_command) == set(COMMANDS),
            f"{label} command inventory changed")
    command_layout_sets: dict[str, set[str]] = {}
    for command in COMMANDS:
        row = by_command[command]
        exact_keys(row, REFLECTION_COMMAND_KEYS, f"{label} {command}")
        ids = _ordered_layout_subset(row.get("eligible_layout_ids"),
                                     f"{label} {command} eligible layouts")
        paired = row.get("paired_layout_values")
        require(isinstance(paired, list) and len(paired) == len(ids),
                f"{label} {command} paired values are incomplete")
        for index, item in enumerate(paired):
            exact_keys(
                item,
                {"layout_pair_id", "actual_relative_motion", "predicted_relative_motion", "discrepancy"},
                f"{label} {command} paired value {index}",
            )
            require(item.get("layout_pair_id") == ids[index],
                    f"{label} {command} paired layout order changed")
            actual = _finite(item.get("actual_relative_motion"),
                             f"{label} {command} actual motion")
            predicted = _finite(item.get("predicted_relative_motion"),
                                f"{label} {command} predicted motion")
            discrepancy = _finite(item.get("discrepancy"),
                                  f"{label} {command} discrepancy")
            require(math.isclose(discrepancy, predicted - actual, rel_tol=1e-12, abs_tol=1e-12),
                    f"{label} {command} discrepancy is not source-consistent")
        for key in (
            "actual_reflected_minus_original_motion",
            "predicted_reflected_minus_original_motion",
            "predicted_minus_actual_contrast_discrepancy",
        ):
            _validate_bootstrap_metric(row.get(key), f"{label} {command} {key}", len(ids))
        command_layout_sets[command] = set(ids)

    combined = value.get("combined_command_discrepancy")
    require(isinstance(combined, Mapping), f"{label} combined discrepancy is missing")
    paired = combined.get("paired_layout_values")
    require(isinstance(paired, list), f"{label} combined paired values are missing")
    combined_ids: list[str] = []
    for index, item in enumerate(paired):
        exact_keys(item, {"layout_pair_id", "discrepancy"},
                   f"{label} combined paired value {index}")
        layout = item.get("layout_pair_id")
        require(isinstance(layout, str), f"{label} combined layout is invalid")
        combined_ids.append(layout)
        _finite(item.get("discrepancy"), f"{label} combined {layout} discrepancy")
    _ordered_layout_subset(combined_ids, f"{label} combined layouts")
    require(set(combined_ids) == command_layout_sets["left"] & command_layout_sets["right"],
            f"{label} combined layouts do not equal the command intersection")
    metric = {key: item for key, item in combined.items() if key != "paired_layout_values"}
    _validate_bootstrap_metric(metric, f"{label} combined discrepancy", len(paired))
    return dict(value)


def _validate_forecast(value: Mapping[str, Any], branch: str) -> dict[str, Any]:
    exact_keys(value, FORECAST_KEYS, "forecast figure input")
    verify_signed(value, "forecast figure input")
    require(
        value.get("schema_version") == FORECAST_INPUT_SCHEMA
        and value.get("study_id") == STUDY_ID
        and value.get("cohort_branch") == branch
        and value.get("models_pooled") is False
        and value.get("analysis_seed") == ANALYSIS_SEED
        and value.get("layout_bootstrap_resamples") == BOOTSTRAP_RESAMPLES,
        "forecast figure identity, cohort, or estimator identity changed",
    )
    _sha(value.get("source_final_analysis_sha256"), "forecast final-analysis binding")
    models = value.get("models")
    require(isinstance(models, Mapping) and set(models) == set(BRANCH_MODELS[branch]),
            "forecast models are pooled, missing, or outside the cohort")
    for model in BRANCH_MODELS[branch]:
        row = models[model]
        exact_keys(row, FORECAST_MODEL_KEYS, f"forecast model {model}")
        _validate_skill_metric(row.get("forecast_skill_vs_persistence"),
                               f"{model} forecast skill")
        _validate_reflection(row.get("reflection_layout_effects"),
                             f"{model} reflection layout effects")
    _assert_no_private_paths(value, "forecast figure input")
    return dict(value)


def _validate_scene(value: Mapping[str, Any], branch: str) -> dict[str, Any]:
    exact_keys(value, SCENE_KEYS, "actual scene/timing input")
    verify_signed(value, "actual scene/timing input")
    require(
        value.get("schema_version") == SCENE_INPUT_SCHEMA
        and value.get("study_id") == STUDY_ID
        and value.get("cohort_branch") == branch
        and value.get("scene_coordinate_frame") == "robot_base_m"
        and value.get("scene_source") == "model_blind_physically_gated_confirmation_fixture_freeze"
        and value.get("layout_count") == 24,
        "actual scene/timing identity, cohort, coordinate frame, or count changed",
    )
    _sha(value.get("source_fixture_freeze_sha256"), "scene fixture-freeze binding")
    _sha(value.get("source_final_analysis_sha256"), "scene final-analysis binding")
    rows = value.get("layouts")
    require(isinstance(rows, list) and len(rows) == len(LAYOUTS),
            "actual scene layout inventory is incomplete")
    for index, row in enumerate(rows):
        layout = LAYOUTS[index]
        exact_keys(row, SCENE_ROW_KEYS, f"scene {layout}")
        require(row.get("layout_pair_id") == layout, f"scene layout order changed at {layout}")
        require(type(row.get("environment_seed")) is int,
                f"scene {layout} environment seed is invalid")
        candidate_id = row.get("candidate_id")
        require(isinstance(candidate_id, str) and SAFE_ID_RE.fullmatch(candidate_id) is not None,
                f"scene {layout} candidate ID is invalid")
        _sha(row.get("candidate_payload_sha256"), f"scene {layout} candidate binding")
        _sha(row.get("accepted_gate_record_sha256"), f"scene {layout} gate binding")
        _hash_size(row.get("pose_manifest"), f"scene {layout} pose manifest")
        arms = row.get("layouts")
        require(isinstance(arms, Mapping) and tuple(arms) == ARMS,
                f"scene {layout} arm inventory or order changed")
        for arm in ARMS:
            arm_value = arms[arm]
            exact_keys(arm_value, ARM_KEYS, f"scene {layout}/{arm}")
            positions = arm_value.get("positions_robot_base_m")
            quaternions = arm_value.get("quaternions_wxyz")
            require(isinstance(positions, Mapping) and tuple(positions) == OBJECTS,
                    f"scene {layout}/{arm} position inventory or order changed")
            require(isinstance(quaternions, Mapping) and tuple(quaternions) == OBJECTS,
                    f"scene {layout}/{arm} quaternion inventory or order changed")
            for name in OBJECTS:
                _finite_vector(positions[name], 3, f"scene {layout}/{arm}/{name} position")
                _finite_vector(quaternions[name], 4, f"scene {layout}/{arm}/{name} quaternion")

    timing = value.get("model_timing")
    require(isinstance(timing, Mapping) and set(timing) == set(BRANCH_MODELS[branch]),
            "scene timing models are pooled, missing, or outside the cohort")
    for model in BRANCH_MODELS[branch]:
        row = timing[model]
        exact_keys(row, TIMING_KEYS, f"{model} timing")
        horizon = _finite(row.get("primary_horizon_s"), f"{model} physical horizon")
        tolerance = _finite(row.get("timestamp_tolerance_s"), f"{model} timestamp tolerance")
        frame = row.get("generated_frame_index")
        offset = row.get("target_executed_action_offset")
        cap = row.get("executed_prefix_cap")
        require(horizon >= 0 and tolerance >= 0, f"{model} timing duration is invalid")
        require(type(frame) is int and frame >= 0, f"{model} generated-frame index is invalid")
        require(type(offset) is int and offset >= 0, f"{model} action offset is invalid")
        require(type(cap) is int and cap >= 0,
                f"{model} executed-prefix cap is invalid")
        camera = row.get("camera_id")
        require(isinstance(camera, str) and SAFE_ID_RE.fullmatch(camera) is not None,
                f"{model} camera ID is invalid")
    _assert_no_private_paths(value, "actual scene/timing input")
    return dict(value)


def _load_and_validate(publication_dir: Path) -> dict[str, Any]:
    root = Path(publication_dir)
    _reject_symlink_components(root, "publication directory")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise FigureRenderContractError(f"publication directory is missing: {root}") from error
    require(root.is_dir(), "publication input is not a directory")
    receipt = load_json(root / PUBLICATION_RECEIPT_FILENAME, "publication build receipt")
    branch = _validate_publication_receipt(receipt, root)
    forecast = load_json(root / FORECAST_FILENAME, "forecast figure input")
    scene = load_json(root / SCENE_FILENAME, "actual scene/timing input")
    _validate_forecast(forecast, branch)
    _validate_scene(scene, branch)
    require(
        forecast["source_final_analysis_sha256"] == scene["source_final_analysis_sha256"]
        == receipt["input_bindings"]["final_analysis"]["sha256"],
        "forecast and scene inputs do not bind the same final analysis",
    )
    require(
        scene["source_fixture_freeze_sha256"]
        == receipt["input_bindings"]["confirmation_fixture_freeze"]["sha256"],
        "scene input does not bind the publication fixture freeze",
    )
    return {
        "root": root,
        "branch": branch,
        "models": BRANCH_MODELS[branch],
        "receipt": receipt,
        "forecast": forecast,
        "scene": scene,
    }


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number_text(value: Any) -> str:
    require(type(value) in (int, float), "numeric display received a non-number")
    if type(value) is int:
        return str(value)
    _finite(value, "numeric display value")
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _svg_number(value: float) -> str:
    require(math.isfinite(value), "SVG coordinate is non-finite")
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _svg_document(title: str, description: str, width: int, height: int,
                  body: Iterable[str]) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"  <title id=\"title\">{_e(title)}</title>",
        f"  <desc id=\"desc\">{_e(description)}</desc>",
        "  <style>",
        "    .bg{fill:#fff}.panel{fill:#f8fafc;stroke:#cbd5e1;stroke-width:1}",
        "    .axis{stroke:#64748b;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}",
        "    .zero{stroke:#475569;stroke-width:1.5;stroke-dasharray:5 4}",
        "    .title{font:700 22px sans-serif;fill:#0f172a}",
        "    .subtitle{font:400 12px sans-serif;fill:#475569}",
        "    .label{font:600 12px sans-serif;fill:#1e293b}",
        "    .small{font:400 10px sans-serif;fill:#475569}",
        "    .tiny{font:400 8px sans-serif;fill:#475569}",
        "    .na{font:italic 11px sans-serif;fill:#64748b}",
        "    .ci{stroke:#2563eb;stroke-width:3}.estimate{fill:#1d4ed8}",
        "    .layout{fill:#60a5fa;stroke:#1d4ed8;stroke-width:1}",
        "    .actual{fill:#f59e0b;stroke:#b45309;stroke-width:1}",
        "    .predicted{fill:#14b8a6;stroke:#0f766e;stroke-width:1}",
        "    .discrepancy{fill:#8b5cf6;stroke:#6d28d9;stroke-width:1}",
        "    .banana{fill:#facc15;stroke:#a16207;stroke-width:1.2}",
        "    .bowl{fill:#38bdf8;stroke:#0369a1;stroke-width:1.2}",
        "    .rubiks{fill:#f472b6;stroke:#9d174d;stroke-width:1.2}",
        "    .track{stroke:#94a3b8;stroke-width:2;marker-end:url(#arrow)}",
        "  </style>",
        "  <defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" "
        "markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\">"
        "<path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#94a3b8\"/></marker></defs>",
        "  <rect class=\"bg\" x=\"0\" y=\"0\" width=\"100%\" height=\"100%\"/>",
        *body,
        "</svg>",
        "",
    ]
    payload = "\n".join(lines).encode("utf-8")
    require(not any(marker.encode() in payload for marker in PRIVATE_PATH_MARKERS),
            "generated SVG contains a private path marker")
    return payload


def _domain(values: Iterable[float]) -> tuple[float, float] | None:
    finite = [float(value) for value in values]
    if not finite:
        return None
    low, high = min(finite + [0.0]), max(finite + [0.0])
    span = high - low
    padding = span * 0.08 if span > 0 else max(abs(low) * 0.08, 0.01)
    return low - padding, high + padding


def _scale(value: float, domain: tuple[float, float], left: float, right: float) -> float:
    low, high = domain
    require(high > low, "plot domain collapsed")
    return left + (float(value) - low) * (right - left) / (high - low)


def _axis_lines(domain: tuple[float, float], left: float, right: float,
                top: float, bottom: float) -> list[str]:
    lines: list[str] = []
    for index in range(5):
        value = domain[0] + index * (domain[1] - domain[0]) / 4
        x = _scale(value, domain, left, right)
        lines.append(
            f'  <line class="grid" x1="{_svg_number(x)}" y1="{_svg_number(top)}" '
            f'x2="{_svg_number(x)}" y2="{_svg_number(bottom)}"/>'
        )
        lines.append(
            f'  <text class="tiny" x="{_svg_number(x)}" y="{_svg_number(bottom + 16)}" '
            f'text-anchor="middle">{_e(f"{value:.4g}")}</text>'
        )
    if domain[0] <= 0 <= domain[1]:
        zero = _scale(0.0, domain, left, right)
        lines.append(
            f'  <line class="zero" x1="{_svg_number(zero)}" y1="{_svg_number(top)}" '
            f'x2="{_svg_number(zero)}" y2="{_svg_number(bottom)}"/>'
        )
    return lines


def _render_skill(model: str, forecast: Mapping[str, Any]) -> bytes:
    metric = forecast["models"][model]["forecast_skill_vs_persistence"]
    rows = {row["layout_pair_id"]: row["value"] for row in metric["layout_pair_values"]}
    values = [metric["ci95"][0], metric["ci95"][1], metric["estimate"], *rows.values()]
    domain = _domain(values)
    assert domain is not None
    width, height = 900, 840
    plot_left, plot_right = 180.0, 850.0
    lines = [
        f'  <text class="title" x="36" y="42">{_e(model)} forecast skill vs persistence</text>',
        "  <text class=\"subtitle\" x=\"36\" y=\"65\">Source estimates and layout values only; no estimator recomputation or cross-model pooling.</text>",
        f'  <rect class="panel" x="28" y="84" width="844" height="92" rx="6"/>',
        f'  <text class="label" x="48" y="112">Estimate (95% layout-bootstrap CI)</text>',
    ]
    estimate_y = 145.0
    x_low = _scale(metric["ci95"][0], domain, plot_left, plot_right)
    x_high = _scale(metric["ci95"][1], domain, plot_left, plot_right)
    x_estimate = _scale(metric["estimate"], domain, plot_left, plot_right)
    lines.extend([
        f'  <line class="ci" x1="{_svg_number(x_low)}" y1="{estimate_y}" '
        f'x2="{_svg_number(x_high)}" y2="{estimate_y}"/>',
        f'  <circle class="estimate" cx="{_svg_number(x_estimate)}" cy="{estimate_y}" r="6">'
        f'<title>estimate={_e(_number_text(metric["estimate"]))}; '
        f'ci95=[{_e(_number_text(metric["ci95"][0]))}, {_e(_number_text(metric["ci95"][1]))}]; '
        f'layout_pairs={metric["layout_pairs"]}</title></circle>',
        f'  <text class="small" x="48" y="164">estimate={_e(_number_text(metric["estimate"]))}; '
        f'CI=[{_e(_number_text(metric["ci95"][0]))}, {_e(_number_text(metric["ci95"][1]))}]; '
        f'n={metric["layout_pairs"]}</text>',
        '  <rect class="panel" x="28" y="194" width="844" height="602" rx="6"/>',
        '  <text class="label" x="48" y="222">Per-layout source values</text>',
    ])
    top, bottom = 238.0, 760.0
    lines.extend(_axis_lines(domain, plot_left, plot_right, top, bottom))
    row_height = (bottom - top) / len(LAYOUTS)
    excluded = set(metric["excluded_incomplete_layout_ids"])
    for index, layout in enumerate(LAYOUTS):
        y = top + (index + 0.5) * row_height
        lines.append(f'  <text class="tiny" x="160" y="{_svg_number(y + 3)}" text-anchor="end">{layout}</text>')
        if layout in rows:
            value = rows[layout]
            x = _scale(value, domain, plot_left, plot_right)
            lines.append(
                f'  <circle class="layout" cx="{_svg_number(x)}" cy="{_svg_number(y)}" r="3.5">'
                f'<title>{layout} value={_e(_number_text(value))}</title></circle>'
            )
        else:
            require(layout in excluded, f"{model} plot encountered an undeclared missing layout")
            lines.append(
                f'  <text class="na" x="{_svg_number(plot_left + 8)}" y="{_svg_number(y + 4)}">not estimable</text>'
            )
    lines.append(
        f'  <text class="tiny" x="36" y="820">analysis_seed={ANALYSIS_SEED}; '
        f'bootstrap_resamples={BOOTSTRAP_RESAMPLES}; complete layout IDs are shown individually.</text>'
    )
    return _svg_document(
        f"{model} forecast skill versus persistence",
        "Separate-model source values copied from the signed publication figure input.",
        width, height, lines,
    )


def _metric_values(metric: Mapping[str, Any]) -> list[float]:
    if metric.get("estimate") is None:
        return []
    return [metric["ci95"][0], metric["ci95"][1], metric["estimate"]]


def _render_reflection(model: str, forecast: Mapping[str, Any]) -> bytes:
    reflection = forecast["models"][model]["reflection_layout_effects"]
    forest_rows: list[tuple[str, Mapping[str, Any], str]] = []
    for command in COMMANDS:
        command_row = reflection["by_command"][command]
        forest_rows.extend([
            (f"{command}: actual reflected-original", command_row["actual_reflected_minus_original_motion"], "actual"),
            (f"{command}: predicted reflected-original", command_row["predicted_reflected_minus_original_motion"], "predicted"),
            (f"{command}: predicted-actual discrepancy", command_row["predicted_minus_actual_contrast_discrepancy"], "discrepancy"),
        ])
    forest_rows.append(("combined command discrepancy", reflection["combined_command_discrepancy"], "discrepancy"))
    values: list[float] = []
    for _, metric, _ in forest_rows:
        values.extend(_metric_values(metric))
    paired: dict[str, dict[str, float]] = {layout: {} for layout in LAYOUTS}
    for command in COMMANDS:
        for row in reflection["by_command"][command]["paired_layout_values"]:
            paired[row["layout_pair_id"]][command] = row["discrepancy"]
            values.append(row["discrepancy"])
    for row in reflection["combined_command_discrepancy"]["paired_layout_values"]:
        paired[row["layout_pair_id"]]["combined"] = row["discrepancy"]
        values.append(row["discrepancy"])
    domain = _domain(values)
    width, height = 1050, 940
    plot_left, plot_right = 310.0, 1000.0
    lines = [
        f'  <text class="title" x="36" y="42">{_e(model)} reflection layout effects</text>',
        "  <text class=\"subtitle\" x=\"36\" y=\"65\">Separate-model source contrasts; either sign retained; no missing value is replaced.</text>",
        '  <rect class="panel" x="28" y="84" width="994" height="286" rx="6"/>',
        '  <text class="label" x="48" y="112">Source estimates (95% layout-bootstrap CI)</text>',
    ]
    forest_top, forest_bottom = 130.0, 340.0
    if domain is None:
        lines.append('  <text class="na" x="48" y="160">No reflection contrast values are available; all source metrics are not estimable.</text>')
    else:
        lines.extend(_axis_lines(domain, plot_left, plot_right, forest_top, forest_bottom))
    row_height = (forest_bottom - forest_top) / len(forest_rows)
    for index, (label, metric, css_class) in enumerate(forest_rows):
        y = forest_top + (index + 0.5) * row_height
        lines.append(f'  <text class="tiny" x="292" y="{_svg_number(y + 3)}" text-anchor="end">{_e(label)}</text>')
        if metric.get("estimate") is None:
            lines.append(f'  <text class="na" x="320" y="{_svg_number(y + 4)}">not estimable (0 layout pairs)</text>')
        else:
            assert domain is not None
            low, high = metric["ci95"]
            x_low = _scale(low, domain, plot_left, plot_right)
            x_high = _scale(high, domain, plot_left, plot_right)
            x_est = _scale(metric["estimate"], domain, plot_left, plot_right)
            lines.extend([
                f'  <line class="ci" x1="{_svg_number(x_low)}" y1="{_svg_number(y)}" '
                f'x2="{_svg_number(x_high)}" y2="{_svg_number(y)}"/>',
                f'  <circle class="{css_class}" cx="{_svg_number(x_est)}" cy="{_svg_number(y)}" r="5">'
                f'<title>{_e(label)}; estimate={_e(_number_text(metric["estimate"]))}; '
                f'ci95=[{_e(_number_text(low))}, {_e(_number_text(high))}]; '
                f'layout_pairs={metric["layout_pairs"]}</title></circle>',
            ])
    lines.extend([
        '  <rect class="panel" x="28" y="390" width="994" height="506" rx="6"/>',
        '  <text class="label" x="48" y="418">Per-layout predicted-minus-actual reflection discrepancy</text>',
        '  <text class="tiny" x="48" y="437">circles: left; squares: right; diamonds: combined command value</text>',
    ])
    layout_top, layout_bottom = 454.0, 860.0
    if domain is None:
        lines.append('  <text class="na" x="48" y="478">No per-layout reflection discrepancy values are available.</text>')
    else:
        lines.extend(_axis_lines(domain, plot_left, plot_right, layout_top, layout_bottom))
        layout_height = (layout_bottom - layout_top) / len(LAYOUTS)
        for index, layout in enumerate(LAYOUTS):
            y = layout_top + (index + 0.5) * layout_height
            lines.append(f'  <text class="tiny" x="292" y="{_svg_number(y + 3)}" text-anchor="end">{layout}</text>')
            if not paired[layout]:
                lines.append(f'  <text class="na" x="320" y="{_svg_number(y + 3)}">not estimable</text>')
                continue
            if "left" in paired[layout]:
                x = _scale(paired[layout]["left"], domain, plot_left, plot_right)
                lines.append(f'  <circle class="actual" cx="{_svg_number(x)}" cy="{_svg_number(y - 3)}" r="3"><title>{layout} left discrepancy={_e(_number_text(paired[layout]["left"]))}</title></circle>')
            if "right" in paired[layout]:
                x = _scale(paired[layout]["right"], domain, plot_left, plot_right)
                lines.append(f'  <rect class="predicted" x="{_svg_number(x - 3)}" y="{_svg_number(y)}" width="6" height="6"><title>{layout} right discrepancy={_e(_number_text(paired[layout]["right"]))}</title></rect>')
            if "combined" in paired[layout]:
                x = _scale(paired[layout]["combined"], domain, plot_left, plot_right)
                points = " ".join(
                    f"{_svg_number(px)},{_svg_number(py)}"
                    for px, py in ((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y))
                )
                lines.append(f'  <polygon class="discrepancy" points="{points}"><title>{layout} combined discrepancy={_e(_number_text(paired[layout]["combined"]))}</title></polygon>')
    lines.append(f'  <text class="tiny" x="36" y="922">{_e(reflection["definition"])}</text>')
    return _svg_document(
        f"{model} reflection layout effects",
        "Separate-model reflection contrast inputs, including explicit unavailable values.",
        width, height, lines,
    )


def _scene_coordinate_domains(
    scene: Mapping[str, Any],
) -> tuple[
    tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
]:
    xs: list[float] = []
    ys: list[float] = []
    for row in scene["layouts"]:
        for arm in ARMS:
            for position in row["layouts"][arm]["positions_robot_base_m"].values():
                xs.append(float(position[0]))
                ys.append(float(position[1]))
    require(xs and ys, "scene coordinate inventory is empty")
    def padded(values: list[float]) -> tuple[float, float]:
        low, high = min(values), max(values)
        span = high - low
        padding = span * 0.08 if span > 0 else 0.01
        return low - padding, high + padding
    return (min(xs), max(xs)), (min(ys), max(ys)), padded(xs), padded(ys)


def _object_marker(name: str, x: float, y: float, title: str) -> str:
    if name == "banana":
        return f'  <circle class="banana" cx="{_svg_number(x)}" cy="{_svg_number(y)}" r="5"><title>{_e(title)}</title></circle>'
    if name == "bowl":
        return f'  <rect class="bowl" x="{_svg_number(x - 5)}" y="{_svg_number(y - 5)}" width="10" height="10"><title>{_e(title)}</title></rect>'
    points = " ".join(
        f"{_svg_number(px)},{_svg_number(py)}"
        for px, py in ((x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y))
    )
    return f'  <polygon class="rubiks" points="{points}"><title>{_e(title)}</title></polygon>'


def _render_scenes(scene: Mapping[str, Any]) -> bytes:
    width = 1680
    columns, rows_count = 3, 8
    card_w, card_h = 536, 184
    margin_x, top = 28, 104
    height = top + rows_count * card_h + 62
    x_observed, y_observed, x_domain, y_domain = _scene_coordinate_domains(scene)
    lines = [
        '  <text class="title" x="36" y="42">Actual confirmation fixture layouts</text>',
        '  <text class="subtitle" x="36" y="65">Robot-base x-y projection; exact source x, y, z and quaternion values are retained in each marker title.</text>',
        f'  <text class="small" x="36" y="84">observed x range [{_e(_number_text(x_observed[0]))}, {_e(_number_text(x_observed[1]))}] m; '
        f'observed y range [{_e(_number_text(y_observed[0]))}, {_e(_number_text(y_observed[1]))}] m; '
        f'source layouts={scene["layout_count"]}; display adds symmetric padding only</text>',
    ]
    for index, row in enumerate(scene["layouts"]):
        column, grid_row = index % columns, index // columns
        x0 = margin_x + column * (card_w + 8)
        y0 = top + grid_row * card_h
        lines.extend([
            f'  <rect class="panel" x="{x0}" y="{y0}" width="{card_w}" height="174" rx="6"/>',
            f'  <text class="label" x="{x0 + 12}" y="{y0 + 20}">{row["layout_pair_id"]}</text>',
        ])
        for arm_index, arm in enumerate(ARMS):
            panel_left = x0 + 12 + arm_index * 258
            panel_top = y0 + 34
            panel_right = panel_left + 246
            panel_bottom = panel_top + 116
            lines.extend([
                f'  <text class="tiny" x="{panel_left}" y="{panel_top - 5}">{arm}</text>',
                f'  <line class="axis" x1="{panel_left}" y1="{panel_bottom}" x2="{panel_right}" y2="{panel_bottom}"/>',
                f'  <line class="axis" x1="{panel_left}" y1="{panel_top}" x2="{panel_left}" y2="{panel_bottom}"/>',
            ])
            arm_value = row["layouts"][arm]
            for name in OBJECTS:
                position = arm_value["positions_robot_base_m"][name]
                quaternion = arm_value["quaternions_wxyz"][name]
                x = _scale(position[0], x_domain, panel_left + 6, panel_right - 6)
                y = _scale(position[1], y_domain, panel_bottom - 6, panel_top + 6)
                title = (
                    f'{row["layout_pair_id"]}/{arm}/{name}; '
                    f'position_robot_base_m=[{", ".join(_number_text(item) for item in position)}]; '
                    f'quaternion_wxyz=[{", ".join(_number_text(item) for item in quaternion)}]'
                )
                lines.append(_object_marker(name, x, y, title))
                lines.append(
                    f'  <text class="tiny" x="{_svg_number(x + 7)}" y="{_svg_number(y - 5)}">'
                    f'{_e(name[0].upper())} z={_e(_number_text(position[2]))}</text>'
                )
            lines.append(f'  <text class="tiny" x="{panel_left}" y="{panel_bottom + 13}">x →; y ↑</text>')
    lines.extend([
        f'  <circle class="banana" cx="36" cy="{height - 28}" r="5"/>',
        f'  <text class="small" x="48" y="{height - 24}">banana</text>',
        f'  <rect class="bowl" x="122" y="{height - 33}" width="10" height="10"/>',
        f'  <text class="small" x="138" y="{height - 24}">bowl</text>',
        f'  <polygon class="rubiks" points="205,{height - 34} 211,{height - 28} 205,{height - 22} 199,{height - 28}"/>',
        f'  <text class="small" x="219" y="{height - 24}">rubiks_cube</text>',
        f'  <text class="small" x="360" y="{height - 24}">coordinate frame: robot_base_m; original and reflected arms use one global measured scale.</text>',
    ])
    return _svg_document(
        "Actual confirmation fixture layouts",
        "All 24 model-blind accepted confirmation layout pairs in robot-base coordinates.",
        width, height, lines,
    )


def _render_timing(scene: Mapping[str, Any], models: Sequence[str]) -> bytes:
    width = 1160
    card_height = 250
    height = 112 + len(models) * card_height + 55
    lines = [
        '  <text class="title" x="36" y="42">Qualified forecast / physical / control timing identities</text>',
        '  <text class="subtitle" x="36" y="65">Three independently recorded target domains. Arrow placement is schematic, not a conversion between domains.</text>',
        '  <text class="label" x="36" y="88">No generated-frame ↔ physical-time ↔ executed-action equivalence is inferred.</text>',
    ]
    for index, model in enumerate(models):
        row = scene["model_timing"][model]
        y0 = 108 + index * card_height
        lines.extend([
            f'  <rect class="panel" x="28" y="{y0}" width="1104" height="232" rx="8"/>',
            f'  <text class="title" x="48" y="{y0 + 32}">{model}</text>',
            f'  <text class="small" x="1020" y="{y0 + 30}" text-anchor="end">camera={_e(row["camera_id"])}</text>',
            f'  <text class="small" x="1020" y="{y0 + 48}" text-anchor="end">timestamp tolerance={_e(_number_text(row["timestamp_tolerance_s"]))} s</text>',
        ])
        tracks = (
            ("generated prediction", f'frame index = {_number_text(row["generated_frame_index"])}'),
            ("physical target", f'horizon = {_number_text(row["primary_horizon_s"])} s'),
            ("executed control target", f'action offset = {_number_text(row["target_executed_action_offset"])}'),
        )
        for track_index, (name, value) in enumerate(tracks):
            y = y0 + 82 + track_index * 46
            lines.extend([
                f'  <text class="label" x="48" y="{y + 4}">{_e(name)}</text>',
                f'  <line class="track" x1="230" y1="{y}" x2="500" y2="{y}"/>',
                f'  <rect class="panel" x="524" y="{y - 17}" width="286" height="34" rx="4"/>',
                f'  <text class="label" x="667" y="{y + 4}" text-anchor="middle">{_e(value)}</text>',
            ])
        lines.extend([
            f'  <text class="small" x="838" y="{y0 + 96}">executed prefix cap = {_e(_number_text(row["executed_prefix_cap"]))} actions</text>',
            f'  <text class="tiny" x="838" y="{y0 + 120}">The cap bounds exposed context; it is not used</text>',
            f'  <text class="tiny" x="838" y="{y0 + 136}">to derive the physical horizon or frame index.</text>',
            f'  <text class="tiny" x="48" y="{y0 + 216}">Exact signed fields shown verbatim; arrows carry no shared numerical scale.</text>',
        ])
    lines.append(
        f'  <text class="tiny" x="36" y="{height - 22}">cohort={_e(scene["cohort_branch"])}; '
        f'models shown separately={_e(",".join(models))}</text>'
    )
    return _svg_document(
        "Qualified timing identities",
        "Explicit generated-frame, physical-horizon, and executed-action-offset fields without inferred conversion.",
        width, height, lines,
    )


def _output_names(models: Sequence[str]) -> tuple[str, ...]:
    dynamic: list[str] = list(BASE_OUTPUTS)
    for model in models:
        dynamic.extend((f"forecast_skill_{model.lower()}.svg", f"reflection_layout_effects_{model.lower()}.svg"))
    return tuple(dynamic)


def _render_documents(context: Mapping[str, Any]) -> dict[str, bytes]:
    forecast, scene = context["forecast"], context["scene"]
    documents = {
        "actual_scene_layouts.svg": _render_scenes(scene),
        "qualified_timing.svg": _render_timing(scene, context["models"]),
    }
    for model in context["models"]:
        documents[f"forecast_skill_{model.lower()}.svg"] = _render_skill(model, forecast)
        documents[f"reflection_layout_effects_{model.lower()}.svg"] = _render_reflection(model, forecast)
    require(tuple(documents) == _output_names(context["models"]), "renderer output order changed")
    return documents


def _write_bytes_at(directory_fd: int, name: str, payload: bytes) -> None:
    require(
        SAFE_OUTPUT_NAME_RE.fullmatch(name) is not None and "/" not in name,
        "rendered output name is unsafe",
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise FigureRenderContractError(
            f"cannot exclusively create rendered output {name}: {error}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory_fd(descriptor: int) -> None:
    os.fsync(descriptor)


def _renameat2_noreplace(
    parent_fd: int, source_name: str, target_name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None,
            "atomic no-replace directory publication is unavailable on this platform")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(target_name), 1
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FigureRenderContractError(f"refusing to replace figure output: {target_name}")
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise FigureRenderContractError(
            "atomic no-replace directory publication is unavailable on this filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), target_name)


def _rename_directory_noreplace_at(
    parent_fd: int, source_name: str, target_name: str
) -> None:
    """Atomically publish within one held parent fd without replacing a target."""

    _renameat2_noreplace(parent_fd, source_name, target_name)


def _restore_staging_after_failed_publication(
    parent_fd: int, staging_name: str, target_name: str, staging_fd: int
) -> None:
    """Undo a just-completed publish before reporting a parent identity race."""

    _require_child_matches_fd(parent_fd, target_name, staging_fd)
    _require_staging_child_absent(parent_fd, staging_name)
    _renameat2_noreplace(parent_fd, target_name, staging_name)
    _require_child_matches_fd(parent_fd, staging_name, staging_fd)
    _fsync_directory_fd(parent_fd)


def _validate_svg(path: Path, label: str) -> None:
    try:
        root = ET.fromstring(path.read_bytes())
    except (OSError, ET.ParseError) as error:
        raise FigureRenderContractError(f"generated {label} is not valid SVG XML: {error}") from error
    require(root.tag == "{http://www.w3.org/2000/svg}svg", f"generated {label} root is not SVG")
    payload = path.read_text(encoding="utf-8")
    require("<script" not in payload.lower() and "file://" not in payload,
            f"generated {label} contains active or private content")
    require(not any(marker in payload for marker in PRIVATE_PATH_MARKERS),
            f"generated {label} contains a private path")


def _validate_render_receipt(
    receipt: Mapping[str, Any], expected_outputs: set[str], *,
    publication_root: Path | None = None,
    expected_source_authentication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    exact_keys(receipt, RENDER_RECEIPT_KEYS, "figure render receipt")
    verify_signed(receipt, "figure render receipt")
    branch = receipt.get("cohort_branch")
    require(
        receipt.get("schema_version") == RENDER_RECEIPT_SCHEMA
        and receipt.get("study_id") == STUDY_ID
        and receipt.get("status") == "rendered_source_only_publication_figures"
        and branch in BRANCH_MODELS
        and receipt.get("model_ids") == list(BRANCH_MODELS[branch])
        and receipt.get("models_pooled") is False
        and receipt.get("publication_to_renderer_to_control_ancestry") is True
        and receipt.get("receipt_self_binding") == "payload_sha256"
        and receipt.get("scientific_estimates_recomputed") is False
        and receipt.get("frame_to_action_mapping_inferred") is False
        and receipt.get("labels_created") is False
        and receipt.get("paper_files_edited_or_generated") is False
        and receipt.get("claim_boundary") == RENDER_CLAIM_BOUNDARY,
        "figure render receipt broadens or changes the presentation-only operation",
    )
    for key in ("publication_source_commit", "renderer_source_commit"):
        require(
            isinstance(receipt.get(key), str)
            and COMMIT_RE.fullmatch(receipt[key]) is not None,
            f"figure render receipt {key} is invalid",
        )
    for key in (
        "source_publication_build_receipt_payload_sha256",
        "source_final_analysis_sha256", "source_fixture_freeze_sha256",
    ):
        _sha(receipt.get(key), f"figure render receipt {key}")

    bindings = receipt.get("input_bindings")
    exact_keys(bindings, RENDER_INPUT_BINDING_KEYS, "figure render input bindings")
    for relative, raw in bindings.items():
        _inventory_descriptor(raw, relative, f"figure render input {relative}")
    if publication_root is not None:
        root = Path(publication_root).resolve(strict=True)
        observed_bindings = {
            relative: file_descriptor(root / relative, public_path=relative)
            for relative in sorted(RENDER_INPUT_BINDING_KEYS)
        }
        require(bindings == observed_bindings,
                "figure render input bindings differ from the publication bundle")
        source_build_receipt = load_json(
            root / PUBLICATION_RECEIPT_FILENAME, "source publication build receipt"
        )
        require(
            receipt["source_publication_build_receipt_payload_sha256"]
            == source_build_receipt.get("payload_sha256")
            and receipt["publication_source_commit"]
            == source_build_receipt.get("publication_source_commit"),
            "figure render receipt differs from its source publication receipt",
        )
        source_forecast = load_json(root / FORECAST_FILENAME, "source forecast figure input")
        source_scene = load_json(root / SCENE_FILENAME, "source actual scene/timing input")
        require(
            receipt["source_final_analysis_sha256"]
            == source_forecast.get("source_final_analysis_sha256")
            == source_scene.get("source_final_analysis_sha256")
            and receipt["source_fixture_freeze_sha256"]
            == source_scene.get("source_fixture_freeze_sha256"),
            "figure render receipt differs from its signed scientific source bindings",
        )

    sources = receipt.get("committed_renderer_sources")
    exact_keys(sources, COMMITTED_RENDERER_SOURCE_KEYS,
               "committed renderer sources")
    expected_source_paths = {
        "renderer_source": _repository_relative(
            RENDERER_SOURCE_PATH, REPOSITORY, "renderer source"
        ),
        "renderer_contract": _repository_relative(
            CONTRACT_PATH, REPOSITORY, "renderer contract"
        ),
    }
    for key, relative in expected_source_paths.items():
        descriptor = _inventory_descriptor(sources[key], relative, f"committed {key}")
        source_path = RENDERER_SOURCE_PATH if key == "renderer_source" else CONTRACT_PATH
        require(file_descriptor(source_path, public_path=relative) == descriptor,
                f"committed {key} differs from the current source bytes")
    remote = _validate_authenticated_remote(
        receipt.get("authorized_renderer_remote"), receipt["renderer_source_commit"]
    )
    require(
        remote["remote_ref_commit"] == receipt["authorized_renderer_remote"]["remote_ref_commit"],
        "authenticated renderer control commit changed",
    )
    if expected_source_authentication is not None:
        require(
            receipt["renderer_source_commit"]
            == expected_source_authentication.get("renderer_source_commit")
            and receipt["publication_to_renderer_to_control_ancestry"]
            == expected_source_authentication.get(
                "publication_to_renderer_to_control_ancestry"
            )
            and receipt["authorized_renderer_remote"]
            == expected_source_authentication.get("authorized_renderer_remote")
            and receipt["committed_renderer_sources"]
            == expected_source_authentication.get("committed_renderer_sources"),
            "figure render receipt differs from the authenticated renderer sources",
        )

    inventory = receipt.get("output_inventory")
    require(isinstance(inventory, Mapping) and set(inventory) == expected_outputs,
            "figure render inventory changed")
    require(receipt.get("output_count_including_self_signed_receipt") == len(inventory) + 1,
            "figure render output count changed")
    for relative, raw in inventory.items():
        _inventory_descriptor(raw, relative, f"rendered output {relative}")
    _assert_no_private_paths(receipt, "figure render receipt")
    return dict(receipt)


def _validate_staged(
    target: Path, receipt: Mapping[str, Any], expected_outputs: set[str], *,
    publication_root: Path,
    expected_source_authentication: Mapping[str, Any],
) -> None:
    staged_receipt = load_json(
        target / "render_receipt.json",
        "staged figure render receipt",
        trusted_open_directory_fd_path=True,
    )
    require(staged_receipt == receipt, "staged figure render receipt changed after creation")
    _validate_render_receipt(
        staged_receipt,
        expected_outputs,
        publication_root=publication_root,
        expected_source_authentication=expected_source_authentication,
    )
    inventory = staged_receipt.get("output_inventory")
    assert isinstance(inventory, Mapping)
    for relative, raw in inventory.items():
        descriptor = _inventory_descriptor(raw, relative, f"rendered output {relative}")
        require(file_descriptor(target / relative, public_path=relative) == descriptor,
                f"rendered output changed: {relative}")
        _validate_svg(target / relative, relative)
    all_entries = list(target.rglob("*"))
    require(not any(path.is_symlink() for path in all_entries),
            "render directory contains a symlink")
    actual = {
        path.relative_to(target).as_posix()
        for path in all_entries if path.is_file()
    }
    require(actual == expected_outputs | {"render_receipt.json"},
            "render directory contains an undeclared output")


def _stable_source_authentication(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "renderer_source_commit": value["renderer_source_commit"],
        "publication_to_renderer_to_control_ancestry": value[
            "publication_to_renderer_to_control_ancestry"
        ],
        "committed_renderer_sources": value["committed_renderer_sources"],
        "authorized_renderer_remote": dict(value["authorized_renderer_remote"]),
    }
    result["authorized_renderer_remote"].pop("authenticated_graph_bytes", None)
    return result


def render_publication(publication_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Validate one immutable publication bundle and atomically render SVGs."""

    requested_target = Path(output_dir)
    absolute_target = Path(os.path.abspath(os.fspath(requested_target)))
    require(
        absolute_target != Path(os.sep)
        and SAFE_OUTPUT_NAME_RE.fullmatch(absolute_target.name) is not None,
        "figure output name is unsafe",
    )
    parent_fd, parent_path = _open_directory_nofollow(
        absolute_target.parent, "figure output parent"
    )
    staging_name: str | None = None
    staging_fd: int | None = None
    try:
        _require_same_open_directory(parent_path, parent_fd, "figure output parent")
        _require_child_absent(parent_fd, absolute_target.name, "figure output")
        context = _load_and_validate(Path(publication_dir))
        resolved_target = parent_path / absolute_target.name
        require(
            resolved_target.parent != context["root"]
            and context["root"] not in resolved_target.parents,
            "figure output must not be created inside the immutable publication bundle",
        )
        source_authentication = _authenticate_renderer_sources(context["receipt"])
        documents = _render_documents(context)

        _require_same_open_directory(parent_path, parent_fd, "figure output parent")
        _require_child_absent(parent_fd, absolute_target.name, "figure output")
        staging_name, staging_fd = _create_staging_directory(parent_fd)
        temporary = _opened_directory_path(staging_fd)
        for relative, payload in documents.items():
            _write_bytes_at(staging_fd, relative, payload)
        output_inventory = {
            relative: file_descriptor(temporary / relative, public_path=relative)
            for relative in sorted(documents)
        }
        input_bindings = {
            FORECAST_FILENAME: file_descriptor(
                context["root"] / FORECAST_FILENAME, public_path=FORECAST_FILENAME
            ),
            SCENE_FILENAME: file_descriptor(
                context["root"] / SCENE_FILENAME, public_path=SCENE_FILENAME
            ),
            PUBLICATION_RECEIPT_FILENAME: file_descriptor(
                context["root"] / PUBLICATION_RECEIPT_FILENAME,
                public_path=PUBLICATION_RECEIPT_FILENAME,
            ),
        }
        receipt = sign_document(
            {
                "schema_version": RENDER_RECEIPT_SCHEMA,
                "study_id": STUDY_ID,
                "status": "rendered_source_only_publication_figures",
                "cohort_branch": context["branch"],
                "model_ids": list(context["models"]),
                "models_pooled": False,
                "publication_source_commit": context["receipt"]["publication_source_commit"],
                "renderer_source_commit": source_authentication["renderer_source_commit"],
                "publication_to_renderer_to_control_ancestry": source_authentication[
                    "publication_to_renderer_to_control_ancestry"
                ],
                "authorized_renderer_remote": source_authentication[
                    "authorized_renderer_remote"
                ],
                "committed_renderer_sources": source_authentication[
                    "committed_renderer_sources"
                ],
                "source_publication_build_receipt_payload_sha256": context["receipt"]["payload_sha256"],
                "source_final_analysis_sha256": context["forecast"]["source_final_analysis_sha256"],
                "source_fixture_freeze_sha256": context["scene"]["source_fixture_freeze_sha256"],
                "input_bindings": input_bindings,
                "output_inventory": output_inventory,
                "output_count_including_self_signed_receipt": len(output_inventory) + 1,
                "receipt_self_binding": "payload_sha256",
                "scientific_estimates_recomputed": False,
                "frame_to_action_mapping_inferred": False,
                "labels_created": False,
                "paper_files_edited_or_generated": False,
                "claim_boundary": RENDER_CLAIM_BOUNDARY,
            }
        )
        _write_bytes_at(staging_fd, "render_receipt.json", pretty_bytes(receipt))
        _validate_staged(
            temporary,
            receipt,
            set(documents),
            publication_root=context["root"],
            expected_source_authentication=source_authentication,
        )

        repeated = _load_and_validate(context["root"])
        require(
            repeated["receipt"]["payload_sha256"] == context["receipt"]["payload_sha256"]
            and repeated["forecast"]["payload_sha256"] == context["forecast"]["payload_sha256"]
            and repeated["scene"]["payload_sha256"] == context["scene"]["payload_sha256"],
            "publication inputs drifted during figure rendering",
        )
        repeated_source_authentication = _authenticate_renderer_sources(context["receipt"])
        require(
            _stable_source_authentication(repeated_source_authentication)
            == _stable_source_authentication(source_authentication),
            "renderer source authentication drifted during figure rendering",
        )
        _validate_staged(
            temporary,
            receipt,
            set(documents),
            publication_root=context["root"],
            expected_source_authentication=source_authentication,
        )
        _require_same_open_directory(parent_path, parent_fd, "figure output parent")
        _require_child_absent(parent_fd, absolute_target.name, "figure output")
        _require_child_matches_fd(parent_fd, staging_name, staging_fd)
        _fsync_directory_fd(staging_fd)
        _rename_directory_noreplace_at(
            parent_fd, staging_name, absolute_target.name
        )
        try:
            _require_child_matches_fd(parent_fd, absolute_target.name, staging_fd)
            _require_same_open_directory(parent_path, parent_fd, "figure output parent")
            _fsync_directory_fd(parent_fd)
        except Exception:
            _restore_staging_after_failed_publication(
                parent_fd, staging_name, absolute_target.name, staging_fd
            )
            raise
        staging_name = None
        return receipt
    finally:
        try:
            if staging_name is not None and staging_fd is not None:
                _remove_staging_directory(parent_fd, staging_name, staging_fd)
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    receipt = render_publication(arguments.publication_dir, arguments.output_dir)
    print(json.dumps(
        {
            "status": receipt["status"],
            "cohort_branch": receipt["cohort_branch"],
            "model_ids": receipt["model_ids"],
            "output_count": receipt["output_count_including_self_signed_receipt"],
            "payload_sha256": receipt["payload_sha256"],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
