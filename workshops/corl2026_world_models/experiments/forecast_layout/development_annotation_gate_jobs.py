#!/usr/bin/env python3
"""Fail-closed operations for the development human-annotation gate.

This module does not annotate images and cannot release confirmation.  It turns
an exact, human-reviewed annotation-media preparation into private blinded
packets, exposes one packet batch through a dedicated handoff directory, and
hash-binds the later human response to immutable delivery/collection/revocation
receipts.  The restricted source map never enters the handoff directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterator, Mapping


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
ANNOTATION_PATH = FORECAST_ROOT / "analysis/forecast_annotation_workflow.py"
MEDIA_PATH = FORECAST_ROOT / "analysis/prepare_development_annotation_media.py"
MEDIA_JOBS_PATH = Path(__file__).with_name("development_annotation_media_jobs.py")
CONTRACT_PATH = Path(__file__).with_name("development_annotation_gate_contract.json")

STUDY_ID = "WMF-ABLATION-001"
CONTRACT_SCHEMA = "wmf-development-annotation-gate-contract-v1"
PACKAGE_SCHEMA = "wmf-development-annotation-private-package-v1"
ADJUDICATION_PACKAGE_SCHEMA = "wmf-development-adjudication-private-package-v1"
DELIVERY_SCHEMA = "wmf-development-annotation-delivery-v1"
COLLECTION_SCHEMA = "wmf-development-annotation-collection-v1"
REVOCATION_SCHEMA = "wmf-development-annotation-revocation-v1"
TIME_RECEIPT_SCHEMA = "wmf-development-annotation-time-accounting-v1"
DELIVERY_RULE = "ONE_GLOBAL_SOURCE_FREE_BATCH_WITH_PRIOR_BATCH_COLLECTED_AND_REVOKED"
SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")
SHA_RE = re.compile(r"[0-9a-f]{64}")
SLOTS = ("rater_a", "rater_b", "adjudicator")
MAX_FUTURE_CLOCK_SKEW_SECONDS = 300
DELIVERY_KEYS = {
    "schema_version", "study_id", "stage", "status", "delivery_id",
    "release_sequence", "rater_slot", "batch_ordinal", "batch_count",
    "packet_id", "packet_manifest_sha256", "packet_artifact_sha256",
    "package_receipt_sha256", "prior_revocation_receipt_sha256", "release_rule",
    "released_at_utc", "released_by_code_sha256", "source_identity_present",
    "restricted_map_present", "science_counts", "labels_created_by_job",
    "safe_to_release_confirmation", "claim_boundary", "payload_sha256",
}
COLLECTION_KEYS = {
    "schema_version", "study_id", "stage", "status", "delivery_id",
    "release_sequence", "rater_slot", "batch_ordinal", "packet_id",
    "delivery_receipt_sha256", "response", "collector_code_sha256",
    "rater_code_sha256", "response_timing", "annotation_count_received",
    "labels_created_by_job", "science_counts", "collected_at_utc",
    "safe_to_release_confirmation", "claim_boundary", "payload_sha256",
}
REVOCATION_KEYS = {
    "schema_version", "study_id", "stage", "status", "delivery_id",
    "release_sequence", "rater_slot", "batch_ordinal", "packet_id",
    "delivery_receipt_sha256", "collection_receipt_sha256", "response_sha256",
    "revoked_by_code_sha256", "revoked_at_utc", "controlled_handoff_path_absent",
    "external_distribution_or_retained_copy_revocation_claimed",
    "labels_created_by_job", "science_counts", "safe_to_release_confirmation",
    "claim_boundary", "payload_sha256",
}


class AnnotationGateError(RuntimeError):
    """Raised when an annotation operation cannot prove its gate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationGateError(message)


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None,
            f"cannot load required module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


annotation = _load_module(ANNOTATION_PATH, "wmf_development_annotation_gate_workflow")
media = _load_module(MEDIA_PATH, "wmf_development_annotation_gate_media")
media_jobs = _load_module(MEDIA_JOBS_PATH, "wmf_development_annotation_gate_media_jobs")


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
    }


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "development",
        "artifact_schemas": {
            "private_package": PACKAGE_SCHEMA,
            "adjudication_package": ADJUDICATION_PACKAGE_SCHEMA,
            "delivery": DELIVERY_SCHEMA,
            "collection": COLLECTION_SCHEMA,
            "revocation": REVOCATION_SCHEMA,
            "time_accounting": TIME_RECEIPT_SCHEMA,
        },
        "required_existing_calls": [
            "prepare_development_annotation_media.finalize_reviewed_inventory",
            "forecast_annotation_workflow.validate_freeze",
            "forecast_annotation_workflow.package_packets",
        ],
        "human_input_policy": {
            "pixel_blindness_receipt_required": True,
            "approved_rubric_and_examples_required_through_freeze": True,
            "locked_response_required_for_collection": True,
            "job_may_create_or_fill_labels": False,
            "job_may_invent_reviewer_rater_or_adjudicator": False,
        },
        "publication_policy": {
            "delivery_rule": DELIVERY_RULE,
            "global_active_batch_limit": 1,
            "batch_tree_source_free": True,
            "restricted_map_private": True,
            "delivery_collection_revocation_receipts_immutable": True,
            "prior_batch_collection_and_revocation_required": True,
            "publication_uses_atomic_directory_rename": True,
            "external_distribution_or_access_revocation_claimed": False,
        },
        "response_time_policy": {
            "annotation_seconds_must_fit_session": True,
            "same_rater_batches_must_not_overlap": True,
            "same_rater_batches_follow_frozen_packet_order": True,
            "response_start_not_before_delivery": True,
            "response_lock_not_after_collection": True,
            "collector_timestamp_and_response_hash_bound": True,
            "max_future_clock_skew_seconds": MAX_FUTURE_CLOCK_SKEW_SECONDS,
            "adjudicator_total_reported_when_required": True,
        },
        "confirmation_policy": {
            "confirmation_jobs_released": 0,
            "safe_to_release_confirmation": False,
            "development_summary_consensus_and_human_usability_decision_still_required": True,
        },
        "science_counts": _zero_science_counts(),
    }


def validate_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    value = _load_json(path, "annotation gate contract")
    require(value == _expected_contract(), "annotation gate contract fields changed")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    path = Path(path)
    cursor = path.absolute()
    while True:
        require(not cursor.is_symlink(), f"descriptor path contains a symlink: {path}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    path = path.resolve(strict=True)
    require(path.is_file(), f"descriptor is not a regular file: {path}")
    return {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _verify_descriptor(value: Any, label: str) -> tuple[dict[str, Any], Path]:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256", "bytes"},
            f"{label} descriptor fields changed")
    path_text = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    require(isinstance(path_text, str) and Path(path_text).is_absolute(),
            f"{label} path must be absolute")
    require(isinstance(digest, str) and SHA_RE.fullmatch(digest) is not None,
            f"{label} SHA-256 is invalid")
    require(type(size) is int and size > 0, f"{label} byte count is invalid")
    path = Path(path_text)
    require(path.exists() and not path.is_symlink(), f"{label} is missing or a symlink")
    observed = _descriptor(path)
    require(observed == dict(value), f"{label} bytes or SHA-256 changed")
    return observed, path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            require(key not in result, f"{label} contains duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise AnnotationGateError(f"{label} contains non-finite JSON number: {token}")

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnnotationGateError(f"cannot read {label}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _sign(value: dict[str, Any]) -> dict[str, Any]:
    return annotation.sign_document(value)


def _verify_signed(value: Mapping[str, Any], label: str) -> None:
    try:
        annotation.verify_signed(value, label)
    except BaseException as error:
        raise AnnotationGateError(str(error)) from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any, label: str) -> datetime:
    try:
        return annotation.require_rfc3339_utc(value, label)
    except BaseException as error:
        raise AnnotationGateError(str(error)) from error


def _event_timestamp(value: Any, label: str) -> datetime:
    result = _timestamp(value, label)
    require(
        result <= datetime.now(timezone.utc) + timedelta(seconds=MAX_FUTURE_CLOCK_SKEW_SECONDS),
        f"{label} exceeds the allowed future clock skew",
    )
    return result


def _operator_hash(code: str, label: str) -> str:
    require(isinstance(code, str) and SAFE_ID_RE.fullmatch(code) is not None,
            f"{label} must be a canonical lowercase identifier")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _write_exclusive_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite immutable JSON: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AnnotationGateError(f"refusing to overwrite immutable JSON: {path}") from error
    finally:
        Path(temporary).unlink(missing_ok=True)


def _copy_exclusive(source: Path, destination: Path, *, mode: int = 0o600) -> None:
    source = Path(source).resolve(strict=True)
    require(source.is_file() and not source.is_symlink(), "copy source is not a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    require(_sha256_file(source) == _sha256_file(destination), "copied artifact hash changed")


@contextmanager
def _gate_lock(private_root: Path) -> Iterator[None]:
    lock_path = private_root / ".annotation_gate.lock"
    require(private_root.is_dir(), "private annotation package is missing")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _tree_descriptor(root: Path) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), f"tree is not a directory: {root}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"tree contains a symlink: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"tree contains a non-file: {path}")
        relative = str(path.relative_to(root))
        files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    require(files, f"tree is empty: {root}")
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "artifact_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
    }


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _assert_source_free_batch(root: Path) -> None:
    """Recheck every delivered JSON string with the workflow's blind-text gate."""

    root = Path(root)
    require(root.is_dir() and not root.is_symlink(),
            "source-free packet root is not a real directory")
    root = root.resolve(strict=True)
    tree_entries = sorted(root.rglob("*"))
    for path in tree_entries:
        require(not path.is_symlink(), f"source-free packet contains a symlink: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"source-free packet contains a non-file: {path}")
    packet = _load_json(root / "packet.json", "source-free packet")
    response = _load_json(root / "response_template.json", "source-free response template")
    rubric = _load_json(root / "rubric.json", "source-free rubric")
    examples_name = rubric.get("illustrated_examples", {}).get("manifest_path")
    require(examples_name == "illustrated_examples.json",
            "source-free illustrated-example manifest path changed")
    examples = _load_json(root / examples_name, "source-free illustrated-example manifest")
    expected_files = {
        "packet.json", "response_template.json", "rubric.json", examples_name,
        examples.get("pixel_blindness_receipt_path"),
    }
    items = packet.get("items")
    example_files = examples.get("files")
    require(isinstance(items, list) and isinstance(example_files, list),
            "source-free packet or example inventory changed")
    expected_files.update(item.get("media_file") for item in items)
    expected_files.update(item.get("file") for item in example_files)
    require(all(isinstance(item, str) and item for item in expected_files),
            "source-free packet contains an invalid relative file")
    require(response.get("packet_id") == packet.get("packet_id")
            and response.get("packet_manifest_sha256") == _sha256_file(root / "packet.json"),
            "source-free response template does not bind the packet")
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    require(actual_files == expected_files,
            "source-free batch file inventory differs from the exact packet payload")
    for path in tree_entries:
        if path.is_dir():
            continue
        require(path.suffix in {".json", ".png"},
                f"source-free packet contains an unexpected file type: {path.name}")
        if path.suffix == ".json":
            value = _load_json(path, f"source-free packet JSON {path.name}")
            forbidden_keys = {
                "source_request_id", "source_video_id", "source_image_id",
                "cell_id", "model_id", "layout_pair_id", "condition_id",
                "episode_id", "request_index", "source_records", "media_path",
                "annotation_media_path",
            }
            stack = [value]
            while stack:
                current = stack.pop()
                if isinstance(current, Mapping):
                    require(not (set(current) & forbidden_keys),
                            f"delivered {path.name} contains a restricted source key")
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
            for text in _walk_strings(value):
                lowered = text.casefold()
                require(not any(token in lowered for token in (
                    "cosmos", "dreamzero", "original_left", "original_right",
                    "reflected_left", "reflected_right",
                    "put the rubik's cube to the left",
                    "put the rubik's cube to the right", "/data/",
                    "source_video", "source_request",
                )), f"delivered {path.name} exposes model, condition, command, or source identity")
                words = {word.strip(".,:;_-/()[]{}") for word in lowered.split()}
                require("n3" not in words and "d1" not in words,
                        f"delivered {path.name} exposes a literal model identifier")


def _package_receipt(private_root: Path) -> tuple[dict[str, Any], Path]:
    path = private_root / "package_receipt.json"
    value = _load_json(path, "private package receipt")
    expected_keys = {
        "schema_version", "study_id", "stage", "status", "media_job_receipt",
        "preparation_receipt", "pixel_blindness_receipt", "development_freeze",
        "image_inventory", "packet_tree", "restricted_map", "rater_streams",
        "counts", "science_counts", "labels_created_by_job",
        "restricted_map_private", "safe_to_release_confirmation", "claim_boundary",
        "completed_at_utc", "payload_sha256",
    }
    require(set(value) == expected_keys, "private package receipt fields changed")
    _verify_signed(value, "private package receipt")
    require(value.get("schema_version") == PACKAGE_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("stage") == "development"
            and value.get("status") == "private_packets_ready_for_single_batch_gate"
            and value.get("science_counts") == _zero_science_counts()
            and value.get("labels_created_by_job") == 0
            and value.get("restricted_map_private") is True
            and value.get("safe_to_release_confirmation") is False,
            "private package receipt did not preserve the zero-science boundary")
    for name in (
        "media_job_receipt", "preparation_receipt", "pixel_blindness_receipt",
        "development_freeze", "image_inventory", "restricted_map",
    ):
        _verify_descriptor(value[name], f"private package {name}")
    packet_tree = value.get("packet_tree")
    require(isinstance(packet_tree, Mapping)
            and set(packet_tree) == {"path", "file_count", "total_bytes", "artifact_sha256"},
            "private packet-tree descriptor fields changed")
    packet_root = Path(packet_tree["path"])
    require(packet_root == (private_root / "packets").resolve()
            and Path(value["restricted_map"]["path"])
            == (private_root / "restricted" / "identity_map.json").resolve(),
            "private packet or restricted-map path changed")
    require(_tree_descriptor(packet_root) == {
        key: packet_tree[key] for key in ("file_count", "total_bytes", "artifact_sha256")
    }, "private packet tree changed")
    mapping = _load_json(private_root / "restricted" / "identity_map.json",
                         "private restricted identity map")
    _verify_signed(mapping, "private restricted identity map")
    require(mapping.get("schema_version") == annotation.RESTRICTED_MAP_SCHEMA
            and mapping.get("study_id") == STUDY_ID
            and mapping.get("stage") == "development"
            and mapping.get("visibility")
            == "RESTRICTED ANALYST IDENTITY MAP; NEVER DISTRIBUTE TO RATERS OR BLIND ADJUDICATORS",
            "private restricted identity map scope changed")
    expected_streams = {
        slot: {
            "batch_count": mapping["packets"][slot]["batch_count"],
            "packet_ids": [batch["packet_id"] for batch in mapping["packets"][slot]["batches"]],
            "packet_manifest_sha256": [
                batch["packet_manifest_sha256"] for batch in mapping["packets"][slot]["batches"]
            ],
        }
        for slot in ("rater_a", "rater_b")
    }
    require(value.get("rater_streams") == expected_streams,
            "private package rater streams differ from the restricted map")
    require(Path(mapping.get("packet_root", "")).resolve() == packet_root,
            "restricted map packet root differs from the private package")
    counts = value.get("counts")
    require(isinstance(counts, Mapping) and set(counts) == {
        "selected_requests", "source_image_records", "unique_packet_images",
        "first_pass_raters_required", "human_pixel_blindness_reviews_consumed",
        "human_responses_consumed", "human_labels_consumed",
    } and counts.get("selected_requests") > 0
        and counts.get("source_image_records") > 0
        and counts.get("unique_packet_images") > 0
        and counts.get("first_pass_raters_required") == 2
        and counts.get("human_pixel_blindness_reviews_consumed") == 1
        and counts.get("human_responses_consumed") == 0
        and counts.get("human_labels_consumed") == 0,
            "private package accounting changed")
    _timestamp(value.get("completed_at_utc"), "private package completed_at_utc")
    return value, path


def _adjudication_package_receipt(private_root: Path) -> tuple[dict[str, Any], Path]:
    path = private_root / "adjudication_package_receipt.json"
    value = _load_json(path, "adjudication package receipt")
    expected_keys = {
        "schema_version", "study_id", "stage", "status",
        "source_package_receipt_sha256", "first_pass_response_sets",
        "adjudication_map", "packet_tree", "counts", "labels_created_by_job",
        "science_counts", "safe_to_release_confirmation", "claim_boundary",
        "completed_at_utc", "payload_sha256",
    }
    require(set(value) == expected_keys, "adjudication package receipt fields changed")
    _verify_signed(value, "adjudication package receipt")
    require(value.get("schema_version") == ADJUDICATION_PACKAGE_SCHEMA
            and value.get("study_id") == STUDY_ID
            and value.get("stage") == "development"
            and value.get("status") == "private_blind_adjudication_packets_ready"
            and value.get("source_package_receipt_sha256")
            == _sha256_file(private_root / "package_receipt.json")
            and value.get("labels_created_by_job") == 0
            and value.get("science_counts") == _zero_science_counts()
            and value.get("safe_to_release_confirmation") is False,
            "adjudication package crossed its scope or zero-science boundary")
    sources = value.get("first_pass_response_sets")
    require(isinstance(sources, Mapping) and set(sources) == {"rater_a", "rater_b"},
            "adjudication first-pass response-set inventory changed")
    for slot, reference in sources.items():
        require(isinstance(reference, Mapping)
                and set(reference) == {"path", "artifact_sha256"},
                f"adjudication {slot} response-set descriptor changed")
        response_path = Path(reference["path"])
        require(response_path == (private_root / "first_pass_response_sets" / slot).resolve()
                and response_path.is_dir()
                and annotation.artifact_sha256(response_path) == reference["artifact_sha256"],
                f"adjudication {slot} response-set bytes changed")
    _, map_path = _verify_descriptor(value.get("adjudication_map"), "adjudication map")
    require(map_path == (private_root / "restricted" / "adjudication_map.json").resolve(),
            "adjudication map path changed")
    mapping = _load_json(map_path, "adjudication map")
    _verify_signed(mapping, "adjudication map")
    counts = value.get("counts")
    require(isinstance(counts, Mapping) and set(counts) == {
        "first_pass_images", "adjudication_required", "adjudication_batches",
    } and all(type(counts[key]) is int and counts[key] >= 0 for key in counts)
        and mapping.get("schema_version") == annotation.ADJUDICATION_MAP_SCHEMA
        and mapping.get("study_id") == STUDY_ID
        and mapping.get("stage") == "development"
        and mapping.get("asset_count") == counts.get("adjudication_required")
        and mapping.get("packets", {}).get("adjudicator", {}).get("batch_count")
        == counts.get("adjudication_batches"),
            "adjudication package counts or map scope changed")
    packet_tree = value.get("packet_tree")
    if counts["adjudication_required"]:
        require(isinstance(packet_tree, Mapping)
                and set(packet_tree) == {"path", "file_count", "total_bytes", "artifact_sha256"}
                and Path(packet_tree["path"]) == (private_root / "adjudication_packets").resolve()
                and _tree_descriptor(Path(packet_tree["path"])) == {
                    key: packet_tree[key]
                    for key in ("file_count", "total_bytes", "artifact_sha256")
                }, "adjudication packet tree changed")
    else:
        require(packet_tree is None and counts["adjudication_batches"] == 0,
                "no-disagreement adjudication package unexpectedly has packets")
    _timestamp(value.get("completed_at_utc"), "adjudication package completed_at_utc")
    return value, path


def package_private_packets(
    *,
    study_commit: str,
    media_job_receipt: Mapping[str, Any],
    pixel_blindness_receipt: Mapping[str, Any],
    development_freeze: Mapping[str, Any],
    private_root: Path,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate future human artifacts and build private first-pass packets."""

    validate_contract()
    require(re.fullmatch(r"[0-9a-f]{40}", study_commit) is not None,
            "study_commit must be an exact lowercase Git commit")
    media_descriptor, media_receipt_path = _verify_descriptor(
        media_job_receipt, "terminal annotation-media job receipt"
    )
    review_descriptor, review_path = _verify_descriptor(
        pixel_blindness_receipt, "human pixel-blindness receipt"
    )
    freeze_descriptor, freeze_path = _verify_descriptor(
        development_freeze, "development annotation freeze"
    )
    raw_media_receipt = _load_json(media_receipt_path, "terminal annotation-media job receipt")
    index_descriptor = raw_media_receipt.get("outputs", {}).get("publish_tranche_index")
    try:
        media_jobs._validate_terminal_preparation_job(
            raw_media_receipt,
            receipt_descriptor=media_descriptor,
            index_descriptor=index_descriptor,
            study_commit=study_commit,
        )
    except BaseException as error:
        raise AnnotationGateError(f"terminal annotation-media receipt failed: {error}") from error
    preparation_root = Path(raw_media_receipt["outputs"]["output_root"]).resolve(strict=True)
    preparation_receipt_path = preparation_root / "preparation_receipt.json"
    preparation_descriptor = _descriptor(preparation_receipt_path)
    require(preparation_descriptor == raw_media_receipt["outputs"]["preparation_receipt"],
            "preparation receipt differs from terminal job output")
    freeze = _load_json(freeze_path, "development annotation freeze")
    try:
        freeze_info = annotation.validate_freeze(
            freeze, freeze_path=freeze_path, stage="development"
        )
    except BaseException as error:
        raise AnnotationGateError(f"development annotation freeze failed: {error}") from error
    require(freeze_info["cohort_branch"] == "full_two_model"
            and freeze_info["qualified_model_ids"] == ["N3", "D1"],
            "formal development packet gate requires the exact N3/D1 cohort")
    completed_at_utc = completed_at_utc or _utc_now()
    completed_at = _event_timestamp(completed_at_utc, "package completed_at_utc")
    review = _load_json(review_path, "human pixel-blindness receipt")
    reviewed_at = _timestamp(review.get("reviewed_at"), "pixel-blindness reviewed_at")
    media_completed = _timestamp(
        raw_media_receipt.get("completed_at_utc"), "annotation-media completed_at_utc"
    )
    require(media_completed <= reviewed_at <= completed_at,
            "pixel-blindness review time precedes media completion or exceeds packaging time")
    example_reviewed_at = _timestamp(
        _load_json(
            freeze_info["examples_path"], "illustrated-example manifest"
        ).get("reviewed_at"),
        "illustrated-example reviewed_at",
    )
    example_blindness_reviewed_at = _timestamp(
        _load_json(
            freeze_info["examples_blindness_path"],
            "illustrated-example pixel-blindness receipt",
        ).get("reviewed_at"),
        "illustrated-example pixel-blindness reviewed_at",
    )
    require(example_reviewed_at <= completed_at
            and example_blindness_reviewed_at <= completed_at,
            "illustrated-example human review time exceeds packaging time")

    private_root = Path(private_root).absolute()
    require(not private_root.exists(), "private annotation package already exists")
    private_root.mkdir(parents=True, mode=0o700)
    os.chmod(private_root, 0o700)
    try:
        inventory_path = private_root / "image_inventory.json"
        try:
            media.finalize_reviewed_inventory(
                preparation_dir=preparation_root,
                review_path=review_path,
                review_sha256=review_descriptor["sha256"],
                output_path=inventory_path,
            )
        except BaseException as error:
            raise AnnotationGateError(f"reviewed inventory finalization failed: {error}") from error
        packet_root = private_root / "packets"
        restricted_path = private_root / "restricted" / "identity_map.json"
        try:
            result = annotation.package_packets(
                selection_path=preparation_root / "request_selection.json",
                image_inventory_path=inventory_path,
                freeze_path=freeze_path,
                packet_root=packet_root,
                restricted_map_path=restricted_path,
            )
        except BaseException as error:
            raise AnnotationGateError(f"private packet packaging failed: {error}") from error
        os.chmod(private_root / "restricted", 0o700)
        os.chmod(restricted_path, 0o600)
        mapping = _load_json(restricted_path, "restricted identity map")
        annotation.verify_signed(mapping, "restricted identity map")
        require(mapping.get("stage") == "development"
                and mapping.get("cohort_branch") == "full_two_model"
                and mapping.get("qualified_model_ids") == ["N3", "D1"],
                "restricted identity map differs from the formal development cohort")
        streams = {
            slot: {
                "batch_count": mapping["packets"][slot]["batch_count"],
                "packet_ids": [batch["packet_id"] for batch in mapping["packets"][slot]["batches"]],
                "packet_manifest_sha256": [
                    batch["packet_manifest_sha256"] for batch in mapping["packets"][slot]["batches"]
                ],
            }
            for slot in ("rater_a", "rater_b")
        }
        receipt = _sign({
            "schema_version": PACKAGE_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "private_packets_ready_for_single_batch_gate",
            "media_job_receipt": media_descriptor,
            "preparation_receipt": preparation_descriptor,
            "pixel_blindness_receipt": review_descriptor,
            "development_freeze": freeze_descriptor,
            "image_inventory": _descriptor(inventory_path),
            "packet_tree": {"path": str(packet_root.resolve()), **_tree_descriptor(packet_root)},
            "restricted_map": _descriptor(restricted_path),
            "rater_streams": streams,
            "counts": {
                "selected_requests": result["selected_requests"],
                "source_image_records": result["source_image_records"],
                "unique_packet_images": result["unique_packet_images"],
                "first_pass_raters_required": 2,
                "human_pixel_blindness_reviews_consumed": 1,
                "human_responses_consumed": 0,
                "human_labels_consumed": 0,
            },
            "science_counts": _zero_science_counts(),
            "labels_created_by_job": 0,
            "restricted_map_private": True,
            "safe_to_release_confirmation": False,
            "claim_boundary": (
                "Human-reviewed source-free media were mechanically randomized into private packets. "
                "No response, label, measurement threshold, consensus, or confirmation release was created."
            ),
            "completed_at_utc": completed_at_utc,
        })
        _write_exclusive_json(private_root / "package_receipt.json", receipt)
        _package_receipt(private_root)
        return receipt
    except BaseException:
        # Preserve every partial artifact for diagnosis; absence of the terminal
        # package receipt makes the attempt ineligible for release.
        raise


def _mapping_for_slot(private_root: Path, slot: str) -> tuple[dict[str, Any], Path]:
    require(slot in SLOTS, "annotation slot is invalid")
    _package_receipt(private_root)
    if slot in {"rater_a", "rater_b"}:
        path = private_root / "restricted" / "identity_map.json"
        expected_schema = annotation.RESTRICTED_MAP_SCHEMA
    else:
        adjudication_receipt = private_root / "adjudication_package_receipt.json"
        require(adjudication_receipt.is_file(), "adjudication packets are not prepared")
        receipt, _ = _adjudication_package_receipt(private_root)
        _, path = _verify_descriptor(receipt.get("adjudication_map"), "adjudication map")
        expected_schema = annotation.ADJUDICATION_MAP_SCHEMA
    mapping = _load_json(path, f"{slot} restricted mapping")
    _verify_signed(mapping, f"{slot} restricted mapping")
    require(mapping.get("schema_version") == expected_schema
            and mapping.get("stage") == "development",
            f"{slot} mapping schema or stage changed")
    return mapping, path


def _batch_context(private_root: Path, slot: str, ordinal: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    require(type(ordinal) is int and ordinal >= 1, "batch ordinal must be positive")
    mapping, _ = _mapping_for_slot(private_root, slot)
    stream = mapping.get("packets", {}).get(slot)
    require(isinstance(stream, Mapping)
            and stream.get("delivery_rule") == "ONE_BATCH_AT_A_TIME_WITH_PRIOR_BATCH_COLLECTED_AND_REVOKED",
            f"{slot} packet delivery rule changed")
    batches = stream.get("batches")
    require(isinstance(batches, list) and stream.get("batch_count") == len(batches),
            f"{slot} packet inventory changed")
    require(ordinal <= len(batches), f"{slot} batch ordinal is outside the frozen stream")
    batch = batches[ordinal - 1]
    try:
        packet, _ = annotation._validate_packet_artifacts(mapping, slot=slot, batch=batch)
    except BaseException as error:
        raise AnnotationGateError(f"{slot} batch failed source packet validation: {error}") from error
    packet_root = Path(mapping["packet_root"]).resolve(strict=True)
    packet_path = (packet_root / batch["packet_relative_path"]).resolve(strict=True)
    return mapping, batch, packet_path.parent


def _receipt_directories(private_root: Path) -> list[Path]:
    archive = private_root / "revoked_deliveries"
    require(not archive.is_symlink(), "revoked-delivery archive changed")
    if not archive.exists():
        return []
    require(archive.is_dir(), "revoked-delivery archive changed")
    entries = sorted(archive.iterdir())
    require(all(path.is_dir() and not path.is_symlink() for path in entries),
            "revoked-delivery archive contains a non-directory or symlink entry")
    return entries


def _require_exact_entries(
    root: Path, *, files: set[str], directories: set[str], label: str,
) -> None:
    require(root.is_dir() and not root.is_symlink(), f"{label} is not a directory")
    require(not (files & directories), f"{label} type declaration overlaps")
    entries = {path.name: path for path in root.iterdir()}
    require(set(entries) == files | directories, f"{label} inventory changed")
    for name in sorted(files):
        path = entries[name]
        require(path.is_file() and not path.is_symlink(),
                f"{label} entry is not a regular file: {name}")
    for name in sorted(directories):
        path = entries[name]
        require(path.is_dir() and not path.is_symlink(),
                f"{label} entry is not a directory: {name}")


def _validate_lifecycle_record(
    *, private_root: Path, record_id: str, published_root: Path,
    delivery_path: Path, private_delivery_path: Path, response_path: Path,
    collection_path: Path, delivery: Mapping[str, Any],
    collection: Mapping[str, Any], revocation: Mapping[str, Any],
) -> None:
    """Replay one complete lifecycle record without changing any artifact."""

    for value, label in (
        (delivery, "archived delivery receipt"),
        (collection, "archived collection receipt"),
        (revocation, "archived revocation receipt"),
    ):
        _verify_signed(value, label)
    require(set(delivery) == DELIVERY_KEYS, "archived delivery receipt fields changed")
    require(set(collection) == COLLECTION_KEYS, "archived collection receipt fields changed")
    require(set(revocation) == REVOCATION_KEYS, "archived revocation receipt fields changed")
    require(delivery.get("schema_version") == DELIVERY_SCHEMA
            and collection.get("schema_version") == COLLECTION_SCHEMA
            and revocation.get("schema_version") == REVOCATION_SCHEMA,
            "archived lifecycle schema changed")
    require(delivery.get("status") == "single_source_free_batch_active"
            and delivery.get("release_rule") == DELIVERY_RULE
            and delivery.get("source_identity_present") is False
            and delivery.get("restricted_map_present") is False
            and collection.get("status") == "locked_human_response_collected"
            and revocation.get("status") == "controlled_filesystem_handoff_revoked"
            and revocation.get("controlled_handoff_path_absent") is True
            and revocation.get("external_distribution_or_retained_copy_revocation_claimed") is False,
            "archived lifecycle status or claim boundary changed")
    require(all(value.get("study_id") == STUDY_ID
                and value.get("stage") == "development"
                and value.get("labels_created_by_job") == 0
                and value.get("science_counts") == _zero_science_counts()
                and value.get("safe_to_release_confirmation") is False
                for value in (delivery, collection, revocation)),
            "archived lifecycle crossed the zero-science boundary")
    require(delivery["delivery_id"] == collection["delivery_id"]
            == revocation["delivery_id"] == record_id,
            "archived lifecycle identifiers differ")
    identity_fields = ("release_sequence", "rater_slot", "batch_ordinal", "packet_id")
    require(all(collection.get(key) == delivery.get(key)
                and revocation.get(key) == delivery.get(key) for key in identity_fields),
            "archived lifecycle identity fields differ")
    require(_sha256_file(delivery_path)
            == collection.get("delivery_receipt_sha256")
            == revocation.get("delivery_receipt_sha256"),
            "archived delivery receipt hash binding changed")
    require(_sha256_file(private_delivery_path) == revocation.get("delivery_receipt_sha256"),
            "archived private delivery witness changed")
    require(_sha256_file(collection_path) == revocation.get("collection_receipt_sha256"),
            "archived collection receipt hash binding changed")
    response_descriptor = collection.get("response")
    require(isinstance(response_descriptor, Mapping)
            and set(response_descriptor) == {"file", "sha256", "bytes"}
            and response_descriptor.get("file") == "response.json"
            and response_descriptor.get("sha256") == _sha256_file(response_path)
            and response_descriptor.get("bytes") == response_path.stat().st_size
            and revocation.get("response_sha256") == response_descriptor.get("sha256"),
            "archived human response hash binding changed")
    mapping, batch, _ = _batch_context(
        private_root, delivery["rater_slot"], delivery["batch_ordinal"]
    )
    require(delivery.get("batch_count") == mapping["packets"][delivery["rater_slot"]]["batch_count"]
            and delivery.get("packet_id") == batch["packet_id"]
            and delivery.get("packet_manifest_sha256") == batch["packet_manifest_sha256"]
            and delivery.get("package_receipt_sha256")
            == _sha256_file(private_root / "package_receipt.json"),
            "archived delivery differs from its frozen packet stream")
    published_batch = published_root / "batch"
    _assert_source_free_batch(published_batch)
    require(annotation.artifact_sha256(published_batch) == delivery.get("packet_artifact_sha256"),
            "archived published packet bytes changed")
    response_value = _load_json(response_path, "archived human response")
    try:
        _, dimensions = annotation._validate_packet_artifacts(
            mapping, slot=delivery["rater_slot"], batch=batch
        )
        annotations = annotation._validate_response(
            response_value, mapping=mapping, slot=delivery["rater_slot"],
            packet=batch, dimensions=dimensions,
        )
    except BaseException as error:
        raise AnnotationGateError(f"archived human response failed replay: {error}") from error
    response_timing = collection.get("response_timing")
    started_at = _timestamp(response_value.get("started_at"), "archived response started_at")
    completed_at = _timestamp(response_value.get("completed_at"), "archived response completed_at")
    locked_at = _timestamp(response_value.get("locked_at"), "archived response locked_at")
    released_at = _timestamp(delivery.get("released_at_utc"), "archived delivery released_at_utc")
    collected_at = _timestamp(collection.get("collected_at_utc"), "archived collection collected_at_utc")
    revoked_at = _timestamp(revocation.get("revoked_at_utc"), "archived revocation revoked_at_utc")
    require(released_at <= started_at <= completed_at <= locked_at <= collected_at <= revoked_at,
            "archived lifecycle timestamps are out of order")
    require(response_timing == {
        "started_at": response_value["started_at"],
        "completed_at": response_value["completed_at"],
        "locked_at": response_value["locked_at"],
        "wall_seconds": (completed_at - started_at).total_seconds(),
        "annotation_seconds": math.fsum(
            item["annotation_seconds"] for item in annotations.values()
        ),
    } and collection.get("annotation_count_received") == len(annotations),
            "archived collection timing or annotation count does not replay")
    require(collection.get("rater_code_sha256")
            == hashlib.sha256(response_value["rater_code"].encode("utf-8")).hexdigest(),
            "archived rater identity hash does not replay")


def _load_archived_delivery(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_exact_entries(
        path,
        files={
            "delivery_receipt_private.json", "response.json",
            "collection_receipt.json", "revocation_receipt.json",
        },
        directories={"published_batch"},
        label="archived delivery",
    )
    _require_exact_entries(
        path / "published_batch",
        files={"delivery_receipt.json"},
        directories={"batch"},
        label="archived published batch",
    )
    delivery = _load_json(path / "published_batch" / "delivery_receipt.json", "archived delivery receipt")
    collection = _load_json(path / "collection_receipt.json", "archived collection receipt")
    revocation = _load_json(path / "revocation_receipt.json", "archived revocation receipt")
    private_root = path.parents[1]
    _validate_lifecycle_record(
        private_root=private_root,
        record_id=path.name,
        published_root=path / "published_batch",
        delivery_path=path / "published_batch" / "delivery_receipt.json",
        private_delivery_path=path / "delivery_receipt_private.json",
        response_path=path / "response.json",
        collection_path=path / "collection_receipt.json",
        delivery=delivery,
        collection=collection,
        revocation=revocation,
    )
    return delivery, collection, revocation


def _validate_lifecycle_history(
    records: list[tuple[str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    require(all(SHA_RE.fullmatch(receipt_sha256) is not None
                for receipt_sha256, _, _, _ in records),
            "archived revocation receipt hash is invalid")
    sequences = [delivery.get("release_sequence") for _, delivery, _, _ in records]
    require(sequences == list(range(1, len(sequences) + 1)),
            "archived deliveries are missing, duplicated, or out of global order")
    for index, (_, delivery, _, _) in enumerate(records):
        expected_prior = None if index == 0 else records[index - 1][0]
        require(delivery.get("prior_revocation_receipt_sha256") == expected_prior,
                "archived delivery revocation chain changed")
    for prior, current in zip(records, records[1:]):
        require(_timestamp(prior[3].get("revoked_at_utc"), "prior revoked_at_utc")
                <= _timestamp(current[1].get("released_at_utc"), "current released_at_utc"),
                "archived release precedes its prior revocation")
    for slot in SLOTS:
        rows = [(delivery, collection) for _, delivery, collection, _ in records
                if delivery["rater_slot"] == slot]
        require([delivery["batch_ordinal"] for delivery, _ in rows]
                == list(range(1, len(rows) + 1)),
                f"{slot} archived batch order changed")
        require(len({collection["rater_code_sha256"] for _, collection in rows}) <= 1,
                f"{slot} archived responses identify multiple raters")
        for prior, current in zip(rows, rows[1:]):
            require(_timestamp(prior[1]["response_timing"]["locked_at"], "prior locked_at")
                    <= _timestamp(current[1]["response_timing"]["started_at"], "current started_at"),
                    f"{slot} archived response sessions overlap or are out of order")
    identities = {
        slot: {collection["rater_code_sha256"] for _, delivery, collection, _ in records
               if delivery["rater_slot"] == slot}
        for slot in SLOTS
    }
    require(not (identities["rater_a"] & identities["rater_b"]),
            "archived first-pass slots identify the same rater")
    require(not (identities["adjudicator"]
                 & (identities["rater_a"] | identities["rater_b"])),
            "archived adjudicator identifies a first-pass rater")


def _lifecycle(private_root: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    result = []
    for path in _receipt_directories(private_root):
        delivery, collection, revocation = _load_archived_delivery(path)
        result.append((path, delivery, collection, revocation))
    _validate_lifecycle_history([
        (_sha256_file(path / "revocation_receipt.json"), delivery, collection, revocation)
        for path, delivery, collection, revocation in result
    ])
    return result


def _validate_public_root(public_root: Path) -> None:
    require(public_root.is_dir() and not public_root.is_symlink(), "handoff root is missing or a symlink")
    names = sorted(path.name for path in public_root.iterdir())
    require(names in ([], ["active"]), "handoff root contains content outside the sole active batch")


def _inflight_delivery_witnesses(private_root: Path) -> list[Path]:
    root = private_root / "delivery_intents"
    if not root.exists():
        return []
    require(root.is_dir() and not root.is_symlink(), "delivery-intent root changed")
    paths = sorted(root.iterdir())
    require(all(path.is_file() and not path.is_symlink() for path in paths),
            "delivery-intent root contains unexpected content")
    return paths


def _pending_collection_directories(private_root: Path) -> list[Path]:
    root = private_root / "pending_collections"
    if not root.exists():
        require(not root.is_symlink(), "pending-collection root changed")
        return []
    require(root.is_dir() and not root.is_symlink(), "pending-collection root changed")
    paths = sorted(root.iterdir())
    require(all(path.is_dir() and not path.is_symlink() for path in paths),
            "pending-collection root contains unexpected content")
    return paths


def release_batch(
    *, private_root: Path, public_root: Path, slot: str, ordinal: int,
    released_by: str, released_at_utc: str | None = None,
) -> dict[str, Any]:
    """Atomically expose exactly one already-blinded packet batch."""

    validate_contract()
    private_root = Path(private_root).resolve(strict=True)
    public_root = Path(public_root).absolute()
    public_root.mkdir(parents=True, exist_ok=True)
    public_root = public_root.resolve(strict=True)
    require(private_root != public_root
            and not private_root.is_relative_to(public_root)
            and not public_root.is_relative_to(private_root),
            "private and handoff roots must be disjoint")
    require(private_root.stat().st_dev == public_root.stat().st_dev,
            "private and handoff roots must share a filesystem for atomic revocation")
    released_by_hash = _operator_hash(released_by, "released_by")
    released_at_utc = released_at_utc or _utc_now()
    released_at = _event_timestamp(released_at_utc, "released_at_utc")
    with _gate_lock(private_root):
        _validate_public_root(public_root)
        require(not (public_root / "active").exists(), "one annotation batch is already active")
        require(not _inflight_delivery_witnesses(private_root),
                "an unreconciled delivery witness blocks another release")
        require(not _pending_collection_directories(private_root),
                "an unreconciled collected response blocks another release")
        lifecycle = _lifecycle(private_root)
        prior_for_slot = [delivery for _, delivery, _, _ in lifecycle if delivery["rater_slot"] == slot]
        require(ordinal == len(prior_for_slot) + 1,
                f"{slot} batch release is not the next frozen ordinal")
        mapping, batch, source_batch = _batch_context(private_root, slot, ordinal)
        _assert_source_free_batch(source_batch)
        package, package_path = _package_receipt(private_root)
        batch_count = mapping["packets"][slot]["batch_count"]
        prior = lifecycle[-1][1] if lifecycle else None
        delivery_id = f"delivery_{len(lifecycle) + 1:04d}_{slot}_{ordinal:04d}_{batch['packet_id'].removeprefix('packet_')}"
        receipt = _sign({
            "schema_version": DELIVERY_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "single_source_free_batch_active",
            "delivery_id": delivery_id,
            "release_sequence": len(lifecycle) + 1,
            "rater_slot": slot,
            "batch_ordinal": ordinal,
            "batch_count": batch_count,
            "packet_id": batch["packet_id"],
            "packet_manifest_sha256": batch["packet_manifest_sha256"],
            "packet_artifact_sha256": annotation.artifact_sha256(source_batch),
            "package_receipt_sha256": _sha256_file(package_path),
            "prior_revocation_receipt_sha256": (
                _sha256_file(lifecycle[-1][0] / "revocation_receipt.json") if prior else None
            ),
            "release_rule": DELIVERY_RULE,
            "released_at_utc": released_at_utc,
            "released_by_code_sha256": released_by_hash,
            "source_identity_present": False,
            "restricted_map_present": False,
            "science_counts": _zero_science_counts(),
            "labels_created_by_job": 0,
            "safe_to_release_confirmation": False,
            "claim_boundary": (
                "One source-free annotation batch is present in the controlled filesystem handoff. "
                "No external delivery, retained-copy deletion, label, or confirmation release is claimed."
            ),
        })
        require(_timestamp(package["completed_at_utc"], "package completed_at_utc") <= released_at,
                "release time precedes private packet completion")
        if prior is not None:
            prior_revoked = _timestamp(lifecycle[-1][3]["revoked_at_utc"], "prior revoked_at_utc")
            require(prior_revoked <= released_at, "release time precedes prior revocation")
        staging_parent = private_root / "delivery_staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        require(not any(staging_parent.iterdir()),
                "an incomplete private delivery staging tree requires reconciliation")
        staging = Path(tempfile.mkdtemp(prefix=".annotation-batch-", dir=staging_parent))
        intent_parent = private_root / "delivery_intents"
        intent_parent.mkdir(parents=True, exist_ok=True)
        intent_path = intent_parent / f"{delivery_id}.json"
        try:
            shutil.copytree(source_batch, staging / "batch", dirs_exist_ok=False)
            _assert_source_free_batch(staging / "batch")
            _write_exclusive_json(staging / "delivery_receipt.json", receipt, mode=0o444)
            require(annotation.artifact_sha256(staging / "batch") == receipt["packet_artifact_sha256"],
                    "staged source-free batch differs from the private packet")
            require(not any(path.is_symlink() for path in staging.rglob("*")),
                    "source-free batch publication contains a symlink")
            _copy_exclusive(staging / "delivery_receipt.json", intent_path, mode=0o400)
            os.replace(staging, public_root / "active")
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            if not (public_root / "active").exists():
                intent_path.unlink(missing_ok=True)
            raise
        _validate_public_root(public_root)
        return receipt


def _active_delivery(
    public_root: Path, *, private_root: Path | None = None
) -> tuple[Path, dict[str, Any], Path]:
    _validate_public_root(public_root)
    active = public_root / "active"
    require(active.is_dir() and not active.is_symlink(), "no annotation batch is active")
    _require_exact_entries(
        active,
        files={"delivery_receipt.json"},
        directories={"batch"},
        label="active annotation handoff",
    )
    receipt_path = active / "delivery_receipt.json"
    receipt = _load_json(receipt_path, "active delivery receipt")
    _verify_signed(receipt, "active delivery receipt")
    require(set(receipt) == DELIVERY_KEYS, "active delivery receipt fields changed")
    require(receipt.get("schema_version") == DELIVERY_SCHEMA
            and receipt.get("status") == "single_source_free_batch_active"
            and receipt.get("release_rule") == DELIVERY_RULE
            and receipt.get("rater_slot") in SLOTS
            and receipt.get("source_identity_present") is False
            and receipt.get("restricted_map_present") is False
            and receipt.get("science_counts") == _zero_science_counts()
            and receipt.get("labels_created_by_job") == 0
            and receipt.get("safe_to_release_confirmation") is False,
            "active delivery receipt changed")
    batch_dir = active / "batch"
    _assert_source_free_batch(batch_dir)
    require(annotation.artifact_sha256(batch_dir) == receipt.get("packet_artifact_sha256"),
            "active packet bytes changed")
    if private_root is not None:
        witnesses = _inflight_delivery_witnesses(private_root)
        require(len(witnesses) == 1
                and witnesses[0].name == f"{receipt['delivery_id']}.json"
                and _sha256_file(witnesses[0]) == _sha256_file(receipt_path),
                "active delivery lacks its exact private immutable witness")
    return active, receipt, receipt_path


def collect_response(
    *, private_root: Path, public_root: Path, response: Mapping[str, Any],
    collector_code: str, collected_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate and privately collect one externally authored locked response."""

    validate_contract()
    private_root = Path(private_root).resolve(strict=True)
    public_root = Path(public_root).resolve(strict=True)
    response_descriptor, response_path = _verify_descriptor(response, "human response")
    collector_hash = _operator_hash(collector_code, "collector_code")
    collected_at_utc = collected_at_utc or _utc_now()
    collected_at = _event_timestamp(collected_at_utc, "collected_at_utc")
    with _gate_lock(private_root):
        active, delivery, delivery_path = _active_delivery(
            public_root, private_root=private_root
        )
        require(not _pending_collection_directories(private_root),
                "an unreconciled pending collection blocks response collection")
        slot = delivery["rater_slot"]
        ordinal = delivery["batch_ordinal"]
        mapping, batch, _ = _batch_context(private_root, slot, ordinal)
        require(batch["packet_id"] == delivery["packet_id"]
                and batch["packet_manifest_sha256"] == delivery["packet_manifest_sha256"],
                "active delivery differs from the frozen batch")
        response_value = _load_json(response_path, "human response")
        try:
            _, dimensions = annotation._validate_packet_artifacts(
                mapping, slot=slot, batch=batch
            )
            annotations = annotation._validate_response(
                response_value, mapping=mapping, slot=slot,
                packet=batch, dimensions=dimensions,
            )
        except BaseException as error:
            raise AnnotationGateError(f"human response failed validation: {error}") from error
        started_at = _timestamp(response_value["started_at"], "response started_at")
        completed_at = _timestamp(response_value["completed_at"], "response completed_at")
        locked_at = _timestamp(response_value["locked_at"], "response locked_at")
        delivered_at = _timestamp(delivery["released_at_utc"], "delivery released_at_utc")
        require(delivered_at <= started_at <= completed_at <= locked_at <= collected_at,
                "response timing precedes delivery or exceeds collector timestamp")
        lifecycle = _lifecycle(private_root)
        prior_same_slot = [collection for _, prior, collection, _ in lifecycle
                           if prior["rater_slot"] == slot]
        if prior_same_slot:
            require(_timestamp(prior_same_slot[-1]["response_timing"]["locked_at"],
                               "prior response locked_at") <= started_at,
                    f"{slot} response overlaps or precedes its prior batch")
            require(prior_same_slot[-1]["rater_code_sha256"]
                    == hashlib.sha256(response_value["rater_code"].encode("utf-8")).hexdigest(),
                    f"{slot} responses identify different raters")
        rater_hash = hashlib.sha256(response_value["rater_code"].encode("utf-8")).hexdigest()
        if slot in {"rater_a", "rater_b"}:
            other = "rater_b" if slot == "rater_a" else "rater_a"
            other_hashes = {
                collection["rater_code_sha256"]
                for _, prior, collection, _ in lifecycle
                if prior["rater_slot"] == other
            }
            require(rater_hash not in other_hashes,
                    "first-pass response identifies the same human as the other rater slot")
        if slot == "adjudicator":
            first_pass_hashes = {
                collection["rater_code_sha256"]
                for _, prior, collection, _ in lifecycle
                if prior["rater_slot"] in {"rater_a", "rater_b"}
            }
            require(rater_hash not in first_pass_hashes,
                    "adjudicator response identifies a first-pass rater")
        destination = private_root / "pending_collections" / delivery["delivery_id"]
        require(not destination.exists(), "active delivery already has a collected response")
        pending_parent = private_root / "pending_collections"
        pending_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{delivery['delivery_id']}.",
                                        dir=pending_parent))
        try:
            _copy_exclusive(response_path, staging / "response.json", mode=0o400)
            _copy_exclusive(delivery_path, staging / "delivery_receipt.json", mode=0o400)
            private_response = _descriptor(staging / "response.json")
            require(private_response["sha256"] == response_descriptor["sha256"]
                    and private_response["bytes"] == response_descriptor["bytes"],
                    "private response copy differs from collected input")
            receipt = _sign({
                "schema_version": COLLECTION_SCHEMA,
                "study_id": STUDY_ID,
                "stage": "development",
                "status": "locked_human_response_collected",
                "delivery_id": delivery["delivery_id"],
                "release_sequence": delivery["release_sequence"],
                "rater_slot": slot,
                "batch_ordinal": ordinal,
                "packet_id": delivery["packet_id"],
                "delivery_receipt_sha256": _sha256_file(delivery_path),
                "response": {
                    "file": "response.json",
                    "sha256": private_response["sha256"],
                    "bytes": private_response["bytes"],
                },
                "collector_code_sha256": collector_hash,
                "rater_code_sha256": rater_hash,
                "response_timing": {
                    "started_at": response_value["started_at"],
                    "completed_at": response_value["completed_at"],
                    "locked_at": response_value["locked_at"],
                    "wall_seconds": (completed_at - started_at).total_seconds(),
                    "annotation_seconds": math.fsum(
                        item["annotation_seconds"] for item in annotations.values()
                    ),
                },
                "annotation_count_received": len(annotations),
                "labels_created_by_job": 0,
                "science_counts": _zero_science_counts(),
                "collected_at_utc": collected_at_utc,
                "safe_to_release_confirmation": False,
                "claim_boundary": (
                    "An externally authored locked response was validated and copied byte-for-byte. "
                    "The collector created no label and confirmation remains held."
                ),
            })
            _write_exclusive_json(staging / "collection_receipt.json", receipt, mode=0o400)
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return receipt


def revoke_delivery(
    *, private_root: Path, public_root: Path, revoked_by: str,
    revoked_at_utc: str | None = None,
) -> dict[str, Any]:
    """Remove the active filesystem handoff and preserve its bytes privately."""

    validate_contract()
    private_root = Path(private_root).resolve(strict=True)
    public_root = Path(public_root).resolve(strict=True)
    revoked_by_hash = _operator_hash(revoked_by, "revoked_by")
    revoked_at_utc = revoked_at_utc or _utc_now()
    revoked_at = _event_timestamp(revoked_at_utc, "revoked_at_utc")
    with _gate_lock(private_root):
        active, delivery, delivery_path = _active_delivery(
            public_root, private_root=private_root
        )
        pending = private_root / "pending_collections" / delivery["delivery_id"]
        pending_entries = _pending_collection_directories(private_root)
        require(pending in pending_entries, "active batch has no collected response")
        require(pending_entries == [pending],
                "pending-collection root contains content beyond the active delivery")
        _require_exact_entries(
            pending,
            files={"response.json", "delivery_receipt.json", "collection_receipt.json"},
            directories=set(),
            label="pending collection",
        )
        collection_path = pending / "collection_receipt.json"
        collection = _load_json(collection_path, "pending collection receipt")
        _verify_signed(collection, "pending collection receipt")
        require(set(collection) == COLLECTION_KEYS, "pending collection receipt fields changed")
        require(collection.get("schema_version") == COLLECTION_SCHEMA
                and collection.get("study_id") == STUDY_ID
                and collection.get("stage") == "development"
                and collection.get("status") == "locked_human_response_collected"
                and collection.get("delivery_id") == delivery["delivery_id"]
                and all(collection.get(key) == delivery.get(key) for key in (
                    "release_sequence", "rater_slot", "batch_ordinal", "packet_id",
                ))
                and collection.get("labels_created_by_job") == 0
                and collection.get("science_counts") == _zero_science_counts()
                and collection.get("safe_to_release_confirmation") is False,
                "pending collection does not bind the active delivery")
        active_delivery_sha256 = _sha256_file(delivery_path)
        pending_delivery_path = pending / "delivery_receipt.json"
        require(collection.get("delivery_receipt_sha256") == active_delivery_sha256
                and _sha256_file(pending_delivery_path) == active_delivery_sha256,
                "pending private delivery copy does not bind the active delivery")
        response_path = pending / "response.json"
        response_descriptor = collection.get("response")
        require(isinstance(response_descriptor, Mapping)
                and set(response_descriptor) == {"file", "sha256", "bytes"}
                and response_descriptor.get("file") == "response.json"
                and response_descriptor.get("sha256") == _sha256_file(response_path)
                and response_descriptor.get("bytes") == response_path.stat().st_size,
                "pending response bytes do not bind the collection receipt")
        collected_at = _timestamp(collection["collected_at_utc"], "collected_at_utc")
        require(collected_at <= revoked_at, "revocation time precedes collection")
        lifecycle = _lifecycle(private_root)
        witness = private_root / "delivery_intents" / f"{delivery['delivery_id']}.json"
        receipt = _sign({
            "schema_version": REVOCATION_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "controlled_filesystem_handoff_revoked",
            "delivery_id": delivery["delivery_id"],
            "release_sequence": delivery["release_sequence"],
            "rater_slot": delivery["rater_slot"],
            "batch_ordinal": delivery["batch_ordinal"],
            "packet_id": delivery["packet_id"],
            "delivery_receipt_sha256": active_delivery_sha256,
            "collection_receipt_sha256": _sha256_file(collection_path),
            "response_sha256": _sha256_file(response_path),
            "revoked_by_code_sha256": revoked_by_hash,
            "revoked_at_utc": revoked_at_utc,
            "controlled_handoff_path_absent": True,
            "external_distribution_or_retained_copy_revocation_claimed": False,
            "labels_created_by_job": 0,
            "science_counts": _zero_science_counts(),
            "safe_to_release_confirmation": False,
            "claim_boundary": (
                "The controlled filesystem handoff was removed and archived privately. "
                "No claim is made about external delivery systems or copies retained by a human."
            ),
        })
        _validate_lifecycle_record(
            private_root=private_root,
            record_id=delivery["delivery_id"],
            published_root=active,
            delivery_path=delivery_path,
            private_delivery_path=witness,
            response_path=response_path,
            collection_path=collection_path,
            delivery=delivery,
            collection=collection,
            revocation=receipt,
        )
        candidate_receipt_sha256 = hashlib.sha256(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        ).hexdigest()
        prospective = [
            (path.name, _sha256_file(path / "revocation_receipt.json"), prior,
             prior_collection, prior_revocation)
            for path, prior, prior_collection, prior_revocation in lifecycle
        ]
        prospective.append((
            delivery["delivery_id"], candidate_receipt_sha256, delivery, collection, receipt,
        ))
        prospective.sort(key=lambda row: row[0])
        require(len({row[0] for row in prospective}) == len(prospective),
                "prospective archive delivery identifier is duplicated")
        _validate_lifecycle_history([
            (receipt_sha256, candidate_delivery, candidate_collection,
             candidate_revocation)
            for _, receipt_sha256, candidate_delivery, candidate_collection,
            candidate_revocation in prospective
        ])
        archive_parent = private_root / "revoked_deliveries"
        archive_parent.mkdir(parents=True, exist_ok=True)
        destination = archive_parent / delivery["delivery_id"]
        require(not destination.exists(), "delivery archive already exists")
        staging = archive_parent / f".{delivery['delivery_id']}.revoking"
        require(not staging.exists(), "an incomplete revocation already requires reconciliation")
        staging.mkdir(mode=0o700)
        try:
            os.replace(active, staging / "published_batch")
            os.replace(witness, staging / "delivery_receipt_private.json")
            os.replace(pending / "response.json", staging / "response.json")
            os.replace(collection_path, staging / "collection_receipt.json")
            (pending / "delivery_receipt.json").unlink()
            pending.rmdir()
            _write_exclusive_json(staging / "revocation_receipt.json", receipt, mode=0o400)
            require(not (public_root / "active").exists(), "active handoff still exists after revocation")
            os.replace(staging, destination)
        except BaseException:
            # A .revoking directory is intentionally terminal-blocking evidence;
            # never silently re-release after an interrupted revocation.
            raise
        _lifecycle(private_root)
        return receipt


def _materialize_response_set(private_root: Path, slot: str, destination: Path) -> Path:
    mapping, _ = _mapping_for_slot(private_root, slot)
    batches = mapping["packets"][slot]["batches"]
    lifecycle = _lifecycle(private_root)
    collected = {
        delivery["batch_ordinal"]: path / "response.json"
        for path, delivery, _, _ in lifecycle if delivery["rater_slot"] == slot
    }
    require(set(collected) == set(range(1, len(batches) + 1)),
            f"{slot} does not have every frozen batch collected and revoked")
    require(not destination.exists(), f"{slot} immutable response set already exists")
    destination.mkdir(parents=True, mode=0o700)
    for ordinal, batch in enumerate(batches, start=1):
        _copy_exclusive(collected[ordinal], destination / f"{batch['packet_id']}.json")
    try:
        annotation._validate_response_set(destination, mapping=mapping, slot=slot)
    except BaseException as error:
        raise AnnotationGateError(f"{slot} materialized response set failed: {error}") from error
    return destination


def prepare_adjudication(
    *, private_root: Path, public_root: Path, completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Mechanically package only disagreements after both human first passes."""

    validate_contract()
    private_root = Path(private_root).resolve(strict=True)
    public_root = Path(public_root).resolve(strict=True)
    completed_at_utc = completed_at_utc or _utc_now()
    _event_timestamp(completed_at_utc, "adjudication package completed_at_utc")
    with _gate_lock(private_root):
        _validate_public_root(public_root)
        require(not (public_root / "active").exists(), "cannot package adjudication while a batch is active")
        require(not _inflight_delivery_witnesses(private_root)
                and not _pending_collection_directories(private_root),
                "cannot package adjudication with an unreconciled annotation lifecycle")
        lifecycle = _lifecycle(private_root)
        if lifecycle:
            require(_timestamp(lifecycle[-1][3]["revoked_at_utc"], "latest revocation time")
                    <= _timestamp(completed_at_utc, "adjudication package completed_at_utc"),
                    "adjudication packaging time precedes a first-pass revocation")
        receipt_path = private_root / "adjudication_package_receipt.json"
        require(not receipt_path.exists(), "adjudication package already exists")
        response_root = private_root / "first_pass_response_sets"
        require(not response_root.exists(), "first-pass response-set output already exists")
        rater_a = _materialize_response_set(private_root, "rater_a", response_root / "rater_a")
        rater_b = _materialize_response_set(private_root, "rater_b", response_root / "rater_b")
        packet_root = private_root / "adjudication_packets"
        map_path = private_root / "restricted" / "adjudication_map.json"
        try:
            result = annotation.package_adjudication(
                restricted_map_path=private_root / "restricted" / "identity_map.json",
                rater_a_response_path=rater_a,
                rater_b_response_path=rater_b,
                packet_root=packet_root,
                adjudication_map_path=map_path,
            )
        except BaseException as error:
            raise AnnotationGateError(f"adjudication packet packaging failed: {error}") from error
        os.chmod(map_path, 0o600)
        receipt = _sign({
            "schema_version": ADJUDICATION_PACKAGE_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "private_blind_adjudication_packets_ready",
            "source_package_receipt_sha256": _sha256_file(private_root / "package_receipt.json"),
            "first_pass_response_sets": {
                "rater_a": {"path": str(rater_a), "artifact_sha256": annotation.artifact_sha256(rater_a)},
                "rater_b": {"path": str(rater_b), "artifact_sha256": annotation.artifact_sha256(rater_b)},
            },
            "adjudication_map": _descriptor(map_path),
            "packet_tree": ({"path": str(packet_root), **_tree_descriptor(packet_root)}
                            if result["adjudication_required_count"] else None),
            "counts": {
                "first_pass_images": result["first_pass_image_count"],
                "adjudication_required": result["adjudication_required_count"],
                "adjudication_batches": result["adjudication_batch_count"],
            },
            "labels_created_by_job": 0,
            "science_counts": _zero_science_counts(),
            "safe_to_release_confirmation": False,
            "claim_boundary": (
                "Only first-pass disagreements were mechanically packaged for a blind independent adjudicator. "
                "No adjudication label, consensus, usability decision, or confirmation release was created."
            ),
            "completed_at_utc": completed_at_utc,
        })
        _write_exclusive_json(receipt_path, receipt)
        return receipt


def write_time_accounting(
    *, private_root: Path, output_path: Path, completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Report human-entered annotation time; never infer or manufacture it."""

    validate_contract()
    private_root = Path(private_root).resolve(strict=True)
    output_path = Path(output_path).absolute()
    completed_at_utc = completed_at_utc or _utc_now()
    _event_timestamp(completed_at_utc, "time accounting completed_at_utc")
    with _gate_lock(private_root):
        require(not output_path.exists(), "annotation time receipt already exists")
        lifecycle = _lifecycle(private_root)
        if lifecycle:
            require(_timestamp(lifecycle[-1][3]["revoked_at_utc"], "latest revocation time")
                    <= _timestamp(completed_at_utc, "time accounting completed_at_utc"),
                    "time accounting timestamp precedes the latest revocation")
        by_slot: dict[str, dict[str, Any]] = {}
        for slot in SLOTS:
            rows = [(path, delivery, collection) for path, delivery, collection, _ in lifecycle
                    if delivery["rater_slot"] == slot]
            by_slot[slot] = {
                "response_count": len(rows),
                "annotation_count": sum(collection["annotation_count_received"] for _, _, collection in rows),
                "annotation_seconds_total": math.fsum(
                    collection["response_timing"]["annotation_seconds"] for _, _, collection in rows
                ),
                "wall_seconds_total": math.fsum(
                    collection["response_timing"]["wall_seconds"] for _, _, collection in rows
                ),
                "collection_receipt_sha256": [
                    _sha256_file(path / "collection_receipt.json") for path, _, _ in rows
                ],
            }
        require(by_slot["rater_a"]["response_count"] > 0
                and by_slot["rater_b"]["response_count"] > 0,
                "time accounting requires both first-pass human response streams")
        adjudication_path = private_root / "adjudication_package_receipt.json"
        require(adjudication_path.is_file(), "time accounting requires adjudication packaging decision")
        adjudication, _ = _adjudication_package_receipt(private_root)
        required = adjudication["counts"]["adjudication_required"]
        expected_adjudication_batches = adjudication["counts"]["adjudication_batches"]
        require(by_slot["adjudicator"]["response_count"] == expected_adjudication_batches,
                "adjudicator response count does not match required frozen batches")
        require((required == 0) == (by_slot["adjudicator"]["response_count"] == 0),
                "adjudicator timing is absent or present contrary to the disagreement decision")
        receipt = _sign({
            "schema_version": TIME_RECEIPT_SCHEMA,
            "study_id": STUDY_ID,
            "stage": "development",
            "status": "human_entered_time_accounted",
            "package_receipt_sha256": _sha256_file(private_root / "package_receipt.json"),
            "adjudication_package_receipt_sha256": _sha256_file(adjudication_path),
            "by_slot": by_slot,
            "annotation_seconds": {
                "rater_a_total": by_slot["rater_a"]["annotation_seconds_total"],
                "rater_b_total": by_slot["rater_b"]["annotation_seconds_total"],
                "adjudicator_total": by_slot["adjudicator"]["annotation_seconds_total"],
            },
            "labels_created_by_job": 0,
            "science_counts": _zero_science_counts(),
            "safe_to_release_confirmation": False,
            "completed_at_utc": completed_at_utc,
            "claim_boundary": (
                "Totals are sums of human-entered per-image seconds in collected hash-bound responses. "
                "They are not GPU, model, simulator, action, label-generation, or confirmation evidence."
            ),
        })
        _write_exclusive_json(output_path, receipt)
        return receipt


def _read_descriptor(path: Path, sha256: str, size: int) -> dict[str, Any]:
    return {"path": str(Path(path).resolve()), "sha256": sha256, "bytes": size}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package-private")
    package.add_argument("--study-commit", required=True)
    for name in ("media-job-receipt", "pixel-blindness-receipt", "development-freeze"):
        package.add_argument(f"--{name}", type=Path, required=True)
        package.add_argument(f"--{name}-sha256", required=True)
        package.add_argument(f"--{name}-bytes", type=int, required=True)
    package.add_argument("--private-root", type=Path, required=True)
    release = commands.add_parser("release-batch")
    release.add_argument("--private-root", type=Path, required=True)
    release.add_argument("--public-root", type=Path, required=True)
    release.add_argument("--slot", choices=SLOTS, required=True)
    release.add_argument("--ordinal", type=int, required=True)
    release.add_argument("--released-by", required=True)
    collect = commands.add_parser("collect-response")
    collect.add_argument("--private-root", type=Path, required=True)
    collect.add_argument("--public-root", type=Path, required=True)
    collect.add_argument("--response", type=Path, required=True)
    collect.add_argument("--response-sha256", required=True)
    collect.add_argument("--response-bytes", type=int, required=True)
    collect.add_argument("--collector-code", required=True)
    revoke = commands.add_parser("revoke-delivery")
    revoke.add_argument("--private-root", type=Path, required=True)
    revoke.add_argument("--public-root", type=Path, required=True)
    revoke.add_argument("--revoked-by", required=True)
    adjudicate = commands.add_parser("prepare-adjudication")
    adjudicate.add_argument("--private-root", type=Path, required=True)
    adjudicate.add_argument("--public-root", type=Path, required=True)
    accounting = commands.add_parser("write-time-accounting")
    accounting.add_argument("--private-root", type=Path, required=True)
    accounting.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package-private":
        result = package_private_packets(
            study_commit=args.study_commit,
            media_job_receipt=_read_descriptor(args.media_job_receipt, args.media_job_receipt_sha256,
                                               args.media_job_receipt_bytes),
            pixel_blindness_receipt=_read_descriptor(args.pixel_blindness_receipt,
                                                     args.pixel_blindness_receipt_sha256,
                                                     args.pixel_blindness_receipt_bytes),
            development_freeze=_read_descriptor(args.development_freeze,
                                                args.development_freeze_sha256,
                                                args.development_freeze_bytes),
            private_root=args.private_root,
        )
    elif args.command == "release-batch":
        result = release_batch(private_root=args.private_root, public_root=args.public_root,
                               slot=args.slot, ordinal=args.ordinal, released_by=args.released_by)
    elif args.command == "collect-response":
        result = collect_response(
            private_root=args.private_root, public_root=args.public_root,
            response=_read_descriptor(args.response, args.response_sha256, args.response_bytes),
            collector_code=args.collector_code,
        )
    elif args.command == "revoke-delivery":
        result = revoke_delivery(private_root=args.private_root, public_root=args.public_root,
                                 revoked_by=args.revoked_by)
    elif args.command == "prepare-adjudication":
        result = prepare_adjudication(private_root=args.private_root, public_root=args.public_root)
    else:
        result = write_time_accounting(private_root=args.private_root, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
