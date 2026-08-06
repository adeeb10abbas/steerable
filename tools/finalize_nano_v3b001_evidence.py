#!/usr/bin/env python3
"""Close the complete Nano V3-B001 result over its retained evidence.

The aggregate compiler deliberately writes statistics, not a publication
evidence index.  This tool verifies that aggregate by reproducing it from the
exact 108 post-close raw batches, verifies every file record reachable from
those behavioral rows, and emits one compact hash manifest.  Optional queue
``attempt_events.jsonl`` files are never guessed into scientific records:
only already-complete infrastructure records are normalized, while all other
event rows remain explicitly named, hash-bound non-denominator sources.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (  # noqa: E402
    AMENDMENT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REPOSITORY,
    RELATIONS,
    SEEDS,
    STUDY_ID,
    load_release_bundle,
)
from tools.compile_nano_v3b001_results import (  # noqa: E402
    AGGREGATE_MANIFEST_SCHEMA,
    BATCH_MANIFEST_SCHEMA,
    EPISODE_FILENAME,
    INFRASTRUCTURE_FILENAME,
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA,
    compile_nano_v3b001_results,
)
from tools.vla_wam_v3_episode_schema import (  # noqa: E402
    BEHAVIORAL_SCHEMA_VERSION,
    INFRASTRUCTURE_SCHEMA_VERSION,
    validate_raw_episode_record,
)


FINAL_MANIFEST_FILENAME = "nano_v3b001_final_evidence_manifest.json"
NORMALIZED_INFRASTRUCTURE_FILENAME = (
    "nano_v3b001_normalized_infrastructure_attempts.jsonl"
)
FINAL_MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-final-evidence-v1"
_SHA256_HEX = frozenset("0123456789abcdef")


class EvidenceFinalizationError(RuntimeError):
    """Raised when the retained evidence cannot be closed without guessing."""


def _fail(message: str) -> None:
    raise EvidenceFinalizationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_HEX)
    )


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        _fail(f"missing or empty retained evidence file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _virtual_file_record(path: Path, payload: bytes) -> dict[str, Any]:
    if not payload:
        _fail(f"refusing to describe an empty output: {path}")
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }


def _canonical_json(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        else:
            text = json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
    except (TypeError, ValueError) as exc:
        raise EvidenceFinalizationError(f"non-canonical JSON value: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            _fail(f"duplicate JSON key is prohibited: {key}")
        output[key] = value
    return output


def _load_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceFinalizationError(f"cannot parse {label}: {exc}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = _load_json_bytes(Path(path).read_bytes(), label)
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def _resolve_declared_path(value: str, *, source: Path, label: str) -> Path:
    declared = Path(value)
    if declared.is_absolute():
        return declared.resolve()
    candidate = (source.parent / declared).resolve()
    if not candidate.exists():
        _fail(
            f"relative {label} does not resolve beside its containing JSONL: "
            f"{value!r} from {source}"
        )
    return candidate


def _verify_claimed_file(
    claim: Mapping[str, Any], *, source: Path, label: str
) -> tuple[Path, dict[str, Any]]:
    path_value = claim.get("path")
    expected_sha256 = claim.get("sha256")
    expected_bytes = claim.get("bytes")
    if not isinstance(path_value, str) or not path_value:
        _fail(f"{label} lacks a non-empty path")
    if not _is_sha256(expected_sha256):
        _fail(f"{label} lacks a lowercase SHA-256")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        _fail(f"{label} lacks a positive byte count")
    path = _resolve_declared_path(path_value, source=source, label=label)
    actual = _file_record(path)
    if actual["sha256"] != expected_sha256 or actual["bytes"] != expected_bytes:
        _fail(f"tampered {label}: {path}")
    return path, actual


def _discover(roots: Iterable[Path], filename: str) -> list[Path]:
    found: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            _fail(f"discovery root is not a directory: {root}")
        found.extend(sorted(path.resolve() for path in root.rglob(filename) if path.is_file()))
    return found


def _require_unique_paths(paths: Sequence[Path], label: str) -> list[Path]:
    resolved = [Path(path).resolve() for path in paths]
    seen: set[Path] = set()
    duplicates: list[str] = []
    for path in resolved:
        if path in seen:
            duplicates.append(str(path))
        seen.add(path)
    if duplicates:
        _fail(f"duplicate {label} input paths: {sorted(set(duplicates))}")
    return sorted(resolved)


def _load_behavioral_batch(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load exactly one behavioral row and its post-close manifest."""

    path = Path(path).resolve()
    jsonl_record = _file_record(path)
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest_record = _file_record(manifest_path)
    manifest = _load_json(manifest_path, "behavioral batch manifest")
    expected_manifest = {
        "schema_version": BATCH_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "jsonl_sha256": jsonl_record["sha256"],
        "jsonl_bytes": jsonl_record["bytes"],
        "row_count": 1,
        "record_schema_versions": [BEHAVIORAL_SCHEMA_VERSION],
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            _fail(f"behavioral batch manifest mismatch for {key}: {path}")
    declared = manifest.get("jsonl_path")
    if not isinstance(declared, str) or Path(declared).resolve() != path:
        _fail(f"behavioral batch manifest does not bind its JSONL: {path}")
    lines = path.read_bytes().splitlines()
    if len(lines) != 1 or not lines[0].strip():
        _fail(f"behavioral batch must contain exactly one non-empty row: {path}")
    parsed = _load_json_bytes(lines[0], f"behavioral row {path}")
    if not isinstance(parsed, dict):
        _fail(f"behavioral row must be an object: {path}")
    try:
        normalized = validate_raw_episode_record(parsed)
    except Exception as exc:
        raise EvidenceFinalizationError(
            f"invalid behavioral row {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        normalized.get("schema_version") != BEHAVIORAL_SCHEMA_VERSION
        or normalized.get("record_type") != "behavioral_episode"
        or normalized.get("behavioral_result_valid") is not True
    ):
        _fail(f"nonbehavioral row supplied as released evidence: {path}")
    raw_result = normalized.get("artifacts", {}).get("raw_result_jsonl", {})
    raw_path = raw_result.get("path") if isinstance(raw_result, dict) else None
    if not isinstance(raw_path, str) or Path(raw_path).resolve() != path:
        _fail(f"behavioral row does not bind its containing JSONL: {path}")
    return normalized, jsonl_record, manifest_record


def _expected_cell_ids() -> set[str]:
    return {
        f"v3b001:nano:seed{seed}:{arm}:{relation}"
        for seed in SEEDS
        for arm in ("control", "position_mirrored")
        for relation in RELATIONS
    }


def _collect_artifact_claims(
    value: Any,
    *,
    source_jsonl: Path,
    cell_id: str,
    location: str,
    artifacts: dict[Path, dict[str, Any]],
    evidence_kind: str = "behavioral",
) -> None:
    """Recursively verify every nested ``path/sha256/bytes`` file record."""

    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            path, actual = _verify_claimed_file(
                value,
                source=source_jsonl,
                label=f"{cell_id}:{location}",
            )
            reference = {
                "cell_id": cell_id,
                "evidence_kind": evidence_kind,
                "location": location,
                "source_jsonl": str(source_jsonl),
            }
            existing = artifacts.get(path)
            if existing is None:
                artifacts[path] = {**actual, "references": [reference]}
            else:
                if (
                    existing["sha256"] != actual["sha256"]
                    or existing["bytes"] != actual["bytes"]
                ):
                    _fail(f"conflicting artifact claims for {path}")
                if reference not in existing["references"]:
                    existing["references"].append(reference)
            return
        for key in sorted(value):
            child_location = f"{location}.{key}" if location else key
            _collect_artifact_claims(
                value[key],
                source_jsonl=source_jsonl,
                cell_id=cell_id,
                location=child_location,
                artifacts=artifacts,
                evidence_kind=evidence_kind,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_artifact_claims(
                child,
                source_jsonl=source_jsonl,
                cell_id=cell_id,
                location=f"{location}[{index}]",
                artifacts=artifacts,
                evidence_kind=evidence_kind,
            )


def _compiled_infrastructure_sources(
    compiled_directory: Path,
    summary: Mapping[str, Any],
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    infrastructure = summary.get("infrastructure_evidence")
    if not isinstance(infrastructure, dict):
        _fail("compiled summary lacks infrastructure_evidence")
    aggregate = infrastructure.get("aggregate_jsonl")
    aggregate_path = compiled_directory / INFRASTRUCTURE_FILENAME
    manifest_path = aggregate_path.with_name(aggregate_path.name + ".manifest.json")
    if aggregate is None:
        if aggregate_path.exists() or manifest_path.exists():
            _fail("unbound compiled infrastructure output exists")
        if infrastructure.get("provided_attempt_count") != 0:
            _fail("compiled summary has an infrastructure count without an aggregate")
        return [], {}
    if not isinstance(aggregate, dict):
        _fail("compiled infrastructure aggregate descriptor must be an object")
    aggregate_record = _file_record(aggregate_path)
    manifest_record = _file_record(manifest_path)
    expected = {
        "path": INFRASTRUCTURE_FILENAME,
        "sha256": aggregate_record["sha256"],
        "bytes": aggregate_record["bytes"],
        "manifest_path": INFRASTRUCTURE_FILENAME + ".manifest.json",
        "manifest_sha256": manifest_record["sha256"],
    }
    if aggregate != expected:
        _fail("compiled summary does not bind its infrastructure aggregate")
    manifest = _load_json(manifest_path, "compiled infrastructure aggregate manifest")
    if manifest.get("schema_version") != AGGREGATE_MANIFEST_SCHEMA:
        _fail("compiled infrastructure aggregate manifest has the wrong schema")
    sources = manifest.get("source_batches")
    if not isinstance(sources, list):
        _fail("compiled infrastructure aggregate lacks source batches")
    paths: list[Path] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            _fail(f"compiled infrastructure source {index} is not an object")
        jsonl = source.get("jsonl")
        batch = source.get("batch_manifest")
        if not isinstance(jsonl, dict) or not isinstance(batch, dict):
            _fail(f"compiled infrastructure source {index} is incomplete")
        jsonl_path, _ = _verify_claimed_file(
            jsonl, source=manifest_path, label=f"compiled infrastructure source {index}"
        )
        batch_path, _ = _verify_claimed_file(
            batch,
            source=manifest_path,
            label=f"compiled infrastructure batch {index}",
        )
        if batch_path != jsonl_path.with_name(jsonl_path.name + ".manifest.json"):
            _fail("compiled infrastructure source manifest is not adjacent to its JSONL")
        paths.append(jsonl_path)
    paths = _require_unique_paths(paths, "compiled infrastructure source")
    return paths, {
        "aggregate_jsonl": aggregate_record,
        "aggregate_manifest": manifest_record,
    }


def _verify_compiler_outputs(
    compiled_directory: Path,
    behavioral_jsonls: Sequence[Path],
) -> tuple[dict[str, Any], Any, dict[str, dict[str, Any]]]:
    """Recompile in a temporary directory and require byte-identical outputs."""

    compiled_directory = Path(compiled_directory).resolve()
    if not compiled_directory.is_dir():
        _fail(f"compiled output directory is missing: {compiled_directory}")
    summary_path = compiled_directory / SUMMARY_FILENAME
    episodes_path = compiled_directory / EPISODE_FILENAME
    episodes_manifest_path = episodes_path.with_name(episodes_path.name + ".manifest.json")
    summary_record = _file_record(summary_path)
    episodes_record = _file_record(episodes_path)
    episodes_manifest_record = _file_record(episodes_manifest_path)
    summary = _load_json(summary_path, "compiled Nano summary")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("study_id") != STUDY_ID
        or summary.get("amendment_id") != AMENDMENT_ID
        or summary.get("model_id") != MODEL_ID
        or summary.get("arena") != "droid_robolab"
    ):
        _fail("compiled summary is not the exact Nano V3-B001 result")
    behavioral = summary.get("behavioral_evidence")
    if not isinstance(behavioral, dict):
        _fail("compiled summary lacks behavioral_evidence")
    if (
        behavioral.get("valid_episode_count") != 108
        or behavioral.get("matched_seed_count") != 27
    ):
        _fail("compiled summary is not a complete 108-cell result")
    aggregate = behavioral.get("aggregate_jsonl")
    expected_aggregate = {
        "path": EPISODE_FILENAME,
        "sha256": episodes_record["sha256"],
        "bytes": episodes_record["bytes"],
        "manifest_path": EPISODE_FILENAME + ".manifest.json",
        "manifest_sha256": episodes_manifest_record["sha256"],
    }
    if aggregate != expected_aggregate:
        _fail("compiled summary does not bind its behavioral aggregate")
    release = summary.get("release")
    if not isinstance(release, dict):
        _fail("compiled summary lacks its release identity")
    manifest_value = release.get("manifest_path")
    manifest_sha256 = release.get("manifest_sha256")
    if not isinstance(manifest_value, str) or not _is_sha256(manifest_sha256):
        _fail("compiled summary release manifest identity is invalid")
    release_path = Path(manifest_value).resolve()
    release_record = _file_record(release_path)
    if release_record["sha256"] != manifest_sha256:
        _fail("released manifest changed after result compilation")
    try:
        release_bundle = load_release_bundle(
            release_path, expected_manifest_sha256=manifest_sha256
        )
    except Exception as exc:
        raise EvidenceFinalizationError(f"cannot load the released V3-B001 bundle: {exc}") from exc
    uncertainty = summary.get("uncertainty_contract")
    if not isinstance(uncertainty, dict):
        _fail("compiled summary lacks uncertainty_contract")
    replicates = uncertainty.get("bootstrap_replicates")
    master_seed = uncertainty.get("bootstrap_master_seed")
    if type(replicates) is not int or replicates < 1 or type(master_seed) is not int:
        _fail("compiled summary has an invalid bootstrap configuration")
    infrastructure_sources, infrastructure_outputs = _compiled_infrastructure_sources(
        compiled_directory, summary
    )
    with tempfile.TemporaryDirectory(prefix="nano-v3b001-finalizer-") as temporary:
        reproduced = compile_nano_v3b001_results(
            release_manifest=release_path,
            release_manifest_sha256=manifest_sha256,
            behavioral_jsonls=behavioral_jsonls,
            infrastructure_jsonls=infrastructure_sources,
            output_directory=Path(temporary) / "reproduced",
            bootstrap_replicates=replicates,
            bootstrap_seed=master_seed,
        )
        expected_paths = {
            "summary": summary_path,
            "episodes": episodes_path,
            "episodes_manifest": episodes_manifest_path,
        }
        if infrastructure_outputs:
            expected_paths.update(
                {
                    "infrastructure": compiled_directory / INFRASTRUCTURE_FILENAME,
                    "infrastructure_manifest": (
                        compiled_directory
                        / (INFRASTRUCTURE_FILENAME + ".manifest.json")
                    ),
                }
            )
        if set(reproduced) != set(expected_paths):
            _fail("compiled output file set does not reproduce exactly")
        for name, expected_path in expected_paths.items():
            if reproduced[name].read_bytes() != expected_path.read_bytes():
                _fail(f"compiled {name} is tampered or not reproducible")
    outputs = {
        "summary": summary_record,
        "aggregate_behavioral_jsonl": episodes_record,
        "aggregate_behavioral_manifest": episodes_manifest_record,
        **infrastructure_outputs,
    }
    return summary, release_bundle, outputs


def _extract_attempt_event_candidate(row: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(row, dict):
        return None, "event row is not a JSON object"
    if row.get("record_type") == "infrastructure_attempt":
        return row, "direct infrastructure record"
    nested = row.get("infrastructure_record")
    if isinstance(nested, dict):
        return nested, "nested infrastructure_record"
    return None, "no complete infrastructure record is present"


def _validate_release_bound_infrastructure(record: Mapping[str, Any], release: Any) -> None:
    cell_id = record.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail("derived infrastructure record lacks registered_cell_id")
    try:
        cell = release.cell(cell_id)
    except Exception as exc:
        raise EvidenceFinalizationError(
            f"derived infrastructure record is outside the V3-B001 release: {cell_id}"
        ) from exc
    expected = {
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "pair_id": cell.row["matched_block_id"],
        "environment_seed": cell.seed,
        "policy_seed": cell.seed,
        "prompt": cell.row["prompt"],
        "prompt_family": "direct_command",
        "predicate_id": cell.row["success_predicate_id"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            _fail(f"derived infrastructure record mismatches released {key}: {cell_id}")
    checkpoint = record.get("checkpoint")
    if checkpoint != {"id": MODEL_REPOSITORY, "revision": CHECKPOINT_REVISION}:
        _fail(f"derived infrastructure record has the wrong checkpoint: {cell_id}")


def _normalize_attempt_events(
    paths: Sequence[Path],
    *,
    output_jsonl: Path,
    release: Any,
    artifacts: dict[Path, dict[str, Any]],
) -> tuple[bytes | None, bytes | None, dict[str, Any]]:
    source_summaries: list[dict[str, Any]] = []
    converted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    attempt_ids: set[str] = set()
    total_unconvertible = 0
    for path in _require_unique_paths(paths, "attempt-event"):
        source = _file_record(path)
        lines = path.read_bytes().splitlines()
        converted_count = 0
        unconvertible_count = 0
        unconvertible_reasons: Counter[str] = Counter()
        for row_number, line in enumerate(lines, 1):
            row_sha256 = _sha256_bytes(line)
            if not line.strip():
                unconvertible_count += 1
                unconvertible_reasons["blank event row"] += 1
                continue
            try:
                row = _load_json_bytes(line, f"attempt event {path}:{row_number}")
            except EvidenceFinalizationError as exc:
                unconvertible_count += 1
                unconvertible_reasons[f"invalid JSON event: {exc}"] += 1
                continue
            candidate, source_kind = _extract_attempt_event_candidate(row)
            if candidate is None:
                unconvertible_count += 1
                unconvertible_reasons[source_kind] += 1
                continue
            try:
                normalized = validate_raw_episode_record(candidate)
                _validate_release_bound_infrastructure(normalized, release)
            except Exception as exc:
                unconvertible_count += 1
                reason = (
                    "incomplete or invalid infrastructure record: "
                    f"{type(exc).__name__}: {exc}"
                )
                unconvertible_reasons[reason] += 1
                continue
            attempt_id = normalized["attempt_id"]
            if attempt_id in attempt_ids:
                _fail(f"duplicate convertible infrastructure attempt_id: {attempt_id}")
            attempt_ids.add(attempt_id)
            _collect_artifact_claims(
                normalized,
                source_jsonl=path,
                cell_id=normalized["registered_cell_id"],
                location="infrastructure",
                artifacts=artifacts,
                evidence_kind="infrastructure",
            )
            rewritten = copy.deepcopy(normalized)
            rewritten["artifacts"]["raw_result_jsonl"] = {
                "path": str(output_jsonl.resolve()),
                "integrity_scope": "batch_manifest_after_close",
            }
            if "final_evidence_normalization_source" in rewritten:
                _fail("infrastructure record already uses reserved normalization provenance")
            rewritten["final_evidence_normalization_source"] = {
                "path": source["path"],
                "sha256": source["sha256"],
                "bytes": source["bytes"],
                "row_number": row_number,
                "row_sha256": row_sha256,
                "source_kind": source_kind,
            }
            try:
                rewritten = validate_raw_episode_record(rewritten)
            except Exception as exc:
                raise EvidenceFinalizationError(
                    f"normalized infrastructure record became invalid: {attempt_id}: {exc}"
                ) from exc
            converted.append((rewritten, rewritten["final_evidence_normalization_source"]))
            converted_count += 1
        source_summaries.append(
            {
                **source,
                "row_count": len(lines),
                "converted_record_count": converted_count,
                "unconvertible_event_count": unconvertible_count,
                "unconvertible_reason_counts": dict(
                    sorted(unconvertible_reasons.items())
                ),
                "behavioral_denominator_included": False,
            }
        )
        total_unconvertible += unconvertible_count
    converted.sort(
        key=lambda item: (
            item[0]["registered_cell_id"],
            item[0]["attempt_id"],
            item[1]["path"],
            item[1]["row_number"],
        )
    )
    normalized_payload: bytes | None = None
    normalized_manifest_payload: bytes | None = None
    normalized_descriptor: dict[str, Any] | None = None
    if converted:
        normalized_payload = b"".join(_canonical_json(record) for record, _ in converted)
        normalized_manifest = {
            "schema_version": BATCH_MANIFEST_SCHEMA,
            "study_id": STUDY_ID,
            "jsonl_path": str(output_jsonl.resolve()),
            "jsonl_sha256": _sha256_bytes(normalized_payload),
            "jsonl_bytes": len(normalized_payload),
            "row_count": len(converted),
            "record_schema_versions": [INFRASTRUCTURE_SCHEMA_VERSION],
        }
        normalized_manifest_payload = _canonical_json(normalized_manifest)
        normalized_descriptor = {
            "jsonl": _virtual_file_record(output_jsonl, normalized_payload),
            "batch_manifest": _virtual_file_record(
                output_jsonl.with_name(output_jsonl.name + ".manifest.json"),
                normalized_manifest_payload,
            ),
            "row_count": len(converted),
        }
    source_summaries.sort(key=lambda item: item["path"])
    return normalized_payload, normalized_manifest_payload, {
        "behavioral_denominator_included": False,
        "normalized": normalized_descriptor,
        "source_files": source_summaries,
        "unconvertible_event_count": total_unconvertible,
        "policy": (
            "Only complete records that independently satisfy the frozen V3 infrastructure "
            "schema and exact V3-B001 release identity are normalized. Other event rows remain "
            "hash-bound non-denominator sources; no missing schema field is fabricated."
        ),
    }


def _prepare_output_directory(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists():
        if not path.is_dir():
            _fail(f"output path exists and is not a directory: {path}")
        contents = list(path.iterdir())
        if contents:
            _fail(f"output directory must be empty; refusing overwrite: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with Path(path).open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise EvidenceFinalizationError(f"refusing to overwrite final evidence: {path}") from exc


def finalize_nano_v3b001_evidence(
    *,
    compiled_output_directory: Path,
    output_directory: Path,
    behavioral_roots: Sequence[Path] = (),
    behavioral_jsonls: Sequence[Path] = (),
    attempt_event_roots: Sequence[Path] = (),
    attempt_event_jsonls: Sequence[Path] = (),
) -> dict[str, Path]:
    """Verify and close one complete, immutable Nano V3-B001 evidence slice."""

    behavioral_paths = _require_unique_paths(
        [*behavioral_jsonls, *_discover(behavioral_roots, "raw_episode.jsonl")],
        "behavioral",
    )
    if len(behavioral_paths) != 108:
        _fail(f"expected exactly 108 behavioral JSONLs, received {len(behavioral_paths)}")
    by_cell: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]] = {}
    artifacts: dict[Path, dict[str, Any]] = {}
    for path in behavioral_paths:
        record, jsonl_record, manifest_record = _load_behavioral_batch(path)
        cell_id = record["registered_cell_id"]
        if cell_id in by_cell:
            _fail(f"duplicate behavioral registered_cell_id: {cell_id}")
        by_cell[cell_id] = (record, jsonl_record, manifest_record, path)
        _collect_artifact_claims(
            record,
            source_jsonl=path,
            cell_id=cell_id,
            location="record",
            artifacts=artifacts,
        )
    if set(by_cell) != _expected_cell_ids():
        missing = sorted(_expected_cell_ids() - set(by_cell))
        extra = sorted(set(by_cell) - _expected_cell_ids())
        _fail(f"behavioral cell set mismatch; missing={missing}, extra={extra}")

    summary, release, compiled_outputs = _verify_compiler_outputs(
        Path(compiled_output_directory), behavioral_paths
    )
    if set(release.by_cell_id) != _expected_cell_ids():
        _fail("compiled summary release is not the exact 108-cell V3-B001 queue")

    output_path = Path(output_directory).resolve()
    normalized_infrastructure_path = output_path / NORMALIZED_INFRASTRUCTURE_FILENAME
    event_paths = _require_unique_paths(
        [*attempt_event_jsonls, *_discover(attempt_event_roots, "attempt_events.jsonl")],
        "attempt-event",
    )
    (
        normalized_infrastructure_payload,
        normalized_infrastructure_manifest_payload,
        infrastructure,
    ) = _normalize_attempt_events(
        event_paths,
        output_jsonl=normalized_infrastructure_path,
        release=release,
        artifacts=artifacts,
    )

    behavioral_sources = [
        {
            "registered_cell_id": cell_id,
            "raw_episode_jsonl": item[1],
            "batch_manifest": item[2],
        }
        for cell_id, item in sorted(by_cell.items())
    ]
    referenced_artifacts = []
    for path, record in sorted(artifacts.items(), key=lambda item: str(item[0])):
        record["references"].sort(
            key=lambda ref: (
                ref["cell_id"],
                ref["evidence_kind"],
                ref["location"],
                ref["source_jsonl"],
            )
        )
        referenced_artifacts.append(record)

    release_record = _file_record(Path(summary["release"]["manifest_path"]))
    code_records = {
        "aggregate_compiler": _file_record(ROOT / "tools/compile_nano_v3b001_results.py"),
        "episode_schema": _file_record(ROOT / "tools/vla_wam_v3_episode_schema.py"),
        "final_evidence_closer": _file_record(Path(__file__)),
    }
    final_manifest = {
        "schema_version": FINAL_MANIFEST_SCHEMA,
        "status": "complete_hash_closed_108_cell_evidence",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "release_manifest": release_record,
        "runtime_identity_sha256": summary["runtime_identity_sha256"],
        "counts": {
            "behavioral_episode_count": 108,
            "matched_seed_count": 27,
            "raw_batch_count": len(behavioral_sources),
            "unique_referenced_artifact_count": len(referenced_artifacts),
            "artifact_reference_count": sum(
                len(record["references"]) for record in referenced_artifacts
            ),
            "normalized_infrastructure_attempt_count": (
                0
                if infrastructure["normalized"] is None
                else infrastructure["normalized"]["row_count"]
            ),
            "unconvertible_attempt_event_count": infrastructure[
                "unconvertible_event_count"
            ],
        },
        "compiled_outputs": compiled_outputs,
        "behavioral_sources": behavioral_sources,
        "referenced_artifacts": referenced_artifacts,
        "infrastructure": infrastructure,
        "statistics_contract": {
            "primary": "paired signed final lateral offset over all 27 four-cell blocks",
            "secondary": "requested-side margin only for the named all-four-correct complete-case subset",
            "valid_behavioral_failures_included": True,
            "infrastructure_in_behavioral_denominator": False,
            "summary": compiled_outputs["summary"],
        },
        "hash_policy": (
            "The aggregate compiler output was reproduced byte-for-byte from exactly 108 "
            "post-close behavioral batches. Every raw JSONL, batch manifest, and nested "
            "path/SHA-256/byte artifact record was rehashed from storage. Attempt events "
            "without complete schema-valid infrastructure records remain separately named "
            "source hashes and never enter the behavioral denominator."
        ),
        "reproducible_code": code_records,
    }
    final_manifest_payload = _canonical_json(final_manifest, pretty=True)

    output_path = _prepare_output_directory(output_path)
    outputs: dict[str, Path] = {}
    if normalized_infrastructure_payload is not None:
        if normalized_infrastructure_manifest_payload is None:
            _fail("internal error: normalized infrastructure manifest is missing")
        _write_exclusive(
            normalized_infrastructure_path, normalized_infrastructure_payload
        )
        normalized_manifest_path = normalized_infrastructure_path.with_name(
            normalized_infrastructure_path.name + ".manifest.json"
        )
        _write_exclusive(
            normalized_manifest_path, normalized_infrastructure_manifest_payload
        )
        outputs["normalized_infrastructure"] = normalized_infrastructure_path
        outputs["normalized_infrastructure_manifest"] = normalized_manifest_path
    final_manifest_path = output_path / FINAL_MANIFEST_FILENAME
    _write_exclusive(final_manifest_path, final_manifest_payload)
    outputs["final_evidence_manifest"] = final_manifest_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-output-directory", type=Path, required=True)
    parser.add_argument("--behavioral-root", type=Path, action="append", default=[])
    parser.add_argument("--behavioral-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--attempt-events-root", type=Path, action="append", default=[])
    parser.add_argument("--attempt-events-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    outputs = finalize_nano_v3b001_evidence(
        compiled_output_directory=args.compiled_output_directory,
        behavioral_roots=args.behavioral_root,
        behavioral_jsonls=args.behavioral_jsonl,
        attempt_event_roots=args.attempt_events_root,
        attempt_event_jsonls=args.attempt_events_jsonl,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                name: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(outputs.items())
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
