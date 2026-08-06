#!/usr/bin/env python3
"""Compile the complete Nano V3-B001 position-reflection result.

This aggregate compiler is intentionally stricter than a plotting script.  It
accepts only the exact hash-bound 108-cell release, requires one valid
behavioral JSONL (and its post-close manifest) for every released cell, and
keeps technical/partial attempts in a separate optional stream.  The primary
estimands are calculated from all 27 four-cell matched blocks; valid failures
are measurements, not missing values.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence

from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    AMENDMENT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REPOSITORY,
    PROMPTS,
    RELATIONS,
    SEEDS,
    STUDY_ID,
    ReleaseBundle,
    RuntimeContractError,
    load_release_bundle,
    sha256_file,
)
from tools.vla_wam_v3_episode_schema import (
    BEHAVIORAL_SCHEMA_VERSION,
    INFRASTRUCTURE_SCHEMA_VERSION,
    validate_raw_episode_record,
)


SUMMARY_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-results-v1"
AGGREGATE_MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-jsonl-manifest-v1"
BATCH_MANIFEST_SCHEMA = "vla-wam-shared-v3-jsonl-batch-manifest-v1"
EPISODE_FILENAME = "nano_v3b001_episodes.jsonl"
INFRASTRUCTURE_FILENAME = "nano_v3b001_infrastructure_attempts.jsonl"
SUMMARY_FILENAME = "nano_v3b001_summary.json"
ARMS = ("control", "position_mirrored")
CONDITIONS = tuple((arm, relation) for arm in ARMS for relation in RELATIONS)
COMPLETE_CASE_SUBSET_ID = "nano_v3b001_all_four_cells_correct"
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_SEED = 3_104_159
_SHA256_HEX = frozenset("0123456789abcdef")


class AggregateCompilationError(RuntimeError):
    """Raised when retained evidence cannot support the frozen aggregate."""


def _fail(message: str) -> None:
    raise AggregateCompilationError(message)


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AggregateCompilationError(f"non-finite or non-serializable result: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise AggregateCompilationError(f"cannot parse {label}: {exc}") from exc


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        _fail(f"missing or empty retained evidence file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_HEX)
    )


def _load_batch(path: Path, *, expected_schema: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify one retained JSONL against its post-close batch manifest."""

    path = Path(path).resolve()
    source = _file_record(path)
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest_record = _file_record(manifest_path)
    manifest = _load_json_bytes(manifest_path.read_bytes(), f"batch manifest {manifest_path}")
    if not isinstance(manifest, dict):
        _fail(f"batch manifest must be an object: {manifest_path}")
    expected = {
        "schema_version": BATCH_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "jsonl_sha256": source["sha256"],
        "jsonl_bytes": source["bytes"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            _fail(f"batch manifest mismatch for {key}: {path}")
    declared_path = manifest.get("jsonl_path")
    if not isinstance(declared_path, str) or Path(declared_path).resolve() != path:
        _fail(f"batch manifest jsonl_path does not bind its source: {path}")

    raw_lines = path.read_bytes().splitlines()
    if not raw_lines or any(not line.strip() for line in raw_lines):
        _fail(f"JSONL must contain only non-empty rows: {path}")
    if manifest.get("row_count") != len(raw_lines):
        _fail(f"batch manifest row_count mismatch: {path}")
    if manifest.get("record_schema_versions") != [expected_schema]:
        _fail(f"batch manifest record schema mismatch: {path}")

    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw_lines, 1):
        parsed = _load_json_bytes(line, f"{path}:{index}")
        if not isinstance(parsed, dict):
            _fail(f"JSONL row must be an object: {path}:{index}")
        try:
            normalized = validate_raw_episode_record(parsed)
        except Exception as exc:
            raise AggregateCompilationError(
                f"invalid retained record {path}:{index}: {type(exc).__name__}: {exc}"
            ) from exc
        if normalized.get("schema_version") != expected_schema:
            _fail(f"unexpected record schema in {path}:{index}")
        raw_result = normalized.get("artifacts", {}).get("raw_result_jsonl", {})
        raw_result_path = raw_result.get("path") if isinstance(raw_result, dict) else None
        if not isinstance(raw_result_path, str) or Path(raw_result_path).resolve() != path:
            _fail(f"record does not bind the exact containing raw JSONL: {path}:{index}")
        rows.append(normalized)
    return rows, {"jsonl": source, "batch_manifest": manifest_record}


def _require_equal(record: Mapping[str, Any], key: str, expected: Any, cell_id: str) -> None:
    if record.get(key) != expected:
        _fail(f"{cell_id} does not match its released {key}")


def _validate_behavioral_cell(
    record: dict[str, Any],
    *,
    release: ReleaseBundle,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if record.get("record_type") != "behavioral_episode" or record.get("behavioral_result_valid") is not True:
        _fail("behavioral inputs may contain only valid behavioral_episode rows")
    cell_id = record.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail("behavioral row lacks registered_cell_id")
    try:
        cell = release.cell(cell_id)
    except RuntimeContractError as exc:
        raise AggregateCompilationError(str(exc)) from exc
    expected = {
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "pair_id": cell.row["matched_block_id"],
        "environment_seed": cell.seed,
        "policy_seed": cell.seed,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_family": "direct_command",
        "predicate_id": cell.row["success_predicate_id"],
        "phase_b_arm": cell.arm,
        "release_manifest_sha256": release.manifest_sha256,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
        "missing_future_policy": "infrastructure_invalid_never_zero",
    }
    for key, value in expected.items():
        _require_equal(record, key, value, cell_id)
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint != {
        "id": MODEL_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
    }:
        _fail(f"{cell_id} does not match the released checkpoint identity")
    runtime = record.get("runtime_identity")
    if not isinstance(runtime, dict) or not _is_sha256(runtime.get("sha256")):
        _fail(f"{cell_id} lacks a valid runtime identity digest")
    measurements = record.get("measurements")
    if not isinstance(measurements, dict):
        _fail(f"{cell_id} lacks schema-derived measurements")
    for field in ("signed_final_lateral_offset_m", "final_requested_signed_margin_m"):
        value = measurements.get(field)
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            _fail(f"{cell_id} lacks finite {field}")
    s = float(measurements["signed_final_lateral_offset_m"])
    expected_margin = s if cell.relation == "left" else -s
    if not math.isclose(
        float(measurements["final_requested_signed_margin_m"]),
        expected_margin,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _fail(f"{cell_id} requested margin is inconsistent with signed lateral offset")
    return {
        "record": record,
        "cell": cell,
        "source": source,
        "s": s,
        "margin": float(measurements["final_requested_signed_margin_m"]),
    }


def _validate_infrastructure_attempt(
    record: dict[str, Any], *, release: ReleaseBundle
) -> dict[str, Any]:
    if record.get("record_type") != "infrastructure_attempt" or record.get("behavioral_result_valid") is not False:
        _fail("infrastructure inputs may contain only nonbehavioral infrastructure_attempt rows")
    cell_id = record.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail("infrastructure row lacks registered_cell_id")
    try:
        cell = release.cell(cell_id)
    except RuntimeContractError as exc:
        raise AggregateCompilationError(str(exc)) from exc
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
        _require_equal(record, key, value, cell_id)
    return record


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _exact_sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    ties = len(values) - positive - negative
    effective = positive + negative
    if effective == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(effective, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**effective))
    return {
        "method": "exact_two_sided_paired_sign_test",
        "null": "positive and negative paired differences are equally probable",
        "positive": positive,
        "negative": negative,
        "ties_excluded": ties,
        "effective_n": effective,
        "p_value": p_value,
    }


def _exact_median_interval(values: Sequence[float], confidence: float = 0.95) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    chosen: tuple[int, float] | None = None
    for k in range(1, n // 2 + 2):
        coverage = 1.0 - 2.0 * sum(math.comb(n, j) for j in range(k)) / (2**n)
        if coverage >= confidence:
            chosen = (k, coverage)
    if chosen is None:
        return {
            "method": "exact_distribution_free_order_statistic_interval",
            "requested_confidence": confidence,
            "lower": None,
            "upper": None,
            "achieved_confidence": 1.0,
            "reason": "sample_too_small_for_finite_interval_at_requested_confidence",
        }
    k, coverage = chosen
    return {
        "method": "exact_distribution_free_order_statistic_interval",
        "requested_confidence": confidence,
        "achieved_confidence": coverage,
        "lower_order_statistic": k,
        "upper_order_statistic": n - k + 1,
        "lower": ordered[k - 1],
        "upper": ordered[n - k],
    }


def _bootstrap_interval(
    values: Sequence[float],
    *,
    label: str,
    statistic: str,
    replicates: int,
    master_seed: int,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    label_seed = int.from_bytes(
        hashlib.sha256(f"{master_seed}:{label}:{statistic}".encode("utf-8")).digest()[:8],
        "big",
    )
    generator = random.Random(label_seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(replicates):
        resample = [values[generator.randrange(n)] for _ in range(n)]
        samples.append(
            statistics.fmean(resample)
            if statistic == "mean"
            else float(statistics.median(resample))
        )
    samples.sort()
    return {
        "method": "matched_seed_nonparametric_percentile_bootstrap",
        "unit_of_resampling": "matched_seed",
        "statistic": statistic,
        "confidence": 0.95,
        "replicates": replicates,
        "seed": label_seed,
        "lower": _percentile(samples, 0.025),
        "upper": _percentile(samples, 0.975),
    }


def _robust_summary(
    values: Sequence[float],
    *,
    label: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite or not all(math.isfinite(value) for value in finite):
        _fail(f"{label} requires at least one finite matched-seed value")
    return {
        "n": len(finite),
        "mean_m": statistics.fmean(finite),
        "median_m": float(statistics.median(finite)),
        "sample_standard_deviation_m": (
            statistics.stdev(finite) if len(finite) > 1 else None
        ),
        "minimum_m": min(finite),
        "maximum_m": max(finite),
        "mean_bootstrap_95": _bootstrap_interval(
            finite,
            label=label,
            statistic="mean",
            replicates=bootstrap_replicates,
            master_seed=bootstrap_seed,
        ),
        "median_bootstrap_95": _bootstrap_interval(
            finite,
            label=label,
            statistic="median",
            replicates=bootstrap_replicates,
            master_seed=bootstrap_seed,
        ),
        "median_exact_interval": _exact_median_interval(finite),
        "paired_sign_test": _exact_sign_test(finite),
    }


def _condition_key(arm: str, relation: str) -> str:
    return f"{arm}:{relation}"


def _aggregate_analysis(
    indexed: Mapping[str, dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    vectors: dict[str, list[float]] = {
        **{f"s:{arm}:{relation}": [] for arm, relation in CONDITIONS},
        **{f"D:{arm}": [] for arm in ARMS},
        **{f"B:{arm}": [] for arm in ARMS},
        "I": [],
        "J": [],
    }
    success_complete: list[dict[str, Any]] = []
    condition_counts: dict[str, dict[str, Any]] = {}
    for arm, relation in CONDITIONS:
        cells = [
            indexed[f"v3b001:nano:seed{seed}:{arm}:{relation}"]["record"]
            for seed in SEEDS
        ]
        condition_counts[_condition_key(arm, relation)] = {
            "episodes": len(cells),
            "successes": sum(record["requested_success"] for record in cells),
            "failure_taxonomy_counts": dict(
                sorted(Counter(record["failure_taxonomy"] for record in cells).items())
            ),
        }

    for seed in SEEDS:
        cells = {
            (arm, relation): indexed[f"v3b001:nano:seed{seed}:{arm}:{relation}"]
            for arm, relation in CONDITIONS
        }
        s = {(arm, relation): cells[(arm, relation)]["s"] for arm, relation in CONDITIONS}
        d = {arm: s[(arm, "left")] - s[(arm, "right")] for arm in ARMS}
        b = {arm: -s[(arm, "right")] - s[(arm, "left")] for arm in ARMS}
        interaction_i = b["position_mirrored"] - b["control"]
        interaction_j = d["position_mirrored"] - d["control"]
        for arm, relation in CONDITIONS:
            vectors[f"s:{arm}:{relation}"].append(s[(arm, relation)])
        for arm in ARMS:
            vectors[f"D:{arm}"].append(d[arm])
            vectors[f"B:{arm}"].append(b[arm])
        vectors["I"].append(interaction_i)
        vectors["J"].append(interaction_j)
        cell_payload = {}
        for arm, relation in CONDITIONS:
            item = cells[(arm, relation)]
            record = item["record"]
            cell_payload[_condition_key(arm, relation)] = {
                "registered_cell_id": record["registered_cell_id"],
                "requested_success": record["requested_success"],
                "failure_taxonomy": record["failure_taxonomy"],
                "signed_final_lateral_offset_m": item["s"],
                "final_requested_signed_margin_m": item["margin"],
                "raw_episode_jsonl_sha256": item["source"]["jsonl"]["sha256"],
            }
        seed_row: dict[str, Any] = {
            "seed": seed,
            "matched_block_id": f"v3b001:nano:seed{seed}",
            "cells": cell_payload,
            "full_sample": {
                "D_control_m": d["control"],
                "D_position_mirrored_m": d["position_mirrored"],
                "B_control_m": b["control"],
                "B_position_mirrored_m": b["position_mirrored"],
                "I_position_reflection_interaction_m": interaction_i,
                "J_redirection_interaction_m": interaction_j,
            },
        }
        if all(item["record"]["requested_success"] for item in cells.values()):
            g = {
                arm: cells[(arm, "right")]["margin"] - cells[(arm, "left")]["margin"]
                for arm in ARMS
            }
            success_row = {
                "seed": seed,
                "G_control_m": g["control"],
                "G_position_mirrored_m": g["position_mirrored"],
                "G_position_reflection_interaction_m": (
                    g["position_mirrored"] - g["control"]
                ),
            }
            seed_row["success_conditional_secondary"] = success_row
            success_complete.append(success_row)
        seed_rows.append(seed_row)

    full_sample = {
        "population": {
            "matched_seed_count": len(SEEDS),
            "behavioral_episode_count": len(SEEDS) * 4,
            "valid_failures_included": True,
            "infrastructure_attempts_included": False,
            "missing_value_imputation": "none",
        },
        "formulas": {
            "s": "signed_final_lateral_offset_m; positive is robot LEFT",
            "D[a,i]": "s[a,i,left] - s[a,i,right]",
            "B[a,i]": "(-s[a,i,right]) - s[a,i,left]",
            "I[i]": "B[position_mirrored,i] - B[control,i]",
            "J[i]": "D[position_mirrored,i] - D[control,i]",
        },
        "interpretation": {
            "positive_D": "the prompt change ordered endpoints LEFT-to-RIGHT",
            "positive_B": "requested-side depth is greater for RIGHT than LEFT",
            "positive_I": "the RIGHT-over-LEFT requested-depth contrast is larger after position reflection",
            "positive_J": "LEFT-to-RIGHT endpoint separation is larger after position reflection",
        },
        "s_by_condition": {
            _condition_key(arm, relation): _robust_summary(
                vectors[f"s:{arm}:{relation}"],
                label=f"s:{arm}:{relation}",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            for arm, relation in CONDITIONS
        },
        "D_by_arm": {
            arm: _robust_summary(
                vectors[f"D:{arm}"],
                label=f"D:{arm}",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            for arm in ARMS
        },
        "B_by_arm": {
            arm: _robust_summary(
                vectors[f"B:{arm}"],
                label=f"B:{arm}",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            for arm in ARMS
        },
        "I_position_reflection_interaction": _robust_summary(
            vectors["I"],
            label="I:position_reflection_interaction",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "J_redirection_interaction": _robust_summary(
            vectors["J"],
            label="J:redirection_interaction",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
    }

    success_secondary: dict[str, Any] = {
        "subset_id": COMPLETE_CASE_SUBSET_ID,
        "inclusion_rule": "all four cells satisfy the frozen requested-success predicate",
        "realized_matched_seed_count": len(success_complete),
        "included_seeds": [row["seed"] for row in success_complete],
        "failures_as_zero": False,
        "unmatched_successful_cells_used": False,
        "formulas": {
            "G[a,i]": "margin[a,i,right] - margin[a,i,left]",
            "interaction": "G[position_mirrored,i] - G[control,i]",
        },
    }
    if success_complete:
        success_secondary["G_by_arm"] = {
            arm: _robust_summary(
                [row[f"G_{arm}_m"] for row in success_complete],
                label=f"G:{arm}:{COMPLETE_CASE_SUBSET_ID}",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            for arm in ARMS
        }
        success_secondary["G_position_reflection_interaction"] = _robust_summary(
            [row["G_position_reflection_interaction_m"] for row in success_complete],
            label=f"G:interaction:{COMPLETE_CASE_SUBSET_ID}",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        success_secondary["seed_level"] = success_complete
    else:
        success_secondary["status"] = "no_all_four_correct_complete_cases"

    all_records = [item["record"] for item in indexed.values()]
    return {
        "condition_outcomes": condition_counts,
        "failure_taxonomy_counts": dict(
            sorted(Counter(record["failure_taxonomy"] for record in all_records).items())
        ),
        "full_sample_primary": full_sample,
        "success_conditional_secondary": success_secondary,
        "seed_level": seed_rows,
    }


def _aggregate_manifest(
    *,
    filename: str,
    payload: bytes,
    row_count: int,
    record_schema: str,
    release: ReleaseBundle,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": AGGREGATE_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "release_manifest_sha256": release.manifest_sha256,
        "jsonl_path": filename,
        "jsonl_sha256": _sha256_bytes(payload),
        "jsonl_bytes": len(payload),
        "row_count": row_count,
        "record_schema_versions": [record_schema],
        "source_batches": list(sources),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with Path(path).open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise AggregateCompilationError(f"refusing to overwrite retained aggregate: {path}") from exc


def compile_nano_v3b001_results(
    *,
    release_manifest: Path,
    release_manifest_sha256: str,
    behavioral_jsonls: Sequence[Path],
    output_directory: Path,
    infrastructure_jsonls: Sequence[Path] = (),
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Validate, analyze, and write one deterministic V3-B001 result slice."""

    if bootstrap_replicates < 1 or type(bootstrap_seed) is not int:
        raise ValueError("bootstrap configuration must use positive replicates and an integer seed")
    try:
        release = load_release_bundle(
            Path(release_manifest),
            expected_manifest_sha256=release_manifest_sha256,
        )
    except RuntimeContractError as exc:
        raise AggregateCompilationError(str(exc)) from exc

    behavioral_paths = [Path(path).resolve() for path in behavioral_jsonls]
    if len(behavioral_paths) != 108:
        _fail(f"expected exactly 108 behavioral cell JSONLs, received {len(behavioral_paths)}")
    indexed: dict[str, dict[str, Any]] = {}
    behavioral_sources: list[dict[str, Any]] = []
    for path in behavioral_paths:
        rows, source = _load_batch(path, expected_schema=BEHAVIORAL_SCHEMA_VERSION)
        if len(rows) != 1:
            _fail(f"each released behavioral cell JSONL must contain exactly one row: {path}")
        item = _validate_behavioral_cell(rows[0], release=release, source=source)
        cell_id = item["record"]["registered_cell_id"]
        if cell_id in indexed:
            _fail(f"duplicate behavioral cell: {cell_id}")
        indexed[cell_id] = item
        behavioral_sources.append({"registered_cell_id": cell_id, **source})
    expected_ids = set(release.by_cell_id)
    observed_ids = set(indexed)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        _fail(f"behavioral cell set is incomplete or extraneous; missing={missing}, extra={extra}")
    runtime_hashes = {
        item["record"]["runtime_identity"]["sha256"] for item in indexed.values()
    }
    if len(runtime_hashes) != 1:
        _fail("the 108 behavioral cells do not share one exact runtime identity")
    for seed in SEEDS:
        for arm in ARMS:
            left = indexed[f"v3b001:nano:seed{seed}:{arm}:left"]["record"]
            right = indexed[f"v3b001:nano:seed{seed}:{arm}:right"]["record"]
            if left["initial_state_sha256"] != right["initial_state_sha256"]:
                _fail(f"seed {seed} {arm} LEFT/RIGHT physical reset mismatch")

    infrastructure_records: list[dict[str, Any]] = []
    infrastructure_sources: list[dict[str, Any]] = []
    attempt_ids: set[str] = set()
    for path in (Path(value).resolve() for value in infrastructure_jsonls):
        rows, source = _load_batch(path, expected_schema=INFRASTRUCTURE_SCHEMA_VERSION)
        for row in rows:
            normalized = _validate_infrastructure_attempt(row, release=release)
            attempt_id = normalized["attempt_id"]
            if attempt_id in attempt_ids:
                _fail(f"duplicate infrastructure attempt: {attempt_id}")
            attempt_ids.add(attempt_id)
            infrastructure_records.append(normalized)
        infrastructure_sources.append({"attempt_count": len(rows), **source})

    analysis = _aggregate_analysis(
        indexed,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    ordered_items = sorted(
        indexed.values(),
        key=lambda item: (
            item["cell"].seed,
            item["cell"].row["execution_order_index_within_seed"],
        ),
    )
    episode_payload = b"".join(_canonical_json(item["record"]) for item in ordered_items)
    behavioral_sources.sort(key=lambda item: item["registered_cell_id"])
    episode_manifest = _aggregate_manifest(
        filename=EPISODE_FILENAME,
        payload=episode_payload,
        row_count=len(ordered_items),
        record_schema=BEHAVIORAL_SCHEMA_VERSION,
        release=release,
        sources=behavioral_sources,
    )
    episode_manifest_payload = _canonical_json(episode_manifest)

    infrastructure_payload: bytes | None = None
    infrastructure_manifest: dict[str, Any] | None = None
    infrastructure_manifest_payload: bytes | None = None
    if infrastructure_records:
        infrastructure_records.sort(
            key=lambda row: (row["registered_cell_id"], row["attempt_id"])
        )
        infrastructure_sources.sort(key=lambda item: item["jsonl"]["path"])
        infrastructure_payload = b"".join(
            _canonical_json(record) for record in infrastructure_records
        )
        infrastructure_manifest = _aggregate_manifest(
            filename=INFRASTRUCTURE_FILENAME,
            payload=infrastructure_payload,
            row_count=len(infrastructure_records),
            record_schema=INFRASTRUCTURE_SCHEMA_VERSION,
            release=release,
            sources=infrastructure_sources,
        )
        infrastructure_manifest_payload = _canonical_json(infrastructure_manifest)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "claim_boundary": release.amendment["analysis_plan"]["claim_boundary"],
        "exact_prompts": PROMPTS,
        "release": {
            "manifest_path": str(Path(release_manifest).resolve()),
            "manifest_sha256": release.manifest_sha256,
            "amendment_sha256": release.amendment_sha256,
            "cells_sha256": release.cells_sha256,
        },
        "runtime_identity_sha256": next(iter(runtime_hashes)),
        "behavioral_evidence": {
            "valid_episode_count": len(ordered_items),
            "matched_seed_count": len(SEEDS),
            "aggregate_jsonl": {
                "path": EPISODE_FILENAME,
                "sha256": episode_manifest["jsonl_sha256"],
                "bytes": episode_manifest["jsonl_bytes"],
                "manifest_path": EPISODE_FILENAME + ".manifest.json",
                "manifest_sha256": _sha256_bytes(episode_manifest_payload),
            },
        },
        "infrastructure_evidence": {
            "provided_attempt_count": len(infrastructure_records),
            "included_in_behavioral_denominator": False,
            "aggregate_jsonl": (
                None
                if infrastructure_manifest is None
                else {
                    "path": INFRASTRUCTURE_FILENAME,
                    "sha256": infrastructure_manifest["jsonl_sha256"],
                    "bytes": infrastructure_manifest["jsonl_bytes"],
                    "manifest_path": INFRASTRUCTURE_FILENAME + ".manifest.json",
                    "manifest_sha256": _sha256_bytes(infrastructure_manifest_payload or b""),
                }
            ),
        },
        "uncertainty_contract": {
            "unit": "matched_seed",
            "bootstrap": "deterministic paired nonparametric percentile",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_master_seed": bootstrap_seed,
            "robust_test": "exact two-sided paired sign test with zero ties excluded",
            "median_interval": "exact distribution-free order-statistic interval",
            "multiplicity_adjustment": "none; estimands are prespecified and reported separately",
        },
        **analysis,
    }
    summary_payload = _canonical_json(summary)

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "episodes": output_directory / EPISODE_FILENAME,
        "episodes_manifest": output_directory / (EPISODE_FILENAME + ".manifest.json"),
        "summary": output_directory / SUMMARY_FILENAME,
    }
    if infrastructure_payload is not None:
        outputs["infrastructure"] = output_directory / INFRASTRUCTURE_FILENAME
        outputs["infrastructure_manifest"] = output_directory / (
            INFRASTRUCTURE_FILENAME + ".manifest.json"
        )
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        _fail(f"refusing to overwrite retained aggregate outputs: {existing}")
    _write_exclusive(outputs["episodes"], episode_payload)
    _write_exclusive(outputs["episodes_manifest"], episode_manifest_payload)
    if infrastructure_payload is not None and infrastructure_manifest_payload is not None:
        _write_exclusive(outputs["infrastructure"], infrastructure_payload)
        _write_exclusive(outputs["infrastructure_manifest"], infrastructure_manifest_payload)
    _write_exclusive(outputs["summary"], summary_payload)
    return outputs


def _discover(roots: Iterable[Path], name: str) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            _fail(f"discovery root is not a directory: {root}")
        found.update(path.resolve() for path in root.rglob(name) if path.is_file())
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--behavioral-root", type=Path, action="append", default=[])
    parser.add_argument("--behavioral-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--infrastructure-root", type=Path, action="append", default=[])
    parser.add_argument("--infrastructure-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    behavioral = sorted(
        set(path.resolve() for path in args.behavioral_jsonl)
        | set(_discover(args.behavioral_root, "raw_episode.jsonl"))
    )
    infrastructure = sorted(
        set(path.resolve() for path in args.infrastructure_jsonl)
        | set(_discover(args.infrastructure_root, "infrastructure_attempts.jsonl"))
    )
    outputs = compile_nano_v3b001_results(
        release_manifest=args.release_manifest,
        release_manifest_sha256=args.release_manifest_sha256,
        behavioral_jsonls=behavioral,
        infrastructure_jsonls=infrastructure,
        output_directory=args.output_directory,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in sorted(outputs.items())},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
