#!/usr/bin/env python3
"""Fail-closed aggregate compiler for the pi0.5 V3-B002 reflection ablation.

The compiler requires the exact 108-cell hash-bound release, emits a raw
episode JSONL with registered diagnostics, and emits pair-derived quantities
in a separate 54-row JSONL.  Infrastructure/partial attempts remain in a
separate stream and never enter any behavioral denominator.
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

import numpy as np

from experiments.v3.pi05_phase_b.contract import (
    AMENDMENT_ID,
    ARMS,
    CHECKPOINT_MANIFEST_SHA256,
    MODEL_ID,
    OPENPI_CONFIG,
    PROMPTS,
    RELATIONS,
    SEEDS,
    STUDY_ID,
    ContractError,
    ReleaseBundle,
    load_release_bundle,
    sha256_file,
)
from experiments.v3.pi05_phase_b.diagnostics import (
    EPISODE_DIAGNOSTICS_FIELD,
    PAIR_DIAGNOSTICS_SCHEMA,
    DiagnosticError,
    attach_episode_diagnostics,
    derive_pair_diagnostics,
)
from tools.vla_wam_v3_episode_schema import (
    BEHAVIORAL_SCHEMA_VERSION,
    INFRASTRUCTURE_SCHEMA_VERSION,
    validate_raw_episode_record,
)


REPORT_SCHEMA = "vla-wam-shared-v3b-pi05-reflection-results-v1"
OUTPUT_MANIFEST_SCHEMA = "vla-wam-shared-v3b-pi05-reflection-output-manifest-v1"
AGGREGATE_MANIFEST_SCHEMA = "vla-wam-shared-v3b-pi05-reflection-jsonl-manifest-v1"
BATCH_MANIFEST_SCHEMA = "vla-wam-shared-v3-jsonl-batch-manifest-v1"
EPISODE_FILENAME = "pi05_v3b002_episodes.jsonl"
PAIR_FILENAME = "pi05_v3b002_pairs.jsonl"
INFRASTRUCTURE_FILENAME = "pi05_v3b002_infrastructure_attempts.jsonl"
REPORT_FILENAME = "pi05_v3b002_report.json"
OUTPUT_MANIFEST_FILENAME = "pi05_v3b002_output_manifest.json"
CONDITIONS = tuple((arm, relation) for arm in ARMS for relation in RELATIONS)
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_SEED = 3_104_159


class CompilationError(RuntimeError):
    """Raised when retained evidence cannot support the registered analysis."""


def _fail(message: str) -> None:
    raise CompilationError(message)


def canonical_json_bytes(value: Any) -> bytes:
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
        raise CompilationError(f"result is not finite canonical JSON: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompilationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    value = Path(path).resolve()
    if not value.is_file() or value.stat().st_size <= 0:
        _fail(f"missing or empty evidence file: {value}")
    return {"path": str(value), "sha256": sha256_file(value), "bytes": value.stat().st_size}


def _load_batch(path: Path, expected_schema: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one JSONL only after verifying its post-close manifest."""

    path = Path(path).resolve()
    source = _file_record(path)
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest_source = _file_record(manifest_path)
    manifest = _load_json(manifest_path, f"batch manifest {manifest_path}")
    expected = {
        "schema_version": BATCH_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "jsonl_sha256": source["sha256"],
        "jsonl_bytes": source["bytes"],
        "record_schema_versions": [expected_schema],
    }
    for key, wanted in expected.items():
        if manifest.get(key) != wanted:
            _fail(f"batch manifest mismatch for {key}: {path}")
    declared = manifest.get("jsonl_path")
    if not isinstance(declared, str) or Path(declared).resolve() != path:
        _fail(f"batch manifest does not bind its exact JSONL: {path}")
    lines = path.read_bytes().splitlines()
    if not lines or any(not line.strip() for line in lines):
        _fail(f"JSONL contains an empty row: {path}")
    if manifest.get("row_count") != len(lines):
        _fail(f"batch manifest row count mismatch: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompilationError(f"cannot parse {path}:{number}: {exc}") from exc
        if not isinstance(raw, dict):
            _fail(f"JSONL row is not an object: {path}:{number}")
        try:
            normalized = validate_raw_episode_record(raw)
        except Exception as exc:
            raise CompilationError(
                f"invalid raw record {path}:{number}: {type(exc).__name__}: {exc}"
            ) from exc
        artifacts = normalized.get("artifacts", {})
        raw_result = artifacts.get("raw_result_jsonl", {}) if isinstance(artifacts, dict) else {}
        raw_path = raw_result.get("path") if isinstance(raw_result, dict) else None
        if not isinstance(raw_path, str) or Path(raw_path).resolve() != path:
            _fail(f"record does not bind its exact containing JSONL: {path}:{number}")
        rows.append(normalized)
    return rows, {"jsonl": source, "batch_manifest": manifest_source}


def _validate_behavioral(
    record: Mapping[str, Any], *, release: ReleaseBundle, source: Mapping[str, Any]
) -> dict[str, Any]:
    if record.get("record_type") != "behavioral_episode" or record.get("behavioral_result_valid") is not True:
        _fail("behavioral inputs may contain only valid behavioral episodes")
    cell_id = record.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail("behavioral record lacks registered_cell_id")
    try:
        cell = release.cell(cell_id)
    except ContractError as exc:
        raise CompilationError(str(exc)) from exc
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
        "future_interface": "actions_only",
        "missing_future_policy": cell.row["missing_future_policy"],
    }
    for key, wanted in expected.items():
        if record.get(key) != wanted:
            _fail(f"{cell_id} does not match released {key}")
    checkpoint = record.get("checkpoint")
    if checkpoint != {
        "id": OPENPI_CONFIG,
        "revision": f"v2a010-manifest-{CHECKPOINT_MANIFEST_SHA256}",
    }:
        _fail(f"{cell_id} does not match the exact historical pi0.5 checkpoint")
    runtime = record.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        _fail(f"{cell_id} lacks runtime identity")
    digest = runtime.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        _fail(f"{cell_id} runtime identity digest is invalid")
    try:
        enriched = attach_episode_diagnostics(record)
    except DiagnosticError as exc:
        raise CompilationError(f"{cell_id}: {exc}") from exc
    return {"cell": cell, "record": enriched, "source": dict(source)}


def _load_executed_actions(record: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    artifacts = record.get("artifacts")
    trace = artifacts.get("executed_action_trace") if isinstance(artifacts, Mapping) else None
    if not isinstance(trace, Mapping):
        _fail("behavioral record lacks executed_action_trace")
    path_value = trace.get("path")
    if not isinstance(path_value, str):
        _fail("executed_action_trace path is missing")
    source = _file_record(Path(path_value))
    if trace.get("sha256") != source["sha256"] or trace.get("bytes") != source["bytes"]:
        _fail("executed_action_trace hash/size binding changed")
    try:
        array = np.load(source["path"], allow_pickle=False)
    except Exception as exc:
        raise CompilationError(f"cannot load executed action trace {source['path']}: {exc}") from exc
    if array.ndim != 2 or array.shape[1] != 8 or array.shape[0] < 1:
        _fail("executed action trace must have shape [N,8]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        _fail("executed action trace must be finite numeric evidence")
    return array, source


def _percentile(ordered: Sequence[float], probability: float) -> float:
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)


def exact_sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ties = len(values) - positive - negative
    effective = positive + negative
    if effective == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(effective, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**effective))
    return {
        "method": "exact_two_sided_paired_sign_test",
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "ties_excluded": ties,
        "effective_n": effective,
        "p_value": p_value,
    }


def _bootstrap_interval(
    values: Sequence[float], *, label: str, statistic: str, replicates: int, master_seed: int
) -> dict[str, Any]:
    if replicates < 10_000:
        raise ValueError("registered analysis requires at least 10,000 bootstrap resamples")
    seed = int.from_bytes(
        hashlib.sha256(f"{master_seed}:{label}:{statistic}".encode()).digest()[:8], "big"
    )
    rng = random.Random(seed)
    finite = [float(value) for value in values]
    samples: list[float] = []
    for _ in range(replicates):
        resample = [finite[rng.randrange(len(finite))] for _ in finite]
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
        "seed": seed,
        "lower": _percentile(samples, 0.025),
        "upper": _percentile(samples, 0.975),
    }


def _exact_median_interval(values: Sequence[float], confidence: float = 0.95) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    chosen: tuple[int, float] | None = None
    for k in range(1, n // 2 + 2):
        coverage = 1.0 - 2.0 * sum(math.comb(n, index) for index in range(k)) / (2**n)
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


def continuous_summary(
    values: Sequence[float], *, label: str, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite or not all(math.isfinite(value) for value in finite):
        _fail(f"{label} requires finite values")
    return {
        "n": len(finite),
        "mean_m": statistics.fmean(finite),
        "mean_bootstrap_95": _bootstrap_interval(
            finite,
            label=label,
            statistic="mean",
            replicates=bootstrap_replicates,
            master_seed=bootstrap_seed,
        ),
        "median_m": float(statistics.median(finite)),
        "median_bootstrap_95": _bootstrap_interval(
            finite,
            label=label,
            statistic="median",
            replicates=bootstrap_replicates,
            master_seed=bootstrap_seed,
        ),
        "median_exact_interval": _exact_median_interval(finite),
        "minimum_m": min(finite),
        "maximum_m": max(finite),
        "sample_standard_deviation_m": statistics.stdev(finite) if len(finite) > 1 else None,
        "paired_sign_test": exact_sign_test(finite),
    }


def exact_layout_swap_permutation(values: Sequence[int]) -> dict[str, Any]:
    """Exact two-sided layout-label permutation using ``abs(sum(DiD))``.

    Swapping control and reflected labels negates a seed's DiD.  Dynamic
    programming counts all 2**n registered within-seed swaps exactly, including
    the two indistinguishable label assignments for a zero DiD.
    """

    if not values or any(type(value) is not int or value not in {-2, -1, 0, 1, 2} for value in values):
        raise ValueError("binary success DiD values must lie in {-2,-1,0,1,2}")
    distribution: Counter[int] = Counter({0: 1})
    for value in values:
        next_distribution: Counter[int] = Counter()
        for total, count in distribution.items():
            next_distribution[total + value] += count
            next_distribution[total - value] += count
        distribution = next_distribution
    observed = sum(values)
    extreme = sum(count for total, count in distribution.items() if abs(total) >= abs(observed))
    total_permutations = 2 ** len(values)
    if sum(distribution.values()) != total_permutations:
        _fail("internal exact-permutation count mismatch")
    return {
        "method": "exact_two_sided_within_seed_control_reflected_label_permutation",
        "test_statistic": "absolute_sum_of_per_seed_success_DiD",
        "observed_signed_sum": observed,
        "observed_absolute_sum": abs(observed),
        "registered_seed_count": len(values),
        "total_permutations": total_permutations,
        "extreme_permutations": extreme,
        "p_value": extreme / total_permutations,
    }


def analyze_pairs(
    pairs: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    """Analyze exactly 54 pair records / 27 complete four-cell blocks."""

    if len(pairs) != 54:
        _fail(f"expected exactly 54 LEFT/RIGHT pair rows, received {len(pairs)}")
    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for pair in pairs:
        key = (pair.get("seed"), pair.get("arm"))
        if key in indexed:
            _fail(f"duplicate pair row: {key}")
        indexed[key] = pair
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(indexed) != expected:
        _fail("pair rows do not form the exact 27-seed x 2-layout design")

    d_by_arm = {arm: [] for arm in ARMS}
    b_by_arm = {arm: [] for arm in ARMS}
    j_values: list[float] = []
    i_values: list[float] = []
    did_values: list[int] = []
    seed_rows: list[dict[str, Any]] = []
    cell_successes = {(arm, relation): 0 for arm, relation in CONDITIONS}
    for seed in SEEDS:
        control = indexed[(seed, "control")]
        reflected = indexed[(seed, "position_mirrored")]
        for arm, pair in (("control", control), ("position_mirrored", reflected)):
            d_by_arm[arm].append(float(pair["endpoint_redirection_D_m"]))
            b_by_arm[arm].append(float(pair["requested_side_depth_contrast_B_m"]))
            cell_successes[(arm, "left")] += int(bool(pair["left_success"]))
            cell_successes[(arm, "right")] += int(bool(pair["right_success"]))
        interaction_j = float(reflected["endpoint_redirection_D_m"]) - float(control["endpoint_redirection_D_m"])
        interaction_i = float(reflected["requested_side_depth_contrast_B_m"]) - float(control["requested_side_depth_contrast_B_m"])
        control_gap = int(bool(control["right_success"])) - int(bool(control["left_success"]))
        reflected_gap = int(bool(reflected["right_success"])) - int(bool(reflected["left_success"]))
        did = reflected_gap - control_gap
        j_values.append(interaction_j)
        i_values.append(interaction_i)
        did_values.append(did)
        seed_rows.append(
            {
                "seed": seed,
                "D_control_m": float(control["endpoint_redirection_D_m"]),
                "D_position_mirrored_m": float(reflected["endpoint_redirection_D_m"]),
                "J_redirection_interaction_m": interaction_j,
                "B_control_m": float(control["requested_side_depth_contrast_B_m"]),
                "B_position_mirrored_m": float(reflected["requested_side_depth_contrast_B_m"]),
                "I_requested_side_depth_interaction_m": interaction_i,
                "control_right_minus_left_success": control_gap,
                "position_mirrored_right_minus_left_success": reflected_gap,
                "binary_success_DiD": did,
            }
        )

    condition_table = {
        arm: {
            relation: {
                "successes": cell_successes[(arm, relation)],
                "episodes": len(SEEDS),
                "failures": len(SEEDS) - cell_successes[(arm, relation)],
            }
            for relation in RELATIONS
        }
        for arm in ARMS
    }
    did_counter = Counter(did_values)
    return {
        "population": {
            "matched_seed_count": len(SEEDS),
            "behavioral_episode_count": 108,
            "matched_left_right_pair_count": 54,
            "valid_behavioral_failures_included": True,
            "infrastructure_attempts_included": False,
            "missing_value_imputation": "none",
        },
        "formulas": {
            "signed_offset": "s; positive robot-base Y is robot LEFT",
            "H1_D[layout,seed]": "s_LEFT - s_RIGHT",
            "H1_J[seed]": "D_position_mirrored - D_control",
            "H2_B[layout,seed]": "requested_depth_RIGHT - requested_depth_LEFT = -s_RIGHT - s_LEFT",
            "H2_I[seed]": "B_position_mirrored - B_control",
            "H3_DiD[seed]": "(success_RIGHT-success_LEFT)_position_mirrored - (success_RIGHT-success_LEFT)_control",
        },
        "H1_endpoint_redirection": {
            "paired_contrast_by_layout": {
                arm: continuous_summary(
                    d_by_arm[arm],
                    label=f"H1:D:{arm}",
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed,
                )
                for arm in ARMS
            },
            "reflected_minus_control_interaction": continuous_summary(
                j_values,
                label="H1:J:position_mirrored_minus_control",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ),
        },
        "H2_requested_side_depth": {
            "paired_contrast_by_layout": {
                arm: continuous_summary(
                    b_by_arm[arm],
                    label=f"H2:B:{arm}",
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed,
                )
                for arm in ARMS
            },
            "reflected_minus_control_interaction": continuous_summary(
                i_values,
                label="H2:I:position_mirrored_minus_control",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ),
        },
        "H3_binary_success": {
            "cell_success_table_2x2": condition_table,
            "per_seed_DiD_distribution": {
                str(value): did_counter.get(value, 0) for value in (-2, -1, 0, 1, 2)
            },
            "mean_DiD": statistics.fmean(did_values),
            "median_DiD": float(statistics.median(did_values)),
            "exact_permutation_test": exact_layout_swap_permutation(did_values),
        },
        "seed_level": seed_rows,
    }


def _aggregate_manifest(
    *, payload: bytes, filename: str, schema: str, row_count: int, release: ReleaseBundle, sources: Sequence[Mapping[str, Any]]
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
        "record_schema_versions": [schema],
        "source_batches": list(sources),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with Path(path).open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise CompilationError(f"refusing to overwrite retained output: {path}") from exc


def compile_pi05_v3b002_results(
    *,
    repo_root: Path,
    release_manifest: Path,
    release_manifest_sha256: str,
    behavioral_jsonls: Sequence[Path],
    output_directory: Path,
    infrastructure_jsonls: Sequence[Path] = (),
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Validate, analyze, hash, and retain one complete V3-B002 result."""

    if bootstrap_replicates < 10_000 or type(bootstrap_seed) is not int:
        raise ValueError("B002 requires >=10,000 bootstrap resamples and an integer seed")
    try:
        release = load_release_bundle(
            Path(repo_root), Path(release_manifest), expected_manifest_sha256=release_manifest_sha256
        )
    except ContractError as exc:
        raise CompilationError(str(exc)) from exc
    paths = [Path(path).resolve() for path in behavioral_jsonls]
    if len(paths) != 108:
        _fail(f"expected exactly 108 behavioral cell JSONLs, received {len(paths)}")
    indexed: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        rows, source = _load_batch(path, BEHAVIORAL_SCHEMA_VERSION)
        if len(rows) != 1:
            _fail(f"each cell JSONL must contain exactly one behavioral row: {path}")
        item = _validate_behavioral(rows[0], release=release, source=source)
        cell_id = item["cell"].cell_id
        if cell_id in indexed:
            _fail(f"duplicate behavioral cell: {cell_id}")
        indexed[cell_id] = item
        sources.append({"registered_cell_id": cell_id, **source})
    if set(indexed) != set(release.by_cell_id):
        _fail("behavioral cell set is incomplete or contains unreleased cells")
    runtime_hashes = {item["record"]["runtime_identity"]["sha256"] for item in indexed.values()}
    if len(runtime_hashes) != 1:
        _fail("all 108 cells must share one exact runtime identity")

    pair_rows: list[dict[str, Any]] = []
    action_sources: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            left = indexed[f"v3b002:pi05:seed{seed}:{arm}:left"]["record"]
            right = indexed[f"v3b002:pi05:seed{seed}:{arm}:right"]["record"]
            left_actions, left_source = _load_executed_actions(left)
            right_actions, right_source = _load_executed_actions(right)
            try:
                pair = derive_pair_diagnostics(
                    seed=seed,
                    arm=arm,
                    left_record=left,
                    right_record=right,
                    left_actions=left_actions,
                    right_actions=right_actions,
                )
            except DiagnosticError as exc:
                raise CompilationError(f"seed {seed} {arm}: {exc}") from exc
            pair["left_raw_episode_jsonl_sha256"] = indexed[left["registered_cell_id"]]["source"]["jsonl"]["sha256"]
            pair["right_raw_episode_jsonl_sha256"] = indexed[right["registered_cell_id"]]["source"]["jsonl"]["sha256"]
            pair["left_executed_action_trace_sha256"] = left_source["sha256"]
            pair["right_executed_action_trace_sha256"] = right_source["sha256"]
            pair_rows.append(pair)
            action_sources.extend(
                [
                    {"registered_cell_id": left["registered_cell_id"], **left_source},
                    {"registered_cell_id": right["registered_cell_id"], **right_source},
                ]
            )

    analysis = analyze_pairs(
        pair_rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    ordered = sorted(
        indexed.values(),
        key=lambda item: (item["cell"].seed, item["cell"].row["execution_order_index_within_seed"]),
    )
    episode_payload = b"".join(canonical_json_bytes(item["record"]) for item in ordered)
    pair_rows.sort(key=lambda row: (row["seed"], ARMS.index(row["arm"])))
    pair_payload = b"".join(canonical_json_bytes(row) for row in pair_rows)
    sources.sort(key=lambda row: row["registered_cell_id"])
    action_sources.sort(key=lambda row: row["registered_cell_id"])
    episode_manifest = _aggregate_manifest(
        payload=episode_payload,
        filename=EPISODE_FILENAME,
        schema=BEHAVIORAL_SCHEMA_VERSION,
        row_count=108,
        release=release,
        sources=sources,
    )
    pair_manifest = _aggregate_manifest(
        payload=pair_payload,
        filename=PAIR_FILENAME,
        schema=PAIR_DIAGNOSTICS_SCHEMA,
        row_count=54,
        release=release,
        sources=action_sources,
    )

    infrastructure_records: list[dict[str, Any]] = []
    infrastructure_sources: list[dict[str, Any]] = []
    attempt_ids: set[str] = set()
    for path in (Path(value).resolve() for value in infrastructure_jsonls):
        rows, source = _load_batch(path, INFRASTRUCTURE_SCHEMA_VERSION)
        for row in rows:
            if row.get("record_type") != "infrastructure_attempt" or row.get("behavioral_result_valid") is not False:
                _fail("infrastructure inputs may contain only nonbehavioral attempts")
            cell_id = row.get("registered_cell_id")
            if not isinstance(cell_id, str) or cell_id not in release.by_cell_id:
                _fail("infrastructure attempt references an unreleased cell")
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or attempt_id in attempt_ids:
                _fail("infrastructure attempt IDs must be unique")
            attempt_ids.add(attempt_id)
            infrastructure_records.append(row)
        infrastructure_sources.append({"attempt_count": len(rows), **source})

    episode_manifest_payload = canonical_json_bytes(episode_manifest)
    pair_manifest_payload = canonical_json_bytes(pair_manifest)
    infrastructure_payload: bytes | None = None
    infrastructure_manifest_payload: bytes | None = None
    if infrastructure_records:
        infrastructure_records.sort(key=lambda row: (row["registered_cell_id"], row["attempt_id"]))
        infrastructure_payload = b"".join(canonical_json_bytes(row) for row in infrastructure_records)
        infrastructure_manifest = _aggregate_manifest(
            payload=infrastructure_payload,
            filename=INFRASTRUCTURE_FILENAME,
            schema=INFRASTRUCTURE_SCHEMA_VERSION,
            row_count=len(infrastructure_records),
            release=release,
            sources=infrastructure_sources,
        )
        infrastructure_manifest_payload = canonical_json_bytes(infrastructure_manifest)

    failure_counts = Counter(
        item["record"][EPISODE_DIAGNOSTICS_FIELD]["failure_category"] for item in indexed.values()
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "exact_prompts": PROMPTS,
        "prospective_registration": {
            "registered_predictions": release.amendment.get("registered_predictions"),
            "known_results_disclosure": release.amendment.get("known_results_disclosure"),
            "analysis_plan": release.amendment.get("analysis_plan"),
        },
        "release": {
            "manifest_path": str(Path(release_manifest).resolve()),
            "manifest_sha256": release.manifest_sha256,
            "amendment_sha256": release.amendment_sha256,
            "cells_sha256": release.cells_sha256,
        },
        "runtime_identity_sha256": next(iter(runtime_hashes)),
        "behavioral_evidence": {
            "episode_count": 108,
            "pair_count": 54,
            "matched_seed_count": 27,
            "valid_failures_retained": True,
            "failure_category_counts": dict(sorted(failure_counts.items())),
            "episode_jsonl": {
                "path": EPISODE_FILENAME,
                "sha256": episode_manifest["jsonl_sha256"],
                "bytes": episode_manifest["jsonl_bytes"],
                "manifest_path": EPISODE_FILENAME + ".manifest.json",
                "manifest_sha256": _sha256_bytes(episode_manifest_payload),
            },
            "pair_jsonl": {
                "path": PAIR_FILENAME,
                "sha256": pair_manifest["jsonl_sha256"],
                "bytes": pair_manifest["jsonl_bytes"],
                "manifest_path": PAIR_FILENAME + ".manifest.json",
                "manifest_sha256": _sha256_bytes(pair_manifest_payload),
            },
        },
        "infrastructure_evidence": {
            "attempt_count": len(infrastructure_records),
            "included_in_behavioral_denominators": False,
        },
        "uncertainty_contract": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_master_seed": bootstrap_seed,
            "bootstrap_unit": "matched_seed",
            "continuous_test": "exact two-sided paired sign test; zero ties excluded",
            "binary_test": "exact within-seed control/reflected layout-label permutation",
            "multiplicity_adjustment": "none; H1/H2/H3 were prospectively registered",
        },
        "analysis": analysis,
    }
    report_payload = canonical_json_bytes(report)

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "episodes": output_directory / EPISODE_FILENAME,
        "episodes_manifest": output_directory / (EPISODE_FILENAME + ".manifest.json"),
        "pairs": output_directory / PAIR_FILENAME,
        "pairs_manifest": output_directory / (PAIR_FILENAME + ".manifest.json"),
        "report": output_directory / REPORT_FILENAME,
        "output_manifest": output_directory / OUTPUT_MANIFEST_FILENAME,
    }
    if infrastructure_payload is not None:
        outputs["infrastructure"] = output_directory / INFRASTRUCTURE_FILENAME
        outputs["infrastructure_manifest"] = output_directory / (INFRASTRUCTURE_FILENAME + ".manifest.json")
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        _fail(f"refusing to overwrite retained outputs: {existing}")
    _write_exclusive(outputs["episodes"], episode_payload)
    _write_exclusive(outputs["episodes_manifest"], episode_manifest_payload)
    _write_exclusive(outputs["pairs"], pair_payload)
    _write_exclusive(outputs["pairs_manifest"], pair_manifest_payload)
    if infrastructure_payload is not None and infrastructure_manifest_payload is not None:
        _write_exclusive(outputs["infrastructure"], infrastructure_payload)
        _write_exclusive(outputs["infrastructure_manifest"], infrastructure_manifest_payload)
    _write_exclusive(outputs["report"], report_payload)
    output_manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "release_manifest_sha256": release.manifest_sha256,
        "files": {
            name: _file_record(path)
            for name, path in sorted(outputs.items())
            if name != "output_manifest"
        },
    }
    _write_exclusive(outputs["output_manifest"], canonical_json_bytes(output_manifest))
    return outputs


def _discover(roots: Iterable[Path], filename: str) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        path = Path(root).resolve()
        if not path.is_dir():
            _fail(f"discovery root is not a directory: {path}")
        found.update(candidate.resolve() for candidate in path.rglob(filename) if candidate.is_file())
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
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
    outputs = compile_pi05_v3b002_results(
        repo_root=args.repo_root,
        release_manifest=args.release_manifest,
        release_manifest_sha256=args.release_manifest_sha256,
        behavioral_jsonls=behavioral,
        infrastructure_jsonls=infrastructure,
        output_directory=args.output_directory,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({name: str(path) for name, path in sorted(outputs.items())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
