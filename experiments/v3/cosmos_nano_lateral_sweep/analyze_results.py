#!/usr/bin/env python3
"""Registered matched-seed analysis for the Nano V3-B005 dose response.

This module consumes the 105 LEFT/RIGHT pair records and the 210 validated
behavioral episode records.  It performs no imputation: missing or duplicate
cells fail closed before any statistic is emitted.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.v3.cosmos_nano_lateral_sweep.compile_pair import SCHEMA as PAIR_SCHEMA
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    AMENDMENT_ID,
    LEVELS,
    MODEL_ID,
    SEEDS,
    STUDY_ID,
    canonical_json_bytes,
    sha256_file,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


REPORT_SCHEMA = "vla-wam-shared-v3b005-nano-dose-response-report-v1"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 3_104_161
FAILURE_CATEGORIES = ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")


class AnalysisError(RuntimeError):
    """Raised when retained evidence cannot support the registered analysis."""


def _fail(message: str) -> None:
    raise AnalysisError(message)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} must be finite")
    return float(value)


def _slope(y: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(LEVELS, dtype=np.float64)
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (len(LEVELS),) or not np.isfinite(values).all():
        _fail("each seed requires seven finite ordered observations")
    centered = x - float(x.mean())
    slope = float(np.dot(centered, values - values.mean()) / np.dot(centered, centered))
    return slope, float(values.mean())


def _exact_sign(values: Sequence[float]) -> dict[str, Any]:
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    ties = len(values) - positives - negatives
    n = positives + negatives
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(positives, negatives) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {
        "positive": positives,
        "negative": negatives,
        "ties": ties,
        "non_tied_n": n,
        "two_sided_p": p,
        "method": "exact_two_sided_paired_sign_test_zero_ties_excluded",
    }


def _bootstrap_mean(values: Sequence[float], *, seed: int, replicates: int) -> dict[str, Any]:
    if replicates < 10_000:
        _fail("registered analysis requires at least 10,000 bootstrap resamples")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        _fail("bootstrap input must be a non-empty finite vector")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(array), size=(replicates, len(array)))
    means = array[draws].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(array.mean()),
        "ci95": [float(low), float(high)],
        "replicates": replicates,
        "unit": "matched_seed",
        "method": "nonparametric_percentile_bootstrap",
    }


def _wilson(successes: int, n: int) -> list[float]:
    if not 0 <= successes <= n or n <= 0:
        _fail("Wilson interval requires 0 <= successes <= n")
    z = 1.959963984540054
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _validate_pairs(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if row.get("schema_version") != PAIR_SCHEMA or row.get("model_id") != MODEL_ID:
            _fail("pair row has the wrong schema or model")
        seed, level = row.get("seed"), row.get("level_index")
        if seed not in SEEDS or type(level) is not int or not 0 <= level < len(LEVELS):
            _fail("pair row lies outside the frozen seed/level grid")
        key = (seed, level)
        if key in indexed:
            _fail(f"duplicate pair row: {key}")
        expected_y = LEVELS[level]
        if not math.isclose(
            _finite(row.get("reference_object_initial_lateral_position_y_m"), "bowl y"),
            expected_y,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(f"pair row changed the frozen level coordinate: {key}")
        for field in ("requested_side_depth_contrast_B_m", "endpoint_redirection_D_m"):
            _finite(row.get(field), f"{key}.{field}")
        if type(row.get("left_success")) is not bool or type(row.get("right_success")) is not bool:
            _fail(f"{key} success flags must be booleans")
        indexed[key] = row
    expected = {(seed, level) for seed in SEEDS for level in range(len(LEVELS))}
    if set(indexed) != expected:
        _fail("pair rows are not the exact 15-seed x 7-level design")
    return indexed


def _validate_episodes(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if (
            row.get("behavioral_result_valid") is not True
            or row.get("model_id") != MODEL_ID
            or row.get("amendment_id") != AMENDMENT_ID
        ):
            _fail("episode row is not a valid V3-B005 behavioral cell")
        seed, level, relation = row.get("environment_seed"), row.get("level_index"), row.get("requested_relation")
        if seed not in SEEDS or type(level) is not int or relation not in {"left", "right"}:
            _fail("episode row lies outside the frozen design")
        key = (seed, level, relation)
        if key in indexed:
            _fail(f"duplicate behavioral cell: {key}")
        taxonomy = row.get("failure_taxonomy")
        if taxonomy not in FAILURE_CATEGORIES:
            _fail(f"unknown failure taxonomy for {key}")
        if type(row.get("requested_success")) is not bool:
            _fail(f"success must be boolean for {key}")
        diagnostics = row.get("nano_v3b005_diagnostics")
        if not isinstance(diagnostics, Mapping):
            _fail(f"missing episode diagnostics for {key}")
        _finite(diagnostics.get("requested_side_depth_m"), f"{key}.requested_side_depth_m")
        indexed[key] = row
    expected = {
        (seed, level, relation)
        for seed in SEEDS
        for level in range(len(LEVELS))
        for relation in ("left", "right")
    }
    if set(indexed) != expected:
        _fail("episode rows are not the exact 210-cell design")
    return indexed


def analyze(
    pair_rows: Iterable[Mapping[str, Any]],
    episode_rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    pairs = _validate_pairs(pair_rows)
    episodes = _validate_episodes(episode_rows)
    primary_slopes: list[float] = []
    primary_centers: list[float] = []
    success_slopes: list[float] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        depth = [float(pairs[(seed, level)]["requested_side_depth_contrast_B_m"]) for level in range(7)]
        gaps = [
            int(pairs[(seed, level)]["right_success"]) - int(pairs[(seed, level)]["left_success"])
            for level in range(7)
        ]
        depth_slope, depth_center = _slope(depth)
        success_slope, success_center = _slope(gaps)
        primary_slopes.append(depth_slope)
        primary_centers.append(depth_center)
        success_slopes.append(success_slope)
        seed_rows.append({
            "seed": seed,
            "depth_contrast_by_level_m": depth,
            "depth_slope_m_per_m": depth_slope,
            "depth_at_center_m": depth_center,
            "success_gap_by_level": gaps,
            "success_gap_slope_per_m": success_slope,
            "success_gap_at_center": success_center,
        })

    level_rows: list[dict[str, Any]] = []
    taxonomy: dict[str, dict[str, dict[str, int]]] = {}
    for level, y_m in enumerate(LEVELS):
        b_values = [float(pairs[(seed, level)]["requested_side_depth_contrast_B_m"]) for seed in SEEDS]
        relation_rows: dict[str, Any] = {}
        taxonomy[str(level)] = {}
        for relation in ("left", "right"):
            cells = [episodes[(seed, level, relation)] for seed in SEEDS]
            successes = sum(bool(cell["requested_success"]) for cell in cells)
            counts = Counter(str(cell["failure_taxonomy"]) for cell in cells)
            taxonomy[str(level)][relation] = {name: counts.get(name, 0) for name in FAILURE_CATEGORIES}
            relation_rows[relation] = {
                "successes": successes,
                "episodes": len(SEEDS),
                "rate": successes / len(SEEDS),
                "wilson_95": _wilson(successes, len(SEEDS)),
            }
        both_success_depth = [
            float(pairs[(seed, level)]["requested_side_depth_contrast_B_m"])
            for seed in SEEDS
            if pairs[(seed, level)]["left_success"] and pairs[(seed, level)]["right_success"]
        ]
        level_rows.append({
            "level_index": level,
            "reference_object_initial_lateral_position_y_m": y_m,
            "depth_contrast_B_m": {
                **_bootstrap_mean(
                    b_values,
                    seed=bootstrap_seed + level,
                    replicates=bootstrap_replicates,
                ),
                "median": float(statistics.median(b_values)),
            },
            "binary_success": relation_rows,
            "both_directions_success_conditional_depth": {
                "realized_pair_n": len(both_success_depth),
                "values_m": both_success_depth,
                "mean_m": float(statistics.fmean(both_success_depth)) if both_success_depth else None,
                "missing_value_policy": "undefined_pairs_omitted_never_zero",
            },
        })

    slope_summary = _bootstrap_mean(
        primary_slopes,
        seed=bootstrap_seed + 100,
        replicates=bootstrap_replicates,
    )
    slope_summary.update({
        "median": float(statistics.median(primary_slopes)),
        "per_seed": primary_slopes,
        "sign_test": _exact_sign(primary_slopes),
    })
    mean_slope = slope_summary["mean"]
    mean_center = float(statistics.fmean(primary_centers))
    candidate = None if mean_slope == 0.0 else float(np.mean(LEVELS) - mean_center / mean_slope)
    in_support = candidate is not None and LEVELS[0] <= candidate <= LEVELS[-1]
    zero_crossing = {
        "reference_object_lateral_y_m": candidate if in_support else None,
        "in_registered_support": in_support,
        "registered_support_m": [LEVELS[0], LEVELS[-1]],
        "unclipped_linear_estimate_m": candidate,
        "policy": "report_only_when_linear_population_fit_crosses_inside_registered_support",
    }
    success_slope_summary = _bootstrap_mean(
        success_slopes,
        seed=bootstrap_seed + 200,
        replicates=bootstrap_replicates,
    )
    success_slope_summary.update({
        "median": float(statistics.median(success_slopes)),
        "per_seed": success_slopes,
        "sign_test": _exact_sign(success_slopes),
    })
    return {
        "schema_version": REPORT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "population": {
            "matched_seed_count": len(SEEDS),
            "level_count": len(LEVELS),
            "matched_pair_count": len(pairs),
            "behavioral_episode_count": len(episodes),
            "valid_behavioral_failures_included": True,
            "infrastructure_attempts_included": False,
            "missing_value_imputation": "none",
        },
        "primary_depth_dose_response": {
            "formula": "within each seed, OLS slope of B=(-s_RIGHT)-s_LEFT on seven bowl-y levels",
            "slope_m_per_m": slope_summary,
            "population_linear_zero_crossing": zero_crossing,
        },
        "binary_success_secondary": {
            "formula": "within each seed, OLS slope of success_RIGHT-success_LEFT on bowl-y",
            "paired_gap_slope_per_m": success_slope_summary,
        },
        "by_level": level_rows,
        "failure_taxonomy_counts": taxonomy,
        "seed_level": seed_rows,
        "uncertainty_contract": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_master_seed": bootstrap_seed,
            "bootstrap_unit": "matched_seed",
            "binary_intervals": "Wilson 95%",
            "slope_test": "exact two-sided sign test with zero slopes excluded",
        },
    }


def _load_pair(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        _fail(f"pair file is not an object: {path}")
    return value


def _load_episode(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        _fail(f"episode JSONL must contain exactly one row: {path}")
    return parse_jsonl_record(lines[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-json", type=Path, action="append", required=True)
    parser.add_argument("--episode-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = analyze(
        (_load_pair(path) for path in args.pair_json),
        (_load_episode(path) for path in args.episode_jsonl),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = canonical_json_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {
        "schema_version": "vla-wam-shared-v3b005-nano-dose-response-report-manifest-v1",
        "path": str(args.output.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "pair_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.pair_json
        ],
        "episode_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.episode_jsonl
        ],
    }
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(json.dumps({"report": str(args.output), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
