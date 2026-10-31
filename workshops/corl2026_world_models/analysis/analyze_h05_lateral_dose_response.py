#!/usr/bin/env python3
"""Authenticate and reproduce the retained H05 lateral dose-response report.

H05 is the completed historical V3-B005 study: 15 matched seed blocks, seven
prospectively fixed bowl-y positions, and LEFT/RIGHT commands at every position.
This script reads only immutable Git blobs.  It authenticates the frozen design,
registered report, 315-source manifest and final audit, then independently
recomputes every report-level slope, bootstrap interval, sign test, success
total and level summary available in the retained report.

The raw 105 pair and 210 episode files named by the source manifest are not
stored in this checkout, so this is report-level arithmetic reproduction rather
than a new raw-evidence audit.  It runs no model, robot, remote job or annotation
workflow and does not advance any forecast-ablation stage gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h05_lateral_dose_response.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h05_lateral_dose_response.csv"

BASE = "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005"
ANALYZER = "experiments/v3/cosmos_nano_lateral_sweep/analyze_results.py"
CELLS = f"{BASE}/nano_lateral_v3b005_cells.jsonl"
QUEUE_MANIFEST = f"{BASE}/nano_lateral_v3b005_manifest.json"
AMENDMENT = f"{BASE}/post_result_nano_lateral_sweep_v3b005_amendment.json"
REPORT = f"{BASE}/results/nano_v3b005_dose_response_report.json"
SOURCE_MANIFEST = REPORT + ".manifest.json"
FINAL_AUDIT = f"{BASE}/results/final_integrity_audit.json"

EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    ANALYZER: {
        "git_blob": "4920f7943cdb175c0f2280ba6299620fd67320d9",
        "sha256": "11fadb0e8e50ee82157a019d2ff04803a28bd214dc987f0d1f14b8d1bb74598c",
        "bytes": 15_426,
    },
    CELLS: {
        "git_blob": "753d8d528223d509b0563db327c624a92c3b8acf",
        "sha256": "a770ae94274eaa85591a3ecd1f0f919b85dadc1c0ac3197c363b31659cb6b132",
        "bytes": 333_982,
    },
    QUEUE_MANIFEST: {
        "git_blob": "5fe4bf3e2668087178de8b11989a80ca616cc1e7",
        "sha256": "47c426f13146591d1a0bde60136e124eb5818cd8d44ef312f0f8fa82ad1623a1",
        "bytes": 4_333,
    },
    AMENDMENT: {
        "git_blob": "e5b970d522f3ab8303a3240ff0a43199a7ec35c1",
        "sha256": "ff23475b53791c42715938d51a303e0ab82de88b1b8a7a30758c008c9919a47b",
        "bytes": 6_139,
    },
    REPORT: {
        "git_blob": "45171ddf0707e0f66e20b10a75da5c8b6fb60dd7",
        "sha256": "a65c659cc5eb24e03baafc8101473f63307a549764090b6403eed5d1405ba999",
        "bytes": 16_128,
    },
    SOURCE_MANIFEST: {
        "git_blob": "ac09ac6fb5c4fd72ac0ac4018f49e9d608fbdd54",
        "sha256": "31a897d74403522a8a8cb8f4e6d04cdeb87bc8c61c21074263d4d9af7edd83ff",
        "bytes": 80_562,
    },
    FINAL_AUDIT: {
        "git_blob": "d9d6971fa32dfcfbd509fbbdc5c41e34553972e7",
        "sha256": "a0024d47d8b83297ac4b6ca079e40eb0b60394055682cdd35470dd03a13c15dc",
        "bytes": 2_656,
    },
}

STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B005"
MODEL_ID = "cosmos3_nano_policy_droid"
SEEDS = tuple(range(9500, 9515))
RELATIONS = ("left", "right")
LEVELS = (
    0.03658219039440155,
    0.06658219039440155,
    0.09658219039440155,
    0.12658219039440155,
    0.15658219039440155,
    0.18658219039440155,
    0.21658219039440155,
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 3_104_161
REPORT_SCHEMA = "vla-wam-shared-v3b005-nano-dose-response-report-v1"
PAIR_RE = re.compile(r"/pair_diagnostics/seed(\d+)_level(\d+)\.json$")
EPISODE_RE = re.compile(
    r"/v3b005__nano__seed(\d+)__level(\d+)__(left|right)/attempt01/raw_episode\.jsonl$"
)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def read_pinned_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    resolved = _git(root, "rev-parse", "--verify", PIN + "^{commit}").decode().strip()
    if resolved != PIN:
        raise ValueError("resolved historical evidence commit differs from fixed pin")

    raw_by_path: dict[str, bytes] = {}
    observed_files: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_SOURCES.items():
        object_id = _git(root, "rev-parse", "--verify", PIN + ":" + path).decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"pinned Git blob mismatch for {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        git_id = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        observed = {
            "git_blob": git_id,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        if observed != expected:
            raise ValueError(f"pinned source bytes disagree with manifest for {path}")
        raw_by_path[path] = raw
        observed_files[path] = observed

    line_bytes = raw_by_path[CELLS].splitlines()
    if len(line_bytes) != 210 or any(not line.strip() for line in line_bytes):
        raise ValueError("registered H05 queue must contain exactly 210 non-empty rows")
    cells = [parse_json(line, label=f"{CELLS}:{index}") for index, line in enumerate(line_bytes, 1)]
    documents = {
        "cells": cells,
        "queue_manifest": parse_json(raw_by_path[QUEUE_MANIFEST], label=QUEUE_MANIFEST),
        "amendment": parse_json(raw_by_path[AMENDMENT], label=AMENDMENT),
        "report": parse_json(raw_by_path[REPORT], label=REPORT),
        "source_manifest": parse_json(raw_by_path[SOURCE_MANIFEST], label=SOURCE_MANIFEST),
        "final_audit": parse_json(raw_by_path[FINAL_AUDIT], label=FINAL_AUDIT),
    }
    return documents, {"commit": PIN, "files": observed_files}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _close(observed: Any, expected: float, label: str, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(_finite(observed, label), expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} disagrees with independent reproduction")


def _slope(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != len(LEVELS):
        raise ValueError("each matched seed must contain all seven ordered levels")
    x = np.asarray(LEVELS, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    centered = x - float(x.mean())
    slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
    return slope, float(y.mean())


def _sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    ties = len(values) - positive - negative
    n = positive + negative
    tail = 1.0 if n == 0 else sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "non_tied_n": n,
        "two_sided_p": min(1.0, 2.0 * tail),
        "method": "exact_two_sided_paired_sign_test_zero_ties_excluded",
    }


def _bootstrap(values: Sequence[float], seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    means = array[draws].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(array.mean()),
        "ci95": [float(low), float(high)],
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "matched_seed",
        "method": "nonparametric_percentile_bootstrap",
    }


def _validate_design(documents: Mapping[str, Any]) -> None:
    amendment = documents["amendment"]
    if amendment.get("status") != "design_frozen_before_v3b005_model_blind_gate_and_before_any_v3b005_model_request":
        raise ValueError("H05 amendment is not the pre-request frozen design")
    design = amendment.get("design")
    if not isinstance(design, dict):
        raise ValueError("H05 amendment has no design object")
    if design.get("matched_seeds") != list(SEEDS) or design.get("ordered_bowl_y_levels_m") != list(LEVELS):
        raise ValueError("H05 amendment changed the registered seed/level grid")
    if design.get("relations") != list(RELATIONS) or design.get("registered_behavioral_episode_ceiling_after_release") != 210:
        raise ValueError("H05 amendment changed relations or episode ceiling")
    plan = amendment.get("analysis_plan", {})
    if plan.get("inferential_unit") != "matched seed; repeated levels are within-seed observations":
        raise ValueError("H05 inferential unit changed")
    if plan.get("bootstrap_master_seed") != BOOTSTRAP_SEED:
        raise ValueError("H05 bootstrap seed changed")

    cells = documents["cells"]
    indexed: set[tuple[int, int, str]] = set()
    prompts = design["exact_prompts"]
    for row in cells:
        seed, level, relation = row.get("environment_seed"), row.get("level_index"), row.get("relation")
        key = (seed, level, relation)
        if seed not in SEEDS or type(level) is not int or not 0 <= level < 7 or relation not in RELATIONS:
            raise ValueError("registered queue row lies outside the frozen H05 grid")
        if key in indexed:
            raise ValueError(f"duplicate registered H05 cell: {key}")
        indexed.add(key)
        expected_cell = f"v3b005:nano:seed{seed}:level{level}:{relation}"
        if row.get("cell_id") != expected_cell or row.get("sampling_seed") != seed:
            raise ValueError(f"registered H05 cell identity mismatch: {key}")
        if row.get("prompt") != prompts[relation] or row.get("model_id") != MODEL_ID:
            raise ValueError(f"registered H05 prompt/model mismatch: {key}")
        _close(row.get("reference_object_initial_lateral_position_y_m"), LEVELS[level], f"{key}.bowl_y")
    expected = {(seed, level, relation) for seed in SEEDS for level in range(7) for relation in RELATIONS}
    if indexed != expected:
        raise ValueError("registered queue is not the exact 15 x 7 x 2 H05 grid")

    queue = documents["queue_manifest"]
    if queue.get("amendment_sha256") != EXPECTED_SOURCES[AMENDMENT]["sha256"]:
        raise ValueError("queue manifest is not bound to the authenticated amendment")
    if queue.get("cells") != {
        "path": "nano_lateral_v3b005_cells.jsonl",
        "row_count": 210,
        "sha256": EXPECTED_SOURCES[CELLS]["sha256"],
    }:
        raise ValueError("queue manifest is not bound to the authenticated 210-cell grid")


def _validate_source_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("sha256") != EXPECTED_SOURCES[REPORT]["sha256"] or manifest.get("bytes") != 16_128:
        raise ValueError("source manifest is not bound to the authenticated report")
    pair_sources = manifest.get("pair_sources")
    episode_sources = manifest.get("episode_sources")
    if not isinstance(pair_sources, list) or not isinstance(episode_sources, list):
        raise ValueError("report source manifest has malformed source lists")
    pair_grid: set[tuple[int, int]] = set()
    episode_grid: set[tuple[int, int, str]] = set()
    source_hashes: list[str] = []
    for item in pair_sources:
        match = PAIR_RE.search(str(item.get("path"))) if isinstance(item, dict) else None
        if match is None:
            raise ValueError("report manifest has an unrecognized pair source")
        pair_grid.add((int(match.group(1)), int(match.group(2))))
        source_hashes.append(str(item.get("sha256")))
    for item in episode_sources:
        match = EPISODE_RE.search(str(item.get("path"))) if isinstance(item, dict) else None
        if match is None:
            raise ValueError("report manifest has an unrecognized episode source")
        episode_grid.add((int(match.group(1)), int(match.group(2)), match.group(3)))
        source_hashes.append(str(item.get("sha256")))
    expected_pairs = {(seed, level) for seed in SEEDS for level in range(7)}
    expected_episodes = {(seed, level, relation) for seed in SEEDS for level in range(7) for relation in RELATIONS}
    if len(pair_sources) != 105 or pair_grid != expected_pairs:
        raise ValueError("source manifest is not the exact 105-pair H05 grid")
    if len(episode_sources) != 210 or episode_grid != expected_episodes:
        raise ValueError("source manifest is not the exact 210-episode H05 grid")
    if len(set(source_hashes)) != 315 or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in source_hashes):
        raise ValueError("source manifest does not retain 315 unique SHA-256 identities")
    return {
        "pair_source_identities": len(pair_sources),
        "episode_source_identities": len(episode_sources),
        "unique_source_sha256": len(set(source_hashes)),
        "raw_source_bytes_present_in_checkout": False,
        "raw_source_bytes_rehashed_by_this_reproduction": False,
    }


def _assert_bootstrap(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    _close(observed.get("mean"), expected["mean"], f"{label}.mean")
    for key in ("replicates", "unit", "method"):
        if observed.get(key) != expected.get(key):
            raise ValueError(f"{label}.{key} disagrees with independent reproduction")
    for index, value in enumerate(expected["ci95"]):
        _close(observed.get("ci95", [None, None])[index], value, f"{label}.ci95[{index}]")


def analyze_documents(documents: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    _validate_design(documents)
    manifest_summary = _validate_source_manifest(documents["source_manifest"])
    report = documents["report"]
    audit = documents["final_audit"]
    if report.get("schema_version") != REPORT_SCHEMA or report.get("model_id") != MODEL_ID:
        raise ValueError("retained H05 report schema/model mismatch")
    expected_population = {
        "matched_seed_count": 15,
        "level_count": 7,
        "matched_pair_count": 105,
        "behavioral_episode_count": 210,
        "valid_behavioral_failures_included": True,
        "infrastructure_attempts_included": False,
        "missing_value_imputation": "none",
    }
    if report.get("population") != expected_population:
        raise ValueError("retained H05 report population changed")
    if audit.get("verdict") != "pass" or audit.get("registered_report", {}).get("sha256") != EXPECTED_SOURCES[REPORT]["sha256"]:
        raise ValueError("final audit does not pass and bind the retained report")
    audit_population = audit.get("population", {})
    if audit_population != {
        "behavioral_cells": 210,
        "matched_pairs": 105,
        "registered_seeds": 15,
        "registered_levels": 7,
        "directions_per_seed_level": 2,
        "model_requests": 1423,
        "decoded_futures": 1423,
        "bridge_failure_count": 0,
    }:
        raise ValueError("final H05 audit population changed")

    seed_rows = report.get("seed_level")
    if not isinstance(seed_rows, list) or len(seed_rows) != 15:
        raise ValueError("retained report must contain all 15 matched-seed rows")
    normalized_seed_rows: list[dict[str, Any]] = []
    normalized_seed_level_rows: list[dict[str, Any]] = []
    primary_slopes: list[float] = []
    primary_centers: list[float] = []
    success_slopes: list[float] = []
    success_gaps_by_level: list[list[int]] = [[] for _ in LEVELS]
    depth_by_level: list[list[float]] = [[] for _ in LEVELS]
    for expected_seed, row in zip(SEEDS, seed_rows):
        if not isinstance(row, dict) or row.get("seed") != expected_seed:
            raise ValueError("retained seed rows are not in the exact registered order")
        depths = [_finite(value, f"seed{expected_seed}.depth") for value in row.get("depth_contrast_by_level_m", [])]
        gaps_raw = row.get("success_gap_by_level")
        if not isinstance(gaps_raw, list) or len(gaps_raw) != 7 or any(type(value) is not int or value not in (-1, 0, 1) for value in gaps_raw):
            raise ValueError(f"seed {expected_seed} has malformed binary success gaps")
        depth_slope, depth_center = _slope(depths)
        success_slope, success_center = _slope(gaps_raw)
        _close(row.get("depth_slope_m_per_m"), depth_slope, f"seed{expected_seed}.depth_slope")
        _close(row.get("depth_at_center_m"), depth_center, f"seed{expected_seed}.depth_center")
        _close(row.get("success_gap_slope_per_m"), success_slope, f"seed{expected_seed}.success_slope")
        _close(row.get("success_gap_at_center"), success_center, f"seed{expected_seed}.success_center")
        # Preserve the authenticated report's canonical binary64 values after
        # validating them against the independent calculation above.  This
        # keeps rendered artifacts stable across NumPy/BLAS implementations
        # whose last-bit dot-product rounding can differ.
        reported_depth_slope = float(row["depth_slope_m_per_m"])
        reported_depth_center = float(row["depth_at_center_m"])
        reported_success_slope = float(row["success_gap_slope_per_m"])
        reported_success_center = float(row["success_gap_at_center"])
        primary_slopes.append(reported_depth_slope)
        primary_centers.append(reported_depth_center)
        success_slopes.append(reported_success_slope)
        for level, value in enumerate(depths):
            depth_by_level[level].append(value)
            success_gaps_by_level[level].append(gaps_raw[level])
        normalized_seed_rows.append({
            "seed": expected_seed,
            "depth_slope_m_per_m": reported_depth_slope,
            "depth_at_center_m": reported_depth_center,
            "success_gap_slope_per_m": reported_success_slope,
            "success_gap_at_center": reported_success_center,
        })
        normalized_seed_level_rows.extend({
            "seed": expected_seed,
            "level_index": level,
            "bowl_y_m": LEVELS[level],
            "depth_contrast_B_m": depths[level],
            "success_right_minus_left": gaps_raw[level],
            "depth_slope_m_per_m": reported_depth_slope,
            "success_gap_slope_per_m": reported_success_slope,
        } for level in range(7))

    primary = report["primary_depth_dose_response"]["slope_m_per_m"]
    primary_bootstrap = _bootstrap(primary_slopes, BOOTSTRAP_SEED + 100)
    _assert_bootstrap(primary, primary_bootstrap, "primary slope")
    _close(primary.get("median"), statistics.median(primary_slopes), "primary median")
    if primary.get("sign_test") != _sign_test(primary_slopes):
        raise ValueError("primary slope sign test disagrees with independent reproduction")

    secondary = report["binary_success_secondary"]["paired_gap_slope_per_m"]
    secondary_bootstrap = _bootstrap(success_slopes, BOOTSTRAP_SEED + 200)
    _assert_bootstrap(secondary, secondary_bootstrap, "success-gap slope")
    _close(secondary.get("median"), statistics.median(success_slopes), "success slope median")
    if secondary.get("sign_test") != _sign_test(success_slopes):
        raise ValueError("success-gap sign test disagrees with independent reproduction")

    mean_slope = primary_bootstrap["mean"]
    crossing = statistics.fmean(LEVELS) - statistics.fmean(primary_centers) / mean_slope
    reported_crossing = report["primary_depth_dose_response"]["population_linear_zero_crossing"]
    _close(reported_crossing.get("unclipped_linear_estimate_m"), crossing, "linear zero crossing")
    if reported_crossing.get("in_registered_support") is not False or reported_crossing.get("reference_object_lateral_y_m") is not None:
        raise ValueError("out-of-support linear crossing was incorrectly promoted")

    by_level = report.get("by_level")
    if not isinstance(by_level, list) or len(by_level) != 7:
        raise ValueError("retained report must contain all seven level summaries")
    level_rows: list[dict[str, Any]] = []
    total_success = {"left": 0, "right": 0}
    for level, (y_m, row) in enumerate(zip(LEVELS, by_level)):
        if row.get("level_index") != level:
            raise ValueError("retained level summaries are out of order")
        _close(row.get("reference_object_initial_lateral_position_y_m"), y_m, f"level{level}.y")
        values = depth_by_level[level]
        depth_summary = row["depth_contrast_B_m"]
        reproduced_level_bootstrap = _bootstrap(values, BOOTSTRAP_SEED + level)
        _assert_bootstrap(depth_summary, reproduced_level_bootstrap, f"level{level}.depth")
        _close(depth_summary.get("median"), statistics.median(values), f"level{level}.median")
        success = row["binary_success"]
        for relation in RELATIONS:
            successes = success[relation]["successes"]
            if type(successes) is not int or success[relation]["episodes"] != 15:
                raise ValueError(f"level{level}.{relation} success denominator changed")
            total_success[relation] += successes
        total_level_gap = sum(success_gaps_by_level[level])
        if total_level_gap != success["right"]["successes"] - success["left"]["successes"]:
            raise ValueError(f"level{level} success gaps disagree with reported counts")
        conditional = row["both_directions_success_conditional_depth"]
        conditional_values = [_finite(value, f"level{level}.conditional_depth") for value in conditional["values_m"]]
        if conditional.get("realized_pair_n") != len(conditional_values):
            raise ValueError(f"level{level} conditional-depth denominator changed")
        _close(conditional.get("mean_m"), statistics.fmean(conditional_values), f"level{level}.conditional_mean")
        level_rows.append({
            "level_index": level,
            "bowl_y_m": y_m,
            "depth_contrast_mean_m": reproduced_level_bootstrap["mean"],
            "depth_contrast_ci95_m": reproduced_level_bootstrap["ci95"],
            "depth_contrast_median_m": statistics.median(values),
            "left_successes": success["left"]["successes"],
            "right_successes": success["right"]["successes"],
            "right_minus_left_successes": total_level_gap,
            "both_success_conditional_pair_n": len(conditional_values),
            "both_success_conditional_depth_mean_m": statistics.fmean(conditional_values),
        })
    if total_success != {"left": 93, "right": 99}:
        raise ValueError("H05 aggregate LEFT/RIGHT success counts changed")

    return {
        "schema_version": "gm-forecast-ablation-retained-h05-dose-response-v1",
        "provenance": provenance,
        "design": {
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "model_id": MODEL_ID,
            "matched_seed_count": 15,
            "controlled_level_count": 7,
            "relations": list(RELATIONS),
            "registered_levels_m": list(LEVELS),
            "registered_behavioral_episode_ceiling": 210,
            "inferential_unit": "matched_seed",
        },
        "authenticated_historical_population": {
            **expected_population,
            "left_successes": 93,
            "right_successes": 99,
            "model_requests": 1423,
            "decoded_futures": 1423,
            "infrastructure_invalid_cells": 0,
        },
        "source_manifest": manifest_summary,
        "primary_depth_dose_response": {
            "estimand": "within-seed OLS slope of B=(-signed_offset_RIGHT)-signed_offset_LEFT over seven registered bowl-y levels",
            **primary_bootstrap,
            "median": statistics.median(primary_slopes),
            "per_seed": primary_slopes,
            "sign_test": _sign_test(primary_slopes),
            "linear_zero_crossing": {
                "unclipped_m": crossing,
                "in_registered_support": False,
                "reported_value_m": None,
                "registered_support_m": [LEVELS[0], LEVELS[-1]],
            },
        },
        "binary_success_secondary": {
            "estimand": "within-seed OLS slope of success_RIGHT-success_LEFT over seven registered bowl-y levels",
            **secondary_bootstrap,
            "median": statistics.median(success_slopes),
            "per_seed": success_slopes,
            "sign_test": _sign_test(success_slopes),
        },
        "by_level": level_rows,
        "matched_seeds": normalized_seed_rows,
        "matched_seed_levels": normalized_seed_level_rows,
        "historical_final_audit": {
            "verdict": "pass",
            "unique_files_hashed": audit["hash_and_decode_audit"]["unique_files_hashed"],
            "unique_bytes_hashed": audit["hash_and_decode_audit"]["unique_bytes_hashed"],
            "viewport_frames_opened": audit["hash_and_decode_audit"]["viewport_frames_opened"],
            "mismatch_count": audit["hash_and_decode_audit"]["mismatch_count"],
        },
        "reproduction": {
            "report_level_seed_rows_recomputed": 15,
            "level_rows_recomputed": 7,
            "bootstrap_replicates_per_summary": BOOTSTRAP_REPLICATES,
            "bootstrap_master_seed": BOOTSTRAP_SEED,
            "primary_and_secondary_bootstrap_exactly_reproduced": True,
            "seven_level_bootstraps_exactly_reproduced": True,
            "sign_tests_exactly_reproduced": True,
            "success_totals_exactly_reproduced": True,
        },
        "new_activity": {
            "model_requests": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "request_rows": 0,
            "request_selections": 0,
            "annotation_media": 0,
            "human_responses": 0,
            "labels": 0,
            "forecast_accuracy_estimates": 0,
            "retrieval_jobs": 0,
            "queue_descriptors": 0,
            "runnable_manifests": 0,
            "confirmation_inference_cells": 0,
        },
        "gate_state": {
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "camera_alignment_ready": False,
            "development_annotation_gate_ready": False,
            "annotation_media_ready": False,
            "human_annotation_ready": False,
            "resource_freeze_ready": False,
            "confirmation_release_ready": False,
            "safe_to_release_new_work": False,
        },
        "claim_boundary": {
            "status": "retrospective authentication and report-level arithmetic reproduction",
            "new_model_inference": False,
            "new_robot_episodes": False,
            "new_human_labels": False,
            "forecast_accuracy_evidence": False,
            "raw_pair_and_episode_bytes_revalidated_now": False,
            "matched_seeds_are_independent_scenes": False,
            "controlled_positions_are_independent_scenes": False,
            "confirmation_test_or_stage_gate": False,
            "changes_232_cell_core_bound": False,
            "boundaries": [
                "The 210 episodes are repeated positions and directions within 15 matched seed blocks, not 210 independent scenes.",
                "The 210 behavioral episodes and 1,423 model requests are authenticated historical-population counts, not current-turn activity.",
                "The report source manifest preserves 315 raw-file paths and SHA-256 identities, but those raw bytes are outside this checkout and were not rehashed here.",
                "Success-conditional depth omits undefined pairs and never encodes a failed direction as zero.",
                "The primary depth slope is retained executed-behavior evidence, not generated-future accuracy, annotation evidence, or a forecast-ablation confirmation result.",
                "The linear zero crossing lies outside registered support and is not reported as an in-support crossing.",
            ],
        },
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    documents, provenance = read_pinned_sources(root)
    return analyze_documents(documents, provenance)


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    fields = [
        "seed", "level_index", "bowl_y_m", "depth_contrast_B_m",
        "success_right_minus_left", "depth_slope_m_per_m", "success_gap_slope_per_m",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["matched_seed_levels"])
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    report = build_report(args.repo_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(render_json(report), encoding="utf-8")
    args.csv_output.write_text(render_csv(report), encoding="utf-8")
    print(json.dumps({
        "json_output": str(args.json_output),
        "csv_output": str(args.csv_output),
        "matched_seeds": len(report["matched_seeds"]),
        "episodes": report["authenticated_historical_population"]["behavioral_episode_count"],
        "primary_slope_mean_m_per_m": report["primary_depth_dose_response"]["mean"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
