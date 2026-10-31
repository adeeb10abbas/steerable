#!/usr/bin/env python3
"""Authenticate and reproduce the retained H06 graded-symmetry analysis.

H06 is the completed Cosmos3-Nano slice of V3-E004.  The primary graded
estimand was frozen before inference: within each of the 27 historical core
seeds, regress the RIGHT-minus-LEFT requested-side depth gap on realised object
layout asymmetry over levels 0.25, 0.50, 0.75 and 1.00.  Level zero is excluded
because the registered intervention changes companion-object inventory there.

This script reads immutable compact Git evidence only.  It authenticates and
strictly parses the full 4,096-episode/2,048-pair V3-E004 tables, independently
rebuilds the 216-episode/108-pair H06 primary population, and reconciles the
arithmetic with the registered report.  Original simulator media and raw
episode files remain external.  No model, robot, remote job or annotation is
run, and no forecast-ablation stage gate advances.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h06_graded_symmetry.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h06_graded_symmetry.csv"

BASE = "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
REGISTRATION = f"{BASE}/registration.json"
EVIDENCE_MANIFEST = f"{BASE}/evidence_manifest.json"
EPISODES = f"{BASE}/results/episodes.jsonl"
PAIRS = f"{BASE}/results/pairs.jsonl"
RESULTS = f"{BASE}/results/results.json"
UPSTREAM_ANALYSIS = "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/analysis.py"

EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    REGISTRATION: {
        "git_blob": "758a77c1227b43c41c66e3ffea27882c78382f02",
        "sha256": "3e3786510ebba933d0c7a4606afa51627ea6bf706f9e64b34cfd74b29e318db2",
        "bytes": 16_100,
        "lines": 359,
    },
    EVIDENCE_MANIFEST: {
        "git_blob": "7c2d6a26aaac6fd2fcee957fa9439c4c8d444717",
        "sha256": "21db6089f67533f270064047e62123e2c69389783aeb314f7a24164a688290d5",
        "bytes": 13_178,
        "lines": 294,
    },
    EPISODES: {
        "git_blob": "5b7b2e5d1118f38d96a753499cfb583fcb9ce800",
        "sha256": "3f5528a686f0f1f04767ba97414429ef0aff30576edc3d440d611facd363f328",
        "bytes": 78_300_627,
        "lines": 4_096,
    },
    PAIRS: {
        "git_blob": "3ed2574b169e1fd1b1774febbc5470f81c6de409",
        "sha256": "fe2d7d8879314122a48aa269374118a45d13f204c9c83691735920e917d68bcc",
        "bytes": 3_981_781,
        "lines": 2_048,
    },
    RESULTS: {
        "git_blob": "7f65771c434d95a60e1856c4774de38608f0416f",
        "sha256": "be8cd0c9a2458a7eefec0ed7760d1e6f6f19e4162d8c427f12f8a113d3290493",
        "bytes": 212_799,
        "lines": 6_592,
    },
    UPSTREAM_ANALYSIS: {
        "git_blob": "fe258d889f4ce164e5c938b02214a902100e0621",
        "sha256": "82d0552d1197cc00c08f3888137fd2b75031a4e69dcc2b50411b9e628a05e5db",
        "bytes": 13_506,
        "lines": 236,
    },
}

MODEL = "cosmos3_nano_policy_droid"
STUDY = "vla_wam_language_steerability_v3"
AMENDMENT = "V3-E004"
SEEDS = tuple(range(9400, 9427))
LEVELS = (0.25, 0.50, 0.75, 1.00)
RELATIONS = ("left", "right")
LEVEL_CODES = {0.25: "025", 0.50: "050", 0.75: "075", 1.00: "100"}
BOOTSTRAP_REPLICATES = 20_000


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


def _selected(row: Mapping[str, Any]) -> bool:
    return (
        row.get("model_id") == MODEL
        and row.get("sampling_seed") in SEEDS
        and row.get("symmetry_level_s") in LEVELS
    )


def read_pinned_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read, byte-authenticate and strictly parse the immutable source table."""

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
        observed = {
            "git_blob": object_id,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "lines": len(raw.splitlines()),
        }
        if observed != expected:
            raise ValueError(f"pinned source bytes disagree with fixed identity for {path}")
        raw_by_path[path] = raw
        observed_files[path] = observed

    episode_rows: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    episode_models: Counter[str] = Counter()
    episode_level_relation: Counter[str] = Counter()
    episode_successes: Counter[str] = Counter()
    for number, raw_line in enumerate(raw_by_path[EPISODES].splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"blank compact episode row at {number}")
        row = parse_json(raw_line, label=f"{EPISODES}:{number}")
        cell_id = row.get("registered_cell_id")
        if not isinstance(cell_id, str) or cell_id in episode_ids:
            raise ValueError(f"missing or duplicate compact episode identity at row {number}")
        episode_ids.add(cell_id)
        model = str(row.get("model_id"))
        episode_models[model] += 1
        level = row.get("symmetry_level_s")
        relation = row.get("relation")
        if isinstance(level, (int, float)) and not isinstance(level, bool) and relation in RELATIONS:
            count_key = f"{model}|{float(level):.2f}|{relation}"
            episode_level_relation[count_key] += 1
            if row.get("success") is True:
                episode_successes[count_key] += 1
        if _selected(row):
            episode_rows.append(row)

    pair_rows: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    pair_models: Counter[str] = Counter()
    for number, raw_line in enumerate(raw_by_path[PAIRS].splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"blank compact pair row at {number}")
        row = parse_json(raw_line, label=f"{PAIRS}:{number}")
        pair_id = row.get("matched_pair_id")
        if not isinstance(pair_id, str) or pair_id in pair_ids:
            raise ValueError(f"missing or duplicate compact pair identity at row {number}")
        pair_ids.add(pair_id)
        pair_models[str(row.get("model_id"))] += 1
        if _selected(row):
            pair_rows.append(row)

    documents = {
        "registration": parse_json(raw_by_path[REGISTRATION], label=REGISTRATION),
        "evidence_manifest": parse_json(raw_by_path[EVIDENCE_MANIFEST], label=EVIDENCE_MANIFEST),
        "results": parse_json(raw_by_path[RESULTS], label=RESULTS),
        "episodes": episode_rows,
        "pairs": pair_rows,
        "full_counts": {
            "episodes": len(episode_ids),
            "pairs": len(pair_ids),
            "episodes_by_model": dict(sorted(episode_models.items())),
            "pairs_by_model": dict(sorted(pair_models.items())),
            "episodes_by_model_level_relation": dict(sorted(episode_level_relation.items())),
            "successes_by_model_level_relation": dict(sorted(episode_successes.items())),
        },
    }
    return documents, {"commit": PIN, "files": observed_files}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _close(observed: Any, expected: float, label: str, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(_finite(observed, label), expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} disagrees with independent reproduction")


def _seed_from_label(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def _bootstrap_mean(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap input must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(BOOTSTRAP_REPLICATES, array.size), replace=True)
    low, high = np.quantile(sampled.mean(axis=1), (0.025, 0.975))
    return {
        "mean": float(array.mean()),
        "ci95": [float(low), float(high)],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "unit": "matched_seed",
        "method": "nonparametric_percentile_bootstrap",
    }


def _sign_test(values: Iterable[float]) -> dict[str, Any]:
    values = list(values)
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


def _validate_registration(documents: Mapping[str, Any]) -> None:
    registration = documents["registration"]
    if registration.get("study_id") != STUDY or registration.get("amendment_id") != AMENDMENT:
        raise ValueError("H06 registration identity changed")
    if registration.get("status") != "prospectively_registered_zero_e004_model_requests_or_behavioral_episodes":
        raise ValueError("H06 registration is not the frozen pre-inference record")
    if registration.get("behavioral_episode_count_before_registration") != 0 or registration.get("model_request_count_before_registration") != 0:
        raise ValueError("H06 registration was not made before model activity")
    design = registration.get("design", {})
    if design.get("core_seeds") != list(SEEDS):
        raise ValueError("H06 registered core seeds changed")
    if design.get("model_levels_and_pairs", {}).get(MODEL) != {
        "0.00": 521,
        "0.25": 27,
        "0.50": 27,
        "0.75": 27,
        "1.00": 521,
    }:
        raise ValueError("H06 registered Nano level allocation changed")
    inventory = registration.get("layout", {}).get("inventory_transition", {})
    if inventory.get("inventory_transition") is not True or inventory.get("H3_primary_levels") != list(LEVELS):
        raise ValueError("H06 inventory-matched positive-level design changed")
    if registration.get("analysis_plan", {}).get("dose_response") != (
        "Per-seed slope on realised A for pi05/Nano inventory-matched positive levels, seed bootstrap CI, and slope signs."
    ):
        raise ValueError("H06 registered dose-response plan changed")

    manifest = documents["evidence_manifest"]
    required = {
        "study_id": STUDY,
        "amendment_id": AMENDMENT,
        "status": "hash_closed_compact_evidence",
        "registered_behavioral_cells": 4_096,
        "valid_behavioral_episodes": 4_096,
        "episodes_sha256": EXPECTED_SOURCES[EPISODES]["sha256"],
        "pairs_sha256": EXPECTED_SOURCES[PAIRS]["sha256"],
        "results_sha256": EXPECTED_SOURCES[RESULTS]["sha256"],
        "registration_sha256": EXPECTED_SOURCES[REGISTRATION]["sha256"],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"H06 evidence manifest changed: {key}")

    counts = documents["full_counts"]
    if counts.get("episodes") != 4_096 or counts.get("pairs") != 2_048:
        raise ValueError("full V3-E004 compact table is incomplete")
    if counts.get("episodes_by_model", {}).get(MODEL) != 2_246:
        raise ValueError("full H06 compact episode population is not 2,246")
    if counts.get("pairs_by_model", {}).get(MODEL) != 1_123:
        raise ValueError("full H06 compact pair population is not 1,123")


def _normalize_primary(documents: Mapping[str, Any]) -> list[dict[str, Any]]:
    episodes = documents["episodes"]
    pairs = documents["pairs"]
    if len(episodes) != 216 or len(pairs) != 108:
        raise ValueError("H06 primary population must contain exactly 216 episodes and 108 pairs")

    by_cell: dict[tuple[int, float, str], Mapping[str, Any]] = {}
    for row in episodes:
        seed = row.get("sampling_seed")
        level = row.get("symmetry_level_s")
        relation = row.get("relation")
        key = (seed, level, relation)
        if seed not in SEEDS or level not in LEVELS or relation not in RELATIONS:
            raise ValueError(f"episode outside H06 primary grid: {key}")
        if key in by_cell:
            raise ValueError(f"duplicate H06 primary episode: {key}")
        expected_pair = f"v3e004:nano:seed{seed}:s{LEVEL_CODES[level]}"
        expected_cell = expected_pair + f":{relation}"
        if row.get("matched_pair_id") != expected_pair or row.get("registered_cell_id") != expected_cell:
            raise ValueError(f"mismatched H06 primary episode identity: {key}")
        if row.get("study_id") != STUDY or row.get("amendment_id") != AMENDMENT:
            raise ValueError(f"mismatched H06 source identity: {key}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"H06 success must be Boolean: {key}")
        for field in ("requested_side_depth", "signed_final_lateral_offset", "asymmetry_metric_A"):
            _finite(row.get(field), f"{key}.{field}")
        by_cell[key] = row

    expected_cells = {(seed, level, relation) for seed in SEEDS for level in LEVELS for relation in RELATIONS}
    if set(by_cell) != expected_cells:
        raise ValueError("H06 primary episode grid is incomplete")

    by_pair: dict[tuple[int, float], Mapping[str, Any]] = {}
    for row in pairs:
        seed = row.get("sampling_seed")
        level = row.get("symmetry_level_s")
        key = (seed, level)
        if seed not in SEEDS or level not in LEVELS or key in by_pair:
            raise ValueError(f"invalid or duplicate H06 compact pair: {key}")
        expected_pair = f"v3e004:nano:seed{seed}:s{LEVEL_CODES[level]}"
        if row.get("matched_pair_id") != expected_pair:
            raise ValueError(f"mismatched H06 compact pair identity: {key}")
        left, right = by_cell[(seed, level, "left")], by_cell[(seed, level, "right")]
        if row.get("left_cell_id") != left.get("registered_cell_id") or row.get("right_cell_id") != right.get("registered_cell_id"):
            raise ValueError(f"H06 pair-to-episode join changed: {key}")
        if row.get("left_success") is not left.get("success") or row.get("right_success") is not right.get("success"):
            raise ValueError(f"H06 compact pair success disagrees with episode rows: {key}")
        A = 0.5 * (
            _finite(left.get("asymmetry_metric_A"), f"{key}.left.A")
            + _finite(right.get("asymmetry_metric_A"), f"{key}.right.A")
        )
        _close(left.get("asymmetry_metric_A"), A, f"{key}.left.A", tolerance=1e-9)
        _close(right.get("asymmetry_metric_A"), A, f"{key}.right.A", tolerance=1e-9)
        _close(row.get("asymmetry_metric_A"), A, f"{key}.pair.A", tolerance=1e-9)
        endpoint = _finite(left.get("signed_final_lateral_offset"), f"{key}.left.endpoint") - _finite(
            right.get("signed_final_lateral_offset"), f"{key}.right.endpoint"
        )
        _close(row.get("endpoint_redirection_left_minus_right_m"), endpoint, f"{key}.endpoint")
        by_pair[key] = row

    expected_pairs = {(seed, level) for seed in SEEDS for level in LEVELS}
    if set(by_pair) != expected_pairs:
        raise ValueError("H06 primary pair grid is incomplete")

    contrasts: list[dict[str, Any]] = []
    for seed in SEEDS:
        for level in LEVELS:
            left, right = by_cell[(seed, level, "left")], by_cell[(seed, level, "right")]
            contrasts.append(
                {
                    "seed": seed,
                    "symmetry_level_s": level,
                    "realised_asymmetry_A": _finite(by_pair[(seed, level)].get("asymmetry_metric_A"), "A"),
                    "requested_depth_gap_R_minus_L_m": _finite(right.get("requested_side_depth"), "right depth")
                    - _finite(left.get("requested_side_depth"), "left depth"),
                    "binary_success_gap_R_minus_L": int(right["success"]) - int(left["success"]),
                    "left_success": left["success"],
                    "right_success": right["success"],
                }
            )
    return contrasts


def _reconcile_registered_report(
    report: Mapping[str, Any],
    full_counts: Mapping[str, Any],
    contrasts: Sequence[Mapping[str, Any]],
    depth: Mapping[str, Any],
    binary: Mapping[str, Any],
) -> None:
    checkpoint = report.get("checkpoints", {}).get(MODEL, {})
    if checkpoint.get("valid_episodes") != 2_246 or checkpoint.get("complete_pairs") != 1_123:
        raise ValueError("registered H06 report population changed")
    registered = checkpoint.get("analysis", {}).get("dose_response_on_realised_A", {})
    if registered.get("primary_levels") != list(LEVELS) or registered.get("s0_excluded_due_registered_inventory_transition") is not True:
        raise ValueError("registered H06 report primary levels changed")
    rows = registered.get("seed_slopes")
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise ValueError("registered H06 report seed slopes are incomplete")
    by_seed = {row["seed"]: row for row in rows}
    for row in depth["per_seed"]:
        expected = by_seed.get(row["seed"])
        if expected is None:
            raise ValueError("registered H06 report omits a core seed")
        _close(expected.get("depth_gap_m_per_A"), row["slope_m_per_A"], "registered depth slope")
        _close(expected.get("binary_gap_per_A"), binary["per_seed_by_seed"][row["seed"]], "registered binary slope")
    for name, reproduced in (("depth_gap_m", depth), ("binary_gap", binary)):
        source = registered.get(name, {})
        _close(source.get("mean_slope"), reproduced["mean"], f"registered {name} mean")
        _close(source.get("median_slope"), reproduced["median"], f"registered {name} median")
        _close(source.get("sign", {}).get("exact_two_sided_p"), reproduced["sign_test"]["two_sided_p"], f"registered {name} sign p")
        source_bootstrap = source.get("bootstrap_mean95", {})
        _close(source_bootstrap.get("low"), reproduced["ci95"][0], f"registered {name} bootstrap low")
        _close(source_bootstrap.get("high"), reproduced["ci95"][1], f"registered {name} bootstrap high")

    progress = checkpoint.get("descriptive_progress", {})
    for level in (0.0,) + LEVELS:
        key = f"{level:.2f}"
        for relation in RELATIONS:
            source = progress.get(f"{key}/{relation}", {})
            count_key = f"{MODEL}|{key}|{relation}"
            observed_n = full_counts.get("episodes_by_model_level_relation", {}).get(count_key, 0)
            observed_successes = full_counts.get("successes_by_model_level_relation", {}).get(count_key, 0)
            if source.get("valid_episodes") != observed_n or source.get("successes") != observed_successes:
                raise ValueError(f"registered H06 level success count changed: {key}/{relation}")


def analyze_documents(documents: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    _validate_registration(documents)
    contrasts = _normalize_primary(documents)

    depth_slopes: list[dict[str, Any]] = []
    binary_slopes: dict[int, float] = {}
    for seed in SEEDS:
        rows = [row for row in contrasts if row["seed"] == seed]
        A = np.asarray([row["realised_asymmetry_A"] for row in rows], dtype=np.float64)
        depth_values = np.asarray([row["requested_depth_gap_R_minus_L_m"] for row in rows], dtype=np.float64)
        binary_values = np.asarray([row["binary_success_gap_R_minus_L"] for row in rows], dtype=np.float64)
        depth_slope = float(np.polyfit(A, depth_values, 1)[0])
        binary_slope = float(np.polyfit(A, binary_values, 1)[0])
        depth_slopes.append({"seed": seed, "slope_m_per_A": depth_slope})
        binary_slopes[seed] = binary_slope

    depth_values = [row["slope_m_per_A"] for row in depth_slopes]
    binary_values = [binary_slopes[seed] for seed in SEEDS]
    depth_bootstrap = _bootstrap_mean(
        depth_values, seed=_seed_from_label(MODEL + ":dose:depth_gap_m")
    )
    binary_bootstrap = _bootstrap_mean(
        binary_values, seed=_seed_from_label(MODEL + ":dose:binary_gap")
    )
    primary_depth = {
        "estimand": "within-seed OLS slope of requested-side depth gap B=(-y_RIGHT)-y_LEFT versus realised object-layout asymmetry A",
        "mean": depth_bootstrap["mean"],
        "median": float(statistics.median(depth_values)),
        "ci95": depth_bootstrap["ci95"],
        "replicates": depth_bootstrap["replicates"],
        "bootstrap_seed": depth_bootstrap["seed"],
        "unit": depth_bootstrap["unit"],
        "method": depth_bootstrap["method"],
        "sign_test": _sign_test(depth_values),
        "per_seed": depth_slopes,
        "interpretation": "A positive slope means the RIGHT-minus-LEFT requested-side depth gap shrinks as realised asymmetry decreases toward symmetry.",
    }
    binary_secondary = {
        "estimand": "within-seed OLS slope of success_RIGHT-success_LEFT versus realised object-layout asymmetry A",
        "mean": binary_bootstrap["mean"],
        "median": float(statistics.median(binary_values)),
        "ci95": binary_bootstrap["ci95"],
        "replicates": binary_bootstrap["replicates"],
        "bootstrap_seed": binary_bootstrap["seed"],
        "unit": binary_bootstrap["unit"],
        "method": binary_bootstrap["method"],
        "sign_test": _sign_test(binary_values),
        "per_seed_by_seed": binary_slopes,
    }

    level_summaries = []
    for level in LEVELS:
        rows = [row for row in contrasts if row["symmetry_level_s"] == level]
        level_summaries.append(
            {
                "symmetry_level_s": level,
                "matched_pairs": len(rows),
                "realised_asymmetry_A_mean": float(np.mean([row["realised_asymmetry_A"] for row in rows])),
                "requested_depth_gap_R_minus_L_mean_m": float(
                    np.mean([row["requested_depth_gap_R_minus_L_m"] for row in rows])
                ),
                "left_successes": sum(bool(row["left_success"]) for row in rows),
                "right_successes": sum(bool(row["right_success"]) for row in rows),
            }
        )

    _reconcile_registered_report(
        documents["results"], documents["full_counts"], contrasts, primary_depth, binary_secondary
    )
    for row in contrasts:
        seed = row["seed"]
        row["depth_slope_m_per_A"] = next(item["slope_m_per_A"] for item in depth_slopes if item["seed"] == seed)
        row["binary_success_slope_per_A"] = binary_slopes[seed]

    activity = {
        "annotation_media": 0,
        "behavioral_actions": 0,
        "behavioral_cells": 0,
        "confirmation_inference_cells": 0,
        "forecast_accuracy_estimates": 0,
        "human_responses": 0,
        "labels": 0,
        "model_requests": 0,
        "queue_descriptors": 0,
        "request_rows": 0,
        "request_selections": 0,
        "retrieval_jobs": 0,
        "runnable_manifests": 0,
    }
    gates = {
        "annotation_media_ready": False,
        "camera_alignment_ready": False,
        "confirmation_release_ready": False,
        "current_GM_PVC_worker_state_reconciled": False,
        "development_annotation_gate_ready": False,
        "human_annotation_ready": False,
        "live_quiescence_inferred": False,
        "resource_freeze_ready": False,
        "safe_to_release_new_work": False,
    }
    return {
        "schema_version": "wmf-retained-h06-graded-symmetry-v1",
        "provenance": dict(provenance),
        "design": {
            "study_id": STUDY,
            "amendment_id": AMENDMENT,
            "model_id": MODEL,
            "matched_seeds": list(SEEDS),
            "inventory_matched_levels": list(LEVELS),
            "excluded_level": 0.0,
            "excluded_level_reason": "Companion objects are absent at s=0 and active at all positive levels; s=0 is not inventory matched.",
            "inferential_unit": "matched seed",
        },
        "authenticated_historical_population": {
            "full_h06_behavioral_episodes": 2_246,
            "full_h06_matched_pairs": 1_123,
            "primary_behavioral_episodes": 216,
            "primary_matched_pairs": 108,
            "primary_matched_seeds": 27,
            "primary_levels": 4,
            "valid_behavioral_failures_included": True,
            "infrastructure_attempts_included": False,
            "missing_value_imputation": "none",
        },
        "primary_depth_dose_response": primary_depth,
        "binary_success_secondary": binary_secondary,
        "by_level": level_summaries,
        "matched_seed_levels": contrasts,
        "reproduction": {
            "full_compact_episode_rows_strictly_parsed": 4_096,
            "full_compact_pair_rows_strictly_parsed": 2_048,
            "primary_episode_rows_rejoined": 216,
            "primary_pair_rows_rejoined": 108,
            "per_seed_slopes_recomputed": 27,
            "registered_bootstraps_exactly_reproduced": True,
            "registered_sign_tests_exactly_reproduced": True,
            "registered_full_level_successes_exactly_reproduced": True,
        },
        "source_manifest": {
            "compact_files_byte_rehashed_now": True,
            "original_raw_episode_and_media_bytes_rehashed_now": False,
            "original_raw_episode_and_media_status": "external hash-bound paths only",
            "authenticated_file_count": len(EXPECTED_SOURCES),
            "authenticated_bytes": sum(item["bytes"] for item in EXPECTED_SOURCES.values()),
        },
        "claim_boundary": {
            "status": "retrospective compact-row authentication and independent arithmetic reproduction",
            "new_model_inference": False,
            "new_robot_episodes": False,
            "new_human_labels": False,
            "forecast_accuracy_evidence": False,
            "confirmation_test_or_stage_gate": False,
            "changes_232_cell_core_bound": False,
            "primary_rows_are_independent_scenes": False,
            "object_layout_symmetry_is_robot_camera_or_embodiment_symmetry": False,
            "position_only_reflection_replication": False,
            "equivalence_claim": False,
            "boundaries": [
                "The 216 primary episodes are repeated commands and levels within 27 matched seed blocks, not 216 independent scenes.",
                "Level zero is excluded prospectively because its object inventory differs from every positive level.",
                "Object-layout symmetry does not assert symmetry of the robot, reset posture, camera rig or embodiment.",
                "The compact Git rows are byte-authenticated; original raw simulator files and media remain external and were not rehashed here.",
                "This is retained executed-behavior evidence, not generated-future accuracy, annotation evidence, confirmation or a release-gate advance.",
                "No equivalence claim passes, and H06 is not a replication or generalization test of position-only reflection.",
            ],
        },
        "new_activity": activity,
        "gate_state": gates,
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    fields = [
        "seed",
        "symmetry_level_s",
        "realised_asymmetry_A",
        "requested_depth_gap_R_minus_L_m",
        "binary_success_gap_R_minus_L",
        "left_success",
        "right_success",
        "depth_slope_m_per_A",
        "binary_success_slope_per_A",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report["matched_seed_levels"]:
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    documents, provenance = read_pinned_sources(ROOT)
    report = analyze_documents(documents, provenance)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(render_json(report), encoding="utf-8")
    args.csv.write_text(render_csv(report), encoding="utf-8")
    print(render_json(report), end="")


if __name__ == "__main__":
    main()
