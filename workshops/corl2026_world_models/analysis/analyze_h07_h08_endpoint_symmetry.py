#!/usr/bin/env python3
"""Authenticate and reproduce retained H07/H08 endpoint-symmetry evidence.

H07 and H08 are the completed DreamZero-s2 and Cosmos3 Edge slices of the
prospectively registered V3-E004 object-layout study.  For each checkpoint,
27 matched seed blocks cross the reference (s=0) and symmetric (s=1) layouts
with LEFT/RIGHT commands: 108 compact episodes and 54 compact pairs.

This script reads only immutable Git objects.  It byte-authenticates and
strictly parses the complete 4,096-episode/2,048-pair V3-E004 tables, rebuilds
all 216 H07/H08 episodes and 108 pairs, and independently reproduces the
registered s=1 minus s=0 interactions for requested-side depth, binary success
and endpoint redirection.  It runs no model, robot, remote job, retrieval or
annotation workflow and advances no forecast-ablation stage gate.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h07_h08_endpoint_symmetry.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h07_h08_endpoint_symmetry.csv"

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

STUDY = "vla_wam_language_steerability_v3"
AMENDMENT = "V3-E004"
SEEDS = tuple(range(9400, 9427))
LEVELS = (0.0, 1.0)
RELATIONS = ("left", "right")
BOOTSTRAP_REPLICATES = 20_000
MODELS: dict[str, dict[str, str]] = {
    "dreamzero_droid_action_cfg": {
        "cohort": "H07",
        "cell_prefix": "dreamzero",
        "label": "DreamZero-s2",
    },
    "cosmos3_edge_policy_droid": {
        "cohort": "H08",
        "cell_prefix": "edge",
        "label": "Cosmos3 Edge",
    },
}
METRICS = ("binary_gap", "depth_gap_m", "endpoint_redirection_m")
REPORT_LEVEL_KEYS = {
    "binary_gap": "binary_gap_R_minus_L",
    "depth_gap_m": "requested_depth_gap_R_minus_L_m",
    "endpoint_redirection_m": "endpoint_redirection_LEFT_minus_RIGHT_m",
}


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
        row.get("model_id") in MODELS
        and row.get("environment_seed") in SEEDS
        and row.get("symmetry_level_s") in LEVELS
    )


def read_pinned_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Byte-authenticate sources and strictly parse both complete compact tables."""

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
    episode_source_rows: dict[str, list[int]] = {model: [] for model in MODELS}
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
        if _selected(row):
            selected = dict(row)
            selected["_source_row"] = number
            episode_rows.append(selected)
            episode_source_rows[model].append(number)

    pair_rows: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    pair_models: Counter[str] = Counter()
    pair_source_rows: dict[str, list[int]] = {model: [] for model in MODELS}
    for number, raw_line in enumerate(raw_by_path[PAIRS].splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"blank compact pair row at {number}")
        row = parse_json(raw_line, label=f"{PAIRS}:{number}")
        pair_id = row.get("matched_pair_id")
        if not isinstance(pair_id, str) or pair_id in pair_ids:
            raise ValueError(f"missing or duplicate compact pair identity at row {number}")
        pair_ids.add(pair_id)
        model = str(row.get("model_id"))
        pair_models[model] += 1
        if _selected(row):
            selected = dict(row)
            selected["_source_row"] = number
            pair_rows.append(selected)
            pair_source_rows[model].append(number)

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
        },
        "source_rows": {
            model: {
                "episodes": episode_source_rows[model],
                "pairs": pair_source_rows[model],
            }
            for model in MODELS
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


def _sign_summary(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    zero = len(values) - positive - negative
    n = positive + negative
    tail = 1.0 if n == 0 else sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n
    return {
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "exact_two_sided_p": min(1.0, 2.0 * tail),
    }


def _exact_sign_flip(values: Sequence[float], *, tolerance: float = 1e-14) -> dict[str, Any]:
    """Reproduce the exact 2^n layout-label randomization without enumeration."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not 0 < array.size <= 27 or not np.isfinite(array).all():
        raise ValueError("exact sign-flip input must contain 1..27 finite values")
    observed = abs(float(array.sum()))
    n = int(array.size)
    permutations = 2**n
    if observed <= tolerance:
        return {"n": n, "permutations": permutations, "extreme": permutations, "exact_two_sided_p": 1.0}

    def signed_sums(part: np.ndarray) -> list[float]:
        sums = [0.0]
        for value in part:
            number = float(value)
            sums = [item + number for item in sums] + [item - number for item in sums]
        return sums

    split = n // 2
    left = signed_sums(array[:split])
    right = sorted(signed_sums(array[split:]))
    threshold = observed - tolerance
    extreme = sum(
        bisect_right(right, -threshold - item) + len(right) - bisect_left(right, threshold - item)
        for item in left
    )
    return {
        "n": n,
        "permutations": permutations,
        "extreme": extreme,
        "exact_two_sided_p": extreme / permutations,
    }


def _validate_registration(documents: Mapping[str, Any]) -> None:
    registration = documents["registration"]
    if registration.get("study_id") != STUDY or registration.get("amendment_id") != AMENDMENT:
        raise ValueError("H07/H08 registration identity changed")
    if registration.get("status") != "prospectively_registered_zero_e004_model_requests_or_behavioral_episodes":
        raise ValueError("H07/H08 registration is not the frozen pre-inference record")
    if registration.get("behavioral_episode_count_before_registration") != 0 or registration.get("model_request_count_before_registration") != 0:
        raise ValueError("H07/H08 registration was not made before model activity")
    if registration.get("design", {}).get("core_seeds") != list(SEEDS):
        raise ValueError("H07/H08 registered core seeds changed")
    allocations = registration.get("design", {}).get("model_levels_and_pairs", {})
    for model in MODELS:
        if allocations.get(model) != {"0.00": 27, "1.00": 27}:
            raise ValueError(f"registered H07/H08 level allocation changed for {model}")
    expected_estimands = {
        "binary_gap": "Y_i,l = success_RIGHT_i,l - success_LEFT_i,l",
        "endpoint_redirection": "D_i,l = signed_endpoint_LEFT_i,l - signed_endpoint_RIGHT_i,l",
        "interaction": "mean(seed-level estimand at s=1 minus same-seed estimand at s=0)",
        "requested_depth_gap": "B_i,l = (-signed_endpoint_RIGHT_i,l) - signed_endpoint_LEFT_i,l",
    }
    if registration.get("estimands") != expected_estimands:
        raise ValueError("registered H07/H08 estimands changed")
    if registration.get("analysis_plan", {}).get("interaction") != (
        "Seed-matched s1-minus-s0 bootstrap CI plus exact 2^27 within-seed layout-label permutation on the historical core; augmented endpoints are used for TOST/CI only."
    ):
        raise ValueError("registered H07/H08 interaction plan changed")
    inventory = registration.get("layout", {}).get("inventory_transition", {})
    if inventory.get("inventory_transition") is not True or inventory.get("s0") != (
        "companions absent; exact hash-bound B001 control inventory"
    ):
        raise ValueError("registered H07/H08 inventory transition changed")

    manifest = documents["evidence_manifest"]
    required_manifest = {
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
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"H07/H08 evidence manifest changed: {key}")

    counts = documents["full_counts"]
    if counts.get("episodes") != 4_096 or counts.get("pairs") != 2_048:
        raise ValueError("full V3-E004 compact table is incomplete")
    for model in MODELS:
        if counts.get("episodes_by_model", {}).get(model) != 108:
            raise ValueError(f"full compact episode population changed for {model}")
        if counts.get("pairs_by_model", {}).get(model) != 54:
            raise ValueError(f"full compact pair population changed for {model}")

    results = documents["results"]
    required_results = {
        "study_id": STUDY,
        "amendment_id": AMENDMENT,
        "status": "complete_hash_closed",
        "registered_behavioral_cells": 4_096,
        "valid_behavioral_episodes": 4_096,
    }
    for key, expected in required_results.items():
        if results.get(key) != expected:
            raise ValueError(f"registered V3-E004 report changed: {key}")


def _normalize_pairs(documents: Mapping[str, Any]) -> list[dict[str, Any]]:
    episodes = documents["episodes"]
    pairs = documents["pairs"]
    if len(episodes) != 216 or len(pairs) != 108:
        raise ValueError("H07/H08 population must contain exactly 216 episodes and 108 pairs")

    by_cell: dict[tuple[str, int, float, str], Mapping[str, Any]] = {}
    for row in episodes:
        model = row.get("model_id")
        seed = row.get("environment_seed")
        level = row.get("symmetry_level_s")
        relation = row.get("relation")
        key = (model, seed, level, relation)
        if model not in MODELS or seed not in SEEDS or level not in LEVELS or relation not in RELATIONS:
            raise ValueError(f"episode outside H07/H08 grid: {key}")
        if key in by_cell:
            raise ValueError(f"duplicate H07/H08 episode: {key}")
        if row.get("sampling_seed") != seed:
            raise ValueError(f"sampling/environment seed mismatch: {key}")
        level_code = "000" if level == 0.0 else "100"
        pair_id = f"v3e004:{MODELS[model]['cell_prefix']}:seed{seed}:s{level_code}"
        if row.get("matched_pair_id") != pair_id or row.get("registered_cell_id") != f"{pair_id}:{relation}":
            raise ValueError(f"mismatched H07/H08 episode identity: {key}")
        if row.get("study_id") != STUDY or row.get("amendment_id") != AMENDMENT or row.get("arena") != "droid_robolab":
            raise ValueError(f"mismatched H07/H08 source identity: {key}")
        if row.get("requested_relation") != relation or type(row.get("success")) is not bool:
            raise ValueError(f"invalid H07/H08 relation or success value: {key}")
        y = _finite(row.get("signed_final_lateral_offset"), f"{key}.signed endpoint")
        depth = _finite(row.get("requested_side_depth"), f"{key}.requested depth")
        expected_depth = y if relation == "left" else -y
        _close(depth, expected_depth, f"{key}.requested-depth sign convention")
        _finite(row.get("asymmetry_metric_A"), f"{key}.asymmetry")
        for field in (
            "request0_pair_identity_sha256",
            "request0_observation_payload_sha256",
            "request0_reset_contract_sha256",
        ):
            if not isinstance(row.get(field), str) or len(row[field]) != 64:
                raise ValueError(f"invalid paired request identity {field}: {key}")
        by_cell[key] = row

    expected_cells = {
        (model, seed, level, relation)
        for model in MODELS
        for seed in SEEDS
        for level in LEVELS
        for relation in RELATIONS
    }
    if set(by_cell) != expected_cells:
        raise ValueError("H07/H08 episode grid is incomplete")

    by_pair: dict[tuple[str, int, float], Mapping[str, Any]] = {}
    for row in pairs:
        model = row.get("model_id")
        seed = row.get("environment_seed")
        level = row.get("symmetry_level_s")
        key = (model, seed, level)
        if model not in MODELS or seed not in SEEDS or level not in LEVELS or key in by_pair:
            raise ValueError(f"invalid or duplicate H07/H08 compact pair: {key}")
        if row.get("sampling_seed") != seed:
            raise ValueError(f"pair sampling/environment seed mismatch: {key}")
        level_code = "000" if level == 0.0 else "100"
        expected_pair = f"v3e004:{MODELS[model]['cell_prefix']}:seed{seed}:s{level_code}"
        if row.get("matched_pair_id") != expected_pair:
            raise ValueError(f"mismatched H07/H08 compact pair identity: {key}")
        left = by_cell[(model, seed, level, "left")]
        right = by_cell[(model, seed, level, "right")]
        if row.get("left_cell_id") != left.get("registered_cell_id") or row.get("right_cell_id") != right.get("registered_cell_id"):
            raise ValueError(f"H07/H08 pair-to-episode join changed: {key}")
        if row.get("left_success") is not left.get("success") or row.get("right_success") is not right.get("success"):
            raise ValueError(f"H07/H08 compact pair success disagrees with episodes: {key}")
        if row.get("identical_reset") is not True or row.get("action_distinct") is not True:
            raise ValueError(f"H07/H08 paired reset/action contract changed: {key}")
        for field in (
            "request0_pair_identity_sha256",
            "request0_observation_payload_sha256",
            "request0_reset_contract_sha256",
        ):
            values = {row.get(field), left.get(field), right.get(field)}
            if len(values) != 1:
                raise ValueError(f"H07/H08 paired request identity disagrees for {field}: {key}")
        A = 0.5 * (
            _finite(left.get("asymmetry_metric_A"), f"{key}.left.A")
            + _finite(right.get("asymmetry_metric_A"), f"{key}.right.A")
        )
        _close(left.get("asymmetry_metric_A"), A, f"{key}.left.A", tolerance=1e-9)
        _close(right.get("asymmetry_metric_A"), A, f"{key}.right.A", tolerance=1e-9)
        _close(row.get("asymmetry_metric_A"), A, f"{key}.pair.A", tolerance=1e-9)
        endpoint = _finite(left.get("signed_final_lateral_offset"), f"{key}.left endpoint") - _finite(
            right.get("signed_final_lateral_offset"), f"{key}.right endpoint"
        )
        _close(row.get("endpoint_redirection_left_minus_right_m"), endpoint, f"{key}.endpoint")
        _close(row.get("endpoint_shift_right_minus_left_m"), -endpoint, f"{key}.endpoint inverse")
        by_pair[key] = row

    expected_pairs = {
        (model, seed, level) for model in MODELS for seed in SEEDS for level in LEVELS
    }
    if set(by_pair) != expected_pairs:
        raise ValueError("H07/H08 compact pair grid is incomplete")

    normalized: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            for level in LEVELS:
                left = by_cell[(model, seed, level, "left")]
                right = by_cell[(model, seed, level, "right")]
                pair = by_pair[(model, seed, level)]
                normalized.append(
                    {
                        "cohort": MODELS[model]["cohort"],
                        "model_id": model,
                        "seed": seed,
                        "symmetry_level_s": level,
                        "realised_asymmetry_A": _finite(pair.get("asymmetry_metric_A"), "pair A"),
                        "binary_gap": int(right["success"]) - int(left["success"]),
                        "depth_gap_m": _finite(right.get("requested_side_depth"), "right depth")
                        - _finite(left.get("requested_side_depth"), "left depth"),
                        "endpoint_redirection_m": _finite(
                            pair.get("endpoint_redirection_left_minus_right_m"), "endpoint redirection"
                        ),
                        "left_success": left["success"],
                        "right_success": right["success"],
                        "left_episode_source_row": left["_source_row"],
                        "right_episode_source_row": right["_source_row"],
                        "pair_source_row": pair["_source_row"],
                    }
                )
    return normalized


def _reconcile_summary(
    reproduced: Mapping[str, Any], source: Mapping[str, Any], *, label: str
) -> None:
    _close(source.get("mean"), reproduced["mean"], f"{label}.mean")
    _close(source.get("median"), reproduced["median"], f"{label}.median")
    source_bootstrap = source.get("bootstrap_mean95", {})
    if source_bootstrap.get("seed") != reproduced["bootstrap"]["seed"]:
        raise ValueError(f"{label}.bootstrap seed changed")
    if source_bootstrap.get("resamples") != BOOTSTRAP_REPLICATES:
        raise ValueError(f"{label}.bootstrap replicate count changed")
    _close(source_bootstrap.get("point"), reproduced["mean"], f"{label}.bootstrap point")
    _close(source_bootstrap.get("low"), reproduced["bootstrap"]["ci95"][0], f"{label}.bootstrap low")
    _close(source_bootstrap.get("high"), reproduced["bootstrap"]["ci95"][1], f"{label}.bootstrap high")
    if source.get("sign") != reproduced["sign"]:
        raise ValueError(f"{label}.sign summary changed")


def analyze_documents(documents: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    _validate_registration(documents)
    pairs = _normalize_pairs(documents)
    results = documents["results"]
    cohorts: dict[str, Any] = {}
    interaction_rows: list[dict[str, Any]] = []

    for model, model_info in MODELS.items():
        checkpoint = results.get("checkpoints", {}).get(model, {})
        if checkpoint.get("valid_episodes") != 108 or checkpoint.get("complete_pairs") != 54:
            raise ValueError(f"registered H07/H08 report population changed for {model}")
        if checkpoint.get("core_s0_s1_complete") is not True:
            raise ValueError(f"registered s0/s1 core is incomplete for {model}")
        claim_gate = checkpoint.get("claim_gate", {})
        if claim_gate.get("H4_endpoint_positive_control_by_level") != {"0.00": True, "1.00": True}:
            raise ValueError(f"registered endpoint positive-control gate changed for {model}")
        for equivalence in claim_gate.get("equivalence_claims", {}).values():
            if equivalence.get("publication_equivalence_claim_allowed") is not False:
                raise ValueError(f"unexpected H07/H08 equivalence claim for {model}")

        model_pairs = [row for row in pairs if row["model_id"] == model]
        report_analysis = checkpoint.get("analysis", {})
        level_outputs: dict[str, Any] = {}
        for level in LEVELS:
            level_rows = [row for row in model_pairs if row["symmetry_level_s"] == level]
            if len(level_rows) != 27:
                raise ValueError(f"H07/H08 level population changed for {model}:{level}")
            source_level = report_analysis.get("levels", {}).get(f"{level:.2f}", {})
            if source_level.get("pairs") != 27:
                raise ValueError(f"registered H07/H08 pair count changed for {model}:{level}")
            left_successes = sum(row["left_success"] for row in level_rows)
            right_successes = sum(row["right_success"] for row in level_rows)
            if source_level.get("left_success", {}).get("successes") != left_successes:
                raise ValueError(f"registered LEFT success count changed for {model}:{level}")
            if source_level.get("right_success", {}).get("successes") != right_successes:
                raise ValueError(f"registered RIGHT success count changed for {model}:{level}")
            metric_outputs: dict[str, Any] = {}
            for metric in METRICS:
                values = [float(row[metric]) for row in level_rows]
                # Upstream labels are binary/depth/endpoint rather than field names.
                label_suffix = {
                    "binary_gap": "binary",
                    "depth_gap_m": "depth",
                    "endpoint_redirection_m": "endpoint",
                }[metric]
                bootstrap = _bootstrap_mean(values, seed=_seed_from_label(f"{model}:{level:.2f}:{label_suffix}"))
                reproduced = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "bootstrap": bootstrap,
                    "sign": _sign_summary(values),
                }
                _reconcile_summary(
                    reproduced,
                    source_level.get(REPORT_LEVEL_KEYS[metric], {}),
                    label=f"{model}:{level:.2f}:{metric}",
                )
                metric_outputs[metric] = reproduced
            level_outputs[f"{level:.2f}"] = {
                "matched_pairs": 27,
                "left_successes": left_successes,
                "right_successes": right_successes,
                "realised_asymmetry_A_mean": float(
                    np.mean([row["realised_asymmetry_A"] for row in level_rows])
                ),
                "metrics": metric_outputs,
            }

        interaction_outputs: dict[str, Any] = {}
        source_interactions = report_analysis.get("interaction_s1_minus_s0_core", {})
        by_seed_level = {(row["seed"], row["symmetry_level_s"]): row for row in model_pairs}
        for metric in METRICS:
            values = [
                float(by_seed_level[(seed, 1.0)][metric])
                - float(by_seed_level[(seed, 0.0)][metric])
                for seed in SEEDS
            ]
            bootstrap = _bootstrap_mean(
                values, seed=_seed_from_label(f"{model}:interaction:{metric}")
            )
            permutation = _exact_sign_flip(values)
            reproduced = {
                "estimand": f"same-seed {metric} at s=1 minus s=0",
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "bootstrap": bootstrap,
                "exact_layout_label_permutation": permutation,
                "seed_values": [
                    {"seed": seed, "s1_minus_s0": value}
                    for seed, value in zip(SEEDS, values)
                ],
            }
            source = source_interactions.get(metric, {})
            _close(source.get("mean"), reproduced["mean"], f"{model}:{metric}.interaction mean")
            _close(source.get("median"), reproduced["median"], f"{model}:{metric}.interaction median")
            source_bootstrap = source.get("bootstrap_mean95", {})
            if source_bootstrap.get("seed") != bootstrap["seed"] or source_bootstrap.get("resamples") != BOOTSTRAP_REPLICATES:
                raise ValueError(f"registered interaction bootstrap design changed for {model}:{metric}")
            _close(source_bootstrap.get("low"), bootstrap["ci95"][0], f"{model}:{metric}.interaction low")
            _close(source_bootstrap.get("high"), bootstrap["ci95"][1], f"{model}:{metric}.interaction high")
            if source.get("exact_layout_label_permutation") != permutation:
                raise ValueError(f"registered exact interaction test changed for {model}:{metric}")
            source_seed_values = source.get("seed_values")
            if not isinstance(source_seed_values, list) or len(source_seed_values) != len(SEEDS):
                raise ValueError(f"registered interaction seed values incomplete for {model}:{metric}")
            for actual, expected in zip(reproduced["seed_values"], source_seed_values):
                if actual["seed"] != expected.get("seed"):
                    raise ValueError(f"registered interaction seed order changed for {model}:{metric}")
                _close(expected.get("s1_minus_s0"), actual["s1_minus_s0"], f"{model}:{metric}.seed interaction")
            interaction_outputs[metric] = reproduced

        for seed in SEEDS:
            s0 = by_seed_level[(seed, 0.0)]
            s1 = by_seed_level[(seed, 1.0)]
            interaction_rows.append(
                {
                    "cohort": model_info["cohort"],
                    "model_id": model,
                    "seed": seed,
                    "s0_realised_asymmetry_A": s0["realised_asymmetry_A"],
                    "s1_realised_asymmetry_A": s1["realised_asymmetry_A"],
                    "s0_left_success": s0["left_success"],
                    "s0_right_success": s0["right_success"],
                    "s1_left_success": s1["left_success"],
                    "s1_right_success": s1["right_success"],
                    "s0_binary_gap": s0["binary_gap"],
                    "s1_binary_gap": s1["binary_gap"],
                    "binary_gap_s1_minus_s0": s1["binary_gap"] - s0["binary_gap"],
                    "s0_depth_gap_m": s0["depth_gap_m"],
                    "s1_depth_gap_m": s1["depth_gap_m"],
                    "depth_gap_s1_minus_s0_m": s1["depth_gap_m"] - s0["depth_gap_m"],
                    "s0_endpoint_redirection_m": s0["endpoint_redirection_m"],
                    "s1_endpoint_redirection_m": s1["endpoint_redirection_m"],
                    "endpoint_redirection_s1_minus_s0_m": s1["endpoint_redirection_m"]
                    - s0["endpoint_redirection_m"],
                    "s0_pair_source_row": s0["pair_source_row"],
                    "s1_pair_source_row": s1["pair_source_row"],
                }
            )

        source_rows = documents["source_rows"][model]
        cohorts[model_info["cohort"]] = {
            "model_id": model,
            "configuration_label": model_info["label"],
            "authenticated_population": {
                "behavioral_episodes": 108,
                "matched_pairs": 54,
                "matched_seeds": 27,
                "levels": [0.0, 1.0],
                "valid_behavioral_failures_included": True,
                "infrastructure_attempts_included": False,
                "missing_value_imputation": "none",
            },
            "source_rows": {
                "episodes_file": EPISODES,
                "episode_rows": source_rows["episodes"],
                "pairs_file": PAIRS,
                "pair_rows": source_rows["pairs"],
            },
            "levels": level_outputs,
            "interactions_s1_minus_s0": interaction_outputs,
            "registered_claim_gate": {
                "endpoint_positive_control_passed_at_both_levels": True,
                "binary_equivalence_claim_allowed": False,
                "depth_equivalence_claim_allowed": False,
            },
        }

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
        "schema_version": "wmf-retained-h07-h08-endpoint-symmetry-v1",
        "provenance": dict(provenance),
        "design": {
            "study_id": STUDY,
            "amendment_id": AMENDMENT,
            "cohorts": ["H07", "H08"],
            "matched_seeds": list(SEEDS),
            "levels": [0.0, 1.0],
            "inferential_unit": "matched seed block",
            "registered_estimands": documents["registration"]["estimands"],
        },
        "cohorts": cohorts,
        "matched_seed_interactions": interaction_rows,
        "reproduction": {
            "full_compact_episode_rows_strictly_parsed": 4_096,
            "full_compact_pair_rows_strictly_parsed": 2_048,
            "h07_h08_episode_rows_rejoined": 216,
            "h07_h08_pair_rows_rejoined": 108,
            "registered_level_summaries_exactly_reproduced": True,
            "registered_interaction_bootstraps_exactly_reproduced": True,
            "registered_exact_layout_label_tests_exactly_reproduced": True,
            "registered_seed_interactions_exactly_reproduced": True,
        },
        "source_manifest": {
            "compact_files_byte_rehashed_now": True,
            "original_raw_episode_and_media_bytes_rehashed_now": False,
            "original_raw_episode_and_media_status": "external hash-bound paths only",
            "authenticated_file_count": len(EXPECTED_SOURCES),
            "authenticated_bytes": sum(item["bytes"] for item in EXPECTED_SOURCES.values()),
        },
        "claim_boundary": {
            "status": "retrospective compact-row authentication and independent arithmetic reproduction of prospectively registered estimands",
            "new_model_inference": False,
            "new_robot_episodes": False,
            "new_human_labels": False,
            "forecast_accuracy_evidence": False,
            "confirmation_test_or_stage_gate": False,
            "changes_232_cell_core_bound": False,
            "rows_are_independent_scenes": False,
            "object_layout_symmetry_is_robot_camera_or_embodiment_symmetry": False,
            "position_only_reflection_replication": False,
            "equivalence_claim": False,
            "boundaries": [
                "The 216 episodes are repeated commands, levels and checkpoints within 27 matched seed labels, not 216 independent scenes or independent model-noise draws.",
                "The s=0 reference omits companion objects while s=1 activates them; the registered s0-to-s1 intervention has an explicit object-inventory transition.",
                "Object-layout symmetry does not assert symmetry of the robot, reset posture, camera rig or embodiment.",
                "Paired request-zero observation, pair and reset-contract hashes are matched; native full-state hashes can differ and whole-state identity is not claimed.",
                "The compact Git rows are byte-authenticated; original raw simulator files and media remain external and were not rehashed here.",
                "H07 is the historical custom DreamZero-s2 configuration, not official conditional-path D1; H08 is a separate Cosmos3 Edge checkpoint excluded from the new core run matrix.",
                "No binary or depth equivalence claim passes the registered power/interval gates.",
                "This is retained executed-behavior evidence, not generated-future accuracy, annotation evidence, confirmation or a release-gate advance.",
            ],
        },
        "new_activity": activity,
        "gate_state": gates,
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    fields = [
        "cohort",
        "model_id",
        "seed",
        "symmetry_level_s",
        "realised_asymmetry_A",
        "left_success",
        "right_success",
        "binary_gap_R_minus_L",
        "depth_gap_R_minus_L_m",
        "endpoint_redirection_LEFT_minus_RIGHT_m",
        "binary_gap_s1_minus_s0",
        "depth_gap_s1_minus_s0_m",
        "endpoint_redirection_s1_minus_s0_m",
        "pair_source_row",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report["matched_seed_interactions"]:
        for level, prefix in ((0.0, "s0"), (1.0, "s1")):
            writer.writerow(
                {
                    "cohort": row["cohort"],
                    "model_id": row["model_id"],
                    "seed": row["seed"],
                    "symmetry_level_s": level,
                    "realised_asymmetry_A": row[f"{prefix}_realised_asymmetry_A"],
                    "left_success": row[f"{prefix}_left_success"],
                    "right_success": row[f"{prefix}_right_success"],
                    "binary_gap_R_minus_L": row[f"{prefix}_binary_gap"],
                    "depth_gap_R_minus_L_m": row[f"{prefix}_depth_gap_m"],
                    "endpoint_redirection_LEFT_minus_RIGHT_m": row[
                        f"{prefix}_endpoint_redirection_m"
                    ],
                    "binary_gap_s1_minus_s0": row["binary_gap_s1_minus_s0"],
                    "depth_gap_s1_minus_s0_m": row["depth_gap_s1_minus_s0_m"],
                    "endpoint_redirection_s1_minus_s0_m": row[
                        "endpoint_redirection_s1_minus_s0_m"
                    ],
                    "pair_source_row": row[f"{prefix}_pair_source_row"],
                }
            )
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
