"""Pure statistical analysis for completed V3-E004 episode rows."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FAILURE_CATEGORIES = ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")


class AnalysisError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _finite(values: Iterable[Any], label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, f"{label} must be a nonempty vector")
    _require(np.isfinite(array).all(), f"{label} contains nonfinite values")
    return array


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    _require(isinstance(successes, int) and isinstance(trials, int), "Wilson counts must be integers")
    _require(0 <= successes <= trials and trials > 0, "invalid Wilson counts")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return {"successes": successes, "trials": trials, "proportion": p, "wilson95_low": center - half, "wilson95_high": center + half}


def exact_two_sided_binomial(low_count: int, high_count: int) -> float:
    """Two-sided p for a paired sign/McNemar test under p=.5."""

    _require(min(low_count, high_count) >= 0, "negative binomial count")
    n = low_count + high_count
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(low_count, high_count) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: str,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, float | int | str]:
    array = _finite(values, "bootstrap values")
    _require(statistic in {"mean", "median"}, "unsupported bootstrap statistic")
    _require(0.0 < confidence < 1.0 and resamples >= 10_000, "invalid bootstrap design")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, array.size), replace=True)
    distribution = sampled.mean(axis=1) if statistic == "mean" else np.median(sampled, axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(distribution, [alpha, 1.0 - alpha])
    point = float(array.mean()) if statistic == "mean" else float(np.median(array))
    return {"statistic": statistic, "point": point, "confidence": confidence, "low": float(low), "high": float(high), "resamples": resamples, "seed": seed}


def sign_summary(values: Sequence[float], *, tolerance: float = 0.0) -> dict[str, int | float]:
    array = _finite(values, "sign values")
    positive = int(np.sum(array > tolerance))
    negative = int(np.sum(array < -tolerance))
    zero = int(array.size - positive - negative)
    return {"positive": positive, "negative": negative, "zero": zero, "exact_two_sided_p": exact_two_sided_binomial(positive, negative)}


def exact_sign_flip_permutation(values: Sequence[float], *, tolerance: float = 1e-14) -> dict[str, int | float]:
    """Exact paired randomization p without materializing all 2^n sums."""

    array = _finite(values, "permutation values")
    n = int(array.size)
    _require(n <= 27, "exact registered layout-label permutation is limited to the 27-seed core")
    observed = abs(float(array.sum()))
    if observed <= tolerance:
        return {"n": n, "permutations": 2**n, "extreme": 2**n, "exact_two_sided_p": 1.0}

    def signed_sums(part: np.ndarray) -> np.ndarray:
        sums = np.array([0.0], dtype=np.float64)
        for value in part:
            sums = np.concatenate((sums + value, sums - value))
        return sums

    split = n // 2
    left = signed_sums(array[:split])
    right = signed_sums(array[split:])
    extreme = 0
    chunk = 512
    threshold = observed - tolerance
    for start in range(0, left.size, chunk):
        totals = left[start : start + chunk, None] + right[None, :]
        extreme += int(np.count_nonzero(np.abs(totals) >= threshold))
    permutations = 2**n
    return {"n": n, "permutations": permutations, "extreme": extreme, "exact_two_sided_p": extreme / permutations}


def equivalence_summary(values: Sequence[float], *, margin: float, resamples: int, seed: int, power_status: str) -> dict[str, Any]:
    array = _finite(values, "equivalence values")
    _require(margin >= 0.0, "equivalence margin is negative")
    if margin == 0.0:
        return {"margin": 0.0, "status": "margin_zero_equivalence_not_defined", "equivalent": False, "power_status": power_status}
    ci90 = bootstrap_interval(array, statistic="mean", confidence=0.90, resamples=resamples, seed=seed)
    equivalent = bool(ci90["low"] > -margin and ci90["high"] < margin)
    # Empirical one-sided bootstrap tests after centering at each boundary.
    rng = np.random.default_rng(seed + 1)
    centered = array - array.mean()
    draws = rng.choice(centered, size=(resamples, array.size), replace=True).mean(axis=1) + array.mean()
    p_lower = float(np.mean(draws <= -margin))
    p_upper = float(np.mean(draws >= margin))
    return {"margin": margin, "ci90": ci90, "tost_bootstrap_p_lower": p_lower, "tost_bootstrap_p_upper": p_upper, "equivalent": equivalent, "power_status": power_status}


def seed_from_label(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def pair_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, float], dict[str, Mapping[str, Any]]]:
    pairs: dict[tuple[int, float], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["environment_seed"]), float(row["symmetry_level_s"]))
        relation = str(row["relation"])
        _require(relation in {"left", "right"}, "unknown relation")
        _require(relation not in pairs[key], f"duplicate relation for {key}")
        pairs[key][relation] = row
    _require(all(set(pair) == {"left", "right"} for pair in pairs.values()), "incomplete LEFT/RIGHT pair")
    return pairs


def compile_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    margins: Mapping[str, float],
    power_status: Mapping[str, str],
    core_seeds: Sequence[int] = tuple(range(9400, 9427)),
    resamples: int = 20_000,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("model_id") == model_id]
    _require(selected, f"no rows for {model_id}")
    pairs = pair_rows(selected)
    levels = sorted({level for _, level in pairs})
    per_level: dict[str, Any] = {}
    pair_estimands: dict[tuple[int, float], dict[str, float]] = {}
    for level in levels:
        level_pairs = sorted(((seed, pair) for (seed, item_level), pair in pairs.items() if item_level == level))
        left_success = sum(bool(pair["left"]["success"]) for _, pair in level_pairs)
        right_success = sum(bool(pair["right"]["success"]) for _, pair in level_pairs)
        discordant_l_only = sum(bool(pair["left"]["success"]) and not bool(pair["right"]["success"]) for _, pair in level_pairs)
        discordant_r_only = sum(bool(pair["right"]["success"]) and not bool(pair["left"]["success"]) for _, pair in level_pairs)
        binary: list[float] = []
        depth: list[float] = []
        endpoint: list[float] = []
        A_values: list[float] = []
        taxonomy = {relation: Counter() for relation in ("left", "right")}
        for seed, pair in level_pairs:
            left, right = pair["left"], pair["right"]
            y = float(bool(right["success"])) - float(bool(left["success"]))
            b = float(right["requested_side_depth"]) - float(left["requested_side_depth"])
            d = float(left["signed_final_lateral_offset"]) - float(right["signed_final_lateral_offset"])
            A = 0.5 * (float(left["asymmetry_metric_A"]) + float(right["asymmetry_metric_A"]))
            _require(math.isclose(float(left["asymmetry_metric_A"]), float(right["asymmetry_metric_A"]), rel_tol=0.0, abs_tol=1e-9), "A differs within matched pair")
            pair_estimands[(seed, level)] = {"binary_gap": y, "depth_gap_m": b, "endpoint_redirection_m": d, "A": A}
            binary.append(y)
            depth.append(b)
            endpoint.append(d)
            A_values.append(A)
            for relation in ("left", "right"):
                category = str(pair[relation]["failure_category"])
                _require(category in FAILURE_CATEGORIES, f"unknown failure category: {category}")
                taxonomy[relation][category] += 1
        label = f"{model_id}:{level:.2f}"
        per_level[f"{level:.2f}"] = {
            "pairs": len(level_pairs),
            "left_success": wilson(left_success, len(level_pairs)),
            "right_success": wilson(right_success, len(level_pairs)),
            "mcnemar": {"left_only_success": discordant_l_only, "right_only_success": discordant_r_only, "exact_two_sided_p": exact_two_sided_binomial(discordant_l_only, discordant_r_only)},
            "binary_gap_R_minus_L": {"mean": float(np.mean(binary)), "median": float(np.median(binary)), "sign": sign_summary(binary), "bootstrap_mean95": bootstrap_interval(binary, statistic="mean", confidence=0.95, resamples=resamples, seed=seed_from_label(label + ":binary"))},
            "requested_depth_gap_R_minus_L_m": {"mean": float(np.mean(depth)), "median": float(np.median(depth)), "sign": sign_summary(depth), "bootstrap_mean95": bootstrap_interval(depth, statistic="mean", confidence=0.95, resamples=resamples, seed=seed_from_label(label + ":depth"))},
            "endpoint_redirection_LEFT_minus_RIGHT_m": {"mean": float(np.mean(endpoint)), "median": float(np.median(endpoint)), "sign": sign_summary(endpoint), "bootstrap_mean95": bootstrap_interval(endpoint, statistic="mean", confidence=0.95, resamples=resamples, seed=seed_from_label(label + ":endpoint"))},
            "realised_A": {"mean": float(np.mean(A_values)), "min": float(np.min(A_values)), "max": float(np.max(A_values))},
            "failure_taxonomy": {relation: {category: taxonomy[relation][category] for category in FAILURE_CATEGORIES} for relation in ("left", "right")},
        }

    core = set(int(seed) for seed in core_seeds)
    shared = sorted(seed for seed in core if (seed, 0.0) in pair_estimands and (seed, 1.0) in pair_estimands)
    _require(len(shared) == len(core), "core s0/s1 interaction is incomplete")
    interactions: dict[str, Any] = {}
    for estimand in ("binary_gap", "depth_gap_m", "endpoint_redirection_m"):
        values = [pair_estimands[(seed, 1.0)][estimand] - pair_estimands[(seed, 0.0)][estimand] for seed in shared]
        interactions[estimand] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "bootstrap_mean95": bootstrap_interval(values, statistic="mean", confidence=0.95, resamples=resamples, seed=seed_from_label(model_id + ":interaction:" + estimand)),
            "exact_layout_label_permutation": exact_sign_flip_permutation(values),
            "seed_values": [{"seed": seed, "s1_minus_s0": value} for seed, value in zip(shared, values)],
        }

    s1_pairs = [pair_estimands[(seed, 1.0)] for seed, level in pairs if level == 1.0]
    equivalence = {
        "binary_gap": equivalence_summary([pair["binary_gap"] for pair in s1_pairs], margin=float(margins["binary_gap"]), resamples=resamples, seed=seed_from_label(model_id + ":equivalence:binary"), power_status=power_status["binary_gap"]),
        "depth_gap_m": equivalence_summary([pair["depth_gap_m"] for pair in s1_pairs], margin=float(margins["depth_gap_m"]), resamples=resamples, seed=seed_from_label(model_id + ":equivalence:depth"), power_status=power_status["depth_gap_m"]),
    }

    dose_response: dict[str, Any] | None = None
    positive_levels = [level for level in levels if level > 0.0]
    if len(positive_levels) >= 3:
        slope_rows: dict[str, list[float]] = {"binary_gap": [], "depth_gap_m": []}
        seed_rows = []
        for seed in sorted(core):
            if not all((seed, level) in pair_estimands for level in positive_levels):
                continue
            A = np.asarray([pair_estimands[(seed, level)]["A"] for level in positive_levels])
            row = {"seed": seed}
            for estimand in slope_rows:
                values = np.asarray([pair_estimands[(seed, level)][estimand] for level in positive_levels])
                slope = float(np.polyfit(A, values, 1)[0])
                slope_rows[estimand].append(slope)
                row[estimand + "_per_A"] = slope
            seed_rows.append(row)
        dose_response = {"primary_levels": positive_levels, "s0_excluded_due_registered_inventory_transition": True, "seed_slopes": seed_rows}
        for estimand, slopes in slope_rows.items():
            dose_response[estimand] = {"mean_slope": float(np.mean(slopes)), "median_slope": float(np.median(slopes)), "sign": sign_summary(slopes), "bootstrap_mean95": bootstrap_interval(slopes, statistic="mean", confidence=0.95, resamples=resamples, seed=seed_from_label(model_id + ":dose:" + estimand))}

    return {"model_id": model_id, "levels": per_level, "interaction_s1_minus_s0_core": interactions, "equivalence_at_s1": equivalence, "dose_response_on_realised_A": dose_response}
