#!/usr/bin/env python3
"""Compile V3-E005 RoboTwin evidence with the preregistered H4-first gate.

The compiler is deliberately arena-specific.  It accepts only the frozen
LingBot-VA RoboTwin queue, keeps infrastructure failures outside behavioral
denominators, and never imports or pools a DROID success predicate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"
FAILURES = ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")
REQUIRED_RESAMPLES = 20_000
H4_THRESHOLD_M = 0.05


class CompileError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompileError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"V3-E005:{label}".encode()).digest()[:8], "big")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite_number(value: Any, field: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{field} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} is not finite")
    return result


def optional_number(value: Any, field: str) -> float | None:
    return None if value is None else finite_number(value, field)


def percentile(values: Sequence[float], probability: float) -> float:
    require(values, "percentile of empty values")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def wilson(successes: int, trials: int) -> dict[str, Any]:
    if trials == 0:
        return {"successes": 0, "trials": 0, "proportion": None, "wilson95_low": None, "wilson95_high": None}
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return {
        "successes": successes,
        "trials": trials,
        "proportion": p,
        "wilson95_low": max(0.0, center - half),
        "wilson95_high": min(1.0, center + half),
    }


def exact_sign_test(values: Sequence[float], *, tolerance: float = 1e-15) -> dict[str, Any]:
    positive = sum(value > tolerance for value in values)
    negative = sum(value < -tolerance for value in values)
    zero = len(values) - positive - negative
    n = positive + negative
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"positive": positive, "negative": negative, "zero": zero, "exact_two_sided_p": p}


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    require(len(left) == len(right), "McNemar inputs differ in length")
    left_only = sum(bool(l) and not bool(r) for l, r in zip(left, right))
    right_only = sum(bool(r) and not bool(l) for l, r in zip(left, right))
    test = exact_sign_test([1.0] * right_only + [-1.0] * left_only)
    return {
        "left_only_success": left_only,
        "right_only_success": right_only,
        "exact_two_sided_p": test["exact_two_sided_p"],
    }


def _signed_subset_sums(values: Sequence[float]) -> list[float]:
    sums = [0.0]
    for value in values:
        sums = [current + value for current in sums] + [current - value for current in sums]
    return sums


def exact_sign_flip_test(values: Sequence[float], *, tolerance: float = 1e-12) -> dict[str, Any]:
    """Exact two-sided within-seed label-swap test via meet in the middle."""

    active = [float(value) for value in values if abs(float(value)) > tolerance]
    if not active:
        return {"exact_two_sided_p": 1.0, "permutations": 1, "nonzero_seed_effects": 0}
    observed = abs(sum(active))
    split = len(active) // 2
    left = _signed_subset_sums(active[:split])
    right = sorted(_signed_subset_sums(active[split:]))
    # Count |left + right| >= observed without materialising the Cartesian product.
    import bisect

    extreme = 0
    for value in left:
        upper = bisect.bisect_left(right, observed - value - tolerance)
        lower = bisect.bisect_right(right, -observed - value + tolerance)
        extreme += len(right) - upper
        extreme += lower
    total = len(left) * len(right)
    return {
        "exact_two_sided_p": min(1.0, extreme / total),
        "permutations": total,
        "nonzero_seed_effects": len(active),
    }


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    resamples: int,
    confidence: float,
    seed: int,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
) -> dict[str, Any]:
    require(resamples >= REQUIRED_RESAMPLES, "registered compilation requires at least 20,000 resamples")
    require(len(values) == len(clusters) and values, "cluster bootstrap inputs invalid")
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        by_cluster[str(cluster)].append(float(value))
    names = sorted(by_cluster)
    require(len(names) == 7, f"registered E005 analysis requires seven scene clusters, got {len(names)}")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        sampled: list[float] = []
        for _ in names:
            sampled.extend(by_cluster[rng.choice(names)])
        draws.append(float(statistic(sampled)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(statistic(values)),
        "low": percentile(draws, alpha),
        "high": percentile(draws, 1.0 - alpha),
        "confidence": confidence,
        "resamples": resamples,
        "clusters": len(names),
        "cluster_unit": "scene_cluster_id",
        "seed": seed,
        "statistic": "mean" if statistic is statistics.fmean else "median",
    }


def occlusion_clear(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, Mapping):
        return bool(value) and not any(bool(item) for item in value.values())
    return False


def validate_runtime_binding(raw: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    binding = raw.get("runtime_identity_requirement") or raw.get("runtime_identity") or raw.get("runtime_binding")
    if not isinstance(binding, Mapping):
        # The canonical E005 episode is already compiled by the fail-closed
        # runtime bridge.  It carries the runtime-manifest digest plus the
        # identity fields that can be compared directly to the registration.
        direct = {
            "model_id": raw.get("model_id"),
            "checkpoint_revision": raw.get("checkpoint_revision"),
            "checkpoint_manifest_sha256": raw.get("checkpoint_manifest_sha256"),
            "external_repository_commit": raw.get("external_repository_commit"),
            "simulator_repository_commit": raw.get("simulator_repository_commit"),
        }
        for key, value in direct.items():
            require(value == expected.get(key), f"runtime identity mismatch: {key}")
        runtime_digest = raw.get("runtime_identity_sha256")
        require(
            isinstance(runtime_digest, str)
            and len(runtime_digest) == 64
            and all(character in "0123456789abcdef" for character in runtime_digest),
            "raw episode lacks a valid runtime identity digest",
        )
        return {**dict(expected), "runtime_identity_sha256": runtime_digest}
    keys = (
        "model_id",
        "checkpoint_id",
        "checkpoint_revision",
        "checkpoint_manifest_sha256",
        "runtime_payload_sha256",
        "environment_lock_sha256",
        "adapter_contract_sha256",
        "external_repository_commit",
        "simulator_repository_commit",
    )
    for key in keys:
        require(binding.get(key) == expected.get(key), f"runtime identity mismatch: {key}")
    return {key: binding[key] for key in keys}


def normalize_episode(
    raw: Mapping[str, Any],
    queue_row: Mapping[str, Any],
    *,
    source_path: Path,
    source_line: int,
    registration_sha256: str,
    queue_sha256: str,
) -> dict[str, Any]:
    cell_id = str(raw.get("cell_id") or raw.get("registered_cell_id") or "")
    require(cell_id == queue_row["cell_id"], f"raw/queue cell mismatch: {cell_id}")
    require(
        raw.get("schema_version") == "vla-wam-shared-v3e005-lingbot-robotwin-episode-v1",
        f"raw episode schema mismatch: {cell_id}",
    )
    require((raw.get("amendment_id") or "V3-E005") == "V3-E005", f"wrong amendment: {cell_id}")
    require((raw.get("arena") or queue_row["arena"]) == "robotwin", f"non-RoboTwin row rejected: {cell_id}")
    require((raw.get("model_id") or queue_row["model_id"]) == "lingbot_va_robotwin", f"checkpoint drift: {cell_id}")
    for key in ("environment_seed", "sampling_seed", "symmetry_level_s", "relation", "scene_id", "prompt", "prompt_sha256"):
        require(raw.get(key, queue_row[key]) == queue_row[key], f"queue binding mismatch: {cell_id}/{key}")
    require(
        raw.get("matched_layout_pair_id", queue_row["matched_layout_pair_id"])
        == queue_row["matched_layout_pair_id"],
        f"queue binding mismatch: {cell_id}/matched_layout_pair_id",
    )
    if "registration_sha256" in raw:
        require(raw["registration_sha256"] == registration_sha256, f"registration hash mismatch: {cell_id}")
    if "queue_sha256" in raw:
        require(raw["queue_sha256"] == queue_sha256, f"queue hash mismatch: {cell_id}")
    runtime = validate_runtime_binding(raw, queue_row["runtime_identity_requirement"])
    success = raw.get("success")
    require(type(success) is bool, f"success is not boolean: {cell_id}")
    category = str(raw.get("failure_category"))
    require(category in FAILURES, f"unknown failure category: {cell_id}/{category}")
    require((category == "correct") is success, f"success/taxonomy mismatch: {cell_id}")
    action_distinct = raw.get("action_distinct")
    require(action_distinct is None or type(action_distinct) is bool, f"invalid action_distinct: {cell_id}")
    endpoint_shift = optional_number(raw.get("endpoint_shift"), f"{cell_id}/endpoint_shift")
    artifacts = raw.get("source_artifacts") or raw.get("raw_artifacts") or raw.get("artifacts")
    require(isinstance(artifacts, Mapping), f"source artifact bindings unavailable: {cell_id}")
    required_artifacts = {
        "result",
        "trajectory",
        "simulator_viewport_video",
        "executed_action_trace",
        "live_reset_snapshot",
    }
    require(required_artifacts <= set(artifacts), f"source artifact inventory incomplete: {cell_id}")
    for name in required_artifacts:
        artifact = artifacts[name]
        require(isinstance(artifact, Mapping), f"artifact binding malformed: {cell_id}/{name}")
        digest = artifact.get("sha256")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"artifact digest malformed: {cell_id}/{name}",
        )
        require(isinstance(artifact.get("bytes"), int) and artifact["bytes"] > 0, f"artifact bytes invalid: {cell_id}/{name}")
        require(isinstance(artifact.get("path"), str) and artifact["path"], f"artifact path missing: {cell_id}/{name}")
    source = {
        "path": str(source_path.resolve()),
        "bytes": source_path.stat().st_size,
        "sha256": sha256(source_path),
        "line": source_line,
    }
    row = {
        "schema_version": "vla-wam-shared-v3e005-compact-episode-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E005",
        "cell_id": cell_id,
        "matched_seed_id": queue_row["matched_seed_id"],
        "matched_pair_id": queue_row["matched_layout_pair_id"],
        "model_id": "lingbot_va_robotwin",
        "arena": "robotwin",
        "scene_id": queue_row["scene_id"],
        "scene_cluster_id": queue_row["scene_cluster_id"],
        "anchor_task": queue_row["anchor_task"],
        "environment_seed": int(queue_row["environment_seed"]),
        "sampling_seed": int(queue_row["sampling_seed"]),
        "symmetry_level_s": float(queue_row["symmetry_level_s"]),
        "layout": queue_row["layout"],
        "relation": queue_row["relation"],
        "prompt": queue_row["prompt"],
        "prompt_sha256": queue_row["prompt_sha256"],
        "success_predicate_id": "frozen_v3_robotwin_relation_aware_success",
        "outcome_coordinate_contract": queue_row["outcome_coordinate_contract"],
        "registration_sha256": registration_sha256,
        "queue_sha256": queue_sha256,
        "runtime_identity": runtime,
        "success": success,
        "failure_category": category,
        "signed_final_lateral_offset": finite_number(raw.get("signed_final_lateral_offset"), f"{cell_id}/signed_final_lateral_offset"),
        "requested_side_depth": finite_number(raw.get("requested_side_depth"), f"{cell_id}/requested_side_depth"),
        "cone_entry_step": None if raw.get("cone_entry_step") is None else int(raw["cone_entry_step"]),
        "cone_entry_sustained": bool(raw.get("cone_entry_sustained", False)),
        "endpoint_shift": endpoint_shift,
        "action_distinct": action_distinct,
        "episode_length": int(raw.get("episode_length")),
        "time_to_first_contact": optional_number(raw.get("time_to_first_contact"), f"{cell_id}/time_to_first_contact"),
        "grasp_step": None if raw.get("grasp_step") is None else int(raw["grasp_step"]),
        "cumulative_lateral_path": finite_number(raw.get("cumulative_lateral_path"), f"{cell_id}/cumulative_lateral_path"),
        "peak_lateral_excursion": finite_number(raw.get("peak_lateral_excursion"), f"{cell_id}/peak_lateral_excursion"),
        "asymmetry_metric_A": finite_number(raw.get("asymmetry_metric_A"), f"{cell_id}/asymmetry_metric_A"),
        "position_residual": finite_number(raw.get("position_residual"), f"{cell_id}/position_residual"),
        "orientation_residual": finite_number(raw.get("orientation_residual"), f"{cell_id}/orientation_residual"),
        "midline_residual": finite_number(raw.get("midline_residual"), f"{cell_id}/midline_residual"),
        "occlusion_check": raw.get("occlusion_check"),
        "realised_object_poses": raw.get("realised_object_poses"),
        "arm_reset_pose": raw.get("arm_reset_pose"),
        "mirrored_asset_identity_verified": raw.get("mirrored_asset_identity_verified"),
        "mirrored_yaw_verified": raw.get("mirrored_yaw_verified"),
        "object_layout_symmetric_not_robot_or_embodiment": True,
        "missing_measurement_policy": "NR remains null and is never converted to zero",
        "raw_artifacts": dict(artifacts),
        "source_raw_episode": source,
    }
    require(row["episode_length"] >= 0, f"negative episode length: {cell_id}")
    require(isinstance(row["realised_object_poses"], Mapping), f"realised object poses unavailable: {cell_id}")
    require(isinstance(row["arm_reset_pose"], Mapping), f"arm reset pose unavailable: {cell_id}")
    if row["symmetry_level_s"] == 1.0:
        require(row["position_residual"] < 0.001, f"position tolerance failed: {cell_id}")
        require(row["orientation_residual"] < math.radians(0.5), f"orientation tolerance failed: {cell_id}")
        require(row["midline_residual"] < 0.001, f"midline tolerance failed: {cell_id}")
        require(occlusion_clear(row["occlusion_check"]), f"occlusion gate failed: {cell_id}")
        require(row["mirrored_asset_identity_verified"] is True, f"asset identity gate failed: {cell_id}")
        require(row["mirrored_yaw_verified"] is True, f"mirrored yaw gate failed: {cell_id}")
    return row


def source_rows(paths: Sequence[Path]) -> tuple[list[tuple[dict[str, Any], Path, int]], list[dict[str, Any]]]:
    behavioral: list[tuple[dict[str, Any], Path, int]] = []
    invalid: list[dict[str, Any]] = []
    for path in sorted({Path(item).resolve() for item in paths}):
        require(path.is_file(), f"input file missing: {path}")
        if path.name in {"infrastructure_invalid.json", "bridge_failure.json"}:
            value = load_json(path)
            invalid.append({"source": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}, "record": value})
            continue
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                if str(value.get("cell_id") or value.get("registered_cell_id") or "").startswith("v3e005:"):
                    behavioral.append((value, path, line_number))
        else:
            value = load_json(path)
            if str(value.get("cell_id") or value.get("registered_cell_id") or "").startswith("v3e005:"):
                behavioral.append((value, path, 1))
    return behavioral, invalid


def discover_inputs(roots: Sequence[Path]) -> list[Path]:
    names = {
        "raw_episode.jsonl",
        "infrastructure_invalid.json",
        "bridge_failure.json",
    }
    return sorted(path for root in roots for path in Path(root).rglob("*") if path.is_file() and path.name in names)


def build_pairs(episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    # The scientific pairing key is seed + scene + layout level.  Do not rely
    # on directory placement or direction ordering to recover the pair.
    grouped: dict[tuple[int, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in episodes:
        key = (int(row["environment_seed"]), str(row["scene_id"]), float(row["symmetry_level_s"]))
        require(row["relation"] not in grouped[key], f"duplicate direction within pairing key: {key}/{row['relation']}")
        grouped[key][row["relation"]] = row
    pairs: list[dict[str, Any]] = []
    for pairing_key, directions in sorted(grouped.items()):
        if set(directions) != {"left", "right"}:
            continue
        left, right = directions["left"], directions["right"]
        pair_id = left["matched_pair_id"]
        require(left["environment_seed"] == right["environment_seed"], f"pair seed mismatch: {pair_id}")
        require(left["scene_cluster_id"] == right["scene_cluster_id"], f"pair scene mismatch: {pair_id}")
        require(left["matched_pair_id"] == right["matched_pair_id"], f"registered pair id mismatch: {pairing_key}")
        derived_endpoint = left["signed_final_lateral_offset"] - right["signed_final_lateral_offset"]
        supplied_endpoints = [value for value in (left["endpoint_shift"], right["endpoint_shift"]) if value is not None]
        require(
            all(math.isclose(value, derived_endpoint, rel_tol=1e-12, abs_tol=1e-12) for value in supplied_endpoints),
            f"runtime endpoint shift differs from derived matched endpoint: {pair_id}",
        )
        supplied_actions = [value for value in (left["action_distinct"], right["action_distinct"]) if value is not None]
        require(len(set(supplied_actions)) <= 1, f"runtime action-distinct pair mismatch: {pair_id}")
        pairs.append(
            {
                "schema_version": "vla-wam-shared-v3e005-matched-pair-v1",
                "amendment_id": "V3-E005",
                "matched_pair_id": pair_id,
                "matched_seed_id": left["matched_seed_id"],
                "model_id": "lingbot_va_robotwin",
                "arena": "robotwin",
                "scene_id": left["scene_id"],
                "scene_cluster_id": left["scene_cluster_id"],
                "environment_seed": left["environment_seed"],
                "symmetry_level_s": left["symmetry_level_s"],
                "left_cell_id": left["cell_id"],
                "right_cell_id": right["cell_id"],
                "left_success": left["success"],
                "right_success": right["success"],
                "binary_gap_R_minus_L": int(right["success"]) - int(left["success"]),
                "requested_depth_gap_R_minus_L_m": right["requested_side_depth"] - left["requested_side_depth"],
                "endpoint_redirection_LEFT_minus_RIGHT_m": derived_endpoint,
                "action_distinct": supplied_actions[0] if supplied_actions else None,
            }
        )
    return pairs


def h4_analysis(pairs: Sequence[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    passed = True
    for level in (0.0, 1.0):
        rows = [row for row in pairs if row["symmetry_level_s"] == level]
        require(len(rows) == 27, f"H4 requires 27 complete pairs at level {level}")
        values = [row["endpoint_redirection_LEFT_minus_RIGHT_m"] for row in rows]
        clusters = [row["scene_cluster_id"] for row in rows]
        interval = cluster_bootstrap(
            values,
            clusters,
            resamples=resamples,
            confidence=0.95,
            seed=stable_seed(f"H4:{level}:mean"),
        )
        level_pass = interval["point"] > H4_THRESHOLD_M and interval["low"] > 0.0
        passed = passed and level_pass
        levels[f"{level:.2f}"] = {
            "pairs": len(rows),
            "mean_m": statistics.fmean(values),
            "median_m": statistics.median(values),
            "scene_clustered_bootstrap_mean95": interval,
            "exact_sign_test": exact_sign_test(values),
            "threshold_m": H4_THRESHOLD_M,
            "pass": level_pass,
            "seed_level_effects": [
                {"environment_seed": row["environment_seed"], "scene_cluster_id": row["scene_cluster_id"], "effect_m": row["endpoint_redirection_LEFT_minus_RIGHT_m"]}
                for row in rows
            ],
        }
    return {
        "role": "positive_control_hard_gate_evaluated_first",
        "recorded_before_h1_h3": True,
        "threshold_m": H4_THRESHOLD_M,
        "levels": levels,
        "outcome": "pass" if passed else "fail",
        "hard_gate_passed": passed,
        "h1_h3_disposition": "interpretation_enabled" if passed else "withheld_due_h4_failure",
    }


def per_level_analysis(rows: Sequence[dict[str, Any]], *, resamples: int, label: str) -> dict[str, Any]:
    values_depth = [row["requested_depth_gap_R_minus_L_m"] for row in rows]
    clusters = [row["scene_cluster_id"] for row in rows]
    left = [row["left_success"] for row in rows]
    right = [row["right_success"] for row in rows]
    return {
        "pairs": len(rows),
        "left_success": wilson(sum(left), len(left)),
        "right_success": wilson(sum(right), len(right)),
        "binary_gap_R_minus_L": statistics.fmean(row["binary_gap_R_minus_L"] for row in rows),
        "exact_mcnemar": exact_mcnemar(left, right),
        "requested_depth_gap_R_minus_L_m": {
            "mean": statistics.fmean(values_depth),
            "median": statistics.median(values_depth),
            "scene_clustered_bootstrap_mean95": cluster_bootstrap(
                values_depth, clusters, resamples=resamples, confidence=0.95, seed=stable_seed(f"H1:{label}:depth:mean")
            ),
            "scene_clustered_bootstrap_median95": cluster_bootstrap(
                values_depth,
                clusters,
                resamples=resamples,
                confidence=0.95,
                seed=stable_seed(f"H1:{label}:depth:median"),
                statistic=statistics.median,
            ),
            "exact_sign_test": exact_sign_test(values_depth),
        },
    }


def h1_analysis(pairs: Sequence[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    by_seed: dict[int, dict[float, dict[str, Any]]] = defaultdict(dict)
    levels: dict[str, Any] = {}
    for level in (0.0, 1.0):
        rows = [row for row in pairs if row["symmetry_level_s"] == level]
        levels[f"{level:.2f}"] = per_level_analysis(rows, resamples=resamples, label=f"{level:.2f}")
        for row in rows:
            by_seed[int(row["environment_seed"])][level] = row
    require(len(by_seed) == 27 and all(set(item) == {0.0, 1.0} for item in by_seed.values()), "H1 grid incomplete")
    binary: list[float] = []
    depth: list[float] = []
    clusters: list[str] = []
    seed_effects: list[dict[str, Any]] = []
    for seed, item in sorted(by_seed.items()):
        b = item[1.0]["binary_gap_R_minus_L"] - item[0.0]["binary_gap_R_minus_L"]
        d = item[1.0]["requested_depth_gap_R_minus_L_m"] - item[0.0]["requested_depth_gap_R_minus_L_m"]
        binary.append(b)
        depth.append(d)
        clusters.append(item[0.0]["scene_cluster_id"])
        seed_effects.append({"environment_seed": seed, "scene_cluster_id": clusters[-1], "binary_interaction": b, "depth_interaction_m": d})
    return {
        "status": "reported_after_h4_pass",
        "levels": levels,
        "interaction_s1_minus_s0": {
            "binary": {
                "mean": statistics.fmean(binary),
                "median": statistics.median(binary),
                "scene_clustered_bootstrap_mean95": cluster_bootstrap(
                    binary, clusters, resamples=resamples, confidence=0.95, seed=stable_seed("H1:interaction:binary")
                ),
                "exact_within_seed_layout_label_permutation": exact_sign_flip_test(binary),
            },
            "requested_depth_m": {
                "mean": statistics.fmean(depth),
                "median": statistics.median(depth),
                "scene_clustered_bootstrap_mean95": cluster_bootstrap(
                    depth, clusters, resamples=resamples, confidence=0.95, seed=stable_seed("H1:interaction:depth")
                ),
                "exact_within_seed_layout_label_permutation": exact_sign_flip_test(depth),
            },
            "seed_effects": seed_effects,
        },
    }


def h2_analysis(registration: Mapping[str, Any], pairs: Sequence[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    rows = [row for row in pairs if row["symmetry_level_s"] == 1.0]
    clusters = [row["scene_cluster_id"] for row in rows]
    binary = [row["binary_gap_R_minus_L"] for row in rows]
    depth = [row["requested_depth_gap_R_minus_L_m"] for row in rows]
    frozen = registration["predictions"]["H2"]
    binary_ci = cluster_bootstrap(binary, clusters, resamples=resamples, confidence=0.90, seed=stable_seed("H2:binary:90"))
    depth_ci = cluster_bootstrap(depth, clusters, resamples=resamples, confidence=0.90, seed=stable_seed("H2:depth:90"))
    return {
        "status": "reported_after_h4_pass_no_equivalence_claim",
        "nondetection_is_not_equivalence": True,
        "binary": {
            "estimate": statistics.fmean(binary),
            "scene_clustered_bootstrap_mean90": binary_ci,
            "registered": frozen["binary"],
            "publication_equivalence_claim_allowed": False,
            "reason": "zero_margin_tost_undefined_underpowered_no_equivalence_claim",
        },
        "requested_depth_m": {
            "estimate": statistics.fmean(depth),
            "scene_clustered_bootstrap_mean90": depth_ci,
            "registered": frozen["requested_depth_m"],
            "interval_within_margin": depth_ci["low"] > -frozen["requested_depth_m"]["margin"] and depth_ci["high"] < frozen["requested_depth_m"]["margin"],
            "publication_equivalence_claim_allowed": False,
            "reason": "registered_design_underpowered_no_equivalence_claim",
        },
    }


def failure_summary(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "reported_after_h4_pass", "levels": {}}
    for level in (0.0, 1.0):
        level_rows = [row for row in episodes if row["symmetry_level_s"] == level]
        directions: dict[str, Any] = {}
        combined = Counter(row["failure_category"] for row in level_rows)
        failures = sum(count for category, count in combined.items() if category != "correct")
        for relation in ("left", "right"):
            counts = Counter(row["failure_category"] for row in level_rows if row["relation"] == relation)
            denominator = sum(counts.values())
            directions[relation] = {
                "counts": {name: counts[name] for name in FAILURES},
                "row_normalized": {name: counts[name] / denominator for name in FAILURES},
            }
        result["levels"][f"{level:.2f}"] = {
            "directions": directions,
            "failure_only_counts": {name: combined[name] for name in FAILURES if name != "correct"},
            "failure_only_shares": {
                name: (combined[name] / failures if failures else None) for name in FAILURES if name != "correct"
            },
            "failure_episodes": failures,
        }
    result["s1_minus_s0_failure_share"] = {}
    for category in FAILURES[1:]:
        s0 = result["levels"]["0.00"]["failure_only_shares"][category]
        s1 = result["levels"]["1.00"]["failure_only_shares"][category]
        result["s1_minus_s0_failure_share"][category] = None if s0 is None or s1 is None else s1 - s0
    return result


def compile_records(
    registration: Mapping[str, Any],
    queue: Sequence[dict[str, Any]],
    records: Sequence[tuple[dict[str, Any], Path, int]],
    invalid: Sequence[dict[str, Any]],
    *,
    resamples: int,
    require_complete: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    require(resamples >= REQUIRED_RESAMPLES, "registered bootstrap/permutation compilation requires at least 20,000 resamples")
    registration_sha = sha256(BASE / "registration.json")
    queue_sha = sha256(BASE / "queue.jsonl")
    queue_by_id = {row["cell_id"]: row for row in queue}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw, source_path, line in records:
        cell_id = str(raw.get("cell_id") or raw.get("registered_cell_id") or "")
        require(cell_id in queue_by_id, f"unexpected E005 cell: {cell_id}")
        candidates[cell_id].append(
            normalize_episode(
                raw,
                queue_by_id[cell_id],
                source_path=source_path,
                source_line=line,
                registration_sha256=registration_sha,
                queue_sha256=queue_sha,
            )
        )
    episodes: list[dict[str, Any]] = []
    for cell_id, rows in sorted(candidates.items()):
        require(len(rows) == 1, f"ambiguous duplicate behavioral evidence: {cell_id} ({len(rows)} candidates)")
        episodes.append(rows[0])
    pairs = build_pairs(episodes)
    complete = len(episodes) == len(queue) == 108 and len(pairs) == 54
    if require_complete:
        require(complete, f"complete E005 results require 108 cells/54 pairs, got {len(episodes)}/{len(pairs)}")
    analysis_order = ["H4"]
    if complete:
        h4 = h4_analysis(pairs, resamples=resamples)
    else:
        h4 = {
            "role": "positive_control_hard_gate_evaluated_first",
            "recorded_before_h1_h3": True,
            "threshold_m": H4_THRESHOLD_M,
            "outcome": "not_evaluable_incomplete",
            "hard_gate_passed": False,
            "h1_h3_disposition": "withheld_until_complete_h4",
            "levels": {},
        }
    hypotheses: dict[str, Any] = {}
    if complete and h4["hard_gate_passed"]:
        analysis_order.extend(["H1", "H2", "H3"])
        h1 = h1_analysis(pairs, resamples=resamples)
        hypotheses["H1"] = h1
        hypotheses["H2"] = h2_analysis(registration, pairs, resamples=resamples)
        hypotheses["H3"] = failure_summary(episodes)
    else:
        reason = "withheld_due_h4_failure" if complete else "withheld_until_complete_h4"
        for name in ("H1", "H2", "H3"):
            hypotheses[name] = {"status": reason, "estimands_reported": False}
    missing = sorted(set(queue_by_id) - {row["cell_id"] for row in episodes})
    result = {
        "schema_version": "vla-wam-shared-v3e005-results-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E005",
        "status": "complete_hash_closed" if complete else "partial_progress_no_publication_claims",
        "arena": "robotwin",
        "model_id": "lingbot_va_robotwin",
        "arena_boundary": "RoboTwin-only estimands; DROID is never imported or pooled.",
        "registration_sha256": registration_sha,
        "queue_sha256": queue_sha,
        "registered_behavioral_cells": len(queue),
        "valid_behavioral_episodes": len(episodes),
        "complete_matched_pairs": len(pairs),
        "infrastructure_invalid_attempts": len(invalid),
        "coverage": {"complete": complete, "missing_cells": len(missing), "missing_cell_ids": missing},
        "bootstrap_resamples": resamples,
        "scene_cluster_count": 7,
        "scene_cluster_warning": registration["analysis"]["nested_scene_warning"],
        "analysis_order": analysis_order,
        "h4_gate": h4,
        "hypotheses": hypotheses,
        "margins": registration["predictions"]["H2"],
        "publication_claim_status": (
            "h1_h3_interpretation_enabled_subject_to_registered_limits"
            if complete and h4["hard_gate_passed"]
            else "h1_h3_withheld_due_h4_failure"
            if complete
            else "withheld_until_complete_h4"
        ),
        "failure_taxonomy": list(FAILURES),
        "missing_measurement_policy": "NR remains null and is never converted to zero",
    }
    return result, episodes, pairs, list(invalid)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))


def compile_to_directory(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    resamples: int = REQUIRED_RESAMPLES,
    require_complete: bool = False,
) -> dict[str, Any]:
    registration = load_json(BASE / "registration.json")
    queue = load_jsonl(BASE / "queue.jsonl")
    records, invalid = source_rows(inputs)
    result, episodes, pairs, invalid_rows = compile_records(
        registration, queue, records, invalid, resamples=resamples, require_complete=require_complete
    )
    write_json(output_dir / "results.json", result)
    write_jsonl(output_dir / "episodes.jsonl", episodes)
    write_jsonl(output_dir / "pairs.jsonl", pairs)
    write_jsonl(output_dir / "infrastructure_invalid.jsonl", invalid_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--raw-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=BASE / "results")
    parser.add_argument("--bootstrap-resamples", type=int, default=REQUIRED_RESAMPLES)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    inputs = list(args.input_jsonl) + discover_inputs(args.raw_root)
    if not inputs:
        raise SystemExit("no E005 raw inputs supplied")
    try:
        result = compile_to_directory(
            inputs=inputs,
            output_dir=args.output_dir,
            resamples=args.bootstrap_resamples,
            require_complete=args.require_complete,
        )
    except (CompileError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": result["status"],
                "valid_behavioral_episodes": result["valid_behavioral_episodes"],
                "h4_outcome": result["h4_gate"]["outcome"],
                "publication_claim_status": result["publication_claim_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
