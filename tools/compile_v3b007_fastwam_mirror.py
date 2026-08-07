#!/usr/bin/env python3
"""Compile the complete FastWAM RoboTwin V3-B007 mirror cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


QUEUE_SHA256 = "2ffe2f99e4d6c4b3d80c24fab7276b21bb83de86d92b8a3438ce38a7ba9e1ae3"
MODEL_BLIND_GATE_SHA256 = "e092917893591490f1b1ee2ab2f9c6bd4cd9cc560fa5702d49dd6974a301d6ad"
MODEL_SPECIFIC_GATE_SHA256 = "798bad08e76a6b313de370c75b67e78c5bb70d62441ee53bc7de988ee3b5e766"
REGISTRATION_SHA256 = "84d14a5c6a02c5f6655384d2ed1ef6e3cdaab05341136d81a3b0e727268ecc8e"
PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
SEEDS = tuple(range(9900, 9927))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model-specific-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def relation_region(step: dict, relation: str) -> bool:
    dx = float(step["object_minus_target_x"])
    dy = float(step["object_minus_target_y"])
    distance = math.hypot(dx, dy)
    side = dx < 0.0 if relation == "left" else dx > 0.0
    return bool(0.08 < distance < 0.20 and side and abs(dy) < 0.05)


def first_sustained(values: list[bool], length: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run == length:
            return index - length + 1
    return None


def final_sustained(values: list[bool], length: int = 3) -> bool:
    return len(values) >= length and all(values[-length:])


def taxonomy(result: dict, trajectory: list[dict]) -> tuple[str, dict]:
    relation = result["requested_relation"]
    opposite = "right" if relation == "left" else "left"
    initial_z = float(trajectory[0]["object_xyz"][2])
    pickup = [float(step["object_xyz"][2]) >= initial_z + 0.03 for step in trajectory]
    requested = [relation_region(step, relation) for step in trajectory]
    opposite_mask = [relation_region(step, opposite) for step in trajectory]
    first_pickup = first_sustained(pickup)
    first_requested = first_sustained(requested)
    first_opposite = first_sustained(opposite_mask)
    if result["requested_success"]:
        category = "correct"
    elif first_pickup is None:
        category = "pick_failed"
    elif final_sustained(opposite_mask):
        category = "wrong_side"
    elif final_sustained(requested) and not bool(trajectory[-1]["grippers_open"]):
        category = "release_failed"
    else:
        category = "transport_failed"
    positions = np.asarray([step["object_xyz"] for step in trajectory], dtype=np.float64)
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0
    measures = {
        "verified_pickup": first_pickup is not None,
        "first_verified_pickup_step": first_pickup,
        "first_sustained_requested_entry_step": first_requested,
        "first_sustained_opposite_entry_step": first_opposite,
        "final_requested_region_sustained": final_sustained(requested),
        "final_opposite_region_sustained": final_sustained(opposite_mask),
        "final_detached_release": bool(trajectory[-1]["grippers_open"]),
        "maximum_pickup_height_m": float(positions[:, 2].max() - initial_z),
        "object_path_length_m": path_length,
    }
    return category, measures


def stable_rng(namespace: str) -> tuple[np.random.Generator, int]:
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
    return np.random.default_rng(seed), seed


def sign_test(values: list[float]) -> dict:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {
        "method": "exact_two_sided_paired_sign_test",
        "positive": positive,
        "negative": negative,
        "ties_excluded": len(values) - n,
        "effective_n": n,
        "p_value": p_value,
    }


def summarize(values: list[float], namespace: str, replicates: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng, seed = stable_rng(namespace)
    samples = rng.integers(0, len(array), size=(replicates, len(array)))
    boot = array[samples].mean(axis=1)
    return {
        "n": len(values),
        "mean_m": float(array.mean()),
        "median_m": float(np.median(array)),
        "sample_standard_deviation_m": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum_m": float(array.min()),
        "maximum_m": float(array.max()),
        "mean_bootstrap_95": {
            "method": "matched_seed_nonparametric_percentile_bootstrap",
            "replicates": replicates,
            "seed": seed,
            "lower": float(np.quantile(boot, 0.025)),
            "upper": float(np.quantile(boot, 0.975)),
            "confidence": 0.95,
        },
        "paired_sign_test": sign_test(values),
    }


def exact_sign_flip_test(values: list[int]) -> dict:
    observed = abs(sum(values))
    counts = Counter({0: 1})
    for value in values:
        next_counts: Counter[int] = Counter()
        magnitude = abs(int(value))
        for total, count in counts.items():
            next_counts[total + magnitude] += count
            next_counts[total - magnitude] += count
        counts = next_counts
    denominator = 2 ** len(values)
    extreme = sum(count for total, count in counts.items() if abs(total) >= observed)
    return {
        "method": "exact_within_seed_layout_label_sign_flip_permutation",
        "statistic": "absolute_sum_of_success_difference_in_differences",
        "observed_sum": sum(values),
        "assignments": denominator,
        "extreme_assignments": extreme,
        "p_value": extreme / denominator,
    }


def action_pair(left_path: Path, right_path: Path) -> dict:
    with np.load(left_path) as payload:
        left = np.asarray(payload["executed"], dtype=np.float64)
    with np.load(right_path) as payload:
        right = np.asarray(payload["executed"], dtype=np.float64)
    count = min(10, len(left), len(right))
    rms = float(np.sqrt(np.mean(np.square(left[:count] - right[:count])))) if count else None
    return {
        "first_actions_compared": count,
        "first_10_action_rms": rms,
        "action_distinct_first_10": bool(rms is not None and rms > 0.0),
    }


def main() -> None:
    args = parse_args()
    study = args.study_root.resolve()
    raw = args.raw_root.resolve()
    output = args.output_dir.resolve()
    gate_path = args.model_specific_gate.resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    registration = study / "artifacts/vla_wam_shared_v3/prospective_tier_b/fastwam_robotwin_mirror_v3b007.json"
    queue_path = study / "artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3b007/v3b007_cells.jsonl"
    blind_gate = study / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b007/model_blind_gate_report.json"
    for path, expected in ((registration, REGISTRATION_SHA256), (queue_path, QUEUE_SHA256), (blind_gate, MODEL_BLIND_GATE_SHA256), (gate_path, MODEL_SPECIFIC_GATE_SHA256)):
        require(sha256(path) == expected, f"frozen input hash mismatch: {path}")
    rows = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
    require(len(rows) == 108, "released queue must contain 108 cells")
    by_cell = {row["cell_id"]: row for row in rows}
    require(len(by_cell) == 108, "released cell IDs are not unique")
    runtime_manifest = json.loads((raw / "manifest.json").read_text())
    require(runtime_manifest.get("behavioral_episode_count") == 108 and runtime_manifest.get("whole_seeds_complete") == 27, "runtime queue is not complete")
    for seed in SEEDS:
        marker = json.loads((raw / f"seed_{seed}_complete.json").read_text())
        require(marker.get("status") == "complete_four_valid_behavioral_cells" and marker.get("behavioral_episode_count") == 4, f"seed {seed} is incomplete")

    episodes = []
    evidence_files = [registration, queue_path, blind_gate, gate_path, raw / "manifest.json", raw / "behavioral_episodes.jsonl", raw / "queue_progress.jsonl"]
    indexed: dict[tuple[int, str, str], dict] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for relation in RELATIONS:
                task_name = f"v3b007_{arm}"
                condition = f"{arm}__{relation}"
                condition_dir = raw / task_name / f"environment_seed_{seed}" / f"sampling_seed_{seed}" / condition
                result_path = condition_dir / "result.json"
                trajectory_path = condition_dir / "trajectory.json"
                action_path = condition_dir / "action_trace.npz"
                video_path = condition_dir / "simulator.mp4"
                for path in (result_path, trajectory_path, action_path, video_path):
                    require(path.is_file() and path.stat().st_size > 0, f"missing raw artifact: {path}")
                    evidence_files.append(path)
                result = json.loads(result_path.read_text())
                trajectory = json.loads(trajectory_path.read_text())
                cell_id = f"v3b007:fastwam:robotwin:seed{seed}:{arm}:{relation}"
                queue = by_cell[cell_id]
                require(result.get("prompt") == PROMPTS[relation] == queue["prompt"], f"prompt drift: {cell_id}")
                require(result.get("requested_relation") == relation and result.get("sampling_seed") == seed and result.get("environment_seed") == seed, f"seed/relation drift: {cell_id}")
                require(len(trajectory) == result["actions_executed"] + 1 and trajectory[-1]["success"] == result["requested_success"], f"trajectory/result mismatch: {cell_id}")
                with np.load(action_path) as payload:
                    actions = np.asarray(payload["executed"])
                require(list(actions.shape) == result["action_trace"]["shape"] and len(actions) == result["actions_executed"], f"action trace mismatch: {cell_id}")
                require(sha256(action_path) == result["action_trace"]["sha256"], f"action trace hash mismatch: {cell_id}")
                category, measures = taxonomy(result, trajectory)
                source_dx = float(trajectory[-1]["object_minus_target_x"])
                signed_lateral = -source_dx
                requested_depth = signed_lateral if relation == "left" else -signed_lateral
                episode = {
                    "schema_version": "vla-wam-shared-v3b007-compact-episode-v1",
                    "study_id": "vla_wam_language_steerability_v3",
                    "amendment_id": "V3-B007",
                    "arena": "robotwin",
                    "model_id": "fastwam_robotwin",
                    "registered_cell_id": cell_id,
                    "seed": seed,
                    "arm": arm,
                    "requested_relation": relation,
                    "prompt": PROMPTS[relation],
                    "requested_success": bool(result["requested_success"]),
                    "failure_category": category,
                    "actions_executed": int(result["actions_executed"]),
                    "right_censored": bool(not result["requested_success"] and result["actions_executed"] == 400),
                    "signed_final_lateral_offset_m": signed_lateral,
                    "requested_side_depth_m": requested_depth,
                    "source_sapien_object_minus_reference_x_m": source_dx,
                    **measures,
                    "artifacts": {
                        "source_result": record(result_path),
                        "source_trajectory": record(trajectory_path),
                        "executed_action_trace": record(action_path),
                        "viewport_video": record(video_path),
                    },
                    "future_evidence": [],
                    "future_interface": "action_only_no_decodable_future",
                }
                episodes.append(episode)
                indexed[(seed, arm, relation)] = {"episode": episode, "action_path": action_path, "initial_hash": result["v3b007"]["initial_physical_fingerprint_sha256"]}

    pairs = []
    metrics: dict[str, list[float]] = defaultdict(list)
    success_did = []
    for seed in SEEDS:
        for arm in ARMS:
            left = indexed[(seed, arm, "left")]
            right = indexed[(seed, arm, "right")]
            require(left["initial_hash"] == right["initial_hash"], f"matched reset drift: seed {seed} {arm}")
            s_left = left["episode"]["signed_final_lateral_offset_m"]
            s_right = right["episode"]["signed_final_lateral_offset_m"]
            redirection = s_left - s_right
            depth_gap = (-s_right) - s_left
            action = action_pair(left["action_path"], right["action_path"])
            pair = {
                "schema_version": "vla-wam-shared-v3b007-matched-pair-v1",
                "seed": seed,
                "arm": arm,
                "endpoint_redirection_left_minus_right_m": redirection,
                "right_minus_left_requested_side_depth_m": depth_gap,
                "left_success": left["episode"]["requested_success"],
                "right_success": right["episode"]["requested_success"],
                **action,
            }
            pairs.append(pair)
            metrics[f"redirection:{arm}"].append(redirection)
            metrics[f"depth_gap:{arm}"].append(depth_gap)
        control = {relation: indexed[(seed, "control", relation)]["episode"] for relation in RELATIONS}
        reflected = {relation: indexed[(seed, "position_mirrored", relation)]["episode"] for relation in RELATIONS}
        metrics["redirection_interaction"].append(
            (reflected["left"]["signed_final_lateral_offset_m"] - reflected["right"]["signed_final_lateral_offset_m"])
            - (control["left"]["signed_final_lateral_offset_m"] - control["right"]["signed_final_lateral_offset_m"])
        )
        metrics["depth_interaction"].append(
            ((-reflected["right"]["signed_final_lateral_offset_m"]) - reflected["left"]["signed_final_lateral_offset_m"])
            - ((-control["right"]["signed_final_lateral_offset_m"]) - control["left"]["signed_final_lateral_offset_m"])
        )
        did = (
            int(reflected["right"]["requested_success"])
            - int(reflected["left"]["requested_success"])
            - int(control["right"]["requested_success"])
            + int(control["left"]["requested_success"])
        )
        success_did.append(did)

    episodes_path = output / "fastwam_v3b007_episodes.jsonl"
    episodes_path.write_bytes(b"".join(canonical(row) for row in episodes))
    pairs_path = output / "fastwam_v3b007_pairs.jsonl"
    pairs_path.write_bytes(b"".join(canonical(row) for row in pairs))
    condition_outcomes = {}
    for arm in ARMS:
        for relation in RELATIONS:
            selected = [row for row in episodes if row["arm"] == arm and row["requested_relation"] == relation]
            condition_outcomes[f"{arm}:{relation}"] = {
                "episodes": len(selected),
                "successes": sum(row["requested_success"] for row in selected),
                "failure_taxonomy_counts": dict(sorted(Counter(row["failure_category"] for row in selected).items())),
                "mean_signed_final_lateral_offset_m": float(np.mean([row["signed_final_lateral_offset_m"] for row in selected])),
                "mean_requested_side_depth_m": float(np.mean([row["requested_side_depth_m"] for row in selected])),
            }
    summary = {
        "schema_version": "vla-wam-shared-v3b007-fastwam-mirror-summary-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B007",
        "model_id": "fastwam_robotwin",
        "arena": "robotwin",
        "status": "complete_27_matched_seeds_108_valid_episodes",
        "population": {"matched_seeds": 27, "behavioral_episodes": 108, "valid_failures_included": True, "infrastructure_attempts_included": False},
        "exact_prompts": PROMPTS,
        "condition_outcomes": condition_outcomes,
        "failure_taxonomy_counts": dict(sorted(Counter(row["failure_category"] for row in episodes).items())),
        "full_sample_primary": {
            "endpoint_redirection_by_arm": {arm: summarize(metrics[f"redirection:{arm}"], f"V3-B007:redirection:{arm}", args.bootstrap_replicates) for arm in ARMS},
            "requested_side_depth_gap_by_arm": {arm: summarize(metrics[f"depth_gap:{arm}"], f"V3-B007:depth:{arm}", args.bootstrap_replicates) for arm in ARMS},
            "endpoint_redirection_interaction": summarize(metrics["redirection_interaction"], "V3-B007:redirection_interaction", args.bootstrap_replicates),
            "requested_side_depth_interaction": summarize(metrics["depth_interaction"], "V3-B007:depth_interaction", args.bootstrap_replicates),
        },
        "binary_success_difference_in_differences": {
            "distribution": {str(value): success_did.count(value) for value in range(-2, 3)},
            "mean": float(np.mean(success_did)),
            "median": float(np.median(success_did)),
            "exact_permutation_test": exact_sign_flip_test(success_did),
        },
        "action_sensitivity": {
            "pairs_with_distinct_first_10_actions": sum(row["action_distinct_first_10"] for row in pairs),
            "pair_count": len(pairs),
        },
        "formulas": {
            "s": "negative native SAPIEN object-minus-reference X; positive is robot LEFT",
            "endpoint_redirection": "s_LEFT - s_RIGHT within arm and seed",
            "requested_side_depth_gap": "(-s_RIGHT) - s_LEFT within arm and seed",
            "interaction": "position_mirrored minus control",
        },
        "claim_boundary": "This is a RoboTwin negative-control mirror at one FastWAM checkpoint and one fixed pair03 scene. It is never pooled with DROID and cannot establish absence of geometry effects outside this checkpoint/arena.",
        "behavioral_evidence": {"episodes": record(episodes_path), "pairs": record(pairs_path)},
    }
    summary_path = output / "fastwam_v3b007_summary.json"
    summary_path.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n")

    if (raw / "infrastructure_failures.jsonl").is_file():
        infrastructure = {
            "schema_version": "vla-wam-shared-v3b007-infrastructure-ledger-v1",
            "behavioral_denominator_contribution": 0,
            "attempt_stream": record(raw / "infrastructure_failures.jsonl"),
            "classification": "wrapper-only partial seed attempts excluded until the same whole seed later completed",
        }
    else:
        infrastructure = {"schema_version": "vla-wam-shared-v3b007-infrastructure-ledger-v1", "behavioral_denominator_contribution": 0, "attempt_stream": None}
    infrastructure_path = output / "infrastructure_ledger.json"
    infrastructure_path.write_text(json.dumps(infrastructure, allow_nan=False, indent=2, sort_keys=True) + "\n")
    if (raw / "thermal_events_remaining.jsonl").is_file():
        evidence_files.append(raw / "thermal_events_remaining.jsonl")
    for name in ("runtime_interventions_fastwam_robotwin.json", "invalid_attempts_fastwam_robotwin.json"):
        if (raw / name).is_file():
            evidence_files.append(raw / name)
    evidence = {
        "schema_version": "vla-wam-shared-v3b007-evidence-hash-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B007",
        "arena": "robotwin",
        "raw_file_count": len(evidence_files),
        "raw_files": [record(path) for path in sorted(set(evidence_files))],
        "compact_files": [record(path) for path in (episodes_path, pairs_path, summary_path, infrastructure_path)],
        "denominator_boundary": "Exactly 108 valid RoboTwin episodes; no DROID pooling.",
    }
    evidence_path = output / "evidence_hash_manifest.json"
    evidence_path.write_text(json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": record(summary_path), "episodes": record(episodes_path), "pairs": record(pairs_path), "evidence": record(evidence_path)}, indent=2))


if __name__ == "__main__":
    main()
