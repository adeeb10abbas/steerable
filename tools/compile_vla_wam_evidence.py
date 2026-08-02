#!/usr/bin/env python3
"""Compile closed-loop VLA/WAM evidence from a frozen run manifest.

The compiler keeps RoboLab binary success, paper-style final progression, and
the stricter pick-then-place diagnostic separate. It fails closed on missing
episodes unless ``--allow-incomplete`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import beta


TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(a: np.ndarray, b: np.ndarray, steps: int) -> float:
    length = min(len(a), len(b), steps)
    if length == 0:
        raise ValueError("Cannot compute RMS over an empty trajectory")
    return float(np.sqrt(np.mean(np.square(a[:length].astype(np.float64) - b[:length]))))


def _beta_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("Beta interval requires at least one observation")
    return [
        float(beta.ppf(0.025, successes + 1, total - successes + 1)),
        float(beta.ppf(0.975, successes + 1, total - successes + 1)),
    ]


def _initial_arrays(group: h5py.Group) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def collect(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset):
            arrays[name] = np.asarray(item)

    group.visititems(collect)
    return arrays


def _initial_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _relation_mask(cube: np.ndarray, bowl: np.ndarray, direction: str) -> np.ndarray:
    delta = cube - bowl
    horizontal_norm = np.linalg.norm(delta[:, :2], axis=1)
    sign = 1.0 if direction == "left" else -1.0
    cosine = np.divide(
        sign * delta[:, 1],
        horizontal_norm,
        out=np.zeros_like(horizontal_norm),
        where=horizontal_norm > 1e-8,
    )
    return (cosine >= math.cos(math.radians(45.0))) & (np.abs(delta[:, 2]) <= 0.1)


def _first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _load_result_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = root / "episode_results.jsonl"
    if not path.exists():
        return {}
    index = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        key = (value["env_name"], int(value["run"]))
        if key in index:
            raise RuntimeError(f"Duplicate episode_results record for {key} in {path}")
        index[key] = value
    return index


def _event_mentions_cube(event: dict[str, Any]) -> bool:
    return "rubiks_cube" in event.get("info", "").lower()


def _load_episode(
    condition: dict[str, Any], direction: str, run: int, seed: int, result_index: dict
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    root = Path(condition["output_root"])
    task = TASKS[direction]
    task_dir = root / task
    hdf_path = task_dir / f"run_{run}.hdf5"
    log_path = task_dir / f"log_{run}_env0.json"
    if not hdf_path.exists() or not log_path.exists():
        missing = [str(path) for path in (hdf_path, log_path) if not path.exists()]
        raise FileNotFoundError(", ".join(missing))

    with h5py.File(hdf_path, "r") as handle:
        demo = handle["data/demo_0"]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        if "bbox/centroid/rubiks_cube" in demo:
            cube = np.asarray(demo["bbox/centroid/rubiks_cube"], dtype=np.float64)
            bowl = np.asarray(demo["bbox/centroid/bowl"], dtype=np.float64)
        else:
            cube = np.asarray(
                demo["states/rigid_object/rubiks_cube/root_pose"][:, :3], dtype=np.float64
            )
            bowl = np.asarray(
                demo["states/rigid_object/bowl/root_pose"][:, :3], dtype=np.float64
            )
        initial = _initial_arrays(demo["initial_state"])

    log = json.loads(log_path.read_text())
    events = log.get("events", [])
    pickup_events = [
        event
        for event in events
        if event.get("name") == "OBJECT_GRABBED_SUCCESS" and _event_mentions_cube(event)
    ]
    interaction_events = [
        event
        for event in events
        if event.get("name")
        in {"OBJECT_GRABBED_SUCCESS", "GRIPPER_HIT_OBJECT", "OBJECT_BUMPED"}
        and _event_mentions_cube(event)
    ]
    pickup_step = int(pickup_events[0]["step"]) if pickup_events else None
    interaction_step = int(interaction_events[0]["step"]) if interaction_events else None
    relation = _relation_mask(cube, bowl, direction)
    first_relation_step = _first_true(relation)
    final_relation = bool(relation[-1])
    post_pick_transition = False
    if pickup_step is not None:
        start = min(max(pickup_step, 1), len(relation) - 1)
        post_pick_transition = bool(np.any(relation[start:] & ~relation[start - 1 : -1]))

    picked = pickup_step is not None
    binary_success = bool(log["success"])
    paper_progression = (int(picked) + int(binary_success)) / 2.0
    relation_only_progression = (int(picked) + int(final_relation)) / 2.0
    strict_completion = picked and binary_success and post_pick_transition
    strict_progression = (int(picked) + int(strict_completion)) / 2.0
    delta_y = float(cube[-1, 1] - bowl[-1, 1])
    requested_signed_offset = delta_y if direction == "left" else -delta_y
    root_result = result_index.get((task, run), {})
    if not root_result:
        raise RuntimeError(f"Missing episode_results record for {condition['id']} {task} run {run}")
    expected_instruction = condition["expected_instruction"][direction]
    if root_result.get("instruction") != expected_instruction:
        raise RuntimeError(
            f"Instruction mismatch for {condition['id']} {direction} run {run}: "
            f"expected {expected_instruction!r}, got {root_result.get('instruction')!r}"
        )
    if root_result.get("instruction_type") != condition["instruction_type"]:
        raise RuntimeError(
            f"Instruction type mismatch for {condition['id']} {direction} run {run}: "
            f"expected {condition['instruction_type']!r}, "
            f"got {root_result.get('instruction_type')!r}"
        )
    timing = root_result.get("timing", {})
    steps = int(log["final_step"])
    policy_inference_s = timing.get("policy_inference_s")
    estimated_policy_requests = math.ceil(steps / int(condition["open_loop_horizon"]))

    row = {
        "condition_id": condition["id"],
        "model_id": condition["model_id"],
        "model_class": condition["model_class"],
        "wording": condition["wording"],
        "analysis_tier": condition["analysis_tier"],
        "controller": condition["controller"],
        "open_loop_horizon": int(condition["open_loop_horizon"]),
        "direction": direction,
        "run": run,
        "episode_seed": seed,
        "valid": True,
        "binary_success": binary_success,
        "steps": steps,
        "correct_cube_interacted": interaction_step is not None,
        "first_interaction_step": interaction_step,
        "correct_cube_grabbed": picked,
        "pickup_step": pickup_step,
        "final_requested_relation": final_relation,
        "first_requested_relation_step": first_relation_step,
        "post_pick_relation_transition": post_pick_transition,
        "paper_progression": paper_progression,
        "relation_only_progression": relation_only_progression,
        "strict_pick_then_place_progression": strict_progression,
        "final_cube_minus_bowl_y_m": delta_y,
        "requested_signed_final_offset_m": requested_signed_offset,
        "initial_requested_relation": bool(relation[0]),
        "first_recorded_cube_centroid_m": cube[0].tolist(),
        "first_recorded_bowl_centroid_m": bowl[0].tolist(),
        "initial_state_sha256": _initial_fingerprint(initial),
        "raw_robolab_score": root_result.get("score"),
        "policy_inference_s": policy_inference_s,
        "policy_inference_avg_ms": timing.get("policy_inference_avg_ms"),
        "estimated_policy_requests": estimated_policy_requests,
        "mean_policy_request_s": (
            float(policy_inference_s) / estimated_policy_requests
            if policy_inference_s is not None and estimated_policy_requests
            else None
        ),
        "env_step_s": timing.get("env_step_s"),
        "wall_total_s": timing.get("wall_total_s"),
        "command_history": timing.get("command_history"),
        "hdf5_path": str(hdf_path.resolve()),
        "log_path": str(log_path.resolve()),
    }
    return row, actions, initial


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["condition_id"],
            row["model_id"],
            row["model_class"],
            row["wording"],
            row["analysis_tier"],
            row["controller"],
            row["open_loop_horizon"],
            row["direction"],
        )
        groups[key].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        successes = sum(row["binary_success"] for row in group)
        grabbed = sum(row["correct_cube_grabbed"] for row in group)
        relations = sum(row["final_requested_relation"] for row in group)
        strict = sum(row["strict_pick_then_place_progression"] == 1.0 for row in group)
        wall = [row["wall_total_s"] for row in group if row["wall_total_s"] is not None]
        inference = [
            row["policy_inference_avg_ms"]
            for row in group
            if row["policy_inference_avg_ms"] is not None
        ]
        request_latency = [
            row["mean_policy_request_s"]
            for row in group
            if row["mean_policy_request_s"] is not None
        ]
        summaries.append(
            {
                "condition_id": key[0],
                "model_id": key[1],
                "model_class": key[2],
                "wording": key[3],
                "analysis_tier": key[4],
                "controller": key[5],
                "open_loop_horizon": key[6],
                "direction": key[7],
                "successes": successes,
                "episodes": len(group),
                "success_rate": successes / len(group),
                "success_beta11_interval_95": _beta_interval(successes, len(group)),
                "grabbed_count": grabbed,
                "final_relation_count": relations,
                "strict_completion_count": strict,
                "mean_paper_progression": float(
                    np.mean([row["paper_progression"] for row in group])
                ),
                "mean_relation_only_progression": float(
                    np.mean([row["relation_only_progression"] for row in group])
                ),
                "mean_strict_progression": float(
                    np.mean([row["strict_pick_then_place_progression"] for row in group])
                ),
                "mean_requested_signed_final_offset_m": float(
                    np.mean([row["requested_signed_final_offset_m"] for row in group])
                ),
                "median_steps": float(np.median([row["steps"] for row in group])),
                "mean_policy_inference_avg_ms": float(np.mean(inference)) if inference else None,
                "mean_policy_request_s": (
                    float(np.mean(request_latency)) if request_latency else None
                ),
                "mean_wall_total_s": float(np.mean(wall)) if wall else None,
            }
        )
    return summaries


def _action_contrasts(
    manifest: dict[str, Any], actions: dict[tuple[str, str, int], np.ndarray]
) -> list[dict[str, Any]]:
    contrasts = []
    for condition in manifest["conditions"]:
        condition_id = condition["id"]
        seeds = condition["episode_seeds"]
        horizon = int(condition["open_loop_horizon"])
        opposite = [
            _rms(actions[(condition_id, "left", run)], actions[(condition_id, "right", run)], horizon)
            for run in range(len(seeds))
            if (condition_id, "left", run) in actions and (condition_id, "right", run) in actions
        ]
        noise = []
        for direction in TASKS:
            available = [
                actions[(condition_id, direction, run)]
                for run in range(len(seeds))
                if (condition_id, direction, run) in actions
            ]
            noise.extend(_rms(a, b, horizon) for a, b in itertools.combinations(available, 2))
        effect = float(np.mean(opposite)) if opposite else None
        noise_floor = float(np.mean(noise)) if noise else None
        contrasts.append(
            {
                "condition_id": condition_id,
                "model_id": condition["model_id"],
                "wording": condition["wording"],
                "controller": condition["controller"],
                "open_loop_horizon": horizon,
                "paired_opposite_prompt_rms": opposite,
                "mean_opposite_prompt_effect": effect,
                "all_pair_same_prompt_different_seed_and_render_rms": noise,
                "mean_same_prompt_seed_and_render_variation": noise_floor,
                "effect_to_same_prompt_variation_ratio": (
                    effect / noise_floor if effect is not None and noise_floor else None
                ),
            }
        )
    return contrasts


def _write_csv(path: Path, rows: list[dict[str, Any]], excluded: set[str] | None = None) -> None:
    if not rows:
        return
    excluded = excluded or set()
    fieldnames = [key for key in rows[0] if key not in excluded]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VLA-WAM shared benchmark results",
        "",
        f"Manifest SHA-256: `{summary['manifest_sha256']}`.",
        "",
        "## Closed-loop outcomes",
        "",
        "| Tier | Model | Wording | Horizon | Direction | Success | 95% Beta(1,1) | Paper progression | Strict progression | Signed offset |",
        "| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary["group_summaries"]:
        interval = row["success_beta11_interval_95"]
        lines.append(
            f"| {row['analysis_tier']} | {row['model_id']} | {row['wording']} | "
            f"{row['open_loop_horizon']} | {row['direction']} | "
            f"{row['successes']}/{row['episodes']} ({row['success_rate']:.0%}) | "
            f"[{interval[0]:.1%}, {interval[1]:.1%}] | "
            f"{row['mean_paper_progression']:.3f} | "
            f"{row['mean_strict_progression']:.3f} | "
            f"{row['mean_requested_signed_final_offset_m']:+.3f} m |"
        )
    lines.extend(
        [
            "",
            "Paper progression is the mean of persistent correct-cube pickup credit and released-object success in the requested location. Relation-only progress is retained in the machine-readable output as a secondary diagnostic. Strict progression additionally requires a post-pick relation transition.",
            "",
            "## First-action opposite-prompt separation versus same-prompt variation",
            "",
            "| Model/condition | Opposite-prompt RMS | Same-prompt seed + renderer RMS | Ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["action_contrasts"]:
        effect = row["mean_opposite_prompt_effect"]
        noise = row["mean_same_prompt_seed_and_render_variation"]
        ratio = row["effect_to_same_prompt_variation_ratio"]
        lines.append(
            f"| {row['model_id']} / {row['condition_id']} | "
            f"{effect:.5f} | {noise:.5f} | {ratio:.3f} |"
            if effect is not None and noise is not None and ratio is not None
            else f"| {row['model_id']} / {row['condition_id']} | insufficient | insufficient | insufficient |"
        )
    lines.extend(
        [
            "",
            "Realtime rendering was not pixel-repeatable across resets despite one exact simulator-state fingerprint. Opposite-prompt and same-prompt first-action distances therefore both include renderer variation. The ratio is a sensitivity diagnostic, not an isolated causal language effect; the frozen-observation probe supplies that test.",
        ]
    )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "These estimates apply to the two pinned checkpoints in this shared spatial task. They do not establish a VLA-versus-WAM class difference.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected_episodes = int(manifest["expected_episode_count"])
    if any(condition["controller"] != "static" for condition in manifest["conditions"]):
        raise RuntimeError("Direct-language manifest contains a non-static controller")
    declared_episodes = sum(
        len(TASKS) * len(condition["episode_seeds"])
        for condition in manifest["conditions"]
    )
    if declared_episodes != expected_episodes:
        raise RuntimeError(
            f"Manifest declares {declared_episodes} episodes but expects {expected_episodes}"
        )
    rows: list[dict[str, Any]] = []
    actions: dict[tuple[str, str, int], np.ndarray] = {}
    missing: list[dict[str, Any]] = []
    initial_fingerprints: dict[str, int] = Counter()
    for condition in manifest["conditions"]:
        result_index = _load_result_index(Path(condition["output_root"]))
        expected_result_keys = {
            (task, run)
            for task in TASKS.values()
            for run in range(len(condition["episode_seeds"]))
        }
        extra_result_keys = sorted(set(result_index) - expected_result_keys)
        if extra_result_keys:
            raise RuntimeError(
                f"Unexpected completed episodes in {condition['id']}: {extra_result_keys}"
            )
        for direction in TASKS:
            for run, seed in enumerate(condition["episode_seeds"]):
                try:
                    row, action, _ = _load_episode(
                        condition, direction, run, int(seed), result_index
                    )
                except FileNotFoundError as error:
                    missing.append(
                        {
                            "condition_id": condition["id"],
                            "direction": direction,
                            "run": run,
                            "episode_seed": int(seed),
                            "error": str(error),
                        }
                    )
                    continue
                if row["initial_requested_relation"]:
                    raise RuntimeError(
                        f"Non-neutral initial state for {condition['id']} {direction} run {run}"
                    )
                rows.append(row)
                actions[(condition["id"], direction, run)] = action
                initial_fingerprints[row["initial_state_sha256"]] += 1
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Missing {len(missing)} registered episodes; rerun or pass --allow-incomplete"
        )
    if not missing and len(rows) != expected_episodes:
        raise RuntimeError(
            f"Compiled {len(rows)} episodes but manifest expects {expected_episodes}"
        )

    tier_counts = Counter(row["analysis_tier"] for row in rows)

    summary = {
        "schema_version": 1,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "complete": not missing,
        "expected_episode_count": expected_episodes,
        "episode_count": len(rows),
        "analysis_tier_episode_counts": dict(tier_counts),
        "direct_task_language_only": True,
        "missing_episodes": missing,
        "initial_state_fingerprint_counts": dict(initial_fingerprints),
        "paper_progression_operationalization": (
            "persistent correct-cube OBJECT_GRABBED_SUCCESS plus RoboLab requested-side "
            "success, which requires the final 45-degree relation and gripper detachment; "
            "each item contributes one half"
        ),
        "relation_only_progression_operationalization": (
            "persistent correct-cube pickup plus final requested relation regardless of "
            "gripper detachment; declared secondary diagnostic"
        ),
        "strict_progression_operationalization": (
            "pickup credit plus a requested-relation false-to-true transition after pickup "
            "that remains true at the final recorded state"
        ),
        "group_summaries": _group_summary(rows),
        "action_contrasts": _action_contrasts(manifest, actions),
        "episodes": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(args.output_dir / "closed_loop_summary.json", summary)
    _write_csv(
        args.output_dir / "episodes.csv", rows, excluded={"command_history"}
    )
    _write_csv(
        args.output_dir / "group_summaries.csv",
        summary["group_summaries"],
        excluded={"success_beta11_interval_95"},
    )
    (args.output_dir / "CLOSED_LOOP_RESULTS.md").write_text(_markdown(summary))
    print(
        f"compiled {len(rows)} episodes; missing={len(missing)}; output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
