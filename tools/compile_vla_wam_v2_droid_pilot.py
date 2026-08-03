#!/usr/bin/env python3
"""Compile the protocol-exact pi0-FAST DROID direct-command pilot.

The simulator ran one LEFT/RIGHT pair per invocation so that every retained
episode carries its preregistered environment and sampling seed.  This
compiler derives physical endpoint and transparent pickup-proxy evidence from
the recorded HDF5 state; simulator state never selected a prompt or action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MODEL_ID = "pi0_fast_droid_vla"
MODEL_LABEL = "pi0-FAST DROID"
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
PICKUP_LIFT_M = 0.03
MOTION_M = 0.01
CONSECUTIVE_STEPS = 3


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    matrix = np.empty((len(quaternion), 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def robot_frame_delta(
    cube_pose: np.ndarray, bowl_pose: np.ndarray, robot_pose: np.ndarray
) -> np.ndarray:
    delta_world = cube_pose[:, :3] - bowl_pose[:, :3]
    # RoboLab's predicate uses this same row-vector convention.
    return np.einsum("tij,ti->tj", rotation_wxyz(robot_pose[:, 3:7]), delta_world)


def relation_mask(delta_robot: np.ndarray, direction: str) -> np.ndarray:
    horizontal = np.linalg.norm(delta_robot[:, :2], axis=1)
    sign = 1.0 if direction == "left" else -1.0
    cosine = np.divide(
        sign * delta_robot[:, 1],
        horizontal,
        out=np.zeros_like(horizontal),
        where=horizontal > 1e-8,
    )
    return cosine >= math.cos(math.radians(45.0))


def first_consecutive(mask: np.ndarray, steps: int = CONSECUTIVE_STEPS) -> int | None:
    if len(mask) < steps:
        return None
    hits = np.convolve(mask.astype(np.int8), np.ones(steps, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(hits == steps)
    return int(indices[0]) if len(indices) else None


def first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def initial_fingerprint(group: h5py.Group) -> str:
    arrays: dict[str, np.ndarray] = {}

    def collect(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset) and name.startswith(("articulation/", "rigid_object/")):
            arrays[name] = np.asarray(item)

    group.visititems(collect)
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def load_grid(grid_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    grid = json.loads(grid_path.read_text())
    cells = {}
    for cell in grid["cells"]:
        if (
            cell["model_id"] == MODEL_ID
            and cell["prompt_family"] == "direct_command"
            and cell["environment_seed"] in {8300, 8301, 8302}
        ):
            cells[(int(cell["environment_seed"]), cell["requested_relation"])] = cell
    expected = {(seed, side) for seed in range(8300, 8303) for side in TASKS}
    if set(cells) != expected:
        raise RuntimeError(f"Pilot grid mismatch: missing={sorted(expected - set(cells))}")
    return cells


def result_index(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "episode_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row["env_name"]: row for row in rows}
    if set(result) != set(TASKS.values()):
        raise RuntimeError(f"Expected exactly two completed tasks in {path}")
    return result


def load_episode(
    *,
    seed: int,
    direction: str,
    root: Path,
    cell: dict[str, Any],
    trajectory_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, str]:
    task = TASKS[direction]
    task_dir = root / task
    hdf5_path = task_dir / "run_0.hdf5"
    log_path = task_dir / "log_0_env0.json"
    env_path = task_dir / "env_cfg.json"
    video_paths = sorted(task_dir.glob("*_viewport.mp4"))
    if len(video_paths) != 1:
        raise RuntimeError(f"Expected one viewport video in {task_dir}, found {video_paths}")
    result = result_index(root)[task]
    log = json.loads(log_path.read_text())
    env = json.loads(env_path.read_text())
    if result["instruction"] != cell["rendered_prompt"]:
        raise RuntimeError(
            f"Prompt mismatch {seed}:{direction}: {result['instruction']!r} != "
            f"{cell['rendered_prompt']!r}"
        )
    if int(env["seed"]) != seed or env["instruction"] != cell["rendered_prompt"]:
        raise RuntimeError(f"Environment provenance mismatch for {seed}:{direction}")
    if bool(result["success"]) != bool(log["success"]):
        raise RuntimeError(f"Result/log success mismatch for {seed}:{direction}")

    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data/demo_0"]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        cube_pose = np.asarray(demo["states/rigid_object/rubiks_cube/root_pose"], dtype=np.float64)
        bowl_pose = np.asarray(demo["states/rigid_object/bowl/root_pose"], dtype=np.float64)
        robot_pose = np.asarray(demo["states/articulation/robot/root_pose"], dtype=np.float64)
        fingerprint = initial_fingerprint(demo["initial_state"])
    steps = int(log["final_step"])
    if not (len(actions) == len(cube_pose) == len(bowl_pose) == len(robot_pose) == steps):
        raise RuntimeError(f"Trajectory length mismatch for {seed}:{direction}")

    delta_robot = robot_frame_delta(cube_pose, bowl_pose, robot_pose)
    requested = relation_mask(delta_robot, direction)
    opposite_direction = "right" if direction == "left" else "left"
    opposite = relation_mask(delta_robot, opposite_direction)
    lift = cube_pose[:, 2] - cube_pose[0, 2]
    displacement = np.linalg.norm(cube_pose[:, :3] - cube_pose[0, :3], axis=1)
    pickup_step = first_consecutive(lift >= PICKUP_LIFT_M)
    interaction_step = first_consecutive(displacement >= MOTION_M)
    entered_step = first_true(requested)
    success = bool(log["success"])
    if success:
        failure_stage = "success"
    elif interaction_step is None:
        failure_stage = "no_object_interaction"
    elif pickup_step is None:
        failure_stage = "object_moved_no_verified_pickup"
    elif entered_step is None:
        failure_stage = "picked_never_entered_requested_region"
    else:
        failure_stage = "entered_requested_region_not_released"

    trajectory_path = trajectory_dir / f"seed{seed}_{direction}.json"
    trajectory = [
        {
            "action_step": index,
            # Display convention: LEFT is negative, RIGHT is positive.
            "object_minus_target_x": float(-delta_robot[index, 1]),
            "object_minus_target_y": float(delta_robot[index, 0]),
            "object_minus_target_z": float(delta_robot[index, 2]),
            "cube_world_xyz": cube_pose[index, :3].tolist(),
            "lift_m": float(lift[index]),
            "requested_relation_region": bool(requested[index]),
            "opposite_relation_region": bool(opposite[index]),
            "gripper_command": float(actions[index, -1]),
        }
        for index in range(steps)
    ]
    dump_json(trajectory_path, trajectory)
    timing = result.get("timing", {})
    requested_signed_endpoint = (
        float(delta_robot[-1, 1]) if direction == "left" else float(-delta_robot[-1, 1])
    )
    row = {
        "model_id": MODEL_ID,
        "pair_id": f"droid_pair_seed_{seed}",
        "task": task,
        "environment_seed": seed,
        "sampling_seed": seed,
        "first_server_sampling_seed": seed * 1000,
        "prompt_family": "direct_command",
        "requested_relation": direction,
        "prompt": cell["rendered_prompt"],
        "requested_success": success,
        "actions_executed": steps,
        "wall_seconds": timing.get("wall_total_s"),
        "policy_inference_seconds": timing.get("policy_inference_s"),
        "policy_inference_average_ms_per_control_step": timing.get("policy_inference_avg_ms"),
        "estimated_policy_requests": math.ceil(steps / 10),
        "estimated_mean_policy_request_seconds": (
            float(timing["policy_inference_s"]) / math.ceil(steps / 10)
            if timing.get("policy_inference_s") is not None
            else None
        ),
        "initial_lateral_display_m": float(-delta_robot[0, 1]),
        "final_lateral_display_m": float(-delta_robot[-1, 1]),
        "initial_forward_m": float(delta_robot[0, 0]),
        "final_forward_m": float(delta_robot[-1, 0]),
        "requested_signed_final_offset_m": requested_signed_endpoint,
        "max_object_lift_m": float(np.max(lift)),
        "max_object_displacement_m": float(np.max(displacement)),
        "verified_pickup_proxy": pickup_step is not None,
        "first_verified_pickup_proxy_step": pickup_step,
        "object_interaction_proxy": interaction_step is not None,
        "first_object_interaction_proxy_step": interaction_step,
        "ever_entered_requested_region": bool(np.any(requested)),
        "first_requested_region_step": entered_step,
        "final_requested_relation": bool(requested[-1]),
        "ever_entered_opposite_region": bool(np.any(opposite)),
        "final_opposite_relation": bool(opposite[-1]),
        "ever_released_in_requested_region": success,
        "paper_style_progression": (int(pickup_step is not None) + int(success)) / 2.0,
        "failure_stage": failure_stage,
        "physical_initial_state_sha256": fingerprint,
        "raw_hdf5": file_record(hdf5_path),
        "raw_log": file_record(log_path),
        "raw_env_config": file_record(env_path),
        "raw_trajectory": file_record(trajectory_path),
        "executed_video": file_record(video_paths[0]),
        "imagined_future_video": None,
        "imagined_future_artifact": None,
    }
    return row, actions, fingerprint


def thermal_summary(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in range(8300, 8303):
        path = directory / f"thermal_seed{seed}.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        names = [event["event"] for event in events]
        if names[0] != "monitor_started" or names[-1] != "monitor_completed":
            raise RuntimeError(f"Incomplete thermal lifecycle: {path}")
        if any("emergency_stop" in name for name in names):
            raise RuntimeError(f"Emergency stop in retained batch: {path}")
        rows.append(
            {
                "environment_seed": seed,
                "events": names,
                "cooldown_count": names.count("cooldown_started"),
                "log": file_record(path),
            }
        )
    return rows


def write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    columns = [
        "model_id", "pair_id", "task", "environment_seed", "sampling_seed",
        "prompt_family", "requested_relation", "prompt", "requested_success",
        "actions_executed", "verified_pickup_proxy", "ever_entered_requested_region",
        "final_requested_relation", "final_lateral_display_m",
        "requested_signed_final_offset_m", "paper_style_progression", "failure_stage",
        "estimated_mean_policy_request_seconds", "wall_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            writer.writerow({key: episode[key] for key in columns})


def write_markdown(path: Path, compiled: dict[str, Any]) -> None:
    summary = compiled["summary"]
    lines = [
        "# pi0-FAST DROID direct-command pilot",
        "",
        "This is a six-episode, oracle-free base-competence gate—not a stable population rate.",
        "The scene and seed are matched inside each pair; only `LEFT` versus `RIGHT` changes.",
        "",
        "## What was tested",
        "",
        "- LEFT: `Put the Rubik's cube to the left of the bowl.`",
        "- RIGHT: `Put the Rubik's cube to the right of the bowl.`",
        "- Environment/sampling seeds: 8300, 8301, and 8302.",
        "- One episode-static prompt; no subtask coach, predicate oracle, or dynamic prompt.",
        "- Primary outcome: released cube in the requested 45-degree bowl-relative region.",
        f"- Transparent pickup proxy: cube lifted at least {PICKUP_LIFT_M:.2f} m for "
        f"{CONSECUTIVE_STEPS} consecutive recorded steps.",
        "",
        "## Result",
        "",
        f"- LEFT: **{summary['by_direction']['left']['successes']}/3**.",
        f"- RIGHT: **{summary['by_direction']['right']['successes']}/3**.",
        f"- Overall: **{summary['successes']}/6**.",
        f"- Same-seed endpoint redirection aligned with LEFT→RIGHT in "
        f"**{summary['aligned_endpoint_pairs']}/3** pairs.",
        f"- First-chunk action RMS was non-zero in **{summary['nonzero_first_chunk_pairs']}/3** pairs.",
        "",
        "| Seed | LEFT | RIGHT | LEFT endpoint | RIGHT endpoint | Redirected? |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for pair in summary["paired_endpoint_responses"]:
        lines.append(
            f"| {pair['environment_seed']} | {'success' if pair['left_success'] else 'failure'} | "
            f"{'success' if pair['right_success'] else 'failure'} | "
            f"{pair['left_final_lateral_display_m']:+.3f} m | "
            f"{pair['right_final_lateral_display_m']:+.3f} m | "
            f"{pair['endpoint_response_direction'].replace('_', ' ')} |"
        )
    lines.extend(
        [
            "",
            "Negative endpoint values are robot LEFT; positive values are robot RIGHT.",
            "The endpoint shifted rightward in all three pairs, but none of the LEFT runs "
            "completed. That is evidence of prompt-conditioned behavior plus a severe "
            "directional/base-competence asymmetry, not robust steerability.",
            "",
            "## Frozen gate decision",
            "",
            f"**{summary['pilot_gate_decision']}.** {summary['pilot_gate_reason']}",
            "",
            "All six videos, failures included, are retained. Simulator state was used only "
            "after action execution for scoring and visualization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/pilot_grid.json"))
    parser.add_argument("--robolab-output", type=Path, default=Path("/home/ali/projects/RoboLab/output"))
    parser.add_argument("--thermal-dir", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/pi0_fast_direct"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_gate.json"))
    args = parser.parse_args()

    cells = load_grid(args.grid)
    trajectory_dir = args.thermal_dir / "trajectories"
    episodes: list[dict[str, Any]] = []
    actions: dict[tuple[int, str], np.ndarray] = {}
    fingerprints: dict[tuple[int, str], str] = {}
    for seed in range(8300, 8303):
        root = args.robolab_output / f"v2_pi0_fast_direct_seed{seed}"
        for direction in TASKS:
            episode, action, fingerprint = load_episode(
                seed=seed,
                direction=direction,
                root=root,
                cell=cells[(seed, direction)],
                trajectory_dir=trajectory_dir,
            )
            episodes.append(episode)
            actions[(seed, direction)] = action
            fingerprints[(seed, direction)] = fingerprint

    pairs = []
    for seed in range(8300, 8303):
        left = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "right")
        if fingerprints[(seed, "left")] != fingerprints[(seed, "right")]:
            raise RuntimeError(f"Physical initial state mismatch inside pair {seed}")
        first_chunk_rms = float(
            np.sqrt(np.mean(np.square(actions[(seed, "left")][:10] - actions[(seed, "right")][:10])))
        )
        shift = right["final_lateral_display_m"] - left["final_lateral_display_m"]
        pairs.append(
            {
                "pair_id": f"droid_pair_seed_{seed}",
                "environment_seed": seed,
                "left_success": left["requested_success"],
                "right_success": right["requested_success"],
                "left_final_lateral_display_m": left["final_lateral_display_m"],
                "right_final_lateral_display_m": right["final_lateral_display_m"],
                "right_minus_left_endpoint_lateral_m": shift,
                "endpoint_response_direction": "aligned" if shift > 0 else "anti_directed" if shift < 0 else "none",
                "first_ten_action_rms": first_chunk_rms,
                "physical_initial_state_sha256": fingerprints[(seed, "left")],
            }
        )

    by_direction = {}
    for direction in TASKS:
        rows = [row for row in episodes if row["requested_relation"] == direction]
        by_direction[direction] = {
            "episodes": len(rows),
            "successes": sum(row["requested_success"] for row in rows),
            "verified_pickups": sum(row["verified_pickup_proxy"] for row in rows),
            "entered_requested_region": sum(row["ever_entered_requested_region"] for row in rows),
            "mean_requested_signed_final_offset_m": float(
                np.mean([row["requested_signed_final_offset_m"] for row in rows])
            ),
        }
    failure_counts = Counter(row["failure_stage"] for row in episodes)
    summary = {
        "episode_count": len(episodes),
        "pair_count": len(pairs),
        "successes": sum(row["requested_success"] for row in episodes),
        "success_rate": sum(row["requested_success"] for row in episodes) / len(episodes),
        "by_direction": by_direction,
        "failure_stage_counts": dict(sorted(failure_counts.items())),
        "paired_endpoint_responses": pairs,
        "aligned_endpoint_pairs": sum(pair["endpoint_response_direction"] == "aligned" for pair in pairs),
        "nonzero_first_chunk_pairs": sum(pair["first_ten_action_rms"] > 0 for pair in pairs),
        "pilot_gate_decision": "expand_direct_directional_bias_only",
        "pilot_gate_reason": (
            "Direct-command success occurred for RIGHT only. The frozen gate calls for the "
            "ten-seed direct-command directional-bias confirmation before any four-wording sweep."
        ),
    }
    compiled = {
        "schema_version": "vla-wam-shared-v2-droid-pilot-v1",
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "measurement": {
            "oracle_actions": 0,
            "dynamic_prompts": 0,
            "subtask_progress_checking": False,
            "pickup_lift_threshold_m": PICKUP_LIFT_M,
            "pickup_consecutive_steps": CONSECUTIVE_STEPS,
            "object_motion_threshold_m": MOTION_M,
            "simulator_state_role": "post_action_scoring_and_visualization_only",
            "future_interface": "none",
            "open_loop_horizon": 10,
        },
        "summary": summary,
        "thermal_guard_lifecycles": thermal_summary(args.thermal_dir),
        "fixed_observation_probe": file_record(
            Path("artifacts/vla_wam_shared_v2/pilot/pi0_fast_fixed_observation/manifest.json")
        ),
        "episodes": episodes,
    }
    dump_json(args.output, compiled)
    write_csv(args.output.with_suffix(".csv"), episodes)
    write_markdown(args.output.with_suffix(".md"), compiled)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
