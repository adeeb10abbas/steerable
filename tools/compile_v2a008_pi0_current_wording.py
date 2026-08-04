#!/usr/bin/env python3
"""Compile the separate V2-A008 current-stack pi0-FAST wording replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


OPENPI_COMMIT = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
CONFIG = "pi0_fast_droid_jointpos_polaris"
FAMILIES = (
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
)
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
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
    return np.einsum(
        "tij,ti->tj", rotation_wxyz(robot_pose[:, 3:7]), delta_world
    )


def load_registry(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    data = json.loads(path.read_text())
    if (
        data.get("schema_version")
        != "vla-wam-v2a008-pi0-current-stack-registry-v1"
        or data.get("amendment_id") != "V2-A008"
        or data.get("status")
        != "frozen_before_current_stack_model_load_or_behavioral_inference"
    ):
        raise ValueError(f"Not the frozen V2-A008 current-stack registry: {path}")
    cells: dict[tuple[int, str, str], dict[str, Any]] = {}
    for cell in data["cells"]:
        if (
            cell["model_id"] == "pi0_fast_current_stack_droid_vla"
            and cell["prompt_family"] in FAMILIES
            and 8300 <= int(cell["environment_seed"]) <= 8309
        ):
            key = (
                int(cell["environment_seed"]),
                cell["prompt_family"],
                cell["requested_relation"],
            )
            if key in cells:
                raise ValueError(f"Duplicate registry cell: {key}")
            if int(cell["sampling_seed_base"]) != key[0]:
                raise ValueError(f"Registry seed mismatch: {key}")
            cells[key] = cell
    expected = {
        (seed, family, relation)
        for seed in range(8300, 8310)
        for family in FAMILIES
        for relation in TASKS
    }
    if set(cells) != expected:
        raise ValueError(
            f"Registry must contain exactly the 60 V2-A008 cells; "
            f"missing={sorted(expected - set(cells))}, extra={sorted(set(cells) - expected)}"
        )
    return cells


def single_result(path: Path, task: str) -> tuple[Path, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one result in {path}, found {len(rows)}")
    row_task = rows[0].get("task_name", rows[0].get("env_name"))
    if row_task != task:
        raise ValueError(f"Unexpected task in {path}: {row_task!r} != {task!r}")
    return path, rows[0]


def load_cell(
    raw_root: Path,
    trace_root: Path,
    seed: int,
    family: str,
    relation: str,
    registry: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    task = TASKS[relation]
    run_root = raw_root / f"v2a008_pi0_current_seed{seed}_{family}_{relation}"
    result_path, result = single_result(run_root / "episode_results.jsonl", task)
    task_root = run_root / task
    env_path = task_root / "env_cfg.json"
    log_path = task_root / "log_0_env0.json"
    h5_path = task_root / "run_0.hdf5"
    videos = sorted(task_root.glob("*_viewport.mp4"))
    if len(videos) != 1:
        raise ValueError(f"Expected one viewport video in {task_root}, found {videos}")
    env = json.loads(env_path.read_text())
    log = json.loads(log_path.read_text())
    prompt = registry["rendered_prompt"]
    if result["instruction"] != prompt or env["instruction"] != prompt:
        raise ValueError(f"Prompt mismatch in {seed}/{family}/{relation}")
    if int(env["seed"]) != seed:
        raise ValueError(f"Environment seed mismatch in {seed}/{family}/{relation}")
    if bool(result["success"]) != bool(log["success"]):
        raise ValueError(f"Result/log mismatch in {seed}/{family}/{relation}")

    stem = f"seed{seed}_{family}_{relation}"
    trace_path = trace_root / f"{stem}_action_trace.json"
    trace = json.loads(trace_path.read_text())
    action_path = Path(trace["executed_actions"]["path"])
    chunks_path = Path(trace["returned_action_chunks"]["path"])
    actions = np.load(action_path, allow_pickle=False)
    chunks = np.load(chunks_path, allow_pickle=False)
    if sha256(action_path) != trace["executed_actions"]["sha256"]:
        raise ValueError(f"Executed-action hash mismatch: {action_path}")
    if sha256(chunks_path) != trace["returned_action_chunks"]["sha256"]:
        raise ValueError(f"Returned-chunk hash mismatch: {chunks_path}")
    if trace["prompt"] != prompt or trace["sampling_seed_base"] != seed:
        raise ValueError(f"Trace provenance mismatch in {seed}/{family}/{relation}")
    expected_request_seeds = [seed * 1000 + i for i in range(len(chunks))]
    if trace["request_sampling_seeds"] != expected_request_seeds:
        raise ValueError(f"Request seed schedule mismatch in {seed}/{family}/{relation}")

    with h5py.File(h5_path) as handle:
        demo = handle["data/demo_0"]
        h5_actions = np.asarray(demo["actions"], dtype=np.float32)
        cube = np.asarray(demo["states/rigid_object/rubiks_cube/root_pose"])
        bowl = np.asarray(demo["states/rigid_object/bowl/root_pose"])
        robot = np.asarray(demo["states/articulation/robot/root_pose"])
        initial_cube = np.asarray(
            demo["initial_state/rigid_object/rubiks_cube/root_pose"][-1]
        )
        initial_bowl = np.asarray(
            demo["initial_state/rigid_object/bowl/root_pose"][-1]
        )
    if not np.array_equal(actions, h5_actions):
        raise ValueError(f"Client/simulator actions differ in {seed}/{family}/{relation}")
    step_count = int(log["final_step"])
    if not (
        len(actions) == len(cube) == len(bowl) == len(robot) == step_count
    ):
        raise ValueError(f"Step-count mismatch in {seed}/{family}/{relation}")

    endpoint = cube[-1, :3] - bowl[-1, :3]
    delta_robot = robot_frame_delta(cube, bowl, robot)
    episode = {
        "cell_id": registry["cell_id"],
        "environment_seed": seed,
        "sampling_seed_base": seed,
        "prompt_family": family,
        "requested_relation": relation,
        "prompt": prompt,
        "success": bool(result["success"]),
        "executed_action_count": step_count,
        "policy_request_count": len(chunks),
        "initial_cube_world_xyz": initial_cube[:3].tolist(),
        "initial_bowl_world_xyz": initial_bowl[:3].tolist(),
        "endpoint_cube_world_xyz": cube[-1, :3].tolist(),
        "endpoint_bowl_world_xyz": bowl[-1, :3].tolist(),
        "endpoint_cube_minus_bowl_world_xyz": endpoint.tolist(),
        # Match the frozen DROID compiler: LEFT is negative and RIGHT positive.
        "endpoint_lateral_display_m": float(-delta_robot[-1, 1]),
        "files": {
            "episode_results": file_record(result_path),
            "environment": file_record(env_path),
            "episode_log": file_record(log_path),
            "trajectory": file_record(h5_path),
            "viewport_video": file_record(videos[0]),
            "action_trace": file_record(trace_path),
            "executed_actions": file_record(action_path),
            "returned_action_chunks": file_record(chunks_path),
        },
    }
    return episode, actions, initial_cube, initial_bowl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--action-trace-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    episodes: list[dict[str, Any]] = []
    actions: dict[tuple[int, str, str], np.ndarray] = {}
    initial: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key in sorted(registry):
        episode, action, cube, bowl = load_cell(
            args.raw_root, args.action_trace_root, *key, registry[key]
        )
        episodes.append(episode)
        actions[key] = action
        initial[key] = (cube, bowl)

    pairs: list[dict[str, Any]] = []
    for seed in range(8300, 8310):
        for family in FAMILIES:
            left_key = (seed, family, "left")
            right_key = (seed, family, "right")
            for index, name in enumerate(("cube", "bowl")):
                if not np.array_equal(initial[left_key][index], initial[right_key][index]):
                    raise ValueError(f"Paired initial {name} differs for {seed}/{family}")
            overlap = min(len(actions[left_key]), len(actions[right_key]), 10)
            if overlap == 0 or actions[left_key].shape[1:] != actions[right_key].shape[1:]:
                raise ValueError(f"Cannot compare paired actions for {seed}/{family}")
            delta = actions[left_key][:overlap] - actions[right_key][:overlap]
            pair_episodes = {
                row["requested_relation"]: row
                for row in episodes
                if row["environment_seed"] == seed and row["prompt_family"] == family
            }
            left_lateral = pair_episodes["left"]["endpoint_lateral_display_m"]
            right_lateral = pair_episodes["right"]["endpoint_lateral_display_m"]
            endpoint_shift = right_lateral - left_lateral
            pairs.append(
                {
                    "environment_seed": seed,
                    "prompt_family": family,
                    "left_success": pair_episodes["left"]["success"],
                    "right_success": pair_episodes["right"]["success"],
                    "left_endpoint_lateral_display_m": left_lateral,
                    "right_endpoint_lateral_display_m": right_lateral,
                    "right_minus_left_endpoint_shift_m": endpoint_shift,
                    "endpoint_ordering_aligned": endpoint_shift > 0,
                    "first_ten_executed_action_rms": float(np.sqrt(np.mean(delta**2))),
                    "first_ten_executed_action_rms_steps_used": overlap,
                    "executed_actions_distinct": bool(np.any(delta != 0)),
                }
            )

    success = Counter((row["prompt_family"], row["requested_relation"]) for row in episodes if row["success"])
    result = {
        "schema_version": "vla-wam-v2a008-pi0-current-wording-result-v1",
        "amendment_id": "V2-A008",
        "status": "complete_60_of_60_valid_current_stack_cells",
        "claim_boundary": (
            "Separate post-result current-stack replication. These outcomes are not "
            "pooled with or substituted for historical pi0-FAST evidence."
        ),
        "model": "pi0-FAST DROID current stack",
        "openpi_commit": OPENPI_COMMIT,
        "robolab_commit": ROBOLAB_COMMIT,
        "openpi_config": CONFIG,
        "study_commit": args.study_commit,
        "checkpoint_manifest": file_record(args.checkpoint_manifest),
        "registry": file_record(args.registry),
        "valid_episode_count": len(episodes),
        "episodes": episodes,
        "pairs": pairs,
        "summary": {
            "by_prompt_family_and_direction": {
                family: {
                    relation: {
                        "successes": success[(family, relation)],
                        "trials": 10,
                    }
                    for relation in TASKS
                }
                for family in FAMILIES
            },
            "aligned_endpoint_pair_count": sum(p["endpoint_ordering_aligned"] for p in pairs),
            "distinct_executed_action_pair_count": sum(p["executed_actions_distinct"] for p in pairs),
            "pair_count": len(pairs),
        },
        "infrastructure_invalid_attempts": [],
    }
    if len(episodes) != 60 or len(pairs) != 30:
        raise AssertionError("V2-A008 compiler accounting failure")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
