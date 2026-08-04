#!/usr/bin/env python3
"""Compile one or more frozen Cosmos3 Edge DROID seed pairs.

The compiler validates static prompt/seed provenance, the matched physical
reset, simulator success, viewport video, executed actions, and every exposed
33-frame decoded future. Raw arrays remain on the ali-owned PVC; the compiled
JSON contains content hashes and compact post-action trajectory summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

MODEL_ID = "cosmos3_edge_droid_wam"
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
PICKUP_LIFT_M = 0.03
MOTION_M = 0.01
CONSECUTIVE_STEPS = 3
ACTION_HORIZON = 32


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
    if not path.is_file():
        raise RuntimeError(f"Missing evidence file: {path}")
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


def robot_frame_delta(cube_pose: np.ndarray, bowl_pose: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    delta_world = cube_pose[:, :3] - bowl_pose[:, :3]
    return np.einsum("tij,ti->tj", rotation_wxyz(robot_pose[:, 3:7]), delta_world)


def relation_mask(delta_robot: np.ndarray, direction: str) -> np.ndarray:
    horizontal = np.linalg.norm(delta_robot[:, :2], axis=1)
    sign = 1.0 if direction == "left" else -1.0
    cosine = np.divide(
        sign * delta_robot[:, 1], horizontal,
        out=np.zeros_like(horizontal), where=horizontal > 1e-8,
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


def result_index(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "episode_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row["env_name"]: row for row in rows}
    if set(result) != set(TASKS.values()):
        raise RuntimeError(f"Expected exactly two completed tasks in {path}")
    return result


def trace_metadata(raw_root: Path, seed: int, direction: str) -> tuple[dict[str, Any], Path]:
    candidates = list(raw_root.glob(f"seed{seed}/simulator_attempt*/actions/seed{seed}_{direction}_executed_actions.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one valid trace metadata file for {seed}:{direction}, found {candidates}")
    path = candidates[0]
    metadata = json.loads(path.read_text())
    if metadata["prompt"] != PROMPTS[direction] or metadata["sampling_seed_base"] != seed:
        raise RuntimeError(f"Trace prompt/seed mismatch for {seed}:{direction}")
    return metadata, path


def load_episode(
    *, seed: int, direction: str, output_root: Path, raw_root: Path, trajectory_dir: Path,
    output_prefix: str = "v2_cosmos_edge", policy_id: str = "cosmos3_v2",
) -> tuple[dict[str, Any], np.ndarray, str]:
    task = TASKS[direction]
    root = output_root / f"{output_prefix}_seed{seed}_neutral"
    task_dir = root / task
    hdf5_path = task_dir / "run_0.hdf5"
    log_path = task_dir / "log_0_env0.json"
    env_path = task_dir / "env_cfg.json"
    videos = sorted(task_dir.glob("*_viewport.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"Expected one viewport video in {task_dir}, found {videos}")
    result = result_index(root)[task]
    log = json.loads(log_path.read_text())
    env = json.loads(env_path.read_text())
    if result["instruction"] != PROMPTS[direction] or env["instruction"] != PROMPTS[direction]:
        raise RuntimeError(f"Static prompt mismatch for {seed}:{direction}")
    if int(env["seed"]) != seed or result["policy"] != policy_id:
        raise RuntimeError(f"Environment/policy provenance mismatch for {seed}:{direction}")
    if bool(result["success"]) != bool(log["success"]):
        raise RuntimeError(f"Result/log success mismatch for {seed}:{direction}")

    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data/demo_0"]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        cube_pose = np.asarray(demo["states/rigid_object/rubiks_cube/root_pose"], dtype=np.float64)
        bowl_pose = np.asarray(demo["states/rigid_object/bowl/root_pose"], dtype=np.float64)
        robot_pose = np.asarray(demo["states/articulation/robot/root_pose"], dtype=np.float64)
        fingerprint = initial_fingerprint(demo["initial_state"])
    steps = int(log["final_step"])
    if not (len(actions) == len(cube_pose) == len(bowl_pose) == len(robot_pose) == steps):
        raise RuntimeError(f"Trajectory length mismatch for {seed}:{direction}")

    metadata, metadata_path = trace_metadata(raw_root, seed, direction)
    executed_path = Path(metadata["executed_actions"]["path"])
    executed = np.load(executed_path, allow_pickle=False)
    if executed.shape != actions.shape or not np.array_equal(executed, actions):
        raise RuntimeError(f"Instrumented actions do not exactly match simulator HDF5: {seed}:{direction}")
    if sha256(executed_path) != metadata["executed_actions"]["sha256"]:
        raise RuntimeError(f"Executed-action hash mismatch: {seed}:{direction}")
    expected_requests = math.ceil(steps / ACTION_HORIZON)
    if len(metadata["requests"]) != expected_requests:
        raise RuntimeError(f"Request/future count mismatch for {seed}:{direction}")
    request_evidence = []
    for index, request in enumerate(metadata["requests"]):
        action_path, future_path = Path(request["action_path"]), Path(request["future_path"])
        action, future = np.load(action_path, allow_pickle=False), np.load(future_path, allow_pickle=False)
        if request["request_index"] != index or action.shape != (32, 8) or future.shape[0] != 33 or future.shape[-1] != 3:
            raise RuntimeError(f"Returned action/future contract mismatch: {seed}:{direction}:{index}")
        if sha256(action_path) != request["action_sha256"] or sha256(future_path) != request["future_sha256"]:
            raise RuntimeError(f"Returned action/future hash mismatch: {seed}:{direction}:{index}")
        request_evidence.append({
            "request_index": index,
            "returned_action": file_record(action_path),
            "decoded_future": file_record(future_path),
            "returned_action_shape": list(action.shape),
            "decoded_future_shape": list(future.shape),
        })

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
    dump_json(trajectory_path, [
        {
            "action_step": index,
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
    ])
    timing = result.get("timing", {})
    return ({
        "model_id": MODEL_ID,
        "pair_id": f"droid_pair_seed_{seed}",
        "task": task,
        "environment_seed": seed,
        "sampling_seed": seed,
        "prompt_family": "direct_command",
        "prompt_controller": "episode_static",
        "oracle_actions": 0,
        "dynamic_prompt_switches": 0,
        "requested_relation": direction,
        "prompt": PROMPTS[direction],
        "requested_success": success,
        "actions_executed": steps,
        "policy_request_count": expected_requests,
        "decoded_future_count": len(request_evidence),
        "wall_seconds": timing.get("wall_total_s"),
        "policy_inference_seconds": timing.get("policy_inference_s"),
        "initial_lateral_display_m": float(-delta_robot[0, 1]),
        "final_lateral_display_m": float(-delta_robot[-1, 1]),
        "requested_signed_final_offset_m": float(delta_robot[-1, 1] if direction == "left" else -delta_robot[-1, 1]),
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
        "failure_stage": failure_stage,
        "physical_initial_state_sha256": fingerprint,
        "raw_hdf5": file_record(hdf5_path),
        "raw_log": file_record(log_path),
        "raw_env_config": file_record(env_path),
        "raw_trajectory": file_record(trajectory_path),
        "executed_video": file_record(videos[0]),
        "executed_action_trace_metadata": file_record(metadata_path),
        "executed_action_trace": file_record(executed_path),
        "imagined_future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
        "imagined_future_requests": request_evidence,
    }, actions, fingerprint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--robolab-output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled-at-git-head", required=True)
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if any(seed not in {8300, 8301, 8302} for seed in seeds):
        raise RuntimeError("Authorized seeds are exactly 8300, 8301, and 8302")

    episodes: list[dict[str, Any]] = []
    actions: dict[tuple[int, str], np.ndarray] = {}
    fingerprints: dict[tuple[int, str], str] = {}
    for seed in seeds:
        for direction in TASKS:
            episode, action, fingerprint = load_episode(
                seed=seed, direction=direction, output_root=args.robolab_output,
                raw_root=args.raw_root, trajectory_dir=args.trajectory_dir,
            )
            episodes.append(episode)
            actions[(seed, direction)] = action
            fingerprints[(seed, direction)] = fingerprint

    pairs = []
    for seed in seeds:
        left = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "right")
        if fingerprints[(seed, "left")] != fingerprints[(seed, "right")]:
            raise RuntimeError(f"Physical initial state mismatch inside pair {seed}")
        overlap = min(len(actions[(seed, "left")]), len(actions[(seed, "right")]))
        delta = actions[(seed, "left")][:overlap].astype(np.float64) - actions[(seed, "right")][:overlap].astype(np.float64)
        first = delta[: min(ACTION_HORIZON, overlap)]
        shift = right["final_lateral_display_m"] - left["final_lateral_display_m"]
        pairs.append({
            "pair_id": f"droid_pair_seed_{seed}",
            "environment_seed": seed,
            "left_success": left["requested_success"],
            "right_success": right["requested_success"],
            "left_final_lateral_display_m": left["final_lateral_display_m"],
            "right_final_lateral_display_m": right["final_lateral_display_m"],
            "right_minus_left_endpoint_lateral_m": shift,
            "endpoint_response_direction": "aligned" if shift > 0 else "anti_directed" if shift < 0 else "none",
            "first_chunk_action_rms": float(np.sqrt(np.mean(np.square(first)))),
            "overlap_action_rms": float(np.sqrt(np.mean(np.square(delta)))),
            "executed_actions_distinct": bool(not np.array_equal(actions[(seed, "left")][:overlap], actions[(seed, "right")][:overlap])),
            "physical_initial_state_sha256": fingerprints[(seed, "left")],
        })

    by_direction = {}
    for direction in TASKS:
        rows = [row for row in episodes if row["requested_relation"] == direction]
        by_direction[direction] = {
            "episodes": len(rows),
            "successes": sum(row["requested_success"] for row in rows),
            "verified_pickups": sum(row["verified_pickup_proxy"] for row in rows),
            "entered_requested_region": sum(row["ever_entered_requested_region"] for row in rows),
        }
    successes = sum(row["requested_success"] for row in episodes)
    complete = seeds == [8300, 8301, 8302]
    left_successes = by_direction["left"]["successes"]
    right_successes = by_direction["right"]["successes"]
    competence = (
        "both_directions" if left_successes and right_successes
        else "left_only" if left_successes else "right_only" if right_successes else "zero_direction"
    )
    payload = {
        "schema_version": "vla-wam-shared-v2-cosmos3-edge-droid-slice-v1",
        "status": "complete" if complete else "coherent_partial_slice",
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "compiled_at_git_head": args.compiled_at_git_head,
        "model_id": MODEL_ID,
        "checkpoint_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777",
        "server_repository_commit": "a904d2d36b774a51dd06ff9ff906816b1a04f579",
        "amendment_id": "V2-A005",
        "arena": "droid_robolab",
        "seeds": seeds,
        "measurement": {
            "oracle_actions": 0,
            "dynamic_prompts": 0,
            "subtask_progress_checking": False,
            "prompt_controller": "episode_static",
            "simulator_state_role": "post_action_scoring_and_visualization_only",
            "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
            "open_loop_horizon": ACTION_HORIZON,
        },
        "summary": {
            "episode_count": len(episodes),
            "pair_count": len(pairs),
            "successes": successes,
            "success_rate": successes / len(episodes),
            "by_direction": by_direction,
            "failure_stage_counts": dict(sorted(Counter(row["failure_stage"] for row in episodes).items())),
            "aligned_endpoint_pairs": sum(pair["endpoint_response_direction"] == "aligned" for pair in pairs),
            "nonzero_first_chunk_pairs": sum(pair["first_chunk_action_rms"] > 0 for pair in pairs),
            "competence_gate": competence if complete else "pending_remaining_preregistered_pairs",
        },
        "pairs": pairs,
        "episodes": episodes,
        "claim_boundary": "This v2 six-cell replication remains separate from the existing Cosmos3 Edge v1 80-episode DROID grid and every RoboTwin denominator.",
    }
    dump_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
