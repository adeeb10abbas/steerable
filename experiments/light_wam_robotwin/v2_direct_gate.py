#!/usr/bin/env python3
"""Run the frozen V2-A006 Light-WAM RoboTwin probe and six-cell gate.

The fixed-observation probe restores every Python, NumPy, Torch, and CUDA RNG
state before LEFT-repeat and RIGHT calls.  The behavioral gate then changes
only the static episode prompt inside each frozen environment/sampling pair.
Infrastructure exceptions are retained separately and never become failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml


SCENES = (
    {
        "pair_id": "robotwin_pair_00",
        "task": "place_a2b_left",
        "environment_seed": 4_300_000,
        "sampling_seed": 8_400,
        "prompts": {
            "left": "Put the blue soap to the left of the tea-box.",
            "right": "Put the blue soap to the right of the tea-box.",
        },
    },
    {
        "pair_id": "robotwin_pair_01",
        "task": "place_a2b_right",
        "environment_seed": 4_300_001,
        "sampling_seed": 8_401,
        "prompts": {
            "left": "Put the brown woodenblock to the left of the black phone.",
            "right": "Put the brown woodenblock to the right of the black phone.",
        },
    },
    {
        "pair_id": "robotwin_pair_02",
        "task": "place_a2b_left",
        "environment_seed": 4_300_002,
        "sampling_seed": 8_402,
        "prompts": {
            "left": "Put the box with cards inside to the left of the red coffee-box.",
            "right": "Put the box with cards inside to the right of the red coffee-box.",
        },
    },
)

RELATION_BY_TASK = {"place_a2b_left": "left", "place_a2b_right": "right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed-probe", "episodes"), required=True)
    parser.add_argument("--light-wam-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--max-actions", type=int, default=400)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_task_args(robotwin_root: Path, task_name: str, task_config: str) -> dict[str, Any]:
    with (robotwin_root / "task_config" / f"{task_config}.yml").open() as handle:
        task_args = yaml.safe_load(handle)
    with (robotwin_root / "task_config" / "_embodiment_config.yml").open() as handle:
        embodiments = yaml.safe_load(handle)
    with (robotwin_root / "task_config" / "_camera_config.yml").open() as handle:
        cameras = yaml.safe_load(handle)

    embodiment_name = task_args["embodiment"][0]
    robot_file = embodiments[embodiment_name]["file_path"]
    with (robotwin_root / robot_file / "config.yml").open() as handle:
        embodiment_config = yaml.safe_load(handle)
    head_camera = cameras[task_args["camera"]["head_camera_type"]]
    task_args.update(
        task_name=task_name,
        task_config=task_config,
        eval_mode=True,
        save_data=False,
        collect_data=False,
        render_freq=0,
        head_camera_h=head_camera["h"],
        head_camera_w=head_camera["w"],
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=embodiment_config,
        right_embodiment_config=embodiment_config,
        eval_video_record_mode="action_step",
        eval_video_frame_stride=1,
    )
    return task_args


def task_class(task_name: str):
    module = __import__(f"envs.{task_name}", fromlist=[task_name])
    return getattr(module, task_name)


def relation_metrics(env, relation: str) -> dict[str, Any]:
    movable = np.asarray(env.object.get_pose().p, dtype=np.float64)
    reference = np.asarray(env.target_object.get_pose().p, dtype=np.float64)
    delta = movable - reference
    distance_xy = float(np.linalg.norm(delta[:2]))
    relation_ok = bool(delta[0] < 0.0) if relation == "left" else bool(delta[0] > 0.0)
    grippers_open = bool(env.robot.is_left_gripper_open() and env.robot.is_right_gripper_open())
    region = bool(0.08 < distance_xy < 0.2 and relation_ok and abs(float(delta[1])) < 0.05)
    return {
        "success": bool(region and grippers_open),
        "relation_region": region,
        "object_xyz": movable.tolist(),
        "target_xyz": reference.tolist(),
        "object_minus_target_x": float(delta[0]),
        "object_minus_target_y": float(delta[1]),
        "distance_xy": distance_xy,
        "grippers_open": grippers_open,
    }


def install_relation_checker(env, relation: str) -> Callable[[], bool]:
    def check_success(self) -> bool:
        return bool(relation_metrics(self, relation)["success"])

    env.check_success = types.MethodType(check_success, env)
    return env.check_success


def observation_arrays(observation: dict[str, Any]) -> list[np.ndarray]:
    obs = observation["observation"]
    return [
        np.asarray(obs["head_camera"]["rgb"]),
        np.asarray(obs["left_camera"]["rgb"]),
        np.asarray(obs["right_camera"]["rgb"]),
        np.asarray(observation["joint_action"]["vector"]),
    ]


def open_video_writer(path: Path, frame: np.ndarray) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
            "-vcodec", "libx264", "-crf", "23", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")
    return process


def build_policy(args: argparse.Namespace):
    light_root = args.light_wam_root.resolve()
    sys.path[:0] = [str(light_root), str(light_root / "src")]
    from experiments.robotwin.lightwam_policy.deploy_policy import get_model

    os.chdir(light_root)
    return get_model(
        {
            "sim_cfg_name": "sim_robotwin.yaml",
            "sim_task": None,
            "use_training_run_config": True,
            "training_config_path": str(args.training_config.resolve()),
            "ckpt_setting": str(args.checkpoint.resolve()),
            "dataset_stats_path": str(args.dataset_stats.resolve()),
            "device": "cuda",
            "mixed_precision": "bf16",
            "action_horizon": 32,
            "replan_steps": 24,
            "num_inference_steps": 10,
            "sigma_shift": 5.0,
            "seed": 8_400,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "timing_enabled": True,
        }
    )


def setup_environment(args: argparse.Namespace, scene: dict[str, Any], episode_index: int, output: Path | None):
    robotwin_root = args.robotwin_root.resolve()
    os.chdir(robotwin_root)
    sys.path[:0] = [
        str(robotwin_root),
        str(robotwin_root / "script"),
        str(robotwin_root / "description" / "utils"),
    ]
    task_args = load_task_args(robotwin_root, scene["task"], args.task_config)
    task_args["eval_video_save_dir"] = str(output) if output is not None else None
    env = task_class(scene["task"])()
    env.setup_demo(
        now_ep_num=episode_index,
        seed=scene["environment_seed"],
        is_test=True,
        **task_args,
    )
    return env


def run_fixed_probe(args: argparse.Namespace, policy) -> None:
    scene = SCENES[0]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = setup_environment(args, scene, 0, None)
    try:
        observation = env.get_obs()
        arrays = observation_arrays(observation)
        observation_sha = array_digest(arrays)
        np.savez_compressed(
            output_dir / "fixed_observation.npz",
            head=arrays[0], left=arrays[1], right=arrays[2], state=arrays[3],
        )
        seed_all(scene["sampling_seed"])
        rng = capture_rng()
        actions: dict[str, np.ndarray] = {}
        for condition, prompt in (
            ("left", scene["prompts"]["left"]),
            ("left_exact_repeat", scene["prompts"]["left"]),
            ("right", scene["prompts"]["right"]),
        ):
            restore_rng(rng)
            if array_digest(observation_arrays(observation)) != observation_sha:
                raise RuntimeError("Fixed observation mutated between prompt-only calls")
            actions[condition] = np.asarray(policy._infer_action_chunk(observation, prompt))

        action_path = output_dir / "fixed_probe_actions.npz"
        np.savez_compressed(action_path, **actions)
        repeat_delta = np.abs(actions["left"] - actions["left_exact_repeat"])
        prompt_delta = np.abs(actions["left"] - actions["right"])
        result = {
            "schema_version": "vla-wam-shared-v2-light-wam-fixed-probe-v1",
            "status": "complete",
            "pair_id": scene["pair_id"],
            "task": scene["task"],
            "environment_seed": scene["environment_seed"],
            "sampling_seed": scene["sampling_seed"],
            "prompts": scene["prompts"],
            "observation_sha256": observation_sha,
            "observation_npz_sha256": sha256_file(output_dir / "fixed_observation.npz"),
            "actions_npz_sha256": sha256_file(action_path),
            "action_shape": list(actions["left"].shape),
            "all_actions_finite": bool(all(np.isfinite(value).all() for value in actions.values())),
            "left_repeat_array_equal": bool(np.array_equal(actions["left"], actions["left_exact_repeat"])),
            "left_repeat_allclose_atol_1e_6": bool(np.allclose(actions["left"], actions["left_exact_repeat"], atol=1e-6, rtol=0.0)),
            "left_repeat_max_abs_delta": float(repeat_delta.max(initial=0.0)),
            "left_right_array_distinct": bool(not np.array_equal(actions["left"], actions["right"])),
            "left_right_mean_abs_delta": float(prompt_delta.mean()),
            "left_right_max_abs_delta": float(prompt_delta.max(initial=0.0)),
            "future_interface": "action_only_infer_action",
            "imagined_future_evidence": None,
        }
        (output_dir / "fixed_probe.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close_env(clear_cache=False)


def run_episode(
    args: argparse.Namespace,
    policy,
    scene: dict[str, Any],
    relation: str,
    episode_index: int,
    expected_initial_sha: str | None,
) -> tuple[dict[str, Any], str]:
    condition = f"direct_command__{relation}"
    condition_dir = (
        args.output_dir.resolve() / scene["pair_id"] / scene["task"]
        / f"environment_seed_{scene['environment_seed']}"
        / f"sampling_seed_{scene['sampling_seed']}" / condition
    )
    condition_dir.mkdir(parents=True, exist_ok=True)
    env = setup_environment(args, scene, episode_index, condition_dir)
    video_writer = None
    started = time.perf_counter()
    executed_actions: list[np.ndarray] = []
    trajectory: list[dict[str, Any]] = []
    try:
        observation = env.get_obs()
        initial_sha = array_digest(observation_arrays(observation))
        if expected_initial_sha is not None and initial_sha != expected_initial_sha:
            raise RuntimeError(
                f"Paired initial observation mismatch: {initial_sha} != {expected_initial_sha}"
            )
        prompt = scene["prompts"][relation]
        env.set_instruction(prompt)
        install_relation_checker(env, relation)
        policy.reset()
        policy.seed = scene["sampling_seed"]
        seed_all(scene["sampling_seed"])

        initial = relation_metrics(env, relation)
        initial_z = initial["object_xyz"][2]
        trajectory.append({"action_step": 0, **initial})
        frame = env._get_eval_video_frame(live=False)
        simulator_video = condition_dir / "simulator.mp4"
        video_writer = open_video_writer(simulator_video, frame)
        env._set_eval_video_ffmpeg(video_writer)

        original_take_action = env.take_action

        def traced_take_action(self, action, action_type=None, **kwargs):
            value = np.asarray(action, dtype=np.float32).copy()
            executed_actions.append(value)
            result = original_take_action(value, action_type=action_type, **kwargs)
            trajectory.append({"action_step": int(self.take_action_cnt), **relation_metrics(self, relation)})
            return result

        env.take_action = types.MethodType(traced_take_action, env)
        while env.take_action_cnt < min(env.step_lim, args.max_actions) and not env.eval_success:
            next_observation = env.get_obs() if policy.should_request_observation() else None
            policy.step(env, next_observation)

        requested = relation_metrics(env, relation)
        native_relation = RELATION_BY_TASK[scene["task"]]
        native = relation_metrics(env, native_relation)
        opposite_relation = "right" if relation == "left" else "left"
        opposite = relation_metrics(env, opposite_relation)
        action_array = np.asarray(executed_actions, dtype=np.float32)
        action_path = condition_dir / "action_trace.npz"
        np.savez_compressed(action_path, executed=action_array)
        (condition_dir / "trajectory.json").write_text(json.dumps(trajectory) + "\n")
        max_lift = max((row["object_xyz"][2] - initial_z for row in trajectory), default=0.0)
        result = {
            "schema_version": "vla-wam-shared-v2-light-wam-episode-v1",
            "model_id": "light_wam_robotwin",
            "pair_id": scene["pair_id"],
            "task": scene["task"],
            "environment_seed": scene["environment_seed"],
            "sampling_seed": scene["sampling_seed"],
            "prompt_family": "direct_command",
            "requested_relation": relation,
            "prompt": prompt,
            "prompt_controller": "episode_static",
            "oracle_actions": 0,
            "dynamic_prompt_switches": 0,
            "initial_observation_sha256": initial_sha,
            "object_name": str(env.selected_modelname_A),
            "object_model_id": int(env.selected_model_id_A),
            "target_name": str(env.selected_modelname_B),
            "target_model_id": int(env.selected_model_id_B),
            "actions_executed": int(action_array.shape[0]),
            "requested_success": requested["success"],
            "native_task_success": native["success"],
            "opposite_relation_success": opposite["success"],
            "initial": initial,
            "final": requested,
            "max_object_lift_m": float(max_lift),
            "wall_seconds": time.perf_counter() - started,
            "action_trace": {
                "path": str(action_path),
                "sha256": sha256_file(action_path),
                "count": int(action_array.shape[0]),
                "shape": list(action_array.shape),
            },
            "trajectory_path": str(condition_dir / "trajectory.json"),
            "simulator_video": str(simulator_video),
            "future_interface": "action_only_infer_action",
            "imagined_future_video": None,
            "imagined_future_artifact": None,
        }
        (condition_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"{scene['pair_id']} {relation}: success={requested['success']} "
            f"dx={requested['object_minus_target_x']:+.4f} actions={action_array.shape[0]}",
            flush=True,
        )
        return result, initial_sha
    finally:
        if video_writer is not None:
            env._del_eval_video_ffmpeg()
        env.close_env(clear_cache=False)


def write_episode_summaries(output_dir: Path, results: list[dict[str, Any]], invalid: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w") as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")
    with (output_dir / "invalid_attempts.jsonl").open("w") as handle:
        for record in invalid:
            handle.write(json.dumps(record) + "\n")
    if results:
        columns = (
            "pair_id", "task", "environment_seed", "sampling_seed", "requested_relation",
            "requested_success", "actions_executed", "prompt",
        )
        with (output_dir / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row[key] for key in columns} for row in results)
    manifest = {
        "schema_version": "vla-wam-shared-v2-light-wam-raw-manifest-v1",
        "valid_episode_count": len(results),
        "requested_success_count": sum(bool(row["requested_success"]) for row in results),
        "invalid_attempt_count": len(invalid),
        "expected_episode_count": 6,
        "future_interface": "action_only_infer_action",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_episodes(args: argparse.Namespace, policy) -> None:
    results: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    episode_index = 0
    for scene in SCENES:
        initial_sha: str | None = None
        for relation in ("left", "right"):
            try:
                result, observed_sha = run_episode(
                    args, policy, scene, relation, episode_index, initial_sha
                )
                results.append(result)
                initial_sha = observed_sha
            except Exception as error:
                invalid.append(
                    {
                        "pair_id": scene["pair_id"],
                        "requested_relation": relation,
                        "environment_seed": scene["environment_seed"],
                        "sampling_seed": scene["sampling_seed"],
                        "stage": "infrastructure_or_runtime",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "included_in_model_denominator": False,
                    }
                )
                print(traceback.format_exc(), file=sys.stderr, flush=True)
            finally:
                episode_index += 1
                write_episode_summaries(args.output_dir.resolve(), results, invalid)


def main() -> None:
    args = parse_args()
    for path in (
        args.light_wam_root, args.robotwin_root, args.checkpoint,
        args.training_config, args.dataset_stats,
    ):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    policy = build_policy(args)
    if args.mode == "fixed-probe":
        run_fixed_probe(args, policy)
    else:
        run_episodes(args, policy)


if __name__ == "__main__":
    main()
