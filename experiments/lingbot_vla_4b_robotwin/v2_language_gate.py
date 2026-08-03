#!/usr/bin/env python3
"""Frozen six-cell LingBot-VLA RoboTwin direct-command gate.

This integration is intentionally external to the steerable evidence checkout.
It loads the released LingBot-VLA policy directly, preserves the official
50-action RoboTwin chunk, and changes only the episode-static prompt inside a
matched LEFT/RIGHT pair.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml

from deploy.lingbot_vla_policy import LingbotVLAServer


RELATION_BY_TASK = {
    "place_a2b_left": "left",
    "place_a2b_right": "right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--mode", choices=("probe", "episode"), required=True)
    parser.add_argument("--fixed-observation", type=Path)
    parser.add_argument("--task", choices=sorted(RELATION_BY_TASK))
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--requested-relation", choices=("left", "right"), required=True)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--use-length", type=int, default=50)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_trace_record(path: Path, actions: np.ndarray) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "count": int(actions.shape[0]),
        "shape": [int(size) for size in actions.shape],
    }


def opposite(relation: str) -> str:
    return "right" if relation == "left" else "left"


def task_class(task_name: str):
    module = __import__(f"envs.{task_name}", fromlist=[task_name])
    return getattr(module, task_name)


def load_robotwin_setup_kwargs(task_name: str) -> dict:
    """Mirror RoboTwin's demo_clean configuration loading without LeRobot extensions."""
    from envs import CONFIGS_PATH  # noqa: PLC0415

    with open(Path(CONFIGS_PATH) / "demo_clean.yml", encoding="utf-8") as handle:
        setup = yaml.safe_load(handle)
    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", encoding="utf-8") as handle:
        embodiment_types = yaml.safe_load(handle)
    embodiment = setup.get("embodiment", ["aloha-agilex"])
    if len(embodiment) != 1:
        raise ValueError(f"Expected the frozen single dual-arm embodiment, got {embodiment}")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    setup["left_robot_file"] = robot_file
    setup["right_robot_file"] = robot_file
    setup["dual_arm_embodied"] = True
    with open(Path(robot_file) / "config.yml", encoding="utf-8") as handle:
        setup["left_embodiment_config"] = yaml.safe_load(handle)
    with open(Path(robot_file) / "config.yml", encoding="utf-8") as handle:
        setup["right_embodiment_config"] = yaml.safe_load(handle)
    with open(Path(CONFIGS_PATH) / "_camera_config.yml", encoding="utf-8") as handle:
        camera_config = yaml.safe_load(handle)
    head_camera = setup["camera"]["head_camera_type"]
    setup["head_camera_h"] = camera_config[head_camera]["h"]
    setup["head_camera_w"] = camera_config[head_camera]["w"]
    setup["render_freq"] = 0
    setup["task_name"] = task_name
    setup["task_config"] = "demo_clean"
    return setup


def relation_metrics(env, relation: str) -> dict:
    object_pose = np.asarray(env.object.get_pose().p, dtype=np.float64)
    target_pose = np.asarray(env.target_object.get_pose().p, dtype=np.float64)
    delta = object_pose - target_pose
    distance_xy = float(np.linalg.norm(delta[:2]))
    relation_ok = bool(delta[0] < 0.0) if relation == "left" else bool(delta[0] > 0.0)
    grippers_open = bool(env.robot.is_left_gripper_open() and env.robot.is_right_gripper_open())
    relation_region = bool(
        0.08 < distance_xy < 0.2 and relation_ok and abs(float(delta[1])) < 0.05
    )
    success = bool(relation_region and grippers_open)
    return {
        "success": success,
        "relation_region": relation_region,
        "object_xyz": object_pose.tolist(),
        "target_xyz": target_pose.tolist(),
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


def open_video_writer(path: Path, width: int, height: int, ffmpeg: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")
    return process


def load_model(args: argparse.Namespace) -> LingbotVLAServer:
    os.environ["QWEN25_PATH"] = str(args.qwen_path.resolve())
    seed_everything(args.sampling_seed)
    return LingbotVLAServer(
        path_to_pi_model=str(args.checkpoint.resolve()),
        use_length=args.use_length,
        use_bf16=True,
        use_fp32=False,
        num_denoising_step=args.num_denoising_steps,
        use_compile=False,
    )


def reset_model(model: LingbotVLAServer, robot_name: str = "robotwin") -> None:
    """Run the official reset from its repository root for relative configs."""
    previous = Path.cwd()
    try:
        os.chdir(Path(__file__).resolve().parents[1])
        model.reset(robot_name)
    finally:
        os.chdir(previous)


def model_input(observation: dict, prompt: str) -> dict:
    cameras = observation["observation"]
    return {
        "observation.images.cam_high": np.asarray(cameras["head_camera"]["rgb"], dtype=np.uint8),
        "observation.images.cam_left_wrist": np.asarray(
            cameras["left_camera"]["rgb"], dtype=np.uint8
        ),
        "observation.images.cam_right_wrist": np.asarray(
            cameras["right_camera"]["rgb"], dtype=np.uint8
        ),
        "observation.state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "task": prompt,
    }


def validate_action_chunk(output: dict, use_length: int) -> np.ndarray:
    if set(output) != {"action"}:
        raise RuntimeError(f"Expected only native 'action' output, got keys={sorted(output)}")
    action = np.asarray(output["action"], dtype=np.float32)
    if action.shape != (use_length, 14):
        raise RuntimeError(f"Expected action shape {(use_length, 14)}, got {action.shape}")
    if not np.isfinite(action).all():
        raise RuntimeError("Action chunk contains NaN or infinity")
    return action


def frozen_prompt(args: argparse.Namespace, env) -> str:
    tools = args.study_root.resolve() / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from vla_wam_v2_protocol import (  # noqa: PLC0415
        canonical_short_object_name,
        first_seen_object_description,
        load_protocol,
        render_prompt,
    )

    movable = first_seen_object_description(
        args.robotwin_root.resolve(), str(env.selected_modelname_A), int(env.selected_model_id_A)
    )
    reference = first_seen_object_description(
        args.robotwin_root.resolve(), str(env.selected_modelname_B), int(env.selected_model_id_B)
    )
    protocol = load_protocol(args.study_root.resolve() / "artifacts/vla_wam_shared_v2/protocol.json")
    return render_prompt(
        protocol,
        family_id="direct_command",
        direction=args.requested_relation,
        movable=movable,
        movable_short=canonical_short_object_name(str(env.selected_modelname_A)),
        reference=reference,
        arena="robotwin_place_a2b",
    )


def probe_prompt(args: argparse.Namespace, metadata: dict) -> str:
    tools = args.study_root.resolve() / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from vla_wam_v2_protocol import load_protocol, render_prompt  # noqa: PLC0415

    protocol = load_protocol(args.study_root.resolve() / "artifacts/vla_wam_shared_v2/protocol.json")
    return render_prompt(
        protocol,
        family_id="direct_command",
        direction=args.requested_relation,
        movable=metadata["movable"],
        movable_short=metadata["object_name"].replace("_", " "),
        reference=metadata["reference"],
        arena="robotwin_place_a2b",
    )


def run_probe(args: argparse.Namespace) -> None:
    if args.fixed_observation is None:
        raise ValueError("--fixed-observation is required for probe mode")
    fixed_root = args.fixed_observation.resolve()
    metadata = json.loads((fixed_root / "metadata.json").read_text())
    with np.load(fixed_root / "observation.npz") as data:
        item = {
            "observation.images.cam_high": data["cam_high"].copy(),
            "observation.images.cam_left_wrist": data["cam_left_wrist"].copy(),
            "observation.images.cam_right_wrist": data["cam_right_wrist"].copy(),
            "observation.state": data["state"].copy(),
            "task": probe_prompt(args, metadata),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    reset_model(model)
    seed_everything(args.sampling_seed)
    started = time.perf_counter()
    action = validate_action_chunk(model.infer(item), args.use_length)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    action_path = args.output_dir / "action.npz"
    np.savez_compressed(action_path, action=action)
    result = {
        "mode": "fixed_observation_probe",
        "requested_relation": args.requested_relation,
        "prompt": item["task"],
        "prompt_utf8_hex": item["task"].encode().hex(),
        "sampling_seed": args.sampling_seed,
        "observation_path": str(fixed_root / "observation.npz"),
        "observation_sha256": metadata["observation_sha256"],
        "action": action_trace_record(action_path, action),
        "action_dtype": str(action.dtype),
        "finite": bool(np.isfinite(action).all()),
        "inference_seconds": elapsed,
        "gpu_peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def run_episode(args: argparse.Namespace) -> None:
    if args.task is None or args.environment_seed is None:
        raise ValueError("--task and --environment-seed are required for episode mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.sampling_seed)
    env = task_class(args.task)()
    video_writer = None
    model = None
    trajectory: list[dict] = []
    predicted_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        setup = load_robotwin_setup_kwargs(args.task)
        setup.update(
            eval_mode=True,
            save_data=False,
            collect_data=False,
            eval_video_save_dir=str(args.output_dir),
        )
        env.setup_demo(seed=args.environment_seed, is_test=True, **setup)
        prompt = frozen_prompt(args, env)
        env.set_instruction(prompt)
        initial = relation_metrics(env, args.requested_relation)
        install_relation_checker(env, args.requested_relation)
        trajectory.append({"action_step": 0, **initial})
        simulator_video = args.output_dir / "simulator.mp4"
        video_writer = open_video_writer(
            simulator_video, int(setup["head_camera_w"]), int(setup["head_camera_h"]), args.ffmpeg
        )
        env._set_eval_video_ffmpeg(video_writer)

        model = load_model(args)
        reset_model(model)
        seed_everything(args.sampling_seed)
        action_limit = min(int(env.step_lim), args.max_actions)
        while env.take_action_cnt < action_limit and not env.eval_success:
            infer_started = time.perf_counter()
            chunk = validate_action_chunk(model.infer(model_input(env.get_obs(), prompt)), args.use_length)
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - infer_started
            predicted_actions.append(chunk.copy())
            for action in chunk:
                if env.take_action_cnt >= action_limit or env.eval_success:
                    break
                executed = np.asarray(action, dtype=np.float64)
                env.take_action(executed)
                executed_actions.append(executed.copy())
                trajectory.append(
                    {"action_step": int(env.take_action_cnt), **relation_metrics(env, args.requested_relation)}
                )

        final = relation_metrics(env, args.requested_relation)
        task_relation = RELATION_BY_TASK[args.task]
        if video_writer is not None:
            env._del_eval_video_ffmpeg()
            video_writer = None
        action_trace_path = args.output_dir / "action_trace.npz"
        executed_trace = np.asarray(executed_actions, dtype=np.float64)
        np.savez_compressed(
            action_trace_path,
            executed=executed_trace,
            predicted_chunks=np.asarray(predicted_actions, dtype=np.float32),
        )
        trajectory_path = args.output_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory) + "\n")
        result = {
            "model_id": "lingbot_vla_4b_robotwin",
            "future_interface": "none",
            "task": args.task,
            "pair_id": {
                4_300_000: "robotwin_pair_00",
                4_300_001: "robotwin_pair_01",
                4_300_002: "robotwin_pair_02",
            }[args.environment_seed],
            "environment_seed": args.environment_seed,
            "sampling_seed": args.sampling_seed,
            "prompt_family": "direct_command",
            "task_relation": task_relation,
            "requested_relation": args.requested_relation,
            "condition_alignment": (
                "native_direction" if task_relation == args.requested_relation else "counterfactual_direction"
            ),
            "prompt": prompt,
            "prompt_utf8_hex": prompt.encode().hex(),
            "object_name": str(env.selected_modelname_A),
            "object_model_id": int(env.selected_model_id_A),
            "target_name": str(env.selected_modelname_B),
            "target_model_id": int(env.selected_model_id_B),
            "actions_executed": int(env.take_action_cnt),
            "requested_success": final["success"],
            "original_task_success": relation_metrics(env, task_relation)["success"],
            "opposite_relation_success": relation_metrics(env, opposite(args.requested_relation))["success"],
            "initial": initial,
            "final": final,
            "inference_seconds": inference_seconds,
            "wall_seconds": time.perf_counter() - started,
            "gpu_peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "simulator_video": str(simulator_video),
            "simulator_video_sha256": sha256_file(simulator_video),
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "action_trace": action_trace_record(action_trace_path, executed_trace),
        }
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if video_writer is not None:
            env._del_eval_video_ffmpeg()
        env.close_env(clear_cache=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    os.chdir(args.robotwin_root.resolve())
    sys.path[:0] = [
        str(args.robotwin_root.resolve()),
        str(args.robotwin_root.resolve() / "script"),
        str(args.robotwin_root.resolve() / "description" / "utils"),
    ]
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "probe":
        run_probe(args)
    else:
        run_episode(args)


if __name__ == "__main__":
    main()
