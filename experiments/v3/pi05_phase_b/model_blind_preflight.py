#!/usr/bin/env python3
"""Fresh zero-request Isaac/RTX preflight for one V3-B002 simulator lane."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--candidate-sha256", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--amendment-id", choices=("V3-B002", "V3-B003"), default="V3-B002")
parser.add_argument("--environment-seed", type=int, default=9400)
parser.add_argument("--pod", required=True)
parser.add_argument("--pod-uid", required=True)
parser.add_argument("--gpu-uuid", required=True)
from robolab.eval.runner import add_common_eval_args  # noqa: E402
add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1 or not args_cli.headless:
    parser.error("preflight requires one headless environment")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("preflight requires realtime/balanced RTX")

root = args_cli.study_root.resolve()
sys.path.insert(0, str(root))
os.environ["VLA_WAM_V3B_FIXTURE_CANDIDATE"] = str(args_cli.candidate.resolve())
os.environ["VLA_WAM_V3B_FIXTURE_SHA256"] = args_cli.candidate_sha256
from experiments.v3.pi05_phase_b.contract import (  # noqa: E402
    OPENPI_COMMIT, PROMPTS, ROBOLAB_COMMIT, STUDY_ID,
    sha256_file,
)
from experiments.v3.pi05_phase_b.runtime import MODEL_BLIND_SCHEMA, adapter_contract_sha256  # noqa: E402

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab  # noqa: E402
import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


TASKS = {
    ("control", "left"): ("control_left.py", "V3B002Pi05ControlLeftTask"),
    ("control", "right"): ("control_right.py", "V3B002Pi05ControlRightTask"),
    ("position_mirrored", "left"): ("position_mirrored_left.py", "V3B002Pi05PositionMirroredLeftTask"),
    ("position_mirrored", "right"): ("position_mirrored_right.py", "V3B002Pi05PositionMirroredRightTask"),
}
OBJECTS = ("rubiks_cube", "bowl", "banana")
CAMERAS = ("over_shoulder_left_camera", "over_shoulder_right_camera", "head_camera", "wrist_cam")
DREAMZERO_SOURCES = {
    "amendment": (
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/"
        "post_result_dreamzero_mirror_v3b003_amendment.json",
        "ba22681ae4d7f748e375617617d9e130e6f1bd5bc0af1e7a995365b145a470fc",
    ),
    "cells": (
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/"
        "dreamzero_mirror_v3b003_cells.jsonl",
        "a6d0f0a5d4c7cdfa5d3de95d44d7b11f42750a76a603ff8c2e44848e34b8f70d",
    ),
    "manifest": (
        "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/"
        "dreamzero_mirror_v3b003_manifest.json",
        "efe50df701193e48b981c025ea3b4d27a80e3bdf83216e38a98a63e27061cb23",
    ),
}


def _numeric(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


def _hold(obs: dict, device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, 8):
        raise RuntimeError("model-blind hold action must be [1,8]")
    return action


def _frame(obs: dict, name: str) -> np.ndarray:
    value = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy(), dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.ptp(value):
        raise RuntimeError(f"blank or malformed RGB view: {name}")
    return value


def _combined(obs: dict) -> np.ndarray:
    frames = [_frame(obs, name) for name in ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")]
    height = min(frame.shape[0] for frame in frames)
    frames = [cv2.resize(frame, (round(frame.shape[1]*height/frame.shape[0]), height)) for frame in frames]
    return np.concatenate(frames, axis=1)


def _fresh_physical_reset(env):
    """Force and attest an independent RoboLab physical reset."""

    counter = getattr(env, "episode_length_buf", None)
    if counter is None or not hasattr(counter, "zero_"):
        raise RuntimeError("RoboLab does not expose a resettable episode_length_buf")
    before = [int(item) for item in _numeric(counter)]
    counter.zero_()
    after_zero = [int(item) for item in _numeric(counter)]
    if after_zero != [0]:
        raise RuntimeError(f"failed to arm fresh physical reset: {after_zero}")
    obs, info = env.reset()
    after_reset = [int(item) for item in _numeric(counter)]
    if after_reset != [0]:
        raise RuntimeError(f"fresh physical reset did not clear episode counter: {after_reset}")
    return obs, info, {
        "episode_length_buf_before_force_reset": before,
        "episode_length_buf_after_zero": after_zero,
        "episode_length_buf_after_reset": after_reset,
    }


def _file(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    try:
        if args_cli.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite preflight: {args_cli.output_dir}")
        if (
            args_cli.candidate_sha256 != "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88"
            or not args_cli.candidate.is_file()
            or sha256_file(args_cli.candidate) != args_cli.candidate_sha256
        ):
            raise RuntimeError("candidate is not the exact Nano B001 fixture")
        candidate = json.loads(args_cli.candidate.read_text(encoding="utf-8"))
        if candidate.get("model_request_count") != 0 or candidate.get("behavioral_episode_count") != 0:
            raise RuntimeError("fixture candidate is not model blind")
        robolab_head = subprocess.check_output(["git", "-C", str(args_cli.robolab_root), "rev-parse", "HEAD"], text=True).strip()
        robolab_diff = subprocess.check_output(["git", "-C", str(args_cli.robolab_root), "status", "--porcelain=v1", "--untracked-files=no"], text=True)
        study_diff = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"], text=True)
        if robolab_head != ROBOLAB_COMMIT or robolab_diff or study_diff:
            raise RuntimeError("preflight requires clean tracked frozen study/RoboLab checkouts")
        if not Path(robolab.__file__).resolve().is_relative_to(args_cli.robolab_root.resolve()):
            raise RuntimeError("effective RoboLab import is outside the pinned worktree")
        design_sources = {}
        if args_cli.amendment_id == "V3-B003":
            for name, (relative, expected) in DREAMZERO_SOURCES.items():
                path = root / relative
                if not path.is_file() or sha256_file(path) != expected:
                    raise RuntimeError(f"DreamZero V3-B003 {name} binding changed")
                design_sources[name] = _file(path)
        gpu_lines = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).splitlines()
        gpu_line = next((line for line in gpu_lines if args_cli.gpu_uuid in line), None)
        if gpu_line is None:
            raise RuntimeError("assigned GPU UUID is not visible")
        args_cli.output_dir.mkdir(parents=True, exist_ok=False)
        set_output_dir(str((args_cli.output_dir/"native").resolve()))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        task_root = root/"experiments/v3/pi05_phase_b/task_files"
        auto_register_droid_envs(
            task=[str(task_root/TASKS[key][0]) for key in TASKS], cameras=WRIST_LEFT_RIGHT_HEAD
        )
        rows = []
        video_records = {}
        for (arm, relation), (_, task_name) in TASKS.items():
            env, env_cfg = create_env(
                task_name, device=args_cli.device, seed=args_cli.environment_seed,
                num_envs=1, instruction_type="default",
                policy=(
                    "v3b003_dreamzero_model_blind_preflight"
                    if args_cli.amendment_id == "V3-B003"
                    else "v3b002_pi05_model_blind_preflight"
                ),
                renderer=args_cli.renderer, rendering_mode=args_cli.rendering_type,
            )
            video_path = args_cli.output_dir/f"{arm}_{relation}_resets.mp4"
            writer = None
            repeats = []
            try:
                if env_cfg.instruction != PROMPTS[relation]:
                    raise RuntimeError("task wrapper prompt bytes changed")
                for repeat in range(3):
                    obs, _, reset_attestation = _fresh_physical_reset(env)
                    hold = _hold(obs, env.device)
                    for _ in range(60):
                        obs, _, terminated, truncated, _ = env.step(hold)
                        if bool(terminated[0]) or bool(truncated[0]):
                            raise RuntimeError("task terminated during 60-step settle")
                    maxima = {name: {"linear_m_s": 0.0, "angular_rad_s": 0.0} for name in OBJECTS}
                    for _ in range(15):
                        obs, _, terminated, truncated, _ = env.step(hold)
                        if bool(terminated[0]) or bool(truncated[0]):
                            raise RuntimeError("task terminated during 15-step stability window")
                        world = get_world(env)
                        for name in OBJECTS:
                            velocity = np.asarray(_numeric(world.get_velocity(name, env_id=0)))
                            maxima[name]["linear_m_s"] = max(maxima[name]["linear_m_s"], float(np.max(np.abs(velocity[:3]))))
                            maxima[name]["angular_rad_s"] = max(maxima[name]["angular_rad_s"], float(np.max(np.abs(velocity[3:]))))
                    if any(value["linear_m_s"] > .02 or value["angular_rad_s"] > .2 for value in maxima.values()):
                        raise RuntimeError("object failed registered 60+15 stability tolerance")
                    world = get_world(env)
                    positions = {name: _numeric(world.get_pose(name, env_id=0)[0]) for name in OBJECTS}
                    expected = candidate["layouts"][arm]["positions_robot_base_m"]
                    if any(max(abs(a-b) for a, b in zip(positions[name], expected[name])) > .003 for name in OBJECTS):
                        raise RuntimeError("live movable-object center missed exact B001 tolerance")
                    left = bool(object_left_of(env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
                    right = bool(object_right_of(env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
                    if left or right:
                        raise RuntimeError("reset is not neutral under frozen predicates")
                    views = {name: {"shape": list(_frame(obs, name).shape), "pixel_range": int(np.ptp(_frame(obs, name)))} for name in CAMERAS}
                    frame = _combined(obs)
                    if writer is None:
                        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("viewport proof writer did not open")
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    repeats.append({
                        "repeat": repeat,
                        "fresh_physical_reset": reset_attestation,
                        "positions_robot_base_m": positions,
                        "stability": maxima,
                        "views": views,
                    })
            finally:
                if writer is not None:
                    writer.release()
                env.close()
            capture = cv2.VideoCapture(str(video_path))
            ok, _ = capture.read()
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if not ok or frame_count != 3:
                raise RuntimeError("viewport proof video did not decode all reset frames")
            video_records[f"{arm}:{relation}"] = {**_file(video_path), "decoded_frame_count": frame_count}
            rows.append({"arm": arm, "relation": relation, "task_name": task_name, "passed": True, "repeat_resets": repeats})
        action_path = args_cli.output_dir/"writer_probe_actions.npy"
        np.save(action_path, np.zeros((15, 8), dtype=np.float32), allow_pickle=False)
        raw_path = args_cli.output_dir/"writer_probe_raw.jsonl"
        raw_path.write_text(json.dumps({"model_blind": True})+"\n")
        if np.load(action_path, allow_pickle=False).shape != (15, 8) or json.loads(raw_path.read_text()) != {"model_blind": True}:
            raise RuntimeError("raw/action writer readback failed")
        dreamzero = args_cli.amendment_id == "V3-B003"
        output = {
            "schema_version": (
                "vla-wam-shared-v3b-dreamzero-model-blind-preflight-v1"
                if dreamzero else MODEL_BLIND_SCHEMA
            ),
            "study_id": STUDY_ID,
            "amendment_id": args_cli.amendment_id,
            "model_id": (
                "dreamzero_droid_action_cfg" if dreamzero else "pi05_current_stack_droid"
            ),
            "passed": True,
            "model_request_count": 0, "behavioral_episode_count": 0,
            "pod": args_cli.pod, "pod_uid": args_cli.pod_uid, "gpu_uuid": args_cli.gpu_uuid,
            "gpu_query": gpu_line, "renderer_backend": "realtime RTX Vulkan",
            "all_required_rgb_views_nonblank": True, "viewport_writer_passed": True,
            "raw_jsonl_writer_passed": True, "action_trace_writer_passed": True,
            "fixture_positions_match": True, "neutral_reset_passed": True,
            "settle_steps": 60, "stable_window_steps": 15,
            "tasks": rows, "viewport_evidence": video_records,
            "fresh_writer_evidence": {"action": _file(action_path), "raw_jsonl": _file(raw_path)},
            "fixture_candidate": _file(args_cli.candidate),
            "design_sources": design_sources,
            "b002_adapter_contract_sha256": (
                None if dreamzero else adapter_contract_sha256(root)
            ),
            "robolab_commit": robolab_head,
            "openpi_commit": None if dreamzero else OPENPI_COMMIT,
            "dreamzero_identity_binding": (
                "V2-A015:dreamzero_action_cfg_s2" if dreamzero else None
            ),
        }
        path = args_cli.output_dir/"model_blind_preflight.json"
        path.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
        print(json.dumps({"path": str(path.resolve()), "sha256": sha256_file(path)}, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
