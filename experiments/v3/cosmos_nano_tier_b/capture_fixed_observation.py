#!/usr/bin/env python3
"""Capture one zero-request fixed observation for a V3-B008/B009 arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--amendment-id", choices=("V3-B008", "V3-B009"), required=True)
BOOTSTRAP.add_argument("--release-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--arm", required=True)
BOOTSTRAP.add_argument("--output", type=Path, required=True)
BOOTSTRAP.add_argument("--pod", required=True)
BOOTSTRAP.add_argument("--pod-uid", required=True)
BOOTSTRAP.add_argument("--gpu-uuid", required=True)
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (  # noqa: E402
    ACTION_DIM,
    CONFIG,
    STUDY_ID,
    load_release,
    sha256_bytes,
    sha256_file,
)


release = load_release(study_root, bootstrap.amendment_id, bootstrap.release_manifest)
if bootstrap.arm not in release.config["arms"]:
    BOOTSTRAP.error("arm is outside the exact release")
cell = release.probe_cell(bootstrap.arm, "left")
if bootstrap.output.exists() or bootstrap.output.with_suffix(".capture.json").exists():
    BOOTSTRAP.error("refusing to overwrite fixed-observation evidence")

parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1 or not args_cli.headless:
    parser.error("fixed capture requires one headless environment")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("fixed capture requires realtime/balanced RTX")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402
from policies.cosmos3.client import Cosmos3Client  # noqa: E402


TASKS = {
    ("V3-B008", "target_start_left"): ("v3b008_target_start_left_left.py", "V3B008TargetStartLeftLeftGateTask"),
    ("V3-B008", "target_start_center"): ("v3b008_target_start_center_left.py", "V3B008TargetStartCenterLeftGateTask"),
    ("V3-B008", "target_start_right"): ("v3b008_target_start_right_left.py", "V3B008TargetStartRightLeftGateTask"),
    ("V3-B009", "cube_target_bowl_reference"): ("v3b009_cube_target_bowl_reference_left.py", "V3B009CubeTargetBowlReferenceLeftGateTask"),
    ("V3-B009", "bowl_target_cube_reference"): ("v3b009_bowl_target_cube_reference_left.py", "V3B009BowlTargetCubeReferenceLeftGateTask"),
}


def _quat_inverse_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    inverse_xyz = -xyz
    return 2 * np.dot(inverse_xyz, v) * inverse_xyz + (w * w - np.dot(inverse_xyz, inverse_xyz)) * v + 2 * w * np.cross(inverse_xyz, v)


def _hold_action(obs: dict[str, Any], device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, ACTION_DIM):
        raise RuntimeError(f"unexpected hold action shape {tuple(action.shape)}")
    return action


def _array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": sha256_bytes(array.tobytes())}


def main() -> None:
    bootstrap.output.parent.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(bootstrap.output.parent / f"native_{bootstrap.arm}"))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    filename, task_name = TASKS[(bootstrap.amendment_id, bootstrap.arm)]
    task_path = study_root / "experiments/v3/prospective_tier_b_gates/task_files" / filename
    auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
    env, env_cfg = create_env(
        task_name,
        device=args_cli.device,
        seed=cell.seed,
        num_envs=1,
        instruction_type="default",
        policy=f"{bootstrap.amendment_id.lower()}_zero_request_fixed_observation",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    try:
        obs, _ = env.reset()
        action = _hold_action(obs, env.device)
        for _ in range(75):
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("fixed-observation environment terminated while settling")
        robot = env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        observed_positions: dict[str, list[float]] = {}
        errors: dict[str, float] = {}
        for name, expected in cell.row["fixture_positions_robot_base_m"].items():
            world = env.scene[name].data.root_pos_w[0].detach().cpu().numpy()
            observed = _quat_inverse_rotate_wxyz(robot_quat, world - robot_pos)
            observed_positions[name] = observed.astype(float).tolist()
            errors[name] = float(np.max(np.abs(observed - np.asarray(expected, dtype=np.float64))))
        if max(errors.values()) > 0.005:
            raise RuntimeError(f"post-settle fixture differs from release: {errors}")
        target = cell.row.get("target_object", "rubiks_cube")
        reference = cell.row.get("reference_object", "bowl")
        left = bool(object_left_of(env, object=target, reference_object=reference, frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
        right = bool(object_right_of(env, object=target, reference_object=reference, frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
        if left or right:
            raise RuntimeError("fixed observation is not neutral under the arm-specific scorer")
        packer = object.__new__(Cosmos3Client)
        packer._image_w = Cosmos3Client.IMAGE_W
        packer._image_h = Cosmos3Client.IMAGE_H
        extracted = Cosmos3Client._extract_observation(packer, obs)
        request = Cosmos3Client._pack_request(packer, extracted, env_cfg.instruction)
        if request.pop("prompt") != cell.row["prompt"]:
            raise RuntimeError("official client packed unexpected prompt bytes")
        arrays = {
            "image": np.asarray(request["observation/image"]),
            "joint_position": np.asarray(request["observation/joint_position"]),
            "gripper_position": np.asarray(request["observation/gripper_position"]),
        }
        if arrays["image"].shape != (540, 640, 3):
            raise RuntimeError(f"official packed image shape changed: {arrays['image'].shape}")
        for name in ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"):
            view = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy())
            if view.ndim != 3 or view.shape[-1] != 3 or not np.ptp(view):
                raise RuntimeError(f"blank or malformed RTX view: {name}")
        np.savez(bootstrap.output, **arrays)
        report = {
            "schema_version": "vla-wam-shared-v3b008-v3b009-nano-fixed-observation-capture-v1",
            "study_id": STUDY_ID,
            "amendment_id": bootstrap.amendment_id,
            "arm": bootstrap.arm,
            "registered_cell_id": cell.cell_id,
            "environment_seed": cell.seed,
            "prompt_used_for_official_packing": env_cfg.instruction,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "settle_steps": 60,
            "stable_window_steps": 15,
            "neutral_reset": True,
            "observed_positions_robot_base_m": observed_positions,
            "max_abs_position_errors_m": errors,
            "observation_npz": {"path": str(bootstrap.output.resolve()), "sha256": sha256_file(bootstrap.output), "bytes": bootstrap.output.stat().st_size},
            "array_fingerprints": {name: _array_record(value) for name, value in arrays.items()},
            "release_manifest_sha256": release.manifest_sha256,
            "model_blind_gate_sha256": release.config["gate_sha256"],
            "pod": bootstrap.pod,
            "pod_uid": bootstrap.pod_uid,
            "gpu_uuid": bootstrap.gpu_uuid,
            "hold_action": [float(value) for value in action[0].detach().cpu().tolist()],
        }
        report_path = bootstrap.output.with_suffix(".capture.json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"observation": report["observation_npz"], "capture_report": str(report_path.resolve())}, indent=2))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

