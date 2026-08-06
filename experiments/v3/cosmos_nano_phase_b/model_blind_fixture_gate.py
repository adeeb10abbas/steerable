#!/usr/bin/env python3
"""Calibrate Nano control/mirror resets and RTX writers without model requests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher


ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
MODEL_ID = "cosmos3_nano_policy_droid"
TASKS = {
    "control_left": (
        "V3BNanoControlLeftCalibrationTask",
        "task_files/control_left.py",
    ),
    "control_right": (
        "V3BNanoControlRightCalibrationTask",
        "task_files/control_right.py",
    ),
    "position_mirrored_left": (
        "V3BNanoPositionMirroredLeftCalibrationTask",
        "task_files/position_mirrored_left.py",
    ),
    "position_mirrored_right": (
        "V3BNanoPositionMirroredRightCalibrationTask",
        "task_files/position_mirrored_right.py",
    ),
}
POSITIONS = ("rubiks_cube", "bowl", "banana")


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--candidate-sha256", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, default=9400)
parser.add_argument("--repeat-resets", type=int, default=3)
parser.add_argument("--settle-steps", type=int, default=60)
parser.add_argument("--stable-window-steps", type=int, default=15)
parser.add_argument("--linear-speed-tolerance-m-s", type=float, default=0.02)
parser.add_argument("--angular-speed-tolerance-rad-s", type=float, default=0.20)
parser.add_argument("--pod", required=True)
parser.add_argument("--pod-uid", required=True)
parser.add_argument("--gpu-uuid", required=True)
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1:
    parser.error("model-blind fixture calibration requires one environment")
if args_cli.repeat_resets < 2:
    parser.error("at least two repeated resets are required")
if args_cli.settle_steps < 1 or args_cli.stable_window_steps < 2:
    parser.error("settling and sustained-stability windows must be positive")
if (
    args_cli.linear_speed_tolerance_m_s <= 0
    or args_cli.angular_speed_tolerance_rad_s <= 0
):
    parser.error("speed tolerances must be positive")
if not args_cli.headless:
    parser.error("model-blind fixture calibration must run headless")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("model-blind fixture calibration requires realtime/balanced RTX")

study_root = args_cli.study_root.resolve()
if not (study_root / "experiments/v3/cosmos_nano_phase_b").is_dir():
    parser.error("study root does not contain the Phase-B calibration package")
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))

os.environ["VLA_WAM_V3B_FIXTURE_CANDIDATE"] = str(args_cli.candidate.resolve())
os.environ["VLA_WAM_V3B_FIXTURE_SHA256"] = args_cli.candidate_sha256

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab  # noqa: E402
import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def numeric(values) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [float(item) for item in values]


def close_vector(left: list[float], right: list[float], tolerance: float) -> bool:
    return len(left) == len(right) and max(abs(a - b) for a, b in zip(left, right)) <= tolerance


def frame_from_obs(obs: dict, name: str) -> np.ndarray:
    frame = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy(), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3 or not np.ptp(frame):
        raise ValueError(f"blank or malformed live RGB view: {name}")
    return frame


def combine_views(obs: dict) -> np.ndarray:
    names = ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")
    frames = [frame_from_obs(obs, name) for name in names]
    height = min(frame.shape[0] for frame in frames)
    resized = [
        cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height))
        for frame in frames
    ]
    return np.concatenate(resized, axis=1)


def hold_action(obs: dict, device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if action.shape != (1, 8):
        raise ValueError(f"unexpected joint-position hold action shape: {tuple(action.shape)}")
    return action


def main() -> None:
    if args_cli.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {args_cli.output_dir}")
    if sha256(args_cli.candidate) != args_cli.candidate_sha256:
        raise ValueError("candidate SHA-256 mismatch before simulator launch")
    candidate = json.loads(args_cli.candidate.read_text())
    if (
        candidate.get("status") != "model_blind_candidate_not_released_for_inference"
        or candidate.get("model_request_count") != 0
        or candidate.get("behavioral_episode_count") != 0
    ):
        raise ValueError("candidate is not the unreleased model-blind fixture input")
    robolab_root = args_cli.robolab_root.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(robolab_root), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
    )
    if commit != ROBOLAB_COMMIT or tracked_diff:
        raise ValueError("calibration requires the clean tracked frozen RoboLab revision")
    study_commit = subprocess.check_output(
        ["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True
    ).strip()
    study_tracked_diff = subprocess.check_output(
        ["git", "-C", str(study_root), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
    )
    if study_tracked_diff:
        raise ValueError("calibration requires a clean tracked study checkout")
    robolab_import = Path(robolab.__file__).resolve()
    if not robolab_import.is_relative_to(robolab_root):
        raise ValueError("effective RoboLab import is outside the pinned worktree")
    gpu_line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()[0]
    if args_cli.gpu_uuid not in gpu_line:
        raise ValueError("live renderer GPU UUID differs from the assigned pod GPU")

    args_cli.output_dir.mkdir(parents=True)
    set_output_dir(str(args_cli.output_dir / "native"))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    task_path = (
        args_cli.study_root.resolve()
        / "experiments/v3/cosmos_nano_phase_b/fixture_tasks.py"
    )
    task_wrappers = {
        label: task_path.parent / relative_path
        for label, (_, relative_path) in TASKS.items()
    }
    auto_register_droid_envs(
        task=[str(task_wrappers[label]) for label in TASKS],
        cameras=WRIST_LEFT_RIGHT_HEAD,
    )

    expected = {
        "control": candidate["layouts"]["control"]["positions_robot_base_m"],
        "position_mirrored": candidate["layouts"]["position_mirrored"]["positions_robot_base_m"],
    }
    if (
        candidate["layouts"]["control"]["quaternions_wxyz_unchanged"]
        != candidate["layouts"]["position_mirrored"]["quaternions_wxyz_unchanged"]
    ):
        raise ValueError("candidate changed initial object quaternions across layouts")
    rows: list[dict] = []
    videos: dict[str, dict[str, object]] = {}
    for label, (task_name, _) in TASKS.items():
        arm, relation = label.rsplit("_", 1)
        env, env_cfg = create_env(
            task_name,
            device=args_cli.device,
            seed=args_cli.environment_seed,
            num_envs=1,
            instruction_type="default",
            policy="v3b_nano_model_blind_fixture_calibration",
            renderer=args_cli.renderer,
            rendering_mode=args_cli.rendering_type,
        )
        writer = None
        video_path = args_cli.output_dir / f"{label}_repeated_resets.mp4"
        try:
            reset_rows = []
            for repeat in range(args_cli.repeat_resets):
                obs, _ = env.reset()
                action = hold_action(obs, env.device)
                for _ in range(args_cli.settle_steps):
                    obs, _, terminated, truncated, _ = env.step(action)
                    if bool(terminated[0]) or bool(truncated[0]):
                        raise ValueError(f"{label} terminated during model-blind settling")
                stability = {
                    name: {"max_linear_speed_m_s": 0.0, "max_angular_speed_rad_s": 0.0}
                    for name in POSITIONS
                }
                for _ in range(args_cli.stable_window_steps):
                    obs, _, terminated, truncated, _ = env.step(action)
                    if bool(terminated[0]) or bool(truncated[0]):
                        raise ValueError(f"{label} terminated during stability window")
                    stable_world = get_world(env)
                    for name in POSITIONS:
                        velocity = numeric(stable_world.get_velocity(name, env_id=0))
                        stability[name]["max_linear_speed_m_s"] = max(
                            stability[name]["max_linear_speed_m_s"],
                            max(abs(item) for item in velocity[:3]),
                        )
                        stability[name]["max_angular_speed_rad_s"] = max(
                            stability[name]["max_angular_speed_rad_s"],
                            max(abs(item) for item in velocity[3:]),
                        )
                world = get_world(env)
                positions = {}
                quaternions = {}
                velocities = {}
                for name in POSITIONS:
                    position, quaternion = world.get_pose(name, env_id=0)
                    positions[name] = numeric(position)
                    quaternions[name] = numeric(quaternion)
                    velocities[name] = numeric(world.get_velocity(name, env_id=0))
                    if not close_vector(positions[name], expected[arm][name], 0.003):
                        raise ValueError(f"{label} live {name} position missed candidate tolerance")
                    if (
                        stability[name]["max_linear_speed_m_s"]
                        > args_cli.linear_speed_tolerance_m_s
                        or stability[name]["max_angular_speed_rad_s"]
                        > args_cli.angular_speed_tolerance_rad_s
                    ):
                        raise ValueError(
                            f"{label} live {name} did not sustain stability: {stability[name]}"
                        )
                left = bool(object_left_of(
                    env,
                    object="rubiks_cube",
                    reference_object="bowl",
                    frame_of_reference="robot",
                    mirrored=False,
                    require_gripper_detached=True,
                    env_id=0,
                ))
                right = bool(object_right_of(
                    env,
                    object="rubiks_cube",
                    reference_object="bowl",
                    frame_of_reference="robot",
                    mirrored=False,
                    require_gripper_detached=True,
                    env_id=0,
                ))
                if left or right:
                    raise ValueError(f"{label} reset is not neutral under both frozen predicates")
                views = {
                    name: {
                        "shape": list(frame_from_obs(obs, name).shape),
                        "dtype": str(frame_from_obs(obs, name).dtype),
                        "pixel_range": int(np.ptp(frame_from_obs(obs, name))),
                    }
                    for name in (
                        "over_shoulder_left_camera",
                        "over_shoulder_right_camera",
                        "head_camera",
                        "wrist_cam",
                    )
                }
                combined = combine_views(obs)
                if writer is None:
                    height, width = combined.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"video writer did not open for {label}")
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                reset_rows.append({
                    "repeat": repeat,
                    "positions_robot_base_m": positions,
                    "quaternions_wxyz": quaternions,
                    "velocities": velocities,
                    "stability_window": stability,
                    "left_predicate_at_reset": left,
                    "right_predicate_at_reset": right,
                    "input_views": views,
                })
            rows.append({
                "label": label,
                "arm": arm,
                "relation": relation,
                "task_name": task_name,
                "prompt": env_cfg.instruction,
                "repeat_resets": reset_rows,
            })
        finally:
            if writer is not None:
                writer.release()
            env.close()
        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, decoded = capture.read()
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if not ok or decoded is None or frame_count != args_cli.repeat_resets:
            raise RuntimeError(f"calibration video decode failed for {label}")
        videos[label] = {**record(video_path), "decoded_frame_count": frame_count}

    by_label = {row["label"]: row for row in rows}
    for arm in ("control", "position_mirrored"):
        left_rows = by_label[f"{arm}_left"]["repeat_resets"]
        right_rows = by_label[f"{arm}_right"]["repeat_resets"]
        for left_row, right_row in zip(left_rows, right_rows):
            for field in ("positions_robot_base_m", "quaternions_wxyz"):
                if left_row[field] != right_row[field]:
                    raise ValueError(f"{arm} LEFT/RIGHT reset fingerprints differ for {field}")
    controls = by_label["control_left"]["repeat_resets"]
    mirrored_rows = by_label["position_mirrored_left"]["repeat_resets"]
    post_settle_quaternion_differences = []
    for repeat, (control, mirrored) in enumerate(zip(controls, mirrored_rows)):
        quaternion_row = {"repeat": repeat, "objects": {}}
        for name in POSITIONS:
            c = control["positions_robot_base_m"][name]
            m = mirrored["positions_robot_base_m"][name]
            if not close_vector([c[0], -c[1], c[2]], m, 0.003):
                raise ValueError(
                    f"live position reflection failed for {name} at repeat {repeat}"
                )
            control_quaternion = control["quaternions_wxyz"][name]
            mirrored_quaternion = mirrored["quaternions_wxyz"][name]
            dot = abs(sum(a * b for a, b in zip(control_quaternion, mirrored_quaternion)))
            quaternion_row["objects"][name] = {
                "max_abs_component_difference": max(
                    abs(a - b) for a, b in zip(control_quaternion, mirrored_quaternion)
                ),
                "absolute_quaternion_dot": min(1.0, dot),
                "angular_distance_rad": 2.0 * math.acos(min(1.0, dot)),
            }
        post_settle_quaternion_differences.append(quaternion_row)

    output = {
        "schema_version": "vla-wam-shared-v3b-nano-position-mirror-model-blind-calibration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "model_id": MODEL_ID,
        "phase": "B_confound_ablation",
        "status": "complete_model_blind_calibration_not_yet_released",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "environment_seed": args_cli.environment_seed,
        "pod": args_cli.pod,
        "pod_uid": args_cli.pod_uid,
        "gpu_uuid": args_cli.gpu_uuid,
        "gpu_query": gpu_line,
        "candidate": record(args_cli.candidate),
        "calibration_driver_source": record(Path(__file__).resolve()),
        "factor_task_source": record(task_path),
        "factor_task_wrappers": {
            label: record(task_wrappers[label]) for label in TASKS
        },
        "robolab": {
            "commit": commit,
            "tracked_diff_empty": True,
            "effective_import": record(robolab_import),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("isaacsim", "isaaclab", "robolab")
            },
        },
        "study_checkout": {
            "commit": study_commit,
            "tracked_diff_empty": True,
        },
        "renderer": {
            "backend": "realtime RTX Vulkan",
            "quality": "balanced",
            "nvidia_icd": record(Path("/etc/vulkan/icd.d/nvidia_icd.json")),
            "all_required_rgb_views_nonblank": True,
        },
        "reset_gate": {
            "repeat_count_per_task": args_cli.repeat_resets,
            "settle_steps": args_cli.settle_steps,
            "settle_steps_basis": (
                "The model-blind 60-step probe reduced movable-object translation below "
                "0.004 m/s while preserving a neutral reset; a longer 180-step probe was "
                "rejected after free settling crossed a task termination boundary"
            ),
            "stable_window_steps": args_cli.stable_window_steps,
            "position_tolerance_m": 0.003,
            "linear_speed_tolerance_m_s": args_cli.linear_speed_tolerance_m_s,
            "angular_speed_tolerance_rad_s": args_cli.angular_speed_tolerance_rad_s,
            "angular_speed_tolerance_basis": (
                "0.02 m/s linear tolerance divided by a conservative 0.10 m object "
                "radius gives 0.20 rad/s, bounding rotational surface speed at the "
                "same scale as translation"
            ),
            "left_right_physical_fingerprints_equal_within_each_arm": True,
            "neither_predicate_true_at_every_reset": True,
            "live_position_reflection_passed_at_every_repeat": True,
            "initial_quaternion_sources_identical_across_layouts": True,
            "post_settle_quaternion_differences_recorded_not_gated": True,
        },
        "post_settle_cross_layout_quaternion_differences": (
            post_settle_quaternion_differences
        ),
        "tasks": rows,
        "viewport_write_gate": videos,
        "claim_boundary": (
            "Model-blind calibration of a positions-only movable-object reflection. "
            "Initial quaternion sources are identical, while any recorded post-settle "
            "orientation difference is a downstream physical consequence of the position "
            "intervention. It is not behavioral evidence, a full scene mirror, or a "
            "reachability claim."
        ),
    }
    output_path = args_cli.output_dir / "model_blind_calibration_report.json"
    output_path.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": record(output_path), "passed": True}, indent=2, sort_keys=True))
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[Nano Phase-B model-blind fixture gate] technical failure: {error}")
        traceback.print_exc()
        simulation_app.close()
        raise
