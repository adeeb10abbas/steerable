#!/usr/bin/env python3
"""Repeated-reset, renderer, and PVC gate for V3-B008/V3-B009 without inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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
    "V3-B008": {
        f"{arm}_{relation}": (
            f"V3B008{''.join(part.title() for part in arm.split('_'))}{relation.title()}GateTask",
            f"task_files/v3b008_{arm}_{relation}.py",
        )
        for arm in ("target_start_right", "target_start_center", "target_start_left")
        for relation in ("left", "right")
    },
    "V3-B009": {
        f"{arm}_{relation}": (
            f"V3B009{''.join(part.title() for part in arm.split('_'))}{relation.title()}GateTask",
            f"task_files/v3b009_{arm}_{relation}.py",
        )
        for arm in ("cube_target_bowl_reference", "bowl_target_cube_reference")
        for relation in ("left", "right")
    },
}
OBJECTS = ("rubiks_cube", "bowl", "banana")


parser = argparse.ArgumentParser()
parser.add_argument("--amendment-id", choices=sorted(TASKS), required=True)
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--candidate-sha256", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, required=True)
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
if args_cli.num_envs != 1 or args_cli.repeat_resets < 2:
    parser.error("gate requires one environment and at least two repeated resets")
if not args_cli.headless or args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("gate requires headless realtime/balanced RTX")

study_root = args_cli.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))
os.environ["VLA_WAM_V3_TIERB_GATE_CANDIDATE"] = str(args_cli.candidate.resolve())
os.environ["VLA_WAM_V3_TIERB_GATE_SHA256"] = args_cli.candidate_sha256

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab  # noqa: E402
import robolab.constants  # noqa: E402
from experiments.v3.prospective_tier_b_gates import droid_fixture_tasks as fixture  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def numeric(values) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [float(item) for item in values]


def frame(obs: dict, name: str) -> np.ndarray:
    value = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy(), dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.ptp(value):
        raise ValueError(f"blank or malformed live RGB view: {name}")
    return value


def montage(obs: dict) -> np.ndarray:
    values = [frame(obs, name) for name in ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")]
    height = min(value.shape[0] for value in values)
    values = [cv2.resize(value, (round(value.shape[1] * height / value.shape[0]), height)) for value in values]
    return np.concatenate(values, axis=1)


def hold_action(obs: dict, device: str) -> torch.Tensor:
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    value = torch.cat((obs["proprio_obs"]["arm_joint_pos"].detach().to(device), gripper), dim=1)
    if value.shape != (1, 8):
        raise ValueError(f"unexpected hold action shape: {tuple(value.shape)}")
    return value


def main() -> None:
    if args_cli.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite gate output: {args_cli.output_dir}")
    if sha256(args_cli.candidate) != args_cli.candidate_sha256:
        raise ValueError("candidate SHA-256 mismatch")
    candidate = json.loads(args_cli.candidate.read_text())
    if candidate.get("status") != "model_blind_candidate_not_released_for_inference":
        raise ValueError("candidate is not unreleased")
    if candidate.get("model_request_count") != 0 or candidate.get("behavioral_episode_count") != 0:
        raise ValueError("candidate contains behavior")

    robolab_root = args_cli.robolab_root.resolve()
    robolab_commit = subprocess.check_output(["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True).strip()
    robolab_diff = subprocess.check_output(["git", "-C", str(robolab_root), "status", "--porcelain=v1", "--untracked-files=no"], text=True)
    study_commit = subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip()
    study_diff = subprocess.check_output(["git", "-C", str(study_root), "status", "--porcelain=v1", "--untracked-files=no"], text=True)
    if robolab_commit != ROBOLAB_COMMIT or robolab_diff or study_diff:
        raise ValueError("gate requires clean pinned RoboLab and clean study checkout")
    if not Path(robolab.__file__).resolve().is_relative_to(robolab_root):
        raise ValueError("effective RoboLab import is outside pinned worktree")
    gpu_line = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"], text=True).splitlines()[0]
    if args_cli.gpu_uuid not in gpu_line:
        raise ValueError("renderer GPU UUID differs from assigned GPU")

    args_cli.output_dir.mkdir(parents=True)
    native = args_cli.output_dir / "native"
    set_output_dir(str(native))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    tasks = TASKS[args_cli.amendment_id]
    task_root = study_root / "experiments/v3/prospective_tier_b_gates"
    wrappers = {label: task_root / relative for label, (_, relative) in tasks.items()}
    auto_register_droid_envs(task=[str(path) for path in wrappers.values()], cameras=WRIST_LEFT_RIGHT_HEAD)

    study = candidate["studies"][args_cli.amendment_id]
    rows = []
    videos = {}
    for label, (task_name, _) in tasks.items():
        arm, relation = label.rsplit("_", 1)
        expected = study["arms"][arm]["positions_robot_base_m"]
        target = study["arms"][arm]["target_object"]
        reference = study["arms"][arm]["reference_object"]
        env, env_cfg = create_env(task_name, device=args_cli.device, seed=args_cli.environment_seed, num_envs=1, instruction_type="default", policy=f"{args_cli.amendment_id.lower()}_model_blind_gate", renderer=args_cli.renderer, rendering_mode=args_cli.rendering_type)
        writer = None
        video_path = args_cli.output_dir / f"{label}_repeated_resets.mp4"
        try:
            repeats = []
            for repeat in range(args_cli.repeat_resets):
                obs, _ = env.reset()
                action = hold_action(obs, env.device)
                for _ in range(args_cli.settle_steps):
                    obs, _, terminated, truncated, _ = env.step(action)
                    if bool(terminated[0]) or bool(truncated[0]):
                        raise ValueError(f"{label} terminated while settling")
                stability = {name: {"max_linear_speed_m_s": 0.0, "max_angular_speed_rad_s": 0.0} for name in OBJECTS}
                for _ in range(args_cli.stable_window_steps):
                    obs, _, terminated, truncated, _ = env.step(action)
                    if bool(terminated[0]) or bool(truncated[0]):
                        raise ValueError(f"{label} terminated during stability window")
                    world = get_world(env)
                    for name in OBJECTS:
                        velocity = numeric(world.get_velocity(name, env_id=0))
                        stability[name]["max_linear_speed_m_s"] = max(stability[name]["max_linear_speed_m_s"], max(abs(v) for v in velocity[:3]))
                        stability[name]["max_angular_speed_rad_s"] = max(stability[name]["max_angular_speed_rad_s"], max(abs(v) for v in velocity[3:]))
                world = get_world(env)
                positions, quaternions = {}, {}
                for name in OBJECTS:
                    position, quaternion = world.get_pose(name, env_id=0)
                    positions[name], quaternions[name] = numeric(position), numeric(quaternion)
                    if max(abs(a - b) for a, b in zip(positions[name], expected[name])) > 0.003:
                        raise ValueError(f"{label} {name} missed position tolerance")
                    if stability[name]["max_linear_speed_m_s"] > args_cli.linear_speed_tolerance_m_s or stability[name]["max_angular_speed_rad_s"] > args_cli.angular_speed_tolerance_rad_s:
                        raise ValueError(f"{label} {name} did not sustain stability")
                left = bool(object_left_of(env, object=target, reference_object=reference, frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
                right = bool(object_right_of(env, object=target, reference_object=reference, frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
                if left or right:
                    raise ValueError(f"{label} reset is not neutral")
                views = {name: {"shape": list(frame(obs, name).shape), "dtype": str(frame(obs, name).dtype), "pixel_range": int(np.ptp(frame(obs, name)))} for name in ("over_shoulder_left_camera", "over_shoulder_right_camera", "head_camera", "wrist_cam")}
                combined = montage(obs)
                if writer is None:
                    height, width = combined.shape[:2]
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"video writer failed for {label}")
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                repeats.append({"repeat": repeat, "positions_robot_base_m": positions, "quaternions_wxyz": quaternions, "stability_window": stability, "left_predicate_at_reset": left, "right_predicate_at_reset": right, "input_views": views})
        finally:
            if writer is not None:
                writer.release()
            env.close()
        capture = cv2.VideoCapture(str(video_path))
        ok, decoded = capture.read()
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not ok or decoded is None or count != args_cli.repeat_resets:
            raise RuntimeError(f"video decode failed for {label}")
        videos[label] = {**record(video_path), "decoded_frame_count": count}
        rows.append({"label": label, "arm": arm, "relation": relation, "target_object": target, "reference_object": reference, "prompt": env_cfg.instruction, "repeat_resets": repeats})

    by_label = {row["label"]: row for row in rows}
    for arm in study["arms"]:
        left_rows = by_label[f"{arm}_left"]["repeat_resets"]
        right_rows = by_label[f"{arm}_right"]["repeat_resets"]
        for left_row, right_row in zip(left_rows, right_rows):
            if left_row["positions_robot_base_m"] != right_row["positions_robot_base_m"] or left_row["quaternions_wxyz"] != right_row["quaternions_wxyz"]:
                raise ValueError(f"{arm} LEFT/RIGHT fingerprints differ")

    output = {
        "schema_version": "vla-wam-shared-v3b-droid-tier-b-model-blind-gate-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": args_cli.amendment_id,
        "model_id": MODEL_ID,
        "status": "passed_model_blind_not_released_for_behavior",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "environment_seed": args_cli.environment_seed,
        "pod": args_cli.pod,
        "pod_uid": args_cli.pod_uid,
        "gpu_uuid": args_cli.gpu_uuid,
        "gpu_query": gpu_line,
        "candidate": record(args_cli.candidate),
        "gate_source": record(Path(__file__).resolve()),
        "fixture_source": record(task_root / "droid_fixture_tasks.py"),
        "task_wrappers": {label: record(path) for label, path in wrappers.items()},
        "robolab": {"commit": robolab_commit, "tracked_diff_empty": True, "effective_import": record(Path(robolab.__file__).resolve()), "versions": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "robolab")}},
        "study_checkout": {"commit": study_commit, "tracked_diff_empty": True},
        "renderer": {"backend": "realtime RTX Vulkan", "quality": "balanced", "nvidia_icd": record(Path("/etc/vulkan/icd.d/nvidia_icd.json")), "all_required_rgb_views_nonblank": True},
        "reset_gate": {"repeat_count_per_task": args_cli.repeat_resets, "settle_steps": args_cli.settle_steps, "stable_window_steps": args_cli.stable_window_steps, "position_tolerance_m": 0.003, "linear_speed_tolerance_m_s": args_cli.linear_speed_tolerance_m_s, "angular_speed_tolerance_rad_s": args_cli.angular_speed_tolerance_rad_s, "left_right_physical_fingerprints_equal_within_each_arm": True, "neither_predicate_true_at_every_reset": True},
        "tasks": rows,
        "viewport_write_gate": videos,
        "release_boundary": "This zero-model-request gate does not release behavioral inference. A hash-bound release amendment and exact queue are still required.",
    }
    output_path = args_cli.output_dir / "model_blind_gate_report.json"
    output_path.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": record(output_path), "passed": True}, indent=2, sort_keys=True))
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[Tier-B DROID model-blind gate] technical failure: {error}")
        traceback.print_exc()
        simulation_app.close()
        raise
