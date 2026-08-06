#!/usr/bin/env python3
"""Run the V3-B004 dense physical scan with zero policy/model requests."""

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
POSITIONS = ("rubiks_cube", "bowl", "banana")
TASKS = {
    "left": ("V3BNanoControlLeftCalibrationTask", "task_files/control_left.py"),
    "right": ("V3BNanoControlRightCalibrationTask", "task_files/control_right.py"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--fixture-candidate", type=Path, required=True)
parser.add_argument("--fixture-candidate-sha256", required=True)
parser.add_argument("--neutrality-correction", type=Path, required=True)
parser.add_argument("--neutrality-correction-sha256", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, default=9500)
parser.add_argument("--repeat-resets", type=int, default=3)
parser.add_argument("--settle-steps", type=int, default=60)
parser.add_argument("--stable-window-steps", type=int, default=15)
parser.add_argument("--boundary-headroom-m", type=float, default=0.02)
parser.add_argument("--position-tolerance-m", type=float, default=0.003)
parser.add_argument("--linear-speed-tolerance-m-s", type=float, default=0.02)
parser.add_argument("--angular-speed-tolerance-rad-s", type=float, default=0.20)
parser.add_argument("--interobject-xy-gap-m", type=float, default=0.002)
parser.add_argument("--pod", required=True)
parser.add_argument("--pod-uid", required=True)
parser.add_argument("--gpu-uuid", required=True)
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1 or args_cli.repeat_resets != 3:
    parser.error("V3-B004 requires one environment and exactly three fresh resets")
if args_cli.settle_steps != 60 or args_cli.stable_window_steps != 15:
    parser.error("V3-B004 requires the frozen 60+15 step stability gate")
if not args_cli.headless:
    parser.error("V3-B004 model-blind calibration must run headless")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("V3-B004 calibration requires realtime/balanced RTX")

study_root = args_cli.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))
from experiments.v3.cosmos_nano_lateral_sweep.calibration_design import (  # noqa: E402
    CONTROL_BOWL_Y_M,
    candidate_key,
    dense_candidates,
    select_largest_radius,
    xy_aabb_separation_m,
)

os.environ["VLA_WAM_V3B_FIXTURE_CANDIDATE"] = str(args_cli.fixture_candidate.resolve())
os.environ["VLA_WAM_V3B_FIXTURE_SHA256"] = args_cli.fixture_candidate_sha256
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


def file_record(path: Path) -> dict[str, object]:
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def numeric(value) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


def corners_array(corners) -> np.ndarray:
    return np.asarray([[float(corner[i]) for i in range(3)] for corner in corners], dtype=float)


def bbox(world, name: str) -> tuple[np.ndarray, np.ndarray]:
    corners = corners_array(world.get_bbox(name, env_id=0))
    return corners.min(axis=0), corners.max(axis=0)


def frame(obs: dict, name: str) -> np.ndarray:
    value = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy(), dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.ptp(value):
        raise ValueError(f"blank or malformed RTX view: {name}")
    return value


def combined_frame(obs: dict) -> np.ndarray:
    frames = [frame(obs, name) for name in (
        "over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"
    )]
    height = min(item.shape[0] for item in frames)
    frames = [cv2.resize(item, (round(item.shape[1] * height / item.shape[0]), height)) for item in frames]
    return np.concatenate(frames, axis=1)


def hold_action(obs: dict, device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if action.shape != (1, 8):
        raise ValueError(f"unexpected hold action shape {tuple(action.shape)}")
    return action


def teleport_bowl_y(env, desired_robot_y_m: float) -> None:
    world = get_world(env)
    current_robot_y = float(numeric(world.get_pose("bowl", env_id=0)[0])[1])
    asset = env.scene.rigid_objects["bowl"]
    state = asset.data.root_state_w.clone()
    pose = state[:, :7].clone()
    pose[:, 1] += desired_robot_y_m - current_robot_y
    asset.write_root_pose_to_sim(pose)
    asset.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]))


def main() -> None:
    if args_cli.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args_cli.output_dir}")
    for path, expected in (
        (args_cli.fixture_candidate, args_cli.fixture_candidate_sha256),
        (args_cli.neutrality_correction, args_cli.neutrality_correction_sha256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"source digest mismatch: {path}")
    correction = json.loads(args_cli.neutrality_correction.read_text())
    fixture = json.loads(args_cli.fixture_candidate.read_text())
    if (
        correction.get("model_request_count_before_correction") != 0
        or correction.get("behavioral_episode_count_before_correction") != 0
        or correction.get("correction", {}).get("new_center_m") != CONTROL_BOWL_Y_M
    ):
        raise ValueError("neutrality correction is not the frozen zero-request source")
    expected_positions = fixture["layouts"]["control"]["positions_robot_base_m"]

    robolab_root = args_cli.robolab_root.resolve()
    robolab_commit = subprocess.check_output(
        ["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True
    ).strip()
    study_commit = subprocess.check_output(
        ["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True
    ).strip()
    for root, expected in ((robolab_root, ROBOLAB_COMMIT), (study_root, study_commit)):
        diff = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
            text=True,
        )
        if diff or (root == robolab_root and expected != robolab_commit):
            raise ValueError(f"calibration requires clean pinned checkout: {root}")
    robolab_import = Path(robolab.__file__).resolve()
    if not robolab_import.is_relative_to(robolab_root):
        raise ValueError("effective RoboLab import is outside the pinned worktree")
    gpu_line = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True).splitlines()[0]
    if args_cli.gpu_uuid not in gpu_line:
        raise ValueError("renderer GPU UUID differs from assigned ali pod GPU")

    args_cli.output_dir.mkdir(parents=True)
    set_output_dir(str(args_cli.output_dir / "native"))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    task_root = study_root / "experiments/v3/cosmos_nano_phase_b"
    wrappers = {relation: task_root / relative for relation, (_, relative) in TASKS.items()}
    auto_register_droid_envs(task=[str(wrappers[key]) for key in TASKS], cameras=WRIST_LEFT_RIGHT_HEAD)

    # Obtain live collision extents and table support without consulting a model.
    probe_env, _ = create_env(
        TASKS["left"][0], device=args_cli.device, seed=args_cli.environment_seed,
        num_envs=1, instruction_type="default", policy="v3b004_zero_request_calibration",
        renderer=args_cli.renderer, rendering_mode=args_cli.rendering_type,
    )
    try:
        probe_obs, _ = probe_env.reset()
        probe_world = get_world(probe_env)
        table_min, table_max = bbox(probe_world, "table")
        bowl_min, bowl_max = bbox(probe_world, "bowl")
        bowl_half_y = float((bowl_max[1] - bowl_min[1]) / 2.0)
        lower_y = float(table_min[1] + bowl_half_y + args_cli.boundary_headroom_m)
        upper_y = float(table_max[1] - bowl_half_y - args_cli.boundary_headroom_m)
        scan_y = dense_candidates(lower_y_m=lower_y, upper_y_m=upper_y)
        _ = combined_frame(probe_obs)
    finally:
        probe_env.close()

    rows: list[dict] = []
    videos: dict[str, dict] = {}
    jsonl_path = args_cli.output_dir / "model_blind_scan_rows.jsonl"
    with jsonl_path.open("x", encoding="utf-8") as jsonl:
        for relation, (task_name, _) in TASKS.items():
            env, env_cfg = create_env(
                task_name, device=args_cli.device, seed=args_cli.environment_seed,
                num_envs=1, instruction_type="default", policy="v3b004_zero_request_calibration",
                renderer=args_cli.renderer, rendering_mode=args_cli.rendering_type,
            )
            writer = None
            video_path = args_cli.output_dir / f"{relation}_dense_scan.mp4"
            try:
                for y_m in scan_y:
                    for repeat in range(args_cli.repeat_resets):
                        failures: list[str] = []
                        obs, _ = env.reset()
                        teleport_bowl_y(env, y_m)
                        action = hold_action(obs, env.device)
                        stability = {
                            name: {"max_linear_speed_m_s": 0.0, "max_angular_speed_rad_s": 0.0}
                            for name in POSITIONS
                        }
                        for step in range(args_cli.settle_steps + args_cli.stable_window_steps):
                            obs, _, terminated, truncated, _ = env.step(action)
                            if bool(terminated[0]) or bool(truncated[0]):
                                failures.append("terminated_during_model_blind_settle")
                                break
                            if step >= args_cli.settle_steps:
                                world = get_world(env)
                                for name in POSITIONS:
                                    velocity = numeric(world.get_velocity(name, env_id=0))
                                    stability[name]["max_linear_speed_m_s"] = max(
                                        stability[name]["max_linear_speed_m_s"], max(abs(v) for v in velocity[:3])
                                    )
                                    stability[name]["max_angular_speed_rad_s"] = max(
                                        stability[name]["max_angular_speed_rad_s"], max(abs(v) for v in velocity[3:])
                                    )
                        world = get_world(env)
                        positions = {name: numeric(world.get_pose(name, env_id=0)[0]) for name in POSITIONS}
                        quaternions = {name: numeric(world.get_pose(name, env_id=0)[1]) for name in POSITIONS}
                        if abs(positions["bowl"][1] - y_m) > args_cli.position_tolerance_m:
                            failures.append("bowl_position_tolerance")
                        for name in ("rubiks_cube", "banana"):
                            if max(abs(a - b) for a, b in zip(positions[name], expected_positions[name])) > args_cli.position_tolerance_m:
                                failures.append(f"{name}_fixed_pose_tolerance")
                        if max(
                            abs(positions["bowl"][axis] - expected_positions["bowl"][axis])
                            for axis in (0, 2)
                        ) > args_cli.position_tolerance_m:
                            failures.append("bowl_fixed_xz_tolerance")
                        for name, value in stability.items():
                            if value["max_linear_speed_m_s"] > args_cli.linear_speed_tolerance_m_s:
                                failures.append(f"{name}_linear_stability")
                            if value["max_angular_speed_rad_s"] > args_cli.angular_speed_tolerance_rad_s:
                                failures.append(f"{name}_angular_stability")
                        left = bool(object_left_of(
                            env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot",
                            mirrored=False, require_gripper_detached=True, env_id=0,
                        ))
                        right = bool(object_right_of(
                            env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot",
                            mirrored=False, require_gripper_detached=True, env_id=0,
                        ))
                        if left or right:
                            failures.append("nonneutral_reset")
                        bowl_box = bbox(world, "bowl")
                        table_box = bbox(world, "table")
                        if bowl_box[0][1] < table_box[0][1] + args_cli.boundary_headroom_m or bowl_box[1][1] > table_box[1][1] - args_cli.boundary_headroom_m:
                            failures.append("table_boundary_headroom")
                        xy_gaps = {}
                        for other in ("rubiks_cube", "banana"):
                            other_box = bbox(world, other)
                            gap = xy_aabb_separation_m(
                                bowl_box[0][:2], bowl_box[1][:2], other_box[0][:2], other_box[1][:2]
                            )
                            xy_gaps[other] = gap
                            if gap < args_cli.interobject_xy_gap_m:
                                failures.append(f"bowl_{other}_collision_projection")
                        views = {}
                        for name in ("over_shoulder_left_camera", "over_shoulder_right_camera", "head_camera", "wrist_cam"):
                            try:
                                image = frame(obs, name)
                                views[name] = {"shape": list(image.shape), "pixel_range": int(np.ptp(image))}
                            except ValueError as exc:
                                failures.append(str(exc))
                        rendered = combined_frame(obs)
                        if writer is None:
                            height, width = rendered.shape[:2]
                            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
                            if not writer.isOpened():
                                raise RuntimeError("dense scan video writer failed to open")
                        writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                        row = {
                            "relation": relation, "prompt": env_cfg.instruction,
                            "candidate_y_m": y_m, "candidate_offset_mm": candidate_key(y_m),
                            "repeat": repeat, "passed": not failures,
                            "failures": sorted(set(failures)), "positions_robot_base_m": positions,
                            "quaternions_wxyz": quaternions, "stability_window": stability,
                            "left_predicate_at_reset": left, "right_predicate_at_reset": right,
                            "bowl_interobject_xy_gap_m": xy_gaps, "input_views": views,
                            "hold_action": numeric(action[0]),
                        }
                        rows.append(row)
                        jsonl.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
                        jsonl.flush()
                        os.fsync(jsonl.fileno())
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
            expected_frames = len(scan_y) * args_cli.repeat_resets
            if not ok or decoded is None or frame_count != expected_frames:
                raise RuntimeError(f"{relation} video persistence failed: {frame_count}/{expected_frames}")
            videos[relation] = {**file_record(video_path), "decoded_frame_count": frame_count}

    passing = []
    fingerprint_differences = []
    for y_m in scan_y:
        candidate_rows = [row for row in rows if row["candidate_y_m"] == y_m]
        left_rows = sorted((row for row in candidate_rows if row["relation"] == "left"), key=lambda row: row["repeat"])
        right_rows = sorted((row for row in candidate_rows if row["relation"] == "right"), key=lambda row: row["repeat"])
        fingerprints_equal = True
        for left_row, right_row in zip(left_rows, right_rows):
            max_position_difference = max(
                abs(a - b)
                for name in POSITIONS
                for a, b in zip(left_row["positions_robot_base_m"][name], right_row["positions_robot_base_m"][name])
            )
            fingerprint_differences.append({
                "candidate_y_m": y_m, "repeat": left_row["repeat"],
                "max_position_difference_m": max_position_difference,
            })
            if max_position_difference > args_cli.position_tolerance_m:
                fingerprints_equal = False
            for name in POSITIONS:
                left_q = left_row["quaternions_wxyz"][name]
                right_q = right_row["quaternions_wxyz"][name]
                sign_invariant_difference = min(
                    max(abs(a - b) for a, b in zip(left_q, right_q)),
                    max(abs(a + b) for a, b in zip(left_q, right_q)),
                )
                if sign_invariant_difference > 0.01:
                    fingerprints_equal = False
        if len(candidate_rows) == 6 and all(row["passed"] for row in candidate_rows) and fingerprints_equal:
            passing.append(y_m)
    radius_mm, levels = select_largest_radius(passing)

    report = {
        "schema_version": "vla-wam-shared-v3b-nano-lateral-model-blind-calibration-v1",
        "study_id": "vla_wam_language_steerability_v3", "amendment_id": "V3-B004",
        "status": "complete_model_blind_numeric_calibration_not_behaviorally_released",
        "passed": True, "model_request_count": 0, "behavioral_episode_count": 0,
        "pod": args_cli.pod, "pod_uid": args_cli.pod_uid, "gpu_uuid": args_cli.gpu_uuid,
        "gpu_query": gpu_line, "environment_seed": args_cli.environment_seed,
        "source": {
            "driver": file_record(Path(__file__).resolve()),
            "fixture_candidate": file_record(args_cli.fixture_candidate),
            "neutrality_correction": file_record(args_cli.neutrality_correction),
            "task_wrappers": {key: file_record(value) for key, value in wrappers.items()},
        },
        "runtime": {
            "study_commit": study_commit, "study_tracked_diff_empty": True,
            "robolab_commit": robolab_commit, "robolab_tracked_diff_empty": True,
            "robolab_import": file_record(robolab_import),
            "versions": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "robolab")},
            "renderer": "realtime RTX Vulkan balanced",
        },
        "live_bounds": {
            "table_min_xyz_m": table_min.tolist(), "table_max_xyz_m": table_max.tolist(),
            "bowl_half_y_m": bowl_half_y, "boundary_headroom_m": args_cli.boundary_headroom_m,
            "candidate_lower_y_m": lower_y, "candidate_upper_y_m": upper_y,
        },
        "dense_scan": {
            "candidate_count": len(scan_y), "candidate_y_m": list(scan_y),
            "passing_candidate_y_m": passing, "repeat_count_per_relation": args_cli.repeat_resets,
            "row_count": len(rows), "rows_jsonl": file_record(jsonl_path),
            "left_right_fingerprint_differences": fingerprint_differences,
        },
        "selection": {
            "center_y_m": CONTROL_BOWL_Y_M, "half_range_mm": radius_mm,
            "ordered_seven_levels_y_m": list(levels),
        },
        "viewport_write_gate": videos,
        "release_boundary": "Numeric levels are calibrated with zero model requests. Commit and hash-bind this report and the exact 210-cell queue before starting a Nano server.",
    }
    report_path = args_cli.output_dir / "model_blind_lateral_calibration_report.json"
    report_path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": file_record(report_path), "levels": list(levels), "passed": True}, indent=2, sort_keys=True))
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[Nano V3-B004 model-blind lateral calibration] technical failure: {error}")
        traceback.print_exc()
        simulation_app.close()
        raise
