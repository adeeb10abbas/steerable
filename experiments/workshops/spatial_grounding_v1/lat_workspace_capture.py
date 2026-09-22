"""Capture actual LAT source geometry and contact inventory with zero policy calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .build_asset_manifest import is_git_worktree
from .lat_candidate_generator import workspace_digest


def _vector(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def _rotate_wxyz(quaternion: list[float], vector: list[float]) -> list[float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    return [
        (1 - 2 * (y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
        2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
        2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz,
    ]


def _root_local_offset(root: list[float], quaternion: list[float], center: list[float]) -> list[float]:
    w, x, y, z = quaternion
    return _rotate_wxyz([w, -x, -y, -z], [center[i] - root[i] for i in range(3)])


def _native_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"available": False, "reason": "observation lacks a native proprio_obs mapping"}
    fields: dict[str, list[float]] = {}
    for name, row in value.items():
        try:
            fields[str(name)] = _vector(row[0])
        except (IndexError, TypeError, ValueError):
            return {"available": False, "reason": f"proprio field {name!r} is not a single-environment numeric vector"}
    return {"available": True, "fields": fields}


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--study-root", type=Path, required=True)
    bootstrap.add_argument("--robolab-root", type=Path, required=True)
    bootstrap.add_argument("--assets-manifest", type=Path, required=True)
    bootstrap.add_argument("--renderer-receipt", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--environment-seed", type=int, default=20260922)
    bootstrap.add_argument("--render-warmup-frames", type=int, default=0)
    known, _ = bootstrap.parse_known_args()
    if known.output.exists():
        raise FileExistsError(f"refusing to overwrite workspace receipt: {known.output}")
    if not is_git_worktree(known.robolab_root) or not known.assets_manifest.is_file() or not known.renderer_receipt.is_file():
        raise ValueError("workspace capture requires pinned RoboLab plus passed asset/renderer receipts")
    if str(known.study_root.resolve()) not in sys.path:
        sys.path.insert(0, str(known.study_root.resolve()))
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import add_common_eval_args

    parser = argparse.ArgumentParser(parents=[bootstrap])
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def render_only_warmup(env: Any, observation: dict[str, Any], count: int, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from .recorder import encode_viewport_video

    if type(count) is not int or not 1 <= count <= 120:
        raise ValueError("render-only diagnostic requires 1..120 frames")
    output.mkdir(parents=True, exist_ok=False)
    cameras = ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")
    initial_time = float(env.sim.current_time)
    snapshots = []
    viewport_frames = []
    for index in range(count + 1):
        if index:
            env.sim.render()
            for camera in cameras:
                env.scene[camera].update(0.0, force_recompute=True)
            observation = env.observation_manager.compute()
        if float(env.sim.current_time) != initial_time:
            raise RuntimeError("render-only diagnostic advanced physical simulation time")
        views = {}
        for camera in cameras:
            image = np.asarray(observation["image_obs"][camera][0].detach().cpu().numpy())
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or not np.ptp(image):
                raise RuntimeError(f"render-only diagnostic has invalid {camera} RGB")
            if camera == cameras[0] or index in {0, 1, 10, 30, 60, count}:
                path = output / f"{camera}-{index:04d}.npy"
                np.save(path, image, allow_pickle=False)
                views[camera] = _record(path)
                if camera == cameras[0]:
                    viewport_frames.append(path)
        snapshots.append({"render_frame": index, "sim_time_s": initial_time, "views": views})
    video = encode_viewport_video(viewport_frames, output / "render_only.mp4", fps=30)
    return observation, {
        "status": "render_only_diagnostic_not_visual_qualification",
        "physics_actions": 0, "simulation_time_unchanged": True,
        "render_frames": count, "snapshots": snapshots,
        "viewport_video": video, "video_timing": "30_fps_display_only_no_physical_time_advance",
    }


def material_asset_paths(stage: Any) -> list[dict[str, Any]]:
    from pxr import Sdf

    result = []
    for prim in stage.Traverse():
        if not any(name in str(prim.GetPath()).lower() for name in ("rubiks_cube", "bowl", "banana")):
            continue
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            if value is not None:
                result.append({
                    "attribute": str(attribute.GetPath()), "authored_asset": value.path,
                    "resolved_path": value.resolvedPath,
                    "resolved_file_exists": Path(value.resolvedPath).is_file() if value.resolvedPath else False,
                })
    return result


def main() -> None:
    args = parse_args()
    args.enable_cameras = True
    if not args.headless or args.num_envs != 1 or args.renderer != "realtime" or args.rendering_type != "balanced":
        raise ValueError("workspace capture requires one headless realtime/balanced RTX environment")
    if not 0 <= args.render_warmup_frames <= 120:
        raise ValueError("render-only diagnostic is bounded to at most 120 frames")
    renderer = json.loads(args.renderer_receipt.read_text(encoding="utf-8"))
    if renderer.get("status") != "passed_zero_model_renderer_preflight" or renderer.get("model_request_count") != 0:
        raise ValueError("workspace capture requires the passed zero-model renderer receipt")
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        import numpy as np
        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        if not Path(robolab.__file__).resolve().is_relative_to(args.robolab_root.resolve()):
            raise RuntimeError("effective RoboLab import is outside the pinned checkout")
        task_path = args.study_root / "experiments/workshops/spatial_grounding_v1/renderer_probe_task.py"
        native = args.output.parent / "native"
        native.mkdir(parents=True, exist_ok=False)
        set_output_dir(str(native))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        auto_register_droid_abs_ik_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        env, _ = create_env(
            "SGWRendererProbeTask", device=args.device, seed=args.environment_seed, num_envs=1,
            instruction_type="default", policy="sgw_01_zero_model_lat_workspace_capture",
            renderer=args.renderer, rendering_mode=args.rendering_type,
        )
        try:
            obs, _ = env.reset()
            warmup = None
            if args.render_warmup_frames:
                import omni.usd
                obs, warmup = render_only_warmup(
                    env, obs, args.render_warmup_frames, args.output.parent / "render_diagnostic",
                )
                warmup["material_assets"] = material_asset_paths(omni.usd.get_context().get_stage())
            world = get_world(env)
            origin = env.scene.env_origins[0].detach().cpu().numpy()
            frames = env.scene["frames"]
            eef_index = frames.data.target_frame_names.index("eef_frame")
            eef_position_env_local = _vector(frames.data.target_pos_w[0, eef_index].detach().cpu().numpy() - origin)
            eef_quaternion_world = _vector(frames.data.target_quat_w[0, eef_index])
            robot = env.scene["robot"].data
            robot_snapshot = {
                "base_position_env_local_xyz_m": _vector(robot.root_pos_w[0].detach().cpu().numpy() - origin),
                "base_quaternion_world_wxyz": _vector(robot.root_quat_w[0]),
                "joint_names": [str(name) for name in env.scene["robot"].joint_names],
                "joint_position_rad": _vector(robot.joint_pos[0]),
                "joint_velocity_rad_s": _vector(robot.joint_vel[0]),
            }
            objects = {}
            for name in ("rubiks_cube", "bowl", "banana", "table"):
                root_position, quaternion = world.get_pose(name, env_id=0)
                corners, geometric_center = world.get_bbox(name, env_id=0)
                corners = np.asarray(
                    [[float(corner[index]) for index in range(3)] for corner in corners],
                    dtype=np.float64,
                )
                root = _vector(root_position)
                rotation = _vector(quaternion)
                center = _vector(geometric_center)
                offset = _root_local_offset(root, rotation, center)
                object_row = {
                    "root_position_env_local_xyz_m": root,
                    "root_quaternion_world_wxyz": rotation,
                    "geometric_center_env_local_xyz_m": center,
                    "geometric_center_offset_root_local_xyz_m": offset,
                    "geometric_center_reconstructed_env_local_xyz_m": [
                        root[i] + _rotate_wxyz(rotation, offset)[i] for i in range(3)
                    ],
                    "bbox_env_local_min_xyz_m": corners.min(axis=0).tolist(),
                    "bbox_env_local_max_xyz_m": corners.max(axis=0).tolist(),
                    "measurement_semantics": {
                        "root_pose": "RoboLab WorldState.get_pose default is_relative=True",
                        "geometric_center": "RoboLab WorldState.get_bbox transformed cached-geometry centroid",
                        "scoring_center": "unvalidated: a later waypoint-validation receipt must explicitly bind the physical-center source",
                    },
                }
                asset = env.scene[name]
                data = getattr(asset, "data", None)
                if hasattr(data, "root_com_pos_w") and hasattr(data, "root_com_vel_w"):
                    object_row["com_position_env_local_xyz_m"] = _vector(
                        data.root_com_pos_w[0].detach().cpu().numpy() - origin
                    )
                    object_row["com_velocity_world_xyz_rad_s"] = _vector(data.root_com_vel_w[0])
                else:
                    object_row["com_measurement"] = {
                        "available": False,
                        "reason": "scene object does not expose IsaacLab RigidObjectData root_com_pos_w/root_com_vel_w",
                    }
                if name in {"rubiks_cube", "bowl"} and "com_position_env_local_xyz_m" not in object_row:
                    raise RuntimeError(f"{name} lacks mandatory rigid-body COM measurements")
                objects[name] = object_row
            sensors = get_contact_sensors(env.scene)
            contact_inventory = sorted(name for name in sensors if not name.endswith("__all_objs"))
            if "rubiks_cube__table" not in contact_inventory:
                raise RuntimeError("workspace scene lacks rubiks_cube__table contact evidence")
            views = {}
            view_root = args.output.parent / "views"
            view_root.mkdir(parents=True, exist_ok=False)
            for camera in ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"):
                frame = np.asarray(obs["image_obs"][camera][0].detach().cpu().numpy(), dtype=np.uint8)
                if frame.ndim != 3 or frame.shape[-1] != 3 or not np.ptp(frame):
                    raise RuntimeError(f"workspace capture has invalid {camera} frame")
                path = view_root / f"{camera}.npy"
                np.save(path, frame, allow_pickle=False)
                views[camera] = {
                    "shape": list(frame.shape),
                    "pixel_range": int(np.ptp(frame)),
                    "lossless_array": _record(path),
                }
        finally:
            env.close()
        receipt = {
            "schema_version": "sgw-01-lat-measured-workspace-v1",
            "measurement_schema_version": "sgw-01-lat-measured-workspace-v2",
            "status": "measured_zero_model_workspace_not_candidate_qualified",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "asset_manifest_sha256": _record(args.assets_manifest)["sha256"],
            "task_asset": "rubiks_cube_banana_bowl.usda",
            "renderer_receipt": _record(args.renderer_receipt),
            "robolab_commit": subprocess.check_output(["git", "-C", str(args.robolab_root), "rev-parse", "HEAD"], text=True).strip(),
            "environment_seed": args.environment_seed,
            "environment_origin_world_xyz_m": _vector(origin),
            "eef_position_env_local_xyz_m": eef_position_env_local,
            "eef_quaternion_world_wxyz": eef_quaternion_world,
            "robot": robot_snapshot,
            "native_observation_proprio": _native_observation(obs.get("proprio_obs")),
            "objects": objects,
            "contact_sensor_inventory": contact_inventory,
            "views": views,
            "render_only_diagnostic": warmup,
            "validated_slots": [],
            "versions": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "robolab")},
        }
        receipt["receipt_sha256"] = workspace_digest(receipt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n")
    finally:
        app.close()


if __name__ == "__main__":
    main()
