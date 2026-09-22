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


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--study-root", type=Path, required=True)
    bootstrap.add_argument("--robolab-root", type=Path, required=True)
    bootstrap.add_argument("--assets-manifest", type=Path, required=True)
    bootstrap.add_argument("--renderer-receipt", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--environment-seed", type=int, default=20260922)
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


def main() -> None:
    args = parse_args()
    args.enable_cameras = True
    if not args.headless or args.num_envs != 1 or args.renderer != "realtime" or args.rendering_type != "balanced":
        raise ValueError("workspace capture requires one headless realtime/balanced RTX environment")
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
            world = get_world(env)
            origin = env.scene.env_origins[0].detach().cpu().numpy()
            frames = env.scene["frames"]
            eef_index = frames.data.target_frame_names.index("eef_frame")
            objects = {}
            for name in ("rubiks_cube", "bowl", "banana", "table"):
                root_position, quaternion = world.get_pose(name, env_id=0)
                corners, geometric_center = world.get_bbox(name, env_id=0)
                corners = np.asarray(
                    [[float(corner[index]) for index in range(3)] for corner in corners],
                    dtype=np.float64,
                )
                objects[name] = {
                    "root_position_env_local_xyz_m": [float(value) for value in root_position.detach().cpu().tolist()],
                    "root_quaternion_world_wxyz": [float(value) for value in quaternion.detach().cpu().tolist()],
                    "geometric_center_env_local_xyz_m": [float(value) for value in geometric_center.tolist()],
                    "bbox_env_local_min_xyz_m": corners.min(axis=0).tolist(),
                    "bbox_env_local_max_xyz_m": corners.max(axis=0).tolist(),
                    "measurement_semantics": {
                        "root_pose": "RoboLab WorldState.get_pose default is_relative=True",
                        "geometric_center": "RoboLab WorldState.get_bbox transformed cached-geometry centroid",
                        "scoring_center": "unvalidated: a later waypoint-validation receipt must explicitly bind the physical-center source",
                    },
                }
            sensors = get_contact_sensors(env.scene)
            contact_inventory = sorted(name for name in sensors if not name.endswith("__all_objs"))
            if "rubiks_cube__table" not in contact_inventory:
                raise RuntimeError("workspace scene lacks rubiks_cube__table contact evidence")
            views = {}
            for camera in ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"):
                frame = np.asarray(obs["image_obs"][camera][0].detach().cpu().numpy(), dtype=np.uint8)
                if frame.ndim != 3 or frame.shape[-1] != 3 or not np.ptp(frame):
                    raise RuntimeError(f"workspace capture has invalid {camera} frame")
                views[camera] = {"shape": list(frame.shape), "pixel_range": int(np.ptp(frame))}
        finally:
            env.close()
        receipt = {
            "schema_version": "sgw-01-lat-measured-workspace-v1",
            "status": "measured_zero_model_workspace_not_candidate_qualified",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "asset_manifest_sha256": _record(args.assets_manifest)["sha256"],
            "task_asset": "rubiks_cube_banana_bowl.usda",
            "renderer_receipt": _record(args.renderer_receipt),
            "robolab_commit": subprocess.check_output(["git", "-C", str(args.robolab_root), "rev-parse", "HEAD"], text=True).strip(),
            "environment_seed": args.environment_seed,
            "environment_origin_world_xyz_m": [float(value) for value in origin],
            "eef_position_env_local_xyz_m": [
                float(value) for value in (frames.data.target_pos_w[0, eef_index].detach().cpu().numpy() - origin)
            ],
            "eef_quaternion_world_wxyz": [
                float(value) for value in frames.data.target_quat_w[0, eef_index].detach().cpu().tolist()
            ],
            "objects": objects,
            "contact_sensor_inventory": contact_inventory,
            "views": views,
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
