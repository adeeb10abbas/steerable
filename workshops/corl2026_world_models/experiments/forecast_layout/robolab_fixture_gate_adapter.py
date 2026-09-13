"""Concrete RoboLab/Isaac adapter for the model-blind fixture gate.

The generic gate injects the exact source/candidate paths and digests into this
factory.  The adapter launches Isaac once, builds the four calibration-only
timeout tasks, and performs reset/hold measurements without importing a policy
client.  Visibility uses live instance counts when a caller extends the camera
stream; the pinned default uses calibrated center projection plus conservative
object-OBB occlusion checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _host(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(isinstance(value, (list, tuple)), "live simulator value is not vector-like")
    output = [float(item) for item in value]
    _require(all(math.isfinite(item) for item in output), "live simulator vector is non-finite")
    return output


def _array_identity(value: Any) -> dict[str, Any]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    _require(not array.dtype.hasobject, "live array has object dtype")
    header = {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"}
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {**header, "value_sha256": digest.hexdigest()}


def _quat_inverse_rotate_wxyz(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    w, x, y, z = (float(item) for item in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    _require(norm > 0.0, "camera quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = (float(item) for item in vector)
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + w * z) * vy + 2 * (x * z - w * y) * vz,
        2 * (x * y - w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + w * x) * vz,
        2 * (x * z + w * y) * vx + 2 * (y * z - w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


def _project(
    camera_center: Sequence[float],
    camera_quaternion_wxyz_ros: Sequence[float],
    target: Sequence[float],
    intrinsic: Sequence[Sequence[float]],
) -> tuple[float, float]:
    local = _quat_inverse_rotate_wxyz(
        camera_quaternion_wxyz_ros,
        [float(target[index]) - float(camera_center[index]) for index in range(3)],
    )
    _require(local[2] > 1e-9, "object center is behind the camera plane")
    return (
        float(intrinsic[0][0]) * local[0] / local[2] + float(intrinsic[0][1]) * local[1] / local[2] + float(intrinsic[0][2]),
        float(intrinsic[1][0]) * local[0] / local[2] + float(intrinsic[1][1]) * local[1] / local[2] + float(intrinsic[1][2]),
    )


def _segment_intersects_aabb_before_target(
    origin: Sequence[float], target: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> bool:
    direction = [float(target[index]) - float(origin[index]) for index in range(3)]
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            if float(origin[axis]) < float(lower[axis]) or float(origin[axis]) > float(upper[axis]):
                return False
            continue
        first = (float(lower[axis]) - float(origin[axis])) / direction[axis]
        second = (float(upper[axis]) - float(origin[axis])) / direction[axis]
        t_min = max(t_min, min(first, second))
        t_max = min(t_max, max(first, second))
        if t_min > t_max:
            return False
    return t_max > 1e-6 and t_min < 1.0 - 1e-6


class RoboLabFixtureGateAdapter:
    """One-candidate live adapter; construct a new instance for each candidate."""

    TASKS = {
        ("original", "left"): "WMFGateOriginalLeftTask",
        ("original", "right"): "WMFGateOriginalRightTask",
        ("reflected", "left"): "WMFGateReflectedLeftTask",
        ("reflected", "right"): "WMFGateReflectedRightTask",
    }

    def __init__(self, config: Mapping[str, Any]) -> None:
        required = (
            "study_root",
            "robolab_root",
            "expected_study_commit",
            "expected_robolab_commit",
            "pod",
            "pod_uid",
            "gpu_uuid",
            "source_contract_path",
            "source_contract_sha256",
            "candidate_pool_path",
            "candidate_pool_sha256",
            "candidate_id",
        )
        missing = [key for key in required if not config.get(key)]
        _require(not missing, f"RoboLab fixture adapter config is missing: {missing}")
        self.config = dict(config)
        self.study_root = Path(config["study_root"]).resolve()
        self.robolab_root = Path(config["robolab_root"]).resolve()
        self.candidate_pool_path = Path(config["candidate_pool_path"]).resolve()
        self.source_contract_path = Path(config["source_contract_path"]).resolve()
        self.candidate_id = str(config["candidate_id"])
        self.device = str(config.get("device", "cuda:0"))
        self.renderer = str(config.get("renderer", "realtime"))
        self.rendering_type = str(config.get("rendering_type", "balanced"))
        _require(self.renderer == "realtime" and self.rendering_type == "balanced", "fixture gate requires realtime/balanced RTX")
        self._verify_sources()

        forecast_root = self.study_root / "workshops/corl2026_world_models/experiments/forecast_layout"
        _require(forecast_root.is_dir(), "study checkout lacks forecast-layout runtime")
        for root in (forecast_root, self.study_root, self.robolab_root):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        os.environ.update({
            "WMF_FORECAST_CANDIDATE_POOL": str(self.candidate_pool_path),
            "WMF_FORECAST_CANDIDATE_POOL_SHA256": str(config["candidate_pool_sha256"]),
            "WMF_FORECAST_CANDIDATE_ID": self.candidate_id,
        })

        from isaaclab.app import AppLauncher

        launch_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(launch_parser)
        launch_args, _ = launch_parser.parse_known_args(["--headless"])
        launch_args.enable_cameras = True
        self._launcher = AppLauncher(launch_args)
        self._simulation_app = self._launcher.app

        import cv2
        import numpy as np
        import torch
        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        from fixture_tasks import observation_identities, success_measurements

        _require(Path(robolab.__file__).resolve().is_relative_to(self.robolab_root), "effective RoboLab import is outside pinned checkout")
        self.cv2, self.np, self.torch = cv2, np, torch
        self.set_output_dir = set_output_dir
        self.create_env = create_env
        self.get_world = get_world
        self.observation_identities = observation_identities
        self.success_measurements = success_measurements
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = False
        task_root = forecast_root / "gate_task_files"
        auto_register_droid_envs(
            task=[str(task_root / name) for name in (
                "original_left.py", "original_right.py", "reflected_left.py", "reflected_right.py"
            )],
            cameras=WRIST_LEFT_RIGHT_HEAD,
        )
        self._closed = False

    def _verify_git(self, root: Path, expected: str, label: str) -> None:
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"], text=True
        )
        _require(actual == expected and not dirty, f"{label} checkout is not the clean expected commit")

    def _verify_sources(self) -> None:
        self._verify_git(self.study_root, str(self.config["expected_study_commit"]), "study")
        self._verify_git(self.robolab_root, str(self.config["expected_robolab_commit"]), "RoboLab")
        _require(_sha256_file(self.source_contract_path) == self.config["source_contract_sha256"], "source-contract digest mismatch")
        _require(_sha256_file(self.candidate_pool_path) == self.config["candidate_pool_sha256"], "candidate-pool digest mismatch")
        source = json.loads(self.source_contract_path.read_text())
        files = [source["scene"]["scene_asset"], *(source["objects"][name]["asset"] for name in ("banana", "bowl", "rubiks_cube"))]
        for row in files:
            path = self.robolab_root / row["path"]
            _require(path.is_file(), f"required RoboLab LFS asset is absent: {path}")
            _require(_sha256_file(path) == row["lfs_oid_sha256"], f"RoboLab asset content digest mismatch: {path}")
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
        self.gpu_line = next((row for row in gpu_rows if str(self.config["gpu_uuid"]) in row), None)
        _require(self.gpu_line is not None, "assigned live GPU UUID is not visible")

    def _hold_action(self, observation: Mapping[str, Any]):
        arm = observation["proprio_obs"]["arm_joint_pos"].detach().to(self.device)
        gripper = observation["proprio_obs"]["gripper_pos"].detach().to(self.device)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        action = self.torch.cat((arm, gripper), dim=1)
        _require(tuple(action.shape) == (1, 8), f"hold action shape changed: {tuple(action.shape)}")
        return action

    def _pose_rows(self, world: Any) -> dict[str, Any]:
        output = {}
        for name in ("banana", "bowl", "rubiks_cube"):
            position, quaternion = world.get_pose(name, env_id=0)
            output[name] = {
                "position_robot_base_m": _host(position),
                "quaternion_wxyz": _host(quaternion),
            }
        return output

    def _camera_rows(self, env: Any, observation: Mapping[str, Any], world: Any, attempt_dir: Path) -> dict[str, Any]:
        object_names = ("banana", "bowl", "rubiks_cube")
        centers = {name: _host(world.get_pose(name, env_id=0)[0]) for name in object_names}
        bounds = {}
        for name in object_names:
            corners, _ = world.get_bbox(name, env_id=0)
            rows = self.np.asarray([[float(value) for value in corner] for corner in corners], dtype=self.np.float64)
            bounds[name] = (rows.min(axis=0).tolist(), rows.max(axis=0).tolist())
        output = {}
        camera_names = ("head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
        for camera_name in camera_names:
            frame = self.np.ascontiguousarray(
                observation["image_obs"][camera_name][0].detach().cpu().numpy(), dtype=self.np.uint8
            )
            _require(frame.ndim == 3 and frame.shape[-1] == 3, f"malformed RGB frame: {camera_name}")
            rgb_sha = hashlib.sha256(frame.tobytes(order="C")).hexdigest()
            image_path = attempt_dir / f"{camera_name}.png"
            wrote = self.cv2.imwrite(str(image_path), self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))
            _require(wrote and image_path.is_file(), f"failed to retain gate RGB: {camera_name}")
            sensor = env.scene[camera_name]
            origin = _host(sensor.data.pos_w[0] - env.scene.env_origins[0])
            quaternion = _host(sensor.data.quat_w_ros[0])
            intrinsic = sensor.data.intrinsic_matrices[0].detach().cpu().tolist()
            projections: dict[str, list[float]] = {}
            unoccluded: dict[str, bool] = {}
            projection_errors: dict[str, str | None] = {}
            for name in object_names:
                try:
                    uv = _project(origin, quaternion, centers[name], intrinsic)
                    in_frame = 0.0 <= uv[0] < frame.shape[1] and 0.0 <= uv[1] < frame.shape[0]
                    blocked = any(
                        _segment_intersects_aabb_before_target(origin, centers[name], *bounds[other])
                        for other in object_names if other != name
                    )
                    projections[name] = [float(uv[0]), float(uv[1])]
                    unoccluded[name] = bool(in_frame and not blocked)
                    projection_errors[name] = None
                except Exception as error:
                    projections[name] = [0.0, 0.0]
                    unoccluded[name] = False
                    projection_errors[name] = f"{type(error).__name__}: {error}"
            geometry = {
                "camera_center_robot_base_m": origin,
                "camera_quaternion_world_wxyz_ros": quaternion,
                "intrinsic_matrix_3x3": intrinsic,
                "image_size_wh": [int(frame.shape[1]), int(frame.shape[0])],
                "object_centers_robot_base_m": centers,
                "object_aabbs_robot_base_m": {
                    name: {"lower": bounds[name][0], "upper": bounds[name][1]} for name in object_names
                },
                "projected_object_centers_uv": projections,
                "projected_unoccluded_by_object": unoccluded,
                "projection_errors": projection_errors,
            }
            geometry_sha = _canonical_sha(geometry)
            output[camera_name] = {
                "shape_hwc": list(frame.shape),
                "dtype": str(frame.dtype),
                "pixel_range": int(self.np.ptp(frame)),
                "rgb_sha256": rgb_sha,
                "rgb_artifact": {
                    "path": str(image_path.resolve()),
                    "sha256": _sha256_file(image_path),
                    "bytes": image_path.stat().st_size,
                },
                "visibility_method": "calibrated_projection_and_obb_occlusion",
                "visibility_source_sha256": geometry_sha,
                "camera_geometry_source_sha256": geometry_sha,
                "projected_object_centers_uv": projections,
                "projected_unoccluded_by_object": unoccluded,
                "geometry": geometry,
            }
        return output

    def _collision_rows(self, world: Any) -> dict[str, Any]:
        pairs = {
            "banana|bowl": ("banana", "bowl"),
            "banana|rubiks_cube": ("banana", "rubiks_cube"),
            "bowl|rubiks_cube": ("bowl", "rubiks_cube"),
            "robot_gripper|banana": ("gripper", "banana"),
            "robot_gripper|bowl": ("gripper", "bowl"),
            "robot_gripper|rubiks_cube": ("gripper", "rubiks_cube"),
        }
        output = {}
        for label, (left, right) in pairs.items():
            contact = bool(world.in_contact(left, right, env_id=0))
            force = _host(world.get_contact_force(left, right, env_id=0))
            evidence = {"bodies": [left, right], "contact": contact, "net_contact_force_world_n": force}
            output[label] = {
                "clear": not contact,
                "evidence_sha256": _canonical_sha(evidence),
                "evidence": evidence,
            }
        return {"query_complete": True, "forbidden_pairs": output}

    def capture_condition(
        self,
        *,
        candidate: Mapping[str, Any],
        layout_arm: str,
        command: str,
        repeat_index: int,
        environment_seed: int,
        gate_contract: Mapping[str, Any],
        attempt_dir: Path,
    ) -> Mapping[str, Any]:
        _require(not self._closed, "fixture gate adapter is closed")
        _require(candidate["candidate_id"] == self.candidate_id, "adapter was constructed for another candidate")
        attempt_dir = Path(attempt_dir)
        _require(not attempt_dir.exists(), f"refusing to overwrite condition evidence: {attempt_dir}")
        attempt_dir.mkdir(parents=True)
        self.set_output_dir(str(attempt_dir / "native"))
        task_name = self.TASKS[(layout_arm, command)]
        env, env_cfg = self.create_env(
            task_name,
            device=self.device,
            seed=environment_seed,
            num_envs=1,
            instruction_type="default",
            policy="wmf_model_blind_fixture_gate_no_policy",
            renderer=self.renderer,
            rendering_mode=self.rendering_type,
        )
        try:
            _require(not hasattr(env_cfg.terminations, "success"), "success termination exists in constructed task")
            observation, info = env.reset()
            hold = self._hold_action(observation)
            for _ in range(int(gate_contract["settle_steps"])):
                observation, _, terminated, truncated, info = env.step(hold)
                _require(not bool(terminated[0]) and not bool(truncated[0]), "task terminated during fixture settling")
            maxima = {
                name: {"max_linear_speed_m_s": 0.0, "max_angular_speed_rad_s": 0.0}
                for name in ("banana", "bowl", "rubiks_cube")
            }
            for _ in range(int(gate_contract["stability_window_steps"])):
                observation, _, terminated, truncated, info = env.step(hold)
                _require(not bool(terminated[0]) and not bool(truncated[0]), "task terminated during stability window")
                world = self.get_world(env)
                for name in maxima:
                    velocity = _host(world.get_velocity(name, env_id=0))
                    maxima[name]["max_linear_speed_m_s"] = max(
                        maxima[name]["max_linear_speed_m_s"], math.sqrt(sum(value * value for value in velocity[:3]))
                    )
                    maxima[name]["max_angular_speed_rad_s"] = max(
                        maxima[name]["max_angular_speed_rad_s"], math.sqrt(sum(value * value for value in velocity[3:]))
                    )
            world = self.get_world(env)
            settled = self._pose_rows(world)
            cameras = self._camera_rows(env, observation, world, attempt_dir)
            collisions = self._collision_rows(world)
            success = self.success_measurements(env)
            identities = self.observation_identities(observation)
            reset_state = {
                "layout_arm": layout_arm,
                "environment_seed": environment_seed,
                "settled_poses": settled,
                "proprio_obs": identities["proprio_obs"],
            }
            configured = {
                name: {
                    "position_robot_base_m": list(candidate["layouts"][layout_arm]["positions_robot_base_m"][name]),
                    "quaternion_wxyz": list(candidate["layouts"][layout_arm]["quaternions_wxyz"][name]),
                }
                for name in ("banana", "bowl", "rubiks_cube")
            }
            capture = {
                "schema_version": "wmf-forecast-layout-live-fixture-capture-v1",
                "study_namespace": "wmf_ablation_001_20260912",
                "candidate_id": candidate["candidate_id"],
                "candidate_payload_sha256": candidate["candidate_payload_sha256"],
                "layout_pair_id": candidate["layout_pair_id"],
                "layout_arm": layout_arm,
                "command": command,
                "repeat_index": repeat_index,
                "environment_seed": environment_seed,
                "model_request_count": 0,
                "behavioral_action_count": 0,
                "configured_poses": configured,
                "settled_poses": settled,
                "settle": {
                    "settle_steps": int(gate_contract["settle_steps"]),
                    "stability_window_steps": int(gate_contract["stability_window_steps"]),
                    "terminated_during_settle": False,
                    "truncated_during_settle": False,
                    "maxima_by_object": maxima,
                },
                "reset_fingerprints": {
                    "reset_state_sha256": _canonical_sha(reset_state),
                    "initial_observation_sha256": identities["combined_sha256"],
                    "initial_camera_rgb_sha256": {
                        name: cameras[name]["rgb_sha256"] for name in cameras
                    },
                },
                "cameras": cameras,
                "collision_checks": collisions,
                "success_predicates": {"left": success["left"], "right": success["right"]},
                "runtime": {
                    "study_commit": self.config["expected_study_commit"],
                    "robolab_commit": self.config["expected_robolab_commit"],
                    "pod": self.config["pod"],
                    "pod_uid": self.config["pod_uid"],
                    "gpu_uuid": self.config["gpu_uuid"],
                    "gpu_query": self.gpu_line,
                    "renderer": "realtime RTX balanced",
                },
            }
            payload = json.dumps(capture, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            capture_path = attempt_dir / "live_capture.json"
            capture_path.write_bytes(payload)
            return capture
        finally:
            env.close()

    def close(self) -> None:
        if not self._closed:
            self._simulation_app.close()
            self._closed = True


def make_adapter(config: Mapping[str, Any]) -> RoboLabFixtureGateAdapter:
    return RoboLabFixtureGateAdapter(config)
