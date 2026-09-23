"""Concrete scoped RoboLab Abs-IK bridge and controller for SGW-01 LAT."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .fixtures import FixtureCandidate, Pose
from .simulator_bridge import (
    Environment,
    ObjectState,
    ResetResult,
    ScriptedController,
    SimulatorBridge,
    SimulatorBridgeError,
    SimulatorSnapshot,
)
from .task_definitions import RoboLabTaskDefinition
from .robolab_measurements import articulation_body_frames, geometric_center_state
from .lat_workspace_capture import render_only_warmup


class RoboLabLatEnvironment:
    def __init__(self, env: Any, candidate: FixtureCandidate, evidence_root: Path) -> None:
        self._env = env
        self._candidate = candidate
        self._evidence_root = evidence_root
        self._reset_index = 0
        self._initial: dict[str, tuple[float, float, float]] | None = None
        step_dt = getattr(env, "step_dt", None)
        if step_dt is None or float(step_dt) <= 0:
            raise SimulatorBridgeError("RoboLab must expose a positive physical step_dt")
        self._step_dt_s = float(step_dt)
        self._steps = 0
        self._observation: Any = None

    def _snapshot(self) -> SimulatorSnapshot:
        from robolab.core.task.conditionals import object_grabbed
        from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
        from robolab.core.world.world_state import get_world

        world = get_world(self._env)
        sensors = get_contact_sensors(self._env.scene)
        support_sensor = sensors.get("rubiks_cube__table")
        if support_sensor is None:
            raise SimulatorBridgeError("RoboLab scene lacks required rubiks_cube__table contact sensor")
        force_matrix = getattr(support_sensor.data, "force_matrix_w", None)
        raw_force = force_matrix if force_matrix is not None else getattr(support_sensor.data, "net_forces_w", None)
        if raw_force is None:
            raise SimulatorBridgeError("rubiks_cube__table contact sensor lacks a force stream")
        support_vectors = np.asarray(raw_force.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3)
        cube_supported = bool(support_vectors.size and np.max(np.linalg.norm(support_vectors, axis=1)) >= 1.0)
        rows: dict[str, ObjectState] = {}
        reset_roots: dict[str, Pose] = {}
        origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        for name in self._candidate.object_poses:
            root_position, quaternion = world.get_pose(name, env_id=0)
            _corners, geometric_center = world.get_bbox(name, env_id=0)
            root_position_values = tuple(float(item) for item in root_position.detach().cpu().tolist())
            geometric_center_values = tuple(float(item) for item in geometric_center.tolist())
            quaternion_values = tuple(float(item) for item in quaternion.detach().cpu().tolist())
            asset = self._env.scene[name]
            com_position_values = tuple(float(item) for item in (asset.data.root_com_pos_w[0].detach().cpu().numpy() - origin))
            com_velocity_values = tuple(float(item) for item in asset.data.root_com_vel_w[0].detach().cpu().tolist())
            center_position, linear_speed, angular_speed = geometric_center_state(
                com_position_env_local_xyz_m=com_position_values,
                geometric_center_env_local_xyz_m=geometric_center_values,
                com_velocity_world=com_velocity_values,
            )
            reset_roots[name] = Pose(root_position_values, quaternion_values)
            rows[name] = ObjectState(
                pose=Pose(center_position, quaternion_values),
                linear_speed_m_s=linear_speed,
                angular_speed_rad_s=angular_speed,
                supported=cube_supported if name == "rubiks_cube" else True,
                attached_to_gripper=(
                    bool(object_grabbed(self._env, object=name, env_id=0))
                    if name == "rubiks_cube"
                    else False
                ),
            )
        return SimulatorSnapshot(
            rows, simulated_time_s=self._steps * self._step_dt_s, reset_root_poses=reset_roots,
            robot_body_frames=articulation_body_frames(self._env.scene["robot"].data),
        )

    def reset(self) -> ResetResult:
        counter = getattr(self._env, "episode_length_buf", None)
        if counter is None or not hasattr(counter, "zero_"):
            raise SimulatorBridgeError("RoboLab must expose episode_length_buf for a physical reset")
        counter.zero_()
        observation, _ = self._env.reset()
        observation, warmup = render_only_warmup(
            self._env, observation, 120, self._evidence_root / f"reset-{self._reset_index + 1:02d}",
        )
        self._observation = observation
        self._steps = 0
        snapshot = self._snapshot()
        self._initial = {name: state.pose.position_m for name, state in snapshot.objects.items()}
        self._reset_index += 1
        camera = "over_shoulder_left_camera"
        image = observation["image_obs"][camera][0].detach().cpu().numpy()
        if image.ndim != 3 or image.shape[-1] != 3 or not np.ptp(image):
            raise SimulatorBridgeError("RoboLab reset did not expose a nonblank stable camera")
        fingerprint = hashlib.sha256(
            json.dumps(
                {name: asdict(state.pose) for name, state in snapshot.objects.items()},
                sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return ResetResult(
            snapshot,
            {
                "reset_id": f"{self._candidate.candidate_id}:{self._reset_index}",
                "camera_id": camera,
                "camera_name": camera,
                "fingerprint": fingerprint,
                "temporal_cache_reset": True,
                "control_step_dt_s": self._step_dt_s,
                "render_only_warmup": warmup,
            },
        )

    def step(self, action: Sequence[float]) -> SimulatorSnapshot:
        import torch

        tensor = action if isinstance(action, torch.Tensor) else torch.as_tensor(action, dtype=torch.float32)
        if tuple(tensor.shape) != (1, 8):
            raise SimulatorBridgeError(f"Abs-IK action must have shape (1, 8), got {tuple(tensor.shape)}")
        observation, _reward, terminated, truncated, _info = self._env.step(tensor.to(self._env.device))
        self._observation = observation
        self._steps += 1
        snapshot = self._snapshot()
        if bool(terminated[0]) or bool(truncated[0]):
            snapshot = replace(snapshot, termination_reason="native_termination_or_truncation")
        return snapshot

    def snapshot(self) -> SimulatorSnapshot:
        return self._snapshot()

    def render_viewport(self) -> np.ndarray:
        if self._observation is None:
            raise SimulatorBridgeError("viewport requested before a physical reset")
        frame = self._observation["image_obs"]["over_shoulder_left_camera"][0].detach().cpu().numpy()
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 or not np.ptp(frame):
            raise SimulatorBridgeError("qualification viewport is not nonblank uint8 RGB")
        return frame.copy()

    def close(self) -> None:
        self._env.close()


class RoboLabLatBridge:
    def __init__(self, *, study_root: Path, robolab_root: Path, evidence_root: Path, device: str, renderer: str, rendering_type: str) -> None:
        self._study_root = Path(study_root).resolve()
        self._robolab_root = Path(robolab_root).resolve()
        self._device = device
        self._evidence_root = evidence_root
        if renderer != "realtime" or rendering_type != "balanced":
            raise SimulatorBridgeError("LAT qualification requires realtime/balanced RTX")

    def create_environment(self, task: RoboLabTaskDefinition, seed: int) -> Environment:
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        payload = json.dumps(task.bridge_config(), sort_keys=True, separators=(",", ":"))
        os.environ["SGW_LAT_CANDIDATE_JSON"] = payload
        os.environ["SGW_LAT_CANDIDATE_SHA256"] = hashlib.sha256(payload.encode()).hexdigest()
        task_path = self._study_root / "experiments/workshops/spatial_grounding_v1/lat_qualification_task.py"
        register_lat_task(auto_register_droid_abs_ik_envs, task_path, WRIST_LEFT_RIGHT_HEAD)
        env, _ = create_env(
            "SGWLatQualificationTask", device=self._device, seed=seed, num_envs=1,
            instruction_type="default", policy="sgw_01_model_blind_lat_controller",
            renderer="realtime", rendering_mode="balanced",
        )
        return RoboLabLatEnvironment(env, task.candidate, self._evidence_root)


def register_lat_task(registrar: Any, task_path: Path, cameras: Any) -> None:
    """Register only the worktree overlay; never write the pinned RoboLab tree."""

    if task_path.name != "lat_qualification_task.py" or not task_path.is_file():
        raise SimulatorBridgeError("LAT qualification task overlay is missing")
    registrar(task=[str(task_path)], cameras=cameras)


class RoboLabLatScriptedController:
    """Execute candidate-recorded world-frame Abs-IK waypoints without a policy."""

    def actions_for_goal(self, environment: Environment, candidate: FixtureCandidate, goal_sign: int) -> Sequence[Sequence[float]]:
        if not isinstance(environment, RoboLabLatEnvironment):
            raise SimulatorBridgeError("LAT controller requires RoboLabLatEnvironment")
        key = "positive" if goal_sign == 1 else "negative"
        waypoints = candidate.metadata.get("abs_ik_waypoints", {}).get(key)
        if not isinstance(waypoints, list) or not waypoints:
            raise SimulatorBridgeError("candidate lacks measured Abs-IK waypoints for the requested LAT goal")
        from robolab.robots.droid import EEF_OFFSET_ROT

        frames = environment._env.scene["frames"]
        index = frames.data.target_frame_names.index("eef_frame")
        eef_quaternion = np.asarray(frames.data.target_quat_w[0, index].detach().cpu().numpy(), dtype=np.float64)
        offset_inverse = np.asarray([EEF_OFFSET_ROT[0], -EEF_OFFSET_ROT[1], -EEF_OFFSET_ROT[2], -EEF_OFFSET_ROT[3]], dtype=np.float64)
        command_quaternion = _quat_mul(eef_quaternion, offset_inverse)
        actions: list[np.ndarray] = []
        for waypoint in waypoints:
            position = np.asarray(waypoint["position_world_xyz_m"], dtype=np.float64)
            hold_steps = int(waypoint["hold_steps"])
            gripper = float(waypoint["gripper_position"])
            if position.shape != (3,) or hold_steps < 1 or len(actions) + hold_steps > 450:
                raise SimulatorBridgeError("invalid measured LAT waypoint sequence")
            command = np.concatenate((position, command_quaternion, [gripper])).astype(np.float32).reshape(1, 8)
            actions.extend(command.copy() for _ in range(hold_steps))
        return actions


def _quat_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ))


def create_bridge(*, robolab_root: Path, assets_manifest: Path, evidence_root: Path, device: str, renderer: str, rendering_type: str, **_: Any) -> RoboLabLatBridge:
    study_root = Path(__file__).resolve().parents[3]
    if not Path(assets_manifest).is_file():
        raise SimulatorBridgeError("measured asset manifest is required")
    return RoboLabLatBridge(study_root=study_root, robolab_root=robolab_root, evidence_root=evidence_root, device=device, renderer=renderer, rendering_type=rendering_type)


def create_controller(**_: Any) -> RoboLabLatScriptedController:
    return RoboLabLatScriptedController()
