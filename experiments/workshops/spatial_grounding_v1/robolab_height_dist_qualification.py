"""Pinned RoboLab bridge factories for the measured HEIGHT/DIST overlays.

The bridge deliberately reuses the model-blind Abs-IK environment/controller
contract.  It is not a policy adapter: a learned policy never sees the
scripted actions, native scoring state, or support geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .fixtures import ACTION_CAP, FixtureCandidate
from .robolab_lat_qualification import RoboLabLatEnvironment
from .simulator_bridge import Environment, SimulatorBridgeError
from .task_definitions import RoboLabTaskDefinition


CALIBRATION_SCHEMA = "sgw-01-lat-closed-pad-midpoint-v1"


class RoboLabFamilyBridge:
    def __init__(self, *, study_root: Path, evidence_root: Path, device: str, renderer: str, rendering_type: str) -> None:
        self._study_root = Path(study_root).resolve()
        self._evidence_root = Path(evidence_root).resolve()
        self._device = device
        if renderer != "realtime" or rendering_type != "balanced":
            raise SimulatorBridgeError("HEIGHT/DIST qualification requires realtime/balanced RTX")

    def create_environment(self, task: RoboLabTaskDefinition, seed: int) -> Environment:
        if task.candidate.family not in {"HEIGHT", "DIST"}:
            raise SimulatorBridgeError("family bridge only accepts HEIGHT or DIST candidates")
        if not task.candidate.metadata.get("goal_supports"):
            raise SimulatorBridgeError("family candidate lacks measured released goal supports")
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        payload = json.dumps(task.bridge_config(), sort_keys=True, separators=(",", ":"))
        os.environ["SGW_FAMILY_CANDIDATE_JSON"] = payload
        os.environ["SGW_FAMILY_CANDIDATE_SHA256"] = hashlib.sha256(payload.encode()).hexdigest()
        task_path = self._study_root / "experiments/workshops/spatial_grounding_v1/family_qualification_task.py"
        if not task_path.is_file():
            raise SimulatorBridgeError("HEIGHT/DIST task overlay is missing")
        auto_register_droid_abs_ik_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        env, _ = create_env(
            "SGWFamilyQualificationTask", device=self._device, seed=seed, num_envs=1,
            instruction_type="default", policy="sgw_01_model_blind_family_controller",
            renderer="realtime", rendering_mode="balanced",
        )
        return RoboLabLatEnvironment(env, task.candidate, self._evidence_root)


def create_bridge(
    *, robolab_root: Path, assets_manifest: Path, evidence_root: Path, device: str, renderer: str, rendering_type: str, **_: Any
) -> RoboLabFamilyBridge:
    if not Path(assets_manifest).is_file():
        raise SimulatorBridgeError("verified actual asset manifest is required")
    if not isinstance(evidence_root, Path):
        raise SimulatorBridgeError("explicit qualification evidence_root is required")
    return RoboLabFamilyBridge(
        study_root=Path(__file__).resolve().parents[3],
        evidence_root=evidence_root,
        device=device,
        renderer=renderer,
        rendering_type=rendering_type,
    )


class RoboLabFamilyScriptedController:
    """Create a 450-action Abs-IK trajectory from a measured 3D support center."""

    def __init__(self, calibration_path: Path) -> None:
        raw = Path(calibration_path).read_bytes()
        self.calibration = json.loads(raw)
        digest = _calibration_digest(self.calibration)
        if (self.calibration.get("schema_version") != CALIBRATION_SCHEMA
                or self.calibration.get("receipt_sha256") != digest):
            raise SimulatorBridgeError("invalid measured Abs-IK calibration identity")

    def actions_for_goal(self, environment: Environment, candidate: FixtureCandidate, goal_sign: int) -> list[np.ndarray]:
        if not isinstance(environment, RoboLabLatEnvironment) or candidate.family not in {"HEIGHT", "DIST"}:
            raise SimulatorBridgeError("family controller requires a HEIGHT/DIST RoboLab environment")
        if goal_sign not in (-1, 1):
            raise SimulatorBridgeError("family controller goal sign is invalid")
        support = _goal_support(candidate, goal_sign)
        target_local = _vector3(support["cube_center_env_local_xyz_m"], "measured goal support center")
        robot = environment._env.scene["robot"]
        if hashlib.sha256(Path(robot.cfg.spawn.usd_path).read_bytes()).hexdigest() != self.calibration["robot_asset"]["sha256"]:
            raise SimulatorBridgeError("actual robot asset differs from measured gripper geometry")
        data = robot.data
        if not np.allclose(data.root_quat_w[0].detach().cpu().numpy(), [1, 0, 0, 0], atol=1e-6):
            raise SimulatorBridgeError("calibrated controller requires identity robot-root orientation")
        index = list(data.body_names).index("base_link")
        origin = environment._env.scene.env_origins[0].detach().cpu().numpy()
        cube_center = np.asarray(candidate.scoring_poses()["rubiks_cube"].position_m) + origin
        return _calibrated_actions_to_target(
            self.calibration,
            cube_center_world_xyz_m=cube_center,
            target_center_world_xyz_m=target_local + origin,
            flange_quaternion_world_wxyz=data.body_quat_w[0, index].detach().cpu().numpy(),
            robot_root_world_xyz_m=data.root_pos_w[0].detach().cpu().numpy(),
        )


def create_controller(*, controller_calibration: Path | None = None, **_: Any) -> RoboLabFamilyScriptedController:
    """Reject legacy candidate waypoints; family qualification requires calibration."""

    if controller_calibration is None:
        raise SimulatorBridgeError("HEIGHT/DIST qualification requires --controller-calibration")
    return RoboLabFamilyScriptedController(Path(controller_calibration))


def _goal_support(candidate: FixtureCandidate, goal_sign: int) -> dict[str, Any]:
    supports = candidate.metadata.get("goal_supports")
    if not isinstance(supports, dict):
        raise SimulatorBridgeError("family candidate lacks measured goal supports")
    if candidate.family == "HEIGHT":
        key = "higher" if goal_sign == 1 else "lower"
    else:
        key = "near_bowl" if goal_sign == 1 else "near_plate"
    support = supports.get(key)
    if not isinstance(support, dict) or not support.get("contact_sensor_id"):
        raise SimulatorBridgeError(f"family candidate lacks measured {key} support")
    return support


def _calibration_digest(value: dict[str, Any]) -> str:
    material = dict(value)
    material.pop("receipt_sha256", None)
    return hashlib.sha256((json.dumps(material, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def _vector3(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise SimulatorBridgeError(f"{label} must be a finite 3-vector")
    return vector


def _calibrated_actions_to_target(
    calibration: dict[str, Any], *, cube_center_world_xyz_m: np.ndarray, target_center_world_xyz_m: np.ndarray,
    flange_quaternion_world_wxyz: np.ndarray, robot_root_world_xyz_m: np.ndarray,
) -> list[np.ndarray]:
    """Keep the measured-TCP recipe while preserving the measured target Z."""

    cube, target = _vector3(cube_center_world_xyz_m, "cube center"), _vector3(target_center_world_xyz_m, "target center")
    quaternion, root = _vector3(flange_quaternion_world_wxyz[1:], "flange quaternion vector"), _vector3(robot_root_world_xyz_m, "robot root")
    if not math.isfinite(float(flange_quaternion_world_wxyz[0])):
        raise SimulatorBridgeError("flange quaternion must be finite")
    tcp = _vector3(calibration.get("virtual_tcp_flange_xyz_m"), "measured virtual TCP")
    lift_height = calibration.get("lift_height_m")
    holds = calibration.get("phase_hold_steps")
    if not isinstance(lift_height, (int, float)) or not math.isfinite(lift_height) or not isinstance(holds, list) or len(holds) != 8:
        raise SimulatorBridgeError("measured calibration lacks trajectory fields")
    if any(type(count) is not int or count < 1 for count in holds) or sum(holds) != ACTION_CAP:
        raise SimulatorBridgeError("measured calibration does not define exactly 450 actions")
    w, x, y, z = flange_quaternion_world_wxyz
    offset = np.asarray((
        (1 - 2 * (y*y + z*z))*tcp[0] + 2*(x*y - z*w)*tcp[1] + 2*(x*z + y*w)*tcp[2],
        2*(x*y + z*w)*tcp[0] + (1 - 2*(x*x + z*z))*tcp[1] + 2*(y*z - x*w)*tcp[2],
        2*(x*z - y*w)*tcp[0] + 2*(y*z + x*w)*tcp[1] + (1 - 2*(x*x + y*y))*tcp[2],
    ))
    lift = np.asarray([0, 0, float(lift_height)])
    points = (cube + lift, cube, cube, cube + lift, target + lift, target, target, target + lift)
    grips = (0, 0, 0.785398, 0.785398, 0.785398, 0.785398, 0, 0)
    actions = []
    for point, grip, count in zip(points, grips, holds, strict=True):
        command = np.concatenate((point - offset - root, flange_quaternion_world_wxyz, [grip])).astype(np.float32).reshape(1, 8)
        if not np.isfinite(command).all():
            raise SimulatorBridgeError("calibrated family command is nonfinite")
        actions.extend(command.copy() for _ in range(count))
    return actions
