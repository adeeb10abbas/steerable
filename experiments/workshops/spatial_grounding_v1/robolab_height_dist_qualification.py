"""Pinned RoboLab bridge factories for the measured HEIGHT/DIST overlays.

The bridge deliberately reuses the model-blind Abs-IK environment/controller
contract.  It is not a policy adapter: a learned policy never sees the
scripted actions, native scoring state, or support geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .fixtures import FixtureCandidate
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
        validate_candidate_inputs(task.candidate)
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
        from .grasp_calibration import SCHEMA
        from .lat_candidate_generator import workspace_digest

        raw = Path(calibration_path).read_bytes()
        self.calibration = json.loads(raw)
        if (self.calibration.get("schema_version") != SCHEMA
                or self.calibration.get("receipt_sha256") != workspace_digest(self.calibration)):
            raise SimulatorBridgeError("invalid measured Abs-IK calibration identity")
        self.identity = {
            "recipe": SCHEMA,
            "calibration_sha256": hashlib.sha256(raw).hexdigest(),
            "calibration": self.calibration,
        }

    def actions_for_goal(self, environment: Environment, candidate: FixtureCandidate, goal_sign: int) -> list[np.ndarray]:
        if not isinstance(environment, RoboLabLatEnvironment) or candidate.family not in {"HEIGHT", "DIST"}:
            raise SimulatorBridgeError("family controller requires a HEIGHT/DIST RoboLab environment")
        if goal_sign not in (-1, 1):
            raise SimulatorBridgeError("family controller goal sign is invalid")
        support = _goal_support(candidate, goal_sign)
        target_local = np.asarray(support["cube_center_env_local_xyz_m"], dtype=np.float64)
        if target_local.shape != (3,) or not np.isfinite(target_local).all():
            raise SimulatorBridgeError("measured goal support center must be finite 3-vector")
        robot = environment._env.scene["robot"]
        if hashlib.sha256(Path(robot.cfg.spawn.usd_path).read_bytes()).hexdigest() != self.calibration["robot_asset"]["sha256"]:
            raise SimulatorBridgeError("actual robot asset differs from measured gripper geometry")
        data = robot.data
        if not np.allclose(data.root_quat_w[0].detach().cpu().numpy(), [1, 0, 0, 0], atol=1e-6):
            raise SimulatorBridgeError("calibrated controller requires identity robot-root orientation")
        index = list(data.body_names).index("base_link")
        origin = environment._env.scene.env_origins[0].detach().cpu().numpy()
        cube_center = np.asarray(candidate.scoring_poses()["rubiks_cube"].position_m) + origin
        from .grasp_calibration import calibrated_actions_to_target

        return calibrated_actions_to_target(
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


def validate_candidate_inputs(candidate: FixtureCandidate) -> None:
    """Reject an incomplete capture before importing or starting RoboLab."""

    scene = candidate.metadata.get("native_scene")
    supports = candidate.metadata.get("goal_supports")
    if not isinstance(scene, dict) or not isinstance(scene.get("asset"), str) or not scene["asset"]:
        raise SimulatorBridgeError("family candidate lacks measured native scene asset")
    if not isinstance(scene.get("object_names"), list) or not all(isinstance(name, str) and name for name in scene["object_names"]):
        raise SimulatorBridgeError("family candidate lacks measured native scene object inventory")
    required = {"rubiks_cube", "bowl"} | ({"plate"} if candidate.family == "DIST" else set())
    if not required.issubset(scene["object_names"]) or "table" not in scene["object_names"]:
        raise SimulatorBridgeError("family native scene lacks a scored object or table")
    if not isinstance(supports, dict) or not supports:
        raise SimulatorBridgeError("family candidate lacks measured released goal supports")
    expected = {"higher", "lower"} if candidate.family == "HEIGHT" else {"near_bowl", "near_plate"}
    if set(supports) != expected:
        raise SimulatorBridgeError("family candidate has incomplete measured goal supports")
    for support in supports.values():
        if not isinstance(support, dict) or not isinstance(support.get("contact_sensor_id"), str) or not support["contact_sensor_id"].strip():
            raise SimulatorBridgeError("family candidate lacks measured nonempty support contact sensor IDs")
        target = np.asarray(support.get("cube_center_env_local_xyz_m"), dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise SimulatorBridgeError("family candidate has invalid measured goal support center")
