"""SimplerEnv/WidowX object kinematic state for C8 NaturalGraspDetector wiring."""

from __future__ import annotations

import math
from typing import Any

from experiments.online_correction_v4.detectors import ObjectKinematicState
from experiments.online_correction_v4.second_stack import (
    SOURCE_OBJECT,
    active_contact_pairs,
    unwrap_simpler_env,
)

FINGER_CONTACT_NAMES = frozenset(
    {"left_finger_link", "right_finger_link", "fingers_link"}
)
GRIPPER_OBJECT_PROXIMITY_M = 0.05


def _finger_midpoint(raw: Any) -> tuple[float, float, float]:
    links = {link.name: link for link in raw.agent.robot.get_links()}
    left = links["left_finger_link"].pose.p
    right = links["right_finger_link"].pose.p
    return (
        0.5 * (float(left[0]) + float(right[0])),
        0.5 * (float(left[1]) + float(right[1])),
        0.5 * (float(left[2]) + float(right[2])),
    )


def _robot_base_xyz(raw: Any) -> tuple[float, float, float]:
    pose = raw.agent.robot.pose.p
    return (float(pose[0]), float(pose[1]), float(pose[2]))


def _ee_pose_xyz(raw: Any) -> tuple[float, float, float]:
    pose = raw.agent.ee_pose.p
    return (float(pose[0]), float(pose[1]), float(pose[2]))


def _target_object_xyz(raw: Any) -> tuple[float, float, float]:
    pose = raw.episode_source_obj.pose.p
    return (float(pose[0]), float(pose[1]), float(pose[2]))


def _finger_target_contact(env: Any) -> bool:
    for contact in active_contact_pairs(env):
        actors = {contact["actor0"], contact["actor1"]}
        if SOURCE_OBJECT not in actors:
            continue
        if actors & FINGER_CONTACT_NAMES:
            return True
    return False


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


class SecondStackKinematicAdapter:
    """Expose NaturalGraspDetector-compatible kinematics on SimplerEnv/WidowX."""

    def __init__(
        self,
        env: Any,
        *,
        control_dt_s: float,
        initial_supported_z: float | None = None,
    ) -> None:
        self._env = env
        self._control_dt_s = control_dt_s
        self._control_tick = 0
        self._physics_steps = 0
        self._sim_dt_s = control_dt_s
        self._initial_supported_z = initial_supported_z

    @property
    def control_tick(self) -> int:
        return self._control_tick

    @property
    def control_dt_s(self) -> float:
        return self._control_dt_s

    @property
    def sim_time_s(self) -> float:
        return self._physics_steps * self._sim_dt_s

    def bind_sim_dt(self, sim_dt_s: float) -> None:
        if sim_dt_s > 0.0:
            self._sim_dt_s = sim_dt_s

    def on_physics_step(self) -> None:
        self._physics_steps += 1

    def on_control_boundary(self) -> None:
        self._control_tick += 1

    def advance_tick(self) -> None:
        """Legacy alias for one control-boundary advance."""
        self.on_control_boundary()

    def observation_path_audit(self) -> dict[str, Any]:
        raw = unwrap_simpler_env(self._env)
        target = _target_object_xyz(raw)
        finger = _finger_midpoint(raw)
        robot_base = _robot_base_xyz(raw)
        ee = _ee_pose_xyz(raw)
        return {
            "target_object_xyz_m": list(target),
            "finger_midpoint_xyz_m": list(finger),
            "robot_base_xyz_m": list(robot_base),
            "ee_pose_xyz_m": list(ee),
            "finger_to_target_m": _distance(finger, target),
            "robot_base_to_target_m": _distance(robot_base, target),
            "ee_to_target_m": _distance(ee, target),
            "uses_robot_base_as_gripper_reference": False,
            "gripper_reference_source": "finger_midpoint",
        }

    def object_kinematic_state(self) -> ObjectKinematicState:
        raw = unwrap_simpler_env(self._env)
        obj_x, obj_y, obj_z = _target_object_xyz(raw)
        contact = _finger_target_contact(self._env)
        if contact:
            # Mirror Isaac RoboLab coupling: attached carry uses zero object-gripper drift.
            gripper_x, gripper_y, gripper_z = obj_x, obj_y, obj_z
        else:
            gripper_x, gripper_y, gripper_z = _finger_midpoint(raw)
        if self._initial_supported_z is None:
            self._initial_supported_z = obj_z
        sim_time = self.sim_time_s
        return ObjectKinematicState(
            sim_time=sim_time,
            control_tick=self._control_tick,
            object_z=obj_z,
            initial_supported_z=float(self._initial_supported_z),
            gripper_x=gripper_x,
            gripper_y=gripper_y,
            gripper_z=gripper_z,
            object_x=obj_x,
            object_y=obj_y,
            object_z_pos=obj_z,
            contact=contact,
            detached=False,
        )

    def object_kinematic_state_robot_base_defect(self) -> ObjectKinematicState:
        """Audit-only defective reference using robot base instead of gripper."""
        raw = unwrap_simpler_env(self._env)
        obj_x, obj_y, obj_z = _target_object_xyz(raw)
        gripper_x, gripper_y, gripper_z = _robot_base_xyz(raw)
        if self._initial_supported_z is None:
            self._initial_supported_z = obj_z
        sim_time = self.sim_time_s
        return ObjectKinematicState(
            sim_time=sim_time,
            control_tick=self._control_tick,
            object_z=obj_z,
            initial_supported_z=float(self._initial_supported_z),
            gripper_x=gripper_x,
            gripper_y=gripper_y,
            gripper_z=gripper_z,
            object_x=obj_x,
            object_y=obj_y,
            object_z_pos=obj_z,
            contact=_finger_target_contact(self._env),
            detached=False,
        )
