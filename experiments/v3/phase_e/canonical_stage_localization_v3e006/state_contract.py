"""Pure state and gate contract for V3-E006 canonical manipulation stages."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .ood_reference import FEATURE_NAMES, normalized_distance, state_feature


STAGES = ("full_reset", "canonical_grasp", "canonical_carry")
MIDLINE_TOLERANCE_M = 0.001
RELATIVE_DRIFT_TOLERANCE_M = 0.002
ARM_JOINT_SPEED_TOLERANCE_RAD_S = 0.01
CUBE_LINEAR_SPEED_TOLERANCE_M_S = 0.01
CUBE_ANGULAR_SPEED_TOLERANCE_RAD_S = 0.05
UNINTENDED_CONTACT_FORCE_TOLERANCE_N = 1.0
SETTLED_WINDOW_STEPS = 10
NORMALIZATION_DECIMALS = 8


class StateContractError(ValueError):
    """A candidate or live state violated the preregistered contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rounded(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _rounded(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        _require(math.isfinite(number), "state contains a non-finite number")
        return round(number, NORMALIZATION_DECIMALS)
    return value


def normalized_state_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only restorable physics state, never request or output fields."""

    expected = {"robot", "objects", "eef"}
    _require(set(state) >= expected, f"state lacks restorable keys: {sorted(expected - set(state))}")
    return _rounded({key: state[key] for key in sorted(expected)})


def normalized_state_sha256(state: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(normalized_state_payload(state)))


def stage_ood(
    state: Mapping[str, Any],
    *,
    stage_reference: Mapping[str, Any],
) -> dict[str, Any]:
    robot = state["robot"]
    cube = state["objects"]["rubiks_cube"]
    eef = state["eef"]
    feature = state_feature(
        arm_joint_positions_rad=robot["joint_position_rad"][:7],
        eef_position_world_m=eef["position_world_m"],
        eef_quaternion_world_wxyz=eef["quaternion_world_wxyz"],
        cube_position_world_m=cube["position_world_m"],
        cube_quaternion_world_wxyz=cube["quaternion_world_wxyz"],
    )
    distance = normalized_distance(
        feature,
        center=stage_reference["direction_balanced_center"],
        scale=stage_reference["robust_scale"],
    )
    maximum = float(stage_reference["acceptance"]["maximum_distance_inclusive"])
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature": [float(value) for value in feature],
        "normalized_distance": distance,
        "maximum_distance_inclusive": maximum,
        "passed": bool(distance <= maximum),
    }


def settled_gate(
    samples: Sequence[Mapping[str, Any]],
    *,
    unintended_contact_pairs: Sequence[str],
) -> dict[str, Any]:
    _require(len(samples) == SETTLED_WINDOW_STEPS, "settled window must contain exactly ten consecutive samples")
    cube_positions = np.asarray([row["cube_position_world_m"] for row in samples], dtype=np.float64)
    eef_positions = np.asarray([row["eef_position_world_m"] for row in samples], dtype=np.float64)
    relative = cube_positions - eef_positions
    relative_drift = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
    midline = float(np.max(np.abs(cube_positions[:, 1])))
    arm_speed = float(max(max(abs(float(x)) for x in row["arm_joint_velocity_rad_s"][:7]) for row in samples))
    cube_linear = float(max(np.linalg.norm(row["cube_linear_velocity_m_s"]) for row in samples))
    cube_angular = float(max(np.linalg.norm(row["cube_angular_velocity_rad_s"]) for row in samples))
    unintended: dict[str, float] = {}
    for pair in unintended_contact_pairs:
        unintended[pair] = float(max(float(row["contact_force_n"].get(pair, 0.0)) for row in samples))
    normal_grasp = all(bool(row["object_grabbed"]) for row in samples)
    checks = {
        "cube_midline": midline < MIDLINE_TOLERANCE_M,
        "cube_gripper_relative_drift": relative_drift < RELATIVE_DRIFT_TOLERANCE_M,
        "arm_joint_speed": arm_speed < ARM_JOINT_SPEED_TOLERANCE_RAD_S,
        "cube_linear_speed": cube_linear < CUBE_LINEAR_SPEED_TOLERANCE_M_S,
        "cube_angular_speed": cube_angular < CUBE_ANGULAR_SPEED_TOLERANCE_RAD_S,
        "normal_gripper_contact": normal_grasp,
        "no_unintended_contacts": all(value <= UNINTENDED_CONTACT_FORCE_TOLERANCE_N for value in unintended.values()),
    }
    return {
        "settled_window_steps": SETTLED_WINDOW_STEPS,
        "observed": {
            "max_cube_midline_residual_m": midline,
            "max_cube_gripper_relative_drift_m": relative_drift,
            "max_arm_joint_speed_rad_s": arm_speed,
            "max_cube_linear_speed_m_s": cube_linear,
            "max_cube_angular_speed_rad_s": cube_angular,
            "max_unintended_contact_force_n_by_pair": unintended,
            "object_grabbed_all_steps": normal_grasp,
        },
        "thresholds": {
            "cube_midline_residual_m_strict": MIDLINE_TOLERANCE_M,
            "cube_gripper_relative_drift_m_strict": RELATIVE_DRIFT_TOLERANCE_M,
            "arm_joint_speed_rad_s_strict": ARM_JOINT_SPEED_TOLERANCE_RAD_S,
            "cube_linear_speed_m_s_strict": CUBE_LINEAR_SPEED_TOLERANCE_M_S,
            "cube_angular_speed_rad_s_strict": CUBE_ANGULAR_SPEED_TOLERANCE_RAD_S,
            "unintended_contact_force_n_inclusive": UNINTENDED_CONTACT_FORCE_TOLERANCE_N,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }

