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
    intended_contact_pair: str = "gripper__rubiks_cube",
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
    intended_contact = float(min(float(row["contact_force_n"].get(intended_contact_pair, 0.0)) for row in samples))
    checks = {
        "cube_midline": midline < MIDLINE_TOLERANCE_M,
        "cube_gripper_relative_drift": relative_drift < RELATIVE_DRIFT_TOLERANCE_M,
        "arm_joint_speed": arm_speed < ARM_JOINT_SPEED_TOLERANCE_RAD_S,
        "cube_linear_speed": cube_linear < CUBE_LINEAR_SPEED_TOLERANCE_M_S,
        "cube_angular_speed": cube_angular < CUBE_ANGULAR_SPEED_TOLERANCE_RAD_S,
        "normal_gripper_contact": normal_grasp,
        "intended_cube_gripper_contact_force": intended_contact > UNINTENDED_CONTACT_FORCE_TOLERANCE_N,
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
            "minimum_intended_cube_gripper_contact_force_n": intended_contact,
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


def _source_values(source: Mapping[str, Any], key: str) -> np.ndarray:
    row = source[key]
    _require(isinstance(row, Mapping) and isinstance(row.get("values"), list), f"reference {key} lacks values")
    value = np.asarray(row["values"], dtype=np.float64)
    _require(np.isfinite(value).all(), f"reference {key} is non-finite")
    return value


def _quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def compare_full_reset_to_e004(
    state: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    reference_file_sha256: str,
) -> dict[str, Any]:
    """Bind and compare the full physics state to a retained E004 s=1 reset."""

    _require(reference.get("schema_version") == "vla-wam-shared-v3e004-request0-reset-contract-v1", "wrong E004 reset-reference schema")
    robot_source = reference["robot"]
    robot = state["robot"]
    joint_delta = float(
        np.max(np.abs(np.asarray(robot["joint_position_rad"], dtype=np.float64) - _source_values(robot_source, "joint_position")))
    )
    joint_velocity_delta = float(
        np.max(np.abs(np.asarray(robot["joint_velocity_rad_s"], dtype=np.float64) - _source_values(robot_source, "joint_velocity")))
    )
    root_position_delta = float(
        np.linalg.norm(np.asarray(robot["root_position_world_m"], dtype=np.float64) - _source_values(robot_source, "root_position"))
    )
    root_orientation_delta = _quaternion_distance(
        robot["root_quaternion_world_wxyz"], _source_values(robot_source, "root_quaternion_wxyz")
    )
    object_rows: dict[str, Any] = {}
    for name, actual in sorted(state["objects"].items()):
        source = reference["rigid_objects"][name]
        object_rows[name] = {
            "position_delta_m": float(
                np.linalg.norm(np.asarray(actual["position_world_m"], dtype=np.float64) - _source_values(source, "root_position"))
            ),
            "orientation_delta_rad": _quaternion_distance(
                actual["quaternion_world_wxyz"], _source_values(source, "root_quaternion_wxyz")
            ),
            "linear_velocity_delta_m_s": float(
                np.linalg.norm(np.asarray(actual["linear_velocity_m_s"], dtype=np.float64) - _source_values(source, "root_linear_velocity"))
            ),
            "angular_velocity_delta_rad_s": float(
                np.linalg.norm(np.asarray(actual["angular_velocity_rad_s"], dtype=np.float64) - _source_values(source, "root_angular_velocity"))
            ),
        }
    comparable = {
        "robot": {
            "joint_position_rad": robot["joint_position_rad"],
            "joint_velocity_rad_s": robot["joint_velocity_rad_s"],
            "root_position_world_m": robot["root_position_world_m"],
            "root_quaternion_world_wxyz": robot["root_quaternion_world_wxyz"],
        },
        "objects": state["objects"],
    }
    candidate_comparable_sha = sha256_bytes(canonical_bytes(_rounded(comparable)))
    thresholds = {
        "max_joint_position_delta_rad_inclusive": 1e-6,
        "max_joint_velocity_delta_rad_s_inclusive": 1e-6,
        "max_robot_root_position_delta_m_inclusive": 1e-9,
        "max_robot_root_orientation_delta_rad_inclusive": 1e-9,
        "max_object_position_delta_m_inclusive": 0.005,
        "max_object_orientation_delta_rad_inclusive": math.radians(2.0),
    }
    passed = (
        joint_delta <= thresholds["max_joint_position_delta_rad_inclusive"]
        and joint_velocity_delta <= thresholds["max_joint_velocity_delta_rad_s_inclusive"]
        and root_position_delta <= thresholds["max_robot_root_position_delta_m_inclusive"]
        and root_orientation_delta <= thresholds["max_robot_root_orientation_delta_rad_inclusive"]
        and all(row["position_delta_m"] <= thresholds["max_object_position_delta_m_inclusive"] for row in object_rows.values())
        and all(row["orientation_delta_rad"] <= thresholds["max_object_orientation_delta_rad_inclusive"] for row in object_rows.values())
    )
    return {
        "reference_file_sha256": reference_file_sha256,
        "reference_reset_contract_sha256": reference["reset_contract_sha256"],
        "candidate_comparable_state_sha256": candidate_comparable_sha,
        "observed": {
            "max_joint_position_delta_rad": joint_delta,
            "max_joint_velocity_delta_rad_s": joint_velocity_delta,
            "robot_root_position_delta_m": root_position_delta,
            "robot_root_orientation_delta_rad": root_orientation_delta,
            "objects": object_rows,
        },
        "thresholds": thresholds,
        "passed": passed,
    }
