"""Pure symmetric SE(3) measured-residual correction for V3-E006-R006."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def normalize_quaternion(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion is malformed")
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("quaternion has zero norm")
    return q / norm


def multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = normalize_quaternion(left)
    bw, bx, by, bz = normalize_quaternion(right)
    return normalize_quaternion(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def inverse(value: Sequence[float]) -> np.ndarray:
    q = normalize_quaternion(value)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def canonical_shortest_arc(value: Sequence[float]) -> np.ndarray:
    q = normalize_quaternion(value)
    if q[0] < 0.0:
        return -q
    if q[0] == 0.0:
        for component in q[1:]:
            if component < 0.0:
                return -q
            if component > 0.0:
                break
    return q


def quaternion_power(value: Sequence[float], gain: float) -> np.ndarray:
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("rotation gain is invalid")
    q = canonical_shortest_arc(value)
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm <= 1e-15:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(q[0]))
    axis = q[1:] / vector_norm
    return normalize_quaternion(
        np.concatenate(([math.cos(gain * angle / 2.0)], axis * math.sin(gain * angle / 2.0)))
    )


def corrected_command(
    *,
    desired_position: Sequence[float],
    desired_quaternion: Sequence[float],
    measured_position: Sequence[float],
    measured_quaternion: Sequence[float],
    current_command_position: Sequence[float],
    current_command_quaternion: Sequence[float],
    translation_gain: float,
    rotation_gain: float,
) -> dict[str, Any]:
    desired_p = np.asarray(desired_position, dtype=np.float64)
    measured_p = np.asarray(measured_position, dtype=np.float64)
    current_p = np.asarray(current_command_position, dtype=np.float64)
    if any(value.shape != (3,) for value in (desired_p, measured_p, current_p)):
        raise ValueError("position vector is malformed")
    if not np.all(np.isfinite(np.concatenate((desired_p, measured_p, current_p)))):
        raise ValueError("position vector is nonfinite")
    if not math.isfinite(translation_gain) or translation_gain < 0.0:
        raise ValueError("translation gain is invalid")
    desired_q = normalize_quaternion(desired_quaternion)
    measured_q = normalize_quaternion(measured_quaternion)
    current_q = normalize_quaternion(current_command_quaternion)
    position_residual = desired_p - measured_p
    error_world = canonical_shortest_arc(multiply(desired_q, inverse(measured_q)))
    delta = quaternion_power(error_world, rotation_gain)
    next_position = current_p + translation_gain * position_residual
    next_quaternion = multiply(delta, current_q)
    return {
        "desired_position_world_m": desired_p.tolist(),
        "desired_quaternion_world_wxyz": desired_q.tolist(),
        "measured_position_world_m": measured_p.tolist(),
        "measured_quaternion_world_wxyz": measured_q.tolist(),
        "current_command_position_world_m": current_p.tolist(),
        "current_command_quaternion_world_wxyz": current_q.tolist(),
        "translation_residual_world_m": position_residual.tolist(),
        "rotation_residual_world_wxyz_shortest_arc": error_world.tolist(),
        "translation_gain": float(translation_gain),
        "rotation_gain": float(rotation_gain),
        "next_command_position_world_m": next_position.tolist(),
        "next_command_quaternion_world_wxyz": next_quaternion.tolist(),
    }


def validate_contract(value: Mapping[str, Any]) -> None:
    expected = {
        "algorithm_version": "symmetric-se3-measured-residual-correction-v1",
        "maximum_correction_rounds": 3,
        "hold_steps_per_round": 30,
        "required_final_consecutive_steps": 10,
        "position_error_m_inclusive": 0.001,
        "orientation_geodesic_error_deg_inclusive": 1.0,
        "translation_gain": 1.0,
        "rotation_gain": 1.0,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R006 residual-correction contract differs: {key}")
