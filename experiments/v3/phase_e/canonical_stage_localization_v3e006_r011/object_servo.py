"""Pure, uniform live object-space SE(3) servo math for V3-E006-R011."""

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
    return normalize_quaternion([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def inverse(value: Sequence[float]) -> np.ndarray:
    q = normalize_quaternion(value)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def rotate(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError("vector is malformed")
    w, xyz = float(q[0]), q[1:]
    return 2 * np.dot(xyz, v) * xyz + (w * w - np.dot(xyz, xyz)) * v + 2 * w * np.cross(xyz, v)


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


def quaternion_to_rotvec(value: Sequence[float]) -> np.ndarray:
    q = canonical_shortest_arc(value)
    norm = float(np.linalg.norm(q[1:]))
    if norm <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, float(q[0]))
    return q[1:] / norm * angle


def rotvec_to_quaternion(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector is malformed")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-15:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return normalize_quaternion(
        np.concatenate(([math.cos(angle / 2.0)], vector / angle * math.sin(angle / 2.0)))
    )


def clip_norm(value: Sequence[float], maximum: float) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(vector)) or not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("norm clip input is invalid")
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum else vector * (maximum / norm)


def object_space_servo_command(
    *,
    live_base_position: Sequence[float],
    live_base_quaternion: Sequence[float],
    live_cube_position: Sequence[float],
    live_cube_quaternion: Sequence[float],
    target_cube_position: Sequence[float],
    target_cube_quaternion: Sequence[float],
    translation_gain: float,
    rotation_gain: float,
    translation_cap_m_per_step: float,
    rotation_cap_deg_per_step: float,
) -> dict[str, Any]:
    """Compute one nonadaptive command from the live cube/base relative transform."""

    positions = [np.asarray(value, dtype=np.float64) for value in (
        live_base_position, live_cube_position, target_cube_position
    )]
    if any(value.shape != (3,) for value in positions) or not np.all(np.isfinite(np.concatenate(positions))):
        raise ValueError("servo position is malformed")
    if not all(math.isfinite(value) and value > 0.0 for value in (
        translation_gain, rotation_gain, translation_cap_m_per_step, rotation_cap_deg_per_step
    )):
        raise ValueError("servo gain/cap is invalid")
    base_p, cube_p, target_p = positions
    base_q = normalize_quaternion(live_base_quaternion)
    cube_q = normalize_quaternion(live_cube_quaternion)
    target_q = normalize_quaternion(target_cube_quaternion)
    base_inv = inverse(base_q)
    relative_p = rotate(base_inv, cube_p - base_p)
    relative_q = multiply(base_inv, cube_q)
    ideal_q = multiply(target_q, inverse(relative_q))
    ideal_p = target_p - rotate(ideal_q, relative_p)
    translation_error = ideal_p - base_p
    translation_correction = clip_norm(
        translation_gain * translation_error, translation_cap_m_per_step
    )
    rotation_error_q = canonical_shortest_arc(multiply(ideal_q, inverse(base_q)))
    rotation_error = quaternion_to_rotvec(rotation_error_q)
    rotation_correction = clip_norm(
        rotation_gain * rotation_error, math.radians(rotation_cap_deg_per_step)
    )
    command_p = base_p + translation_correction
    command_q = multiply(rotvec_to_quaternion(rotation_correction), base_q)
    reconstructed_cube_p = command_p + rotate(command_q, relative_p)
    reconstructed_cube_q = multiply(command_q, relative_q)
    return {
        "live_base_position_world_m": base_p.tolist(),
        "live_base_quaternion_world_wxyz": base_q.tolist(),
        "live_cube_position_world_m": cube_p.tolist(),
        "live_cube_quaternion_world_wxyz": cube_q.tolist(),
        "target_cube_position_world_m": target_p.tolist(),
        "target_cube_quaternion_world_wxyz": target_q.tolist(),
        "live_cube_in_base_translation_m": relative_p.tolist(),
        "live_cube_in_base_quaternion_wxyz": relative_q.tolist(),
        "ideal_base_position_world_m": ideal_p.tolist(),
        "ideal_base_quaternion_world_wxyz": ideal_q.tolist(),
        "translation_error_world_m": translation_error.tolist(),
        "rotation_error_world_rotvec_rad": rotation_error.tolist(),
        "translation_gain": float(translation_gain),
        "rotation_gain": float(rotation_gain),
        "translation_cap_m_per_step": float(translation_cap_m_per_step),
        "rotation_cap_deg_per_step": float(rotation_cap_deg_per_step),
        "applied_translation_correction_world_m": translation_correction.tolist(),
        "applied_rotation_correction_world_rotvec_rad": rotation_correction.tolist(),
        "command_base_position_world_m": command_p.tolist(),
        "command_base_quaternion_world_wxyz": command_q.tolist(),
        "command_reconstructed_cube_position_world_m": reconstructed_cube_p.tolist(),
        "command_reconstructed_cube_quaternion_world_wxyz": reconstructed_cube_q.tolist(),
    }


def validate_contract(value: Mapping[str, Any]) -> None:
    expected = {
        "algorithm_version": "uniform-live-object-space-se3-servo-v1",
        "translation_gain": 0.2,
        "rotation_gain": 0.2,
        "translation_cap_m_per_step": 0.002,
        "rotation_cap_deg_per_step": 2.0,
        "servo_steps": 360,
        "gripper_command": 1.0,
        "early_stop": False,
        "gate_read_during_servo": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R011 object-space servo contract differs: {key}")
