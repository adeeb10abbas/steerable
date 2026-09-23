"""Pinned RoboLab geometry measurements used for SGW physical-center scoring."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def articulation_body_frames(data: Any) -> dict[str, Any]:
    names = list(data.body_names)
    positions = np.asarray(data.body_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
    rotations = np.asarray(data.body_quat_w[0].detach().cpu().numpy(), dtype=np.float64)
    if (not names or len(set(names)) != len(names) or positions.shape != (len(names), 3)
            or rotations.shape != (len(names), 4) or not np.isfinite(positions).all()
            or not np.isfinite(rotations).all()):
        raise ValueError("robot body frame inventory is malformed")
    if not np.allclose(np.linalg.norm(rotations, axis=1), 1, atol=1e-3):
        raise ValueError("robot body quaternions are not normalized")
    return {
        "source": "IsaacLab ArticulationData body_pos_w/body_quat_w; world frame",
        "bodies": {
            name: {"position_world_xyz_m": position.tolist(), "quaternion_world_wxyz": rotation.tolist()}
            for name, position, rotation in zip(names, positions, rotations, strict=True)
        },
        "claim_boundary": "body frame origins, not inferred fingertip contact centers",
    }


def geometric_center_state(
    *,
    com_position_env_local_xyz_m: Sequence[float],
    geometric_center_env_local_xyz_m: Sequence[float],
    com_velocity_world: Sequence[float],
) -> tuple[tuple[float, float, float], float, float]:
    """Return the geometric-center position and rigid-body center speed norms.

    RoboLab's ``get_bbox`` returns the transformed cached-geometry centroid in
    env-local coordinates. IsaacLab compatibility ``root_vel_w`` is a center
    of mass (COM) velocity, so transport its linear component from COM to the
    geometric center before taking protocol norms.
    """

    com = _vector3(com_position_env_local_xyz_m, "COM position")
    center = _vector3(geometric_center_env_local_xyz_m, "geometric center")
    velocity = np.asarray(com_velocity_world, dtype=np.float64)
    if velocity.shape != (6,) or not np.isfinite(velocity).all():
        raise ValueError("COM velocity must be a finite six-vector")
    angular = velocity[3:]
    center_linear = velocity[:3] + np.cross(angular, center - com)
    return (
        tuple(float(value) for value in center),
        float(np.linalg.norm(center_linear)),
        float(np.linalg.norm(angular)),
    )


def _vector3(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite three-vector")
    return vector
