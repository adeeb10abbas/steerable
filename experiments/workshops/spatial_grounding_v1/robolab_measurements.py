"""Pinned RoboLab geometry measurements used for SGW physical-center scoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def physical_center_state(
    *,
    root_position_env_local_xyz_m: Sequence[float],
    geometric_center_env_local_xyz_m: Sequence[float],
    root_velocity_world: Sequence[float],
) -> tuple[tuple[float, float, float], float, float]:
    """Return the geometric-center position and rigid-body center speed norms.

    RoboLab's ``get_pose`` returns an env-local rigid-root pose, while
    ``get_bbox`` returns the transformed cached-geometry centroid in that same
    frame.  They are not interchangeable for asymmetric assets.  The velocity
    API reports rigid-root linear/angular world-axis velocity, so transport the
    linear component to the geometric center before taking protocol norms.
    """

    root = _vector3(root_position_env_local_xyz_m, "root position")
    center = _vector3(geometric_center_env_local_xyz_m, "geometric center")
    velocity = np.asarray(root_velocity_world, dtype=np.float64)
    if velocity.shape != (6,) or not np.isfinite(velocity).all():
        raise ValueError("root velocity must be a finite six-vector")
    angular = velocity[3:]
    center_linear = velocity[:3] + np.cross(angular, center - root)
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
