from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from tools.validate_v3e006_r012_postexecution import (
    frozen,
    validate_live_tensor_geometry,
)


def _fixture() -> tuple[dict, dict, np.ndarray]:
    origin = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    static = {
        "canonical_sha256": "a" * 64,
        "collision_corners_body_m": [
            [x, y, z]
            for x in (-0.01, 0.01)
            for y in (-0.02, 0.02)
            for z in (-0.03, 0.03)
        ],
        "center_body_m": [0.0, 0.0, 0.0],
    }
    position = np.asarray([1.1, -1.8, 3.3], dtype=np.float64)
    reconstructed = frozen.reconstruct_collision_bounds_env_local(
        body_position_env_local=position - origin,
        body_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        collision_corners_body=static["collision_corners_body_m"],
        collision_center_body=static["center_body_m"],
    )
    value = {
        "live_tensor_pose": {
            "position_tensor_world_m": position.tolist(),
            "scene_env_origin_world_m": origin.tolist(),
            "position_env_local_m": (position - origin).tolist(),
            "quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "static_body_local_geometry_sha256": static["canonical_sha256"],
        "reconstructed_bounds_env_local": reconstructed,
    }
    return value, static, origin


def test_supplied_origin_is_checked_then_frozen_math_runs() -> None:
    value, static, origin = _fixture()
    validate_live_tensor_geometry(value, static, origin=origin, label="fixture")


def test_origin_mutation_is_rejected() -> None:
    value, static, origin = _fixture()
    changed = deepcopy(origin)
    changed[0] += 1.0
    with pytest.raises(frozen.ValidationError, match="call-site/embedded origin differs"):
        validate_live_tensor_geometry(value, static, origin=changed, label="fixture")
