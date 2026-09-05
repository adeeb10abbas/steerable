"""Reset attestation verification against the hash-pinned horizontal reset registry."""

from __future__ import annotations

import math
from typing import Any, Mapping

from experiments.online_correction_v4.droid_reset import ResetAttestationError
from experiments.online_correction_v4.droid_task_files.reset_registry import ResetRegistry, load_reset_registry


NEUTRAL_COSINE_THRESHOLD = math.cos(math.radians(45.0))
POSITION_TOLERANCE_M = 0.003
NATIVE_DT_TOLERANCE_S = 1e-6


def load_bound_reset_registry(*, registry_path: str, registry_sha256: str) -> ResetRegistry:
    return load_reset_registry(registry_path=registry_path, registry_sha256=registry_sha256)


def verify_physical_reset_against_registry(
    physical: Mapping[str, Any],
    *,
    registry: ResetRegistry,
    env_seed: int,
) -> dict[str, float]:
    if env_seed not in registry.positions_by_env_seed:
        raise ResetAttestationError(f"env_seed {env_seed} is absent from reset registry")
    expected = registry.positions_by_env_seed[env_seed]
    errors: dict[str, float] = {}
    objects = physical.get("objects")
    if not isinstance(objects, dict):
        raise ResetAttestationError("physical reset payload lacks objects")
    robot = physical.get("robot_position_robot_xyz_m")
    if not isinstance(robot, list) or len(robot) != 3:
        raise ResetAttestationError("physical reset payload lacks robot_position_robot_xyz_m")
    for name, target in expected.items():
        entry = objects.get(name)
        if not isinstance(entry, dict):
            raise ResetAttestationError(f"physical reset payload missing object {name}")
        observed = entry.get("position_robot_xyz_m")
        if not isinstance(observed, list) or len(observed) != 3:
            raise ResetAttestationError(f"physical reset payload missing robot-frame pose for {name}")
        errors[name] = max(abs(float(a) - float(b)) for a, b in zip(observed, target))
    if max(errors.values()) > POSITION_TOLERANCE_M:
        raise ResetAttestationError(
            "physical reset positions exceed registry tolerance: "
            + ", ".join(f"{k}={v:.4f}m" for k, v in sorted(errors.items()))
        )
    return errors


def verify_neutral_horizontal_layout(physical: Mapping[str, Any]) -> None:
    objects = physical.get("objects")
    if not isinstance(objects, dict):
        raise ResetAttestationError("physical reset payload lacks objects for neutrality check")
    cube = objects.get("rubiks_cube", {}).get("position_robot_xyz_m")
    bowl = objects.get("bowl", {}).get("position_robot_xyz_m")
    if not isinstance(cube, list) or not isinstance(bowl, list):
        raise ResetAttestationError("neutral layout check requires rubiks_cube and bowl robot-frame poses")
    delta_y = float(cube[1]) - float(bowl[1])
    delta_x = float(cube[0]) - float(bowl[0])
    horizontal = math.hypot(delta_x, delta_y)
    if horizontal <= 1e-8:
        return
    left = delta_y / horizontal >= NEUTRAL_COSINE_THRESHOLD
    right = -delta_y / horizontal >= NEUTRAL_COSINE_THRESHOLD
    if left or right:
        raise ResetAttestationError("post-settle layout is not neutral for horizontal fixture")


def verify_measured_native_dt(*, measured_s: float, locked_s: float) -> None:
    if measured_s <= 0 or locked_s <= 0:
        raise ResetAttestationError("native control dt must be positive")
    if abs(measured_s - locked_s) > NATIVE_DT_TOLERANCE_S:
        raise ResetAttestationError(
            f"measured native dt {measured_s} != locked native dt {locked_s}"
        )
