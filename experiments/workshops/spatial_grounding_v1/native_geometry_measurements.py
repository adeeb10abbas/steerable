"""Native measurement helpers for SGW engineering evidence, never policy input."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def vector(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def robot_snapshot(scene: Any, origin_world_xyz_m: Any) -> dict[str, Any]:
    """Return the actual articulation root, joints, frames, and USD identity."""

    from .robolab_measurements import articulation_body_frames

    robot = scene["robot"]
    data = robot.data
    origin = vector(origin_world_xyz_m)
    root = vector(data.root_pos_w[0])
    base_position = [root[index] - origin[index] for index in range(3)]
    base_quaternion = vector(data.root_quat_w[0])
    value = {
        "measurement_scope": "native_robot_state_not_policy_input",
        "articulation_root_position_env_local_xyz_m": base_position,
        "articulation_root_quaternion_world_wxyz": base_quaternion,
        # Compatibility aliases match the prior LAT workspace-capture receipt.
        "base_position_env_local_xyz_m": base_position,
        "base_quaternion_world_wxyz": base_quaternion,
        "joint_names": [str(name) for name in robot.joint_names],
        "joint_position_rad": vector(data.joint_pos[0]),
        "joint_velocity_rad_s": vector(data.joint_vel[0]),
        "body_frames": articulation_body_frames(data),
    }
    asset_path = getattr(getattr(getattr(robot, "cfg", None), "spawn", None), "usd_path", None)
    if not isinstance(asset_path, str) or not Path(asset_path).is_file():
        return {
            **value,
            "asset_usd": {"available": False, "reason": "native robot spawn USD path is unavailable"},
        }
    asset = Path(asset_path)
    return {
        **value,
        "asset_usd": {
            "available": True, "path": str(asset.resolve()), "bytes": asset.stat().st_size,
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        },
    }


def camera_extrinsics(scene: Any, camera_names: tuple[str, ...]) -> dict[str, Any]:
    """Read native sensor world extrinsics; report unavailable fields explicitly."""

    rows: dict[str, Any] = {}
    for name in camera_names:
        camera = scene[name]
        data = getattr(camera, "data", None)
        position = getattr(data, "pos_w", None)
        quaternion = getattr(data, "quat_w_world", None)
        if quaternion is None:
            quaternion = getattr(data, "quat_w", None)
        if position is None or quaternion is None:
            rows[name] = {
                "available": False,
                "reason": "native camera sensor does not expose pos_w and world quaternion",
            }
            continue
        rows[name] = {
            "available": True,
            "position_world_xyz_m": vector(position[0]),
            "quaternion_world_wxyz": vector(quaternion[0]),
        }
    return {"measurement_scope": "native_camera_extrinsics_not_policy_input", "cameras": rows}


def aabb_separation(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Conservative axis-aligned separation; overlap is not a collision assertion."""

    try:
        first_minimum = [float(value) for value in first["minimum_xyz_m"]]
        first_maximum = [float(value) for value in first["maximum_xyz_m"]]
        second_minimum = [float(value) for value in second["minimum_xyz_m"]]
        second_maximum = [float(value) for value in second["maximum_xyz_m"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("conservative AABB measurement is malformed") from error
    if any(low > high for low, high in zip(first_minimum + second_minimum, first_maximum + second_maximum, strict=True)):
        raise ValueError("conservative AABB measurement has inverted bounds")
    gaps = [
        max(second_minimum[index] - first_maximum[index], first_minimum[index] - second_maximum[index], 0.0)
        for index in range(3)
    ]
    return {
        "aabb_separation_xyz_m": gaps,
        "aabb_euclidean_separation_m": sum(value * value for value in gaps) ** .5,
        "aabb_overlap": not any(gaps),
        "caveat": "AABB overlap is conservative geometry overlap, not a measured physical collision.",
    }


def collision_geometry_local_bounds(stage: Any) -> dict[str, Any]:
    """Measure USD collision Gprim bounds in their nearest rigid-body frame."""

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        return {"available": False, "reason": "pxr USD collision APIs are unavailable"}
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    transforms = UsdGeom.XformCache()
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            continue
        box = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        relative, resets_stack = transforms.ComputeRelativeTransform(prim, body)
        if resets_stack:
            continue
        minimum, maximum = box.GetMin(), box.GetMax()
        rows.append({
            "geometry_prim": str(prim.GetPath()), "rigid_body_prim": str(body.GetPath()),
            "local_min_xyz_m": [float(value) for value in minimum],
            "local_max_xyz_m": [float(value) for value in maximum],
            "geometry_to_body_matrix_gf": [[float(value) for value in row] for row in relative],
        })
    if not rows:
        return {"available": False, "reason": "no collision Gprims with a rigid-body ancestor were measured"}
    return {
        "available": True, "measurement_scope": "conservative_usd_collision_bounds_not_collision_outcome",
        "rows": rows,
        "caveat": "Local collision bounds require current body poses for world AABBs; overlap is not collision.",
    }
