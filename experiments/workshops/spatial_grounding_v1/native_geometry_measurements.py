"""Native measurement helpers for SGW engineering evidence, never policy input."""

from __future__ import annotations

import hashlib
import math
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


def robot_snapshot(
    scene: Any, origin_world_xyz_m: Any, *, asset_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    if isinstance(asset_identity, Mapping) and asset_identity.get("available") is True:
        return {**value, "asset_usd": dict(asset_identity)}
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


def collision_geometry_local_bounds(stage: Any, *, articulation_path: str | None, body_names: list[str]) -> dict[str, Any]:
    """Measure USD collision Gprim bounds in their nearest rigid-body frame."""

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        return {"available": False, "reason": "pxr USD collision APIs are unavailable"}
    if not isinstance(articulation_path, str) or not articulation_path or len(set(body_names)) != len(body_names):
        return {"available": False, "reason": "native articulation prim path/body-name mapping is unavailable"}
    root = stage.GetPrimAtPath(articulation_path)
    if not root or not root.IsValid():
        return {"available": False, "reason": "native articulation prim path does not resolve in the stage"}
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    transforms = UsdGeom.XformCache()
    rows = []
    known = set(body_names)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim
        while body and body.GetName() not in known:
            body = body.GetParent()
        if not body or body.GetName() not in known:
            return {"available": False, "reason": f"collision geometry has ambiguous/missing native body mapping: {prim.GetPath()}"}
        box = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        relative, resets_stack = transforms.ComputeRelativeTransform(prim, body)
        if resets_stack:
            continue
        minimum, maximum = box.GetMin(), box.GetMax()
        rows.append({
            "geometry_prim": str(prim.GetPath()), "body_name": str(body.GetName()), "rigid_body_prim": str(body.GetPath()),
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


def project_collision_aabbs(inventory: Mapping[str, Any], body_frames: Mapping[str, Any]) -> dict[str, Any]:
    """Project local collision boxes through Gf local and native body transforms."""

    if inventory.get("available") is not True or not isinstance(inventory.get("rows"), list):
        return {"available": False, "reason": "collision inventory is unavailable"}
    bodies = body_frames.get("bodies") if isinstance(body_frames, Mapping) else None
    if not isinstance(bodies, Mapping):
        return {"available": False, "reason": "native body-frame map is unavailable"}
    per_body: dict[str, list[list[float]]] = {}
    for row in inventory["rows"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("body_name"), str):
            return {"available": False, "reason": "collision inventory row is malformed"}
        frame = bodies.get(row["body_name"])
        if not isinstance(frame, Mapping):
            return {"available": False, "reason": f"native body frame missing for {row['body_name']}"}
        try:
            local_minimum = _finite3(row["local_min_xyz_m"])
            local_maximum = _finite3(row["local_max_xyz_m"])
            matrix = row["geometry_to_body_matrix_gf"]
            body_position = _finite3(frame["position_world_xyz_m"])
            body_quaternion = _finite4(frame["quaternion_world_wxyz"])
        except (KeyError, TypeError, ValueError):
            return {"available": False, "reason": f"collision row/body frame malformed for {row['body_name']}"}
        if any(low > high for low, high in zip(local_minimum, local_maximum, strict=True)):
            return {"available": False, "reason": f"collision local bounds are inverted for {row['body_name']}"}
        corners = []
        for x in (local_minimum[0], local_maximum[0]):
            for y in (local_minimum[1], local_maximum[1]):
                for z in (local_minimum[2], local_maximum[2]):
                    body_local = _gf_row_transform((x, y, z), matrix)
                    rotated = _rotate_wxyz(body_quaternion, body_local)
                    corners.append([body_position[index] + rotated[index] for index in range(3)])
        per_body.setdefault(row["body_name"], []).extend(corners)
    return {
        "available": True,
        "body_world_aabbs": {
            name: {"minimum_xyz_m": [min(point[index] for point in points) for index in range(3)],
                   "maximum_xyz_m": [max(point[index] for point in points) for index in range(3)]}
            for name, points in per_body.items()
        },
        "caveat": "Conservative world AABB separation is not a physical collision measurement.",
    }


def clearance_to_objects(robot_aabbs: Mapping[str, Any], objects: Mapping[str, Any]) -> dict[str, Any]:
    """Compare projected robot AABBs to measured object AABBs without a pass threshold."""

    if robot_aabbs.get("available") is not True:
        return {"available": False, "reason": robot_aabbs.get("reason", "robot AABBs unavailable")}
    rows = {}
    for body, box in robot_aabbs["body_world_aabbs"].items():
        for name, object_box in objects.items():
            rows[f"{body}__{name}"] = aabb_separation(box, object_box)
    return {
        "available": True, "separations": rows,
        "caveat": "AABB separation is a conservative lower-bound geometry observation, not collision/safety closure.",
    }


def _gf_row_transform(point: tuple[float, float, float], matrix: Any) -> list[float]:
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 4 or any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in matrix):
        raise ValueError("Gf matrix must be a finite 4x4 row-vector transform")
    values = [[float(value) for value in row] for row in matrix]
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError("Gf matrix must be finite")
    return [sum(point[row] * values[row][column] for row in range(3)) + values[3][column] for column in range(3)]


def _rotate_wxyz(quaternion: list[float], point: list[float]) -> list[float]:
    w, x, y, z = quaternion
    vx, vy, vz = point
    return [
        (1 - 2 * (y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
        2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
        2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz,
    ]


def _finite3(value: Any) -> list[float]:
    values = [float(item) for item in value]
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError("expected finite vector3")
    return values


def _finite4(value: Any) -> list[float]:
    values = [float(item) for item in value]
    if len(values) != 4 or not all(math.isfinite(item) for item in values) or not math.isclose(sum(item * item for item in values), 1.0, abs_tol=1e-3):
        raise ValueError("expected normalized finite quaternion")
    return values
