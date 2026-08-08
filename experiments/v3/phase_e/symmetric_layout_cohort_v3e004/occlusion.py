"""Camera-grounded visibility and reference-occlusion checks for V3-E004.

This module does not infer visibility from object placement labels.  Every
result is computed from captured camera geometry and, when available, live
instance-segmentation evidence.  Missing geometry or a missing camera fails
closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .layout_contract import LayoutContractError


def _vec3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(value) != 3 or not all(type(item) in (int, float) and math.isfinite(float(item)) for item in value):
        raise LayoutContractError(f"{label} must be a finite 3-vector")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class YawOrientedBox:
    center_world_m: tuple[float, float, float]
    half_extents_m: tuple[float, float, float]
    yaw_world_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_world_m", _vec3(self.center_world_m, "box center"))
        object.__setattr__(self, "half_extents_m", _vec3(self.half_extents_m, "box half extents"))
        if not all(value > 0.0 for value in self.half_extents_m):
            raise LayoutContractError("box half extents must be positive")
        if not math.isfinite(float(self.yaw_world_rad)):
            raise LayoutContractError("box yaw must be finite")


@dataclass(frozen=True)
class CameraEvidence:
    camera_name: str
    camera_center_world_m: tuple[float, float, float]
    target_center_world_m: tuple[float, float, float]
    reference_bounds_world: YawOrientedBox
    target_instance_visible_pixels: int | None
    segmentation_source_sha256: str | None
    target_projected_pixel_uv: tuple[float, float] | None = None
    image_size_wh: tuple[int, int] | None = None
    camera_geometry_source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.camera_name:
            raise LayoutContractError("camera name must be nonempty")
        object.__setattr__(self, "camera_center_world_m", _vec3(self.camera_center_world_m, "camera center"))
        object.__setattr__(self, "target_center_world_m", _vec3(self.target_center_world_m, "target center"))
        if self.target_instance_visible_pixels is not None:
            if type(self.target_instance_visible_pixels) is not int or self.target_instance_visible_pixels < 0:
                raise LayoutContractError("target visible pixel count must be a nonnegative integer")
            if not self.segmentation_source_sha256:
                raise LayoutContractError("segmentation pixel evidence requires a source digest")
        if self.target_projected_pixel_uv is not None or self.image_size_wh is not None:
            if self.target_projected_pixel_uv is None or self.image_size_wh is None:
                raise LayoutContractError("projected target evidence requires pixel and image size")
            if len(self.target_projected_pixel_uv) != 2 or not all(
                type(value) in (int, float) and math.isfinite(float(value))
                for value in self.target_projected_pixel_uv
            ):
                raise LayoutContractError("projected target pixel must be finite UV")
            if len(self.image_size_wh) != 2 or not all(type(value) is int and value > 0 for value in self.image_size_wh):
                raise LayoutContractError("image size must contain positive integer width/height")
            if not self.camera_geometry_source_sha256:
                raise LayoutContractError("projected target evidence requires a camera-geometry source digest")


def _rotate_z_minus(vector: Sequence[float], yaw: float) -> tuple[float, float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = vector
    return (c * x + s * y, -s * x + c * y, z)


def _quat_inverse_rotate_wxyz(
    quaternion_wxyz: Sequence[float], vector_world: Sequence[float]
) -> tuple[float, float, float]:
    if len(quaternion_wxyz) != 4 or not all(
        type(value) in (int, float) and math.isfinite(float(value)) for value in quaternion_wxyz
    ):
        raise LayoutContractError("camera quaternion must be a finite wxyz 4-vector")
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise LayoutContractError("camera quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = _vec3(vector_world, "world vector")
    # R(q)^T v, expanded to avoid an optional numerical dependency.
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + w * z) * vy + 2 * (x * z - w * y) * vz,
        2 * (x * y - w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + w * x) * vz,
        2 * (x * z + w * y) * vx + 2 * (y * z - w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


def project_world_target_to_pixel(
    *,
    camera_center_world_m: Sequence[float],
    camera_quaternion_world_wxyz_ros: Sequence[float],
    target_center_world_m: Sequence[float],
    intrinsic_matrix_3x3: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Project a world target using a live ROS-convention camera pose.

    ROS camera axes are +X right, +Y down, +Z forward.  This matches
    IsaacLab's ``quat_w_ros`` camera-pose output.
    """

    camera = _vec3(camera_center_world_m, "camera center")
    target = _vec3(target_center_world_m, "target center")
    if len(intrinsic_matrix_3x3) != 3 or any(len(row) != 3 for row in intrinsic_matrix_3x3):
        raise LayoutContractError("camera intrinsic matrix must be 3x3")
    intrinsic = [
        [float(value) for value in row]
        for row in intrinsic_matrix_3x3
    ]
    if not all(math.isfinite(value) for row in intrinsic for value in row):
        raise LayoutContractError("camera intrinsic matrix must be finite")
    local = _quat_inverse_rotate_wxyz(
        camera_quaternion_world_wxyz_ros,
        tuple(target[index] - camera[index] for index in range(3)),
    )
    if local[2] <= 1e-9:
        raise LayoutContractError("target lies behind or on the live camera plane")
    u = intrinsic[0][0] * local[0] / local[2] + intrinsic[0][1] * local[1] / local[2] + intrinsic[0][2]
    v = intrinsic[1][0] * local[0] / local[2] + intrinsic[1][1] * local[1] / local[2] + intrinsic[1][2]
    return (u, v)


def segment_intersects_box_before_target(
    camera_center_world_m: Sequence[float],
    target_center_world_m: Sequence[float],
    reference_bounds_world: YawOrientedBox,
    *,
    endpoint_epsilon: float = 1e-6,
) -> bool:
    """Return whether the camera-to-target segment enters the reference OBB.

    The slab intersection is evaluated in the reference box's local frame.
    Only intersections strictly before the target endpoint count.  The check
    therefore encodes the stated causal concern: the bowl lies between the
    lens and the cube.
    """

    camera = _vec3(camera_center_world_m, "camera center")
    target = _vec3(target_center_world_m, "target center")
    center = reference_bounds_world.center_world_m
    origin = _rotate_z_minus(tuple(camera[i] - center[i] for i in range(3)), reference_bounds_world.yaw_world_rad)
    end = _rotate_z_minus(tuple(target[i] - center[i] for i in range(3)), reference_bounds_world.yaw_world_rad)
    direction = tuple(end[i] - origin[i] for i in range(3))
    if math.sqrt(sum(value * value for value in direction)) <= endpoint_epsilon:
        raise LayoutContractError("camera and target centers are coincident")
    t_min, t_max = 0.0, 1.0
    for axis, half_extent in enumerate(reference_bounds_world.half_extents_m):
        if abs(direction[axis]) < 1e-15:
            if origin[axis] < -half_extent or origin[axis] > half_extent:
                return False
            continue
        t1 = (-half_extent - origin[axis]) / direction[axis]
        t2 = (half_extent - origin[axis]) / direction[axis]
        near, far = min(t1, t2), max(t1, t2)
        t_min, t_max = max(t_min, near), min(t_max, far)
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_min < 1.0 - endpoint_epsilon and t_max > endpoint_epsilon


def evaluate_camera_evidence(
    evidence: CameraEvidence,
    *,
    minimum_visible_target_pixels: int,
) -> dict[str, Any]:
    if type(minimum_visible_target_pixels) is not int or minimum_visible_target_pixels <= 0:
        raise LayoutContractError("minimum visible target pixels must be a positive registered integer")
    occluded = segment_intersects_box_before_target(
        evidence.camera_center_world_m,
        evidence.target_center_world_m,
        evidence.reference_bounds_world,
    )
    segmentation_visible = (
        evidence.target_instance_visible_pixels is not None
        and evidence.target_instance_visible_pixels >= minimum_visible_target_pixels
    )
    projection_visible = False
    if evidence.target_projected_pixel_uv is not None and evidence.image_size_wh is not None:
        u, v = evidence.target_projected_pixel_uv
        width, height = evidence.image_size_wh
        projection_visible = 0.0 <= u < width and 0.0 <= v < height
    if evidence.target_instance_visible_pixels is None and evidence.target_projected_pixel_uv is None:
        raise LayoutContractError(
            f"{evidence.camera_name}: neither segmentation nor calibrated projection visibility is available"
        )
    visible = segmentation_visible or projection_visible
    return {
        "camera_name": evidence.camera_name,
        "occlusion_check": occluded,
        "target_visible": visible,
        "target_instance_visible_pixels": evidence.target_instance_visible_pixels,
        "minimum_visible_target_pixels": minimum_visible_target_pixels,
        "segmentation_source_sha256": evidence.segmentation_source_sha256,
        "target_projected_pixel_uv": (
            list(evidence.target_projected_pixel_uv) if evidence.target_projected_pixel_uv is not None else None
        ),
        "image_size_wh": list(evidence.image_size_wh) if evidence.image_size_wh is not None else None,
        "camera_geometry_source_sha256": evidence.camera_geometry_source_sha256,
        "camera_center_world_m": list(evidence.camera_center_world_m),
        "target_center_world_m": list(evidence.target_center_world_m),
        "reference_bounds_world": {
            "center_world_m": list(evidence.reference_bounds_world.center_world_m),
            "half_extents_m": list(evidence.reference_bounds_world.half_extents_m),
            "yaw_world_rad": evidence.reference_bounds_world.yaw_world_rad,
        },
        "gate_passed": (not occluded) and visible,
        "method": "camera-to-target segment versus live reference OBB; target visibility from rendered instance pixels or calibrated camera projection",
    }


def evaluate_all_cameras(
    evidence_by_camera: Mapping[str, CameraEvidence],
    *,
    expected_cameras: Sequence[str],
    minimum_visible_target_pixels: int,
) -> dict[str, dict[str, Any]]:
    if set(evidence_by_camera) != set(expected_cameras):
        raise LayoutContractError("camera evidence set differs from registered expected cameras")
    rows = {
        name: evaluate_camera_evidence(
            evidence_by_camera[name], minimum_visible_target_pixels=minimum_visible_target_pixels
        )
        for name in expected_cameras
    }
    if not all(row["gate_passed"] for row in rows.values()):
        failed = [name for name, row in rows.items() if not row["gate_passed"]]
        raise LayoutContractError(f"visibility/occlusion gate failed: {failed}")
    return rows
