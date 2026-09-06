"""Fail-closed G2 reset, camera, and task-frame evidence for V4 horizontal fixtures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.online_correction_v4.droid_reset import (
    validate_reset_attestation_payload,
)
from experiments.online_correction_v4.droid_reset_verify import (
    POSITION_TOLERANCE_M,
    verify_measured_native_dt,
    verify_neutral_horizontal_layout,
    verify_physical_reset_against_registry,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    ResetRegistry,
)
from experiments.online_correction_v4.geometry import TaskFrame

def seed_receipt_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g2-seed-receipt-v1"


def aggregate_receipt_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g2-aggregate-receipt-v1"


def axis_review_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g2-axis-review-v1"


SEED_RECEIPT_SCHEMA = seed_receipt_schema("horizontal")
AGGREGATE_RECEIPT_SCHEMA = aggregate_receipt_schema("horizontal")
AXIS_REVIEW_SCHEMA = axis_review_schema("horizontal")
REQUIRED_POLICY_CAMERAS = (
    "over_shoulder_left_camera",
    "wrist_cam",
    "over_shoulder_right_camera",
)
ROBOLAB_DROID_TASK_AXES = {
    "u_left_robot": (0.0, 1.0, 0.0),
    "u_front_robot": (-1.0, 0.0, 0.0),
    "u_up_robot": (0.0, 0.0, 1.0),
}


class G2GateError(RuntimeError):
    """Raised when model-blind G2 evidence is incomplete or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_vector(value: Any, *, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise G2GateError(f"{label} must be a {length}-vector")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise G2GateError(f"{label} must contain finite values")
    return result


def quaternion_rotate_wxyz(
    quaternion_wxyz: Iterable[float],
    vector_xyz: Iterable[float],
) -> tuple[float, float, float]:
    qw, qx, qy, qz = _numeric_vector(
        tuple(quaternion_wxyz), length=4, label="quaternion"
    )
    vx, vy, vz = _numeric_vector(tuple(vector_xyz), length=3, label="vector")
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        raise G2GateError("robot quaternion has zero norm")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def task_frame_evidence(physical_reset: Mapping[str, Any]) -> dict[str, Any]:
    robot_quaternion = _numeric_vector(
        physical_reset.get("robot_quaternion_world_wxyz"),
        length=4,
        label="robot_quaternion_world_wxyz",
    )
    robot_origin = _numeric_vector(
        physical_reset.get("robot_position_world_xyz_m"),
        length=3,
        label="robot_position_world_xyz_m",
    )
    u_left = quaternion_rotate_wxyz(
        robot_quaternion, ROBOLAB_DROID_TASK_AXES["u_left_robot"]
    )
    u_front = quaternion_rotate_wxyz(
        robot_quaternion, ROBOLAB_DROID_TASK_AXES["u_front_robot"]
    )
    u_up = quaternion_rotate_wxyz(
        robot_quaternion, ROBOLAB_DROID_TASK_AXES["u_up_robot"]
    )
    frame = TaskFrame(
        u_left=u_left,
        u_front=u_front,
        u_up=u_up,
        origin=robot_origin,
    )
    return {
        "convention": "DROID robot-base +Y left, -X front/toward robot, +Z up",
        "u_left_world": list(frame.u_left),
        "u_front_world": list(frame.u_front),
        "u_up_world": list(frame.u_up),
        "origin_world": list(frame.origin),
        "right_handed": True,
        "visual_axis_review_status": "pending_separate_rendered_axis_review",
    }


def camera_view_evidence(raw_observation: Mapping[str, Any]) -> dict[str, Any]:
    image_obs = raw_observation.get("image_obs")
    if not isinstance(image_obs, Mapping):
        raise G2GateError("RoboLab observation lacks image_obs")
    result: dict[str, Any] = {}
    for camera in REQUIRED_POLICY_CAMERAS:
        if camera not in image_obs:
            raise G2GateError(f"RoboLab observation lacks required camera {camera}")
        value = image_obs[camera]
        try:
            import numpy as np

            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 3 or array.shape[-1] != 3:
                raise G2GateError(
                    f"camera {camera} must be one HxWx3 image, got {array.shape}"
                )
            contiguous = np.ascontiguousarray(array)
            if contiguous.dtype != np.uint8:
                raise G2GateError(
                    f"camera {camera} must expose uint8 pixels, got {contiguous.dtype}"
                )
            pixel_range = float(np.ptp(contiguous))
            if not math.isfinite(pixel_range) or pixel_range <= 0:
                raise G2GateError(f"camera {camera} is blank")
            result[camera] = {
                "shape": [int(item) for item in contiguous.shape],
                "dtype": str(contiguous.dtype),
                "pixel_range": pixel_range,
                "raw_array_sha256": sha256_bytes(contiguous.tobytes()),
                "nonblank": True,
                "policy_input_camera": True,
            }
        except ImportError as exc:
            raise G2GateError("numpy is required for camera evidence") from exc
    return result


def project_world_target_to_pixel(
    *,
    camera_center_world_m: Iterable[float],
    camera_quaternion_world_wxyz_ros: Iterable[float],
    target_world_m: Iterable[float],
    intrinsic_matrix_3x3: Iterable[Iterable[float]],
) -> tuple[float, float]:
    """Project a world point with Isaac's ROS camera convention."""
    center = _numeric_vector(
        tuple(camera_center_world_m), length=3, label="camera center"
    )
    quaternion = _numeric_vector(
        tuple(camera_quaternion_world_wxyz_ros),
        length=4,
        label="camera quaternion",
    )
    target = _numeric_vector(tuple(target_world_m), length=3, label="target")
    matrix = tuple(tuple(row) for row in intrinsic_matrix_3x3)
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise G2GateError("camera intrinsic matrix must be 3x3")
    intrinsic = tuple(
        tuple(float(value) for value in row)
        for row in matrix
    )
    if any(not math.isfinite(value) for row in intrinsic for value in row):
        raise G2GateError("camera intrinsic matrix must contain finite values")
    qw, qx, qy, qz = quaternion
    local = quaternion_rotate_wxyz(
        (qw, -qx, -qy, -qz),
        tuple(target[index] - center[index] for index in range(3)),
    )
    if local[2] <= 1e-9:
        raise G2GateError("axis point lies behind or on the camera plane")
    return (
        intrinsic[0][0] * local[0] / local[2]
        + intrinsic[0][1] * local[1] / local[2]
        + intrinsic[0][2],
        intrinsic[1][0] * local[0] / local[2]
        + intrinsic[1][1] * local[1] / local[2]
        + intrinsic[1][2],
    )


def axis_projection_evidence(
    *,
    physical_reset: Mapping[str, Any],
    camera_geometry: Mapping[str, Any],
    camera_views: Mapping[str, Any],
    arrow_length_m: float = 0.12,
    reference_object: str = "bowl",
    fixture_id: str = "horizontal",
) -> dict[str, Any]:
    """Project task-frame axes into the exact policy-camera images."""
    if not math.isfinite(arrow_length_m) or arrow_length_m <= 0:
        raise G2GateError("axis arrow length must be positive")
    task_frame = task_frame_evidence(physical_reset)
    objects = physical_reset.get("objects")
    if not isinstance(objects, Mapping):
        raise G2GateError("physical reset lacks objects for axis projection")
    reference = objects.get(reference_object)
    if not isinstance(reference, Mapping):
        raise G2GateError(
            f"physical reset lacks {reference_object} for axis projection"
        )
    reference_world = _numeric_vector(
        reference.get("position_world_xyz_m"),
        length=3,
        label=f"{reference_object} world position",
    )
    origin = (reference_world[0], reference_world[1], reference_world[2] + 0.10)
    endpoints = {
        "left": tuple(
            origin[index] + arrow_length_m * task_frame["u_left_world"][index]
            for index in range(3)
        ),
        "front": tuple(
            origin[index] + arrow_length_m * task_frame["u_front_world"][index]
            for index in range(3)
        ),
        "up": tuple(
            origin[index] + arrow_length_m * task_frame["u_up_world"][index]
            for index in range(3)
        ),
    }
    if set(camera_geometry) != set(REQUIRED_POLICY_CAMERAS):
        raise G2GateError("camera geometry inventory differs from policy cameras")
    rows: dict[str, Any] = {}
    visible_axis_names: set[str] = set()
    for name in REQUIRED_POLICY_CAMERAS:
        geometry = camera_geometry.get(name)
        view = camera_views.get(name)
        if not isinstance(geometry, Mapping) or not isinstance(view, Mapping):
            raise G2GateError(f"camera projection inputs are incomplete for {name}")
        image_size = geometry.get("image_size_wh")
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise G2GateError(f"camera {name} image size is invalid")
        width, height = (int(image_size[0]), int(image_size[1]))
        shape = view.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or [height, width] != [int(shape[0]), int(shape[1])]
        ):
            raise G2GateError(f"camera {name} geometry and RGB dimensions differ")

        def _project(point: Iterable[float]) -> tuple[float, float]:
            return project_world_target_to_pixel(
                camera_center_world_m=geometry.get("camera_center_world_m", ()),
                camera_quaternion_world_wxyz_ros=geometry.get(
                    "camera_quaternion_world_wxyz_ros", ()
                ),
                target_world_m=point,
                intrinsic_matrix_3x3=geometry.get("intrinsic_matrix_3x3", ()),
            )

        projected_origin = _project(origin)
        projected_endpoints = {
            axis: _project(point) for axis, point in endpoints.items()
        }
        in_frame = {
            axis: (
                0 <= projected_origin[0] < width
                and 0 <= projected_origin[1] < height
                and 0 <= point[0] < width
                and 0 <= point[1] < height
            )
            for axis, point in projected_endpoints.items()
        }
        visible_axis_names.update(axis for axis, visible in in_frame.items() if visible)
        rows[name] = {
            **dict(geometry),
            "rgb_raw_array_sha256": view.get("raw_array_sha256"),
            "axis_origin_world_m": list(origin),
            "axis_origin_pixel_uv": list(projected_origin),
            "axis_endpoint_world_m": {
                axis: list(point) for axis, point in endpoints.items()
            },
            "axis_endpoint_pixel_uv": {
                axis: list(point) for axis, point in projected_endpoints.items()
            },
            "axis_fully_in_frame": in_frame,
        }
    if visible_axis_names != {"left", "front", "up"}:
        raise G2GateError(
            "policy camera projections do not jointly show left/front/up axes"
        )
    return {
        "schema_version": (
            f"v4-{fixture_id.replace('_', '-')}-g2-axis-projection-v1"
        ),
        "arrow_length_m": arrow_length_m,
        "axis_origin_rule": f"0.10m above live {reference_object} center",
        "axes_visible_across_policy_cameras": sorted(visible_axis_names),
        "camera_rows": rows,
    }


def compile_seed_receipt(
    *,
    env_seed: int,
    episode_id: str,
    registry: ResetRegistry,
    reset_attestation: Mapping[str, Any],
    physical_reset: Mapping[str, Any],
    camera_views: Mapping[str, Any],
    camera_geometry: Mapping[str, Any],
    expected_native_control_dt_s: float,
    runtime_identity: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    fixture_id: str = "horizontal",
) -> dict[str, Any]:
    validate_reset_attestation_payload(reset_attestation, episode_id=episode_id)
    position_errors = verify_physical_reset_against_registry(
        physical_reset,
        registry=registry,
        env_seed=env_seed,
    )
    if fixture_id == "horizontal":
        verify_neutral_horizontal_layout(physical_reset)
    measured_dt = float(physical_reset.get("measured_native_control_dt_s", 0.0))
    verify_measured_native_dt(
        measured_s=measured_dt,
        locked_s=expected_native_control_dt_s,
    )
    if set(camera_views) != set(REQUIRED_POLICY_CAMERAS):
        raise G2GateError("camera evidence inventory differs from the frozen policy cameras")
    for camera, evidence in camera_views.items():
        if not isinstance(evidence, Mapping) or evidence.get("nonblank") is not True:
            raise G2GateError(f"camera evidence is incomplete for {camera}")
        if evidence.get("policy_input_camera") is not True:
            raise G2GateError(f"camera {camera} is not attested as a policy-input camera")
    if reset_attestation.get("model_request_count_before_attestation") != 0:
        raise G2GateError("reset attestation records model requests")
    image_artifacts = artifacts.get("policy_camera_images")
    if not isinstance(image_artifacts, Mapping):
        raise G2GateError("policy camera artifacts are missing")
    for camera, evidence in camera_views.items():
        artifact = image_artifacts.get(camera)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("decoded_raw_array_sha256")
            != evidence.get("raw_array_sha256")
        ):
            raise G2GateError(
                f"camera artifact pixels differ from live evidence for {camera}"
            )
    overlay_artifacts = artifacts.get("axis_overlay_images")
    if (
        not isinstance(overlay_artifacts, Mapping)
        or not isinstance(overlay_artifacts.get("montage"), Mapping)
    ):
        raise G2GateError("axis overlay montage artifact is missing")

    task_frame = task_frame_evidence(physical_reset)
    axis_projection = axis_projection_evidence(
        physical_reset=physical_reset,
        camera_geometry=camera_geometry,
        camera_views=camera_views,
        reference_object=registry.object_roles["reference"].scene_object,
        fixture_id=fixture_id,
    )
    return {
        "schema_version": seed_receipt_schema(fixture_id),
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "status": "passed_reset_camera_and_numeric_frame_pending_axis_visual_review",
        "passed_reset_and_camera": True,
        "g2_complete": False,
        "environment_seed": env_seed,
        "registered_episode_id": episode_id,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "reset_registry_sha256": registry.registry_sha256,
        "registry_position_tolerance_m": POSITION_TOLERANCE_M,
        "position_max_error_m_by_object": position_errors,
        "native_control_dt_s": measured_dt,
        "reset_attestation": dict(reset_attestation),
        "physical_reset": dict(physical_reset),
        "task_frame": task_frame,
        "policy_input_cameras": dict(camera_views),
        "camera_geometry": dict(camera_geometry),
        "axis_projection": axis_projection,
        "runtime_identity": dict(runtime_identity),
        "artifacts": dict(artifacts),
        "release_boundary": (
            "This per-seed zero-model-request receipt does not release G2 or policy "
            "inference. G2 requires all registered seeds plus a separate rendered "
            "left/front/up axis review."
        ),
    }


def compile_aggregate_receipt(
    *,
    expected_env_seeds: Iterable[int],
    seed_receipts: Iterable[Mapping[str, Any]],
    axis_review: Mapping[str, Any] | None,
    fixture_id: str = "horizontal",
) -> dict[str, Any]:
    expected = tuple(sorted(set(int(seed) for seed in expected_env_seeds)))
    receipts = list(seed_receipts)
    observed: dict[int, Mapping[str, Any]] = {}
    for receipt in receipts:
        if receipt.get("schema_version") != seed_receipt_schema(fixture_id):
            raise G2GateError("aggregate contains a seed receipt with the wrong schema")
        seed = receipt.get("environment_seed")
        if type(seed) is not int or seed in observed:
            raise G2GateError("aggregate seed receipts contain a missing or duplicate seed")
        observed[seed] = receipt
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    failed = sorted(
        seed
        for seed, receipt in observed.items()
        if receipt.get("passed_reset_and_camera") is not True
        or receipt.get("model_request_count") != 0
        or receipt.get("behavioral_episode_count") != 0
    )
    axis_passed = False
    if isinstance(axis_review, Mapping):
        if axis_review.get("schema_version") != axis_review_schema(fixture_id):
            raise G2GateError("axis review schema differs")
        if axis_review.get("campaign_id") != "online_correction_v4":
            raise G2GateError("axis review campaign differs")
        if axis_review.get("fixture_id") != fixture_id:
            raise G2GateError("axis review fixture differs")
        if axis_review.get("model_request_count") != 0:
            raise G2GateError("axis review records model requests")
        if axis_review.get("behavioral_episode_count") != 0:
            raise G2GateError("axis review records behavioral episodes")
        reviewer = axis_review.get("reviewer_identity")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise G2GateError("axis review lacks reviewer identity")
        reviewed_at = axis_review.get("reviewed_at_utc")
        if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
            raise G2GateError("axis review lacks UTC review time")
        source = axis_review.get("source_axis_overlay")
        if (
            not isinstance(source, Mapping)
            or not isinstance(source.get("sha256"), str)
            or len(source["sha256"]) != 64
            or type(source.get("bytes")) is not int
            or source["bytes"] <= 0
        ):
            raise G2GateError("axis review lacks source overlay identity")
        assertions = axis_review.get("assertions")
        required_assertions = {
            "left_axis_matches_fixed_robot_viewpoint",
            "front_axis_points_toward_robot",
            "up_axis_opposes_gravity",
            "labels_and_arrow_origins_visible",
        }
        if not isinstance(assertions, Mapping) or any(
            assertions.get(key) is not True for key in required_assertions
        ):
            raise G2GateError("axis review assertions are incomplete")
        axis_passed = bool(
            axis_review.get("passed") is True
            and axis_review.get("rendered_left_front_up") is True
        )
    passed = not missing and not unexpected and not failed and axis_passed
    return {
        "schema_version": aggregate_receipt_schema(fixture_id),
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "status": "passed" if passed else "blocked_incomplete",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "expected_seed_count": len(expected),
        "observed_seed_count": len(observed),
        "missing_env_seeds": missing,
        "unexpected_env_seeds": unexpected,
        "failed_env_seeds": failed,
        "axis_review_passed": axis_passed,
        "axis_review": dict(axis_review) if isinstance(axis_review, Mapping) else None,
        "seed_receipt_sha256_by_env_seed": {
            str(seed): sha256_bytes(canonical_json_bytes(dict(receipt)))
            for seed, receipt in sorted(observed.items())
        },
        "authorizes_behavioral_inference": False,
    }
