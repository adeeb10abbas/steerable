"""Model-blind live-reset adapter for every V3-E004 DROID bridge.

The model-specific runners retain their existing policy clients.  Immediately
after the registered settle window, they pass the live RoboLab environment,
raw proprioception, and rendered camera evidence to this adapter.  The adapter
extracts the actual object/arm state, compiles the existing E004 camera/layout
gate, and only then returns a request-zero authorization fingerprint.

Camera extraction remains model-independent but simulator-version-specific;
callers supply rows obtained from the live Isaac sensors.  Missing calibrated
geometry or rendered visibility fails closed in ``compile_live_gate``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .layout_contract import E004Candidate, PoseSE2, canonical_json_bytes, load_candidate, wrap_angle
from .live_gate import compile_live_gate
from .runtime_contract import E004Cell, E004RuntimeBundle, RuntimeContractError, canonical_json_sha256, sha256_file


SNAPSHOT_SCHEMA = "vla-wam-shared-v3e004-live-scene-snapshot-v1"
BOUND_GATE_SCHEMA = "vla-wam-shared-v3e004-bound-live-scene-gate-v1"
LINEAR_SPEED_TOLERANCE_M_S = 0.02
ANGULAR_SPEED_TOLERANCE_RAD_S = 0.2
SETTLE_STEPS = 60
STABILITY_WINDOW_STEPS = 15
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _host_list(value: Any, expected: int, label: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(isinstance(value, (list, tuple)) and len(value) == expected, f"{label} must contain {expected} values")
    output = [float(item) for item in value]
    _require(all(math.isfinite(item) for item in output), f"{label} must be finite")
    return output


def _quaternion_normalized(value: Any, label: str) -> tuple[float, float, float, float]:
    raw = _host_list(value, 4, label)
    norm = math.sqrt(sum(item * item for item in raw))
    _require(norm > 0.0 and math.isfinite(norm), f"{label} is invalid")
    return tuple(item / norm for item in raw)  # type: ignore[return-value]


def _quat_inverse_rotate_wxyz(q_value: Any, vector: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = _quaternion_normalized(q_value, "robot root quaternion")
    vx, vy, vz = (float(item) for item in vector)
    # q^-1 * (0,v) * q, expanded to avoid a simulator math dependency.
    ix = w * vx - y * vz + z * vy
    iy = w * vy - z * vx + x * vz
    iz = w * vz - x * vy + y * vx
    iw = x * vx + y * vy + z * vz
    return (
        ix * w + iw * x + iy * z - iz * y,
        iy * w + iw * y + iz * x - ix * z,
        iz * w + iw * z + ix * y - iy * x,
    )


def _yaw_wxyz(value: Any, label: str) -> float:
    w, x, y, z = _quaternion_normalized(value, label)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _root_pose(scene_object: Any, label: str) -> tuple[list[float], Any]:
    data = getattr(scene_object, "data", None)
    _require(data is not None, f"{label} has no simulator data")
    return (
        _host_list(data.root_pos_w[0], 3, f"{label} root position"),
        data.root_quat_w[0],
    )


def extract_realised_object_poses(
    env: Any,
    *,
    candidate: E004Candidate,
    symmetry_level_s: float,
    scene_object_mapping: Mapping[str, str],
) -> dict[str, PoseSE2]:
    """Extract settled object SE(2) poses in the registered robot-base frame."""

    expected = candidate.layout(symmetry_level_s)
    _require(set(scene_object_mapping) >= set(expected), "scene-object mapping is incomplete for this level")
    _require(len({scene_object_mapping[name] for name in expected}) == len(expected), "scene-object mapping aliases physical prims")
    robot_pos, robot_quat = _root_pose(env.scene["robot"], "robot")
    robot_yaw = _yaw_wxyz(robot_quat, "robot root quaternion")
    output: dict[str, PoseSE2] = {}
    for logical_name, target in expected.items():
        physical_name = str(scene_object_mapping[logical_name])
        world_pos, world_quat = _root_pose(env.scene[physical_name], physical_name)
        delta = [world_pos[index] - robot_pos[index] for index in range(3)]
        robot_xyz = _quat_inverse_rotate_wxyz(robot_quat, delta)
        output[logical_name] = PoseSE2(
            x_m=robot_xyz[0],
            y_m=robot_xyz[1],
            z_m=robot_xyz[2],
            yaw_rad=wrap_angle(_yaw_wxyz(world_quat, f"{physical_name} root quaternion") - robot_yaw),
            asset_identity=target.asset_identity,
        )
    return output


def extract_arm_reset_pose(observation: Mapping[str, Any]) -> dict[str, Any]:
    proprio = observation.get("proprio_obs")
    _require(isinstance(proprio, Mapping), "live observation has no proprio_obs")
    joints = _host_list(proprio.get("arm_joint_pos"), 7, "arm_joint_pos")
    gripper_raw = proprio.get("gripper_pos")
    if hasattr(gripper_raw, "detach"):
        gripper_raw = gripper_raw.detach().cpu().tolist()
    if hasattr(gripper_raw, "tolist"):
        gripper_raw = gripper_raw.tolist()
    while isinstance(gripper_raw, list) and len(gripper_raw) == 1 and isinstance(gripper_raw[0], list):
        gripper_raw = gripper_raw[0]
    _require(isinstance(gripper_raw, list) and gripper_raw, "gripper_pos is missing")
    gripper = [float(item) for item in gripper_raw]
    _require(all(math.isfinite(item) for item in gripper), "gripper_pos must be finite")
    source = {"arm_joint_positions_rad": joints, "gripper_position": gripper}
    return {**source, "measurement_source_sha256": canonical_json_sha256(source)}


def camera_geometry_sha256(row: Mapping[str, Any]) -> str:
    bounds = row.get("reference_bounds_world", {})
    value = {
        "camera_center_world_m": row.get("camera_center_world_m"),
        "camera_quaternion_world_wxyz_ros": row.get("camera_quaternion_world_wxyz_ros"),
        "target_center_world_m": row.get("target_center_world_m"),
        "intrinsic_matrix_3x3": row.get("intrinsic_matrix_3x3"),
        "image_size_wh": row.get("image_size_wh"),
        "reference_bounds_world": bounds,
    }
    return canonical_json_sha256(value)


def bind_camera_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one live camera row after the bridge renders its segmentation."""

    output = dict(row)
    output["camera_geometry_source_sha256"] = camera_geometry_sha256(output)
    pixels = output.get("target_instance_visible_pixels")
    if pixels is not None:
        _require(type(pixels) is int and pixels >= 0, "rendered target pixel count is invalid")
        _require(_SHA256.fullmatch(str(output.get("segmentation_source_sha256", ""))) is not None, "segmentation source digest is required")
    else:
        # The checked-in occlusion contract permits calibrated projection as
        # a fail-closed visibility fallback when this RoboLab camera preset
        # does not expose instance segmentation.  It never treats a missing
        # segmentation stream as zero pixels.
        _require(output.get("target_projected_pixel_uv") is not None, "camera needs segmentation or calibrated projection evidence")
        _require(output.get("segmentation_source_sha256") is None, "missing segmentation must not carry a synthetic digest")
    return output


def validate_settle_stability(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(value.get("settle_steps") == SETTLE_STEPS, "E004 settle step count changed")
    _require(value.get("stability_window_steps") == STABILITY_WINDOW_STEPS, "E004 stability-window count changed")
    maxima = value.get("maxima_by_object")
    _require(isinstance(maxima, Mapping) and maxima, "settle stability maxima are missing")
    normalized: dict[str, dict[str, float]] = {}
    for name, row in maxima.items():
        _require(isinstance(row, Mapping), f"stability row is invalid for {name}")
        linear = float(row.get("linear_speed_m_s"))
        angular = float(row.get("angular_speed_rad_s"))
        _require(
            math.isfinite(linear) and 0.0 <= linear <= LINEAR_SPEED_TOLERANCE_M_S,
            f"{name} failed linear stability: observed={linear:.9g} m/s, "
            f"limit={LINEAR_SPEED_TOLERANCE_M_S:.9g} m/s",
        )
        _require(
            math.isfinite(angular) and 0.0 <= angular <= ANGULAR_SPEED_TOLERANCE_RAD_S,
            f"{name} failed angular stability: observed={angular:.9g} rad/s, "
            f"limit={ANGULAR_SPEED_TOLERANCE_RAD_S:.9g} rad/s",
        )
        normalized[str(name)] = {"linear_speed_m_s": linear, "angular_speed_rad_s": angular}
    return {
        "settle_steps": SETTLE_STEPS,
        "stability_window_steps": STABILITY_WINDOW_STEPS,
        "maxima_by_object": normalized,
        "linear_speed_tolerance_m_s": LINEAR_SPEED_TOLERANCE_M_S,
        "angular_speed_tolerance_rad_s": ANGULAR_SPEED_TOLERANCE_RAD_S,
    }


class ModelBlindLiveGateAdapter:
    """One-shot gate state machine embedded in a model-specific bridge."""

    def __init__(
        self,
        *,
        bundle: E004RuntimeBundle,
        cell: E004Cell,
        snapshot_path: Path,
        gate_path: Path,
        minimum_visible_target_pixels: int,
        orientation_tolerance_attestation: Mapping[str, Any] | None = None,
    ) -> None:
        _require(cell.row["arena"] == "droid_robolab", "live adapter is DROID-only")
        _require(cell.row["execution_mode"] == "new_behavioral_episode", "preserved evidence cannot enter a live runner")
        self.bundle = bundle
        self.cell = cell
        self.snapshot_path = Path(snapshot_path).resolve()
        self.gate_path = Path(gate_path).resolve()
        self.minimum_visible_target_pixels = minimum_visible_target_pixels
        self.orientation_tolerance_attestation = (
            dict(orientation_tolerance_attestation)
            if orientation_tolerance_attestation is not None
            else None
        )
        self._gate_sha256: str | None = None
        self._model_request_count = 0
        self._behavioral_action_count = 0

    def capture_and_compile(
        self,
        *,
        env: Any,
        observation: Mapping[str, Any],
        scene_object_mapping: Mapping[str, str],
        camera_rows: Mapping[str, Mapping[str, Any]],
        settle_stability: Mapping[str, Any],
    ) -> dict[str, Any]:
        _require(self._model_request_count == 0 and self._behavioral_action_count == 0, "live gate must precede requests and behavioral actions")
        _require(not self.snapshot_path.exists() and not self.gate_path.exists(), "refusing to overwrite retained live gate evidence")
        candidate = load_candidate(self.bundle.candidate_path, self.bundle.candidate_sha256)
        poses = extract_realised_object_poses(
            env,
            candidate=candidate,
            symmetry_level_s=self.cell.symmetry_level_s,
            scene_object_mapping=scene_object_mapping,
        )
        bound_cameras = {name: bind_camera_row(row) for name, row in camera_rows.items()}
        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA,
            "study_id": self.cell.row["study_id"],
            "amendment_id": self.cell.row["amendment_id"],
            "registered_cell_id": self.cell.cell_id,
            "registered_cell_sha256": self.cell.row_sha256,
            "registration_sha256": self.bundle.registration_sha256,
            "queue_sha256": self.bundle.queue_sha256,
            "candidate_sha256": self.bundle.candidate_sha256,
            "symmetry_level_s": self.cell.symmetry_level_s,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "realised_object_poses": {name: pose.to_json() for name, pose in sorted(poses.items())},
            "arm_reset_pose": extract_arm_reset_pose(observation),
            "cameras": bound_cameras,
            "settle_stability": validate_settle_stability(settle_stability),
            "orientation_tolerance_attestation": self.orientation_tolerance_attestation,
            "scope_caveat": "Object symmetry is relative to the robot midline; robot kinematics, reset arm pose, cameras, and embodiment are not asserted symmetric.",
        }
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_bytes(canonical_json_bytes(snapshot))
        snapshot_sha = sha256_file(self.snapshot_path)
        compiled = compile_live_gate(
            candidate_path=self.bundle.candidate_path,
            candidate_sha256=self.bundle.candidate_sha256,
            snapshot_path=self.snapshot_path,
            snapshot_sha256=snapshot_sha,
            minimum_visible_target_pixels=self.minimum_visible_target_pixels,
            realisation_orientation_tolerance_rad=(
                self.orientation_tolerance_attestation[
                    "effective_live_orientation_realisation_tolerance_rad"
                ]
                if self.orientation_tolerance_attestation is not None
                else None
            ),
            orientation_tolerance_attestation=self.orientation_tolerance_attestation,
        )
        bound = {
            "schema_version": BOUND_GATE_SCHEMA,
            "study_id": self.cell.row["study_id"],
            "amendment_id": self.cell.row["amendment_id"],
            "status": "passed_and_released_for_exact_cell_request_zero",
            "passed": True,
            "registered_cell_id": self.cell.cell_id,
            "registered_cell_sha256": self.cell.row_sha256,
            "registration_sha256": self.bundle.registration_sha256,
            "queue_sha256": self.bundle.queue_sha256,
            "candidate_sha256": self.bundle.candidate_sha256,
            "snapshot": {"path": str(self.snapshot_path), "sha256": snapshot_sha, "bytes": self.snapshot_path.stat().st_size},
            "compiled_gate": compiled,
            "orientation_tolerance_attestation": self.orientation_tolerance_attestation,
        }
        self.gate_path.parent.mkdir(parents=True, exist_ok=True)
        self.gate_path.write_bytes(canonical_json_bytes(bound))
        self._gate_sha256 = sha256_file(self.gate_path)
        return {**bound, "gate_sha256": self._gate_sha256, "gate_path": str(self.gate_path)}

    def authorize_model_request(self) -> str:
        """Return the immutable live-gate digest for a policy request."""

        _require(self._gate_sha256 is not None and self.gate_path.is_file(), "model request attempted before live gate")
        _require(sha256_file(self.gate_path) == self._gate_sha256, "live gate changed after release")
        self._model_request_count += 1
        return self._gate_sha256

    def authorize_behavioral_action(self) -> None:
        _require(self._model_request_count > 0, "behavioral action attempted before a gated model request")
        self._behavioral_action_count += 1

    @property
    def model_request_count(self) -> int:
        return self._model_request_count

    @property
    def behavioral_action_count(self) -> int:
        return self._behavioral_action_count
