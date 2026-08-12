#!/usr/bin/env python3
"""Construct and gate V3-E006 states without a learned-model request."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--robolab-root", type=Path, required=True)
BOOTSTRAP.add_argument("--e004-candidate", type=Path, required=True)
BOOTSTRAP.add_argument("--e004-candidate-sha256", required=True)
BOOTSTRAP.add_argument("--ood-freeze", type=Path, required=True)
BOOTSTRAP.add_argument("--ood-freeze-sha256", required=True)
BOOTSTRAP.add_argument("--e004-reset-reference", type=Path, required=True)
BOOTSTRAP.add_argument("--e004-reset-reference-sha256", required=True)
BOOTSTRAP.add_argument("--runtime-bindings", type=Path, required=True)
BOOTSTRAP.add_argument("--runtime-bindings-sha256", required=True)
BOOTSTRAP.add_argument("--control-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--paired-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--pod", required=True)
BOOTSTRAP.add_argument("--pod-uid", required=True)
BOOTSTRAP.add_argument("--gpu-uuid", required=True)
BOOTSTRAP.add_argument("--expected-study-commit", required=True)
BOOTSTRAP.add_argument("--expected-robolab-commit", default="0aef241fb088ca21bb4ebd24448940ed56620d17")
BOOTSTRAP.add_argument("--construction-seed", type=int, default=13000)
BOOTSTRAP.add_argument("--steps-per-waypoint", type=int, default=80)
BOOTSTRAP.add_argument("--health-preflight-root", type=Path, required=True)
BOOTSTRAP.add_argument("--health-harness-sha256", required=True)
BOOTSTRAP.add_argument("--health-launch-sha256", required=True)
BOOTSTRAP.add_argument("--health-child-sha256", required=True)
BOOTSTRAP.add_argument("--health-runtime-log-sha256", required=True)
BOOTSTRAP.add_argument("--runtime-log", type=Path, required=True)
BOOTSTRAP.add_argument("--container-image", required=True)
BOOTSTRAP.add_argument("--container-id", required=True)
BOOTSTRAP.add_argument("--driver-version", required=True)
from robolab.eval.runner import add_common_eval_args

add_common_eval_args(BOOTSTRAP)
AppLauncher.add_app_launcher_args(BOOTSTRAP)
args, _ = BOOTSTRAP.parse_known_args()
args.enable_cameras = True
args.num_envs = 1
args.num_runs = 1

study_root = args.study_root.resolve()
robolab_root = args.robolab_root.resolve()
sys.path[:0] = [str(study_root), str(robolab_root)]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_passed_health_preflight(files: Mapping[str, tuple[Path, str]]) -> None:
    harness_path, _ = files["harness_result"]
    child_path, _ = files["preflight_result"]
    try:
        harness = json.loads(harness_path.read_text(encoding="utf-8"))
        child = json.loads(child_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("formal health preflight JSON is missing or invalid") from exc
    if harness.get("status") != "passed_generic_zero_model_health_preflight" or harness.get("passed") is not True:
        raise ValueError("formal health harness did not pass")
    if child.get("status") != "passed_generic_zero_model_cuda_vulkan_isaac_physics_render_health_preflight" or child.get("passed") is not True:
        raise ValueError("formal health child did not pass exact runtime checks")
    for label, payload in (("harness", harness), ("child", child)):
        for key in ("model_request_count", "behavioral_episode_count", "state_candidate_count"):
            if payload.get(key) != 0:
                raise ValueError(f"formal health {label} has nonzero {key}")
    binding = harness.get("child_report")
    if not isinstance(binding, Mapping) or binding != {
        "path": str(child_path.resolve()),
        "bytes": child_path.stat().st_size,
        "sha256": _sha(child_path),
    }:
        raise ValueError("formal health harness child binding differs")


health_files = {
    "harness_result": (args.health_preflight_root / "harness_result.json", args.health_harness_sha256),
    "preflight_launch": (args.health_preflight_root / "preflight_launch.json", args.health_launch_sha256),
    "preflight_result": (args.health_preflight_root / "preflight_result.json", args.health_child_sha256),
    "runtime_log": (args.health_preflight_root / "runtime.log", args.health_runtime_log_sha256),
}
for path, digest in (
    (args.e004_candidate, args.e004_candidate_sha256),
    (args.ood_freeze, args.ood_freeze_sha256),
    (args.e004_reset_reference, args.e004_reset_reference_sha256),
    (args.runtime_bindings, args.runtime_bindings_sha256),
    *(health_files.values()),
):
    if not path.is_file() or _sha(path) != digest:
        BOOTSTRAP.error(f"hash-bound input is missing or changed: {path}")
try:
    _verify_passed_health_preflight(health_files)
except ValueError as exc:
    BOOTSTRAP.error(str(exc))
for path in (args.control_scene_asset, args.paired_scene_asset):
    if not path.is_file():
        BOOTSTRAP.error(f"scene input is missing: {path}")
if args.output_dir.exists():
    BOOTSTRAP.error(f"refusing to overwrite state-construction evidence: {args.output_dir}")
if subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_study_commit:
    BOOTSTRAP.error("study checkout differs from expected state-construction commit")
if subprocess.check_output(["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_robolab_commit:
    BOOTSTRAP.error("RoboLab checkout differs from the pinned commit")

mapping = {name: name for name in ("banana", "banana_right", "bowl", "rubiks_cube")}
os.environ.update(
    {
        "VLA_WAM_V3E004_FIXTURE_CANDIDATE": str(args.e004_candidate.resolve()),
        "VLA_WAM_V3E004_FIXTURE_SHA256": args.e004_candidate_sha256,
        "VLA_WAM_V3E004_SYMMETRY_LEVEL_S": "1.0",
        "VLA_WAM_V3E004_CONTROL_SCENE_ASSET": str(args.control_scene_asset.resolve()),
        "VLA_WAM_V3E004_PAIRED_SCENE_ASSET": str(args.paired_scene_asset.resolve()),
        "VLA_WAM_V3E004_SCENE_OBJECT_MAPPING": json.dumps(mapping, sort_keys=True),
    }
)

simulation_app = AppLauncher(args).app
CURRENT_STAGE = "app_launcher_started"
LAST_REFERENCE_BOUNDS_EVIDENCE: dict[str, Any] | None = None

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.sensors.contact_sensor_utils import get_contact_sensors  # noqa: E402
from robolab.core.task.conditionals import object_grabbed  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402
from robolab.robots.droid import EEF_OFFSET_ROT  # noqa: E402

from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import (  # noqa: E402
    _quat_inverse_wxyz,
    _quat_multiply_wxyz,
    _quat_normalize_wxyz,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006.runtime_contract import (  # noqa: E402
    load_runtime_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006.state_contract import (  # noqa: E402
    canonical_bytes,
    compare_full_reset_to_e004,
    normalized_state_sha256,
    settled_gate,
    stage_ood,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import load_candidate  # noqa: E402
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.occlusion import (  # noqa: E402
    CameraEvidence,
    YawOrientedBox,
    evaluate_all_cameras,
    project_world_target_to_pixel,
)


CAMERAS = ("head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
MOVABLE = ("banana", "banana_right", "bowl", "rubiks_cube")
PERMITTED_CONTACTS = {
    "gripper__rubiks_cube",
    "banana__table",
    "banana_right__table",
    "bowl__table",
}
EXPECTED_CONTACT_SENSORS = {
    "gripper__rubiks_cube",
    "gripper__banana",
    "gripper__banana_right",
    "gripper__bowl",
    "gripper__table",
    "banana__rubiks_cube",
    "banana_right__rubiks_cube",
    "bowl__rubiks_cube",
    "rubiks_cube__table",
    "banana__banana_right",
    "banana__bowl",
    "banana__table",
    "banana_right__bowl",
    "banana_right__table",
    "bowl__table",
}


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha(resolved)}


def _retained_environment() -> dict[str, str | None]:
    return {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "VK_ICD_FILENAMES",
            "LD_LIBRARY_PATH",
            "HOME",
            "XDG_CACHE_HOME",
            "WARP_CACHE_PATH",
            "MPLCONFIGDIR",
            "TMPDIR",
            "PYTHONPATH",
        )
    }


def _base_evidence(*, candidate_gate_passed: bool = False, state_candidate_count: int = 0) -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "schema_version": "vla-wam-shared-v3e006-state-construction-attempt-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E006",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "state_candidate_count": state_candidate_count,
        "behavioral_denominator_included": False,
        "candidate_gate_passed": candidate_gate_passed,
        "failure_stage": CURRENT_STAGE,
        "invocation": sys.argv,
        "construction_source": {**_binding(source), "study_commit": args.expected_study_commit},
        "input_bindings": {
            "e004_candidate": _binding(args.e004_candidate),
            "ood_freeze": _binding(args.ood_freeze),
            "e004_full_reset_reference": _binding(args.e004_reset_reference),
            "runtime_contract": _binding(args.runtime_bindings),
            "control_scene_asset": _binding(args.control_scene_asset),
            "paired_scene_asset": _binding(args.paired_scene_asset),
            "frozen_e004_bounds_source": _binding(
                study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/model_blind_droid_gate.py"
            ),
        },
        "passed_health_preflight": {name: _binding(path) for name, (path, _digest) in health_files.items()},
        "runtime_log": {
            "path": str(args.runtime_log.resolve()),
            "binding_status": "rehash_after_process_exit_by_outer_ledger",
        },
        "environment": _retained_environment(),
        "lane": {
            "pod": args.pod,
            "pod_uid": args.pod_uid,
            "gpu_uuid": args.gpu_uuid,
            "container_image": args.container_image,
            "container_id": args.container_id,
            "driver_version": args.driver_version,
            "device": args.device,
            "python": sys.executable,
        },
        "last_reference_bounds_evidence": LAST_REFERENCE_BOUNDS_EVIDENCE,
    }


def _write_failure(exc: BaseException) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = {}
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "state_construction_failure.json":
            available[str(path.relative_to(args.output_dir))] = _binding(path)
    report = {
        **_base_evidence(),
        "status": "infrastructure_invalid_model_blind_state_construction",
        "passed": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "available_raw_artifacts": available,
    }
    path = args.output_dir / "state_construction_failure.json"
    path.write_bytes(canonical_bytes(report))
    return path


def _host(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return [float(item) for item in array.reshape(-1)]


def _qmul(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    return _quat_multiply_wxyz(left, right)


def _qrotate(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q = _quat_normalize_wxyz(quaternion)
    w, xyz = float(q[0]), q[1:]
    v = np.asarray(vector, dtype=np.float64)
    return 2 * np.dot(xyz, v) * xyz + (w * w - np.dot(xyz, xyz)) * v + 2 * w * np.cross(xyz, v)


def _rotvec_to_quat(rotvec: Sequence[float]) -> np.ndarray:
    value = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    return _quat_normalize_wxyz(np.concatenate(([math.cos(angle / 2)], value / angle * math.sin(angle / 2))))


def _command(position: Sequence[float], eef_quaternion: Sequence[float], grip: float, device: str) -> torch.Tensor:
    command_quaternion = _qmul(eef_quaternion, _quat_inverse_wxyz(np.asarray(EEF_OFFSET_ROT, dtype=np.float64)))
    action = np.concatenate((np.asarray(position), command_quaternion, [grip])).astype(np.float32)
    return torch.from_numpy(action).reshape(1, 8).to(device)


def _contact_forces(env: Any) -> dict[str, float]:
    rows: dict[str, float] = {}
    for name, sensor in sorted(get_contact_sensors(env.scene).items()):
        if name.endswith("__all_objs"):
            continue
        matrix = getattr(sensor.data, "force_matrix_w", None)
        raw = matrix if matrix is not None else getattr(sensor.data, "net_forces_w", None)
        if raw is None:
            raise RuntimeError(f"contact sensor {name} has no force stream")
        value = np.asarray(raw.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3)
        rows[name] = float(np.max(np.linalg.norm(value, axis=1))) if value.size else 0.0
    return rows


def _contact_coverage(env: Any) -> dict[str, Any]:
    inventory = sorted(name for name in get_contact_sensors(env.scene) if not name.endswith("__all_objs"))
    missing = sorted(EXPECTED_CONTACT_SENSORS - set(inventory))
    extra = sorted(set(inventory) - EXPECTED_CONTACT_SENSORS)
    checks = {
        "complete_pairwise_sensor_inventory": not missing,
        "cube_gripper_sensor_present": "gripper__rubiks_cube" in inventory,
        "companion_table_sensors_present": {"banana__table", "banana_right__table"} <= set(inventory),
        "all_sensor_force_streams_live": set(_contact_forces(env)) == set(inventory),
    }
    if not all(checks.values()):
        raise RuntimeError(f"contact-sensor coverage failed closed: missing={missing}, extra={extra}")
    return {
        "inventory": inventory,
        "expected_inventory": sorted(EXPECTED_CONTACT_SENSORS),
        "missing": missing,
        "extra": extra,
        "checks": checks,
        "passed": True,
        "force_threshold_n": 1.0,
    }


def _sample(env: Any, frames: Any, eef_index: int) -> dict[str, Any]:
    cube = env.scene["rubiks_cube"].data
    robot = env.scene["robot"].data
    return {
        "cube_position_world_m": _host(cube.root_pos_w[0]),
        "cube_linear_velocity_m_s": _host(cube.root_lin_vel_w[0]),
        "cube_angular_velocity_rad_s": _host(cube.root_ang_vel_w[0]),
        "eef_position_world_m": _host(frames.data.target_pos_w[0, eef_index]),
        "arm_joint_velocity_rad_s": _host(robot.joint_vel[0])[:7],
        "object_grabbed": bool(object_grabbed(env, object="rubiks_cube", env_id=0)),
        "contact_force_n": _contact_forces(env),
    }


def _capture_state(
    env: Any,
    frames: Any,
    eef_index: int,
    *,
    gripper_command: float,
    contact_coverage: Mapping[str, Any],
    contact_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    robot = env.scene["robot"].data
    objects: dict[str, Any] = {}
    for name in MOVABLE:
        data = env.scene[name].data
        objects[name] = {
            "position_world_m": _host(data.root_pos_w[0]),
            "quaternion_world_wxyz": _host(data.root_quat_w[0]),
            "linear_velocity_m_s": _host(data.root_lin_vel_w[0]),
            "angular_velocity_rad_s": _host(data.root_ang_vel_w[0]),
        }
    return {
        "robot": {
            "joint_names": [str(name) for name in env.scene["robot"].joint_names],
            "root_position_world_m": _host(robot.root_pos_w[0]),
            "root_quaternion_world_wxyz": _host(robot.root_quat_w[0]),
            "root_linear_velocity_m_s": _host(robot.root_lin_vel_w[0]),
            "root_angular_velocity_rad_s": _host(robot.root_ang_vel_w[0]),
            "joint_position_rad": _host(robot.joint_pos[0]),
            "joint_velocity_rad_s": _host(robot.joint_vel[0]),
            "gripper": {
                "joint_names": [str(name) for name in env.scene["robot"].joint_names[7:]],
                "joint_position_rad": _host(robot.joint_pos[0])[7:],
                "joint_velocity_rad_s": _host(robot.joint_vel[0])[7:],
                "normal_binary_command": float(gripper_command),
                "object_grabbed": bool(object_grabbed(env, object="rubiks_cube", env_id=0)),
            },
        },
        "objects": objects,
        "eef": {
            "position_world_m": _host(frames.data.target_pos_w[0, eef_index]),
            "quaternion_world_wxyz": _host(frames.data.target_quat_w[0, eef_index]),
        },
        "contact_evidence": {
            "coverage": dict(contact_coverage),
            "settled_force_snapshots_n": [dict(row["contact_force_n"]) for row in contact_samples],
            "object_grabbed_by_step": [bool(row["object_grabbed"]) for row in contact_samples],
        },
    }


def _physical_prim_path(asset: Any) -> str:
    for owner_name in ("root_physx_view", "_root_physx_view"):
        paths = getattr(getattr(asset, owner_name, None), "prim_paths", None)
        if paths:
            return str(paths[0])
    return str(asset.cfg.prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")


def _materialize_single_env_prim_path(raw_path: str, *, num_envs: int) -> str:
    """Resolve RoboLab's single-environment regex without changing geometry."""
    if num_envs != 1:
        raise RuntimeError("bowl prim-path materialization is defined only for num_envs=1")
    if raw_path.count("env_.*/") > 1:
        raise RuntimeError(f"ambiguous environment regex in prim path: {raw_path}")
    if ".*" in raw_path and "/World/envs/env_.*/" not in raw_path:
        raise RuntimeError(f"unsupported environment regex in prim path: {raw_path}")
    resolved = raw_path.replace("/World/envs/env_.*/", "/World/envs/env_0/")
    if ".*" in resolved or "{" in resolved or "}" in resolved:
        raise RuntimeError(f"unresolved/ambiguous prim path: {raw_path}")
    return resolved


def _reference_bounds(env: Any) -> YawOrientedBox:
    global LAST_REFERENCE_BOUNDS_EVIDENCE
    asset = env.scene["bowl"]
    raw_path = _physical_prim_path(asset)
    prim_path = _materialize_single_env_prim_path(raw_path, num_envs=args.num_envs)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    matches = [path for path in (prim_path,) if stage.GetPrimAtPath(path) and stage.GetPrimAtPath(path).IsValid()]
    if matches != [prim_path] or not prim or not prim.IsValid():
        LAST_REFERENCE_BOUNDS_EVIDENCE = {
            "raw_prim_path": raw_path,
            "resolved_prim_path": prim_path,
            "valid_matches": matches,
            "passed": False,
        }
        raise RuntimeError(f"expected exactly one valid bowl prim, found {matches}")
    local_bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(prim)
    local_range = local_bound.ComputeAlignedRange()
    minimum, maximum = local_range.GetMin(), local_range.GetMax()
    minimum_values = [float(minimum[index]) for index in range(3)]
    maximum_values = [float(maximum[index]) for index in range(3)]
    local_center = np.asarray(
        [(minimum_values[index] + maximum_values[index]) * 0.5 for index in range(3)], dtype=np.float64
    )
    half = tuple((maximum_values[index] - minimum_values[index]) * 0.5 for index in range(3))
    LAST_REFERENCE_BOUNDS_EVIDENCE = {
        "method": "frozen E004 _reference_bounds_world math after deterministic num_envs=1 regex materialization",
        "frozen_e004_source": _binding(
            study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/model_blind_droid_gate.py"
        ),
        "raw_prim_path": raw_path,
        "resolved_prim_path": prim_path,
        "valid_matches": matches,
        "local_minimum_m": minimum_values,
        "local_maximum_m": maximum_values,
        "local_center_m": local_center.tolist(),
        "half_extents_m": list(half),
        "passed": False,
    }
    if not all(math.isfinite(value) for value in (*minimum_values, *maximum_values, *local_center)):
        raise RuntimeError("bowl USD local range/center is nonfinite")
    if not all(math.isfinite(value) and value > 0 for value in half):
        raise RuntimeError("bowl USD local bound is invalid")
    position = np.asarray(_host(asset.data.root_pos_w[0]))
    quaternion = _host(asset.data.root_quat_w[0])
    center = position + _qrotate(quaternion, local_center)
    if not np.all(np.isfinite(center)):
        raise RuntimeError("bowl USD world center is nonfinite")
    w, x, y, z = _quat_normalize_wxyz(quaternion)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    LAST_REFERENCE_BOUNDS_EVIDENCE.update(
        {
            "center_world_m": center.tolist(),
            "yaw_world_rad": yaw,
            "passed": True,
        }
    )
    return YawOrientedBox(tuple(center), half, yaw)


def _save_camera_evidence(env: Any, obs: Mapping[str, Any], stage: str, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    cube = _host(env.scene["rubiks_cube"].data.root_pos_w[0])
    bounds = _reference_bounds(env)
    evidence: dict[str, CameraEvidence] = {}
    bindings: dict[str, Any] = {}
    for name in CAMERAS:
        image = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy(), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3 or not np.ptp(image):
            raise RuntimeError(f"blank or malformed conditioning camera {name}")
        path = root / f"{stage}__{name}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to retain camera image {path}")
        sensor = env.scene[name]
        center = _host(sensor.data.pos_w[0])
        quaternion = _host(sensor.data.quat_w_ros[0])
        intrinsic = np.asarray(sensor.data.intrinsic_matrices[0].detach().cpu().numpy(), dtype=np.float64).tolist()
        geometry = {
            "camera_center_world_m": center,
            "camera_quaternion_world_wxyz_ros": quaternion,
            "intrinsic_matrix_3x3": intrinsic,
            "image_size_wh": [int(image.shape[1]), int(image.shape[0])],
            "extrinsics": {
                "translation_world_m": center,
                "quaternion_world_wxyz_ros": quaternion,
                "convention": "ROS camera axes: +X right, +Y down, +Z forward",
            },
        }
        geometry_sha = hashlib.sha256(canonical_bytes(geometry)).hexdigest()
        pixel = project_world_target_to_pixel(
            camera_center_world_m=center,
            camera_quaternion_world_wxyz_ros=quaternion,
            target_center_world_m=cube,
            intrinsic_matrix_3x3=intrinsic,
        )
        evidence[name] = CameraEvidence(
            camera_name=name,
            camera_center_world_m=tuple(center),
            target_center_world_m=tuple(cube),
            reference_bounds_world=bounds,
            target_instance_visible_pixels=None,
            segmentation_source_sha256=None,
            target_projected_pixel_uv=pixel,
            image_size_wh=(int(image.shape[1]), int(image.shape[0])),
            camera_geometry_source_sha256=geometry_sha,
        )
        bindings[name] = {
            **geometry,
            "target_projected_pixel_uv": list(pixel),
            "rgb": {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)},
            "rgb_nonblank": True,
        }
    gate = evaluate_all_cameras(evidence, expected_cameras=CAMERAS, minimum_visible_target_pixels=32)
    return {
        "bindings": bindings,
        "reference_bounds_evidence": LAST_REFERENCE_BOUNDS_EVIDENCE,
        "gate": gate,
        "policy_conditioning_camera_feeds": {
            "observation/exterior_image_1_left": "over_shoulder_left_camera",
            "observation/wrist_image_left": "wrist_cam",
        },
        "all_policy_conditioning_feeds_retained": all(
            name in bindings for name in ("over_shoulder_left_camera", "wrist_cam")
        ),
        "passed": all(row["gate_passed"] for row in gate.values()),
    }


def _companion_gate(state: Mapping[str, Any], candidate: Any) -> dict[str, Any]:
    nominal = candidate.layout(1.0)
    observed: dict[str, Any] = {}
    passed = True
    for name in ("banana", "banana_right", "bowl"):
        actual = state["objects"][name]
        expected = nominal[name]
        position_error = float(np.linalg.norm(np.asarray(actual["position_world_m"]) - [expected.x_m, expected.y_m, expected.z_m]))
        q = _quat_normalize_wxyz(actual["quaternion_world_wxyz"])
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        orientation_error = abs((yaw - expected.yaw_rad + math.pi) % (2 * math.pi) - math.pi)
        row_passed = position_error < candidate.realisation_position_tolerance_m and orientation_error < candidate.realisation_orientation_tolerance_rad
        observed[name] = {"position_error_m": position_error, "orientation_error_rad": orientation_error, "passed": row_passed}
        passed = passed and row_passed
    return {
        "observed": observed,
        "position_tolerance_m_strict": candidate.realisation_position_tolerance_m,
        "orientation_tolerance_rad_strict": candidate.realisation_orientation_tolerance_rad,
        "passed": passed,
    }


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise RuntimeError("state construction video has no frames")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("video writer failed to open")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    global CURRENT_STAGE
    CURRENT_STAGE = "load_hash_bound_contracts"
    args.output_dir.mkdir(parents=True)
    set_output_dir(str((args.output_dir / "native").resolve()))
    ood = json.loads(args.ood_freeze.read_text(encoding="utf-8"))
    reset_reference = json.loads(args.e004_reset_reference.read_text(encoding="utf-8"))
    runtime_bindings = load_runtime_contract(
        args.runtime_bindings,
        args.runtime_bindings_sha256,
        study_root=study_root,
        external_roots=(robolab_root,),
    )
    candidate = load_candidate(args.e004_candidate, args.e004_candidate_sha256)
    CURRENT_STAGE = "register_exact_e004_environment"
    task_file = study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/task_files/left.py"
    auto_register_droid_abs_ik_envs(task=[str(task_file)], cameras=WRIST_LEFT_RIGHT_HEAD)
    env = None
    primary_failure: BaseException | None = None
    try:
        env, cfg = create_env(
            "V3E004DroidLeftTask",
            device=args.device,
            seed=args.construction_seed,
            num_envs=1,
            instruction_type="default",
            policy="v3e006_model_blind_state_construction",
            renderer=args.renderer,
            rendering_mode=args.rendering_type,
        )
        video_frames: list[np.ndarray] = []
        started = time.time()
        CURRENT_STAGE = "full_reset"
        obs, _ = env.reset()
        frames = env.scene["frames"]
        eef_index = frames.data.target_frame_names.index("eef_frame")
        CURRENT_STAGE = "contact_sensor_coverage"
        contact_coverage = _contact_coverage(env)
        reset_position = _host(frames.data.target_pos_w[0, eef_index])
        reset_quaternion = _host(frames.data.target_quat_w[0, eef_index])
        action = _command(reset_position, reset_quaternion, 0.0, env.device)
        reset_samples: list[dict[str, Any]] = []
        CURRENT_STAGE = "full_reset_settle"
        for step in range(75):
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("environment terminated during E004 reset settle")
            if step % 6 == 0:
                video_frames.append(np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8))
            if step >= 65:
                reset_samples.append(_sample(env, frames, eef_index))
        CURRENT_STAGE = "full_reset_capture_and_gates"
        full_reset = _capture_state(
            env,
            frames,
            eef_index,
            gripper_command=0.0,
            contact_coverage=contact_coverage,
            contact_samples=reset_samples,
        )
        full_reset["normalized_state_sha256"] = normalized_state_sha256(full_reset)
        full_reset["camera_evidence"] = _save_camera_evidence(env, obs, "full_reset", args.output_dir / "cameras")
        full_reset["companion_pose_gate"] = _companion_gate(full_reset, candidate)
        full_reset["e004_full_reset_comparison"] = compare_full_reset_to_e004(
            full_reset,
            reference=reset_reference,
            reference_file_sha256=args.e004_reset_reference_sha256,
        )
        if not all(
            (
                full_reset["camera_evidence"]["passed"],
                full_reset["companion_pose_gate"]["passed"],
                full_reset["e004_full_reset_comparison"]["passed"],
                contact_coverage["passed"],
            )
        ):
            raise RuntimeError("full_reset differs from the retained E004 s=1 reset contract")

        cube_initial = np.asarray(full_reset["objects"]["rubiks_cube"]["position_world_m"], dtype=np.float64)
        cube_quaternion = np.asarray(full_reset["objects"]["rubiks_cube"]["quaternion_world_wxyz"], dtype=np.float64)
        stages: dict[str, Any] = {"full_reset": full_reset}
        for stage, lift in (("canonical_grasp", 0.008), ("canonical_carry", 0.055)):
            CURRENT_STAGE = f"{stage}_construction"
            reference = ood["stages"][stage]
            relative_translation = np.asarray(reference["direction_balanced_center"][7:10], dtype=np.float64)
            relative_rotation = _rotvec_to_quat(reference["direction_balanced_center"][10:13])
            desired_cube = cube_initial.copy()
            desired_cube[1] = 0.0
            desired_cube[2] += lift
            desired_eef_quaternion = _qmul(cube_quaternion, _quat_inverse_wxyz(relative_rotation))
            desired_eef_position = desired_cube - _qrotate(desired_eef_quaternion, relative_translation)
            if stage == "canonical_grasp":
                approach = desired_eef_position.copy()
                approach[2] += 0.10
                waypoints = ((approach, 0.0), (desired_eef_position, 0.0), (desired_eef_position, 0.7853981633974483))
            else:
                waypoints = ((desired_eef_position, 0.7853981633974483),)
            for target, grip in waypoints:
                action = _command(target, desired_eef_quaternion, grip, env.device)
                for step in range(args.steps_per_waypoint):
                    obs, _, terminated, truncated, _ = env.step(action)
                    if bool(terminated[0]) or bool(truncated[0]):
                        raise RuntimeError(f"environment terminated constructing {stage}")
                    if step % 5 == 0:
                        video_frames.append(np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8))
            CURRENT_STAGE = f"{stage}_settle"
            settled: list[dict[str, Any]] = []
            for step in range(80):
                obs, _, terminated, truncated, _ = env.step(action)
                if bool(terminated[0]) or bool(truncated[0]):
                    raise RuntimeError(f"environment terminated settling {stage}")
                if step >= 70:
                    settled.append(_sample(env, frames, eef_index))
                if step % 5 == 0:
                    video_frames.append(np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8))
            sensor_names = set(_contact_forces(env))
            unintended = sorted(sensor_names - PERMITTED_CONTACTS)
            CURRENT_STAGE = f"{stage}_capture_and_gates"
            state = _capture_state(
                env,
                frames,
                eef_index,
                gripper_command=0.7853981633974483,
                contact_coverage=contact_coverage,
                contact_samples=settled,
            )
            state["physics_gate"] = settled_gate(settled, unintended_contact_pairs=unintended)
            state["ood_gate"] = stage_ood(state, stage_reference=reference)
            state["camera_evidence"] = _save_camera_evidence(env, obs, stage, args.output_dir / "cameras")
            state["companion_pose_gate"] = _companion_gate(state, candidate)
            state["construction_target"] = {
                "method": "deterministic RoboLab absolute differential IK plus normal gripper contact; no weld or learned-model request",
                "cube_position_world_m": desired_cube.tolist(),
                "eef_position_world_m": desired_eef_position.tolist(),
                "eef_quaternion_world_wxyz": desired_eef_quaternion.tolist(),
            }
            state["normalized_state_sha256"] = normalized_state_sha256(state)
            state["passed"] = all(
                (
                    state["physics_gate"]["passed"],
                    state["ood_gate"]["passed"],
                    state["camera_evidence"]["passed"],
                    state["companion_pose_gate"]["passed"],
                )
            )
            stages[stage] = state
            if not state["passed"]:
                raise RuntimeError(f"{stage} candidate failed a preregistered state gate")

        CURRENT_STAGE = "write_candidate_video"
        video_path = args.output_dir / "videos" / "model_blind_state_construction.mp4"
        video_path.parent.mkdir(parents=True)
        _write_video(video_path, video_frames)
        output = {
            "schema_version": "vla-wam-shared-v3e006-state-candidate-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": "V3-E006",
            "status": "passed_model_blind_state_construction_not_released_for_inference",
            "passed": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "state_candidate_count": 1,
            "construction_seed": args.construction_seed,
            "construction_prompt_exposure": "environment task prompt exists in the simulator config but is never read by or supplied to the deterministic controller",
            "task_prompt": cfg.instruction,
            "ood_freeze": {"path": str(args.ood_freeze.resolve()), "sha256": args.ood_freeze_sha256, "bytes": args.ood_freeze.stat().st_size},
            "construction_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha(Path(__file__).resolve()),
                "bytes": Path(__file__).stat().st_size,
                "study_commit": args.expected_study_commit,
            },
            "e004_full_reset_reference": {"path": str(args.e004_reset_reference.resolve()), "sha256": args.e004_reset_reference_sha256, "bytes": args.e004_reset_reference.stat().st_size},
            "frozen_e004_runtime_bindings": {"path": str(args.runtime_bindings.resolve()), "sha256": args.runtime_bindings_sha256, "bytes": args.runtime_bindings.stat().st_size, "value": runtime_bindings},
            "e004_candidate": {"path": str(args.e004_candidate.resolve()), "sha256": args.e004_candidate_sha256, "bytes": args.e004_candidate.stat().st_size},
            "scene_assets": {
                "control": {"path": str(args.control_scene_asset.resolve()), "sha256": _sha(args.control_scene_asset), "bytes": args.control_scene_asset.stat().st_size},
                "paired": {"path": str(args.paired_scene_asset.resolve()), "sha256": _sha(args.paired_scene_asset), "bytes": args.paired_scene_asset.stat().st_size},
            },
            "environment": {"pod": args.pod, "pod_uid": args.pod_uid, "gpu_uuid": args.gpu_uuid, "device": args.device},
            "execution_evidence": _base_evidence(candidate_gate_passed=True, state_candidate_count=1),
            "contact_sensor_coverage": contact_coverage,
            "construction_seconds": time.time() - started,
            "stages": stages,
            "video": {"path": str(video_path.resolve()), "sha256": _sha(video_path), "bytes": video_path.stat().st_size},
            "release_boundary": "registration, queue, tests, and an explicit smoke/isolation release must be committed and pushed before request zero",
        }
        output_path = args.output_dir / "state_candidate.json"
        CURRENT_STAGE = "write_passed_candidate"
        output_path.write_bytes(canonical_bytes(output))
        print(json.dumps({"passed": True, "output": str(output_path), "sha256": _sha(output_path), "bytes": output_path.stat().st_size}, indent=2))
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                if primary_failure is None:
                    raise
                print(
                    f"environment close raised after retained construction failure: {type(close_error).__name__}: {close_error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    construction_failure: BaseException | None = None
    try:
        main()
    except BaseException as exc:
        construction_failure = exc
        failure_path = _write_failure(exc)
        print(
            json.dumps(
                {"passed": False, "failure": str(failure_path), "sha256": _sha(failure_path)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except BaseException as close_error:
            if construction_failure is None:
                raise
            print(
                f"SimulationApp.close raised after retained construction failure: {type(close_error).__name__}: {close_error}",
                file=sys.stderr,
            )
    if construction_failure is not None:
        raise construction_failure.with_traceback(construction_failure.__traceback__)
