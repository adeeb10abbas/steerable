#!/usr/bin/env python3
"""Evaluate the preregistered V3-E006-R001 matched-direction repair schedule.

Historical sources came from E004 pi0.5, but this process makes no new model
request and selects only with the unchanged physical/OOD/camera gates.
"""

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
BOOTSTRAP.add_argument("--repair-registration", type=Path, required=True)
BOOTSTRAP.add_argument("--repair-registration-sha256", required=True)
BOOTSTRAP.add_argument("--candidate-schedule", type=Path, required=True)
BOOTSTRAP.add_argument("--candidate-schedule-sha256", required=True)
BOOTSTRAP.add_argument("--original-closure-binding", type=Path, required=True)
BOOTSTRAP.add_argument("--original-closure-binding-sha256", required=True)
BOOTSTRAP.add_argument("--source-push-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--source-push-gate-sha256", required=True)
BOOTSTRAP.add_argument("--control-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--paired-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--pod", required=True)
BOOTSTRAP.add_argument("--pod-uid", required=True)
BOOTSTRAP.add_argument("--gpu-uuid", required=True)
BOOTSTRAP.add_argument("--expected-study-commit", required=True)
BOOTSTRAP.add_argument("--expected-robolab-commit", default="0aef241fb088ca21bb4ebd24448940ed56620d17")
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
    (args.repair_registration, args.repair_registration_sha256),
    (args.candidate_schedule, args.candidate_schedule_sha256),
    (args.original_closure_binding, args.original_closure_binding_sha256),
    (args.source_push_gate, args.source_push_gate_sha256),
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

try:
    repair_registration = json.loads(args.repair_registration.read_text(encoding="utf-8"))
    candidate_schedule = json.loads(args.candidate_schedule.read_text(encoding="utf-8"))
    source_push_gate = json.loads(args.source_push_gate.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    BOOTSTRAP.error(f"repair control artifact is unreadable: {exc}")
if repair_registration.get("repair_amendment_id") != "V3-E006-R001":
    BOOTSTRAP.error("repair registration ID differs")
if repair_registration.get("status") != "prospectively_registered_repair_algorithm_before_any_repair_candidate_or_model_request":
    BOOTSTRAP.error("repair registration status differs")
if repair_registration.get("counts_at_registration") != {
    "behavioral_episodes": 0,
    "model_requests": 0,
    "repair_candidate_evaluations": 0,
}:
    BOOTSTRAP.error("repair registration does not bind zero counts")
if candidate_schedule.get("repair_amendment_id") != "V3-E006-R001" or candidate_schedule.get("candidate_budget") != 8:
    BOOTSTRAP.error("candidate schedule identity/budget differs")
if len(candidate_schedule.get("candidate_pairs", [])) != 8 or [row.get("candidate_rank") for row in candidate_schedule["candidate_pairs"]] != list(range(1, 9)):
    BOOTSTRAP.error("candidate schedule ranks differ")
schedule_registration = candidate_schedule.get("repair_registration", {})
if schedule_registration.get("bytes") != args.repair_registration.stat().st_size or schedule_registration.get("sha256") != args.repair_registration_sha256:
    BOOTSTRAP.error("candidate schedule does not bind this repair registration")
schedule_ood = candidate_schedule.get("ood_freeze", {})
if schedule_ood.get("bytes") != args.ood_freeze.stat().st_size or schedule_ood.get("sha256") != args.ood_freeze_sha256:
    BOOTSTRAP.error("candidate schedule does not bind the frozen OOD reference")
schedule_closure = candidate_schedule.get("original_closure_verification", {}).get("binding", {})
if schedule_closure.get("bytes") != args.original_closure_binding.stat().st_size or schedule_closure.get("sha256") != args.original_closure_binding_sha256:
    BOOTSTRAP.error("candidate schedule does not bind the original closure proof")
rank_one = candidate_schedule["candidate_pairs"][0]
rank_one_observed = {
    "grasp": (
        rank_one["canonical_grasp"]["environment_seed"],
        rank_one["canonical_grasp"]["source_states"]["left"]["state_capture_index"],
        rank_one["canonical_grasp"]["source_states"]["left"]["hdf5_index"],
        rank_one["canonical_grasp"]["source_states"]["right"]["state_capture_index"],
        rank_one["canonical_grasp"]["source_states"]["right"]["hdf5_index"],
    ),
    "carry": (
        rank_one["canonical_carry"]["environment_seed"],
        rank_one["canonical_carry"]["source_states"]["left"]["state_capture_index"],
        rank_one["canonical_carry"]["source_states"]["left"]["hdf5_index"],
        rank_one["canonical_carry"]["source_states"]["right"]["state_capture_index"],
        rank_one["canonical_carry"]["source_states"]["right"]["hdf5_index"],
    ),
}
if rank_one_observed != {"grasp": (9521, 30, 104, 31, 105), "carry": (9442, 39, 113, 38, 112)}:
    BOOTSTRAP.error("rank-one matched-direction anchors differ from registration")
if source_push_gate.get("status") != "passed_before_any_r001_candidate_or_model_request":
    BOOTSTRAP.error("source-push gate did not pass prospectively")
implementation_commit = str(source_push_gate.get("implementation_commit", ""))
if not implementation_commit or subprocess.run(
    ["git", "-C", str(study_root), "merge-base", "--is-ancestor", implementation_commit, args.expected_study_commit],
    check=False,
).returncode:
    BOOTSTRAP.error("source-push implementation commit is not an ancestor of runtime checkout")
if source_push_gate.get("model_request_count") != 0 or source_push_gate.get("repair_candidate_evaluation_count") != 0:
    BOOTSTRAP.error("source-push gate has nonzero prerelease counts")
implementation_files = source_push_gate.get("implementation_files")
if not isinstance(implementation_files, list) or not implementation_files:
    BOOTSTRAP.error("source-push gate has no implementation-file inventory")
for row in implementation_files:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        BOOTSTRAP.error(f"source-push gate contains unsafe path: {relative}")
    actual = study_root / relative
    if not actual.is_file() or actual.stat().st_size != row.get("bytes") or _sha(actual) != row.get("sha256"):
        BOOTSTRAP.error(f"source-push implementation file changed: {relative}")

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
LAST_PARTIAL_STAGES: dict[str, Any] = {}
CANDIDATE_EVALUATION_COUNT = 0

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
        "schema_version": "vla-wam-shared-v3e006-r001-state-repair-attempt-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E006-R001",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "state_candidate_count": state_candidate_count,
        "repair_candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
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
            "repair_registration": _binding(args.repair_registration),
            "candidate_schedule": _binding(args.candidate_schedule),
            "original_v3e006_closure_binding": _binding(args.original_closure_binding),
            "source_push_gate": _binding(args.source_push_gate),
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
        "partial_stage_evidence": LAST_PARTIAL_STAGES,
    }


def _write_failure(exc: BaseException) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = {}
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "state_construction_failure.json":
            available[str(path.relative_to(args.output_dir))] = _binding(path)
    report = {
        **_base_evidence(),
        "status": "infrastructure_invalid_r001_state_repair",
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
    return YawOrientedBox(
        tuple(float(value) for value in center),
        tuple(float(value) for value in half),
        float(yaw),
    )


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


def _fresh_reset_and_gate(
    env: Any,
    *,
    candidate: Any,
    reset_reference: Mapping[str, Any],
    contact_coverage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    global CURRENT_STAGE
    CURRENT_STAGE = f"{label}_fresh_e004_reset"
    obs, _ = env.reset()
    reset_position = _host(frames.data.target_pos_w[0, eef_index])
    reset_quaternion = _host(frames.data.target_quat_w[0, eef_index])
    reset_action = _command(reset_position, reset_quaternion, 0.0, env.device)
    reset_samples: list[dict[str, Any]] = []
    for step in range(75):
        obs, _, terminated, truncated, _ = env.step(reset_action)
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError(f"environment terminated during {label} E004 reset settle")
        if step % 12 == 0:
            video_frames.append(np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8))
        if step >= 65:
            reset_samples.append(_sample(env, frames, eef_index))
    reset_state = _capture_state(
        env,
        frames,
        eef_index,
        gripper_command=0.0,
        contact_coverage=contact_coverage,
        contact_samples=reset_samples,
    )
    reset_state["normalized_state_sha256"] = normalized_state_sha256(reset_state)
    reset_state["camera_evidence"] = _save_camera_evidence(
        env, obs, f"{label}__full_reset", args.output_dir / "cameras"
    )
    reset_state["companion_pose_gate"] = _companion_gate(reset_state, candidate)
    reset_state["e004_full_reset_comparison"] = compare_full_reset_to_e004(
        reset_state,
        reference=reset_reference,
        reference_file_sha256=args.e004_reset_reference_sha256,
    )
    reset_state["passed"] = all(
        (
            reset_state["camera_evidence"]["passed"],
            reset_state["companion_pose_gate"]["passed"],
            reset_state["e004_full_reset_comparison"]["passed"],
            contact_coverage["passed"],
        )
    )
    if not reset_state["passed"]:
        raise RuntimeError(f"{label} fresh reset differs from retained exact E004 s=1 reset")
    return obs, reset_state


def _materialize_and_gate_stage(
    env: Any,
    *,
    stage_name: str,
    schedule_stage: Mapping[str, Any],
    stage_reference: Mapping[str, Any],
    candidate: Any,
    contact_coverage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    global CURRENT_STAGE
    balanced = schedule_stage["direction_balanced_state"]
    joint_position_values = balanced["joint_position_rad"]
    joint_velocity_values = balanced["joint_velocity_rad_s"]
    cube_pose_values = balanced["cube_pose_world_wxyz"]
    cube_velocity_values = balanced["cube_velocity_world"]
    if len(joint_position_values) != len(env.scene["robot"].joint_names) or len(joint_velocity_values) != len(joint_position_values):
        raise RuntimeError("scheduled full joint state does not match the exact E004 robot")
    if len(cube_pose_values) != 7 or len(cube_velocity_values) != 6:
        raise RuntimeError("scheduled cube state has invalid dimensions")
    if not all(math.isfinite(float(value)) for value in (*joint_position_values, *joint_velocity_values, *cube_pose_values, *cube_velocity_values)):
        raise RuntimeError("scheduled direction-balanced state contains a nonfinite number")

    CURRENT_STAGE = f"{label}_atomic_closed_contact_write"
    robot = env.scene["robot"]
    cube = env.scene["rubiks_cube"]
    joint_position = torch.tensor(joint_position_values, dtype=torch.float32, device=env.device).reshape(1, -1)
    joint_velocity = torch.tensor(joint_velocity_values, dtype=torch.float32, device=env.device).reshape(1, -1)
    cube_pose = torch.tensor(cube_pose_values, dtype=torch.float32, device=env.device).reshape(1, 7)
    cube_velocity = torch.tensor(cube_velocity_values, dtype=torch.float32, device=env.device).reshape(1, 6)
    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    robot.set_joint_position_target(joint_position)
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(cube_velocity)
    env.scene.write_data_to_sim()
    env.sim.step(render=True)
    env.scene.update(env.physics_dt)

    actual_fk_position = _host(frames.data.target_pos_w[0, eef_index])
    actual_fk_quaternion = _host(frames.data.target_quat_w[0, eef_index])
    hold_action = _command(actual_fk_position, actual_fk_quaternion, 0.7853981633974483, env.device)
    settled: list[dict[str, Any]] = []
    obs: Mapping[str, Any] | None = None
    CURRENT_STAGE = f"{label}_closed_hold_160"
    for step in range(160):
        obs, _, terminated, truncated, _ = env.step(hold_action)
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError(f"environment terminated during {label} closed hold")
        if step % 10 == 0:
            video_frames.append(np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8))
        if step >= 150:
            settled.append(_sample(env, frames, eef_index))
    if obs is None:
        raise RuntimeError("closed hold produced no observation")

    sensor_names = set(_contact_forces(env))
    unintended = sorted(sensor_names - PERMITTED_CONTACTS)
    CURRENT_STAGE = f"{label}_unchanged_gates"
    state = _capture_state(
        env,
        frames,
        eef_index,
        gripper_command=0.7853981633974483,
        contact_coverage=contact_coverage,
        contact_samples=settled,
    )
    state["physics_gate"] = settled_gate(settled, unintended_contact_pairs=unintended)
    state["ood_gate"] = stage_ood(state, stage_reference=stage_reference)
    state["camera_evidence"] = _save_camera_evidence(env, obs, label, args.output_dir / "cameras")
    state["companion_pose_gate"] = _companion_gate(state, candidate)
    state["construction"] = {
        "method": "atomic direction-balanced full-state write with historical closed gripper and cube normal-contact geometry; one closed sync step; 160-step closed absolute-IK hold at actual post-sync FK pose; no weld, open approach, lift, or learned-model request",
        "candidate_rank": schedule_stage.get("candidate_rank"),
        "stage": stage_name,
        "source_environment_seed": schedule_stage["environment_seed"],
        "source_states": schedule_stage["source_states"],
        "scheduled_direction_balanced_state": balanced,
        "actual_post_sync_fk_hold_target": {
            "position_world_m": actual_fk_position,
            "quaternion_world_wxyz": actual_fk_quaternion,
        },
        "settle_steps": 160,
        "gate_window_final_steps": 10,
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
    return obs, state


def main() -> None:
    global CURRENT_STAGE, LAST_PARTIAL_STAGES, CANDIDATE_EVALUATION_COUNT
    CURRENT_STAGE = "load_hash_bound_r001_contracts"
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
            seed=13000,
            num_envs=1,
            instruction_type="default",
            policy="v3e006_r001_behavior_blind_state_repair",
            renderer=args.renderer,
            rendering_mode=args.rendering_type,
        )
        started = time.time()
        video_frames: list[np.ndarray] = []
        frames = env.scene["frames"]
        eef_index = frames.data.target_frame_names.index("eef_frame")
        CURRENT_STAGE = "pre_candidate_contact_sensor_initialization_reset"
        env.reset()
        CURRENT_STAGE = "contact_sensor_coverage"
        contact_coverage = _contact_coverage(env)
        attempts: list[dict[str, Any]] = []
        accepted: dict[str, Any] | None = None

        for candidate_pair in candidate_schedule["candidate_pairs"]:
            rank = int(candidate_pair["candidate_rank"])
            CANDIDATE_EVALUATION_COUNT = rank
            rank_attempt: dict[str, Any] = {
                "candidate_rank": rank,
                "model_request_count": 0,
                "behavioral_episode_count": 0,
                "stages": {},
            }
            LAST_PARTIAL_STAGES = {"candidate_rank": rank}
            for stage_name in ("canonical_grasp", "canonical_carry"):
                label = f"rank{rank:02d}__{stage_name}"
                _obs, reset_state = _fresh_reset_and_gate(
                    env,
                    candidate=candidate,
                    reset_reference=reset_reference,
                    contact_coverage=contact_coverage,
                    frames=frames,
                    eef_index=eef_index,
                    label=label,
                    video_frames=video_frames,
                )
                schedule_stage = dict(candidate_pair[stage_name])
                schedule_stage["candidate_rank"] = rank
                _obs, state = _materialize_and_gate_stage(
                    env,
                    stage_name=stage_name,
                    schedule_stage=schedule_stage,
                    stage_reference=ood["stages"][stage_name],
                    candidate=candidate,
                    contact_coverage=contact_coverage,
                    frames=frames,
                    eef_index=eef_index,
                    label=label,
                    video_frames=video_frames,
                )
                rank_attempt["stages"][stage_name] = {
                    "fresh_reset": reset_state,
                    "candidate_state": state,
                }
                LAST_PARTIAL_STAGES[stage_name] = rank_attempt["stages"][stage_name]
            rank_attempt["passed"] = all(
                row["candidate_state"]["passed"] for row in rank_attempt["stages"].values()
            )
            attempts.append(rank_attempt)
            if rank_attempt["passed"]:
                accepted = rank_attempt
                break

        CURRENT_STAGE = "retain_r001_candidate_search"
        video_path = args.output_dir / "videos" / "v3e006_r001_state_repair_search.mp4"
        video_path.parent.mkdir(parents=True)
        _write_video(video_path, video_frames)
        passed = accepted is not None
        report = {
            "schema_version": "vla-wam-shared-v3e006-r001-state-repair-result-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": "V3-E006-R001",
            "status": (
                "passed_r001_state_repair_not_released_for_behavior"
                if passed
                else "r001_candidate_budget_exhausted_no_valid_state_pair"
            ),
            "passed": passed,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "repair_candidate_evaluation_count": len(attempts),
            "accepted_candidate_rank": accepted["candidate_rank"] if accepted else None,
            "first_passing_rule_obeyed": accepted is None or all(not row["passed"] for row in attempts[:-1]),
            "candidate_budget": 8,
            "repair_registration": _binding(args.repair_registration),
            "candidate_schedule": _binding(args.candidate_schedule),
            "source_push_gate": _binding(args.source_push_gate),
            "original_v3e006_closure_binding": _binding(args.original_closure_binding),
            "ood_freeze": _binding(args.ood_freeze),
            "construction_source": {
                **_binding(Path(__file__).resolve()),
                "study_commit": args.expected_study_commit,
            },
            "frozen_e004_runtime_bindings": {
                **_binding(args.runtime_bindings),
                "value": runtime_bindings,
            },
            "e004_full_reset_reference": _binding(args.e004_reset_reference),
            "e004_candidate": _binding(args.e004_candidate),
            "scene_assets": {
                "control": _binding(args.control_scene_asset),
                "paired": _binding(args.paired_scene_asset),
            },
            "historical_policy_provenance_disclosure": candidate_schedule[
                "historical_policy_provenance_disclosure"
            ],
            "construction_prompt_exposure": "the exact E004 environment task prompt exists in cfg but is never read by or supplied to the repair controller",
            "task_prompt_retained_for_audit": cfg.instruction,
            "selection_rule": repair_registration["candidate_search"],
            "contact_sensor_coverage": contact_coverage,
            "attempts": attempts,
            "accepted_states": accepted["stages"] if accepted else None,
            "execution_evidence": _base_evidence(
                candidate_gate_passed=passed,
                state_candidate_count=1 if passed else 0,
            ),
            "construction_seconds": time.time() - started,
            "video": _binding(video_path),
            "release_boundary": "behavioral registration, queue, smoke/isolation, and release remain prohibited until this result is independently validated and committed",
        }
        report_path = args.output_dir / "state_repair_result.json"
        report_path.write_bytes(canonical_bytes(report))
        print(
            json.dumps(
                {
                    "passed": passed,
                    "output": str(report_path),
                    "sha256": _sha(report_path),
                    "bytes": report_path.stat().st_size,
                    "candidate_evaluations": len(attempts),
                    "accepted_candidate_rank": report["accepted_candidate_rank"],
                },
                indent=2,
                sort_keys=True,
            )
        )
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
                    f"environment close raised after retained R001 failure: {type(close_error).__name__}: {close_error}",
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
