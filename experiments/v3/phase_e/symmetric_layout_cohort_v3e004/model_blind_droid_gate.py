#!/usr/bin/env python3
"""Run one zero-model-request E004 DROID live scene/RTX preflight.

This command exercises the exact E004 task fixture in Isaac/RoboLab, performs
the registered settle/stability window, records all four live RGB cameras,
extracts calibrated camera geometry plus the bowl's USD bounding box, and
compiles the E004 snapshot/layout/occlusion gate.  It never starts or queries a
learned policy.  Behavioral bridges repeat the same gate in-process immediately
before request zero; this standalone preflight cannot be substituted for that
per-episode gate.
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
import traceback
from typing import Any, Mapping


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--robolab-root", type=Path, required=True)
BOOTSTRAP.add_argument("--registration", type=Path, required=True)
BOOTSTRAP.add_argument("--registration-sha256", required=True)
BOOTSTRAP.add_argument("--queue", type=Path, required=True)
BOOTSTRAP.add_argument("--queue-sha256", required=True)
BOOTSTRAP.add_argument("--candidate", type=Path, required=True)
BOOTSTRAP.add_argument("--candidate-sha256", required=True)
BOOTSTRAP.add_argument("--cell-id", required=True)
BOOTSTRAP.add_argument("--control-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--paired-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--scene-object-mapping", required=True)
BOOTSTRAP.add_argument("--gate-output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--pod", required=True)
BOOTSTRAP.add_argument("--pod-uid", required=True)
BOOTSTRAP.add_argument("--gpu-uuid", required=True)
BOOTSTRAP.add_argument("--expected-study-commit", required=True)
BOOTSTRAP.add_argument("--expected-robolab-commit", default="0aef241fb088ca21bb4ebd24448940ed56620d17")
BOOTSTRAP.add_argument("--minimum-visible-target-pixels", type=int, default=32)
BOOTSTRAP.add_argument("--request0-replay-amendment", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-replay-amendment-sha256", required=True)
BOOTSTRAP.add_argument("--live-orientation-tolerance-amendment", type=Path, required=True)
BOOTSTRAP.add_argument("--live-orientation-tolerance-amendment-sha256", required=True)
BOOTSTRAP.add_argument("--request0-mode", choices=("capture_left", "replay_right"), required=True)
BOOTSTRAP.add_argument("--request0-observation-cache", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-observation-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-reset-contract", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-native-reset-contract", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-attestation", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-observation-cache-sha256")
BOOTSTRAP.add_argument("--request0-observation-manifest-sha256")
BOOTSTRAP.add_argument("--request0-reset-contract-sha256")
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
sys.path.insert(0, str(study_root))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.live_snapshot_adapter import (  # noqa: E402
    ModelBlindLiveGateAdapter,
    bind_camera_row,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.occlusion import (  # noqa: E402
    project_world_target_to_pixel,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (  # noqa: E402
    load_runtime_bundle,
    sha256_file,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.request0_replay import (  # noqa: E402
    build_reset_contract,
    capture_left_observation,
    evidence_envelope,
    load_amendment,
    replay_left_observation_for_right,
    write_capture_attestation,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.r002_orientation_tolerance import (  # noqa: E402
    build_runtime_attestation as build_r002_runtime_attestation,
    load_amendment as load_r002_amendment,
)


bundle = load_runtime_bundle(
    registration_path=bootstrap.registration,
    registration_sha256=bootstrap.registration_sha256,
    queue_path=bootstrap.queue,
    queue_sha256=bootstrap.queue_sha256,
    candidate_path=bootstrap.candidate,
    candidate_sha256=bootstrap.candidate_sha256,
)
cell = bundle.cell(bootstrap.cell_id)
if cell.row["arena"] != "droid_robolab" or cell.row["execution_mode"] != "new_behavioral_episode":
    BOOTSTRAP.error("standalone live gate accepts only new E004 DROID cells")
load_amendment(bootstrap.request0_replay_amendment, bootstrap.request0_replay_amendment_sha256)
r002_amendment = load_r002_amendment(
    bootstrap.live_orientation_tolerance_amendment,
    bootstrap.live_orientation_tolerance_amendment_sha256,
    registration_sha256=bundle.registration_sha256,
    queue_sha256=bundle.queue_sha256,
    candidate_sha256=bundle.candidate_sha256,
)
r002_attestation = build_r002_runtime_attestation(
    amendment=r002_amendment,
    amendment_path=bootstrap.live_orientation_tolerance_amendment,
    amendment_sha256=bootstrap.live_orientation_tolerance_amendment_sha256,
    control_scene_asset=bootstrap.control_scene_asset,
    paired_scene_asset=bootstrap.paired_scene_asset,
    symmetry_level_s=cell.symmetry_level_s,
)
if (cell.relation, bootstrap.request0_mode) not in {
    ("left", "capture_left"),
    ("right", "replay_right"),
}:
    BOOTSTRAP.error("request-zero preflight mode differs from registered relation")
if bootstrap.request0_mode == "capture_left":
    if bootstrap.request0_native_reset_contract.resolve() != bootstrap.request0_reset_contract.resolve():
        BOOTSTRAP.error("LEFT preflight native reset contract must equal the retained LEFT contract")
    for path in (
        bootstrap.request0_observation_cache,
        bootstrap.request0_observation_manifest,
        bootstrap.request0_reset_contract,
        bootstrap.request0_attestation,
    ):
        if path.exists():
            BOOTSTRAP.error(f"refusing to overwrite request-zero preflight evidence: {path}")
else:
    if bootstrap.request0_native_reset_contract.resolve() == bootstrap.request0_reset_contract.resolve():
        BOOTSTRAP.error("RIGHT preflight native reset contract must be retained separately from LEFT")
    for label, path, expected in (
        ("observation cache", bootstrap.request0_observation_cache, bootstrap.request0_observation_cache_sha256),
        ("observation manifest", bootstrap.request0_observation_manifest, bootstrap.request0_observation_manifest_sha256),
        ("LEFT reset contract", bootstrap.request0_reset_contract, bootstrap.request0_reset_contract_sha256),
    ):
        if not expected or not path.is_file() or sha256_file(path) != expected:
            BOOTSTRAP.error(f"RIGHT request-zero preflight {label} binding is missing or changed")
    for path in (bootstrap.request0_native_reset_contract, bootstrap.request0_attestation):
        if path.exists():
            BOOTSTRAP.error(f"refusing to overwrite RIGHT request-zero preflight evidence: {path}")
try:
    scene_mapping = json.loads(bootstrap.scene_object_mapping)
except json.JSONDecodeError as exc:
    BOOTSTRAP.error(f"scene-object mapping is invalid JSON: {exc}")

os.environ.update(
    {
        "VLA_WAM_V3E004_FIXTURE_CANDIDATE": str(bundle.candidate_path),
        "VLA_WAM_V3E004_FIXTURE_SHA256": bundle.candidate_sha256,
        "VLA_WAM_V3E004_SYMMETRY_LEVEL_S": str(cell.symmetry_level_s),
        "VLA_WAM_V3E004_CONTROL_SCENE_ASSET": str(bootstrap.control_scene_asset.resolve()),
        "VLA_WAM_V3E004_PAIRED_SCENE_ASSET": str(bootstrap.paired_scene_asset.resolve()),
        "VLA_WAM_V3E004_SCENE_OBJECT_MAPPING": json.dumps(scene_mapping, sort_keys=True),
    }
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1 or not args_cli.headless:
    parser.error("E004 live gate requires one headless environment")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("E004 live gate requires realtime/balanced RTX")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
import robolab  # noqa: E402
import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


CAMERAS = ("head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")


def _host(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [float(item) for item in value]


def _quat_rotate_wxyz(quaternion: list[float], vector: list[float]) -> list[float]:
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (item / norm for item in (w, x, y, z))
    vx, vy, vz = vector
    ix = w * vx + y * vz - z * vy
    iy = w * vy + z * vx - x * vz
    iz = w * vz + x * vy - y * vx
    iw = -x * vx - y * vy - z * vz
    return [
        ix * w + iw * -x + iy * -z - iz * -y,
        iy * w + iw * -y + iz * -x - ix * -z,
        iz * w + iw * -z + ix * -y - iy * -x,
    ]


def _yaw_wxyz(quaternion: list[float]) -> float:
    w, x, y, z = quaternion
    norm = math.sqrt(sum(item * item for item in quaternion))
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _physical_prim_path(asset: Any) -> str:
    for owner_name in ("root_physx_view", "_root_physx_view"):
        owner = getattr(asset, owner_name, None)
        paths = getattr(owner, "prim_paths", None)
        if paths:
            return str(paths[0])
    cfg = getattr(asset, "cfg", None)
    raw = str(getattr(cfg, "prim_path", ""))
    if "{ENV_REGEX_NS}" in raw:
        return raw.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    if raw:
        return raw
    raise RuntimeError("could not resolve bowl USD prim path")


def _reference_bounds_world(env: Any, physical_name: str) -> dict[str, Any]:
    asset = env.scene[physical_name]
    prim_path = _physical_prim_path(asset)
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"invalid bowl prim path: {prim_path}")
    local_bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(prim)
    local_range = local_bound.ComputeAlignedRange()
    minimum, maximum = local_range.GetMin(), local_range.GetMax()
    center_local = [float((minimum[index] + maximum[index]) * 0.5) for index in range(3)]
    half_extents = [float((maximum[index] - minimum[index]) * 0.5) for index in range(3)]
    if not all(math.isfinite(value) and value > 0 for value in half_extents):
        raise RuntimeError("bowl USD local bound is invalid")
    position = _host(asset.data.root_pos_w[0])
    quaternion = _host(asset.data.root_quat_w[0])
    rotated = _quat_rotate_wxyz(quaternion, center_local)
    return {
        "center_world_m": [position[index] + rotated[index] for index in range(3)],
        "half_extents_m": half_extents,
        "yaw_world_rad": _yaw_wxyz(quaternion),
        "usd_prim_path": prim_path,
    }


def _sensor(env: Any, name: str) -> Any:
    try:
        return env.scene[name]
    except (KeyError, TypeError):
        sensors = getattr(env.scene, "sensors", {})
        if name in sensors:
            return sensors[name]
        raise RuntimeError(f"live camera sensor is unavailable: {name}")


def _rgb(obs: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy(), dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.ptp(value):
        raise RuntimeError(f"blank or malformed live RGB view: {name}")
    return value


def _camera_rows(env: Any, obs: Mapping[str, Any], target_physical: str, reference_physical: str) -> dict[str, dict[str, Any]]:
    target_center = _host(env.scene[target_physical].data.root_pos_w[0])
    bounds = _reference_bounds_world(env, reference_physical)
    bounds = {key: value for key, value in bounds.items() if key != "usd_prim_path"}
    rows: dict[str, dict[str, Any]] = {}
    for name in CAMERAS:
        sensor = _sensor(env, name)
        data = sensor.data
        center = _host(data.pos_w[0])
        quaternion = _host(data.quat_w_ros[0])
        intrinsic_raw = data.intrinsic_matrices[0]
        if hasattr(intrinsic_raw, "detach"):
            intrinsic_raw = intrinsic_raw.detach().cpu().tolist()
        intrinsic = [[float(item) for item in row] for row in intrinsic_raw]
        frame = _rgb(obs, name)
        row = {
            "camera_center_world_m": center,
            "camera_quaternion_world_wxyz_ros": quaternion,
            "target_center_world_m": target_center,
            "intrinsic_matrix_3x3": intrinsic,
            "image_size_wh": [int(frame.shape[1]), int(frame.shape[0])],
            "reference_bounds_world": bounds,
            "target_instance_visible_pixels": None,
            "segmentation_source_sha256": None,
            "rgb_source_sha256": hashlib.sha256(frame.tobytes(order="C")).hexdigest(),
            "rgb_source_shape": list(frame.shape),
            "rgb_source_dtype": str(frame.dtype),
        }
        row["target_projected_pixel_uv"] = list(
            project_world_target_to_pixel(
                camera_center_world_m=center,
                camera_quaternion_world_wxyz_ros=quaternion,
                target_center_world_m=target_center,
                intrinsic_matrix_3x3=intrinsic,
            )
        )
        rows[name] = bind_camera_row(row)
    return rows


def _hold_action(obs: Mapping[str, Any], device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, 8):
        raise RuntimeError(f"hold action is not [1,8]: {tuple(action.shape)}")
    return action


def _write_video(path: Path, obs: Mapping[str, Any]) -> None:
    frames = [_rgb(obs, name) for name in CAMERAS]
    height = min(frame.shape[0] for frame in frames)
    resized = [cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height)) for frame in frames]
    montage = np.concatenate(resized, axis=1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (montage.shape[1], montage.shape[0]))
    if not writer.isOpened():
        raise RuntimeError("RTX gate video writer did not open")
    try:
        for _ in range(4):
            writer.write(cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    ok, decoded = capture.read()
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not ok or decoded is None or count < 1:
        raise RuntimeError("RTX gate video does not decode")


def main() -> None:
    output_dir = bootstrap.gate_output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite model-blind gate: {output_dir}")
    for path in (bootstrap.control_scene_asset, bootstrap.paired_scene_asset):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"scene asset is missing: {path}")
    robolab_root = bootstrap.robolab_root.resolve()
    study_commit = subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip()
    robolab_commit = subprocess.check_output(["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True).strip()
    if study_commit != bootstrap.expected_study_commit or robolab_commit != bootstrap.expected_robolab_commit:
        raise RuntimeError("study or RoboLab revision differs from the gate invocation")
    if not Path(robolab.__file__).resolve().is_relative_to(robolab_root):
        raise RuntimeError("effective RoboLab import is outside the pinned worktree")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if bootstrap.gpu_uuid not in {row.strip() for row in gpu_rows}:
        raise RuntimeError("assigned GPU UUID is not live-visible")
    output_dir.mkdir(parents=True)
    set_output_dir(str(output_dir / "native"))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    task_root = study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/task_files"
    auto_register_droid_envs(task=[str(task_root / "left.py"), str(task_root / "right.py")], cameras=WRIST_LEFT_RIGHT_HEAD)
    task_name = "V3E004DroidLeftTask" if cell.relation == "left" else "V3E004DroidRightTask"
    env, env_cfg = create_env(
        task_name,
        device=args_cli.device,
        seed=cell.environment_seed,
        num_envs=1,
        instruction_type="default",
        policy="v3e004_model_blind_live_gate",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    try:
        obs, _ = env.reset()
        hold = _hold_action(obs, env.device)
        for _ in range(60):
            obs, _, terminated, truncated, _ = env.step(hold)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("task terminated during 60-step settle")
        expected = bundle.cell(cell.cell_id).row
        logical_names = bundle.registration["layout"]
        del expected, logical_names  # identity was already hash-checked; use candidate inventory below.
        candidate = __import__(
            "experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract",
            fromlist=["load_candidate"],
        ).load_candidate(bundle.candidate_path, bundle.candidate_sha256)
        physical_names = [scene_mapping[name] for name in candidate.layout(cell.symmetry_level_s)]
        stability = {name: {"linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0} for name in physical_names}
        for _ in range(15):
            obs, _, terminated, truncated, _ = env.step(hold)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("task terminated during 15-step stability window")
            world = get_world(env)
            for name in physical_names:
                velocity = _host(world.get_velocity(name, env_id=0))
                stability[name]["linear_speed_m_s"] = max(stability[name]["linear_speed_m_s"], max(abs(value) for value in velocity[:3]))
                stability[name]["angular_speed_rad_s"] = max(stability[name]["angular_speed_rad_s"], max(abs(value) for value in velocity[3:]))
        left = bool(object_left_of(env, object=scene_mapping["rubiks_cube"], reference_object=scene_mapping["bowl"], frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
        right = bool(object_right_of(env, object=scene_mapping["rubiks_cube"], reference_object=scene_mapping["bowl"], frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
        if left or right:
            raise RuntimeError("reset is not neutral under the frozen LEFT/RIGHT predicates")
        video = output_dir / "live_rtx_camera_montage.mp4"
        _write_video(video, obs)
        adapter = ModelBlindLiveGateAdapter(
            bundle=bundle,
            cell=cell,
            snapshot_path=output_dir / "live_scene_snapshot.json",
            gate_path=output_dir / "live_scene_gate.json",
            minimum_visible_target_pixels=bootstrap.minimum_visible_target_pixels,
            orientation_tolerance_attestation=r002_attestation,
        )
        camera_rows = _camera_rows(env, obs, scene_mapping["rubiks_cube"], scene_mapping["bowl"])
        gate = adapter.capture_and_compile(
            env=env,
            observation=obs,
            scene_object_mapping=scene_mapping,
            camera_rows=camera_rows,
            settle_stability={"settle_steps": 60, "stability_window_steps": 15, "maxima_by_object": stability},
        )
        reset_contract = build_reset_contract(
            env=env,
            physical_object_names=physical_names,
            camera_rows=camera_rows,
            observation=obs,
        )
        if bootstrap.request0_mode == "capture_left":
            request0 = capture_left_observation(
                observation=obs,
                reset_contract=reset_contract,
                amendment_path=bootstrap.request0_replay_amendment,
                amendment_sha256=bootstrap.request0_replay_amendment_sha256,
                cell_id=cell.cell_id,
                matched_pair_id=cell.matched_pair_id,
                cache_path=bootstrap.request0_observation_cache,
                manifest_path=bootstrap.request0_observation_manifest,
                reset_contract_path=bootstrap.request0_reset_contract,
            )
            write_capture_attestation(
                amendment_path=bootstrap.request0_replay_amendment,
                amendment_sha256=bootstrap.request0_replay_amendment_sha256,
                cell_id=cell.cell_id,
                matched_pair_id=cell.matched_pair_id,
                cache_path=bootstrap.request0_observation_cache,
                manifest_path=bootstrap.request0_observation_manifest,
                reset_contract_path=bootstrap.request0_reset_contract,
                observation_payload_sha256=request0["observation_payload_sha256"],
                reset_contract_payload_sha256=request0["reset_contract"]["payload_sha256"],
                attestation_path=bootstrap.request0_attestation,
            )
            observation_payload_sha = request0["observation_payload_sha256"]
            reset_payload_sha = request0["reset_contract"]["payload_sha256"]
            native_reset_path = bootstrap.request0_reset_contract
        else:
            _, request0 = replay_left_observation_for_right(
                native_observation=obs,
                native_reset_contract=reset_contract,
                amendment_path=bootstrap.request0_replay_amendment,
                amendment_sha256=bootstrap.request0_replay_amendment_sha256,
                cell_id=cell.cell_id,
                matched_pair_id=cell.matched_pair_id,
                cache_path=bootstrap.request0_observation_cache,
                cache_sha256=str(bootstrap.request0_observation_cache_sha256),
                manifest_path=bootstrap.request0_observation_manifest,
                manifest_sha256=str(bootstrap.request0_observation_manifest_sha256),
                reset_contract_path=bootstrap.request0_reset_contract,
                reset_contract_file_sha256=str(bootstrap.request0_reset_contract_sha256),
                native_reset_contract_path=bootstrap.request0_native_reset_contract,
                attestation_path=bootstrap.request0_attestation,
            )
            observation_payload_sha = request0["request0_observation_payload_sha256"]
            reset_payload_sha = request0["right_reset_contract_sha256"]
            native_reset_path = bootstrap.request0_native_reset_contract
        request0_evidence = evidence_envelope(
            mode=bootstrap.request0_mode,
            amendment_path=bootstrap.request0_replay_amendment,
            cache_path=bootstrap.request0_observation_cache,
            manifest_path=bootstrap.request0_observation_manifest,
            reset_contract_path=bootstrap.request0_reset_contract,
            native_reset_contract_path=native_reset_path,
            attestation_path=bootstrap.request0_attestation,
            observation_payload_sha256=observation_payload_sha,
            reset_contract_payload_sha256=reset_payload_sha,
        )
        report = {
            "schema_version": "vla-wam-shared-v3e004-standalone-model-blind-droid-gate-v2",
            "study_id": cell.row["study_id"],
            "amendment_id": cell.row["amendment_id"],
            "status": "passed_model_blind_preflight_not_a_behavioral_release",
            "passed": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "behavioral_action_count": 0,
            "fixture_setup_hold_action_count": 75,
            "registered_cell_id": cell.cell_id,
            "matched_pair_id": cell.matched_pair_id,
            "model_id": cell.model_id,
            "registration_sha256": bundle.registration_sha256,
            "queue_sha256": bundle.queue_sha256,
            "candidate_sha256": bundle.candidate_sha256,
            "pod": bootstrap.pod,
            "pod_uid": bootstrap.pod_uid,
            "gpu_uuid": bootstrap.gpu_uuid,
            "study_commit": study_commit,
            "robolab_commit": robolab_commit,
            "task_prompt": env_cfg.instruction,
            "live_scene_gate": {"path": gate["gate_path"], "sha256": gate["gate_sha256"], "bytes": Path(gate["gate_path"]).stat().st_size},
            "viewport_video": {"path": str(video), "sha256": sha256_file(video), "bytes": video.stat().st_size},
            "request0_replay": request0_evidence,
            "live_orientation_realisation_tolerance_amendment": r002_attestation,
            "release_boundary": "Behavioral bridges must repeat this gate in the same simulator process immediately before model request zero.",
        }
        report_path = output_dir / "model_blind_gate_report.json"
        report_path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), "report_sha256": sha256_file(report_path), "passed": True}, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
