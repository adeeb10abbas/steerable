#!/usr/bin/env python3
"""Execute one hash-released V3-C002 DROID behavioral cell.

The bridge owns one Isaac process and one registered cell.  It recreates the
graded object layout, performs the 60+15 model-blind settle/camera/reset gate
inside the live environment, and only then permits request zero.  The policy
clients retain their native action/future contracts; this file only supplies
the C002 authorization and identity-complete raw simulator export.  The
registered physical goal is read only from queue metadata, never from text.
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
BOOTSTRAP.add_argument("--runtime-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--runtime-manifest-sha256", required=True)
BOOTSTRAP.add_argument("--authorization-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--authorization-gate-sha256", required=True)
BOOTSTRAP.add_argument("--authorization-mode", choices=("model_blind_preflight", "excluded_smoke", "behavioral"), required=True)
BOOTSTRAP.add_argument("--cell-id", required=True)
BOOTSTRAP.add_argument("--lane-pod-uid", required=True)
BOOTSTRAP.add_argument("--lane-gpu-uuid", required=True)
BOOTSTRAP.add_argument("--policy-server-pod-uid", required=True)
BOOTSTRAP.add_argument("--policy-server-gpu-uuid", required=True)
BOOTSTRAP.add_argument("--lane-id", required=True)
BOOTSTRAP.add_argument("--raw-root", required=True)
BOOTSTRAP.add_argument("--container-identity", required=True)
BOOTSTRAP.add_argument("--runtime-identity", required=True)
BOOTSTRAP.add_argument("--server-process-identity", required=True)
BOOTSTRAP.add_argument("--server-lock-identity", required=True)
BOOTSTRAP.add_argument("--live-snapshot", type=Path, required=True)
BOOTSTRAP.add_argument("--live-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--simulator-export", type=Path, required=True)
BOOTSTRAP.add_argument("--raw-event-stream", type=Path, required=True)
BOOTSTRAP.add_argument("--state-capture-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--action-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--future-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--model-endpoint-host", required=True)
BOOTSTRAP.add_argument("--model-endpoint-port", type=int, required=True)
BOOTSTRAP.add_argument("--control-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--paired-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--scene-object-mapping", required=True)
BOOTSTRAP.add_argument("--expected-study-commit", required=True)
BOOTSTRAP.add_argument(
    "--expected-robolab-commit",
    default="0aef241fb088ca21bb4ebd24448940ed56620d17",
)
BOOTSTRAP.add_argument("--minimum-visible-target-pixels", type=int, default=32)
BOOTSTRAP.add_argument("--request0-replay-amendment", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-replay-amendment-sha256", required=True)
BOOTSTRAP.add_argument("--live-orientation-tolerance-amendment", type=Path, required=True)
BOOTSTRAP.add_argument("--live-orientation-tolerance-amendment-sha256", required=True)
BOOTSTRAP.add_argument("--request0-mode", choices=("capture_block", "replay_block"), required=True)
BOOTSTRAP.add_argument("--request0-observation-cache", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-observation-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-reset-contract", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-native-reset-contract", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-replay-attestation", type=Path, required=True)
BOOTSTRAP.add_argument("--request0-observation-cache-sha256")
BOOTSTRAP.add_argument("--request0-observation-manifest-sha256")
BOOTSTRAP.add_argument("--request0-reset-contract-sha256")
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
robolab_root = bootstrap.robolab_root.resolve()
for root in (study_root, robolab_root):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (  # noqa: E402
    AMENDMENT_ID,
    MODEL_ID,
    STUDY_ID,
    ContractError,
    load_cells,
    require_model_blind_preflight_authorization,
    require_released_gate,
    require_smoke_authorization,
    sha256_file,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import (  # noqa: E402
    validate_runtime_manifest,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.droid_behavioral_contract import (  # noqa: E402
    MODEL_SPECS,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.episode_compiler import (  # noqa: E402
    _cone as frozen_cone,
    _failure_category as frozen_failure_category,
    _normalize_steps as frozen_normalize_steps,
    frozen_requested_success,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.live_snapshot_adapter import (  # noqa: E402
    ModelBlindLiveGateAdapter,
    bind_camera_row,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import (  # noqa: E402
    load_candidate,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.occlusion import (  # noqa: E402
    project_world_target_to_pixel,
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


if sha256_file(bootstrap.registration) != bootstrap.registration_sha256:
    BOOTSTRAP.error("C002 registration SHA-256 mismatch")
if sha256_file(bootstrap.queue) != bootstrap.queue_sha256:
    BOOTSTRAP.error("C002 queue SHA-256 mismatch")
registration, cells = load_cells(registration_path=bootstrap.registration, queue_path=bootstrap.queue)
matches = [item for item in cells if item.cell_id == bootstrap.cell_id]
if len(matches) != 1:
    BOOTSTRAP.error("C002 cell is absent or duplicated")
cell = matches[0]
if cell.row["arena"] != "droid_robolab" or cell.row["execution_mode"] != "new_behavioral_episode":
    BOOTSTRAP.error("behavioral adapter accepts only new C002 DROID cells")
if not bootstrap.candidate.is_file() or sha256_file(bootstrap.candidate) != bootstrap.candidate_sha256:
    BOOTSTRAP.error("E004 s=1 candidate binding changed")
if bootstrap.candidate_sha256 != registration["e004_s1_layout"]["candidate_sha256"]:
    BOOTSTRAP.error("candidate differs from the C002 registration")
exact = registration["exact_e004_pi05_runtime"]
spec = MODEL_SPECS[MODEL_ID]
for key, observed in (("action_dim", spec.action_dim), ("action_horizon", spec.action_horizon), ("action_cap", spec.action_cap)):
    if exact["identity_values"][key] != observed:
        BOOTSTRAP.error(f"exact pi05 contract differs for {key}")


class _Bundle:
    def __init__(self) -> None:
        self.registration_path = bootstrap.registration.resolve()
        self.registration_sha256 = bootstrap.registration_sha256
        self.registration = registration
        self.queue_path = bootstrap.queue.resolve()
        self.queue_sha256 = bootstrap.queue_sha256
        self.candidate_path = bootstrap.candidate.resolve()
        self.candidate_sha256 = bootstrap.candidate_sha256

    def cell(self, cell_id: str) -> Any:
        if cell_id != cell.cell_id:
            raise ContractError("cell lookup changed after registration")
        return cell


bundle = _Bundle()


if bootstrap.authorization_mode == "behavioral":
    if not bootstrap.authorization_gate.is_file() or sha256_file(bootstrap.authorization_gate) != bootstrap.authorization_gate_sha256:
        raise ContractError("behavioral release-gate binding changed")
    authorization = require_released_gate(
        registration_path=bootstrap.registration,
        queue_path=bootstrap.queue,
        release_gate_path=bootstrap.authorization_gate,
    )[2]
elif bootstrap.authorization_mode == "excluded_smoke":
    if not bootstrap.authorization_gate.is_file() or sha256_file(bootstrap.authorization_gate) != bootstrap.authorization_gate_sha256:
        raise ContractError("excluded-smoke authorization binding changed")
    authorization = require_smoke_authorization(
        registration_path=bootstrap.registration,
        queue_path=bootstrap.queue,
        authorization_path=bootstrap.authorization_gate,
    )[2]
else:
    if not bootstrap.authorization_gate.is_file() or sha256_file(bootstrap.authorization_gate) != bootstrap.authorization_gate_sha256:
        raise ContractError("model-blind source-push gate binding changed")
    authorization = require_model_blind_preflight_authorization(
        registration_path=bootstrap.registration,
        queue_path=bootstrap.queue,
        source_push_gate_path=bootstrap.authorization_gate,
    )[2]
runtime_manifest = validate_runtime_manifest(
    bootstrap.runtime_manifest,
    bootstrap.runtime_manifest_sha256,
    registration_path=bootstrap.registration,
    queue_path=bootstrap.queue,
    pod_uid=bootstrap.lane_pod_uid,
    gpu_uuid=bootstrap.lane_gpu_uuid,
)
for key, expected_value in (
    ("simulator_pod_uid", bootstrap.lane_pod_uid),
    ("simulator_gpu_uuid", bootstrap.lane_gpu_uuid),
    ("policy_server_pod_uid", bootstrap.policy_server_pod_uid),
    ("policy_server_gpu_uuid", bootstrap.policy_server_gpu_uuid),
    ("server_port", bootstrap.model_endpoint_port),
    ("lane_id", bootstrap.lane_id),
    ("raw_root", bootstrap.raw_root),
    ("container_identity", bootstrap.container_identity),
    ("runtime_identity", bootstrap.runtime_identity),
    ("server_process_identity", bootstrap.server_process_identity),
    ("server_lock_identity", bootstrap.server_lock_identity),
):
    if runtime_manifest["runtime_identity"].get(key) != expected_value:
        BOOTSTRAP.error(f"lane runtime binding differs for {key}")
try:
    scene_mapping = json.loads(bootstrap.scene_object_mapping)
except json.JSONDecodeError as exc:
    BOOTSTRAP.error(f"scene-object mapping is invalid JSON: {exc}")
if not isinstance(scene_mapping, dict):
    BOOTSTRAP.error("scene-object mapping must be an object")
if scene_mapping.get("rubiks_cube") != "rubiks_cube" or scene_mapping.get("bowl") != "bowl":
    BOOTSTRAP.error("the frozen DROID predicate requires identity mapping for cube and bowl")

for path in (bootstrap.control_scene_asset, bootstrap.paired_scene_asset, bootstrap.runtime_manifest):
    if not path.is_file() or path.stat().st_size <= 0:
        BOOTSTRAP.error(f"required live input is missing: {path}")
r002_amendment = load_r002_amendment(
    bootstrap.live_orientation_tolerance_amendment,
    bootstrap.live_orientation_tolerance_amendment_sha256,
    registration_sha256=registration["source_bindings"]["artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/registration.json"]["sha256"],
    queue_sha256=registration["source_bindings"]["artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/queue.jsonl"]["sha256"],
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
load_amendment(
    bootstrap.request0_replay_amendment,
    bootstrap.request0_replay_amendment_sha256,
)
expected_first_condition = cell.row["execution_order"][0]
if (cell.condition == expected_first_condition) != (bootstrap.request0_mode == "capture_block"):
    BOOTSTRAP.error("only the first registered condition may capture the block request-zero observation")
if bootstrap.request0_mode == "capture_block":
    if bootstrap.request0_native_reset_contract.resolve() != bootstrap.request0_reset_contract.resolve():
        BOOTSTRAP.error("block-source native reset contract must be the retained reset contract")
    for path in (
        bootstrap.request0_observation_cache,
        bootstrap.request0_observation_manifest,
        bootstrap.request0_reset_contract,
        bootstrap.request0_native_reset_contract,
        bootstrap.request0_replay_attestation,
    ):
        if path.exists():
            BOOTSTRAP.error(f"refusing to overwrite request-zero evidence: {path}")
else:
    if bootstrap.request0_native_reset_contract.resolve() == bootstrap.request0_reset_contract.resolve():
        BOOTSTRAP.error("RIGHT native reset contract must be retained separately from LEFT")
    for name, path, expected in (
        ("observation cache", bootstrap.request0_observation_cache, bootstrap.request0_observation_cache_sha256),
        ("observation manifest", bootstrap.request0_observation_manifest, bootstrap.request0_observation_manifest_sha256),
        ("reset contract", bootstrap.request0_reset_contract, bootstrap.request0_reset_contract_sha256),
    ):
        if not expected or not path.is_file() or sha256_file(path) != expected:
            BOOTSTRAP.error(f"block replay {name} binding is missing or changed")
    if bootstrap.request0_replay_attestation.exists():
        BOOTSTRAP.error(f"refusing to overwrite block replay attestation: {bootstrap.request0_replay_attestation}")
    if bootstrap.request0_native_reset_contract.exists():
        BOOTSTRAP.error(
            f"refusing to overwrite native block-replay reset contract: {bootstrap.request0_native_reset_contract}"
        )
for path in (bootstrap.live_snapshot, bootstrap.live_gate, bootstrap.simulator_export, bootstrap.raw_event_stream):
    if path.exists():
        BOOTSTRAP.error(f"refusing to overwrite retained evidence: {path}")
for directory in (
    bootstrap.state_capture_dir,
    bootstrap.action_trace_dir,
    bootstrap.future_trace_dir,
    bootstrap.output_dir,
):
    if directory.exists():
        BOOTSTRAP.error(f"refusing to reuse retained output directory: {directory}")

os.environ.update(
    {
        "VLA_WAM_V3E004_FIXTURE_CANDIDATE": str(bundle.candidate_path),
        "VLA_WAM_V3E004_FIXTURE_SHA256": bundle.candidate_sha256,
        "VLA_WAM_V3E004_SYMMETRY_LEVEL_S": str(cell.symmetry_level_s),
        "VLA_WAM_V3E004_CONTROL_SCENE_ASSET": str(bootstrap.control_scene_asset.resolve()),
        "VLA_WAM_V3E004_PAIRED_SCENE_ASSET": str(bootstrap.paired_scene_asset.resolve()),
        "VLA_WAM_V3E004_SCENE_OBJECT_MAPPING": json.dumps(scene_mapping, sort_keys=True),
        "VLA_WAM_V3C002_PROMPT": cell.row["prompt"],
        "VLA_WAM_V3C002_PROMPT_UTF8_HEX": cell.row["prompt_utf8_hex"],
    }
)

import cv2  # noqa: E402,F401 -- RoboLab requires OpenCV before Isaac Lab
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.output_folder_name = str(bootstrap.output_dir.resolve())
if args_cli.video_mode != "viewport" or args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("E004 requires one environment, one run, and viewport video")
if args_cli.enable_subtask or args_cli.instruction_type != "default":
    parser.error("E004 permits only the static registered prompt and no subtask coach")
if not args_cli.headless or args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("E004 requires headless realtime/balanced RTX rendering")
if args_cli.device != "cuda:0":
    parser.error("one E004 Isaac process owns exactly cuda:0 inside its GPU-scoped pod")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
import robolab  # noqa: E402
import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
import robolab.core.environments.runtime as robolab_runtime  # noqa: E402
from robolab.core.task.conditionals import (  # noqa: E402
    object_dropped,
    object_grabbed,
    object_left_of,
    object_right_of,
)
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


CAMERAS = ("head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
_candidate_value = load_candidate(bundle.candidate_path, bundle.candidate_sha256)
PHYSICAL_OBJECTS = tuple(
    str(scene_mapping[name])
    for name in sorted(_candidate_value.layout(cell.symmetry_level_s))
)


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing retained artifact: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_new_json(path: Path, value: Any) -> Path:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _host(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [float(item) for item in value]


def _inverse_rotate_wxyz(quaternion: list[float], vector: list[float]) -> list[float]:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("robot root quaternion is invalid")
    q /= norm
    w, xyz = float(q[0]), q[1:]
    inverse = -xyz
    result = 2 * np.dot(inverse, v) * inverse + (w * w - np.dot(inverse, inverse)) * v + 2 * w * np.cross(inverse, v)
    return [float(item) for item in result]


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
    raw = str(getattr(getattr(asset, "cfg", None), "prim_path", ""))
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
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(prim)
    aligned = bound.ComputeAlignedRange()
    minimum, maximum = aligned.GetMin(), aligned.GetMax()
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


def _camera_rows(env: Any, obs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    target = str(scene_mapping["rubiks_cube"])
    reference = str(scene_mapping["bowl"])
    target_center = _host(env.scene[target].data.root_pos_w[0])
    bounds = _reference_bounds_world(env, reference)
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


def _retain_policy_camera_images(obs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    directory = bootstrap.state_capture_dir / "request0_policy_cameras"
    if directory.exists():
        raise FileExistsError(f"refusing to reuse retained policy-camera directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, Any]] = {}
    for name in CAMERAS:
        path = directory / f"{name}.npy"
        np.save(path, _rgb(obs, name), allow_pickle=False)
        records[name] = _file_record(path)
    return records


def _write_model_blind_camera_video(obs: Mapping[str, Any]) -> Path:
    path = bootstrap.state_capture_dir / "model_blind_policy_camera_montage.mp4"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite model-blind renderer proof: {path}")
    frames = [_rgb(obs, name) for name in CAMERAS]
    height = min(frame.shape[0] for frame in frames)
    resized = [cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height)) for frame in frames]
    montage = np.concatenate(resized, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (montage.shape[1], montage.shape[0]))
    if not writer.isOpened():
        raise RuntimeError("model-blind RTX video writer did not open")
    try:
        for _ in range(4):
            writer.write(cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    okay, decoded = capture.read()
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not okay or decoded is None or count < 1:
        raise RuntimeError("model-blind RTX renderer proof does not decode")
    return path


def _hold_action(obs: Mapping[str, Any], device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, 8):
        raise RuntimeError(f"hold action is not [1,8]: {tuple(action.shape)}")
    return action


class StateCaptureProxy:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.samples: list[dict[str, Any]] = []
        self.runner_resets = 0
        self.physical_resets = 0
        self.cached_reset: tuple[Any, Any] | None = None
        self.started = time.monotonic()
        self.capture_path = bootstrap.state_capture_dir / "state_capture.json"
        self.partial_path = bootstrap.state_capture_dir / "states.partial.jsonl"
        self.adapter = ModelBlindLiveGateAdapter(
            bundle=bundle,
            cell=cell,
            snapshot_path=bootstrap.live_snapshot,
            gate_path=bootstrap.live_gate,
            minimum_visible_target_pixels=bootstrap.minimum_visible_target_pixels,
            orientation_tolerance_attestation=r002_attestation,
        )
        self.initial_state_sha256: str | None = None
        self.native_reset_snapshot_sha256: str | None = None
        self.request0_pair_identity_sha256: str | None = None
        self.request0_evidence: dict[str, Any] | None = None
        self.policy_camera_images: dict[str, dict[str, Any]] | None = None
        self.model_blind_camera_video: Path | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> dict[str, Any]:
        target = self._env.scene[str(scene_mapping["rubiks_cube"])].data.root_pos_w[0]
        reference = self._env.scene[str(scene_mapping["bowl"])].data.root_pos_w[0]
        robot = self._env.scene["robot"].data
        robot_position = _host(robot.root_pos_w[0])
        robot_quaternion = _host(robot.root_quat_w[0])
        target_relative = [a - b for a, b in zip(_host(target), robot_position)]
        reference_relative = [a - b for a, b in zip(_host(reference), robot_position)]
        return {
            "action_step": action_step,
            "object_xyz": _inverse_rotate_wxyz(robot_quaternion, target_relative),
            "reference_xyz": _inverse_rotate_wxyz(robot_quaternion, reference_relative),
            "grippers_open": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
            "object_grabbed": bool(object_grabbed(self._env, object="rubiks_cube", env_id=0)),
        }

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if len(self.samples) > 1:
            raise RuntimeError("reset attempted after C002 behavior began")
        self.runner_resets += 1
        if self.runner_resets == 2:
            if self.cached_reset is None or self.physical_resets != 1 or not bootstrap.live_gate.is_file():
                raise RuntimeError("duplicate runner reset preceded a completed live gate")
            return self.cached_reset
        if self.runner_resets != 1:
            raise RuntimeError("frozen RoboLab runner must reset exactly twice before behavior")
        self.physical_resets += 1
        result = self._env.reset(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("RoboLab reset did not return (observation, info)")
        obs, info = result
        hold = _hold_action(obs, self._env.device)
        for _ in range(60):
            obs, _, terminated, truncated, _ = self._env.step(hold)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("cell terminated during the 60-step settle")
        stability = {
            name: {"linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0}
            for name in PHYSICAL_OBJECTS
        }
        for _ in range(15):
            obs, _, terminated, truncated, _ = self._env.step(hold)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("cell terminated during the 15-step stability window")
            world = get_world(self._env)
            for name in PHYSICAL_OBJECTS:
                velocity = _host(world.get_velocity(name, env_id=0))
                stability[name]["linear_speed_m_s"] = max(
                    stability[name]["linear_speed_m_s"], max(abs(value) for value in velocity[:3])
                )
                stability[name]["angular_speed_rad_s"] = max(
                    stability[name]["angular_speed_rad_s"], max(abs(value) for value in velocity[3:])
                )
        left = bool(object_left_of(self._env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
        right = bool(object_right_of(self._env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
        if left or right:
            raise RuntimeError("live reset is not neutral under the frozen LEFT/RIGHT predicates")
        self._env.episode_length_buf.zero_()
        self.samples = [self._sample(0)]
        camera_rows = _camera_rows(self._env, obs)
        self.adapter.capture_and_compile(
            env=self._env,
            observation=obs,
            scene_object_mapping=scene_mapping,
            camera_rows=camera_rows,
            settle_stability={
                "settle_steps": 60,
                "stability_window_steps": 15,
                "maxima_by_object": stability,
            },
        )
        snapshot = json.loads(bootstrap.live_snapshot.read_text(encoding="utf-8"))
        initial_payload = {
            "sample": self.samples[0],
            "realised_object_poses": snapshot["realised_object_poses"],
            "arm_reset_pose": snapshot["arm_reset_pose"],
            "cameras": snapshot["cameras"],
        }
        self.native_reset_snapshot_sha256 = hashlib.sha256(
            json.dumps(initial_payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        reset_contract = build_reset_contract(
            env=self._env,
            physical_object_names=PHYSICAL_OBJECTS,
            camera_rows=camera_rows,
            observation=obs,
        )
        # Reuse the exact frozen E004 lossless request-zero machinery.  Its
        # public names say LEFT/RIGHT, but C002 uses the operation purely as a
        # language-free block source/replay.  Synthetic legacy IDs satisfy
        # that interface and are retained alongside the actual C002 cell ID.
        legacy_source_id = f"{cell.block_id}:request0:left"
        legacy_target_id = f"{cell.block_id}:request0:right"
        if bootstrap.request0_mode == "capture_block":
            request0 = capture_left_observation(
                observation=obs,
                reset_contract=reset_contract,
                amendment_path=bootstrap.request0_replay_amendment,
                amendment_sha256=bootstrap.request0_replay_amendment_sha256,
                cell_id=legacy_source_id,
                matched_pair_id=cell.matched_pair_id,
                cache_path=bootstrap.request0_observation_cache,
                manifest_path=bootstrap.request0_observation_manifest,
                reset_contract_path=bootstrap.request0_reset_contract,
            )
            write_capture_attestation(
                amendment_path=bootstrap.request0_replay_amendment,
                amendment_sha256=bootstrap.request0_replay_amendment_sha256,
                cell_id=legacy_source_id,
                matched_pair_id=cell.matched_pair_id,
                cache_path=bootstrap.request0_observation_cache,
                manifest_path=bootstrap.request0_observation_manifest,
                reset_contract_path=bootstrap.request0_reset_contract,
                observation_payload_sha256=request0["observation_payload_sha256"],
                reset_contract_payload_sha256=request0["reset_contract"]["payload_sha256"],
                attestation_path=bootstrap.request0_replay_attestation,
            )
            effective_obs = obs
        else:
            effective_obs, request0 = replay_left_observation_for_right(
                native_observation=obs,
                native_reset_contract=reset_contract,
                amendment_path=bootstrap.request0_replay_amendment,
                amendment_sha256=bootstrap.request0_replay_amendment_sha256,
                cell_id=legacy_target_id,
                matched_pair_id=cell.matched_pair_id,
                cache_path=bootstrap.request0_observation_cache,
                cache_sha256=str(bootstrap.request0_observation_cache_sha256),
                manifest_path=bootstrap.request0_observation_manifest,
                manifest_sha256=str(bootstrap.request0_observation_manifest_sha256),
                reset_contract_path=bootstrap.request0_reset_contract,
                reset_contract_file_sha256=str(bootstrap.request0_reset_contract_sha256),
                native_reset_contract_path=bootstrap.request0_native_reset_contract,
                attestation_path=bootstrap.request0_replay_attestation,
            )
        observation_payload_sha = str(
            request0["observation_payload_sha256"]
            if bootstrap.request0_mode == "capture_block"
            else request0["request0_observation_payload_sha256"]
        )
        reset_payload_sha = str(
            request0["reset_contract"]["payload_sha256"]
            if bootstrap.request0_mode == "capture_block"
            else request0["right_reset_contract_sha256"]
        )
        self.request0_evidence = evidence_envelope(
            mode=("capture_left" if bootstrap.request0_mode == "capture_block" else "replay_right"),
            amendment_path=bootstrap.request0_replay_amendment,
            cache_path=bootstrap.request0_observation_cache,
            manifest_path=bootstrap.request0_observation_manifest,
            reset_contract_path=bootstrap.request0_reset_contract,
            native_reset_contract_path=(
                bootstrap.request0_reset_contract
                if bootstrap.request0_mode == "capture_block"
                else bootstrap.request0_native_reset_contract
            ),
            attestation_path=bootstrap.request0_replay_attestation,
            observation_payload_sha256=observation_payload_sha,
            reset_contract_payload_sha256=reset_payload_sha,
        )
        self.request0_evidence.update(
            {
                "c002_mode": bootstrap.request0_mode,
                "c002_cell_id": cell.cell_id,
                "c002_cell_sha256": cell.row_sha256,
                "c002_seed_block_id": cell.block_id,
                "legacy_e004_relation_labels_are_operational_only": True,
            }
        )
        self.request0_pair_identity_sha256 = self.request0_evidence["pair_identity_sha256"]
        # This hash is shared by all four prompt forms and binds the exact
        # physical reset contract plus the exact effective request-zero
        # observation.  The native renderer snapshot remains separately
        # retained for infrastructure diagnostics.
        self.initial_state_sha256 = self.request0_pair_identity_sha256
        self.policy_camera_images = _retain_policy_camera_images(effective_obs)
        if bootstrap.authorization_mode == "model_blind_preflight":
            self.model_blind_camera_video = _write_model_blind_camera_video(effective_obs)
        self.started = time.monotonic()
        self.cached_reset = (effective_obs, info)
        return self.cached_reset

    def authorize_request(self) -> str:
        if (
            self.runner_resets != 2
            or self.physical_resets != 1
            or self.initial_state_sha256 is None
            or self.request0_pair_identity_sha256 is None
            or self.request0_evidence is None
        ):
            raise RuntimeError("model request attempted before the complete live reset gate")
        return self.adapter.authorize_model_request()

    def step(self, action: Any) -> Any:
        self.adapter.authorize_behavioral_action()
        if len(self.samples) - 1 >= spec.action_cap:
            raise RuntimeError("behavioral action exceeded the registered action cap")
        if not self.partial_path.exists():
            bootstrap.state_capture_dir.mkdir(parents=True, exist_ok=True)
            self.partial_path.write_text(
                json.dumps(self.samples[0], allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        result = self._env.step(action)
        sample = self._sample(len(self.samples))
        self.samples.append(sample)
        with self.partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
        return result

    def write_capture(self) -> Path | None:
        if self.capture_path.exists() or not self.samples:
            return self.capture_path if self.capture_path.exists() else None
        actions = len(self.samples) - 1
        if actions != self.adapter.behavioral_action_count:
            raise RuntimeError("live gate action counter differs from captured state transitions")
        detached = bool(object_dropped(self._env, object="rubiks_cube", env_id=0))
        normalized_steps = frozen_normalize_steps(self.samples)
        success = frozen_requested_success(normalized_steps, cell.physical_goal, detached)
        valid = success or actions == spec.action_cap
        value = {
            "schema_version": "vla-wam-shared-v3c002-droid-state-capture-v1",
            "study_id": cell.row["study_id"],
            "amendment_id": cell.row["amendment_id"],
            "registered_cell_id": cell.cell_id,
            "registered_cell_sha256": cell.row_sha256,
            "model_id": cell.model_id,
            "environment_seed": cell.environment_seed,
            "sampling_seed": cell.sampling_seed,
            "requested_relation": cell.relation,
            "prompt": cell.row["prompt"],
            "requested_success": success,
            "right_censored": not success and actions == spec.action_cap,
            "final_detached_release": detached,
            "actions_executed": actions,
            "action_cap": spec.action_cap,
            "model_request_count": self.adapter.model_request_count,
            "runner_pre_action_reset_calls": self.runner_resets,
            "physical_reset_calls": self.physical_resets,
            "initial_state_sha256": self.initial_state_sha256,
            "native_reset_snapshot_sha256": self.native_reset_snapshot_sha256,
            "request0_pair_identity_sha256": self.request0_pair_identity_sha256,
            "request0_replay": self.request0_evidence,
            "steps": self.samples,
            "wall_time_s": time.monotonic() - self.started,
            "behavioral_result_valid_candidate": valid,
            "partial_attempt_reason": None if valid else "episode ended before frozen success or action cap",
        }
        return _write_new_json(self.capture_path, value)

    def close(self) -> Any:
        if bootstrap.authorization_mode != "model_blind_preflight":
            self.write_capture()
        return self._env.close()


study_commit = _git_commit(study_root)
robolab_commit = _git_commit(robolab_root)
if study_commit != bootstrap.expected_study_commit or robolab_commit != bootstrap.expected_robolab_commit:
    raise RuntimeError("study or RoboLab revision differs from the bridge invocation")
if not Path(robolab.__file__).resolve().is_relative_to(robolab_root):
    raise RuntimeError("effective RoboLab import is outside the pinned worktree")
gpu_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
).splitlines()
if bootstrap.lane_gpu_uuid not in {row.strip() for row in gpu_rows}:
    raise RuntimeError("assigned C002 simulator GPU UUID is not live-visible")
observed_runtime = dict(runtime_manifest["runtime_identity"])
if observed_runtime.get("source_commit") != study_commit:
    raise RuntimeError("lane runtime source commit differs from the executed worktree")
if observed_runtime.get("server_port") != bootstrap.model_endpoint_port:
    raise RuntimeError("lane runtime server port differs from the policy endpoint")

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False
set_output_dir(str(bootstrap.output_dir.resolve()))
task_root = study_root / "experiments/v3/phase_c_semantic_equivalence_v3c002/task_files"
auto_register_droid_envs(
    task=[str(task_root / "left.py"), str(task_root / "right.py")],
    cameras=WRIST_LEFT_RIGHT_HEAD,
)
args_cli.task = ["V3C002DroidLeftTask" if cell.physical_goal == "left" else "V3C002DroidRightTask"]


_create_env = robolab_runtime.create_env
proxies: list[StateCaptureProxy] = []
clients: list[Any] = []


def _captured_create_env(*args: Any, **kwargs: Any) -> Any:
    kwargs["seed"] = cell.environment_seed
    env, env_cfg = _create_env(*args, **kwargs)
    if env_cfg.instruction != cell.row["prompt"]:
        raise RuntimeError("RoboLab task prompt bytes differ from the registered cell")
    proxy = StateCaptureProxy(env)
    proxies.append(proxy)
    return proxy, env_cfg


robolab_runtime.create_env = _captured_create_env


def _proxy() -> StateCaptureProxy:
    if len(proxies) != 1:
        raise RuntimeError("exactly one C002 environment must exist before request zero")
    return proxies[0]


class _C002Pi05Client:
    """Construct the exact π0.5 client lazily after policy imports are live."""

    def __new__(cls) -> Any:
        from policies.pi0_family.client import Pi0DroidJointposClient

        class Client(Pi0DroidJointposClient):
            def __init__(self) -> None:
                super().__init__(
                    remote_host=bootstrap.model_endpoint_host,
                    remote_port=bootstrap.model_endpoint_port,
                    policy_variant="pi05",
                    open_loop_horizon=15,
                )
                self.prompt = cell.row["prompt"]
                self.request_index = 0
                self.request_sampling_seeds: list[int] = []
                self.returned_chunks: list[np.ndarray] = []
                self.executed_actions: list[np.ndarray] = []
                self.trace_path: Path | None = None
                self.query_server_entry_count = 0
                self.model_blind_boundary_count = 0

            def _query_server(self, request: dict[str, Any]) -> dict[str, Any]:
                self.query_server_entry_count += 1
                _proxy().authorize_request()
                request_seed = cell.sampling_seed * 1000 + self.request_index
                # Preserve the frozen V2-A010 native request exactly.  The
                # E004 gate binding is retained beside it, not injected into
                # the OpenPI observation dictionary.
                response = super()._query_server({**request, "sampling_seed": request_seed})
                if response.get("v2a010_sampling_seed") != request_seed:
                    raise RuntimeError("π0.5 server did not attest the requested sampling seed")
                self.request_sampling_seeds.append(request_seed)
                self.request_index += 1
                return response

            def _unpack_response(self, response: dict[str, Any]) -> np.ndarray:
                chunk = np.asarray(super()._unpack_response(response), dtype=np.float32)
                if chunk.shape != (15, 8) or not np.isfinite(chunk).all():
                    raise RuntimeError(f"π0.5 response must be finite [15,8], got {chunk.shape}")
                self.returned_chunks.append(chunk.copy())
                return chunk

            def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
                if instruction != self.prompt:
                    raise RuntimeError("π0.5 episode-static prompt changed")
                if bootstrap.authorization_mode == "model_blind_preflight":
                    proxy = _proxy()
                    if proxy.runner_resets != 2 or proxy.physical_resets != 1 or proxy.request0_evidence is None:
                        raise RuntimeError("model-blind boundary reached before the complete C002 reset/request-zero gate")
                    self.model_blind_boundary_count += 1
                    raise ModelBlindBoundaryReached("stopped before _query_server by registered model-blind adapter mode")
                result = super().infer(obs, instruction, env_id=env_id)
                action = np.asarray(result["action"], dtype=np.float32)
                if action.shape != (8,) or not np.isfinite(action).all():
                    raise RuntimeError("π0.5 executed action must be finite [8]")
                self.executed_actions.append(action.copy())
                return result

            def write_trace(self) -> Path | None:
                if self.trace_path is not None or not self.executed_actions:
                    return self.trace_path
                bootstrap.action_trace_dir.mkdir(parents=True, exist_ok=True)
                stem = f"seed{cell.sampling_seed}_{cell.relation}"
                actions_path = bootstrap.action_trace_dir / f"{stem}_executed_actions.npy"
                chunks_path = bootstrap.action_trace_dir / f"{stem}_returned_chunks.npy"
                metadata_path = bootstrap.action_trace_dir / f"{stem}_action_trace.json"
                if any(path.exists() for path in (actions_path, chunks_path, metadata_path)):
                    raise FileExistsError("refusing to overwrite π0.5 C002 action evidence")
                actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
                chunks = np.stack(self.returned_chunks).astype(np.float32, copy=False)
                np.save(actions_path, actions, allow_pickle=False)
                np.save(chunks_path, chunks, allow_pickle=False)
                _write_new_json(
                    metadata_path,
                    {
                        "schema_version": "vla-wam-shared-v3c002-pi05-action-trace-v1",
                        "registered_cell_id": cell.cell_id,
                        "registered_cell_sha256": cell.row_sha256,
                        "prompt": self.prompt,
                        "request_sampling_seeds": self.request_sampling_seeds,
                        "request_events": [
                            {"replan_index": index, "request_seed": seed}
                            for index, seed in enumerate(self.request_sampling_seeds)
                        ],
                        "executed_actions": _file_record(actions_path),
                        "returned_action_chunks": _file_record(chunks_path),
                        "future_interface": "actions_only",
                    },
                )
                self.trace_path = metadata_path.resolve()
                return self.trace_path

            def reset(self, *, env_id: int | None = None) -> None:
                if env_id is None:
                    self.write_trace()
                super().reset(env_id=env_id)

        return Client()


def make_client(_: argparse.Namespace) -> Any:
    if cell.model_id != MODEL_ID:
        raise RuntimeError(f"C002 permits only the registered π0.5 model, not {cell.model_id}")
    client = _C002Pi05Client()
    clients.append(client)
    return client


class ModelBlindBoundaryReached(RuntimeError):
    """Expected zero-request stop immediately before the policy query boundary."""


def _write_model_blind_report() -> Path:
    if len(proxies) != 1 or len(clients) != 1:
        raise RuntimeError("model-blind proof requires exactly one C002 proxy and client")
    proxy, client = proxies[0], clients[0]
    if (
        proxy.runner_resets != 2
        or proxy.physical_resets != 1
        or proxy.adapter.model_request_count != 0
        or proxy.adapter.behavioral_action_count != 0
        or len(proxy.samples) != 1
        or client.query_server_entry_count != 0
        or client.model_blind_boundary_count != 1
        or proxy.request0_evidence is None
        or proxy.policy_camera_images is None
        or proxy.model_blind_camera_video is None
    ):
        raise RuntimeError("C002 model-blind same-process proof counters or retained evidence are invalid")
    bootstrap.raw_event_stream.parent.mkdir(parents=True, exist_ok=True)
    with bootstrap.raw_event_stream.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "record_type": "model_blind_pre_request_boundary",
            "cell_id": cell.cell_id,
            "model_request_count": 0,
            "behavioral_action_count": 0,
            "behavioral_episode_count": 0,
            "stopped_before": "_query_server",
        }, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "vla-wam-shared-v3c002-same-process-model-blind-adapter-gate-v1",
        "status": "passed_same_process_gate_stopped_before_query_server",
        "passed": True,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "cell_id": cell.cell_id,
        "cell_sha256": cell.row_sha256,
        "source_commit": study_commit,
        "robolab_commit": robolab_commit,
        "registration_sha256": bootstrap.registration_sha256,
        "queue_sha256": bootstrap.queue_sha256,
        "runtime_manifest": _file_record(bootstrap.runtime_manifest),
        "candidate": _file_record(bootstrap.candidate),
        "live_scene_snapshot": _file_record(bootstrap.live_snapshot),
        "live_scene_gate": _file_record(bootstrap.live_gate),
        "request0_replay": proxy.request0_evidence,
        "policy_camera_images": proxy.policy_camera_images,
        "renderer_video": _file_record(proxy.model_blind_camera_video),
        "raw_writer_probe": _file_record(bootstrap.raw_event_stream),
        "full_reset": True,
        "physical_scene": True,
        "policy_cameras": True,
        "raw_writer": True,
        "renderer": True,
        "same_process_gate_completed_before_query_server": True,
        "query_server_entry_count": 0,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "behavioral_episode_count": 0,
        "excluded_from_behavioral_denominators": True,
        "scope": "C002 adapter-process model-blind proof only; no inference or behavior authorized",
    }
    return _write_new_json(bootstrap.simulator_export, report)


def _viewport_video() -> Path:
    paths = [
        path.resolve()
        for path in bootstrap.output_dir.rglob("*.mp4")
        if path.is_file() and path.stat().st_size > 0
    ]
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one non-empty viewport video, found {paths}")
    capture = cv2.VideoCapture(str(paths[0]))
    okay, frame = capture.read()
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not okay or frame is None or count < 1:
        raise RuntimeError("viewport video does not decode")
    return paths[0]


def _trace_metadata(client: Any) -> tuple[Path, dict[str, Any]]:
    path = client.write_trace()
    if path is None:
        path = getattr(client, "trace_path", None)
    if path is None or not Path(path).is_file():
        raise RuntimeError("model client did not retain an action trace")
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    executed = value.get("executed_actions")
    if not isinstance(executed, dict) or not isinstance(executed.get("path"), str):
        raise RuntimeError("action trace lacks its executed-action NPY")
    return path, value


def _write_export() -> Path:
    if len(proxies) != 1 or len(clients) != 1:
        raise RuntimeError("one C002 proxy/client is required")
    proxy = proxies[0]
    capture_path = proxy.write_capture()
    if capture_path is None:
        raise RuntimeError("state capture is absent")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if capture.get("behavioral_result_valid_candidate") is not True:
        raise RuntimeError("partial attempt cannot emit a behavioral export")
    trace_path, trace = _trace_metadata(clients[0])
    actions_path = Path(str(trace["executed_actions"]["path"])).resolve()
    actions = np.load(actions_path, allow_pickle=False)
    if actions.ndim != 2 or actions.shape != (capture["actions_executed"], 8) or not np.isfinite(actions).all():
        raise RuntimeError("retained action trace differs from captured simulator steps")
    steps = frozen_normalize_steps(capture["steps"])
    final = steps[-1]
    signed = float(final["object_xyz"][1]) - float(final["reference_xyz"][1])
    depth = signed if cell.physical_goal == "left" else -signed
    detached = bool(capture["final_detached_release"])
    success = frozen_requested_success(steps, cell.physical_goal, detached)
    category = frozen_failure_category(
        success=success,
        steps=steps,
        relation=cell.physical_goal,
        detached_release=detached,
    )
    request_events = trace.get("request_events")
    if not isinstance(request_events, list) or not request_events:
        raise RuntimeError("π0.5 action trace lacks request events")
    bootstrap.raw_event_stream.parent.mkdir(parents=True, exist_ok=True)
    with bootstrap.raw_event_stream.open("x", encoding="utf-8") as handle:
        for event in request_events:
            handle.write(json.dumps(event, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    final_state_path = bootstrap.state_capture_dir / "final_state.json"
    _write_new_json(final_state_path, final)
    if proxy.policy_camera_images is None:
        raise RuntimeError("request-zero policy camera images were not retained")
    episode_runtime = dict(observed_runtime)
    episode_runtime["policy_camera_image_artifact_hashes"] = {
        name: record["sha256"] for name, record in proxy.policy_camera_images.items()
    }
    raw = {
        "schema_version": "vla-wam-shared-v3c002-raw-episode-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "cell_id": cell.cell_id,
        "cell_sha256": cell.row_sha256,
        "registration_sha256": bootstrap.registration_sha256,
        "queue_sha256": bootstrap.queue_sha256,
        "model_id": MODEL_ID,
        "prompt_condition": cell.condition,
        "physical_goal": cell.physical_goal,
        "surface_direction_word": cell.row["surface_direction_word"],
        "prompt": cell.row["prompt"],
        "prompt_utf8_hex": cell.row["prompt_utf8_hex"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "episode_seed": cell.seed,
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "stage_identifier": "full_reset",
        "initial_state_sha256": capture["initial_state_sha256"],
        "request0_pair_identity_sha256": capture["request0_pair_identity_sha256"],
        "request0_replay": capture["request0_replay"],
        "request_events": request_events,
        "state_trace": steps,
        "final_detached_release": detached,
        "reported_frozen_task_success": success,
        "reported_failure_category": category,
        "pickup_grabbed_ever": any(bool(step["object_grabbed"]) for step in steps),
        "first_pickup_step": next((int(step["action_step"]) for step in steps if step["object_grabbed"]), None),
        "contact_detected": None,
        "contact_measurement_status": "not_exposed_by_exact_e004_state_capture",
        "transport_steps_after_pickup": sum(bool(step["object_grabbed"]) for step in steps),
        "placement_in_requested_cone_final": bool(frozen_cone(steps[-1], cell.physical_goal)),
        "signed_final_lateral_offset": signed,
        "requested_side_depth": depth,
        "runtime_identity": episode_runtime,
        "authorization_mode": bootstrap.authorization_mode,
        "excluded_from_behavioral_denominators": bootstrap.authorization_mode == "excluded_smoke",
        "raw_artifacts": {
            "simulator_video": _file_record(_viewport_video()),
            "executed_action_trace": _file_record(actions_path),
            "raw_episode_jsonl": _file_record(bootstrap.raw_event_stream),
            "final_state": _file_record(final_state_path),
            "state_trace": _file_record(capture_path),
            "policy_camera_images": proxy.policy_camera_images,
        },
        "live_gate": _file_record(bootstrap.live_gate),
        "action_trace_metadata": _file_record(trace_path),
        "model_request_count": proxy.adapter.model_request_count,
        "live_gate_behavioral_action_count": proxy.adapter.behavioral_action_count,
        "live_orientation_realisation_tolerance_amendment": r002_attestation,
    }
    path = bootstrap.simulator_export.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained C002 raw episode: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    failure: BaseException | None = None
    try:
        model_blind_stop = False
        try:
            run_evaluation(args_cli, policy=spec.policy_id, client_factory=make_client)
        except ModelBlindBoundaryReached:
            if bootstrap.authorization_mode != "model_blind_preflight":
                raise
            model_blind_stop = True
        except BaseException as exc:
            failure = exc
        finally:
            if bootstrap.authorization_mode != "model_blind_preflight":
                for client in clients:
                    try:
                        client.write_trace()
                    except BaseException as exc:
                        failure = failure or exc
                for proxy in proxies:
                    try:
                        proxy.write_capture()
                    except BaseException as exc:
                        failure = failure or exc
        if failure is not None:
            failure_path = bootstrap.simulator_export.with_name("bridge_failure.json")
            _write_new_json(
                failure_path,
                {
                    "schema_version": "vla-wam-shared-v3c002-infrastructure-attempt-v1",
                    "record_type": "infrastructure_attempt",
                    "behavioral_result_valid": False,
                    "denominator_eligible": False,
                    "registered_cell_id": cell.cell_id,
                    "registered_cell_sha256": cell.row_sha256,
                    "model_id": cell.model_id,
                    "authorization_mode": bootstrap.authorization_mode,
                    "model_request_count": sum(proxy.adapter.model_request_count for proxy in proxies),
                    "behavioral_action_count": sum(proxy.adapter.behavioral_action_count for proxy in proxies),
                    "excluded_from_behavioral_denominators": True,
                    "error_type": type(failure).__name__,
                    "error": str(failure),
                    "traceback": "".join(
                        traceback.format_exception(type(failure), failure, failure.__traceback__)
                    ),
                },
            )
            raise failure
        if bootstrap.authorization_mode == "model_blind_preflight":
            if not model_blind_stop:
                raise RuntimeError("model-blind adapter did not stop before _query_server")
            _write_model_blind_report()
        else:
            _write_export()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
