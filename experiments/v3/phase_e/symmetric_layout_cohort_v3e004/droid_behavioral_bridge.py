#!/usr/bin/env python3
"""Execute one hash-released V3-E004 DROID behavioral cell.

The bridge owns one Isaac process and one registered cell.  It recreates the
graded object layout, performs the 60+15 model-blind settle/camera/reset gate
inside the live environment, and only then permits request zero.  The policy
clients retain their native action/future contracts; this file only supplies
the common E004 authorization and canonical raw simulator export.
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
from typing import Any, Callable, Mapping
import uuid


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--robolab-root", type=Path, required=True)
BOOTSTRAP.add_argument("--registration", type=Path, required=True)
BOOTSTRAP.add_argument("--registration-sha256", required=True)
BOOTSTRAP.add_argument("--registration-commit", required=True)
BOOTSTRAP.add_argument("--queue", type=Path, required=True)
BOOTSTRAP.add_argument("--queue-sha256", required=True)
BOOTSTRAP.add_argument("--candidate", type=Path, required=True)
BOOTSTRAP.add_argument("--candidate-sha256", required=True)
BOOTSTRAP.add_argument("--lane-release", type=Path, required=True)
BOOTSTRAP.add_argument("--lane-release-sha256", required=True)
BOOTSTRAP.add_argument("--runtime-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--cell-id", required=True)
BOOTSTRAP.add_argument("--lane-pod-uid", required=True)
BOOTSTRAP.add_argument("--lane-gpu-uuid", required=True)
BOOTSTRAP.add_argument("--live-snapshot", type=Path, required=True)
BOOTSTRAP.add_argument("--live-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--simulator-export", type=Path, required=True)
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
BOOTSTRAP.add_argument("--dreamzero-server-contract", type=Path)
BOOTSTRAP.add_argument("--dreamzero-future-root", type=Path)
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
robolab_root = bootstrap.robolab_root.resolve()
for root in (study_root, robolab_root):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.droid_behavioral_contract import (  # noqa: E402
    bind_runtime_identity,
    model_spec,
    simulator_export_envelope,
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
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (  # noqa: E402
    RuntimeContractError,
    load_runtime_bundle,
    sha256_file,
    validate_lane_release,
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
    BOOTSTRAP.error("behavioral bridge accepts only new E004 DROID cells")
spec = model_spec(cell, endpoint_port=bootstrap.model_endpoint_port)
lane_release = validate_lane_release(
    bootstrap.lane_release,
    bootstrap.lane_release_sha256,
    bundle=bundle,
    model_id=cell.model_id,
    lane_pod_uid=bootstrap.lane_pod_uid,
    lane_gpu_uuid=bootstrap.lane_gpu_uuid,
)
runtime_release = lane_release.get("runtime_identity")
if not isinstance(runtime_release, dict) or not isinstance(runtime_release.get("sha256"), str):
    BOOTSTRAP.error("lane release does not bind a model runtime manifest")
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
for path in (bootstrap.live_snapshot, bootstrap.live_gate, bootstrap.simulator_export):
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


def _hold_action(obs: Mapping[str, Any], device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, 8):
        raise RuntimeError(f"hold action is not [1,8]: {tuple(action.shape)}")
    return action


def _cone(sample: Mapping[str, Any], relation: str) -> bool:
    obj = np.asarray(sample["object_xyz"], dtype=np.float64)
    ref = np.asarray(sample["reference_xyz"], dtype=np.float64)
    delta = obj - ref
    radius = math.hypot(float(delta[0]), float(delta[1]))
    margin = float(delta[1]) if relation == "left" else -float(delta[1])
    return radius > 1e-8 and margin / radius >= math.cos(math.radians(45.0))


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
        )
        self.initial_state_sha256: str | None = None

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
            raise RuntimeError("reset attempted after E004 behavior began")
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
        gate = self.adapter.capture_and_compile(
            env=self._env,
            observation=obs,
            scene_object_mapping=scene_mapping,
            camera_rows=_camera_rows(self._env, obs),
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
        self.initial_state_sha256 = hashlib.sha256(
            json.dumps(initial_payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.started = time.monotonic()
        self.cached_reset = (obs, info)
        return self.cached_reset

    def authorize_request(self) -> str:
        if self.runner_resets != 2 or self.physical_resets != 1 or self.initial_state_sha256 is None:
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
        requested = [_cone(sample, cell.relation) for sample in self.samples]
        detached = bool(object_dropped(self._env, object="rubiks_cube", env_id=0))
        success = len(requested) >= 3 and all(requested[-3:]) and detached
        valid = success or actions == spec.action_cap
        value = {
            "schema_version": "vla-wam-shared-v3e004-droid-state-capture-v1",
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
            "steps": self.samples,
            "wall_time_s": time.monotonic() - self.started,
            "behavioral_result_valid_candidate": valid,
            "partial_attempt_reason": None if valid else "episode ended before frozen success or action cap",
        }
        return _write_new_json(self.capture_path, value)

    def close(self) -> Any:
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
    raise RuntimeError("assigned E004 GPU UUID is not live-visible")

runtime_identity_path = bootstrap.simulator_export.with_name("runtime_identity.json")
bind_runtime_identity(
    cell=cell,
    bundle=bundle,
    source_path=bootstrap.runtime_manifest,
    source_expected_sha256=str(runtime_release["sha256"]),
    lane_release_path=bootstrap.lane_release,
    lane_release_sha256=bootstrap.lane_release_sha256,
    output_path=runtime_identity_path,
)

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False
set_output_dir(str(bootstrap.output_dir.resolve()))
task_root = study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/task_files"
auto_register_droid_envs(
    task=[str(task_root / "left.py"), str(task_root / "right.py")],
    cameras=WRIST_LEFT_RIGHT_HEAD,
)
args_cli.task = ["V3E004DroidLeftTask" if cell.relation == "left" else "V3E004DroidRightTask"]


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
        raise RuntimeError("exactly one E004 environment must exist before request zero")
    return proxies[0]


class _E004Pi05Client:
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

            def _query_server(self, request: dict[str, Any]) -> dict[str, Any]:
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
                    raise FileExistsError("refusing to overwrite π0.5 E004 action evidence")
                actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
                chunks = np.stack(self.returned_chunks).astype(np.float32, copy=False)
                np.save(actions_path, actions, allow_pickle=False)
                np.save(chunks_path, chunks, allow_pickle=False)
                _write_new_json(
                    metadata_path,
                    {
                        "schema_version": "vla-wam-shared-v3e004-pi05-action-trace-v1",
                        "registered_cell_id": cell.cell_id,
                        "registered_cell_sha256": cell.row_sha256,
                        "prompt": self.prompt,
                        "request_sampling_seeds": self.request_sampling_seeds,
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


class _LazyE004CosmosClient:
    def __init__(self) -> None:
        self.inner: Any | None = None
        self.trace_path: Path | None = None
        self.session_manifest_path = bootstrap.simulator_export.with_name("cosmos_session_manifest.json")

    def _ensure(self) -> Any:
        if self.inner is not None:
            return self.inner
        proxy = _proxy()
        if proxy.runner_resets != 2 or proxy.initial_state_sha256 is None or not bootstrap.live_gate.is_file():
            raise RuntimeError("Cosmos client construction preceded the live reset gate")
        from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.cosmos_client import E004CosmosClient
        from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.cosmos_runtime import (
            build_session_manifest,
            load_registration_bundle,
            load_runtime_identity,
        )

        cosmos_bundle = load_registration_bundle(
            study_root,
            registration_commit=bootstrap.registration_commit,
        )
        cosmos_cell = cosmos_bundle.cell(cell.cell_id, model_id=cell.model_id)
        runtime = load_runtime_identity(bootstrap.runtime_manifest, model_id=cell.model_id)
        session = build_session_manifest(
            bundle=cosmos_bundle,
            cell=cosmos_cell,
            runtime=runtime,
            runtime_manifest_path=bootstrap.runtime_manifest,
            session_id=f"{cell.cell_id}:{uuid.uuid4().hex}",
            attempt_id=str(bootstrap.simulator_export.parent.resolve()),
            gate_paths={
                "static_layout": bootstrap.live_gate,
                "live_camera_reset": bootstrap.live_gate,
                "raw_write": bootstrap.lane_release,
                "renderer": bootstrap.lane_release,
            },
            initial_state_sha256=proxy.initial_state_sha256,
        )
        _write_new_json(self.session_manifest_path, session)
        self.inner = E004CosmosClient(
            bundle=cosmos_bundle,
            cell=cosmos_cell,
            runtime=runtime,
            session_manifest_path=self.session_manifest_path,
            remote_host=bootstrap.model_endpoint_host,
            remote_port=bootstrap.model_endpoint_port,
            sampling_seed_base=cell.sampling_seed,
            action_trace_dir=bootstrap.action_trace_dir,
            future_trace_dir=bootstrap.future_trace_dir,
        )
        return self.inner

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        inner = self._ensure()
        _proxy().authorize_request()
        return inner.infer(obs, instruction, env_id=env_id)

    def reset(self, *, env_id: int | None = None) -> None:
        if self.inner is not None:
            self.inner.reset(env_id=env_id)
            self.trace_path = self.inner.trace_path

    def write_trace(self) -> Path | None:
        if self.inner is None:
            return None
        self.trace_path = self.inner.write_trace()
        return self.trace_path


def _dreamzero_client() -> Any:
    if bootstrap.dreamzero_server_contract is None or bootstrap.dreamzero_future_root is None:
        raise RuntimeError("DreamZero requires --dreamzero-server-contract and --dreamzero-future-root")
    from experiments.v3.dreamzero_droid.client import V3DreamZeroS2Client
    from experiments.v3.dreamzero_phase_b.client import V3B003DreamZeroClient

    class Client(V3B003DreamZeroClient):
        def __init__(self) -> None:
            super().__init__(
                remote_host=bootstrap.model_endpoint_host,
                remote_port=bootstrap.model_endpoint_port,
                environment_seed=cell.environment_seed,
                cell_id=cell.cell_id,
                reset_attestation=bootstrap.live_gate,
                action_trace_dir=bootstrap.action_trace_dir,
                server_contract_path=bootstrap.dreamzero_server_contract,
                release_gate_path=bootstrap.lane_release,
                future_root=bootstrap.dreamzero_future_root,
            )
            self.trace_path: Path | None = None

        def _pack_request(self, extracted_obs: dict[str, Any], instruction: str) -> dict[str, Any]:
            _proxy().authorize_request()
            # Keep DreamZero's native request bytes unchanged.  The per-cell
            # gate digest is already hash-bound by the simulator export.
            return V3DreamZeroS2Client._pack_request(self, extracted_obs, instruction)

        def write_trace(self) -> Path | None:
            path = super().write_trace()
            if path is not None:
                self.trace_path = Path(path).resolve()
            elif self.trace_path is None and self.prompt is not None:
                candidate_path = self._metadata_path()
                if candidate_path.is_file():
                    self.trace_path = candidate_path.resolve()
            return self.trace_path

    return Client()


def make_client(_: argparse.Namespace) -> Any:
    if cell.model_id == "pi05_current_stack_droid":
        client = _E004Pi05Client()
    elif cell.model_id in {"cosmos3_nano_policy_droid", "cosmos3_edge_policy_droid"}:
        client = _LazyE004CosmosClient()
    elif cell.model_id == "dreamzero_droid_action_cfg":
        client = _dreamzero_client()
    else:  # already excluded by the runtime bundle
        raise RuntimeError(f"unsupported E004 DROID model: {cell.model_id}")
    clients.append(client)
    return client


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
        raise RuntimeError("one E004 proxy/client is required")
    capture_path = proxies[0].write_capture()
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
    future_evidence: Any = None
    future_status = "not_exposed_by_action_only_interface"
    if cell.model_id in {"cosmos3_nano_policy_droid", "cosmos3_edge_policy_droid"}:
        future_evidence = {
            "action_future_trace": _file_record(trace_path),
            "requests": trace.get("requests", []),
        }
        future_status = "exposed_and_retained"
    elif cell.model_id == "dreamzero_droid_action_cfg":
        future_evidence = {
            "action_future_trace": _file_record(trace_path),
            "future_manifest": trace.get("future_manifest"),
        }
        future_status = "native_latent_and_official_decoded_future_retained"
    export = simulator_export_envelope(
        cell=cell,
        bundle=bundle,
        steps=capture["steps"],
        requested_success=bool(capture["requested_success"]),
        right_censored=bool(capture["right_censored"]),
        final_detached_release=bool(capture["final_detached_release"]),
        live_gate=_file_record(bootstrap.live_gate),
        runtime_identity=_file_record(runtime_identity_path),
        executed_action_trace=_file_record(actions_path),
        viewport_video=_file_record(_viewport_video()),
        future_evidence=future_evidence,
        future_evidence_status=future_status,
    )
    export.update(
        {
            "state_capture": _file_record(capture_path),
            "action_trace_metadata": _file_record(trace_path),
            "model_request_count": proxies[0].adapter.model_request_count,
            "live_gate_behavioral_action_count": proxies[0].adapter.behavioral_action_count,
        }
    )
    return _write_new_json(bootstrap.simulator_export, export)


def main() -> None:
    failure: BaseException | None = None
    try:
        try:
            run_evaluation(args_cli, policy=spec.policy_id, client_factory=make_client)
        except BaseException as exc:
            failure = exc
        finally:
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
                    "schema_version": "vla-wam-shared-v3e004-droid-bridge-failure-v1",
                    "record_type": "infrastructure_attempt",
                    "behavioral_result_valid": False,
                    "denominator_eligible": False,
                    "registered_cell_id": cell.cell_id,
                    "registered_cell_sha256": cell.row_sha256,
                    "model_id": cell.model_id,
                    "error_type": type(failure).__name__,
                    "error": str(failure),
                    "traceback": "".join(
                        traceback.format_exception(type(failure), failure, failure.__traceback__)
                    ),
                },
            )
            raise failure
        _write_export()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
