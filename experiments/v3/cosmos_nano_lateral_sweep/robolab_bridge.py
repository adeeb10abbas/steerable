#!/usr/bin/env python3
"""Run one exact released V3-B005 Nano cell in RoboLab.

The bridge creates the live-reset attestation immediately before the first
policy request, retains the initial and every post-action state, and writes the
simulator export consumed by ``compile_cell.py``.  A partial attempt preserves
its streams but never emits a behavioral export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback
from typing import Any


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--release-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--release-manifest-sha256", required=True)
BOOTSTRAP.add_argument("--runtime-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--cell-id", required=True)
BOOTSTRAP.add_argument("--release-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--safe-fixture", type=Path, required=True)
BOOTSTRAP.add_argument("--safe-fixture-sha256", required=True)
BOOTSTRAP.add_argument("--state-capture-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--action-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--future-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--reset-attestation", type=Path, required=True)
BOOTSTRAP.add_argument("--simulator-export", type=Path, required=True)
BOOTSTRAP.add_argument("--output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--remote-host", required=True)
BOOTSTRAP.add_argument("--remote-port", type=int, default=18031)
BOOTSTRAP.add_argument("--lane-pod-uid", required=True)
BOOTSTRAP.add_argument("--lane-gpu-uuid", required=True)
BOOTSTRAP.add_argument("--open-loop-horizon", type=int, default=32)
BOOTSTRAP.add_argument("--instruction-controller", choices=["static"], default="static")
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))

from experiments.v3.cosmos_nano_lateral_sweep.live_support import (  # noqa: E402
    verify_behavioral_release_gate,
    verify_live_runtime_identity,
)
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (  # noqa: E402
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    ANGULAR_SPEED_TOLERANCE_RAD_S,
    AMENDMENT_ID,
    LINEAR_SPEED_TOLERANCE_M_S,
    PHASE,
    RESET_SCHEMA,
    SETTLE_EVIDENCE_SCHEMA,
    SETTLE_OBJECTS,
    SETTLE_STEPS,
    STABILITY_WINDOW_STEPS,
    STUDY_ID,
    canonical_json_bytes,
    load_release_bundle,
    sha256_bytes,
    sha256_file,
    validate_settle_stability_evidence,
    validate_reset_attestation,
)
from tools.vla_wam_v3_episode_schema import (  # noqa: E402
    MEASUREMENT_FRAME_ID,
    derive_initial_state_sha256,
)


release = load_release_bundle(
    bootstrap.release_manifest,
    expected_manifest_sha256=bootstrap.release_manifest_sha256,
)
cell = release.cell(bootstrap.cell_id)
runtime_identity = verify_live_runtime_identity(
    bootstrap.runtime_manifest,
    study_root=study_root,
    release=release,
)
release_gate = verify_behavioral_release_gate(
    bootstrap.release_gate,
    release=release,
    runtime=runtime_identity,
)
safe_fixture_path = bootstrap.safe_fixture.resolve()
if (
    not safe_fixture_path.is_file()
    or sha256_file(safe_fixture_path) != bootstrap.safe_fixture_sha256
    or release.safe_fixture_sha256 != bootstrap.safe_fixture_sha256
):
    BOOTSTRAP.error("safe fixture does not match the V3-B005 release")
os.environ["VLA_WAM_V3B005_SAFE_FIXTURE"] = str(safe_fixture_path)
os.environ["VLA_WAM_V3B005_SAFE_FIXTURE_SHA256"] = bootstrap.safe_fixture_sha256
os.environ["VLA_WAM_V3B005_LEVEL_INDEX"] = str(cell.level_index)
safe_fixture = json.loads(safe_fixture_path.read_text(encoding="utf-8"))


def _expected_positions_robot_m() -> dict[str, list[float]]:
    positions = safe_fixture["positions_robot_base_m"]
    bowl = list(positions["bowl_center_at_level_0"])
    bowl[1] = safe_fixture["ordered_bowl_y_levels_m"][cell.level_index]
    return {
        "rubiks_cube": list(positions["rubiks_cube"]),
        "bowl": bowl,
        "banana": list(positions["banana"]),
    }

if bootstrap.open_loop_horizon != ACTION_CHUNK_STEPS:
    BOOTSTRAP.error("the released Nano open-loop horizon is exactly 32")
bridge_failure_path = bootstrap.simulator_export.with_name("bridge_failure.json")
for output in (bootstrap.reset_attestation, bootstrap.simulator_export, bridge_failure_path):
    if output.exists():
        BOOTSTRAP.error(f"refusing to overwrite retained Phase-B evidence: {output}")
for directory in (
    bootstrap.state_capture_dir,
    bootstrap.action_trace_dir,
    bootstrap.future_trace_dir,
):
    if directory.exists():
        BOOTSTRAP.error(f"refusing to reuse retained Phase-B output directory: {directory}")
gpu_inventory = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
    text=True,
)
if bootstrap.lane_gpu_uuid not in gpu_inventory:
    BOOTSTRAP.error("assigned ali-owned lane GPU UUID is not visible in this process")

import cv2  # noqa: E402,F401 -- RoboLab requires this import before Isaac Lab
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.video_mode != "viewport":
    parser.error("every released Phase-B cell requires viewport video")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("a released Phase-B cell requires exactly one environment and one run")
if args_cli.enable_subtask:
    parser.error("progress-conditioned subtask coaching is prohibited")
if args_cli.instruction_type != "default":
    parser.error("Phase-B uses only the exact static direct-command prompt")
if args_cli.output_dir.exists():
    parser.error(f"refusing to reuse retained simulator output directory: {args_cli.output_dir}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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

from experiments.v3.cosmos_nano_lateral_sweep.live_client import V3BNanoLiveClient  # noqa: E402


robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False
set_output_dir(str(args_cli.output_dir.resolve()))

TASK_FILES = {"left": "left.py", "right": "right.py"}
TASK_NAMES = {"left": "V3B005NanoLeftTask", "right": "V3B005NanoRightTask"}
task_root = study_root / "experiments/v3/cosmos_nano_lateral_sweep/task_files"
auto_register_droid_envs(
    task=[str(task_root / TASK_FILES[key]) for key in ("left", "right")],
    cameras=WRIST_LEFT_RIGHT_HEAD,
)
args_cli.task = [TASK_NAMES[cell.relation]]


def _quat_inverse_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,):
        raise RuntimeError("robot root pose has an unexpected shape")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("robot root quaternion is invalid")
    q = q / norm
    w, xyz = q[0], q[1:]
    inverse_xyz = -xyz
    return (
        2 * np.dot(inverse_xyz, v) * inverse_xyz
        + (w * w - np.dot(inverse_xyz, inverse_xyz)) * v
        + 2 * w * np.cross(inverse_xyz, v)
    )


def _hold_action(obs: dict[str, Any], device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, ACTION_DIM):
        raise RuntimeError(f"unexpected model-blind hold-action shape: {tuple(action.shape)}")
    return action


def _numeric(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


class StateCaptureProxy:
    """Retain reset identity and every post-action measurement state."""

    def __init__(self, env: Any, capture_dir: Path) -> None:
        self._env = env
        self._capture_dir = capture_dir
        self._samples: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._partial_started = False
        self._written = False
        self._settle_evidence: dict[str, Any] | None = None
        self._runner_pre_action_reset_calls = 0
        self._physical_reset_calls = 0
        self._settle_gate_runs = 0
        self._cached_reset_result: tuple[Any, Any] | None = None
        stem = cell.cell_id.replace(":", "__")
        self.partial_path = capture_dir / f"{stem}.states.partial.jsonl"
        self.capture_path = capture_dir / f"{stem}.capture.json"
        self.fixture_evidence_path = capture_dir / f"{stem}.fixture_match.json"
        self.settle_evidence_path = capture_dir / f"{stem}.settle_stability.json"
        for path in (
            self.partial_path,
            self.capture_path,
            self.fixture_evidence_path,
            self.settle_evidence_path,
        ):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite Phase-B state evidence: {path}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> dict[str, Any]:
        cube_world = self._env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl_world = self._env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        robot = self._env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        return {
            "action_step": action_step,
            "object_xyz": _quat_inverse_rotate_wxyz(robot_quat, cube_world - robot_pos).tolist(),
            "reference_xyz": _quat_inverse_rotate_wxyz(robot_quat, bowl_world - robot_pos).tolist(),
            "grippers_open": bool(
                object_dropped(self._env, object="rubiks_cube", env_id=0)
            ),
            "object_grabbed": bool(
                object_grabbed(self._env, object="rubiks_cube", env_id=0)
            ),
        }

    def _physical_reset_payload(self) -> dict[str, Any]:
        robot = self._env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        objects: dict[str, Any] = {}
        for name in ("rubiks_cube", "bowl", "banana"):
            asset = self._env.scene[name].data
            position_world = asset.root_pos_w[0].detach().cpu().numpy()
            objects[name] = {
                "position_world_xyz_m": position_world.astype(float).tolist(),
                "position_robot_xyz_m": _quat_inverse_rotate_wxyz(
                    robot_quat, position_world - robot_pos
                ).tolist(),
                "quaternion_world_wxyz": asset.root_quat_w[0].detach().cpu().numpy().astype(float).tolist(),
            }
        return {
            "schema_version": "vla-wam-shared-v3b-nano-physical-reset-v1",
            "registered_cell_id": cell.cell_id,
            "fixture_id": cell.fixture_id,
            "objects": objects,
        }

    def write_reset_attestation(self) -> Path:
        if bootstrap.reset_attestation.exists():
            return bootstrap.reset_attestation.resolve()
        if len(self._samples) != 1 or self._partial_started or self._settle_evidence is None:
            raise RuntimeError("reset attestation must precede every policy request/action")
        if self._runner_pre_action_reset_calls != 2:
            raise RuntimeError("frozen RoboLab runner must perform exactly two pre-action reset calls")
        if self._physical_reset_calls != 1 or self._settle_gate_runs != 1:
            raise RuntimeError("duplicate runner reset must map to one physical reset and one settle gate")
        validate_settle_stability_evidence(self._settle_evidence, cell=cell)
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self.settle_evidence_path.write_text(
            json.dumps(self._settle_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        physical = self._physical_reset_payload()
        expected_positions = _expected_positions_robot_m()
        tolerance = 0.005
        position_errors: dict[str, float] = {}
        for name, expected in expected_positions.items():
            observed = np.asarray(
                physical["objects"][name]["position_robot_xyz_m"], dtype=np.float64
            )
            target = np.asarray(expected, dtype=np.float64)
            position_errors[name] = float(np.max(np.abs(observed - target)))
        fixture_match = max(position_errors.values()) <= tolerance
        sample = self._samples[0]
        delta = np.asarray(sample["object_xyz"]) - np.asarray(sample["reference_xyz"])
        horizontal = math.hypot(float(delta[0]), float(delta[1]))
        left = horizontal > 1e-8 and float(delta[1]) / horizontal >= math.cos(math.radians(45.0))
        right = horizontal > 1e-8 and -float(delta[1]) / horizontal >= math.cos(math.radians(45.0))
        evidence = {
            "schema_version": "vla-wam-shared-v3b-nano-live-fixture-match-v1",
            "registered_cell_id": cell.cell_id,
            "released_fixture_sha256": cell.fixture_sha256,
            "level_index": cell.level_index,
            "reference_object_initial_lateral_position_y_m": cell.level_y_m,
            "position_frame": "robot",
            "position_tolerance_m": tolerance,
            "max_abs_position_error_m": position_errors,
            "positions_match": fixture_match,
            "initial_quaternions_recorded_as_post_settle_mediators": {
                name: value["quaternion_world_wxyz"] for name, value in physical["objects"].items()
            },
            "neutral_reset": not left and not right,
        }
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self.fixture_evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not fixture_match or left or right:
            raise RuntimeError("live reset does not match the released neutral fixture")
        initial_hash = derive_initial_state_sha256(
            {"measurement_frame": MEASUREMENT_FRAME_ID, "steps": self._samples}
        )
        for directory in (bootstrap.reset_attestation.parent, args_cli.output_dir):
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".{cell.cell_id.replace(':', '_')}.write_preflight"
            probe.write_bytes(b"v3b005-live-output-preflight\n")
            probe.unlink()
        reset = {
            "schema_version": RESET_SCHEMA,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "registered_cell_id": cell.cell_id,
            "matched_block_id": cell.row["matched_block_id"],
            "model_id": cell.row["model_id"],
            "level_index": cell.level_index,
            "relation": cell.relation,
            "environment_seed": cell.seed,
            "sampling_seed": cell.seed,
            "fixture_id": cell.fixture_id,
            "released_fixture_sha256": cell.fixture_sha256,
            "prompt_sha256": cell.row["prompt_sha256"],
            "release_fingerprint_sha256": release.release_fingerprint(cell),
            "runtime_identity_sha256": runtime_identity["runtime_identity_sha256"],
            "behavioral_release_gate_sha256": sha256_file(bootstrap.release_gate),
            "lane_pod_uid": bootstrap.lane_pod_uid,
            "lane_gpu_uuid": bootstrap.lane_gpu_uuid,
            "model_request_count_before_attestation": 0,
            "neutral_reset_passed": True,
            "released_fixture_match_passed": True,
            "viewport_writer_preflight_passed": args_cli.video_mode == "viewport",
            "raw_output_preflight_passed": True,
            "model_blind_settle_gate_passed": True,
            "settle_steps": SETTLE_STEPS,
            "stable_window_steps": STABILITY_WINDOW_STEPS,
            "linear_speed_tolerance_m_s": LINEAR_SPEED_TOLERANCE_M_S,
            "angular_speed_tolerance_rad_s": ANGULAR_SPEED_TOLERANCE_RAD_S,
            "episode_length_buf_reset_passed": True,
            "settle_stability_evidence_path": str(self.settle_evidence_path.resolve()),
            "settle_stability_evidence_sha256": sha256_file(self.settle_evidence_path),
            "physical_reset_sha256": sha256_bytes(canonical_json_bytes(physical)),
            "initial_state_sha256": initial_hash,
            "fixture_match_evidence_sha256": sha256_file(self.fixture_evidence_path),
            "fixture_match_evidence_path": str(self.fixture_evidence_path.resolve()),
            "safe_fixture_sha256": bootstrap.safe_fixture_sha256,
        }
        bootstrap.reset_attestation.write_text(
            json.dumps(reset, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_reset_attestation(
            bootstrap.reset_attestation,
            cell=cell,
            release=release,
            runtime=runtime_identity,
        )
        return bootstrap.reset_attestation.resolve()

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if self._partial_started or len(self._samples) > 1 or bootstrap.reset_attestation.exists():
            raise RuntimeError("RoboLab attempted a reset after Phase-B request/action authorization")
        self._runner_pre_action_reset_calls += 1
        if self._runner_pre_action_reset_calls > 2:
            raise RuntimeError("frozen RoboLab runner performed more than two pre-action reset calls")
        if self._runner_pre_action_reset_calls == 2:
            if (
                self._cached_reset_result is None
                or self._physical_reset_calls != 1
                or self._settle_gate_runs != 1
                or self._settle_evidence is None
                or len(self._samples) != 1
            ):
                raise RuntimeError("duplicate runner reset occurred before the physical reset gate completed")
            # Pinned RoboLab calls env.reset() twice consecutively before its
            # first policy request.  After this proxy's required 75-step gate,
            # zeroing episode_length_buf would make the raw second call enter
            # RoboLab's ep_len<=2 artifact branch and perform a fresh Isaac
            # reset.  Preserve the runner API while making only that duplicate
            # call idempotent: return the already-settled observation unchanged.
            self._settle_evidence.update(
                runner_pre_action_reset_calls=self._runner_pre_action_reset_calls,
                physical_reset_calls=self._physical_reset_calls,
                settle_gate_runs=self._settle_gate_runs,
                duplicate_second_reset_idempotent=True,
            )
            validate_settle_stability_evidence(self._settle_evidence, cell=cell)
            return self._cached_reset_result

        self._physical_reset_calls += 1
        result = self._env.reset(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("RoboLab reset did not return (observation, info)")
        obs, info = result
        action = _hold_action(obs, self._env.device)
        self._settle_gate_runs += 1
        for _ in range(SETTLE_STEPS):
            obs, _, terminated, truncated, _ = self._env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("released cell terminated during model-blind settling")
        maxima = {
            name: {
                "max_linear_component_speed_m_s": 0.0,
                "max_angular_component_speed_rad_s": 0.0,
            }
            for name in SETTLE_OBJECTS
        }
        for _ in range(STABILITY_WINDOW_STEPS):
            obs, _, terminated, truncated, _ = self._env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("released cell terminated during model-blind stability window")
            world = get_world(self._env)
            for name in SETTLE_OBJECTS:
                velocity = _numeric(world.get_velocity(name, env_id=0))
                maxima[name]["max_linear_component_speed_m_s"] = max(
                    maxima[name]["max_linear_component_speed_m_s"],
                    max(abs(item) for item in velocity[:3]),
                )
                maxima[name]["max_angular_component_speed_rad_s"] = max(
                    maxima[name]["max_angular_component_speed_rad_s"],
                    max(abs(item) for item in velocity[3:]),
                )
        world = get_world(self._env)
        positions: dict[str, list[float]] = {}
        quaternions: dict[str, list[float]] = {}
        velocities: dict[str, list[float]] = {}
        for name in SETTLE_OBJECTS:
            position, quaternion = world.get_pose(name, env_id=0)
            positions[name] = _numeric(position)
            quaternions[name] = _numeric(quaternion)
            velocities[name] = _numeric(world.get_velocity(name, env_id=0))
        left = bool(object_left_of(
            self._env, object="rubiks_cube", reference_object="bowl",
            frame_of_reference="robot", mirrored=False,
            require_gripper_detached=True, env_id=0,
        ))
        right = bool(object_right_of(
            self._env, object="rubiks_cube", reference_object="bowl",
            frame_of_reference="robot", mirrored=False,
            require_gripper_detached=True, env_id=0,
        ))
        counter = getattr(self._env, "episode_length_buf", None)
        if counter is None or not hasattr(counter, "zero_"):
            raise RuntimeError("RoboLab does not expose a resettable episode_length_buf")
        counter_before = _numeric(counter)
        counter.zero_()
        counter_after = [int(item) for item in _numeric(counter)]
        evidence = {
            "schema_version": SETTLE_EVIDENCE_SCHEMA,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "registered_cell_id": cell.cell_id,
            "settle_steps": SETTLE_STEPS,
            "stable_window_steps": STABILITY_WINDOW_STEPS,
            "linear_speed_tolerance_m_s": LINEAR_SPEED_TOLERANCE_M_S,
            "angular_speed_tolerance_rad_s": ANGULAR_SPEED_TOLERANCE_RAD_S,
            "hold_action_shape": [1, ACTION_DIM],
            "terminated_or_truncated_during_gate": False,
            "stability_window_component_maxima": maxima,
            "post_settle_velocities": velocities,
            "post_settle_positions_world_xyz_m": positions,
            "post_settle_quaternions_world_wxyz": quaternions,
            "neutral_after_settle": not left and not right,
            "episode_length_buf_before_reset": counter_before,
            "episode_length_buf_reset_passed": counter_after == [0],
            "episode_length_buf_after_reset": counter_after,
            "model_request_count_during_gate": 0,
            "runner_pre_action_reset_calls": self._runner_pre_action_reset_calls,
            "physical_reset_calls": self._physical_reset_calls,
            "settle_gate_runs": self._settle_gate_runs,
            "duplicate_second_reset_idempotent": False,
        }
        validate_settle_stability_evidence(
            evidence, cell=cell, runner_reset_contract_complete=False
        )
        self._settle_evidence = evidence
        # Support RoboLab's harmless multiple pre-action initialization resets.
        self._samples = [self._sample(0)]
        self._started = time.monotonic()
        self._written = False
        self._cached_reset_result = (obs, info)
        return self._cached_reset_result

    def step(self, action: Any) -> Any:
        if not bootstrap.reset_attestation.exists():
            raise RuntimeError("env.step occurred before the reset attestation")
        if not self._partial_started:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            self.partial_path.write_text(
                json.dumps(self._samples[0], sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            self._partial_started = True
        result = self._env.step(action)
        sample = self._sample(len(self._samples))
        self._samples.append(sample)
        with self.partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
        return result

    def write_capture(self) -> Path | None:
        if self._written or not self._samples:
            return None
        predicate = object_left_of if cell.relation == "left" else object_right_of
        success = bool(
            predicate(
                self._env,
                object="rubiks_cube",
                reference_object="bowl",
                frame_of_reference="robot",
                mirrored=False,
                require_gripper_detached=True,
                env_id=0,
            )
        )
        actions = len(self._samples) - 1
        complete = success or actions == ACTION_CAP
        capture = {
            "schema_version": "vla-wam-shared-v3b-nano-state-capture-v1",
            "registered_cell_id": cell.cell_id,
            "attempt_id": f"{args_cli.output_folder_name}:level{cell.level_index}:{cell.relation}",
            "environment_seed": cell.seed,
            "sampling_seed": cell.seed,
            "requested_relation": cell.relation,
            "prompt": cell.row["prompt"],
            "requested_success": success,
            "actions_executed": actions,
            "action_cap": ACTION_CAP,
            "right_censored": not success and actions == ACTION_CAP,
            "final_detached_release": bool(
                object_dropped(self._env, object="rubiks_cube", env_id=0)
            ),
            "first_contact_step": None,
            "first_contact_unavailable_reason": (
                "The exact RoboLab Nano integration exposes grasp and detached-release "
                "conditionals but no verified physical contact stream."
            ),
            "wall_time_s": time.monotonic() - self._started,
            "operational_wall_time_valid": True,
            "samples": self._samples,
            "behavioral_result_valid_candidate": complete,
            "partial_attempt_reason": None if complete else "episode ended before success or 450 actions",
        }
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self.capture_path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
        self._written = True
        return self.capture_path

    def close(self) -> Any:
        self.write_capture()
        return self._env.close()


_create_env = robolab_runtime.create_env
proxies: list[StateCaptureProxy] = []


def _create_captured_env(*args: Any, **kwargs: Any) -> Any:
    kwargs["seed"] = cell.seed
    env, env_cfg = _create_env(*args, **kwargs)
    proxy = StateCaptureProxy(env, bootstrap.state_capture_dir)
    proxies.append(proxy)
    return proxy, env_cfg


robolab_runtime.create_env = _create_captured_env
clients: list[V3BNanoLiveClient] = []


def _ensure_reset() -> Path:
    if len(proxies) != 1:
        raise RuntimeError("exactly one RoboLab environment must exist before Nano request")
    return proxies[0].write_reset_attestation()


def make_client(_: argparse.Namespace) -> V3BNanoLiveClient:
    client = V3BNanoLiveClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        sampling_seed_base=cell.seed,
        action_trace_dir=bootstrap.action_trace_dir,
        future_trace_dir=bootstrap.future_trace_dir,
        cell=cell,
        release=release,
        runtime=runtime_identity,
        reset_attestation_path=bootstrap.reset_attestation,
        ensure_reset_attestation=_ensure_reset,
        behavioral_release_gate_sha256=sha256_file(bootstrap.release_gate),
    )
    clients.append(client)
    return client


def _find_viewport_video() -> Path:
    videos = [path for path in args_cli.output_dir.rglob("*.mp4") if path.is_file()]
    if len(videos) != 1 or videos[0].stat().st_size <= 0:
        raise RuntimeError(f"expected exactly one non-empty viewport video, found {videos}")
    return videos[0].resolve()


def _write_export() -> Path:
    if len(proxies) != 1 or len(clients) != 1:
        raise RuntimeError("exactly one proxy/client is required for a released cell")
    capture_path = proxies[0].capture_path
    trace_path = clients[0].trace_path
    if not capture_path.is_file() or trace_path is None or not trace_path.is_file():
        raise RuntimeError("state capture or action/future trace is missing")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if capture.get("behavioral_result_valid_candidate") is not True:
        raise RuntimeError("partial Phase-B attempt cannot emit a behavioral simulator export")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    actions_path = Path(trace["executed_actions"]["path"]).resolve()
    reset = json.loads(bootstrap.reset_attestation.read_text(encoding="utf-8"))
    reset_fingerprint = sha256_bytes(canonical_json_bytes(reset))
    export = {
        "schema_version": "vla-wam-shared-v3b-nano-simulator-export-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "phase": PHASE,
        "registered_cell_id": cell.cell_id,
        "matched_block_id": cell.row["matched_block_id"],
        "model_id": cell.row["model_id"],
        "level_index": cell.level_index,
        "reference_object_initial_lateral_position_y_m": cell.level_y_m,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "environment_seed": cell.seed,
        "sampling_seed": cell.seed,
        "fixture_id": cell.fixture_id,
        "fixture_sha256": cell.fixture_sha256,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "reset_fingerprint_sha256": reset_fingerprint,
        "runtime_identity_sha256": runtime_identity["runtime_identity_sha256"],
        "behavioral_release_gate_sha256": sha256_file(bootstrap.release_gate),
        "lane_pod_uid": bootstrap.lane_pod_uid,
        "lane_gpu_uuid": bootstrap.lane_gpu_uuid,
        "action_space": "joint_position_8d",
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
        "attempt_id": capture["attempt_id"],
        "initial_state_sha256": reset["initial_state_sha256"],
        "steps": capture["samples"],
        "actions_executed": capture["actions_executed"],
        "executed_action_trace_path": str(actions_path),
        "viewport_video_path": str(_find_viewport_video()),
        "policy_requests": trace["requests"],
        "reset_attestation_path": str(bootstrap.reset_attestation.resolve()),
        "source_artifacts": {
            "state_capture": str(capture_path.resolve()),
            "action_future_trace": str(trace_path.resolve()),
            "safe_fixture": str(safe_fixture_path),
            "behavioral_release_gate": str(bootstrap.release_gate.resolve()),
            "fixture_match_evidence": str(proxies[0].fixture_evidence_path.resolve()),
        },
        "requested_success": capture["requested_success"],
        "right_censored": capture["right_censored"],
        "final_detached_release": capture["final_detached_release"],
        "wall_time_s": capture["wall_time_s"],
        "operational_wall_time_valid": capture["operational_wall_time_valid"],
        "first_contact_step": capture["first_contact_step"],
        "first_contact_unavailable_reason": capture["first_contact_unavailable_reason"],
    }
    bootstrap.simulator_export.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.simulator_export.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n")
    return bootstrap.simulator_export


def main() -> None:
    failure: BaseException | None = None
    try:
        try:
            run_evaluation(args_cli, policy="cosmos3_nano_v2", client_factory=make_client)
        except BaseException as exc:
            failure = exc
        finally:
            for client in clients:
                client.write_trace()
            for proxy in proxies:
                proxy.write_capture()
        if failure is not None:
            bridge_failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": "vla-wam-shared-v3b-nano-bridge-failure-v1",
                        "registered_cell_id": cell.cell_id,
                        "denominator_eligible": False,
                        "error_type": type(failure).__name__,
                        "error": str(failure),
                        "traceback": "".join(
                            traceback.format_exception(
                                type(failure), failure, failure.__traceback__
                            )
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            # Isaac's close path can terminate the process successfully, so all
            # denominator-eligibility evidence must be durable before closing it.
            _write_export()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[Nano V3-B005] infrastructure failure: {error}")
        traceback.print_exc()
        raise
