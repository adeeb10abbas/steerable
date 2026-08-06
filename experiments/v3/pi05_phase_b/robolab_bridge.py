#!/usr/bin/env python3
"""Execute one hash-released π0.5 V3-B002 RoboLab cell.

The proxy reproduces Nano V3-B001's two-runner-reset/one-physical-reset
contract, 60-step settle plus 15-step stability window, and fixture check.  It
also retains the real RoboLab ``object_grabbed`` conditional at every state;
contact remains explicitly unavailable and is never inferred from grasp.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--release-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--release-manifest-sha256", required=True)
BOOTSTRAP.add_argument("--runtime-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--release-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--cell-id", required=True)
BOOTSTRAP.add_argument("--lane-pod-uid", required=True)
BOOTSTRAP.add_argument("--lane-gpu-uuid", required=True)
BOOTSTRAP.add_argument("--fixture-candidate", type=Path, required=True)
BOOTSTRAP.add_argument("--fixture-candidate-sha256", required=True)
BOOTSTRAP.add_argument("--state-capture-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--action-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--reset-attestation", type=Path, required=True)
BOOTSTRAP.add_argument("--simulator-export", type=Path, required=True)
BOOTSTRAP.add_argument("--output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--remote-host", required=True)
BOOTSTRAP.add_argument("--remote-port", type=int, default=8001)
BOOTSTRAP.add_argument("--open-loop-horizon", type=int, default=15)
BOOTSTRAP.add_argument("--instruction-controller", choices=["static"], default="static")
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
sys.path.insert(0, str(study_root))

from experiments.v3.pi05_phase_b.contract import (  # noqa: E402
    ACTION_CAP, ACTION_CHUNK_STEPS, ACTION_DIM, AMENDMENT_ID, MODEL_ID, PHASE,
    STUDY_ID, canonical_json_bytes, load_release_bundle, sha256_bytes,
    sha256_file,
)
from experiments.v3.pi05_phase_b.runtime import (  # noqa: E402
    validate_release_gate, validate_runtime_identity,
)
from tools.vla_wam_v3_episode_schema import (  # noqa: E402
    MEASUREMENT_FRAME_ID, derive_initial_state_sha256,
)


release = load_release_bundle(
    study_root, bootstrap.release_manifest,
    expected_manifest_sha256=bootstrap.release_manifest_sha256,
)
cell = release.cell(bootstrap.cell_id)
runtime_value = json.loads(bootstrap.runtime_manifest.read_text(encoding="utf-8"))
runtime_identity = validate_runtime_identity(
    runtime_value, repo_root=study_root, release=release,
)
gate_value = json.loads(bootstrap.release_gate.read_text(encoding="utf-8"))
validate_release_gate(gate_value, release=release, runtime=runtime_identity)
lane_matches = [lane for lane in runtime_identity["live_topology"]["simulator_lanes"]
                if lane["pod_uid"] == bootstrap.lane_pod_uid
                and lane["gpu_uuid"] == bootstrap.lane_gpu_uuid]
if len(lane_matches) != 1:
    BOOTSTRAP.error("bridge lane is not the exact runtime-bound pod UID/GPU UUID")
live_gpu_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
).splitlines()
if bootstrap.lane_gpu_uuid not in {row.strip() for row in live_gpu_rows}:
    BOOTSTRAP.error("runtime-bound simulator GPU UUID is not visible in this pod")
candidate_path = bootstrap.fixture_candidate.resolve()
if (
    bootstrap.fixture_candidate_sha256
    != "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88"
    or not candidate_path.is_file()
    or sha256_file(candidate_path) != bootstrap.fixture_candidate_sha256
):
    BOOTSTRAP.error("fixture candidate is not the exact Nano V3-B001 candidate")
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
os.environ["VLA_WAM_V3B_FIXTURE_CANDIDATE"] = str(candidate_path)
os.environ["VLA_WAM_V3B_FIXTURE_SHA256"] = bootstrap.fixture_candidate_sha256
if bootstrap.open_loop_horizon != ACTION_CHUNK_STEPS:
    BOOTSTRAP.error("π0.5 V3-B002 open-loop horizon is exactly 15")
for path in (bootstrap.reset_attestation, bootstrap.simulator_export):
    if path.exists():
        BOOTSTRAP.error(f"refusing to overwrite retained evidence: {path}")
for directory in (bootstrap.state_capture_dir, bootstrap.action_trace_dir, bootstrap.output_dir):
    if directory.exists():
        BOOTSTRAP.error(f"refusing to reuse retained output directory: {directory}")

import cv2  # noqa: E402,F401
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
    parser.error("each cell requires one environment, one run, and viewport video")
if args_cli.enable_subtask or args_cli.instruction_type != "default":
    parser.error("V3-B002 permits only a static direct command and no subtask coach")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
import robolab.core.environments.runtime as robolab_runtime  # noqa: E402
from robolab.core.task.conditionals import (  # noqa: E402
    object_dropped, object_grabbed, object_left_of, object_right_of,
)
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402
from experiments.v3.pi05_phase_b.client import V3B002Pi05Client  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False
set_output_dir(str(bootstrap.output_dir.resolve()))

TASKS = {
    ("control", "left"): ("control_left.py", "V3B002Pi05ControlLeftTask"),
    ("control", "right"): ("control_right.py", "V3B002Pi05ControlRightTask"),
    ("position_mirrored", "left"): ("position_mirrored_left.py", "V3B002Pi05PositionMirroredLeftTask"),
    ("position_mirrored", "right"): ("position_mirrored_right.py", "V3B002Pi05PositionMirroredRightTask"),
}
task_root = study_root / "experiments/v3/pi05_phase_b/task_files"
registration_order = (
    ("control", "left"), ("control", "right"),
    ("position_mirrored", "left"), ("position_mirrored", "right"),
)
auto_register_droid_envs(
    task=[str(task_root / TASKS[key][0]) for key in registration_order],
    cameras=WRIST_LEFT_RIGHT_HEAD,
)
args_cli.task = [TASKS[(cell.arm, cell.relation)][1]]


def _inverse_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    inverse = -xyz
    return 2 * np.dot(inverse, v) * inverse + (w*w - np.dot(inverse, inverse))*v + 2*w*np.cross(inverse, v)


def _hold_action(obs: dict[str, Any], device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, ACTION_DIM):
        raise RuntimeError("model-blind hold action is not [1,8]")
    return action


def _host_array(value: Any) -> np.ndarray:
    """Convert simulator values without asking CUDA tensors for ``numpy()``."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


class StateCaptureProxy:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.samples: list[dict[str, Any]] = []
        self.runner_resets = 0
        self.physical_resets = 0
        self.settle_runs = 0
        self.cached_reset: tuple[Any, Any] | None = None
        self.stability: dict[str, Any] | None = None
        self.started = time.monotonic()
        self.written = False
        stem = cell.cell_id.replace(":", "__")
        self.partial_path = bootstrap.state_capture_dir / f"{stem}.states.partial.jsonl"
        self.capture_path = bootstrap.state_capture_dir / f"{stem}.capture.json"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> dict[str, Any]:
        cube = self._env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl = self._env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        robot = self._env.scene["robot"].data
        pos = robot.root_pos_w[0].detach().cpu().numpy()
        quat = robot.root_quat_w[0].detach().cpu().numpy()
        return {
            "action_step": action_step,
            "object_xyz": _inverse_rotate(quat, cube-pos).tolist(),
            "reference_xyz": _inverse_rotate(quat, bowl-pos).tolist(),
            "grippers_open": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
            "object_grabbed": bool(object_grabbed(self._env, object="rubiks_cube", env_id=0)),
        }

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if len(self.samples) > 1 or bootstrap.reset_attestation.exists():
            raise RuntimeError("reset attempted after behavioral authorization")
        self.runner_resets += 1
        if self.runner_resets == 2:
            if self.cached_reset is None or self.physical_resets != 1 or self.settle_runs != 1:
                raise RuntimeError("duplicate reset preceded completed physical reset gate")
            return self.cached_reset
        if self.runner_resets != 1:
            raise RuntimeError("frozen runner must call reset exactly twice before actions")
        self.physical_resets += 1
        result = self._env.reset(*args, **kwargs)
        obs, info = result
        hold = _hold_action(obs, self._env.device)
        self.settle_runs += 1
        for _ in range(60):
            obs, _, terminated, truncated, _ = self._env.step(hold)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("cell terminated during 60-step model-blind settle")
        maxima = {name: {"linear": 0.0, "angular": 0.0} for name in ("rubiks_cube", "bowl", "banana")}
        for _ in range(15):
            obs, _, terminated, truncated, _ = self._env.step(hold)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("cell terminated during 15-step stability window")
            world = get_world(self._env)
            for name, row in maxima.items():
                velocity = _host_array(world.get_velocity(name, env_id=0))
                row["linear"] = max(row["linear"], float(np.max(np.abs(velocity[:3]))))
                row["angular"] = max(row["angular"], float(np.max(np.abs(velocity[3:]))))
        if any(row["linear"] > .02 or row["angular"] > .2 for row in maxima.values()):
            raise RuntimeError("live reset failed Nano's registered stability tolerances")
        self._env.episode_length_buf.zero_()
        self.samples = [self._sample(0)]
        self.stability = {"maxima": maxima, "settle_steps": 60, "stable_window_steps": 15}
        self.started = time.monotonic()
        self.cached_reset = (obs, info)
        return self.cached_reset

    def _fixture_match(self) -> bool:
        expected = candidate["layouts"][cell.arm]["positions_robot_base_m"]
        robot = self._env.scene["robot"].data
        rpos = robot.root_pos_w[0].detach().cpu().numpy()
        rquat = robot.root_quat_w[0].detach().cpu().numpy()
        errors = []
        for name, target in expected.items():
            world = self._env.scene[name].data.root_pos_w[0].detach().cpu().numpy()
            observed = _inverse_rotate(rquat, world-rpos)
            errors.append(float(np.max(np.abs(observed-np.asarray(target, dtype=float)))))
        return max(errors) <= .003

    def write_reset_attestation(self) -> Path:
        if bootstrap.reset_attestation.exists():
            return bootstrap.reset_attestation.resolve()
        if self.runner_resets != 2 or self.physical_resets != 1 or self.settle_runs != 1 or len(self.samples) != 1:
            raise RuntimeError("reset attestation requires two logical/one physical reset")
        sample = self.samples[0]
        delta = np.asarray(sample["object_xyz"])-np.asarray(sample["reference_xyz"])
        distance = math.hypot(float(delta[0]), float(delta[1]))
        neutral = distance > 1e-8 and abs(float(delta[1])/distance) < math.cos(math.radians(45))
        if not neutral or not self._fixture_match():
            raise RuntimeError("live reset does not match the exact neutral B001 fixture")
        initial = derive_initial_state_sha256({"measurement_frame": MEASUREMENT_FRAME_ID, "steps": self.samples})
        value = {
            "schema_version": "vla-wam-shared-v3b-pi05-live-reset-v1",
            "study_id": STUDY_ID, "amendment_id": AMENDMENT_ID, "model_id": MODEL_ID,
            "registered_cell_id": cell.cell_id, "environment_seed": cell.seed,
            "arm": cell.arm, "relation": cell.relation, "passed": True,
            "release_fingerprint_sha256": release.release_fingerprint(cell),
            "runtime_identity_sha256": runtime_identity["runtime_identity_sha256"],
            "initial_state_sha256": initial, "fixture_sha256": cell.row["fixture_sha256"],
            "fixture_candidate_sha256": bootstrap.fixture_candidate_sha256,
            "runner_pre_action_reset_calls": 2, "physical_reset_calls": 1,
            "settle_gate_runs": 1, "duplicate_second_reset_idempotent": True,
            "settle_steps": 60, "stable_window_steps": 15,
            "model_request_count_before_attestation": 0,
        }
        value["reset_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(value))
        bootstrap.reset_attestation.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.reset_attestation.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
        return bootstrap.reset_attestation.resolve()

    def step(self, action: Any) -> Any:
        if not bootstrap.reset_attestation.exists():
            raise RuntimeError("env.step occurred before live reset attestation")
        if not self.partial_path.exists():
            bootstrap.state_capture_dir.mkdir(parents=True, exist_ok=True)
            self.partial_path.write_text(json.dumps(self.samples[0], sort_keys=True)+"\n")
        result = self._env.step(action)
        sample = self._sample(len(self.samples))
        self.samples.append(sample)
        with self.partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True)+"\n")
        return result

    def write_capture(self) -> Path | None:
        if self.written or not self.samples:
            return None
        predicate = object_left_of if cell.relation == "left" else object_right_of
        success = bool(predicate(self._env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
        actions = len(self.samples)-1
        complete = success or actions == ACTION_CAP
        value = {
            "schema_version": "vla-wam-shared-v3b-pi05-state-capture-v1",
            "registered_cell_id": cell.cell_id, "attempt_id": f"{cell.cell_id}:attempt01",
            "environment_seed": cell.seed, "sampling_seed": cell.seed,
            "arm": cell.arm, "requested_relation": cell.relation, "prompt": cell.row["prompt"],
            "requested_success": success, "actions_executed": actions, "action_cap": ACTION_CAP,
            "right_censored": (not success and actions == ACTION_CAP),
            "final_detached_release": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
            "first_contact_step": None,
            "first_contact_unavailable_reason": "The pinned RoboLab integration exposes object_grabbed and detached release but no verified physical contact stream; grasp is not substituted for contact.",
            "wall_time_s": time.monotonic()-self.started, "operational_wall_time_valid": True,
            "samples": self.samples, "behavioral_result_valid_candidate": complete,
            "partial_attempt_reason": None if complete else "episode ended before success or 450 actions",
        }
        bootstrap.state_capture_dir.mkdir(parents=True, exist_ok=True)
        self.capture_path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
        self.written = True
        return self.capture_path

    def close(self) -> Any:
        self.write_capture()
        return self._env.close()


create_env = robolab_runtime.create_env
proxies: list[StateCaptureProxy] = []
clients: list[V3B002Pi05Client] = []


def captured_create_env(*args: Any, **kwargs: Any) -> Any:
    kwargs["seed"] = cell.seed
    env, cfg = create_env(*args, **kwargs)
    proxy = StateCaptureProxy(env)
    proxies.append(proxy)
    return proxy, cfg


robolab_runtime.create_env = captured_create_env


def ensure_reset() -> Path:
    if len(proxies) != 1:
        raise RuntimeError("exactly one environment is required before request zero")
    return proxies[0].write_reset_attestation()


def make_client(_: argparse.Namespace) -> V3B002Pi05Client:
    client = V3B002Pi05Client(
        remote_host=args_cli.remote_host, remote_port=args_cli.remote_port,
        sampling_seed_base=cell.seed, action_trace_dir=bootstrap.action_trace_dir,
        prompt=cell.row["prompt"], release_fingerprint_sha256=release.release_fingerprint(cell),
        runtime_identity_sha256=runtime_identity["runtime_identity_sha256"],
        reset_attestation_path=bootstrap.reset_attestation,
        ensure_reset_attestation=ensure_reset,
    )
    clients.append(client)
    return client


def _video() -> Path:
    videos = [path for path in bootstrap.output_dir.rglob("*.mp4") if path.is_file() and path.stat().st_size]
    if len(videos) != 1:
        raise RuntimeError(f"expected exactly one viewport video, found {videos}")
    return videos[0].resolve()


def write_export() -> None:
    if len(proxies) != 1 or len(clients) != 1:
        raise RuntimeError("one proxy/client required")
    capture = json.loads(proxies[0].capture_path.read_text())
    if capture.get("behavioral_result_valid_candidate") is not True or clients[0].trace_path is None:
        raise RuntimeError("partial attempt cannot emit behavioral export")
    trace = json.loads(clients[0].trace_path.read_text())
    reset = json.loads(bootstrap.reset_attestation.read_text())
    export = {
        "schema_version": "vla-wam-shared-v3b-pi05-simulator-export-v1",
        "study_id": STUDY_ID, "amendment_id": AMENDMENT_ID, "phase": PHASE,
        "registered_cell_id": cell.cell_id, "matched_block_id": cell.row["matched_block_id"],
        "model_id": MODEL_ID, "arm": cell.arm, "requested_relation": cell.relation,
        "prompt": cell.row["prompt"], "prompt_sha256": cell.row["prompt_sha256"],
        "environment_seed": cell.seed, "sampling_seed": cell.seed,
        "fixture_id": cell.row["fixture_id"], "fixture_sha256": cell.row["fixture_sha256"],
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "reset_fingerprint_sha256": reset["reset_fingerprint_sha256"],
        "runtime_identity_sha256": runtime_identity["runtime_identity_sha256"],
        "action_space": "joint_position_8d", "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP, "instruction_controller": "static",
        "attempt_id": capture["attempt_id"], "initial_state_sha256": reset["initial_state_sha256"],
        "steps": capture["samples"], "actions_executed": capture["actions_executed"],
        "executed_action_trace": trace["executed_actions"],
        "returned_action_chunks": trace["returned_action_chunks"],
        "request_sampling_seeds": trace["request_sampling_seeds"],
        "viewport_video_path": str(_video()), "action_trace_metadata_path": str(clients[0].trace_path),
        "reset_attestation_path": str(bootstrap.reset_attestation.resolve()),
        "requested_success": capture["requested_success"], "right_censored": capture["right_censored"],
        "final_detached_release": capture["final_detached_release"],
        "wall_time_s": capture["wall_time_s"], "operational_wall_time_valid": True,
        "first_contact_step": None, "first_contact_unavailable_reason": capture["first_contact_unavailable_reason"],
        "future_interface": "actions_only", "missing_future_policy": "action_only_interface_not_applicable_never_zero",
    }
    bootstrap.simulator_export.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.simulator_export.write_text(json.dumps(export, indent=2, sort_keys=True)+"\n")


def main() -> None:
    failure: BaseException | None = None
    try:
        try:
            run_evaluation(args_cli, policy="pi05_v2a010_current", client_factory=make_client)
        except BaseException as exc:
            failure = exc
        finally:
            for client in clients:
                client.write_trace()
            for proxy in proxies:
                proxy.write_capture()
        if failure is not None:
            raise failure
        write_export()
    except BaseException as exc:
        path = bootstrap.simulator_export.with_name("bridge_failure.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps({"registered_cell_id": cell.cell_id, "denominator_eligible": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}, indent=2)+"\n")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
