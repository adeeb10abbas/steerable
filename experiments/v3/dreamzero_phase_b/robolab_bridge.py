#!/usr/bin/env python3
"""Run one released DreamZero V3-B003 cell with full raw retention."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--release-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--release-manifest-sha256", required=True)
BOOTSTRAP.add_argument("--cell-id", required=True)
BOOTSTRAP.add_argument("--runtime-identity", type=Path, required=True)
BOOTSTRAP.add_argument("--release-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--fixture-candidate", type=Path, required=True)
BOOTSTRAP.add_argument("--fixture-candidate-sha256", required=True)
BOOTSTRAP.add_argument("--state-capture-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--action-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--reset-attestation", type=Path, required=True)
BOOTSTRAP.add_argument("--simulator-export", type=Path, required=True)
BOOTSTRAP.add_argument("--output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--remote-host", required=True)
BOOTSTRAP.add_argument("--remote-port", type=int, required=True)
BOOTSTRAP.add_argument("--lane-pod-uid", required=True)
BOOTSTRAP.add_argument("--lane-gpu-uuid", required=True)
BOOTSTRAP.add_argument("--open-loop-horizon", type=int, default=8)
BOOTSTRAP.add_argument("--instruction-controller", choices=("static",), default="static")
bootstrap, _ = BOOTSTRAP.parse_known_args()
sys.path.insert(0, str(bootstrap.study_root.resolve()))

from experiments.v3.dreamzero_droid.adapter import (  # noqa: E402
    CAPTURE_SCHEMA,
    OFFICIAL_NOISE_SEED,
    validate_runtime_identity,
)
from experiments.v3.dreamzero_phase_b.contract import (  # noqa: E402
    EXPECTED_SHA256,
    FIXTURE_CANDIDATE_SHA256,
    PROMPTS,
    load_cell,
    sha256_file,
    validate_release_gate,
)

cell = load_cell(bootstrap.study_root, bootstrap.cell_id)
if bootstrap.release_manifest_sha256 != EXPECTED_SHA256["manifest"]:
    BOOTSTRAP.error("DreamZero V3-B003 manifest argument changed")
if (
    not bootstrap.release_manifest.is_file()
    or sha256_file(bootstrap.release_manifest) != EXPECTED_SHA256["manifest"]
):
    BOOTSTRAP.error("DreamZero V3-B003 release manifest bytes changed")
if (
    bootstrap.fixture_candidate_sha256 != FIXTURE_CANDIDATE_SHA256
    or not bootstrap.fixture_candidate.is_file()
    or sha256_file(bootstrap.fixture_candidate) != FIXTURE_CANDIDATE_SHA256
):
    BOOTSTRAP.error("DreamZero V3-B003 fixture candidate changed")
# The reused, hash-pinned RoboLab task module loads the registered fixture at
# import time.  Bind the already-validated path and digest before any Isaac or
# task imports so the behavioral bridge uses exactly the model-blind gate's
# immutable object coordinates.
os.environ["VLA_WAM_V3B_FIXTURE_CANDIDATE"] = str(
    bootstrap.fixture_candidate.resolve()
)
os.environ["VLA_WAM_V3B_FIXTURE_SHA256"] = bootstrap.fixture_candidate_sha256
if bootstrap.open_loop_horizon != 8 or bootstrap.remote_port == 5000:
    BOOTSTRAP.error("DreamZero V3-B003 requires horizon 8 and an isolated non-5000 port")
runtime_identity = validate_runtime_identity(
    bootstrap.study_root,
    bootstrap.runtime_identity,
    check_live_repositories=True,
)
release_gate = validate_release_gate(
    bootstrap.release_gate,
    repo_root=bootstrap.study_root,
    runtime_identity=bootstrap.runtime_identity,
    lane_pod_uid=bootstrap.lane_pod_uid,
    lane_gpu_uuid=bootstrap.lane_gpu_uuid,
)
future_root = Path(str(release_gate["future_root"])).resolve()
server_contract_path = Path(str(release_gate["server_contract"]["path"])).resolve()

import cv2  # noqa: E402,F401
import numpy as np  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.video_mode != "viewport" or args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("each V3-B003 cell requires one environment, one run, and viewport video")
if args_cli.device != "cuda:0" or args_cli.enable_subtask:
    parser.error("V3-B003 requires cuda:0 and prohibits progress-conditioned coaching")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.core.task.conditionals import object_dropped, object_left_of, object_right_of  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT  # noqa: E402

from experiments.v3.dreamzero_phase_b.client import V3B003DreamZeroClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

TASKS = {
    ("control", "left"): ("control_left.py", "V3B002Pi05ControlLeftTask"),
    ("control", "right"): ("control_right.py", "V3B002Pi05ControlRightTask"),
    ("position_mirrored", "left"): (
        "position_mirrored_left.py", "V3B002Pi05PositionMirroredLeftTask"
    ),
    ("position_mirrored", "right"): (
        "position_mirrored_right.py", "V3B002Pi05PositionMirroredRightTask"
    ),
}
task_file, task_name = TASKS[(cell.arm, cell.relation)]
task_path = args_cli.study_root / "experiments/v3/pi05_phase_b/task_files" / task_file
auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT)
args_cli.task = [task_name]
fixture = json.loads(bootstrap.fixture_candidate.read_text())
expected_positions = fixture["layouts"][cell.arm]["positions_robot_base_m"]


def _numeric(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


def _quat_inverse_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    inverse_xyz = -xyz
    return (
        2 * np.dot(inverse_xyz, vector) * inverse_xyz
        + (w * w - np.dot(inverse_xyz, inverse_xyz)) * vector
        + 2 * w * np.cross(inverse_xyz, vector)
    )


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _in_region(sample: dict[str, Any], relation: str) -> bool:
    delta = np.asarray(sample["object_xyz"]) - np.asarray(sample["reference_xyz"])
    distance = math.hypot(float(delta[0]), float(delta[1]))
    margin = float(delta[1]) if relation == "left" else -float(delta[1])
    return distance > 1e-8 and margin / distance >= math.cos(math.radians(45))


class StateCaptureProxy:
    def __init__(self, env: Any) -> None:
        self._env = env
        self._samples: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._written = False
        self._runner_pre_action_reset_calls = 0
        self._physical_reset_calls = 0
        self._cached_reset_result: tuple[Any, Any] | None = None
        self._reset_attestation_payload: dict[str, Any] | None = None
        self._partial = bootstrap.state_capture_dir / "states.partial.jsonl"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> None:
        cube_world = self._env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl_world = self._env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        robot = self._env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        sample = {
            "action_step": action_step,
            "object_xyz": _quat_inverse_rotate_wxyz(robot_quat, cube_world - robot_pos).tolist(),
            "reference_xyz": _quat_inverse_rotate_wxyz(robot_quat, bowl_world - robot_pos).tolist(),
            "grippers_open": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
        }
        self._samples.append(sample)
        with self._partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if self._partial.exists() or len(self._samples) > 1:
            raise RuntimeError("reset attempted after V3-B003 behavioral authorization")
        self._runner_pre_action_reset_calls += 1
        if self._runner_pre_action_reset_calls == 2:
            if (
                self._cached_reset_result is None
                or self._physical_reset_calls != 1
                or self._reset_attestation_payload is None
                or len(self._samples) != 1
                or bootstrap.reset_attestation.exists()
            ):
                raise RuntimeError("duplicate reset preceded the completed V3-B003 physical reset")
            self._reset_attestation_payload.update({
                "runner_pre_action_reset_calls": 2,
                "physical_reset_calls": 1,
                "duplicate_second_reset_idempotent": True,
            })
            bootstrap.reset_attestation.parent.mkdir(parents=True, exist_ok=True)
            bootstrap.reset_attestation.write_text(
                json.dumps(self._reset_attestation_payload, indent=2, sort_keys=True) + "\n"
            )
            return self._cached_reset_result
        if self._runner_pre_action_reset_calls != 1:
            raise RuntimeError("frozen RoboLab runner must call reset exactly twice before actions")
        if bootstrap.reset_attestation.exists():
            raise FileExistsError("refusing to overwrite V3-B003 reset attestation")
        self._physical_reset_calls += 1
        counter = getattr(self._env, "episode_length_buf", None)
        if counter is None or not hasattr(counter, "zero_"):
            raise RuntimeError("RoboLab reset counter is unavailable")
        before = _numeric(counter)
        counter.zero_()
        result = self._env.reset(*args, **kwargs)
        if [int(item) for item in _numeric(counter)] != [0]:
            raise RuntimeError("fresh V3-B003 physical reset was not attested")
        world = get_world(self._env)
        positions = {
            name: _numeric(world.get_pose(name, env_id=0)[0])
            for name in ("rubiks_cube", "bowl", "banana")
        }
        maximum_error = max(
            abs(observed - expected)
            for name in positions
            for observed, expected in zip(positions[name], expected_positions[name])
        )
        if maximum_error > 0.005:
            raise RuntimeError(f"V3-B003 reset missed the fixture by {maximum_error} m")
        self._reset_attestation_payload = {
            "schema_version": "vla-wam-shared-v3b-dreamzero-reset-attestation-v1",
            "passed": True,
            "registered_cell_id": cell.cell_id,
            "arm": cell.arm,
            "relation": cell.relation,
            "prompt": cell.row["prompt"],
            "environment_seed": cell.seed,
            "model_request_count_at_write": 0,
            "episode_length_buf_before_force_reset": before,
            "positions_robot_base_m": positions,
            "maximum_fixture_position_error_m": maximum_error,
            "fixture_candidate_sha256": FIXTURE_CANDIDATE_SHA256,
            "runner_pre_action_reset_calls": 1,
            "physical_reset_calls": 1,
            "duplicate_second_reset_idempotent": False,
        }
        self._samples = []
        self._started = time.monotonic()
        self._sample(0)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("RoboLab reset did not return (observation, info)")
        self._cached_reset_result = result
        return self._cached_reset_result

    def step(self, action: Any) -> Any:
        if not bootstrap.reset_attestation.exists():
            raise RuntimeError("V3-B003 action occurred before reset attestation")
        if not self._partial.exists():
            bootstrap.state_capture_dir.mkdir(parents=True, exist_ok=True)
            self._partial.write_text(
                json.dumps(self._samples[0], sort_keys=True, separators=(",", ":")) + "\n"
            )
        result = self._env.step(action)
        self._sample(len(self._samples))
        return result

    def write_capture(self) -> Path | None:
        if self._written or not self._samples:
            return None
        relation = cell.relation
        predicate = object_left_of if relation == "left" else object_right_of
        success = bool(predicate(
            self._env, object="rubiks_cube", reference_object="bowl",
            frame_of_reference="robot", mirrored=False,
            require_gripper_detached=True, env_id=0,
        ))
        action_count = len(self._samples) - 1
        object_xyz = [np.asarray(row["object_xyz"], dtype=float) for row in self._samples]
        z0 = float(object_xyz[0][2])
        pickup_step = _first_sustained([float(point[2]) - z0 >= 0.03 for point in object_xyz])
        requested_mask = [_in_region(row, relation) for row in self._samples]
        opposite = "right" if relation == "left" else "left"
        opposite_mask = [_in_region(row, opposite) for row in self._samples]
        entry_step = next((index for index, value in enumerate(requested_mask) if value), None)
        sustained_step = _first_sustained(requested_mask)
        final_delta = object_xyz[-1] - np.asarray(self._samples[-1]["reference_xyz"], dtype=float)
        signed_offset = float(final_delta[1])
        requested_depth = signed_offset if relation == "left" else -signed_offset
        detached = bool(object_dropped(self._env, object="rubiks_cube", env_id=0))
        if success:
            failure_category = "correct"
        elif pickup_step is None:
            failure_category = "pick_failed"
        elif any(opposite_mask) and not any(requested_mask):
            failure_category = "wrong_side"
        elif any(requested_mask) and not detached:
            failure_category = "release_failed"
        else:
            failure_category = "transport_failed"
        path_length = float(sum(
            np.linalg.norm(right - left) for left, right in zip(object_xyz, object_xyz[1:])
        ))
        lateral_path = float(sum(
            abs(float(right[1] - left[1])) for left, right in zip(object_xyz, object_xyz[1:])
        ))
        output = bootstrap.state_capture_dir / "capture.json"
        capture = {
            "schema_version": CAPTURE_SCHEMA,
            "registered_cell_id": cell.cell_id,
            "attempt_id": f"{cell.cell_id}:attempt01",
            "identity_binding": "V2-A015:dreamzero_action_cfg_s2",
            "amendment_id": "V3-B003",
            "arm": cell.arm,
            "environment_seed": cell.seed,
            "policy_seed": cell.seed,
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
            "requested_relation": relation,
            "prompt": cell.row["prompt"],
            "requested_success": success,
            "frozen_failure_stage": failure_category,
            "failure_taxonomy": failure_category,
            "actions_executed": action_count,
            "action_cap": 450,
            "right_censored": not success and action_count == 450,
            "final_detached_release": detached,
            "first_contact_step": None,
            "first_contact_unavailable_reason": "RoboLab integration exposes no verified contact stream.",
            "grasp_step": pickup_step,
            "cone_entry_step": entry_step,
            "cone_entry_sustained": sustained_step is not None,
            "signed_final_lateral_offset_m": signed_offset,
            "requested_side_depth_m": requested_depth,
            "object_path_length_m": path_length,
            "cumulative_lateral_path_m": lateral_path,
            "peak_lateral_excursion_m": float(max(abs(point[1] - object_xyz[0][1]) for point in object_xyz)),
            "episode_length_steps": action_count,
            "wall_time_s": time.monotonic() - self._started,
            "operational_wall_time_valid": True,
            "samples": self._samples,
            "behavioral_result_valid_candidate": success or action_count == 450,
            "partial_attempt_reason": None if success or action_count == 450 else "ended_before_success_or_cap",
        }
        bootstrap.state_capture_dir.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite V3-B003 state capture: {output}")
        output.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
        self._written = True
        return output

    def close(self) -> Any:
        self.write_capture()
        return self._env.close()


_create_env = runtime.create_env
proxies: list[StateCaptureProxy] = []


def _captured_create_env(*args: Any, **kwargs: Any):
    kwargs["seed"] = cell.seed
    env, env_cfg = _create_env(*args, **kwargs)
    if env_cfg.instruction != cell.row["prompt"]:
        raise RuntimeError("registered DreamZero prompt bytes changed")
    proxy = StateCaptureProxy(env)
    proxies.append(proxy)
    return proxy, env_cfg


runtime.create_env = _captured_create_env
clients: list[V3B003DreamZeroClient] = []


def make_client(_: argparse.Namespace) -> V3B003DreamZeroClient:
    client = V3B003DreamZeroClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        environment_seed=cell.seed,
        cell_id=cell.cell_id,
        reset_attestation=bootstrap.reset_attestation,
        action_trace_dir=bootstrap.action_trace_dir,
        server_contract_path=server_contract_path,
        release_gate_path=bootstrap.release_gate,
        future_root=future_root,
    )
    clients.append(client)
    return client


def main() -> None:
    failure: BaseException | None = None
    try:
        try:
            run_evaluation(args_cli, policy="dreamzero_v2", client_factory=make_client)
        except BaseException as error:
            failure = error
        finally:
            for client_instance in clients:
                client_instance.write_trace()
            for proxy in proxies:
                proxy.write_capture()
        if failure is not None:
            # Isaac's close path can terminate the process with exit code zero.
            # Persist the causal exception first so the queue can still classify
            # this attempt as infrastructure-invalid and outside the denominator.
            bridge_failure_path.parent.mkdir(parents=True, exist_ok=True)
            bridge_failure_path.write_text(json.dumps({
                "schema_version": "vla-wam-shared-v3b-dreamzero-infrastructure-failure-v1",
                "registered_cell_id": cell.cell_id,
                "behavioral_result_valid": False,
                "denominator_policy": "excluded_from_behavioral_denominator",
                "error": f"{type(failure).__name__}: {failure}",
                "traceback": "".join(traceback.format_exception(
                    type(failure), failure, failure.__traceback__
                )),
            }, indent=2, sort_keys=True) + "\n")
        else:
            videos = [
                str(path.resolve())
                for path in bootstrap.output_dir.rglob("*.mp4")
                if path.is_file() and path.stat().st_size > 0
            ]
            if len(videos) != 1:
                raise RuntimeError(f"expected one V3-B003 viewport video, found {videos}")
            capture_path = bootstrap.state_capture_dir / "capture.json"
            trace_path = bootstrap.action_trace_dir / f"seed{cell.seed}_{cell.relation}_executed_actions.json"
            if not capture_path.is_file() or not trace_path.is_file():
                raise RuntimeError("V3-B003 state capture or action trace is missing")
            capture = json.loads(capture_path.read_text())
            if capture.get("behavioral_result_valid_candidate") is not True:
                raise RuntimeError("partial V3-B003 attempt cannot emit a behavioral export")
            # Make denominator-eligibility evidence durable before closing Isaac.
            bootstrap.simulator_export.parent.mkdir(parents=True, exist_ok=True)
            bootstrap.simulator_export.write_text(json.dumps({
                "schema_version": "vla-wam-shared-v3b-dreamzero-simulator-export-v1",
                "registered_cell_id": cell.cell_id,
                "capture_path": str(capture_path.resolve()),
                "trace_manifest_path": str(trace_path.resolve()),
                "viewport_video_path": videos[0],
                "reset_attestation_path": str(bootstrap.reset_attestation.resolve()),
                "runtime_identity_path": str(bootstrap.runtime_identity.resolve()),
                "release_gate_path": str(bootstrap.release_gate.resolve()),
            }, indent=2, sort_keys=True) + "\n")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = bootstrap.simulator_export.with_name("bridge_failure.json")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps({
            "schema_version": "vla-wam-shared-v3b-dreamzero-infrastructure-failure-v1",
            "registered_cell_id": bootstrap.cell_id,
            "behavioral_result_valid": False,
            "denominator_policy": "excluded_from_behavioral_denominator",
            "error": f"{type(error).__name__}: {error}",
        }, indent=2, sort_keys=True) + "\n")
        traceback.print_exc()
        raise
