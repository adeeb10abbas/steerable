#!/usr/bin/env python3
"""Run one authorized v3 GR00T pair through the unchanged v2 integration.

The adapter contract is checked before importing Isaac Lab.  This bridge
changes only the registered seed range and adds post-action state capture.  It
reuses the v2 client, v2 seeded server protocol, v2 task/reset/scorer files,
and RoboLab's evaluator.  There is no outcome-dependent early stop beyond the
unchanged success termination and the unchanged 30-second task timeout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--environment-seed", type=int, required=True)
BOOTSTRAP.add_argument("--sampling-seed-base", type=int, required=True)
BOOTSTRAP.add_argument("--runtime-identity", type=Path, required=True)
BOOTSTRAP.add_argument("--release-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--state-capture-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--action-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--remote-host", required=True)
BOOTSTRAP.add_argument("--remote-port", type=int, default=5555)
BOOTSTRAP.add_argument("--open-loop-horizon", type=int, default=8)
BOOTSTRAP.add_argument("--instruction-controller", choices=["static"], default="static")
BOOTSTRAP.add_argument("--condition", choices=["left", "right", "both"], default="both")
bootstrap, _ = BOOTSTRAP.parse_known_args()

sys.path.insert(0, str(bootstrap.study_root / "experiments/v3/groot_droid"))
from adapter import CAPTURE_SCHEMA, PROMPTS, TASKS, preflight  # noqa: E402

if bootstrap.environment_seed != bootstrap.sampling_seed_base:
    BOOTSTRAP.error("environment and sampling seed must match")
if bootstrap.condition != "both":
    BOOTSTRAP.error("a fresh v3 launch must execute the complete matched pair")
if bootstrap.open_loop_horizon != 8:
    BOOTSTRAP.error("the frozen GR00T open-loop horizon is 8")
preflight(
    bootstrap.study_root.resolve(),
    bootstrap.environment_seed,
    bootstrap.runtime_identity,
    bootstrap.release_gate,
    check_live_repositories=True,
)

import cv2  # noqa: E402,F401 -- RoboLab requires this before Isaac Lab
import numpy as np  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.video_mode != "viewport":
    parser.error("v3 GR00T requires viewport video")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("each registered cell requires one environment and one run")
if args_cli.enable_subtask:
    parser.error("progress-conditioned subtask coaching is prohibited")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.core.task.conditionals import (  # noqa: E402
    object_dropped,
    object_left_of,
    object_right_of,
)
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)

sys.path.insert(0, str(args_cli.study_root / "experiments/groot_droid"))
from v2_robolab_client import V2GR00TDroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

left_task = args_cli.study_root / (
    "experiments/groot_droid/robolab_v2_tasks/"
    "rubiks_cube_left_of_bowl_matched.py"
)
right_task = args_cli.study_root / (
    "experiments/groot_droid/robolab_v2_tasks/"
    "rubiks_cube_right_of_bowl_matched.py"
)
auto_register_droid_envs(task=[str(left_task), str(right_task)])
args_cli.task = [TASKS["left"], TASKS["right"]]


def _quat_inverse_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the robot frame for Isaac's wxyz quaternion."""

    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("robot pose must provide a 4D quaternion and 3D vector")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("robot quaternion is invalid")
    q = q / norm
    w, xyz = q[0], q[1:]
    # q^-1 * v * q, using the vector rotation identity.
    inverse_xyz = -xyz
    return (
        2.0 * np.dot(inverse_xyz, v) * inverse_xyz
        + (w * w - np.dot(inverse_xyz, inverse_xyz)) * v
        + 2.0 * w * np.cross(inverse_xyz, v)
    )


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _in_requested_region(sample: dict[str, Any], relation: str) -> bool:
    obj = np.asarray(sample["object_xyz"], dtype=np.float64)
    ref = np.asarray(sample["reference_xyz"], dtype=np.float64)
    delta = obj - ref
    forward, lateral = float(delta[0]), float(delta[1])
    horizontal_distance = math.hypot(forward, lateral)
    requested_margin = lateral if relation == "left" else -lateral
    return (
        horizontal_distance > 1e-8
        and requested_margin / horizontal_distance >= math.cos(math.radians(45.0))
    )


class StateCaptureProxy:
    """Transparent environment proxy retaining initial and post-action states."""

    def __init__(self, env: Any, task_name: str, capture_dir: Path) -> None:
        self._env = env
        self._task_name = task_name
        self._relation = "left" if "Left" in task_name else "right"
        self._capture_dir = capture_dir
        self._samples: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._written = False
        self._stream_path = (
            capture_dir
            / f"seed{args_cli.environment_seed}_{self._relation}_states.partial.jsonl"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> None:
        cube_world = self._env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl_world = self._env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        robot = self._env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        cube_robot = _quat_inverse_rotate_wxyz(robot_quat, cube_world - robot_pos)
        bowl_robot = _quat_inverse_rotate_wxyz(robot_quat, bowl_world - robot_pos)
        detached = bool(object_dropped(self._env, object="rubiks_cube", env_id=0))
        sample = {
            "action_step": action_step,
            "object_xyz": cube_robot.tolist(),
            "reference_xyz": bowl_robot.tolist(),
            # The shared schema names this historical stream grippers_open.
            # Its source is explicitly the same object_dropped conditional used
            # by the frozen task, never an action-command inference.
            "grippers_open": detached,
            "final_detached_release": detached,
        }
        self._samples.append(sample)
        # Append-only recovery stream: a normal exception still leaves every
        # state observed before the interruption on the PVC.
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        with self._stream_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        final_path = self._capture_dir / (
            f"seed{args_cli.environment_seed}_{self._relation}.json"
        )
        if self._stream_path.exists() or final_path.exists():
            raise FileExistsError(
                "refusing to overwrite retained GR00T v3 state evidence: "
                f"{self._stream_path} or {final_path}"
            )
        result = self._env.reset(*args, **kwargs)
        self._samples = []
        self._started = time.monotonic()
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self._stream_path.touch(exist_ok=False)
        self._sample(0)
        return result

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        self._sample(len(self._samples))
        return result

    def _write(self) -> None:
        if self._written or not self._samples:
            return
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        final_detached = bool(self._samples[-1]["final_detached_release"])
        predicate = object_left_of if self._relation == "left" else object_right_of
        requested_success = bool(predicate(
            self._env,
            object="rubiks_cube",
            reference_object="bowl",
            frame_of_reference="robot",
            mirrored=False,
            require_gripper_detached=True,
            env_id=0,
        ))
        actions_executed = len(self._samples) - 1
        action_cap = 450
        complete_behavioral_candidate = requested_success or actions_executed == action_cap
        object_points = [
            np.asarray(sample["object_xyz"], dtype=np.float64) for sample in self._samples
        ]
        initial_object = object_points[0]
        interaction = _first_sustained([
            float(np.linalg.norm(point - initial_object)) >= 0.01
            for point in object_points
        ])
        pickup = _first_sustained([
            float(point[2] - initial_object[2]) >= 0.03 for point in object_points
        ])
        entered_requested = any(
            _in_requested_region(sample, self._relation) for sample in self._samples
        )
        failure_stage = (
            "success"
            if requested_success
            else "no_object_interaction"
            if interaction is None
            else "object_moved_no_verified_pickup"
            if pickup is None
            else "picked_never_entered_requested_region"
            if not entered_requested
            else "entered_requested_region_not_released"
        )
        capture = {
            "schema_version": CAPTURE_SCHEMA,
            "registered_cell_id": (
                f"v3:droid:groot_n17_droid_vla:seed{args_cli.environment_seed}:"
                f"{self._relation}"
            ),
            "attempt_id": (
                f"v3-groot-seed{args_cli.environment_seed}-{self._relation}-attempt0"
            ),
            "environment_seed": args_cli.environment_seed,
            "policy_seed": args_cli.sampling_seed_base,
            "requested_relation": self._relation,
            "prompt": PROMPTS[self._relation],
            "actions_executed": actions_executed,
            "action_cap": action_cap,
            "right_censored": not requested_success and actions_executed == action_cap,
            "requested_success": requested_success,
            "final_detached_release": final_detached,
            "frozen_failure_stage": failure_stage,
            "first_contact_step": None,
            "first_contact_unavailable_reason": (
                "RoboLab frozen GR00T adapter exposes grasp and detached-release "
                "conditionals but no verified physical contact stream"
            ),
            "wall_time_s": time.monotonic() - self._started,
            "operational_wall_time_valid": True,
            "samples": [
                {key: value for key, value in sample.items() if key != "final_detached_release"}
                for sample in self._samples
            ],
            "capture_contract": {
                "coordinates": "robot-base frame from Isaac root pose wxyz inverse rotation",
                "detached_release": "robolab.core.task.conditionals.object_dropped",
                "failure_stage_pickup_proxy": "sustained 3 cm object lift from initial state",
                "contact": "instrumentation_unavailable; never substituted with grasp",
                "partial_state_stream": str(self._stream_path),
            },
            "behavioral_result_valid_candidate": complete_behavioral_candidate,
            "partial_attempt_reason": (
                None if complete_behavioral_candidate
                else "episode ended before frozen success termination or action cap"
            ),
        }
        path = self._capture_dir / f"seed{args_cli.environment_seed}_{self._relation}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite GR00T v3 capture: {path}")
        path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
        self._written = True

    def close(self) -> Any:
        self._write()
        return self._env.close()


_create_env = runtime.create_env


def _seeded_captured_create_env(*args: Any, **kwargs: Any) -> Any:
    kwargs["seed"] = args_cli.environment_seed
    env, env_cfg = _create_env(*args, **kwargs)
    task_name = str(args[0] if args else kwargs.get("task"))
    return StateCaptureProxy(env, task_name, args_cli.state_capture_dir), env_cfg


runtime.create_env = _seeded_captured_create_env


_clients: list[V2GR00TDroidJointposClient] = []


def make_client(_: argparse.Namespace) -> V2GR00TDroidJointposClient:
    client = V2GR00TDroidJointposClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        open_loop_horizon=args_cli.open_loop_horizon,
        sampling_seed_base=args_cli.sampling_seed_base,
        action_trace_dir=args_cli.action_trace_dir,
    )
    _clients.append(client)
    return client


def main() -> None:
    try:
        run_evaluation(args_cli, policy="groot_v2", client_factory=make_client)
    finally:
        # Preserve any in-memory actions on normal Python exceptions. The v2
        # client remains the source of truth for serialization and hashes.
        for client in _clients:
            client._write_trace()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[GR00T v3] technical failure: {error}")
        traceback.print_exc()
        raise
