#!/usr/bin/env python3
"""Run one authorized V3-A002 π0-FAST matched DROID pair.

This runner uses the frozen V3-A002 public old-name-config compatibility
identity.  It preserves the exact direct prompts and task overlays, executes a
ten-action open-loop horizon, retains every returned/executed action and every
scored simulator state, records a viewport video for each cell, and never
terminates a behavioral failure before the frozen 450-action cap.
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
BOOTSTRAP.add_argument("--remote-port", type=int, default=8011)
BOOTSTRAP.add_argument("--open-loop-horizon", type=int, default=10)
BOOTSTRAP.add_argument("--instruction-controller", choices=["static"], default="static")
BOOTSTRAP.add_argument(
    "--condition", choices=["both", "left", "right"], default="both"
)
bootstrap, _ = BOOTSTRAP.parse_known_args()

sys.path.insert(0, str(bootstrap.study_root / "experiments/v3/pi0_fast_old_name_config_bridge"))
from adapter import CAPTURE_SCHEMA, PROMPTS, TASKS, preflight  # noqa: E402

if bootstrap.environment_seed != bootstrap.sampling_seed_base:
    BOOTSTRAP.error("environment and sampling seed bases must match")
if bootstrap.open_loop_horizon != 10:
    BOOTSTRAP.error("the frozen V3-A002 π0-FAST old-name-config bridge horizon is 10")
AUTHORIZATION = preflight(
    bootstrap.study_root.resolve(),
    bootstrap.environment_seed,
    bootstrap.runtime_identity,
    bootstrap.release_gate,
    check_live_repositories=True,
)
SIMULATOR_POD = AUTHORIZATION["runtime_identity"]["target_kubernetes"]["simulator"]["pod"]

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
    parser.error("π0-FAST old-name-config bridge v3 requires viewport video")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("each registered cell requires one environment and one run")
if args_cli.enable_subtask:
    parser.error("progress-conditioned subtask coaching is prohibited")
if args_cli.instruction_type != "default":
    parser.error("π0-FAST old-name-config bridge v3 uses only the frozen direct-command prompt")
expected_output_prefix = (
    f"v3a002_pi0_fast_old_name_config_bridge_seed{args_cli.environment_seed}_"
    f"{args_cli.condition}_attempt"
)
if not args_cli.output_folder_name.startswith(expected_output_prefix):
    parser.error(
        "--output-folder-name must begin with " + repr(expected_output_prefix)
    )

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

from client import V3Pi0FastOldNameConfigClient  # noqa: E402

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
task_paths = {"left": left_task, "right": right_task}
relations = ["left", "right"] if args_cli.condition == "both" else [args_cli.condition]
auto_register_droid_envs(task=[str(task_paths[relation]) for relation in relations])
args_cli.task = [TASKS[relation] for relation in relations]


def _quat_inverse_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("robot root pose has an unexpected shape")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("robot root quaternion is invalid")
    q = q / norm
    w, xyz = q[0], q[1:]
    inverse_xyz = -xyz
    return (
        2 * np.dot(inverse_xyz, v) * inverse_xyz
        + (w * w - np.dot(inverse_xyz, inverse_xyz)) * v
        + 2 * w * np.cross(inverse_xyz, v)
    )


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _in_requested_region(sample: dict[str, Any], relation: str) -> bool:
    """Apply the frozen DROID 45-degree cone to one retained robot-frame state."""

    object_xyz = np.asarray(sample["object_xyz"], dtype=np.float64)
    reference_xyz = np.asarray(sample["reference_xyz"], dtype=np.float64)
    delta = object_xyz - reference_xyz
    forward, lateral = float(delta[0]), float(delta[1])
    horizontal_distance = math.hypot(lateral, forward)
    requested_margin = lateral if relation == "left" else -lateral
    return (
        horizontal_distance > 1e-8
        and requested_margin / horizontal_distance >= math.cos(math.radians(45.0))
    )


class StateCaptureProxy:
    """Transparent env proxy retaining the initial and every post-action state."""

    def __init__(self, env: Any, task_name: str, capture_dir: Path) -> None:
        self._env = env
        self._relation = "left" if "Left" in task_name else "right"
        self._capture_dir = capture_dir
        self._samples: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._written = False
        self._partial_started = False
        self._partial_path = capture_dir / (
            f"seed{args_cli.environment_seed}_{self._relation}_states.partial.jsonl"
        )
        self._final_path = capture_dir / (
            f"seed{args_cli.environment_seed}_{self._relation}.json"
        )
        if self._partial_path.exists() or self._final_path.exists():
            raise FileExistsError(
                "refusing to overwrite retained V3-A002 state evidence: "
                f"{self._partial_path} or {self._final_path}"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> dict[str, Any]:
        cube_world = self._env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl_world = self._env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        robot = self._env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        object_xyz = _quat_inverse_rotate_wxyz(robot_quat, cube_world - robot_pos)
        reference_xyz = _quat_inverse_rotate_wxyz(robot_quat, bowl_world - robot_pos)
        delta = object_xyz - reference_xyz
        forward, lateral = float(delta[0]), float(delta[1])
        distance = math.hypot(forward, lateral)
        requested_margin = lateral if self._relation == "left" else -lateral
        cone_score = requested_margin / distance if distance > 1e-8 else None
        sample = {
            "action_step": action_step,
            "object_xyz": object_xyz.tolist(),
            "reference_xyz": reference_xyz.tolist(),
            "grippers_open": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
            "score_trace": {
                "signed_lateral_offset_m": lateral,
                "forward_offset_m": forward,
                "requested_signed_margin_m": requested_margin,
                "requested_cone_cosine": cone_score,
                "requested_region": bool(
                    cone_score is not None
                    and cone_score >= math.cos(math.radians(45.0))
                ),
                "opposite_region": bool(
                    cone_score is not None
                    and -cone_score >= math.cos(math.radians(45.0))
                ),
            },
        }
        return sample

    def _start_partial_stream(self) -> None:
        if self._partial_started:
            return
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        with self._partial_path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(self._samples[0], sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        self._partial_started = True

    def _append_sample(self, sample: dict[str, Any]) -> None:
        self._samples.append(sample)
        with self._partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if self._partial_started or len(self._samples) > 1:
            raise RuntimeError("RoboLab reset attempted after V3-A002 action execution")
        result = self._env.reset(*args, **kwargs)
        # RoboLab may reset twice before the first action.  Keep only the state
        # from the last pre-action reset and do not create evidence files until
        # the first env.step, so that this harmless initialization is supported.
        self._samples = [self._sample(0)]
        self._started = time.monotonic()
        self._written = False
        return result

    def step(self, action: Any) -> Any:
        if not self._samples:
            raise RuntimeError("V3-A002 env.step occurred before a captured reset")
        self._start_partial_stream()
        result = self._env.step(action)
        self._append_sample(self._sample(len(self._samples)))
        return result

    def write_capture(self) -> Path | None:
        if self._written or not self._samples:
            return None
        predicate = object_left_of if self._relation == "left" else object_right_of
        success = bool(predicate(
            self._env,
            object="rubiks_cube",
            reference_object="bowl",
            frame_of_reference="robot",
            mirrored=False,
            require_gripper_detached=True,
            env_id=0,
        ))
        action_count = len(self._samples) - 1
        xyz = [np.asarray(sample["object_xyz"], dtype=np.float64) for sample in self._samples]
        initial = xyz[0]
        movement = [float(np.linalg.norm(point - initial)) >= 0.01 for point in xyz]
        pickup = [float(point[2] - initial[2]) >= 0.03 for point in xyz]
        interaction_step = _first_sustained(movement)
        pickup_step = _first_sustained(pickup)
        ever_entered_requested = any(
            _in_requested_region(sample, self._relation) for sample in self._samples
        )
        # The shared compiler independently derives the same cone-entry timeline
        # from retained XYZ.  This value preserves the exact v2 legacy stage:
        # entry means any historical entry, not merely final-state membership.
        if success:
            failure_stage = "success"
        elif interaction_step is None:
            failure_stage = "no_object_interaction"
        elif pickup_step is None:
            failure_stage = "object_moved_no_verified_pickup"
        else:
            failure_stage = (
                "entered_requested_region_not_released"
                if ever_entered_requested
                else "picked_never_entered_requested_region"
            )
        complete = success or action_count == 450
        capture = {
            "schema_version": CAPTURE_SCHEMA,
            "registered_cell_id": (
                f"v3:droid:pi0_fast_old_name_config_v3a002:seed{args_cli.environment_seed}:"
                f"{self._relation}"
            ),
            "attempt_id": f"{args_cli.output_folder_name}:{self._relation}",
            "simulator_pod": SIMULATOR_POD,
            "environment_seed": args_cli.environment_seed,
            "policy_seed": args_cli.sampling_seed_base,
            "requested_relation": self._relation,
            "prompt": PROMPTS[self._relation],
            "requested_success": success,
            "frozen_failure_stage": failure_stage,
            "actions_executed": action_count,
            "action_cap": 450,
            "right_censored": not success and action_count == 450,
            "final_detached_release": bool(
                object_dropped(self._env, object="rubiks_cube", env_id=0)
            ),
            "first_contact_step": None,
            "first_contact_unavailable_reason": (
                "The frozen V3-A002 RoboLab integration exposes grasp and "
                "detached-release conditionals but no verified physical contact stream."
            ),
            "wall_time_s": time.monotonic() - self._started,
            "operational_wall_time_valid": True,
            "samples": self._samples,
            "behavioral_result_valid_candidate": complete,
            "partial_attempt_reason": (
                None if complete else "episode ended before success termination or action cap"
            ),
            "capture_contract": {
                "coordinates": "robot base from inverse Isaac root-pose wxyz rotation",
                "contact": "instrumentation_unavailable; grasp is not substituted",
                "partial_state_stream": str(self._partial_path.resolve()),
                "failure_early_stopping": False,
            },
        }
        if self._final_path.exists():
            raise FileExistsError(
                f"refusing to overwrite V3-A002 capture: {self._final_path}"
            )
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self._final_path.write_text(
            json.dumps(capture, indent=2, sort_keys=True) + "\n"
        )
        self._written = True
        return self._final_path

    def close(self) -> Any:
        self.write_capture()
        return self._env.close()


_create_env = runtime.create_env
_proxies: list[StateCaptureProxy] = []


def _seeded_captured_create_env(*args: Any, **kwargs: Any) -> Any:
    kwargs["seed"] = args_cli.environment_seed
    env, env_cfg = _create_env(*args, **kwargs)
    task_name = str(args[0] if args else kwargs.get("task"))
    proxy = StateCaptureProxy(env, task_name, args_cli.state_capture_dir)
    _proxies.append(proxy)
    return proxy, env_cfg


runtime.create_env = _seeded_captured_create_env
_clients: list[V3Pi0FastOldNameConfigClient] = []


def make_client(_: argparse.Namespace) -> V3Pi0FastOldNameConfigClient:
    client = V3Pi0FastOldNameConfigClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        sampling_seed_base=args_cli.sampling_seed_base,
        action_trace_dir=args_cli.action_trace_dir,
    )
    _clients.append(client)
    return client


def main() -> None:
    try:
        run_evaluation(
            args_cli,
            policy="pi0_fast_old_name_config_v3a002",
            client_factory=make_client,
        )
    finally:
        for client in _clients:
            client.write_trace()
        for proxy in _proxies:
            proxy.write_capture()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[π0-FAST old-name-config bridge v3] technical failure: {error}")
        traceback.print_exc()
        raise
