#!/usr/bin/env python3
"""Run one authorized DreamZero s=2 v3 matched DROID pair on the RTX lane."""

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
BOOTSTRAP.add_argument("--sampling-seed", type=int, required=True)
BOOTSTRAP.add_argument("--runtime-identity", type=Path, required=True)
BOOTSTRAP.add_argument("--release-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--state-capture-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--action-trace-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--remote-host", required=True)
BOOTSTRAP.add_argument("--remote-port", type=int, required=True)
BOOTSTRAP.add_argument(
    "--simulator-lane", choices=["raytrace-rtxpro6000-ali"], required=True
)
BOOTSTRAP.add_argument("--open-loop-horizon", type=int, default=8)
BOOTSTRAP.add_argument("--instruction-controller", choices=["static"], default="static")
BOOTSTRAP.add_argument("--condition", choices=["both"], default="both")
bootstrap, _ = BOOTSTRAP.parse_known_args()

sys.path.insert(0, str(bootstrap.study_root / "experiments/v3/dreamzero_droid"))
from adapter import (  # noqa: E402
    ACTION_CFG_STYLE_SCALE,
    CAPTURE_SCHEMA,
    GATE_SCHEMA,
    IDENTITY_BINDING,
    OFFICIAL_NOISE_SEED,
    PROMPTS,
    TASKS,
    preflight,
)

if bootstrap.environment_seed != bootstrap.sampling_seed:
    BOOTSTRAP.error("DreamZero environment and registered sampling seed labels must match")
if bootstrap.open_loop_horizon != 8:
    BOOTSTRAP.error("the exact DreamZero s=2 open-loop horizon is 8")
if bootstrap.remote_port == 5000:
    BOOTSTRAP.error("protected pre-existing DreamZero port 5000 is prohibited")
preflight(
    bootstrap.study_root.resolve(),
    bootstrap.environment_seed,
    bootstrap.runtime_identity,
    bootstrap.release_gate,
    check_live_repositories=True,
)
release_gate = json.loads(bootstrap.release_gate.read_text())
if release_gate.get("schema_version") != GATE_SCHEMA:
    BOOTSTRAP.error("release gate schema changed after preflight")
future_root = Path(str(release_gate["future_root"])).resolve()
server_contract_path = Path(str(release_gate["server_contract_path"])).resolve()

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
    parser.error("DreamZero v3 requires viewport video")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("each registered DreamZero cell requires one environment and one run")
if args_cli.device != "cuda:0":
    parser.error("the ali-owned RTX simulator lane must be exposed as cuda:0")
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
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT  # noqa: E402

from client import V3DreamZeroS2Client  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

left_task = args_cli.study_root / (
    "experiments/dreamzero_droid/robolab_v2_tasks/"
    "rubiks_cube_left_of_bowl_matched.py"
)
right_task = args_cli.study_root / (
    "experiments/dreamzero_droid/robolab_v2_tasks/"
    "rubiks_cube_right_of_bowl_matched.py"
)
auto_register_droid_envs(
    task=[str(left_task), str(right_task)], cameras=WRIST_LEFT_RIGHT
)
args_cli.task = [TASKS["left"], TASKS["right"]]


def _quat_inverse_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("DreamZero robot root pose has an unexpected shape")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("DreamZero robot root quaternion is invalid")
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
    """Capture initial plus every post-action object/reference state."""

    def __init__(self, env: Any, task_name: str, capture_dir: Path) -> None:
        self._env = env
        self._relation = "left" if "Left" in task_name else "right"
        self._capture_dir = capture_dir
        self._samples: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._written = False
        self._partial_path = capture_dir / (
            f"seed{args_cli.environment_seed}_{self._relation}_states.partial.jsonl"
        )

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
            "object_xyz": _quat_inverse_rotate_wxyz(
                robot_quat, cube_world - robot_pos
            ).tolist(),
            "reference_xyz": _quat_inverse_rotate_wxyz(
                robot_quat, bowl_world - robot_pos
            ).tolist(),
            "grippers_open": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
        }
        self._samples.append(sample)
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        with self._partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        final_path = self._capture_dir / (
            f"seed{args_cli.environment_seed}_{self._relation}.json"
        )
        if self._partial_path.exists() or final_path.exists():
            raise FileExistsError(
                "refusing to overwrite retained DreamZero v3 state evidence: "
                f"{self._partial_path} or {final_path}"
            )
        result = self._env.reset(*args, **kwargs)
        self._samples = []
        self._started = time.monotonic()
        self._written = False
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self._partial_path.touch(exist_ok=False)
        self._sample(0)
        return result

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        self._sample(len(self._samples))
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
        if success:
            failure_stage = "success"
        elif interaction_step is None:
            failure_stage = "no_object_interaction"
        elif pickup_step is None:
            failure_stage = "object_moved_no_verified_pickup"
        else:
            failure_stage = (
                "entered_requested_region_not_released"
                if ever_entered_requested else "picked_never_entered_requested_region"
            )
        complete = success or action_count == 450
        capture = {
            "schema_version": CAPTURE_SCHEMA,
            "registered_cell_id": (
                f"v3:droid:dreamzero_droid_action_cfg:seed{args_cli.environment_seed}:"
                f"{self._relation}"
            ),
            "attempt_id": (
                f"v3-dreamzero-s2-seed{args_cli.environment_seed}-"
                f"{self._relation}-attempt0"
            ),
            "identity_binding": IDENTITY_BINDING,
            "environment_seed": args_cli.environment_seed,
            "policy_seed": args_cli.sampling_seed,
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
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
                "The frozen DreamZero RoboLab integration exposes grasp and "
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
                "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
            },
        }
        output = self._capture_dir / f"seed{args_cli.environment_seed}_{self._relation}.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite DreamZero v3 capture: {output}")
        output.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
        self._written = True
        return output

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
_clients: list[V3DreamZeroS2Client] = []


def make_client(_: argparse.Namespace) -> V3DreamZeroS2Client:
    client = V3DreamZeroS2Client(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        environment_seed=args_cli.environment_seed,
        sampling_seed_label=args_cli.sampling_seed,
        action_trace_dir=args_cli.action_trace_dir,
        server_contract_path=server_contract_path,
        release_gate_path=args_cli.release_gate,
        future_root=future_root,
    )
    _clients.append(client)
    return client


def main() -> None:
    try:
        run_evaluation(args_cli, policy="dreamzero_v2", client_factory=make_client)
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
        print(f"[DreamZero s=2 v3] technical failure: {error}")
        traceback.print_exc()
        raise
