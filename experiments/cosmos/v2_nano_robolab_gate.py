#!/usr/bin/env python3
"""Run exactly one frozen Cosmos3 Nano LEFT/RIGHT DROID seed pair."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401 -- RoboLab requires this before Isaac Lab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--sampling-seed-base", type=int, required=True)
parser.add_argument("--action-trace-dir", type=Path, required=True)
parser.add_argument("--future-trace-dir", type=Path, required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, default=18011)
parser.add_argument("--open-loop-horizon", type=int, default=32)
parser.add_argument("--instruction-controller", choices=["static"], default="static")

from robolab.eval.runner import add_common_eval_args, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if args_cli.open_loop_horizon != 32:
    parser.error("The frozen Nano DROID gate requires --open-loop-horizon 32")
if args_cli.video_mode != "viewport":
    parser.error("The frozen Nano DROID gate requires --video-mode viewport")
if args_cli.environment_seed not in {8300, 8301, 8302}:
    parser.error("The authorized environment seeds are exactly 8300, 8301, 8302")
if args_cli.sampling_seed_base != args_cli.environment_seed:
    parser.error("Environment and sampling seed integers must match within a DROID pair")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("Each frozen task cell requires one environment and one run")
if args_cli.enable_subtask:
    parser.error("Pass --disable-subtask; progress-conditioned coaching is forbidden")

left_task = args_cli.study_root / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py"
right_task = args_cli.study_root / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py"
for task_path in (left_task, right_task):
    if not task_path.is_file():
        parser.error(f"Missing frozen task file: {task_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants
from robolab.core.environments import runtime
from robolab.registrations.droid.auto_env_registrations_jointpos import (
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import (
    WRIST_LEFT_RIGHT_HEAD,
)

sys.path.insert(0, str(args_cli.study_root / "experiments/cosmos"))
from v2_nano_robolab_client import V2NanoCosmos3Client

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

auto_register_droid_envs(task=[str(left_task), str(right_task)], cameras=WRIST_LEFT_RIGHT_HEAD)
args_cli.task = ["RubiksCubeLeftOfBowlMatchedTask", "RubiksCubeRightOfBowlMatchedTask"]

_create_env = runtime.create_env


def _seeded_create_env(*args, **kwargs):
    kwargs["seed"] = args_cli.environment_seed
    return _create_env(*args, **kwargs)


runtime.create_env = _seeded_create_env


def make_client(_: argparse.Namespace) -> V2NanoCosmos3Client:
    return V2NanoCosmos3Client(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        sampling_seed_base=args_cli.sampling_seed_base,
        action_trace_dir=args_cli.action_trace_dir,
        future_trace_dir=args_cli.future_trace_dir,
    )


def main() -> None:
    run_evaluation(args_cli, policy="cosmos3_nano_v2", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Cosmos3 Nano V2-A011] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
