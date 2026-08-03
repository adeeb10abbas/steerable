#!/usr/bin/env python3
"""Run exactly one frozen GR00T LEFT/RIGHT DROID seed pair in RoboLab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import cv2  # noqa: F401 -- RoboLab requires this before Isaac Lab
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--sampling-seed-base", type=int, required=True)
parser.add_argument("--action-trace-dir", type=Path, required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, default=5555)
parser.add_argument("--open-loop-horizon", type=int, default=8)
parser.add_argument("--instruction-controller", choices=["static"], default="static")
parser.add_argument(
    "--condition",
    choices=["left", "right", "both"],
    default="both",
    help="Run one frozen condition or the complete pair; useful after an infra partial.",
)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if args_cli.open_loop_horizon != 8:
    parser.error("The frozen GR00T DROID gate requires --open-loop-horizon 8")
if args_cli.video_mode != "viewport":
    parser.error("The frozen GR00T DROID gate requires --video-mode viewport")
if args_cli.environment_seed not in {8300, 8301, 8302}:
    parser.error("The authorized environment seeds are exactly 8300, 8301, 8302")
if args_cli.sampling_seed_base != args_cli.environment_seed:
    parser.error("Environment and sampling seed integers must match within a DROID pair")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("Each frozen task cell requires one environment and one run")
if args_cli.enable_subtask:
    parser.error("Pass --disable-subtask; progress-conditioned coaching is forbidden")

left_task = (
    args_cli.study_root
    / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py"
)
right_task = (
    args_cli.study_root
    / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py"
)
for task_path in (left_task, right_task):
    if not task_path.is_file():
        parser.error(f"Missing frozen task file: {task_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)

sys.path.insert(0, str(args_cli.study_root / "experiments/groot_droid"))
from v2_robolab_client import V2GR00TDroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

task_paths = {
    "left": [str(left_task)],
    "right": [str(right_task)],
    "both": [str(left_task), str(right_task)],
}
task_names = {
    "left": ["RubiksCubeLeftOfBowlMatchedTask"],
    "right": ["RubiksCubeRightOfBowlMatchedTask"],
    "both": [
        "RubiksCubeLeftOfBowlMatchedTask",
        "RubiksCubeRightOfBowlMatchedTask",
    ],
}
auto_register_droid_envs(task=task_paths[args_cli.condition])
args_cli.task = task_names[args_cli.condition]

# The upstream generic runner does not expose a seed flag.  Patch only this
# process's runtime factory so env_cfg.json records the frozen environment seed.
_create_env = runtime.create_env


def _seeded_create_env(*args, **kwargs):
    kwargs["seed"] = args_cli.environment_seed
    return _create_env(*args, **kwargs)


runtime.create_env = _seeded_create_env


def make_client(_: argparse.Namespace) -> V2GR00TDroidJointposClient:
    return V2GR00TDroidJointposClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        open_loop_horizon=args_cli.open_loop_horizon,
        sampling_seed_base=args_cli.sampling_seed_base,
        action_trace_dir=args_cli.action_trace_dir,
    )


def main() -> None:
    run_evaluation(args_cli, policy="groot_v2", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[GR00T v2] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
