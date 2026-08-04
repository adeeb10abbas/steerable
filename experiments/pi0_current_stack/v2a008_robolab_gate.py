#!/usr/bin/env python3
"""Run one V2-A008 current-stack pi0-FAST wording cell in RoboLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import cv2  # noqa: F401 -- required before Isaac Lab
from isaaclab.app import AppLauncher


EXPECTED_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FAMILIES = {
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
}

parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--registry", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--sampling-seed-base", type=int, required=True)
parser.add_argument("--prompt-family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--requested-relation", choices=["left", "right"], required=True)
parser.add_argument("--action-trace-dir", type=Path, required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, default=8000)
parser.add_argument("--open-loop-horizon", type=int, default=10)
parser.add_argument("--instruction-controller", choices=["static"], default="static")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if not 8300 <= args_cli.environment_seed <= 8309:
    parser.error("V2-A008 environment seeds are exactly 8300 through 8309")
if args_cli.sampling_seed_base != args_cli.environment_seed:
    parser.error("Environment and sampling seed bases must match")
if args_cli.open_loop_horizon != 10:
    parser.error("V2-A008 requires --open-loop-horizon 10")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("V2-A008 runs exactly one environment and one run per invocation")
if args_cli.video_mode != "viewport":
    parser.error("V2-A008 requires --video-mode viewport")
if args_cli.enable_subtask:
    parser.error("Pass --disable-subtask; progress-conditioned coaching is forbidden")
if args_cli.instruction_type != "default":
    parser.error("Do not pass --instruction-type; --prompt-family owns the frozen prompt")

registry = json.loads(args_cli.registry.read_text())
if (
    registry.get("schema_version")
    != "vla-wam-v2a008-pi0-current-stack-registry-v1"
    or registry.get("amendment_id") != "V2-A008"
):
    parser.error(f"Not the frozen V2-A008 registry: {args_cli.registry}")
matches = [
    row
    for row in registry["cells"]
    if row["environment_seed"] == args_cli.environment_seed
    and row["sampling_seed_base"] == args_cli.sampling_seed_base
    and row["prompt_family"] == args_cli.prompt_family
    and row["requested_relation"] == args_cli.requested_relation
]
if len(matches) != 1:
    parser.error(f"Expected one matching frozen registry cell, found {len(matches)}")
registry_cell = matches[0]
if args_cli.output_folder_name != registry_cell["output_folder_name"]:
    parser.error(
        "--output-folder-name must equal the frozen registry value "
        f"{registry_cell['output_folder_name']!r}"
    )

actual_commit = subprocess.check_output(
    ["git", "-C", str(args_cli.robolab_root), "rev-parse", "HEAD"], text=True
).strip()
if actual_commit != EXPECTED_ROBOLAB_COMMIT:
    parser.error(
        f"Current-stack RoboLab commit changed: {actual_commit}; "
        f"expected {EXPECTED_ROBOLAB_COMMIT}"
    )

task_path = (
    args_cli.study_root
    / "experiments/pi0_current_stack/robolab_v2_tasks"
    / f"rubiks_cube_{args_cli.requested_relation}_of_bowl_matched.py"
)
if not task_path.is_file():
    parser.error(f"Missing frozen V2-A008 task overlay: {task_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)

sys.path.insert(0, str(args_cli.study_root / "experiments/pi0_current_stack"))
from v2a008_robolab_client import V2A008Pi0FastClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

auto_register_droid_envs(task=[str(task_path)])
task_name = (
    "RubiksCubeLeftOfBowlMatchedTask"
    if args_cli.requested_relation == "left"
    else "RubiksCubeRightOfBowlMatchedTask"
)
args_cli.task = [task_name]
args_cli.instruction_type = args_cli.prompt_family

_create_env = runtime.create_env


def _seeded_create_env(*args, **kwargs):
    kwargs["seed"] = args_cli.environment_seed
    return _create_env(*args, **kwargs)


runtime.create_env = _seeded_create_env


def make_client(_: argparse.Namespace) -> V2A008Pi0FastClient:
    return V2A008Pi0FastClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        open_loop_horizon=args_cli.open_loop_horizon,
        sampling_seed_base=args_cli.sampling_seed_base,
        prompt_family=args_cli.prompt_family,
        requested_relation=args_cli.requested_relation,
        expected_prompt=registry_cell["rendered_prompt"],
        action_trace_dir=args_cli.action_trace_dir,
    )


def main() -> None:
    run_evaluation(args_cli, policy="pi0_fast_v2a008_current", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pi0-FAST V2-A008] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
