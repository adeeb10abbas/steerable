#!/usr/bin/env python3
"""Run one frozen V2-A010 pi0.5 direct-command media cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import cv2  # noqa: F401
from isaaclab.app import AppLauncher


ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--registry", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--sampling-seed-base", type=int, required=True)
parser.add_argument("--requested-relation", choices=["left", "right"], required=True)
parser.add_argument("--action-trace-dir", type=Path, required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, default=8001)
parser.add_argument("--open-loop-horizon", type=int, default=15)
parser.add_argument("--instruction-controller", choices=["static"], default="static")
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.environment_seed not in {8300, 8301, 8302}:
    parser.error("V2-A010 seeds are exactly 8300, 8301, 8302")
if args_cli.sampling_seed_base != args_cli.environment_seed:
    parser.error("Environment and sampling seed bases must match")
if args_cli.open_loop_horizon != 15 or args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("V2-A010 requires horizon 15, one environment, and one run")
if args_cli.video_mode != "viewport" or args_cli.enable_subtask:
    parser.error("V2-A010 requires viewport video and --disable-subtask")
if args_cli.instruction_type != "default":
    parser.error("V2-A010 uses only the frozen direct-command default prompt")
head = subprocess.check_output(
    ["git", "-C", str(args_cli.robolab_root), "rev-parse", "HEAD"], text=True
).strip()
if head != ROBOLAB_COMMIT:
    parser.error(f"Unexpected RoboLab commit: {head}")
registry = json.loads(args_cli.registry.read_text())
matches = [row for row in registry["cells"] if row["environment_seed"] == args_cli.environment_seed and row["requested_relation"] == args_cli.requested_relation]
if len(matches) != 1:
    parser.error(f"Expected one registry cell, found {len(matches)}")
cell = matches[0]
if args_cli.output_folder_name != cell["output_folder_name"]:
    parser.error(f"Use frozen --output-folder-name {cell['output_folder_name']}")
task_path = args_cli.study_root / "experiments/groot_droid/robolab_v2_tasks" / f"rubiks_cube_{args_cli.requested_relation}_of_bowl_matched.py"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

sys.path.insert(0, str(args_cli.study_root / "experiments/pi05_current_stack"))
from v2a010_robolab_client import V2A010Pi05Client  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
auto_register_droid_envs(task=[str(task_path)])
args_cli.task = [cell["anchor_task"]]
_create_env = runtime.create_env


def _seeded_create_env(*args, **kwargs):
    kwargs["seed"] = args_cli.environment_seed
    return _create_env(*args, **kwargs)


runtime.create_env = _seeded_create_env


def make_client(_: argparse.Namespace) -> V2A010Pi05Client:
    return V2A010Pi05Client(
        remote_host=args_cli.remote_host, remote_port=args_cli.remote_port,
        open_loop_horizon=15, sampling_seed_base=args_cli.sampling_seed_base,
        requested_relation=args_cli.requested_relation,
        expected_prompt=cell["rendered_prompt"], action_trace_dir=args_cli.action_trace_dir,
    )


def main() -> None:
    run_evaluation(args_cli, policy="pi05_v2a010_current", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pi0.5 V2-A010] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
