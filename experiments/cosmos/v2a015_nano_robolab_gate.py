#!/usr/bin/env python3
"""Run one frozen V2-A015 Cosmos3 Nano g=1 LEFT/RIGHT DROID pair."""

from __future__ import annotations

import argparse
import json
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
parser.add_argument("--fixed-observation-gate", type=Path, required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, default=18021)
parser.add_argument("--open-loop-horizon", type=int, default=32)
parser.add_argument("--instruction-controller", choices=["static"], default="static")

from robolab.eval.runner import add_common_eval_args, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if args_cli.open_loop_horizon != 32:
    parser.error("The frozen V2-A015 Nano DROID gate requires --open-loop-horizon 32")
if args_cli.video_mode != "viewport":
    parser.error("The frozen V2-A015 Nano DROID gate requires --video-mode viewport")
if args_cli.environment_seed not in {8300, 8301, 8302}:
    parser.error("The authorized environment seeds are exactly 8300, 8301, 8302")
if args_cli.sampling_seed_base != args_cli.environment_seed:
    parser.error("Environment and sampling seed integers must match within a DROID pair")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("Each frozen task cell requires one environment and one run")
if args_cli.enable_subtask:
    parser.error("Pass --disable-subtask; progress-conditioned coaching is forbidden")

fixed_gate = json.loads(args_cli.fixed_observation_gate.read_text())
expected_fixed_gate = {
    "schema_version": (
        "vla-wam-shared-v2-cosmos3-nano-policy-droid-v2a015-g1-"
        "fixed-observation-v1"
    ),
    "status": "passed",
    "amendment_id": "V2-A015",
    "arm_id": "cosmos3_nano_no_cfg_g1",
    "guidance": 1.0,
    "baseline_guidance": 3.0,
}
for key, value in expected_fixed_gate.items():
    if fixed_gate.get(key) != value:
        parser.error(
            f"Fixed-observation release gate mismatch for {key}: "
            f"expected={value!r}, observed={fixed_gate.get(key)!r}"
        )
fixed_metrics = fixed_gate.get("metrics", {})
for key in (
    "left_repeat_action_bit_identical",
    "left_repeat_future_bit_identical",
    "left_right_action_distinct",
    "left_right_future_distinct",
):
    if fixed_metrics.get(key) is not True:
        parser.error(f"Fixed-observation release gate did not pass {key}")
if fixed_gate.get("baseline_result") != {
    "artifact": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_direct_gate.json"
    ),
    "sha256": "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93",
    "reported_result": "LEFT 3/3; RIGHT 3/3; 3/3 aligned endpoint pairs",
}:
    parser.error("Fixed-observation release gate does not bind the frozen g=3 baseline")
if [record.get("condition") for record in fixed_gate.get("records", [])] != [
    "left",
    "left_exact_repeat",
    "right",
]:
    parser.error("Fixed-observation release gate does not contain the exact three requests")

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
from v2a015_nano_robolab_client import V2A015NanoCosmos3Client

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


def make_client(_: argparse.Namespace) -> V2A015NanoCosmos3Client:
    return V2A015NanoCosmos3Client(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        sampling_seed_base=args_cli.sampling_seed_base,
        action_trace_dir=args_cli.action_trace_dir,
        future_trace_dir=args_cli.future_trace_dir,
    )


def main() -> None:
    # Keep the official V2-A011 runner integration identifier unchanged.  The
    # derived client binds every raw request and final trace to V2-A015/g=1.
    run_evaluation(args_cli, policy="cosmos3_nano_v2", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Cosmos3 Nano V2-A015 g=1] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
