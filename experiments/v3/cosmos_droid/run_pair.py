#!/usr/bin/env python3
"""Run one exact authorized v3 Cosmos3 DROID LEFT/RIGHT seed pair."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401 -- RoboLab requires this import before Isaac Lab
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--model-id", choices=[
    "cosmos3_edge_policy_droid", "cosmos3_nano_policy_droid"
], required=True)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--sampling-seed-base", type=int, required=True)
parser.add_argument("--runtime-manifest", type=Path, required=True)
parser.add_argument("--release-manifest", type=Path, required=True)
parser.add_argument("--action-trace-dir", type=Path, required=True)
parser.add_argument("--future-trace-dir", type=Path, required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, required=True)
parser.add_argument("--open-loop-horizon", type=int, default=32)
parser.add_argument("--instruction-controller", choices=["static"], default="static")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

study_root = args_cli.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))

from experiments.v3.cosmos_droid.client import V3CosmosDroidClient  # noqa: E402
from experiments.v3.cosmos_droid.contract import (  # noqa: E402
    MODEL_CONTRACTS,
    load_authorized_pair,
    verify_release_gate,
    verify_runtime_identity,
)


pair = load_authorized_pair(study_root, args_cli.model_id, args_cli.environment_seed)
if args_cli.sampling_seed_base != pair.seed:
    parser.error("Environment and sampling seed must equal the registered pair seed")
if args_cli.open_loop_horizon != 32:
    parser.error("The frozen Cosmos action horizon is 32")
if args_cli.video_mode != "viewport":
    parser.error("Every v3 behavioral cell requires viewport video")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("Each registered cell requires one environment and one run")
if args_cli.enable_subtask:
    parser.error("Pass --disable-subtask; progress-conditioned coaching is forbidden")
if args_cli.remote_port != MODEL_CONTRACTS[args_cli.model_id]["server_port"]:
    parser.error("Remote port differs from the model-specific v2 serving contract")

runtime_identity = verify_runtime_identity(study_root, args_cli.model_id, args_cli.runtime_manifest)
verify_release_gate(
    args_cli.release_manifest,
    pair=pair,
    runtime_identity_sha256=runtime_identity["runtime_identity_sha256"],
)
for raw_dir in (args_cli.action_trace_dir, args_cli.future_trace_dir):
    raw_dir.mkdir(parents=True, exist_ok=False)
    probe = raw_dir / ".write_preflight"
    probe.write_bytes(b"v3-cosmos-behavioral-write-preflight\n")
    probe.unlink()

left_task = study_root / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py"
right_task = study_root / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

auto_register_droid_envs(task=[str(left_task), str(right_task)], cameras=WRIST_LEFT_RIGHT_HEAD)
args_cli.task = ["RubiksCubeLeftOfBowlMatchedTask", "RubiksCubeRightOfBowlMatchedTask"]

_create_env = runtime.create_env


def _seeded_create_env(*args, **kwargs):
    kwargs["seed"] = pair.seed
    return _create_env(*args, **kwargs)


runtime.create_env = _seeded_create_env


def make_client(_: argparse.Namespace) -> V3CosmosDroidClient:
    return V3CosmosDroidClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        sampling_seed_base=pair.seed,
        action_trace_dir=args_cli.action_trace_dir,
        future_trace_dir=args_cli.future_trace_dir,
        pair=pair,
        runtime_identity=runtime_identity,
    )


def main() -> None:
    policy_id = (
        "cosmos3_v2" if args_cli.model_id == "cosmos3_edge_policy_droid"
        else "cosmos3_nano_v2"
    )
    run_evaluation(args_cli, policy=policy_id, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Cosmos3 v3] infrastructure failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
