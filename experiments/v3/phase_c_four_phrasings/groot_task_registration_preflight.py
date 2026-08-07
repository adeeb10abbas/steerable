#!/usr/bin/env python3
"""Register and reset all eight GR00T Phase-C tasks without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import cv2  # noqa: F401 -- RoboLab requires this import before Isaac Lab
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--bridge-preflight", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1:
    parser.error("task-registration preflight requires one environment")
if args_cli.video_mode != "viewport":
    parser.error("task-registration preflight must initialize the viewport renderer")

study_root = args_cli.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))

from experiments.v3.phase_c_four_phrasings.contract import canonical_json_bytes, sha256_file  # noqa: E402

bridge = json.loads(args_cli.bridge_preflight.read_text())
if (
    bridge.get("schema_version") != "vla-wam-shared-v3c-groot-seed-block-preflight-v1"
    or bridge.get("passed") is not True
    or bridge.get("model_request_count") != 0
    or bridge.get("behavioral_episode_count") != 0
    or len(bridge.get("cells", [])) != 8
):
    parser.error("bridge preflight is incomplete or not model-blind")
if args_cli.output.exists():
    parser.error(f"refusing to overwrite retained task-registration evidence: {args_cli.output}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


def main() -> None:
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    task_files = [cell["task_file"] for cell in bridge["cells"]]
    auto_register_droid_envs(task=task_files, cameras=WRIST_LEFT_RIGHT_HEAD)
    rows = []
    for cell in bridge["cells"]:
        env, env_cfg = create_env(
            cell["task_name"],
            device=args_cli.device,
            seed=bridge["seed"],
            num_envs=1,
            instruction_type="default",
            policy="v3c001_groot_zero_action_registration",
            renderer=args_cli.renderer,
            rendering_mode=args_cli.rendering_type,
        )
        try:
            env.reset()
            env.reset()
            if env_cfg.instruction != cell["prompt"]:
                raise ValueError(f"registered task prompt changed: {cell['task_name']}")
            cube = env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
            bowl = env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
            left = bool(object_left_of(
                env, object="rubiks_cube", reference_object="bowl",
                frame_of_reference="robot", mirrored=False,
                require_gripper_detached=True, env_id=0,
            ))
            right = bool(object_right_of(
                env, object="rubiks_cube", reference_object="bowl",
                frame_of_reference="robot", mirrored=False,
                require_gripper_detached=True, env_id=0,
            ))
            if left or right:
                raise ValueError(f"registered reset is not neutral: {cell['task_name']}")
            rows.append({
                "registered_cell_id": cell["registered_cell_id"],
                "within_seed_execution_order": cell["within_seed_execution_order"],
                "task_name": cell["task_name"],
                "prompt": env_cfg.instruction,
                "cube_world_xyz": cube.tolist(),
                "bowl_world_xyz": bowl.tolist(),
                "left_predicate_at_reset": left,
                "right_predicate_at_reset": right,
                "model_requests": 0,
                "actions_executed": 0,
            })
        finally:
            env.close()
    cube_positions = np.asarray([row["cube_world_xyz"] for row in rows])
    bowl_positions = np.asarray([row["bowl_world_xyz"] for row in rows])
    cube_spread = float(np.max(np.ptp(cube_positions, axis=0)))
    bowl_spread = float(np.max(np.ptp(bowl_positions, axis=0)))
    tolerance = 0.003
    if cube_spread > tolerance or bowl_spread > tolerance:
        raise ValueError(
            f"task wrappers do not share a matched reset: cube={cube_spread}, bowl={bowl_spread}"
        )
    report = {
        "schema_version": "vla-wam-shared-v3c-groot-live-task-registration-v1",
        "experiment_id": "V3-C001",
        "model_id": "groot_n17_droid_vla",
        "seed": bridge["seed"],
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "executed_action_count": 0,
        "renderer_initialized": True,
        "matched_reset_tolerance_m": tolerance,
        "max_cube_position_spread_m": cube_spread,
        "max_bowl_position_spread_m": bowl_spread,
        "bridge_preflight": {
            "path": str(args_cli.bridge_preflight),
            "sha256": sha256_file(args_cli.bridge_preflight),
        },
        "cells": rows,
        "next_status": "live_prompt_aware_action_state_video_writer_still_required",
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[GR00T V3-C001 task registration] infrastructure failure: {error}")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
