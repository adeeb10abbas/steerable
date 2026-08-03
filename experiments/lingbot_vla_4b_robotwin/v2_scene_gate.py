#!/usr/bin/env python3
"""Create and close the frozen LingBot RoboTwin scene without taking actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--requested-relation", choices=("left", "right"), required=True)
    args = parser.parse_args()

    robotwin_root = args.robotwin_root.resolve()
    os.chdir(robotwin_root)
    sys.path[:0] = [
        str(robotwin_root),
        str(robotwin_root / "script"),
        str(robotwin_root / "description" / "utils"),
    ]
    from experiment.lingbot_vla_4b_language_gate import (  # noqa: PLC0415
        frozen_prompt,
        load_robotwin_setup_kwargs,
        relation_metrics,
        seed_everything,
        task_class,
    )

    seed_everything(args.sampling_seed)
    env = task_class(args.task)()
    try:
        setup = load_robotwin_setup_kwargs(args.task)
        setup.update(eval_mode=True, save_data=False, collect_data=False)
        env.setup_demo(seed=args.environment_seed, is_test=True, **setup)
        prompt_args = argparse.Namespace(
            study_root=args.study_root,
            robotwin_root=robotwin_root,
            requested_relation=args.requested_relation,
        )
        prompt = frozen_prompt(prompt_args, env)
        observation = env.get_obs()
        cameras = observation["observation"]
        state = np.asarray(observation["joint_action"]["vector"])
        result = {
            "gate": "integrated_robotwin_scene",
            "task": args.task,
            "environment_seed": args.environment_seed,
            "sampling_seed": args.sampling_seed,
            "prompt": prompt,
            "cam_high_shape": list(cameras["head_camera"]["rgb"].shape),
            "cam_left_wrist_shape": list(cameras["left_camera"]["rgb"].shape),
            "cam_right_wrist_shape": list(cameras["right_camera"]["rgb"].shape),
            "state_shape": list(state.shape),
            "state_finite": bool(np.isfinite(state).all()),
            "initial_relation_metrics": relation_metrics(env, args.requested_relation),
        }
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
