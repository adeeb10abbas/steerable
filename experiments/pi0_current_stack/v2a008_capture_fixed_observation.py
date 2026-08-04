#!/usr/bin/env python3
"""Capture the neutral current-stack pi0-FAST fixed-observation fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

import cv2  # noqa: F401 -- required before Isaac Lab
import numpy as np
from isaaclab.app import AppLauncher


EXPECTED_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"

parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--registry", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, default=8300)
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.environment_seed != 8300 or args_cli.num_envs != 1:
    parser.error("V2-A008 fixed-observation capture requires seed 8300 and one environment")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args_cli.robolab_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != EXPECTED_ROBOLAB_COMMIT:
        raise ValueError(f"Unexpected current-stack RoboLab commit: {actual_commit}")
    registry = json.loads(args_cli.registry.read_text())
    cell = next(
        row for row in registry["cells"]
        if row["environment_seed"] == 8300
        and row["prompt_family"] == "short_command"
        and row["requested_relation"] == "left"
    )
    task_path = (
        args_cli.study_root
        / "experiments/pi0_current_stack/robolab_v2_tasks/"
        "rubiks_cube_left_of_bowl_matched.py"
    )
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(args_cli.output_dir))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    auto_register_droid_envs(task=[str(task_path)])
    env, env_cfg = create_env(
        cell["anchor_task"],
        device=args_cli.device,
        seed=8300,
        num_envs=1,
        instruction_type="short_command",
        policy="pi0_fast_v2a008_fixture",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    try:
        env.reset()
        obs, _ = env.reset()
        if env_cfg.instruction != cell["rendered_prompt"]:
            raise ValueError("Fixture prompt differs from the frozen registry")
        left_at_reset = bool(
            object_left_of(
                env, object="rubiks_cube", reference_object="bowl",
                frame_of_reference="robot", mirrored=False,
                require_gripper_detached=True, env_id=0,
            )
        )
        right_at_reset = bool(
            object_right_of(
                env, object="rubiks_cube", reference_object="bowl",
                frame_of_reference="robot", mirrored=False,
                require_gripper_detached=True, env_id=0,
            )
        )
        if left_at_reset or right_at_reset:
            raise ValueError("V2-A008 fixture reset must satisfy neither LEFT nor RIGHT")
        helper = object.__new__(Pi0DroidJointposClient)
        extracted = helper._extract_observation(obs, env_id=0)
        request = helper._pack_request(extracted, cell["rendered_prompt"])
        prompt = request.pop("prompt")
        if prompt != cell["rendered_prompt"]:
            raise ValueError("Packed prompt differs from the frozen registry")
        arrays = {key: np.asarray(value) for key, value in request.items()}
        required = {
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
            "observation/joint_position",
            "observation/gripper_position",
        }
        if set(arrays) != required:
            raise ValueError(f"Unexpected pi0 observation keys: {sorted(arrays)}")
        fixture_path = args_cli.output_dir / "seed8300_fixed_observation.npz"
        np.savez(fixture_path, **arrays)
        cube = env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl = env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        manifest = {
            "schema_version": "vla-wam-v2a008-pi0-current-fixed-observation-v1",
            "registry_path": str(args_cli.registry),
            "registry_sha256": sha256(args_cli.registry),
            "robolab_commit": actual_commit,
            "environment_seed": 8300,
            "task": cell["anchor_task"],
            "prompt": cell["rendered_prompt"],
            "reset_count": 2,
            "neutral_reset_contract": {
                "cube_world_xyz": cube.tolist(),
                "bowl_world_xyz": bowl.tolist(),
                "left_predicate_at_reset": left_at_reset,
                "right_predicate_at_reset": right_at_reset,
            },
            "fixture_path": str(fixture_path),
            "fixture_sha256": sha256(fixture_path),
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(arrays.items())
            },
        }
        manifest_path = args_cli.output_dir / "seed8300_fixed_observation.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pi0-FAST V2-A008 fixture] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
