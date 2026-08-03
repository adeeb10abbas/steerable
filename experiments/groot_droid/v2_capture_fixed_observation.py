#!/usr/bin/env python3
"""Capture the exact neutral-scene observation used by the GR00T repeat gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import cv2  # noqa: F401 -- RoboLab requires this before Isaac Lab
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, default=8300)
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if args_cli.environment_seed not in {8300, 8301, 8302}:
    parser.error("The authorized fixed-observation seeds are 8300, 8301, 8302")
if args_cli.num_envs != 1:
    parser.error("The frozen repeat-gate observation requires one environment")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)

sys.path.insert(0, str(args_cli.study_root / "experiments/groot_droid"))
from policies.gr00t.client import GR00TDroidJointposClient  # noqa: E402


PROMPT = "Put the Rubik's cube to the left of the bowl."
TASK = "RubiksCubeLeftOfBowlMatchedTask"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    task_path = (
        args_cli.study_root
        / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py"
    )
    if not task_path.is_file():
        raise FileNotFoundError(task_path)

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(args_cli.output_dir))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    auto_register_droid_envs(task=[str(task_path)])

    env, env_cfg = create_env(
        TASK,
        device=args_cli.device,
        seed=args_cli.environment_seed,
        num_envs=1,
        instruction_type="default",
        policy="groot_v2_repeat_gate_fixture",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    try:
        # Match RoboLab's episode start exactly.
        env.reset()
        obs, _ = env.reset()
        if env_cfg.instruction != PROMPT:
            raise ValueError(f"Unexpected frozen prompt: {env_cfg.instruction!r}")

        cube_xyz = (
            env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        )
        bowl_xyz = env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        expected_cube_xy = np.array(
            [0.303364634513855, 0.12396888434886932], dtype=np.float32
        )
        expected_bowl_xy = np.array(
            [0.4425783157348633, 0.12660105526447296], dtype=np.float32
        )
        if not np.allclose(cube_xyz[:2], expected_cube_xy, atol=1e-3, rtol=0):
            raise ValueError(f"Cube reset is not the frozen neutral pose: {cube_xyz}")
        if not np.allclose(bowl_xyz[:2], expected_bowl_xy, atol=1e-3, rtol=0):
            raise ValueError(f"Bowl reset is not the frozen neutral pose: {bowl_xyz}")
        left_at_reset = bool(
            object_left_of(
                env,
                object="rubiks_cube",
                reference_object="bowl",
                frame_of_reference="robot",
                mirrored=False,
                require_gripper_detached=True,
                env_id=0,
            )
        )
        right_at_reset = bool(
            object_right_of(
                env,
                object="rubiks_cube",
                reference_object="bowl",
                frame_of_reference="robot",
                mirrored=False,
                require_gripper_detached=True,
                env_id=0,
            )
        )
        if left_at_reset or right_at_reset:
            raise ValueError(
                "Frozen matched reset must satisfy neither LEFT nor RIGHT; "
                f"got left={left_at_reset}, right={right_at_reset}"
            )

        helper = object.__new__(GR00TDroidJointposClient)
        extracted = helper._extract_observation(obs, env_id=0)
        request = helper._pack_request(extracted, PROMPT)
        numeric_request = {
            key: np.asarray(value)
            for key, value in request.items()
            if key != "annotation.language.language_instruction"
        }
        expected = {
            "video.exterior_image_1_left": ((1, 1, 180, 320, 3), "uint8"),
            "video.wrist_image_left": ((1, 1, 180, 320, 3), "uint8"),
            "state.eef_9d": ((1, 1, 9), "float32"),
            "state.joint_position": ((1, 1, 7), "float32"),
            "state.gripper_position": ((1, 1, 1), "float32"),
        }
        for key, (shape, dtype) in expected.items():
            arr = numeric_request[key]
            if arr.shape != shape or str(arr.dtype) != dtype:
                raise ValueError(
                    f"Observation contract mismatch for {key}: {arr.shape}/{arr.dtype}"
                )

        stem = f"seed{args_cli.environment_seed}_fixed_observation"
        npz_path = args_cli.output_dir / f"{stem}.npz"
        np.savez(npz_path, **numeric_request)
        exterior_npy_path = args_cli.output_dir / f"{stem}_exterior.npy"
        exterior_png_path = args_cli.output_dir / f"{stem}_exterior.png"
        wrist_npy_path = args_cli.output_dir / f"{stem}_wrist.npy"
        np.save(
            exterior_npy_path,
            numeric_request["video.exterior_image_1_left"],
            allow_pickle=False,
        )
        np.save(
            wrist_npy_path,
            numeric_request["video.wrist_image_left"],
            allow_pickle=False,
        )
        exterior_rgb = numeric_request["video.exterior_image_1_left"][0, 0]
        if not cv2.imwrite(
            str(exterior_png_path), cv2.cvtColor(exterior_rgb, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"Failed to write {exterior_png_path}")

        robolab_root = Path.cwd().resolve()
        robolab_commit = subprocess.check_output(
            ["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True
        ).strip()
        source_paths = [
            task_path,
            robolab_root / "assets/scenes/rubiks_cube_banana_bowl.usda",
            robolab_root / "assets/objects/ycb/bowl.usd",
            robolab_root / "assets/objects/ycb/banana.usd",
            robolab_root / "assets/objects/hot3d/rubiks_cube.usd",
            robolab_root / "assets/objects/ycb/textures/obj_000013.png",
            robolab_root / "assets/objects/ycb/textures/obj_000010.png",
            robolab_root / "assets/objects/hot3d/textures/obj_000030.png",
            robolab_root / "assets/fixtures/franka_table.usd",
            robolab_root / "assets/fixtures/table_maple.usd",
            robolab_root / "assets/fixtures/Props/instaceable_meshes.usd",
            robolab_root / "assets/backgrounds/default/home_office.exr",
            robolab_root / "assets/robots/franka_robotiq_2f_85_flattened.usd",
        ]
        for source_path in source_paths:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
        manifest_path = args_cli.output_dir / f"{stem}.json"
        manifest = {
            "schema_version": "vla-wam-shared-v2-groot-fixed-observation-v1",
            "environment_seed": args_cli.environment_seed,
            "task": TASK,
            "prompt": PROMPT,
            "reset_count": 2,
            "neutral_reset_contract": {
                "cube_world_xyz": cube_xyz.tolist(),
                "bowl_world_xyz": bowl_xyz.tolist(),
                "cube_minus_bowl_world_xyz": (cube_xyz - bowl_xyz).tolist(),
                "left_predicate_at_reset": left_at_reset,
                "right_predicate_at_reset": right_at_reset,
                "requires_neither_predicate": True,
            },
            "renderer": args_cli.renderer,
            "rendering_type": args_cli.rendering_type,
            "npz_path": str(npz_path),
            "npz_sha256": _sha256(npz_path),
            "exterior_image": {
                "npy_path": str(exterior_npy_path),
                "npy_sha256": _sha256(exterior_npy_path),
                "png_path": str(exterior_png_path),
                "png_sha256": _sha256(exterior_png_path),
            },
            "wrist_image": {
                "npy_path": str(wrist_npy_path),
                "npy_sha256": _sha256(wrist_npy_path),
            },
            "source_contract": {
                "robolab_root": str(robolab_root),
                "robolab_commit": robolab_commit,
                "files": {
                    str(path): _sha256(path) for path in source_paths
                },
            },
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(value.min()),
                    "max": float(value.max()),
                }
                for key, value in sorted(numeric_request.items())
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[GR00T v2 fixture] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
