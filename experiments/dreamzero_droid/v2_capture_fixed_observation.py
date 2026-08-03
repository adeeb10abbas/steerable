#!/usr/bin/env python3
"""Prove the DreamZero neutral reset, request contract, and RTX viewport writer."""

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
import torch
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
if args_cli.environment_seed != 8300:
    parser.error("The DreamZero fixed-observation gate is frozen at seed 8300")
if args_cli.num_envs != 1:
    parser.error("The DreamZero fixed-observation gate requires one environment")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_viewport_cams  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)

sys.path.insert(0, str(args_cli.study_root / "experiments/dreamzero_droid"))
from v2_robolab_client import (  # noqa: E402
    LEFT,
    V2DreamZeroDroidClient,
)


TASK = "RubiksCubeLeftOfBowlMatchedTask"
EXPECTED_CUBE_XY = np.array([0.303364634513855, 0.12396888434886932])
EXPECTED_BOWL_XY = np.array([0.4425783157348633, 0.12660105526447296])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    task_path = (
        args_cli.study_root
        / "experiments/dreamzero_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py"
    )
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
        policy="dreamzero_v2_contract_gate",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    try:
        print("[DreamZero v2 fixture] environment created", flush=True)
        env.reset()
        obs, _ = env.reset()
        print("[DreamZero v2 fixture] exact reset captured", flush=True)
        if env_cfg.instruction != LEFT:
            raise ValueError(f"Frozen prompt bytes changed: {env_cfg.instruction!r}")
        cube_xyz = env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl_xyz = env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        if not np.allclose(cube_xyz[:2], EXPECTED_CUBE_XY, atol=1e-3, rtol=0):
            raise ValueError(f"Cube reset is not frozen neutral pose: {cube_xyz}")
        if not np.allclose(bowl_xyz[:2], EXPECTED_BOWL_XY, atol=1e-3, rtol=0):
            raise ValueError(f"Bowl reset is not frozen neutral pose: {bowl_xyz}")
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
            raise ValueError("Matched reset must begin outside both requested predicates")
        print("[DreamZero v2 fixture] neutral predicates verified", flush=True)

        helper = object.__new__(V2DreamZeroDroidClient)
        helper.cam2_source = "right"
        helper.resize = "pad"
        helper.image_height = 180
        helper.image_width = 320
        helper._env_session_id = {}
        extracted = helper._extract_observation(obs, env_id=0)
        request = helper._pack_request(extracted, LEFT)
        expected = {
            "observation/exterior_image_0_left": ((180, 320, 3), "uint8"),
            "observation/exterior_image_1_left": ((180, 320, 3), "uint8"),
            "observation/wrist_image_left": ((180, 320, 3), "uint8"),
            "observation/joint_position": ((7,), "float64"),
            "observation/cartesian_position": ((6,), "float64"),
            "observation/gripper_position": ((1,), "float64"),
        }
        arrays = {key: np.asarray(request[key]) for key in expected}
        for key, (shape, dtype) in expected.items():
            if arrays[key].shape != shape or str(arrays[key].dtype) != dtype:
                raise ValueError(f"Request contract mismatch for {key}: {arrays[key].shape}/{arrays[key].dtype}")
        fixture_path = args_cli.output_dir / "seed8300_fixed_observation.npz"
        np.savez(fixture_path, **arrays)
        print("[DreamZero v2 fixture] wire fixture retained", flush=True)

        viewport_path = args_cli.output_dir / "rtx_viewport_persistence_gate.mp4"
        video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
        writer = VideoWriter(str(viewport_path), video_fps)
        try:
            for _ in range(8):
                frame = unpack_viewport_cams(obs, env_id=0).get("combined_image")
                writer.write(frame)
                hold = np.concatenate(
                    [
                        np.asarray(obs["proprio_obs"]["arm_joint_pos"][0].cpu()),
                        np.asarray(obs["proprio_obs"]["gripper_pos"][0].cpu()),
                    ]
                ).astype(np.float32)
                obs, _, _, _, _ = env.step(torch.as_tensor(hold[None], device=env.device))
        finally:
            writer.release()
        capture = cv2.VideoCapture(str(viewport_path))
        decoded_frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded_frames += 1
        capture.release()
        if decoded_frames != 8:
            raise ValueError(f"Viewport writer produced {decoded_frames} frames, expected 8")
        print("[DreamZero v2 fixture] viewport video decoded", flush=True)

        robolab_root = Path.cwd().resolve()
        robolab_commit = subprocess.check_output(
            ["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True
        ).strip()
        manifest = {
            "schema_version": "vla-wam-shared-v2-dreamzero-fixed-observation-v1",
            "status": "passed",
            "environment_seed": args_cli.environment_seed,
            "task": TASK,
            "prompt": LEFT,
            "prompt_utf8_sha256": hashlib.sha256(LEFT.encode()).hexdigest(),
            "reset_count": 2,
            "neutral_reset_contract": {
                "cube_world_xyz": cube_xyz.tolist(),
                "bowl_world_xyz": bowl_xyz.tolist(),
                "left_predicate_at_reset": left_at_reset,
                "right_predicate_at_reset": right_at_reset,
            },
            "request_contract": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(arrays.items())
            },
            "fixture": {
                "path": str(fixture_path),
                "sha256": _sha256(fixture_path),
            },
            "renderer": {
                "pod": "raytrace-rtxpro6000-ali",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "mode": args_cli.renderer,
                "rendering_type": args_cli.rendering_type,
                "omniverse_eula_accepted_by_user": True,
                "eula_environment_variable": "OMNI_KIT_ACCEPT_EULA=YES",
                "viewport_video": str(viewport_path),
                "viewport_video_sha256": _sha256(viewport_path),
                "viewport_video_bytes": viewport_path.stat().st_size,
                "decoded_frame_count": decoded_frames,
            },
            "source_contract": {
                "robolab_root": str(robolab_root),
                "robolab_commit": robolab_commit,
                "task_path": str(task_path),
                "task_sha256": _sha256(task_path),
            },
            "claim_boundary": "No model was loaded or queried by this reset/renderer gate.",
        }
        manifest_path = args_cli.output_dir / "fixed_observation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = {
            "schema_version": "vla-wam-shared-v2-dreamzero-renderer-failure-v1",
            "status": "technical_failure",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "claim_boundary": "No model was loaded or queried by this reset/renderer gate.",
        }
        failure_path = args_cli.output_dir / "technical_failure.json"
        failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"[DreamZero v2 fixture] technical failure retained: {failure_path}", flush=True)
        print(failure["traceback"], flush=True)
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[DreamZero v2 fixture] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
