#!/usr/bin/env python3
"""Capture one model-blind, officially packed V3-B005 probe observation.

Run this script in a fresh process once for each registered probe level 0, 3,
and 6.  It performs no policy request and no behavioral episode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from isaaclab.app import AppLauncher


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--manifest-sha256", required=True)
BOOTSTRAP.add_argument("--safe-fixture", type=Path, required=True)
BOOTSTRAP.add_argument("--safe-fixture-sha256", required=True)
BOOTSTRAP.add_argument("--physical-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--level-index", type=int, choices=(0, 3, 6), required=True)
BOOTSTRAP.add_argument("--environment-seed", type=int, default=9500)
BOOTSTRAP.add_argument("--output", type=Path, required=True)
BOOTSTRAP.add_argument("--pod", required=True)
BOOTSTRAP.add_argument("--pod-uid", required=True)
BOOTSTRAP.add_argument("--gpu-uuid", required=True)
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))

from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (  # noqa: E402
    ACTION_DIM,
    LEVELS,
    PHYSICAL_GATE_SCHEMA,
    PROBE_LEVELS,
    SETTLE_STEPS,
    STABILITY_WINDOW_STEPS,
    load_json,
    load_release_bundle,
    sha256_bytes,
    sha256_file,
)

release = load_release_bundle(
    bootstrap.manifest,
    expected_manifest_sha256=bootstrap.manifest_sha256,
)
if bootstrap.environment_seed != 9500 or bootstrap.level_index not in PROBE_LEVELS:
    BOOTSTRAP.error("fixed observations use seed 9500 and levels 0, 3, and 6 only")
if (
    not bootstrap.safe_fixture.is_file()
    or sha256_file(bootstrap.safe_fixture) != bootstrap.safe_fixture_sha256
    or bootstrap.safe_fixture_sha256 != release.safe_fixture_sha256
):
    BOOTSTRAP.error("safe fixture differs from the frozen V3-B005 fixture")
physical_gate = load_json(bootstrap.physical_gate, "V3-B005 physical gate")
if (
    physical_gate.get("schema_version") != PHYSICAL_GATE_SCHEMA
    or physical_gate.get("passed") is not True
    or sha256_file(bootstrap.physical_gate) != release.hashes["physical_gate"]
):
    BOOTSTRAP.error("physical gate differs from the passed V3-B005 gate")
if bootstrap.output.exists():
    BOOTSTRAP.error(f"refusing to overwrite {bootstrap.output}")

os.environ["VLA_WAM_V3B005_SAFE_FIXTURE"] = str(bootstrap.safe_fixture.resolve())
os.environ["VLA_WAM_V3B005_SAFE_FIXTURE_SHA256"] = bootstrap.safe_fixture_sha256
os.environ["VLA_WAM_V3B005_LEVEL_INDEX"] = str(bootstrap.level_index)

parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1 or not args_cli.headless:
    parser.error("fixed-observation capture requires one headless environment")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("fixed-observation capture requires realtime/balanced RTX")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402
from policies.cosmos3.client import Cosmos3Client  # noqa: E402


def _numeric(value: object) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]  # type: ignore[arg-type]


def _hold_action(obs: dict, device: str) -> torch.Tensor:
    arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(device)
    gripper = obs["proprio_obs"]["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    if tuple(action.shape) != (1, ACTION_DIM):
        raise RuntimeError(f"unexpected hold-action shape {tuple(action.shape)}")
    return action


def _array_record(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_bytes(array.tobytes()),
    }


def main() -> None:
    output_dir = bootstrap.output.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(output_dir / "native"))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    task_path = study_root / "experiments/v3/cosmos_nano_lateral_sweep/task_files/left.py"
    auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
    env, env_cfg = create_env(
        "V3B005NanoLeftTask",
        device=args_cli.device,
        seed=bootstrap.environment_seed,
        num_envs=1,
        instruction_type="default",
        policy="v3b005_zero_request_fixed_observation",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    try:
        counter = getattr(env, "episode_length_buf", None)
        if counter is None or not hasattr(counter, "zero_"):
            raise RuntimeError("RoboLab episode counter is unavailable")
        counter.zero_()
        obs, _ = env.reset()
        action = _hold_action(obs, env.device)
        for _ in range(SETTLE_STEPS + STABILITY_WINDOW_STEPS):
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("fixed-observation environment terminated while settling")
        # Force the probe to use the exact request preprocessing owned by the
        # official checkpoint client.  object.__new__ avoids opening a socket.
        packer = object.__new__(Cosmos3Client)
        extracted = Cosmos3Client._extract_observation(packer, obs)
        request = Cosmos3Client._pack_request(packer, extracted, env_cfg.instruction)
        arrays = {
            "image": np.asarray(request["observation/image"]),
            "joint_position": np.asarray(request["observation/joint_position"]),
            "gripper_position": np.asarray(request["observation/gripper_position"]),
        }
        if arrays["image"].shape != (540, 640, 3):
            raise RuntimeError(f"official packed image shape changed: {arrays['image'].shape}")
        for name in ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"):
            view = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy())
            if view.ndim != 3 or view.shape[-1] != 3 or not np.ptp(view):
                raise RuntimeError(f"blank or malformed RTX view: {name}")
        np.savez(bootstrap.output, **arrays)
        report = {
            "schema_version": "vla-wam-shared-v3b005-nano-fixed-observation-capture-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": "V3-B005",
            "level_index": bootstrap.level_index,
            "reference_object_initial_lateral_position_y_m": LEVELS[bootstrap.level_index],
            "environment_seed": bootstrap.environment_seed,
            "prompt_used_for_official_packing": env_cfg.instruction,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "settle_steps": SETTLE_STEPS,
            "stable_window_steps": STABILITY_WINDOW_STEPS,
            "observation_npz": {
                "path": str(bootstrap.output.resolve()),
                "sha256": sha256_file(bootstrap.output),
                "bytes": bootstrap.output.stat().st_size,
            },
            "array_fingerprints": {name: _array_record(value) for name, value in arrays.items()},
            "release_manifest_sha256": release.manifest_sha256,
            "safe_fixture_sha256": release.safe_fixture_sha256,
            "physical_gate_sha256": release.hashes["physical_gate"],
            "pod": bootstrap.pod,
            "pod_uid": bootstrap.pod_uid,
            "gpu_uuid": bootstrap.gpu_uuid,
            "hold_action": _numeric(action[0]),
        }
        report_path = bootstrap.output.with_suffix(".capture.json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"observation": report["observation_npz"], "capture_report": str(report_path)}, indent=2))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
