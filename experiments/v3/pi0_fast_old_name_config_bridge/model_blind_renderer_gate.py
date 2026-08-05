#!/usr/bin/env python3
"""Launch the frozen RoboLab reset and verify live RTX rendering without inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import traceback

import cv2
import numpy as np
from isaaclab.app import AppLauncher


MODEL_ID = "pi0_fast_old_name_config_v3a002"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
PROMPT = "Put the Rubik's cube to the left of the bowl."
OPENPI_CLIENT_ROOT = Path(
    "/data/users/ali/vla_wam/external/openpi-235044ed/packages/openpi-client/src"
)
PYTHONPATH = [
    "/data/users/ali/vla_wam/src/steerable",
    "/data/users/ali/vla_wam/external/RoboLab-pi0fast-bridge-0aef241-clean01",
    "/data/users/ali/vla_wam/external/openpi-235044ed/packages/openpi-client/src",
]


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, default=8310)
parser.add_argument("--pod", required=True)
parser.add_argument("--pod-uid", required=True)
parser.add_argument("--gpu-uuid", required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, default=8011)
from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.environment_seed != 8310 or args_cli.num_envs != 1:
    parser.error("V3-A002 renderer gate requires seed 8310 and one environment")
if not args_cli.headless:
    parser.error("V3-A002 renderer gate must run headless")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("V3-A002 renderer gate requires realtime/balanced RTX rendering")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import openpi_client  # noqa: E402
import policies.pi0_family.client as pi0_client_module  # noqa: E402
import robolab  # noqa: E402
import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    if args_cli.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite renderer-gate output: {args_cli.output_dir}")
    if os.environ.get("PYTHONPATH", "").split(":") != PYTHONPATH:
        raise ValueError("renderer gate requires the exact frozen PYTHONPATH order")
    commit = subprocess.check_output(
        ["git", "-C", str(args_cli.robolab_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(args_cli.robolab_root), "status", "--porcelain=v1"], text=True
    )
    if commit != ROBOLAB_COMMIT or status:
        raise ValueError("renderer gate requires the clean frozen RoboLab revision")
    client_path = Path(openpi_client.__file__).resolve()
    if not client_path.is_relative_to(OPENPI_CLIENT_ROOT.resolve()):
        raise ValueError(f"openpi_client resolved outside frozen 235044ed source: {client_path}")
    robolab_import = Path(robolab.__file__).resolve()
    policy_client_import = Path(pi0_client_module.__file__).resolve()
    for label, path in (
        ("robolab", robolab_import),
        ("RoboLab pi0 client", policy_client_import),
    ):
        if not path.is_relative_to(args_cli.robolab_root.resolve()):
            raise ValueError(f"{label} resolved outside the frozen RoboLab worktree: {path}")

    task_path = args_cli.study_root / (
        "experiments/groot_droid/robolab_v2_tasks/"
        "rubiks_cube_left_of_bowl_matched.py"
    )
    args_cli.output_dir.mkdir(parents=True)
    set_output_dir(str(args_cli.output_dir))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    auto_register_droid_envs(task=[str(task_path)])
    env, env_cfg = create_env(
        "RubiksCubeLeftOfBowlMatchedTask",
        device=args_cli.device,
        seed=args_cli.environment_seed,
        num_envs=1,
        instruction_type="default",
        policy="pi0_fast_old_name_config_v3a002_renderer_gate",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    try:
        env.reset()
        obs, _ = env.reset()
        if env_cfg.instruction != PROMPT:
            raise ValueError("renderer-gate prompt differs from the frozen direct prompt")
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
            raise ValueError("renderer-gate reset must satisfy neither LEFT nor RIGHT")

        helper = object.__new__(Pi0DroidJointposClient)
        extracted = helper._extract_observation(obs, env_id=0)
        request = helper._pack_request(extracted, PROMPT)
        if request.pop("prompt") != PROMPT:
            raise ValueError("packed prompt differs from the frozen direct prompt")
        arrays = {key: np.asarray(value) for key, value in request.items()}
        required = {
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
            "observation/joint_position",
            "observation/gripper_position",
        }
        if set(arrays) != required or any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("live renderer observation contract is invalid")
        frame = np.asarray(arrays["observation/exterior_image_1_left"], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3 or not np.ptp(frame):
            raise ValueError("live RTX camera frame is blank or malformed")

        video = args_cli.output_dir / "live_neutral_reset_viewport.mp4"
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError("renderer-gate MP4 writer did not open")
        try:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(10):
                writer.write(bgr)
        finally:
            writer.release()
        decoder = cv2.VideoCapture(str(video))
        try:
            ok, decoded = decoder.read()
            frame_count = int(decoder.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            decoder.release()
        if not ok or decoded is None or frame_count < 1:
            raise RuntimeError("live renderer-gate MP4 failed decode")

        gpu_line = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ], text=True,
        ).splitlines()[0]
        if args_cli.gpu_uuid not in gpu_line:
            raise ValueError("live renderer GPU UUID differs from the assigned pod GPU")
        metadata_client = WebsocketClientPolicy(
            host=args_cli.remote_host, port=args_cli.remote_port
        )
        try:
            server_metadata = metadata_client.get_server_metadata()
        finally:
            metadata_client._ws.close()  # noqa: SLF001 - metadata-only gate owns it
        expected_metadata = {
            "pi0_fast_old_name_config_bridge": "v3a002",
            "openpi_commit": "235044ed8a1502c0a18338eedc5d7adfe705af05",
            "openpi_tree": "03a4387bedbc0fa1467c367c60fc24e28b61ec6c",
            "openpi_config": "pi0_fast_droid_jointpos",
            "max_token_len": 250,
            "checkpoint_assets_rule": "checkpoint_local_assets_only",
            "sampling_contract": "required_request_field:sampling_seed",
        }
        for key, wanted in expected_metadata.items():
            if server_metadata.get(key) != wanted:
                raise ValueError(f"metadata-only WebSocket gate mismatch for {key}")
        manifest = {
            "schema_version": "vla-wam-shared-v3-pi0-fast-old-name-config-model-blind-renderer-gate-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "model_id": MODEL_ID,
            "passed": True,
            "status": "passed",
            "pod": args_cli.pod,
            "pod_uid": args_cli.pod_uid,
            "gpu_uuid": args_cli.gpu_uuid,
            "model_request_count": 0,
            "environment_seed": args_cli.environment_seed,
            "prompt": PROMPT,
            "neutral_reset_contract": {
                "left_predicate_at_reset": left,
                "right_predicate_at_reset": right,
            },
            "renderer": {"backend": "realtime RTX Vulkan", "quality": "balanced"},
            "nvidia_icd": {
                "path": "/etc/vulkan/icd.d/nvidia_icd.json",
                "sha256": sha256(Path("/etc/vulkan/icd.d/nvidia_icd.json")),
            },
            "gpu_query": gpu_line,
            "policy_endpoint": f"{args_cli.remote_host}:{args_cli.remote_port}",
            "websocket_metadata_only_handshake": {
                "passed": True,
                "inference_requests_sent": 0,
                "server_metadata": expected_metadata,
            },
            "robolab_commit": commit,
            "effective_imports": {
                "pythonpath": PYTHONPATH,
                "modules": {
                    "openpi_client": record(client_path),
                    "robolab": record(robolab_import),
                    "policies.pi0_family.client": record(policy_client_import),
                    "openpi_client.websocket_client_policy": record(
                        OPENPI_CLIENT_ROOT / "openpi_client/websocket_client_policy.py"
                    ),
                    "openpi_client.msgpack_numpy": record(
                        OPENPI_CLIENT_ROOT / "openpi_client/msgpack_numpy.py"
                    ),
                },
            },
            "simulator_versions": {
                name: importlib.metadata.version(name)
                for name in ("isaacsim", "isaaclab", "robolab")
            },
            "observation_arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(arrays.items())
            },
            "viewport_video": record(video),
            "video_decode_frame_count": frame_count,
        }
        output = args_cli.output_dir / "renderer_gate_manifest.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[π0-FAST V3-A002 renderer gate] technical failure: {error}")
        traceback.print_exc()
        simulation_app.close()
        raise
