#!/usr/bin/env python3
"""Run the frozen three-request Cosmos3 Nano fixed-observation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from openpi_client import websocket_client_policy

LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
CONDITIONS = (("left", LEFT), ("left_exact_repeat", LEFT), ("right", RIGHT))
FROZEN_GROUNDED_OBSERVATION_SHA256 = "2a431b0fa288890b3509b314c0351c91123d5f64b237678fed972848e29cd55b"
MODEL_ID = "cosmos3_nano_policy_droid"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _save_video(path: Path, video: np.ndarray, fps: float = 15.0) -> None:
    height, width = video.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in video:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--conditioning-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18011)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    source_plan = json.loads(args.source_plan.read_text())
    if registry["checkpoint"]["revision"] != CHECKPOINT_REVISION:
        raise ValueError("Registry checkpoint revision changed")
    if registry["checkpoint"]["hash_gate_passed"] is not True:
        raise ValueError("Checkpoint payload hash gate has not passed")
    prompts = registry["behavioral_queue"]["prompts"]
    if prompts != {"left": LEFT, "right": RIGHT}:
        raise ValueError("Registry prompt bytes do not match the frozen gate")
    if registry["fixed_observation_gate"]["conditions"] != ["left", "left_exact_repeat", "right"]:
        raise ValueError("Registry fixed-observation conditions changed")
    conditioning_png_sha256 = _sha256(args.conditioning_image)
    if conditioning_png_sha256 != FROZEN_GROUNDED_OBSERVATION_SHA256:
        raise ValueError("Input is not the committed frozen grounded observation")

    image_bgr = cv2.imread(str(args.conditioning_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.conditioning_image)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if image.shape != (540, 640, 3) or image.dtype != np.uint8:
        raise ValueError(f"Unexpected conditioning image contract: {image.shape}/{image.dtype}")
    source = source_plan["source"]
    joint = np.asarray(source["joint_position"], dtype=np.float32)
    gripper = np.asarray(source["gripper_position"], dtype=np.float32)
    observation_hashes = {
        "image_raw_rgb_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
        "joint_position_raw_sha256": hashlib.sha256(joint.tobytes()).hexdigest(),
        "gripper_position_raw_sha256": hashlib.sha256(gripper.tobytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    actions: dict[str, np.ndarray] = {}
    futures: dict[str, np.ndarray] = {}
    records = []
    for condition, prompt in CONDITIONS:
        request = {
            "observation/image": image,
            "observation/joint_position": joint,
            "observation/gripper_position": gripper,
            "prompt": prompt,
            "sampling_seed": 8300,
        }
        started = time.perf_counter()
        response = client.infer(request)
        action = np.asarray(response.get("action", response.get("actions")), dtype=np.float32)
        video = np.asarray(response.get("video"), dtype=np.uint8)
        if response.get("sampling_seed") != 8300:
            raise ValueError(f"{condition}: server did not echo sampling_seed=8300")
        if action.shape != (32, 8):
            raise ValueError(f"{condition}: expected action [32,8], got {action.shape}")
        if video.ndim != 4 or video.shape[0] != 33 or video.shape[-1] != 3:
            raise ValueError(f"{condition}: expected 33-frame RGB future, got {video.shape}")
        action_path = args.output_dir / f"{condition}_action.npy"
        future_path = args.output_dir / f"{condition}_future.npy"
        mp4_path = args.output_dir / f"{condition}_future.mp4"
        np.save(action_path, action, allow_pickle=False)
        np.save(future_path, video, allow_pickle=False)
        _save_video(mp4_path, video)
        actions[condition] = action
        futures[condition] = video
        records.append({
            "condition": condition,
            "prompt": prompt,
            "requested_sampling_seed": 8300,
            "server_sampling_seed": 8300,
            "wall_seconds": time.perf_counter() - started,
            "action_shape": list(action.shape),
            "future_shape": list(video.shape),
            "action_sha256": _sha256(action_path),
            "future_npy_sha256": _sha256(future_path),
            "future_mp4_sha256": _sha256(mp4_path),
        })
    metrics = {
        "left_repeat_action_rms": _rms(actions["left"], actions["left_exact_repeat"]),
        "left_repeat_future_pixel_mae": _mae(futures["left"], futures["left_exact_repeat"]),
        "left_right_action_rms": _rms(actions["left"], actions["right"]),
        "left_right_future_pixel_mae": _mae(futures["left"], futures["right"]),
    }
    passed = (
        metrics["left_repeat_action_rms"] == 0.0
        and metrics["left_repeat_future_pixel_mae"] == 0.0
        and metrics["left_right_action_rms"] > 0.0
        and metrics["left_right_future_pixel_mae"] > 0.0
    )
    manifest = {
        "schema_version": "vla-wam-shared-v2-cosmos3-nano-policy-droid-fixed-observation-v1",
        "status": "passed" if passed else "failed",
        "model_id": MODEL_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "amendment_id": "V2-A011",
        "registry_sha256": _sha256(args.registry),
        "source_plan_sha256": _sha256(args.source_plan),
        "conditioning_png_sha256": conditioning_png_sha256,
        "observation_hashes": observation_hashes,
        "server": f"ws://{args.host}:{args.port}",
        "records": records,
        "metrics": metrics,
        "claim_boundary": "Sensitivity and determinism diagnostic only; not robot success.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
