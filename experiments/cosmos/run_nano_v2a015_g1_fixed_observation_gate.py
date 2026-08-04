#!/usr/bin/env python3
"""Run the frozen three-request Cosmos3 Nano V2-A015 g=1 release gate."""

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
FROZEN_SOURCE_PLAN_SHA256 = "c5025b51f57224308d16338db38ab57b06e7025ba9e585a05993e01a79952fc3"
FROZEN_OBSERVATION_HASHES = {
    "image_raw_rgb_sha256": "6261ce5ab21383342c2012c14f7ff97d3dcd74e5f4202f2b3444355cc7ba3332",
    "joint_position_raw_sha256": "6661cda587dce95d7e39db70d98bbed73498aecc82f0a1407ea41bf3fd839b85",
    "gripper_position_raw_sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
}
MODEL_ID = "cosmos3_nano_policy_droid"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
AMENDMENT_ID = "V2-A015"
ARM_ID = "cosmos3_nano_no_cfg_g1"
GUIDANCE = 1.0
BASELINE_GUIDANCE = 3.0
BASELINE_RESULT_ARTIFACT = (
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "cosmos3_nano_policy_droid_direct_gate.json"
)
BASELINE_RESULT_SHA256 = "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93"


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


def _cosmos_arm(amendment: dict) -> dict:
    arms = [arm for arm in amendment.get("arms", []) if arm.get("arm_id") == ARM_ID]
    if len(arms) != 1:
        raise ValueError(f"Expected exactly one {ARM_ID!r} arm in {AMENDMENT_ID}")
    arm = arms[0]
    expected = {
        "model_id": MODEL_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "guidance": GUIDANCE,
        "baseline_guidance": BASELINE_GUIDANCE,
        "num_steps": 4,
        "shift": 5.0,
        "action_chunk_shape": [32, 8],
        "future_contract": "decoded 33-frame RGB future for every policy request",
        "behavioral_episode_count": 6,
    }
    for key, value in expected.items():
        if arm.get(key) != value:
            raise ValueError(
                f"V2-A015 arm changed for {key}: expected={value!r}, observed={arm.get(key)!r}"
            )
    return arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--conditioning-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18021)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    amendment = json.loads(args.amendment.read_text())
    source_plan = json.loads(args.source_plan.read_text())
    if amendment.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("The supplied amendment is not V2-A015")
    _cosmos_arm(amendment)
    baseline_disclosure = amendment["known_result_disclosure"]["cosmos3_nano_baseline"]
    expected_baseline_disclosure = {
        "guidance": BASELINE_GUIDANCE,
        "artifact": BASELINE_RESULT_ARTIFACT,
        "sha256": BASELINE_RESULT_SHA256,
        "result": "LEFT 3/3; RIGHT 3/3; 3/3 aligned endpoint pairs",
    }
    for key, value in expected_baseline_disclosure.items():
        if baseline_disclosure.get(key) != value:
            raise ValueError(
                f"V2-A015 baseline disclosure changed for {key}: "
                f"expected={value!r}, observed={baseline_disclosure.get(key)!r}"
            )
    if _sha256(args.baseline_result) != BASELINE_RESULT_SHA256:
        raise ValueError("The supplied g=3 baseline result does not match its frozen hash")
    baseline = json.loads(args.baseline_result.read_text())
    if baseline.get("amendment_id") != "V2-A011":
        raise ValueError("The frozen g=3 baseline result is not V2-A011")
    if baseline.get("model_id") != MODEL_ID or baseline.get("summary", {}).get("successes") != 6:
        raise ValueError("The frozen g=3 baseline result identity or 6/6 summary changed")
    if registry["checkpoint"]["revision"] != CHECKPOINT_REVISION:
        raise ValueError("Registry checkpoint revision changed")
    if registry["checkpoint"]["hash_gate_passed"] is not True:
        raise ValueError("Checkpoint payload hash gate has not passed")
    if registry["serving_contract"]["guidance"] != BASELINE_GUIDANCE:
        raise ValueError("The archived registry no longer identifies the g=3 baseline")
    prompts = registry["behavioral_queue"]["prompts"]
    if prompts != {"left": LEFT, "right": RIGHT}:
        raise ValueError("Registry prompt bytes do not match the frozen gate")
    if amendment["behavioral_grid"]["prompts"] != {"left": LEFT, "right": RIGHT}:
        raise ValueError("V2-A015 prompt bytes do not match the frozen gate")
    if registry["fixed_observation_gate"]["conditions"] != [
        "left",
        "left_exact_repeat",
        "right",
    ]:
        raise ValueError("Registry fixed-observation conditions changed")
    conditioning_png_sha256 = _sha256(args.conditioning_image)
    if conditioning_png_sha256 != FROZEN_GROUNDED_OBSERVATION_SHA256:
        raise ValueError("Input is not the committed frozen grounded observation")
    if _sha256(args.source_plan) != FROZEN_SOURCE_PLAN_SHA256:
        raise ValueError("Input is not the frozen V2-A011 source-state plan")

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
    if observation_hashes != FROZEN_OBSERVATION_HASHES:
        raise ValueError("Decoded image/state arrays do not match the frozen fixture")
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
        expected_response_metadata = {
            "sampling_seed": 8300,
            "amendment_id": AMENDMENT_ID,
            "arm_id": ARM_ID,
            "guidance": GUIDANCE,
            "baseline_guidance": BASELINE_GUIDANCE,
            "baseline_result_artifact": BASELINE_RESULT_ARTIFACT,
            "baseline_result_sha256": BASELINE_RESULT_SHA256,
        }
        for key, value in expected_response_metadata.items():
            if response.get(key) != value:
                raise ValueError(
                    f"{condition}: server metadata mismatch for {key}: "
                    f"expected={value!r}, observed={response.get(key)!r}"
                )
        action = np.asarray(response.get("action", response.get("actions")), dtype=np.float32)
        video = np.asarray(response.get("video"), dtype=np.uint8)
        if action.shape != (32, 8):
            raise ValueError(f"{condition}: expected action [32,8], got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError(f"{condition}: action contains a non-finite value")
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
        records.append(
            {
                "condition": condition,
                "prompt": prompt,
                "amendment_id": AMENDMENT_ID,
                "arm_id": ARM_ID,
                "guidance": GUIDANCE,
                "baseline_guidance": BASELINE_GUIDANCE,
                "baseline_result_artifact": BASELINE_RESULT_ARTIFACT,
                "baseline_result_sha256": BASELINE_RESULT_SHA256,
                "requested_sampling_seed": 8300,
                "server_sampling_seed": 8300,
                "wall_seconds": time.perf_counter() - started,
                "action_shape": list(action.shape),
                "action_finite": True,
                "future_shape": list(video.shape),
                "action_sha256": _sha256(action_path),
                "future_npy_sha256": _sha256(future_path),
                "future_mp4_sha256": _sha256(mp4_path),
            }
        )
    metrics = {
        "left_repeat_action_bit_identical": bool(
            np.array_equal(actions["left"], actions["left_exact_repeat"])
        ),
        "left_repeat_future_bit_identical": bool(
            np.array_equal(futures["left"], futures["left_exact_repeat"])
        ),
        "left_right_action_distinct": bool(not np.array_equal(actions["left"], actions["right"])),
        "left_right_future_distinct": bool(not np.array_equal(futures["left"], futures["right"])),
        "left_repeat_action_rms": _rms(actions["left"], actions["left_exact_repeat"]),
        "left_repeat_future_pixel_mae": _mae(futures["left"], futures["left_exact_repeat"]),
        "left_right_action_rms": _rms(actions["left"], actions["right"]),
        "left_right_future_pixel_mae": _mae(futures["left"], futures["right"]),
    }
    passed = (
        metrics["left_repeat_action_bit_identical"]
        and metrics["left_repeat_future_bit_identical"]
        and metrics["left_right_action_distinct"]
        and metrics["left_right_future_distinct"]
    )
    manifest = {
        "schema_version": (
            "vla-wam-shared-v2-cosmos3-nano-policy-droid-v2a015-g1-"
            "fixed-observation-v1"
        ),
        "status": "passed" if passed else "failed",
        "model_id": MODEL_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "amendment_id": AMENDMENT_ID,
        "arm_id": ARM_ID,
        "guidance": GUIDANCE,
        "baseline_guidance": BASELINE_GUIDANCE,
        "baseline_result": {
            "artifact": BASELINE_RESULT_ARTIFACT,
            "sha256": BASELINE_RESULT_SHA256,
            "reported_result": baseline_disclosure["result"],
        },
        "amendment_sha256": _sha256(args.amendment),
        "registry_sha256": _sha256(args.registry),
        "source_plan_sha256": _sha256(args.source_plan),
        "conditioning_png_sha256": conditioning_png_sha256,
        "observation_hashes": observation_hashes,
        "server": f"ws://{args.host}:{args.port}",
        "records": records,
        "metrics": metrics,
        "claim_boundary": (
            "Determinism and prompt-sensitivity release gate for the V2-A015 g=1 arm; "
            "not robot success and not a replacement for the archived g=3 baseline."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
