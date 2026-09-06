#!/usr/bin/env python3
"""Run the zero-behavior C7 Nano policy-session qualification probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from openpi_client import websocket_client_policy


CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
POLICY_ID = "cosmos3_nano_droid"
FIXTURE_ID = "object_pair"
LEFT_PROMPT = (
    "Place the sponge so that the sponge is left of the tray. "
    "Use the robot's fixed viewpoint for left, right, front, and behind."
)
RIGHT_PROMPT = (
    "Place the sponge so that the sponge is right of the tray. "
    "Use the robot's fixed viewpoint for left, right, front, and behind."
)
CONDITIONS = (
    ("left", LEFT_PROMPT),
    ("left_fresh_session_exact_repeat", LEFT_PROMPT),
    ("right", RIGHT_PROMPT),
)
JOINT_POSITION = np.asarray(
    [0.0, -0.6283185307179586, 0.0, -2.5132741228718345, 0.0, 1.8849555921538759, 0.0],
    dtype=np.float32,
)
GRIPPER_POSITION = np.asarray([0.0], dtype=np.float32)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _fresh_request(
    *,
    host: str,
    port: int,
    image: np.ndarray,
    prompt: str,
    sampling_seed: int,
) -> tuple[dict[str, Any], float]:
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    started = time.monotonic()
    try:
        response = client.infer(
            {
                "observation/image": image,
                "observation/joint_position": JOINT_POSITION,
                "observation/gripper_position": GRIPPER_POSITION,
                "prompt": prompt,
                "sampling_seed": sampling_seed,
                "action_step_start": 0,
            }
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if not isinstance(response, dict):
        raise RuntimeError("Nano response must be an object")
    return response, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--seed-registry", type=Path, required=True)
    parser.add_argument("--seed-registry-sha256", required=True)
    parser.add_argument("--conditioning-image", type=Path, required=True)
    parser.add_argument("--conditioning-image-sha256", required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G4 output: {args.output}")
    if _sha256_file(args.conditioning_image) != args.conditioning_image_sha256:
        raise RuntimeError("G4 conditioning image digest mismatch")
    if _sha256_file(args.seed_registry) != args.seed_registry_sha256:
        raise RuntimeError("G4 seed registry digest mismatch")

    checkpoint_registry = json.loads(args.checkpoint_registry.read_text(encoding="utf-8"))
    checkpoint = checkpoint_registry.get("checkpoint", {})
    if checkpoint.get("revision") != CHECKPOINT_REVISION:
        raise RuntimeError("G4 checkpoint revision mismatch")
    if checkpoint.get("hash_gate_passed") is not True:
        raise RuntimeError("G4 checkpoint hash gate has not passed")

    seed_registry = json.loads(args.seed_registry.read_text(encoding="utf-8"))
    if seed_registry.get("scope") != "g4_policy_session_only":
        raise RuntimeError("G4 seed registry scope mismatch")
    if args.sampling_seed not in seed_registry.get("allowed_sampling_seeds", []):
        raise RuntimeError("G4 sampling seed is not registered")

    image_bgr = cv2.imread(str(args.conditioning_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.conditioning_image)
    source_shape = list(image_bgr.shape)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 540), interpolation=cv2.INTER_AREA)
    if image.shape != (540, 640, 3) or image.dtype != np.uint8:
        raise RuntimeError("G4 conditioning image transform failed")

    args.output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    actions: dict[str, np.ndarray] = {}
    futures: dict[str, np.ndarray] = {}
    try:
        for condition, prompt in CONDITIONS:
            response, wall_seconds = _fresh_request(
                host=args.host,
                port=args.port,
                image=image,
                prompt=prompt,
                sampling_seed=args.sampling_seed,
            )
            action = np.asarray(response.get("action", response.get("actions")), dtype=np.float32)
            future = np.asarray(response.get("video"), dtype=np.uint8)
            if action.shape != (32, 8) or not np.isfinite(action).all():
                raise RuntimeError(f"{condition}: invalid Nano action contract {action.shape}")
            if future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
                raise RuntimeError(f"{condition}: invalid Nano future contract {future.shape}")
            if response.get("sampling_seed") != args.sampling_seed:
                raise RuntimeError(f"{condition}: Nano server did not echo sampling seed")
            if response.get("v4_seed_registry_sha256") != args.seed_registry_sha256:
                raise RuntimeError(f"{condition}: Nano server seed registry binding mismatch")
            if response.get("v4_seed_scope") != "g4_policy_session_only":
                raise RuntimeError(f"{condition}: Nano server qualification scope mismatch")
            action_path = args.output / f"{condition}.action.npy"
            future_path = args.output / f"{condition}.future.npy"
            np.save(action_path, action, allow_pickle=False)
            np.save(future_path, future, allow_pickle=False)
            actions[condition] = action
            futures[condition] = future
            records.append(
                {
                    "condition": condition,
                    "prompt": prompt,
                    "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    "sampling_seed": args.sampling_seed,
                    "fresh_client_session": True,
                    "action_shape": list(action.shape),
                    "future_shape": list(future.shape),
                    "wall_seconds": wall_seconds,
                    "action": _artifact(action_path),
                    "future": _artifact(future_path),
                }
            )

        repeat_action_equal = bool(
            np.array_equal(actions["left"], actions["left_fresh_session_exact_repeat"])
        )
        repeat_future_equal = bool(
            np.array_equal(futures["left"], futures["left_fresh_session_exact_repeat"])
        )
        prompt_action_rms = float(
            np.sqrt(
                np.mean(
                    (
                        actions["left"].astype(np.float64)
                        - actions["right"].astype(np.float64)
                    )
                    ** 2
                )
            )
        )
        prompt_future_mae = float(
            np.mean(
                np.abs(
                    futures["left"].astype(np.float64)
                    - futures["right"].astype(np.float64)
                )
            )
        )
        if not math.isfinite(prompt_action_rms) or not math.isfinite(prompt_future_mae):
            raise RuntimeError("G4 prompt comparison metrics are nonfinite")
        passed = repeat_action_equal and repeat_future_equal
        receipt = {
            "schema_version": "v4-object-pair-g4-nano-policy-session-receipt-v1",
            "campaign_id": "online_correction_v4",
            "gate": "G4",
            "fixture_id": FIXTURE_ID,
            "policy_id": POLICY_ID,
            "status": "passed" if passed else "failed",
            "passed": passed,
            "qualification_scope": "policy_session_only_no_behavioral_episode",
            "behavioral_episode_count": 0,
            "model_request_count": len(CONDITIONS),
            "fresh_client_session_per_request": True,
            "checkpoint_registry": _artifact(args.checkpoint_registry),
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_content_manifest_sha256": checkpoint.get("present_metadata_sha256"),
            "seed_registry": _artifact(args.seed_registry),
            "conditioning_input": {
                **_artifact(args.conditioning_image),
                "source_shape": source_shape,
                "submitted_shape": list(image.shape),
                "submitted_rgb_sha256": _sha256_bytes(np.ascontiguousarray(image).tobytes()),
                "joint_position_sha256": _sha256_bytes(JOINT_POSITION.tobytes()),
                "gripper_position_sha256": _sha256_bytes(GRIPPER_POSITION.tobytes()),
            },
            "records": records,
            "checks": {
                "action_shape_32x8_and_finite": True,
                "future_33_frame_rgb_present": True,
                "sampling_seed_echoed": True,
                "fresh_session_exact_repeat_actions_equal": repeat_action_equal,
                "fresh_session_exact_repeat_futures_equal": repeat_future_equal,
                "static_prompt_bytes_bound": True,
            },
            "reported_not_gated_language_diagnostics": {
                "left_right_action_rms": prompt_action_rms,
                "left_right_future_pixel_mae": prompt_future_mae,
                "note": "G4 does not require a positive prompt effect.",
            },
            "release_boundary": (
                "A pass establishes the C7 Nano G4 policy-session interface only. "
                "G5-G8 and an immutable released runtime lock remain required before "
                "confirmatory policy episodes."
            ),
        }
        receipt_path = args.output / "g4_policy_session_receipt.json"
        receipt_path.write_bytes(_canonical_json_bytes(receipt))
        print(
            json.dumps(
                {
                    "passed": passed,
                    "receipt": _artifact(receipt_path),
                    "model_request_count": len(CONDITIONS),
                },
                sort_keys=True,
            )
        )
        if not passed:
            raise SystemExit(20)
    except Exception:
        failure_path = args.output / "infrastructure_failure.json"
        if not failure_path.exists():
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": "v4-object-pair-g4-infrastructure-failure-v1",
                        "behavioral_episode_count": 0,
                        "model_request_count_completed": len(records),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
