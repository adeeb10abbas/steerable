#!/usr/bin/env python3
"""Repeat/sensitivity preflight for the exact Cosmos3 v3 Phase-A stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.v3.cosmos_droid.contract import (
    MODEL_CONTRACTS,
    AuthorizedPair,
    ContractError,
    load_authorized_pair,
    sha256_file,
    verify_runtime_identity,
)


CONDITIONS = ("left", "left_exact_repeat", "right")
FROZEN_GROUNDED_OBSERVATION_SHA256 = (
    "2a431b0fa288890b3509b314c0351c91123d5f64b237678fed972848e29cd55b"
)


def _rms(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first.astype(np.float64) - second.astype(np.float64)))))


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float64) - second.astype(np.float64))))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def evaluate_responses(
    *, pair: AuthorizedPair, runtime: dict[str, Any],
    responses: dict[str, dict[str, Any]], conditioning_image_sha256: str,
) -> dict[str, Any]:
    """Validate three server responses and return a release manifest."""

    if set(responses) != set(CONDITIONS):
        raise ContractError("gate requires exactly left, left_exact_repeat, and right responses")
    if conditioning_image_sha256 != FROZEN_GROUNDED_OBSERVATION_SHA256:
        raise ContractError("gate did not use the exact frozen grounded observation")
    observation_hashes = [response.get("observation_hashes") for response in responses.values()]
    if not all(isinstance(value, dict) for value in observation_hashes):
        raise ContractError("every gate response must retain observation hashes")
    if len({json.dumps(value, sort_keys=True) for value in observation_hashes}) != 1:
        raise ContractError("gate requests did not use byte-identical observations")
    actions: dict[str, np.ndarray] = {}
    futures: dict[str, np.ndarray] = {}
    rows = []
    for condition in CONDITIONS:
        relation = "left" if condition.startswith("left") else "right"
        response = responses[condition]
        if "video" not in response or response["video"] is None:
            raise ContractError(f"{condition}: decoded future is missing")
        action = np.asarray(response.get("action", response.get("actions")), dtype=np.float32)
        future = np.asarray(response["video"], dtype=np.uint8)
        if action.shape != (32, 8):
            raise ContractError(f"{condition}: expected action [32,8], got {action.shape}")
        if future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
            raise ContractError(f"{condition}: expected 33-frame RGB future, got {future.shape}")
        if MODEL_CONTRACTS[pair.model_id]["sampling_seed_echo_required"]:
            if response.get("sampling_seed") != pair.seed:
                raise ContractError(f"{condition}: sampling seed echo mismatch")
        actions[condition] = action
        futures[condition] = future
        rows.append({
            "condition": condition,
            "relation": relation,
            "registered_cell_id": pair.cell(relation)["cell_id"],
            "prompt": pair.cell(relation)["prompt"],
            "sampling_seed": pair.seed,
            "server_sampling_seed": response.get("sampling_seed"),
            "action_shape": list(action.shape),
            "action_array_sha256": _array_sha256(action),
            "future_shape": list(future.shape),
            "future_array_sha256": _array_sha256(future),
            "future_evidence_status": "exposed_and_retained",
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
    return {
        "schema_version": "vla-wam-shared-v3-cosmos-fixed-observation-v1",
        "study_id": pair.left["study_id"],
        "model_id": pair.model_id,
        "status": "passed" if passed else "failed",
        "queue_sha256": pair.queue_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "conditioning_image_sha256": conditioning_image_sha256,
        "observation_hashes": observation_hashes[0],
        "pair_seed_used_for_preflight": pair.seed,
        "conditions": list(CONDITIONS),
        "records": rows,
        "metrics": metrics,
        "future_evidence_policy": (
            "Every exposed future is retained. Missing/latent-only future evidence is "
            "an infrastructure contract failure and is never converted to zero."
        ),
        "claim_boundary": "Repeatability and prompt sensitivity only; not robot success.",
    }


def collect_responses(
    *, pair: AuthorizedPair, image: np.ndarray, joint: np.ndarray, gripper: np.ndarray,
    infer: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Issue the three byte-identical-observation requests."""

    if image.shape != (540, 640, 3) or image.dtype != np.uint8:
        raise ContractError("conditioning image must be uint8 RGB [540,640,3]")
    responses = {}
    observation_hash = {
        "image": _array_sha256(image),
        "joint": _array_sha256(joint),
        "gripper": _array_sha256(gripper),
    }
    for condition in CONDITIONS:
        relation = "left" if condition.startswith("left") else "right"
        request = {
            "observation/image": image.copy(),
            "observation/joint_position": joint.copy(),
            "observation/gripper_position": gripper.copy(),
            "prompt": pair.cell(relation)["prompt"],
            "sampling_seed": pair.seed,
        }
        started = time.perf_counter()
        response = dict(infer(request))
        response["wall_time_s"] = time.perf_counter() - started
        response["observation_hashes"] = observation_hash
        responses[condition] = response
    return responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--seed", type=int, default=8303)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--conditioning-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    if args.port != MODEL_CONTRACTS[args.model_id]["server_port"]:
        parser.error("port differs from the frozen model-specific contract")
    pair = load_authorized_pair(args.study_root, args.model_id, args.seed)
    runtime = verify_runtime_identity(args.study_root, args.model_id, args.runtime_manifest)

    import cv2
    from openpi_client import websocket_client_policy

    conditioning_sha = sha256_file(args.conditioning_image)
    if conditioning_sha != FROZEN_GROUNDED_OBSERVATION_SHA256:
        raise ContractError("conditioning image is not the exact v2 grounded observation")
    image_bgr = cv2.imread(str(args.conditioning_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.conditioning_image)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    source_plan = json.loads(args.source_plan.read_text())
    source = source_plan["source"]
    joint = np.asarray(source["joint_position"], dtype=np.float32)
    gripper = np.asarray(source["gripper_position"], dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_probe = args.output_dir / ".write_preflight"
    write_probe.write_bytes(b"v3-cosmos-fixed-observation-write-preflight\n")
    write_probe.unlink()
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    responses = collect_responses(
        pair=pair, image=image, joint=joint, gripper=gripper, infer=client.infer,
    )

    for condition, response in responses.items():
        action_path = args.output_dir / f"{condition}_action.npy"
        future_path = args.output_dir / f"{condition}_future.npy"
        np.save(action_path, np.asarray(response.get("action", response.get("actions"))), allow_pickle=False)
        np.save(future_path, np.asarray(response["video"]), allow_pickle=False)
    manifest = evaluate_responses(
        pair=pair, runtime=runtime, responses=responses,
        conditioning_image_sha256=conditioning_sha,
    )
    by_condition = {row["condition"]: row for row in manifest["records"]}
    for condition in CONDITIONS:
        action_path = args.output_dir / f"{condition}_action.npy"
        future_path = args.output_dir / f"{condition}_future.npy"
        by_condition[condition]["retained_action"] = {
            "path": str(action_path), "sha256": sha256_file(action_path),
            "bytes": action_path.stat().st_size,
        }
        by_condition[condition]["retained_future"] = {
            "path": str(future_path), "sha256": sha256_file(future_path),
            "bytes": future_path.stat().st_size,
        }
    manifest.update(
        conditioning_image={"path": str(args.conditioning_image), "sha256": conditioning_sha},
        source_plan={"path": str(args.source_plan), "sha256": sha256_file(args.source_plan)},
    )
    output = args.output_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] != "passed":
        raise SystemExit(20)


if __name__ == "__main__":
    main()
