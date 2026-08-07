#!/usr/bin/env python3
"""Collect the V3-C001 fixed-observation requests from exact live servers.

The collector never opens a simulator and never executes an action.  It sends
the 12 registered requests for one checkpoint against one byte-identical
observation, retaining the returned arrays on the PVC.  Model-specific server
dependencies are imported only by the selected backend so this module remains
unit-testable in the repository environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contract import (
    EXPERIMENT_ID,
    MODEL_CONTRACTS,
    PROMPT_FORMS,
    canonical_json_bytes,
    load_jsonl,
    prompt_sha256,
    sha256_file,
)


class LiveGateError(ValueError):
    """Raised before a malformed live request can reach a model server."""


def _array_record(path: Path, array: np.ndarray) -> dict[str, Any]:
    np.save(path, np.asarray(array), allow_pickle=False)
    retained = np.load(path, allow_pickle=False)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(retained.shape),
        "dtype": str(retained.dtype),
    }


def _selected_requests(path: Path, model_id: str) -> list[dict[str, Any]]:
    selected = [row for row in load_jsonl(path) if row.get("model_id") == model_id]
    if len(selected) != 12:
        raise LiveGateError(f"expected 12 registered requests for {model_id}, found {len(selected)}")
    observed = {
        (row.get("prompt_family"), row.get("condition")) for row in selected
    }
    expected = {
        (form, condition)
        for form in PROMPT_FORMS
        for condition in ("left", "left_exact_repeat", "right")
    }
    if observed != expected:
        raise LiveGateError("request registry does not contain the exact four 3-request probes")
    for row in selected:
        if row.get("experiment_id") != EXPERIMENT_ID or row.get("behavioral_episode") is not False:
            raise LiveGateError("fixed-observation registry identity changed")
        if row.get("prompt_sha256") != prompt_sha256(row.get("prompt", "")):
            raise LiveGateError("fixed-observation prompt bytes changed")
    return selected


def _load_groot_observation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        observation = {key: np.asarray(archive[key]) for key in archive.files}
    expected = {
        "video.exterior_image_1_left": ((1, 1, 180, 320, 3), "uint8"),
        "video.wrist_image_left": ((1, 1, 180, 320, 3), "uint8"),
        "state.eef_9d": ((1, 1, 9), "float32"),
        "state.joint_position": ((1, 1, 7), "float32"),
        "state.gripper_position": ((1, 1, 1), "float32"),
    }
    if set(observation) != set(expected):
        raise LiveGateError("GR00T fixed observation keys changed")
    for key, (shape, dtype) in expected.items():
        if observation[key].shape != shape or str(observation[key].dtype) != dtype:
            raise LiveGateError(f"GR00T fixed observation contract changed for {key}")
    return observation


def _groot_infer(host: str, port: int, observation: dict[str, np.ndarray], prompt: str, seed: int) -> np.ndarray:
    from policies.gr00t.client import (  # type: ignore[import-not-found]
        GR00TDroidJointposClient,
        GR00TPolicyClient,
        _MsgSerializer,
    )

    client = GR00TPolicyClient(host=host, port=port)
    request_observation = {key: value.copy() for key, value in observation.items()}
    request_observation["annotation.language.language_instruction"] = [prompt]
    rpc = {
        "endpoint": "get_action",
        "data": {"observation": request_observation, "options": {"sampling_seed": seed}},
    }
    try:
        client.socket.send(_MsgSerializer.to_bytes(rpc))
        message = client.socket.recv()
    finally:
        client.close()
    response = _MsgSerializer.from_bytes(message)
    if isinstance(response, dict) and "error" in response:
        raise RuntimeError(f"GR00T server error: {response['error']}")
    helper = object.__new__(GR00TDroidJointposClient)
    actions = GR00TDroidJointposClient._unpack_response(helper, tuple(response))
    if actions.shape != (40, 8):
        raise LiveGateError(f"GR00T returned action shape {actions.shape}, expected [40,8]")
    return np.asarray(actions, dtype=np.float32)


def _load_cosmos_observation(image_path: Path, source_plan_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise LiveGateError(f"cannot read Cosmos conditioning image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    source_plan = json.loads(source_plan_path.read_text())
    source = source_plan.get("source", {})
    joint = np.asarray(source.get("joint_position"), dtype=np.float32)
    gripper = np.asarray(source.get("gripper_position"), dtype=np.float32)
    if image.shape != (540, 640, 3) or image.dtype != np.uint8:
        raise LiveGateError("Cosmos conditioning image must be uint8 RGB [540,640,3]")
    if joint.size == 0 or gripper.size == 0:
        raise LiveGateError("Cosmos source plan lacks joint/gripper state")
    return image, joint, gripper


def _cosmos_infer(host: str, port: int, observation: tuple[np.ndarray, np.ndarray, np.ndarray], prompt: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from openpi_client import websocket_client_policy  # type: ignore[import-not-found]

    image, joint, gripper = observation
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    response = client.infer({
        "observation/image": image.copy(),
        "observation/joint_position": joint.copy(),
        "observation/gripper_position": gripper.copy(),
        "prompt": prompt,
        "sampling_seed": seed,
    })
    actions = np.asarray(response.get("action", response.get("actions")), dtype=np.float32)
    future = np.asarray(response.get("video"), dtype=np.uint8)
    if actions.shape != (32, 8):
        raise LiveGateError(f"Cosmos returned action shape {actions.shape}, expected [32,8]")
    if future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
        raise LiveGateError(f"Cosmos returned future shape {future.shape}, expected 33-frame RGB")
    return actions, future


def collect(
    *,
    requests_path: Path,
    model_id: str,
    backend: str,
    host: str,
    port: int,
    sampling_seed: int,
    output_dir: Path,
    observation_path: Path,
    source_plan_path: Path | None = None,
) -> Path:
    requests = _selected_requests(requests_path, model_id)
    if output_dir.exists():
        raise LiveGateError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    probe = output_dir / ".write_preflight"
    probe.write_bytes(b"v3-c001-fixed-observation-write-preflight\n")
    probe.unlink()

    if backend == "groot_zmq":
        if model_id != "groot_n17_droid_vla":
            raise LiveGateError("groot_zmq backend is only valid for GR00T")
        observation: Any = _load_groot_observation(observation_path)
        observation_sha = sha256_file(observation_path)
    elif backend == "cosmos_websocket":
        if model_id not in {"cosmos3_edge_policy_droid", "cosmos3_nano_policy_droid"}:
            raise LiveGateError("cosmos_websocket backend requires Edge or Nano")
        if source_plan_path is None:
            raise LiveGateError("Cosmos collection requires --source-plan")
        observation = _load_cosmos_observation(observation_path, source_plan_path)
        observation_sha = hashlib.sha256(
            b"".join(np.ascontiguousarray(value).tobytes() for value in observation)
        ).hexdigest()
    else:
        raise LiveGateError(f"unknown backend: {backend}")

    response_path = output_dir / "responses.jsonl"
    with response_path.open("x", encoding="utf-8") as stream:
        for request in requests:
            stem = f"{request['prompt_family']}_{request['condition']}"
            if backend == "groot_zmq":
                actions = _groot_infer(host, port, observation, request["prompt"], sampling_seed)
                future = None
            else:
                actions, future = _cosmos_infer(host, port, observation, request["prompt"], sampling_seed)
            row = {
                "schema_version": "vla-wam-shared-v3c-four-phrasings-live-response-v1",
                "experiment_id": EXPERIMENT_ID,
                "model_id": model_id,
                "prompt_family": request["prompt_family"],
                "condition": request["condition"],
                "prompt": request["prompt"],
                "prompt_sha256": request["prompt_sha256"],
                "observation_sha256": observation_sha,
                "sampling_seed": sampling_seed,
                "actions": _array_record(output_dir / f"{stem}_actions.npy", actions),
                "behavioral_episode": False,
                "executed_action_count": 0,
            }
            if future is not None:
                row["decoded_future"] = _array_record(output_dir / f"{stem}_future.npy", future)
            stream.write(canonical_json_bytes(row).decode())
            stream.flush()
    return response_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--backend", choices=("groot_zmq", "cosmos_websocket"), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=8500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path)
    args = parser.parse_args()
    output = collect(
        requests_path=args.requests,
        model_id=args.model_id,
        backend=args.backend,
        host=args.host,
        port=args.port,
        sampling_seed=args.sampling_seed,
        output_dir=args.output_dir,
        observation_path=args.observation,
        source_plan_path=args.source_plan,
    )
    print(json.dumps({
        "status": "collected_not_yet_evaluated",
        "model_id": args.model_id,
        "behavioral_episode_count": 0,
        "responses": {"path": str(output), "sha256": sha256_file(output)},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
