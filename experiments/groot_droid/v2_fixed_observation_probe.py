#!/usr/bin/env python3
"""Run the required exact-repeat and prompt-sensitivity probe for GR00T N1.7."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from policies.gr00t.client import (
    _MsgSerializer,
    _as_action_chunk,
    _get_action_value,
)


LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_info(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _numeric_info(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numeric_info(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _combined_action(action_dict: dict[str, Any]) -> np.ndarray:
    joints = _as_action_chunk(
        _get_action_value(action_dict, "joint_position"), name="joint_position"
    )
    gripper = _as_action_chunk(
        _get_action_value(action_dict, "gripper_position"), name="gripper_position"
    )
    combined = np.concatenate([joints, gripper], axis=1).astype(np.float32)
    if combined.shape != (40, 8):
        raise ValueError(f"Expected GR00T action shape (40, 8), got {combined.shape}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=5555)
    parser.add_argument("--sampling-seed", type=int, default=8300000)
    args = parser.parse_args()

    fixture_manifest = json.loads(args.fixture_manifest.read_text())
    if fixture_manifest["npz_sha256"] != _sha256(args.fixture):
        raise ValueError("Fixed-observation fixture hash does not match its manifest")
    if fixture_manifest["prompt"] != LEFT:
        raise ValueError("Fixed-observation manifest does not contain the frozen LEFT prompt")

    with np.load(args.fixture, allow_pickle=False) as archive:
        base_request = {key: archive[key] for key in archive.files}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 600_000)
    socket.setsockopt(zmq.SNDTIMEO, 30_000)
    socket.connect(f"tcp://{args.remote_host}:{args.remote_port}")
    records: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    try:
        for label, prompt in (("left_a", LEFT), ("left_b", LEFT), ("right", RIGHT)):
            request = dict(base_request)
            request["annotation.language.language_instruction"] = [prompt]
            rpc = {
                "endpoint": "get_action",
                "data": {
                    "observation": request,
                    "options": {"sampling_seed": args.sampling_seed},
                },
            }
            socket.send(_MsgSerializer.to_bytes(rpc))
            response = _MsgSerializer.from_bytes(socket.recv())
            if isinstance(response, dict) and "error" in response:
                raise RuntimeError(f"Server error during {label}: {response['error']}")
            action_dict, info = tuple(response)
            action = _combined_action(action_dict)
            action_path = args.output_dir / f"{label}_action.npy"
            np.save(action_path, action, allow_pickle=False)
            arrays[label] = action
            records[label] = {
                "prompt": prompt,
                "sampling_seed": args.sampling_seed,
                "action_path": str(action_path),
                "action_sha256": _sha256(action_path),
                "shape": list(action.shape),
                "dtype": str(action.dtype),
                "server_info": _numeric_info(info),
            }
    finally:
        socket.close(linger=0)
        context.term()

    repeat_rms = float(np.sqrt(np.mean((arrays["left_a"] - arrays["left_b"]) ** 2)))
    prompt_rms = float(np.sqrt(np.mean((arrays["left_a"] - arrays["right"]) ** 2)))
    repeat_max_abs = float(np.max(np.abs(arrays["left_a"] - arrays["left_b"])))
    prompt_max_abs = float(np.max(np.abs(arrays["left_a"] - arrays["right"])))
    passed = repeat_rms == 0.0 and repeat_max_abs == 0.0 and prompt_rms > 0.0
    manifest = {
        "schema_version": "vla-wam-shared-v2-groot-exact-repeat-probe-v1",
        "fixture_manifest": str(args.fixture_manifest),
        "fixture_sha256": _sha256(args.fixture),
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "sampling_seed": args.sampling_seed,
        "records": records,
        "metrics": {
            "left_exact_repeat_rms": repeat_rms,
            "left_exact_repeat_max_abs": repeat_max_abs,
            "left_vs_right_rms": prompt_rms,
            "left_vs_right_max_abs": prompt_max_abs,
        },
        "contract": {
            "action_shape": [40, 8],
            "repeat_requires_bit_exact": True,
            "prompt_difference_requires_nonzero_rms": True,
        },
        "passed": passed,
    }
    manifest_path = args.output_dir / "exact_repeat_probe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
