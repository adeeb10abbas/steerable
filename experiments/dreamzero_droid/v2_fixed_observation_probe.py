#!/usr/bin/env python3
"""Run the frozen DreamZero exact-repeat and prompt-only sensitivity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import websockets.sync.client

from policies.dreamzero.client import MsgPackNumpy

from v2_robolab_client import LEFT, OFFICIAL_NOISE_SEED, RIGHT


CONDITIONS = (("left_a", LEFT), ("left_b", LEFT), ("right", RIGHT))
EXPECTED_ACTION_SHAPE = (24, 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _connect(uri: str, packer: MsgPackNumpy):
    connection = websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=300,
        ping_interval=60,
        ping_timeout=600,
    )
    metadata = packer.unpack(connection.recv(timeout=300))
    return connection, metadata


def _recv_array(connection: Any, packer: MsgPackNumpy) -> np.ndarray:
    raw = connection.recv(timeout=600)
    if isinstance(raw, str):
        raise RuntimeError(f"DreamZero server error:\n{raw}")
    response = packer.unpack(raw)
    if isinstance(response, dict):
        response = response.get("actions", response)
    action = np.asarray(response, dtype=np.float32)
    if action.shape != EXPECTED_ACTION_SHAPE:
        raise ValueError(
            f"DreamZero returned {action.shape}; expected {EXPECTED_ACTION_SHAPE}"
        )
    if not np.isfinite(action).all():
        raise ValueError("DreamZero returned a non-finite fixed-observation action")
    return action


def _wait_for_manifest(path: Path, timeout_seconds: float = 300.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for DreamZero future manifest: {path}")


def _load_latent(manifest: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if manifest.get("request_count") != 1 or len(manifest.get("requests", [])) != 1:
        raise ValueError("Each fixed-observation session must retain exactly one request")
    record = manifest["requests"][0]
    latent_entry = record["latent_video"]
    latent_path = Path(latent_entry["path"])
    if _sha256(latent_path) != latent_entry["sha256"]:
        raise ValueError(f"Latent future hash mismatch: {latent_path}")
    latent_tensor = torch.load(latent_path, map_location="cpu", weights_only=True)
    latent = latent_tensor.float().numpy()
    if list(latent_tensor.shape) != latent_entry["shape"]:
        raise ValueError(f"Latent future shape mismatch: {latent_path}")
    return latent, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    args = parser.parse_args()
    if args.remote_port == 5000:
        parser.error("V2-A007 prohibits requests to the pre-existing port 5000")

    fixture_manifest = json.loads(args.fixture_manifest.read_text())
    fixture_entry = fixture_manifest.get("fixture", {})
    expected_fixture_hash = fixture_entry.get("sha256")
    if expected_fixture_hash != _sha256(args.fixture):
        raise ValueError("Fixed-observation fixture hash does not match its manifest")
    if fixture_manifest.get("prompt") != LEFT:
        raise ValueError("Fixed-observation manifest does not contain the frozen LEFT prompt")
    if fixture_manifest.get("status") != "passed":
        raise ValueError("Renderer/reset gate did not pass")

    with np.load(args.fixture, allow_pickle=False) as archive:
        base_request = {key: archive[key] for key in archive.files}
    required = {
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/cartesian_position",
        "observation/gripper_position",
    }
    if set(base_request) != required:
        raise ValueError(
            "Fixed-observation keys changed: "
            f"missing={sorted(required - set(base_request))}, "
            f"extra={sorted(set(base_request) - required)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.future_root.glob("episode_*")):
        raise ValueError(
            "Fixed-observation future root must be empty so episode indices are auditable"
        )

    packer = MsgPackNumpy()
    uri = f"ws://{args.remote_host}:{args.remote_port}"
    connection, server_metadata = _connect(uri, packer)
    arrays: dict[str, np.ndarray] = {}
    latents: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, Any]] = {}
    session_id = "dreamzero-v2-a007-fixed-observation"
    try:
        for episode_index, (label, prompt) in enumerate(CONDITIONS):
            request = dict(base_request)
            request.update(
                {
                    "prompt": prompt,
                    "session_id": session_id,
                    "endpoint": "infer",
                }
            )
            connection.send(packer.pack(request))
            action = _recv_array(connection, packer)
            action_path = args.output_dir / f"{label}_official_action.npy"
            np.save(action_path, action, allow_pickle=False)

            connection.send(
                packer.pack(
                    {
                        "endpoint": "reset",
                        "session_ids": [session_id],
                    }
                )
            )
            reset_reply = connection.recv(timeout=600)
            if isinstance(reset_reply, str) and reset_reply.lower().startswith("error"):
                raise RuntimeError(f"DreamZero reset failed: {reset_reply}")

            future_manifest_path = (
                args.future_root
                / f"episode_{episode_index:03d}"
                / "future_manifest.json"
            )
            future_manifest = _wait_for_manifest(future_manifest_path)
            latent, server_record = _load_latent(future_manifest)
            if server_record.get("prompt") != prompt:
                raise ValueError(f"Server retained the wrong prompt for {label}")
            server_action = Path(server_record["official_action"]["path"])
            if _sha256(server_action) != server_record["official_action"]["sha256"]:
                raise ValueError(f"Server-side action hash mismatch for {label}")
            server_action_array = np.load(server_action, allow_pickle=False)
            if not np.array_equal(action, server_action_array):
                raise ValueError(
                    f"Measurement instrumentation changed the returned action for {label}"
                )

            arrays[label] = action
            latents[label] = latent
            records[label] = {
                "prompt": prompt,
                "sampling_seed_label": 8300,
                "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
                "session_id": session_id,
                "action_path": str(action_path),
                "action_sha256": _sha256(action_path),
                "shape": list(action.shape),
                "dtype": str(action.dtype),
                "future_manifest": str(future_manifest_path),
                "future_manifest_sha256": _sha256(future_manifest_path),
                "latent_path": server_record["latent_video"]["path"],
                "latent_sha256": server_record["latent_video"]["sha256"],
                "official_decode_count": len(
                    future_manifest.get("official_reset_decode", [])
                ),
            }
    finally:
        connection.close()

    metrics = {
        "left_exact_repeat_action_array_equal": bool(
            np.array_equal(arrays["left_a"], arrays["left_b"])
        ),
        "left_exact_repeat_action_rms": _rms(arrays["left_a"], arrays["left_b"]),
        "left_exact_repeat_action_max_abs": _max_abs(
            arrays["left_a"], arrays["left_b"]
        ),
        "left_vs_right_action_rms": _rms(arrays["left_a"], arrays["right"]),
        "left_vs_right_action_max_abs": _max_abs(
            arrays["left_a"], arrays["right"]
        ),
        "left_exact_repeat_latent_array_equal": bool(
            np.array_equal(latents["left_a"], latents["left_b"])
        ),
        "left_exact_repeat_latent_rms": _rms(
            latents["left_a"], latents["left_b"]
        ),
        "left_exact_repeat_latent_max_abs": _max_abs(
            latents["left_a"], latents["left_b"]
        ),
        "left_vs_right_latent_rms": _rms(latents["left_a"], latents["right"]),
        "left_vs_right_latent_max_abs": _max_abs(
            latents["left_a"], latents["right"]
        ),
    }
    passed = bool(
        metrics["left_exact_repeat_action_array_equal"]
        and metrics["left_exact_repeat_latent_array_equal"]
        and metrics["left_vs_right_action_rms"] > 0.0
        and metrics["left_vs_right_latent_rms"] > 0.0
    )
    manifest = {
        "schema_version": "vla-wam-shared-v2-dreamzero-exact-repeat-probe-v1",
        "status": "passed" if passed else "failed",
        "fixture_manifest": str(args.fixture_manifest),
        "fixture_sha256": _sha256(args.fixture),
        "future_root": str(args.future_root),
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "server_metadata": server_metadata,
        "sampling_seed_label": 8300,
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "records": records,
        "metrics": metrics,
        "contract": {
            "action_shape": list(EXPECTED_ACTION_SHAPE),
            "clean_session_reset_between_requests": True,
            "repeat_requires_bit_exact_action_and_latent": True,
            "prompt_difference_requires_nonzero_action_and_latent_rms": True,
            "measurement_only_action_preservation_verified": True,
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
