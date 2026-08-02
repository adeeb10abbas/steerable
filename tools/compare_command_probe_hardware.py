#!/usr/bin/env python3
"""Compare two Cosmos command probes run on different physical policy GPUs.

The same frozen observation, prompt list, and request seed isolate inference
hardware sensitivity. This is an exclusion/provenance audit, not a model
steerability metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"Shape mismatch: {left.shape} versus {right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"Shape mismatch: {left.shape} versus {right.shape}")
    return float(np.mean(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu0-probe", type=Path, required=True)
    parser.add_argument("--gpu1-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu0_manifest_path = args.gpu0_probe / "manifest.json"
    gpu1_manifest_path = args.gpu1_probe / "manifest.json"
    gpu0 = _load(gpu0_manifest_path)
    gpu1 = _load(gpu1_manifest_path)
    for name, manifest in (("gpu0", gpu0), ("gpu1", gpu1)):
        if manifest["model"] != "cosmos":
            raise RuntimeError(f"{name} probe is not Cosmos")
        if len(manifest["records"]) != 16:
            raise RuntimeError(f"{name} probe has {len(manifest['records'])}, expected 16")

    invariant_fields = (
        "plan_sha256",
        "conditioning_png_sha256",
        "conditioning_raw_rgb_sha256",
        "sampling_seed",
    )
    invariants = {field: gpu0[field] == gpu1[field] for field in invariant_fields}
    if not all(invariants.values()):
        raise RuntimeError(f"Frozen probe invariant mismatch: {invariants}")
    gpu0_records = {row["condition"]: row for row in gpu0["records"]}
    gpu1_records = {row["condition"]: row for row in gpu1["records"]}
    if set(gpu0_records) != set(gpu1_records):
        raise RuntimeError("Probe condition sets differ")

    rows = []
    for condition in [row["condition"] for row in gpu0["records"]]:
        left_record = gpu0_records[condition]
        right_record = gpu1_records[condition]
        if left_record["prompt"] != right_record["prompt"]:
            raise RuntimeError(f"Prompt mismatch for {condition}")
        if left_record["server_sampling_seed"] != gpu0["sampling_seed"]:
            raise RuntimeError(f"GPU0 server seed mismatch for {condition}")
        if right_record["server_sampling_seed"] != gpu1["sampling_seed"]:
            raise RuntimeError(f"GPU1 server seed mismatch for {condition}")

        gpu0_action_path = args.gpu0_probe / f"{condition}_action.npy"
        gpu1_action_path = args.gpu1_probe / f"{condition}_action.npy"
        gpu0_future_path = args.gpu0_probe / f"{condition}_future.mp4"
        gpu1_future_path = args.gpu1_probe / f"{condition}_future.mp4"
        gpu0_action = np.load(gpu0_action_path)
        gpu1_action = np.load(gpu1_action_path)
        gpu0_video = _read_video(gpu0_future_path)
        gpu1_video = _read_video(gpu1_future_path)
        rows.append(
            {
                "condition": condition,
                "style": left_record["style"],
                "prompt": left_record["prompt"],
                "action_shape": list(gpu0_action.shape),
                "future_shape": list(gpu0_video.shape),
                "action_rms_gpu0_vs_gpu1": _rms(gpu0_action, gpu1_action),
                "action_max_abs_gpu0_vs_gpu1": float(
                    np.max(np.abs(gpu0_action.astype(np.float64) - gpu1_action))
                ),
                "decoded_future_mae_0_255_gpu0_vs_gpu1": _mae(gpu0_video, gpu1_video),
                "decoded_future_rms_0_255_gpu0_vs_gpu1": _rms(gpu0_video, gpu1_video),
                "action_file_hash_match": _sha256(gpu0_action_path) == _sha256(gpu1_action_path),
                "future_file_hash_match": _sha256(gpu0_future_path) == _sha256(gpu1_future_path),
            }
        )

    action_rms = [row["action_rms_gpu0_vs_gpu1"] for row in rows]
    future_mae = [row["decoded_future_mae_0_255_gpu0_vs_gpu1"] for row in rows]
    output = {
        "schema_version": 1,
        "status": "excluded_hardware_sensitivity_audit",
        "purpose": "Quantify physical-policy-GPU sensitivity using the exact same frozen observation, 16 prompts, and request seed. These distances are not steerability evidence.",
        "gpu0_probe": str(args.gpu0_probe.resolve()),
        "gpu1_probe": str(args.gpu1_probe.resolve()),
        "gpu0_manifest_sha256": _sha256(gpu0_manifest_path),
        "gpu1_manifest_sha256": _sha256(gpu1_manifest_path),
        "invariants": invariants,
        "conditions": len(rows),
        "action_hash_matches": sum(row["action_file_hash_match"] for row in rows),
        "future_hash_matches": sum(row["future_file_hash_match"] for row in rows),
        "mean_action_rms_gpu0_vs_gpu1": float(np.mean(action_rms)),
        "median_action_rms_gpu0_vs_gpu1": float(np.median(action_rms)),
        "max_action_rms_gpu0_vs_gpu1": float(np.max(action_rms)),
        "mean_decoded_future_mae_0_255_gpu0_vs_gpu1": float(np.mean(future_mae)),
        "median_decoded_future_mae_0_255_gpu0_vs_gpu1": float(np.median(future_mae)),
        "max_decoded_future_mae_0_255_gpu0_vs_gpu1": float(np.max(future_mae)),
        "rows": rows,
        "claim_boundary": "The comparison measures numerical sensitivity to the physical inference card. The definitive Cosmos probe and all definitive Cosmos rollouts use policy GPU 1; GPU0 probe outputs are retained only as excluded provenance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        f"compared {len(rows)} conditions; action hash matches={output['action_hash_matches']}; "
        f"future hash matches={output['future_hash_matches']}"
    )


if __name__ == "__main__":
    main()
