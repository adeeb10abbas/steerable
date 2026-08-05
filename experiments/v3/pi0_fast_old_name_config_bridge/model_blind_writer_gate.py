#!/usr/bin/env python3
"""Verify the frozen neutral fixture and raw-output writers without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


MODEL_ID = "pi0_fast_old_name_config_v3a002"
FIXTURE_SHA256 = "ce8be012347718a162bf0d92ba2fb71a01c570a3462d72ef2c16a86082131778"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pod", required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite writer-gate output: {args.output_dir}")
    manifest = json.loads(args.fixture_manifest.read_text())
    if manifest.get("fixture_sha256") != FIXTURE_SHA256 or sha256(args.fixture) != FIXTURE_SHA256:
        raise ValueError("frozen V3-A002 fixture hash mismatch")
    neutral = manifest.get("neutral_reset_contract", {})
    if neutral.get("left_predicate_at_reset") is not False or neutral.get("right_predicate_at_reset") is not False:
        raise ValueError("fixture is not neutral under both frozen predicates")

    args.output_dir.mkdir(parents=True)
    with np.load(args.fixture, allow_pickle=False) as payload:
        frame = np.asarray(payload["observation/exterior_image_1_left"], dtype=np.uint8)
    height, width = frame.shape[:2]
    video = args.output_dir / "neutral_reset_viewport_write_test.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 writer did not open")
    try:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for _ in range(10):
            writer.write(bgr)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(video))
    try:
        ok, decoded = capture.read()
        decoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not ok or decoded is None or decoded_count < 1:
        raise RuntimeError("written MP4 failed decode verification")

    action = args.output_dir / "neutral_hold_action.npy"
    np.save(action, np.zeros((1, 8), dtype=np.float32), allow_pickle=False)
    loaded = np.load(action, allow_pickle=False)
    if loaded.shape != (1, 8) or loaded.dtype != np.float32:
        raise RuntimeError("written action failed round-trip verification")

    row = {
        "schema_version": "vla-wam-shared-v3-pi0-fast-old-name-config-model-blind-writer-gate-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "model_id": MODEL_ID,
        "pod": args.pod,
        "model_request_count": 0,
        "environment_seed": 8300,
        "neutral_reset_verified": True,
        "fixture": record(args.fixture),
        "viewport_video": record(video),
        "neutral_hold_action": record(action),
        "round_trip": {
            "video_decode_passed": True,
            "decoded_frame_count": decoded_count,
            "action_load_passed": True,
            "jsonl_write_pending": True,
        },
    }
    jsonl = args.output_dir / "model_blind_writer.jsonl"
    jsonl.write_text(json.dumps(row, sort_keys=True) + "\n")
    decoded_row = json.loads(jsonl.read_text())
    if decoded_row.get("model_request_count") != 0 or decoded_row.get("pod") != args.pod:
        raise RuntimeError("JSONL round-trip failed")

    output = {
        "schema_version": "vla-wam-shared-v3-pi0-fast-old-name-config-model-blind-writer-gate-manifest-v1",
        "passed": True,
        "pod": args.pod,
        "model_request_count": 0,
        "fixture_manifest": record(args.fixture_manifest),
        "viewport_video": record(video),
        "neutral_hold_action": record(action),
        "writer_jsonl": record(jsonl),
        "checks": {
            "neutral_reset_verified": True,
            "video_write_and_decode_passed": True,
            "action_write_and_load_passed": True,
            "jsonl_write_and_parse_passed": True,
        },
    }
    output_path = args.output_dir / "writer_gate_manifest.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
