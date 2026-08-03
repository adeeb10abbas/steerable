#!/usr/bin/env python3
"""Render one selected GR00T LEFT/RIGHT pair as compact publication media."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--left-endpoint-y", type=float, required=True)
    parser.add_argument("--right-endpoint-y", type=float, required=True)
    args = parser.parse_args()

    left = cv2.VideoCapture(str(args.left_video))
    right = cv2.VideoCapture(str(args.right_video))
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Could not open both source videos")
    fps_left = left.get(cv2.CAP_PROP_FPS)
    fps_right = right.get(cv2.CAP_PROP_FPS)
    if abs(fps_left - fps_right) > 1e-6:
        raise ValueError(f"FPS mismatch: {fps_left} versus {fps_right}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_left,
        (1280, 360),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open publication video writer")

    count = 0
    try:
        while True:
            ok_left, frame_left = left.read()
            ok_right, frame_right = right.read()
            if ok_left != ok_right:
                raise ValueError("Source videos have different frame counts")
            if not ok_left:
                break
            frame_left = cv2.resize(frame_left, (640, 360), interpolation=cv2.INTER_AREA)
            frame_right = cv2.resize(
                frame_right, (640, 360), interpolation=cv2.INTER_AREA
            )
            frame = np.concatenate([frame_left, frame_right], axis=1)
            for x, title, endpoint in (
                (12, "LEFT prompt", args.left_endpoint_y),
                (652, "RIGHT prompt", args.right_endpoint_y),
            ):
                cv2.rectangle(frame, (x - 6, 8), (x + 270, 62), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    title,
                    (x, 31),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"endpoint lateral y={endpoint:+.3f} m",
                    (x, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            count += 1
    finally:
        left.release()
        right.release()
        writer.release()
    if count == 0 or not args.output.is_file():
        raise RuntimeError("Publication video contains no frames")

    manifest = {
        "schema_version": "vla-wam-shared-v2-groot-paired-media-v1",
        "model": "nvidia/GR00T-N1.7-DROID",
        "environment_seed": args.seed,
        "selection_reason": "Seed 8301 has the largest observed LEFT-minus-RIGHT endpoint lateral separation among the three frozen pairs.",
        "frame_count": count,
        "fps": fps_left,
        "layout": "LEFT prompt on the left; RIGHT prompt on the right",
        "source_videos": {
            "LEFT": {"path": str(args.left_video), "sha256": _sha256(args.left_video)},
            "RIGHT": {
                "path": str(args.right_video),
                "sha256": _sha256(args.right_video),
            },
        },
        "endpoint_lateral_y": {
            "LEFT": args.left_endpoint_y,
            "RIGHT": args.right_endpoint_y,
        },
        "publication_video": {
            "path": str(args.output),
            "repository_path": args.repository_path,
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
        "caption": "GR00T N1.7 DROID seed 8301: both static direct-command episodes fail, but matched LEFT and RIGHT prompts produce distinct actions and requested-order endpoint separation.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
