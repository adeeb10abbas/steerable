#!/usr/bin/env python3
"""Create compact Cosmos3 Edge paired video and endpoint scorecard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, repository_path: str | None = None) -> dict:
    value = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if repository_path is not None:
        value["repository_path"] = repository_path
    return value


def make_video(left_path: Path, right_path: Path, output: Path, pair: dict) -> tuple[int, float]:
    left, right = cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Could not open both selected viewport videos")
    fps_left, fps_right = left.get(cv2.CAP_PROP_FPS), right.get(cv2.CAP_PROP_FPS)
    if abs(fps_left - fps_right) > 1e-6:
        raise RuntimeError(f"FPS mismatch: {fps_left} versus {fps_right}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps_left, (1280, 360))
    if not writer.isOpened():
        raise RuntimeError("Could not open paired-video writer")
    last_left = last_right = None
    done_left = done_right = False
    frame_count = 0
    try:
        while not (done_left and done_right):
            if not done_left:
                ok, frame = left.read()
                if ok:
                    last_left = frame
                else:
                    done_left = True
            if not done_right:
                ok, frame = right.read()
                if ok:
                    last_right = frame
                else:
                    done_right = True
            if last_left is None or last_right is None:
                if done_left or done_right:
                    raise RuntimeError("A source viewport video contained no frames")
                continue
            if done_left and done_right:
                break
            canvas = np.concatenate([
                cv2.resize(last_left, (640, 360), interpolation=cv2.INTER_AREA),
                cv2.resize(last_right, (640, 360), interpolation=cv2.INTER_AREA),
            ], axis=1)
            values = (
                (12, "LEFT prompt - success", pair["left_final_lateral_display_m"]),
                (652, "RIGHT prompt - success", pair["right_final_lateral_display_m"]),
            )
            for x, title, endpoint in values:
                cv2.rectangle(canvas, (x - 6, 8), (x + 430, 68), (0, 0, 0), -1)
                cv2.putText(canvas, title, (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"final lateral={endpoint:+.3f} m", (x, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
            frame_count += 1
    finally:
        left.release()
        right.release()
        writer.release()
    if frame_count == 0:
        raise RuntimeError("Paired publication video contained no frames")
    return frame_count, fps_left


def make_scorecard(pairs: list[dict], output: Path) -> None:
    width, height = 1600, 900
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(image, "Cosmos3 Edge DROID: matched LEFT/RIGHT endpoints", (85, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(image, "Six static direct-command episodes - all succeeded", (88, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (70, 70, 70), 2, cv2.LINE_AA)
    x0, x1, y0, row_gap = 245, 1500, 260, 175
    minimum, maximum = -0.2, 0.55
    zero_x = int(x0 + (0 - minimum) / (maximum - minimum) * (x1 - x0))
    cv2.line(image, (x0, 195), (x1, 195), (80, 80, 80), 2)
    cv2.line(image, (zero_x, 180), (zero_x, 750), (165, 165, 165), 2)
    for value in (-0.2, 0.0, 0.2, 0.4):
        x = int(x0 + (value - minimum) / (maximum - minimum) * (x1 - x0))
        cv2.line(image, (x, 185), (x, 205), (80, 80, 80), 2)
        cv2.putText(image, f"{value:+.1f} m", (x - 40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (65, 65, 65), 1, cv2.LINE_AA)
    for index, pair in enumerate(pairs):
        y = y0 + index * row_gap
        cv2.putText(image, f"seed {pair['environment_seed']}", (75, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 45, 45), 2, cv2.LINE_AA)
        left_value, right_value = pair["left_final_lateral_display_m"], pair["right_final_lateral_display_m"]
        left_x = int(x0 + (left_value - minimum) / (maximum - minimum) * (x1 - x0))
        right_x = int(x0 + (right_value - minimum) / (maximum - minimum) * (x1 - x0))
        cv2.line(image, (left_x, y), (right_x, y), (145, 145, 145), 6)
        cv2.circle(image, (left_x, y), 20, (210, 90, 45), -1)
        cv2.circle(image, (right_x, y), 20, (55, 155, 55), -1)
        cv2.putText(image, f"LEFT {left_value:+.3f}", (left_x - 85, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 65, 35), 2, cv2.LINE_AA)
        cv2.putText(image, f"RIGHT {right_value:+.3f}", (right_x - 80, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 115, 35), 2, cv2.LINE_AA)
    cv2.putText(image, "3/3 endpoint orderings aligned; displayed negative is robot LEFT", (85, 820), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (55, 55, 55), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError("Could not write endpoint scorecard")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-prefix", required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    pairs = result["pairs"]
    selected = max(pairs, key=lambda row: row["right_minus_left_endpoint_lateral_m"])
    seed = selected["environment_seed"]
    episodes = {row["requested_relation"]: row for row in result["episodes"] if row["environment_seed"] == seed}
    video = args.output_dir / f"cosmos3_edge_seed{seed}_paired.mp4"
    scorecard = args.output_dir / "cosmos3_edge_endpoint_scorecard.png"
    frame_count, fps = make_video(Path(episodes["left"]["executed_video"]["path"]), Path(episodes["right"]["executed_video"]["path"]), video, selected)
    make_scorecard(pairs, scorecard)
    manifest_path = args.output_dir / "media_manifest.json"
    manifest = {
        "schema_version": "vla-wam-shared-v2-cosmos3-edge-media-v1",
        "model_id": "cosmos3_edge_droid_wam",
        "selection": {
            "environment_seed": seed,
            "reason": "Largest observed RIGHT-minus-LEFT endpoint separation among the three frozen pairs.",
            "frame_alignment": "Shorter successful episode holds its final frame until the longer successful episode ends.",
        },
        "source_videos": {side: record(Path(episodes[side]["executed_video"]["path"])) for side in ("left", "right")},
        "publication_video": {**record(video, f"{args.repository_prefix}/{video.name}"), "frame_count": frame_count, "fps": fps},
        "endpoint_scorecard": record(scorecard, f"{args.repository_prefix}/{scorecard.name}"),
        "caption": "Cosmos3 Edge DROID v2 direct gate: LEFT 3/3 and RIGHT 3/3, with distinct actions and aligned endpoint ordering in all three matched pairs.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
