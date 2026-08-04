#!/usr/bin/env python3
"""Build one bounded, hash-bearing paired actual-rollout video for V2-A010."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def local_record(path: Path, repository_path: str) -> dict[str, Any]:
    payload = record(path)
    payload["path"] = repository_path
    return payload


def episode_for(result: dict[str, Any], seed: int, relation: str) -> dict[str, Any]:
    matches = [
        row
        for row in result["episodes"]
        if row["environment_seed"] == seed and row["requested_relation"] == relation
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {seed}/{relation} episode, found {len(matches)}")
    return matches[0]


def render_pair(left_path: Path, right_path: Path, output: Path, left: dict[str, Any], right: dict[str, Any]) -> tuple[int, float]:
    captures = [cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("could not open both selected viewport MP4s")
    fps = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
    if not all(value > 0 for value in fps) or abs(fps[0] - fps[1]) > 1e-6:
        raise RuntimeError(f"invalid or mismatched FPS: {fps}")
    width, height = 640, 360
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps[0], (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {output}")
    frames: list[np.ndarray | None] = [None, None]
    ended = [False, False]
    count = 0
    labels = (("LEFT", left), ("RIGHT", right))
    try:
        while not all(ended):
            for index, capture in enumerate(captures):
                if ended[index]:
                    continue
                ok, frame = capture.read()
                if ok:
                    frames[index] = frame
                else:
                    ended[index] = True
            if any(frame is None for frame in frames):
                if any(ended):
                    raise RuntimeError("selected viewport MP4 had no decodable frames")
                continue
            panel = np.concatenate(
                [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames],
                axis=1,
            )
            for index, (relation, episode) in enumerate(labels):
                x = index * width + 12
                outcome = "SUCCESS" if episode["success"] else "FAILURE"
                endpoint = episode["endpoint_lateral_display_m"]
                cv2.rectangle(panel, (x - 6, 8), (x + 550, 70), (0, 0, 0), -1)
                cv2.putText(panel, f"{relation} static prompt - {outcome}", (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, f"final lateral = {endpoint:+.3f} m", (x, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(panel)
            count += 1
    finally:
        for capture in captures:
            capture.release()
        writer.release()
    if count == 0:
        raise RuntimeError("publication video has zero frames")
    return count, fps[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-media-dir", required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    if (
        result.get("schema_version") != "vla-wam-v2a010-pi05-current-result-v1"
        or result.get("amendment_id") != "V2-A010"
        or result.get("status") != "complete_6_of_6_valid_current_stack_cells"
        or result.get("valid_episode_count") != 6
    ):
        raise ValueError("not a complete V2-A010 result")
    left, right = episode_for(result, args.seed, "left"), episode_for(result, args.seed, "right")
    left_source = Path(left["files"]["viewport_video"]["path"])
    right_source = Path(right["files"]["viewport_video"]["path"])
    for source, expected in ((left_source, left["files"]["viewport_video"]), (right_source, right["files"]["viewport_video"])):
        actual = record(source)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"source viewport provenance mismatch: {source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.output_dir / f"pi05_current_stack_v2a010_seed{args.seed}_paired_actual.mp4"
    poster = args.output_dir / f"pi05_current_stack_v2a010_seed{args.seed}_paired_actual_poster.jpg"
    frames, fps = render_pair(left_source, right_source, video, left, right)
    capture = cv2.VideoCapture(str(video))
    ok, first = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("could not extract paired-video poster")
    if not cv2.imwrite(str(poster), first):
        raise RuntimeError("could not write paired-video poster")

    repo = args.repository_media_dir.rstrip("/")
    media = {
        "schema_version": "vla-wam-v2a010-pi05-current-stack-media-v1",
        "status": "complete_selected_seed8300_actual_rollout",
        "amendment_id": "V2-A010",
        "claim_boundary": "This is a current-stack V2-A010 actual simulator rollout, not recovered historical pi0.5 footage. It is VLA action-policy evidence and has no imagined-future media.",
        "source_result": record(args.result),
        "selection_rule": "Lowest environment seed with a complete valid matched LEFT/RIGHT pair; seed 8300. The shorter RIGHT rollout holds its final frame while LEFT continues.",
        "source_videos": {"left": left["files"]["viewport_video"], "right": right["files"]["viewport_video"]},
        "publication_video": local_record(video, f"{repo}/{video.name}"),
        "poster": local_record(poster, f"{repo}/{poster.name}"),
        "frame_count": frames,
        "fps": fps,
        "gallery_entries": [
            {
                "id": f"pi05_current_stack_v2a010_seed{args.seed}",
                "arena": "droid",
                "arena_label": "DROID / RoboLab",
                "model_label": "π0.5 DROID — current-stack V2-A010",
                "category": "VLA",
                "future_interface": "Actions only; no decoded visual future",
                "evidence_status": "Valid current-stack behavioral pair; both episodes succeeded; selected publication rollout",
                "pair_label": f"seed {args.seed} matched pair",
                "seed": args.seed,
                "video": local_record(video, f"{repo}/{video.name}"),
                "poster": local_record(poster, f"{repo}/{poster.name}"),
                "directions": [
                    {"relation": "LEFT", "prompt": left["prompt"], "outcome": f"{'success' if left['success'] else 'failure'} after {left['executed_action_count']} actions"},
                    {"relation": "RIGHT", "prompt": right["prompt"], "outcome": f"{'success' if right['success'] else 'failure'} after {right['executed_action_count']} actions"},
                ],
                "selection_note": "Lowest-seed valid matched V2-A010 pair. This is actual simulator execution only; it is not historical v1 pi0.5 footage and has no imagined-future counterpart.",
                "source_manifest": f"{repo}/media_manifest.json"
            }
        ]
    }
    manifest = args.output_dir / "media_manifest.json"
    manifest.write_text(json.dumps(media, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest), "video": str(video), "poster": str(poster)}, sort_keys=True))


if __name__ == "__main__":
    main()
