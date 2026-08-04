#!/usr/bin/env python3
"""Publish the preselected Light-WAM pair00 execution video.

This is a post-hoc media operation over hash-locked valid episodes. It does not
run the model, simulator, or scoring code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, published_path: str | None = None) -> dict[str, Any]:
    return {
        "path": published_path or str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def duration(ffmpeg: Path, video: Path) -> float:
    probe = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(video)],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if not match:
        raise RuntimeError(f"ffmpeg did not report a duration for {video}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-result", type=Path, required=True)
    parser.add_argument("--published-result-path", required=True)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-output-path", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.compiled_result.read_text())
    if (
        result.get("schema_version")
        != "vla-wam-shared-v2-light-wam-robotwin-slice-v1"
        or result.get("status") != "complete"
        or result.get("valid_episode_count") != 6
    ):
        raise RuntimeError("Light-WAM compiled-result contract mismatch")

    pair = {
        episode["requested_relation"]: episode
        for episode in result["episodes"]
        if episode["pair_id"] == "robotwin_pair_00"
    }
    if set(pair) != {"left", "right"}:
        raise RuntimeError("Light-WAM pair00 is incomplete")
    if not pair["left"]["requested_success"] or pair["right"]["requested_success"]:
        raise RuntimeError("Preselected Light-WAM outcome pair has changed")

    expected = {item["path"]: item for item in result["evidence_files"]}
    for relation, path in (("left", args.left_video), ("right", args.right_video)):
        source = expected.get(str(path))
        if not source:
            raise RuntimeError(f"Source video is absent from compiled evidence: {path}")
        if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
            raise RuntimeError(f"Source video hash mismatch: {path}")
        if pair[relation]["simulator_video"] != str(path):
            raise RuntimeError(f"Episode video path mismatch for {relation}")

    left_duration = duration(args.ffmpeg, args.left_video)
    right_duration = duration(args.ffmpeg, args.right_video)
    output_duration = max(left_duration, right_duration)
    left_pad = max(0.0, output_duration - left_duration)
    right_pad = max(0.0, output_duration - right_duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(args.left_video), "-i", str(args.right_video),
        "-filter_complex",
        (
            f"[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={left_pad:.3f}[l];"
            f"[1:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={right_pad:.3f}[r];"
            "[l][r]hstack=inputs=2[v]"
        ),
        "-map", "[v]", "-t", f"{output_duration:.3f}", "-r", "10",
        "-c:v", "libx264", "-preset", "medium", "-crf", "25",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    subprocess.run(command, check=True)

    manifest = {
        "schema_version": "vla-wam-shared-v2-light-wam-media-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_selected_pair00",
        "model_id": "light_wam_robotwin",
        "media_kind": "simulator_execution_not_imagined_future",
        "selection_policy": "First frozen pair containing a LEFT success and matched RIGHT failure.",
        "behavioral_denominator_change": 0,
        "source_result": record(args.compiled_result, args.published_result_path),
        "source_videos": {
            "LEFT": record(args.left_video),
            "RIGHT": record(args.right_video),
        },
        "composition": {
            "layout": "LEFT command on the left; matched RIGHT command on the right",
            "shorter_episode_policy": "hold_final_frame",
            "left_duration_seconds": left_duration,
            "right_duration_seconds": right_duration,
            "output_duration_seconds": output_duration,
            "ffmpeg": record(args.ffmpeg),
            "command": command[:-1] + [args.published_output_path],
        },
        "publication_video": record(args.output, args.published_output_path),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
