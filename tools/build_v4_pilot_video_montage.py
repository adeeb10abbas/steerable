#!/usr/bin/env python3
"""Verify selected V4 pilot videos and render a three-frame review montage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queue(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = str(row["episode_id"])
        if episode_id in rows:
            raise ValueError(f"duplicate queue episode: {episode_id}")
        rows[episode_id] = row
    return rows


def parse_selections(values: list[str]) -> dict[str, list[str]]:
    selections: dict[str, list[str]] = {}
    for value in values:
        lane, separator, attempt = value.partition("=")
        if (
            separator != "="
            or not lane.startswith("g7")
            or not attempt.startswith("attempt")
        ):
            raise ValueError(f"invalid lane selection: {value}")
        attempts = selections.setdefault(lane, [])
        if attempt in attempts:
            raise ValueError(f"duplicate lane-attempt selection: {value}")
        attempts.append(attempt)
    if not selections:
        raise ValueError("at least one lane selection is required")
    return selections


def discover_attempts(
    *,
    raw_root: Path,
    queue: dict[str, dict[str, Any]],
    selections: dict[str, list[str]],
) -> dict[str, Path]:
    attempts: dict[str, Path] = {}
    for complete_path in raw_root.rglob("COMPLETE.json"):
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        lane = next(
            (
                candidate
                for candidate in selections
                if f"/lane-{candidate}/" in str(complete_path)
            ),
            None,
        )
        if (
            lane is None
            or complete.get("attempt_id") not in selections[lane]
        ):
            continue
        episode_id = str(complete.get("episode_id"))
        if episode_id not in queue:
            raise ValueError(f"selected attempt is outside pilot queue: {episode_id}")
        if complete.get("status") != "valid":
            continue
        if episode_id in attempts:
            raise ValueError(f"duplicate selected attempt: {episode_id}")
        attempts[episode_id] = complete_path.parent
    missing = sorted(set(queue) - set(attempts))
    unexpected = sorted(set(attempts) - set(queue))
    if missing or unexpected:
        raise ValueError(
            f"pilot coverage differs: missing={missing}, unexpected={unexpected}"
        )
    return attempts


def render_montage(
    *,
    raw_root: Path,
    queue_path: Path,
    selections: dict[str, list[str]],
    montage_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    queue = load_queue(queue_path)
    attempts = discover_attempts(
        raw_root=raw_root,
        queue=queue,
        selections=selections,
    )
    panels = []
    records = []
    for episode_id in sorted(attempts):
        attempt_path = attempts[episode_id]
        episode = json.loads(
            (attempt_path / "episode.json").read_text(encoding="utf-8")
        )
        video = episode.get("viewport_video")
        if not isinstance(video, dict):
            raise ValueError(f"viewport video record missing: {episode_id}")
        video_path = attempt_path / str(video["video_uri"])
        observed_sha = sha256_file(video_path)
        if observed_sha != video.get("video_sha256"):
            raise ValueError(f"viewport video hash differs: {episode_id}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"viewport video cannot be decoded: {episode_id}")
        observed_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frames = int(video["frame_count"])
        if observed_frames != expected_frames:
            raise ValueError(
                f"viewport frame count differs for {episode_id}: "
                f"{observed_frames} != {expected_frames}"
            )
        sampled = []
        sample_indices = (
            0,
            max(0, (observed_frames - 1) // 2),
            max(0, observed_frames - 1),
        )
        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(
                    f"viewport frame {index} cannot be decoded: {episode_id}"
                )
            frame = cv2.resize(frame, (192, 108))
            sampled.append(frame)
        capture.release()
        panel = np.concatenate(sampled, axis=1)
        row = queue[episode_id]
        label = (
            f"{episode_id.split('-pilot-')[-1][:2]} "
            f"{row['factors']['goal']} {row['factors']['scenario']} "
            f"{episode['failure_label']}"
        )
        cv2.rectangle(panel, (0, 0), (576, 18), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
        records.append(
            {
                "episode_id": episode_id,
                "attempt_id": episode["attempt_id"],
                "goal": row["factors"]["goal"],
                "scenario": row["factors"]["scenario"],
                "end_reason": episode["end_reason"],
                "failure_label": episode["failure_label"],
                "video": {
                    "path": str(video_path),
                    "sha256": observed_sha,
                    "frame_count": observed_frames,
                    "fps": float(video["fps"]),
                    "size_bytes": video_path.stat().st_size,
                    "sampled_frame_indices": list(sample_indices),
                },
            }
        )
    columns = 4
    blank = np.zeros_like(panels[0])
    while len(panels) % columns:
        panels.append(blank)
    montage = np.concatenate(
        [
            np.concatenate(panels[index : index + columns], axis=1)
            for index in range(0, len(panels), columns)
        ],
        axis=0,
    )
    montage_path.parent.mkdir(parents=True, exist_ok=True)
    if montage_path.exists() or inventory_path.exists():
        raise FileExistsError("montage outputs already exist")
    if not cv2.imwrite(str(montage_path), montage):
        raise OSError(f"cannot write montage: {montage_path}")
    scenario_counts: dict[str, int] = {}
    for record in records:
        scenario = record["scenario"]
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    static_count = sum(
        count
        for scenario, count in scenario_counts.items()
        if scenario in {"original_sham", "destination_static"}
    )
    motion_count = scenario_counts.get("move_stop", 0)
    inventory = {
        "schema_version": "v4-pilot-video-montage-inventory-v1",
        "campaign_id": "online_correction_v4",
        "queue": {
            "path": str(queue_path),
            "sha256": sha256_file(queue_path),
            "episode_count": len(queue),
        },
        "selected_attempts_by_lane": selections,
        "valid_episode_count": len(records),
        "static_episode_count": static_count,
        "motion_episode_count": motion_count,
        "scenario_counts": scenario_counts,
        "videos_all_hash_verified_and_decoded": True,
        "montage": {
            "path": str(montage_path),
            "sha256": sha256_file(montage_path),
            "width": int(montage.shape[1]),
            "height": int(montage.shape[0]),
        },
        "records": records,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "release_boundary": (
            "This inventory proves video integrity and creates review material. "
            "A separate visual review and G7 aggregate are still required."
        ),
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    try:
        inventory = render_montage(
            raw_root=args.raw_root.resolve(),
            queue_path=args.queue.resolve(),
            selections=parse_selections(args.select),
            montage_path=args.montage.resolve(),
            inventory_path=args.inventory.resolve(),
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "inventory": str(args.inventory),
                "montage": str(args.montage),
                "video_count": inventory["valid_episode_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
