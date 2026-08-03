from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .constants import DEFAULT_FPS
from .io import write_json


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _write_vtt(path: Path, segments: list[dict[str, Any]], fps: int) -> None:
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = int(segment["start_frame"]) / fps
        end = (int(segment["end_frame"]) + 1) / fps
        label = str(segment["subtask"]).replace("-->", "to")
        lines.extend(
            [
                str(int(segment["segment_index"]) + 1),
                f"{_timestamp(start)} --> {_timestamp(end)}",
                label,
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "steerable-res1-visual-audit/0.1"}
    )
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Downloaded empty video from {url}")
    os.replace(partial, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_visual_audit(
    audit_csv: Path,
    output_dir: Path,
    *,
    fps: int = DEFAULT_FPS,
) -> dict[str, Any]:
    """Download the locked audit videos and render direct-alignment captions."""

    with audit_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No audit rows found in {audit_csv}")

    videos_dir = output_dir / "videos"
    captions_dir = output_dir / "captions"
    expected_video_names = {
        f"episode_{int(row['lerobot_episode_index']):06d}.mp4" for row in rows
    }
    expected_caption_names = {
        f"episode_{int(row['lerobot_episode_index']):06d}.vtt" for row in rows
    }
    # These directories are generated caches owned by this command. Remove only
    # files from an older locked sample; reviewer CSVs live outside this tree.
    for stale in videos_dir.glob("episode_*.mp4"):
        if stale.name not in expected_video_names:
            stale.unlink()
    for stale in captions_dir.glob("episode_*.vtt"):
        if stale.name not in expected_caption_names:
            stale.unlink()

    records: list[dict[str, Any]] = []
    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        episode = int(row["lerobot_episode_index"])
        filename = f"episode_{episode:06d}.mp4"
        video_path = videos_dir / filename
        caption_path = captions_dir / f"episode_{episode:06d}.vtt"
        _download(row["video_url"], video_path)
        segments = json.loads(row["segments"])
        _write_vtt(caption_path, segments, fps)
        relative_video = video_path.relative_to(output_dir)
        relative_caption = caption_path.relative_to(output_dir)
        segment_rows = "".join(
            "<tr>"
            f"<td>{int(segment['segment_index'])}</td>"
            f"<td>{int(segment['start_frame'])}-{int(segment['end_frame'])}</td>"
            f"<td>{html.escape(str(segment['subtask']))}</td>"
            "</tr>"
            for segment in segments
        )
        cards.append(
            f"""
            <article>
              <h2>{index:02d}. Episode {episode} / steering {html.escape(row['steering_trajectory_id'])}</h2>
              <p><strong>Source:</strong> {html.escape(row['source_collection'])}<br>
                 <strong>Task:</strong> {html.escape(row['task_instruction'])}</p>
              <video controls preload="metadata">
                <source src="{html.escape(str(relative_video))}" type="video/mp4">
                <track kind="captions" src="{html.escape(str(relative_caption))}" srclang="en" label="Direct step i" default>
              </video>
              <table><thead><tr><th>Segment</th><th>Frames</th><th>Direct label</th></tr></thead>
              <tbody>{segment_rows}</tbody></table>
            </article>
            """
        )
        records.append(
            {
                "episode": episode,
                "steering_trajectory_id": int(row["steering_trajectory_id"]),
                "video": str(video_path),
                "caption": str(caption_path),
                "bytes": video_path.stat().st_size,
                "sha256": _sha256(video_path),
                "segments": len(segments),
            }
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RES-1 direct frame-alignment audit</title>
  <style>
    body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; font: 16px/1.45 system-ui, sans-serif; color: #17202a; }}
    .warning {{ padding: 1rem; background: #fff4d6; border-left: 5px solid #c98200; }}
    article {{ margin: 2.5rem 0; padding-top: 1rem; border-top: 1px solid #ccd1d1; }}
    video {{ width: min(100%, 720px); background: #111; }}
    table {{ border-collapse: collapse; margin-top: 1rem; width: min(100%, 900px); }}
    th, td {{ border: 1px solid #ccd1d1; padding: .35rem .55rem; text-align: left; }}
    th {{ background: #eef2f3; }}
  </style>
</head>
<body>
  <h1>RES-1 locked 20-video alignment audit</h1>
  <p class="warning"><strong>Human judgment required.</strong> Captions show the candidate mapping
  <code>LeRobot frame i -&gt; annotation step i</code>. The corresponding raw Bridge observation
  index is <code>i+1</code>. Record decisions in <code>../visual_alignment_audit.csv</code>;
  this page does not auto-approve boundaries.</p>
  {''.join(cards)}
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "fps": fps,
        "mapping_under_review": "lerobot_frame_i_to_annotation_step_i",
        "raw_bridge_observation_index": "i+1",
        "audit_csv": str(audit_csv),
        "index_html": str(index_path),
        "video_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "records": records,
    }
    write_json(output_dir / "video_manifest.json", manifest)
    return manifest
