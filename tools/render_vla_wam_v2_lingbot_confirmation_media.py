#!/usr/bin/env python3
"""Render one compact, full-rollout LingBot-VA paired confirmation clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(workspace)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def result_label(episode: dict[str, Any]) -> str:
    outcome = "SUCCESS" if episode["requested_success"] else "FAILURE"
    return f"{outcome} after {episode['actions_executed']} actions"


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def label_cards(left_text: str, right_text: str, header: Path, footer: Path) -> None:
    header_image = Image.new("RGB", (960, 60), "#172A3A")
    draw = ImageDraw.Draw(header_image)
    draw.rectangle((0, 0, 479, 59), fill="#A85C05")
    draw.rectangle((480, 0, 959, 59), fill="#1D568B")
    font = load_font(22, bold=True)
    for x0, text in [(0, left_text), (480, right_text)]:
        bounds = draw.textbbox((0, 0), text, font=font)
        width = bounds[2] - bounds[0]
        draw.text((x0 + (480 - width) / 2, 17), text, fill="white", font=font)
    header_image.save(header)

    footer_image = Image.new("RGB", (960, 34), "#172A3A")
    footer_draw = ImageDraw.Draw(footer_image)
    footer_font = load_font(17, bold=True)
    text = "FULL ROLLOUTS · TIME NORMALIZED TO EPISODE PROGRESS"
    bounds = footer_draw.textbbox((0, 0), text, font=footer_font)
    footer_draw.text(((960 - (bounds[2] - bounds[0])) / 2, 7), text, fill="white", font=footer_font)
    footer_image.save(footer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--slice",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/directional_confirmation/lingbot_va_pairs03_09_slice.json"),
    )
    parser.add_argument("--pair-id", default="robotwin_pair_03")
    parser.add_argument("--left-video", type=Path)
    parser.add_argument("--right-video", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation"),
    )
    parser.add_argument(
        "--media-index",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/media_index.json"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    slice_path = args.slice if args.slice.is_absolute() else workspace / args.slice
    output_dir = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir
    index_path = args.media_index if args.media_index.is_absolute() else workspace / args.media_index
    payload = json.loads(slice_path.read_text())
    if payload["model_id"] != "lingbot_va_robotwin":
        raise RuntimeError("Renderer requires the LingBot-VA prospective slice")

    selected = [episode for episode in payload["episodes"] if episode["pair_id"] == args.pair_id]
    by_relation = {episode["requested_relation"]: episode for episode in selected}
    if set(by_relation) != {"left", "right"}:
        raise RuntimeError(f"{args.pair_id} does not have one valid LEFT and RIGHT episode")
    left = by_relation["left"]
    right = by_relation["right"]
    if left["physical_initial_state_sha256"] != right["physical_initial_state_sha256"]:
        raise RuntimeError("Selected episodes do not share the same physical initial-state hash")
    response = next(
        row for row in payload["summary"]["paired_endpoint_responses"] if row["pair_id"] == args.pair_id
    )
    if response["first_ten_executed_action_rms"] is None:
        raise RuntimeError("Selected pair lacks the prospective action-trace comparison")

    left_path = args.left_video or Path(left["executed_video"]["path"])
    right_path = args.right_video or Path(right["executed_video"]["path"])
    for path, episode in [(left_path, left), (right_path, right)]:
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != episode["executed_video"]["sha256"]:
            raise RuntimeError(f"Source simulator video hash mismatch: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_number = args.pair_id.removeprefix("robotwin_pair_")
    stem = f"lingbot_va_pair{pair_number}_left_right_normalized_full_rollouts"
    video_path = output_dir / f"{stem}.mp4"
    poster_path = output_dir / f"{stem}_poster.jpg"
    captions_path = output_dir / f"{stem}.vtt"

    # RoboTwin writes one 10-fps simulator frame per executed action. Scale each
    # complete episode independently to a common publication duration so neither
    # a 400-action failure nor an early success is truncated.
    left_pts_scale = args.duration_seconds * 10.0 / left["actions_executed"]
    right_pts_scale = args.duration_seconds * 10.0 / right["actions_executed"]
    left_text = f"ASKED LEFT | {result_label(left)}"
    right_text = f"ASKED RIGHT | {result_label(right)}"
    with tempfile.TemporaryDirectory(prefix="lingbot-paired-labels-") as temporary:
        header_path = Path(temporary) / "header.png"
        footer_path = Path(temporary) / "footer.png"
        label_cards(left_text, right_text, header_path, footer_path)
        filter_graph = (
            f"[0:v]setpts=PTS*{left_pts_scale:.12f},fps=20,scale=480:360:flags=lanczos[l];"
            f"[1:v]setpts=PTS*{right_pts_scale:.12f},fps=20,scale=480:360:flags=lanczos[r];"
            "[l][r]hstack=inputs=2:shortest=1[pair];"
            "[2:v]scale=960:60[header];[3:v]scale=960:34[footer];"
            "[header][pair][footer]vstack=inputs=3:shortest=1[out]"
        )
        run(
            [
                args.ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(left_path),
                "-i",
                str(right_path),
                "-loop",
                "1",
                "-i",
                str(header_path),
                "-loop",
                "1",
                "-i",
                str(footer_path),
                "-filter_complex",
                filter_graph,
                "-map",
                "[out]",
                "-t",
                f"{args.duration_seconds:.3f}",
                "-an",
                "-r",
                "20",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ]
        )
    run(
        [
            args.ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{args.duration_seconds / 2:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(poster_path),
        ]
    )
    captions_path.write_text(
        "WEBVTT\n\n"
        f"00:00:00.000 --> 00:00:{args.duration_seconds:06.3f}\n"
        f"LingBot-VA {args.pair_id}: full simulator rollouts normalized to episode progress. "
        f"LEFT: {result_label(left).lower()}. RIGHT: {result_label(right).lower()}.\n"
    )

    item_id = stem
    item = {
        "id": item_id,
        "model_id": "lingbot_va_robotwin",
        "pair_id": args.pair_id,
        "environment_seed": left["environment_seed"],
        "sampling_seed": left["sampling_seed"],
        "selection_rule": "First prospective LingBot-VA confirmation pair (pair03); selected by pair order, not outcome.",
        "layout": "LEFT command on the left; RIGHT command on the right",
        "temporal_alignment": "Each complete simulator rollout is independently time-normalized to 10 seconds of episode progress; no source frames are intentionally omitted.",
        "left_result": result_label(left).lower().replace(" ", "_"),
        "right_result": result_label(right).lower().replace(" ", "_"),
        "paired_endpoint_response": response["endpoint_response_direction"],
        "first_ten_executed_action_rms": response["first_ten_executed_action_rms"],
        "source_simulator_videos": {
            "left": left["executed_video"],
            "right": right["executed_video"],
        },
        "video": {
            **file_record(video_path, workspace),
            "codec": "h264",
            "width": 960,
            "height": 454,
            "duration_seconds": args.duration_seconds,
            "frames": round(args.duration_seconds * 20),
        },
        "poster": file_record(poster_path, workspace),
        "captions": file_record(captions_path, workspace),
        "accessibility": "Burned-in command, outcome, action count, and time-normalization disclosure; VTT captions repeat the results.",
    }
    index = json.loads(index_path.read_text())
    index["selection_policy"] = (
        "Compact same-seed confirmation pairs selected after each valid model slice. "
        "The LingBot-VA item uses the first prospective pair by frozen pair order, independent of outcome. Raw videos remain on the ali PVC."
    )
    index["items"] = sorted(
        [existing for existing in index["items"] if existing["id"] != item_id] + [item],
        key=lambda existing: existing["id"],
    )
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(item, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
