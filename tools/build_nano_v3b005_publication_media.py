#!/usr/bin/env python3
"""Build bounded actual-versus-local-prediction media for Nano V3-B005.

The outcome-independent slice is the lowest registered seed at the two sweep
extremes and center.  Each video preserves the complete simulator rollout and
every exposed 33-frame request-local prediction horizon.  The predictions are
stitched only for inspection; visible labels state that they are independent
local horizons and not one continuous full-task imagination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator

import cv2
import imageio_ffmpeg
import numpy as np

from experiments.v3.cosmos_nano_lateral_sweep.analyze_results import REPORT_SCHEMA
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import LEVELS, SEEDS


OUTPUT_FPS = 15.0
SELECTED_LEVELS = (0, 3, 6)
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
PANEL_WIDTH = CANVAS_WIDTH // 2
HEADER_HEIGHT = 126
PANEL_HEADER_HEIGHT = 56
CONTENT_HEIGHT = 430
FOOTER_HEIGHT = CANVAS_HEIGHT - HEADER_HEIGHT - PANEL_HEADER_HEIGHT - CONTENT_HEIGHT
SEPARATOR_FRAMES = 8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, display_path: str | None = None) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}")
    return {
        "path": display_path or str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_record(expected: dict[str, Any], label: str) -> Path:
    path = Path(expected["path"])
    actual = record(path)
    require(actual["bytes"] == expected.get("bytes"), f"{label} byte count changed")
    require(actual["sha256"] == expected.get("sha256"), f"{label} SHA-256 changed")
    return path


def read_episode(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 1, f"expected one episode row: {path}")
    row = json.loads(lines[0])
    require(row.get("behavioral_result_valid") is True, f"invalid behavioral episode: {path}")
    return row


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        frame,
        (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.full((height, width, 3), (16, 18, 20), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return output


def prediction_frames(row: dict[str, Any]) -> Iterator[tuple[np.ndarray, str]]:
    requests = row["future_requests"]
    require(requests, f"{row['registered_cell_id']} has no retained prediction horizon")
    require(
        [item["request_index"] for item in requests] == list(range(len(requests))),
        f"{row['registered_cell_id']} request indices are not contiguous",
    )
    for request in requests:
        expected = request["decoded_future"]
        path = verify_record(expected, f"{row['registered_cell_id']} request {request['request_index']}")
        future = np.load(path, mmap_mode="r", allow_pickle=False)
        require(
            future.dtype == np.uint8
            and future.ndim == 4
            and future.shape[0] == 33
            and future.shape[-1] == 3,
            f"unexpected decoded-future array: {path}: {future.shape}/{future.dtype}",
        )
        label = (
            f"LOCAL HORIZON {request['request_index'] + 1}/{len(requests)}"
            f"  |  begins at action step {request['action_step_start']}"
        )
        slate = np.full((CONTENT_HEIGHT, PANEL_WIDTH, 3), (53, 27, 72), dtype=np.uint8)
        put_text(slate, label, (30, 118), 0.62, (255, 255, 255), 2)
        put_text(slate, "Next: all 33 decoded prediction frames", (30, 176), 0.58, (145, 238, 202), 2)
        put_text(slate, "Independent request-local future; not execution", (30, 237), 0.52, (224, 211, 236), 1)
        for _ in range(SEPARATOR_FRAMES):
            yield slate, label
        for frame_index, rgb in enumerate(future):
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            panel = letterbox(bgr, PANEL_WIDTH, CONTENT_HEIGHT)
            badge = f"{label}  |  frame {frame_index + 1}/33"
            cv2.rectangle(panel, (10, 10), (PANEL_WIDTH - 10, 46), (8, 10, 12), -1)
            put_text(panel, badge, (20, 35), 0.47, (255, 255, 255), 1)
            yield panel, label


def compose(
    row: dict[str, Any],
    pair: dict[str, Any],
    output: Path,
    poster: Path,
) -> dict[str, Any]:
    viewport_expected = row["artifacts"]["viewport_video"]
    viewport = verify_record(viewport_expected, f"{row['registered_cell_id']} viewport")
    capture = cv2.VideoCapture(str(viewport))
    require(capture.isOpened(), f"could not open viewport: {viewport}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    require(abs(fps - OUTPUT_FPS) < 0.01, f"unexpected viewport FPS: {fps}")
    actual_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    require(actual_count > 0, f"viewport has no declared frames: {viewport}")
    predictions = prediction_frames(row)
    prediction_count = len(row["future_requests"]) * (33 + SEPARATOR_FRAMES)
    frame_count = max(actual_count, prediction_count)

    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + "_intermediate_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    require(writer.isOpened(), f"could not open video writer: {intermediate}")
    latest_actual: np.ndarray | None = None
    latest_prediction: np.ndarray | None = None
    first_canvas: np.ndarray | None = None
    emitted_predictions = 0
    try:
        for index in range(frame_count):
            if index < actual_count:
                ok, source = capture.read()
                require(ok, f"viewport ended before declared frame count: {viewport}")
                latest_actual = letterbox(source, PANEL_WIDTH, CONTENT_HEIGHT)
            if index < prediction_count:
                latest_prediction, _ = next(predictions)
                emitted_predictions += 1
            require(latest_actual is not None and latest_prediction is not None, "media timeline has an empty panel")

            canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (239, 239, 235), dtype=np.uint8)
            cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, HEADER_HEIGHT), (42, 52, 59), -1)
            cv2.rectangle(
                canvas,
                (0, HEADER_HEIGHT),
                (PANEL_WIDTH, HEADER_HEIGHT + PANEL_HEADER_HEIGHT),
                (47, 84, 77),
                -1,
            )
            cv2.rectangle(
                canvas,
                (PANEL_WIDTH, HEADER_HEIGHT),
                (CANVAS_WIDTH, HEADER_HEIGHT + PANEL_HEADER_HEIGHT),
                (94, 50, 122),
                -1,
            )
            outcome = "CORRECT" if row["requested_success"] else row["failure_taxonomy"].replace("_", " ").upper()
            put_text(
                canvas,
                f"COSMOS3 NANO DOSE-RESPONSE  |  seed {row['environment_seed']}  |  level {row['level_index']}  |  {outcome}",
                (24, 32),
                0.62,
                (214, 236, 228),
                2,
            )
            put_text(canvas, "EXACT STATIC PROMPT", (24, 67), 0.43, (198, 205, 210), 1)
            put_text(canvas, row["prompt"], (214, 68), 0.64, (255, 255, 255), 2)
            put_text(
                canvas,
                f"Reference-object lateral position: {row['reference_object_initial_lateral_position_y_m']:+.3f} m",
                (24, 105),
                0.48,
                (207, 214, 218),
                1,
            )
            put_text(canvas, "ACTUAL SIMULATOR ROLLOUT - EXECUTED", (22, HEADER_HEIGHT + 36), 0.54, (255, 255, 255), 2)
            put_text(
                canvas,
                "STITCHED LOCAL PREDICTIONS - NOT EXECUTION",
                (PANEL_WIDTH + 22, HEADER_HEIGHT + 36),
                0.54,
                (255, 255, 255),
                2,
            )
            top = HEADER_HEIGHT + PANEL_HEADER_HEIGHT
            canvas[top : top + CONTENT_HEIGHT, :PANEL_WIDTH] = latest_actual
            canvas[top : top + CONTENT_HEIGHT, PANEL_WIDTH:] = latest_prediction
            footer = top + CONTENT_HEIGHT
            put_text(
                canvas,
                "Right panel preserves every exposed 33-frame request-local horizon in order; it is not one continuous full-task imagination.",
                (24, footer + 30),
                0.45,
                (47, 55, 60),
                1,
            )
            put_text(
                canvas,
                f"Matched endpoint redirection D: {float(pair['endpoint_redirection_D_m']):+.3f} m"
                f"  |  requested-side depth contrast B: {float(pair['requested_side_depth_contrast_B_m']):+.3f} m",
                (24, footer + 60),
                0.47,
                (47, 55, 60),
                1,
            )
            writer.write(canvas)
            if first_canvas is None:
                first_canvas = canvas.copy()
    finally:
        capture.release()
        writer.release()
    require(emitted_predictions == prediction_count, "not every prediction frame was retained")
    require(first_canvas is not None, "publication video has no frames")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    intermediate.unlink()
    require(output.is_file() and output.stat().st_size > 0, "H.264 output was not created")
    require(cv2.imwrite(str(poster), first_canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9]), "poster write failed")
    check = cv2.VideoCapture(str(output))
    require(check.isOpened(), f"could not reopen publication video: {output}")
    decoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    require(decoded_frames == frame_count, f"publication frame-count mismatch: {decoded_frames} != {frame_count}")
    return {
        "source_actual_rollout": viewport_expected,
        "source_prediction_horizons": [request["decoded_future"] for request in row["future_requests"]],
        "actual_frame_count": actual_count,
        "prediction_timeline_frame_count": prediction_count,
        "publication_frame_count": frame_count,
        "duration_seconds": frame_count / OUTPUT_FPS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--behavior-root", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-media-dir", required=True)
    args = parser.parse_args()
    report_payload = json.loads(args.report.read_text(encoding="utf-8"))
    require(report_payload.get("schema_version") == REPORT_SCHEMA, "unexpected report schema")
    require(report_payload.get("population", {}).get("behavioral_episode_count") == 210, "report is not complete")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed = min(SEEDS)
    selected: list[dict[str, Any]] = []
    for level in SELECTED_LEVELS:
        pair_path = args.pair_root / f"seed{seed}_level{level}.json"
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        require(pair.get("seed") == seed and pair.get("level_index") == level, "pair identity mismatch")
        require(
            abs(float(pair["reference_object_initial_lateral_position_y_m"]) - LEVELS[level]) < 1e-12,
            "pair changed the frozen level coordinate",
        )
        for relation in ("left", "right"):
            cell_dir = args.behavior_root / f"v3b005__nano__seed{seed}__level{level}__{relation}" / "attempt01"
            episode_path = cell_dir / "raw_episode.jsonl"
            row = read_episode(episode_path)
            expected_cell = f"v3b005:nano:seed{seed}:level{level}:{relation}"
            require(row.get("registered_cell_id") == expected_cell, "episode identity mismatch")
            stem = f"nano_v3b005_seed{seed}_level{level}_{relation}_actual_vs_local_predictions"
            video = args.output_dir / f"{stem}.mp4"
            poster = args.output_dir / f"{stem}_poster.png"
            timeline = compose(row, pair, video, poster)
            selected.append(
                {
                    "registered_cell_id": expected_cell,
                    "environment_seed": seed,
                    "level_index": level,
                    "reference_object_initial_lateral_position_y_m": LEVELS[level],
                    "requested_relation": relation,
                    "exact_prompt": row["prompt"],
                    "requested_success": row["requested_success"],
                    "failure_taxonomy": row["failure_taxonomy"],
                    "source_episode": record(episode_path),
                    "source_pair": record(pair_path),
                    **timeline,
                    "publication_video": record(
                        video,
                        f"{args.repository_media_dir.rstrip('/')}/{video.name}",
                    ),
                    "poster": record(
                        poster,
                        f"{args.repository_media_dir.rstrip('/')}/{poster.name}",
                    ),
                }
            )

    manifest = {
        "schema_version": "vla-wam-shared-v3b005-nano-dose-response-publication-media-v1",
        "status": "complete_outcome_independent_extreme_center_actual_and_local_predictions",
        "study_id": report_payload["study_id"],
        "amendment_id": report_payload["amendment_id"],
        "model_id": report_payload["model_id"],
        "selection": {
            "rule": "Lowest registered seed at frozen levels 0, 3, and 6; both LEFT and RIGHT cells; outcome not used.",
            "seed": seed,
            "levels": list(SELECTED_LEVELS),
            "selected_cell_count": len(selected),
            "outcome_used_for_selection": False,
            "all_exposed_prediction_frames_retained": True,
        },
        "media_semantics": {
            "left_panel": "Complete executed simulator rollout.",
            "right_panel": "Every exposed 33-frame request-local decoded prediction horizon in request order.",
            "continuity_boundary": "Stitching is for inspection and does not convert local horizons into a continuous full-task imagination.",
            "padding": "The shorter panel holds its final frame; neither the actual rollout nor prediction timeline is truncated.",
            "browser_encoding": "H.264/yuv420p/faststart/no audio",
        },
        "source_report": record(args.report),
        "renderer": record(Path(__file__), "tools/build_nano_v3b005_publication_media.py"),
        "selected_media": selected,
    }
    manifest_path = args.output_dir / "nano_v3b005_publication_media_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "selected_media": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
