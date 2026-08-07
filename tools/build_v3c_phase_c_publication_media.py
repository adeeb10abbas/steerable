#!/usr/bin/env python3
"""Build bounded, prompt-explicit publication media for one V3-C001 seed.

For each of the four preregistered prompt families, the renderer places the
matched LEFT and RIGHT cells side by side.  The actual-rollout video preserves
both complete simulator trajectories.  For Cosmos checkpoints, a separate
companion video preserves every exposed 33-frame request-local prediction
horizon in order.  Prediction videos are explicitly labelled as stitched local
horizons, never as continuous full-task imagination.
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

from tools.compile_v3c_phase_c_results import MODELS, PROMPT_FAMILIES


CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
PANEL_WIDTH = CANVAS_WIDTH // 2
GLOBAL_HEADER_HEIGHT = 52
PROMPT_HEIGHT = 104
PANEL_HEADER_HEIGHT = 46
CONTENT_HEIGHT = 466
PREDICTION_FPS = 15.0
PREDICTION_SEPARATOR_FRAMES = 10
MODEL_LABELS = {
    "groot_n17_droid_vla": "GR00T N1.7 DROID",
    "cosmos3_edge_policy_droid": "COSMOS3 EDGE POLICY DROID",
    "cosmos3_nano_policy_droid": "COSMOS3 NANO POLICY DROID",
}
FAMILY_LABELS = {
    "direct_command": "DIRECT INSTRUCTION",
    "short_command": "SHORTENED INSTRUCTION",
    "goal_as_outcome": "GOAL STATEMENT",
    "desired_plus_negated_opposite": "CONTRASTIVE INSTRUCTION",
}


class MediaError(ValueError):
    """Raised when selected source evidence is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MediaError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, display_path: str | None = None) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}")
    return {
        "path": display_path or str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verified_artifact(record: dict[str, Any], label: str) -> Path:
    require(isinstance(record, dict), f"missing artifact record: {label}")
    path = Path(record.get("path", ""))
    actual = file_record(path)
    require(actual["bytes"] == record.get("bytes"), f"{label} byte count changed: {path}")
    require(actual["sha256"] == record.get("sha256"), f"{label} SHA-256 changed: {path}")
    return path


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def wrap_text(text: str, max_width: int, scale: float, thickness: int = 1) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


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


def _outcome(cell: dict[str, Any]) -> str:
    if cell.get("requested_success") is True:
        return "CORRECT"
    return str(cell.get("failure_taxonomy", "unknown")).replace("_", " ").upper()


def _base_canvas(
    model_id: str,
    family: str,
    seed: int,
    left: dict[str, Any],
    right: dict[str, Any],
    media_kind: str,
) -> np.ndarray:
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (239, 239, 235), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, GLOBAL_HEADER_HEIGHT), (42, 52, 59), -1)
    put_text(
        canvas,
        f"{MODEL_LABELS[model_id]}  |  {FAMILY_LABELS[family]}  |  MATCHED SEED {seed}",
        (22, 34),
        0.58,
        (246, 248, 247),
        2,
    )
    for panel_index, (relation, registered) in enumerate((("LEFT", left), ("RIGHT", right))):
        x0 = panel_index * PANEL_WIDTH
        color = (47, 84, 77) if panel_index == 0 else (82, 68, 50)
        cv2.rectangle(
            canvas,
            (x0, GLOBAL_HEADER_HEIGHT),
            (x0 + PANEL_WIDTH, GLOBAL_HEADER_HEIGHT + PROMPT_HEIGHT),
            (250, 248, 243),
            -1,
        )
        put_text(canvas, f"PROMPT REQUESTS {relation}", (x0 + 18, 78), 0.42, color, 1)
        # OpenCV's built-in font is ASCII-only.  Render the registered prompt
        # byte-for-byte without decorative Unicode quotation marks so the
        # publication frame never substitutes visible replacement glyphs.
        lines = wrap_text(registered["prompt"], PANEL_WIDTH - 36, 0.47, 1)
        require(len(lines) <= 3, f"prompt does not fit publication header: {registered['prompt']}")
        for line_index, line in enumerate(lines):
            put_text(canvas, line, (x0 + 18, 105 + 25 * line_index), 0.47, (27, 34, 37), 1)
        panel_y0 = GLOBAL_HEADER_HEIGHT + PROMPT_HEIGHT
        cv2.rectangle(canvas, (x0, panel_y0), (x0 + PANEL_WIDTH, panel_y0 + PANEL_HEADER_HEIGHT), color, -1)
        label = "ACTUAL EXECUTION" if media_kind == "actual" else "MODEL PREDICTIONS - NOT EXECUTION"
        put_text(canvas, f"{label}  |  {_outcome(registered)}", (x0 + 18, panel_y0 + 30), 0.48, (255, 255, 255), 2)
    return canvas


def _encode(intermediate: Path, output: Path, expected_frames: int, expected_fps: float) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
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
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    intermediate.unlink()
    require(output.is_file() and output.stat().st_size > 0, f"H.264 output was not created: {output}")
    capture = cv2.VideoCapture(str(output))
    require(capture.isOpened(), f"could not reopen publication video: {output}")
    decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded, _ = capture.read()
    capture.release()
    require(decoded and decoded_frames == expected_frames, f"publication frame-count mismatch: {output}")
    require(abs(decoded_fps - expected_fps) < 0.01, f"publication FPS mismatch: {output}")


def _actual_source(cell: dict[str, Any]) -> tuple[Path, cv2.VideoCapture, int, float]:
    path = verified_artifact(cell["artifacts"]["viewport_video"], "viewport video")
    capture = cv2.VideoCapture(str(path))
    require(capture.isOpened(), f"could not open viewport video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    require(count > 0 and fps > 0, f"invalid viewport metadata: {path}")
    return path, capture, count, fps


def build_actual_pair(
    model_id: str,
    family: str,
    seed: int,
    left: dict[str, Any],
    right: dict[str, Any],
    output: Path,
    poster: Path,
) -> dict[str, Any]:
    _, left_capture, left_count, left_fps = _actual_source(left)
    _, right_capture, right_count, right_fps = _actual_source(right)
    require(abs(left_fps - right_fps) < 0.01, "matched viewport FPS differs")
    frame_count = max(left_count, right_count)
    intermediate = output.with_name(output.stem + "_intermediate_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), left_fps, (CANVAS_WIDTH, CANVAS_HEIGHT)
    )
    require(writer.isOpened(), f"could not open video writer: {intermediate}")
    latest: list[np.ndarray | None] = [None, None]
    first_canvas: np.ndarray | None = None
    try:
        for frame_index in range(frame_count):
            for panel_index, (capture, count) in enumerate(
                ((left_capture, left_count), (right_capture, right_count))
            ):
                if frame_index < count:
                    ok, frame = capture.read()
                    require(ok, "viewport ended before its declared frame count")
                    latest[panel_index] = letterbox(frame, PANEL_WIDTH, CONTENT_HEIGHT)
            require(all(frame is not None for frame in latest), "actual timeline began with an empty panel")
            canvas = _base_canvas(model_id, family, seed, left, right, "actual")
            top = GLOBAL_HEADER_HEIGHT + PROMPT_HEIGHT + PANEL_HEADER_HEIGHT
            canvas[top : top + CONTENT_HEIGHT, :PANEL_WIDTH] = latest[0]
            canvas[top : top + CONTENT_HEIGHT, PANEL_WIDTH:] = latest[1]
            footer = top + CONTENT_HEIGHT
            put_text(
                canvas,
                "Both complete simulator rollouts are shown; the shorter trajectory holds its final frame.",
                (22, footer + 31),
                0.43,
                (47, 55, 60),
                1,
            )
            writer.write(canvas)
            if first_canvas is None:
                first_canvas = canvas.copy()
    finally:
        left_capture.release()
        right_capture.release()
        writer.release()
    require(first_canvas is not None, "actual publication video has no frames")
    _encode(intermediate, output, frame_count, left_fps)
    require(cv2.imwrite(str(poster), first_canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9]), "poster write failed")
    return {
        "source_left_viewport": left["artifacts"]["viewport_video"],
        "source_right_viewport": right["artifacts"]["viewport_video"],
        "source_frame_counts": {"left": left_count, "right": right_count},
        "publication_frame_count": frame_count,
        "fps": left_fps,
        "duration_seconds": frame_count / left_fps,
    }


def _prediction_frames(cell: dict[str, Any]) -> tuple[Iterator[np.ndarray], int, dict[str, Any]]:
    trace_record = cell["artifacts"].get("decoded_future_trace")
    trace_path = verified_artifact(trace_record, "decoded future trace")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    requests = trace.get("requests")
    require(isinstance(requests, list) and requests, f"no retained prediction horizons: {trace_path}")
    require(
        [request.get("request_index") for request in requests] == list(range(len(requests))),
        f"prediction request indices are not contiguous: {trace_path}",
    )

    def frames() -> Iterator[np.ndarray]:
        for request in requests:
            future_path = Path(request.get("future_path", ""))
            require(future_path.is_file(), f"missing retained future: {future_path}")
            require(sha256_file(future_path) == request.get("future_sha256"), f"future hash changed: {future_path}")
            future = np.load(future_path, allow_pickle=False, mmap_mode="r")
            require(
                future.dtype == np.uint8
                and future.ndim == 4
                and future.shape[0] == 33
                and future.shape[-1] == 3,
                f"unexpected decoded-future array: {future_path}: {future.shape}/{future.dtype}",
            )
            slate = np.full((CONTENT_HEIGHT, PANEL_WIDTH, 3), (53, 27, 72), dtype=np.uint8)
            put_text(
                slate,
                f"LOCAL HORIZON {request['request_index'] + 1}/{len(requests)}",
                (26, 175),
                0.66,
                (255, 255, 255),
                2,
            )
            put_text(slate, "Next: all 33 decoded frames", (26, 224), 0.54, (145, 238, 202), 2)
            put_text(slate, "Independent request; not a continuous rollout", (26, 271), 0.46, (224, 211, 236), 1)
            for _ in range(PREDICTION_SEPARATOR_FRAMES):
                yield slate
            for frame_index, rgb in enumerate(future):
                panel = letterbox(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR), PANEL_WIDTH, CONTENT_HEIGHT)
                cv2.rectangle(panel, (8, 8), (PANEL_WIDTH - 8, 45), (8, 10, 12), -1)
                put_text(
                    panel,
                    f"LOCAL HORIZON {request['request_index'] + 1}/{len(requests)}  |  FRAME {frame_index + 1}/33",
                    (18, 34),
                    0.44,
                    (255, 255, 255),
                    1,
                )
                yield panel

    count = len(requests) * (33 + PREDICTION_SEPARATOR_FRAMES)
    return frames(), count, trace_record


def build_prediction_pair(
    model_id: str,
    family: str,
    seed: int,
    left: dict[str, Any],
    right: dict[str, Any],
    output: Path,
    poster: Path,
) -> dict[str, Any]:
    left_frames, left_count, left_trace = _prediction_frames(left)
    right_frames, right_count, right_trace = _prediction_frames(right)
    frame_count = max(left_count, right_count)
    intermediate = output.with_name(output.stem + "_intermediate_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        PREDICTION_FPS,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    require(writer.isOpened(), f"could not open video writer: {intermediate}")
    latest: list[np.ndarray | None] = [None, None]
    first_canvas: np.ndarray | None = None
    emitted = [0, 0]
    try:
        for frame_index in range(frame_count):
            if frame_index < left_count:
                latest[0] = next(left_frames)
                emitted[0] += 1
            if frame_index < right_count:
                latest[1] = next(right_frames)
                emitted[1] += 1
            require(all(frame is not None for frame in latest), "prediction timeline began with an empty panel")
            canvas = _base_canvas(model_id, family, seed, left, right, "prediction")
            top = GLOBAL_HEADER_HEIGHT + PROMPT_HEIGHT + PANEL_HEADER_HEIGHT
            canvas[top : top + CONTENT_HEIGHT, :PANEL_WIDTH] = latest[0]
            canvas[top : top + CONTENT_HEIGHT, PANEL_WIDTH:] = latest[1]
            footer = top + CONTENT_HEIGHT
            put_text(
                canvas,
                "Every exposed 33-frame local horizon is retained in request order; stitching does not imply full-task continuity.",
                (22, footer + 31),
                0.40,
                (47, 55, 60),
                1,
            )
            writer.write(canvas)
            if first_canvas is None:
                first_canvas = canvas.copy()
    finally:
        writer.release()
    require(emitted == [left_count, right_count], "not every prediction frame was emitted")
    require(first_canvas is not None, "prediction publication video has no frames")
    _encode(intermediate, output, frame_count, PREDICTION_FPS)
    require(cv2.imwrite(str(poster), first_canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9]), "poster write failed")
    return {
        "source_left_future_trace": left_trace,
        "source_right_future_trace": right_trace,
        "source_timeline_frame_counts": {"left": left_count, "right": right_count},
        "publication_frame_count": frame_count,
        "fps": PREDICTION_FPS,
        "duration_seconds": frame_count / PREDICTION_FPS,
    }


def _registration_rows(path: Path, model_id: str, seed: int) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("model_id") == model_id and row.get("seed") == seed]
    require(len(selected) == 8, "registration must contain the selected model's complete eight-cell seed block")
    by_id = {row["registered_cell_id"]: row for row in selected}
    require(len(by_id) == 8, "registered cell ids are not unique")
    return by_id


def _merge_cells(
    report: dict[str, Any], registration: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in report["cells"]:
        registered = registration.get(cell.get("registered_cell_id"))
        require(registered is not None, f"cell is absent from registration: {cell.get('registered_cell_id')}")
        require(
            cell.get("prompt_family") == registered.get("prompt_family")
            and cell.get("relation") == registered.get("relation"),
            "executed condition differs from registration",
        )
        merged = {**cell, "prompt": registered["prompt"]}
        output[(cell["prompt_family"], cell["relation"])] = merged
    require(
        set(output) == {(family, relation) for family in PROMPT_FAMILIES for relation in ("left", "right")},
        "whole-seed report is not a complete four-phrasing matched block",
    )
    return output


def _publication_record(path: Path, repository_media_dir: str) -> dict[str, Any]:
    return file_record(path, f"{repository_media_dir.rstrip('/')}/{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-seed-report", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-media-dir", required=True)
    parser.add_argument("--seed", type=int, default=8500)
    args = parser.parse_args()

    report = json.loads(args.whole_seed_report.read_text(encoding="utf-8"))
    model_id = report.get("model_id")
    require(model_id in MODELS, f"unsupported Phase-C model: {model_id}")
    require(report.get("passed") is True, "whole-seed report did not pass")
    require(report.get("seed") == args.seed, "whole-seed report seed differs from selection")
    require(report.get("behavioral_episode_count") == 8, "whole-seed report is incomplete")
    require(report.get("infrastructure_episode_count") == 0, "selected seed contains infrastructure episodes")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registration = _registration_rows(args.registration, model_id, args.seed)
    cells = _merge_cells(report, registration)
    media: list[dict[str, Any]] = []
    for family in PROMPT_FAMILIES:
        left = cells[(family, "left")]
        right = cells[(family, "right")]
        base = f"{model_id}_v3c001_seed{args.seed}_{family}"
        actual_video = args.output_dir / f"{base}_matched_actual.mp4"
        actual_poster = args.output_dir / f"{base}_matched_actual_poster.png"
        actual_timeline = build_actual_pair(
            model_id, family, args.seed, left, right, actual_video, actual_poster
        )
        item: dict[str, Any] = {
            "prompt_family": family,
            "selection_rule": "Lowest preregistered seed; both matched directions; outcome not used.",
            "left_registered_cell_id": left["registered_cell_id"],
            "right_registered_cell_id": right["registered_cell_id"],
            "left_exact_prompt": left["prompt"],
            "right_exact_prompt": right["prompt"],
            "left_requested_success": left["requested_success"],
            "right_requested_success": right["requested_success"],
            "left_failure_taxonomy": left["failure_taxonomy"],
            "right_failure_taxonomy": right["failure_taxonomy"],
            "actual_timeline": actual_timeline,
            "actual_video": _publication_record(actual_video, args.repository_media_dir),
            "actual_poster": _publication_record(actual_poster, args.repository_media_dir),
        }
        if MODELS[model_id] == "decoded_future_required":
            prediction_video = args.output_dir / f"{base}_matched_local_predictions.mp4"
            prediction_poster = args.output_dir / f"{base}_matched_local_predictions_poster.png"
            prediction_timeline = build_prediction_pair(
                model_id, family, args.seed, left, right, prediction_video, prediction_poster
            )
            item.update(
                {
                    "prediction_timeline": prediction_timeline,
                    "prediction_video": _publication_record(prediction_video, args.repository_media_dir),
                    "prediction_poster": _publication_record(prediction_poster, args.repository_media_dir),
                }
            )
        media.append(item)

    manifest = {
        "schema_version": "vla-wam-shared-v3c-phase-c-publication-media-v1",
        "status": "complete_bounded_outcome_independent_seed",
        "experiment_id": "V3-C001",
        "model_id": model_id,
        "seed": args.seed,
        "selection": {
            "rule": "Lowest preregistered seed; all four prompt families; both matched directions; outcome not used.",
            "prompt_family_count": 4,
            "selected_cell_count": 8,
            "outcome_used_for_selection": False,
        },
        "media_semantics": {
            "actual": "Complete matched LEFT and RIGHT simulator rollouts; shorter side holds its final frame.",
            "predictions": (
                "For Cosmos only: every exposed 33-frame request-local prediction horizon in request order."
            ),
            "continuity_boundary": (
                "Prediction stitching is for inspection and does not convert local horizons into continuous full-task imagination."
            ),
            "prompts": "Exact UTF-8 episode prompts remain visible above every trajectory.",
            "browser_encoding": "H.264/yuv420p/faststart/no audio",
        },
        "source_whole_seed_report": file_record(args.whole_seed_report),
        "source_registration": file_record(args.registration),
        "renderer": file_record(Path(__file__), "tools/build_v3c_phase_c_publication_media.py"),
        "media": media,
    }
    manifest_path = args.output_dir / f"{model_id}_v3c001_publication_media_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "media_groups": len(media)}, sort_keys=True))


if __name__ == "__main__":
    main()
