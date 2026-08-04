#!/usr/bin/env python3
"""Build bounded Cosmos3-Super prediction and action-trajectory media."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME_COUNT = 17
ACTION_SHAPE = (16, 10)
WIDTH = 640
HEIGHT = 480
FPS = 5
HEADER_HEIGHT = 60
ACTION_HEIGHT = 180
CANVAS_WIDTH = WIDTH * 2
CANVAS_HEIGHT = HEADER_HEIGHT + HEIGHT + ACTION_HEIGHT


def decode(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"{path}: expected exactly one video stream")
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"{path}: expected {FRAME_COUNT} frames, got {len(frames)}")
    if any(frame.shape != (HEIGHT, WIDTH, 3) for frame in frames):
        raise ValueError(f"{path}: expected every frame to be {WIDTH}x{HEIGHT} RGB")
    return frames


def load_action(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if payload.get("shape") != list(ACTION_SHAPE):
        raise ValueError(f"{path}: expected reported action shape {list(ACTION_SHAPE)}")
    action = np.asarray(payload.get("data"), dtype=np.float32)
    if action.shape != ACTION_SHAPE or not np.isfinite(action).all():
        raise ValueError(f"{path}: expected finite action data with shape {ACTION_SHAPE}")
    return action


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    roots = (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts"))
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def trajectory_panel(draw: ImageDraw.ImageDraw, left: np.ndarray, right: np.ndarray) -> None:
    top = HEADER_HEIGHT + HEIGHT
    draw.rectangle((0, top, CANVAS_WIDTH, CANVAS_HEIGHT), fill="#0f172a")
    draw.text(
        (18, top + 8),
        "RETURNED 16x10 ACTION TRAJECTORIES — LEFT cyan / RIGHT magenta — NOT EXECUTED",
        fill="#f8fafc",
        font=font(17, bold=True),
    )
    plot_width = 235
    plot_height = 52
    x_gap = 14
    y_gap = 12
    x_start = 18
    y_start = top + 43
    for dimension in range(ACTION_SHAPE[1]):
        column = dimension % 5
        row = dimension // 5
        x0 = x_start + column * (plot_width + x_gap)
        y0 = y_start + row * (plot_height + y_gap)
        x1 = x0 + plot_width
        y1 = y0 + plot_height
        draw.rounded_rectangle((x0, y0, x1, y1), radius=5, fill="#111827", outline="#334155")
        draw.text((x0 + 5, y0 + 3), f"d{dimension}", fill="#cbd5e1", font=font(12, bold=True))
        values = np.concatenate((left[:, dimension], right[:, dimension]))
        low = float(values.min())
        high = float(values.max())
        if high == low:
            high = low + 1.0
        pad = 0.05 * (high - low)
        low -= pad
        high += pad
        graph_left = x0 + 28
        graph_right = x1 - 6
        graph_top = y0 + 7
        graph_bottom = y1 - 7
        if low <= 0.0 <= high:
            zero_y = graph_bottom - (0.0 - low) / (high - low) * (graph_bottom - graph_top)
            draw.line((graph_left, zero_y, graph_right, zero_y), fill="#475569", width=1)

        def points(series: np.ndarray) -> list[tuple[float, float]]:
            return [
                (
                    graph_left + index / (ACTION_SHAPE[0] - 1) * (graph_right - graph_left),
                    graph_bottom
                    - (float(value) - low) / (high - low) * (graph_bottom - graph_top),
                )
                for index, value in enumerate(series)
            ]

        draw.line(points(left[:, dimension]), fill="#22d3ee", width=2, joint="curve")
        draw.line(points(right[:, dimension]), fill="#f472b6", width=2, joint="curve")


def compose(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_action: np.ndarray,
    right_action: np.ndarray,
) -> np.ndarray:
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#111827")
    canvas.paste(Image.fromarray(left_frame), (0, HEADER_HEIGHT))
    canvas.paste(Image.fromarray(right_frame), (WIDTH, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 8), "LEFT prediction", fill="#22d3ee", font=font(23, bold=True))
    draw.text((WIDTH + 18, 8), "RIGHT prediction", fill="#f472b6", font=font(23, bold=True))
    warning = "COSMOS3-SUPER IMAGE-ONLY PROBE | GENERATED FUTURES | NO ACTUAL ROLLOUT"
    warning_font = font(14, bold=True)
    warning_box = draw.textbbox((0, 0), warning, font=warning_font)
    warning_width = warning_box[2] - warning_box[0]
    draw.text(
        ((CANVAS_WIDTH - warning_width) // 2, 39),
        warning,
        fill="#fbbf24",
        font=warning_font,
    )
    draw.line((WIDTH, 0, WIDTH, HEADER_HEIGHT + HEIGHT), fill="#f8fafc", width=2)
    trajectory_panel(draw, left_action, right_action)
    return np.asarray(canvas)


def encode(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = CANVAS_WIDTH
        stream.height = CANVAS_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "slow"}
        stream.time_base = Fraction(1, FPS)
        for index, pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-action", type=Path, required=True)
    parser.add_argument("--right-action", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, required=True)
    args = parser.parse_args()

    left_frames = decode(args.left)
    right_frames = decode(args.right)
    left_action = load_action(args.left_action)
    right_action = load_action(args.right_action)
    paired = [
        compose(left, right, left_action, right_action)
        for left, right in zip(left_frames, right_frames, strict=True)
    ]
    encode(args.output, paired)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(paired[len(paired) // 2]).save(args.poster, quality=90, optimize=True)


if __name__ == "__main__":
    main()
