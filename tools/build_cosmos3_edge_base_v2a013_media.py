#!/usr/bin/env python3
"""Build bounded side-by-side V2-A013 model-prediction publication media."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME_COUNT = 17
WIDTH = 640
HEIGHT = 480
FPS = 5
HEADER_HEIGHT = 54


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


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    canvas = Image.new("RGB", (WIDTH * 2, HEIGHT + HEADER_HEIGHT), "#111827")
    canvas.paste(Image.fromarray(left), (0, HEADER_HEIGHT))
    canvas.paste(Image.fromarray(right), (WIDTH, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    label_font = font(22)
    warning_font = font(14)
    draw.text((18, 8), "LEFT: cube left of bowl", fill="white", font=label_font)
    draw.text((WIDTH + 18, 8), "RIGHT: cube right of bowl", fill="white", font=label_font)
    warning = "MODEL PREDICTION ONLY | NO SIMULATOR ROLLOUT"
    warning_box = draw.textbbox((0, 0), warning, font=warning_font)
    warning_width = warning_box[2] - warning_box[0]
    draw.text(
        ((WIDTH * 2 - warning_width) // 2, 36),
        warning,
        fill="#fbbf24",
        font=warning_font,
    )
    draw.line((WIDTH, 0, WIDTH, HEIGHT + HEADER_HEIGHT), fill="#f8fafc", width=2)
    return np.asarray(canvas)


def encode(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = WIDTH * 2
        stream.height = HEIGHT + HEADER_HEIGHT
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, required=True)
    args = parser.parse_args()

    left_frames = decode(args.left)
    right_frames = decode(args.right)
    paired = [compose(left, right) for left, right in zip(left_frames, right_frames)]
    encode(args.output, paired)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(paired[len(paired) // 2]).save(args.poster, quality=90, optimize=True)


if __name__ == "__main__":
    main()
