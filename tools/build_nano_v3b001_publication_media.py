#!/usr/bin/env python3
"""Build complete actual-versus-local-prediction media for Nano V3-B001.

The builder consumes the final, hash-bound 108-episode aggregate and selects
the same outcome-independent evidence block as the scientific renderer: the
minimum released seed and all four control/position-reflected × LEFT/RIGHT
cells.  Every selected viewport MP4 and every exposed decoded-future NPY must
be supplied locally and must match its aggregate byte count and SHA-256.

For each selected cell, the complete simulator rollout is shown beside every
33-frame local prediction horizon in request order.  Explicit boundary slates
identify request index and action-step start.  The shorter timeline holds its
last frame; neither the actual duration nor a model-future frame is truncated.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Iterator, Mapping, Sequence


try:  # Supports both ``python -m tools...`` and direct script execution.
    from tools import render_nano_v3b001_results as result_renderer
except ImportError:  # pragma: no cover - exercised by the CLI smoke test.
    import render_nano_v3b001_results as result_renderer  # type: ignore[no-redef]


MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-publication-media-v1"
OUTPUT_FPS = 15
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
PANEL_WIDTH = CANVAS_WIDTH // 2
HEADER_HEIGHT = 128
PANEL_TITLE_HEIGHT = 58
CONTENT_HEIGHT = 426
CONTENT_TOP = HEADER_HEIGHT + PANEL_TITLE_HEIGHT
FOOTER_TOP = CONTENT_TOP + CONTENT_HEIGHT
SEPARATOR_FRAMES = 8
FUTURE_FRAME_COUNT = 33
ACTUAL_LABEL = "ACTUAL SIMULATOR ROLLOUT — EXECUTED BEHAVIOR"
PREDICTION_LABEL = "MODEL LOCAL PREDICTION — NOT EXECUTION"
CONTINUITY_NOTICE = (
    "Stitched request-local horizons are not one continuous full-task imagination."
)


class NanoPublicationMediaError(RuntimeError):
    """Raised when source evidence or publication media fails closed."""


def _fail(message: str) -> None:
    raise NanoPublicationMediaError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}")
    return {
        "path": display_path if display_path is not None else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NanoPublicationMediaError(f"cannot serialize media manifest: {exc}") from exc


def _run(
    command: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    binary_output: bool = False,
) -> subprocess.CompletedProcess[Any]:
    completed = subprocess.run(
        list(command),
        input=input_bytes,
        check=False,
        capture_output=True,
        text=input_bytes is None and not binary_output,
    )
    if completed.returncode:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        _fail(f"command failed ({completed.returncode}): {' '.join(command)}\n{str(stderr)[-4000:]}")
    return completed


def _verify_asset(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and path.stat().st_size > 0, f"{label} is missing or empty: {path}")
    _require(type(expected.get("bytes")) is int, f"{label} aggregate byte count is invalid")
    _require(path.stat().st_size == expected["bytes"], f"{label} byte count disagrees with aggregate")
    digest = sha256_file(path)
    _require(digest == expected.get("sha256"), f"{label} SHA-256 disagrees with aggregate")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


@dataclass(frozen=True)
class NpyRgb:
    path: Path
    shape: tuple[int, int, int, int]
    payload_offset: int
    payload_bytes: int


@dataclass(frozen=True)
class FutureInput:
    request_index: int
    action_step_start: int
    expected_record: dict[str, Any]
    local_record: dict[str, Any]
    array: NpyRgb


@dataclass(frozen=True)
class CellInput:
    row: dict[str, Any]
    actual_path: Path
    actual_expected_record: dict[str, Any]
    actual_local_record: dict[str, Any]
    futures: tuple[FutureInput, ...]


def inspect_npy_uint8_rgb(path: Path) -> NpyRgb:
    """Inspect a C-contiguous uint8 ``[33,H,W,3]`` NPY without NumPy."""

    path = Path(path).resolve()
    with path.open("rb") as handle:
        _require(handle.read(6) == b"\x93NUMPY", f"decoded future is not NPY: {path}")
        version = handle.read(2)
        _require(len(version) == 2, f"truncated NPY version: {path}")
        if version[0] == 1:
            raw_length = handle.read(2)
            _require(len(raw_length) == 2, f"truncated NPY header length: {path}")
            header_length = struct.unpack("<H", raw_length)[0]
            encoding = "latin1"
        elif version[0] in {2, 3}:
            raw_length = handle.read(4)
            _require(len(raw_length) == 4, f"truncated NPY header length: {path}")
            header_length = struct.unpack("<I", raw_length)[0]
            encoding = "utf-8" if version[0] == 3 else "latin1"
        else:
            _fail(f"unsupported NPY version {tuple(version)}: {path}")
        header_raw = handle.read(header_length)
        _require(len(header_raw) == header_length, f"truncated NPY header: {path}")
        try:
            header = ast.literal_eval(header_raw.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise NanoPublicationMediaError(f"invalid NPY header: {path}: {exc}") from exc
        _require(isinstance(header, dict), f"NPY header is not a mapping: {path}")
        _require(header.get("descr") in {"|u1", "<u1", ">u1"}, f"NPY is not uint8: {path}")
        _require(header.get("fortran_order") is False, f"Fortran-order NPY is prohibited: {path}")
        shape_value = header.get("shape")
        _require(
            isinstance(shape_value, tuple)
            and len(shape_value) == 4
            and all(type(value) is int and value > 0 for value in shape_value),
            f"invalid decoded-future NPY shape: {path}",
        )
        shape = tuple(shape_value)
        _require(
            shape[0] == FUTURE_FRAME_COUNT and shape[-1] == 3,
            f"decoded future must have shape [33,H,W,3]: {path}: {shape}",
        )
        offset = handle.tell()
    payload = math.prod(shape)
    _require(
        path.stat().st_size == offset + payload,
        f"decoded-future NPY payload length changed: {path}",
    )
    return NpyRgb(path=path, shape=shape, payload_offset=offset, payload_bytes=payload)


def _iter_npy_rgb(array: NpyRgb) -> Iterator[bytes]:
    _, height, width, channels = array.shape
    frame_bytes = height * width * channels
    count = 0
    with array.path.open("rb") as handle:
        handle.seek(array.payload_offset)
        while count < array.shape[0]:
            payload = handle.read(frame_bytes)
            _require(len(payload) == frame_bytes, f"decoded future truncated during read: {array.path}")
            count += 1
            yield payload
        _require(handle.read(1) == b"", f"decoded future gained trailing bytes: {array.path}")
    _require(count == FUTURE_FRAME_COUNT, f"decoded future did not expose all 33 frames: {array.path}")


def collect_inputs(
    *,
    summary_path: Path,
    episodes_path: Path,
    actual_rollout_assets: Mapping[str, Path],
    decoded_prediction_assets: Mapping[tuple[str, int], Path],
) -> tuple[Any, tuple[CellInput, ...]]:
    """Load final evidence and verify the complete deterministic media slice."""

    evidence = result_renderer.load_evidence(summary_path, episodes_path)
    selected_rows = evidence.selected_rows
    _require(len(selected_rows) == 4, "minimum-seed selection did not yield all four cells")
    expected_actual = {row["registered_cell_id"] for row in selected_rows}
    expected_predictions = {
        (row["registered_cell_id"], request["request_index"])
        for row in selected_rows
        for request in row["future_requests"]
    }
    _require(
        set(actual_rollout_assets) == expected_actual,
        "actual-rollout mappings must contain exactly all four selected cells",
    )
    _require(
        set(decoded_prediction_assets) == expected_predictions,
        "decoded-prediction mappings must contain every selected exposed horizon and no extras",
    )

    cells: list[CellInput] = []
    for row in selected_rows:
        cell_id = row["registered_cell_id"]
        expected_actual_record = dict(row["artifacts"]["viewport_video"])
        actual_path = Path(actual_rollout_assets[cell_id]).resolve()
        actual_local = _verify_asset(actual_path, expected_actual_record, f"{cell_id} actual rollout")
        futures: list[FutureInput] = []
        for request in row["future_requests"]:
            request_index = request["request_index"]
            action_step = request.get("action_step_start")
            _require(
                type(action_step) is int and action_step >= 0,
                f"{cell_id} request {request_index} action-step start is invalid",
            )
            key = (cell_id, request_index)
            local_path = Path(decoded_prediction_assets[key]).resolve()
            expected = dict(request["decoded_future"])
            local = _verify_asset(
                local_path,
                expected,
                f"{cell_id} decoded prediction request {request_index}",
            )
            array = inspect_npy_uint8_rgb(local_path)
            _require(
                list(array.shape) == request["decoded_future_shape"],
                f"{cell_id} request {request_index} NPY shape disagrees with aggregate",
            )
            futures.append(
                FutureInput(
                    request_index=request_index,
                    action_step_start=action_step,
                    expected_record=expected,
                    local_record=local,
                    array=array,
                )
            )
        _require(futures, f"{cell_id} has no exposed local-prediction horizon")
        _require(
            [item.request_index for item in futures] == list(range(len(futures))),
            f"{cell_id} future request order is not contiguous",
        )
        cells.append(
            CellInput(
                row=row,
                actual_path=actual_path,
                actual_expected_record=expected_actual_record,
                actual_local_record=actual_local,
                futures=tuple(futures),
            )
        )
    return evidence, tuple(cells)


def _parse_time(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _inspect_stream(ffmpeg: Path, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    stderr = completed.stderr
    video_line = next((line for line in stderr.splitlines() if "Video:" in line), None)
    _require(video_line is not None, f"ffmpeg found no video stream: {path}")
    codec = re.search(r"Video:\s*([^\s,]+)", video_line)
    dimensions = re.search(r"\b(\d{2,5})x(\d{2,5})\b", video_line)
    fps = re.search(r"\b(\d+(?:\.\d+)?)\s+fps\b", video_line)
    pixel = re.search(r"\b(yuv[0-9a-z]+|rgb[0-9a-z]+|bgr[0-9a-z]+|gbrp[0-9a-z]+)\b", video_line)
    duration_match = re.search(r"Duration:\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", stderr)
    _require(codec is not None and dimensions is not None and fps is not None, f"cannot parse video stream: {video_line}")
    _require(duration_match is not None, f"video duration is unavailable: {path}")
    return {
        "codec_name": codec.group(1),
        "pixel_format": pixel.group(1) if pixel is not None else None,
        "width": int(dimensions.group(1)),
        "height": int(dimensions.group(2)),
        "fps": float(fps.group(1)),
        "duration_seconds": _parse_time(duration_match.group(1)),
        "has_audio": any("Audio:" in line for line in stderr.splitlines()),
        "inspection_video_line": video_line.strip(),
    }


def _full_decode(ffmpeg: Path, path: Path) -> dict[str, Any]:
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-f",
        "null",
        "-",
    ]
    completed = _run(command)
    progress: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            progress[key] = value
    _require(progress.get("progress") == "end", f"full decode did not reach EOF: {path}")
    frame_text = progress.get("frame")
    _require(frame_text is not None and frame_text.isdigit(), f"full decode frame count is unavailable: {path}")
    frame_count = int(frame_text)
    _require(frame_count > 0, f"video has no decodable frames: {path}")
    decoded_duration = 0.0
    if progress.get("out_time_us", "").isdigit():
        decoded_duration = int(progress["out_time_us"]) / 1_000_000.0
    return {
        "frame_count": frame_count,
        "decoded_duration_seconds": decoded_duration,
        "command": command,
    }


def probe_video(ffmpeg: Path, path: Path) -> dict[str, Any]:
    stream = _inspect_stream(ffmpeg, path)
    decode = _full_decode(ffmpeg, path)
    return {**stream, **decode}


def _mp4_atom_offsets(path: Path) -> dict[str, int]:
    offsets: dict[str, int] = {}
    total = Path(path).stat().st_size
    offset = 0
    with Path(path).open("rb") as handle:
        while offset + 8 <= total:
            handle.seek(offset)
            size_raw = handle.read(4)
            atom_raw = handle.read(4)
            if len(size_raw) != 4 or len(atom_raw) != 4:
                break
            size = struct.unpack(">I", size_raw)[0]
            atom = atom_raw.decode("latin1", "replace")
            header = 8
            if size == 1:
                extended = handle.read(8)
                _require(len(extended) == 8, f"truncated extended MP4 atom: {path}")
                size = struct.unpack(">Q", extended)[0]
                header = 16
            elif size == 0:
                size = total - offset
            _require(size >= header, f"invalid MP4 atom size at {offset}: {path}")
            offsets.setdefault(atom, offset)
            offset += size
    return offsets


def _sample_decodes(
    ffmpeg: Path,
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    _require(frame_count >= 3, f"publication video has fewer than three frames: {path}")
    indices = sorted({0, frame_count // 2, frame_count - 1})
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select={expression}",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    payload = _run(command, binary_output=True).stdout
    _require(isinstance(payload, bytes), "sample decode did not return raw bytes")
    frame_bytes = width * height * 3
    _require(len(payload) == frame_bytes * len(indices), f"sample decode frame count changed: {path}")
    return [
        {
            "frame_index": index,
            "decoded_rgb_sha256": hashlib.sha256(
                payload[position * frame_bytes : (position + 1) * frame_bytes]
            ).hexdigest(),
        }
        for position, index in enumerate(indices)
    ]


def validate_publication_video(
    ffmpeg: Path,
    path: Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    probe = probe_video(ffmpeg, path)
    _require(probe["codec_name"] == "h264", f"publication codec is not H.264: {path}")
    _require(probe["pixel_format"] == "yuv420p", f"publication pixel format is not yuv420p: {path}")
    _require(
        (probe["width"], probe["height"]) == (CANVAS_WIDTH, CANVAS_HEIGHT),
        f"publication dimensions changed: {path}",
    )
    _require(math.isclose(probe["fps"], OUTPUT_FPS, abs_tol=0.01), f"publication FPS changed: {path}")
    _require(probe["has_audio"] is False, f"publication video unexpectedly contains audio: {path}")
    _require(probe["frame_count"] == expected_frames, f"publication frame count changed: {path}")
    expected_duration = expected_frames / OUTPUT_FPS
    _require(
        abs(probe["duration_seconds"] - expected_duration) <= 1.0 / OUTPUT_FPS + 0.011,
        f"publication duration changed: {path}",
    )
    atoms = _mp4_atom_offsets(path)
    _require(
        "moov" in atoms and "mdat" in atoms and atoms["moov"] < atoms["mdat"],
        f"publication MP4 is not fast-start: {path}",
    )
    samples = _sample_decodes(
        ffmpeg,
        path,
        frame_count=probe["frame_count"],
        width=probe["width"],
        height=probe["height"],
    )
    return {
        **probe,
        "duration_contract_seconds": expected_duration,
        "faststart_atom_offsets": atoms,
        "decoded_frame_samples": samples,
        "validation_policy": (
            "H.264/yuv420p/no-audio, moov-before-mdat fast-start, full-stream decode, "
            "exact frame count and duration, and first/middle/last RGB frame decode"
        ),
    }


class _Fonts:
    def __init__(self, font_path: Path) -> None:
        try:
            from PIL import ImageFont
        except ImportError as exc:  # pragma: no cover - dependency failure.
            raise NanoPublicationMediaError("Pillow is required for publication labels") from exc
        self._image_font = ImageFont
        self.path = Path(font_path).resolve()
        _require(self.path.is_file(), f"publication font does not exist: {self.path}")
        self.cache: dict[int, Any] = {}

    def get(self, size: int) -> Any:
        if size not in self.cache:
            self.cache[size] = self._image_font.truetype(str(self.path), size=size)
        return self.cache[size]


def _image_modules() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - dependency failure.
        raise NanoPublicationMediaError("Pillow is required for publication media") from exc
    return Image, ImageDraw


def _letterbox(image: Any, width: int, height: int) -> Any:
    Image, _ = _image_modules()
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "#0B1116")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def _draw_banner(image: Any, text: str, *, fonts: _Fonts, fill: str) -> Any:
    _, ImageDraw = _image_modules()
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = fonts.get(17)
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    x = max(12, (output.width - text_width) // 2)
    y = output.height - 42
    draw.rounded_rectangle((x - 10, y - 7, x + text_width + 10, y + 27), 7, fill="#050708")
    draw.text((x, y), text, font=font, fill=fill)
    return output


def _base_canvas(cell: CellInput, fonts: _Fonts) -> Any:
    Image, ImageDraw = _image_modules()
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#F4F0E8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, CANVAS_WIDTH, HEADER_HEIGHT), fill="#15232D")
    draw.rectangle((0, HEADER_HEIGHT, PANEL_WIDTH, CONTENT_TOP), fill="#203A35")
    draw.rectangle((PANEL_WIDTH, HEADER_HEIGHT, CANVAS_WIDTH, CONTENT_TOP), fill="#4D2B63")
    arm_label = cell.row["phase_b_arm"].replace("_", " ").upper()
    outcome = (
        "CORRECT"
        if cell.row["requested_success"]
        else cell.row["failure_taxonomy"].replace("_", " ").upper()
    )
    draw.text(
        (28, 15),
        f"NANO V3-B001 · SEED {cell.row['environment_seed']} · ARM: {arm_label} · OUTCOME: {outcome}",
        font=fonts.get(19),
        fill="#B8DCCF",
    )
    draw.text((28, 51), "EXACT STATIC PROMPT", font=fonts.get(16), fill="#D2D9DE")
    draw.text((225, 45), cell.row["prompt"], font=fonts.get(28), fill="#FFFFFF")
    draw.text(
        (24, HEADER_HEIGHT + 17),
        ACTUAL_LABEL,
        font=fonts.get(20),
        fill="#FFFFFF",
    )
    draw.text(
        (PANEL_WIDTH + 24, HEADER_HEIGHT + 17),
        PREDICTION_LABEL,
        font=fonts.get(20),
        fill="#FFFFFF",
    )
    draw.line((PANEL_WIDTH, HEADER_HEIGHT, PANEL_WIDTH, FOOTER_TOP), fill="#F4F0E8", width=3)
    draw.text(
        (28, FOOTER_TOP + 18),
        "The right panel stitches every exposed 33-frame request-local horizon in request order.",
        font=fonts.get(18),
        fill="#25323A",
    )
    draw.text(
        (28, FOOTER_TOP + 52),
        "Separators show request index and action-step start. " + CONTINUITY_NOTICE,
        font=fonts.get(18),
        fill="#25323A",
    )
    draw.text(
        (28, FOOTER_TOP + 82),
        "The shorter side holds its final frame so neither the complete actual duration nor any future frame is cut.",
        font=fonts.get(15),
        fill="#5C6870",
    )
    return image


def _prediction_separator(cell: CellInput, future: FutureInput, fonts: _Fonts) -> Any:
    Image, ImageDraw = _image_modules()
    image = Image.new("RGB", (PANEL_WIDTH, CONTENT_HEIGHT), "#251431")
    draw = ImageDraw.Draw(image)
    count = len(cell.futures)
    draw.text(
        (38, 74),
        f"REQUEST {future.request_index + 1:03d} / {count:03d}",
        font=fonts.get(34),
        fill="#FFFFFF",
    )
    draw.text(
        (38, 132),
        f"ACTION-STEP START: {future.action_step_start}",
        font=fonts.get(24),
        fill="#D8BEED",
    )
    draw.text((38, 215), "Next: 33 decoded RGB frames", font=fonts.get(25), fill="#C8F27C")
    draw.text(
        (38, 267),
        "Boundary between independent local horizons",
        font=fonts.get(19),
        fill="#F1D79A",
    )
    draw.text(
        (38, 326),
        PREDICTION_LABEL,
        font=fonts.get(19),
        fill="#FFFFFF",
    )
    return image


def _prediction_frames(cell: CellInput, fonts: _Fonts) -> Iterator[tuple[str, Any]]:
    Image, ImageDraw = _image_modules()
    for future in cell.futures:
        separator = _prediction_separator(cell, future, fonts)
        for _ in range(SEPARATOR_FRAMES):
            yield "separator", separator
        _, height, width, _ = future.array.shape
        emitted = 0
        for emitted, payload in enumerate(_iter_npy_rgb(future.array), 1):
            source = Image.frombytes("RGB", (width, height), payload)
            panel = _letterbox(source, PANEL_WIDTH, CONTENT_HEIGHT)
            draw = ImageDraw.Draw(panel)
            badge = (
                f"REQUEST {future.request_index + 1:03d} · ACTION STEP {future.action_step_start} "
                f"· FRAME {emitted:02d}/33"
            )
            draw.rounded_rectangle((14, 14, 532, 48), 7, fill="#080A0CCC")
            draw.text((26, 21), badge, font=fonts.get(15), fill="#FFFFFF")
            yield "future", panel
        _require(emitted == FUTURE_FRAME_COUNT, f"request {future.request_index} lost future frames")


def _decode_actual_to_raw(
    ffmpeg: Path,
    source: Path,
    output: Path,
) -> tuple[dict[str, Any], int, list[str]]:
    source_probe = probe_video(ffmpeg, source)
    filter_graph = (
        f"fps={OUTPUT_FPS}:round=up,"
        f"scale={PANEL_WIDTH}:{CONTENT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={PANEL_WIDTH}:{CONTENT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    command = [
        str(ffmpeg),
        "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filter_graph,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        str(output),
    ]
    _run(command)
    frame_bytes = PANEL_WIDTH * CONTENT_HEIGHT * 3
    _require(output.is_file() and output.stat().st_size % frame_bytes == 0, "actual raw decode is incomplete")
    frames = output.stat().st_size // frame_bytes
    _require(frames > 0, f"actual rollout decoded zero frames: {source}")
    minimum_duration_frames = max(1, math.ceil(source_probe["duration_seconds"] * OUTPUT_FPS - 1e-9))
    _require(
        frames + 1 >= minimum_duration_frames,
        f"actual resampling ended before the complete source duration: {source}",
    )
    return source_probe, int(frames), command


def _encoder_command(ffmpeg: Path, output: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
        "-framerate",
        str(OUTPUT_FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-metadata",
        "creation_time=",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-g",
        str(OUTPUT_FPS * 2),
        "-sc_threshold",
        "0",
        str(output),
    ]


def _compose_cell(
    *,
    ffmpeg: Path,
    cell: CellInput,
    font_path: Path,
    work_directory: Path,
    output: Path,
) -> dict[str, Any]:
    Image, _ = _image_modules()
    fonts = _Fonts(font_path)
    actual_raw = work_directory / "actual.rgb"
    source_probe, actual_frames, decode_command = _decode_actual_to_raw(
        ffmpeg,
        cell.actual_path,
        actual_raw,
    )
    prediction_frames = len(cell.futures) * (SEPARATOR_FRAMES + FUTURE_FRAME_COUNT)
    actual_duration_frames = max(
        actual_frames,
        math.ceil(source_probe["duration_seconds"] * OUTPUT_FPS - 1e-9),
    )
    output_frames = max(actual_duration_frames, prediction_frames)
    _require(output_frames >= 3, "comparison video requires at least three frames")
    base = _base_canvas(cell, fonts)
    prediction_iterator = iter(_prediction_frames(cell, fonts))
    encoder_command = _encoder_command(ffmpeg, output)
    process = subprocess.Popen(
        encoder_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    frame_bytes = PANEL_WIDTH * CONTENT_HEIGHT * 3
    last_actual: Any | None = None
    last_prediction: Any | None = None
    future_written = 0
    separators_written = 0
    actual_read = 0
    try:
        with actual_raw.open("rb") as actual_handle:
            for frame_index in range(output_frames):
                if frame_index < actual_frames:
                    payload = actual_handle.read(frame_bytes)
                    _require(len(payload) == frame_bytes, "actual raw frame stream truncated")
                    last_actual = Image.frombytes("RGB", (PANEL_WIDTH, CONTENT_HEIGHT), payload)
                    actual_read += 1
                _require(last_actual is not None, "actual rollout yielded no first frame")
                actual_panel = last_actual
                if frame_index >= actual_duration_frames:
                    actual_panel = _draw_banner(
                        last_actual,
                        "HELD FINAL ACTUAL FRAME — prediction timeline continues",
                        fonts=fonts,
                        fill="#C8F27C",
                    )

                if frame_index < prediction_frames:
                    kind, last_prediction = next(prediction_iterator)
                    if kind == "future":
                        future_written += 1
                    else:
                        separators_written += 1
                _require(last_prediction is not None, "prediction timeline yielded no first frame")
                prediction_panel = last_prediction
                if frame_index >= prediction_frames:
                    prediction_panel = _draw_banner(
                        last_prediction,
                        "HELD FINAL PREDICTION FRAME — actual rollout continues",
                        fonts=fonts,
                        fill="#F1D79A",
                    )

                canvas = base.copy()
                canvas.paste(actual_panel, (0, CONTENT_TOP))
                canvas.paste(prediction_panel, (PANEL_WIDTH, CONTENT_TOP))
                _require(process.stdin is not None, "ffmpeg encoder stdin is unavailable")
                try:
                    process.stdin.write(canvas.tobytes())
                except BrokenPipeError as exc:
                    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
                    raise NanoPublicationMediaError(f"ffmpeg encoder closed early: {stderr[-4000:]}") from exc
            _require(actual_handle.read(1) == b"", "actual raw decode contains unaccounted frames")
        _require(actual_read == actual_frames, "not every resampled actual frame was consumed")
        _require(
            future_written == len(cell.futures) * FUTURE_FRAME_COUNT,
            "not every exposed model-future frame was written",
        )
        _require(
            separators_written == len(cell.futures) * SEPARATOR_FRAMES,
            "request boundary separators were not written exactly",
        )
        try:
            next(prediction_iterator)
        except StopIteration:
            pass
        else:
            _fail("prediction timeline contains unaccounted frames")
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        returncode = process.wait()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        if process.stderr is not None:
            process.stderr.close()
        _require(returncode == 0, f"ffmpeg encoder failed ({returncode}): {stderr[-4000:]}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        output.unlink(missing_ok=True)
        raise
    _require(output.is_file() and output.stat().st_size > 0, "publication encoder produced no video")
    return {
        "source_actual_probe": source_probe,
        "actual_resampled_frames": actual_frames,
        "actual_duration_frames": actual_duration_frames,
        "request_count": len(cell.futures),
        "future_frames_written": future_written,
        "separator_frames_written": separators_written,
        "prediction_timeline_frames": prediction_frames,
        "output_frame_count": output_frames,
        "actual_held_frames": max(0, output_frames - actual_duration_frames),
        "prediction_held_frames": max(0, output_frames - prediction_frames),
        "output_duration_seconds": output_frames / OUTPUT_FPS,
        "actual_decode_command": decode_command,
        "encoder_command": encoder_command,
    }


def _render_poster(ffmpeg: Path, video: Path, output: Path, frame_index: int) -> list[str]:
    command = [
        str(ffmpeg),
        "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        "1",
        "-compression_level",
        "9",
        "-map_metadata",
        "-1",
        str(output),
    ]
    _run(command)
    _require(output.is_file() and output.stat().st_size > 0, "poster extraction produced no output")
    Image, _ = _image_modules()
    try:
        with Image.open(output) as image:
            _require(image.format == "PNG", "poster is not PNG")
            _require(image.size == (CANVAS_WIDTH, CANVAS_HEIGHT), "poster dimensions changed")
            image.verify()
    except OSError as exc:
        raise NanoPublicationMediaError(f"poster cannot be decoded: {output}: {exc}") from exc
    _run([str(ffmpeg), "-v", "error", "-i", str(output), "-frames:v", "1", "-f", "null", "-"])
    return command


def _normalize_command(command: Sequence[str], *, staging_root: Path) -> list[str]:
    root = str(staging_root.resolve())
    return [str(value).replace(root, "$V3B001_MEDIA_STAGING") for value in command]


def _stem(cell: CellInput) -> str:
    return (
        f"nano_v3b001_seed{cell.row['environment_seed']}_"
        f"{cell.row['phase_b_arm']}_{cell.row['requested_relation']}"
    )


def _ensure_output_target(output_directory: Path) -> Path:
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        _require(output_directory.is_dir(), f"output path is not a directory: {output_directory}")
        _require(not any(output_directory.iterdir()), f"output directory must be empty: {output_directory}")
    else:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
    return output_directory


def _publish_stage(stage: Path, output_directory: Path) -> None:
    if output_directory.exists():
        _require(not any(output_directory.iterdir()), f"output directory became non-empty: {output_directory}")
        for source in sorted(stage.iterdir(), key=lambda item: item.name):
            target = output_directory / source.name
            _require(not target.exists(), f"refusing to overwrite publication media: {target}")
            os.replace(source, target)
        stage.rmdir()
    else:
        os.replace(stage, output_directory)


def build_publication_media(
    *,
    summary_path: Path,
    episodes_path: Path,
    output_directory: Path,
    actual_rollout_assets: Mapping[str, Path],
    decoded_prediction_assets: Mapping[tuple[str, int], Path],
    ffmpeg_path: Path,
    font_file: Path,
) -> dict[str, Any]:
    """Build and validate all four selected comparison videos and posters."""

    evidence, cells = collect_inputs(
        summary_path=summary_path,
        episodes_path=episodes_path,
        actual_rollout_assets=actual_rollout_assets,
        decoded_prediction_assets=decoded_prediction_assets,
    )
    ffmpeg_path = Path(ffmpeg_path).resolve()
    font_file = Path(font_file).resolve()
    _require(ffmpeg_path.is_file() and os.access(ffmpeg_path, os.X_OK), f"ffmpeg is not executable: {ffmpeg_path}")
    _Fonts(font_file)
    version = _run([str(ffmpeg_path), "-version"]).stdout.splitlines()[0]
    _require("ffmpeg version" in version, "supplied encoder is not ffmpeg")
    output_directory = _ensure_output_target(output_directory)

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.media-", dir=output_directory.parent)
    ).resolve()
    stage = staging_root / "deliverable"
    stage.mkdir()
    selected_manifest: list[dict[str, Any]] = []
    try:
        for cell in cells:
            cell_work = staging_root / ("work-" + _stem(cell))
            cell_work.mkdir()
            video = stage / (_stem(cell) + "_actual_vs_local_predictions.mp4")
            poster = stage / (_stem(cell) + "_poster.png")
            composition = _compose_cell(
                ffmpeg=ffmpeg_path,
                cell=cell,
                font_path=font_file,
                work_directory=cell_work,
                output=video,
            )
            validation = validate_publication_video(
                ffmpeg_path,
                video,
                expected_frames=composition["output_frame_count"],
            )
            poster_index = min(SEPARATOR_FRAMES, composition["output_frame_count"] - 1)
            poster_command = _render_poster(ffmpeg_path, video, poster, poster_index)
            row = cell.row
            selected_manifest.append(
                {
                    "registered_cell_id": row["registered_cell_id"],
                    "environment_seed": row["environment_seed"],
                    "arm": row["phase_b_arm"],
                    "requested_relation": row["requested_relation"],
                    "exact_prompt": row["prompt"],
                    "requested_success": row["requested_success"],
                    "failure_taxonomy": row["failure_taxonomy"],
                    "source_actual_rollout": {
                        "aggregate_record": cell.actual_expected_record,
                        "verified_local_asset": cell.actual_local_record,
                    },
                    "source_local_prediction_horizons_in_order": [
                        {
                            "request_index": future.request_index,
                            "action_step_start": future.action_step_start,
                            "shape": list(future.array.shape),
                            "aggregate_record": future.expected_record,
                            "verified_local_asset": future.local_record,
                        }
                        for future in cell.futures
                    ],
                    "timeline": {
                        key: value
                        for key, value in composition.items()
                        if key not in {"actual_decode_command", "encoder_command", "source_actual_probe"}
                    },
                    "source_actual_probe": {
                        key: value
                        for key, value in composition["source_actual_probe"].items()
                        if key != "command"
                    },
                    "publication_video": _file_record(video, display_path=video.name),
                    "poster": {
                        **_file_record(poster, display_path=poster.name),
                        "source_publication_frame_index": poster_index,
                        "pixel_dimensions": [CANVAS_WIDTH, CANVAS_HEIGHT],
                    },
                    "output_validation": {
                        key: value for key, value in validation.items() if key != "command"
                    },
                    "normalized_commands": {
                        "actual_complete_decode_and_resample": _normalize_command(
                            composition["actual_decode_command"], staging_root=staging_root
                        ),
                        "publication_encode": _normalize_command(
                            composition["encoder_command"], staging_root=staging_root
                        ),
                        "poster": _normalize_command(poster_command, staging_root=staging_root),
                    },
                }
            )

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete_minimum_seed_all_four_cells_actual_and_local_predictions",
            "study_id": result_renderer.STUDY_ID,
            "amendment_id": result_renderer.AMENDMENT_ID,
            "model_id": result_renderer.MODEL_ID,
            "arena": result_renderer.ARENA,
            "exact_prompts": result_renderer.PROMPTS,
            "claim_boundary": evidence.summary["claim_boundary"],
            "selection": {
                "rule_id": "lowest_released_seed_all_four_cells_all_exposed_predictions_v1",
                "rule": (
                    "Select the minimum released environment seed and retain all four "
                    "control/position-reflected × LEFT/RIGHT cells irrespective of outcome; "
                    "for every selected cell retain the complete actual rollout and every exposed "
                    "decoded local prediction in request-index order."
                ),
                "selected_seed": evidence.selected_seed,
                "selected_cell_count": len(cells),
                "outcome_used_for_selection": False,
                "request_used_for_selection": False,
                "no_substitution_for_missing_media": True,
                "no_outcome_hiding_audit": {
                    "selected_successes": sum(cell.row["requested_success"] for cell in cells),
                    "selected_failures": sum(not cell.row["requested_success"] for cell in cells),
                    "full_cohort_condition_outcomes": evidence.condition_outcomes,
                    "full_cohort_failure_taxonomy_counts": evidence.taxonomy_counts,
                },
            },
            "media_semantics": {
                "visible_labels": {
                    "actual": ACTUAL_LABEL,
                    "prediction": PREDICTION_LABEL,
                    "continuity_notice": CONTINUITY_NOTICE,
                },
                "left_panel": "Complete actual simulator rollout; executed robot behavior.",
                "right_panel": (
                    "Every exposed 33-frame decoded local model-prediction horizon, stitched in "
                    "request order with explicit request-index/action-step separators; not execution."
                ),
                "continuity_boundary": (
                    "Stitching preserves all request-local evidence for viewing but does not turn "
                    "independent local horizons into one continuous full-task imagination."
                ),
                "padding": (
                    "The shorter side holds its final frame until the longer side ends; no complete "
                    "actual duration or decoded-future frame is truncated."
                ),
            },
            "sources": {
                "summary": _file_record(Path(summary_path)),
                "episode_aggregate": _file_record(Path(episodes_path)),
            },
            "renderer": {
                "tool": _file_record(
                    Path(__file__),
                    display_path="tools/build_nano_v3b001_publication_media.py",
                ),
                "ffmpeg": _file_record(ffmpeg_path),
                "ffmpeg_version": version,
                "font": _file_record(font_file),
                "canvas": {
                    "pixel_dimensions": [CANVAS_WIDTH, CANVAS_HEIGHT],
                    "fps": OUTPUT_FPS,
                    "prediction_separator_frames_per_request": SEPARATOR_FRAMES,
                },
                "encoding": {
                    "codec": "H.264 via libx264",
                    "pixel_format": "yuv420p",
                    "movflags": "+faststart",
                    "audio": "none",
                    "threads": 1,
                    "metadata": "stripped with -map_metadata -1",
                },
            },
            "selected_media": selected_manifest,
        }
        manifest_path = stage / "nano_v3b001_publication_media_manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest))
        shutil.rmtree(cell_work, ignore_errors=True)
        _publish_stage(stage, output_directory)
        shutil.rmtree(staging_root, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return {
        "manifest": output_directory / "nano_v3b001_publication_media_manifest.json",
        "videos": sorted(output_directory.glob("*.mp4")),
        "posters": sorted(output_directory.glob("*.png")),
    }


def _parse_actual_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        _require("=" in value, "--actual-rollout-asset requires CELL_ID=PATH")
        cell_id, raw_path = value.split("=", 1)
        _require(cell_id and raw_path and cell_id not in result, "invalid or duplicate actual asset mapping")
        result[cell_id] = Path(raw_path).expanduser().resolve()
    return result


def _parse_prediction_specs(values: Sequence[str]) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for value in values:
        _require("=" in value, "--decoded-prediction-asset requires CELL_ID@REQUEST_INDEX=PATH")
        raw_key, raw_path = value.split("=", 1)
        _require("@" in raw_key and raw_path, "invalid decoded-prediction asset mapping")
        cell_id, raw_index = raw_key.rsplit("@", 1)
        try:
            request_index = int(raw_index)
        except ValueError as exc:
            raise NanoPublicationMediaError("decoded-prediction request index must be an integer") from exc
        key = (cell_id, request_index)
        _require(cell_id and request_index >= 0 and key not in result, "invalid or duplicate prediction mapping")
        result[key] = Path(raw_path).expanduser().resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--font-file", type=Path, required=True)
    parser.add_argument(
        "--actual-rollout-asset",
        action="append",
        default=[],
        metavar="CELL_ID=PATH",
        help="Required once for each of the four selected complete viewport videos",
    )
    parser.add_argument(
        "--decoded-prediction-asset",
        action="append",
        default=[],
        metavar="CELL_ID@REQUEST_INDEX=PATH",
        help="Required once for every exposed decoded-future NPY in the selected cells",
    )
    args = parser.parse_args()
    outputs = build_publication_media(
        summary_path=args.summary,
        episodes_path=args.episodes,
        output_directory=args.output_directory,
        actual_rollout_assets=_parse_actual_specs(args.actual_rollout_asset),
        decoded_prediction_assets=_parse_prediction_specs(args.decoded_prediction_asset),
        ffmpeg_path=args.ffmpeg,
        font_file=args.font_file,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": _file_record(outputs["manifest"]),
                "videos": [_file_record(path) for path in outputs["videos"]],
                "posters": [_file_record(path) for path in outputs["posters"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
