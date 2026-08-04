#!/usr/bin/env python3
"""Build complete, hash-bound V2-A015 CFG publication media.

The renderer deliberately accepts the compiled result and exactly three pair
manifests on the command line.  It never searches a rollout directory.  The
pair manifests identify the raw candidates; the compiled result establishes
which six cells are valid behavioral evidence.

Each arm emits four bounded files: one all-seed actual-execution composite,
one all-seed prediction/imagination composite, and one poster for each.  Raw
viewport videos, NumPy futures, and DreamZero official decodes remain outside
Git.  A compact manifest binds every input and output by SHA-256.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator


SEEDS = (8300, 8301, 8302)
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}


@dataclass(frozen=True)
class ArmContract:
    arm_id: str
    model_id: str
    result_schema: str
    pair_schema: str
    setting_label: str
    prediction_kind: str
    prediction_filename_label: str
    prediction_fps: int
    claim_boundary: str


ARM_CONTRACTS = {
    "dreamzero_action_cfg_s2": ArmContract(
        arm_id="dreamzero_action_cfg_s2",
        model_id="dreamzero_droid_action_cfg",
        result_schema="vla-wam-shared-v2-dreamzero-v2a015-s2-result-v1",
        pair_schema="vla-wam-shared-v2-dreamzero-v2a015-pair-collection-v1",
        setting_label="CFG-style negative-branch action guidance s=2; video CFG=5",
        prediction_kind="MODEL IMAGINATION — NOT EXECUTION",
        prediction_filename_label="imagination",
        prediction_fps=5,
        claim_boundary=(
            "The actual composite contains complete simulator viewport executions. "
            "The imagination composite contains every complete official_reset_decode "
            "listed by each of the six DreamZero behavioral cells. These official "
            "model decodes are not simulator execution, task outcomes, or additional "
            "behavioral episodes. The s=2 arm is derived CFG-style negative-branch "
            "action guidance, not an official DreamZero action-CFG mode."
        ),
    ),
    "cosmos3_nano_g1": ArmContract(
        arm_id="cosmos3_nano_no_cfg_g1",
        model_id="cosmos3_nano_policy_droid",
        result_schema="vla-wam-shared-v2-cosmos3-nano-v2a015-g1-result-v1",
        pair_schema="vla-wam-shared-v2-cosmos3-nano-v2a015-g1-pair-collection-v1",
        setting_label="joint action/video CFG g=1 (CFG blend removed); baseline g=3",
        prediction_kind="MODEL PREDICTION — LOCAL HORIZONS, NOT EXECUTION",
        prediction_filename_label="local_predictions",
        prediction_fps=15,
        claim_boundary=(
            "The actual composite contains complete simulator viewport executions. "
            "The prediction composite contains every retained 33-frame RGB future in "
            "request order. Each request is a local model-prediction horizon; joining "
            "them for review does not make a continuous imagined rollout, simulator "
            "execution, task outcome, or additional behavioral episode."
        ),
    ),
}

PANEL_WIDTH = 640
CONTENT_HEIGHT = 360
HEADER_HEIGHT = 120
PANEL_HEIGHT = CONTENT_HEIGHT + HEADER_HEIGHT
CANVAS_WIDTH = PANEL_WIDTH * 2
SEED_SLATE_SECONDS = 1.2
REQUEST_SLATE_SECONDS = 0.9
FONT_FILE: Path | None = None


@dataclass(frozen=True)
class VerifiedFile:
    path: Path
    bytes: int
    sha256: str

    def record(self, display_path: str | None = None) -> dict[str, Any]:
        return {
            "path": display_path or str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass
class CellEvidence:
    seed: int
    relation: str
    prompt: str
    requested_success: bool
    actions_executed: int
    viewport: VerifiedFile
    prediction_sources: list[VerifiedFile]
    prediction_shapes: list[tuple[int, int, int, int]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_file(path: Path) -> VerifiedFile:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Missing file: {resolved}")
    return VerifiedFile(resolved, resolved.stat().st_size, sha256(resolved))


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Missing JSON evidence: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON evidence: {path}: {exc}") from exc


def resolve_record(
    record: dict[str, Any], base: Path, label: str, *, require_bytes: bool = True
) -> VerifiedFile:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} is not a file record")
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{label} lacks a path")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RuntimeError(f"{label} lacks a SHA-256")
    if require_bytes and (not isinstance(expected_bytes, int) or expected_bytes < 0):
        raise RuntimeError(f"{label} lacks an integer byte count")
    if expected_bytes is not None and (
        not isinstance(expected_bytes, int) or expected_bytes < 0
    ):
        raise RuntimeError(f"{label} has an invalid byte count")
    candidate = Path(raw_path).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    observed = local_file(path)
    if (
        (expected_bytes is not None and observed.bytes != expected_bytes)
        or observed.sha256 != expected_hash
    ):
        raise RuntimeError(f"{label} provenance mismatch: {path}")
    return observed


def same_file(left: VerifiedFile, right: VerifiedFile, label: str) -> None:
    if left != right:
        raise RuntimeError(
            f"{label} does not bind the same path, bytes, and SHA-256: "
            f"{left.path} versus {right.path}"
        )


def _episode_map(result: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 6:
        raise RuntimeError("Compiled result must contain exactly six episodes")
    mapped: dict[tuple[int, str], dict[str, Any]] = {}
    for row in episodes:
        try:
            key = (int(row["environment_seed"]), str(row["requested_relation"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Compiled episode lacks a valid seed/relation") from exc
        if key in mapped:
            raise RuntimeError(f"Duplicate compiled episode: {key}")
        mapped[key] = row
    expected = {(seed, relation) for seed in SEEDS for relation in RELATIONS}
    if set(mapped) != expected:
        raise RuntimeError(f"Compiled result is not the exact V2-A015 grid: {sorted(mapped)}")
    return mapped


def _manifest_map(
    paths: list[Path], contract: ArmContract
) -> tuple[dict[int, tuple[Path, dict[str, Any]]], list[VerifiedFile]]:
    if len(paths) != 3 or len({path.resolve() for path in paths}) != 3:
        raise RuntimeError("Exactly three distinct --pair-manifest inputs are required")
    mapped: dict[int, tuple[Path, dict[str, Any]]] = {}
    records: list[VerifiedFile] = []
    for supplied in paths:
        path = supplied.resolve()
        payload = load_json(path)
        expected = {
            "schema_version": contract.pair_schema,
            "status": "complete_behavioral_pair_candidate",
            "amendment_id": "V2-A015",
            "arm_id": contract.arm_id,
            "model_id": contract.model_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"Pair manifest {path} mismatch for {key}: "
                    f"expected={value!r}, observed={payload.get(key)!r}"
                )
        seed = int(payload.get("environment_seed", -1))
        if seed not in SEEDS or seed in mapped:
            raise RuntimeError(f"Unauthorized or duplicate pair seed: {seed}")
        cells = payload.get("cells")
        if not isinstance(cells, list) or len(cells) != 2:
            raise RuntimeError(f"Pair {seed} must contain exactly LEFT and RIGHT cells")
        observed = {(int(cell.get("environment_seed", -1)), cell.get("requested_relation")) for cell in cells}
        if observed != {(seed, "left"), (seed, "right")}:
            raise RuntimeError(f"Pair {seed} is not an exact LEFT/RIGHT pair")
        for cell in cells:
            relation = cell["requested_relation"]
            if cell.get("prompt") != PROMPTS[relation]:
                raise RuntimeError(f"Pair {seed}/{relation} changed the exact prompt")
        mapped[seed] = (path, payload)
        records.append(local_file(path))
    if set(mapped) != set(SEEDS):
        raise RuntimeError("Pair manifests must be exactly seeds 8300, 8301, and 8302")
    return mapped, records


def _check_result_provenance(
    result: dict[str, Any], result_path: Path, supplied: list[VerifiedFile]
) -> None:
    records = result.get("provenance", {}).get("pair_manifests")
    if not isinstance(records, list) or len(records) != 3:
        raise RuntimeError("Compiled result does not bind exactly three pair manifests")
    compiled = [
        resolve_record(record, result_path.parent, f"compiled pair manifest {index}")
        for index, record in enumerate(records)
    ]
    compiled_keys = {(item.path, item.bytes, item.sha256) for item in compiled}
    supplied_keys = {(item.path, item.bytes, item.sha256) for item in supplied}
    if compiled_keys != supplied_keys:
        raise RuntimeError("Supplied pair manifests differ from compiled-result provenance")


def inspect_npy_uint8_rgb(path: Path) -> tuple[tuple[int, int, int, int], int, int]:
    """Return (shape, data offset, payload bytes) for a strict C-order RGB NPY."""

    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise RuntimeError(f"Decoded future is not an NPY file: {path}")
        version = stream.read(2)
        if len(version) != 2:
            raise RuntimeError(f"Truncated NPY version: {path}")
        major, minor = version
        if major == 1:
            length_data = stream.read(2)
            if len(length_data) != 2:
                raise RuntimeError(f"Truncated NPY header length: {path}")
            header_length = struct.unpack("<H", length_data)[0]
            encoding = "latin1"
        elif major in (2, 3):
            length_data = stream.read(4)
            if len(length_data) != 4:
                raise RuntimeError(f"Truncated NPY header length: {path}")
            header_length = struct.unpack("<I", length_data)[0]
            encoding = "utf8" if major == 3 else "latin1"
        else:
            raise RuntimeError(f"Unsupported NPY version {major}.{minor}: {path}")
        header_data = stream.read(header_length)
        if len(header_data) != header_length:
            raise RuntimeError(f"Truncated NPY header: {path}")
        try:
            header = ast.literal_eval(header_data.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Invalid NPY header: {path}") from exc
        data_offset = stream.tell()
    if not isinstance(header, dict):
        raise RuntimeError(f"NPY header is not a dictionary: {path}")
    if header.get("descr") not in ("|u1", "<u1") or header.get("fortran_order") is not False:
        raise RuntimeError(f"Future must be C-order uint8 RGB: {path}")
    shape_raw = header.get("shape")
    if not isinstance(shape_raw, tuple) or len(shape_raw) != 4:
        raise RuntimeError(f"Future must have shape [33,H,W,3]: {path}")
    try:
        shape = tuple(int(value) for value in shape_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid NPY shape: {path}") from exc
    if shape[0] != 33 or shape[-1] != 3 or min(shape) <= 0:
        raise RuntimeError(f"Future must have shape [33,H,W,3], got {shape}: {path}")
    payload_bytes = math.prod(shape)
    if path.stat().st_size != data_offset + payload_bytes:
        raise RuntimeError(f"NPY payload length mismatch: {path}")
    return shape, data_offset, payload_bytes


def extract_npy_payload(source: Path, output: Path) -> tuple[int, int, int, int]:
    shape, offset, payload_bytes = inspect_npy_uint8_rgb(source)
    with source.open("rb") as reader, output.open("xb") as writer:
        reader.seek(offset)
        remaining = payload_bytes
        while remaining:
            block = reader.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError(f"Unexpected end of NPY payload: {source}")
            writer.write(block)
            remaining -= len(block)
        if reader.read(1):
            raise RuntimeError(f"Unexpected trailing NPY bytes: {source}")
    return shape


def collect_evidence(
    arm: str, result_path: Path, pair_paths: list[Path]
) -> tuple[ArmContract, VerifiedFile, list[VerifiedFile], list[CellEvidence]]:
    contract = ARM_CONTRACTS[arm]
    result_path = result_path.resolve()
    result = load_json(result_path)
    expected = {
        "schema_version": contract.result_schema,
        "status": "complete",
        "amendment_id": "V2-A015",
        "arm_id": contract.arm_id,
        "model_id": contract.model_id,
        "exact_prompts": PROMPTS,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(
                f"Compiled result mismatch for {key}: "
                f"expected={value!r}, observed={result.get(key)!r}"
            )
    if result.get("summary", {}).get("valid_episode_count") != 6:
        raise RuntimeError("Compiled result is not the complete six-cell denominator")
    episodes = _episode_map(result)
    manifests, manifest_records = _manifest_map(pair_paths, contract)
    _check_result_provenance(result, result_path, manifest_records)

    evidence: list[CellEvidence] = []
    for seed in SEEDS:
        pair_path, pair = manifests[seed]
        pair_cells = {cell["requested_relation"]: cell for cell in pair["cells"]}
        for relation in RELATIONS:
            row = episodes[(seed, relation)]
            cell = pair_cells[relation]
            if row.get("prompt") != PROMPTS[relation]:
                raise RuntimeError(f"Compiled prompt changed for {seed}/{relation}")
            if not isinstance(row.get("requested_success"), bool):
                raise RuntimeError(f"Compiled outcome is not boolean for {seed}/{relation}")
            actions = row.get("actions_executed")
            if not isinstance(actions, int) or not 0 < actions <= 450:
                raise RuntimeError(f"Invalid action count for {seed}/{relation}: {actions!r}")

            pair_viewport = resolve_record(
                cell.get("simulator_artifacts", {}).get("viewport_video", {}),
                pair_path.parent,
                f"pair viewport {seed}/{relation}",
            )
            result_viewport = resolve_record(
                row.get("simulator_artifacts", {}).get("viewport_video", {}),
                result_path.parent,
                f"compiled viewport {seed}/{relation}",
            )
            same_file(pair_viewport, result_viewport, f"viewport {seed}/{relation}")

            prediction_sources: list[VerifiedFile] = []
            prediction_shapes: list[tuple[int, int, int, int]] = []
            if arm == "dreamzero_action_cfg_s2":
                pair_future = resolve_record(
                    cell.get("future_manifest", {}),
                    pair_path.parent,
                    f"pair future manifest {seed}/{relation}",
                    require_bytes=False,
                )
                result_future = resolve_record(
                    row.get("future_manifest", {}),
                    result_path.parent,
                    f"compiled future manifest {seed}/{relation}",
                )
                same_file(pair_future, result_future, f"future manifest {seed}/{relation}")
                future = load_json(pair_future.path)
                decodes = future.get("official_reset_decode")
                compiled_decodes = row.get("official_decoded_futures")
                if not isinstance(decodes, list) or not decodes:
                    raise RuntimeError(f"No official_reset_decode for {seed}/{relation}")
                if (
                    not isinstance(compiled_decodes, list)
                    or row.get("official_decoded_future_count") != len(decodes)
                    or cell.get("future_manifest", {}).get("official_decode_count") != len(decodes)
                    or len(compiled_decodes) != len(decodes)
                ):
                    raise RuntimeError(f"Incomplete official decode accounting for {seed}/{relation}")
                for index, (raw_record, compiled_record) in enumerate(
                    zip(decodes, compiled_decodes, strict=True)
                ):
                    raw_decode = resolve_record(
                        raw_record,
                        pair_future.path.parent,
                        f"official decode {seed}/{relation}/{index}",
                        require_bytes=False,
                    )
                    compiled_decode = resolve_record(
                        compiled_record,
                        result_path.parent,
                        f"compiled official decode {seed}/{relation}/{index}",
                    )
                    same_file(raw_decode, compiled_decode, f"official decode {seed}/{relation}/{index}")
                    prediction_sources.append(raw_decode)
            else:
                raw_requests = cell.get("model_requests")
                compiled_requests = row.get("imagined_future_requests")
                if (
                    not isinstance(raw_requests, list)
                    or not isinstance(compiled_requests, list)
                    or not raw_requests
                    or len(raw_requests) != len(compiled_requests)
                    or cell.get("decoded_future_count") != len(raw_requests)
                    or row.get("decoded_future_count") != len(raw_requests)
                ):
                    raise RuntimeError(f"Incomplete Cosmos future accounting for {seed}/{relation}")
                for index, (raw_request, compiled_request) in enumerate(
                    zip(raw_requests, compiled_requests, strict=True)
                ):
                    if (
                        raw_request.get("request_index") != index
                        or compiled_request.get("request_index") != index
                        or compiled_request.get("prompt") != PROMPTS[relation]
                    ):
                        raise RuntimeError(f"Cosmos request order/prompt mismatch: {seed}/{relation}/{index}")
                    raw_future = resolve_record(
                        raw_request.get("decoded_future", {}),
                        pair_path.parent,
                        f"pair decoded future {seed}/{relation}/{index}",
                    )
                    compiled_future = resolve_record(
                        compiled_request.get("decoded_future", {}),
                        result_path.parent,
                        f"compiled decoded future {seed}/{relation}/{index}",
                    )
                    same_file(raw_future, compiled_future, f"decoded future {seed}/{relation}/{index}")
                    shape, _, _ = inspect_npy_uint8_rgb(raw_future.path)
                    prediction_sources.append(raw_future)
                    prediction_shapes.append(shape)

            evidence.append(
                CellEvidence(
                    seed=seed,
                    relation=relation,
                    prompt=PROMPTS[relation],
                    requested_success=row["requested_success"],
                    actions_executed=actions,
                    viewport=pair_viewport,
                    prediction_sources=prediction_sources,
                    prediction_shapes=prediction_shapes,
                )
            )
    return contract, local_file(result_path), manifest_records, evidence


def run(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        input=input_bytes,
        check=False,
        capture_output=True,
        text=input_bytes is None,
    )
    if result.returncode:
        stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode("utf8", "replace")
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{stderr[-4000:]}")
    return result  # type: ignore[return-value]


def _runtime_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Rendering requires OpenCV, NumPy, and Pillow; use the pinned RoboLab media environment"
        ) from exc
    return cv2, np, Image, (ImageDraw, ImageFont)


def video_info(path: Path) -> dict[str, Any]:
    cv2, _, _, _ = _runtime_modules()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video stream contract: {path}")
    return {
        "fps": fps,
        "frame_count": frames,
        "duration_s": frames / fps,
        "width": width,
        "height": height,
    }


def iter_video_frames(path: Path, expected_frames: int) -> Iterator[Any]:
    cv2, _, _, _ = _runtime_modules()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            yield frame
    finally:
        capture.release()
    if decoded != expected_frames:
        raise RuntimeError(
            f"Complete video decode count mismatch: {path}: expected={expected_frames}, decoded={decoded}"
        )


@lru_cache(maxsize=None)
def _font(size: int) -> Any:
    _, _, _, (_, image_font) = _runtime_modules()
    if FONT_FILE is None:
        raise RuntimeError("Publication font has not been initialized")
    return image_font.truetype(str(FONT_FILE), size=size)


def _draw_lines(frame: Any, rows: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]]) -> Any:
    cv2, np, image, (image_draw, _) = _runtime_modules()
    rgb = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB)
    canvas = image.fromarray(rgb)
    draw = image_draw.Draw(canvas)
    for position, text, size, color in rows:
        draw.text(position, text, font=_font(size), fill=color)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def make_panel(frame: Any, *, title: str, prompt: str, detail: str) -> Any:
    cv2, np, _, _ = _runtime_modules()
    height, width = frame.shape[:2]
    scale = min(PANEL_WIDTH / width, CONTENT_HEIGHT / height)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:HEADER_HEIGHT] = (24, 17, 11)
    y = HEADER_HEIGHT + (CONTENT_HEIGHT - resized.shape[0]) // 2
    x = (PANEL_WIDTH - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return _draw_lines(
        panel,
        [
            ((16, 12), title, 20, (226, 235, 244)),
            ((16, 45), prompt, 18, (255, 255, 255)),
            ((16, 78), detail, 14, (184, 199, 216)),
        ],
    )


def held_panel(panel: Any) -> Any:
    cv2, np, image, (image_draw, _) = _runtime_modules()
    rgb = cv2.cvtColor(np.ascontiguousarray(panel.copy()), cv2.COLOR_BGR2RGB)
    canvas = image.fromarray(rgb)
    draw = image_draw.Draw(canvas)
    text = "HELD FINAL FRAME — paired side continues"
    font = _font(17)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    x, y = (PANEL_WIDTH - width) // 2, PANEL_HEIGHT - 48
    draw.rectangle((x - 10, y - 7, x + width + 10, y + 28), fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 221, 117))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def seed_slate(seed: int, contract: ArmContract, kind: str) -> Any:
    _, np, _, _ = _runtime_modules()
    frame = np.zeros((PANEL_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    frame[:] = (35, 25, 16)
    return _draw_lines(
        frame,
        [
            ((56, 62), f"V2-A015 · seed {seed} · {kind}", 34, (255, 255, 255)),
            ((56, 120), contract.setting_label, 23, (198, 212, 225)),
            ((56, 202), f"LEFT prompt: {PROMPTS['left']}", 21, (245, 163, 64)),
            ((56, 247), f"RIGHT prompt: {PROMPTS['right']}", 21, (101, 169, 255)),
        ],
    )


def request_slate(cell: CellEvidence, index: int, count: int) -> Any:
    _, np, _, _ = _runtime_modules()
    frame = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    frame[:] = (31, 16, 21)
    return _draw_lines(
        frame,
        [
            ((28, 65), f"{cell.relation.upper()} · seed {cell.seed} · request {index + 1}/{count}", 25, (255, 255, 255)),
            ((28, 125), cell.prompt, 18, (215, 228, 239)),
            ((28, 210), "33-frame LOCAL prediction horizon", 23, (192, 255, 114)),
            ((28, 258), "Boundary slate: not a continuous imagined rollout", 17, (255, 214, 124)),
        ],
    )


def _encoder_command(ffmpeg: Path, fps: int, output: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{CANVAS_WIDTH}x{PANEL_HEIGHT}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
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
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-metadata",
        "title=",
        "-metadata",
        "comment=",
        str(output),
    ]


class RawEncoder:
    def __init__(self, ffmpeg: Path, fps: int, output: Path):
        self.command = _encoder_command(ffmpeg, fps, output)
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_count = 0

    def write(self, frame: Any) -> None:
        _, np, _, _ = _runtime_modules()
        if frame.shape != (PANEL_HEIGHT, CANVAS_WIDTH, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"Invalid publication frame: {frame.shape}/{frame.dtype}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg encoder stdin is unavailable")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            stderr = self.process.stderr.read().decode("utf8", "replace") if self.process.stderr else ""
            raise RuntimeError(f"ffmpeg encoder closed early: {stderr[-4000:]}") from exc
        self.frame_count += 1

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        returncode = self.process.wait()
        stderr = self.process.stderr.read().decode("utf8", "replace") if self.process.stderr else ""
        if returncode:
            raise RuntimeError(f"ffmpeg encoder failed ({returncode}): {stderr[-4000:]}")
        if self.frame_count <= 0:
            raise RuntimeError("ffmpeg encoder received zero frames")


def pair_panels(
    left: Iterator[Any], right: Iterator[Any]
) -> Iterator[tuple[Any, Any, bool, bool]]:
    iterators = [iter(left), iter(right)]
    last: list[Any | None] = [None, None]
    ended = [False, False]
    while True:
        fresh = [False, False]
        for index, iterator in enumerate(iterators):
            if ended[index]:
                continue
            try:
                last[index] = next(iterator)
                fresh[index] = True
            except StopIteration:
                ended[index] = True
        if all(ended):
            break
        if any(value is None for value in last):
            raise RuntimeError("One side of a matched pair contains zero decodable frames")
        yield last[0], last[1], ended[0] and not fresh[0], ended[1] and not fresh[1]


def actual_panel_frames(cell: CellEvidence, info: dict[str, Any], contract: ArmContract) -> Iterator[Any]:
    outcome = "SUCCESS" if cell.requested_success else "VALID FAILURE"
    for frame in iter_video_frames(cell.viewport.path, info["frame_count"]):
        yield make_panel(
            frame,
            title=f"ACTUAL EXECUTION · {cell.relation.upper()}",
            prompt=cell.prompt,
            detail=(
                f"seed {cell.seed} · {outcome} · {cell.actions_executed} executed actions\n"
                f"{contract.setting_label}"
            ),
        )


def render_actual(
    *,
    ffmpeg: Path,
    contract: ArmContract,
    cells: list[CellEvidence],
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cv2, _, _, _ = _runtime_modules()
    cell_map = {(cell.seed, cell.relation): cell for cell in cells}
    infos = {(cell.seed, cell.relation): video_info(cell.viewport.path) for cell in cells}
    source_fps = [info["fps"] for info in infos.values()]
    fps = round(source_fps[0])
    if fps <= 0 or any(abs(value - fps) > 1e-6 for value in source_fps):
        raise RuntimeError(f"Viewport FPS values are not one common integer rate: {source_fps}")
    temporary = output.with_name(output.name + ".staging.mp4")
    encoder = RawEncoder(ffmpeg, fps, temporary)
    padding: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            slate = seed_slate(seed, contract, "ACTUAL SIMULATOR EXECUTION")
            for _ in range(round(SEED_SLATE_SECONDS * fps)):
                encoder.write(slate)
            left = cell_map[(seed, "left")]
            right = cell_map[(seed, "right")]
            left_info, right_info = infos[(seed, "left")], infos[(seed, "right")]
            target_frames = max(left_info["frame_count"], right_info["frame_count"])
            for relation, info in (("left", left_info), ("right", right_info)):
                pad_frames = target_frames - info["frame_count"]
                padding.append(
                    {
                        "seed": seed,
                        "relation": relation,
                        "source_frame_count": info["frame_count"],
                        "paired_frame_count": target_frames,
                        "held_final_frame_padding_frames": pad_frames,
                        "held_final_frame_padding_s": pad_frames / fps,
                        "label_shown_during_padding": pad_frames > 0,
                    }
                )
            written = 0
            for left_panel, right_panel, left_held, right_held in pair_panels(
                actual_panel_frames(left, left_info, contract),
                actual_panel_frames(right, right_info, contract),
            ):
                if left_held:
                    left_panel = held_panel(left_panel)
                if right_held:
                    right_panel = held_panel(right_panel)
                encoder.write(cv2.hconcat([left_panel, right_panel]))
                written += 1
            if written != target_frames:
                raise RuntimeError(
                    f"Actual pair frame count mismatch for seed {seed}: "
                    f"expected={target_frames}, rendered={written}"
                )
    except BaseException:
        if encoder.process.poll() is None:
            encoder.process.kill()
        encoder.process.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        del cv2
    encoder.close()
    temporary.replace(output)
    return (
        {
            "command": encoder.command,
            "input_probes": {
                f"seed{seed}_{relation}": infos[(seed, relation)]
                for seed in SEEDS
                for relation in RELATIONS
            },
            "output_fps": fps,
            "output_frame_count": encoder.frame_count,
            "composition_backend": "OpenCV complete-frame decoder + Pillow raster labels + raw BGR ffmpeg pipe",
        },
        padding,
    )


def dream_prediction_frames(
    cell: CellEvidence, contract: ArmContract, infos: list[dict[str, Any]]
) -> Iterator[Any]:
    count = len(cell.prediction_sources)
    for index, (source, info) in enumerate(
        zip(cell.prediction_sources, infos, strict=True)
    ):
        for frame in iter_video_frames(source.path, info["frame_count"]):
            yield make_panel(
                frame,
                title=f"{contract.prediction_kind} · {cell.relation.upper()}",
                prompt=cell.prompt,
                detail=(
                    f"seed {cell.seed} · official reset decode {index + 1}/{count}\n"
                    f"{contract.setting_label}"
                ),
            )


def cosmos_prediction_frames(cell: CellEvidence, contract: ArmContract) -> Iterator[Any]:
    cv2, np, _, _ = _runtime_modules()
    count = len(cell.prediction_sources)
    slate_frames = round(REQUEST_SLATE_SECONDS * contract.prediction_fps)
    for index, (source, expected_shape) in enumerate(
        zip(cell.prediction_sources, cell.prediction_shapes, strict=True)
    ):
        slate = request_slate(cell, index, count)
        for _ in range(slate_frames):
            yield slate
        future = np.load(source.path, allow_pickle=False, mmap_mode="r")
        if tuple(future.shape) != expected_shape or future.dtype != np.uint8:
            raise RuntimeError(f"Cosmos future changed after provenance validation: {source.path}")
        for rgb in future:
            frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            yield make_panel(
                frame,
                title=f"MODEL PREDICTION · LOCAL HORIZON · {cell.relation.upper()}",
                prompt=cell.prompt,
                detail=(
                    f"seed {cell.seed} · request {index + 1}/{count} · 33 frames\n"
                    f"{contract.setting_label}"
                ),
            )


def render_prediction(
    *,
    ffmpeg: Path,
    contract: ArmContract,
    cells: list[CellEvidence],
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    cv2, _, _, _ = _runtime_modules()
    cell_map = {(cell.seed, cell.relation): cell for cell in cells}
    input_probes: dict[tuple[int, str], list[dict[str, Any]]] = {}
    request_counts: dict[str, int] = {}
    source_frames: dict[tuple[int, str], int] = {}
    slate_frames = round(REQUEST_SLATE_SECONDS * contract.prediction_fps)
    for cell in cells:
        key = (cell.seed, cell.relation)
        request_counts[f"seed{cell.seed}_{cell.relation}"] = len(cell.prediction_sources)
        if contract.arm_id == "dreamzero_action_cfg_s2":
            infos = [video_info(source.path) for source in cell.prediction_sources]
            if any(abs(info["fps"] - contract.prediction_fps) > 1e-6 for info in infos):
                raise RuntimeError(f"DreamZero official-decode FPS changed: {key}: {infos}")
            input_probes[key] = infos
            source_frames[key] = sum(info["frame_count"] for info in infos)
        else:
            infos = [
                {
                    "frame_count": shape[0],
                    "fps": contract.prediction_fps,
                    "duration_s": shape[0] / contract.prediction_fps,
                    "width": shape[2],
                    "height": shape[1],
                    "source_container": "npy_c_order_uint8",
                }
                for shape in cell.prediction_shapes
            ]
            input_probes[key] = infos
            source_frames[key] = len(infos) * slate_frames + sum(
                info["frame_count"] for info in infos
            )

    temporary = output.with_name(output.name + ".staging.mp4")
    encoder = RawEncoder(ffmpeg, contract.prediction_fps, temporary)
    padding: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            slate = seed_slate(seed, contract, contract.prediction_kind)
            for _ in range(round(SEED_SLATE_SECONDS * contract.prediction_fps)):
                encoder.write(slate)
            left, right = cell_map[(seed, "left")], cell_map[(seed, "right")]
            target_frames = max(source_frames[(seed, relation)] for relation in RELATIONS)
            for relation in RELATIONS:
                count = source_frames[(seed, relation)]
                pad_frames = target_frames - count
                padding.append(
                    {
                        "seed": seed,
                        "relation": relation,
                        "source_frame_count_including_request_slates": count,
                        "paired_frame_count": target_frames,
                        "held_final_frame_padding_frames": pad_frames,
                        "held_final_frame_padding_s": pad_frames / contract.prediction_fps,
                        "label_shown_during_padding": pad_frames > 0,
                    }
                )
            left_iterator = (
                dream_prediction_frames(left, contract, input_probes[(seed, "left")])
                if contract.arm_id == "dreamzero_action_cfg_s2"
                else cosmos_prediction_frames(left, contract)
            )
            right_iterator = (
                dream_prediction_frames(right, contract, input_probes[(seed, "right")])
                if contract.arm_id == "dreamzero_action_cfg_s2"
                else cosmos_prediction_frames(right, contract)
            )
            written = 0
            for left_panel, right_panel, left_held, right_held in pair_panels(
                left_iterator, right_iterator
            ):
                if left_held:
                    left_panel = held_panel(left_panel)
                if right_held:
                    right_panel = held_panel(right_panel)
                encoder.write(cv2.hconcat([left_panel, right_panel]))
                written += 1
            if written != target_frames:
                raise RuntimeError(
                    f"Prediction pair frame count mismatch for seed {seed}: "
                    f"expected={target_frames}, rendered={written}"
                )
    except BaseException:
        if encoder.process.poll() is None:
            encoder.process.kill()
        encoder.process.wait()
        temporary.unlink(missing_ok=True)
        raise
    encoder.close()
    temporary.replace(output)
    return (
        {
            "command": encoder.command,
            "input_probes": {
                f"seed{seed}_{relation}": input_probes[(seed, relation)]
                for seed in SEEDS
                for relation in RELATIONS
            },
            "output_fps": contract.prediction_fps,
            "output_frame_count": encoder.frame_count,
            "composition_backend": "Complete source decode + Pillow raster labels + raw BGR ffmpeg pipe",
            "cosmos_request_boundary_slate_frames": (
                slate_frames if contract.arm_id == "cosmos3_nano_no_cfg_g1" else 0
            ),
        },
        padding,
        request_counts,
    )


def _mp4_atom_offsets(path: Path) -> dict[str, int]:
    offsets: dict[str, int] = {}
    with path.open("rb") as stream:
        offset = 0
        total = path.stat().st_size
        while offset + 8 <= total:
            stream.seek(offset)
            size_raw = stream.read(4)
            atom = stream.read(4).decode("latin1", "replace")
            size = struct.unpack(">I", size_raw)[0]
            header = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    break
                size = struct.unpack(">Q", extended)[0]
                header = 16
            elif size == 0:
                size = total - offset
            if size < header:
                break
            offsets.setdefault(atom, offset)
            offset += size
    return offsets


def validate_output(ffmpeg: Path, path: Path) -> dict[str, Any]:
    cv2, np, _, _ = _runtime_modules()
    inspection = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    stderr = inspection.stderr
    video_line = next((line for line in stderr.splitlines() if "Video:" in line), None)
    if video_line is None:
        raise RuntimeError(f"ffmpeg did not report a video stream: {path}")
    codec_match = re.search(r"Video:\s*([^\s,]+)", video_line)
    pixel_match = re.search(r",\s*(yuv420p)(?:\([^)]*\))?\s*,", video_line)
    dimensions = re.search(r",\s*(\d+)x(\d+)(?:\s|\[|,)", video_line)
    if (
        codec_match is None
        or codec_match.group(1) != "h264"
        or pixel_match is None
        or dimensions is None
        or int(dimensions.group(1)) != CANVAS_WIDTH
        or int(dimensions.group(2)) != PANEL_HEIGHT
    ):
        raise RuntimeError(f"Publication encoding contract failed: {path}: {video_line}")
    info = video_info(path)
    frame_count = info["frame_count"]
    if frame_count < 3:
        raise RuntimeError(f"Publication video has fewer than three frames: {path}")
    indices = sorted({0, frame_count // 2, frame_count - 1})
    capture = cv2.VideoCapture(str(path))
    decoded = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode publication frame {index}: {path}")
            decoded.append(
                {
                    "frame_index": index,
                    "decoded_bgr_sha256": hashlib.sha256(
                        np.ascontiguousarray(frame).tobytes()
                    ).hexdigest(),
                }
            )
    finally:
        capture.release()
    full_decode_command = [
        str(ffmpeg), "-v", "error", "-i", str(path), "-an", "-f", "null", "-"
    ]
    run(full_decode_command)
    if re.search(
        r"^\s*(title|comment|description|copyright|creation_time)\s*:",
        stderr,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        raise RuntimeError(f"Publication video retained descriptive source metadata: {path}")
    atoms = _mp4_atom_offsets(path)
    if "moov" not in atoms or "mdat" not in atoms or atoms["moov"] > atoms["mdat"]:
        raise RuntimeError(f"Publication video is not fast-start MP4: {path}")
    return {
        **info,
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "frame_count": frame_count,
        "decoded_frame_indices": indices,
        "decoded_frame_samples": decoded,
        "full_decode_command": full_decode_command,
        "faststart_atom_offsets": atoms,
        "metadata_policy": "map_metadata=-1; no title/comment/description/copyright/creation_time tags",
    }


def render_poster(ffmpeg: Path, video: Path, frame_count: int, output: Path) -> list[str]:
    middle = frame_count // 2
    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{middle})",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-map_metadata",
        "-1",
        str(output),
    ]
    run(command)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Poster extraction failed: {output}")
    run([str(ffmpeg), "-v", "error", "-i", str(output), "-frames:v", "1", "-f", "null", "-"])
    return command


def output_record(path: Path, *, final_dir: Path, record_prefix: str | None) -> dict[str, Any]:
    observed = local_file(path)
    display = str(final_dir / path.name)
    if record_prefix:
        display = f"{record_prefix.rstrip('/')}/{path.name}"
    return observed.record(display)


def _normalized_command(command: list[str], staging_root: Path) -> list[str]:
    root = str(staging_root.resolve())
    return [value.replace(root, "$V2A015_MEDIA_STAGING") for value in command]


def build(args: argparse.Namespace) -> dict[str, Any]:
    global FONT_FILE

    contract, result_record, pair_records, cells = collect_evidence(
        args.arm, args.result, args.pair_manifest
    )
    ffmpeg = local_file(args.ffmpeg.resolve())
    font_file = local_file(args.font_file.resolve())
    FONT_FILE = font_file.path
    _font.cache_clear()
    cv2, np, _, _ = _runtime_modules()
    import PIL

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite media output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{contract.arm_id}-media-", dir=output_dir.parent) as temporary_root:
        staging_root = Path(temporary_root).resolve()
        build_dir = staging_root / "deliverable"
        build_dir.mkdir()
        actual_name = f"{contract.arm_id}_all_seeds_actual.mp4"
        prediction_name = f"{contract.arm_id}_all_seeds_{contract.prediction_filename_label}.mp4"
        actual = build_dir / actual_name
        prediction = build_dir / prediction_name
        actual_poster = build_dir / actual_name.replace(".mp4", "_poster.jpg")
        prediction_poster = build_dir / prediction_name.replace(".mp4", "_poster.jpg")

        actual_render, actual_padding = render_actual(
            ffmpeg=ffmpeg.path,
            contract=contract,
            cells=cells,
            output=actual,
        )
        prediction_render, prediction_padding, request_counts = render_prediction(
            ffmpeg=ffmpeg.path,
            contract=contract,
            cells=cells,
            output=prediction,
        )
        actual_validation = validate_output(ffmpeg.path, actual)
        prediction_validation = validate_output(ffmpeg.path, prediction)
        actual_poster_command = render_poster(
            ffmpeg.path, actual, actual_validation["frame_count"], actual_poster
        )
        prediction_poster_command = render_poster(
            ffmpeg.path, prediction, prediction_validation["frame_count"], prediction_poster
        )

        ffmpeg_version = run([str(ffmpeg.path), "-version"]).stdout.splitlines()[0]
        input_cells = []
        for cell in cells:
            input_cells.append(
                {
                    "environment_seed": cell.seed,
                    "relation": cell.relation,
                    "prompt": cell.prompt,
                    "requested_success": cell.requested_success,
                    "actions_executed": cell.actions_executed,
                    "complete_viewport_video": cell.viewport.record(),
                    "prediction_source_count": len(cell.prediction_sources),
                    "prediction_sources_in_order": [item.record() for item in cell.prediction_sources],
                    "prediction_shapes": [list(shape) for shape in cell.prediction_shapes],
                }
            )

        manifest = {
            "schema_version": "vla-wam-shared-v2-v2a015-cfg-media-v1",
            "status": "complete_all_six_cells_actual_and_prediction_media",
            "amendment_id": "V2-A015",
            "arm_id": contract.arm_id,
            "model_id": contract.model_id,
            "claim_boundary": contract.claim_boundary,
            "exact_prompts": PROMPTS,
            "setting_label": contract.setting_label,
            "selection_policy": (
                "No outcome-based or request-based selection. Exact supplied pair manifests "
                "for seeds 8300--8302 are matched against compiled-result provenance; all six "
                "complete viewport videos and all exposed prediction/imagination sources are retained."
            ),
            "source_result": result_record.record(),
            "source_pair_manifests": [item.record() for item in sorted(pair_records, key=lambda row: str(row.path))],
            "input_cells": input_cells,
            "request_or_decode_counts": request_counts,
            "padding": {
                "policy": (
                    "Within each matched seed, both sides start together. The shorter side holds "
                    "its final decoded frame with an explicit label until the longer side ends; "
                    "no source video or prediction is truncated."
                ),
                "actual": actual_padding,
                "prediction_or_imagination": prediction_padding,
            },
            "outputs": {
                "actual_video": output_record(actual, final_dir=output_dir, record_prefix=args.record_prefix),
                "actual_poster": output_record(actual_poster, final_dir=output_dir, record_prefix=args.record_prefix),
                "prediction_or_imagination_video": output_record(prediction, final_dir=output_dir, record_prefix=args.record_prefix),
                "prediction_or_imagination_poster": output_record(prediction_poster, final_dir=output_dir, record_prefix=args.record_prefix),
            },
            "output_validation": {
                "actual": actual_validation,
                "prediction_or_imagination": prediction_validation,
                "policy": "H.264/yuv420p, 1280x480, fast-start, and successful first/middle/last frame decodes",
            },
            "renderer": {
                "tool": local_file(Path(__file__)).record(),
                "ffmpeg": ffmpeg.record(),
                "ffmpeg_version": ffmpeg_version,
                "runtime_versions": {
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                    "pillow": PIL.__version__,
                    "label_font": font_file.record(),
                },
                "encoding": {
                    "codec": "libx264",
                    "pixel_format": "yuv420p",
                    "movflags": "+faststart",
                    "threads": 1,
                    "metadata": "stripped with -map_metadata -1",
                },
                "commands": {
                    "actual": {
                        **actual_render,
                        "command": _normalized_command(actual_render["command"], staging_root),
                    },
                    "prediction_or_imagination": {
                        **prediction_render,
                        "command": _normalized_command(prediction_render["command"], staging_root),
                    },
                    "actual_poster": _normalized_command(actual_poster_command, staging_root),
                    "prediction_or_imagination_poster": _normalized_command(prediction_poster_command, staging_root),
                },
            },
        }
        manifest_path = build_dir / "media_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(build_dir, output_dir)
    return {
        "status": "complete",
        "arm_id": contract.arm_id,
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "media_manifest.json"),
        "actual_video": str(output_dir / actual_name),
        "prediction_or_imagination_video": str(output_dir / prediction_name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_CONTRACTS))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-prefix")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--font-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
