#!/usr/bin/env python3
"""Build one bounded, browser-compatible V3-A002 paired rollout video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


SUMMARY_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-summary-v3"
MANIFEST_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-hash-manifest-v3"
AMENDMENT_SCHEMA = "vla-wam-shared-v3-post-result-pi0-fast-old-name-config-amendment-v1"
MODEL_ID = "pi0_fast_old_name_config_v3a002"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def repository_record(path: Path, repository_path: str) -> dict[str, Any]:
    payload = record(path)
    payload["path"] = repository_path
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def raw_viewport_record(manifest: dict[str, Any], seed: int, relation: str) -> dict[str, Any]:
    relation_marker = "RubiksCubeLeftOfBowlMatchedTask" if relation == "left" else "RubiksCubeRightOfBowlMatchedTask"
    matches = [
        item
        for item in manifest["raw_source_artifacts"]
        if f"seed{seed}_" in item["path"]
        and relation_marker in item["path"]
        and item["path"].endswith("_viewport.mp4")
    ]
    require(len(matches) == 1, f"expected one {seed}/{relation} viewport record, found {len(matches)}")
    return matches[0]


def verify_source(expected: dict[str, Any]) -> Path:
    source = Path(expected["path"])
    require(source.is_file(), f"missing source video: {source}")
    actual = record(source)
    require(actual["bytes"] == expected["bytes"], f"source byte-count mismatch: {source}")
    require(actual["sha256"] == expected["sha256"], f"source SHA-256 mismatch: {source}")
    return source


def put_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_pair(
    left_path: Path,
    right_path: Path,
    output: Path,
    poster: Path,
    seed: int,
    prompts: dict[str, str],
    pair: dict[str, Any],
) -> tuple[int, float, float]:
    captures = [cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))]
    require(all(capture.isOpened() for capture in captures), "could not open both source viewport videos")
    source_fps = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
    require(all(value > 0 for value in source_fps), f"invalid source FPS values: {source_fps}")
    require(abs(source_fps[0] - source_fps[1]) < 1e-6, f"mismatched source FPS values: {source_fps}")

    width, view_height, header_height, footer_height = 640, 360, 118, 114
    canvas_height = header_height + view_height + footer_height
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + "_intermediate_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps[0],
        (width * 2, canvas_height),
    )
    require(writer.isOpened(), f"could not open intermediate writer: {intermediate}")

    latest: list[np.ndarray | None] = [None, None]
    ended = [False, False]
    frame_count = 0
    first_canvas: np.ndarray | None = None
    relations = ("left", "right")
    colors = ((0, 165, 255), (220, 120, 35))
    try:
        while not all(ended):
            for index, capture in enumerate(captures):
                if ended[index]:
                    continue
                ok, frame = capture.read()
                if ok:
                    latest[index] = frame
                else:
                    ended[index] = True
            if any(frame is None for frame in latest):
                require(not any(ended), "a source viewport video had no decodable frames")
                continue

            canvas = np.zeros((canvas_height, width * 2, 3), dtype=np.uint8)
            put_text(canvas, f"pi0-FAST compatibility cohort | matched seed {seed}", (24, 32), 0.75, (245, 245, 245), 2)
            put_text(canvas, "ACTUAL SIMULATOR EXECUTION | same reset; only the static prompt changes", (24, 62), 0.50, (190, 190, 190), 1)
            for index, relation in enumerate(relations):
                x = index * width
                success = bool(pair[f"{relation}_success"])
                outcome = "SUCCESS" if success else "FAILURE"
                put_text(canvas, f"{relation.upper()} request | {outcome}", (x + 24, 88), 0.59, colors[index], 2)
                put_text(canvas, f'Prompt: "{prompts[relation]}"', (x + 24, 109), 0.38, (235, 235, 235), 1)
                canvas[header_height : header_height + view_height, x : x + width] = cv2.resize(
                    latest[index], (width, view_height), interpolation=cv2.INTER_AREA
                )

            footer_y = header_height + view_height
            shift = float(pair["right_minus_left_endpoint_shift_m"])
            rms = float(pair["action_rms_common_prefix"])
            prefix = int(pair["common_prefix_actions"])
            put_text(canvas, "Matched-pair diagnostics", (24, footer_y + 32), 0.58, (245, 245, 245), 2)
            put_text(
                canvas,
                f"RIGHT - LEFT final lateral shift: {shift:+.3f} m ({pair['endpoint_ordering'].replace('_', ' ')})",
                (24, footer_y + 61),
                0.52,
                (220, 220, 220),
                1,
            )
            put_text(
                canvas,
                f"Executed-action RMS over {prefix} common steps: {rms:.3f} (native mixed 8-D coordinates)",
                (24, footer_y + 88),
                0.47,
                (180, 180, 180),
                1,
            )
            put_text(canvas, "Action-only policy: no imagined-future video exists for this checkpoint.", (664, footer_y + 61), 0.48, (180, 180, 180), 1)
            writer.write(canvas)
            if first_canvas is None:
                first_canvas = canvas.copy()
            frame_count += 1
    finally:
        for capture in captures:
            capture.release()
        writer.release()

    require(frame_count > 0 and first_canvas is not None, "publication video has no frames")
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
    require(cv2.imwrite(str(poster), first_canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]), "could not write poster")
    duration_seconds = frame_count / source_fps[0]
    return frame_count, source_fps[0], duration_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8311)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-media-dir", required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    evidence = json.loads(args.evidence_manifest.read_text())
    amendment = json.loads(args.amendment.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA, "unexpected summary schema")
    require(summary.get("model_id") == MODEL_ID and summary.get("behavioral_episodes") == 40, "unexpected summary identity or count")
    require(evidence.get("schema_version") == MANIFEST_SCHEMA, "unexpected evidence-manifest schema")
    require(evidence.get("model_id") == MODEL_ID and evidence.get("raw_inputs_read_only") is True, "unexpected evidence identity or mutability")
    require(amendment.get("schema_version") == AMENDMENT_SCHEMA and amendment.get("amendment_id") == "V3-A002", "unexpected amendment")

    pair_matches = [pair for pair in summary["pairs"] if pair["seed"] == args.seed]
    require(len(pair_matches) == 1, f"expected one summary pair for seed {args.seed}")
    pair = pair_matches[0]
    prompts = amendment["three_request_gate"]["prompts"]
    left_record = raw_viewport_record(evidence, args.seed, "left")
    right_record = raw_viewport_record(evidence, args.seed, "right")
    left_path, right_path = verify_source(left_record), verify_source(right_record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pi0_fast_v3a002_seed{args.seed}_paired_actual"
    video = args.output_dir / f"{stem}.mp4"
    poster = args.output_dir / f"{stem}_poster.jpg"
    frames, fps, duration = render_pair(left_path, right_path, video, poster, args.seed, prompts, pair)

    repository_dir = args.repository_media_dir.rstrip("/")
    media = {
        "schema_version": "vla-wam-shared-v3-pi0-fast-old-name-config-media-v1",
        "study_id": summary["study_id"],
        "amendment_id": "V3-A002",
        "model_id": MODEL_ID,
        "status": "complete_selected_matched_actual_rollout",
        "selection_rule": "Lowest seed in the completed V3-A002 cohort with LEFT failure and RIGHT success: seed 8311.",
        "claim_boundary": "Actual simulator execution from the separate public compatibility cohort. It is not recovered historical pi0-FAST evidence, is not pooled with the historical denominator, and has no imagined-future counterpart.",
        "browser_encoding": "H.264 / yuv420p / faststart / no audio",
        "seed": args.seed,
        "directions": {
            "left": {"prompt": prompts["left"], "success": bool(pair["left_success"]), "source_video": left_record},
            "right": {"prompt": prompts["right"], "success": bool(pair["right_success"]), "source_video": right_record},
        },
        "matched_pair_diagnostics": {
            "endpoint_ordering": pair["endpoint_ordering"],
            "right_minus_left_endpoint_shift_m": pair["right_minus_left_endpoint_shift_m"],
            "action_rms_common_prefix": pair["action_rms_common_prefix"],
            "common_prefix_actions": pair["common_prefix_actions"],
            "action_rms_unit": summary["action_rms_common_prefix"]["unit"],
        },
        "source_summary": repository_record(
            args.summary,
            "artifacts/vla_wam_shared_v3/results/pi0_fast_old_name_config_v3a002_summary.json",
        ),
        "source_evidence_manifest": repository_record(
            args.evidence_manifest,
            "artifacts/vla_wam_shared_v3/results/pi0_fast_old_name_config_v3a002_evidence_hash_manifest.json",
        ),
        "source_amendment": repository_record(
            args.amendment,
            "artifacts/vla_wam_shared_v3/post_result_pi0_fast_old_name_config_amendment.json",
        ),
        "renderer": repository_record(Path(__file__), "tools/build_v3a002_pi0_fast_media.py"),
        "publication_video": repository_record(video, f"{repository_dir}/{video.name}"),
        "poster": repository_record(poster, f"{repository_dir}/{poster.name}"),
        "frame_count": frames,
        "fps": fps,
        "duration_seconds": duration,
    }
    manifest_path = args.output_dir / "media_manifest.json"
    manifest_path.write_text(json.dumps(media, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "poster": str(poster), "video": str(video)}, sort_keys=True))


if __name__ == "__main__":
    main()
