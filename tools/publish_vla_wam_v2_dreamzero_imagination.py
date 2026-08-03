#!/usr/bin/env python3
"""Archive every official DreamZero decode and publish paired imagination clips.

The DreamZero server writes one official reset-path decode for each retained
session.  V2-A007 has three fixed-observation probe sessions and six valid
behavioral sessions.  This tool validates those nine raw MP4s against their
hash-bearing manifests, copies the exact bytes into the bounded Git media
archive, and builds three derived LEFT/RIGHT behavioral comparison clips.

The resulting videos are model predictions, not simulator executions and not
additional behavioral episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = Path(
    "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/vla_wam_shared_v2/media/dreamzero_droid/imagination")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
SEEDS = (8300, 8301, 8302)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Missing JSON evidence: {path}")
    return json.loads(path.read_text())


def validate_record(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"Byte mismatch for {label}: {path}")
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {label}: {path}")
    return path


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(repo_root.resolve())),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def parse_duration(stderr: str) -> float:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if match is None:
        raise RuntimeError("ffmpeg did not report an input duration")
    hours, minutes, seconds = match.groups()
    value = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid duration: {value}")
    return value


def duration(ffmpeg: Path, video: Path) -> float:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(video)],
        check=False,
        capture_output=True,
        text=True,
    )
    return parse_duration(result.stderr)


def archive_exact(source: Path, target: Path, expected_sha: str) -> None:
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite archived decode: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256(target) != expected_sha:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Copied decode hash mismatch: {target}")


def compose_pair(ffmpeg: Path, left: Path, right: Path, output: Path) -> dict[str, Any]:
    left_duration = duration(ffmpeg, left)
    right_duration = duration(ffmpeg, right)
    total_duration = max(left_duration, right_duration)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite paired imagination clip: {output}")
    temporary = output.with_name(output.name + ".staging.mp4")
    if temporary.exists():
        raise RuntimeError(f"Refusing stale staging file: {temporary}")
    filter_graph = (
        f"[0:v]setpts=PTS-STARTPTS,scale=640:352:force_original_aspect_ratio=decrease,"
        f"pad=640:352:(ow-iw)/2:(oh-ih)/2:black,tpad=stop_mode=clone:"
        f"stop_duration={total_duration:.6f}[l];"
        f"[1:v]setpts=PTS-STARTPTS,scale=640:352:force_original_aspect_ratio=decrease,"
        f"pad=640:352:(ow-iw)/2:(oh-ih)/2:black,tpad=stop_mode=clone:"
        f"stop_duration={total_duration:.6f}[r];"
        "[l][r]hstack=inputs=2[v]"
    )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-an",
        "-t",
        f"{total_duration:.6f}",
        "-r",
        "5",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(output)
    return {
        "left_duration_s": left_duration,
        "right_duration_s": right_duration,
        "duration_s": total_duration,
        "fps": 5,
        "layout": "LEFT-command official decode on the left; RIGHT-command official decode on the right",
        "temporal_alignment": "Both complete official decodes start together; the shorter decode holds its final frame.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    result_path = args.result if args.result.is_absolute() else repo_root / args.result
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    ffmpeg = args.ffmpeg.resolve()
    if not ffmpeg.is_file():
        raise RuntimeError(f"Missing ffmpeg: {ffmpeg}")
    manifest_path = output_dir / "imagination_media_manifest.json"
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite imagination archive: {output_dir}")

    result = load_json(result_path)
    if (
        result.get("schema_version")
        != "vla-wam-shared-v2-dreamzero-droid-direct-gate-v1"
        or result.get("status") != "complete"
        or result.get("valid_episode_count") != 6
        or result.get("future_retention_audit", {}).get("total_official_reset_decode_count") != 9
    ):
        raise RuntimeError("DreamZero compiled-result contract mismatch")

    episode_map = {
        (int(row["environment_seed"]), row["requested_relation"]): row
        for row in result["episodes"]
    }
    expected_cells = {(seed, relation) for seed in SEEDS for relation in PROMPTS}
    if set(episode_map) != expected_cells:
        raise RuntimeError("DreamZero result does not contain the exact six behavioral cells")

    output_dir.mkdir(parents=True)
    originals_dir = output_dir / "official_decodes"
    pairs_dir = output_dir / "paired"
    originals_dir.mkdir()
    pairs_dir.mkdir()

    archived: list[dict[str, Any]] = []
    behavioral: dict[tuple[int, str], Path] = {}
    for seed in SEEDS:
        for relation in PROMPTS:
            row = episode_map[(seed, relation)]
            decodes = row.get("official_decoded_futures", [])
            if row.get("official_decoded_future_count") != 1 or len(decodes) != 1:
                raise RuntimeError(f"Expected one behavioral decode for {seed}/{relation}")
            source_record = decodes[0]
            source = validate_record(source_record, f"behavioral decode {seed}/{relation}")
            target = originals_dir / f"behavioral_seed{seed}_{relation}_official_decode.mp4"
            archive_exact(source, target, source_record["sha256"])
            behavioral[(seed, relation)] = target
            archived.append(
                {
                    "id": f"behavioral_seed{seed}_{relation}",
                    "scope": "valid_behavioral_episode",
                    "environment_seed": seed,
                    "sampling_seed": int(row["sampling_seed"]),
                    "relation": relation,
                    "prompt": PROMPTS[relation],
                    "policy_request_count": int(row["policy_request_count"]),
                    "source_video": source_record,
                    "archived_video": file_record(target, repo_root),
                }
            )

    probe_record = result["provenance"]["exact_repeat_probe"]
    probe_path = validate_record(probe_record, "fixed-observation probe")
    probe = load_json(probe_path)
    if probe.get("schema_version") != "vla-wam-shared-v2-dreamzero-exact-repeat-probe-v1" or not probe.get("passed"):
        raise RuntimeError("DreamZero fixed-observation probe contract mismatch")
    probe_specs = (
        ("left_a", "fixed_probe_left_a", "left"),
        ("left_b", "fixed_probe_left_b_exact_repeat", "left"),
        ("right", "fixed_probe_right", "right"),
    )
    for record_key, archive_id, relation in probe_specs:
        item = probe["records"][record_key]
        future_path = Path(item["future_manifest"]).resolve()
        if sha256(future_path) != item["future_manifest_sha256"]:
            raise RuntimeError(f"Probe future-manifest hash mismatch: {record_key}")
        future = load_json(future_path)
        decodes = future.get("official_reset_decode", [])
        if item.get("official_decode_count") != 1 or len(decodes) != 1:
            raise RuntimeError(f"Expected one probe decode: {record_key}")
        source_record = decodes[0]
        source = validate_record(source_record, f"probe decode {record_key}")
        target = originals_dir / f"{archive_id}_official_decode.mp4"
        archive_exact(source, target, source_record["sha256"])
        archived.append(
            {
                "id": archive_id,
                "scope": "fixed_observation_diagnostic",
                "probe_record": record_key,
                "sampling_seed_label": int(item["sampling_seed_label"]),
                "relation": relation,
                "prompt": item["prompt"],
                "source_future_manifest": {
                    "path": str(future_path),
                    "bytes": future_path.stat().st_size,
                    "sha256": sha256(future_path),
                },
                "source_video": source_record,
                "archived_video": file_record(target, repo_root),
            }
        )

    pair_entries = []
    for seed in SEEDS:
        output = pairs_dir / f"dreamzero_seed{seed}_left_right_imagined_futures.mp4"
        composition = compose_pair(
            ffmpeg,
            behavioral[(seed, "left")],
            behavioral[(seed, "right")],
            output,
        )
        left = episode_map[(seed, "left")]
        right = episode_map[(seed, "right")]
        pair_entries.append(
            {
                "id": f"dreamzero_droid_seed{seed}_imagined_futures",
                "arena": "droid",
                "arena_label": "DROID / RoboLab",
                "model_label": "DreamZero DROID — imagined futures",
                "category": "WAM IMAGINATION",
                "media_kind": "model_prediction_not_execution",
                "future_interface": "Official reset-path decode of retained DreamZero latent video predictions",
                "evidence_status": "Hash-validated official model decode; not simulator execution or an additional episode",
                "pair_label": f"seed {seed} matched behavioral sessions",
                "seed": seed,
                "video": file_record(output, repo_root),
                "directions": [
                    {
                        "relation": "LEFT",
                        "prompt": PROMPTS["left"],
                        "outcome": f"official imagined-video decode across {left['policy_request_count']} policy requests",
                    },
                    {
                        "relation": "RIGHT",
                        "prompt": PROMPTS["right"],
                        "outcome": f"official imagined-video decode across {right['policy_request_count']} policy requests",
                    },
                ],
                "selection_note": "Derived view of both complete official decodes. LEFT is displayed on the left and RIGHT on the right. No predicted frame is scored as executed behavior.",
                "source_manifest": str(manifest_path.relative_to(repo_root)),
                "composition": composition,
            }
        )

    ffmpeg_version = subprocess.run(
        [str(ffmpeg), "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    manifest = {
        "schema_version": "vla-wam-shared-v2-dreamzero-imagination-media-v1",
        "status": "complete_all_official_decodes_archived",
        "model_id": "dreamzero_droid",
        "amendment_id": "V2-A007",
        "claim_boundary": "These are official model-predicted video decodes. They are not simulator executions, task outcomes, or additional behavioral episodes.",
        "source_result": file_record(result_path, repo_root),
        "source_probe": {
            "path": str(probe_path),
            "bytes": probe_path.stat().st_size,
            "sha256": sha256(probe_path),
        },
        "archive_policy": "All nine official reset-path MP4s are copied byte-for-byte: six behavioral-session decodes and three fixed-observation probe decodes. No outcome-based selection.",
        "official_decode_count": len(archived),
        "behavioral_decode_count": sum(item["scope"] == "valid_behavioral_episode" for item in archived),
        "fixed_observation_probe_decode_count": sum(item["scope"] == "fixed_observation_diagnostic" for item in archived),
        "official_decodes": archived,
        "gallery_entries": pair_entries,
        "renderer": {
            "ffmpeg": {
                "path": str(ffmpeg),
                "bytes": ffmpeg.stat().st_size,
                "sha256": sha256(ffmpeg),
            },
            "ffmpeg_version": ffmpeg_version,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "official_decode_count": len(archived),
                "behavioral_decode_count": 6,
                "probe_decode_count": 3,
                "paired_gallery_clip_count": len(pair_entries),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
