#!/usr/bin/env python3
"""Verify the retained H01 publication videos without materializing them.

The verifier is deliberately separate from the package producer.  It pins the
committed compact manifest, authenticates the upstream Git blobs, streams each
MP4 through ffprobe and ffmpeg, and emits a deterministic publication-QA
receipt.  It does not create annotation media, labels, timing maps, or science.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


MEDIA_EVIDENCE_COMMIT = "34239ca102cc7b1a819e4c5c7c32b24e3bdc43f0"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
EVIDENCE_MANIFEST_PATH = (
    "workshops/corl2026_world_models/paper/media/"
    "representative_h01_videos.json"
)
EXPECTED_EVIDENCE_MANIFEST = {
    "git_blob": "cf16c8176ff66a13fc3dd21df8c7e9380d7eddcf",
    "bytes": 8060,
    "sha256": "34f626e91c44f51bdbfd73cfa4ffc48d68fe645d1e85f94e863c56b3451b12ba",
}
EXPECTED_SOURCE_COMMIT = "ce561e66f82e95055e39d3d7711691982f6b2086"
EXPECTED_SOURCE_MANIFEST = {
    "path": (
        "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/"
        "publication_media/nano_v3b001_publication_media_manifest.json"
    ),
    "git_blob": "ab487db038551f52be52dcc506901c28ca1c35cd",
    "bytes": 47015,
    "sha256": "2a9023b976b6a877b2da266ce1333f6639d89a971a4e082f50a7af1f0cf246ea",
}
EXPECTED_SELECTION_RULE = (
    "lowest_released_seed_all_four_cells_all_exposed_predictions_v1"
)
EXPECTED_VIDEO_INDEX = {
    "entry_count": 4,
    "canonical_bytes": 1224,
    "sha256": "c22cf0539603aac259d99006b4fc1fa240b277e6423dbb10e9e81586d253e87a",
}
EXPECTED_PACKAGE_INDEX = {
    "entry_count": 5,
    "canonical_bytes": 1303,
    "sha256": "e80f7eed465b8df4b479a08caaef1412c0030c2a31e2ff17e29eb6fbe9c12cfd",
}
EXPECTED_VIDEO_BYTES = 12_475_015
EXPECTED_PACKAGE_BYTES = 12_522_030


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(root), *args]
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def framed_git_sha1(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def authenticate_blob(
    root: Path,
    commit: str,
    path: str,
    expected: dict,
) -> tuple[bytes, dict]:
    oid = git(root, "rev-parse", f"{commit}:{path}").decode("ascii").strip()
    raw = git(root, "cat-file", "blob", oid)
    observed = {
        "path": path,
        "git_blob": oid,
        "bytes": len(raw),
        "sha256": sha256(raw),
    }
    require(oid == framed_git_sha1(raw), f"Git blob framing mismatch: {path}")
    for key, value in expected.items():
        require(observed.get(key) == value, f"Pinned identity changed: {path}: {key}")
    return raw, observed


def run_media_tool(command: list[str], raw: bytes, label: str) -> bytes:
    completed = subprocess.run(
        command,
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        completed.returncode == 0,
        f"{label} failed with code {completed.returncode}: "
        + completed.stderr.decode("utf-8", "replace"),
    )
    require(not completed.stderr, f"{label} emitted stderr")
    return completed.stdout


def probe_video(raw: bytes, label: str) -> dict:
    output = run_media_tool(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,pix_fmt,width,height,"
                "avg_frame_rate,r_frame_rate,nb_read_frames,duration:"
                "format=duration"
            ),
            "-of",
            "json",
            "pipe:0",
        ],
        raw,
        f"ffprobe {label}",
    )
    parsed = json.loads(output)
    streams = parsed.get("streams", [])
    videos = [row for row in streams if row.get("codec_type") == "video"]
    audios = [row for row in streams if row.get("codec_type") == "audio"]
    require(len(videos) == 1, f"{label} must contain exactly one video stream")
    require(not audios, f"{label} unexpectedly contains audio")
    require(len(streams) == 1, f"{label} unexpectedly contains another stream")
    stream = videos[0]
    frame_rate = Fraction(stream["avg_frame_rate"])
    require(frame_rate == 15, f"{label} frame rate changed")
    require(stream["r_frame_rate"] == "15/1", f"{label} nominal rate changed")
    frame_count = int(stream["nb_read_frames"])
    duration_text = stream["duration"]
    duration_microseconds = int(round(float(duration_text) * 1_000_000))
    require(
        abs(duration_microseconds - round(frame_count * 1_000_000 / 15)) <= 1,
        f"{label} frame count and duration disagree",
    )
    return {
        "codec_name": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps_numerator": frame_rate.numerator,
        "fps_denominator": frame_rate.denominator,
        "has_audio": False,
        "frame_count": frame_count,
        "duration_seconds": float(duration_text),
        "duration_microseconds": duration_microseconds,
    }


def decode_video(raw: bytes, label: str, probe: dict) -> dict:
    output = run_media_tool(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        raw,
        f"ffmpeg {label}",
    )
    progress: dict[str, str] = {}
    for line in output.decode("utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            progress[key] = value
    require(progress.get("progress") == "end", f"{label} decode did not finish")
    decoded_frames = int(progress["frame"])
    decoded_duration_microseconds = int(progress["out_time_us"])
    duplicate_frames = int(progress["dup_frames"])
    dropped_frames = int(progress["drop_frames"])
    require(decoded_frames == probe["frame_count"], f"{label} decode frame mismatch")
    require(
        decoded_duration_microseconds == probe["duration_microseconds"],
        f"{label} decode duration mismatch",
    )
    require(duplicate_frames == 0, f"{label} decode duplicated frames")
    require(dropped_frames == 0, f"{label} decode dropped frames")
    return {
        "full_stream_decode_passed": True,
        "decoded_frame_count": decoded_frames,
        "decoded_duration_microseconds": decoded_duration_microseconds,
        "duplicate_frames": duplicate_frames,
        "dropped_frames": dropped_frames,
    }


def canonical_video_index(videos: list[dict]) -> tuple[bytes, dict]:
    rows = sorted(
        (
            row["registered_cell_id"],
            row["git_object"]["path"],
            row["git_object"]["bytes"],
            row["git_object"]["git_blob"],
            row["git_object"]["sha256"],
        )
        for row in videos
    )
    raw = "".join(
        f"{cell}\0{path}\0{size}\0{oid}\0{digest}\n"
        for cell, path, size, oid, digest in rows
    ).encode("utf-8")
    return raw, {
        "canonicalization": (
            "lexicographically sorted UTF-8 "
            "cell_id NUL path NUL bytes NUL git_blob NUL sha256 LF rows"
        ),
        "entry_count": len(rows),
        "canonical_bytes": len(raw),
        "sha256": sha256(raw),
    }


def canonical_package_index(source_manifest: dict, videos: list[dict]) -> tuple[bytes, dict]:
    identities = [source_manifest] + [row["git_object"] for row in videos]
    rows = sorted(
        (
            row["path"],
            row["bytes"],
            row["git_blob"],
            row["sha256"],
        )
        for row in identities
    )
    raw = "".join(
        f"{path}\0{size}\0{oid}\0{digest}\n"
        for path, size, oid, digest in rows
    ).encode("utf-8")
    return raw, {
        "canonicalization": (
            "lexicographically sorted UTF-8 "
            "path NUL bytes NUL git_blob NUL sha256 LF rows"
        ),
        "entry_count": len(rows),
        "canonical_bytes": len(raw),
        "sha256": sha256(raw),
    }


def build_receipt(root: Path = ROOT) -> dict:
    evidence_commit = git(
        root, "rev-parse", f"{MEDIA_EVIDENCE_COMMIT}^{{commit}}"
    ).decode("ascii").strip()
    require(evidence_commit == MEDIA_EVIDENCE_COMMIT, "Media evidence pin changed")
    manifest_raw, evidence_manifest_identity = authenticate_blob(
        root,
        MEDIA_EVIDENCE_COMMIT,
        EVIDENCE_MANIFEST_PATH,
        EXPECTED_EVIDENCE_MANIFEST,
    )
    package = json.loads(manifest_raw)
    require(package["source"]["commit"] == EXPECTED_SOURCE_COMMIT, "Source pin changed")
    require(
        package["selection"]["rule_id"] == EXPECTED_SELECTION_RULE,
        "Selection rule changed",
    )
    require(package["selection"]["selected_seed"] == 9400, "Selected seed changed")
    require(package["selection"]["selected_cell_count"] == 4, "Cell count changed")
    require(
        package["selection"]["outcome_used_for_selection"] is False
        and package["selection"]["request_used_for_selection"] is False
        and package["selection"]["no_substitution_for_missing_media"] is True,
        "Selection boundary changed",
    )

    source_expected = package["source"]["publication_manifest"]
    require(source_expected == EXPECTED_SOURCE_MANIFEST, "Source manifest identity changed")
    _, source_manifest_identity = authenticate_blob(
        root,
        EXPECTED_SOURCE_COMMIT,
        source_expected["path"],
        source_expected,
    )

    videos = []
    for declared in package["videos"]:
        cell_id = declared["registered_cell_id"]
        expected = declared["git_object"]
        raw, identity = authenticate_blob(
            root,
            EXPECTED_SOURCE_COMMIT,
            expected["path"],
            expected,
        )
        probe = probe_video(raw, cell_id)
        historical = declared["historical_full_stream_qualification"]
        require(
            {
                "codec_name": probe["codec_name"],
                "pixel_format": probe["pixel_format"],
                "width": probe["width"],
                "height": probe["height"],
                "fps": probe["fps_numerator"] / probe["fps_denominator"],
                "has_audio": probe["has_audio"],
                "frame_count": probe["frame_count"],
                "decoded_duration_seconds": probe["duration_seconds"],
            }
            == {
                key: historical[key]
                for key in (
                    "codec_name",
                    "pixel_format",
                    "width",
                    "height",
                    "fps",
                    "has_audio",
                    "frame_count",
                    "decoded_duration_seconds",
                )
            },
            f"Current decode differs from historical qualification: {cell_id}",
        )
        decode = decode_video(raw, cell_id, probe)
        videos.append(
            {
                "registered_cell_id": cell_id,
                "git_object": identity,
                "ffprobe": probe,
                "ffmpeg": decode,
            }
        )
    videos.sort(key=lambda row: row["registered_cell_id"])

    _, video_index = canonical_video_index(videos)
    _, package_index = canonical_package_index(source_manifest_identity, videos)
    require(
        all(video_index[key] == value for key, value in EXPECTED_VIDEO_INDEX.items()),
        "Video canonical index changed",
    )
    require(
        all(package_index[key] == value for key, value in EXPECTED_PACKAGE_INDEX.items()),
        "Package canonical index changed",
    )
    video_bytes = sum(row["git_object"]["bytes"] for row in videos)
    package_bytes = source_manifest_identity["bytes"] + video_bytes
    require(video_bytes == EXPECTED_VIDEO_BYTES, "Video byte total changed")
    require(package_bytes == EXPECTED_PACKAGE_BYTES, "Package byte total changed")

    audit = package["selection"]["no_outcome_hiding_audit"]
    condition_outcomes = audit["full_cohort_condition_outcomes"]
    full_episodes = sum(row["episodes"] for row in condition_outcomes.values())
    full_successes = sum(row["successes"] for row in condition_outcomes.values())
    full_failures = full_episodes - full_successes
    require(
        (full_episodes, full_successes, full_failures) == (108, 102, 6),
        "Cohort totals changed",
    )
    require(
        (audit["selected_successes"], audit["selected_failures"]) == (4, 0),
        "Selected outcome disclosure changed",
    )

    return {
        "schema_version": "wmf-h01-representative-media-verification-v1",
        "status": "cached_git_blobs_authenticated_and_full_stream_decoded",
        "basis": {
            "media_evidence_commit": MEDIA_EVIDENCE_COMMIT,
            "evidence_manifest": evidence_manifest_identity,
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "source_publication_manifest": source_manifest_identity,
        },
        "selection_disclosure": {
            "rule_id": EXPECTED_SELECTION_RULE,
            "selected_seed": 9400,
            "selected_cell_count": 4,
            "outcome_used_for_selection": False,
            "request_used_for_selection": False,
            "substitution_for_missing_media": False,
            "selected_successes": 4,
            "selected_failures": 0,
            "full_cohort_episodes": full_episodes,
            "full_cohort_successes": full_successes,
            "full_cohort_failures": full_failures,
        },
        "videos": videos,
        "canonical_indices": {
            "videos": video_index,
            "source_manifest_and_videos": package_index,
        },
        "totals": {
            "video_count": len(videos),
            "video_bytes": video_bytes,
            "source_manifest_and_video_file_count": len(videos) + 1,
            "source_manifest_and_video_bytes": package_bytes,
            "evidence_manifest_files_authenticated": 1,
            "source_publication_manifest_files_authenticated": 1,
            "video_git_blobs_authenticated": len(videos),
            "video_full_stream_decodes_passed": len(videos),
            "materialized_video_files": 0,
        },
        "zero_activity": {
            "annotation_media_created": 0,
            "human_responses_created": 0,
            "labels_created": 0,
            "request_selections_created": 0,
            "forecast_accuracy_estimates_created": 0,
            "model_requests_issued": 0,
            "behavioral_episodes_launched": 0,
        },
        "claim_boundary": {
            "use": "Current cached-byte and full-decode publication-media QA only.",
            "human_visual_review_performed": False,
            "not_annotation_media": True,
            "not_annotation_evidence": True,
            "not_human_label_evidence": True,
            "not_scientific_timing_maps": True,
            "not_physical_time_alignment": True,
            "not_forecast_accuracy_evidence": True,
            "not_outcome_coverage_evidence": True,
            "does_not_advance_any_science_or_release_gate": True,
            "does_not_verify_remote_raw_or_pvc_availability": True,
            "cache_presence_is_point_in_time": True,
            "annotation_exclusion_reason": (
                "The composites juxtapose actual and predicted panels with visible semantic "
                "labels, so they violate the source-free blinded-rater input contract."
            ),
            "timing_exclusion_reason": (
                "The right panel stitches request-local decoded horizons; frame counts and "
                "playback duration do not create a physical-time or action-index mapping."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE / "paper/media/representative_h01_video_verification.json",
    )
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
