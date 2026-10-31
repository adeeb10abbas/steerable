#!/usr/bin/env python3
"""Authenticate the declared H01 representative videos from immutable Git blobs.

This is a presentation-media package, not a timing reconstruction.  It makes no
network calls, performs no model inference, and does not materialize the videos
in the sparse worktree.  The output contains authenticated Git pointers so the
four already-tracked MP4s can be retrieved without duplicating their bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MEDIA_DIR = (
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/"
    "publication_media"
)
SOURCE_MANIFEST = f"{MEDIA_DIR}/nano_v3b001_publication_media_manifest.json"
SOURCE_EPISODES = (
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/"
    "nano_v3b001_episodes.jsonl"
)
EXPECTED_SOURCE = {
    SOURCE_MANIFEST: {
        "git_blob": "ab487db038551f52be52dcc506901c28ca1c35cd",
        "bytes": 47015,
        "sha256": "2a9023b976b6a877b2da266ce1333f6639d89a971a4e082f50a7af1f0cf246ea",
    },
    SOURCE_EPISODES: {
        "git_blob": "5c2f4a6fa4795ec054dbce2d4d68554245ceb4cb",
        "bytes": 5822218,
        "sha256": "ce67aecb2705394e8161fa4b65df7130e62648e2f6d8b6a93408c8da685a7207",
    },
}
EXPECTED_VIDEOS = {
    "v3b001:nano:seed9400:control:left": {
        "filename": "nano_v3b001_seed9400_control_left_actual_vs_local_predictions.mp4",
        "git_blob": "0f12b9e60c0697929b90b5303b77e78567095056",
        "bytes": 1662892,
        "sha256": "78e9c2111faf3f4c180724314e8d0fe8a61dbdfb78e0fefa44d34434318a0b62",
        "actions_executed": 122,
    },
    "v3b001:nano:seed9400:control:right": {
        "filename": "nano_v3b001_seed9400_control_right_actual_vs_local_predictions.mp4",
        "git_blob": "59d744da63e3bb8ff0f6ba041631cf6b5fb33e6e",
        "bytes": 4944505,
        "sha256": "67422e7d8d52dcdc74325d3d08e37cbd5b6c45bdef3721f4323625166065d84b",
        "actions_executed": 353,
    },
    "v3b001:nano:seed9400:position_mirrored:left": {
        "filename": "nano_v3b001_seed9400_position_mirrored_left_actual_vs_local_predictions.mp4",
        "git_blob": "a2a51663dbc8741a77d9b352c30e06a10032b85f",
        "bytes": 2903051,
        "sha256": "db73e5c504f6aa802008b4679165861b575b371604b84be0a6bdb12cd9e38df2",
        "actions_executed": 217,
    },
    "v3b001:nano:seed9400:position_mirrored:right": {
        "filename": "nano_v3b001_seed9400_position_mirrored_right_actual_vs_local_predictions.mp4",
        "git_blob": "1406493af421f8fdd10321fc2ebc092eef6a8a7b",
        "bytes": 2964567,
        "sha256": "296c8128b98674c94b402a56206314e48dc28e11ff62eedec55c0ea1ad5a46df",
        "actions_executed": 193,
    },
}


def git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        input=input_bytes,
    )


def authenticate_blob(root: Path, path: str, expected: dict) -> tuple[bytes, dict]:
    oid = git(root, "rev-parse", f"{PIN}:{path}").decode().strip()
    raw = git(root, "cat-file", "blob", oid)
    computed_oid = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()
    observed = {
        "path": path,
        "git_blob": oid,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if oid != computed_oid:
        raise ValueError(f"Git blob framing mismatch: {path}")
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"Pinned identity changed for {path}: {key}")
    return raw, observed


def build_package(root: Path = ROOT) -> dict:
    commit = git(root, "rev-parse", f"{PIN}^{{commit}}").decode().strip()
    if commit != PIN:
        raise ValueError("Evidence commit does not match its full pin")

    manifest_raw, manifest_identity = authenticate_blob(
        root, SOURCE_MANIFEST, EXPECTED_SOURCE[SOURCE_MANIFEST]
    )
    episodes_raw, episodes_identity = authenticate_blob(
        root, SOURCE_EPISODES, EXPECTED_SOURCE[SOURCE_EPISODES]
    )
    manifest = json.loads(manifest_raw)
    episodes = [json.loads(line) for line in episodes_raw.splitlines() if line.strip()]

    selection = manifest.get("selection", {})
    expected_rule_id = "lowest_released_seed_all_four_cells_all_exposed_predictions_v1"
    if selection.get("rule_id") != expected_rule_id:
        raise ValueError("Historical media selection rule changed")
    if selection.get("selected_seed") != 9400 or selection.get("selected_cell_count") != 4:
        raise ValueError("Historical representative block changed")
    if not (
        selection.get("outcome_used_for_selection") is False
        and selection.get("request_used_for_selection") is False
        and selection.get("no_substitution_for_missing_media") is True
    ):
        raise ValueError("Representative selection boundary changed")
    if len(episodes) != 108:
        raise ValueError("H01 episode cohort no longer contains exactly 108 rows")
    if min(row["environment_seed"] for row in episodes) != 9400:
        raise ValueError("Selected seed is no longer the minimum retained H01 seed")

    media_by_cell = {
        row["registered_cell_id"]: row for row in manifest.get("selected_media", [])
    }
    episode_by_cell = {row["registered_cell_id"]: row for row in episodes}
    if len(episode_by_cell) != len(episodes):
        raise ValueError("H01 episode table contains duplicate registered cell IDs")
    if set(media_by_cell) != set(EXPECTED_VIDEOS):
        raise ValueError("Selected media do not cover the exact four-condition block")

    videos = []
    for cell_id, expected in EXPECTED_VIDEOS.items():
        media = media_by_cell[cell_id]
        episode = episode_by_cell.get(cell_id)
        if episode is None:
            raise ValueError(f"Selected cell missing from H01 episode table: {cell_id}")
        path = f"{MEDIA_DIR}/{expected['filename']}"
        _, identity = authenticate_blob(
            root,
            path,
            {key: expected[key] for key in ("git_blob", "bytes", "sha256")},
        )
        publication = media.get("publication_video", {})
        if publication != {
            "bytes": expected["bytes"],
            "path": expected["filename"],
            "sha256": expected["sha256"],
        }:
            raise ValueError(f"Historical publication manifest differs: {cell_id}")
        if episode.get("actions_executed") != expected["actions_executed"]:
            raise ValueError(f"Episode action count changed: {cell_id}")
        if not (
            episode.get("behavioral_result_valid") is True
            and episode.get("requested_success") is True
            and episode.get("final_detached_release") is True
        ):
            raise ValueError(f"Selected episode status changed: {cell_id}")
        probe = media.get("output_validation", {})
        required_probe = {
            "codec_name": "h264",
            "pixel_format": "yuv420p",
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "has_audio": False,
        }
        if any(probe.get(key) != value for key, value in required_probe.items()):
            raise ValueError(f"Historical full-stream qualification changed: {cell_id}")
        videos.append({
            "registered_cell_id": cell_id,
            "arm": episode["phase_b_arm"],
            "requested_relation": episode["requested_relation"],
            "environment_seed": episode["environment_seed"],
            "actions_executed": episode["actions_executed"],
            "behavioral_result_valid": True,
            "requested_success": True,
            "final_detached_release": True,
            "git_object": identity,
            "historical_full_stream_qualification": {
                **required_probe,
                "frame_count": probe["frame_count"],
                "decoded_duration_seconds": probe["decoded_duration_seconds"],
                "validation_policy": probe["validation_policy"],
            },
        })

    return {
        "schema_version": "wmf-h01-representative-media-package-v1",
        "status": "authenticated_git_blob_pointers_not_worktree_copies",
        "source": {
            "commit": PIN,
            "publication_manifest": manifest_identity,
            "episode_table": episodes_identity,
        },
        "selection": {
            "rule_id": expected_rule_id,
            "rule": selection["rule"],
            "selected_seed": 9400,
            "selected_cell_count": 4,
            "all_four_registered_conditions_included": True,
            "outcome_used_for_selection": False,
            "request_used_for_selection": False,
            "no_substitution_for_missing_media": True,
            "no_outcome_hiding_audit": selection["no_outcome_hiding_audit"],
        },
        "videos": videos,
        "totals": {
            "video_count": len(videos),
            "video_bytes": sum(row["git_object"]["bytes"] for row in videos),
            "source_manifest_and_video_file_count": len(videos) + 1,
            "source_manifest_and_video_bytes": (
                manifest_identity["bytes"]
                + sum(row["git_object"]["bytes"] for row in videos)
            ),
            "new_model_requests": 0,
            "new_behavioral_episodes": 0,
            "new_human_labels": 0,
        },
        "claim_boundary": {
            "use": "Representative H01 presentation media for the four-condition minimum-seed block.",
            "not_scientific_timing_maps": True,
            "not_physical_time_alignment": True,
            "not_forecast_accuracy_evidence": True,
            "not_annotation_evidence": True,
            "does_not_complete_a5_or_any_forecast_gate": True,
            "reason": (
                "Each video juxtaposes the complete actual rollout with stitched, request-local "
                "decoded horizons; stitching does not create a continuous full-task prediction."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE / "paper/media/representative_h01_videos.json",
    )
    args = parser.parse_args()
    package = build_package()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(package, indent=2) + "\n")
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
