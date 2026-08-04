#!/usr/bin/env python3
"""Build one bounded actual-versus-prediction pair for Cosmos3 Nano V2-A011.

The source arrays, all raw rollouts, and all model futures remain on the
ali-owned PVC.  This tool emits only two selected MP4s, one poster, and a
hash-bearing manifest: a paired simulator execution and the first same-seed
LEFT/RIGHT decoded future.  The latter is visibly labelled as a model
prediction and never as execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RESULT_SCHEMA = "vla-wam-shared-v2-cosmos3-nano-policy-droid-result-v1"
RESULT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def repo_record(path: Path, repository_dir: str) -> dict[str, Any]:
    value = record(path)
    value["path"] = f"{repository_dir.rstrip('/')}/{path.name}"
    return value


def verify_source(recorded: dict[str, Any]) -> Path:
    path = Path(recorded["path"])
    observed = record(path)
    if observed["bytes"] != recorded["bytes"] or observed["sha256"] != recorded["sha256"]:
        raise RuntimeError(f"source provenance mismatch: {path}")
    return path


def episode_for(result: dict[str, Any], seed: int, relation: str) -> dict[str, Any]:
    matches = [
        row for row in result["episodes"]
        if row["environment_seed"] == seed and row["requested_relation"] == relation
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {seed}/{relation} episode, found {len(matches)}")
    return matches[0]


def draw_label(frame: np.ndarray, panel: int, title: str, subtitle: str) -> None:
    x = panel * 640 + 12
    cv2.rectangle(frame, (x - 6, 8), (x + 610, 70), (0, 0, 0), -1)
    cv2.putText(frame, title, (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (x, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_actual_pair(left_path: Path, right_path: Path, output: Path, left: dict[str, Any], right: dict[str, Any]) -> tuple[int, float]:
    captures = [cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("could not open selected simulator viewport MP4s")
    fps = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
    if not all(value > 0 for value in fps) or abs(fps[0] - fps[1]) > 1e-6:
        raise RuntimeError(f"invalid or mismatched viewport FPS: {fps}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps[0], (1280, 360))
    if not writer.isOpened():
        raise RuntimeError(f"could not open actual-video writer: {output}")
    frames: list[np.ndarray | None] = [None, None]
    ended = [False, False]
    count = 0
    try:
        while not all(ended):
            for index, capture in enumerate(captures):
                if ended[index]:
                    continue
                ok, image = capture.read()
                if ok:
                    frames[index] = image
                else:
                    ended[index] = True
            if any(image is None for image in frames):
                if any(ended):
                    raise RuntimeError("selected simulator video had no decodable frames")
                continue
            canvas = np.concatenate([
                cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA) for image in frames
            ], axis=1)
            for index, (relation, episode) in enumerate((("LEFT", left), ("RIGHT", right))):
                draw_label(
                    canvas,
                    index,
                    f"ACTUAL EXECUTION - {relation} - {'SUCCESS' if episode['requested_success'] else 'FAILURE'}",
                    f"final lateral = {episode['final_lateral_display_m']:+.3f} m",
                )
            writer.write(canvas)
            count += 1
    finally:
        for capture in captures:
            capture.release()
        writer.release()
    if not count:
        raise RuntimeError("actual pair contains zero frames")
    return count, fps[0]


def render_prediction_pair(left: np.ndarray, right: np.ndarray, output: Path, fps: float = 15.0) -> int:
    if left.shape != right.shape or left.ndim != 4 or left.shape[-1] != 3 or left.shape[0] != 33:
        raise RuntimeError(f"unexpected decoded-future shapes: {left.shape} and {right.shape}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
    if not writer.isOpened():
        raise RuntimeError(f"could not open prediction-video writer: {output}")
    try:
        for left_frame, right_frame in zip(left, right, strict=True):
            canvas = np.concatenate([
                cv2.resize(left_frame, (640, 360), interpolation=cv2.INTER_AREA),
                cv2.resize(right_frame, (640, 360), interpolation=cv2.INTER_AREA),
            ], axis=1)
            draw_label(canvas, 0, "MODEL PREDICTION - LEFT - NOT EXECUTION", "first policy request; 33 decoded RGB frames")
            draw_label(canvas, 1, "MODEL PREDICTION - RIGHT - NOT EXECUTION", "first policy request; 33 decoded RGB frames")
            writer.write(canvas)
    finally:
        writer.release()
    return int(left.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--fixed-observation", type=Path, required=True)
    parser.add_argument("--raw-layout-event", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-media-dir", required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("amendment_id") != "V2-A011"
        or result.get("checkpoint_revision") != RESULT_REVISION
        or result.get("summary", {}).get("episode_count") != 6
    ):
        raise ValueError("not the complete V2-A011 Cosmos3 Nano result")
    fixed = json.loads(args.fixed_observation.read_text())
    if fixed.get("status") != "passed" or fixed.get("checkpoint_revision") != RESULT_REVISION:
        raise ValueError("V2-A011 fixed-observation gate did not pass")
    layout = json.loads(args.raw_layout_event.read_text())
    if layout.get("status") != "measurement_only_layout_compatibility" or layout.get("behavioral_data_modified") is not False:
        raise ValueError("raw layout event is not a measurement-only compatibility record")

    left, right = episode_for(result, args.seed, "left"), episode_for(result, args.seed, "right")
    left_video = verify_source(left["executed_video"])
    right_video = verify_source(right["executed_video"])
    left_future_record = left["imagined_future_requests"][0]["decoded_future"]
    right_future_record = right["imagined_future_requests"][0]["decoded_future"]
    left_future_path = verify_source(left_future_record)
    right_future_path = verify_source(right_future_record)
    left_future = np.load(left_future_path, allow_pickle=False)
    right_future = np.load(right_future_path, allow_pickle=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual = args.output_dir / f"cosmos3_nano_v2a011_seed{args.seed}_paired_actual.mp4"
    prediction = args.output_dir / f"cosmos3_nano_v2a011_seed{args.seed}_paired_model_prediction.mp4"
    poster = args.output_dir / f"cosmos3_nano_v2a011_seed{args.seed}_paired_actual_poster.jpg"
    actual_frames, actual_fps = render_actual_pair(left_video, right_video, actual, left, right)
    prediction_frames = render_prediction_pair(left_future, right_future, prediction)
    capture = cv2.VideoCapture(str(actual))
    ok, first = capture.read()
    capture.release()
    if not ok or not cv2.imwrite(str(poster), first):
        raise RuntimeError("could not write actual-rollout poster")

    repository_dir = args.repository_media_dir.rstrip("/")
    manifest = {
        "schema_version": "vla-wam-v2a011-cosmos3-nano-policy-droid-media-v1",
        "status": "complete_selected_seed8300_actual_and_prediction_pair",
        "amendment_id": "V2-A011",
        "claim_boundary": "The actual MP4 is simulator execution. The paired prediction MP4 decodes the first returned future for the same seed and static command; it is not execution, a task outcome, or another behavioral trial.",
        "selection_rule": "Lowest environment seed with a complete valid matched LEFT/RIGHT pair: seed 8300. For bounded prediction media, select request index 0 from each paired episode.",
        "source_result": record(args.result),
        "fixed_observation_gate": record(args.fixed_observation),
        "raw_layout_compatibility_event": record(args.raw_layout_event),
        "source_videos": {"left": left["executed_video"], "right": right["executed_video"]},
        "source_decoded_futures": {"left_request_000": left_future_record, "right_request_000": right_future_record},
        "publication_video": repo_record(actual, repository_dir),
        "prediction_video": repo_record(prediction, repository_dir),
        "poster": repo_record(poster, repository_dir),
        "actual_frame_count": actual_frames,
        "actual_fps": actual_fps,
        "prediction_frame_count": prediction_frames,
        "prediction_fps": 15.0,
        "gallery_entries": [{
            "id": f"cosmos3_nano_v2a011_seed{args.seed}",
            "arena": "droid",
            "arena_label": "DROID / RoboLab",
            "model_label": "Cosmos3 Nano Policy DROID — V2-A011",
            "category": "WAM",
            "future_interface": "Actions plus decoded RGB futures; 15 futures retained for this selected pair (4 LEFT, 11 RIGHT)",
            "evidence_status": "Valid current-stack behavioral pair; both episodes succeeded; selected actual-versus-prediction comparison",
            "pair_label": f"seed {args.seed} matched pair",
            "seed": args.seed,
            "video": repo_record(actual, repository_dir),
            "poster": repo_record(poster, repository_dir),
            "comparison_media": {
                "kind": "model_prediction_not_execution",
                "label": "MODEL PREDICTION — NOT EXECUTION",
                "video": repo_record(prediction, repository_dir),
                "source_decoded_futures": {"left_request_000": left_future_record, "right_request_000": right_future_record},
                "note": "Same seed and static LEFT/RIGHT prompts; first returned decoded future from each behavioral episode."
            },
            "directions": [
                {"relation": "LEFT", "prompt": left["prompt"], "outcome": f"success after {left['actions_executed']} actions"},
                {"relation": "RIGHT", "prompt": right["prompt"], "outcome": f"success after {right['actions_executed']} actions"},
            ],
            "selection_note": "Actual execution (left) and model prediction (right) are shown together below. Both seed-8300 episodes succeeded; the prediction is a decoded future, not a rollout.",
            "source_manifest": f"{repository_dir}/media_manifest.json"
        }]
    }
    manifest_path = args.output_dir / "media_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "actual": str(actual), "prediction": str(prediction)}, sort_keys=True))


if __name__ == "__main__":
    main()
