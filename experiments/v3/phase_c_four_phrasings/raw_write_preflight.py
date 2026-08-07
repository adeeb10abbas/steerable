#!/usr/bin/env python3
"""Exercise every V3-C001 raw-output writer without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import EXPERIMENT_ID, MODEL_CONTRACTS, canonical_json_bytes, sha256_file


class WritePreflightError(ValueError):
    """Raised when a registered output cannot be written and verified."""


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise WritePreflightError(f"empty or missing output: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run(*, model_id: str, output_dir: Path) -> dict[str, Any]:
    if model_id not in MODEL_CONTRACTS:
        raise WritePreflightError(f"unregistered model_id: {model_id}")
    if output_dir.exists():
        raise WritePreflightError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    actions_path = output_dir / "executed_actions.npy"
    np.save(actions_path, np.zeros((1, 8), dtype=np.float32), allow_pickle=False)
    state_path = output_dir / "state_trace.jsonl"
    state_path.write_bytes(canonical_json_bytes({
        "schema_version": "vla-wam-shared-v3c-write-preflight-state-v1",
        "action_step": 0,
        "model_blind_sentinel": True,
    }))
    episode_path = output_dir / "episode.jsonl"
    episode_path.write_bytes(canonical_json_bytes({
        "schema_version": "vla-wam-shared-v3c-write-preflight-episode-v1",
        "behavioral_episode": False,
        "model_request_count": 0,
    }))

    video_path = output_dir / "viewport.mp4"
    try:
        import cv2

        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64)
        )
        if not writer.isOpened():
            raise WritePreflightError("OpenCV could not open the MP4 writer")
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
    except Exception as error:
        raise WritePreflightError(f"viewport MP4 write failed: {error}") from error

    files = {
        "simulator_viewport_video": _record(video_path),
        "executed_actions": _record(actions_path),
        "state_trace": _record(state_path),
        "behavioral_jsonl": _record(episode_path),
    }
    report = {
        "schema_version": "vla-wam-shared-v3c-four-phrasings-raw-write-preflight-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "outputs": list(files),
        "files": files,
        "scope": (
            "Model-blind format/PVC write preflight. The model-specific Phase-A "
            "release proof separately establishes simulator viewport capture."
        ),
    }
    report_path = output_dir / "raw_write_preflight.json"
    report_path.write_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(model_id=args.model_id, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
