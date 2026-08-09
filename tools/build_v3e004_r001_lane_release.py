#!/usr/bin/env python3
"""Bind a passed LEFT-capture/RIGHT-replay preflight into an E004 lane release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.request0_replay import (  # noqa: E402
    LANE_PREFLIGHT_SCHEMA,
    file_record,
    load_amendment,
    validate_lane_preflight,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (  # noqa: E402
    load_runtime_bundle,
    sha256_file,
    validate_lane_release,
)


def _load(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("registration", "queue", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--base-lane-release", type=Path, required=True)
    parser.add_argument("--base-lane-release-sha256", required=True)
    parser.add_argument("--request0-replay-amendment", type=Path, required=True)
    parser.add_argument("--request0-replay-amendment-sha256", required=True)
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_runtime_bundle(
        registration_path=args.registration,
        registration_sha256=args.registration_sha256,
        queue_path=args.queue,
        queue_sha256=args.queue_sha256,
        candidate_path=args.candidate,
        candidate_sha256=args.candidate_sha256,
    )
    release = validate_lane_release(
        args.base_lane_release,
        args.base_lane_release_sha256,
        bundle=bundle,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    load_amendment(
        args.request0_replay_amendment,
        args.request0_replay_amendment_sha256,
    )
    left, right = _load(args.left_report), _load(args.right_report)
    left_identity = left.get("request0_replay", {}).get("pair_identity_sha256")
    right_identity = right.get("request0_replay", {}).get("pair_identity_sha256")
    if not isinstance(left_identity, str) or left_identity != right_identity:
        raise ValueError("LEFT/RIGHT preflight pair identities differ")
    output = dict(release)
    output["request0_replay_preflight"] = {
        "schema_version": LANE_PREFLIGHT_SCHEMA,
        "amendment": file_record(args.request0_replay_amendment),
        "left_report": file_record(args.left_report),
        "right_report": file_record(args.right_report),
        "pair_identity_sha256": left_identity,
    }
    output["gates"] = {
        **release["gates"],
        "request0_left_capture_right_replay_preflight": True,
    }
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite lane release: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_lane_preflight(
        output,
        amendment_sha256=args.request0_replay_amendment_sha256,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "pair_identity_sha256": left_identity,
                "model_request_count": 0,
                "behavioral_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
