#!/usr/bin/env python3
"""Recover the three allow-listed E004 post-behavior failures offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.offline_recovery import (  # noqa: E402
    CANDIDATE_SHA256,
    QUEUE_SHA256,
    REGISTRATION_SHA256,
    production_specs,
    recover_attempt,
    select_specs,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import load_runtime_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--ali-vla-wam-root", type=Path, default=Path("/data/users/ali/vla_wam"))
    parser.add_argument("--video-root", type=Path, default=Path("/data/users/ali/vla_wam/external/RoboLab-11142d4/output"))
    parser.add_argument(
        "--only",
        nargs="+",
        default=["pi05_s100_left", "dreamzero_s100_left", "nano_s100_left"],
        choices=["pi05_s100_left", "dreamzero_s100_left", "nano_s100_left"],
    )
    args = parser.parse_args()
    artifact = args.repo_root / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
    bundle = load_runtime_bundle(
        registration_path=artifact / "registration.json",
        registration_sha256=REGISTRATION_SHA256,
        queue_path=artifact / "queue.jsonl",
        queue_sha256=QUEUE_SHA256,
        candidate_path=artifact / "layout/candidate.json",
        candidate_sha256=CANDIDATE_SHA256,
    )
    results = [
        recover_attempt(bundle=bundle, spec=spec, repo_root=args.repo_root, video_root=args.video_root)
        for spec in select_specs(production_specs(args.ali_vla_wam_root), args.only)
    ]
    print(json.dumps({"status": "complete", "recoveries": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
