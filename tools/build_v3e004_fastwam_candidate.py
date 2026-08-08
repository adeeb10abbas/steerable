#!/usr/bin/env python3
"""Emit the model-blind FastWAM RoboTwin V3-E004 layout candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.fastwam_robotwin import (  # noqa: E402
    candidate_payload,
    canonical_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite candidate: {output}")
    payload = canonical_json_bytes(candidate_payload())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "candidate": str(output),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "model_request_count": 0,
                "behavioral_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
