#!/usr/bin/env python3
"""Emit the model-blind seven-scene V3-E005 layout candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.cross_arena_geometry_v3e005.runtime_contract import (  # noqa: E402
    QUEUE_SHA256,
    REGISTRATION_SHA256,
    load_registered_bundle,
)
from experiments.v3.phase_e.cross_arena_geometry_v3e005.scene_contract import (  # noqa: E402
    candidate_payload,
    canonical_json_bytes,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/layout/scene_candidate.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    study_root = args.study_root.expanduser().resolve()
    bundle = load_registered_bundle(study_root)
    if bundle.registration_sha256 != REGISTRATION_SHA256:
        raise SystemExit("registration SHA-256 drift")
    if bundle.queue_sha256 != QUEUE_SHA256:
        raise SystemExit("queue SHA-256 drift")
    output = args.output.expanduser().resolve()
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
                "registration_sha256": bundle.registration_sha256,
                "queue_sha256": bundle.queue_sha256,
                "scene_count": 7,
                "registered_cell_count": len(bundle.cells),
                "model_request_count": 0,
                "model_action_request_count": 0,
                "behavioral_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
