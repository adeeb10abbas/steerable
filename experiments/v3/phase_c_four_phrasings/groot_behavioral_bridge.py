#!/usr/bin/env python3
"""Fail-closed preflight entry point for the GR00T V3-C001 live bridge.

This first bridge slice validates the released model identity, the entire
eight-cell seed block, exact prompt-to-task routing (including contrastive
prompts that contain both direction words), task-source hashes, condition
order, and fresh raw destinations.  It deliberately performs zero model
requests and zero simulator episodes.  Live execution remains disabled until
the prompt-aware action/state/video writer is validated against this output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import canonical_json_bytes
from .groot_behavioral_contract import validate_seed_block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--registration-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", required=True)
    args = parser.parse_args()
    report = validate_seed_block(
        study_root=args.study_root.resolve(),
        execution_plan=args.execution_plan,
        release_manifest=args.release_manifest,
        registration_manifest=args.registration_manifest,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite retained preflight: {args.output}")
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
