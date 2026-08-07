#!/usr/bin/env python3
"""Create a zero-request, whole-seed Cosmos V3-C001 bridge preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import canonical_json_bytes
from .cosmos_behavioral_contract import COSMOS_MODELS, validate_seed_block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--registration-manifest", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(sorted(COSMOS_MODELS)), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", required=True)
    args = parser.parse_args()
    report = validate_seed_block(
        study_root=args.study_root.resolve(),
        execution_plan=args.execution_plan,
        release_manifest=args.release_manifest,
        registration_manifest=args.registration_manifest,
        model_id=args.model_id,
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
