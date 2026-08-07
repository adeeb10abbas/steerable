#!/usr/bin/env python3
"""Bind a verified Phase-A Nano runtime to one isolated Tier-B release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (
    load_release,
    sha256_file,
    write_runtime_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--amendment-id", choices=("V3-B008", "V3-B009"), required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument(
        "--base-runtime-manifest",
        type=Path,
        required=True,
        help="Verified Phase-A or V3-B005 Nano runtime identity",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = load_release(args.study_root, args.amendment_id, args.release_manifest)
    runtime = write_runtime_identity(
        study_root=args.study_root,
        release=release,
        base_runtime_manifest=args.base_runtime_manifest,
        output=args.output,
    )
    print(json.dumps({
        "path": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
