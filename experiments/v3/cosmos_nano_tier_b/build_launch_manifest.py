#!/usr/bin/env python3
"""Hash-bind the live Nano Tier-B client stack before behavioral inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (
    CONFIG,
    STUDY_ID,
    load_release,
    load_runtime,
    sha256_file,
    validate_behavioral_release_gate,
)


LIVE_FILES = (
    "experiments/v3/cosmos_nano_tier_b/live_client.py",
    "experiments/v3/cosmos_nano_tier_b/robolab_bridge.py",
    "experiments/v3/cosmos_nano_tier_b/compile_cell.py",
    "experiments/v3/cosmos_nano_tier_b/queue.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--amendment-id", choices=tuple(CONFIG), required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--behavioral-release-gate", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.study_root.resolve()
    release = load_release(root, args.amendment_id, args.release_manifest)
    runtime = load_runtime(args.runtime_manifest, study_root=root, release=release)
    validate_behavioral_release_gate(
        args.behavioral_release_gate, release=release, runtime=runtime
    )
    candidate_sha256 = release.cells[0].row["candidate_sha256"]
    if sha256_file(args.candidate) != candidate_sha256:
        raise ValueError("candidate differs from the exact released SHA-256")
    task_files = sorted({
        f"experiments/v3/prospective_tier_b_gates/task_files/"
        f"{args.amendment_id.lower().replace('-', '')}_{cell.arm}_{cell.relation}.py"
        for cell in release.cells
    })
    sources = []
    for relative in (*LIVE_FILES, *task_files):
        path = root / relative
        sources.append({
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    payload = {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-behavior-launch-v1",
        "study_id": STUDY_ID,
        "amendment_id": args.amendment_id,
        "model_id": "cosmos3_nano_policy_droid",
        "status": "behavior_launch_hash_bound_zero_cells_launched",
        "authorized_behavioral_cells": release.config["cells"],
        "launched_behavioral_cells": 0,
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "behavioral_release_gate_sha256": sha256_file(args.behavioral_release_gate),
        "candidate_sha256": candidate_sha256,
        "live_sources": sources,
        "execution_contract": "one isolated behavior-only server, one serial client, whole-seed ordering",
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
