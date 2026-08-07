#!/usr/bin/env python3
"""Release Nano Tier-B behavior only after the exact fixed-observation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (
    CONFIG,
    MODEL_ID,
    STUDY_ID,
    ContractError,
    load_json,
    load_release,
    load_runtime,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--amendment-id", choices=tuple(CONFIG), required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--fixed-observation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = load_release(args.study_root, args.amendment_id, args.release_manifest)
    runtime = load_runtime(args.runtime_manifest, study_root=args.study_root, release=release)
    fixed_path = args.fixed_observation_report.resolve()
    fixed = load_json(fixed_path, "fixed-observation report")
    expected = {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-fixed-observation-v1",
        "study_id": STUDY_ID,
        "amendment_id": args.amendment_id,
        "model_id": MODEL_ID,
        "status": "passed",
        "release_gate_passed": True,
        "probe_only": True,
        "behavioral_episode_count": 0,
        "model_request_count": len(release.config["arms"]) * 3,
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "model_blind_gate_sha256": release.config["gate_sha256"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "server_port": release.config["port"],
    }
    for key, wanted in expected.items():
        if fixed.get(key) != wanted:
            raise ContractError(f"fixed-observation report mismatch for {key}")
    if set(fixed.get("metrics", {})) != set(release.config["arms"]):
        raise ContractError("fixed-observation report arm set mismatch")
    if not all(value.get("passed") is True for value in fixed["metrics"].values()):
        raise ContractError("not every arm passed fixed-observation repeatability and sensitivity")
    payload = {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-behavioral-release-v1",
        "study_id": STUDY_ID,
        "amendment_id": args.amendment_id,
        "model_id": MODEL_ID,
        "status": "passed_behavioral_release",
        "behavioral_release": True,
        "authorized_behavioral_cells": release.config["cells"],
        "launched_behavioral_cells_at_release": 0,
        "completed_behavioral_cells_at_release": 0,
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "model_blind_gate_sha256": release.config["gate_sha256"],
        "server_port": release.config["port"],
        "fixed_observation_gate": {
            "path": str(fixed_path),
            "sha256": sha256_file(fixed_path),
            "status": "passed",
            "model_request_count": fixed["model_request_count"],
            "behavioral_episode_count": 0,
        },
        "server_isolation": "fresh dedicated process; one behavior client; no shared mutable session",
        "denominator_policy": "retain all valid behavioral failures; separate infrastructure failures and partial attempts",
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "behavioral_release": True}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

