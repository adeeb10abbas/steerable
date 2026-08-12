#!/usr/bin/env python3
"""Validate retained V3-E006 zero-request infrastructure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_binding(binding: Mapping[str, Any], *, verify_raw: bool) -> None:
    if not {"path", "bytes", "sha256"} <= set(binding):
        raise AssertionError("incomplete evidence binding")
    if not isinstance(binding["bytes"], int) or binding["bytes"] < 0:
        raise AssertionError("invalid evidence byte count")
    if not isinstance(binding["sha256"], str) or len(binding["sha256"]) != 64:
        raise AssertionError("invalid evidence SHA-256")
    path = Path(str(binding["path"]))
    if verify_raw:
        if not path.is_file() or path.stat().st_size != binding["bytes"] or sha256(path) != binding["sha256"]:
            raise AssertionError(f"target evidence differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    artifact = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/gates/model_blind_infrastructure_invalid.jsonl"
    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2:
        raise AssertionError("expected the two retained model-blind infrastructure-invalid attempts")
    commits = set()
    for row in rows:
        if row.get("schema_version") != "vla-wam-shared-v3e006-model-blind-infrastructure-invalid-v2":
            raise AssertionError("invalid infrastructure schema")
        for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count", "state_candidate_count"):
            if row.get(key) != 0:
                raise AssertionError(f"infrastructure attempt has nonzero {key}")
        if row.get("behavioral_denominator_included") is not False or row.get("candidate_gate_passed") is not False:
            raise AssertionError("infrastructure attempt entered a scientific denominator")
        if len(row.get("invocation", {}).get("argv", [])) < 20:
            raise AssertionError("full invocation not retained")
        commits.add(row["invocation"]["study_commit"])
        for binding in row["input_bindings"].values():
            verify_binding(binding, verify_raw=args.verify_raw)
        verify_binding(row["construction_source"], verify_raw=False)
        for binding in row["raw_sources"].values():
            verify_binding(binding, verify_raw=args.verify_raw)
    if commits != {
        "011e61396d7831001e9614f3929108a0202535fc",
        "0c733d29b36dbadd4eba4009a1c3887ef50367a8",
    }:
        raise AssertionError("retained infrastructure commit set differs")

    lineage_path = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/source_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    for row in lineage["commits"]:
        completed = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{row['sha']}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode:
            raise AssertionError(f"source-lineage SHA is not a commit: {row['sha']}")
    print(json.dumps({"passed": True, "retained_attempts": len(rows), "verified_raw": args.verify_raw}, sort_keys=True))


if __name__ == "__main__":
    main()
