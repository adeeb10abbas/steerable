#!/usr/bin/env python3
"""Validate the compact V3-E006-R007 closure and optionally its raw bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007/results"
TERMINAL = "r007_candidate_budget_exhausted_no_valid_state_pair"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def verify(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    require(path.is_file() and path.stat().st_size == row.get("bytes") and sha256(path) == row.get("sha256"), f"{label} binding differs")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    closure = root / CLOSURE.relative_to(ROOT)
    results = json.loads((closure / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((closure / "evidence_manifest.json").read_text(encoding="utf-8"))
    require(results.get("status") == manifest.get("status") == TERMINAL, "terminal status differs")
    require(results.get("passed") is False and results.get("accepted_candidate_rank") is None, "closure acceptance differs")
    require(results.get("model_request_count") == results.get("behavioral_episode_count") == 0, "closure counts differ")
    require(results.get("diagnostic_evaluation_count") == results.get("registered_diagnostic_budget") == 4, "diagnostic count differs")
    require(results.get("candidate_pair_evaluation_count") == results.get("registered_candidate_budget") == 4, "candidate count differs")
    require([row.get("candidate_rank") for row in results.get("candidate_attempts", [])] == [1, 2, 3, 4], "rank order differs")
    require(all(row.get("passed") is False for row in results["candidate_attempts"]), "closure contains a passing candidate")
    for attempt in results["candidate_attempts"]:
        require(set(attempt.get("stages", {})) == {"canonical_grasp", "canonical_carry"}, "stage set differs")
        for stage in attempt["stages"].values():
            require(stage.get("passed") is False, "exhausted stage unexpectedly passed")
            require(stage.get("ood_gate", {}).get("passed") is True, "OOD gate did not pass")
            require(stage.get("camera_gate_passed") is True, "camera gate did not pass")
            require(stage.get("companion_gate", {}).get("passed") is True, "companion gate did not pass")
            require(stage.get("frame_identity_passed") is True, "frame identity did not pass")
    verify(root, manifest["repo_result"], "repo result")
    verify(root, manifest["repo_target_validation_receipt"], "repo receipt")
    verify(root, manifest["decision_memo"], "decision memo")
    verify(root, manifest["closure_tool"], "closure tool")
    amendment = results.get("postexecution_validator_amendment")
    require(amendment == manifest.get("postexecution_validator_amendment"), "validator amendment differs")
    if args.verify_raw:
        for name, row in manifest["raw_evidence"].items():
            path = verify(root, row, f"raw {name}")
            if name == "failed_zero_byte_postexecution_validation":
                require(path.stat().st_size == 0, "failed wrapper receipt is not zero bytes")
    print(json.dumps({"passed": True, "results": manifest["repo_result"], "manifest_sha256": sha256(closure / "evidence_manifest.json"), "status": TERMINAL}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
