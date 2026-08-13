#!/usr/bin/env python3
"""Validate the compact, hash-bound V3-E006-R005 exhaustion closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e"
    / "canonical_stage_localization_v3e006_r005/results"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def verify_binding(record: Mapping[str, Any], *, verify_raw: bool) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        require(not verify_raw and Path(str(record.get("path", ""))).is_absolute(), f"missing binding: {path}")
        return path
    require(path.stat().st_size == record.get("bytes"), f"binding bytes changed: {path}")
    require(sha256_file(path) == record.get("sha256"), f"binding SHA changed: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    result_path = args.closure_root / "results.json"
    manifest_path = args.closure_root / "evidence_manifest.json"
    receipt_path = args.closure_root / "target_validation_receipt.json"
    memo_path = args.closure_root / "DECISION_MEMO.md"
    result = load(result_path)
    manifest = load(manifest_path)
    receipt = load(receipt_path)
    require(result.get("schema_version") == "vla-wam-shared-v3e006-r005-state-repair-closure-v2", "closure schema changed")
    require(result.get("status") == "r005_candidate_budget_exhausted_no_valid_state_pair", "closure status changed")
    require(result.get("passed") is False and result.get("accepted_candidate_rank") is None, "exhaustion acceptance changed")
    require(result.get("registered_diagnostic_budget") == result.get("diagnostic_evaluation_count") == 4, "diagnostic count changed")
    require(result.get("diagnostics_all_passed") is True and len(result.get("known_reachable_diagnostics", [])) == 4, "diagnostics differ")
    require(result.get("registered_candidate_budget") == result.get("candidate_pair_evaluation_count") == 4, "candidate count changed")
    attempts = result.get("candidate_attempts")
    require(isinstance(attempts, list) and [row.get("candidate_rank") for row in attempts] == [1, 2, 3, 4], "candidate rank order changed")
    for row in attempts:
        require(row.get("passed") is False, "exhausted rank unexpectedly passed")
        require(row.get("model_request_count") == row.get("behavioral_episode_count") == 0, "rank is not zero-model")
        stages = row.get("stages")
        require(isinstance(stages, Mapping) and set(stages) == {"canonical_grasp", "canonical_carry"}, "stage inventory changed")
        for state in stages.values():
            require(isinstance(state, Mapping), "stage summary is invalid")
            require(state.get("passed") is False, "exhausted stage unexpectedly passed")
            physics = state.get("physics_gate")
            require(isinstance(physics, Mapping) and physics.get("passed") is False, "stage physics outcome changed")
            require(state.get("ood_gate_passed") is True, "stage OOD gate changed")
            require(state.get("camera_gate_passed") is True, "stage camera gate changed")
            require(state.get("companion_gate_passed") is True, "stage companion gate changed")
    require(result.get("first_passing_rule_obeyed") is True, "selection rule changed")
    require(result.get("model_request_count") == result.get("behavioral_episode_count") == 0, "closure is not zero-model")
    require(result.get("behavioral_activation_released") is False, "behavior was released")
    require(receipt.get("passed") is True, "target validation receipt did not pass")
    require(manifest.get("status") == result["status"], "manifest status differs")
    require(manifest.get("passed") is False and manifest.get("accepted_candidate_rank") is None, "manifest acceptance differs")
    require(manifest.get("model_request_count") == manifest.get("behavioral_episode_count") == 0, "manifest is not zero-model")
    for name in ("repo_result", "repo_target_validation_receipt", "decision_memo", "closure_tool"):
        verify_binding(manifest[name], verify_raw=True)
    for record in manifest.get("raw_evidence", {}).values():
        if record is not None:
            verify_binding(record, verify_raw=args.verify_raw)
    for name in ("raw_result", "raw_harness", "raw_launch", "raw_runtime_log", "raw_target_validation_receipt"):
        verify_binding(result[name], verify_raw=args.verify_raw)
    require(memo_path.is_file() and "zero model requests" in memo_path.read_text(encoding="utf-8"), "decision memo changed")
    print(json.dumps({
        "passed": True,
        "status": result["status"],
        "candidate_pair_count": len(attempts),
        "result_sha256": sha256_file(result_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
