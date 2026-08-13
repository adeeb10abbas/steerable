#!/usr/bin/env python3
"""Validate the hash-closed R004 construction-time-limit exhaustion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path, row: dict, label: str) -> None:
    if not path.is_file() or path.stat().st_size != row.get("bytes") or sha256(path) != row.get("sha256"):
        raise ValueError(f"{label} binding differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    base = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r004"
    results_path = base / "results/results.json"
    manifest_path = base / "results/evidence_manifest.json"
    results = json.loads(results_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "status": "r004_candidate_budget_exhausted_construction_time_limit_before_materialization",
        "passed": False, "diagnostics_all_passed": True, "diagnostic_evaluation_count": 4,
        "candidate_pair_evaluation_count": 4, "stage_solve_count": 8,
        "accepted_candidate_rank": None, "state_candidate_count": 0,
        "model_request_count": 0, "behavioral_episode_count": 0,
        "behavioral_activation_released": False, "scientific_gate_thresholds_unchanged": True,
    }
    for key, value in expected.items():
        if results.get(key) != value:
            raise ValueError(f"R004 closure field differs: {key}")
    termination = results.get("termination_signature", {})
    if termination != {
        "all_eight_stages_identical": True, "common_step_counter": 450,
        "max_episode_length": 450, "phase": "registered_waypoint_07_of_08_correction_round_01",
        "phase_step_one_based": 15, "success": False, "terminated": False,
        "time_out": True, "truncated": True,
    }:
        raise ValueError("R004 termination signature differs")
    for name, row in manifest["local_closure_files"].items():
        verify(root / row["path"], row, f"local {name}")
    if args.verify_raw:
        for name, row in manifest["raw_evidence"].items():
            verify(Path(row["path"]), row, f"raw {name}")
    print(json.dumps({"passed": True, "results_sha256": sha256(results_path),
                      "manifest_sha256": sha256(manifest_path), "raw_verified": args.verify_raw,
                      "model_request_count": 0, "behavioral_episode_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
