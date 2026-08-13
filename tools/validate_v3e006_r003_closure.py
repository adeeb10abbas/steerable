#!/usr/bin/env python3
"""Validate the compact V3-E006-R003 diagnostic closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def verify(path: Path, binding: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if path.stat().st_size != binding.get("bytes") or sha256(path) != binding.get("sha256"):
        raise ValueError(f"{label} binding differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    artifact = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003"
    results_path = artifact / "results/results.json"
    memo_path = artifact / "DECISION_MEMO.md"
    manifest_path = artifact / "results/evidence_manifest.json"
    results, manifest = load(results_path), load(manifest_path)
    expected = {
        "status": "r003_known_reachable_diagnostic_failed_candidates_not_evaluated",
        "passed": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "diagnostic_evaluation_count": 1,
        "candidate_pair_evaluation_count": 0,
        "state_candidate_count": 0,
        "accepted_candidate_rank": None,
        "behavioral_activation_released": False,
        "scientific_gate_thresholds_unchanged": True,
    }
    for key, value in expected.items():
        if results.get(key) != value:
            raise ValueError(f"R003 closure field differs: {key}")
    failure = results.get("diagnostic_failure", {})
    if not (
        failure.get("diagnostic_index_one_based") == 1
        and failure.get("stage") == "canonical_grasp"
        and failure.get("source_side") == "left"
        and failure.get("position_tolerance_m_inclusive") == 0.001
        and failure.get("orientation_tolerance_deg_inclusive") == 1.0
        and failure.get("final_position_error_m") == 0.0012277967696529603
        and failure.get("final_orientation_geodesic_error_deg") == 0.2210353088374109
        and failure.get("fresh_reset_passed") is True
        and failure.get("camera_evidence_passed") is True
        and failure.get("all_base_link_to_eef_frame_identity_checks_passed") is True
        and failure.get("environment_closed_before_next_environment") is True
    ):
        raise ValueError("R003 diagnostic failure summary differs")
    verify(results_path, manifest["local_closure_files"]["results"], "closure results")
    verify(memo_path, manifest["local_closure_files"]["decision_memo"], "closure memo")
    if manifest.get("status") != "hash_closed_registered_diagnostic_failure":
        raise ValueError("R003 evidence manifest status differs")
    if manifest.get("model_request_count") != 0 or manifest.get("behavioral_episode_count") != 0:
        raise ValueError("R003 evidence manifest counts differ")
    if manifest.get("raw_evidence") != results.get("raw_evidence"):
        raise ValueError("R003 raw evidence inventories differ")
    if args.verify_raw:
        for name, row in manifest["raw_evidence"].items():
            verify(Path(str(row["path"])), row, f"raw {name}")
    print(json.dumps({
        "passed": True,
        "results": {"path": str(results_path), "bytes": results_path.stat().st_size, "sha256": sha256(results_path)},
        "manifest": {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": sha256(manifest_path)},
        "diagnostic_evaluation_count": 1,
        "candidate_pair_evaluation_count": 0,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
