#!/usr/bin/env python3
"""Validate R010's fail-closed, pre-action geometry-oracle closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.close_v3e006_r010 import TERMINAL, summarize_preflight


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r010/results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    require(
        path.is_file()
        and path.stat().st_size == row.get("bytes")
        and sha256(path) == row.get("sha256"),
        f"{label} binding differs",
    )
    return path


def validate_compact_contract(results: Mapping[str, Any]) -> None:
    require(
        results.get("status") == TERMINAL
        and results.get("passed") is False
        and results.get("geometry_attachment_preflight_count") == 1
        and results.get("diagnostic_evaluation_count") == 0
        and results.get("candidate_pair_evaluation_count") == 0
        and results.get("accepted_candidate_rank") is None
        and results.get("accepted_state_hashes") is None,
        "R010 closure terminal/counts differ",
    )
    require(
        results.get("model_request_count") == results.get("behavioral_episode_count") == 0
        and results.get("behavioral_activation_released") is False
        and results.get("mechanically_valid_fail_closed_execution") is True
        and results.get("relative_bound_attachment_validated") is False
        and results.get("relative_bound_controller_evaluated") is False
        and results.get("intended_r010_construction_scientifically_exhausted") is False,
        "R010 closure interpretation/release differs",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    closure = root / CLOSURE.relative_to(ROOT)
    results = json.loads((closure / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((closure / "evidence_manifest.json").read_text(encoding="utf-8"))
    validate_compact_contract(results)
    require(manifest.get("status") == TERMINAL, "manifest terminal differs")
    for key, label in (
        ("repo_result", "repo results"),
        ("repo_target_validation_receipt", "repo receipt"),
        ("decision_memo", "decision memo"),
        ("closure_tool", "closure tool"),
        ("closure_validator", "closure validator"),
        ("repair_registration", "registration"),
        ("candidate_schedule", "schedule"),
        ("source_push_gate", "source gate"),
    ):
        verify(root, manifest[key], label)
    require(
        manifest.get("model_request_count") == manifest.get("behavioral_episode_count") == 0
        and manifest.get("behavioral_release_permitted") is False,
        "manifest counts/release differ",
    )
    if args.verify_raw:
        raw_paths = {
            key: verify(root, row, f"raw {key}")
            for key, row in manifest["raw_evidence"].items()
        }
        cross = {
            "raw_result": "child_result",
            "raw_preflight": "geometry_attachment_preflight",
            "raw_harness": "harness",
            "raw_launch": "launch",
            "raw_runtime_log": "runtime_log",
            "authoritative_target_validation_receipt": "authoritative_validation",
        }
        for result_key, manifest_key in cross.items():
            require(
                results.get(result_key) == manifest["raw_evidence"][manifest_key],
                f"result/manifest cross-binding differs: {result_key}",
            )
        raw = json.loads(raw_paths["child_result"].read_text(encoding="utf-8"))
        preflight = json.loads(
            raw_paths["geometry_attachment_preflight"].read_text(encoding="utf-8")
        )
        harness = json.loads(raw_paths["harness"].read_text(encoding="utf-8"))
        receipt = json.loads(
            raw_paths["authoritative_validation"].read_text(encoding="utf-8")
        )
        require(
            raw.get("status") == harness.get("child_status") == TERMINAL
            and harness.get("process_completed") is True
            and harness.get("process_exit_code") == 0
            and harness.get("scientific_gate_passed") is False,
            "raw/harness terminal differs",
        )
        require(
            raw.get("passed") is False
            and raw.get("geometry_attachment_preflight_count") == 1
            and raw.get("r010_live_diagnostic_count") == 0
            and raw.get("repair_candidate_evaluation_count") == 0
            and raw.get("state_candidate_count") == 0
            and raw.get("known_reachable_diagnostics") == []
            and raw.get("attempts") == []
            and raw.get("accepted_candidate_rank") is None
            and raw.get("accepted_states") is None
            and raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0,
            "raw pre-action counts differ",
        )
        require(raw.get("geometry_attachment_preflight") == preflight, "raw preflight copy differs")
        require(
            summarize_preflight(preflight) == results.get("geometry_oracle_finding"),
            "geometry-oracle finding does not regenerate",
        )
        evidence = receipt.get("candidate_evidence", {})
        require(
            receipt.get("passed") is True
            and evidence.get("passed") is True
            and evidence.get("geometry_attachment_preflight_passed") is False
            and evidence.get("child_report") == manifest["raw_evidence"]["child_result"]
            and evidence.get("geometry_attachment_preflight")
            == manifest["raw_evidence"]["geometry_attachment_preflight"],
            "target receipt differs",
        )
        repo_receipt = verify(root, manifest["repo_target_validation_receipt"], "repo receipt")
        require(
            repo_receipt.read_bytes() == raw_paths["authoritative_validation"].read_bytes(),
            "copied target receipt differs",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "status": TERMINAL,
                "results": manifest["repo_result"],
                "manifest_sha256": sha256(closure / "evidence_manifest.json"),
                "relative_bound_attachment_validated": False,
                "behavioral_release": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
