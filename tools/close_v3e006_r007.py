#!/usr/bin/env python3
"""Hash-close the validated V3-E006-R007 zero-model state-search exhaustion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007"
    / "results"
)
TERMINAL = "r007_candidate_budget_exhausted_no_valid_state_pair"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


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


def binding(path: Path, *, repo_relative: bool = False) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing bound file: {path}")
    bound_path = str(path.relative_to(ROOT)) if repo_relative else str(path)
    return {
        "path": bound_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def gate_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    physics = state.get("physics_gate")
    require(isinstance(physics, Mapping), "stage lacks physics gate")
    return {
        "passed": state.get("passed"),
        "normalized_state_sha256": state.get("normalized_state_sha256"),
        "physics_gate": physics,
        "ood_gate": state.get("ood_gate"),
        "camera_gate_passed": state.get("camera_evidence", {}).get("passed"),
        "companion_gate": state.get("companion_pose_gate"),
        "frame_identity_passed": state.get("base_link_to_eef_frame_identity", {}).get("passed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authoritative-validation-receipt", type=Path, required=True)
    parser.add_argument("--interim-validation-receipt", type=Path, required=True)
    parser.add_argument("--failed-zero-byte-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw_root = args.candidate_root.resolve()
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite R007 closure: {output}")

    launch_path = raw_root / "launch.json"
    harness_path = raw_root / "harness_result.json"
    runtime_path = raw_root / "runtime.log"
    harness = load(harness_path)
    child = harness.get("child_report")
    require(isinstance(child, Mapping), "harness lacks child report")
    child_path = Path(str(child["path"])).resolve()
    require(binding(child_path)["sha256"] == child.get("sha256"), "child binding changed")
    raw = load(child_path)
    require(raw.get("status") == TERMINAL and raw.get("passed") is False, "R007 terminal status differs")
    require(raw.get("accepted_candidate_rank") is None and raw.get("accepted_states") is None, "R007 accepted a state")
    require(raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0, "R007 is not zero-model")
    diagnostics = raw.get("known_reachable_diagnostics")
    attempts = raw.get("attempts")
    require(isinstance(diagnostics, list) and len(diagnostics) == 4, "R007 diagnostics differ")
    require(all(row.get("passed") is True for row in diagnostics), "R007 diagnostics did not all pass")
    require(isinstance(attempts, list) and len(attempts) == 4, "R007 candidate count differs")
    require([row.get("candidate_rank") for row in attempts] == [1, 2, 3, 4], "R007 rank order differs")
    require(all(row.get("passed") is False for row in attempts), "R007 exhaustion contains a passing pair")

    authoritative = load(args.authoritative_validation_receipt)
    evidence = authoritative.get("candidate_evidence")
    require(authoritative.get("passed") is True and isinstance(evidence, Mapping) and evidence.get("passed") is True, "authoritative raw validation failed")
    require(evidence.get("child_report", {}).get("sha256") == sha256_file(child_path), "authoritative receipt binds another result")
    amendment = authoritative.get("postexecution_validator_amendment", {}).get("amendment")
    require(isinstance(amendment, Mapping), "authoritative receipt lacks validator amendment")
    interim = load(args.interim_validation_receipt)
    require(interim.get("passed") is True, "interim validation did not pass")
    require(args.failed_zero_byte_receipt.stat().st_size == 0, "failed wrapper receipt is not the retained zero-byte file")

    summaries = []
    for attempt in attempts:
        stages = attempt.get("stages")
        require(isinstance(stages, Mapping), "candidate pair lacks stages")
        summaries.append(
            {
                "candidate_rank": attempt["candidate_rank"],
                "passed": False,
                "stages": {
                    stage: gate_summary(stages[stage]["candidate_state"])
                    for stage in ("canonical_grasp", "canonical_carry")
                },
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    compact = {
        "schema_version": "vla-wam-shared-v3e006-r007-state-repair-closure-v1",
        "amendment_id": "V3-E006-R007",
        "status": TERMINAL,
        "passed": False,
        "accepted_candidate_rank": None,
        "accepted_state_hashes": None,
        "registered_diagnostic_budget": 4,
        "diagnostic_evaluation_count": 4,
        "diagnostics_all_passed": True,
        "registered_candidate_budget": 4,
        "candidate_pair_evaluation_count": 4,
        "candidate_attempts": summaries,
        "first_passing_rule_obeyed": raw.get("first_passing_rule_obeyed"),
        "selection_rule": raw.get("selection_rule"),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
        "raw_result": binding(child_path),
        "raw_harness": binding(harness_path),
        "raw_launch": binding(launch_path),
        "raw_runtime_log": binding(runtime_path),
        "interim_target_validation_receipt": binding(args.interim_validation_receipt),
        "failed_zero_byte_postexecution_receipt": binding(args.failed_zero_byte_receipt),
        "authoritative_target_validation_receipt": binding(args.authoritative_validation_receipt),
        "postexecution_validator_amendment": amendment,
        "registration": raw.get("repair_registration"),
        "candidate_schedule": raw.get("candidate_schedule"),
        "source_push_gate_at_execution": raw.get("source_push_gate"),
        "source_commit_at_execution": load(launch_path).get("study_commit"),
    }
    result_path = output / "results.json"
    result_path.write_bytes(canonical_bytes(compact))
    receipt_path = output / "target_validation_receipt.json"
    receipt_path.write_bytes(args.authoritative_validation_receipt.read_bytes())
    memo_path = output / "DECISION_MEMO.md"
    memo_path.write_text(
        "# V3-E006-R007 state-construction decision\n\n"
        "The four registered reachable-pose diagnostics passed. All four registered grasp/carry "
        "candidate pairs failed at least one unchanged physics gate; no state was accepted. "
        "OOD, camera, companion, and frame gates passed for all eight evaluated stages.\n\n"
        "R007 made zero model requests and zero behavioral episodes. Behavioral activation remains blocked. "
        "The first validator invocation rejected a redundant retained `candidate_rank` annotation. An additive, "
        "post-execution validator amendment required that exact annotation and delegated every remaining check "
        "to the frozen validator; the untouched raw result then passed target revalidation. Both the interim "
        "receipt and the zero-byte failed wrapper receipt are retained alongside the authoritative receipt.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vla-wam-shared-v3e006-r007-closure-manifest-v1",
        "repair_amendment_id": "V3-E006-R007",
        "status": TERMINAL,
        "passed": False,
        "accepted_candidate_rank": None,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "repo_result": binding(result_path, repo_relative=True),
        "repo_target_validation_receipt": binding(receipt_path, repo_relative=True),
        "decision_memo": binding(memo_path, repo_relative=True),
        "raw_evidence": {
            "launch": binding(launch_path),
            "harness": binding(harness_path),
            "child_result": binding(child_path),
            "runtime_log": binding(runtime_path),
            "interim_validation": binding(args.interim_validation_receipt),
            "failed_zero_byte_postexecution_validation": binding(args.failed_zero_byte_receipt),
            "authoritative_validation": binding(args.authoritative_validation_receipt),
        },
        "postexecution_validator_amendment": amendment,
        "closure_tool": binding(Path(__file__), repo_relative=True),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "invocation": [sys.executable, *sys.argv],
    }
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    print(json.dumps({
        "result": binding(result_path),
        "receipt": binding(receipt_path),
        "memo": binding(memo_path),
        "manifest": binding(manifest_path),
        "status": TERMINAL,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
