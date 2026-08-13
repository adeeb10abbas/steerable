#!/usr/bin/env python3
"""Hash-close the validated V3-E006-R012 zero-model state-search exhaustion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012"
)
DEFAULT_OUTPUT = ARTIFACT / "results"
TERMINAL = "r012_candidate_budget_exhausted_no_valid_state_pair"


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
        "frame_identity_passed": state.get(
            "base_link_to_eef_frame_identity", {}
        ).get("passed"),
    }


def stage_pass_counts(attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
    states = [
        attempt["stages"][stage]["candidate_state"]
        for attempt in attempts
        for stage in ("canonical_grasp", "canonical_carry")
    ]
    physics_checks = sorted(states[0]["physics_gate"]["checks"])
    require(
        all(sorted(state["physics_gate"]["checks"]) == physics_checks for state in states),
        "physics check taxonomy differs across stages",
    )
    return {
        "evaluated_stage_count": len(states),
        "full_state_pass_count": sum(state.get("passed") is True for state in states),
        "physics_gate_pass_count": sum(
            state["physics_gate"].get("passed") is True for state in states
        ),
        "ood_gate_pass_count": sum(
            state.get("ood_gate", {}).get("passed") is True for state in states
        ),
        "camera_gate_pass_count": sum(
            state.get("camera_evidence", {}).get("passed") is True for state in states
        ),
        "companion_gate_pass_count": sum(
            state.get("companion_pose_gate", {}).get("passed") is True
            for state in states
        ),
        "frame_identity_pass_count": sum(
            state.get("base_link_to_eef_frame_identity", {}).get("passed") is True
            for state in states
        ),
        "physics_check_pass_counts": {
            key: sum(
                state["physics_gate"]["checks"].get(key) is True for state in states
            )
            for key in physics_checks
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authoritative-validation-receipt", type=Path, required=True)
    parser.add_argument("--failed-zero-byte-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw_root = args.candidate_root.resolve()
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite R012 closure: {output}")

    launch_path = raw_root / "launch.json"
    harness_path = raw_root / "harness_result.json"
    runtime_path = raw_root / "runtime.log"
    harness = load(harness_path)
    child = harness.get("child_report")
    require(isinstance(child, Mapping), "harness lacks child report")
    child_path = Path(str(child["path"])).resolve()
    require(binding(child_path)["sha256"] == child.get("sha256"), "child binding changed")
    raw = load(child_path)
    require(raw.get("status") == TERMINAL and raw.get("passed") is False, "R012 terminal status differs")
    require(
        raw.get("accepted_candidate_rank") is None
        and raw.get("accepted_states") is None,
        "R012 accepted a state",
    )
    require(
        raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0,
        "R012 is not zero-model",
    )
    diagnostics = raw.get("known_reachable_diagnostics")
    attempts = raw.get("attempts")
    require(isinstance(diagnostics, list) and len(diagnostics) == 4, "R012 diagnostics differ")
    require(all(row.get("passed") is True for row in diagnostics), "R012 diagnostics did not all pass")
    require(isinstance(attempts, list) and len(attempts) == 4, "R012 candidate count differs")
    require([row.get("candidate_rank") for row in attempts] == [1, 2, 3, 4], "R012 rank order differs")
    require(all(row.get("passed") is False for row in attempts), "R012 exhaustion contains a passing pair")

    authoritative = load(args.authoritative_validation_receipt)
    evidence = authoritative.get("candidate_evidence")
    require(
        authoritative.get("passed") is True
        and isinstance(evidence, Mapping)
        and evidence.get("passed") is True,
        "authoritative raw validation failed",
    )
    require(
        evidence.get("child_report", {}).get("sha256") == sha256_file(child_path),
        "authoritative receipt binds another result",
    )
    authoritative_amendment = authoritative.get(
        "postexecution_validator_amendment", {}
    ).get("amendment")
    require(isinstance(authoritative_amendment, Mapping), "authoritative receipt lacks validator amendment")
    local_amendment_path = ARTIFACT / "postexecution_validator_amendment_v1.json"
    local_amendment = binding(local_amendment_path, repo_relative=True)
    require(
        {
            "bytes": authoritative_amendment.get("bytes"),
            "sha256": authoritative_amendment.get("sha256"),
        }
        == {"bytes": local_amendment["bytes"], "sha256": local_amendment["sha256"]},
        "authoritative receipt binds another validator amendment",
    )
    require(
        args.failed_zero_byte_receipt.stat().st_size == 0,
        "failed frozen receipt is not the retained zero-byte file",
    )

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
        "schema_version": "vla-wam-shared-v3e006-r012-state-repair-closure-v1",
        "amendment_id": "V3-E006-R012",
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
        "stage_gate_pass_counts": stage_pass_counts(attempts),
        "first_passing_rule_obeyed": raw.get("first_passing_rule_obeyed"),
        "selection_rule": raw.get("selection_rule"),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
        "raw_result": binding(child_path),
        "raw_harness": binding(harness_path),
        "raw_launch": binding(launch_path),
        "raw_runtime_log": binding(runtime_path),
        "failed_zero_byte_frozen_validation_receipt": binding(
            args.failed_zero_byte_receipt
        ),
        "authoritative_target_validation_receipt": binding(
            args.authoritative_validation_receipt
        ),
        "postexecution_validator_amendment": local_amendment,
        "registration": raw.get("repair_registration"),
        "candidate_schedule": raw.get("candidate_schedule"),
        "source_push_gate_at_execution": raw.get("source_push_gate"),
        "geometry_attachment_preflight": raw.get("geometry_attachment_preflight_receipt"),
        "source_commit_at_execution": load(launch_path).get("study_commit"),
    }
    result_path = output / "results.json"
    result_path.write_bytes(canonical_bytes(compact))
    receipt_path = output / "target_validation_receipt.json"
    receipt_path.write_bytes(args.authoritative_validation_receipt.read_bytes())
    memo_path = output / "DECISION_MEMO.md"
    counts = compact["stage_gate_pass_counts"]
    memo_path.write_text(
        "# V3-E006-R012 state-construction decision\n\n"
        "All four registered reachable-pose diagnostics passed. All four registered grasp/carry "
        "candidate pairs failed at least one unchanged gate, so no state was accepted. Across the "
        f"eight evaluated stages, physics passed {counts['physics_gate_pass_count']}/8, OOD passed "
        f"{counts['ood_gate_pass_count']}/8, camera passed {counts['camera_gate_pass_count']}/8, "
        f"companion passed {counts['companion_gate_pass_count']}/8, and frame identity passed "
        f"{counts['frame_identity_pass_count']}/8. The exact unchanged check pass counts are "
        f"{counts['physics_check_pass_counts']}.\n\n"
        "R012 made zero model requests and zero behavioral episodes. Behavioral activation remains "
        "blocked. The frozen validator initially passed the already-validated scene origin as an "
        "unsupported helper keyword. An additive post-execution amendment requires that supplied "
        "origin to equal the retained tensor-pose origin, then delegates every remaining check to "
        "the frozen validator; the untouched raw result passed target revalidation. The zero-byte failed "
        "receipt and authoritative receipt are both retained.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vla-wam-shared-v3e006-r012-closure-manifest-v1",
        "repair_amendment_id": "V3-E006-R012",
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
            "geometry_attachment_preflight": binding(
                Path(str(raw["geometry_attachment_preflight_receipt"]["path"]))
            ),
            "failed_zero_byte_frozen_validation": binding(
                args.failed_zero_byte_receipt
            ),
            "authoritative_validation": binding(
                args.authoritative_validation_receipt
            ),
        },
        "postexecution_validator_amendment": local_amendment,
        "closure_tool": binding(Path(__file__), repo_relative=True),
        "closure_validator": binding(
            ROOT / "tools/validate_v3e006_r012_closure.py", repo_relative=True
        ),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "invocation": [sys.executable, *sys.argv],
    }
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    print(
        json.dumps(
            {
                "result": binding(result_path),
                "receipt": binding(receipt_path),
                "memo": binding(memo_path),
                "manifest": binding(manifest_path),
                "status": TERMINAL,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
