#!/usr/bin/env python3
"""Validate the compact V3-E006-R012 closure and optionally its raw bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012/results"
)
TERMINAL = "r012_candidate_budget_exhausted_no_valid_state_pair"


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
    require(
        path.is_file()
        and path.stat().st_size == row.get("bytes")
        and sha256(path) == row.get("sha256"),
        f"{label} binding differs",
    )
    return path


def gate_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    physics = state.get("physics_gate")
    require(isinstance(physics, Mapping), "raw stage lacks physics gate")
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
        "raw physics check taxonomy differs",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    closure = root / CLOSURE.relative_to(ROOT)
    results = json.loads((closure / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (closure / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    require(
        results.get("status") == manifest.get("status") == TERMINAL,
        "terminal status differs",
    )
    require(
        results.get("passed") is False
        and results.get("accepted_candidate_rank") is None
        and results.get("accepted_state_hashes") is None,
        "closure acceptance differs",
    )
    require(
        results.get("model_request_count")
        == results.get("behavioral_episode_count")
        == 0,
        "closure counts differ",
    )
    require(
        results.get("diagnostic_evaluation_count")
        == results.get("registered_diagnostic_budget")
        == 4,
        "diagnostic count differs",
    )
    require(
        results.get("candidate_pair_evaluation_count")
        == results.get("registered_candidate_budget")
        == 4,
        "candidate count differs",
    )
    require(
        [row.get("candidate_rank") for row in results.get("candidate_attempts", [])]
        == [1, 2, 3, 4],
        "rank order differs",
    )
    require(
        all(row.get("passed") is False for row in results["candidate_attempts"]),
        "closure contains a passing candidate",
    )
    for attempt in results["candidate_attempts"]:
        require(
            set(attempt.get("stages", {}))
            == {"canonical_grasp", "canonical_carry"},
            "stage set differs",
        )
        for stage in attempt["stages"].values():
            require(stage.get("passed") is False, "exhausted stage unexpectedly passed")
            require(stage.get("camera_gate_passed") is True, "camera gate did not pass")
            require(stage.get("frame_identity_passed") is True, "frame identity did not pass")
    counts = results.get("stage_gate_pass_counts", {})
    require(
        counts.get("evaluated_stage_count") == 8
        and counts.get("full_state_pass_count") == 0
        and counts.get("physics_gate_pass_count") == 0
        and counts.get("camera_gate_pass_count") == 8
        and counts.get("frame_identity_pass_count") == 8,
        "compact stage counts differ",
    )
    require(
        counts.get("physics_check_pass_counts", {}).get("normal_gripper_contact") == 2
        and counts.get("physics_check_pass_counts", {}).get(
            "intended_cube_gripper_contact_force"
        )
        == 2,
        "contact failure count differs",
    )
    verify(root, manifest["repo_result"], "repo result")
    verify(root, manifest["repo_target_validation_receipt"], "repo receipt")
    verify(root, manifest["decision_memo"], "decision memo")
    verify(root, manifest["closure_tool"], "closure tool")
    verify(root, manifest["closure_validator"], "closure validator")
    amendment = results.get("postexecution_validator_amendment")
    require(
        amendment == manifest.get("postexecution_validator_amendment"),
        "validator amendment differs",
    )
    verify(root, amendment, "postexecution validator amendment")
    if args.verify_raw:
        raw_paths = {}
        for name, row in manifest["raw_evidence"].items():
            path = verify(root, row, f"raw {name}")
            raw_paths[name] = path
            if name == "failed_zero_byte_frozen_validation":
                require(path.stat().st_size == 0, "failed frozen receipt is not zero bytes")
        correspondence = {
            "raw_result": "child_result",
            "raw_harness": "harness",
            "raw_launch": "launch",
            "raw_runtime_log": "runtime_log",
            "failed_zero_byte_frozen_validation_receipt": "failed_zero_byte_frozen_validation",
            "authoritative_target_validation_receipt": "authoritative_validation",
            "geometry_attachment_preflight": "geometry_attachment_preflight",
        }
        for result_key, manifest_key in correspondence.items():
            require(
                results.get(result_key) == manifest["raw_evidence"][manifest_key],
                f"raw cross-binding differs: {result_key}",
            )
        raw = json.loads(raw_paths["child_result"].read_text(encoding="utf-8"))
        harness = json.loads(raw_paths["harness"].read_text(encoding="utf-8"))
        authoritative = json.loads(
            raw_paths["authoritative_validation"].read_text(encoding="utf-8")
        )
        require(
            raw.get("status") == harness.get("child_status") == TERMINAL,
            "raw terminal status differs",
        )
        require(
            harness.get("process_completed") is True
            and harness.get("scientific_gate_passed") is False,
            "raw harness completion differs",
        )
        require(
            raw.get("passed") is False
            and raw.get("accepted_candidate_rank") is None
            and raw.get("accepted_states") is None,
            "raw acceptance differs",
        )
        require(
            raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0,
            "raw model/behavior counts differ",
        )
        require(
            results.get("first_passing_rule_obeyed")
            == raw.get("first_passing_rule_obeyed"),
            "first-pass rule differs",
        )
        require(
            results.get("selection_rule") == raw.get("selection_rule"),
            "selection rule differs",
        )
        diagnostics = raw.get("known_reachable_diagnostics")
        attempts = raw.get("attempts")
        require(
            isinstance(diagnostics, list)
            and len(diagnostics) == 4
            and all(row.get("passed") is True for row in diagnostics),
            "raw diagnostics differ",
        )
        require(
            isinstance(attempts, list)
            and [row.get("candidate_rank") for row in attempts] == [1, 2, 3, 4],
            "raw rank order differs",
        )
        regenerated = []
        for attempt in attempts:
            require(attempt.get("passed") is False, "raw attempt unexpectedly passed")
            stages = attempt.get("stages")
            require(isinstance(stages, Mapping), "raw stage mapping absent")
            regenerated.append(
                {
                    "candidate_rank": attempt["candidate_rank"],
                    "passed": False,
                    "stages": {
                        stage: gate_summary(stages[stage]["candidate_state"])
                        for stage in ("canonical_grasp", "canonical_carry")
                    },
                }
            )
        require(
            regenerated == results.get("candidate_attempts"),
            "compact gate summaries do not regenerate from raw",
        )
        require(
            stage_pass_counts(attempts) == results.get("stage_gate_pass_counts"),
            "compact pass counts do not regenerate from raw",
        )
        evidence = authoritative.get("candidate_evidence", {})
        require(
            authoritative.get("passed") is True and evidence.get("passed") is True,
            "authoritative receipt did not pass",
        )
        require(
            evidence.get("child_report") == manifest["raw_evidence"]["child_result"],
            "authoritative receipt binds another child",
        )
        authoritative_amendment = authoritative.get(
            "postexecution_validator_amendment", {}
        ).get("amendment", {})
        require(
            authoritative_amendment.get("bytes") == amendment.get("bytes")
            and authoritative_amendment.get("sha256") == amendment.get("sha256"),
            "authoritative amendment binding differs",
        )
        repo_receipt = verify(
            root, manifest["repo_target_validation_receipt"], "repo receipt"
        )
        require(
            repo_receipt.read_bytes()
            == raw_paths["authoritative_validation"].read_bytes(),
            "repo receipt is not authoritative receipt",
        )
        amendment_path = verify(root, amendment, "postexecution validator amendment")
        amendment_value = json.loads(amendment_path.read_text(encoding="utf-8"))
        require(
            amendment_value.get("raw_result")
            == manifest["raw_evidence"]["child_result"],
            "amendment raw result differs",
        )
        require(
            amendment_value.get("model_request_count")
            == amendment_value.get("behavioral_episode_count")
            == 0,
            "amendment counts differ",
        )
    print(
        json.dumps(
            {
                "passed": True,
                "results": manifest["repo_result"],
                "manifest_sha256": sha256(closure / "evidence_manifest.json"),
                "status": TERMINAL,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
