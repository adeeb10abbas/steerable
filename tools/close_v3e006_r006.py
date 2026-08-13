#!/usr/bin/env python3
"""Hash-close a passing or exhausted R006 zero-model state-search result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006/results"
TERMINAL = {
    "passed_r006_state_repair_not_released_for_behavior",
    "r006_candidate_budget_exhausted_no_valid_state_pair",
    "r006_known_reachable_diagnostic_failed_candidates_not_evaluated",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
    bound_path = str(path)
    if repo_relative:
        bound_path = str(path.relative_to(ROOT))
    return {"path": bound_path, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--target-validation-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.candidate_root.resolve()
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite R006 closure: {output}")
    launch_path = root / "launch.json"
    harness_path = root / "harness_result.json"
    runtime_path = root / "runtime.log"
    launch, harness = load(launch_path), load(harness_path)
    child_binding = harness.get("child_report")
    require(isinstance(child_binding, Mapping), "harness lacks child report")
    child_path = Path(str(child_binding["path"])).resolve()
    require(binding(child_path)["sha256"] == child_binding.get("sha256"), "child report binding changed")
    result = load(child_path)
    require(result.get("status") in TERMINAL, "R006 result is not a registered terminal result")
    require(result.get("model_request_count") == result.get("behavioral_episode_count") == 0, "R006 closure is not zero-model")
    receipt = load(args.target_validation_receipt)
    evidence = receipt.get("candidate_evidence")
    require(receipt.get("passed") is True and isinstance(evidence, Mapping) and evidence.get("passed") is True, "target raw validation did not pass")
    require(evidence.get("child_report", {}).get("sha256") == sha256_file(child_path), "target receipt binds another result")
    output.mkdir(parents=True, exist_ok=False)
    result_copy = output / "results.json"
    receipt_copy = output / "target_validation_receipt.json"
    diagnostics = result.get("known_reachable_diagnostics")
    attempts = result.get("attempts")
    require(isinstance(diagnostics, list), "R006 diagnostics are absent")
    require(isinstance(attempts, list), "R006 candidate attempts are absent")
    diagnostic_summary = [
        {
            "diagnostic_index_one_based": row.get("diagnostic_index_one_based"),
            "stage": row.get("stage"),
            "source_side": row.get("source_side"),
            "passed": row.get("passed"),
        }
        for row in diagnostics
    ]
    attempt_summary: list[dict[str, Any]] = []
    for attempt in attempts:
        stages = attempt.get("stages")
        require(isinstance(stages, Mapping), "R006 attempt lacks stages")
        stage_summary: dict[str, Any] = {}
        for stage_name, stage_row in stages.items():
            require(isinstance(stage_row, Mapping), "R006 stage row is invalid")
            state = stage_row.get("candidate_state")
            require(isinstance(state, Mapping), "R006 stage lacks candidate state")
            physics = state.get("physics_gate")
            require(isinstance(physics, Mapping), "R006 stage lacks physics gate")
            construction = state.get("construction")
            post_write = None
            if isinstance(construction, Mapping):
                atomic_write = construction.get("atomic_write")
                if isinstance(atomic_write, Mapping):
                    post_write = atomic_write.get("post_write_fk")
            stage_summary[str(stage_name)] = {
                "passed": state.get("passed"),
                "physics_gate": physics,
                "ood_gate_passed": (
                    state.get("ood_gate", {}).get("passed")
                    if isinstance(state.get("ood_gate"), Mapping)
                    else None
                ),
                "camera_gate_passed": (
                    state.get("camera_evidence", {}).get("passed")
                    if isinstance(state.get("camera_evidence"), Mapping)
                    else None
                ),
                "companion_gate_passed": (
                    state.get("companion_pose_gate", {}).get("passed")
                    if isinstance(state.get("companion_pose_gate"), Mapping)
                    else None
                ),
                "post_write_fk": post_write,
                "normalized_state_sha256": state.get("normalized_state_sha256"),
            }
        attempt_summary.append({
            "candidate_rank": attempt.get("candidate_rank"),
            "passed": attempt.get("passed"),
            "model_request_count": attempt.get("model_request_count"),
            "behavioral_episode_count": attempt.get("behavioral_episode_count"),
            "stages": stage_summary,
        })
    compact_result = {
        "schema_version": "vla-wam-shared-v3e006-r006-state-repair-closure-v2",
        "amendment_id": "V3-E006-R006",
        "status": result["status"],
        "passed": bool(result.get("passed")),
        "accepted_candidate_rank": result.get("accepted_candidate_rank"),
        "accepted_state_hashes": None,
        "registered_diagnostic_budget": int(result.get("diagnostic_budget")),
        "diagnostic_evaluation_count": len(diagnostics),
        "diagnostics_all_passed": all(row.get("passed") is True for row in diagnostics),
        "known_reachable_diagnostics": diagnostic_summary,
        "registered_candidate_budget": int(result.get("candidate_budget")),
        "candidate_pair_evaluation_count": len(attempts),
        "candidate_attempts": attempt_summary,
        "first_passing_rule_obeyed": result.get("first_passing_rule_obeyed"),
        "selection_rule": result.get("selection_rule"),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
        "release_boundary": result.get("release_boundary"),
        "raw_result": binding(child_path),
        "raw_harness": binding(harness_path),
        "raw_launch": binding(launch_path),
        "raw_runtime_log": binding(runtime_path),
        "raw_target_validation_receipt": binding(args.target_validation_receipt),
        "registration": result.get("repair_registration"),
        "candidate_schedule": result.get("candidate_schedule"),
        "source_push_gate": result.get("source_push_gate"),
        "source_commit_at_execution": launch.get("study_commit"),
    }
    result_copy.write_bytes(canonical_bytes(compact_result))
    receipt_copy.write_bytes(args.target_validation_receipt.read_bytes())
    memo = (
        "# V3-E006-R006 state-construction decision\n\n"
        f"Registered terminal status: `{result['status']}`. Passed: `{bool(result.get('passed'))}`. "
        f"Accepted candidate rank: `{result.get('accepted_candidate_rank')}`.\n\n"
        "The construction run made zero model requests and zero behavioral episodes. "
        "A passing state pair authorizes only the separately registered B001 activation; it does not itself authorize inference.\n"
    )
    memo_path = output / "DECISION_MEMO.md"
    memo_path.write_text(memo, encoding="utf-8")
    accepted = result.get("accepted_states")
    state_hashes = None
    if isinstance(accepted, Mapping):
        state_hashes = {
            stage: row["candidate_state"]["normalized_state_sha256"]
            for stage, row in accepted.items()
        }
    manifest = {
        "schema_version": "vla-wam-shared-v3e006-r006-closure-manifest-v1",
        "repair_amendment_id": "V3-E006-R006",
        "status": result["status"],
        "passed": bool(result.get("passed")),
        "accepted_candidate_rank": result.get("accepted_candidate_rank"),
        "accepted_state_hashes": state_hashes,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "repo_result": binding(result_copy, repo_relative=True),
        "repo_target_validation_receipt": binding(receipt_copy, repo_relative=True),
        "decision_memo": binding(memo_path, repo_relative=True),
        "raw_evidence": {
            "launch": binding(launch_path), "harness": binding(harness_path),
            "child_result": binding(child_path), "runtime_log": binding(runtime_path),
            "video": binding(Path(str(result["video"]["path"]))) if isinstance(result.get("video"), Mapping) else None,
        },
        "registration": binding(Path(str(result["repair_registration"]["path"]))),
        "candidate_schedule": binding(Path(str(result["candidate_schedule"]["path"]))),
        "source_push_gate": binding(Path(str(result["source_push_gate"]["path"]))),
        "closure_tool": binding(Path(__file__), repo_relative=True),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "invocation": [sys.executable, *sys.argv],
    }
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    print(json.dumps({
        "result": binding(result_copy), "receipt": binding(receipt_copy),
        "manifest": binding(manifest_path), "status": result["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
