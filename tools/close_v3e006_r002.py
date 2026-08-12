#!/usr/bin/env python3
"""Hash-close the exhausted, zero-request V3-E006-R002 state search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r002"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path, *, relative: bool = False) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing evidence: {path}")
    name = str(path.relative_to(REPO_ROOT)) if relative else str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def copied_binding(path: Path, recorded_path: str) -> dict[str, Any]:
    row = binding(path)
    row["path"] = recorded_path
    return row


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def write_new(path: Path, value: str | Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite closure evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def solve_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attempts = result.get("attempts")
    require(isinstance(attempts, list) and len(attempts) == 8, "R002 did not evaluate eight ranks")
    for expected_rank, attempt in enumerate(attempts, 1):
        require(attempt.get("candidate_rank") == expected_rank and attempt.get("passed") is False, "R002 rank record changed")
        stages = attempt.get("stages")
        require(isinstance(stages, Mapping) and set(stages) == {"canonical_grasp", "canonical_carry"}, "R002 stage inventory changed")
        for stage in ("canonical_grasp", "canonical_carry"):
            ik = stages[stage].get("ik_solution")
            require(isinstance(ik, Mapping) and ik.get("passed") is False, "unexpected passing R002 stage")
            errors = ik.get("errors")
            require(isinstance(errors, list) and len(errors) == 240, "R002 IK trace length changed")
            best_position = min(errors, key=lambda row: float(row["position_error_m"]))
            best_orientation = min(errors, key=lambda row: float(row["orientation_geodesic_error_deg"]))
            final = errors[-1]
            rows.append(
                {
                    "candidate_rank": expected_rank,
                    "stage": stage,
                    "construction_method": attempt["construction_method"],
                    "failure_reason": ik["failure_reason"],
                    "completed_steps": ik["completed_steps"],
                    "achieved_consecutive_steps": ik["achieved_consecutive_steps"],
                    "solution_finite": ik["solution_finite"],
                    "solution_inside_soft_joint_limits": ik["solution_inside_soft_joint_limits"],
                    "target_position_world_m": ik["target"]["position_world_m"],
                    "target_quaternion_world_wxyz": ik["target"]["quaternion_world_wxyz"],
                    "minimum_position_error_m": best_position["position_error_m"],
                    "minimum_position_error_step": best_position["step_one_based"],
                    "orientation_error_at_minimum_position_deg": errors[int(best_position["step_one_based"]) - 1]["orientation_geodesic_error_deg"],
                    "minimum_orientation_error_deg": best_orientation["orientation_geodesic_error_deg"],
                    "minimum_orientation_error_step": best_orientation["step_one_based"],
                    "position_error_at_minimum_orientation_m": errors[int(best_orientation["step_one_based"]) - 1]["position_error_m"],
                    "final_position_error_m": final["position_error_m"],
                    "final_orientation_error_deg": final["orientation_geodesic_error_deg"],
                }
            )
    require(len(rows) == 16, "R002 closure does not contain sixteen stage solves")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--target-validation-receipt", type=Path, required=True)
    parser.add_argument("--target-validation-recorded-path", required=True)
    args = parser.parse_args()

    result = load(args.result)
    harness = load(args.harness)
    receipt = load(args.target_validation_receipt)
    require(result.get("status") == "r002_candidate_budget_exhausted_no_valid_state_pair", "R002 is not exhausted")
    require(result.get("passed") is False and result.get("accepted_candidate_rank") is None, "R002 acceptance changed")
    require(result.get("model_request_count") == result.get("behavioral_episode_count") == 0, "R002 was not zero-request")
    require(harness.get("status") == "completed_r002_candidate_search", "R002 harness did not complete")
    require(harness.get("process_completed") is True and harness.get("process_exit_code") == 0, "R002 child process did not close cleanly")
    require(receipt.get("passed") is True, "target raw validation did not pass")
    candidate_evidence = receipt.get("candidate_evidence", {})
    child = candidate_evidence.get("child_report", {})
    require(child.get("sha256") == sha256_file(args.result), "target receipt binds another result")
    for label, local, embedded in (
        ("result", args.result, child),
        ("harness", args.harness, candidate_evidence.get("harness", {})),
        ("launch", args.launch, candidate_evidence.get("launch", {})),
        ("runtime log", args.runtime_log, harness.get("runtime_log", {})),
    ):
        require(embedded.get("bytes") == local.stat().st_size, f"copied {label} byte count changed")
        require(embedded.get("sha256") == sha256_file(local), f"copied {label} digest changed")

    output = ARTIFACT_ROOT / "results"
    summary = {
        "schema_version": "vla-wam-shared-v3e006-r002-state-repair-closure-v1",
        "study_id": result["study_id"],
        "amendment_id": result["amendment_id"],
        "status": result["status"],
        "passed": False,
        "accepted_candidate_rank": None,
        "candidate_pair_count": 8,
        "stage_solve_count": 16,
        "first_passing_rule_obeyed": result["first_passing_rule_obeyed"],
        "selection_rule": result["selection_rule"],
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
        "scientific_gate_thresholds_unchanged": True,
        "diagnostic_scope": "All sixteen stage attempts failed at the frozen IK convergence boundary before physical/OOD/camera/companion evaluation. The retained metrics may motivate a solver-only amendment; they do not authorize threshold changes.",
        "raw_evidence": {
            "result": copied_binding(args.result, str(child["path"])),
            "harness": copied_binding(args.harness, str(candidate_evidence["harness"]["path"])),
            "launch": copied_binding(args.launch, str(candidate_evidence["launch"]["path"])),
            "runtime_log": copied_binding(args.runtime_log, str(harness["runtime_log"]["path"])),
            "target_validation_receipt": copied_binding(
                args.target_validation_receipt, args.target_validation_recorded_path
            ),
        },
        "stage_solves": solve_rows(result),
    }
    results_path = output / "results.json"
    write_new(results_path, summary)
    memo_path = output / "DECISION_MEMO.md"
    write_new(
        memo_path,
        "# V3-E006-R002 decision memo\n\n"
        "The prospective eight-pair construction budget was exhausted without a valid grasp/carry state pair. "
        "All sixteen stage solves completed 240 exact Abs-IK steps but achieved zero consecutive steps inside "
        "the registered 1 mm / 1 degree solver tolerance; no physical, OOD, camera, or companion threshold was "
        "relaxed. No model request or behavioral episode occurred, and behavioral activation remained blocked.\n\n"
        "The uniform pattern is an engineering diagnosis of the reset-to-final-target solver formulation only. "
        "It is not evidence that the unchanged scientific state gates are unattainable and is not a behavioral result.\n",
    )
    manifest_path = output / "evidence_manifest.json"
    manifest = {
        "schema_version": "vla-wam-shared-v3e006-r002-state-repair-evidence-manifest-v1",
        "status": "hash_closed_exhausted_zero_request_state_search",
        "study_id": result["study_id"],
        "amendment_id": result["amendment_id"],
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "local_artifacts": {
            "results": binding(results_path, relative=True),
            "decision_memo": binding(memo_path, relative=True),
            "closure_tool": binding(Path(__file__), relative=True),
            "registration": binding(ARTIFACT_ROOT / "repair_registration.json", relative=True),
            "candidate_schedule": binding(ARTIFACT_ROOT / "gates/candidate_schedule.json", relative=True),
            "source_push_gate": binding(ARTIFACT_ROOT / "source_push_gate.json", relative=True),
        },
        "raw_artifacts": summary["raw_evidence"],
    }
    write_new(manifest_path, manifest)
    print(json.dumps({"results": binding(results_path), "memo": binding(memo_path), "manifest": binding(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
