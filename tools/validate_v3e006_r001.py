#!/usr/bin/env python3
"""Fail-closed validator for the prospective V3-E006-R001 repair amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPAIR_REL = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r001")
ORIGINAL_REL = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006")
ORIGINAL_COMMIT = "e13e7b22b048075bf0d3cf44a892c70853ce8a7e"
STATE_CONTRACT_SHA256 = "2476b28d2867c1b87f477fd5f89e545616be00d860d4144f8cbdb70af10f3c18"
OOD_REFERENCE_SHA256 = "4df1ebf0061096a74b5eccd10b2a144e840f52fd50469b8bdae9369d1696fd04"
OOD_FREEZE_SHA256 = "4d3a02c1d96be1ddd98f47a15cf41ad1d2a6c54c3007f3846742cfcbf31873f4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> Any:
    require(path.is_file(), f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_binding(row: Mapping[str, Any], *, path: Path | None = None, ignore_path: bool = False) -> None:
    actual = Path(path if path is not None else str(row.get("path", "")))
    require(actual.is_file(), f"bound file is missing: {actual}")
    observed = binding(actual)
    require(observed["bytes"] == row.get("bytes"), f"bound bytes differ: {actual}")
    require(observed["sha256"] == row.get("sha256"), f"bound digest differs: {actual}")
    if not ignore_path:
        require(observed["path"] == row.get("path"), f"bound path differs: {actual}")


def tracked_original_paths(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", ORIGINAL_COMMIT], text=True
    ).splitlines()
    return sorted(
        path
        for path in output
        if path.startswith(f"{ORIGINAL_REL}/")
        or path.startswith("experiments/v3/phase_e/canonical_stage_localization_v3e006/")
        or path
        in {
            "tests/test_v3e006_state_contract.py",
            "tests/test_v3e006_stop_package.py",
            "tools/validate_v3e006.py",
            "tools/validate_v3e006_infrastructure_evidence.py",
        }
    )


def validate_package(root: Path) -> dict[str, Any]:
    repair_root = root / REPAIR_REL
    registration_path = repair_root / "repair_registration.json"
    closure_path = repair_root / "gates/original_v3e006_closure_binding.json"
    schedule_path = repair_root / "gates/candidate_schedule.json"
    registration = load(registration_path)
    closure = load(closure_path)
    schedule = load(schedule_path)
    require(registration.get("schema_version") == "vla-wam-shared-v3e006-r001-prospective-state-repair-registration-v1", "wrong registration schema")
    require(registration.get("repair_amendment_id") == "V3-E006-R001", "wrong repair ID")
    require(registration.get("status") == "prospectively_registered_repair_algorithm_before_any_repair_candidate_or_model_request", "wrong registration status")
    require(registration.get("registered_at_utc") == "2026-08-12T20:02:00Z", "registration timestamp differs")
    require(registration.get("counts_at_registration") == {"behavioral_episodes": 0, "model_requests": 0, "repair_candidate_evaluations": 0}, "registration counts are nonzero")
    require(registration["candidate_search"].get("algorithm_version") == "matched-direction-midpoint-search-v2", "algorithm version differs")
    require(registration["candidate_search"].get("maximum_candidate_pairs") == 8, "candidate budget differs")
    require(registration.get("supersedes_nothing") is True, "repair must not supersede original closure")
    require(registration["user_authorized_override"].get("original_stop_remains_true_for_original_candidate") is True, "original stop is not preserved")

    require(closure.get("schema_version") == "vla-wam-shared-v3e006-r001-original-closure-binding-v1", "wrong closure schema")
    require(closure.get("original_closure_commit") == ORIGINAL_COMMIT, "wrong original closure commit")
    files = closure.get("files")
    require(isinstance(files, list) and len(files) == closure.get("file_count") == 28, "closure inventory count differs")
    require([row["path"] for row in files] == tracked_original_paths(root), "closure inventory is not the exact tracked original tree")
    for row in files:
        verify_binding(row, path=root / row["path"], ignore_path=True)
    require(not subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", ORIGINAL_COMMIT, "--", str(ORIGINAL_REL), "experiments/v3/phase_e/canonical_stage_localization_v3e006", "tests/test_v3e006_state_contract.py", "tests/test_v3e006_stop_package.py", "tools/validate_v3e006.py", "tools/validate_v3e006_infrastructure_evidence.py"],
        text=True,
    ).strip(), "an original V3-E006 path was modified")

    frozen = registration["frozen_inputs"]
    verify_binding(frozen["original_closure_binding"], path=closure_path, ignore_path=True)
    for name in ("state_contract", "ood_reference", "ood_freeze", "original_results", "original_evidence", "runtime_contract", "e004_reset", "e004_candidate"):
        verify_binding(frozen[name], path=root / frozen[name]["path"], ignore_path=True)
    require(frozen["state_contract"]["sha256"] == STATE_CONTRACT_SHA256, "state contract hash differs")
    require(frozen["ood_reference"]["sha256"] == OOD_REFERENCE_SHA256, "OOD implementation hash differs")
    require(frozen["ood_freeze"]["sha256"] == OOD_FREEZE_SHA256, "OOD freeze hash differs")

    require(schedule.get("schema_version") == "vla-wam-shared-v3e006-r001-matched-direction-candidate-schedule-v1", "wrong schedule schema")
    require(schedule.get("status") == "frozen_before_any_live_repair_candidate_or_new_model_request", "wrong schedule status")
    require(schedule.get("model_request_count") == schedule.get("repair_behavioral_episode_count") == 0, "schedule counts are nonzero")
    require(schedule.get("candidate_budget") == 8, "schedule budget differs")
    require(schedule["repair_registration"]["bytes"] == registration_path.stat().st_size and schedule["repair_registration"]["sha256"] == sha256(registration_path), "schedule registration binding differs")
    require(schedule["ood_freeze"]["bytes"] == frozen["ood_freeze"]["bytes"] and schedule["ood_freeze"]["sha256"] == frozen["ood_freeze"]["sha256"], "schedule OOD binding differs")
    require(schedule["original_closure_verification"]["binding"]["bytes"] == closure_path.stat().st_size and schedule["original_closure_verification"]["binding"]["sha256"] == sha256(closure_path), "schedule closure binding differs")
    pairs = schedule.get("candidate_pairs")
    require(isinstance(pairs, list) and len(pairs) == 8, "schedule must contain eight candidate pairs")
    require([row["candidate_rank"] for row in pairs] == list(range(1, 9)), "candidate order differs")
    rank = pairs[0]
    grasp = rank["canonical_grasp"]
    carry = rank["canonical_carry"]
    require((grasp["environment_seed"], grasp["source_states"]["left"]["state_capture_index"], grasp["source_states"]["left"]["hdf5_index"], grasp["source_states"]["right"]["state_capture_index"], grasp["source_states"]["right"]["hdf5_index"]) == (9521, 30, 104, 31, 105), "rank-one grasp anchor differs")
    require((carry["environment_seed"], carry["source_states"]["left"]["state_capture_index"], carry["source_states"]["left"]["hdf5_index"], carry["source_states"]["right"]["state_capture_index"], carry["source_states"]["right"]["hdf5_index"]) == (9442, 39, 113, 38, 112), "rank-one carry anchor differs")
    for pair in pairs:
        for stage in ("canonical_grasp", "canonical_carry"):
            row = pair[stage]
            require(set(row["source_states"]) == {"left", "right"}, "candidate source is not direction-paired")
            require(row["stage"] == stage, "candidate stage differs")
            require(abs(float(row["ranking_metrics"]["absolute_midpoint_cube_y_before_zero_m"])) < 0.001, "candidate midpoint is not within registered tolerance")
            require(float(row["direction_balanced_state"]["cube_pose_world_wxyz"][1]) == 0.0, "candidate cube y is not exactly centered")
            require(row["direction_balanced_state"]["joint_velocity_rad_s"] == [0.0] * 13, "candidate joint velocity is not zero")
            require(row["direction_balanced_state"]["cube_velocity_world"] == [0.0] * 6, "candidate cube velocity is not zero")
            for direction in ("left", "right"):
                source = row["source_states"][direction]
                for key in ("raw_episode", "state_capture", "hdf5_trace"):
                    bound = source["provenance"][key]
                    require(set(bound) >= {"path", "bytes", "sha256"}, f"source {key} lacks binding")
    return {
        "passed": True,
        "repair_registration_sha256": sha256(registration_path),
        "candidate_schedule_sha256": sha256(schedule_path),
        "original_closure_binding_sha256": sha256(closure_path),
        "candidate_pairs": len(pairs),
        "original_files_verified": len(files),
    }


def validate_candidate_root(root: Path) -> dict[str, Any]:
    harness_path = root / "harness_result.json"
    launch_path = root / "launch.json"
    harness = load(harness_path)
    launch = load(launch_path)
    require(harness.get("model_request_count") == 0 and harness.get("behavioral_episode_count") == 0, "harness behavioral/model counts are nonzero")
    verify_binding(harness["launch"], path=launch_path, ignore_path=True)
    verify_binding(harness["runtime_log"], ignore_path=False)
    verify_binding(harness["child_report"], ignore_path=False)
    child = load(Path(harness["child_report"]["path"]))
    require(child.get("amendment_id") == "V3-E006-R001", "child amendment differs")
    require(child.get("model_request_count") == child.get("behavioral_episode_count") == 0, "child behavioral/model counts are nonzero")
    require(child.get("candidate_budget") == 8, "child candidate budget differs")
    attempts = child.get("attempts")
    require(isinstance(attempts, list) and 1 <= len(attempts) <= 8, "child candidate attempts differ")
    require([row["candidate_rank"] for row in attempts] == list(range(1, len(attempts) + 1)), "child candidate attempts are not sequential")
    passed_ranks = [row["candidate_rank"] for row in attempts if row["passed"]]
    require(len(passed_ranks) <= 1 and (not passed_ranks or passed_ranks == [attempts[-1]["candidate_rank"]]), "first-pass stopping rule violated")
    require(child.get("accepted_candidate_rank") == (passed_ranks[0] if passed_ranks else None), "accepted candidate rank differs")
    for attempt in attempts:
        require(attempt.get("model_request_count") == attempt.get("behavioral_episode_count") == 0, "attempt counts are nonzero")
        require(set(attempt["stages"]) == {"canonical_grasp", "canonical_carry"}, "attempt stage set differs")
        for stage in attempt["stages"].values():
            require(stage["fresh_reset"]["passed"] is True, "fresh reset gate failed")
            state = stage["candidate_state"]
            require(all(key in state for key in ("physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate", "normalized_state_sha256")), "candidate state lacks gate/hash evidence")
    return {
        "passed": True,
        "harness_sha256": sha256(harness_path),
        "child_sha256": sha256(Path(harness["child_report"]["path"])),
        "scientific_gate_passed": child.get("passed") is True,
        "accepted_candidate_rank": child.get("accepted_candidate_rank"),
        "candidate_evaluations": len(attempts),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate-root", type=Path)
    args = parser.parse_args()
    result = {"package": validate_package(args.root.resolve())}
    if args.candidate_root is not None:
        result["candidate"] = validate_candidate_root(args.candidate_root.resolve())
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"V3-E006-R001 validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
