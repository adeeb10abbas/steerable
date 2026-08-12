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
ATTEMPT01_BINDINGS = {
    "launch": (16931, "94bb3b07c824665b35fdfe16c73a801ecd8c78e8fd0cfe5b4a609f425f3e80c2"),
    "harness_result": (1482, "b96715b562d0e1865fe71f3249dc69fb92fb00ec722781f7bc5561f07f6eb64f"),
    "runtime_log": (21997, "cc188cfb6edc86f423e0cecd6c27b9ce1acf23240a7be85d3d9e511fa8503363"),
    "child_failure": (61209, "ad0dad25fc081bcadcf888a2c3c057546948216ec7abac6313e507452fa556dd"),
}


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

    lifecycle_path = repair_root / "gates/runtime_lifecycle_repair_v1.json"
    lifecycle = load(lifecycle_path)
    require(
        lifecycle.get("schema_version")
        == "vla-wam-shared-v3e006-r001-prospective-runtime-lifecycle-repair-v1",
        "wrong runtime lifecycle repair schema",
    )
    require(
        lifecycle.get("status")
        == "prospectively_frozen_after_infrastructure_invalid_attempt01_before_any_retry_or_model_request",
        "runtime lifecycle repair was not frozen prospectively",
    )
    require(
        lifecycle.get("counts_at_freeze")
        == {
            "model_requests": 0,
            "behavioral_episodes": 0,
            "accepted_state_candidates": 0,
            "completed_candidate_pairs": 0,
            "infrastructure_invalid_search_attempts": 1,
        },
        "runtime lifecycle repair counts differ",
    )
    lifecycle_frozen = lifecycle["frozen_scientific_contracts"]
    require(lifecycle_frozen.get("repair_registration_sha256") == sha256(registration_path), "lifecycle repair registration binding differs")
    require(lifecycle_frozen.get("candidate_schedule_sha256") == sha256(schedule_path), "lifecycle repair schedule binding differs")
    require(lifecycle_frozen.get("original_closure_binding_sha256") == sha256(closure_path), "lifecycle repair closure binding differs")
    require(lifecycle_frozen.get("state_contract_sha256") == STATE_CONTRACT_SHA256, "lifecycle repair state gate differs")
    require(lifecycle_frozen.get("ood_reference_sha256") == OOD_REFERENCE_SHA256, "lifecycle repair OOD code differs")
    require(lifecycle_frozen.get("ood_freeze_sha256") == OOD_FREEZE_SHA256, "lifecycle repair OOD freeze differs")
    require(lifecycle_frozen.get("candidate_budget") == 8, "lifecycle repair candidate budget differs")

    infrastructure_path = repair_root / "infrastructure_attempts.jsonl"
    infrastructure_rows = [
        json.loads(line)
        for line in infrastructure_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(infrastructure_rows) == 1, "R001 infrastructure ledger must retain exactly attempt01 before retry")
    infrastructure = infrastructure_rows[0]
    require(infrastructure.get("attempt_id") == "v3e006-r001-state-repair-attempt01", "wrong R001 infrastructure attempt ID")
    require(infrastructure.get("model_request_count") == infrastructure.get("behavioral_episode_count") == 0, "attempt01 counts are nonzero")
    require(infrastructure.get("state_candidate_count") == infrastructure.get("completed_candidate_pair_count") == 0, "attempt01 was incorrectly counted as a candidate")
    require(infrastructure.get("behavioral_denominator_included") is False and infrastructure.get("candidate_denominator_included") is False, "attempt01 entered a denominator")
    require(infrastructure.get("harness_misclassification", {}).get("scientific_completion") is False, "attempt01 harness defect was not disclosed")
    for name, (expected_bytes, expected_sha) in ATTEMPT01_BINDINGS.items():
        row = infrastructure["raw_bindings"][name]
        require(row.get("bytes") == expected_bytes and row.get("sha256") == expected_sha, f"attempt01 {name} binding differs")

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
    v5_path = repair_root / "source_push_gate_v5.json"
    source_gate_path = v5_path if v5_path.is_file() else repair_root / "source_push_gate_v4.json"
    source_gate_result = None
    if source_gate_path.is_file():
        source_gate = load(source_gate_path)
        is_lifecycle_gate = source_gate_path == v5_path
        require(source_gate.get("schema_version") == "vla-wam-shared-v3e006-r001-source-push-gate-v1", "wrong source-push schema")
        if is_lifecycle_gate:
            require(source_gate.get("status") == "passed_after_attempt01_infrastructure_invalid_before_single_retry_or_model_request", "lifecycle source-push gate did not pass")
            require(source_gate.get("model_request_count") == source_gate.get("behavioral_episode_count") == 0, "lifecycle source-push model/behavior counts are nonzero")
            require(source_gate.get("repair_candidate_evaluation_count") == source_gate.get("infrastructure_invalid_search_attempt_count") == 1, "lifecycle source-push attempt history differs")
            require(source_gate.get("completed_candidate_pair_count") == source_gate.get("accepted_state_candidate_count") == 0, "lifecycle source-push incorrectly counts a completed/accepted candidate")
        else:
            require(source_gate.get("status") == "passed_before_any_r001_candidate_or_model_request", "source-push gate did not pass")
            require(source_gate.get("model_request_count") == source_gate.get("behavioral_episode_count") == source_gate.get("repair_candidate_evaluation_count") == 0, "source-push counts are nonzero")
        implementation_commit = str(source_gate.get("implementation_commit", ""))
        require(bool(implementation_commit), "source-push implementation commit is absent")
        subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{implementation_commit}^{{commit}}"], check=True)
        inventory = source_gate.get("implementation_files")
        require(isinstance(inventory, list) and inventory, "source-push implementation inventory is absent")
        for row in inventory:
            relative = Path(str(row.get("path", "")))
            require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe source-push path: {relative}")
            verify_binding(row, path=root / relative, ignore_path=True)
        prior_gate_path = repair_root / ("source_push_gate_v4.json" if is_lifecycle_gate else "source_push_gate_v3.json")
        prior_gate = source_gate.get("supersedes")
        require(isinstance(prior_gate, Mapping), "latest source-push gate lacks supersession binding")
        verify_binding(prior_gate, path=prior_gate_path, ignore_path=True)
        prior_value = load(prior_gate_path)
        require(prior_value.get("status") == "passed_before_any_r001_candidate_or_model_request", "superseded source-push gate did not pass")
        require(prior_value.get("model_request_count") == prior_value.get("behavioral_episode_count") == prior_value.get("repair_candidate_evaluation_count") == 0, "superseded source-push counts are nonzero")
        prior_commit = str(prior_value.get("implementation_commit", ""))
        require(bool(prior_commit), "superseded source-push implementation commit is absent")
        require(not subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", prior_commit, implementation_commit], check=False).returncode, "source-push ancestry differs")
        archive_owner = source_gate
        if is_lifecycle_gate:
            lifecycle_binding = source_gate.get("runtime_lifecycle_repair")
            infrastructure_binding = source_gate.get("attempt01_infrastructure_ledger")
            require(isinstance(lifecycle_binding, Mapping), "v5 lacks lifecycle repair binding")
            require(isinstance(infrastructure_binding, Mapping), "v5 lacks attempt01 ledger binding")
            verify_binding(lifecycle_binding, path=lifecycle_path, ignore_path=True)
            verify_binding(infrastructure_binding, path=infrastructure_path, ignore_path=True)
            require(
                source_gate.get("prospective_change_scope")
                == "runtime lifecycle isolation, terminal diagnostics, and failure classification only; candidate schedule and all scientific gates remain unchanged",
                "v5 prospective lifecycle scope differs",
            )
            archive_owner = prior_value
            v3_path = repair_root / "source_push_gate_v3.json"
            v3_binding = prior_value.get("supersedes")
            require(isinstance(v3_binding, Mapping), "v4 source-push gate lacks v3 binding")
            verify_binding(v3_binding, path=v3_path, ignore_path=True)
            v3_value = load(v3_path)
            require(v3_value.get("model_request_count") == v3_value.get("behavioral_episode_count") == v3_value.get("repair_candidate_evaluation_count") == 0, "v3 source-push counts are nonzero")
            v3_commit = str(v3_value.get("implementation_commit", ""))
            require(bool(v3_commit) and not subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", v3_commit, prior_commit], check=False).returncode, "v3→v4 source ancestry differs")
            prior_value = v3_value
            prior_commit = v3_commit
        v2_binding = prior_value.get("supersedes")
        require(isinstance(v2_binding, Mapping), "v3 source-push gate lacks corrected-v2 binding")
        v2_path = repair_root / "source_push_gate_v2.json"
        verify_binding(v2_binding, path=v2_path, ignore_path=True)
        v2_value = load(v2_path)
        require(v2_value.get("model_request_count") == v2_value.get("behavioral_episode_count") == v2_value.get("repair_candidate_evaluation_count") == 0, "corrected-v2 source-push counts are nonzero")
        v2_commit = str(v2_value.get("implementation_commit", ""))
        require(bool(v2_commit) and not subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", v2_commit, prior_commit], check=False).returncode, "corrected-v2→v3 source ancestry differs")
        v1_binding = v2_value.get("supersedes")
        require(isinstance(v1_binding, Mapping), "corrected-v2 source-push gate lacks v1 binding")
        v1_path = repair_root / "source_push_gate.json"
        verify_binding(v1_binding, path=v1_path, ignore_path=True)
        v1_value = load(v1_path)
        require(v1_value.get("model_request_count") == v1_value.get("behavioral_episode_count") == v1_value.get("repair_candidate_evaluation_count") == 0, "v1 source-push counts are nonzero")
        v1_commit = str(v1_value.get("implementation_commit", ""))
        require(bool(v1_commit) and not subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", v1_commit, v2_commit], check=False).returncode, "v1→corrected-v2 source ancestry differs")
        archive_binding = archive_owner.get("original_pushed_v2_archive")
        archive_path = repair_root / "source_push_gate_v2_superseded_invalid.json"
        require(isinstance(archive_binding, Mapping), "v4 lacks original pushed v2 archive binding")
        verify_binding(archive_binding, path=archive_path, ignore_path=True)
        require(archive_binding.get("sha256") == "c63645794688d155254373387f5fcf284911e7ab55235442ae23b45f9fff0c23", "original pushed v2 archive digest differs")
        archived_v2 = load(archive_path)
        require(archived_v2.get("model_request_count") == archived_v2.get("behavioral_episode_count") == archived_v2.get("repair_candidate_evaluation_count") == 0, "archived original-v2 counts are nonzero")
        archive_commit = str(archived_v2.get("implementation_commit", ""))
        require(archive_commit == v2_commit, "archived and corrected v2 implementation commits differ")
        bad_v1_binding = archived_v2.get("supersedes")
        require(isinstance(bad_v1_binding, Mapping) and bad_v1_binding.get("sha256") == sha256(v1_path), "archived original-v2 v1 digest differs")
        require(bad_v1_binding.get("bytes") == 3322 and v1_path.stat().st_size == 2851, "archived original-v2 defect is not the registered sole byte-count mismatch")
        require(archive_owner.get("archived_v2_defect") == "superseded v1 byte count recorded as 3322 instead of actual 2851; v1 SHA-256 and all zero counts were correct", "v4 archived-v2 defect statement differs")
        source_gate_result = {"sha256": sha256(source_gate_path), "implementation_commit": implementation_commit, "files": len(inventory), "status": "latest_source_gate_and_original_freeze_archive_verified"}
    return {
        "passed": True,
        "repair_registration_sha256": sha256(registration_path),
        "candidate_schedule_sha256": sha256(schedule_path),
        "original_closure_binding_sha256": sha256(closure_path),
        "candidate_pairs": len(pairs),
        "original_files_verified": len(files),
        "infrastructure_attempts_retained": len(infrastructure_rows),
        "runtime_lifecycle_repair_sha256": sha256(lifecycle_path),
        "source_push_gate": source_gate_result,
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
    verify_binding(launch["harness_source"], ignore_path=False)
    require(harness.get("harness_source") == launch.get("harness_source"), "harness source differs from launch")
    for row in launch.get("input_bindings", {}).values():
        verify_binding(row, ignore_path=False)
    for row in launch.get("formal_health_preflight", {}).values():
        verify_binding(row, ignore_path=False)
    child = load(Path(harness["child_report"]["path"]))
    require(child.get("amendment_id") == "V3-E006-R001", "child amendment differs")
    require(child.get("model_request_count") == child.get("behavioral_episode_count") == 0, "child behavioral/model counts are nonzero")
    child_evidence = child.get("execution_evidence", child)
    require(isinstance(child_evidence, Mapping), "child execution evidence is absent")
    verify_binding(child["construction_source"], ignore_path=False)
    require(
        child["construction_source"]["sha256"] == launch["input_bindings"]["repair_source"]["sha256"]
        and child["construction_source"]["bytes"] == launch["input_bindings"]["repair_source"]["bytes"],
        "child construction source differs from launch",
    )
    require(child["construction_source"].get("study_commit") == launch.get("study_commit"), "child study commit differs from launch")
    input_name_map = {
        "e004_candidate": "e004_candidate",
        "ood_freeze": "ood_freeze",
        "e004_full_reset_reference": "e004_reset_reference",
        "runtime_contract": "runtime_contract",
        "repair_registration": "repair_registration",
        "candidate_schedule": "candidate_schedule",
        "original_v3e006_closure_binding": "original_closure_binding",
        "source_push_gate": "source_push_gate",
        "control_scene_asset": "control_scene_asset",
        "paired_scene_asset": "paired_scene_asset",
    }
    for child_name, launch_name in input_name_map.items():
        child_row = child_evidence["input_bindings"][child_name]
        launch_row = launch["input_bindings"][launch_name]
        require(
            child_row["bytes"] == launch_row["bytes"] and child_row["sha256"] == launch_row["sha256"],
            f"child {child_name} binding differs from launch",
        )
    require(child_evidence.get("passed_health_preflight") == launch.get("formal_health_preflight"), "child formal-health bundle differs from launch")
    child_lane = child_evidence.get("lane", {})
    for key in ("pod", "pod_uid", "gpu_uuid", "container_image", "container_id", "driver_version"):
        require(child_lane.get(key) == launch.get("lane", {}).get(key), f"child lane {key} differs from launch")
    require(child_evidence.get("runtime_log", {}).get("path") == harness["runtime_log"]["path"], "child runtime-log path differs from harness")
    invocation = child_evidence.get("invocation")
    child_argv = launch.get("child_argv")
    require(isinstance(invocation, list) and isinstance(child_argv, list) and len(child_argv) >= 4, "child invocation evidence is absent")
    require(invocation[1:] == child_argv[3:], "child observed argv differs from launch argv")
    if child.get("status") == "infrastructure_invalid_r001_state_repair":
        require(child.get("passed") is False and child.get("candidate_gate_passed") is False, "infrastructure child was marked passed")
        require(child.get("state_candidate_count") == 0, "infrastructure child counted a state candidate")
        require(child.get("behavioral_denominator_included") is False, "infrastructure child entered behavioral denominator")
        require(Path(harness["child_report"]["path"]).name == "state_construction_failure.json", "infrastructure child has wrong report name")
        for row in child.get("input_bindings", {}).values():
            verify_binding(row, ignore_path=False)
        for row in child.get("passed_health_preflight", {}).values():
            verify_binding(row, ignore_path=False)
        verify_binding(child["construction_source"], ignore_path=False)
        for row in child.get("available_raw_artifacts", {}).values():
            verify_binding(row, ignore_path=False)
        historical_attempt01 = sha256(harness_path) == ATTEMPT01_BINDINGS["harness_result"][1]
        if historical_attempt01:
            require(sha256(launch_path) == ATTEMPT01_BINDINGS["launch"][1], "attempt01 launch digest differs")
            require(sha256(Path(harness["runtime_log"]["path"])) == ATTEMPT01_BINDINGS["runtime_log"][1], "attempt01 log digest differs")
            require(sha256(Path(harness["child_report"]["path"])) == ATTEMPT01_BINDINGS["child_failure"][1], "attempt01 child digest differs")
            require(harness.get("status") == "completed_r001_candidate_search" and harness.get("process_completed") is True and harness.get("process_exit_code") == 0, "attempt01 historical harness defect differs")
            classification = "verified_historical_attempt01_infrastructure_with_disclosed_outer_harness_misclassification"
        else:
            require(harness.get("status") == "infrastructure_invalid_r001_state_repair", "new infrastructure failure was misclassified")
            require(harness.get("process_completed") is False, "new infrastructure failure marked process complete")
            require(isinstance(harness.get("process_exit_code"), int) and harness["process_exit_code"] != 0, "new failed child did not exit nonzero")
            classification = "verified_infrastructure_invalid_zero_model_zero_behavior"
        return {
            "passed": True,
            "scientific_gate_passed": False,
            "evidence_classification": classification,
            "harness_sha256": sha256(harness_path),
            "child_sha256": sha256(Path(harness["child_report"]["path"])),
            "repair_candidate_evaluations": child.get("repair_candidate_evaluation_count"),
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "state_candidate_count": 0,
        }
    require(harness.get("status") == "completed_r001_candidate_search" and harness.get("process_completed") is True, "normal search result was not process-complete")
    require(child.get("status") in {"passed_r001_state_repair_not_released_for_behavior", "r001_candidate_budget_exhausted_no_valid_state_pair"}, "wrong normal child status")
    require(child.get("candidate_budget") == 8, "child candidate budget differs")
    for key, launch_name in (
        ("repair_registration", "repair_registration"),
        ("candidate_schedule", "candidate_schedule"),
        ("source_push_gate", "source_push_gate"),
        ("original_v3e006_closure_binding", "original_closure_binding"),
        ("ood_freeze", "ood_freeze"),
        ("e004_full_reset_reference", "e004_reset_reference"),
        ("e004_candidate", "e004_candidate"),
    ):
        verify_binding(child[key], ignore_path=False)
        require(
            child[key]["bytes"] == launch["input_bindings"][launch_name]["bytes"]
            and child[key]["sha256"] == launch["input_bindings"][launch_name]["sha256"],
            f"normal child {key} differs from launch",
        )
    verify_binding(child["frozen_e004_runtime_bindings"], ignore_path=False)
    for key, launch_name in (("control", "control_scene_asset"), ("paired", "paired_scene_asset")):
        verify_binding(child["scene_assets"][key], ignore_path=False)
        require(
            child["scene_assets"][key]["bytes"] == launch["input_bindings"][launch_name]["bytes"]
            and child["scene_assets"][key]["sha256"] == launch["input_bindings"][launch_name]["sha256"],
            f"normal child {key} scene differs from launch",
        )
    verify_binding(child["video"], ignore_path=False)
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
            lifecycle = stage["environment_lifecycle"]
            require(lifecycle.get("created") is True, "stage did not create a dedicated environment")
            require(lifecycle.get("contact_sensors_initialized_in_this_environment") is True, "stage did not initialize local contacts")
            require(lifecycle.get("fresh_reset_completed_in_this_environment") is True, "stage did not complete its fresh reset")
            require(lifecycle.get("closed_before_next_stage") is True, "stage environment was not closed")
            require(stage["fresh_reset"]["passed"] is True, "fresh reset gate failed")
            for row in stage["fresh_reset"]["camera_evidence"]["bindings"].values():
                verify_binding(row["rgb"], ignore_path=False)
            state = stage["candidate_state"]
            require(all(key in state for key in ("physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate", "normalized_state_sha256")), "candidate state lacks gate/hash evidence")
            for row in state["camera_evidence"]["bindings"].values():
                verify_binding(row["rgb"], ignore_path=False)
    lifecycle = child.get("environment_lifecycle")
    require(isinstance(lifecycle, list) and len(lifecycle) == 2 * len(attempts) <= 16, "dedicated environment lifecycle count differs")
    require([row["environment_ordinal"] for row in lifecycle] == list(range(1, len(lifecycle) + 1)), "environment lifecycle order differs")
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
