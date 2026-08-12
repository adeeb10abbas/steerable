#!/usr/bin/env python3
"""Fail-closed validator for prospective and raw V3-E006-R003 evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r003.predecessor_contract import (
    validate_r002_exhaustion,
)


R002_CLOSURE_COMMIT = "27d1bfd844808f7f336bbb4e25552a9c859fd08a"
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
REGISTRATION_STATUS = "prospectively_registered_before_any_r003_live_diagnostic_candidate_or_model_request"
SOURCE_GATE_STATUS = "passed_before_first_r003_live_diagnostic_candidate_or_model_request"
TERMINAL_STATUSES = {
    "passed_r003_state_repair_not_released_for_behavior": True,
    "r003_candidate_budget_exhausted_no_valid_state_pair": False,
    "r003_known_reachable_diagnostic_failed_candidates_not_evaluated": False,
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_binding(path: Path, expected: Mapping[str, Any], label: str) -> None:
    require(path.is_file(), f"{label} is missing: {path}")
    require(path.stat().st_size == expected.get("bytes"), f"{label} bytes differ")
    require(sha256(path) == expected.get("sha256"), f"{label} digest differs")


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {path}: {exc}") from exc


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name]
    require(len(nodes) == 1, f"{path} must contain exactly one {name}")
    return ast.dump(nodes[0], include_attributes=False)


def normalize_quaternion(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    require(quaternion.shape == (4,) and np.all(np.isfinite(quaternion)), "quaternion differs")
    norm = float(np.linalg.norm(quaternion))
    require(norm > 0.0, "zero quaternion")
    return quaternion / norm


def qmul(left: Any, right: Any) -> np.ndarray:
    lw, lx, ly, lz = normalize_quaternion(left)
    rw, rx, ry, rz = normalize_quaternion(right)
    return normalize_quaternion(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def shortest_slerp(left: Any, right: Any, fraction: float) -> np.ndarray:
    a, b = normalize_quaternion(left), normalize_quaternion(right)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        return normalize_quaternion((1.0 - fraction) * a + fraction * b)
    theta = math.acos(dot)
    return normalize_quaternion(
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * a
        + math.sin(fraction * theta) / math.sin(theta) * b
    )


def validate_scientific_selection(report: Mapping[str, Any], harness: Mapping[str, Any]) -> None:
    """Recompute the complete first-pass/exhaustion decision from raw stages."""

    require(report.get("status") in TERMINAL_STATUSES, "child terminal status differs")
    expected_terminal_pass = TERMINAL_STATUSES[str(report["status"])]
    require(report.get("passed") is expected_terminal_pass, "child pass boolean differs")
    diagnostics = report.get("known_reachable_diagnostics")
    require(isinstance(diagnostics, list) and 1 <= len(diagnostics) <= 4, "diagnostics differ")
    require(
        [row.get("diagnostic_index_one_based") for row in diagnostics]
        == list(range(1, len(diagnostics) + 1)),
        "diagnostic order differs",
    )
    for row in diagnostics:
        hold = row.get("pose_hold")
        require(isinstance(hold, Mapping), "diagnostic hold is absent")
        require(row.get("passed") is hold.get("passed"), "diagnostic pass differs from hold")
        require(
            hold.get("passed")
            is (
                hold.get("completed_steps") == hold.get("hold_steps") == 30
                and hold.get("final_window_passed") is True
                and hold.get("all_states_finite") is True
                and hold.get("all_arm_states_inside_live_soft_joint_limits") is True
                and hold.get("all_base_link_to_eef_frame_identity_checks_passed") is True
                and hold.get("termination") is None
            ),
            "diagnostic pass does not recompute",
        )
    require(
        report.get("r003_live_diagnostic_count") == len(diagnostics),
        "child diagnostic count differs",
    )
    require(
        harness.get("r003_live_diagnostic_count") == len(diagnostics),
        "harness diagnostic count differs",
    )
    attempts = report.get("attempts")
    if report.get("status") == "r003_known_reachable_diagnostic_failed_candidates_not_evaluated":
        require(not all(row["passed"] for row in diagnostics), "diagnostic failure status has no failure")
        require(isinstance(attempts, list) and attempts == [], "failed diagnostic evaluated candidates")
        require(report.get("repair_candidate_evaluation_count") == 0, "failed diagnostic candidate count differs")
        require(report.get("accepted_candidate_rank") is None, "failed diagnostic accepted a rank")
        require(harness.get("process_completed") is True, "diagnostic terminal was not complete")
        require(harness.get("scientific_gate_passed") is False, "diagnostic harness scientific flag differs")
        require(harness.get("child_status") == report.get("status"), "diagnostic harness child status differs")
        return

    require(len(diagnostics) == 4 and all(row["passed"] for row in diagnostics), "candidates ran before four passed diagnostics")
    require(isinstance(attempts, list) and 1 <= len(attempts) <= 4, "child attempt count differs")
    require([row.get("candidate_rank") for row in attempts] == list(range(1, len(attempts) + 1)), "child rank order differs")
    require(report.get("repair_candidate_evaluation_count") == len(attempts), "child evaluated count differs")
    require(harness.get("repair_candidate_evaluation_count") == len(attempts), "harness evaluated count differs")
    recomputed_rank_passes: list[bool] = []
    for attempt in attempts:
        require(set(attempt.get("stages", {})) == {"canonical_grasp", "canonical_carry"}, "child stages differ")
        stage_passes: list[bool] = []
        for stage in ("canonical_grasp", "canonical_carry"):
            row = attempt["stages"][stage]
            state = row.get("candidate_state")
            require(isinstance(state, Mapping), f"{stage} candidate state is absent")
            evaluated = all(
                isinstance(state.get(name), Mapping)
                for name in ("physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate")
            )
            if evaluated:
                expected_stage_pass = all(
                    state[name].get("passed") is True
                    for name in ("physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate")
                )
                require(state.get("passed") is expected_stage_pass, f"{stage} stage pass differs from gates")
            else:
                require(state.get("passed") is False, f"unevaluated {stage} stage was marked passed")
                require(isinstance(state.get("candidate_rejection"), str), f"unevaluated {stage} lacks rejection")
                expected_stage_pass = False
            stage_passes.append(expected_stage_pass)
        expected_rank_pass = all(stage_passes)
        require(attempt.get("passed") is expected_rank_pass, "rank pass differs from both stages")
        recomputed_rank_passes.append(expected_rank_pass)

    passing_indices = [index for index, value in enumerate(recomputed_rank_passes) if value]
    require(len(passing_indices) <= 1, "more than one rank passed after first-pass stopping")
    if expected_terminal_pass:
        require(passing_indices == [len(attempts) - 1], "passing rank was not the last/first passing rank")
        accepted_rank = len(attempts)
        require(report.get("accepted_candidate_rank") == accepted_rank, "accepted rank differs")
        require(report.get("accepted_states") == attempts[-1]["stages"], "accepted states differ from passing rank")
        require(all(not value for value in recomputed_rank_passes[:-1]), "a prior rank passed")
    else:
        require(len(attempts) == report.get("candidate_budget") == 4, "exhaustion did not complete all four ranks")
        require(not any(recomputed_rank_passes), "exhaustion contains a passing rank")
        require(report.get("accepted_candidate_rank") is None, "exhaustion has accepted rank")
        require(report.get("accepted_states") is None, "exhaustion has accepted states")
    require(report.get("first_passing_rule_obeyed") is True, "first-pass rule flag differs")
    require(harness.get("process_completed") is True, "normal search was not process complete")
    require(harness.get("status") == "completed_r003_candidate_search", "normal harness status differs")
    require(harness.get("scientific_gate_passed") is expected_terminal_pass, "harness scientific flag differs")
    require(harness.get("child_status") == report.get("status"), "harness child status differs")


def validate_static(
    root: Path, *, require_source_gate: bool = True, verify_raw: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    relative = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003")
    artifact = root / relative
    registration_path = artifact / "repair_registration.json"
    schedule_path = artifact / "gates/candidate_schedule.json"
    source_gate_v1_path = artifact / "source_push_gate.json"
    source_gate_v2_path = artifact / "source_push_gate_v2.json"
    source_gate_path = source_gate_v2_path if source_gate_v2_path.is_file() else source_gate_v1_path
    registration = load(registration_path)
    schedule = load(schedule_path)

    require(registration.get("repair_amendment_id") == "V3-E006-R003", "registration ID differs")
    require(registration.get("status") == REGISTRATION_STATUS, "registration status differs")
    require(registration.get("counts_at_registration") == {
        "r003_live_diagnostics": 0, "r003_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0
    }, "registration counts differ")
    require(registration.get("predecessor_repair_amendment_id") == "V3-E006-R002", "predecessor ID differs")
    for commit in (BASE, R002_CLOSURE_COMMIT):
        require(
            subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                           check=False).returncode == 0,
            f"source lineage commit does not exist: {commit}",
        )
    frozen = registration["frozen_inputs"]
    for name, row in frozen.items():
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = root / path
        verify_binding(path, row, f"registration frozen input {name}")

    require(schedule.get("status") == "frozen_before_any_r003_live_diagnostic_candidate_or_model_request", "schedule status differs")
    require(schedule.get("candidate_budget") == 4 and schedule.get("diagnostic_budget") == 4, "schedule budgets differ")
    require([row.get("candidate_rank") for row in schedule.get("candidate_pairs", [])] == list(range(1, 5)), "schedule ranks differ")
    require(
        [row.get("diagnostic_index_one_based") for row in schedule.get("known_reachable_diagnostics", [])]
        == list(range(1, 5)), "diagnostic ranks differ"
    )
    require(
        schedule.get("model_request_count") == schedule.get("behavioral_episode_count")
        == schedule.get("r003_live_candidate_evaluation_count")
        == schedule.get("r003_live_diagnostic_count") == 0,
        "schedule counts differ",
    )
    verify_binding(registration_path, schedule["repair_registration"], "schedule registration")
    for name, row in schedule["r002_predecessor"].items():
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = root / path
        verify_binding(path, row, f"R002 predecessor {name}")
    r002_results = load(root / schedule["r002_predecessor"]["results"]["path"])
    try:
        validate_r002_exhaustion(r002_results)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    require(schedule.get("r002_closure_commit") == R002_CLOSURE_COMMIT, "R002 closure commit differs")
    require(
        schedule.get("r002_raw_result", {}).get("sha256")
        == "afb8c3ba2b53f1513bd22fd6135b16cbfe3e4dd9de3fe3d818ac05e458311fe7",
        "R002 raw result binding differs",
    )
    require([row["construction_method"] for row in schedule["candidate_pairs"]] == ["direct_contact_initialization"] * 4, "construction method dispatch differs")
    r002_schedule = load(root / frozen["r002_candidate_schedule"]["path"])
    expected_selectors = [
        ("left_observed", "reflected_right_observed"),
        ("reflected_right_observed", "left_observed"),
        ("left_observed", "left_observed"),
        ("reflected_right_observed", "reflected_right_observed"),
    ]
    observed_selectors: list[tuple[str, str]] = []
    for pair in schedule["candidate_pairs"]:
        predecessor_pair = r002_schedule["candidate_pairs"][pair["candidate_rank"] - 1]
        observed_selectors.append(
            (
                pair["canonical_grasp"]["contact_transform_selector"],
                pair["canonical_carry"]["contact_transform_selector"],
            )
        )
        for stage in ("canonical_grasp", "canonical_carry"):
            row = pair[stage]
            old = predecessor_pair[stage]
            require(row["target_cube_pose"]["position_world_m"][1] == 0.0, "target cube not centered")
            require(row["target_cube_pose"] == old["target_cube_pose"], "world cube target changed")
            require(
                row["centerline_constrained_base_link_ik_target"]
                == old["centerline_constrained_eef_ik_target"],
                "numeric base_link target changed from mislabeled R002 value",
            )
            require(
                row["selected_observed_cube_in_base_link_transform"]
                == old["selected_observed_cube_in_eef_transform"],
                "numeric contact transform changed",
            )
            residual = row["se3_reconstruction"]
            require(residual["cube_midline_residual_m"] <= 1e-12, "reconstructed cube not centered")
            require(residual["position_residual_m"] <= 1e-12, "SE3 position residual differs")
            require(residual["rotation_matrix_frobenius_residual"] <= 1e-12, "SE3 rotation residual differs")
            require(set(row["both_direction_sources"]) == {"left", "right"}, "direction source pair incomplete")
            solver = row["r003_solver_initialization"]
            require(len(solver["waypoints"]) == 8, "waypoint count differs")
            require([waypoint["fraction"] for waypoint in solver["waypoints"]] == [index / 8 for index in range(1, 9)], "waypoint fractions differ")
            require(all(waypoint["hold_steps"] == 30 and waypoint["required_final_consecutive_steps"] == 10 for waypoint in solver["waypoints"]), "waypoint holds differ")
            require(solver["waypoints"][-1]["position_world_m"] == row["centerline_constrained_base_link_ik_target"]["position_world_m"], "waypoint endpoint differs")
            source = row["both_direction_sources"][solver["source_side"]]
            source_position = np.asarray(source["base_link_position_world_m"], dtype=np.float64)
            source_quaternion = source["base_link_quaternion_world_wxyz"]
            target_position = np.asarray(
                row["centerline_constrained_base_link_ik_target"]["position_world_m"],
                dtype=np.float64,
            )
            target_quaternion = row["centerline_constrained_base_link_ik_target"][
                "quaternion_world_wxyz"
            ]
            for index, waypoint in enumerate(solver["waypoints"], start=1):
                fraction = index / 8.0
                require(
                    np.array_equal(
                        np.asarray(waypoint["position_world_m"], dtype=np.float64),
                        (1.0 - fraction) * source_position + fraction * target_position,
                    ),
                    "waypoint position interpolation differs",
                )
                require(
                    abs(
                        float(
                            np.dot(
                                normalize_quaternion(waypoint["quaternion_world_wxyz"]),
                                shortest_slerp(source_quaternion, target_quaternion, fraction),
                            )
                        )
                    )
                    >= 1.0 - 1e-14,
                    "waypoint shortest-arc SLERP differs",
                )
            for source in row["both_direction_sources"].values():
                require("base_link_position_world_m" in source and "eef_position_world_m" not in source, "candidate source frame label differs")
    require(observed_selectors == expected_selectors, "direction-balanced selector order differs")

    for diagnostic in schedule["known_reachable_diagnostics"]:
        source = diagnostic["source"]
        require(diagnostic["recorded_pose_frame"] == "robot/base_link", "diagnostic frame differs")
        require("base_link_position_world_m" in source and "eef_position_world_m" not in source, "diagnostic source frame differs")
        require(diagnostic["hold_steps"] == 30 and diagnostic["required_final_consecutive_steps"] == 10, "diagnostic hold differs")
        expected_eef_quaternion = qmul(
            source["base_link_quaternion_world_wxyz"], [0.5, -0.5, 0.5, -0.5]
        )
        observed_eef_quaternion = normalize_quaternion(
            source["expected_eef_frame_quaternion_world_wxyz"]
        )
        require(
            abs(float(np.dot(observed_eef_quaternion, expected_eef_quaternion)))
            >= 1.0 - 1e-14,
            "diagnostic expected eef quaternion differs from base offset composition",
        )
        require(
            source["expected_eef_frame_position_world_m"]
            == source["base_link_position_world_m"],
            "diagnostic expected eef position differs despite zero offset",
        )
    recorder = schedule["controller_source_bindings"]["robolab_post_step_end_effector_pose_recorder"]
    require("base_link" in recorder.get("semantic_assertion", ""), "recorder semantics absent")

    state_contract = root / frozen["state_contract"]["path"]
    ood_reference = root / frozen["ood_reference"]["path"]
    require(sha256(state_contract) == "2476b28d2867c1b87f477fd5f89e545616be00d860d4144f8cbdb70af10f3c18", "state contract not unchanged")
    require(sha256(ood_reference) == "4df1ebf0061096a74b5eccd10b2a144e840f52fd50469b8bdae9369d1696fd04", "OOD source not unchanged")
    r003_text = (root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/state_repair_gate.py").read_text(encoding="utf-8")
    r003_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/state_repair_gate.py"
    r002_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/state_repair_gate.py"
    for name in ("_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence", "_companion_gate", "_fresh_reset_and_gate"):
        require(function_ast(r002_source, name) == function_ast(r003_source, name), f"unchanged helper differs: {name}")
    require("def _command_base_link(" in r003_text and "_command_base_link(position, quaternion, 1.0" in r003_text, "direct base_link action path absent")
    command_tree = ast.parse(r003_text)
    command_node = next(
        row for row in command_tree.body
        if isinstance(row, ast.FunctionDef) and row.name == "_command_base_link"
    )
    command_dump = ast.dump(command_node, include_attributes=False)
    require("EEF_OFFSET" not in command_dump and "_quat_inverse" not in command_dump, "base_link command applies an offset")
    require("_run_known_reachable_diagnostic(" in r003_text, "diagnostic dispatch absent")
    require("open_approach_close_lift" not in r003_text, "unregistered R002 fallback remains reachable")
    require("requests.post" not in r003_text and "httpx" not in r003_text and "policy_server" not in r003_text, "model endpoint exists in R003 constructor")

    source_gate_summary = None
    if require_source_gate:
        require(source_gate_path.is_file(), "R003 source-push gate is missing")
        source_gate = load(source_gate_path)
        require(source_gate.get("status") == SOURCE_GATE_STATUS, "source-push status differs")
        require(source_gate.get("model_request_count") == source_gate.get("behavioral_episode_count") == source_gate.get("r003_live_candidate_evaluation_count") == source_gate.get("r003_live_diagnostic_count") == 0, "source-push counts differ")
        if source_gate_path == source_gate_v2_path:
            require(source_gate.get("schema_version") == "vla-wam-shared-v3e006-r003-source-push-gate-v2", "source-push v2 schema differs")
            require(source_gate.get("infrastructure_invalid_search_attempt_count") == 1, "source-push invalid-attempt count differs")
            verify_binding(source_gate_v1_path, source_gate["supersedes_source_push_gate_v1"], "source-push v1 predecessor")
            ledger_binding = source_gate.get("infrastructure_attempts")
            require(isinstance(ledger_binding, Mapping), "source-push infrastructure ledger binding absent")
            ledger_path = root / str(ledger_binding["path"])
            verify_binding(ledger_path, ledger_binding, "source-push infrastructure ledger")
            ledger = load(ledger_path)
            require(ledger.get("attempt_count") == 1, "infrastructure ledger attempt count differs")
            require(ledger.get("model_request_count") == ledger.get("behavioral_episode_count") == ledger.get("state_candidate_count") == 0, "infrastructure ledger counts differ")
            attempt = ledger.get("attempts", [{}])[0]
            require(attempt.get("status") == "infrastructure_invalid_pre_AppLauncher_no_diagnostic_or_candidate", "invalid-attempt status differs")
            require(attempt.get("app_launcher_started") is False, "invalid attempt passed AppLauncher boundary")
            require(attempt.get("model_request_count") == attempt.get("behavioral_episode_count") == attempt.get("diagnostic_evaluation_count") == attempt.get("candidate_pair_evaluation_count") == attempt.get("state_candidate_count") == 0, "invalid-attempt counts differ")
            raw_bindings = attempt.get("raw_bindings")
            require(
                isinstance(raw_bindings, Mapping)
                and set(raw_bindings)
                == {"expanded_invocation", "wrapper_log", "launch", "harness_result", "runtime_log"},
                "invalid-attempt raw binding inventory differs",
            )
            expected_attempt_hashes = {
                "expanded_invocation": "2c24849450ab388015ec057680ade47f4a6b1ddc693c702379d5701da84bd876",
                "wrapper_log": "87cede0a42126be38ea387b64f2a5966967361092adec0200179e5efdda9b93a",
                "launch": "edcee7c0934a4d0f36d1222a3144d2bbabfb70464ca1a8c94ad2670230d72ff3",
                "harness_result": "2f5b9741768c08ead6803db7bfee7227ce4cdb02fb7846d7fdde70a66cfc55b0",
                "runtime_log": "0d2ebef90fa60a03b7f008fadfa496612668a1f2f9a534efba0bf048e07af708",
            }
            require(
                {name: row.get("sha256") for name, row in raw_bindings.items()}
                == expected_attempt_hashes,
                "invalid-attempt raw hashes differ",
            )
            output_root = Path(str(attempt.get("output_root", "")))
            require(
                raw_bindings["launch"]["path"] == str(output_root / "launch.json")
                and raw_bindings["harness_result"]["path"]
                == str(output_root / "harness_result.json")
                and raw_bindings["runtime_log"]["path"] == str(output_root / "runtime.log"),
                "invalid-attempt output-root bindings differ",
            )
            if verify_raw:
                for name, row in raw_bindings.items():
                    verify_binding(Path(str(row["path"])), row, f"invalid-attempt raw {name}")
                invalid_receipt = validate_candidate_root(root, output_root)
                require(
                    invalid_receipt.get("status")
                    == "retained_infrastructure_invalid_before_AppLauncher",
                    "invalid-attempt raw receipt status differs",
                )
        else:
            require(source_gate.get("infrastructure_invalid_search_attempt_count") == 0, "source-push v1 invalid count differs")
        implementation_commit = str(source_gate.get("implementation_commit", ""))
        require(
            subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{implementation_commit}^{{commit}}"], check=False).returncode == 0,
            "source-push implementation commit does not exist",
        )
        for row in source_gate.get("implementation_files", []):
            relative_path = Path(str(row["path"]))
            require(not relative_path.is_absolute() and ".." not in relative_path.parts, "unsafe source-push path")
            verify_binding(root / relative_path, row, f"source-push inventory {relative_path}")
        require(len(source_gate.get("implementation_files", [])) >= 9, "source-push inventory is incomplete")
        verify_binding(registration_path, source_gate["repair_registration"], "source-push registration")
        verify_binding(schedule_path, source_gate["candidate_schedule"], "source-push schedule")
        source_gate_summary = binding(source_gate_path)

    return {
        "passed": True,
        "registration": binding(registration_path),
        "candidate_schedule": binding(schedule_path),
        "r002_predecessor_results": schedule["r002_predecessor"]["results"],
        "source_push_gate": source_gate_summary,
        "candidate_pair_count": 4,
        "diagnostic_count": 4,
        "unchanged_helper_count": 6,
    }


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    launch_path = candidate_root / "launch.json"
    harness_path = candidate_root / "harness_result.json"
    runtime_path = candidate_root / "runtime.log"
    launch = load(launch_path)
    harness = load(harness_path)
    require(harness.get("model_request_count") == harness.get("behavioral_episode_count") == 0, "raw harness counts differ")
    verify_binding(launch_path, harness["launch"], "raw harness launch")
    verify_binding(runtime_path, harness["runtime_log"], "raw harness runtime log")
    for name, expected in launch.get("input_bindings", {}).items():
        verify_binding(Path(str(expected["path"])), expected, f"raw launch input {name}")
    for name, expected in launch.get("formal_health_preflight", {}).items():
        verify_binding(Path(str(expected["path"])), expected, f"raw formal health {name}")
    verify_binding(Path(str(launch["harness_source"]["path"])), launch["harness_source"], "raw harness source")
    report_binding = harness.get("child_report")
    if report_binding is None:
        require(harness.get("status") == "infrastructure_invalid_r003_state_repair", "pre-child harness status differs")
        require(harness.get("process_completed") is False, "pre-child harness cannot be complete")
        require(harness.get("scientific_gate_passed") is False, "pre-child harness cannot pass science")
        require(harness.get("process_exit_code") == 2, "pre-child bootstrap exit code differs")
        require(harness.get("child_status") is None, "pre-child harness has a child status")
        require(harness.get("r003_live_diagnostic_count") == harness.get("repair_candidate_evaluation_count") == 0, "pre-child harness counts differ")
        require(
            "R002 predecessor was not a zero-request finite exhaustion"
            in str(harness.get("failure_log_tail", "")),
            "pre-child bootstrap cause differs",
        )
        require(not (candidate_root / "raw").exists(), "pre-child attempt unexpectedly created raw child evidence")
        return {
            "passed": True,
            "candidate_root": str(candidate_root),
            "status": "retained_infrastructure_invalid_before_AppLauncher",
            "launch": binding(launch_path),
            "harness": binding(harness_path),
            "runtime_log": binding(runtime_path),
            "child_report": None,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "r003_live_diagnostic_count": 0,
            "repair_candidate_evaluation_count": 0,
        }
    require(isinstance(report_binding, Mapping), "raw harness child binding differs")
    report_path = Path(str(report_binding["path"]))
    verify_binding(report_path, report_binding, "raw child report")
    report = load(report_path)
    if harness.get("process_completed") is True:
        require(report_path.name == "state_repair_result.json", "completed child report filename differs")
        require(report.get("model_request_count") == report.get("behavioral_episode_count") == 0, "child counts differ")
        validate_scientific_selection(report, harness)
        diagnostics = report["known_reachable_diagnostics"]
        for diagnostic in diagnostics:
            lifecycle = diagnostic["environment_lifecycle"]
            require(lifecycle["closed_before_next_environment"] is True, "diagnostic env not closed")
            require(diagnostic["fresh_reset"]["passed"] is True, "diagnostic fresh reset failed")
            for camera in diagnostic["fresh_reset"]["camera_evidence"]["bindings"].values():
                verify_binding(Path(str(camera["rgb"]["path"])), camera["rgb"], "diagnostic reset camera")
            initialization = diagnostic["historical_state_initialization"]
            require(initialization["base_link_to_eef_frame_identity"]["passed"] is True, "diagnostic initial frame identity failed")
            for row in diagnostic["pose_hold"]["errors"]:
                require(row["base_link_to_eef_frame_identity"]["passed"] is True, "diagnostic frame identity row failed")
            for row in diagnostic["pose_hold"]["construction_action_trace"]:
                require(row["base_link_to_eef_frame_identity"]["passed"] is True, "diagnostic trace frame identity failed")
        attempts = report["attempts"]
        for attempt in attempts:
            for stage in attempt["stages"].values():
                require(stage["ik_solve_environment"]["fresh_reset"]["passed"] is True, "IK reset failed")
                require(stage["ik_solve_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "IK env not closed")
                for camera in stage["ik_solve_environment"]["fresh_reset"]["camera_evidence"]["bindings"].values():
                    verify_binding(Path(str(camera["rgb"]["path"])), camera["rgb"], "IK reset camera")
                if stage["ik_solution"]["passed"]:
                    require(stage["materialization_environment"]["fresh_reset"]["passed"] is True, "material reset failed")
                    require(stage["materialization_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "material env not closed")
                    for camera in stage["materialization_environment"]["fresh_reset"]["camera_evidence"]["bindings"].values():
                        verify_binding(Path(str(camera["rgb"]["path"])), camera["rgb"], "material reset camera")
                    state = stage["candidate_state"]
                    if "camera_evidence" in state:
                        for camera in state["camera_evidence"]["bindings"].values():
                            verify_binding(Path(str(camera["rgb"]["path"])), camera["rgb"], "candidate camera")
        for key in ("repair_registration", "candidate_schedule", "source_push_gate",
                    "original_v3e006_closure_binding", "ood_freeze", "e004_full_reset_reference",
                    "e004_candidate", "r002_predecessor_results", "construction_source", "video"):
            verify_binding(Path(str(report[key]["path"])), report[key], f"normal report {key}")
        verify_binding(
            Path(str(report["frozen_e004_runtime_bindings"]["path"])),
            report["frozen_e004_runtime_bindings"], "normal report runtime contract",
        )
        for scene in report["scene_assets"].values():
            verify_binding(Path(str(scene["path"])), scene, "normal report scene")
    else:
        require(report_path.name == "state_construction_failure.json", "invalid child report filename differs")
        require(report.get("status") == "infrastructure_invalid_r003_state_repair", "invalid child status differs")
        require(report.get("model_request_count") == report.get("behavioral_episode_count") == report.get("state_candidate_count") == 0, "invalid child counts differ")
        for artifact in report.get("available_raw_artifacts", {}).values():
            verify_binding(Path(str(artifact["path"])), artifact, "invalid available artifact")
    child_evidence = report.get("execution_evidence", report)
    require(isinstance(child_evidence, Mapping), "child execution evidence is absent")
    for name, expected in child_evidence.get("input_bindings", {}).items():
        verify_binding(Path(str(expected["path"])), expected, f"child input {name}")
    for name, expected in child_evidence.get("passed_health_preflight", {}).items():
        verify_binding(Path(str(expected["path"])), expected, f"child formal health {name}")
    historical = child_evidence.get("historical_source_verification_before_AppLauncher")
    require(isinstance(historical, Mapping) and historical, "child historical sources absent")
    for source_name, source in historical.items():
        for artifact_name, expected in source.get("bindings", {}).items():
            verify_binding(
                Path(str(expected["path"])),
                expected,
                f"child historical source {source_name}/{artifact_name}",
            )
    controller_sources = child_evidence.get("controller_source_verification_before_AppLauncher")
    require(isinstance(controller_sources, Mapping) and controller_sources, "child controller sources absent")
    for source_name, expected in controller_sources.items():
        verify_binding(
            Path(str(expected["path"])), expected, f"child controller source {source_name}"
        )
    require(child_evidence.get("passed_health_preflight") == launch.get("formal_health_preflight"), "child health differs from launch")
    require(child_evidence.get("lane") == {**launch.get("lane", {}), "device": child_evidence.get("lane", {}).get("device"), "python": child_evidence.get("lane", {}).get("python")}, "child lane differs from launch")
    return {
        "passed": True,
        "candidate_root": str(candidate_root),
        "launch": binding(launch_path),
        "harness": binding(harness_path),
        "child_report": binding(report_path),
        "child_status": report.get("status"),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pre-source-gate", action="store_true")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    result = validate_static(
        args.study_root,
        require_source_gate=not args.pre_source_gate,
        verify_raw=args.verify_raw,
    )
    if args.candidate_root is not None:
        result["candidate_evidence"] = validate_candidate_root(args.study_root, args.candidate_root)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
