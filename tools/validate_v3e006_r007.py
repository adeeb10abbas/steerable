#!/usr/bin/env python3
"""Fail-closed validator for prospective and raw V3-E006-R007 evidence."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r007.predecessor_contract import (
    validate_r006_exhaustion_closure,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r007.residual_correction import (
    corrected_command,
    normalize_quaternion,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r007.source_gate_contract import (
    SCHEMA as SOURCE_GATE_SCHEMA,
    STATUS as SOURCE_GATE_STATUS,
    validate_source_gate,
)


BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R006_CLOSURE_COMMIT = "125e8f0d231ebd2e3c7d0d9b54dce83e1080cea1"
R006_RESULTS_SHA256 = "3c58721d11f669243690aaf3619121d1c348bf788ca56aacd2a009f727065e63"
R006_RAW_RESULT_SHA256 = "7eae75c38a7b65ba4b8fbc44f3ca4c565c3af5675134c93570b1dc0e85176011"
REGISTRATION_STATUS = (
    "prospectively_registered_before_any_r007_live_diagnostic_candidate_or_model_request"
)
TERMINAL_STATUSES = {
    "passed_r007_state_repair_not_released_for_behavior": True,
    "r007_candidate_budget_exhausted_no_valid_state_pair": False,
    "r007_known_reachable_diagnostic_failed_candidates_not_evaluated": False,
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


def resolve_bound(root: Path, expected: Mapping[str, Any], label: str) -> Path:
    path = Path(str(expected.get("path", "")))
    if not path.is_absolute():
        path = root / path
    verify_binding(path, expected, label)
    return path


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {path}: {exc}") from exc


def canonical_sha(value: Any) -> str:
    payload = (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name]
    require(len(nodes) == 1, f"{path} must contain exactly one {name}")
    return ast.dump(nodes[0], include_attributes=False)


def qclose(left: Sequence[float], right: Sequence[float], atol: float = 1e-12) -> bool:
    a, b = normalize_quaternion(left), normalize_quaternion(right)
    return min(float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b))) <= atol


def pclose(left: Sequence[float], right: Sequence[float], atol: float = 1e-12) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=atol))


def slerp_wxyz(left: Sequence[float], right: Sequence[float], fraction: float) -> np.ndarray:
    """Recompute the frozen shortest-arc quaternion interpolation."""

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


def expected_open_contact_commands(
    construction: Mapping[str, Any], expected_stage: Mapping[str, Any]
) -> list[tuple[str, int, float, list[float]]]:
    """Recompute every registered float32 8-D action without trusting raw output."""

    start = construction.get("start_base_link_pose_world_wxyz")
    require(isinstance(start, list) and len(start) == 7, "open-contact start pose differs")
    require(np.isfinite(np.asarray(start, dtype=np.float64)).all(), "open-contact start pose is nonfinite")
    targets = expected_stage.get("r007_open_contact_targets")
    require(isinstance(targets, Mapping), "registered open-contact targets absent")

    def pose(name: str) -> tuple[np.ndarray, np.ndarray]:
        value = targets.get(name)
        require(isinstance(value, Mapping), f"registered target absent: {name}")
        position = np.asarray(value.get("position_world_m"), dtype=np.float64)
        quaternion = normalize_quaternion(value.get("quaternion_world_wxyz"))
        require(position.shape == (3,) and np.isfinite(position).all(), f"target position differs: {name}")
        require(np.isfinite(quaternion).all(), f"target quaternion differs: {name}")
        return position, quaternion

    start_p, start_q = np.asarray(start[:3], dtype=np.float64), normalize_quaternion(start[3:])
    approach_p, approach_q = pose("approach_base_link_pose")
    contact_p, contact_q = pose("contact_base_link_pose_at_exact_reset_cube")
    final_p, final_q = pose("stage_target_base_link_pose")
    phases = (
        ("open_approach", 120, start_p, start_q, approach_p, approach_q, 0.0),
        ("open_descent", 120, approach_p, approach_q, contact_p, contact_q, 0.0),
        ("normal_close", 90, contact_p, contact_q, contact_p, contact_q, 1.0),
        ("closed_lift_to_registered_stage_target", 180, contact_p, contact_q, final_p, final_q, 1.0),
        ("closed_settle_at_registered_stage_target", 300, final_p, final_q, final_p, final_q, 1.0),
    )
    rows: list[tuple[str, int, float, list[float]]] = []
    for phase, count, from_p, from_q, to_p, to_q, grip in phases:
        for step in range(1, count + 1):
            fraction = float(step) / float(count)
            position = (1.0 - fraction) * from_p + fraction * to_p
            quaternion = slerp_wxyz(from_q, to_q, fraction)
            action = np.concatenate((position, quaternion, [grip])).astype(np.float32)
            rows.append((phase, step, grip, [float(value) for value in action]))
    require(len(rows) == 810, "recomputed open-contact action count differs")
    return rows


def validate_construction_horizon_activation(lifecycle: Mapping[str, Any], label: str) -> None:
    activation = lifecycle.get("construction_horizon_activation")
    require(isinstance(activation, Mapping), f"{label} horizon activation absent")
    require(
        activation.get("status")
        == "registered_construction_timeout_extended_before_first_reset_or_step",
        f"{label} horizon activation status differs",
    )
    require(activation.get("only_mutated_field") == "env.cfg.episode_length_s", f"{label} mutation differs")
    require(activation.get("original_max_episode_length_steps") == 450, f"{label} original horizon differs")
    require(activation.get("registered_max_episode_length_steps") == 900, f"{label} registered horizon differs")
    require(activation.get("original_episode_length_s") == 30.0, f"{label} original seconds differ")
    require(activation.get("registered_episode_length_s") == 60.0, f"{label} registered seconds differ")
    require(
        math.isclose(float(activation.get("step_dt_s")), 1.0 / 15.0, rel_tol=0.0, abs_tol=1e-12),
        f"{label} step dt differs",
    )
    require(activation.get("common_step_counter_before_and_after") == 0, f"{label} stepped before activation")
    require(activation.get("episode_length_buf_before_and_after") == [0.0], f"{label} episode counter differs")
    require(activation.get("termination_config_byte_equal") is True, f"{label} termination config changed")
    require(
        activation.get("termination_contract_before") == activation.get("termination_contract_after"),
        f"{label} termination snapshots differ",
    )
    require(
        activation.get("termination_contract_before", {}).get("active_terms") == ["success", "time_out"],
        f"{label} termination terms differ",
    )
    require(activation.get("registered_worst_case_steps") == 885, f"{label} worst case differs")
    require(activation.get("registered_margin_steps") == 15, f"{label} margin differs")
    require(activation.get("behavioral_horizon_mutated") is False, f"{label} behavioral horizon changed")


def validate_pose_hold(hold: Mapping[str, Any], label: str) -> bool:
    contract = hold.get("residual_correction_contract")
    require(isinstance(contract, Mapping), f"{label} correction contract absent")
    try:
        validate_contract(contract)
    except ValueError as exc:
        raise ValidationError(f"{label}: {exc}") from exc
    require(
        canonical_sha(contract) == hold.get("residual_correction_contract_sha256"),
        f"{label} correction digest differs",
    )
    require(hold.get("maximum_correction_rounds") == 3, f"{label} round budget differs")
    require(hold.get("hold_steps") == hold.get("hold_steps_per_round") == 30, f"{label} hold differs")
    require(hold.get("required_final_consecutive_steps") == 10, f"{label} window differs")
    require(hold.get("position_error_m_inclusive") == 0.001, f"{label} position threshold differs")
    require(
        hold.get("orientation_geodesic_error_deg_inclusive") == 1.0,
        f"{label} orientation threshold differs",
    )
    rounds = hold.get("correction_rounds")
    require(isinstance(rounds, list) and 1 <= len(rounds) <= 3, f"{label} rounds differ")
    require(hold.get("completed_correction_rounds") == len(rounds), f"{label} round count differs")
    desired = hold.get("target_base_link_pose_world_wxyz")
    require(isinstance(desired, list) and len(desired) == 7, f"{label} desired target differs")
    desired_p, desired_q = desired[:3], desired[3:]
    all_errors: list[Mapping[str, Any]] = []
    all_trace: list[Mapping[str, Any]] = []
    aggregate_finite = True
    aggregate_inside = True
    aggregate_frame = True
    observed_termination = None
    for index, row in enumerate(rounds, start=1):
        require(row.get("round_one_based") == index, f"{label} round order differs")
        require(pclose(row.get("desired_target_position_world_m"), desired_p), f"{label} desired position changed")
        require(qclose(row.get("desired_target_quaternion_world_wxyz"), desired_q), f"{label} desired quaternion changed")
        errors = row.get("errors")
        require(isinstance(errors, list) and 0 <= len(errors) <= 30, f"{label} round errors differ")
        require(row.get("completed_steps") == len(errors), f"{label} completed steps differ")
        round_trace = row.get("construction_action_trace")
        require(isinstance(round_trace, list) and len(round_trace) == len(errors), f"{label} trace length differs")
        command_p = row.get("command_position_world_m")
        command_q = row.get("command_quaternion_world_wxyz")
        for error in errors:
            require(pclose(error.get("desired_target_position_world_m"), desired_p), f"{label} error desired position changed")
            require(qclose(error.get("desired_target_quaternion_world_wxyz"), desired_q), f"{label} error desired quaternion changed")
            require(pclose(error.get("command_position_world_m"), command_p), f"{label} command position drifted within round")
            require(qclose(error.get("command_quaternion_world_wxyz"), command_q), f"{label} command quaternion drifted within round")
            desired_position = np.asarray(error["desired_target_position_world_m"], dtype=np.float64)
            measured_position = np.asarray(error["measured_position_world_m"], dtype=np.float64)
            recomputed_position_error = float(np.linalg.norm(measured_position - desired_position))
            require(
                math.isclose(float(error.get("position_error_m")), recomputed_position_error,
                             rel_tol=0.0, abs_tol=1e-12),
                f"{label} logged position error differs",
            )
            desired_quaternion = normalize_quaternion(error["desired_target_quaternion_world_wxyz"])
            measured_quaternion = normalize_quaternion(error["measured_quaternion_world_wxyz"])
            dot = min(1.0, max(-1.0, abs(float(np.dot(desired_quaternion, measured_quaternion)))))
            recomputed_orientation_error = math.degrees(2.0 * math.acos(dot))
            require(
                math.isclose(float(error.get("orientation_geodesic_error_deg")),
                             recomputed_orientation_error, rel_tol=0.0, abs_tol=1e-9),
                f"{label} logged orientation error differs",
            )
            row_finite = error.get("finite") is True
            row_inside = error.get("arm_inside_soft_joint_limits") is True
            row_frame = error.get("base_link_to_eef_frame_identity", {}).get("passed") is True
            aggregate_finite = aggregate_finite and row_finite
            aggregate_inside = aggregate_inside and row_inside
            aggregate_frame = aggregate_frame and row_frame
        final = errors[-10:]
        recomputed_window = len(final) == 10 and all(
            value.get("position_error_m") <= 0.001
            and value.get("orientation_geodesic_error_deg") <= 1.0
            for value in final
        )
        require(row.get("final_window_passed") is recomputed_window, f"{label} round window flag differs")
        round_unsafe = bool(
            row.get("termination") is not None
            or not aggregate_finite
            or not aggregate_inside
            or not aggregate_frame
            or not errors
        )
        if row.get("termination") is not None:
            require(observed_termination is None, f"{label} multiple terminations")
            observed_termination = row["termination"]
        correction = row.get("measured_residual_correction")
        if not recomputed_window and not round_unsafe and index < 3:
            require(isinstance(correction, Mapping), f"{label} missing correction")
            last = errors[-1]
            expected = corrected_command(
                desired_position=desired_p,
                desired_quaternion=desired_q,
                measured_position=last["measured_position_world_m"],
                measured_quaternion=last["measured_quaternion_world_wxyz"],
                current_command_position=command_p,
                current_command_quaternion=command_q,
                translation_gain=1.0,
                rotation_gain=1.0,
            )
            require(pclose(correction.get("next_command_position_world_m"), expected["next_command_position_world_m"]), f"{label} translation correction differs")
            require(qclose(correction.get("next_command_quaternion_world_wxyz"), expected["next_command_quaternion_world_wxyz"]), f"{label} rotation correction differs")
            require(index < len(rounds), f"{label} correction lacks next round")
            require(pclose(rounds[index]["command_position_world_m"], expected["next_command_position_world_m"]), f"{label} next position command differs")
            require(qclose(rounds[index]["command_quaternion_world_wxyz"], expected["next_command_quaternion_world_wxyz"]), f"{label} next quaternion command differs")
        else:
            require(correction is None, f"{label} correction after pass/final round")
        all_errors.extend(errors)
        all_trace.extend(round_trace)
        if recomputed_window or round_unsafe:
            require(index == len(rounds), f"{label} did not stop at pass/unsafe round")
    require(hold.get("completed_steps") == len(all_errors), f"{label} aggregate step count differs")
    require(hold.get("errors") == all_errors, f"{label} aggregate errors differ")
    require(hold.get("construction_action_trace") == all_trace, f"{label} aggregate trace differs")
    require(hold.get("termination") == observed_termination, f"{label} termination differs")
    require(hold.get("all_states_finite") is aggregate_finite, f"{label} finite aggregate differs")
    require(hold.get("all_arm_states_inside_live_soft_joint_limits") is aggregate_inside,
            f"{label} soft-limit aggregate differs")
    require(hold.get("all_base_link_to_eef_frame_identity_checks_passed") is aggregate_frame,
            f"{label} frame aggregate differs")
    expected_pass = bool(
        rounds[-1].get("final_window_passed") is True
        and observed_termination is None
        and aggregate_finite
        and aggregate_inside
        and aggregate_frame
    )
    require(hold.get("final_window_passed") is rounds[-1].get("final_window_passed"), f"{label} final window differs")
    require(hold.get("desired_target_invariant_across_rounds") is True, f"{label} target-invariance flag differs")
    require(hold.get("passed") is expected_pass, f"{label} pass does not recompute")
    return expected_pass


def validate_scientific_selection(
    report: Mapping[str, Any], harness: Mapping[str, Any], schedule: Mapping[str, Any]
) -> None:
    require(report.get("status") in TERMINAL_STATUSES, "child terminal status differs")
    terminal_pass = TERMINAL_STATUSES[str(report["status"])]
    require(report.get("passed") is terminal_pass, "child pass boolean differs")
    require(report.get("model_request_count") == report.get("behavioral_episode_count") == 0, "child counts differ")
    diagnostics = report.get("known_reachable_diagnostics")
    require(isinstance(diagnostics, list) and 1 <= len(diagnostics) <= 4, "diagnostics differ")
    require(
        [row.get("diagnostic_index_one_based") for row in diagnostics]
        == list(range(1, len(diagnostics) + 1)),
        "diagnostic order differs",
    )
    diagnostic_passes = []
    for row in diagnostics:
        hold_pass = validate_pose_hold(row.get("pose_hold", {}), "diagnostic pose hold")
        require(row.get("passed") is hold_pass, "diagnostic pass differs from hold")
        diagnostic_passes.append(hold_pass)
    require(report.get("r007_live_diagnostic_count") == len(diagnostics), "child diagnostic count differs")
    require(harness.get("r007_live_diagnostic_count") == len(diagnostics), "harness diagnostic count differs")
    attempts = report.get("attempts")
    if report.get("status") == "r007_known_reachable_diagnostic_failed_candidates_not_evaluated":
        require(not all(diagnostic_passes), "diagnostic failure status has no failure")
        require(attempts == [], "failed diagnostic evaluated candidates")
        require(report.get("repair_candidate_evaluation_count") == 0, "failed diagnostic candidate count differs")
        require(report.get("accepted_candidate_rank") is None, "failed diagnostic accepted rank")
    else:
        require(len(diagnostics) == 4 and all(diagnostic_passes), "candidate ran before four passed diagnostics")
        require(isinstance(attempts, list) and 1 <= len(attempts) <= 4, "attempt count differs")
        require([row.get("candidate_rank") for row in attempts] == list(range(1, len(attempts) + 1)), "rank order differs")
        rank_passes: list[bool] = []
        for attempt in attempts:
            rank = int(attempt.get("candidate_rank", -1))
            require(1 <= rank <= 4, "candidate rank differs")
            expected_pair = schedule["candidate_pairs"][rank - 1]
            require(expected_pair.get("candidate_rank") == rank, "frozen candidate rank differs")
            stages = attempt.get("stages", {})
            require(set(stages) == {"canonical_grasp", "canonical_carry"}, "stage set differs")
            stage_passes = []
            for stage_name, stage in stages.items():
                state = stage.get("candidate_state", {})
                evaluated = all(isinstance(state.get(key), Mapping) for key in (
                    "physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate"
                ))
                if evaluated:
                    validate_open_contact_state(
                        state, stage_name, expected_pair[stage_name], candidate_rank=rank
                    )
                expected = evaluated and all(state[key].get("passed") is True for key in (
                    "physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate"
                ))
                require(state.get("passed") is expected, f"{stage_name} pass differs")
                stage_passes.append(expected)
            rank_pass = all(stage_passes)
            require(attempt.get("passed") is rank_pass, "rank pass differs")
            rank_passes.append(rank_pass)
        require(report.get("repair_candidate_evaluation_count") == len(attempts), "evaluated count differs")
        if terminal_pass:
            require(rank_passes[-1] and not any(rank_passes[:-1]), "first-pass selection differs")
            require(report.get("accepted_candidate_rank") == len(attempts), "accepted rank differs")
            require(report.get("accepted_states") == attempts[-1]["stages"], "accepted states differ")
        else:
            require(len(attempts) == 4 and not any(rank_passes), "exhaustion differs")
            require(report.get("accepted_candidate_rank") is None and report.get("accepted_states") is None, "exhaustion accepted state")
        require(report.get("first_passing_rule_obeyed") is True, "first-pass flag differs")
    require(harness.get("process_completed") is True, "harness process not complete")
    require(harness.get("status") == "completed_r007_candidate_search", "harness status differs")
    require(harness.get("scientific_gate_passed") is terminal_pass, "harness scientific flag differs")
    require(harness.get("child_status") == report.get("status"), "harness child status differs")


def validate_terminal_selection_rule(
    report: Mapping[str, Any], schedule: Mapping[str, Any]
) -> None:
    require(
        report.get("selection_rule") == schedule.get("selection_rule"),
        "terminal selection rule is not the exact frozen candidate schedule",
    )


def validate_open_contact_state(
    state: Mapping[str, Any],
    label: str,
    expected_stage: Mapping[str, Any],
    *,
    candidate_rank: int,
) -> None:
    construction = state.get("construction")
    require(isinstance(construction, Mapping), f"{label} construction absent")
    require(construction.get("method") == "exact_reset_open_approach_normal_close_lift", f"{label} method differs")
    require(construction.get("stage") == label, f"{label} construction stage differs")
    require(construction.get("candidate_rank") == candidate_rank, f"{label} candidate rank differs")
    require(
        construction.get("registered_stage_schedule") == expected_stage,
        f"{label} registered stage schedule differs",
    )
    require(
        construction.get("registered_targets") == expected_stage.get("r007_open_contact_targets"),
        f"{label} registered targets differ",
    )
    require(construction.get("gate_window_final_steps") == 10, f"{label} gate window differs")
    require(construction.get("phase_steps") == {
        "open_approach": 120, "open_descent": 120, "normal_close": 90,
        "closed_lift_to_registered_stage_target": 180,
        "closed_settle_at_registered_stage_target": 300,
    }, f"{label} phase counts differ")
    require(construction.get("episode_length_buf_before_candidate_actions") == [75], f"{label} pre-action counter differs")
    require(construction.get("episode_length_buf_after_candidate_actions") == [885], f"{label} post-action counter differs")
    require(construction.get("post_reset_joint_state_write_count") == 0, f"{label} joint state rewritten")
    require(construction.get("post_reset_object_state_write_count") == 0, f"{label} object state rewritten")
    trace = construction.get("construction_action_trace")
    require(isinstance(trace, list) and len(trace) == 810, f"{label} trace count differs")
    recomputed = expected_open_contact_commands(construction, expected_stage)
    vector_fields = {
        "eef_position_world_m": 3,
        "eef_quaternion_world_wxyz": 4,
        "base_link_position_world_m": 3,
        "base_link_quaternion_world_wxyz": 4,
        "joint_position_rad": 13,
        "joint_velocity_rad_s": 13,
        "cube_position_world_m": 3,
        "cube_quaternion_world_wxyz": 4,
        "cube_linear_velocity_m_s": 3,
        "cube_angular_velocity_rad_s": 3,
    }
    for index, (row, (phase, phase_step, grip, expected_action)) in enumerate(
        zip(trace, recomputed, strict=True), start=1
    ):
        require(row.get("phase") == phase, f"{label} trace {index} phase differs")
        require(row.get("phase_step_one_based") == phase_step, f"{label} trace {index} step differs")
        actual_action = row.get("command_action_8d")
        require(isinstance(actual_action, list) and len(actual_action) == 8, f"{label} trace {index} action differs")
        require(
            np.array_equal(
                np.asarray(actual_action, dtype=np.float32),
                np.asarray(expected_action, dtype=np.float32),
            ),
            f"{label} trace {index} 7-D pose command differs",
        )
        require(float(actual_action[7]) == grip, f"{label} trace {index} gripper differs")
        for field, width in vector_fields.items():
            value = row.get(field)
            require(
                isinstance(value, list)
                and len(value) == width
                and np.isfinite(np.asarray(value, dtype=np.float64)).all(),
                f"{label} trace {index} {field} differs",
            )
        frame = row.get("base_link_to_eef_frame_identity")
        require(isinstance(frame, Mapping) and frame.get("passed") is True, f"{label} trace {index} frame identity failed")
        require(
            float(frame.get("position_composition_residual_m", math.inf)) <= 1e-6
            and float(frame.get("orientation_composition_residual_deg", math.inf)) <= 1e-4,
            f"{label} trace {index} frame residual differs",
        )

    samples = construction.get("settled_gate_samples")
    require(isinstance(samples, list) and len(samples) == 10, f"{label} settled samples differ")
    for index, (sample, row) in enumerate(zip(samples, trace[-10:], strict=True), start=1):
        shared = (
            ("cube_position_world_m", "cube_position_world_m"),
            ("cube_linear_velocity_m_s", "cube_linear_velocity_m_s"),
            ("cube_angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
            ("eef_position_world_m", "base_link_position_world_m"),
            ("base_link_quaternion_world_wxyz", "base_link_quaternion_world_wxyz"),
            ("live_eef_frame_position_world_m", "eef_position_world_m"),
            ("live_eef_frame_quaternion_world_wxyz", "eef_quaternion_world_wxyz"),
        )
        for sample_key, trace_key in shared:
            require(sample.get(sample_key) == row.get(trace_key), f"{label} final sample {index} {sample_key} differs")
        require(
            sample.get("arm_joint_velocity_rad_s") == row.get("joint_velocity_rad_s", [])[:7],
            f"{label} final sample {index} arm velocity differs",
        )
        require(
            sample.get("base_link_to_eef_frame_identity")
            == row.get("base_link_to_eef_frame_identity"),
            f"{label} final sample {index} frame evidence differs",
        )
        require(isinstance(sample.get("contact_force_n"), Mapping), f"{label} final sample {index} contacts absent")
        require(
            all(math.isfinite(float(value)) for value in sample["contact_force_n"].values()),
            f"{label} final sample {index} contact force is nonfinite",
        )
        require(isinstance(sample.get("object_grabbed"), bool), f"{label} final sample {index} grasp differs")

    contact = state.get("contact_evidence", {})
    require(
        contact.get("settled_force_snapshots_n") == [row["contact_force_n"] for row in samples],
        f"{label} final contact samples differ",
    )
    require(
        contact.get("object_grabbed_by_step") == [row["object_grabbed"] for row in samples],
        f"{label} final grasp samples differ",
    )
    last = trace[-1]
    require(state.get("robot", {}).get("joint_position_rad") == last["joint_position_rad"], f"{label} final joint position differs")
    require(state.get("robot", {}).get("joint_velocity_rad_s") == last["joint_velocity_rad_s"], f"{label} final joint velocity differs")
    cube = state.get("objects", {}).get("rubiks_cube", {})
    for state_key, trace_key in (
        ("position_world_m", "cube_position_world_m"),
        ("quaternion_world_wxyz", "cube_quaternion_world_wxyz"),
        ("linear_velocity_m_s", "cube_linear_velocity_m_s"),
        ("angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
    ):
        require(cube.get(state_key) == last[trace_key], f"{label} final cube {state_key} differs")
    require(state.get("eef", {}).get("position_world_m") == last["base_link_position_world_m"], f"{label} final base position differs")
    require(state.get("eef", {}).get("quaternion_world_wxyz") == last["base_link_quaternion_world_wxyz"], f"{label} final base quaternion differs")


def validate_static(
    root: Path, *, require_source_gate: bool = True, verify_retry_history: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    artifact = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007"
    registration_path = artifact / "repair_registration.json"
    schedule_path = artifact / "gates/candidate_schedule.json"
    source_gate_path = artifact / "source_push_gate.json"
    registration, schedule = load(registration_path), load(schedule_path)
    require(registration.get("repair_amendment_id") == "V3-E006-R007", "registration ID differs")
    require(registration.get("status") == REGISTRATION_STATUS, "registration status differs")
    require(registration.get("predecessor_repair_amendment_id") == "V3-E006-R006", "predecessor ID differs")
    require(registration.get("counts_at_registration") == {
        "r007_live_diagnostics": 0, "r007_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }, "registration counts differ")
    for commit in (BASE, R006_CLOSURE_COMMIT):
        require(subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0, f"lineage commit absent: {commit}")
    for name, row in registration["frozen_inputs"].items():
        resolve_bound(root, row, f"registration frozen input {name}")
    predecessor = registration["r006_predecessor"]
    require(predecessor.get("closure_commit") == R006_CLOSURE_COMMIT, "R006 closure commit differs")
    require(predecessor["results"].get("sha256") == R006_RESULTS_SHA256, "R006 results digest differs")
    require(predecessor["raw_result"].get("sha256") == R006_RAW_RESULT_SHA256, "R006 raw digest differs")
    predecessor_results_path = resolve_bound(root, predecessor["results"], "R006 predecessor results")
    try:
        validate_r006_exhaustion_closure(load(predecessor_results_path))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    require(schedule.get("repair_amendment_id") == "V3-E006-R007", "schedule ID differs")
    require(schedule.get("status") == "frozen_before_any_r007_live_diagnostic_candidate_or_model_request", "schedule status differs")
    require(schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4, "budgets differ")
    require([row.get("candidate_rank") for row in schedule.get("candidate_pairs", [])] == [1, 2, 3, 4], "rank order differs")
    require([row.get("diagnostic_index_one_based") for row in schedule.get("known_reachable_diagnostics", [])] == [1, 2, 3, 4], "diagnostic order differs")
    require(all(schedule.get(key) == 0 for key in (
        "model_request_count", "behavioral_episode_count", "r007_live_diagnostic_count",
        "r007_live_candidate_evaluation_count",
    )), "schedule counts differ")
    verify_binding(registration_path, schedule["repair_registration"], "schedule registration")
    require(schedule.get("r006_predecessor") == predecessor, "schedule predecessor differs")
    correction = schedule.get("residual_correction_contract", {})
    try:
        validate_contract(correction)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    correction_sha = canonical_sha(correction)
    require(correction_sha == schedule.get("residual_correction_contract_sha256"), "correction digest differs")
    horizon = schedule.get("construction_horizon_contract", {})
    horizon_sha = canonical_sha(horizon)
    require(horizon_sha == schedule.get("construction_horizon_contract_sha256"), "horizon digest differs")
    require(horizon.get("original_max_episode_length_steps") == 450, "original horizon differs")
    require(horizon.get("registered_max_episode_length_steps") == 900, "registered horizon differs")
    require(horizon.get("original_episode_length_s") == 30.0, "original seconds differ")
    require(horizon.get("registered_episode_length_s") == 60.0, "registered seconds differ")
    require(horizon.get("required_step_dt_s") == 1.0 / 15.0, "step dt differs")
    require(horizon.get("registered_max_episode_length_steps") == 900, "horizon roles differ")
    worst = horizon.get("worst_case_derivation", {})
    require(worst == {
        "fresh_reset_steps": 75,
        "waypoint_count": 8,
        "maximum_correction_rounds_per_waypoint": 3,
        "hold_steps_per_round": 30,
        "worst_case_steps": 795,
        "registered_margin_steps": 105,
    }, "worst-case derivation differs")
    require(75 + 8 * 3 * 30 == 795 < 900, "registered horizon does not cover worst case")
    require(horizon.get("behavioral_episode_horizon_unchanged") is True, "behavioral horizon changed")

    open_contact = schedule.get("open_contact_construction_contract", {})
    open_contact_sha = canonical_sha(open_contact)
    require(open_contact_sha == schedule.get("open_contact_construction_contract_sha256"), "open-contact digest differs")
    require(registration.get("open_contact_construction_contract") == open_contact, "registration open-contact contract differs")
    require(registration.get("open_contact_construction_contract_sha256") == open_contact_sha, "registration open-contact digest differs")
    require(open_contact.get("phase_steps") == {
        "open_approach": 120, "open_descent": 120, "normal_close": 90,
        "closed_lift_to_registered_stage_target": 180,
        "closed_settle_at_registered_stage_target": 300,
    }, "open-contact phase counts differ")
    require(open_contact.get("candidate_action_steps") == 810, "candidate action total differs")
    require(open_contact.get("worst_case_materialization_steps") == 885, "materialization total differs")
    require(75 + 810 == 885 and 900 - 885 == 15, "open-contact horizon arithmetic differs")

    r006_schedule_path = resolve_bound(root, schedule["r006_target_solver_horizon_schedule"], "R006 target/solver/horizon schedule")
    r006_schedule = load(r006_schedule_path)
    require(schedule["known_reachable_diagnostics"] == r006_schedule["known_reachable_diagnostics"], "R006 diagnostics changed")
    stripped_pairs = deepcopy(schedule["candidate_pairs"])
    for pair in stripped_pairs:
        pair["construction_method"] = "direct_contact_initialization"
        for stage_name in ("canonical_grasp", "canonical_carry"):
            pair[stage_name].pop("r007_open_contact_targets")
    require(stripped_pairs == r006_schedule["candidate_pairs"], "R006 candidate targets/order changed beyond registered construction fields")
    require(correction == r006_schedule["residual_correction_contract"], "R006 residual correction changed")
    require(horizon == r006_schedule["construction_horizon_contract"], "R006 construction horizon changed")
    require(
        schedule.get("residual_correction_contract_sha256")
        == r006_schedule.get("residual_correction_contract_sha256"),
        "R006 residual-correction digest changed",
    )

    r006_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/state_repair_gate.py"
    r007_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r007/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_finalize_unchanged_gates",
    ):
        require(function_ast(r006_source, name) == function_ast(r007_source, name), f"unchanged helper differs: {name}")
    source_text = r007_source.read_text(encoding="utf-8")
    require("requests.post" not in source_text and "httpx" not in source_text and "policy_server" not in source_text, "model endpoint exists")
    tree = ast.parse(source_text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    pose_dump = ast.dump(functions["_run_registered_pose_hold"], include_attributes=False)
    require("corrected_command" in pose_dump and "maximum_rounds" in pose_dump, "correction loop absent")
    require("translation_gain" in pose_dump and "rotation_gain" in pose_dump, "registered gains absent")
    require("candidate_schedule" in pose_dump, "correction does not load frozen schedule")
    for caller in ("_run_known_reachable_diagnostic", "_solve_registered_ik"):
        require("_run_registered_pose_hold" in ast.dump(functions[caller], include_attributes=False), f"{caller} bypasses correction")
    command_dump = ast.dump(functions["_command_base_link"], include_attributes=False)
    require("EEF_OFFSET" not in command_dump and "_quat_inverse" not in command_dump, "base-link command applies offset")
    materialize_dump = ast.dump(functions["_open_contact_materialize_and_gate"], include_attributes=False)
    for prohibited in ("write_joint_state_to_sim", "write_root_pose_to_sim", "write_root_velocity_to_sim"):
        require(prohibited not in materialize_dump, f"post-reset state write entered R007: {prohibited}")
    require("810" in materialize_dump and "885" in materialize_dump, "open-contact action/counter total differs")
    require(source_text.count("state = _open_contact_materialize_and_gate(") == 1, "open-contact path is not sole candidate dispatch")
    require("state = _direct_materialize_and_gate(" not in source_text, "legacy injected-state path remains reachable")

    source_gate_summary = None
    if require_source_gate:
        require(source_gate_path.is_file(), "R007 source-push gate missing")
        source_gate = load(source_gate_path)
        require(source_gate.get("schema_version") == SOURCE_GATE_SCHEMA, "source-gate schema differs")
        require(source_gate.get("status") == SOURCE_GATE_STATUS, "source-gate status differs")
        try:
            validate_source_gate(
                source_gate, study_root=root, verify_raw_history=verify_retry_history
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        implementation = str(source_gate.get("implementation_commit", ""))
        require(subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{implementation}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0, "implementation commit absent")
        inventory = source_gate.get("implementation_files")
        require(
            isinstance(inventory, list)
            and len(inventory) == source_gate.get("implementation_file_count") == 13,
            "source inventory differs",
        )
        for row in inventory:
            relative = Path(str(row.get("path", "")))
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
            verify_binding(root / relative, row, f"source inventory {relative}")
        verify_binding(registration_path, source_gate["repair_registration"], "source-gate registration")
        verify_binding(schedule_path, source_gate["candidate_schedule"], "source-gate schedule")
        source_gate_summary = binding(source_gate_path)
    return {
        "passed": True,
        "registration": binding(registration_path),
        "candidate_schedule": binding(schedule_path),
        "r006_predecessor_results": predecessor["results"],
        "residual_correction_contract_sha256": correction_sha,
        "construction_horizon_contract_sha256": horizon_sha,
        "open_contact_construction_contract_sha256": open_contact_sha,
        "candidate_action_steps": 810,
        "candidate_pair_count": 4,
        "diagnostic_count": 4,
        "source_push_gate": source_gate_summary,
    }


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    launch_path, harness_path, runtime_path = (
        candidate_root / "launch.json", candidate_root / "harness_result.json",
        candidate_root / "runtime.log",
    )
    launch, harness = load(launch_path), load(harness_path)
    require(harness.get("model_request_count") == harness.get("behavioral_episode_count") == 0, "harness counts differ")
    verify_binding(launch_path, harness["launch"], "harness launch")
    verify_binding(runtime_path, harness["runtime_log"], "harness runtime")
    for name, row in launch.get("input_bindings", {}).items():
        verify_binding(Path(str(row["path"])), row, f"launch input {name}")
    for name, row in launch.get("formal_health_preflight", {}).items():
        verify_binding(Path(str(row["path"])), row, f"health input {name}")
    verify_binding(Path(str(launch["harness_source"]["path"])), launch["harness_source"], "harness source")
    child_binding = harness.get("child_report")
    require(isinstance(child_binding, Mapping), "child report absent")
    report_path = Path(str(child_binding["path"])); verify_binding(report_path, child_binding, "child report")
    report = load(report_path)
    if harness.get("process_completed") is True:
        require(report_path.name == "state_repair_result.json", "terminal report filename differs")
        schedule = load(
            root
            / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007/gates/candidate_schedule.json"
        )
        validate_terminal_selection_rule(report, schedule)
        validate_scientific_selection(report, harness, schedule)
        for diagnostic in report["known_reachable_diagnostics"]:
            validate_construction_horizon_activation(
                diagnostic["environment_lifecycle"], "diagnostic environment"
            )
            require(diagnostic["environment_lifecycle"]["closed_before_next_environment"] is True, "diagnostic env not closed")
            require(diagnostic["fresh_reset"]["passed"] is True, "diagnostic reset failed")
            for camera in diagnostic["fresh_reset"]["camera_evidence"]["bindings"].values():
                verify_binding(Path(str(camera["rgb"]["path"])), camera["rgb"], "diagnostic camera")
        for attempt in report.get("attempts", []):
            for stage in attempt["stages"].values():
                validate_construction_horizon_activation(
                    stage["materialization_environment"]["environment_lifecycle"],
                    "materialization environment",
                )
                require(stage["materialization_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "material env not closed")
                require(stage["materialization_environment"]["fresh_reset"]["passed"] is True, "material reset failed")
        for key in (
            "repair_registration", "candidate_schedule", "source_push_gate",
            "original_v3e006_closure_binding", "r006_predecessor_results", "ood_freeze",
            "e004_full_reset_reference", "e004_candidate", "construction_source", "video",
        ):
            verify_binding(Path(str(report[key]["path"])), report[key], f"report {key}")
        verify_binding(Path(str(report["frozen_e004_runtime_bindings"]["path"])), report["frozen_e004_runtime_bindings"], "report runtime")
        for row in report["scene_assets"].values():
            verify_binding(Path(str(row["path"])), row, "report scene")
    else:
        require(report_path.name == "state_construction_failure.json", "invalid report filename differs")
        require(report.get("status") == "infrastructure_invalid_r007_state_repair", "invalid status differs")
        require(report.get("model_request_count") == report.get("behavioral_episode_count") == report.get("state_candidate_count") == 0, "invalid counts differ")
        for row in report.get("available_raw_artifacts", {}).values():
            verify_binding(Path(str(row["path"])), row, "invalid raw artifact")
    evidence = report.get("execution_evidence", report)
    require(evidence.get("passed_health_preflight") == launch.get("formal_health_preflight"), "child health differs")
    for group in ("input_bindings", "passed_health_preflight"):
        for name, row in evidence.get(group, {}).items():
            verify_binding(Path(str(row["path"])), row, f"child {group}/{name}")
    for source in evidence.get("historical_source_verification_before_AppLauncher", {}).values():
        for name, row in source.get("bindings", {}).items():
            verify_binding(Path(str(row["path"])), row, f"historical source {name}")
    for name, row in evidence.get("controller_source_verification_before_AppLauncher", {}).items():
        verify_binding(Path(str(row["path"])), row, f"controller source {name}")
    return {
        "passed": True, "candidate_root": str(candidate_root), "launch": binding(launch_path),
        "harness": binding(harness_path), "child_report": binding(report_path),
        "child_status": report.get("status"), "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--pre-source-gate", action="store_true")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    result = validate_static(
        args.study_root,
        require_source_gate=not args.pre_source_gate,
        verify_retry_history=args.verify_raw,
    )
    if args.candidate_root is not None:
        require(args.verify_raw, "--candidate-root requires --verify-raw")
        result["candidate_evidence"] = validate_candidate_root(args.study_root, args.candidate_root)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
