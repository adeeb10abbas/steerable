#!/usr/bin/env python3
"""Static and target-raw validator for V3-E006-R008."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r008.object_servo import (
    object_space_servo_command,
    validate_contract as validate_servo_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r008.predecessor_contract import (
    R007_CLOSURE_COMMIT,
    R007_RESULTS_SHA256,
    validate_r007_exhaustion_closure,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r008.source_gate_contract import (
    SCHEMA as SOURCE_SCHEMA,
    STATUS as SOURCE_STATUS,
    validate_source_gate,
)


ROOT = Path(__file__).resolve().parents[1]
ART = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008")
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"


class ValidationError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ValidationError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(path: Path, row: Mapping[str, Any], label: str) -> Path:
    require(path.is_file() and path.stat().st_size == row.get("bytes") and sha(path) == row.get("sha256"), f"{label} binding differs")
    return path


def resolve(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    return verify(path, row, label)


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)
    return ast.dump(node, include_attributes=False)


def quat_normalize(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    require(q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 0, "quaternion malformed")
    return q / np.linalg.norm(q)


def slerp(left: Sequence[float], right: Sequence[float], fraction: float) -> np.ndarray:
    a, b = quat_normalize(left), quat_normalize(right)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 1 - 1e-12:
        return quat_normalize((1 - fraction) * a + fraction * b)
    theta = math.acos(dot)
    return quat_normalize(
        math.sin((1 - fraction) * theta) / math.sin(theta) * a
        + math.sin(fraction * theta) / math.sin(theta) * b
    )


def same_float32(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(np.array_equal(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


def expected_precontact(construction: Mapping[str, Any]) -> list[tuple[str, int, list[float]]]:
    targets = construction["registered_precontact_targets"]
    start = construction["start_base_link_pose_world_wxyz"]
    points = {
        "start": (np.asarray(start[:3]), quat_normalize(start[3:])),
        "approach": (
            np.asarray(targets["approach_base_link_pose"]["position_world_m"]),
            quat_normalize(targets["approach_base_link_pose"]["quaternion_world_wxyz"]),
        ),
        "contact": (
            np.asarray(targets["contact_base_link_pose_at_exact_reset_cube"]["position_world_m"]),
            quat_normalize(targets["contact_base_link_pose_at_exact_reset_cube"]["quaternion_world_wxyz"]),
        ),
    }
    phases = (
        ("open_approach", 120, "start", "approach", 0.0),
        ("open_descent", 120, "approach", "contact", 0.0),
        ("normal_close", 90, "contact", "contact", 1.0),
    )
    rows = []
    for phase, count, left, right, grip in phases:
        p0, q0 = points[left]; p1, q1 = points[right]
        for index in range(1, count + 1):
            f = index / count
            action = np.concatenate(((1-f)*p0 + f*p1, slerp(q0, q1, f), [grip])).astype(np.float32)
            rows.append((phase, index, action.tolist()))
    return rows


def validate_construction_lifecycle(lifecycle: Mapping[str, Any], label: str) -> None:
    activation = lifecycle.get("construction_horizon_activation")
    require(isinstance(activation, Mapping), f"{label} horizon activation absent")
    require(
        activation.get("status")
        == "registered_construction_timeout_extended_before_first_reset_or_step",
        f"{label} horizon activation status differs",
    )
    require(activation.get("only_mutated_field") == "env.cfg.episode_length_s", f"{label} mutation differs")
    require(activation.get("original_max_episode_length_steps") == 450, f"{label} original horizon differs")
    require(activation.get("registered_max_episode_length_steps") == 1500, f"{label} registered horizon differs")
    require(activation.get("original_episode_length_s") == 30.0, f"{label} original seconds differ")
    require(activation.get("registered_episode_length_s") == 100.0, f"{label} registered seconds differs")
    require(
        math.isclose(float(activation.get("step_dt_s")), 1.0 / 15.0, rel_tol=0.0, abs_tol=1e-12),
        f"{label} step dt differs",
    )
    require(activation.get("common_step_counter_before_and_after") == 0, f"{label} stepped before activation")
    require(activation.get("episode_length_buf_before_and_after") in ([0], [0.0]), f"{label} episode counter differs")
    require(activation.get("termination_config_byte_equal") is True, f"{label} termination config changed")
    require(
        activation.get("termination_contract_before") == activation.get("termination_contract_after"),
        f"{label} termination snapshots differ",
    )
    require(
        activation.get("termination_contract_before", {}).get("active_terms") == ["success", "time_out"],
        f"{label} termination terms differ",
    )
    require(activation.get("registered_worst_case_steps") == 1365, f"{label} worst case differs")
    require(activation.get("registered_margin_steps") == 135, f"{label} margin differs")
    require(activation.get("behavioral_horizon_mutated") is False, f"{label} behavioral horizon changed")


def quaternion_equivalent(left: Sequence[float], right: Sequence[float], *, atol: float = 1e-12) -> bool:
    a, b = quat_normalize(left), quat_normalize(right)
    return bool(abs(float(np.dot(a, b))) >= 1.0 - atol)


def validate_candidate_state(
    state: Mapping[str, Any],
    expected_stage: Mapping[str, Any],
    rank: int,
    schedule: Mapping[str, Any],
) -> None:
    construction = state.get("construction")
    require(isinstance(construction, Mapping), "candidate construction absent")
    require(construction.get("method") == "exact_reset_open_close_uniform_object_servo_q_handoff", "construction method differs")
    retained = deepcopy(expected_stage); retained["candidate_rank"] = rank
    require(construction.get("registered_stage_schedule") == retained, "retained stage schedule differs")
    require(construction.get("registered_target_cube_pose") == expected_stage["r008_target_cube_pose"], "cube target differs")
    for retained_key, schedule_key in (
        ("object_space_servo_contract", "object_space_servo_contract"),
        ("joint_handoff_contract", "joint_handoff_contract"),
        ("construction_lifecycle_contract", "construction_lifecycle_contract"),
    ):
        require(
            construction.get(retained_key) == schedule.get(schedule_key),
            f"retained {retained_key} differs from global schedule",
        )
    require(construction.get("phase_steps") == {
        "open_approach": 120, "open_descent": 120, "normal_close": 90,
        "closed_object_space_servo": 360, "captured_q_normal_joint_settle": 600,
    }, "phase counts differ")
    require(construction.get("episode_length_buf_before_candidate_actions") == [75], "start counter differs")
    require(construction.get("episode_length_buf_before_handoff") == [765], "handoff counter differs")
    require(construction.get("episode_length_buf_after_candidate_actions") == [1365], "final counter differs")
    require(construction.get("post_reset_joint_state_write_count") == construction.get("post_reset_object_state_write_count") == 0, "post-reset state write differs")
    require(construction.get("captured_joint_target_write_count") == 1, "q target write count differs")
    require(construction.get("cartesian_action_manager_apply_count_during_joint_settle") == 0, "Cartesian settle action differs")
    require(construction.get("joint_or_object_state_write_count") == 0, "settle state write differs")
    trace = construction.get("construction_action_trace")
    servo_trace = construction.get("object_space_servo_trace")
    require(isinstance(trace, list) and len(trace) == 1290, "trace length differs")
    require(isinstance(servo_trace, list) and servo_trace == trace[330:690], "servo subtrace differs")
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
    for index, row in enumerate(trace, start=1):
        for field, width in vector_fields.items():
            value = row.get(field)
            require(
                isinstance(value, list)
                and len(value) == width
                and np.isfinite(np.asarray(value, dtype=np.float64)).all(),
                f"trace {index} {field} differs",
            )
        frame = row.get("base_link_to_eef_frame_identity")
        require(isinstance(frame, Mapping) and frame.get("passed") is True, f"trace {index} frame identity failed")
        require(
            float(frame.get("position_composition_residual_m", math.inf)) <= 1e-6
            and float(frame.get("orientation_composition_residual_deg", math.inf)) <= 1e-4,
            f"trace {index} frame residual differs",
        )
    for row, (phase, step, action) in zip(trace[:330], expected_precontact(construction), strict=True):
        require(row.get("phase") == phase and row.get("phase_step_one_based") == step, "precontact phase differs")
        require(same_float32(row.get("command_action_8d", []), action), "precontact command differs")
    contract = construction["object_space_servo_contract"]
    for index, row in enumerate(servo_trace, start=1):
        evidence = row.get("pre_action_object_space_servo")
        require(isinstance(evidence, Mapping), "servo pre-action evidence absent")
        previous = trace[329 + index - 1]
        require(
            np.allclose(
                evidence.get("live_base_position_world_m"),
                previous.get("base_link_position_world_m"), atol=1e-12, rtol=0,
            ),
            f"servo {index} base pre-state is not chained",
        )
        require(
            quaternion_equivalent(
                evidence.get("live_base_quaternion_world_wxyz", []),
                previous.get("base_link_quaternion_world_wxyz", []),
            ),
            f"servo {index} base quaternion pre-state is not chained",
        )
        require(
            np.allclose(
                evidence.get("live_cube_position_world_m"),
                previous.get("cube_position_world_m"), atol=1e-12, rtol=0,
            ),
            f"servo {index} cube pre-state is not chained",
        )
        require(
            quaternion_equivalent(
                evidence.get("live_cube_quaternion_world_wxyz", []),
                previous.get("cube_quaternion_world_wxyz", []),
            ),
            f"servo {index} cube quaternion pre-state is not chained",
        )
        require(
            evidence.get("target_cube_position_world_m")
            == construction["registered_target_cube_pose"]["position_world_m"],
            f"servo {index} target position differs from frozen target",
        )
        require(
            quaternion_equivalent(
                evidence.get("target_cube_quaternion_world_wxyz", []),
                construction["registered_target_cube_pose"]["quaternion_world_wxyz"],
            ),
            f"servo {index} target quaternion differs from frozen target",
        )
        recomputed = object_space_servo_command(
            live_base_position=evidence["live_base_position_world_m"],
            live_base_quaternion=evidence["live_base_quaternion_world_wxyz"],
            live_cube_position=evidence["live_cube_position_world_m"],
            live_cube_quaternion=evidence["live_cube_quaternion_world_wxyz"],
            target_cube_position=evidence["target_cube_position_world_m"],
            target_cube_quaternion=evidence["target_cube_quaternion_world_wxyz"],
            translation_gain=contract["translation_gain"],
            rotation_gain=contract["rotation_gain"],
            translation_cap_m_per_step=contract["translation_cap_m_per_step"],
            rotation_cap_deg_per_step=contract["rotation_cap_deg_per_step"],
        )
        for key, expected in recomputed.items():
            actual = evidence.get(key)
            if isinstance(expected, list):
                require(np.allclose(actual, expected, atol=1e-12, rtol=0), f"servo {index} {key} differs")
            else:
                require(actual == expected, f"servo {index} {key} differs")
        expected_action = [*recomputed["command_base_position_world_m"], *recomputed["command_base_quaternion_world_wxyz"], 1.0]
        require(same_float32(row.get("command_action_8d", []), expected_action), f"servo {index} command differs")
    target = construction.get("captured_joint_position_target_rad")
    require(isinstance(target, list) and len(target) == 13 and np.isfinite(target).all(), "captured q differs")
    require(
        same_float32(target, servo_trace[-1].get("joint_position_rad", [])),
        "captured q is not the exact servo-step-360 observed q",
    )
    for index, row in enumerate(trace[690:], start=1):
        require(row.get("phase") == "captured_q_normal_joint_settle", "settle phase differs")
        require(row.get("phase_step_one_based") == index, "settle step differs")
        require(row.get("normal_joint_position_target_rad") == target, "settle q target changed")
        require(row.get("cartesian_action_manager_applied") is False, "Cartesian action applied during settle")
        for key in (
            "joint_position_rad", "joint_velocity_rad_s", "cube_position_world_m",
            "cube_quaternion_world_wxyz", "cube_linear_velocity_m_s", "cube_angular_velocity_rad_s",
        ):
            require(np.isfinite(np.asarray(row.get(key), dtype=np.float64)).all(), f"settle {key} nonfinite")
        require(row.get("base_link_to_eef_frame_identity", {}).get("passed") is True, "frame identity failed")
    samples = construction.get("settled_gate_samples")
    require(isinstance(samples, list) and len(samples) == 10, "final gate window differs")
    for sample, row in zip(samples, trace[-10:], strict=True):
        for sample_key, trace_key in (
            ("cube_position_world_m", "cube_position_world_m"),
            ("cube_linear_velocity_m_s", "cube_linear_velocity_m_s"),
            ("cube_angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
            ("eef_position_world_m", "base_link_position_world_m"),
            ("base_link_quaternion_world_wxyz", "base_link_quaternion_world_wxyz"),
            ("live_eef_frame_position_world_m", "eef_position_world_m"),
            ("live_eef_frame_quaternion_world_wxyz", "eef_quaternion_world_wxyz"),
        ):
            require(sample.get(sample_key) == row.get(trace_key), f"final sample {sample_key} differs")
        require(sample.get("arm_joint_velocity_rad_s") == row.get("joint_velocity_rad_s")[:7], "final arm velocity differs")
        require(sample.get("base_link_to_eef_frame_identity") == row.get("base_link_to_eef_frame_identity"), "final frame sample differs")
        require(isinstance(sample.get("object_grabbed"), bool), "final grasp evidence differs")
        require(isinstance(sample.get("contact_force_n"), Mapping), "final contact evidence differs")
        require(
            all(math.isfinite(float(value)) for value in sample["contact_force_n"].values()),
            "final contact force is nonfinite",
        )
    last = trace[-1]
    require(state.get("robot", {}).get("joint_position_rad") == last.get("joint_position_rad"), "final robot q differs")
    require(state.get("robot", {}).get("joint_velocity_rad_s") == last.get("joint_velocity_rad_s"), "final robot qd differs")
    cube = state.get("objects", {}).get("rubiks_cube", {})
    for state_key, trace_key in (
        ("position_world_m", "cube_position_world_m"),
        ("quaternion_world_wxyz", "cube_quaternion_world_wxyz"),
        ("linear_velocity_m_s", "cube_linear_velocity_m_s"),
        ("angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
    ):
        require(cube.get(state_key) == last.get(trace_key), f"final cube {state_key} differs")
    require(state.get("eef", {}).get("position_world_m") == last.get("base_link_position_world_m"), "final base position differs")
    require(state.get("eef", {}).get("quaternion_world_wxyz") == last.get("base_link_quaternion_world_wxyz"), "final base quaternion differs")
    contact = state.get("contact_evidence", {})
    require(contact.get("settled_force_snapshots_n") == [row["contact_force_n"] for row in samples], "contact samples differ")
    require(contact.get("object_grabbed_by_step") == [row["object_grabbed"] for row in samples], "grasp samples differ")
    require(state.get("physics_gate", {}).get("settled_window_steps") == 10, "physics window differs")
    require(state.get("passed") is all((
        state.get("physics_gate", {}).get("passed"),
        state.get("ood_gate", {}).get("passed"),
        state.get("camera_evidence", {}).get("passed"),
        state.get("companion_pose_gate", {}).get("passed"),
    )), "candidate pass recomputation differs")


def validate_static(root: Path, *, source_gate_required: bool) -> dict[str, Any]:
    root = root.resolve(); artifact = root / ART
    registration = load(artifact / "repair_registration.json")
    schedule = load(artifact / "gates/candidate_schedule.json")
    require(registration.get("repair_amendment_id") == "V3-E006-R008", "registration ID differs")
    require(registration.get("predecessor_repair_amendment_id") == "V3-E006-R007", "predecessor differs")
    require(registration.get("counts_at_registration") == {
        "r008_live_diagnostics": 0, "r008_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }, "registration counts differ")
    for commit in (BASE, R007_CLOSURE_COMMIT):
        require(subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode == 0, f"missing commit {commit}")
    predecessor_path = resolve(root, registration["r007_predecessor"]["results"], "R007 predecessor")
    require(sha(predecessor_path) == R007_RESULTS_SHA256, "R007 results digest differs")
    validate_r007_exhaustion_closure(load(predecessor_path))
    require(schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4, "budgets differ")
    require([row.get("candidate_rank") for row in schedule["candidate_pairs"]] == [1,2,3,4], "rank order differs")
    require(schedule.get("r007_predecessor") == registration.get("r007_predecessor"), "predecessor schedule differs")
    for name, contract in (
        ("object_space_servo", schedule["object_space_servo_contract"]),
        ("joint_handoff", schedule["joint_handoff_contract"]),
        ("construction_lifecycle", schedule["construction_lifecycle_contract"]),
    ):
        require(canonical_sha(contract) == schedule[f"{name}_contract_sha256"], f"{name} digest differs")
    validate_servo_contract(schedule["object_space_servo_contract"])
    require(schedule["joint_handoff_contract"]["settle_steps"] == 600, "handoff settle differs")
    require(schedule["construction_lifecycle_contract"]["worst_case_steps"] == 1365, "horizon derivation differs")
    require(schedule["construction_lifecycle_contract"]["registered_max_episode_length_steps"] == 1500, "construction horizon differs")
    r007_schedule = load(resolve(root, schedule["r007_candidate_schedule"], "R007 schedule"))
    require("construction_horizon_contract" not in schedule, "inherited 900-step horizon remains active")
    require("open_contact_construction_contract" not in schedule, "inherited 900-step open-contact contract remains active")
    archived = schedule.get("archived_predecessor_contracts", {})
    require(
        archived.get("status") == "archived_lineage_only_not_active_r008_runtime_evidence",
        "predecessor contracts are not explicitly archived",
    )
    require(
        archived.get("r005_construction_horizon_contract")
        == r007_schedule.get("construction_horizon_contract")
        and archived.get("r005_construction_horizon_contract_sha256")
        == r007_schedule.get("construction_horizon_contract_sha256"),
        "archived R005 horizon differs",
    )
    require(
        archived.get("r007_open_contact_construction_contract")
        == r007_schedule.get("open_contact_construction_contract")
        and archived.get("r007_open_contact_construction_contract_sha256")
        == r007_schedule.get("open_contact_construction_contract_sha256"),
        "archived R007 construction contract differs",
    )
    stripped = deepcopy(schedule["candidate_pairs"])
    for pair in stripped:
        pair["construction_method"] = "exact_reset_open_approach_normal_close_lift"
        for stage in ("canonical_grasp", "canonical_carry"):
            pair[stage].pop("r008_target_cube_pose")
            pair[stage].pop("r008_precontact_targets")
    require(stripped == r007_schedule["candidate_pairs"], "candidate sources/targets/order changed")
    require(schedule["known_reachable_diagnostics"] == r007_schedule["known_reachable_diagnostics"], "diagnostics changed")
    r007_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r007/state_repair_gate.py"
    r008_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_finalize_unchanged_gates",
    ):
        require(function_ast(r007_source, name) == function_ast(r008_source, name), f"scientific helper differs: {name}")
    source = r008_source.read_text(encoding="utf-8")
    require(
        "from experiments.v3.phase_e.canonical_stage_localization_v3e006.state_contract import" in source
        and "from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import" in source,
        "frozen scientific gate imports differ",
    )
    require("requests.post" not in source and "policy_server" not in source, "model endpoint exists")
    material = function_ast(r008_source, "_open_contact_materialize_and_gate")
    for prohibited in ("write_joint_state_to_sim", "write_root_pose_to_sim", "write_root_velocity_to_sim"):
        require(prohibited not in material, f"state write entered materializer: {prohibited}")
    require("object_space_servo_command" in material and "_normal_joint_equilibrium_step" in material, "R008 controller topology absent")
    gate_summary = None
    if source_gate_required:
        gate_path = artifact / "source_push_gate.json"
        require(gate_path.is_file(), "source gate absent")
        gate = load(gate_path)
        require(gate.get("schema_version") == SOURCE_SCHEMA and gate.get("status") == SOURCE_STATUS, "source gate differs")
        try:
            validate_source_gate(gate, study_root=root, verify_raw_history=True)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        gate_summary = binding(gate_path)
    return {
        "passed": True,
        "registration": binding(artifact / "repair_registration.json"),
        "candidate_schedule": binding(artifact / "gates/candidate_schedule.json"),
        "source_push_gate": gate_summary,
        "candidate_pair_count": 4,
        "diagnostic_count": 4,
        "object_space_servo_contract_sha256": schedule["object_space_servo_contract_sha256"],
        "joint_handoff_contract_sha256": schedule["joint_handoff_contract_sha256"],
        "construction_lifecycle_contract_sha256": schedule["construction_lifecycle_contract_sha256"],
    }


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    launch_path = candidate_root / "launch.json"; harness_path = candidate_root / "harness_result.json"; runtime_path = candidate_root / "runtime.log"
    launch, harness = load(launch_path), load(harness_path)
    verify(launch_path, harness["launch"], "harness launch"); verify(runtime_path, harness["runtime_log"], "runtime log")
    require(harness.get("model_request_count") == harness.get("behavioral_episode_count") == 0, "harness counts differ")
    child_path = verify(Path(harness["child_report"]["path"]), harness["child_report"], "child report")
    report = load(child_path)
    schedule = load(root / ART / "gates/candidate_schedule.json")
    attempts = report.get("attempts")
    require(report.get("r008_live_diagnostic_count") == 4, "diagnostic count differs")
    for diagnostic in report.get("known_reachable_diagnostics", []):
        require(diagnostic.get("passed") is True, "reachable diagnostic failed")
        validate_construction_lifecycle(diagnostic.get("environment_lifecycle", {}), "diagnostic environment")
        require(diagnostic.get("environment_lifecycle", {}).get("closed_before_next_environment") is True, "diagnostic env not closed")
        require(diagnostic.get("fresh_reset", {}).get("passed") is True, "diagnostic reset failed")
        for camera in diagnostic.get("fresh_reset", {}).get("camera_evidence", {}).get("bindings", {}).values():
            verify(Path(camera["rgb"]["path"]), camera["rgb"], "diagnostic camera")
    require(isinstance(attempts, list) and [row.get("candidate_rank") for row in attempts] == list(range(1,len(attempts)+1)), "attempt order differs")
    for attempt in attempts:
        rank = attempt["candidate_rank"]; expected_pair = schedule["candidate_pairs"][rank-1]
        for stage in ("canonical_grasp", "canonical_carry"):
            state = attempt["stages"][stage]["candidate_state"]
            validate_candidate_state(state, expected_pair[stage], rank, schedule)
            materialization = attempt["stages"][stage]["materialization_environment"]
            validate_construction_lifecycle(materialization["environment_lifecycle"], "materialization environment")
            require(materialization["environment_lifecycle"]["closed_before_next_environment"] is True, "materialization env not closed")
            require(materialization["fresh_reset"].get("passed") is True, "materialization reset failed")
            for camera in state.get("camera_evidence", {}).get("bindings", {}).values():
                verify(Path(camera["rgb"]["path"]), camera["rgb"], "candidate camera")
        require(attempt.get("passed") is all(attempt["stages"][stage]["candidate_state"]["passed"] for stage in ("canonical_grasp","canonical_carry")), "pair pass differs")
    passed = [row for row in attempts if row.get("passed")]
    if report.get("passed"):
        require(len(passed) == 1 and passed[0] is attempts[-1], "first pass differs")
        require(report.get("accepted_candidate_rank") == attempts[-1]["candidate_rank"], "accepted rank differs")
    else:
        require(not passed and len(attempts) == 4 and report.get("accepted_candidate_rank") is None, "exhaustion differs")
    for group in (launch.get("input_bindings", {}), launch.get("formal_health_preflight", {})):
        for row in group.values():
            verify(Path(row["path"]), row, "launch evidence")
    for key in (
        "repair_registration", "candidate_schedule", "source_push_gate",
        "original_v3e006_closure_binding", "r007_predecessor_results", "ood_freeze",
        "e004_full_reset_reference", "e004_candidate", "construction_source", "video",
    ):
        verify(Path(report[key]["path"]), report[key], f"report {key}")
    verify(Path(report["frozen_e004_runtime_bindings"]["path"]), report["frozen_e004_runtime_bindings"], "runtime binding")
    for row in report["scene_assets"].values():
        verify(Path(row["path"]), row, "scene asset")
    return {"passed": True, "launch": binding(launch_path), "harness": binding(harness_path), "child_report": binding(child_path), "model_request_count": 0, "behavioral_episode_count": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--pre-source-gate", action="store_true")
    parser.add_argument("--verify-raw", action="store_true")
    parser.add_argument("--candidate-root", type=Path)
    args = parser.parse_args()
    result = validate_static(args.study_root, source_gate_required=not args.pre_source_gate)
    if args.candidate_root:
        require(args.verify_raw, "candidate root requires --verify-raw")
        result["candidate_evidence"] = validate_candidate_root(args.study_root.resolve(), args.candidate_root.resolve())
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
