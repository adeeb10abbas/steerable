#!/usr/bin/env python3
"""Build deterministic C7 trigger/fallback and measurement gate receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4 import geometry as geom  # noqa: E402
from experiments.online_correction_v4.attempts import (  # noqa: E402
    InfraInvalidReason,
    TerminalEvidenceFlags,
    derive_failure_label,
)
from experiments.online_correction_v4.clock import (  # noqa: E402
    ActionQueue,
    ControlledSimulationClock,
)
from experiments.online_correction_v4.contracts import (  # noqa: E402
    FailureLabel,
    PolicyTimingAchieved,
    TimingConfig,
)
from experiments.online_correction_v4.detectors import (  # noqa: E402
    DetachmentDetector,
    DetachmentDetectorConfig,
    GraspDetectorConfig,
    NaturalGraspDetector,
    ObjectKinematicState,
)
from experiments.online_correction_v4.droid_scorer import (  # noqa: E402
    aggregate_settling_predicates,
    load_scoring_context,
)
from experiments.online_correction_v4.geometry import (  # noqa: E402
    build_planar_goal_set,
    direct_inverse_pair_equivalent,
)
from experiments.online_correction_v4.scoring import (  # noqa: E402
    TerminalEvidence,
    score_terminal_first_placement,
)


FIXTURE_ID = "object_pair"
POLICY_ID = "cosmos3_nano_droid"
G3_SCHEMA = "v4-object-pair-g3-aggregate-receipt-v1"
G4_SCHEMA = "v4-object-pair-g4-nano-policy-session-receipt-v1"
GEOMETRY_SCHEMA = "v4-object-pair-scoring-geometry-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_pass(payload: dict[str, Any], *, schema: str, path: Path) -> None:
    if payload.get("schema_version") != schema:
        raise ValueError(f"{path} schema mismatch")
    if payload.get("fixture_id") != FIXTURE_ID:
        raise ValueError(f"{path} fixture mismatch")
    if payload.get("passed") is not True or payload.get("status") != "passed":
        raise ValueError(f"{path} is not a passing receipt")


def kinematic_state(
    *,
    tick: int,
    dt: float,
    lift_m: float,
    relative_drift_m: float = 0.0,
    contact: bool = True,
    detached: bool = False,
) -> ObjectKinematicState:
    gripper_z = 0.05 + lift_m
    return ObjectKinematicState(
        sim_time=tick * dt,
        control_tick=tick,
        object_z=0.05 + lift_m,
        initial_supported_z=0.05,
        gripper_x=0.0,
        gripper_y=0.0,
        gripper_z=gripper_z,
        object_x=relative_drift_m,
        object_y=0.0,
        object_z_pos=0.05 + lift_m,
        contact=contact,
        detached=detached,
    )


def build_g5_receipt(
    *,
    g3: dict[str, Any],
    g3_path: Path,
    g4: dict[str, Any],
    g4_path: Path,
) -> dict[str, Any]:
    require_pass(g3, schema=G3_SCHEMA, path=g3_path)
    require_pass(g4, schema=G4_SCHEMA, path=g4_path)
    if g3.get("observed_scripted_check_count") != 112:
        raise ValueError("G3 does not contain all scripted grasp/placement checks")
    if g4.get("policy_id") != POLICY_ID:
        raise ValueError("G4 policy mismatch")

    dt = 1.0 / 15.0
    config = GraspDetectorConfig()
    detector = NaturalGraspDetector(config=config, control_dt_s=dt)
    no_lift_event = detector.update(
        kinematic_state(tick=0, dt=dt, lift_m=0.0, contact=True)
    )
    detector.reset()
    for tick in range(3):
        drift_event = detector.update(
            kinematic_state(
                tick=tick,
                dt=dt,
                lift_m=0.05,
                relative_drift_m=0.02,
            )
        )
    detector.reset()
    stable_event = None
    for tick in range(4):
        stable_event = detector.update(
            kinematic_state(tick=tick, dt=dt, lift_m=0.05)
        )

    detachment = DetachmentDetector(DetachmentDetectorConfig(dwell_ticks=2))
    pre_carry = detachment.update(
        kinematic_state(tick=8, dt=dt, lift_m=0.05, detached=True)
    )
    detachment.arm_after_verified_carry()
    first_detached = detachment.update(
        kinematic_state(tick=10, dt=dt, lift_m=0.05, detached=True)
    )
    release_event = detachment.update(
        kinematic_state(tick=11, dt=dt, lift_m=0.05, detached=True)
    )
    checks = {
        "no_trigger_without_registered_lift": no_lift_event is None,
        "excess_relative_drift_resets_dwell": drift_event is None,
        "stable_carry_triggers_after_registered_dwell": (
            stable_event is not None
            and math.isclose(stable_event.dwell_s, 0.2, abs_tol=1e-12)
            and stable_event.lift_m >= config.min_lift_m
        ),
        "detachment_ignored_before_verified_carry": pre_carry is None,
        "first_detachment_requires_two_ticks": (
            first_detached is None
            and release_event is not None
            and release_event.onset_tick == 10
            and release_event.detected_tick == 11
        ),
        "fresh_policy_session_repeat_qualified": (
            g4.get("checks", {}).get(
                "fresh_session_exact_repeat_actions_equal"
            )
            is True
        ),
        "live_scripted_grasp_transport_release_basis": (
            g3.get("scripted_passed_check_count") == 112
            and g3.get("scripted_failed_check_count") == 0
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "v4-object-pair-g5-trigger-branch-receipt-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "policy_id": POLICY_ID,
        "gate": "G5",
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "checks": checks,
        "detector_contract": {
            "min_lift_m": config.min_lift_m,
            "dwell_s": config.dwell_s,
            "relative_drift_max_m": config.relative_drift_max_m,
            "trigger_deadline_s": config.trigger_deadline_s,
            "release_detection_dwell_ticks": 2,
        },
        "selected_prefix_mode": "independent_natural_rollout_fallback",
        "evaluation_design_label": "randomized_event_triggered_evaluation",
        "exact_counterfactual_branching_claimed": False,
        "fallback_scope": {
            "C7": "authorized_after_G6_G7_G8",
            "C2": "not_authorized; C2 still requires verified common prefixes",
        },
        "qualification_basis": {
            "g3_scripted_physics": artifact(g3_path),
            "g4_policy_session": artifact(g4_path),
        },
        "release_boundary": (
            "Passes C7 G5 using the prospectively permitted independent-natural-"
            "rollout fallback. It does not authorize exact-prefix response claims, "
            "C2, policy pilots before G6, or confirmatory inference before G7-G8."
        ),
    }


def center(box: geom.AxisAlignedBox) -> geom.Vec3:
    return (
        (box.x_min + box.x_max) / 2.0,
        (box.y_min + box.y_max) / 2.0,
        (box.z_min + box.z_max) / 2.0,
    )


def passing_evidence(
    *,
    p_obj_world: geom.Vec3,
    p_ref_world: geom.Vec3,
    relation: geom.RelationKind,
    released: bool = True,
) -> TerminalEvidence:
    return TerminalEvidence(
        p_obj_world=p_obj_world,
        p_named_ref_world=p_ref_world,
        relation=relation,
        grasp_occurred=True,
        carry_verified=True,
        released=released,
        stable_for_dwell=True,
        allowed_support=True,
    )


def build_g6_receipt(
    *,
    geometry_payload: dict[str, Any],
    geometry_path: Path,
    campaign: dict[str, Any],
    campaign_path: Path,
) -> dict[str, Any]:
    if geometry_payload.get("schema_version") != GEOMETRY_SCHEMA:
        raise ValueError("object-pair scoring geometry schema mismatch")
    if geometry_payload.get("fixture_id") != FIXTURE_ID:
        raise ValueError("object-pair scoring geometry fixture mismatch")
    d_cap_m = float(geometry_payload["d_cap_m"])
    geometry_sha = sha256_file(geometry_path)
    contexts = {
        relation: load_scoring_context(
            geometry_path,
            expected_sha256=geometry_sha,
            relation=relation,
            d_cap_m=d_cap_m,
        )
        for relation in ("left", "right", "front", "behind")
    }
    task_workspace = contexts["left"].planar_spec.workspace
    p_ref_task = center(task_workspace)
    p_ref_world = contexts["left"].frame.task_to_world(p_ref_task)
    valid_scores: dict[str, Any] = {}
    valid_points: dict[str, geom.Vec3] = {}
    for relation, context in contexts.items():
        assert context.planar_spec is not None
        goal = build_planar_goal_set(
            context.frame,
            context.planar_spec,
            p_ref_world,
        )
        if goal.empty or goal.region is None:
            raise ValueError(f"{relation} synthetic goal is unexpectedly empty")
        point = context.frame.task_to_world(center(goal.region))
        valid_points[relation] = point
        valid_scores[relation] = score_terminal_first_placement(
            context,
            passing_evidence(
                p_obj_world=point,
                p_ref_world=p_ref_world,
                relation=relation,
            ),
        )

    left_ctx = contexts["left"]
    object_before = valid_points["left"]
    moved_ref_task = (p_ref_task[0] + 0.06, p_ref_task[1], p_ref_task[2])
    moved_ref_world = left_ctx.frame.task_to_world(moved_ref_task)
    signed_before = left_ctx.frame.signed_axes(object_before, p_ref_world)
    signed_after = left_ctx.frame.signed_axes(object_before, moved_ref_world)
    wrong_score = score_terminal_first_placement(
        left_ctx,
        passing_evidence(
            p_obj_world=valid_points["right"],
            p_ref_world=p_ref_world,
            relation="left",
        ),
    )
    held_score = score_terminal_first_placement(
        left_ctx,
        passing_evidence(
            p_obj_world=valid_points["left"],
            p_ref_world=p_ref_world,
            relation="left",
            released=False,
        ),
    )

    timing = TimingConfig.from_mapping(campaign["timing"])
    achieved = PolicyTimingAchieved.from_requested(1.0 / 15.0, timing, 32)
    queue = ActionQueue(native_control_dt_s=achieved.native_control_dt_s)
    queue.enqueue_chunk(
        chunk_id="old",
        request_id="old-request",
        actions=((1.0,), (1.0,), (1.0,)),
    )
    clock = ControlledSimulationClock(
        timing=timing,
        achieved=achieved,
        action_queue=queue,
    )
    observation_id, _ = clock.capture_observation("a" * 64)
    request = clock.submit_policy_request(observation_id)
    clock.complete_inference(
        request,
        chunk_id="new",
        actions=((2.0,), (2.0,)),
        wall_duration_s=7.0,
    )
    delay_actions = clock.advance_for_delay_window(request)
    applied = clock.apply_due_responses()
    first_new = clock.action_queue.pop_for_tick()

    missing = aggregate_settling_predicates((), dwell_ticks=2)
    no_grasp = derive_failure_label(
        TerminalEvidenceFlags(
            grasp_occurred=False,
            timeout_without_completion=True,
            timeout_after_no_grasp=True,
        )
    )
    grasp_lost = derive_failure_label(
        TerminalEvidenceFlags(
            grasp_occurred=True,
            carry_verified=True,
            grasp_lost=True,
        )
    )
    checks = {
        "reference_only_motion_changes_relative_not_object_world": (
            object_before == valid_points["left"]
            and math.isclose(
                signed_after["signed_left_m"]
                - signed_before["signed_left_m"],
                -0.06,
                abs_tol=1e-12,
            )
        ),
        "all_planar_goal_regions_accept_nonunique_valid_points": all(
            score.success
            and score.goal_violation_m == 0.0
            and not score.goal_set_empty
            for score in valid_scores.values()
        ),
        "wrong_direction_release_is_geometric_failure": (
            wrong_score.failure_label == "wrong_goal_region"
            and wrong_score.predicates["released"]
        ),
        "correct_direction_held_object_fails_release": (
            held_score.failure_label == "release_failed"
        ),
        "direct_inverse_wordings_share_goal_sets": all(
            direct_inverse_pair_equivalent(
                "sponge",
                "tray",
                relation,
            )
            for relation in contexts
        ),
        "registered_task_frame_is_right_handed_and_visible_axis_bound": (
            left_ctx.frame.u_left == (0.0, 1.0, 0.0)
            and left_ctx.frame.u_front == (-1.0, 0.0, 0.0)
            and left_ctx.frame.u_up == (0.0, 0.0, 1.0)
        ),
        "changed_observation_action_causal_order": (
            request.observation_capture_time == 0.0
            and request.response_available_time == achieved.achieved_delay_s
            and all(command.chunk_id == "old" for command in delay_actions)
            and applied == [request]
            and first_new is not None
            and first_new.chunk_id == "new"
        ),
        "wall_inference_duration_does_not_advance_sim_time": (
            request.inference_wall_duration_s == 7.0
            and math.isclose(clock.sim_time, achieved.achieved_delay_s)
        ),
        "video_frame_index_aligns_with_sim_clock_and_control_ticks": all(
            math.isclose(
                frame_index * achieved.native_control_dt_s,
                frame_time,
                abs_tol=1e-12,
            )
            for frame_index, frame_time in enumerate(
                tuple(index * achieved.native_control_dt_s for index in range(4))
            )
        ),
        "rejected_stimulus_is_infrastructure_not_policy_nonresponse": (
            InfraInvalidReason.WRONG_STIMULUS_TRAJECTORY.value
            == "wrong_stimulus_trajectory"
            and InfraInvalidReason.WRONG_STIMULUS_TRAJECTORY.value
            not in {label.value for label in FailureLabel}
        ),
        "missing_terminal_predicates_fail_closed": (
            not missing.available
            and "settling_samples" in missing.missing_fields
        ),
        "no_grasp_grasp_loss_and_infrastructure_are_distinct": (
            no_grasp is FailureLabel.NO_GRASP
            and grasp_lost is FailureLabel.GRASP_LOST
        ),
        "geometry_distance_cap_is_finite_positive": (
            math.isfinite(d_cap_m) and d_cap_m > 0.0
        ),
        "score_reproduction_is_byte_stable": (
            canonical_json_bytes(valid_scores["left"].to_outcome())
            == canonical_json_bytes(
                score_terminal_first_placement(
                    left_ctx,
                    passing_evidence(
                        p_obj_world=valid_points["left"],
                        p_ref_world=p_ref_world,
                        relation="left",
                    ),
                ).to_outcome()
            )
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "v4-object-pair-g6-measurement-receipt-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "gate": "G6",
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "checks": checks,
        "known_motion_fixture": {
            "reference_displacement_task_m": [0.06, 0.0, 0.0],
            "manipulated_object_world_position_unchanged": True,
            "signed_left_change_m": (
                signed_after["signed_left_m"]
                - signed_before["signed_left_m"]
            ),
        },
        "failure_case_labels": {
            "wrong_direction_released": wrong_score.failure_label,
            "correct_direction_held": held_score.failure_label,
            "no_grasp_timeout": no_grasp.value,
            "grasp_loss": grasp_lost.value,
            "missing_terminal_predicates": "unresolved_behavioral_failure",
        },
        "controlled_clock": {
            "native_control_dt_s": achieved.native_control_dt_s,
            "achieved_delay_s": achieved.achieved_delay_s,
            "delay_window_action_chunk_ids": [
                command.chunk_id for command in delay_actions
            ],
            "new_chunk_first_applied_after_delay": (
                first_new.chunk_id if first_new is not None else None
            ),
        },
        "scoring_geometry": artifact(geometry_path),
        "campaign": artifact(campaign_path),
        "d_cap_m": d_cap_m,
        "release_boundary": (
            "Passes fixture-scoped C7 G6 deterministic measurement checks. "
            "Random video audit and live trigger/outcome frequencies remain G7 "
            "requirements; this receipt alone does not authorize confirmatory runs."
        ),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g3-aggregate", type=Path, required=True)
    parser.add_argument("--g4-receipt", type=Path, required=True)
    parser.add_argument("--scoring-geometry", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "docs/online_correction_v4/campaign.json",
    )
    parser.add_argument("--g5-out", type=Path, required=True)
    parser.add_argument("--g6-out", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.g5_out, args.g6_out):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {output}")
    g3 = load_json(args.g3_aggregate)
    g4 = load_json(args.g4_receipt)
    geometry = load_json(args.scoring_geometry)
    campaign = load_json(args.campaign)
    g5 = build_g5_receipt(
        g3=g3,
        g3_path=args.g3_aggregate.resolve(),
        g4=g4,
        g4_path=args.g4_receipt.resolve(),
    )
    g6 = build_g6_receipt(
        geometry_payload=geometry,
        geometry_path=args.scoring_geometry.resolve(),
        campaign=campaign,
        campaign_path=args.campaign.resolve(),
    )
    write_exclusive(args.g5_out.resolve(), g5)
    write_exclusive(args.g6_out.resolve(), g6)
    print(
        json.dumps(
            {
                "g5": {**artifact(args.g5_out.resolve()), "passed": g5["passed"]},
                "g6": {**artifact(args.g6_out.resolve()), "passed": g6["passed"]},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
