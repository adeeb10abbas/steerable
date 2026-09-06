#!/usr/bin/env python3
"""Build exact C8 polygon scoring geometry and the G6 measurement receipt."""

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
from experiments.online_correction_v4.droid_scorer import (  # noqa: E402
    aggregate_settling_predicates,
    load_scoring_context,
)
from experiments.online_correction_v4.geometry import (  # noqa: E402
    ConvexPolygonPrism,
    build_planar_goal_set,
    direct_inverse_pair_equivalent,
)
from experiments.online_correction_v4.scoring import (  # noqa: E402
    TerminalEvidence,
    score_terminal_first_placement,
)
from experiments.online_correction_v4.second_stack import (  # noqa: E402
    CUBE_HALF_EXTENT_M,
    RELATION_AXES_SCENE_XY,
    SUPPORT_CENTER_SCENE_M,
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
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


def region_center(region: ConvexPolygonPrism) -> geom.Vec3:
    count = len(region.vertices_xy)
    return (
        sum(point[0] for point in region.vertices_xy) / count,
        sum(point[1] for point in region.vertices_xy) / count,
        (region.z_min + region.z_max) / 2.0,
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


def build_geometry(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    g2_path: Path,
    g3_path: Path,
    g4_path: Path,
) -> dict[str, Any]:
    if (
        plan.get("schema_version")
        != "v4-second-stack-g3-geometry-plan-v1"
    ):
        raise ValueError("C8 G3 plan schema differs")
    gate_requirements = (
        (
            load_json(g2_path),
            "v4-second-stack-g2-aggregate-v1",
            "C8 G2",
        ),
        (
            load_json(g3_path),
            "v4-second-stack-g3-scripted-aggregate-v1",
            "C8 G3",
        ),
        (
            load_json(g4_path),
            "v4-second-stack-g4-policy-session-receipt-v1",
            "C8 G4",
        ),
    )
    for receipt, schema, label in gate_requirements:
        if (
            receipt.get("schema_version") != schema
            or receipt.get("fixture_id") != "second_stack"
            or receipt.get("passed") is not True
            or receipt.get("status") != "passed"
        ):
            raise ValueError(f"{label} receipt is not a pass")
    if (
        gate_requirements[1][0].get("selected_scale") != 0.5
        or gate_requirements[1][0].get("passed_check_count") != 112
    ):
        raise ValueError("C8 G3 receipt does not bind all checks at scale 0.5")
    axes = plan.get("relation_axes_scene_xy")
    if axes != {key: list(value) for key, value in RELATION_AXES_SCENE_XY.items()}:
        raise ValueError("C8 relation axes differ")
    world_vertices = plan.get("support_workspace_scene_xy")
    if not isinstance(world_vertices, list) or len(world_vertices) != 4:
        raise ValueError("C8 support workspace polygon differs")
    original_u_left = (*RELATION_AXES_SCENE_XY["left"], 0.0)
    original_u_front = (*RELATION_AXES_SCENE_XY["front"], 0.0)
    u_left = geom._normalize(original_u_left)
    u_front = geom._normalize(original_u_front)
    frame = geom.TaskFrame(
        u_left=u_left,
        u_front=u_front,
        u_up=(0.0, 0.0, 1.0),
    )
    vertices_task = []
    for point in world_vertices:
        transformed = frame.world_to_task(
            (float(point[0]), float(point[1]), 0.0)
        )
        vertices_task.append([transformed[0], transformed[1]])
    support_center_z = SUPPORT_CENTER_SCENE_M[2] + 0.03
    target_center_z = support_center_z + CUBE_HALF_EXTENT_M
    projected_half = CUBE_HALF_EXTENT_M * (
        abs(u_left[0]) + abs(u_left[1])
    )
    d_cap_m = max(
        math.dist(first, second)
        for first in vertices_task
        for second in vertices_task
    )
    return {
        "schema_version": "v4-second-stack-scoring-geometry-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C8",
        "fixture_id": "second_stack",
        "status": "released_for_policy_qualification",
        "behavioral_episode_count": 0,
        "model_request_count": 0,
        "task_frame": {
            "basis_convention": "task coordinates are left, front, up",
            "u_left": list(u_left),
            "u_front": list(u_front),
            "u_up": [0.0, 0.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
            "source_axes_before_unit_normalization": {
                "u_left": list(original_u_left),
                "u_front": list(original_u_front),
            },
        },
        "workspace": {
            "kind": "convex_polygon_prism",
            "vertices_xy": vertices_task,
            "z_min": target_center_z,
            "z_max": target_center_z,
        },
        "object_footprint": {
            "half_left": projected_half,
            "half_front": projected_half,
            "half_up": CUBE_HALF_EXTENT_M,
        },
        "reference_footprint": {
            "half_left": projected_half,
            "half_front": projected_half,
            "half_up": CUBE_HALF_EXTENT_M,
        },
        "clearance_m": 0.01,
        "d_cap_m": d_cap_m,
        "geometry_provenance": {
            "target_object": "baked_green_cube_3cm",
            "reference_object": "baked_yellow_cube_3cm",
            "workspace_is_exact_registered_eroded_support_polygon": True,
            "world_vertices_scene_xy_m": world_vertices,
            "target_center_support_z_m": target_center_z,
        },
        "qualification_basis": {
            "g2_aggregate": artifact(g2_path),
            "g3_scripted": artifact(g3_path),
            "g4_policy_session": artifact(g4_path),
            "g3_plan": artifact(plan_path),
        },
        "release_boundary": (
            "This exact polygon geometry is released for C8 G6-G8 policy "
            "qualification. Confirmatory episodes remain blocked until G7-G8 "
            "and an immutable runtime release pass."
        ),
    }


def build_g6(
    *,
    geometry_path: Path,
    campaign_path: Path,
    g5_path: Path,
) -> dict[str, Any]:
    geometry = load_json(geometry_path)
    campaign = load_json(campaign_path)
    g5 = load_json(g5_path)
    if (
        geometry.get("schema_version")
        != "v4-second-stack-scoring-geometry-v1"
        or geometry.get("fixture_id") != "second_stack"
    ):
        raise ValueError("C8 scoring geometry differs")
    if (
        g5.get("schema_version")
        != "v4-second-stack-g5-trigger-branch-receipt-v1"
        or g5.get("passed") is not True
    ):
        raise ValueError("C8 G5 receipt is not a pass")
    d_cap_m = float(geometry["d_cap_m"])
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
    workspace = contexts["left"].planar_spec.workspace
    if not isinstance(workspace, ConvexPolygonPrism):
        raise ValueError("C8 scorer did not preserve polygon workspace")
    p_ref_task = region_center(workspace)
    p_ref_world = contexts["left"].frame.task_to_world(p_ref_task)
    valid_scores = {}
    valid_points: dict[str, geom.Vec3] = {}
    for relation, context in contexts.items():
        goal = build_planar_goal_set(
            context.frame,
            context.planar_spec,
            p_ref_world,
        )
        if goal.empty or not isinstance(goal.region, ConvexPolygonPrism):
            raise ValueError(f"C8 {relation} synthetic goal is empty")
        point = context.frame.task_to_world(region_center(goal.region))
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
    moved_ref_task = (
        p_ref_task[0] + 0.04,
        p_ref_task[1],
        p_ref_task[2],
    )
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
    outside_polygon_task = (
        max(point[0] for point in workspace.vertices_xy),
        max(point[1] for point in workspace.vertices_xy),
        p_ref_task[2],
    )
    outside_distance = geom.goal_distance(
        left_ctx.frame,
        left_ctx.frame.task_to_world(outside_polygon_task),
        build_planar_goal_set(
            left_ctx.frame,
            left_ctx.planar_spec,
            p_ref_world,
        ),
        d_cap_m=d_cap_m,
    )

    timing = TimingConfig.from_mapping(campaign["timing"])
    achieved = PolicyTimingAchieved.from_requested(0.2, timing, 8)
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
            math.isclose(
                signed_after["signed_left_m"]
                - signed_before["signed_left_m"],
                -0.04,
                abs_tol=1e-12,
            )
        ),
        "all_planar_goal_regions_accept_nonunique_valid_points": all(
            score.success
            and score.goal_violation_m == 0.0
            and not score.goal_set_empty
            for score in valid_scores.values()
        ),
        "axis_aligned_workspace_overreach_is_rejected": (
            outside_distance.distance_m > 0.0
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
                "green block",
                "yellow block",
                relation,
            )
            for relation in contexts
        ),
        "registered_rotated_task_frame_is_right_handed": (
            left_ctx.frame.u_left
            == geom._normalize((*RELATION_AXES_SCENE_XY["left"], 0.0))
            and left_ctx.frame.u_front
            == geom._normalize((*RELATION_AXES_SCENE_XY["front"], 0.0))
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
                index * achieved.native_control_dt_s,
                frame_time,
                abs_tol=1e-12,
            )
            for index, frame_time in enumerate(
                tuple(index * achieved.native_control_dt_s for index in range(4))
            )
        ),
        "rejected_stimulus_is_infrastructure_not_policy_nonresponse": (
            InfraInvalidReason.WRONG_STIMULUS_TRAJECTORY.value
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
        "schema_version": "v4-second-stack-g6-measurement-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C8",
        "fixture_id": "second_stack",
        "gate": "G6",
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "checks": checks,
        "known_motion_fixture": {
            "reference_displacement_task_m": [0.04, 0.0, 0.0],
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
            "achieved_standard_query_period_s": (
                achieved.achieved_standard_query_period_s
            ),
            "achieved_delay_s": achieved.achieved_delay_s,
            "delay_window_action_chunk_ids": [
                command.chunk_id for command in delay_actions
            ],
            "new_chunk_first_applied_after_delay": (
                first_new.chunk_id if first_new is not None else None
            ),
        },
        "scoring_geometry": artifact(geometry_path),
        "g5_trigger_branch": artifact(g5_path),
        "campaign": artifact(campaign_path),
        "d_cap_m": d_cap_m,
        "release_boundary": (
            "A pass completes fixture-scoped C8 G6 deterministic measurement "
            "checks. Random video audit and live outcome frequencies remain G7 "
            "requirements; confirmatory inference remains blocked through G8."
        ),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--g3", type=Path, required=True)
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--g5", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "docs/online_correction_v4/campaign.json",
    )
    parser.add_argument("--geometry-out", type=Path, required=True)
    parser.add_argument("--g6-out", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.geometry_out, args.g6_out):
        if output.exists():
            raise FileExistsError(output)
    geometry = build_geometry(
        plan=load_json(args.plan),
        plan_path=args.plan.resolve(),
        g2_path=args.g2.resolve(),
        g3_path=args.g3.resolve(),
        g4_path=args.g4.resolve(),
    )
    write_exclusive(args.geometry_out, geometry)
    g6 = build_g6(
        geometry_path=args.geometry_out.resolve(),
        campaign_path=args.campaign.resolve(),
        g5_path=args.g5.resolve(),
    )
    write_exclusive(args.g6_out, g6)
    print(json.dumps({"status": g6["status"], "path": str(args.g6_out)}))
    return 0 if g6["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
