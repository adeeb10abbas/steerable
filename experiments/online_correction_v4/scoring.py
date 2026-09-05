"""Terminal first-placement scoring for online correction V4.

Consumes settled physical evidence and geometry goal sets. Does not access policy
inputs or privileged oracle targets.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from experiments.online_correction_v4 import geometry as geom

FailureLabel = Literal[
    "success",
    "no_grasp",
    "grasp_lost",
    "transport_incomplete",
    "wrong_goal_region",
    "release_failed",
    "support_or_containment_failed",
    "timeout_without_completion",
    "collision_caused_terminal_failure",
    "model_output_invalid",
    "unresolved_behavioral_failure",
]

FailureStage = Literal[
    "none",
    "pickup",
    "transport",
    "wrong_relation",
    "release",
    "timeout",
    "collision",
    "other",
]

ReferenceMembership = Literal["named", "other", "both", "neither"]


@dataclass(frozen=True)
class TerminalEvidence:
    """Frozen passive-settling evidence at first-placement confirmation."""

    p_obj_world: geom.Vec3
    p_named_ref_world: geom.Vec3
    relation: geom.RelationKind
    p_other_ref_world: geom.Vec3 | None = None
    grasp_occurred: bool = False
    carry_verified: bool = False
    released: bool = False
    stable_for_dwell: bool = False
    allowed_support: bool = False
    allowed_containment: bool = False
    boundary_violation: bool = False
    collision_terminal_failure: bool = False
    timeout_without_completion: bool = False
    grasp_lost: bool = False
    transport_incomplete: bool = False
    model_output_invalid: bool = False
    unresolved_behavioral_failure: bool = False


@dataclass(frozen=True)
class ScoringContext:
    frame: geom.TaskFrame
    d_cap_m: float
    planar_spec: geom.PlanarRelationSpec | None = None
    shelf_spec: geom.ShelfRelationSpec | None = None
    containment_spec: geom.ContainmentSpec | None = None
    other_planar_spec: geom.PlanarRelationSpec | None = None
    geometric_tol_m: float = 0.0


@dataclass
class TerminalScore:
    success: bool
    failure_label: FailureLabel
    failure_stage: FailureStage
    predicates: dict[str, bool]
    goal_violation_m: float
    goal_violation_capped_m: float
    goal_set_empty: bool
    goal_violation_cap_applied: bool
    D_cap_m: float
    signed_left_m: float
    signed_front_m: float
    signed_up_m: float
    reference_membership: ReferenceMembership | None = None
    response_projection_distance_m: float | None = None
    response_projection_capped_m: float | None = None
    component_id: str | None = None
    goal_set_empty_cause: str | None = None

    def to_outcome(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["all_completion_predicates"] = dict(self.predicates)
        return payload


DETAILED_TO_COARSE: dict[FailureLabel, FailureStage] = {
    "success": "none",
    "no_grasp": "pickup",
    "grasp_lost": "transport",
    "transport_incomplete": "transport",
    "wrong_goal_region": "wrong_relation",
    "release_failed": "release",
    "support_or_containment_failed": "release",
    "timeout_without_completion": "timeout",
    "collision_caused_terminal_failure": "collision",
    "model_output_invalid": "other",
    "unresolved_behavioral_failure": "other",
}


def failure_stage_for_label(label: FailureLabel) -> FailureStage:
    return DETAILED_TO_COARSE[label]


def _named_goal(ctx: ScoringContext, evidence: TerminalEvidence) -> geom.GoalSetResult:
    relation = geom.canonical_relation(evidence.relation)
    if relation == "inside":
        if ctx.containment_spec is None:
            raise ValueError("containment_spec required for inside relation")
        return geom.build_containment_goal_set(ctx.frame, ctx.containment_spec, evidence.p_named_ref_world)
    if relation in geom.VERTICAL_RELATIONS:
        if ctx.shelf_spec is None:
            raise ValueError("shelf_spec required for above/below relation")
        return geom.build_shelf_goal_set(ctx.frame, ctx.shelf_spec, evidence.p_named_ref_world)
    if ctx.planar_spec is None:
        raise ValueError("planar_spec required for horizontal relation")
    return geom.build_planar_goal_set(ctx.frame, ctx.planar_spec, evidence.p_named_ref_world)


def _other_goal(ctx: ScoringContext, evidence: TerminalEvidence) -> geom.GoalSetResult | None:
    if evidence.p_other_ref_world is None or ctx.other_planar_spec is None:
        return None
    return geom.build_planar_goal_set(ctx.frame, ctx.other_planar_spec, evidence.p_other_ref_world)


def _geometric_relation_correct(
    ctx: ScoringContext,
    evidence: TerminalEvidence,
    goal: geom.GoalSetResult,
) -> bool:
    if goal.empty:
        return False
    relation = geom.canonical_relation(evidence.relation)
    if relation == "inside":
        return geom.inside_containment(goal, ctx.frame, evidence.p_obj_world, partial_tol=ctx.geometric_tol_m)
    return geom.point_in_goal_set(ctx.frame, evidence.p_obj_world, goal, tol=ctx.geometric_tol_m)


def _allowed_support_or_containment(ctx: ScoringContext, evidence: TerminalEvidence, goal: geom.GoalSetResult) -> bool:
    relation = geom.canonical_relation(evidence.relation)
    if relation == "inside":
        if not evidence.allowed_containment:
            return False
        if ctx.containment_spec is None:
            return evidence.allowed_containment
        return geom.containment_support_ok(
            ctx.containment_spec,
            ctx.frame,
            evidence.p_obj_world,
            evidence.p_named_ref_world,
        )
    if relation in geom.VERTICAL_RELATIONS:
        if not evidence.allowed_support:
            return False
        if ctx.shelf_spec is None:
            return evidence.allowed_support
        return geom.shelf_support_ok(ctx.frame, ctx.shelf_spec, evidence.p_obj_world)
    return evidence.allowed_support


def _required_manipulation_occurred(evidence: TerminalEvidence) -> bool:
    return evidence.grasp_occurred and evidence.carry_verified


def _no_registered_terminal_violation(evidence: TerminalEvidence) -> bool:
    return not evidence.boundary_violation


def derive_failure_label(predicates: dict[str, bool], evidence: TerminalEvidence) -> FailureLabel:
    if predicates["success"]:
        return "success"
    if evidence.model_output_invalid:
        return "model_output_invalid"
    if evidence.unresolved_behavioral_failure:
        return "unresolved_behavioral_failure"
    if evidence.collision_terminal_failure:
        return "collision_caused_terminal_failure"
    if not evidence.grasp_occurred:
        return "no_grasp"
    if evidence.grasp_lost:
        return "grasp_lost"
    if evidence.transport_incomplete or not evidence.carry_verified:
        return "transport_incomplete"
    if not evidence.released:
        return "release_failed"
    if not predicates["geometric_relation_correct"]:
        return "wrong_goal_region"
    if not predicates["allowed_support_or_containment"]:
        return "support_or_containment_failed"
    if not evidence.stable_for_dwell:
        return "support_or_containment_failed"
    if not predicates["no_registered_terminal_violation"]:
        return "collision_caused_terminal_failure"
    if evidence.timeout_without_completion:
        return "timeout_without_completion"
    return "unresolved_behavioral_failure"


def score_terminal_first_placement(
    ctx: ScoringContext,
    evidence: TerminalEvidence,
    *,
    include_response_projection: bool = False,
) -> TerminalScore:
    goal = _named_goal(ctx, evidence)
    distance = geom.goal_distance(ctx.frame, evidence.p_obj_world, goal, d_cap_m=ctx.d_cap_m)
    signed = ctx.frame.signed_axes(evidence.p_obj_world, evidence.p_named_ref_world)

    geometric_ok = _geometric_relation_correct(ctx, evidence, goal)
    manipulation_ok = _required_manipulation_occurred(evidence)
    support_ok = _allowed_support_or_containment(ctx, evidence, goal)
    predicates = {
        "geometric_relation_correct": geometric_ok,
        "required_manipulation_occurred": manipulation_ok,
        "released": evidence.released,
        "allowed_support_or_containment": support_ok,
        "stable_for_registered_dwell": evidence.stable_for_dwell,
        "no_registered_terminal_violation": _no_registered_terminal_violation(evidence),
    }
    predicates["success"] = all(predicates.values())

    failure_label = derive_failure_label(predicates, evidence)
    failure_stage = failure_stage_for_label(failure_label)

    membership: ReferenceMembership | None = None
    other = _other_goal(ctx, evidence)
    if other is not None:
        membership = geom.reference_membership(
            named_goal=goal,
            other_goal=other,
            frame=ctx.frame,
            p_obj_world=evidence.p_obj_world,
            tol=ctx.geometric_tol_m,
        )

    response_distance_m: float | None = None
    response_capped_m: float | None = None
    if include_response_projection:
        response = geom.response_projection_distance(
            ctx.frame, evidence.p_obj_world, goal, d_cap_m=ctx.d_cap_m
        )
        response_distance_m = response.distance_m
        response_capped_m = response.capped_distance_m

    uncapped = distance.distance_m if math.isfinite(distance.distance_m) else float("inf")
    return TerminalScore(
        success=predicates["success"],
        failure_label=failure_label,
        failure_stage=failure_stage,
        predicates=predicates,
        goal_violation_m=uncapped,
        goal_violation_capped_m=distance.capped_distance_m,
        goal_set_empty=distance.goal_set_empty,
        goal_violation_cap_applied=distance.cap_applied,
        D_cap_m=ctx.d_cap_m,
        signed_left_m=signed["signed_left_m"],
        signed_front_m=signed["signed_front_m"],
        signed_up_m=signed["signed_up_m"],
        reference_membership=membership,
        response_projection_distance_m=response_distance_m,
        response_projection_capped_m=response_capped_m,
        component_id=distance.component_id,
        goal_set_empty_cause=goal.empty_cause,
    )


def score_response_horizon(
    ctx: ScoringContext,
    evidence: TerminalEvidence,
    *,
    p_obj_world: geom.Vec3,
    p_ref_world: geom.Vec3 | None = None,
) -> geom.DistanceResult:
    goal = _named_goal(ctx, evidence)
    response_goal = geom.GoalSetResult(
        region=goal.region,
        empty=goal.empty,
        empty_cause=goal.empty_cause,
        projection_kind="response_planar",
        component_id=goal.component_id,
    )
    return geom.response_projection_distance(ctx.frame, p_obj_world, response_goal, d_cap_m=ctx.d_cap_m)


@dataclass(frozen=True)
class ResponseHorizonRequest:
    """Evaluate capped planar response distance at ``t_event_planned + horizon_s``."""

    t_event_planned_s: float
    horizon_s: float
    sham_trajectory: geom.TrajectorySeries
    move_trajectory: geom.TrajectorySeries
    sham_terminal_extension: geom.TrajectorySample | None = None
    move_terminal_extension: geom.TrajectorySample | None = None


@dataclass
class ResponseHorizonScore:
    query_time_s: float
    d_cap_sham_m: float
    d_cap_move_m: float
    h_response_m: float
    goal_set_empty: bool
    sham_terminal_extension_applied: bool
    move_terminal_extension_applied: bool
    shared_p_ref_world: geom.Vec3
    sham_p_obj_world: geom.Vec3
    move_p_obj_world: geom.Vec3


def _shared_moving_planar_goal(
    ctx: ScoringContext,
    p_ref_world: geom.Vec3,
) -> geom.GoalSetResult:
    if ctx.planar_spec is None:
        raise ValueError("planar_spec required for response-horizon scoring")
    return geom.build_response_planar_goal_set(ctx.frame, ctx.planar_spec, p_ref_world)


def score_response_horizon_pair(
    ctx: ScoringContext,
    request: ResponseHorizonRequest,
) -> ResponseHorizonScore:
    query_time_s = request.t_event_planned_s + request.horizon_s
    sham_resolved = geom.resolve_trajectory_sample(
        request.sham_trajectory,
        query_time_s,
        terminal_extension=request.sham_terminal_extension,
    )
    move_resolved = geom.resolve_trajectory_sample(
        request.move_trajectory,
        query_time_s,
        terminal_extension=request.move_terminal_extension,
    )
    shared_ref = move_resolved.sample.p_named_ref_world
    goal = _shared_moving_planar_goal(ctx, shared_ref)
    response_goal = geom.GoalSetResult(
        region=goal.region,
        empty=goal.empty,
        empty_cause=goal.empty_cause,
        projection_kind="response_planar",
        component_id=goal.component_id,
    )
    sham_distance = geom.response_projection_distance(
        ctx.frame,
        sham_resolved.sample.p_obj_world,
        response_goal,
        d_cap_m=ctx.d_cap_m,
    )
    move_distance = geom.response_projection_distance(
        ctx.frame,
        move_resolved.sample.p_obj_world,
        response_goal,
        d_cap_m=ctx.d_cap_m,
    )
    return ResponseHorizonScore(
        query_time_s=query_time_s,
        d_cap_sham_m=sham_distance.capped_distance_m,
        d_cap_move_m=move_distance.capped_distance_m,
        h_response_m=sham_distance.capped_distance_m - move_distance.capped_distance_m,
        goal_set_empty=sham_distance.goal_set_empty or move_distance.goal_set_empty,
        sham_terminal_extension_applied=sham_resolved.terminal_extension_applied,
        move_terminal_extension_applied=move_resolved.terminal_extension_applied,
        shared_p_ref_world=shared_ref,
        sham_p_obj_world=sham_resolved.sample.p_obj_world,
        move_p_obj_world=move_resolved.sample.p_obj_world,
    )
