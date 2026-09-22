"""Pure physical scoring for SGW-01.

The scorer consumes simulator observations only after an episode has finished.
It never decides when the controller stops and never substitutes an inferred
endpoint for a missing or safety-censored observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence


class OutcomeStatus(str, Enum):
    VALID_MODEL = "valid_model"
    SAFETY_CENSORED = "safety_censored"
    INFRA_INVALID = "infrastructure_invalid"


@dataclass(frozen=True)
class FrozenScoringConfig:
    relation_margin_m: float = 0.03
    initial_neutral_tolerance_m: float = 0.005
    reference_motion_limit_m: float = 0.005
    pickup_height_m: float = 0.03
    pickup_consecutive_steps: int = 3
    final_stability_seconds: float = 0.5
    linear_speed_limit_m_s: float = 0.02
    angular_speed_limit_rad_s: float = 0.2
    action_cap: int = 450


@dataclass(frozen=True)
class GoalSpec:
    family: str
    physical_goal_sign: int
    form: str | None = None

    def __post_init__(self) -> None:
        if self.family not in {"LAT", "HEIGHT", "DIST"}:
            raise ValueError(f"unknown family: {self.family}")
        if self.physical_goal_sign not in {-1, 1}:
            raise ValueError("physical_goal_sign must be -1 or 1")


@dataclass(frozen=True)
class RelationMeasurement:
    relation_m: float
    requested_margin_m: float
    target_reference: str


@dataclass(frozen=True)
class EpisodeScore:
    status: OutcomeStatus
    requested_success: bool | None
    terminal_margin_m: float | None
    relation_m: float | None
    pickup_step: int | None
    first_success_step: int | None
    terminal_step: int | None
    anchor_max_drift_m: float | None
    reference_motion_ok: bool | None
    release_detached: bool | None
    failure_stage: str | None
    safety_censored: bool
    infrastructure_reason: str | None = None


def canonical_status(score: EpisodeScore) -> str:
    """Map scorer outcomes to the recorder's durable result vocabulary."""
    if score.status is OutcomeStatus.INFRA_INVALID:
        return "technical_invalid"
    if score.status is OutcomeStatus.SAFETY_CENSORED:
        return "censored"
    return "valid_success" if score.requested_success else "valid_model_failure"


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        return (float(value["x"]), float(value["y"]), float(value["z"]))
    if len(value) != 3:
        raise ValueError("position must contain x, y, z")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def relation_m(family: str, cube: Any, bowl: Any, plate: Any | None = None) -> float:
    """Return the protocol's signed physical relation in metres."""
    c, b = _xyz(cube), _xyz(bowl)
    if family == "LAT":
        return c[1] - b[1]
    if family == "HEIGHT":
        return c[2] - b[2]
    if family == "DIST":
        if plate is None:
            raise ValueError("DIST requires a plate position")
        p = _xyz(plate)
        return math.dist(c, p) - math.dist(c, b)
    raise ValueError(f"unknown family: {family}")


def _max_anchor_drift(states: Sequence[Mapping[str, Any]]) -> float:
    if not states:
        return 0.0
    first = states[0]
    max_drift = 0.0
    for state in states:
        for name in ("bowl", "plate"):
            first_value = first.get(name, first.get(f"{name}_xyz_m"))
            state_value = state.get(name, state.get(f"{name}_xyz_m"))
            if first_value is not None and state_value is not None:
                max_drift = max(max_drift, math.dist(_xyz(state_value), _xyz(first_value)))
    return max_drift


def _pickup_step(states: Sequence[Mapping[str, Any]], cfg: FrozenScoringConfig) -> int | None:
    run = 0
    for index, state in enumerate(states):
        held = bool(state.get("gripper_holding", state.get("held", False)))
        height = float(state.get("cube_height_lift_m", state.get("cube_lift_m", 0.0)))
        if held and height >= cfg.pickup_height_m:
            run += 1
            if run >= cfg.pickup_consecutive_steps:
                return index - cfg.pickup_consecutive_steps + 1
        else:
            run = 0
    return None


def _first_success(events: Iterable[Mapping[str, Any]]) -> int | None:
    steps = [int(event["step"]) for event in events if bool(event.get("success", False))]
    return min(steps) if steps else None


def score_episode(
    episode: Mapping[str, Any],
    goal: GoalSpec,
    config: FrozenScoringConfig | None = None,
) -> EpisodeScore:
    """Score a completed trace, retaining censored and infrastructure outcomes."""
    cfg = config or FrozenScoringConfig()
    if episode.get("status") == OutcomeStatus.INFRA_INVALID.value or episode.get("infrastructure_reason"):
        return EpisodeScore(
            OutcomeStatus.INFRA_INVALID, None, None, None, None, None, None,
            None, None, None, None, False, str(episode.get("infrastructure_reason", "invalid")),
        )
    states = list(episode.get("states", ()))
    if not states:
        raise ValueError("valid scoring requires at least one state")
    terminal_observed = bool(episode.get("terminal_observed", len(states) >= cfg.action_cap))
    safety = bool(episode.get("termination_reason") == "safety" or episode.get("safety_terminated"))
    anchor_drift = _max_anchor_drift(states)
    reference_ok = anchor_drift <= cfg.reference_motion_limit_m
    pickup = _pickup_step(states, cfg)
    first_success = _first_success(episode.get("success_events", ()))
    if safety or not terminal_observed:
        return EpisodeScore(
            OutcomeStatus.SAFETY_CENSORED, False, None, None, pickup, first_success,
            None, anchor_drift, reference_ok, None, "safety_censored", True,
        )
    final = states[-1]
    required = ("cube", "bowl")
    if any(key not in final and f"{key}_xyz_m" not in final for key in required):
        raise ValueError("terminal state lacks cube or bowl position")
    cube = final.get("cube", final.get("cube_xyz_m"))
    bowl = final.get("bowl", final.get("bowl_xyz_m"))
    plate = final.get("plate", final.get("plate_xyz_m"))
    rel = relation_m(goal.family, cube, bowl, plate)
    margin = goal.physical_goal_sign * rel
    detached = bool(final.get("final_detached_release", final.get("release_detached", False)))
    stable = bool(final.get("stable_for_seconds", False) and final.get("supported", True))
    success = bool(pickup is not None and margin >= cfg.relation_margin_m and reference_ok and detached and stable)
    if pickup is None:
        stage = "pick_failed"
    elif not reference_ok:
        stage = "anchor_disturbance"
    elif margin < cfg.relation_margin_m:
        stage = "wrong_side"
    elif not detached:
        stage = "release_failed"
    elif not stable:
        stage = "transport_failed"
    else:
        stage = None
    return EpisodeScore(
        OutcomeStatus.VALID_MODEL, success, margin, rel, pickup, first_success,
        len(states) - 1, anchor_drift, reference_ok, detached, stage, False,
    )
