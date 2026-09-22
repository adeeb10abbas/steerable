"""Pure physical scoring for SGW-01.

The scorer consumes simulator observations only after an episode has finished.
It never decides when the controller stops and never substitutes an inferred
endpoint for a missing or safety-censored observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
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


def result_payload(
    score: EpisodeScore,
    *,
    release_id: str,
    cell_id: str,
    attempt_id: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    """Build the recorder's JSON-safe result payload without scientific defaults."""
    payload = asdict(score)
    payload["status"] = canonical_status(score)
    payload["completed_at_utc"] = completed_at_utc
    payload["release_id"] = release_id
    payload["cell_id"] = cell_id
    payload["attempt_id"] = attempt_id
    payload["schema_version"] = "sgw-01-result-v1"
    payload["status"] = str(payload["status"])
    payload["outcome_status"] = str(score.status.value)
    if payload["status"] == "technical_invalid" and not str(payload.get("infrastructure_reason", "")).strip():
        raise ValueError("technical_invalid result requires infrastructure_reason")
    return payload


def _xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        result = (float(value["x"]), float(value["y"]), float(value["z"]))
    else:
        if len(value) != 3:
            raise ValueError("position must contain x, y, z")
        result = tuple(float(v) for v in value)  # type: ignore[assignment]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("position must contain finite coordinates")
    return result  # type: ignore[return-value]


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


def _max_anchor_drift(states: Sequence[Mapping[str, Any]], required: Sequence[str]) -> float:
    if not states:
        return 0.0
    first = states[0]
    max_drift = 0.0
    for state in states:
        for name in ("bowl", "plate"):
            first_value = first.get(name, first.get(f"{name}_xyz_m"))
            state_value = state.get(name, state.get(f"{name}_xyz_m"))
            if name in required and (first_value is None or state_value is None):
                raise ValueError(f"missing {name} position in timestep")
            if first_value is not None and state_value is not None:
                max_drift = max(max_drift, math.dist(_xyz(state_value), _xyz(first_value)))
    return max_drift


def _pickup_step(
    states: Sequence[Mapping[str, Any]],
    cfg: FrozenScoringConfig,
    *,
    initial_cube_z: float | None = None,
) -> int | None:
    if initial_cube_z is None and states:
        initial = states[0].get("cube", states[0].get("cube_xyz_m"))
        if initial is not None:
            initial_cube_z = _xyz(initial)[2]
    run = 0
    for index, state in enumerate(states[1:], start=1):
        position = state.get("cube", state.get("cube_xyz_m"))
        if position is None or initial_cube_z is None:
            continue
        height = _xyz(position)[2] - initial_cube_z
        reported_height = state.get("cube_height_lift_m", state.get("cube_lift_m"))
        if reported_height is not None and not math.isclose(float(reported_height), height, abs_tol=1e-4):
            raise ValueError("reported pickup height disagrees with cube geometry")
        if height >= cfg.pickup_height_m:
            run += 1
            if run >= cfg.pickup_consecutive_steps:
                return index - cfg.pickup_consecutive_steps + 1
        else:
            run = 0
    return None


def _first_success(events: Iterable[Mapping[str, Any]]) -> int | None:
    steps = [int(event["step"]) for event in events if bool(event.get("success", False))]
    return min(steps) if steps else None


def _validate_trace(states: Sequence[Mapping[str, Any]], cfg: FrozenScoringConfig) -> None:
    if len(states) != cfg.action_cap + 1:
        raise ValueError("trace must contain reset state plus action steps 1..450")
    indexed = [state.get("action_step") for state in states]
    if any(step is not None for step in indexed):
        if indexed != list(range(cfg.action_cap + 1)):
            raise ValueError("action steps must be contiguous from reset step 0 through action 450")
    times = [state.get("sim_time_s") for state in states]
    if any(time is None for time in times):
        raise ValueError("sim_time_s must be recorded for every timestep")
    numeric = [float(time) for time in times]
    if any(not math.isfinite(time) for time in numeric) or any(
        b <= a for a, b in zip(numeric, numeric[1:])
    ):
        raise ValueError("simulated times must be finite and strictly increasing")


def _stable_release(
    states: Sequence[Mapping[str, Any]],
    goal: GoalSpec,
    cfg: FrozenScoringConfig,
) -> bool:
    if not states or not bool(states[-1].get("final_detached_release", states[-1].get("release_detached", False))):
        return False
    if not all("sim_time_s" in state for state in states):
        return False
    times = [float(state["sim_time_s"]) for state in states]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        return False
    end = float(states[-1]["sim_time_s"])
    boundary = end - cfg.final_stability_seconds
    preceding = [index for index, time in enumerate(times) if time <= boundary + 1e-9]
    if not preceding:
        return False
    # Include the sample at or immediately before the window boundary. A
    # control period need not divide half a second exactly.
    window = states[preceding[-1]:]
    if end - float(window[0]["sim_time_s"]) < cfg.final_stability_seconds - 1e-9:
        return False
    return all(
        bool(state.get("supported", False))
        and float(state.get("linear_speed_m_s", math.inf)) < cfg.linear_speed_limit_m_s
        and float(state.get("angular_speed_rad_s", math.inf)) < cfg.angular_speed_limit_rad_s
        and bool(state.get("final_detached_release", state.get("release_detached", False)))
        and goal.physical_goal_sign * relation_m(
            goal.family,
            state.get("cube", state.get("cube_xyz_m")),
            state.get("bowl", state.get("bowl_xyz_m")),
            state.get("plate", state.get("plate_xyz_m")),
        ) >= cfg.relation_margin_m
        for state in window
    )


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
    safety = bool(episode.get("termination_reason") == "safety" or episode.get("safety_terminated"))
    if not safety:
        try:
            _validate_trace(states, cfg)
        except ValueError as error:
            return EpisodeScore(
                OutcomeStatus.INFRA_INVALID, None, None, None, None, None, None,
                None, None, None, "invalid_trace", False, str(error),
            )
    terminal_observed = len(states) >= cfg.action_cap + 1 and bool(
        episode.get("terminal_observed", False)
        or episode.get("termination_reason") == "action_cap"
        or len(states) == cfg.action_cap
    )
    required_anchors = ("bowl", "plate") if goal.family == "DIST" else ("bowl",)
    anchor_drift = _max_anchor_drift(states, required_anchors)
    reference_ok = anchor_drift <= cfg.reference_motion_limit_m
    initial_cube = episode.get("initial_cube_xyz_m", episode.get("initial_cube"))
    initial_cube_z = _xyz(initial_cube)[2] if initial_cube is not None else None
    pickup = _pickup_step(states, cfg, initial_cube_z=initial_cube_z)
    first_success = _first_success(episode.get("success_events", ()))
    if safety:
        return EpisodeScore(
            OutcomeStatus.SAFETY_CENSORED, False, None, None, pickup, first_success,
            None, anchor_drift, reference_ok, None, "safety_censored", True,
        )
    if not terminal_observed:
        return EpisodeScore(
            OutcomeStatus.INFRA_INVALID, None, None, None, pickup, first_success,
            None, anchor_drift, reference_ok, None, "missing_action_450_endpoint", False,
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
    stable = _stable_release(states, goal, cfg)
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
        cfg.action_cap, anchor_drift, reference_ok, detached, stage, False,
    )
