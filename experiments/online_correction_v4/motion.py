"""Minimum-jerk reference motion profiles for V4 interventions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Optional, Sequence


def minimum_jerk_scalar(u: float) -> float:
    """Scalar minimum-jerk interpolation S(u)=10u^3-15u^4+6u^5 on [0,1]."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return ((6.0 * u - 15.0) * u + 10.0) * u * u * u


def peak_speed(delta_m: float, duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    return 1.875 * abs(delta_m) / duration_s


class MotionDirectionError(ValueError):
    """Raised when a live motion vector cannot be derived for a fixture/goal pair."""


class MotionProfileKind(str, Enum):
    ORIGINAL_SHAM = "original_sham"
    MOVE_STOP = "move_stop"
    MOVE_A = "move_A"
    DESTINATION_STATIC = "destination_static"
    SLOW_DRIFT = "slow_drift"
    FAST_DRIFT = "fast_drift"
    REVERSAL = "reversal"


@dataclass(frozen=True)
class MotionSegment:
    start_time_s: float
    end_time_s: float
    start_fraction: float
    end_fraction: float


@dataclass(frozen=True)
class ReferenceMotionState:
    sim_time: float
    displacement_m: float
    fraction: float
    velocity_m_s: float
    truncated: bool
    truncation_reason: Optional[str]
    frozen: bool


@dataclass
class ReferenceMotionController:
    profile: MotionProfileKind
    displacement_m: float
    move_stop_duration_s: float = 0.5
    slow_drift_duration_s: float = 4.0
    fast_drift_duration_s: float = 1.0
    reversal_waypoints: Sequence[tuple[float, float]] = (
        (0.0, 0.0),
        (2.0, 1.0),
        (4.0, -0.5),
    )
    event_onset_s: Optional[float] = None
    destination_fraction: float = 1.0
    frozen_at: Optional[float] = None
    frozen_displacement_m: Optional[float] = None
    truncated_at: Optional[float] = None
    truncation_reason: Optional[str] = None

    @classmethod
    def from_scenario(
        cls,
        scenario: str,
        *,
        displacement_m: float,
        motion_config: dict,
    ) -> ReferenceMotionController:
        profile_map = {
            "original_sham": MotionProfileKind.ORIGINAL_SHAM,
            "move_stop": MotionProfileKind.MOVE_STOP,
            "move_A": MotionProfileKind.MOVE_A,
            "destination_static": MotionProfileKind.DESTINATION_STATIC,
            "slow_drift": MotionProfileKind.SLOW_DRIFT,
            "fast_drift": MotionProfileKind.FAST_DRIFT,
            "reversal": MotionProfileKind.REVERSAL,
        }
        if scenario not in profile_map:
            raise ValueError(f"unsupported scenario {scenario!r}")
        waypoints = tuple(
            (float(w["time_s"]), float(w["displacement_units"]))
            for w in motion_config["reversal_waypoints"]
        )
        return cls(
            profile=profile_map[scenario],
            displacement_m=displacement_m,
            move_stop_duration_s=float(motion_config["move_stop_duration_s"]),
            slow_drift_duration_s=float(motion_config["slow_drift_duration_s"]),
            fast_drift_duration_s=float(motion_config["fast_drift_duration_s"]),
            reversal_waypoints=waypoints,
        )

    def schedule_event(self, onset_s: float) -> None:
        self.event_onset_s = onset_s

    def freeze_at(self, sim_time: float, *, reason: str) -> None:
        if self.frozen_at is not None:
            return
        state = self.pose_at(sim_time)
        self.frozen_at = sim_time
        self.frozen_displacement_m = state.displacement_m
        self.truncated_at = sim_time
        self.truncation_reason = reason

    def _segments(self) -> list[MotionSegment]:
        if self.profile is MotionProfileKind.REVERSAL:
            points = list(self.reversal_waypoints)
            segments: list[MotionSegment] = []
            for (t0, f0), (t1, f1) in zip(points, points[1:]):
                segments.append(MotionSegment(t0, t1, f0, f1))
            return segments
        duration = {
            MotionProfileKind.MOVE_STOP: self.move_stop_duration_s,
            MotionProfileKind.MOVE_A: self.move_stop_duration_s,
            MotionProfileKind.SLOW_DRIFT: self.slow_drift_duration_s,
            MotionProfileKind.FAST_DRIFT: self.fast_drift_duration_s,
            MotionProfileKind.ORIGINAL_SHAM: self.move_stop_duration_s,
        }.get(self.profile, 0.0)
        if self.profile is MotionProfileKind.DESTINATION_STATIC:
            return []
        return [MotionSegment(0.0, duration, 0.0, 1.0)]

    def _profile_fraction(self, local_time: float) -> tuple[float, float]:
        if self.profile is MotionProfileKind.DESTINATION_STATIC:
            return self.destination_fraction, 0.0
        if self.profile is MotionProfileKind.ORIGINAL_SHAM:
            return 0.0, 0.0
        segments = self._segments()
        if not segments:
            return 0.0, 0.0
        if local_time <= segments[0].start_time_s:
            return segments[0].start_fraction, 0.0
        for segment in segments:
            if local_time <= segment.end_time_s + 1e-12:
                span = segment.end_time_s - segment.start_time_s
                if span <= 0:
                    return segment.end_fraction, 0.0
                u = (local_time - segment.start_time_s) / span
                delta_f = segment.end_fraction - segment.start_fraction
                s = minimum_jerk_scalar(u)
                ds_du = 30.0 * u * u - 60.0 * u * u * u + 30.0 * u * u * u * u
                fraction = segment.start_fraction + delta_f * s
                velocity = delta_f * ds_du / span
                return fraction, velocity
        last = segments[-1]
        return last.end_fraction, 0.0

    def pose_at(self, sim_time: float) -> ReferenceMotionState:
        if self.frozen_at is not None:
            disp = self.frozen_displacement_m or 0.0
            return ReferenceMotionState(
                sim_time=sim_time,
                displacement_m=disp,
                fraction=disp / self.displacement_m if self.displacement_m else 0.0,
                velocity_m_s=0.0,
                truncated=True,
                truncation_reason=self.truncation_reason,
                frozen=True,
            )
        if self.profile is MotionProfileKind.DESTINATION_STATIC:
            disp = self.destination_fraction * self.displacement_m
            return ReferenceMotionState(
                sim_time=sim_time,
                displacement_m=disp,
                fraction=self.destination_fraction,
                velocity_m_s=0.0,
                truncated=False,
                truncation_reason=None,
                frozen=False,
            )
        if self.event_onset_s is None:
            return ReferenceMotionState(
                sim_time=sim_time,
                displacement_m=0.0,
                fraction=0.0,
                velocity_m_s=0.0,
                truncated=False,
                truncation_reason=None,
                frozen=False,
            )
        local_time = max(0.0, sim_time - self.event_onset_s)
        fraction, velocity = self._profile_fraction(local_time)
        disp = fraction * self.displacement_m
        truncated = self.truncated_at is not None and sim_time + 1e-12 >= self.truncated_at
        return ReferenceMotionState(
            sim_time=sim_time,
            displacement_m=disp,
            fraction=fraction,
            velocity_m_s=velocity * self.displacement_m,
            truncated=truncated,
            truncation_reason=self.truncation_reason if truncated else None,
            frozen=False,
        )

    def sample_path(self, start: float, end: float, step_s: float) -> list[ReferenceMotionState]:
        if step_s <= 0:
            raise ValueError("step_s must be positive")
        times: list[float] = []
        t = start
        while t <= end + 1e-12:
            times.append(round(t, 12))
            t += step_s
        if times[-1] < end - 1e-12:
            times.append(end)
        return [self.pose_at(t) for t in times]

    @staticmethod
    def displacement_vector(
        *,
        goal: str,
        fixture: str,
        physical_sign: int,
        diagonal_signs: Optional[tuple[int, int]] = None,
    ) -> tuple[float, float]:
        """Return a unit direction in task-plane coordinates for registered fixtures."""
        if fixture == "reference_binding" and diagonal_signs is not None:
            left, front = diagonal_signs
            norm = math.sqrt(2.0)
            return (left / norm, front / norm)
        axis_map = {
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "front": (0.0, 1.0),
            "behind": (0.0, -1.0),
        }
        if goal not in axis_map:
            if fixture in ("vertical", "containment"):
                return (float(physical_sign), 0.0)
            raise ValueError(f"unsupported goal direction {goal!r}")
        dx, dy = axis_map[goal]
        return (dx * physical_sign, dy * physical_sign)

    @staticmethod
    def resolve_live_direction(
        *,
        fixture: str,
        goal: str,
        counterbalance: Mapping[str, Any],
        supported_fixtures: Iterable[str] = ("horizontal",),
    ) -> tuple[float, float]:
        """Derive the task-plane motion unit vector for a registered manifest row."""
        if fixture not in set(supported_fixtures):
            raise MotionDirectionError(
                f"fixture {fixture!r} has no released live motion binding; "
                "derive direction only for supported fixtures"
            )
        try:
            physical_sign = int(counterbalance["physical_translation_sign"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MotionDirectionError("counterbalance.physical_translation_sign is required") from exc
        if physical_sign not in (-1, 1):
            raise MotionDirectionError(
                f"counterbalance.physical_translation_sign must be +/-1, got {physical_sign!r}"
            )
        diagonal_signs: tuple[int, int] | None = None
        if fixture == "reference_binding":
            raw = counterbalance.get("physical_A_diagonal_signs")
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise MotionDirectionError(
                    "reference_binding requires counterbalance.physical_A_diagonal_signs"
                )
            diagonal_signs = (int(raw[0]), int(raw[1]))
        try:
            return ReferenceMotionController.displacement_vector(
                goal=goal,
                fixture=fixture,
                physical_sign=physical_sign,
                diagonal_signs=diagonal_signs,
            )
        except ValueError as exc:
            raise MotionDirectionError(str(exc)) from exc

    def world_offset(
        self,
        sim_time: float,
        *,
        direction: tuple[float, float],
    ) -> tuple[float, float]:
        state = self.pose_at(sim_time)
        return (direction[0] * state.displacement_m, direction[1] * state.displacement_m)
