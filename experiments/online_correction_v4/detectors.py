"""Natural grasp and first-detachment detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.online_correction_v4.contracts import DetachmentEvent, NaturalGraspEvent


@dataclass(frozen=True)
class ObjectKinematicState:
    sim_time: float
    control_tick: int
    object_z: float
    initial_supported_z: float
    gripper_x: float
    gripper_y: float
    gripper_z: float
    object_x: float
    object_y: float
    object_z_pos: float
    contact: bool = False
    detached: bool = False


@dataclass(frozen=True)
class GraspDetectorConfig:
    min_lift_m: float = 0.04
    dwell_s: float = 0.20
    relative_drift_max_m: float = 0.01
    trigger_deadline_s: float = 40.0


@dataclass(frozen=True)
class DetachmentDetectorConfig:
    dwell_ticks: int = 2


@dataclass
class CarrySample:
    start_time: float
    start_tick: int
    reference_x: float
    reference_y: float
    reference_z: float


@dataclass
class NaturalGraspDetector:
    config: GraspDetectorConfig
    control_dt_s: float
    _carry_start: Optional[CarrySample] = None
    _eligible_event: Optional[NaturalGraspEvent] = None
    _failed_deadline: bool = False
    _grasp_occurred: bool = False

    @property
    def eligible(self) -> bool:
        return self._eligible_event is not None

    @property
    def grasp_occurred(self) -> bool:
        return self._grasp_occurred

    @property
    def carry_verified(self) -> bool:
        return self._eligible_event is not None

    @property
    def event(self) -> Optional[NaturalGraspEvent]:
        return self._eligible_event

    def update(self, state: ObjectKinematicState) -> Optional[NaturalGraspEvent]:
        if self._eligible_event is not None:
            return self._eligible_event
        lift = state.object_z_pos - state.initial_supported_z
        rel_drift = (
            (state.object_x - state.gripper_x) ** 2
            + (state.object_y - state.gripper_y) ** 2
            + (state.object_z_pos - state.gripper_z) ** 2
        ) ** 0.5
        carrying = (state.contact or lift >= self.config.min_lift_m * 0.5) and lift >= self.config.min_lift_m
        stable = rel_drift <= self.config.relative_drift_max_m
        if carrying:
            self._grasp_occurred = True
        if carrying and stable:
            if self._carry_start is None:
                self._carry_start = CarrySample(
                    start_time=state.sim_time,
                    start_tick=state.control_tick,
                    reference_x=state.gripper_x,
                    reference_y=state.gripper_y,
                    reference_z=state.gripper_z,
                )
            dwell = state.sim_time - self._carry_start.start_time
            if dwell + 1e-12 >= self.config.dwell_s:
                self._eligible_event = NaturalGraspEvent(
                    t_eligible=state.sim_time,
                    control_tick=state.control_tick,
                    lift_m=lift,
                    dwell_s=dwell,
                    relative_drift_m=rel_drift,
                )
                return self._eligible_event
        else:
            self._carry_start = None
        if state.sim_time + 1e-12 >= self.config.trigger_deadline_s:
            self._failed_deadline = True
        return None

    def reset(self) -> None:
        self._carry_start = None
        self._eligible_event = None
        self._failed_deadline = False
        self._grasp_occurred = False


@dataclass
class DetachmentDetector:
    config: DetachmentDetectorConfig
    carry_verified: bool = False
    _onset_time: Optional[float] = None
    _onset_tick: Optional[int] = None
    _consecutive_ticks: int = 0
    _event: Optional[DetachmentEvent] = None

    @property
    def armed(self) -> bool:
        return self.carry_verified

    def arm_after_verified_carry(self) -> None:
        self.carry_verified = True

    @property
    def detected(self) -> bool:
        return self._event is not None

    @property
    def event(self) -> Optional[DetachmentEvent]:
        return self._event

    def update(self, state: ObjectKinematicState) -> Optional[DetachmentEvent]:
        if self._event is not None:
            return self._event
        if not self.carry_verified:
            return None
        if state.detached:
            if self._onset_time is None:
                self._onset_time = state.sim_time
                self._onset_tick = state.control_tick
            self._consecutive_ticks += 1
            if self._consecutive_ticks >= self.config.dwell_ticks:
                self._event = DetachmentEvent(
                    t_onset=self._onset_time,
                    t_detected=state.sim_time,
                    onset_tick=self._onset_tick if self._onset_tick is not None else state.control_tick,
                    detected_tick=state.control_tick,
                )
                return self._event
        else:
            self._onset_time = None
            self._onset_tick = None
            self._consecutive_ticks = 0
        return None

    def reset(self) -> None:
        self.carry_verified = False
        self._onset_time = None
        self._onset_tick = None
        self._consecutive_ticks = 0
        self._event = None
