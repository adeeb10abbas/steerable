"""Controlled simulation clock and action-queue semantics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Deque, Iterable, Optional

from experiments.online_correction_v4.contracts import PolicyTimingAchieved, TimingConfig


def quantize_upward(requested_s: float, native_dt_s: float) -> float:
    """Round requested duration upward to an integer number of native control ticks."""
    if native_dt_s <= 0:
        raise ValueError("native_dt_s must be positive")
    if requested_s < 0:
        raise ValueError("requested_s must be nonnegative")
    ticks = math.ceil(requested_s / native_dt_s - 1e-10)
    return ticks * native_dt_s


class EmptyBufferRule(str, Enum):
    HOLD_LAST_VALID = "hold_last_valid"


class QueueReplacementRule(str, Enum):
    REPLACE_REMAINING = "replace_remaining"


class QuerySchedule(str, Enum):
    STANDARD = "standard"
    FAST_AFTER_GRASP = "fast_after_grasp"


@dataclass(frozen=True)
class ActionCommand:
    action_index: int
    chunk_id: str
    request_id: str
    values: tuple[float, ...]


@dataclass
class PolicyRequest:
    request_id: str
    observation_id: str
    observation_capture_time: float
    submit_time: float
    response_available_time: Optional[float] = None
    inference_wall_duration_s: Optional[float] = None
    response_chunk_id: Optional[str] = None
    staged_actions: Optional[list[tuple[float, ...]]] = None
    applied: bool = False
    discarded: bool = False


@dataclass
class ActionQueue:
    native_control_dt_s: float
    empty_buffer_rule: EmptyBufferRule = EmptyBufferRule.HOLD_LAST_VALID
    replacement_rule: QueueReplacementRule = QueueReplacementRule.REPLACE_REMAINING
    initial_command: Optional[ActionCommand] = None
    _pending: Deque[ActionCommand] = field(default_factory=deque, init=False, repr=False)
    _last_valid: Optional[ActionCommand] = field(default=None, init=False, repr=False)
    _executed: list[ActionCommand] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_command is not None:
            self._last_valid = self.initial_command

    def enqueue_chunk(
        self,
        *,
        chunk_id: str,
        request_id: str,
        actions: Iterable[tuple[float, ...]],
        start_index: int = 0,
    ) -> None:
        commands = [
            ActionCommand(
                action_index=start_index + offset,
                chunk_id=chunk_id,
                request_id=request_id,
                values=tuple(action),
            )
            for offset, action in enumerate(actions)
        ]
        if self.replacement_rule is QueueReplacementRule.REPLACE_REMAINING:
            self._pending = deque(commands)
        else:
            self._pending.extend(commands)

    def pop_for_tick(self) -> Optional[ActionCommand]:
        if self._pending:
            command = self._pending.popleft()
            self._last_valid = command
            self._executed.append(command)
            return command
        if self.empty_buffer_rule is EmptyBufferRule.HOLD_LAST_VALID:
            return self._last_valid
        return None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def executed(self) -> list[ActionCommand]:
        return list(self._executed)


@dataclass
class ControlledSimulationClock:
    """Deterministic controlled-time scheduler for observation/query/action semantics."""

    timing: TimingConfig
    achieved: PolicyTimingAchieved
    schedule: QuerySchedule = QuerySchedule.STANDARD
    fast_schedule_active: bool = False
    sim_time: float = 0.0
    control_tick: int = 0
    sim_paused: bool = False
    policy_phase_active: bool = True
    passive_settling_active: bool = False
    action_queue: ActionQueue = field(default_factory=lambda: ActionQueue(native_control_dt_s=0.05))
    pending_requests: list[PolicyRequest] = field(default_factory=list)
    completed_requests: list[PolicyRequest] = field(default_factory=list)
    next_query_time: Optional[float] = None
    event_phase_fraction: float = 0.0
    natural_grasp_time: Optional[float] = None
    event_onset_time: Optional[float] = None
    active_cap_time: Optional[float] = None
    settling_end_time: Optional[float] = None
    episode_end_time: Optional[float] = None
    _request_counter: int = field(default=0, init=False, repr=False)
    _observation_counter: int = field(default=0, init=False, repr=False)
    _on_tick: Optional[Callable[[float, int, Optional[ActionCommand]], None]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.action_queue.native_control_dt_s != self.achieved.native_control_dt_s:
            self.action_queue.native_control_dt_s = self.achieved.native_control_dt_s
        self.next_query_time = self.sim_time

    @property
    def query_period(self) -> float:
        if self.schedule is QuerySchedule.FAST_AFTER_GRASP and self.fast_schedule_active:
            return self.achieved.achieved_fast_query_period_s
        return self.achieved.achieved_standard_query_period_s

    def register_tick_callback(
        self, callback: Callable[[float, int, Optional[ActionCommand]], None]
    ) -> None:
        self._on_tick = callback

    def reset(self) -> None:
        self.sim_time = 0.0
        self.control_tick = 0
        self.sim_paused = False
        self.policy_phase_active = True
        self.passive_settling_active = False
        self.pending_requests.clear()
        self.completed_requests.clear()
        self.next_query_time = 0.0
        self.natural_grasp_time = None
        self.event_onset_time = None
        self.active_cap_time = None
        self.settling_end_time = None
        self.episode_end_time = None
        self.fast_schedule_active = False
        self._request_counter = 0
        self._observation_counter = 0
        self.action_queue = ActionQueue(native_control_dt_s=self.achieved.native_control_dt_s)

    def _new_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter:05d}"

    def _new_observation_id(self) -> str:
        self._observation_counter += 1
        return f"obs-{self._observation_counter:05d}"

    def plan_event_onset_after_grasp(self) -> float:
        if self.natural_grasp_time is None:
            raise RuntimeError("natural grasp must be registered before event onset")
        boundary = self._next_standard_query_boundary_at_or_after(self.natural_grasp_time)
        phase_offset = self.event_phase_fraction * self.achieved.achieved_standard_query_period_s
        onset = quantize_upward(boundary + phase_offset, self.achieved.native_control_dt_s)
        self.event_onset_time = onset
        self.active_cap_time = min(
            self.timing.episode_cap_s,
            onset + self.timing.post_event_cap_s,
        )
        return onset

    def _next_standard_query_boundary_at_or_after(self, time_s: float) -> float:
        period = self.achieved.achieved_standard_query_period_s
        if time_s <= 0:
            return 0.0
        ticks = math.ceil(time_s / period - 1e-10)
        return ticks * period

    def register_natural_grasp(self, grasp_time: float) -> None:
        self.natural_grasp_time = grasp_time
        if self.schedule is QuerySchedule.FAST_AFTER_GRASP:
            self.fast_schedule_active = True

    def start_passive_settling(self, detection_time: float) -> None:
        if self.passive_settling_active:
            return
        self.policy_phase_active = False
        self.passive_settling_active = True
        self.settling_end_time = detection_time + self.timing.release_settling_s
        if self.active_cap_time is None:
            self.active_cap_time = min(self.timing.episode_cap_s, detection_time)

    def stop_policy_phase(self, stop_time: float, *, timeout: bool = False) -> None:
        self.policy_phase_active = False
        if timeout:
            self.start_passive_settling(stop_time)

    def due_for_query(self) -> bool:
        return (
            self.policy_phase_active
            and not self.sim_paused
            and self.next_query_time is not None
            and math.isclose(self.sim_time, self.next_query_time, abs_tol=1e-12)
        )

    def capture_observation(self, state_hash: str) -> tuple[str, float]:
        observation_id = self._new_observation_id()
        return observation_id, self.sim_time

    def submit_policy_request(self, observation_id: str) -> PolicyRequest:
        request = PolicyRequest(
            request_id=self._new_request_id(),
            observation_id=observation_id,
            observation_capture_time=self.sim_time,
            submit_time=self.sim_time,
        )
        self.pending_requests.append(request)
        self.sim_paused = True
        return request

    def complete_inference(
        self,
        request: PolicyRequest,
        *,
        chunk_id: str,
        actions: Iterable[tuple[float, ...]],
        wall_duration_s: float,
    ) -> None:
        if request not in self.pending_requests:
            raise ValueError("unknown pending request")
        request.inference_wall_duration_s = wall_duration_s
        request.response_chunk_id = chunk_id
        request.staged_actions = [tuple(action) for action in actions]
        request.response_available_time = request.observation_capture_time + self.achieved.achieved_delay_s
        self.pending_requests.remove(request)
        self.completed_requests.append(request)
        self.sim_paused = False

    def advance_for_delay_window(self, request: PolicyRequest) -> list[ActionCommand]:
        """Simulate [t, t+delay) using the pre-response action queue."""
        target = request.observation_capture_time + self.achieved.achieved_delay_s
        executed: list[ActionCommand] = []
        while self.sim_time + 1e-12 < target:
            executed.append(self._advance_one_control_tick(apply_new_chunks=False))
        return [cmd for cmd in executed if cmd is not None]

    def apply_due_responses(self) -> list[PolicyRequest]:
        applied: list[PolicyRequest] = []
        for request in self.completed_requests:
            if request.applied or request.discarded:
                continue
            if request.response_available_time is None or request.staged_actions is None:
                continue
            if self.sim_time + 1e-12 >= request.response_available_time:
                self.action_queue.enqueue_chunk(
                    chunk_id=request.response_chunk_id or request.request_id,
                    request_id=request.request_id,
                    actions=request.staged_actions,
                    start_index=len(self.action_queue.executed),
                )
                request.applied = True
                applied.append(request)
        return applied

    def _advance_one_control_tick(self, *, apply_new_chunks: bool = True) -> Optional[ActionCommand]:
        if apply_new_chunks:
            self.apply_due_responses()
        command = None
        if self.policy_phase_active or self.passive_settling_active:
            command = self.action_queue.pop_for_tick()
        self.control_tick += 1
        self.sim_time = round(self.control_tick * self.achieved.native_control_dt_s, 12)
        if self._on_tick is not None:
            self._on_tick(self.sim_time, self.control_tick, command)
        if self.passive_settling_active and self.settling_end_time is not None:
            if self.sim_time + 1e-12 >= self.settling_end_time:
                self.episode_end_time = self.sim_time
                self.passive_settling_active = False
        elif (
            self.policy_phase_active
            and self.sim_time + 1e-12 >= self.timing.episode_cap_s
            and self.event_onset_time is None
        ):
            self.stop_policy_phase(self.sim_time, timeout=True)
        elif self.active_cap_time is not None and self.policy_phase_active:
            if self.sim_time + 1e-12 >= self.active_cap_time:
                self.stop_policy_phase(self.sim_time, timeout=True)
        return command

    def advance_to_next_query(self) -> list[dict[str, Any]]:
        """Advance one query cycle including inference delay semantics."""
        events: list[dict[str, Any]] = []
        if not self.policy_phase_active:
            while self.passive_settling_active:
                self._advance_one_control_tick()
            return events
        if self.next_query_time is None:
            raise RuntimeError("next_query_time is unset")
        while self.sim_time + 1e-12 < self.next_query_time:
            cmd = self._advance_one_control_tick()
            events.append({"kind": "control_tick", "sim_time": self.sim_time, "command": cmd})
        obs_id, obs_time = self.capture_observation("state")
        request = self.submit_policy_request(obs_id)
        events.append(
            {
                "kind": "query",
                "sim_time": obs_time,
                "observation_id": obs_id,
                "request_id": request.request_id,
            }
        )
        return events

    def finish_query_cycle(
        self,
        request: PolicyRequest,
        *,
        chunk_id: str,
        actions: Iterable[tuple[float, ...]],
        wall_duration_s: float,
        advance_reference: Optional[Callable[[float], None]] = None,
    ) -> dict[str, Any]:
        """Complete inference and simulate the emulated delay without double-counting."""
        self.complete_inference(
            request,
            chunk_id=chunk_id,
            actions=actions,
            wall_duration_s=wall_duration_s,
        )
        delay_actions = self.advance_for_delay_window(request)
        if advance_reference is not None:
            start = request.observation_capture_time
            end = request.response_available_time or start
            step = self.achieved.native_control_dt_s
            t = start
            while t + 1e-12 < end:
                t = min(t + step, end)
                advance_reference(t)
        applied = self.apply_due_responses()
        self.next_query_time = request.observation_capture_time + self.query_period
        return {
            "delay_actions": delay_actions,
            "applied_requests": applied,
            "response_available_time": request.response_available_time,
            "next_query_time": self.next_query_time,
        }

    def run_deterministic_schedule(
        self,
        *,
        num_queries: int,
        chunk_factory: Callable[[PolicyRequest], tuple[str, list[tuple[float, ...]]]],
        wall_duration_factory: Callable[[PolicyRequest], float],
        advance_reference: Optional[Callable[[float], None]] = None,
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for _ in range(num_queries):
            if not self.policy_phase_active:
                break
            events = self.advance_to_next_query()
            trace.extend(events)
            request = self.pending_requests[-1]
            chunk_id, actions = chunk_factory(request)
            wall = wall_duration_factory(request)
            result = self.finish_query_cycle(
                request,
                chunk_id=chunk_id,
                actions=actions,
                wall_duration_s=wall,
                advance_reference=advance_reference,
            )
            trace.append({"kind": "response", "request_id": request.request_id, **result})
        return trace
