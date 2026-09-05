"""Typed contracts for the V4 online-correction runtime core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class CampaignPhase(str, Enum):
    DESIGN = "DESIGN"
    IMPLEMENTING = "IMPLEMENTING"
    QUALIFYING = "QUALIFYING"
    PILOTING = "PILOTING"
    FROZEN = "FROZEN"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    COMPLETE = "COMPLETE"
    BLOCKED_ACCESS = "BLOCKED_ACCESS"
    BLOCKED_RUNTIME = "BLOCKED_RUNTIME"
    BLOCKED_SETUP = "BLOCKED_SETUP"


class AttemptStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VALID = "valid"
    INFRA_INVALID = "infra_invalid"
    BLOCKED = "blocked"


class FailureStage(str, Enum):
    NONE = "none"
    PICKUP = "pickup"
    TRANSPORT = "transport"
    WRONG_RELATION = "wrong_relation"
    RELEASE = "release"
    TIMEOUT = "timeout"
    COLLISION = "collision"
    OTHER = "other"


class FailureLabel(str, Enum):
    SUCCESS = "success"
    NO_GRASP = "no_grasp"
    GRASP_LOST = "grasp_lost"
    TRANSPORT_INCOMPLETE = "transport_incomplete"
    WRONG_GOAL_REGION = "wrong_goal_region"
    RELEASE_FAILED = "release_failed"
    SUPPORT_OR_CONTAINMENT_FAILED = "support_or_containment_failed"
    TIMEOUT_WITHOUT_COMPLETION = "timeout_without_completion"
    COLLISION_CAUSED_TERMINAL_FAILURE = "collision_caused_terminal_failure"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    UNRESOLVED_BEHAVIORAL_FAILURE = "unresolved_behavioral_failure"


DETAILED_TO_COARSE: dict[FailureLabel, FailureStage] = {
    FailureLabel.SUCCESS: FailureStage.NONE,
    FailureLabel.NO_GRASP: FailureStage.PICKUP,
    FailureLabel.GRASP_LOST: FailureStage.TRANSPORT,
    FailureLabel.TRANSPORT_INCOMPLETE: FailureStage.TRANSPORT,
    FailureLabel.WRONG_GOAL_REGION: FailureStage.WRONG_RELATION,
    FailureLabel.RELEASE_FAILED: FailureStage.RELEASE,
    FailureLabel.SUPPORT_OR_CONTAINMENT_FAILED: FailureStage.RELEASE,
    FailureLabel.TIMEOUT_WITHOUT_COMPLETION: FailureStage.TIMEOUT,
    FailureLabel.COLLISION_CAUSED_TERMINAL_FAILURE: FailureStage.COLLISION,
    FailureLabel.MODEL_OUTPUT_INVALID: FailureStage.OTHER,
    FailureLabel.UNRESOLVED_BEHAVIORAL_FAILURE: FailureStage.OTHER,
}


def failure_stage_for_label(label: FailureLabel) -> FailureStage:
    return DETAILED_TO_COARSE[label]


class GroupExecutionState(str, Enum):
    UNLEASED = "unleased"
    LEASED = "leased"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class TimingConfig:
    emulated_observation_action_delay_s: float = 0.10
    standard_query_period_s: float = 0.50
    fast_query_period_s: float = 0.25
    episode_cap_s: float = 60.0
    trigger_deadline_s: float = 40.0
    post_event_cap_s: float = 20.0
    release_detection_dwell_ticks: int = 2
    release_settling_s: float = 1.0
    natural_grasp_min_lift_m: float = 0.04
    natural_grasp_dwell_s: float = 0.20
    kinematic_grasp_relative_drift_max_m: float = 0.01
    changed_observation_reference_displacement_m: float = 0.001
    event_phase_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)

    @classmethod
    def from_mapping(cls, timing: Mapping[str, Any]) -> TimingConfig:
        fractions = tuple(float(v) for v in timing["event_phase_fractions"])
        return cls(
            emulated_observation_action_delay_s=float(timing["emulated_observation_action_delay_s"]),
            standard_query_period_s=float(timing["standard_query_period_s"]),
            fast_query_period_s=float(timing["fast_query_period_s"]),
            episode_cap_s=float(timing["episode_cap_s"]),
            trigger_deadline_s=float(timing["trigger_deadline_s"]),
            post_event_cap_s=float(timing["post_event_cap_s"]),
            release_detection_dwell_ticks=int(timing["release_detection_dwell_ticks"]),
            release_settling_s=float(timing["release_settling_s"]),
            natural_grasp_min_lift_m=float(timing["natural_grasp_min_lift_m"]),
            natural_grasp_dwell_s=float(timing["natural_grasp_dwell_s"]),
            kinematic_grasp_relative_drift_max_m=float(timing["kinematic_grasp_relative_drift_max_m"]),
            changed_observation_reference_displacement_m=float(
                timing["changed_observation_reference_displacement_m"]
            ),
            event_phase_fractions=fractions,
        )


@dataclass(frozen=True)
class PolicyTimingAchieved:
    native_control_dt_s: float
    achieved_delay_s: float
    achieved_standard_query_period_s: float
    achieved_fast_query_period_s: float
    prediction_horizon_actions: int

    @classmethod
    def from_requested(
        cls,
        native_control_dt_s: float,
        timing: TimingConfig,
        prediction_horizon_actions: int,
    ) -> PolicyTimingAchieved:
        from experiments.online_correction_v4.clock import quantize_upward

        return cls(
            native_control_dt_s=native_control_dt_s,
            achieved_delay_s=quantize_upward(timing.emulated_observation_action_delay_s, native_control_dt_s),
            achieved_standard_query_period_s=quantize_upward(
                timing.standard_query_period_s, native_control_dt_s
            ),
            achieved_fast_query_period_s=quantize_upward(timing.fast_query_period_s, native_control_dt_s),
            prediction_horizon_actions=prediction_horizon_actions,
        )

    def action_horizon_s(self) -> float:
        """Total executed-action horizon available from one predicted chunk."""
        return self.prediction_horizon_actions * self.native_control_dt_s

    def required_queue_coverage_s(self, timing: TimingConfig) -> float:
        """Largest registered query interval plus emulated delay."""
        largest_query = max(
            self.achieved_standard_query_period_s,
            self.achieved_fast_query_period_s,
        )
        return largest_query + self.achieved_delay_s

    def covers_registered_interval(self, timing: TimingConfig) -> bool:
        required = self.required_queue_coverage_s(timing)
        return self.action_horizon_s() + 1e-12 >= required

    # Backward-compatible alias; prefer required_queue_coverage_s(timing).
    def queue_coverage_s(self, _query_period_s: float | None = None) -> float:
        return self.action_horizon_s()


@dataclass(frozen=True)
class EpisodeManifestRow:
    schema_version: int
    manifest_type: str
    runtime_bound: bool
    episode_id: str
    campaign: str
    family: str
    fixture: str
    block_id: int
    block_key: str
    env_seed: int
    policy_seed: int
    cohort: str
    priority: str
    factors: dict[str, str]
    prefix_group_id: str
    execution_group: str
    execution_order_key: str
    config_sha256: str
    reuse_episode_ids: tuple[str, ...]
    counterbalance: dict[str, Any]
    prompt_recipe: dict[str, Any]
    execution_order: int = 0

    @classmethod
    def from_manifest_dict(cls, row: Mapping[str, Any]) -> EpisodeManifestRow:
        return cls(
            schema_version=int(row["schema_version"]),
            manifest_type=str(row["manifest_type"]),
            runtime_bound=bool(row["runtime_bound"]),
            episode_id=str(row["episode_id"]),
            campaign=str(row["campaign"]),
            family=str(row["family"]),
            fixture=str(row["fixture"]),
            block_id=int(row["block_id"]),
            block_key=str(row["block_key"]),
            env_seed=int(row["env_seed"]),
            policy_seed=int(row["policy_seed"]),
            cohort=str(row["cohort"]),
            priority=str(row["priority"]),
            factors=dict(row["factors"]),
            prefix_group_id=str(row["prefix_group_id"]),
            execution_group=str(row["execution_group"]),
            execution_order_key=str(row["execution_order_key"]),
            config_sha256=str(row["config_sha256"]),
            reuse_episode_ids=tuple(str(v) for v in row.get("reuse_episode_ids", ())),
            counterbalance=dict(row["counterbalance"]),
            prompt_recipe=dict(row["prompt_recipe"]),
            execution_order=int(row.get("execution_order", 0)),
        )


@dataclass(frozen=True)
class NaturalGraspEvent:
    t_eligible: float
    control_tick: int
    lift_m: float
    dwell_s: float
    relative_drift_m: float


@dataclass(frozen=True)
class DetachmentEvent:
    t_onset: float
    t_detected: float
    onset_tick: int
    detected_tick: int


@dataclass
class EpisodeRuntimeFlags:
    trigger_eligible: bool = False
    event_delivered: bool = False
    event_observed: bool = False
    motion_truncated_by_release: bool = False
    passive_settling_started: bool = False
    policy_phase_active: bool = True


ACCEPTED_LEDGER_SCHEMA_VERSION = "online-correction-v4-accepted-ledger-v1"
REJECTED_ATTEMPT_SCHEMA_VERSION = "online-correction-v4-rejected-attempt-v1"
ACCEPTED_LEDGER_MANIFEST_SCHEMA_VERSION = "online-correction-v4-accepted-ledger-manifest-v1"
ATTEMPT_SELECTION_RULE = "latest_verified_valid_by_attempt_id"


@dataclass
class EpisodeTimingRecord:
    sim_time: float = 0.0
    t_reset: float = 0.0
    t_trigger_eligible: Optional[float] = None
    t_event_planned: Optional[float] = None
    t_intervention_command: Optional[float] = None
    t_motion_actual_onset: Optional[float] = None
    t_detach_onset: Optional[float] = None
    t_detach_detected: Optional[float] = None
    t_active_end: Optional[float] = None
    t_settling_start: Optional[float] = None
    t_complete: Optional[float] = None
    t_episode_end: Optional[float] = None
    query_times: list[float] = field(default_factory=list)
    request_records: list[dict[str, Any]] = field(default_factory=list)
