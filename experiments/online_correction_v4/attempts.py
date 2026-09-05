"""Attempt classification and infrastructure retry rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from experiments.online_correction_v4.contracts import (
    AttemptStatus,
    FailureLabel,
    FailureStage,
    failure_stage_for_label,
)


class InfraInvalidReason(str, Enum):
    CHECKPOINT_HASH_MISMATCH = "checkpoint_hash_mismatch"
    RESET_ATTESTATION_FAILED = "reset_attestation_failed"
    MALFORMED_ACTION_INTERFACE = "malformed_action_interface"
    SIMULATOR_CRASH = "simulator_crash"
    MISSING_MANDATORY_STREAM = "missing_mandatory_stream"
    WRONG_RESET = "wrong_reset"
    SCHEDULER_TIMING_VIOLATION = "scheduler_timing_violation"
    WRONG_STIMULUS_TRAJECTORY = "wrong_stimulus_trajectory"
    SCORER_RUNTIME_CORRUPTION = "scorer_runtime_corruption"
    CLUSTER_PREEMPTION_INCOMPLETE = "cluster_preemption_incomplete"


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    lane_quarantine_threshold: int = 5

    def allows_retry(self, prior_infra_attempts: int) -> bool:
        return prior_infra_attempts < self.max_retries


@dataclass(frozen=True)
class AttemptRecord:
    episode_id: str
    attempt_id: str
    status: AttemptStatus
    reason: Optional[str] = None
    infra_invalid_reason: Optional[InfraInvalidReason] = None
    prior_infra_attempts: int = 0
    failure_label: Optional[FailureLabel] = None
    failure_stage: Optional[FailureStage] = None


@dataclass
class AttemptClassifier:
    retry_policy: RetryPolicy = RetryPolicy()

    def classify_behavioral_failure(
        self,
        *,
        failure_label: FailureLabel,
        episode_id: str = "",
        attempt_id: str = "",
    ) -> AttemptRecord:
        return AttemptRecord(
            episode_id=episode_id,
            attempt_id=attempt_id,
            status=AttemptStatus.VALID,
            reason=failure_label.value,
            failure_label=failure_label,
            failure_stage=failure_stage_for_label(failure_label),
        )

    def classify_infra_invalid(
        self,
        *,
        episode_id: str,
        attempt_id: str,
        reason: InfraInvalidReason,
        prior_infra_attempts: int,
    ) -> AttemptRecord:
        return AttemptRecord(
            episode_id=episode_id,
            attempt_id=attempt_id,
            status=AttemptStatus.INFRA_INVALID,
            reason=reason.value,
            infra_invalid_reason=reason,
            prior_infra_attempts=prior_infra_attempts,
            failure_label=FailureLabel.UNRESOLVED_BEHAVIORAL_FAILURE,
            failure_stage=FailureStage.OTHER,
        )

    def authorize_retry(self, record: AttemptRecord) -> bool:
        if record.status is not AttemptStatus.INFRA_INVALID:
            return False
        return self.retry_policy.allows_retry(record.prior_infra_attempts)

    def next_attempt_id(self, episode_id: str, prior_attempt_ids: list[str]) -> str:
        index = len(prior_attempt_ids) + 1
        return f"{episode_id}--a{index:03d}"


@dataclass(frozen=True)
class TerminalEvidenceFlags:
    success: bool = False
    grasp_occurred: bool = False
    carry_verified: bool = False
    grasp_lost: bool = False
    released: bool = False
    trigger_eligible: bool = False
    event_delivered: bool = False
    transport_incomplete: bool = False
    geometric_relation_correct: bool = False
    allowed_support: bool = False
    allowed_containment: bool = False
    stable_for_dwell: bool = False
    boundary_violation: bool = False
    collision_terminal_failure: bool = False
    model_output_invalid: bool = False
    unresolved_behavioral_failure: bool = False
    timeout_without_completion: bool = False
    timeout_after_no_grasp: bool = False


def derive_failure_label(flags: TerminalEvidenceFlags) -> FailureLabel:
    """Earliest decisive failure under the frozen predicate hierarchy."""
    if flags.success:
        return FailureLabel.SUCCESS
    if flags.model_output_invalid:
        return FailureLabel.MODEL_OUTPUT_INVALID
    if flags.unresolved_behavioral_failure:
        return FailureLabel.UNRESOLVED_BEHAVIORAL_FAILURE
    if flags.collision_terminal_failure or flags.boundary_violation:
        return FailureLabel.COLLISION_CAUSED_TERMINAL_FAILURE
    if not flags.grasp_occurred:
        return FailureLabel.NO_GRASP
    if flags.grasp_lost:
        return FailureLabel.GRASP_LOST
    if flags.transport_incomplete or not flags.carry_verified:
        return FailureLabel.TRANSPORT_INCOMPLETE
    if not flags.released:
        return FailureLabel.RELEASE_FAILED
    if not flags.geometric_relation_correct:
        return FailureLabel.WRONG_GOAL_REGION
    if not (flags.allowed_support or flags.allowed_containment) or not flags.stable_for_dwell:
        return FailureLabel.SUPPORT_OR_CONTAINMENT_FAILED
    if flags.timeout_without_completion:
        return FailureLabel.TIMEOUT_WITHOUT_COMPLETION
    return FailureLabel.UNRESOLVED_BEHAVIORAL_FAILURE


def classify_terminal_outcome(flags: TerminalEvidenceFlags) -> tuple[AttemptStatus, FailureLabel, FailureStage, dict[str, Any]]:
    label = derive_failure_label(flags)
    stage = failure_stage_for_label(label)
    meta: dict[str, Any] = {
        "failure_label": label.value,
        "failure_stage": stage.value,
        "timeout_after_no_grasp": flags.timeout_after_no_grasp,
    }
    if label is FailureLabel.SUCCESS:
        meta["success"] = True
    return AttemptStatus.VALID, label, stage, meta
