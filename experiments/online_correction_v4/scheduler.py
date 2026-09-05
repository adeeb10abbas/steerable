"""Dependency-aware campaign scheduler and gate transitions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from experiments.online_correction_v4.contracts import CampaignPhase, GroupExecutionState
from experiments.online_correction_v4.leases import (
    DeadOwnerVerificationReceipt,
    GroupLease,
    GroupLeaseStore,
    LeaseConflict,
    StaleLeaseTakeoverDenied,
)
from experiments.online_correction_v4.registry import CampaignRegistry, ExecutionGroup


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING = "pending"


@dataclass(frozen=True)
class SchedulerTransition:
    previous: CampaignPhase
    new: CampaignPhase
    reason: str


FAMILY_CONTROL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "C3": ("C1",),
    "C4": ("C1", "C3"),
}


@dataclass
class CampaignScheduler:
    registry: CampaignRegistry
    lease_store: Optional[GroupLeaseStore] = None
    phase: CampaignPhase = CampaignPhase.IMPLEMENTING
    released_families: set[str] = field(default_factory=set)
    blocked_families: dict[str, str] = field(default_factory=dict)
    gate_status: dict[str, GateStatus] = field(default_factory=dict)
    accepted_episodes: set[str] = field(default_factory=set)
    completed_episodes: set[str] = field(default_factory=set)
    infra_invalid_counts: Counter[str] = field(default_factory=Counter)
    group_states: dict[str, GroupExecutionState] = field(default_factory=dict)
    active_leases: dict[str, GroupLease] = field(default_factory=dict)
    interrupted_groups: set[str] = field(default_factory=set)
    history: list[SchedulerTransition] = field(default_factory=list)

    def _transition(self, new_phase: CampaignPhase, reason: str) -> None:
        self.history.append(SchedulerTransition(previous=self.phase, new=new_phase, reason=reason))
        self.phase = new_phase

    def set_release_state(
        self,
        *,
        released: set[str],
        blocked: dict[str, str],
    ) -> list[str]:
        errors: list[str] = []
        declared = {row.family for row in self.registry.rows}
        if released & set(blocked):
            errors.append("released and blocked families overlap")
        if released | set(blocked) != declared:
            errors.append("released/blocked families must partition the manifest")
        for family in released:
            for dependency in FAMILY_CONTROL_DEPENDENCIES.get(family, ()):
                if dependency not in released:
                    errors.append(
                        f"{family}: control dependency {dependency} must be released"
                    )
        if not errors:
            self.released_families = set(released)
            self.blocked_families = dict(blocked)
            if self.phase is CampaignPhase.IMPLEMENTING:
                self._transition(CampaignPhase.QUALIFYING, "release state configured")
        return errors

    def record_gate(self, gate_id: str, status: GateStatus) -> None:
        self.gate_status[gate_id] = status

    def gates_ready_for_pilot(self, required: tuple[str, ...] = ("G0", "G1", "G2", "G3", "G4", "G5", "G6")) -> bool:
        return all(self.gate_status.get(g) is GateStatus.PASSED for g in required)

    def advance_to_piloting(self) -> None:
        if not self.gates_ready_for_pilot():
            raise RuntimeError("required qualification gates have not passed")
        self._transition(CampaignPhase.PILOTING, "engineering pilots authorized")

    def advance_to_frozen(self) -> None:
        if self.gate_status.get("G7") is not GateStatus.PASSED:
            raise RuntimeError("G7 engineering pilot gate must pass before freeze")
        self._transition(CampaignPhase.FROZEN, "main queue frozen")

    def advance_to_running(self) -> None:
        if self.phase is not CampaignPhase.FROZEN:
            raise RuntimeError("campaign must be frozen before running")
        self._transition(CampaignPhase.RUNNING, "main dispatch started")

    def family_dispatchable(self, family_id: str) -> bool:
        if family_id in self.blocked_families:
            return False
        if family_id not in self.released_families:
            return False
        for dependency in FAMILY_CONTROL_DEPENDENCIES.get(family_id, ()):
            if dependency not in self.released_families:
                return False
        return True

    def _group_remaining_episodes(self, group: ExecutionGroup) -> list[str]:
        remaining = []
        for row in group.rows:
            if not self.family_dispatchable(row.family):
                continue
            if row.episode_id in self.completed_episodes:
                continue
            remaining.append(row.episode_id)
        return remaining

    def dispatchable_groups(self) -> list[ExecutionGroup]:
        groups: list[ExecutionGroup] = []
        for group in self.registry.iter_groups():
            families = {row.family for row in group.rows}
            if not all(self.family_dispatchable(f) for f in families):
                continue
            ok, _ = self.registry.group_dependencies_satisfied(
                group, accepted_episode_ids=self.accepted_episodes
            )
            if not ok:
                continue
            if group.group_id in self.active_leases:
                continue
            if self._group_remaining_episodes(group):
                state = self.group_states.get(group.group_id, GroupExecutionState.UNLEASED)
                if state in {GroupExecutionState.UNLEASED, GroupExecutionState.PARTIAL}:
                    groups.append(group)
        return groups

    def acquire_group_lease(
        self,
        *,
        group_id: str,
        owner_lane: str,
        attempt_id: str,
        manifest_sha256: str,
        dead_owner_receipt: Optional[DeadOwnerVerificationReceipt] = None,
    ) -> GroupLease:
        if self.lease_store is None:
            raise RuntimeError("lease store is not configured")
        if group_id in self.active_leases:
            raise LeaseConflict(f"group {group_id} is already leased in scheduler state")
        lease = self.lease_store.acquire(
            group_id=group_id,
            owner_lane=owner_lane,
            attempt_id=attempt_id,
            manifest_sha256=manifest_sha256,
            dead_owner_receipt=dead_owner_receipt,
        )
        self.active_leases[group_id] = lease
        self.group_states[group_id] = GroupExecutionState.LEASED
        return lease

    def release_group_lease(self, group_id: str) -> None:
        lease = self.active_leases.pop(group_id, None)
        if lease is None:
            return
        if self.lease_store is not None:
            self.lease_store.release(lease)
        group = self.registry.by_execution_group[group_id]
        if self._group_remaining_episodes(group):
            self.group_states[group_id] = GroupExecutionState.PARTIAL
        else:
            self.group_states[group_id] = GroupExecutionState.COMPLETE

    def mark_group_interrupted(self, group_id: str) -> None:
        self.interrupted_groups.add(group_id)
        if group_id in self.active_leases:
            self.release_group_lease(group_id)
        self.group_states[group_id] = GroupExecutionState.PARTIAL

    def record_valid_episode(self, episode_id: str) -> None:
        row = self.registry.get(episode_id)
        if not self.family_dispatchable(row.family):
            raise RuntimeError(f"episode {episode_id} belongs to an unreleased family")
        self.accepted_episodes.add(episode_id)
        self.completed_episodes.add(episode_id)
        group_id = row.execution_group
        group = self.registry.by_execution_group[group_id]
        if not self._group_remaining_episodes(group):
            self.group_states[group_id] = GroupExecutionState.COMPLETE

    def record_infra_invalid(self, episode_id: str) -> None:
        self.infra_invalid_counts[episode_id] += 1

    def coverage_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in self.registry.rows:
            summary[row.family]["assigned"] += 1
            if row.episode_id in self.accepted_episodes:
                summary[row.family]["accepted"] += 1
            elif row.family in self.blocked_families:
                summary[row.family]["blocked"] += 1
        return {family: dict(counts) for family, counts in summary.items()}

    def remaining_episode_ids(self) -> list[str]:
        blocked_families = set(self.blocked_families)
        remaining = []
        for row in self.registry.rows:
            if row.family in blocked_families:
                continue
            if row.family not in self.released_families:
                continue
            if row.episode_id not in self.completed_episodes:
                remaining.append(row.episode_id)
        return remaining

    def can_close_campaign(self) -> bool:
        if self.phase not in {CampaignPhase.RUNNING, CampaignPhase.VERIFYING}:
            return False
        return not self.remaining_episode_ids()

    def advance_to_complete(self) -> None:
        if not self.can_close_campaign():
            raise RuntimeError("campaign still has pending released cells")
        self._transition(CampaignPhase.COMPLETE, "all released cells accepted or blocked")
