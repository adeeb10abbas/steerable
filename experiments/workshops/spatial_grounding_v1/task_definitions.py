"""Goal-independent RoboLab task definitions for SGW-01 qualifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fixtures import ACTION_CAP, FixtureCandidate


@dataclass(frozen=True)
class RoboLabTaskDefinition:
    """Input to the bridge; it never exposes a success predicate to a policy."""

    candidate: FixtureCandidate
    action_cap: int = ACTION_CAP
    goal_termination: bool = False

    def __post_init__(self) -> None:
        if self.action_cap != ACTION_CAP or self.goal_termination:
            raise ValueError("SGW-01 fixtures require a goal-independent 450-action task")

    def bridge_config(self) -> dict[str, Any]:
        return self.candidate.task_payload()


def build_task_definition(candidate: FixtureCandidate) -> RoboLabTaskDefinition:
    """Build a candidate-bound definition after its asset identity is verified."""

    candidate.validate_neutral_start()
    return RoboLabTaskDefinition(candidate=candidate)
