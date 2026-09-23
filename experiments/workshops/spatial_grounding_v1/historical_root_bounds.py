"""Necessary root nonmatches for independently justified position intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .historical_root_nonmatch import (
    CommonFrameContract,
    RootNonmatchResult,
    RootPosition,
    _validated_position,
    prove_root_position_nonmatch,
)


@dataclass(frozen=True)
class RootPositionBounds:
    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]
    frame_id: str


def prove_bounded_root_nonmatch(
    historical: Mapping[str, RootPositionBounds],
    prospective: Mapping[str, RootPosition],
    *,
    required_actors: Sequence[str],
    frame: CommonFrameContract,
) -> RootNonmatchResult:
    """Compare against each interval's nearest point, never its nominal center.

    Returned component differences are lower bounds on root separation, not
    observed historical displacements. Bound provenance remains the caller's
    responsibility; neither a match nor historical coverage is established.
    """
    if not isinstance(historical, Mapping):
        return RootNonmatchResult("unresolved", "malformed historical root-bound mapping")
    lower_roots = {
        actor: RootPosition(bound.lower_m, bound.frame_id)
        for actor, bound in historical.items() if isinstance(bound, RootPositionBounds)
    }
    checked = prove_root_position_nonmatch(
        lower_roots, prospective, required_actors=required_actors, frame=frame,
    )
    if not checked.max_component_differences_m:
        return checked
    nearest = {}
    for actor, _ in checked.max_component_differences_m:
        bound = historical[actor]
        low = _validated_position(bound.lower_m)
        high = _validated_position(bound.upper_m)
        position = _validated_position(prospective[actor].position_m)
        if low is None or high is None or position is None or any(a > b for a, b in zip(low, high)):
            return RootNonmatchResult("unresolved", f"invalid root-position interval for actor: {actor}")
        nearest[actor] = RootPosition(
            tuple(max(a, min(value, b)) for a, b, value in zip(low, high, position)),
            bound.frame_id,
        )
    result = prove_root_position_nonmatch(
        nearest, prospective, required_actors=required_actors, frame=frame,
    )
    return RootNonmatchResult(
        result.status,
        "minimum root-position separation from the bound exceeds tolerance"
        if result.status == "nonmatch" else "historical root bound may overlap reset tolerance",
        result.differing_actors,
        result.max_component_differences_m,
    )
