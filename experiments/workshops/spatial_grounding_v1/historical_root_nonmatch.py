"""Fail-closed root-position nonmatch primitive.

This module is intentionally provenance-agnostic and is not wired into the
historical inventory or comparator. Callers must supply independently
hash-audited observations and an explicit common coordinate-frame contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


DEFAULT_RESET_POSITION_TOLERANCE_M = 0.003


@dataclass(frozen=True)
class CommonFrameContract:
    """The frame identity shared by both root-position observations."""

    frame_id: str


@dataclass(frozen=True)
class RootPosition:
    """An actor root position expressed in the declared common frame."""

    position_m: tuple[float, float, float]
    frame_id: str


@dataclass(frozen=True)
class RootNonmatchResult:
    """A conservative result; ``duplicate`` is intentionally not represented."""

    status: str
    reason: str
    differing_actors: tuple[str, ...] = ()
    max_component_differences_m: tuple[tuple[str, float], ...] = ()


def prove_root_position_nonmatch(
    historical: Mapping[str, RootPosition],
    prospective: Mapping[str, RootPosition],
    *,
    required_actors: Sequence[str],
    frame: CommonFrameContract,
    tolerance_m: float = DEFAULT_RESET_POSITION_TOLERANCE_M,
) -> RootNonmatchResult:
    """Prove only the necessary root-position condition for a nonmatch.

    Every required actor must be present, finite, three-dimensional, and
    expressed in exactly ``frame.frame_id`` on both sides. A result is
    ``nonmatch`` iff at least one actor's maximum absolute coordinate
    difference is strictly greater than ``tolerance_m``. All other cases are
    ``unresolved``. Root agreement never proves duplication because centroid,
    orientation, asset, and population evidence are outside this primitive.
    """

    if not isinstance(frame.frame_id, str) or not frame.frame_id:
        return RootNonmatchResult("unresolved", "invalid common-frame contract")
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        return RootNonmatchResult("unresolved", "invalid position tolerance")

    actors = tuple(dict.fromkeys(required_actors))
    if not actors or any(not isinstance(actor, str) or not actor for actor in actors):
        return RootNonmatchResult("unresolved", "required actor set is empty or invalid")

    differences: list[tuple[str, float]] = []
    for actor in actors:
        old = historical.get(actor)
        new = prospective.get(actor)
        if old is None or new is None:
            return RootNonmatchResult(
                "unresolved", f"missing required actor root: {actor}"
            )
        if old.frame_id != frame.frame_id or new.frame_id != frame.frame_id:
            return RootNonmatchResult(
                "unresolved", f"coordinate-frame mismatch for actor: {actor}"
            )
        old_position = _validated_position(old.position_m)
        new_position = _validated_position(new.position_m)
        if old_position is None or new_position is None:
            return RootNonmatchResult(
                "unresolved", f"invalid root position for actor: {actor}"
            )
        differences.append(
            (actor, max(abs(a - b) for a, b in zip(old_position, new_position)))
        )

    differing = tuple(actor for actor, difference in differences if difference > tolerance_m)
    if not differing:
        return RootNonmatchResult(
            "unresolved",
            "all required root positions are within tolerance; duplication is not established",
            max_component_differences_m=tuple(differences),
        )
    return RootNonmatchResult(
        "nonmatch",
        "required actor root-position separation exceeds tolerance",
        differing_actors=differing,
        max_component_differences_m=tuple(differences),
    )


def _validated_position(
    position: Sequence[float],
) -> tuple[float, float, float] | None:
    if isinstance(position, (str, bytes)):
        return None
    try:
        if len(position) != 3:
            return None
        values = tuple(float(value) for value in position)
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None
