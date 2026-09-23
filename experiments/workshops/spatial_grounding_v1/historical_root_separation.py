"""Necessary root-layout exclusions in independently qualified metric frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .fixtures import RESET_POSITION_TOLERANCE_M
from .historical_root_nonmatch import RootPosition, _validated_position


@dataclass(frozen=True)
class MetricFrameContract:
    """Caller-audited orthonormal metre frames and per-root Euclidean error bounds."""

    historical_frame_id: str
    prospective_frame_id: str
    historical_root_error_m: float
    prospective_root_error_m: float


@dataclass(frozen=True)
class RootSeparationResult:
    status: str
    reason: str
    historical_distance_m: float | None = None
    prospective_distance_m: float | None = None
    separation_difference_lower_bound_m: float | None = None
    necessary_match_distance_bound_m: float | None = None


def prove_root_separation_nonmatch(
    historical: Mapping[str, RootPosition],
    prospective: Mapping[str, RootPosition],
    *,
    actor_pair: tuple[str, str],
    frames: MetricFrameContract,
) -> RootSeparationResult:
    """Disprove a match without knowing the transform between two metric frames.

    Matching both actor roots within the fixed componentwise tolerance requires
    their Euclidean separation distances to differ by at most twice sqrt(3)
    times that tolerance. Account conservatively for both roots' measurement
    errors on both sides. Distance agreement never proves a duplicate.
    """
    if not isinstance(frames, MetricFrameContract):
        return RootSeparationResult("unresolved", "missing qualified metric-frame contract")
    if any(not isinstance(value, str) or not value
           for value in (frames.historical_frame_id, frames.prospective_frame_id)):
        return RootSeparationResult("unresolved", "invalid metric-frame identity")
    errors = _validated_position((
        frames.historical_root_error_m, frames.prospective_root_error_m, 0.0,
    ))
    if errors is None or min(errors) < 0:
        return RootSeparationResult("unresolved", "missing or invalid per-root error bound")
    if (not isinstance(actor_pair, tuple) or len(actor_pair) != 2
            or any(not isinstance(actor, str) or not actor for actor in actor_pair)
            or actor_pair[0] == actor_pair[1]):
        return RootSeparationResult("unresolved", "two distinct actor identities are required")
    distances = []
    for roots, frame_id in (
        (historical, frames.historical_frame_id), (prospective, frames.prospective_frame_id),
    ):
        if not isinstance(roots, Mapping):
            return RootSeparationResult("unresolved", "malformed root-position mapping")
        positions = []
        for actor in actor_pair:
            root = roots.get(actor)
            if not isinstance(root, RootPosition) or root.frame_id != frame_id:
                return RootSeparationResult("unresolved", f"missing or unqualified actor root: {actor}")
            position = _validated_position(root.position_m)
            if position is None:
                return RootSeparationResult("unresolved", f"invalid actor root: {actor}")
            positions.append(position)
        distance = math.dist(*positions)
        if not math.isfinite(distance):
            return RootSeparationResult("unresolved", "nonfinite actor separation")
        distances.append(distance)
    old, new = distances
    uncertainty = 2 * errors[0] + 2 * errors[1]
    difference = abs(old - new)
    numeric_guard = 16 * (math.ulp(old) + math.ulp(new) + math.ulp(difference) + math.ulp(uncertainty))
    if not math.isfinite(uncertainty) or not math.isfinite(numeric_guard):
        return RootSeparationResult("unresolved", "nonfinite uncertainty bound")
    lower = max(0.0, math.nextafter(difference - uncertainty - numeric_guard, -math.inf))
    threshold = 2 * math.sqrt(3) * RESET_POSITION_TOLERANCE_M
    upper = math.nextafter(threshold + 16 * math.ulp(threshold), math.inf)
    if lower > upper:
        return RootSeparationResult(
            "nonmatch", "actor separation violates a necessary root-position match condition",
            old, new, lower, upper,
        )
    return RootSeparationResult(
        "unresolved", "separation distances do not disprove a match; duplication is not established",
        old, new, lower, upper,
    )
