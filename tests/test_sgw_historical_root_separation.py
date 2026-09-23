import math

import pytest

from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import RootPosition
from experiments.workshops.spatial_grounding_v1.historical_root_separation import (
    MetricFrameContract, prove_root_separation_nonmatch,
)


FRAMES = MetricFrameContract("old_robot_base", "new_world", 0.0, 0.0)
PAIR = ("rubiks_cube", "bowl")


def roots(first, second, frame):
    return {PAIR[0]: RootPosition(tuple(first), frame), PAIR[1]: RootPosition(tuple(second), frame)}


def compare(old, new, frames=FRAMES):
    return prove_root_separation_nonmatch(old, new, actor_pair=PAIR, frames=frames)


def test_distance_nonmatch_is_independent_of_frame_translation_and_rotation():
    old = roots((0, 0, 0), (0.14, 0, 0), "old_robot_base")
    new = roots((100, -40, 70), (100, -40, 70.3), "new_world")
    result = compare(old, new)
    assert result.status == "nonmatch"
    assert result.separation_difference_lower_bound_m > 0.15999


def test_equal_separations_are_not_a_duplicate_claim():
    result = compare(roots((0, 0, 0), (0.14, 0, 0), "old_robot_base"),
                     roots((2, 3, 4), (2, 3.14, 4), "new_world"))
    assert result.status == "unresolved"


@pytest.mark.parametrize("pair", [None, "rubiks_cube", ("bowl",), ("bowl", "bowl"), ("", "bowl")])
def test_unidentified_actor_pair_cannot_exclude(pair):
    result = prove_root_separation_nonmatch(
        roots((0, 0, 0), (0.14, 0, 0), "old_robot_base"),
        roots((0, 0, 0), (0.3, 0, 0), "new_world"),
        actor_pair=pair, frames=FRAMES,
    )
    assert result.status == "unresolved"


def test_missing_metric_contract_cannot_exclude():
    result = compare(roots((0, 0, 0), (0.14, 0, 0), "old_robot_base"),
                     roots((0, 0, 0), (0.3, 0, 0), "new_world"), None)
    assert result.status == "unresolved"


@pytest.mark.parametrize("factor,expected", [(1 - 1e-8, "unresolved"), (1, "unresolved"), (1 + 1e-8, "nonmatch")])
def test_exact_frozen_component_tolerance_boundary(factor, expected):
    old = roots((0, 0, 0), (0.1, 0.1, 0.1), "old_robot_base")
    new = roots((-0.003 * factor,) * 3, (0.1 + 0.003 * factor,) * 3, "new_world")
    assert compare(old, new).status == expected


def test_both_roots_error_bounds_are_accounted_on_both_sides():
    old = roots((0, 0, 0), (0.14, 0, 0), "old_robot_base")
    new = roots((0, 0, 0), (0.17, 0, 0), "new_world")
    assert compare(old, new).status == "nonmatch"
    frames = MetricFrameContract("old_robot_base", "new_world", 0.005, 0.005)
    assert compare(old, new, frames).status == "unresolved"


@pytest.mark.parametrize("error", [None, True, -0.001, math.nan, math.inf, "0", 10**1000])
def test_missing_or_invalid_precision_contract_never_excludes(error):
    frames = MetricFrameContract("old_robot_base", "new_world", error, 0)
    assert compare(roots((0, 0, 0), (1, 0, 0), "old_robot_base"),
                   roots((0, 0, 0), (2, 0, 0), "new_world"), frames).status == "unresolved"


@pytest.mark.parametrize("mutation", ["missing", "frame", "nonfinite", "boolean", "overflow"])
def test_every_actor_and_frame_must_be_valid(mutation):
    old = roots((0, 0, 0), (0.14, 0, 0), "old_robot_base")
    new = roots((0, 0, 0), (0.3, 0, 0), "new_world")
    if mutation == "missing":
        del old["bowl"]
    elif mutation == "frame":
        old["bowl"] = RootPosition((0.14, 0, 0), "unknown")
    elif mutation == "nonfinite":
        old["bowl"] = RootPosition((math.nan, 0, 0), "old_robot_base")
    elif mutation == "boolean":
        old["bowl"] = RootPosition((True, 0, 0), "old_robot_base")
    else:
        old = roots((-1e308, 0, 0), (1e308, 0, 0), "old_robot_base")
    assert compare(old, new).status == "unresolved"
