import math

import pytest

from experiments.workshops.spatial_grounding_v1.historical_root_bounds import (
    RootPositionBounds,
    prove_bounded_root_nonmatch,
)
from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import (
    CommonFrameContract,
    RootPosition,
)


FRAME = CommonFrameContract("world_axes_m")


def bounds():
    return {"cube": RootPositionBounds(
        (math.nextafter(-0.005, -math.inf),) * 3,
        (math.nextafter(0.005, math.inf),) * 3,
        FRAME.frame_id,
    )}


@pytest.mark.parametrize("x,status", [
    (0.0, "unresolved"),
    (0.005, "unresolved"),
    (0.0079, "unresolved"),
    (0.008, "unresolved"),
    (0.0081, "nonmatch"),
    (-0.008, "unresolved"),
    (-0.0081, "nonmatch"),
])
def test_five_mm_historical_bound_is_added_to_fixed_three_mm_reset_tolerance(x, status):
    result = prove_bounded_root_nonmatch(
        bounds(), {"cube": RootPosition((x, 0, 0), FRAME.frame_id)},
        required_actors=("cube",), frame=FRAME,
    )
    assert result.status == status
    assert result.status != "duplicate"


@pytest.mark.parametrize("upper", [(math.nan, 1, 1), (math.inf, 1, 1), (-1, -1, -1), (1, 1), ("1", 1, 1)])
def test_invalid_later_actor_blocks_an_earlier_nonmatch(upper):
    old = {**bounds(), "bowl": RootPositionBounds((0, 0, 0), upper, FRAME.frame_id)}
    new = {actor: RootPosition((1, 1, 1), FRAME.frame_id) for actor in old}
    result = prove_bounded_root_nonmatch(old, new, required_actors=("cube", "bowl"), frame=FRAME)
    assert result.status == "unresolved"
    assert result.max_component_differences_m == ()


def test_wrong_frame_and_missing_actor_remain_unresolved():
    new = {"cube": RootPosition((1, 1, 1), "another_frame")}
    assert prove_bounded_root_nonmatch(
        bounds(), new, required_actors=("cube",), frame=FRAME,
    ).status == "unresolved"
    assert prove_bounded_root_nonmatch(
        bounds(), {}, required_actors=("cube",), frame=FRAME,
    ).status == "unresolved"


def test_reported_distance_is_to_nearest_point_not_center():
    result = prove_bounded_root_nonmatch(
        bounds(), {"cube": RootPosition((0.009, 0.002, -0.001), FRAME.frame_id)},
        required_actors=("cube",), frame=FRAME,
    )
    assert result.status == "nonmatch"
    assert dict(result.max_component_differences_m)["cube"] == pytest.approx(0.004)


def test_diagonal_offset_does_not_replace_componentwise_tolerance_with_euclidean_distance():
    result = prove_bounded_root_nonmatch(
        bounds(), {"cube": RootPosition((0.007, 0.007, 0.007), FRAME.frame_id)},
        required_actors=("cube",), frame=FRAME,
    )
    assert result.status == "unresolved"
    assert dict(result.max_component_differences_m)["cube"] == pytest.approx(0.002)


@pytest.mark.parametrize("actors", [(), "cube", ("cube", "cube"), ("cube", "missing")])
def test_invalid_or_missing_required_actors_do_not_allow_partial_nonmatches(actors):
    result = prove_bounded_root_nonmatch(
        bounds(), {"cube": RootPosition((1, 1, 1), FRAME.frame_id)},
        required_actors=actors, frame=FRAME,
    )
    assert result.status == "unresolved"
