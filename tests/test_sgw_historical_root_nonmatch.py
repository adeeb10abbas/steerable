from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import (
    CommonFrameContract,
    RootPosition,
    prove_root_position_nonmatch,
)


FRAME = CommonFrameContract("robot_base_object_root_m")


def pose(x, y=0.0, z=0.0, frame_id="robot_base_object_root_m"):
    return RootPosition((x, y, z), frame_id)


def test_strictly_greater_than_three_mm_proves_nonmatch():
    result = prove_root_position_nonmatch(
        {"cube": pose(0.0)},
        {"cube": pose(0.003001)},
        required_actors=("cube",),
        frame=FRAME,
    )
    assert result.status == "nonmatch"
    assert result.differing_actors == ("cube",)


def test_three_mm_boundary_is_not_a_nonmatch():
    result = prove_root_position_nonmatch(
        {"cube": pose(0.0)},
        {"cube": pose(0.003)},
        required_actors=("cube",),
        frame=FRAME,
    )
    assert result.status == "unresolved"
    assert result.differing_actors == ()


def test_near_match_never_becomes_duplicate():
    result = prove_root_position_nonmatch(
        {"cube": pose(0.0), "bowl": pose(0.2)},
        {"cube": pose(0.001), "bowl": pose(0.201)},
        required_actors=("cube", "bowl"),
        frame=FRAME,
    )
    assert result.status == "unresolved"
    assert "duplication" in result.reason


def test_missing_actor_and_invalid_pose_are_unresolved():
    missing = prove_root_position_nonmatch(
        {"cube": pose(0.0)},
        {},
        required_actors=("cube", "bowl"),
        frame=FRAME,
    )
    invalid = prove_root_position_nonmatch(
        {"cube": RootPosition((float("nan"), 0.0, 0.0), FRAME.frame_id)},
        {"cube": pose(1.0)},
        required_actors=("cube",),
        frame=FRAME,
    )
    assert missing.status == invalid.status == "unresolved"


def test_frame_mismatch_is_fail_closed():
    result = prove_root_position_nonmatch(
        {"cube": pose(0.0, frame_id="world")},
        {"cube": pose(1.0)},
        required_actors=("cube",),
        frame=FRAME,
    )
    assert result.status == "unresolved"
    assert "frame" in result.reason


def test_all_required_actors_must_be_checked():
    result = prove_root_position_nonmatch(
        {"cube": pose(0.0), "bowl": pose(0.0)},
        {"cube": pose(0.004), "bowl": pose(0.0)},
        required_actors=("cube", "bowl"),
        frame=FRAME,
    )
    assert result.status == "nonmatch"
    assert result.differing_actors == ("cube",)


def test_malformed_containers_are_unresolved():
    assert prove_root_position_nonmatch(
        [], {}, required_actors=("cube",), frame=FRAME
    ).status == "unresolved"
    assert prove_root_position_nonmatch(
        {}, {}, required_actors="cube", frame=FRAME
    ).status == "unresolved"
    assert prove_root_position_nonmatch(
        {}, {}, required_actors=(["cube"],), frame=FRAME
    ).status == "unresolved"
    assert prove_root_position_nonmatch(
        {}, {}, required_actors=(), frame=FRAME
    ).status == "unresolved"
    assert prove_root_position_nonmatch(
        {}, {}, required_actors=("cube", "cube"), frame=FRAME
    ).status == "unresolved"
    assert prove_root_position_nonmatch(
        {}, {}, required_actors=("cube",), frame="not-a-contract"
    ).status == "unresolved"


def test_invalid_coordinate_values_are_unresolved():
    for value in ((True, 0.0, 0.0), ("0.0", 0.0, 0.0), (0.0, 0.0), (float("inf"), 0.0, 0.0)):
        result = prove_root_position_nonmatch(
            {"cube": RootPosition(value, FRAME.frame_id)},
            {"cube": pose(1.0)},
            required_actors=("cube",),
            frame=FRAME,
        )
        assert result.status == "unresolved"


def test_invalid_later_actor_blocks_earlier_separation():
    result = prove_root_position_nonmatch(
        {"cube": pose(0.0), "bowl": RootPosition((True, 0.0, 0.0), FRAME.frame_id)},
        {"cube": pose(0.004), "bowl": pose(0.0)},
        required_actors=("cube", "bowl"),
        frame=FRAME,
    )
    assert result.status == "unresolved"
    assert result.differing_actors == ()


def test_tolerance_is_not_caller_configurable():
    result = prove_root_position_nonmatch(
        {"cube": pose(0.0)},
        {"cube": pose(0.002)},
        required_actors=("cube",),
        frame=FRAME,
    )
    assert result.status == "unresolved"


def test_unordered_and_overflowing_coordinates_are_unresolved():
    for value in ({0, 1, 2}, {0: 1, 1: 2, 2: 3}, (10**1000, 0, 0)):
        result = prove_root_position_nonmatch(
            {"cube": RootPosition(value, FRAME.frame_id)},
            {"cube": pose(0)},
            required_actors=("cube",),
            frame=FRAME,
        )
        assert result.status == "unresolved"
    assert prove_root_position_nonmatch(
        {"cube": pose(-1e308)}, {"cube": pose(1e308)},
        required_actors=("cube",), frame=FRAME,
    ).status == "unresolved"
