from __future__ import annotations

from experiments.workshops.spatial_grounding_v1.fixtures import (
    FixtureCandidate,
    FixtureError,
    Pose,
    candidate_order,
    select_qualified_layouts,
)


def candidate(identifier: str, *, family: str = "LAT", side: str | None = None) -> FixtureCandidate:
    poses = {
        "rubiks_cube": {"position_m": [0.4, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
        "bowl": {"position_m": [0.5, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
    }
    metadata = {}
    if family == "DIST":
        poses["plate"] = {"position_m": [0.3, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]}
        metadata["bowl_side"] = side
    if family == "HEIGHT":
        metadata["upper_support_side"] = side
    return FixtureCandidate.from_json({
        "candidate_id": identifier, "family": family, "seed": 1, "task_asset": "measured_scene.usda",
        "asset_manifest_sha256": "a" * 64, "object_poses": poses, "metadata": metadata,
    })


def test_candidate_rejects_non_neutral_pose() -> None:
    value = candidate("bad").task_payload()
    value.update({"seed": 1, "asset_manifest_sha256": "a" * 64, "object_poses": value["object_poses"], "metadata": {}})
    value["object_poses"]["rubiks_cube"]["position_m"][1] = 0.006
    try:
        FixtureCandidate.from_json(value)
    except FixtureError as error:
        assert "neutral" in str(error)
    else:
        raise AssertionError("candidate must reject a non-neutral reset")


def test_lat_selection_is_hash_ordered_and_complete() -> None:
    candidates = [candidate(f"c{index:02d}") for index in range(29)]
    selected = select_qualified_layouts("LAT", 7, candidates, {item.candidate_id for item in candidates})
    assert list(selected) == ["LAT-P01", *[f"LAT-D{i:02d}" for i in range(1, 5)], *[f"LAT-C{i:02d}" for i in range(1, 25)]]
    assert list(selected.values()) == [item.candidate_id for item in candidate_order(7, candidates)]


def test_dist_requires_frozen_counterbalance() -> None:
    candidates = [candidate(f"left{index}", family="DIST", side="left") for index in range(14)]
    candidates += [candidate(f"right{index}", family="DIST", side="right") for index in range(13)]
    try:
        select_qualified_layouts("DIST", 3, candidates, {item.candidate_id for item in candidates})
    except FixtureError as error:
        assert "counterbalance" in str(error)
    else:
        raise AssertionError("unbalanced DIST candidates must not be selected")
