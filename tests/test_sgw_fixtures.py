from __future__ import annotations

from pathlib import Path

from experiments.workshops.spatial_grounding_v1.fixtures import (
    FixtureCandidate,
    FixtureError,
    candidate_order,
    select_qualified_layouts,
)
from experiments.workshops.spatial_grounding_v1.build_asset_manifest import is_git_worktree, referenced_usd_assets
from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import materialize_lat_candidates, workspace_digest


def candidate(identifier: str, *, family: str = "LAT", side: str | None = None, x: float = 0.4) -> FixtureCandidate:
    poses = {
        "rubiks_cube": {"position_m": [x, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
        "bowl": {"position_m": [x + 0.1, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
    }
    metadata = {"scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]}}
    if family == "DIST":
        poses["plate"] = {"position_m": [x - 0.1, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]}
        metadata["bowl_side"] = side
        metadata["scoring_center_offsets_root_local_m"]["plate"] = [0, 0, 0]
    if family == "HEIGHT":
        metadata["upper_support_side"] = side
    return FixtureCandidate.from_json({
        "candidate_id": identifier, "family": family, "seed": 1, "task_asset": "measured_scene.usda",
        "asset_manifest_sha256": "a" * 64, "object_poses": poses, "metadata": metadata,
    })


def test_candidate_rejects_non_neutral_pose() -> None:
    value = {
        "candidate_id": "bad", "family": "LAT", "seed": 1, "task_asset": "measured_scene.usda",
        "asset_manifest_sha256": "a" * 64, "metadata": {"scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]}},
        "object_poses": {
            "rubiks_cube": {"position_m": [0.4, 0.006, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
            "bowl": {"position_m": [0.5, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
    }
    try:
        FixtureCandidate.from_json(value)
    except FixtureError as error:
        assert "neutral" in str(error)
    else:
        raise AssertionError("candidate must reject a non-neutral reset")


def test_lat_selection_is_hash_ordered_and_complete() -> None:
    candidates = [candidate(f"c{index:02d}", x=0.4 + index * 0.01) for index in range(29)]
    selected = select_qualified_layouts("LAT", 7, candidates, {item.candidate_id for item in candidates})
    assert list(selected) == ["LAT-P01", *[f"LAT-D{i:02d}" for i in range(1, 5)], *[f"LAT-C{i:02d}" for i in range(1, 25)]]
    assert list(selected.values()) == [item.candidate_id for item in candidate_order(7, candidates)]


def test_dist_requires_frozen_counterbalance() -> None:
    candidates = [
        candidate(f"left{index}", family="DIST", side="left", x=0.4 + index * 0.01)
        for index in range(14)
    ]
    candidates += [
        candidate(f"right{index}", family="DIST", side="right", x=0.6 + index * 0.01)
        for index in range(13)
    ]
    try:
        select_qualified_layouts("DIST", 3, candidates, {item.candidate_id for item in candidates})
    except FixtureError as error:
        assert "counterbalance" in str(error)
    else:
        raise AssertionError("unbalanced DIST candidates must not be selected")


def test_candidates_within_reset_tolerance_are_duplicates() -> None:
    first = candidate("first")
    second = candidate("second", x=0.402)
    try:
        candidate_order(1, [first, second])
    except FixtureError as error:
        assert "within the reset tolerance" in str(error)
    else:
        raise AssertionError("near-identical layouts must be rejected")


def test_root_reset_and_scoring_center_are_explicitly_separate() -> None:
    value = {
        "candidate_id": "offset", "family": "LAT", "seed": 1, "task_asset": "measured.usda",
        "asset_manifest_sha256": "a" * 64,
        "object_poses": {
            "rubiks_cube": {"position_m": [0.4, 0.1, 0.1], "quaternion_wxyz": [0, 0, 0, 1]},
            "bowl": {"position_m": [0.5, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "metadata": {"scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0.1, 0], "bowl": [0, 0, 0]}},
    }
    candidate_value = FixtureCandidate.from_json(value)
    assert candidate_value.object_poses["rubiks_cube"].position_m == (0.4, 0.1, 0.1)
    assert candidate_value.scoring_poses()["rubiks_cube"].position_m == (0.4, 0.0, 0.1)


def test_asset_manifest_resolves_actual_usda_references(tmp_path) -> None:
    root = tmp_path / "robolab"
    scene = root / "assets/scenes/scene.usda"
    object_usd = root / "assets/objects/cube.usd"
    texture = root / "assets/textures/cube.png"
    scene.parent.mkdir(parents=True)
    object_usd.parent.mkdir(parents=True)
    texture.parent.mkdir(parents=True)
    scene.write_text('@../objects/cube.usd@\n', encoding="utf-8")
    object_usd.write_text('@../textures/cube.png@\n', encoding="utf-8")
    texture.write_bytes(b"texture")
    assert referenced_usd_assets(scene, root) == [scene.resolve(), object_usd.resolve(), texture.resolve()]


def test_git_worktree_identity_accepts_dot_git_file_and_rejects_plain_directory(tmp_path) -> None:
    repository_root = Path(__file__).parents[1]
    assert (repository_root / ".git").is_file()
    assert is_git_worktree(repository_root)
    plain_directory = tmp_path / "not-a-checkout"
    plain_directory.mkdir()
    assert not is_git_worktree(plain_directory)


def test_lat_candidates_use_only_measured_workspace_slots() -> None:
    slot = {
        "slot_id": "actual-slot-1",
        "object_poses": {
            "rubiks_cube": {"position_m": [0.4, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
            "bowl": {"position_m": [0.5, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "center_source": "pinned_robolab_geometric_center",
        "scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]},
        "abs_ik_waypoints": {"positive": [{"position_world_xyz_m": [0.4, 0.0, 0.2], "gripper_position": 0, "hold_steps": 1}], "negative": [{"position_world_xyz_m": [0.4, 0.0, 0.2], "gripper_position": 0, "hold_steps": 1}]},
    }
    candidates = materialize_lat_candidates({
        "schema_version": "sgw-01-lat-measured-workspace-v1",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "asset_manifest_sha256": "a" * 64,
        "task_asset": "rubiks_cube_banana_bowl.usda",
        "receipt_sha256": "b" * 64,
        "validated_slots": [slot],
    }, seed=4)
    assert candidates[0]["metadata"]["source_slot_id"] == "actual-slot-1"


def test_lat_candidates_require_an_explicit_geometric_center_binding() -> None:
    slot = {
        "slot_id": "unbound-center",
        "object_poses": {
            "rubiks_cube": {"position_m": [0.4, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
            "bowl": {"position_m": [0.5, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "abs_ik_waypoints": {"positive": [{"position_world_xyz_m": [0.4, 0.0, 0.2], "gripper_position": 0, "hold_steps": 1}], "negative": [{"position_world_xyz_m": [0.4, 0.0, 0.2], "gripper_position": 0, "hold_steps": 1}]},
    }
    try:
        materialize_lat_candidates({
            "schema_version": "sgw-01-lat-measured-workspace-v1",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "asset_manifest_sha256": "a" * 64,
            "task_asset": "rubiks_cube_banana_bowl.usda",
            "receipt_sha256": "b" * 64,
            "validated_slots": [slot],
        }, seed=4)
    except ValueError as error:
        assert "geometric-center" in str(error)
    else:
        raise AssertionError("unbound center semantics must fail closed")


def test_workspace_digest_excludes_its_self_reference() -> None:
    receipt = {"schema_version": "sgw-01-lat-measured-workspace-v1", "receipt_sha256": "wrong", "validated_slots": []}
    first = workspace_digest(receipt)
    receipt["receipt_sha256"] = first
    assert workspace_digest(receipt) == first
