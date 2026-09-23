import json
from argparse import Namespace

import pytest

from experiments.workshops.spatial_grounding_v1.height_dist_proposals import main, propose_family_layouts
from experiments.workshops.spatial_grounding_v1.model_blind_qualification import _load_selected_candidate


def _pose(position):
    return {"position_m": position, "quaternion_wxyz": [1, 0, 0, 0]}


def _workspace():
    return {
        "receipt_sha256": "a" * 64,
        "asset_manifest_sha256": "b" * 64,
        "task_asset": "measured_scene.usda",
    }


def _scene(names):
    return {"asset": "measured_scene.usda", "object_names": [*names, "table", "upper_platform", "lower_platform"]}


def _height_row():
    return {
        "layout_id": "h-001",
        "object_root_poses": {"rubiks_cube": _pose([.3, 0, .1]), "bowl": _pose([.5, 0, .1])},
        "scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]},
        "scoring_centers_env_local_xyz_m": {"rubiks_cube": [.3, 0, .1], "bowl": [.5, 0, .1]},
        "native_scene": _scene(["rubiks_cube", "bowl"]),
        "upper_support_side": "left",
        "goal_supports": {
            "higher": {"support_surface_id": "upper_platform_top", "contact_sensor_id": "rubiks_cube__upper_platform",
                       "cube_center_env_local_xyz_m": [.3, -.1, .14]},
            "lower": {"support_surface_id": "lower_platform_top", "contact_sensor_id": "rubiks_cube__lower_platform",
                      "cube_center_env_local_xyz_m": [.3, .1, .06]},
        },
    }


def _dist_row():
    return {
        "layout_id": "d-001",
        "object_root_poses": {
            "rubiks_cube": _pose([.5, 0, .1]), "bowl": _pose([.4, 0, .1]), "plate": _pose([.6, 0, .1]),
        },
        "scoring_center_offsets_root_local_m": {
            "rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0], "plate": [0, 0, 0],
        },
        "scoring_centers_env_local_xyz_m": {
            "rubiks_cube": [.5, 0, .1], "bowl": [.4, 0, .1], "plate": [.6, 0, .1],
        },
        "native_scene": _scene(["rubiks_cube", "bowl", "plate"]),
        "bowl_side": "right",
        "goal_supports": {
            "near_bowl": {"support_surface_id": "table-left", "contact_sensor_id": "rubiks_cube__table_left",
                          "cube_center_env_local_xyz_m": [.43, 0, .1]},
            "near_plate": {"support_surface_id": "table-right", "contact_sensor_id": "rubiks_cube__table_right",
                           "cube_center_env_local_xyz_m": [.57, 0, .1]},
        },
    }


def test_height_materializes_only_measured_neutral_support_layout():
    workspace = _workspace()
    workspace["height_layout_measurements"] = [_height_row()]

    candidate = propose_family_layouts(workspace, family="HEIGHT", seed=7, count=1)[0]

    assert candidate["metadata"]["status"] == "unqualified_proposal_starting_physical_validation"
    assert candidate["metadata"]["upper_support_side"] == "left"
    assert candidate["metadata"]["controller_status"].startswith("requires_measured")


def test_dist_requires_measured_plate_and_landing_supports():
    workspace = _workspace()
    row = _dist_row()
    del row["goal_supports"]["near_plate"]["support_surface_id"]
    workspace["dist_layout_measurements"] = [row]

    with pytest.raises(ValueError, match="surface identity"):
        propose_family_layouts(workspace, family="DIST", seed=7, count=1)


def test_dist_materializes_measured_perpendicular_bisector_layout():
    workspace = _workspace()
    workspace["dist_layout_measurements"] = [_dist_row()]

    candidate = propose_family_layouts(workspace, family="DIST", seed=7, count=1)[0]

    assert set(candidate["object_poses"]) == {"rubiks_cube", "bowl", "plate"}
    assert candidate["metadata"]["bowl_side"] == "right"


def test_missing_family_measurements_is_an_explicit_blocker():
    with pytest.raises(ValueError, match="HEIGHT requires a measured"):
        propose_family_layouts(_workspace(), family="HEIGHT", seed=7, count=1)


def test_measured_center_must_close_from_root_and_offset():
    workspace = _workspace()
    row = _height_row()
    row["scoring_centers_env_local_xyz_m"]["bowl"] = [.5, 0, .101]
    workspace["height_layout_measurements"] = [row]

    with pytest.raises(ValueError, match="does not close"):
        propose_family_layouts(workspace, family="HEIGHT", seed=7, count=1)


def test_duplicate_capture_layout_ids_are_rejected():
    workspace = _workspace()
    workspace["height_layout_measurements"] = [_height_row(), _height_row()]

    with pytest.raises(ValueError, match="duplicate layout IDs"):
        propose_family_layouts(workspace, family="HEIGHT", seed=7, count=1)


def test_cli_writes_explicitly_unqualified_proposals(tmp_path, monkeypatch):
    workspace = _workspace()
    workspace["height_layout_measurements"] = [_height_row()]
    source = tmp_path / "workspace.json"
    output = tmp_path / "height.json"
    source.write_text(json.dumps(workspace))
    monkeypatch.setattr(
        "sys.argv",
        ["height_dist_proposals.py", "--workspace-receipt", str(source), "--family", "HEIGHT",
         "--output", str(output), "--seed", "7", "--count", "1"],
    )

    main()

    value = json.loads(output.read_text())
    assert value["status"] == "proposed_unqualified"
    assert value["candidate_count"] == 1

    candidate = _load_selected_candidate(Namespace(proposal_file=output, candidate_id=value["candidates"][0]["candidate_id"]))
    assert candidate.family == "HEIGHT"
