from copy import deepcopy
import math
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.lat_workspace_capture import _rotate_wxyz
from experiments.workshops.spatial_grounding_v1.paper_informed_geometry import from_files, translation_preview


def object_row(center, *, quaternion=(1, 0, 0, 0), offset=(0, 0, 0), half=0.03):
    rotated = _rotate_wxyz(list(quaternion), list(offset))
    return {
        "geometric_center_env_local_xyz_m": list(center),
        "root_position_env_local_xyz_m": [center[i] - rotated[i] for i in range(3)],
        "geometric_center_offset_root_local_xyz_m": list(offset),
        "root_quaternion_world_wxyz": list(quaternion),
        "bbox_env_local_min_xyz_m": [x - half for x in center],
        "bbox_env_local_max_xyz_m": [x + half for x in center],
    }


@pytest.fixture
def objects():
    return {
        "rubiks_cube": object_row([0.30, 0.10, 0.15], quaternion=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
                                 offset=[0.04, -0.02, 0.03]),
        "bowl": object_row([0.40, 0.10, 0.15]),
        "table": object_row([0.5, 0, -0.4], half=0.5),
    }


def test_center_target_is_not_authoring_x_or_actor_root_radius(objects):
    original = deepcopy(objects)
    robot = [0.10, -0.10, 0.05]
    report = translation_preview(objects, robot)
    center = report["proposed_geometric_centers_env_local_xyz_m"]["rubiks_cube"]
    assert math.dist(center, robot) == pytest.approx(0.50, abs=1e-12)
    assert center[0] != pytest.approx(0.50)
    assert report["proposed_cube_actor_root_distance_m"] != pytest.approx(0.50)
    assert report["translation_env_local_xyz_m"][2] == 0
    assert report["translated_actor_poses"]["rubiks_cube"]["quaternion_wxyz"] == objects["rubiks_cube"]["root_quaternion_world_wxyz"]
    before = [objects["bowl"]["geometric_center_env_local_xyz_m"][i] -
              objects["rubiks_cube"]["geometric_center_env_local_xyz_m"][i] for i in range(3)]
    after = [report["proposed_geometric_centers_env_local_xyz_m"]["bowl"][i] - center[i] for i in range(3)]
    assert after == pytest.approx(before, abs=1e-15)
    assert objects == original
    assert "table" not in report["translated_actor_poses"]
    assert report["release_permitted"] is report["new_scene_materialized"] is False
    assert report["new_registered_candidates"] == report["new_model_requests"] == report["new_simulator_runs"] == 0


@pytest.mark.parametrize("center,match", [
    ([0.1, 0.1, 0.51], "height"),
    ([0, 0, 0.1], "direction"),
])
def test_unavailable_horizontal_solution_is_not_silently_guessed(objects, center, match):
    objects["rubiks_cube"] = object_row(center)
    with pytest.raises(ValueError, match=match):
        translation_preview(objects, [0, 0, 0])


@pytest.mark.parametrize("root", [[True, 0, 0], [math.nan, 0, 0], [0, 0], None])
def test_robot_root_must_be_explicit_finite_measurement(objects, root):
    with pytest.raises(ValueError, match="measured robot root"):
        translation_preview(objects, root)


def test_bad_measured_offset_fails_closed(objects):
    objects["rubiks_cube"]["geometric_center_offset_root_local_xyz_m"][0] += 0.01
    with pytest.raises(ValueError, match="inconsistent"):
        translation_preview(objects, [0, 0, 0])


def test_outside_table_is_reported_not_promoted_to_qualification(objects):
    objects["bowl"] = object_row([1.5, 0, 0.15])
    value = translation_preview(objects, [0, 0, 0])
    assert "bowl" in value["objects_extending_outside_measured_table"]
    assert value["table_edge_clearance_xy_m"]["bowl"] < 0
    assert value["release_permitted"] is False


def test_exact_measured_workspace_can_drive_a_nonmaterialized_calculation():
    path = Path("artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json")
    value = from_files(path, path)
    assert value["proposed_cube_geometric_center_distance_m"] == pytest.approx(0.50, abs=1e-12)
    assert value["input_bindings"]["robot_workspace"]["sha256"] == "1ef79d38f5004c40b2c5161a8c42aca2298dcbbf60d67ff3e48a19642c567e0e"
    assert value["new_scene_materialized"] is False


def test_native_family_capture_cannot_guess_a_missing_robot_binding():
    base = Path("artifacts/workshops/spatial_grounding_v1/infrastructure")
    with pytest.raises(ValueError, match="overlay-to-workspace"):
        from_files(base / "a40-20260922r-workspace.json", base / "native-metric-20260923cg/candidate_capture.json")
