from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1 import paper_engineering as engineering
from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate
from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
from experiments.workshops.spatial_grounding_v1.lat_workspace_capture import _rotate_wxyz
from experiments.workshops.spatial_grounding_v1.paper_informed_geometry import translation_preview

BASE = Path("artifacts/workshops/spatial_grounding_v1")
WORKSPACE = BASE / "infrastructure/a40-20260922z-workspace.json"


@pytest.fixture
def measured_translation(tmp_path):
    workspace = json.loads(WORKSPACE.read_bytes())
    source = next(row for row in json.loads((BASE / "proposals/lat-20260922.json").read_bytes())["candidates"]
                  if row["candidate_id"] == "LAT-CANDIDATE-057")
    objects = deepcopy(workspace["objects"])
    for name, pose in source["object_poses"].items():
        row = objects[name]
        rotated = _rotate_wxyz(pose["quaternion_wxyz"], row["geometric_center_offset_root_local_xyz_m"])
        center = [pose["position_m"][i] + rotated[i] for i in range(3)]
        delta = [center[i] - row["geometric_center_env_local_xyz_m"][i] for i in range(3)]
        for key in ("bbox_env_local_min_xyz_m", "bbox_env_local_max_xyz_m", "geometric_center_env_local_xyz_m"):
            row[key] = [row[key][i] + delta[i] for i in range(3)]
        row["root_position_env_local_xyz_m"] = pose["position_m"]
        row["root_quaternion_world_wxyz"] = pose["quaternion_wxyz"]
    preview = translation_preview(objects, workspace["robot"]["base_position_env_local_xyz_m"])
    delta = preview["translation_env_local_xyz_m"]
    for name, row in objects.items():
        if name == "table":
            continue
        row["root_position_env_local_xyz_m"] = preview["translated_actor_poses"][name]["position_m"]
        for key in ("bbox_env_local_min_xyz_m", "bbox_env_local_max_xyz_m", "geometric_center_env_local_xyz_m"):
            row[key] = [row[key][i] + delta[i] for i in range(3)]
    base_scene = tmp_path / "synthetic-base.usda"
    base_scene.write_text('#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n')
    manifest_path = engineering.write_overlay(
        root=tmp_path / "translated", base_scene=base_scene, workspace_path=WORKSPACE,
        poses=preview["translated_actor_poses"], phase="translated",
    )
    manifest = json.loads(manifest_path.read_bytes())
    capture = {
        "family": "LAT", "objects": objects, "robot_snapshot": workspace["robot"],
        "environment_origin_world_xyz_m": workspace["environment_origin_world_xyz_m"],
        "model_request_count": 0, "behavioral_episode_count": 0,
        "overlay_manifest": engineering.record(manifest_path), "overlay_manifest_sha256": manifest["manifest_sha256"],
        "asset_manifest_sha256": workspace["asset_manifest_sha256"], "robolab_commit": workspace["robolab_commit"],
        "test_only": True,
    }
    capture["receipt_sha256"] = workspace_digest(capture)
    path = manifest_path.parent / "capture.json"
    path.write_text(json.dumps(capture, indent=2) + "\n")
    return source, preview, path, manifest_path


def test_measured_layout_keeps_goal_displacements_and_calibrated_recipe(measured_translation):
    source, preview, capture, manifest = measured_translation
    candidate, screen = engineering.materialize_lat_candidate(
        capture_path=capture, manifest_path=manifest, source_candidate=source,
    )
    value = FixtureCandidate.from_json(candidate)
    assert value.candidate_id == engineering.ATTEMPT
    assert value.family == "LAT"
    assert abs(value.relation_m()) < 1e-12
    assert screen["reasons"] == []
    assert value.metadata["engineering_geometry_measurements"] is True
    assert math.dist(value.scoring_poses()["rubiks_cube"].position_m, [0, 0, 0]) == pytest.approx(.5, abs=1e-12)
    for goal in ("positive", "negative"):
        old = source["metadata"]["abs_ik_waypoints"][goal]
        new = candidate["metadata"]["abs_ik_waypoints"][goal]
        assert sum(point["hold_steps"] for point in new) == 450
        for axis in (0, 1):
            assert new[-1]["position_world_xyz_m"][axis] - new[0]["position_world_xyz_m"][axis] == pytest.approx(
                old[-1]["position_world_xyz_m"][axis] - old[0]["position_world_xyz_m"][axis], abs=1e-12)
    usd = (manifest.parent / "scene.usda").read_text()
    assert 'over "robot"' not in usd and 'over "table"' not in usd and "camera" not in usd
    assert 'over "banana"' in usd
    assert candidate["object_poses"]["rubiks_cube"] == preview["translated_actor_poses"]["rubiks_cube"]


def test_engineering_native_scene_rejects_unbound_task_poses(measured_translation):
    source, _, capture, manifest = measured_translation
    candidate, _ = engineering.materialize_lat_candidate(
        capture_path=capture, manifest_path=manifest, source_candidate=source,
    )
    scene = candidate["metadata"]["native_scene"]
    engineering.validate_native_scene(scene, candidate["object_poses"])
    bad = deepcopy(candidate["object_poses"])
    bad["bowl"]["position_m"][0] += .02
    with pytest.raises(ValueError, match="task poses differ"):
        engineering.validate_native_scene(scene, bad)
    with capture.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="input bytes differ"):
        engineering.validate_native_scene(scene)


def test_overlay_refuses_to_move_robot_camera_or_table(tmp_path):
    for name in ("robot", "wrist_cam", "table"):
        with pytest.raises(ValueError, match="only measured non-table"):
            engineering.write_overlay(
                root=tmp_path / name, base_scene=tmp_path / "base.usda", workspace_path=WORKSPACE,
                poses={"rubiks_cube": {}, "bowl": {}, name: {}}, phase="translated",
            )


def test_measured_footprint_rejection_is_retained(measured_translation):
    source, _, path, manifest = measured_translation
    capture = json.loads(path.read_bytes())
    capture["objects"]["table"]["bbox_env_local_max_xyz_m"][0] = .5
    capture["receipt_sha256"] = workspace_digest(capture)
    path.write_text(json.dumps(capture) + "\n")
    candidate, screen = engineering.materialize_lat_candidate(
        capture_path=path, manifest_path=manifest, source_candidate=source,
    )
    assert screen["status"] == "measured_footprint_rejection"
    assert candidate["metadata"]["geometric_screen_status"] == "rejected"
    assert screen["release_permitted"] is False


def test_registration_does_not_authorize_policy_or_extra_trials(tmp_path):
    path = tmp_path / "registration.json"
    for changes in ({"model_requests": 1}, {"scripted_trials": 7}, {"maximum_new_engineering_layouts": 2}):
        value = {
            "schema_version": engineering.SCHEMA, "attempt_id": engineering.ATTEMPT,
            "source_candidate_id": "LAT-CANDIDATE-057", "robolab_commit": engineering.ROBOLAB_COMMIT,
            "scripted_trials": 6, "actions_per_trial": 450, "model_requests": 0,
            "release_permitted": False, "maximum_new_engineering_layouts": 1, **changes,
        }
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="bounded zero-model"):
            engineering.load_registration(path)


def test_native_configuration_comparison_reports_measurements_and_missing_cameras():
    state = {
        "robot_snapshot": {
            "asset_usd": {"available": True, "sha256": "1" * 64},
            "joint_names": ["joint"], "joint_position_rad": [0.1],
            "base_position_env_local_xyz_m": [0, 0, 0], "base_quaternion_world_wxyz": [1, 0, 0, 0],
        },
        "camera_extrinsics": {"cameras": {name: {
            "available": True, "position_world_xyz_m": [1, 0, 0], "quaternion_world_wxyz": [1, 0, 0, 0],
        } for name in ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")}},
    }
    changed = deepcopy(state)
    changed["camera_extrinsics"]["cameras"]["wrist_cam"]["position_world_xyz_m"][0] += .001
    result = engineering.compare_native_configuration(state, changed)
    assert result["pose_errors"]["wrist_cam"]["max_component_position_error_m"] == pytest.approx(.001)
    changed["camera_extrinsics"]["cameras"]["wrist_cam"]["position_world_xyz_m"][0] += .003
    with pytest.raises(ValueError, match="reset tolerances"):
        engineering.compare_native_configuration(state, changed)
    changed = deepcopy(state)
    changed["camera_extrinsics"]["cameras"]["wrist_cam"] = {"available": False, "reason": "missing world quaternion"}
    with pytest.raises(ValueError, match="extrinsics unavailable: wrist_cam"):
        engineering.compare_native_configuration(state, changed)
