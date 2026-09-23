import hashlib
import math
from types import SimpleNamespace

import pytest

from experiments.workshops.spatial_grounding_v1.native_geometry_measurements import (
    aabb_separation, camera_extrinsics, clearance_to_objects, native_articulation_path,
    project_collision_aabbs, robot_snapshot,
)


class Tensor:
    def __init__(self, value): self.value = value
    def __getitem__(self, _): return Tensor(self.value)
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.value


def test_robot_snapshot_uses_articulation_root_and_asset_bytes(tmp_path, monkeypatch):
    asset = tmp_path / "robot.usd"
    asset.write_bytes(b"native robot")
    robot = type("Robot", (), {})()
    robot.joint_names = ["joint-1"]
    robot.cfg = type("Cfg", (), {"spawn": type("Spawn", (), {"usd_path": str(asset)})()})()
    robot.data = type("Data", (), {
        "root_pos_w": [Tensor([1.2, 2.3, 3.4])], "root_quat_w": [Tensor([1, 0, 0, 0])],
        "joint_pos": [Tensor([.1])], "joint_vel": [Tensor([.2])],
        "body_names": ["actual_body"], "body_pos_w": [Tensor([[1.2, 2.3, 3.4]])],
        "body_quat_w": [Tensor([[1, 0, 0, 0]])],
    })()
    monkeypatch.setattr(
        "experiments.workshops.spatial_grounding_v1.robolab_measurements.articulation_body_frames",
        lambda _: {"bodies": [{"name": "actual_body"}]},
    )
    receipt = robot_snapshot({"robot": robot}, Tensor([1, 2, 3]))
    assert receipt["articulation_root_position_env_local_xyz_m"] == pytest.approx([.2, .3, .4])
    assert receipt["base_position_env_local_xyz_m"] == pytest.approx([.2, .3, .4])
    assert receipt["asset_usd"]["sha256"] == hashlib.sha256(b"native robot").hexdigest()
    assert receipt["body_frames"]["bodies"][0]["name"] == "actual_body"


def test_camera_extrinsics_and_unavailable_state_are_explicit():
    available = type("Camera", (), {"data": type("Data", (), {
        "pos_w": [Tensor([1, 2, 3])], "quat_w_world": [Tensor([1, 0, 0, 0])],
    })()})()
    unavailable = type("Camera", (), {"data": object()})()
    receipt = camera_extrinsics({"left": available, "right": unavailable}, ("left", "right"))
    assert receipt["cameras"]["left"]["position_world_xyz_m"] == [1.0, 2.0, 3.0]
    assert receipt["cameras"]["right"]["available"] is False


def test_conservative_aabb_overlap_is_not_labeled_collision():
    overlap = aabb_separation(
        {"minimum_xyz_m": [0, 0, 0], "maximum_xyz_m": [1, 1, 1]},
        {"minimum_xyz_m": [.5, .5, .5], "maximum_xyz_m": [2, 2, 2]},
    )
    assert overlap["aabb_overlap"] is True
    assert "not a measured physical collision" in overlap["caveat"]
    separated = aabb_separation(
        {"minimum_xyz_m": [0, 0, 0], "maximum_xyz_m": [1, 1, 1]},
        {"minimum_xyz_m": [2, 1, 1], "maximum_xyz_m": [3, 2, 2]},
    )
    assert separated["aabb_euclidean_separation_m"] == pytest.approx(1)


def test_projected_collision_bounds_use_matrix_and_native_body_pose():
    inventory = {
        "available": True,
        "rows": [{
            "body_name": "finger", "local_min_xyz_m": [0, 0, 0], "local_max_xyz_m": [1, 1, 1],
            "geometry_to_body_matrix_gf": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]],
        }],
    }
    frames = {"bodies": {"finger": {"position_world_xyz_m": [10, 0, 0], "quaternion_world_wxyz": [1, 0, 0, 0]}}}
    projected = project_collision_aabbs(inventory, frames)
    assert projected["body_world_aabbs"]["finger"]["minimum_xyz_m"] == [11, 2, 3]
    assert projected["body_world_aabbs"]["finger"]["maximum_xyz_m"] == [12, 3, 4]
    clearance = clearance_to_objects(projected, {
        "bowl": {"minimum_xyz_m": [14, 2, 3], "maximum_xyz_m": [15, 3, 4]},
    })
    assert clearance["separations"]["finger__bowl"]["aabb_euclidean_separation_m"] == pytest.approx(2)


@pytest.mark.parametrize("inventory,frames", [
    ({"available": True, "rows": [{"body_name": "missing", "local_min_xyz_m": [0, 0, 0], "local_max_xyz_m": [1, 1, 1], "geometry_to_body_matrix_gf": [[1, 0, 0, 0]] * 4}]}, {"bodies": {}}),
    ({"available": True, "rows": [{"body_name": "finger", "local_min_xyz_m": [0, 0, float("nan")], "local_max_xyz_m": [1, 1, 1], "geometry_to_body_matrix_gf": [[1, 0, 0, 0]] * 4}]}, {"bodies": {"finger": {"position_world_xyz_m": [0, 0, 0], "quaternion_world_wxyz": [1, 0, 0, 0]}}}),
])
def test_collision_projection_reports_unavailable_for_missing_or_nonfinite_inputs(inventory, frames):
    assert project_collision_aabbs(inventory, frames)["available"] is False


def test_only_resolved_native_articulation_path_is_accepted():
    robot = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"))
    assert native_articulation_path(robot) is None
    robot.root_physx_view = SimpleNamespace(prim_paths=["/World/envs/env_0/Robot"])
    assert native_articulation_path(robot) == "/World/envs/env_0/Robot"
    robot.root_physx_view.prim_paths = [robot.cfg.prim_path]
    assert native_articulation_path(robot) is None


def test_unspecified_camera_quaternion_convention_is_not_guessed():
    camera = SimpleNamespace(data=SimpleNamespace(pos_w=[[0, 0, 0]], quat_w=[[1, 0, 0, 0]]))
    assert camera_extrinsics({"camera": camera}, ("camera",))["cameras"]["camera"]["available"] is False


def test_rotation_projects_all_corners_and_empty_bounds_are_not_clear():
    inventory = {"available": True, "rows": [{
        "body_name": "finger", "local_min_xyz_m": [0, 0, 0], "local_max_xyz_m": [2, 1, 1],
        "geometry_to_body_matrix_gf": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
    }]}
    frames = {"bodies": {"finger": {
        "position_world_xyz_m": [4, 5, 6], "quaternion_world_wxyz": [math.sqrt(.5), 0, 0, math.sqrt(.5)],
    }}}
    projected = project_collision_aabbs(inventory, frames)
    assert projected["body_world_aabbs"]["finger"]["minimum_xyz_m"] == pytest.approx([3, 6, 6])
    assert projected["body_world_aabbs"]["finger"]["maximum_xyz_m"] == pytest.approx([4, 8, 7])
    assert clearance_to_objects(projected, {})["available"] is False
    assert project_collision_aabbs({"available": True, "rows": []}, frames)["available"] is False
    with pytest.raises(ValueError, match="malformed"):
        aabb_separation({"minimum_xyz_m": [math.nan, 0, 0], "maximum_xyz_m": [1, 1, 1]},
                        {"minimum_xyz_m": [2, 2, 2], "maximum_xyz_m": [3, 3, 3]})
