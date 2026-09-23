import hashlib

import pytest

from experiments.workshops.spatial_grounding_v1.native_geometry_measurements import (
    aabb_separation, camera_extrinsics, robot_snapshot,
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
