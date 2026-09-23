from types import SimpleNamespace

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.lat_workspace_capture import render_only_warmup
from experiments.workshops.spatial_grounding_v1.robolab_measurements import articulation_body_frames


class Array:
    def __init__(self, data):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


def test_body_frame_measurement_preserves_names_and_rejects_malformed_geometry():
    data = SimpleNamespace(
        body_names=["base_link", "finger"],
        body_pos_w=[Array(np.asarray([[0.3, 0, 0.4], [0.3, 0, 0.25]]))],
        body_quat_w=[Array(np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]]))],
    )
    receipt = articulation_body_frames(data)
    assert receipt["bodies"]["base_link"]["position_world_xyz_m"] == [0.3, 0, 0.4]
    assert receipt["bodies"]["finger"]["position_world_xyz_m"] == [0.3, 0, 0.25]
    assert "not inferred fingertip" in receipt["claim_boundary"]
    data.body_names = ["base_link", "base_link"]
    with pytest.raises(ValueError, match="inventory"):
        articulation_body_frames(data)


def test_render_diagnostic_refreshes_cameras_without_physics_and_retains_video(tmp_path):
    renders = []
    updates = []
    cameras = ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")
    obs = {"image_obs": {name: [Array(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))] for name in cameras}}
    env = SimpleNamespace(
        sim=SimpleNamespace(current_time=0.0, render=lambda: renders.append(True)),
        scene={name: SimpleNamespace(update=lambda dt, **kwargs: updates.append((dt, kwargs))) for name in cameras},
        observation_manager=SimpleNamespace(compute=lambda: obs),
    )
    final, receipt = render_only_warmup(env, obs, 3, tmp_path / "warmup")
    assert final is obs
    assert len(renders) == 3 and len(updates) == 9
    assert all(dt == 0 and kwargs == {"force_recompute": True} for dt, kwargs in updates)
    assert receipt["physics_actions"] == 0
    assert receipt["viewport_video"]["frame_count"] == 4
    assert all(row["sim_time_s"] == 0 for row in receipt["snapshots"])
    with pytest.raises(FileExistsError):
        render_only_warmup(env, obs, 3, tmp_path / "warmup")


def test_render_diagnostic_rejects_physical_time_advance(tmp_path):
    sim = SimpleNamespace(current_time=0.0)
    sim.render = lambda: setattr(sim, "current_time", 0.1)
    cameras = ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")
    obs = {"image_obs": {name: [Array(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))] for name in cameras}}
    env = SimpleNamespace(
        sim=sim, scene={name: SimpleNamespace(update=lambda *a, **kw: None) for name in cameras},
        observation_manager=SimpleNamespace(compute=lambda: obs),
    )
    with pytest.raises(RuntimeError, match="advanced physical"):
        render_only_warmup(env, obs, 1, tmp_path / "warmup")
