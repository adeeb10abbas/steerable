import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.gripper_geometry_capture import capture_gripper_motion, finger_geometry


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return Tensor(self.value[index])

    def __sub__(self, other):
        return Tensor(self.value - other.value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def environment(monkeypatch, *, terminate_at=None):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float32=np.float32, as_tensor=lambda value, **kwargs: np.asarray(value),
    ))
    data = SimpleNamespace(
        body_names=["base_link", "left_inner_finger", "right_inner_finger"],
        body_pos_w=Tensor([[[0.3, 0, 0.4], [0.3, -0.04, 0.25], [0.3, 0.04, 0.25]]]),
        body_quat_w=Tensor([[[1, 0, 0, 0]] * 3]),
        root_pos_w=Tensor([[0, 0, 0]]), root_quat_w=Tensor([[1, 0, 0, 0]]),
        joint_pos=Tensor([[0.0]]),
    )
    observation = {"image_obs": {"over_shoulder_left_camera": Tensor(np.arange(192, dtype=np.uint8).reshape(1, 8, 8, 3))}}
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data, joint_names=["finger_joint"])},
        sim=SimpleNamespace(current_time=0.0), step_dt=1 / 15, device="cpu", actions=[],
    )
    for name in ("rubiks_cube", "bowl", "banana"):
        env.scene[name] = SimpleNamespace(data=SimpleNamespace(root_pos_w=Tensor([[0.5, 0, 0.1]])))
    def step(action):
        env.actions.append(action.copy())
        env.sim.current_time += env.step_dt
        data.joint_pos.value[0, 0] = action[0, 7]
        return observation, None, [len(env.actions) == terminate_at], [False], {}
    env.step = step
    return env, observation


def test_empty_gripper_probe_records_exact_static_commands_and_complete_video(monkeypatch, tmp_path):
    env, observation = environment(monkeypatch)
    geometry = [{"body": "left_inner_finger", "test_only": True}]
    _, receipt = capture_gripper_motion(env, observation, tmp_path / "probe", geometry=geometry)
    assert receipt["observed_controller_actions"] == receipt["issued_controller_commands"] == 60
    assert json.loads((tmp_path / "probe/geometry.json").read_text())["finger_geometry"] == geometry
    assert json.loads((tmp_path / "probe/receipt.json").read_text())["finger_geometry"] == geometry
    assert receipt["viewport_video"]["frame_count"] == len(receipt["records"]) == 61
    assert receipt["model_requests"] == receipt["behavioral_episodes"] == 0
    assert receipt["records"][30]["joint_position_rad"] == pytest.approx([0.785398])
    assert receipt["records"][60]["joint_position_rad"] == [0]
    for index, action in enumerate(env.actions):
        np.testing.assert_allclose(action[0, :7], [0.3, 0, 0.4, 1, 0, 0, 0])
        assert action[0, 7] == pytest.approx(0.785398 if index < 30 else 0)
    assert len(list((tmp_path / "probe").glob("command-*.json"))) == 60


def test_interrupted_gripper_probe_keeps_partial_recording(monkeypatch, tmp_path):
    env, observation = environment(monkeypatch, terminate_at=5)
    with pytest.raises(RuntimeError, match="action 5"):
        capture_gripper_motion(env, observation, tmp_path / "probe", geometry=[])
    receipt = json.loads((tmp_path / "probe/receipt.json").read_text())
    assert receipt["status"] == "infrastructure_invalid_gripper_probe"
    assert receipt["observed_controller_actions"] == receipt["issued_controller_commands"] == 5
    assert receipt["viewport_video"]["frame_count"] == 6


def test_failed_step_distinguishes_issued_commands_from_observed_actions(monkeypatch, tmp_path):
    env, observation = environment(monkeypatch)
    def fail_step(action):
        raise RuntimeError("simulator unavailable")
    env.step = fail_step
    with pytest.raises(RuntimeError, match="simulator unavailable"):
        capture_gripper_motion(env, observation, tmp_path / "probe", geometry=[])
    receipt = json.loads((tmp_path / "probe/receipt.json").read_text())
    assert receipt["issued_controller_commands"] == 1
    assert receipt["observed_controller_actions"] == 0
    assert receipt["viewport_video"]["frame_count"] == 1


def test_gripper_probe_retains_and_rejects_wrong_physical_time(monkeypatch, tmp_path):
    env, observation = environment(monkeypatch)
    env.step_dt = 1 / 20
    original_step = env.step
    def wrong_time_step(action):
        result = original_step(action)
        env.sim.current_time += 0.01
        return result
    env.step = wrong_time_step
    with pytest.raises(RuntimeError, match="physical-time mismatch"):
        capture_gripper_motion(env, observation, tmp_path / "probe", geometry=[])
    receipt = json.loads((tmp_path / "probe/receipt.json").read_text())
    assert receipt["status"] == "infrastructure_invalid_gripper_probe"
    assert receipt["issued_controller_commands"] == receipt["observed_controller_actions"] == 1
    assert receipt["viewport_video"]["frame_count"] == 2


def test_finger_bounds_are_measured_relative_to_the_actual_body():
    pytest.importorskip("pxr", reason="native USD geometry test requires the optional USD reader")
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    root = "/World/envs/env_0/robot/Gripper/Robotiq_2F_85"
    body = UsdGeom.Xform.Define(stage, root + "/left_inner_finger")
    body.AddTranslateOp().Set((0.3, 0.1, 0.4))
    shape = UsdGeom.Cube.Define(stage, str(body.GetPath()) + "/pad")
    shape.GetSizeAttr().Set(0.02)
    shape.AddTranslateOp().Set((0, 0, 0.04))
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    rows = finger_geometry(stage, ["left_inner_finger"])
    assert len(rows) == 1 and rows[0]["collision_geometry"]
    assert rows[0]["body"] == "left_inner_finger"
    assert rows[0]["body_local_bounds_center_m"] == pytest.approx([0, 0, 0.04])
