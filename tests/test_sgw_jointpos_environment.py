import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import robolab_jointpos_environment as jointpos
from experiments.workshops.spatial_grounding_v1.adapters import AdapterError, NanoPolicyAdapter, ProductionAdapter
from experiments.workshops.spatial_grounding_v1.contract import Cell, load_release
from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate
from experiments.workshops.spatial_grounding_v1.policy_observations import CAMERAS
from experiments.workshops.spatial_grounding_v1.recorder import AttemptRecorder
from experiments.workshops.spatial_grounding_v1.worker import _canonical_outcome, _load_scorer
from tests.test_sgw_contract import make_release
from tests.test_sgw_production_e2e import _transport


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def clone(self):
        return Tensor(self.value.copy())

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def tolist(self):
        return self.value.tolist()

    def to(self, _device):
        return self

    def zero_(self):
        self.value.fill(0)

    @property
    def shape(self):
        return self.value.shape


def candidate():
    return FixtureCandidate.from_json(candidate_json())


def candidate_json():
    return {
        "candidate_id": "LAT-SYNTHETIC-NATIVE-BOUNDARY", "family": "LAT", "seed": 1,
        "task_asset": "rubiks_cube_banana_bowl.usda", "asset_manifest_sha256": "a" * 64,
        "object_poses": {
            "rubiks_cube": {"position_m": [.3, 0, .08], "quaternion_wxyz": [1, 0, 0, 0]},
            "bowl": {"position_m": [.5, 0, .1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "metadata": {"scoring_center_offsets_root_local_m": {
            "rubiks_cube": [0, 0, .02], "bowl": [0, 0, 0],
        }},
    }


def install_native_boundary(monkeypatch, *, tensor_module=None):
    """Substitute Isaac/RoboLab tensors and physics, not SGW measurement/scoring."""
    ns = types.SimpleNamespace
    class JointPositionActionCfg:
        asset_name = "robot"
        use_default_offset = False
    class Scene(dict):
        env_origins = Tensor([[1., 2., 0.]])
    class Native:
        step_dt = 1 / 15
        device = "cpu"

        def __init__(self):
            self.index, self.closed, self.renders = 0, 0, 0
            self.actions = []
            self.drift = 0.
            self.angular_speed = 0.
            self.support_force = 3.
            self.terminal_at = None
            self.episode_length_buf = Tensor([99])
            self.cfg = ns(actions=ns(body=JointPositionActionCfg()))
            self.action_manager = ns(total_action_dim=8)
            self.scene = Scene({
                name: ns(update=lambda *_a, **_k: None) for name in CAMERAS
            })
            self.scene["robot"] = ns(data=ns(
                body_names=["base_link"],
                body_pos_w=Tensor([[[1., 2., .3]]]),
                body_quat_w=Tensor([[[1., 0., 0., 0.]]]),
            ))
            self.sim = ns(current_time=0., render=self.render)
            self.observation_manager = ns(compute=self.observation)
            self.update_objects()

        def center(self, name):
            if name == "bowl":
                return np.array([.5, 0, .1])
            return np.array([.3 + self.drift, .1 if self.index >= 4 else 0,
                             .14 if 1 <= self.index <= 3 else .1])

        def root(self, name):
            return self.center(name) - ([0, 0, .02] if name == "rubiks_cube" else np.zeros(3))

        def update_objects(self):
            for name in ("rubiks_cube", "bowl"):
                self.scene[name] = ns(data=ns(
                    root_com_pos_w=Tensor([self.center(name) + [1., 2., 0.] - [.02, 0, 0]]),
                    root_com_vel_w=Tensor([[0, 0, 0, 0, 0, self.angular_speed]]),
                ))

        def render(self):
            self.renders += 1

        def observation(self):
            frame = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(1, 16, 16, 3)
            return {
                "image_obs": {name: Tensor(frame + number) for number, name in enumerate(CAMERAS)},
                "proprio_obs": {"arm_joint_pos": Tensor(np.zeros((1, 7), dtype=np.float32)),
                               "gripper_pos": Tensor(np.ones((1, 1), dtype=np.float32))},
                "oracle_object_positions": {"rubiks_cube": self.center("rubiks_cube")},
            }

        def reset(self):
            assert self.episode_length_buf.value[0] == 0
            self.index = 0
            self.update_objects()
            return self.observation(), {}

        def step(self, action):
            assert action.shape == (1, 8)
            self.actions.append(action.numpy().copy())
            self.index += 1
            self.sim.current_time += self.step_dt
            self.update_objects()
            return self.observation(), 0., [self.index == self.terminal_at], [False], {}

        def close(self):
            self.closed += 1

    def world(env):
        return ns(
            get_pose=lambda name, env_id: (Tensor(env.root(name)), Tensor([1., 0, 0, 0])),
            get_bbox=lambda name, env_id: (None, env.center(name)),
        )
    torch = types.ModuleType("torch")
    torch.Tensor, torch.float32 = Tensor, np.float32
    torch.from_numpy = Tensor
    torch.as_tensor = lambda value, dtype: Tensor(np.asarray(value, dtype=dtype))
    monkeypatch.setitem(sys.modules, "torch", tensor_module if tensor_module is not None else torch)
    modules = {
        "isaaclab.envs.mdp": {"JointPositionActionCfg": JointPositionActionCfg},
        "robolab.core.world.world_state": {"get_world": world},
        "robolab.core.task.conditionals": {"object_grabbed": lambda env, **_: 1 <= env.index <= 3},
        "robolab.core.sensors.contact_sensor_utils": {"get_contact_sensors": lambda scene: scene["sensors"]},
    }
    for name, attrs in modules.items():
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent not in sys.modules:
                module = types.ModuleType(parent)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, parent, module)
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(sys.modules[name.rsplit(".", 1)[0]], name.rsplit(".", 1)[1], module, raising=False)

    def make_native():
        env = Native()
        class Force:
            @property
            def force_matrix_w(self):
                return Tensor([[[[0, 0, env.support_force]]]])
        env.scene["sensors"] = {
            f"rubiks_cube__{name}": ns(data=Force()) for name in ("banana", "bowl", "table")
        }
        return env
    return make_native


def test_measured_jointpos_boundary_preserves_support_com_transport_and_resets(monkeypatch, tmp_path):
    native = install_native_boundary(monkeypatch)()
    env = jointpos.JointPositionEnvironment(native, candidate=candidate(), cell_id="test", evidence_root=tmp_path)
    reset = env.reset()
    assert native.renders == 120 and not native.actions
    assert reset.receipt["render_only_warmup"]["viewport_video"]["frame_count"] == 121
    assert reset.snapshot["cube_xyz_m"] == (.3, 0., .1)
    assert "oracle_object_positions" not in env.policy_observation()
    native.support_force, native.angular_speed = 0., .5
    env.step(np.zeros(8))
    assert env.snapshot()["supported"] is False
    assert env.snapshot()["linear_speed_m_s"] == pytest.approx(.01)
    native.drift = .004
    with pytest.raises(AdapterError, match="3 mm"):
        env.reset()
    env.close()
    env.close()
    assert native.closed == 1


def test_jointpos_production_records_all_450_actions_with_real_scorer(monkeypatch, tmp_path):
    factory = install_native_boundary(monkeypatch)
    release = load_release(make_release(tmp_path))
    cell = Cell({**release.partition("N3", "LAT", "P")[0].row, "sampling_seed": 1, "physical_goal_sign": 1, "form": "D"})
    native = factory()
    recorder = AttemptRecorder(release, cell, "attempt-001")
    recorder.begin()
    resets = []
    adapter = ProductionAdapter(
        NanoPolicyAdapter, transport=_transport, transport_factory=lambda **_: _transport,
        runtime_handle=types.SimpleNamespace(reset=lambda: resets.append(True), close=lambda: None),
        environment_factory=lambda cell, evidence_root: jointpos.JointPositionEnvironment(
            native, candidate=candidate(), cell_id=cell.cell_id, evidence_root=evidence_root,
        ),
    )
    reset = adapter.reset(cell, recorder)
    outcome = _canonical_outcome(adapter.run_episode(cell, recorder, reset), cell, _load_scorer())
    assert outcome["status"] == "valid_success"
    assert outcome["terminal_step"] == len(native.actions) == 450
    assert outcome["viewport_artifact"]["fps"] == 15
    assert len(list((recorder.path / "observations").glob("*.npy"))) == 451
    assert len(resets) == 1
    stored = json.loads((recorder.path / "states/reset.json").read_text())
    assert stored["reset"]["physical_reset_receipt"]["native_action_mode"] == "joint_position"
    adapter.close()
    assert native.closed == 1


def test_jointpos_binding_checks_actual_assets_candidate_and_release(monkeypatch, tmp_path):
    scene = tmp_path / "scene.usda"
    scene.write_text("measured")
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"scene": {"path": str(scene), "bytes": scene.stat().st_size,
                                           "sha256": jointpos._sha256(scene)}, "assets": []}))
    proposal = tmp_path / "candidate.json"
    proposal.write_text(json.dumps({**candidate_json(), "asset_manifest_sha256": jointpos._sha256(assets)}))
    row = {"cell_id": "cell-1", "status": "RELEASED", "family": "LAT", "layout_id": "LAT-P01",
           "fixture_sha256": "b" * 64, "prompt": "static", "prompt_sha256": hashlib.sha256(b"static").hexdigest()}
    binding_file = tmp_path / "binding.json"
    binding_file.write_text(json.dumps({
        "source_root": str(tmp_path), "robolab_root": str(tmp_path),
        "source_commit": "c" * 40, "robolab_commit": jointpos.D1_ROBOLAB_CLIENT_COMMIT,
        "assets_manifest": str(assets), "assets_manifest_sha256": jointpos._sha256(assets),
        "cells": {"cell-1": {**row, "scene_seed": 0, "candidate_path": str(proposal),
                             "candidate_file_sha256": jointpos._sha256(proposal)}},
    }))
    monkeypatch.setenv("SGW01_ENV_BINDING", str(binding_file))
    monkeypatch.setenv("SGW01_ENV_BINDING_SHA256", jointpos._sha256(binding_file))
    monkeypatch.setattr(jointpos, "_verify_git_checkout", lambda *args: None)
    binding = jointpos.JointPositionBinding.load()
    assert binding.cell(row)[0]["scene_seed"] == 0
    with pytest.raises(AdapterError, match="prompt"):
        binding.cell({**row, "prompt": "switched"})
    with pytest.raises(AdapterError, match="released"):
        binding.cell({**row, "status": "PLANNED"})
    proposal.write_text("{}")
    with pytest.raises(AdapterError, match="candidate file"):
        binding.cell(row)
    scene.write_text("changed!")
    with pytest.raises(AdapterError, match="asset payload"):
        jointpos.JointPositionBinding.load()


def test_production_runtime_is_closed_even_if_simulator_cleanup_fails():
    calls = []
    adapter = ProductionAdapter(
        NanoPolicyAdapter, transport=_transport, transport_factory=lambda **_: _transport,
        runtime_handle=types.SimpleNamespace(close=lambda: calls.append("runtime_closed")),
    )
    def failed_close():
        raise OSError("simulator cleanup failed")
    adapter.environment = types.SimpleNamespace(close=failed_close)
    with pytest.raises(OSError, match="cleanup failed"):
        adapter.close()
    assert calls == ["runtime_closed"] and adapter.environment is None
