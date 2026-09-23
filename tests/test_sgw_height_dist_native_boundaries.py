import json
import os
import sys
import types

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate
from experiments.workshops.spatial_grounding_v1.robolab_height_dist_qualification import (
    create_bridge,
    create_controller,
)
from experiments.workshops.spatial_grounding_v1.robolab_lat_qualification import RoboLabLatEnvironment
from experiments.workshops.spatial_grounding_v1.simulator_bridge import SimulatorBridgeError
from experiments.workshops.spatial_grounding_v1.task_definitions import build_task_definition
import experiments.workshops.spatial_grounding_v1.robolab_height_dist_qualification as family_bridge
import experiments.workshops.spatial_grounding_v1.model_blind_qualification as qualification


def _candidate():
    return FixtureCandidate.from_json({
        "candidate_id": "HEIGHT-CANDIDATE-001",
        "family": "HEIGHT",
        "seed": 7,
        "task_asset": "measured.usda",
        "asset_manifest_sha256": "a" * 64,
        "object_poses": {
            "rubiks_cube": {"position_m": [.3, 0, .1], "quaternion_wxyz": [1, 0, 0, 0]},
            "bowl": {"position_m": [.5, 0, .1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "metadata": {
            "scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]},
            "native_scene": {
                "asset": "measured.usda",
                "object_names": ["rubiks_cube", "bowl", "table", "upper_platform", "lower_platform"],
            },
            "goal_supports": {
                "higher": {"contact_sensor_id": "rubiks_cube__upper", "cube_center_env_local_xyz_m": [.4, 0, .21]},
                "lower": {"contact_sensor_id": "rubiks_cube__lower", "cube_center_env_local_xyz_m": [.4, 0, .04]},
            },
        },
    })


def test_family_support_sensor_selection_uses_declared_surfaces_not_table():
    environment = object.__new__(RoboLabLatEnvironment)
    environment._candidate = _candidate()

    assert environment._support_sensor_names() == ("rubiks_cube__upper", "rubiks_cube__lower")


@pytest.mark.parametrize("tensor_corners", (False, True))
def test_family_snapshot_accepts_native_single_env_bbox_vectors(monkeypatch, tmp_path, tensor_corners):
    import itertools
    import torch
    from pxr import Gf
    from experiments.workshops.spatial_grounding_v1 import robolab_lat_qualification

    corners = [Gf.Vec3d(*point) for point in itertools.product((0, 1), (0, 2), (0, 3))]
    if tensor_corners:
        corners = torch.tensor([list(point) for point in corners])

    class World:
        def get_pose(self, name, env_id):
            assert env_id == 0
            return torch.zeros(3), torch.tensor([1., 0, 0, 0])

        def get_bbox(self, name, env_id):
            assert env_id == 0
            return corners, np.array([.5, 1., 1.5])

    class Scene:
        env_origins = [torch.zeros(3)]

        def __getitem__(self, name):
            return types.SimpleNamespace(data=types.SimpleNamespace(
                root_com_pos_w=torch.zeros(1, 3), root_com_vel_w=torch.zeros(1, 6),
            ))

    sensors = {name: types.SimpleNamespace(data=types.SimpleNamespace(
        force_matrix_w=torch.tensor([[[[0., 0, 2.]]]]),
    )) for name in ("rubiks_cube__upper", "rubiks_cube__lower")}
    for name, attrs in {
        "robolab.core.task.conditionals": {"object_grabbed": lambda *args, **kwargs: False},
        "robolab.core.sensors.contact_sensor_utils": {"get_contact_sensors": lambda scene: sensors},
        "robolab.core.world.world_state": {"get_world": lambda env: World()},
    }.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(robolab_lat_qualification, "articulation_body_frames", lambda data: {})
    env = RoboLabLatEnvironment(
        types.SimpleNamespace(scene=Scene(), step_dt=.1), _candidate(), tmp_path,
    )
    snapshot = env._snapshot()
    assert snapshot.objects["rubiks_cube"].supported
    assert set(snapshot.context_measurements) == {"table", "upper_platform", "lower_platform"}
    for row in snapshot.context_measurements.values():
        assert row["bbox_env_local_min_xyz_m"] == (0., 0., 0.)
        assert row["bbox_env_local_max_xyz_m"] == (1., 2., 3.)
        assert row["geometric_center_env_local_xyz_m"] == (.5, 1., 1.5)


def test_family_bridge_requires_explicit_evidence_root_before_native_start(tmp_path):
    manifest = tmp_path / "assets.json"
    manifest.write_text("{}")

    with pytest.raises(TypeError):
        create_bridge(
            robolab_root=tmp_path, assets_manifest=manifest, device="cuda:0",
            renderer="realtime", rendering_type="balanced",
        )

    bridge = create_bridge(
        robolab_root=tmp_path, assets_manifest=manifest, evidence_root=tmp_path / "evidence",
        device="cuda:0", renderer="realtime", rendering_type="balanced",
    )
    assert bridge._evidence_root == (tmp_path / "evidence").resolve()
    missing_scene = FixtureCandidate.from_json({**_candidate().task_payload(), "seed": 7, "metadata": {
        "scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]},
        "goal_supports": _candidate().metadata["goal_supports"],
    }})
    with pytest.raises(SimulatorBridgeError, match="native scene"):
        bridge.create_environment(build_task_definition(missing_scene), seed=7)


def test_family_bridge_passes_candidate_payload_to_mocked_native_factory(tmp_path, monkeypatch):
    manifest = tmp_path / "assets.json"
    manifest.write_text("{}")
    bridge = create_bridge(
        robolab_root=tmp_path, assets_manifest=manifest, evidence_root=tmp_path / "evidence",
        device="cuda:0", renderer="realtime", rendering_type="balanced",
    )
    calls = {}
    runtime = types.ModuleType("robolab.core.environments.runtime")
    runtime.create_env = lambda *args, **kwargs: (calls.setdefault("native_env", object()), calls.setdefault("create", (args, kwargs)))
    registrations = types.ModuleType("robolab.registrations.droid.auto_env_registrations_abs_ik")
    registrations.auto_register_droid_abs_ik_envs = lambda **kwargs: calls.setdefault("register", kwargs)
    cameras = types.ModuleType("robolab.registrations.droid.camera_presets")
    cameras.WRIST_LEFT_RIGHT_HEAD = "cameras"
    for name, module in {
        "robolab": types.ModuleType("robolab"),
        "robolab.core": types.ModuleType("robolab.core"),
        "robolab.core.environments": types.ModuleType("robolab.core.environments"),
        "robolab.core.environments.runtime": runtime,
        "robolab.registrations": types.ModuleType("robolab.registrations"),
        "robolab.registrations.droid": types.ModuleType("robolab.registrations.droid"),
        "robolab.registrations.droid.auto_env_registrations_abs_ik": registrations,
        "robolab.registrations.droid.camera_presets": cameras,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    received = {}
    monkeypatch.setattr(
        family_bridge,
        "RoboLabLatEnvironment",
        lambda env, candidate, evidence_root: received.update(
            env=env, candidate=candidate, evidence_root=evidence_root
        ) or "wrapped-environment",
    )

    value = bridge.create_environment(build_task_definition(_candidate()), seed=17)

    assert value == "wrapped-environment"
    assert calls["create"][0] == ("SGWFamilyQualificationTask",)
    assert calls["create"][1]["seed"] == 17
    assert calls["register"]["task"][0].endswith("family_qualification_task.py")
    assert json.loads(os.environ["SGW_FAMILY_CANDIDATE_JSON"])["goal_supports"]["higher"]["contact_sensor_id"] == "rubiks_cube__upper"
    assert received["evidence_root"] == (tmp_path / "evidence").resolve()


def test_family_controller_requires_calibration_and_preserves_measured_target_z(tmp_path):
    with pytest.raises(SimulatorBridgeError, match="controller-calibration"):
        create_controller()

    asset = tmp_path / "robot.usd"
    asset.write_bytes(b"verified robot")
    calibration = {
        "schema_version": "sgw-01-lat-closed-pad-midpoint-v1",
        "virtual_tcp_flange_xyz_m": [0, 0, 0],
        "lift_height_m": .12,
        "phase_hold_steps": [1] * 7 + [443],
        "robot_asset": {"sha256": __import__("hashlib").sha256(asset.read_bytes()).hexdigest()},
    }
    from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
    calibration["receipt_sha256"] = workspace_digest(calibration)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    controller = create_controller(controller_calibration=path)
    assert controller.calibration["receipt_sha256"] == calibration["receipt_sha256"]
    assert controller.identity["calibration_sha256"]

    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    class TwoD:
        def __getitem__(self, index):
            assert index == (0, 0)
            return Tensor([1, 0, 0, 0])

    class Data:
        root_quat_w = [Tensor([1, 0, 0, 0])]
        body_names = ["base_link"]
        body_quat_w = TwoD()
        root_pos_w = [Tensor([0, 0, 0])]

    class Robot:
        data = Data()
        class cfg:
            class spawn:
                usd_path = str(asset)

    class Scene:
        env_origins = [Tensor([0, 0, 0])]
        def __getitem__(self, key):
            return {"robot": Robot()}[key]

    environment = object.__new__(RoboLabLatEnvironment)
    environment._env = type("Native", (), {
        "scene": Scene(),
    })()
    actions = controller.actions_for_goal(environment, _candidate(), 1)
    assert len(actions) == 450
    # Index 5 is the direct lowering-to-target phase; it must retain .21 m,
    # not overwrite height with the initial cube z.
    assert actions[5][0, 2] == pytest.approx(.21)


def test_main_rejects_incomplete_family_candidate_before_applauncher(tmp_path, monkeypatch):
    manifest = tmp_path / "assets.json"
    manifest.write_bytes(b"assets")
    candidate = _candidate()
    candidate = FixtureCandidate.from_json({
        **candidate.task_payload(),
        "seed": 7,
        "asset_manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
        "metadata": {
            "scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]},
            "goal_supports": candidate.metadata["goal_supports"],
        },
    })
    args = type("Args", (), {
        "headless": True, "renderer": "realtime", "rendering_type": "balanced",
        "assets_manifest": manifest, "family": "HEIGHT", "output_root": tmp_path / "out",
        "robolab_root": tmp_path, "device": "cuda:0", "bridge_factory": "unused:factory",
        "controller_factory": "unused:factory", "controller_calibration": None, "seed": 7,
    })()
    monkeypatch.setattr(qualification, "parse_args", lambda: args)
    monkeypatch.setattr(qualification, "_load_selected_candidate", lambda _: candidate)

    with pytest.raises(SimulatorBridgeError, match="native scene"):
        qualification.main()
