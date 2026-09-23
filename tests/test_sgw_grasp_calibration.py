import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate
from experiments.workshops.spatial_grounding_v1.grasp_calibration import calibrated_actions, derive_calibration
from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
from experiments.workshops.spatial_grounding_v1.lat_workspace_capture import _rotate_wxyz
from experiments.workshops.spatial_grounding_v1.robolab_lat_qualification import RoboLabLatEnvironment, RoboLabLatScriptedController


def measured_workspace():
    records = []
    for index in range(61):
        closed = 1 <= index <= 30
        distance, half_width = (0.144, 0.01) if closed else (0.13, 0.05)
        bodies = {
            name: {"position_world_xyz_m": position, "quaternion_world_wxyz": [1, 0, 0, 0]}
            for name, position in (
                ("base_link", [1, 2, 3]),
                ("left_inner_finger", [1 + distance, 2 + half_width, 3]),
                ("right_inner_finger", [1 + distance, 2 - half_width, 3]),
            )
        }
        records.append({
            "action_step": index, "sim_time_s": index / 15,
            "robot_body_frames": {"bodies": bodies},
            "joint_names": ["finger_joint"], "joint_position_rad": [0.785398 if closed else 0],
        })
    workspace = {
        "model_request_count": 0, "behavioral_episode_count": 0, "robolab_commit": "test-source",
        "robot": {"asset_usd": {"sha256": "a" * 64}},
        "gripper_calibration": {
            "status": "measured_empty_gripper_motion_not_fixture_qualification",
            "model_requests": 0, "behavioral_episodes": 0,
            "issued_controller_commands": 60, "observed_controller_actions": 60,
            "records": records, "control_step_dt_s": 1 / 15,
            "finger_geometry": [
                {"body": f"{side}_inner_finger", "prim": f"/{side}/Defeatured_2F_85_PAD_OPEN_fingertipsstep",
                 "body_local_bounds_center_m": [0, 0, 0], "collision_geometry": False}
                for side in ("left", "right")
            ],
        },
    }
    workspace["receipt_sha256"] = workspace_digest(workspace)
    return workspace


def candidate():
    path = Path(__file__).parents[1] / "artifacts/workshops/spatial_grounding_v1/proposals/lat-20260922.json"
    return FixtureCandidate.from_json(next(
        row for row in json.loads(path.read_text())["candidates"] if row["candidate_id"] == "LAT-CANDIDATE-032"
    ))


def test_calibration_uses_measured_closed_pad_midpoint_and_keeps_claim_boundary():
    value = derive_calibration(measured_workspace(), "b" * 64)
    assert value["virtual_tcp_flange_xyz_m"] == pytest.approx([0.144, 0, 0])
    assert value["measured_samples"]["open"]["midpoint_flange_xyz_m"] == pytest.approx([0.13, 0, 0])
    assert value["measured_samples"]["closed"]["pad_center_separation_m"] == pytest.approx(0.02)
    assert value["receipt_sha256"] == workspace_digest(value)
    assert value["model_requests"] == value["behavioral_episodes"] == 0
    assert "not a measured contact surface" in value["claim_boundary"]
    assert sum(value["phase_hold_steps"]) == 450


@pytest.mark.parametrize("failure", ["incomplete", "wrong_joint", "wrong_time", "missing_pad"])
def test_calibration_rejects_incomplete_or_unmeasured_probe(failure):
    value = measured_workspace()
    probe = value["gripper_calibration"]
    if failure == "incomplete":
        probe["observed_controller_actions"] = 59
    elif failure == "wrong_joint":
        probe["records"][30]["joint_position_rad"] = [0]
    elif failure == "wrong_time":
        probe["records"][5]["sim_time_s"] = 0
    else:
        probe["finger_geometry"].pop()
    value["receipt_sha256"] = workspace_digest(value)
    with pytest.raises(ValueError):
        derive_calibration(value, "b" * 64)


@pytest.mark.parametrize("sign", [1, -1])
def test_static_controller_corrects_position_and_releases_then_retreats(sign):
    fixture = candidate()
    before = copy.deepcopy(fixture)
    calibration = derive_calibration(measured_workspace(), "b" * 64)
    q = np.array([2**-0.5, 0, 2**-0.5, 0])
    robot_root = np.array([0.01, 0.02, 0.03])
    commands = calibrated_actions(
        fixture, sign, calibration, flange_quaternion_world_wxyz=q,
        environment_origin_world_xyz_m=np.zeros(3), robot_root_world_xyz_m=robot_root,
    )
    assert len(commands) == 450 and all(a.shape == (1, 8) for a in commands)
    offset = np.array(_rotate_wxyz(q, calibration["virtual_tcp_flange_xyz_m"]))
    cube = np.array(fixture.scoring_poses()["rubiks_cube"].position_m)
    np.testing.assert_allclose(commands[20][0, :3] + robot_root + offset, cube, atol=1e-7)
    assert commands[40][0, 7] == pytest.approx(0.785398)
    assert commands[120][0, 7] == commands[-1][0, 7] == 0
    assert commands[-1][0, 2] - commands[120][0, 2] == pytest.approx(0.12)
    assert sign * (commands[-1][0, 1] + robot_root[1] - cube[1]) > 0.12
    assert fixture == before
    calibration["virtual_tcp_flange_xyz_m"][0] += 0.01
    with pytest.raises(ValueError, match="identity"):
        calibrated_actions(
            fixture, sign, calibration, flange_quaternion_world_wxyz=q,
            environment_origin_world_xyz_m=np.zeros(3), robot_root_world_xyz_m=robot_root,
        )


def test_native_controller_binds_actual_robot_asset_and_calibration(tmp_path):
    def tensor(value):
        value = np.asarray(value)
        result = SimpleNamespace()
        result.detach = result.cpu = lambda: result
        result.numpy = lambda: value
        return result
    asset = tmp_path / "robot.usd"
    asset.write_bytes(b"synthetic test asset")
    calibration = derive_calibration(measured_workspace(), "b" * 64)
    calibration["robot_asset"]["sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    calibration["receipt_sha256"] = workspace_digest(calibration)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    robot = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(asset))),
        data=SimpleNamespace(
            body_names=["base_link"], root_quat_w=[tensor([1, 0, 0, 0])],
            root_pos_w=[tensor([0, 0, 0])],
            body_quat_w={(0, 0): tensor([2**-0.5, 0, 2**-0.5, 0])},
        ),
    )
    class Scene(dict):
        env_origins = [tensor([0, 0, 0])]
    env = object.__new__(RoboLabLatEnvironment)
    env._env = SimpleNamespace(scene=Scene(robot=robot))
    controller = RoboLabLatScriptedController(path)
    assert controller.identity["calibration_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(controller.actions_for_goal(env, candidate(), 1)) == 450
    asset.write_bytes(b"different asset")
    with pytest.raises(RuntimeError, match="actual robot asset"):
        controller.actions_for_goal(env, candidate(), 1)
