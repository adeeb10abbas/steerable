import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate
from experiments.workshops.spatial_grounding_v1.robolab_height_dist_qualification import (
    _calibrated_actions_to_target,
    _calibration_digest,
    create_bridge,
    create_controller,
)
from experiments.workshops.spatial_grounding_v1.robolab_lat_qualification import RoboLabLatEnvironment
from experiments.workshops.spatial_grounding_v1.simulator_bridge import SimulatorBridgeError


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


def test_family_controller_requires_calibration_and_preserves_measured_target_z(tmp_path):
    with pytest.raises(SimulatorBridgeError, match="controller-calibration"):
        create_controller()

    calibration = {
        "schema_version": "sgw-01-lat-closed-pad-midpoint-v1",
        "virtual_tcp_flange_xyz_m": [0, 0, 0],
        "lift_height_m": .12,
        "phase_hold_steps": [1] * 7 + [443],
        "robot_asset": {"sha256": "a" * 64},
    }
    calibration["receipt_sha256"] = _calibration_digest(calibration)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    controller = create_controller(controller_calibration=path)
    assert controller.calibration["receipt_sha256"] == calibration["receipt_sha256"]

    actions = _calibrated_actions_to_target(
        calibration,
        cube_center_world_xyz_m=np.array([.3, 0, .1]),
        target_center_world_xyz_m=np.array([.4, 0, .21]),
        flange_quaternion_world_wxyz=np.array([1, 0, 0, 0]),
        robot_root_world_xyz_m=np.zeros(3),
    )
    assert len(actions) == 450
    # Index 5 is the direct lowering-to-target phase; it must retain .21 m,
    # not overwrite height with the initial cube z.
    assert actions[5][0, 2] == pytest.approx(.21)
