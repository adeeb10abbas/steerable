from __future__ import annotations

import hashlib
import json
import math

import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import (
    ASYMMETRY_LEVELS,
    LayoutContractError,
    PoseSE2,
    SymmetryWeights,
    build_candidate,
    canonical_json_bytes,
    evaluate_layout,
    interpolate_pose,
    load_candidate,
    pose_map_sha256,
    symmetry_residuals,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.candidate_builder import (
    INPUT_SCHEMA,
    build_from_spec,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.static_gate import (
    evaluate_static_candidate,
)


ASSET_CUBE = "ycb/rubiks_cube.usd#sha256:cube"
ASSET_BOWL = "ycb/bowl.usd#sha256:bowl"
ASSET_BANANA = "ycb/banana.usd#sha256:banana"


def make_candidate():
    control = {
        "rubiks_cube": PoseSE2(0.303, 0.124, 0.081, 0.0, ASSET_CUBE),
        "bowl": PoseSE2(0.443, 0.127, 0.077, 0.0, ASSET_BOWL),
        "banana": PoseSE2(0.539, -0.076, 0.068, 0.35, ASSET_BANANA),
    }
    symmetric = {
        "rubiks_cube": PoseSE2(0.303, 0.0, 0.081, 0.0, ASSET_CUBE),
        "bowl": PoseSE2(0.443, 0.0, 0.077, 0.0, ASSET_BOWL),
        "banana": PoseSE2(0.539, -0.22, 0.068, 0.35, ASSET_BANANA),
        "banana_right": PoseSE2(0.539, 0.22, 0.068, -0.35, ASSET_BANANA),
    }
    companions = {
        # This is a registered counterfactual interpolation anchor only.  The
        # companion is physically absent from the exact s=0 control.
        "banana_right": PoseSE2(0.539, 0.076, 0.068, -0.35, ASSET_BANANA)
    }
    control_digest = pose_map_sha256(control)
    return build_candidate(
        control_poses=control,
        symmetric_poses=symmetric,
        companion_counterfactual_s0_poses=companions,
        mirror_pairs=[("banana", "banana_right")],
        midline_objects=["rubiks_cube", "bowl"],
        target_object="rubiks_cube",
        reference_object="bowl",
        expected_cameras=["head", "wrist", "shoulder_left", "shoulder_right"],
        robot_base_xy_m=(0.0, 0.0),
        weights=SymmetryWeights(position_inverse_m=10.0, orientation_inverse_rad=1.0),
        s0_frozen_control_attestation={
            "inventory_policy": "exact_b001_inventory_and_poses",
            "inventory_transition": True,
            "source_fixture_id": "V3-B001/control",
            "source_fixture_sha256": "1" * 64,
            "source_queue_sha256": "2" * 64,
            "source_inventory": sorted(control),
            "control_poses_sha256": control_digest,
            "dose_response_primary_levels": [0.25, 0.5, 0.75, 1.0],
            "s0_analysis_role": "anchored_reference_not_in_primary_H3_slope",
            "design_limitation": "H1 s0-to-s1 includes the registered same-asset companion activation.",
        },
    )


def arm_pose():
    return {
        "arm_joint_positions_rad": [0.0] * 7,
        "gripper_position": 0.0,
        "measurement_source_sha256": "a" * 64,
        "eef_position_robot_xyz_m": [0.2, -0.1, 0.4],
    }


def test_planar_slerp_uses_short_arc_across_pi():
    a = PoseSE2(0.0, 0.0, 0.1, math.radians(170), "asset")
    b = PoseSE2(1.0, 1.0, 0.1, math.radians(-170), "asset")
    middle = interpolate_pose(a, b, 0.5)
    assert abs(abs(middle.yaw_rad) - math.pi) < 1e-12
    assert middle.x_m == pytest.approx(0.5)
    assert middle.y_m == pytest.approx(0.5)


def test_s0_is_exact_three_object_control_and_positive_levels_activate_companion():
    candidate = make_candidate()
    assert set(candidate.layout(0.0)) == {"rubiks_cube", "bowl", "banana"}
    for level in ASYMMETRY_LEVELS[1:]:
        assert set(candidate.layout(level)) == {"rubiks_cube", "bowl", "banana", "banana_right"}
    payload = candidate.to_json()
    assert payload["levels"]["0.00"]["inventory_transition"] is False
    assert payload["levels"]["0.25"]["inventory_transition"] is True
    assert payload["companion_activation_policy"]["H3_primary_levels"] == [0.25, 0.5, 0.75, 1.0]


def test_candidate_round_trip_is_hash_bound(tmp_path):
    candidate = make_candidate()
    path = tmp_path / "candidate.json"
    path.write_bytes(canonical_json_bytes(candidate.to_json()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_candidate(path, digest)
    assert loaded.to_json() == candidate.to_json()
    path.write_text(path.read_text().replace("0.25", "0.24", 1))
    with pytest.raises(LayoutContractError, match="SHA-256"):
        load_candidate(path, digest)


def test_candidate_accepts_only_machine_epsilon_derived_roundoff():
    payload = make_candidate().to_json()
    payload["levels"]["0.50"]["asymmetry_metric_A"] += 4e-16
    # Hash verification is performed by load_candidate; this unit isolates
    # cross-Python semantic reconstruction of an already hash-bound payload.
    from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import candidate_from_json

    assert candidate_from_json(payload).to_json()["levels"]["0.50"]["symmetry_level_s"] == 0.5
    payload["levels"]["0.50"]["asymmetry_metric_A"] += 1e-10
    with pytest.raises(LayoutContractError, match="derived field changed"):
        candidate_from_json(payload)


def test_full_pose_symmetry_catches_copied_yaw():
    candidate = make_candidate()
    poses = candidate.layout(1.0)
    broken = dict(poses)
    right = broken["banana_right"]
    broken["banana_right"] = PoseSE2(
        right.x_m, right.y_m, right.z_m, poses["banana"].yaw_rad, right.asset_identity
    )
    residual = symmetry_residuals(
        broken, [("banana", "banana_right")], ["rubiks_cube", "bowl"]
    )
    assert residual["position_residual_m"] == 0.0
    assert residual["orientation_residual_rad"] == pytest.approx(0.7)


def test_asset_identity_mismatch_fails_closed():
    candidate = make_candidate()
    poses = dict(candidate.layout(1.0))
    right = poses["banana_right"]
    poses["banana_right"] = PoseSE2(right.x_m, right.y_m, right.z_m, right.yaw_rad, "pear")
    with pytest.raises(LayoutContractError, match="different assets"):
        candidate.residuals(poses)


def test_live_s1_gate_logs_A_residuals_and_arm_pose():
    candidate = make_candidate()
    cameras = {name: False for name in candidate.expected_cameras}
    visible = {name: True for name in candidate.expected_cameras}
    row = evaluate_layout(
        candidate,
        symmetry_level_s=1.0,
        realised_object_poses=candidate.layout(1.0),
        occlusion_check_by_camera=cameras,
        target_visible_by_camera=visible,
        arm_reset_pose=arm_pose(),
    )
    assert row["asymmetry_metric_A"] == pytest.approx(0.0)
    assert row["position_residual_m"] == pytest.approx(0.0)
    assert row["orientation_residual_rad"] == pytest.approx(0.0)
    assert row["object_layout_symmetric_not_embodiment"] is True
    assert len(row["arm_reset_pose"]["arm_joint_positions_rad"]) == 7


def test_live_gate_rejects_undeclared_inventory_and_missing_camera():
    candidate = make_candidate()
    poses = dict(candidate.layout(0.25))
    poses["pear"] = PoseSE2(0.2, 0.2, 0.1, 0.0, "pear")
    cameras = {name: False for name in candidate.expected_cameras}
    visible = {name: True for name in candidate.expected_cameras}
    with pytest.raises(LayoutContractError, match="inventory"):
        evaluate_layout(
            candidate,
            symmetry_level_s=0.25,
            realised_object_poses=poses,
            occlusion_check_by_camera=cameras,
            target_visible_by_camera=visible,
            arm_reset_pose=arm_pose(),
        )
    cameras.pop("wrist")
    with pytest.raises(LayoutContractError, match="camera set"):
        evaluate_layout(
            candidate,
            symmetry_level_s=0.25,
            realised_object_poses=candidate.layout(0.25),
            occlusion_check_by_camera=cameras,
            target_visible_by_camera=visible,
            arm_reset_pose=arm_pose(),
        )


def test_realised_pose_must_match_requested_level():
    candidate = make_candidate()
    poses = dict(candidate.layout(0.5))
    cube = poses["rubiks_cube"]
    poses["rubiks_cube"] = PoseSE2(
        cube.x_m, cube.y_m + 0.01, cube.z_m, cube.yaw_rad, cube.asset_identity
    )
    with pytest.raises(LayoutContractError, match="requested s"):
        evaluate_layout(
            candidate,
            symmetry_level_s=0.5,
            realised_object_poses=poses,
            occlusion_check_by_camera={name: False for name in candidate.expected_cameras},
            target_visible_by_camera={name: True for name in candidate.expected_cameras},
            arm_reset_pose=arm_pose(),
        )


def test_candidate_json_is_strict_finite_json():
    # Guard against accidentally introducing NaN into a hash-bearing candidate.
    json.loads(canonical_json_bytes(make_candidate().to_json()), parse_constant=lambda token: pytest.fail(token))


def test_registered_builder_and_static_gate(tmp_path):
    candidate = make_candidate()
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = {
        "schema_version": INPUT_SCHEMA,
        "registered_before_inference": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "source_bindings": {"source.json": source_hash},
        "control_poses": {
            name: pose.to_json() for name, pose in candidate.control_poses.items()
        },
        "symmetric_poses": {
            name: pose.to_json() for name, pose in candidate.symmetric_poses.items()
        },
        "companion_counterfactual_s0_poses": {
            name: pose.to_json()
            for name, pose in candidate.companion_counterfactual_s0_poses.items()
        },
        "mirror_pairs": [list(pair) for pair in candidate.mirror_pairs],
        "midline_objects": list(candidate.midline_objects),
        "target_object": candidate.target_object,
        "reference_object": candidate.reference_object,
        "expected_cameras": list(candidate.expected_cameras),
        "robot_base_xy_m": list(candidate.robot_base_xy_m),
        "asymmetry_weights": candidate.weights.to_json(),
        "s0_frozen_control_attestation": dict(candidate.s0_frozen_control_attestation),
        "realisation_position_tolerance_m": candidate.realisation_position_tolerance_m,
        "realisation_orientation_tolerance_rad": candidate.realisation_orientation_tolerance_rad,
    }
    payload = build_from_spec(spec, repo_root=tmp_path)
    path = tmp_path / "candidate.json"
    path.write_bytes(canonical_json_bytes(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = evaluate_static_candidate(path, digest)
    assert report["passed"] is True
    assert report["levels"][0]["active_inventory"] == sorted(candidate.control_poses)
    assert report["H3_primary_levels"] == [0.25, 0.5, 0.75, 1.0]
    assert report["levels"][-1]["asymmetry_metric_A"] == pytest.approx(0.0)


def test_builder_rejects_changed_bound_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("changed\n")
    candidate = make_candidate()
    spec = {
        "schema_version": INPUT_SCHEMA,
        "registered_before_inference": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "source_bindings": {"source.json": "0" * 64},
        "control_poses": {name: pose.to_json() for name, pose in candidate.control_poses.items()},
    }
    with pytest.raises(LayoutContractError, match="source binding changed"):
        build_from_spec(spec, repo_root=tmp_path)
