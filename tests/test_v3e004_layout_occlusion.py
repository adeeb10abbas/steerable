from __future__ import annotations

import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import LayoutContractError
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.occlusion import (
    CameraEvidence,
    YawOrientedBox,
    evaluate_all_cameras,
    evaluate_camera_evidence,
    project_world_target_to_pixel,
    segment_intersects_box_before_target,
)


def box_at(x: float):
    return YawOrientedBox(
        center_world_m=(x, 0.0, 0.1),
        half_extents_m=(0.04, 0.07, 0.04),
        yaw_world_rad=0.0,
    )


def camera(name: str, *, bowl_x: float = 0.45, target_x: float = 0.30):
    return CameraEvidence(
        camera_name=name,
        camera_center_world_m=(0.0, 0.0, 0.1),
        target_center_world_m=(target_x, 0.0, 0.1),
        reference_bounds_world=box_at(bowl_x),
        target_instance_visible_pixels=250,
        segmentation_source_sha256="b" * 64,
    )


def test_reference_behind_target_is_not_occlusion():
    assert not segment_intersects_box_before_target(
        (0.0, 0.0, 0.1), (0.30, 0.0, 0.1), box_at(0.45)
    )
    row = evaluate_camera_evidence(camera("head"), minimum_visible_target_pixels=20)
    assert row["occlusion_check"] is False
    assert row["target_visible"] is True
    assert row["gate_passed"] is True


def test_reference_between_camera_and_target_is_occlusion():
    assert segment_intersects_box_before_target(
        (0.0, 0.0, 0.1), (0.50, 0.0, 0.1), box_at(0.25)
    )
    row = evaluate_camera_evidence(
        camera("head", bowl_x=0.25, target_x=0.50), minimum_visible_target_pixels=20
    )
    assert row["occlusion_check"] is True
    assert row["gate_passed"] is False


def test_calibrated_projection_can_supply_visibility_without_segmentation():
    evidence = CameraEvidence(
        camera_name="wrist",
        camera_center_world_m=(0.0, 0.1, 0.4),
        target_center_world_m=(0.30, 0.0, 0.1),
        reference_bounds_world=box_at(0.45),
        target_instance_visible_pixels=None,
        segmentation_source_sha256=None,
        target_projected_pixel_uv=(320.0, 240.0),
        image_size_wh=(640, 480),
        camera_geometry_source_sha256="c" * 64,
    )
    row = evaluate_camera_evidence(evidence, minimum_visible_target_pixels=20)
    assert row["target_visible"] is True
    assert row["gate_passed"] is True


def test_projection_is_recomputed_from_live_extrinsics_and_intrinsics():
    # Identity ROS camera looks along +Z.  A target at camera (x,y,z)=(1,2,10)
    # projects under fx=fy=100, cx=320, cy=240 to (330,260).
    pixel = project_world_target_to_pixel(
        camera_center_world_m=(0.0, 0.0, 0.0),
        camera_quaternion_world_wxyz_ros=(1.0, 0.0, 0.0, 0.0),
        target_center_world_m=(1.0, 2.0, 10.0),
        intrinsic_matrix_3x3=((100.0, 0.0, 320.0), (0.0, 100.0, 240.0), (0.0, 0.0, 1.0)),
    )
    assert pixel == pytest.approx((330.0, 260.0))


def test_missing_segmentation_and_projection_fails_closed():
    evidence = CameraEvidence(
        camera_name="head",
        camera_center_world_m=(0.0, 0.0, 0.1),
        target_center_world_m=(0.30, 0.0, 0.1),
        reference_bounds_world=box_at(0.45),
        target_instance_visible_pixels=None,
        segmentation_source_sha256=None,
    )
    with pytest.raises(LayoutContractError, match="neither segmentation"):
        evaluate_camera_evidence(evidence, minimum_visible_target_pixels=20)


def test_all_camera_gate_rejects_missing_or_failed_camera():
    evidence = {name: camera(name) for name in ("head", "wrist")}
    with pytest.raises(LayoutContractError, match="camera evidence set"):
        evaluate_all_cameras(
            evidence, expected_cameras=("head", "wrist", "shoulder"), minimum_visible_target_pixels=20
        )
    evidence["wrist"] = camera("wrist", bowl_x=0.25, target_x=0.50)
    with pytest.raises(LayoutContractError, match="gate failed"):
        evaluate_all_cameras(
            evidence, expected_cameras=("head", "wrist"), minimum_visible_target_pixels=20
        )
