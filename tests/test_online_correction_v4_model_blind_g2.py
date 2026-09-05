"""Tests for V4 horizontal model-blind G2 receipt validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.droid_contract import (
    canonical_json_bytes as contract_json_bytes,
    sha256_bytes as contract_sha256,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    ObjectRoleBinding,
    ResetRegistry,
)
from experiments.online_correction_v4.model_blind_g2 import (
    G2GateError,
    REQUIRED_POLICY_CAMERAS,
    compile_aggregate_receipt,
    compile_seed_receipt,
    task_frame_evidence,
)


def _registry() -> ResetRegistry:
    return ResetRegistry(
        schema_version="v4-droid-horizontal-reset-registry-v1",
        fixture_id="horizontal",
        status="model_blind_candidate_not_released_for_inference",
        model_request_count=0,
        behavioral_episode_count=0,
        scene_asset="rubiks_cube_banana_bowl.usda",
        scene_metadata_sha256="8" * 64,
        contact_objects=("rubiks_cube", "banana", "bowl", "table"),
        object_roles={
            "target": ObjectRoleBinding("target", "rubiks_cube", "asset-target"),
            "reference": ObjectRoleBinding("reference", "bowl", "asset-reference"),
            "distractor": ObjectRoleBinding("distractor", "banana", "asset-distractor"),
        },
        positions_by_env_seed={
            2100000000: {
                "rubiks_cube": (0.30, 0.12, 0.08),
                "bowl": (0.44, 0.126, 0.077),
                "banana": (0.54, -0.075, 0.068),
            }
        },
        registry_path="/tmp/registry.json",
        registry_sha256="5" * 64,
    )


def _attestation(episode_id: str) -> dict:
    value = {
        "schema_version": "v4-droid-reset-attestation-v1",
        "study_id": "online_correction_v4",
        "registered_episode_id": episode_id,
        "environment_seed": 2100000000,
        "fixture_id": "horizontal",
        "model_request_count_before_attestation": 0,
        "runner_pre_action_reset_calls": 2,
        "physical_reset_calls": 1,
        "settle_gate_runs": 1,
        "duplicate_second_reset_idempotent": True,
    }
    value["reset_fingerprint_sha256"] = contract_sha256(
        contract_json_bytes(value)
    )
    return value


def _physical() -> dict:
    return {
        "environment_seed": 2100000000,
        "objects": {
            "rubiks_cube": {
                "position_robot_xyz_m": [0.30, 0.12, 0.08],
                "position_world_xyz_m": [0.0, 0.0, 1.0],
            },
            "bowl": {
                "position_robot_xyz_m": [0.44, 0.126, 0.077],
                "position_world_xyz_m": [0.0, 0.0, 1.0],
            },
            "banana": {
                "position_robot_xyz_m": [0.54, -0.075, 0.068],
                "position_world_xyz_m": [0.0, 0.0, 1.0],
            },
        },
        "robot_position_robot_xyz_m": [0.0, 0.0, 0.0],
        "robot_position_world_xyz_m": [1.0, 2.0, 0.0],
        "robot_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
        "measured_native_control_dt_s": 0.05,
    }


def _cameras() -> dict:
    return {
        name: {
            "shape": [64, 64, 3],
            "dtype": "uint8",
            "pixel_range": 255.0,
            "raw_array_sha256": "6" * 64,
            "nonblank": True,
            "policy_input_camera": True,
        }
        for name in REQUIRED_POLICY_CAMERAS
    }


def _camera_geometry() -> dict:
    return {
        name: {
            "camera_center_world_m": [0.0, 0.0, 0.0],
            "camera_quaternion_world_wxyz_ros": [1.0, 0.0, 0.0, 0.0],
            "intrinsic_matrix_3x3": [
                [20.0, 0.0, 32.0],
                [0.0, 20.0, 32.0],
                [0.0, 0.0, 1.0],
            ],
            "image_size_wh": [64, 64],
        }
        for name in REQUIRED_POLICY_CAMERAS
    }


def _artifacts() -> dict:
    return {
        "policy_camera_images": {
            name: {"decoded_raw_array_sha256": "6" * 64}
            for name in REQUIRED_POLICY_CAMERAS
        },
        "axis_overlay_images": {
            "montage": {"path": "/tmp/axis.png", "sha256": "7" * 64, "bytes": 1}
        },
    }


class ModelBlindG2Tests(unittest.TestCase):
    def test_task_frame_uses_droid_robot_axes(self) -> None:
        frame = task_frame_evidence(_physical())
        self.assertEqual(frame["u_left_world"], [0.0, 1.0, 0.0])
        self.assertEqual(frame["u_front_world"], [-1.0, 0.0, 0.0])
        self.assertEqual(frame["u_up_world"], [0.0, 0.0, 1.0])
        self.assertTrue(frame["right_handed"])

    def test_seed_receipt_passes_reset_camera_but_not_full_g2(self) -> None:
        episode_id = "online-correction-v4-g2-horizontal-2100000000"
        receipt = compile_seed_receipt(
            env_seed=2100000000,
            episode_id=episode_id,
            registry=_registry(),
            reset_attestation=_attestation(episode_id),
            physical_reset=_physical(),
            camera_views=_cameras(),
            camera_geometry=_camera_geometry(),
            expected_native_control_dt_s=0.05,
            runtime_identity={"study_commit": "a" * 40},
            artifacts=_artifacts(),
        )
        self.assertTrue(receipt["passed_reset_and_camera"])
        self.assertFalse(receipt["g2_complete"])
        self.assertEqual(receipt["model_request_count"], 0)
        self.assertEqual(receipt["registry_position_tolerance_m"], 0.005)

    def test_seed_receipt_rejects_missing_policy_camera(self) -> None:
        episode_id = "online-correction-v4-g2-horizontal-2100000000"
        cameras = _cameras()
        cameras.pop("wrist_cam")
        with self.assertRaises(G2GateError):
            compile_seed_receipt(
                env_seed=2100000000,
                episode_id=episode_id,
                registry=_registry(),
                reset_attestation=_attestation(episode_id),
                physical_reset=_physical(),
                camera_views=cameras,
                camera_geometry=_camera_geometry(),
                expected_native_control_dt_s=0.05,
                runtime_identity={},
                artifacts=_artifacts(),
            )

    def test_aggregate_requires_complete_seed_coverage_and_axis_review(self) -> None:
        seed_receipt = {
            "schema_version": "v4-horizontal-g2-seed-receipt-v1",
            "environment_seed": 2100000000,
            "passed_reset_and_camera": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        }
        blocked = compile_aggregate_receipt(
            expected_env_seeds=(2100000000,),
            seed_receipts=(seed_receipt,),
            axis_review=None,
        )
        self.assertFalse(blocked["passed"])
        passed = compile_aggregate_receipt(
            expected_env_seeds=(2100000000,),
            seed_receipts=(seed_receipt,),
            axis_review={
                "schema_version": "v4-horizontal-g2-axis-review-v1",
                "campaign_id": "online_correction_v4",
                "fixture_id": "horizontal",
                "passed": True,
                "rendered_left_front_up": True,
                "model_request_count": 0,
                "behavioral_episode_count": 0,
                "reviewer_identity": "unit-test",
                "reviewed_at_utc": "2026-09-05T00:00:00Z",
                "source_axis_overlay": {
                    "path": "/tmp/axis.png",
                    "sha256": "7" * 64,
                    "bytes": 1,
                },
                "assertions": {
                    "left_axis_matches_fixed_robot_viewpoint": True,
                    "front_axis_points_toward_robot": True,
                    "up_axis_opposes_gravity": True,
                    "labels_and_arrow_origins_visible": True,
                },
            },
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(passed["authorizes_behavioral_inference"])

    def test_aggregate_rejects_duplicate_seed(self) -> None:
        receipt = {
            "schema_version": "v4-horizontal-g2-seed-receipt-v1",
            "environment_seed": 2100000000,
            "passed_reset_and_camera": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        }
        with self.assertRaises(G2GateError):
            compile_aggregate_receipt(
                expected_env_seeds=(2100000000,),
                seed_receipts=(receipt, dict(receipt)),
                axis_review=None,
            )


if __name__ == "__main__":
    unittest.main()
