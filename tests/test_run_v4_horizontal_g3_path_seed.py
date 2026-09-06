"""Focused tests for the horizontal G3 path-seed runner helpers."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json"
)
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_v4_horizontal_g3_path_seed",
    ROOT / "tools/run_v4_horizontal_g3_path_seed.py",
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def _geometry() -> dict:
    from experiments.online_correction_v4.droid_g3 import horizontal_geometry_from_scene

    frame = {
        "u_left_world": [1.0, 0.0, 0.0],
        "u_front_world": [0.0, 1.0, 0.0],
        "u_up_world": [0.0, 0.0, 1.0],
        "origin_world": [0.0, 0.0, 0.0],
    }
    scene = {
        "table_world_aabb_m": {
            "min_xyz": [-0.5, -0.4, -0.1],
            "max_xyz": [0.5, 0.4, 0.0],
        },
        "objects": {
            "rubiks_cube": {
                "world_aabb_m": {
                    "min_xyz": [-0.02, -0.02, 0.0],
                    "max_xyz": [0.02, 0.02, 0.04],
                }
            },
            "bowl": {
                "world_aabb_m": {
                    "min_xyz": [-0.05, -0.05, 0.0],
                    "max_xyz": [0.05, 0.05, 0.05],
                }
            },
        },
    }
    return horizontal_geometry_from_scene(
        task_frame_evidence=frame,
        scene_state=scene,
        support_edge_margin_m=0.005,
    )


class G3PathSeedSamplingTests(unittest.TestCase):
    def test_path_sample_times_include_endpoint(self) -> None:
        times = runner.path_sample_times(0.5, 0.02)
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 0.5)
        self.assertGreater(len(times), 1)
        self.assertAlmostEqual(times[1] - times[0], 0.02)

    def test_path_sample_times_adds_endpoint_when_not_on_grid(self) -> None:
        times = runner.path_sample_times(0.51, 0.02)
        self.assertAlmostEqual(times[-1], 0.51)
        self.assertNotAlmostEqual(times[-2], 0.51)

    def test_motion_onset_is_zero_except_destination_static(self) -> None:
        self.assertIsNone(runner.motion_onset_s_for_scenario("destination_static"))
        self.assertEqual(runner.motion_onset_s_for_scenario("move_stop"), 0.0)
        self.assertEqual(runner.motion_onset_s_for_scenario("original_sham"), 0.0)

    def test_configure_motion_controller_schedules_onset_for_moving_profiles(self) -> None:
        motion = json.loads(CAMPAIGN.read_text(encoding="utf-8"))["motion"]
        moving = runner.configure_motion_controller(
            "move_stop",
            displacement_m=0.12,
            motion_config=motion,
        )
        static = runner.configure_motion_controller(
            "destination_static",
            displacement_m=0.12,
            motion_config=motion,
        )
        self.assertEqual(moving.event_onset_s, 0.0)
        self.assertIsNone(static.event_onset_s)
        self.assertAlmostEqual(moving.pose_at(0.5).displacement_m, 0.12)
        self.assertAlmostEqual(static.pose_at(0.0).displacement_m, 0.12)


class G3PathSeedGeometryTests(unittest.TestCase):
    def test_expected_reference_world_position_matches_robot_frame_offset(self) -> None:
        baseline = (0.1, 0.2, 0.03)
        quaternion = (1.0, 0.0, 0.0, 0.0)
        expected = runner.expected_reference_world_position(
            baseline_world=baseline,
            robot_quaternion_wxyz=quaternion,
            direction_task=(1.0, 0.0),
            displacement_m=0.12,
        )
        self.assertAlmostEqual(expected[0], 0.1)
        self.assertAlmostEqual(expected[1], 0.32)
        self.assertAlmostEqual(expected[2], 0.03)

    def test_stationary_drift_is_euclidean(self) -> None:
        self.assertAlmostEqual(
            runner.stationary_drift_m((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)),
            5.0,
        )


class G3PathSeedEvaluationTests(unittest.TestCase):
    def test_evaluate_path_sample_passes_clean_support(self) -> None:
        geometry = _geometry()
        contract = json.loads(CAMPAIGN.read_text(encoding="utf-8"))["fixtures"][
            "horizontal"
        ]["model_blind_g3_geometry"]
        baseline = (0.0, 0.0, 0.025)
        planned = runner.expected_reference_world_position(
            baseline_world=baseline,
            robot_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            direction_task=(1.0, 0.0),
            displacement_m=0.06,
        )
        sample, reasons = runner.evaluate_path_sample(
            goal="left",
            geometry=geometry,
            geometry_contract=contract,
            baseline_reference_world=baseline,
            baseline_target_world=(0.12, 0.0, 0.02),
            baseline_distractor_world=(-0.12, 0.0, 0.02),
            direction_task=(1.0, 0.0),
            planned_displacement_m=0.06,
            planned_reference_world=planned,
            measured_reference_world=planned,
            measured_target_world=(0.12, 0.0, 0.02),
            measured_distractor_world=(-0.12, 0.0, 0.02),
            contacts={
                "support_valid": True,
                "reference_robot_contact": False,
                "unmodeled_collision": False,
            },
            target_object="rubiks_cube",
        )
        self.assertEqual(reasons, [])
        self.assertTrue(sample["support_valid"])
        self.assertTrue(sample["legal_goal_nonempty"])

    def test_evaluate_path_sample_records_pose_and_drift_violations(self) -> None:
        geometry = _geometry()
        contract = json.loads(CAMPAIGN.read_text(encoding="utf-8"))["fixtures"][
            "horizontal"
        ]["model_blind_g3_geometry"]
        baseline = (0.0, 0.0, 0.025)
        planned = runner.expected_reference_world_position(
            baseline_world=baseline,
            robot_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            direction_task=(1.0, 0.0),
            displacement_m=0.06,
        )
        _, reasons = runner.evaluate_path_sample(
            goal="left",
            geometry=geometry,
            geometry_contract=contract,
            baseline_reference_world=baseline,
            baseline_target_world=(0.12, 0.0, 0.02),
            baseline_distractor_world=(-0.12, 0.0, 0.02),
            direction_task=(1.0, 0.0),
            planned_displacement_m=0.06,
            planned_reference_world=planned,
            measured_reference_world=(
                planned[0] + 0.01,
                planned[1],
                planned[2],
            ),
            measured_target_world=(0.12, 0.01, 0.02),
            measured_distractor_world=(-0.12, 0.0, 0.02),
            contacts={
                "support_valid": False,
                "reference_robot_contact": True,
                "unmodeled_collision": True,
            },
            target_object="rubiks_cube",
        )
        self.assertIn("reference_pose_error_exceeds_contract", reasons)
        self.assertIn("target_drift_exceeds_contract", reasons)
        self.assertIn("support_invalid", reasons)
        self.assertIn("reference_robot_contact", reasons)
        self.assertIn("unmodeled_collision", reasons)

    def test_summarize_path_check_aggregates_unique_reasons(self) -> None:
        observation = runner.summarize_path_check(
            goal="left",
            scenario="move_stop",
            planned_duration_s=0.5,
            sample_interval_s=0.02,
            sample_records=[
                {
                    "support_valid": False,
                    "reachable_workspace": True,
                    "legal_goal_nonempty": True,
                    "reference_robot_contact": False,
                    "unmodeled_collision": True,
                    "reasons": ["unmodeled_collision", "support_invalid"],
                },
                {
                    "support_valid": True,
                    "reachable_workspace": False,
                    "legal_goal_nonempty": True,
                    "reference_robot_contact": True,
                    "unmodeled_collision": False,
                    "reasons": ["reference_robot_contact", "support_invalid"],
                },
            ],
            measured_trajectory_evidence={
                "path": "measured.json",
                "sha256": "a" * 64,
                "bytes": 1,
            },
            reference_trajectory_evidence={
                "path": "reference.json",
                "sha256": "b" * 64,
                "bytes": 1,
            },
        )
        self.assertFalse(observation["collision_free"])
        self.assertFalse(observation["support_valid"])
        self.assertFalse(observation["reachable_workspace"])
        self.assertTrue(observation["legal_goal_nonempty"])
        self.assertTrue(observation["reference_robot_contact"])
        self.assertEqual(
            observation["reasons"],
            [
                "unmodeled_collision",
                "support_invalid",
                "reference_robot_contact",
            ],
        )


class G3PathSeedGateValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))

    def test_validate_gate_inputs_accepts_pinned_plan(self) -> None:
        from experiments.online_correction_v4.droid_task_files.binding import sha256_file

        runner.validate_gate_inputs(
            plan=self.plan,
            campaign=self.campaign,
            campaign_path=CAMPAIGN,
            campaign_sha256=sha256_file(CAMPAIGN),
            plan_path=PLAN,
            plan_sha256=sha256_file(PLAN),
            reset_registry_path=REGISTRY,
            reset_registry_sha256=sha256_file(REGISTRY),
            environment_seed=2100000000,
            scale=1.0,
            sha256_file=sha256_file,
        )

    def test_validate_gate_inputs_rejects_hash_mismatch(self) -> None:
        from experiments.online_correction_v4.droid_task_files.binding import sha256_file

        with self.assertRaises(RuntimeError):
            runner.validate_gate_inputs(
                plan=self.plan,
                campaign=self.campaign,
                campaign_path=CAMPAIGN,
                campaign_sha256="0" * 64,
                plan_path=PLAN,
                plan_sha256=sha256_file(PLAN),
                reset_registry_path=REGISTRY,
                reset_registry_sha256=sha256_file(REGISTRY),
                environment_seed=2100000000,
                scale=1.0,
                sha256_file=sha256_file,
            )

    def test_build_goal_area_cases_returns_four_relations(self) -> None:
        geometry = _geometry()
        cases = runner.build_goal_area_cases(
            geometry=geometry,
            baseline_reference_world=(0.0, 0.0, 0.025),
            direction_by_goal={
                "left": [1.0, 0.0],
                "right": [1.0, 0.0],
                "front": [0.0, 1.0],
                "behind": [0.0, 1.0],
            },
            displacement_m=0.12,
            robot_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            clearance_m=0.01,
            minimum_shrinking_area_fraction=0.20,
        )
        self.assertEqual(len(cases), 4)
        self.assertEqual([case["relation"] for case in cases], [
            "left",
            "right",
            "front",
            "behind",
        ])


class G3PathSeedImportSafetyTests(unittest.TestCase):
    def test_runner_module_imports_without_robolab(self) -> None:
        blocked = {
            "isaaclab",
            "isaaclab.app",
            "robolab",
            "robolab.constants",
        }
        original_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".", 1)[0]
            if name in blocked or root in {"isaaclab", "robolab", "omni"}:
                raise ImportError(f"blocked import {name}")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            spec = importlib.util.spec_from_file_location(
                "g3_path_seed_isolated",
                ROOT / "tools/run_v4_horizontal_g3_path_seed.py",
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertAlmostEqual(
                module.path_sample_times(0.04, 0.02),
                [0.0, 0.02, 0.04],
            )


if __name__ == "__main__":
    unittest.main()
