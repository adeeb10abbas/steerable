"""Pure tests for the horizontal DROID G3 feasibility helpers."""

from __future__ import annotations

import math
import unittest

from experiments.online_correction_v4.droid_g3 import (
    DroidG3Error,
    bounds_world_to_task,
    classify_contacts,
    goal_area_case,
    horizontal_geometry_from_scene,
    physics_sampling_stride,
    reference_is_supported,
    scenario_duration_s,
    task_frame_from_evidence,
)


FRAME = {
    "u_left_world": [1.0, 0.0, 0.0],
    "u_front_world": [0.0, 1.0, 0.0],
    "u_up_world": [0.0, 0.0, 1.0],
    "origin_world": [0.0, 0.0, 0.0],
}
MOTION = {
    "move_stop_duration_s": 0.5,
    "slow_drift_duration_s": 4.0,
    "fast_drift_duration_s": 1.0,
    "reversal_waypoints": [
        {"time_s": 0.0, "displacement_units": 0.0},
        {"time_s": 2.0, "displacement_units": 1.0},
        {"time_s": 4.0, "displacement_units": -0.5},
    ],
}


def _bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> dict:
    return {
        "min_xyz": [x_min, y_min, z_min],
        "max_xyz": [x_max, y_max, z_max],
        "prim_path": "/World/test",
    }


def _scene() -> dict:
    return {
        "table_world_aabb_m": _bounds(-0.5, 0.5, -0.4, 0.4, -0.1, 0.0),
        "objects": {
            "rubiks_cube": {
                "world_aabb_m": _bounds(-0.02, 0.02, -0.02, 0.02, 0.0, 0.04)
            },
            "bowl": {
                "world_aabb_m": _bounds(-0.05, 0.05, -0.05, 0.05, 0.0, 0.05)
            },
        },
    }


class DroidG3SamplingTests(unittest.TestCase):
    def test_sampling_stride_uses_physics_substeps_below_cap(self) -> None:
        stride, interval = physics_sampling_stride(
            1.0 / 120.0, maximum_interval_s=0.02
        )
        self.assertEqual(stride, 2)
        self.assertAlmostEqual(interval, 1.0 / 60.0)

    def test_sampling_rejects_coarse_physics(self) -> None:
        with self.assertRaises(DroidG3Error):
            physics_sampling_stride(0.025, maximum_interval_s=0.02)

    def test_scenario_durations_are_frozen(self) -> None:
        expected = {
            "original_sham": 0.5,
            "destination_static": 0.5,
            "move_stop": 0.5,
            "slow_drift": 4.0,
            "fast_drift": 1.0,
            "reversal": 4.0,
        }
        self.assertEqual(
            {
                scenario: scenario_duration_s(scenario, MOTION)
                for scenario in expected
            },
            expected,
        )


class DroidG3GeometryTests(unittest.TestCase):
    def test_bounds_transform_uses_registered_task_frame(self) -> None:
        rotated = task_frame_from_evidence(
            {
                "u_left_world": [0.0, 1.0, 0.0],
                "u_front_world": [-1.0, 0.0, 0.0],
                "u_up_world": [0.0, 0.0, 1.0],
                "origin_world": [0.0, 0.0, 0.0],
            }
        )
        result = bounds_world_to_task(
            rotated, _bounds(-0.2, 0.2, -0.4, 0.4, 0.0, 0.1)
        )
        self.assertAlmostEqual(result.x_min, -0.4)
        self.assertAlmostEqual(result.x_max, 0.4)
        self.assertAlmostEqual(result.y_min, -0.2)
        self.assertAlmostEqual(result.y_max, 0.2)

    def test_shrinking_goal_area_must_remove_twenty_percent(self) -> None:
        geometry = horizontal_geometry_from_scene(
            task_frame_evidence=FRAME,
            scene_state=_scene(),
            support_edge_margin_m=0.005,
        )
        shrinking = goal_area_case(
            geometry=geometry,
            relation="left",
            original_reference_world=(0.0, 0.0, 0.025),
            endpoint_reference_world=(0.12, 0.0, 0.025),
            clearance_m=0.01,
            minimum_shrinking_area_fraction=0.20,
        )
        self.assertTrue(shrinking["shrinking_direction"])
        self.assertGreater(shrinking["removed_area_fraction"], 0.20)
        self.assertTrue(shrinking["passes_information_gate"])

        expanding = goal_area_case(
            geometry=geometry,
            relation="right",
            original_reference_world=(0.0, 0.0, 0.025),
            endpoint_reference_world=(0.12, 0.0, 0.025),
            clearance_m=0.01,
            minimum_shrinking_area_fraction=0.20,
        )
        self.assertFalse(expanding["shrinking_direction"])
        self.assertTrue(expanding["passes_information_gate"])

    def test_reference_support_uses_full_footprint(self) -> None:
        geometry = horizontal_geometry_from_scene(
            task_frame_evidence=FRAME,
            scene_state=_scene(),
            support_edge_margin_m=0.005,
        )
        self.assertTrue(
            reference_is_supported(
                frame=geometry["frame"],
                reference_position_world=(0.0, 0.0, 0.025),
                table_bounds_task=geometry["table_bounds_task"],
                reference_footprint=geometry["reference_footprint"],
                edge_margin_m=0.005,
            )
        )
        self.assertFalse(
            reference_is_supported(
                frame=geometry["frame"],
                reference_position_world=(0.49, 0.0, 0.025),
                table_bounds_task=geometry["table_bounds_task"],
                reference_footprint=geometry["reference_footprint"],
                edge_margin_m=0.005,
            )
        )


class DroidG3ContactTests(unittest.TestCase):
    def test_only_supported_table_contacts_pass(self) -> None:
        result = classify_contacts(
            {
                "rubiks_cube__table": 1.0,
                "banana__table": 2.0,
                "bowl__table": 3.0,
                "robot_all__bowl": 0.0,
            },
            active_force_threshold_n=0.05,
        )
        self.assertTrue(result["support_valid"])
        self.assertFalse(result["reference_robot_contact"])
        self.assertFalse(result["unmodeled_collision"])

    def test_robot_reference_or_object_contact_fails(self) -> None:
        result = classify_contacts(
            {
                "rubiks_cube__table": 1.0,
                "banana__table": 2.0,
                "bowl__table": 3.0,
                "robot_all__bowl": 0.2,
                "rubiks_cube__bowl": 0.3,
            },
            active_force_threshold_n=0.05,
        )
        self.assertTrue(result["reference_robot_contact"])
        self.assertTrue(result["unmodeled_collision"])
        self.assertEqual(
            result["disallowed_contact_pairs"],
            ["robot_all__bowl", "rubiks_cube__bowl"],
        )

    def test_nan_contact_force_is_rejected(self) -> None:
        with self.assertRaises(DroidG3Error):
            classify_contacts(
                {"bowl__table": math.nan},
                active_force_threshold_n=0.05,
            )


if __name__ == "__main__":
    unittest.main()
