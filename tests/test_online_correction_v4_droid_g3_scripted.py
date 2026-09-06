"""Unit tests for the V4 horizontal G3 scripted controller."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from experiments.online_correction_v4 import geometry as geom
from experiments.online_correction_v4.adapters import TerminalPhysicalPredicates
from experiments.online_correction_v4.droid_g3_scripted import (
    DroidG3ScriptedError,
    ScriptedControllerConfig,
    _placement_feedback_target,
    minimum_jerk_waypoints,
    run_scripted_horizontal_check,
    select_robust_target_task,
    select_robust_target_world,
    trajectory_json_compatible,
)


def _goal_for_relation(relation: str) -> geom.GoalSetResult:
    frame = geom.TaskFrame.identity()
    workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
    foot_obj = geom.ObjectFootprint(0.02, 0.02, 0.02)
    foot_ref = geom.ObjectFootprint(0.05, 0.05, 0.02)
    spec = geom.PlanarRelationSpec(
        relation=relation,  # type: ignore[arg-type]
        clearance_m=0.01,
        workspace=workspace,
        object_footprint=foot_obj,
        reference_footprint=foot_ref,
    )
    return geom.build_planar_goal_set(frame, spec, (0.0, 0.0, 0.04))


class _FakeWorld:
    def __init__(self, poses: dict[str, tuple[float, float, float]]) -> None:
        self.poses = poses

    def get_pose(self, name: str, *, env_id: int = 0) -> tuple[list[float], list[float]]:
        del env_id
        position = self.poses[name]
        return [float(v) for v in position], [1.0, 0.0, 0.0, 0.0]


class _FakeFrameData:
    def __init__(self, position: tuple[float, float, float]) -> None:
        self.target_frame_names = ["eef_frame"]
        self.target_pos_w = [[list(position)]]
        self.target_quat_w = [[[1.0, 0.0, 0.0, 0.0]]]


class _FakeEnv:
    def __init__(
        self,
        *,
        poses: dict[str, tuple[float, float, float]] | None = None,
        grabbed_after: int = 0,
        dropped_after: int | None = None,
        terminate_at: int | None = None,
        truncate_at: int | None = None,
        terminal_predicates: TerminalPhysicalPredicates | None = None,
        final_object_pose: tuple[float, float, float] | None = None,
    ) -> None:
        self.poses = dict(
            poses
            or {
                "rubiks_cube": (0.20, 0.05, 0.04),
                "bowl": (0.0, 0.0, 0.04),
                "robot": (0.0, 0.0, 0.0),
            }
        )
        self.robot_position = (0.10, 0.02, 0.20)
        self.tick = 0
        self.grabbed_after = grabbed_after
        self.dropped_after = dropped_after
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.terminal_predicates = terminal_predicates or TerminalPhysicalPredicates(
            available=True,
            allowed_support=True,
            stable_for_dwell=True,
        )
        self.final_object_pose = final_object_pose
        self.callback_invocations: list[tuple[int, float]] = []
        world = _FakeWorld(self.poses)
        frames = SimpleNamespace(data=_FakeFrameData(self.robot_position))
        self.backend = SimpleNamespace(
            env=SimpleNamespace(scene={"frames": frames}),
            modules={
                "get_world": lambda _env: world,
                "object_grabbed": self._object_grabbed,
                "object_dropped": self._object_dropped,
                "eef_offset_rotation": (1.0, 0.0, 0.0, 0.0),
            },
        )
        self.control_dt_s = 0.1

    def _object_grabbed(self, _env: object, *, object: str, env_id: int = 0) -> bool:
        del _env, object, env_id
        return self.tick >= self.grabbed_after

    def _object_dropped(self, _env: object, *, object: str, env_id: int = 0) -> bool:
        del _env, object, env_id
        if self.dropped_after is None:
            return False
        return self.tick >= self.dropped_after

    def step(self, action: tuple[float, ...]) -> tuple[dict[str, object], dict[str, object]]:
        del action
        self.tick += 1
        self.robot_position = (
            self.robot_position[0] + 0.001,
            self.robot_position[1],
            self.robot_position[2] + (0.05 if self.tick > 8 else 0.0),
        )
        frames = self.backend.env.scene["frames"]
        frames.data = _FakeFrameData(self.robot_position)
        if self.tick >= 7:
            cube_x, cube_y, _cube_z = self.poses["rubiks_cube"]
            self.poses["rubiks_cube"] = (cube_x, cube_y, 0.09)
        terminated = self.terminate_at is not None and self.tick >= self.terminate_at
        truncated = self.truncate_at is not None and self.tick >= self.truncate_at
        return {}, {"terminated": terminated, "truncated": truncated}

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        if self.final_object_pose is not None:
            self.poses["rubiks_cube"] = self.final_object_pose
        return self.terminal_predicates


class WaypointTests(unittest.TestCase):
    def test_minimum_jerk_endpoints(self) -> None:
        start = (0.0, 0.0, 0.1)
        end = (0.2, -0.1, 0.3)
        points = minimum_jerk_waypoints(start, end, 10)
        self.assertEqual(len(points), 10)
        for axis in range(3):
            self.assertAlmostEqual(points[0][axis], start[axis], places=9)
            self.assertAlmostEqual(points[-1][axis], end[axis], places=9)

    def test_minimum_jerk_is_monotonic_per_axis(self) -> None:
        start = (0.0, 0.0, 0.1)
        end = (0.2, -0.1, 0.3)
        points = minimum_jerk_waypoints(start, end, 12)
        for axis in range(3):
            delta = end[axis] - start[axis]
            values = [point[axis] for point in points]
            if delta >= 0.0:
                self.assertTrue(all(left <= right + 1e-12 for left, right in zip(values, values[1:])))
            else:
                self.assertTrue(all(left >= right - 1e-12 for left, right in zip(values, values[1:])))

    def test_place_feedback_corrects_xy_and_clamps(self) -> None:
        corrected = _placement_feedback_target(
            nominal_target=(0.20, 0.10, 0.30),
            placement_target=(0.20, 0.10, 0.05),
            object_position=(0.24, 0.08, 0.12),
            gain=1.0,
            max_correction_m=0.03,
        )
        self.assertAlmostEqual(corrected[0], 0.17)
        self.assertAlmostEqual(corrected[1], 0.12)
        self.assertAlmostEqual(corrected[2], 0.30)


class TargetSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = geom.TaskFrame.identity()
        self.cube = (0.20, 0.05, 0.04)
        self.table_top = 0.0
        self.half_up = 0.02
        self.inset = 0.015

    def test_all_four_relations_select_inside_goal(self) -> None:
        expected_x = {
            "left": 0.0 + 0.05 + 0.02 + 0.01 + self.inset,
            "right": -0.0 - 0.05 - 0.02 - 0.01 - self.inset,
        }
        expected_y = {
            "front": 0.0 + 0.05 + 0.02 + 0.01 + self.inset,
            "behind": -0.0 - 0.05 - 0.02 - 0.01 - self.inset,
        }
        for relation in ("left", "right", "front", "behind"):
            goal = _goal_for_relation(relation)
            target_task = select_robust_target_task(
                frame=self.frame,
                goal=goal,
                cube_position_world=self.cube,
                relation=relation,
                inset_m=self.inset,
                table_top_z_task=self.table_top,
                object_half_up=self.half_up,
            )
            if relation in expected_x:
                self.assertAlmostEqual(target_task[0], expected_x[relation], places=9)
                self.assertAlmostEqual(target_task[1], self.cube[1], places=9)
            else:
                self.assertAlmostEqual(target_task[1], expected_y[relation], places=9)
                self.assertAlmostEqual(target_task[0], self.cube[0], places=9)
            self.assertAlmostEqual(target_task[2], self.table_top + self.half_up, places=9)
            self.assertTrue(goal.region is not None and goal.region.point_inside(target_task))

    def test_narrow_nonempty_goal_uses_opposite_region_edge(self) -> None:
        goal = replace(
            _goal_for_relation("left"),
            region=geom.AxisAlignedBox(0.10, 0.11, -0.05, 0.05, 0.0, 0.08),
        )
        target = select_robust_target_task(
            frame=self.frame,
            goal=goal,
            cube_position_world=self.cube,
            relation="left",
            inset_m=self.inset,
            table_top_z_task=self.table_top,
            object_half_up=self.half_up,
        )
        self.assertAlmostEqual(target[0], 0.11)
        self.assertTrue(goal.region is not None and goal.region.point_inside(target))

    def test_world_target_roundtrip(self) -> None:
        goal = _goal_for_relation("left")
        world = select_robust_target_world(
            frame=self.frame,
            goal=goal,
            cube_position_world=self.cube,
            relation="left",
            inset_m=self.inset,
            table_top_z_task=self.table_top,
            object_half_up=self.half_up,
        )
        task = self.frame.world_to_task(world)
        self.assertTrue(goal.region is not None and goal.region.point_inside(task))


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_nonpositive_ticks(self) -> None:
        payload = {
            "phase_ticks": {
                "approach": 0,
                "descend": 5,
                "close_dwell": 5,
                "lift": 5,
                "transport": 5,
                "place_descend": 5,
                "open_dwell": 5,
                "retreat": 5,
                "settle": 5,
            },
            "geometry_offsets": {
                "approach_height_m": 0.12,
                "descend_offset_m": 0.025,
                "lift_height_m": 0.12,
                "transport_height_m": 0.12,
                "place_descend_offset_m": 0.04,
                "retreat_height_m": 0.10,
                "target_inset_m": 0.015,
            },
        }
        with self.assertRaises(DroidG3ScriptedError):
            ScriptedControllerConfig.from_mapping(payload)

    def test_rejects_nonfinite_offsets(self) -> None:
        payload = {
            "phase_ticks": {"approach": 5, "descend": 5, "close_dwell": 5, "lift": 5, "transport": 5, "place_descend": 5, "open_dwell": 5, "retreat": 5, "settle": 5},
            "geometry_offsets": {
                "approach_height_m": math.nan,
                "descend_offset_m": 0.025,
                "lift_height_m": 0.12,
                "transport_height_m": 0.12,
                "place_descend_offset_m": 0.04,
                "retreat_height_m": 0.10,
                "target_inset_m": 0.015,
            },
        }
        with self.assertRaises(DroidG3ScriptedError):
            ScriptedControllerConfig.from_mapping(payload)


class ScriptedRunTests(unittest.TestCase):
    def _small_config(self) -> ScriptedControllerConfig:
        return ScriptedControllerConfig.from_mapping(
            {
                "phase_ticks": {phase: 2 for phase in (
                    "approach",
                    "descend",
                    "close_dwell",
                    "lift",
                    "transport",
                    "place_descend",
                    "open_dwell",
                    "retreat",
                    "settle",
                )},
                "geometry_offsets": {
                    "approach_height_m": 0.12,
                    "descend_offset_m": 0.025,
                    "lift_height_m": 0.12,
                    "transport_height_m": 0.12,
                    "place_descend_offset_m": 0.04,
                    "retreat_height_m": 0.10,
                    "target_inset_m": 0.015,
                },
                "min_grasp_lift_m": 0.04,
            }
        )

    def test_successful_run_records_trajectory_and_passes_stages(self) -> None:
        goal = _goal_for_relation("left")
        env = _FakeEnv(
            grabbed_after=3,
            dropped_after=13,
            final_object_pose=(0.30, 0.05, 0.02),
        )
        result = run_scripted_horizontal_check(
            env,
            relation="left",
            goal=goal,
            frame=geom.TaskFrame.identity(),
            config=self._small_config(),
            table_top_z_task=0.0,
            object_half_up=0.02,
        )
        json.loads(trajectory_json_compatible(result))
        self.assertEqual(result["model_request_count"], 0)
        self.assertEqual(result["tick_count"], 18)
        self.assertTrue(all(result["stages"].values()))
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["trajectory"]), 18)
        self.assertEqual(result["trajectory"][0]["phase"], "approach")
        self.assertEqual(result["trajectory"][-1]["phase"], "settle")

    def test_stage_distinction_when_grasp_or_release_fail(self) -> None:
        goal = _goal_for_relation("left")
        env = _FakeEnv(grabbed_after=999, dropped_after=999)
        result = run_scripted_horizontal_check(
            env,
            relation="left",
            goal=goal,
            frame=geom.TaskFrame.identity(),
            config=self._small_config(),
            table_top_z_task=0.0,
            object_half_up=0.02,
        )
        self.assertFalse(result["stages"]["grasped"])
        self.assertFalse(result["stages"]["transported"])
        self.assertFalse(result["stages"]["released"])
        self.assertIn("no_contact_after_close", result["reasons"])

    def test_moving_reference_callback_invoked_each_tick(self) -> None:
        goal = _goal_for_relation("front")
        env = _FakeEnv(grabbed_after=1, dropped_after=12)

        def callback(tick: int, sim_time_s: float) -> None:
            env.callback_invocations.append((tick, sim_time_s))

        run_scripted_horizontal_check(
            env,
            relation="front",
            goal=goal,
            frame=geom.TaskFrame.identity(),
            config=self._small_config(),
            table_top_z_task=0.0,
            object_half_up=0.02,
            reference_motion_callback=callback,
        )
        self.assertEqual(len(env.callback_invocations), 18)
        self.assertEqual(env.callback_invocations[0], (1, 0.1))
        self.assertEqual(env.callback_invocations[-1][0], 18)

    def test_termination_failure_marks_early_exit(self) -> None:
        goal = _goal_for_relation("right")
        env = _FakeEnv(terminate_at=3)
        result = run_scripted_horizontal_check(
            env,
            relation="right",
            goal=goal,
            frame=geom.TaskFrame.identity(),
            config=self._small_config(),
            table_top_z_task=0.0,
            object_half_up=0.02,
        )
        self.assertTrue(result["terminated_early"])
        self.assertEqual(result["termination_reason"], "simulator_terminated")
        self.assertFalse(result["passed"])
        self.assertEqual(result["tick_count"], 3)

    def test_truncation_failure(self) -> None:
        goal = _goal_for_relation("behind")
        env = _FakeEnv(truncate_at=2)
        result = run_scripted_horizontal_check(
            env,
            relation="behind",
            goal=goal,
            frame=geom.TaskFrame.identity(),
            config=self._small_config(),
            table_top_z_task=0.0,
            object_half_up=0.02,
        )
        self.assertTrue(result["terminated_early"])
        self.assertEqual(result["termination_reason"], "simulator_truncated")

    def test_goal_satisfied_uses_final_object_center(self) -> None:
        goal = _goal_for_relation("left")
        env = _FakeEnv(
            grabbed_after=1,
            dropped_after=12,
            final_object_pose=(0.05, 0.05, 0.02),
        )
        result = run_scripted_horizontal_check(
            env,
            relation="left",
            goal=goal,
            frame=geom.TaskFrame.identity(),
            config=self._small_config(),
            table_top_z_task=0.0,
            object_half_up=0.02,
        )
        self.assertFalse(result["stages"]["goal_satisfied"])
        self.assertIn("object_center_outside_goal_set", result["reasons"])

    def test_module_import_has_no_robolab_dependency(self) -> None:
        import importlib
        import sys

        module_name = "experiments.online_correction_v4.droid_g3_scripted"
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("robolab", source)
        self.assertNotIn("isaac", source.lower())


if __name__ == "__main__":
    unittest.main()
